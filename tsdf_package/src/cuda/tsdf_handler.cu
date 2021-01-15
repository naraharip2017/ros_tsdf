#include "cuda/tsdf_handler.cuh"

//used to determine num blocks when executing cuda kernel
const int threadsPerCudaBlock = 128;

//error function for cpu called after kernel calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//error function for dynamic parallelism
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

//used for debugging, current index represents num of heap blocks in heap
__global__
void printHashTableAndBlockHeap(HashTable * hashTable_d, BlockHeap * blockHeap_d){
  // HashEntry * hashEntries = hashTable_d->hashEntries;
  // for(size_t i=0;i<NUM_BUCKETS; ++i){
  //   printf("Bucket: %lu\n", (unsigned long)i);
  //   for(size_t it = 0; it<HASH_ENTRIES_PER_BUCKET; ++it){
  //     HashEntry hashEntry = hashEntries[it+i*HASH_ENTRIES_PER_BUCKET];
  //     Vector3f position = hashEntry.position;
  //     if (hashEntry.isFree()){
  //       printf("  Hash Entry with   Position: (N,N,N)   Offset: %d   Pointer: %d\n", hashEntry.offset, hashEntry.pointer);
  //     }
  //     else{
  //       printf("  Hash Entry with   Position: (%f,%f,%f)   Offset: %d   Pointer: %d\n", position(0), position(1), position(2), hashEntry.offset, hashEntry.pointer);
  //     }
  //   }
  //   printf("%s\n", "--------------------------------------------------------");
  // }

  // printf("Block Heap Free List: ");
  // int * freeBlocks = blockHeap_d->freeBlocks;
  // for(size_t i = 0; i<NUM_HEAP_BLOCKS; ++i){
  //   printf("%d  ", freeBlocks[i]);
  // }
  // printf("\n");
  printf("Current Block Heap Free Index: %d\n", blockHeap_d->currentIndex);
}

/*
* given voxel block center point returns hashed bucket index for the block
*/
__device__
size_t retrieveHashIndexFromPoint(Vector3f point){
  return abs((((int)point(0)*PRIME_ONE) ^ ((int)point(1)*PRIME_TWO) ^ ((int)point(2)*PRIME_THREE)) % NUM_BUCKETS);
}

/*
* calculates floor in scale given
*/
__device__ 
float FloorFun(float x, float scale){
  return floor(x*scale) / scale;
}

/*
* Given world point return center of volume given by volume_size
*/
__device__
Vector3f getVolumeCenterFromPoint(Vector3f point, float volume_size){
  float scale = 1/volume_size;
  float half_volume_size = volume_size / 2;
  Vector3f volume_center;
  volume_center(0) = FloorFun(point(0), scale) + half_volume_size;
  volume_center(1) = FloorFun(point(1), scale) + half_volume_size;
  volume_center(2) = FloorFun(point(2), scale) + half_volume_size;
  return volume_center;
}

/*
* Check if two Vector3f are equal
*/
__device__
bool checkFloatingPointVectorsEqual(Vector3f A, Vector3f B, float epsilon){
  Vector3f diff = A-B;
  //have to use an epsilon value due to floating point precision errors
  if((fabs(diff(0)) < epsilon) && (fabs(diff(1)) < epsilon) && (fabs(diff(2)) < epsilon))
    return true;

  return false;
}

/*
* calculate the line between point_d and origin_transformed_d then get points truncation distance away from point_d on the line. Set truncation_start and truncation_end to those points
* truncation_start will be closer to origin
*/
__device__
inline void getTruncationLineEndPoints(pcl::PointXYZ & point_d, Vector3f * origin_transformed_d, Vector3f & truncation_start, Vector3f & truncation_end){
  Vector3f u = *origin_transformed_d;
  Vector3f point_d_vector(point_d.x, point_d.y, point_d.z);
  Vector3f v = point_d_vector - u; //direction
  //equation of line is u+tv
  float vMag = sqrt(pow(v(0), 2) + pow(v(1),2) + pow(v(2), 2));
  Vector3f v_normalized = v / vMag;
  truncation_start = point_d_vector - TRUNCATION_DISTANCE*v_normalized;
  
  truncation_end = point_d_vector + TRUNCATION_DISTANCE*v_normalized;

  //set truncation_start to whichever point is closer to the origin
  float distance_tStart_origin = pow(truncation_start(0) - u(0), 2) + pow(truncation_start(1) - u(1),2) + pow(truncation_start(2) - u(2), 2);
  float distance_tEnd_origin = pow(truncation_end(0) - u(0), 2) + pow(truncation_end(1) - u(1),2) + pow(truncation_end(2) - u(2), 2);

  if(distance_tEnd_origin < distance_tStart_origin){
    Vector3f temp = truncation_start;
    truncation_start = truncation_end;
    truncation_end = temp;
  }
}

/*
* Given the truncation start volume, volume size, and ray defined by u and t traverse the world till reaching truncation_end_vol
* Read for more information: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
*/
__device__
inline void traverseVolume(Vector3f & truncation_start_vol, Vector3f & truncation_end_vol, const float & volume_size, Vector3f & u, Vector3f & v, Vector3f * traversed_vols, int * traversed_vols_size){
  float half_volume_size = volume_size / 2;
  const float epsilon = volume_size / 4;
  const float volume_size_plus_epsilon = volume_size + epsilon;
  const float volume_size_minus_epsilon = volume_size - epsilon;
  float stepX = v(0) > 0 ? volume_size : -1 * volume_size;
  float stepY = v(1) > 0 ? volume_size : -1 * volume_size;
  float stepZ = v(2) > 0 ? volume_size : -1 * volume_size;
  Vector3f truncation_vol_start_center = getVolumeCenterFromPoint(truncation_start_vol, volume_size);
  float tMaxX = fabs(v(0) < 0 ? (truncation_vol_start_center(0) - half_volume_size - u(0)) / v(0) : (truncation_vol_start_center(0) + half_volume_size - u(0)) / v(0));
  float tMaxY = fabs(v(1) < 0 ? (truncation_vol_start_center(1) - half_volume_size - u(1)) / v(1) : (truncation_vol_start_center(1) + half_volume_size - u(1)) / v(1));
  float tMaxZ = fabs(v(2) < 0 ? (truncation_vol_start_center(2) - half_volume_size - u(2)) / v(2) : (truncation_vol_start_center(2) + half_volume_size - u(2)) / v(2));
  float tDeltaX = fabs(volume_size / v(0));
  float tDeltaY = fabs(volume_size / v(1));
  float tDeltaZ = fabs(volume_size / v(2));
  Vector3f current_vol(truncation_start_vol(0), truncation_start_vol(1), truncation_start_vol(2));
  Vector3f current_vol_center = getVolumeCenterFromPoint(current_vol, volume_size);
  Vector3f truncation_vol_end_center = getVolumeCenterFromPoint(truncation_end_vol, volume_size);

  // printf("-----------------------------------------------------\n");
  // printf("truncation_vol_start_center: (%f,%f,%f)\n", truncation_vol_start_center(0), truncation_vol_start_center(1), truncation_vol_start_center(2));
  // printf("truncation_vol_end_center: (%f,%f,%f)\n", truncation_vol_end_center(0), truncation_vol_end_center(1), truncation_vol_end_center(2));
  // printf("-----------------------------------------------------\n");
  int insert_index;
  int current_index = 0;
  while(!checkFloatingPointVectorsEqual(current_vol_center, truncation_vol_end_center, epsilon)){
    //add traversed volume to list of traversed volume and update size atomically
    insert_index = atomicAdd(&(*traversed_vols_size), 1);
    traversed_vols[insert_index] = current_vol_center;
    // printf("current_vol: (%f,%f,%f)\n", current_vol(0), current_vol(1), current_vol(2));
    // printf("current_vol_center: (%f,%f,%f)\n", current_vol_center(0), current_vol_center(1), current_vol_center(2));
    // printf("tMaxX: %f, tMaxy: %f, tMaxZ:%f\n", tMaxX, tMaxY, tMaxZ);
    // printf("deltaX: %f, deltaY: %f, deltaZ:%f\n", tDeltaX, tDeltaY, tDeltaZ);
    current_index ++;
    if(current_index > 30){
      printf("current_vol_center: (%f,%f,%f), end_center: (%f,%f,%f), start_center: (%f,%f,%f), end: (%f,%f,%f), start: (%f,%f,%f)\n", current_vol_center(0), current_vol_center(1), current_vol_center(2), truncation_vol_end_center(0), truncation_vol_end_center(1), truncation_vol_end_center(2), truncation_vol_start_center(0), truncation_vol_start_center(1), truncation_vol_start_center(2), truncation_end_vol(0), truncation_end_vol(1), truncation_end_vol(2), truncation_start_vol(0), truncation_start_vol(1), truncation_start_vol(2));

      break;
    }

    if(tMaxX < tMaxY){
      if(tMaxX < tMaxZ)
      {
        current_vol(0) += stepX;
        tMaxX += tDeltaX;
      }
      else if(tMaxX > tMaxZ){
        current_vol(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else{
        current_vol(0) += stepX;
        current_vol(2) += stepZ;
        tMaxX += tDeltaX;
        tMaxZ += tDeltaZ;
      }
    }
    else if(tMaxX > tMaxY){
      if(tMaxY < tMaxZ){
        current_vol(1) += stepY;
        tMaxY += tDeltaY;
      }
      else if(tMaxY > tMaxZ){
        current_vol(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else{
        current_vol(1) += stepY;
        current_vol(2) += stepZ;
        tMaxY += tDeltaY;
        tMaxZ += tDeltaZ;
      }
    }
    else{
      if(tMaxZ < tMaxX){
        current_vol(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else if(tMaxZ > tMaxX){
        current_vol(0) += stepX;
        current_vol(1) += stepY;
        tMaxX += tDeltaX;
        tMaxY += tDeltaY;
      }
      else{ 
        current_vol(0) += stepX;
        current_vol(1) += stepY;
        current_vol(2) += stepZ;
        tMaxX += tDeltaX;
        tMaxY += tDeltaY;
        tMaxZ += tDeltaZ;
      }
    } 
    Vector3f temp_current_vol_center = current_vol_center;
    current_vol_center = getVolumeCenterFromPoint(current_vol, volume_size);
    Vector3f diff;
    diff(0) = fabs(temp_current_vol_center(0) - current_vol_center(0));
    diff(1) = fabs(temp_current_vol_center(1) - current_vol_center(1));
    diff(2) = fabs(temp_current_vol_center(2) - current_vol_center(2));
    // printf("diff: (%f,%f,%f)\n",diff(0), diff(1), diff(2));
    if((diff(0) < volume_size_minus_epsilon && diff(1) < volume_size_minus_epsilon && diff(2) < volume_size_minus_epsilon) || (diff(0) > volume_size_plus_epsilon || diff(1) > volume_size_plus_epsilon || diff(2) > volume_size_plus_epsilon)){
      return;
    }
    // printf("-----------------------------------------------------\n");
  }      

  //add traversed volume to list of traversed volume and update size atomically
  insert_index = atomicAdd(&(*traversed_vols_size), 1);
  traversed_vols[insert_index] = current_vol_center;
  // printf("current_vol: (%f,%f,%f)\n", current_vol(0), current_vol(1), current_vol(2));
  // printf("current_vol_center: (%f,%f,%f)\n", current_vol_center(0), current_vol_center(1), current_vol_center(2));

}

/*
* Kernel will get voxel blocks along ray between origin and a point in lidar cloud within truncation distance of point
*/
__global__
void getVoxelBlocksForPoint(pcl::PointXYZ * points_d, Vector3f * pointCloudVoxelBlocks_d, int * pointer_d, Vector3f * origin_transformed_d, int * size_d){
  int threadIndex = (blockIdx.x*threadsPerCudaBlock + threadIdx.x);
  if(threadIndex>=*size_d){
    return;
  }
  pcl::PointXYZ point_d = points_d[threadIndex];
  Vector3f truncation_start;
  Vector3f truncation_end;

  getTruncationLineEndPoints(point_d, origin_transformed_d, truncation_start, truncation_end);
  //equation of line is u+tv
  Vector3f u = truncation_start;
  Vector3f v = truncation_end - truncation_start;

  traverseVolume(truncation_start, truncation_end, VOXEL_BLOCK_SIZE, u, v, pointCloudVoxelBlocks_d, pointer_d);
  return;
}

/*
* Given block coordinates return the position of the block in the block heap or return -1 if not allocated
*/
__device__ 
int getBlockPositionForBlockCoordinates(Vector3f & voxelBlockCoordinates, size_t & bucket_index, size_t & currentGlobalIndex, HashEntry * hashEntries){

  HashEntry hashEntry;

  //check the hashed bucket for the block
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hashEntry = hashEntries[currentGlobalIndex+i];
    if(checkFloatingPointVectorsEqual(hashEntry.position, voxelBlockCoordinates, BLOCK_EPSILON)){
      return hashEntry.pointer;
    }
  }

  currentGlobalIndex+=HASH_ENTRIES_PER_BUCKET-1;

  //check the linked list if necessary
  while(hashEntry.offset!=0){
    int offset = hashEntry.offset;
    currentGlobalIndex+=offset;
    if(currentGlobalIndex>=HASH_TABLE_SIZE){
      currentGlobalIndex %= HASH_TABLE_SIZE;
    }
    hashEntry = hashEntries[currentGlobalIndex];
    if(checkFloatingPointVectorsEqual(hashEntry.position, voxelBlockCoordinates, BLOCK_EPSILON)){
      return hashEntry.pointer;
    }
  }

  //block not allocated in hashTable
  return -1;
}

/*
* If allocating a new block then first attempt to insert in the bucket it hashed to
*/
__device__
inline bool attemptHashedBucketVoxelBlockCreation(size_t & hashedBucketIndex, BlockHeap * blockHeap_d, Vector3f & voxelBlockCoordinates, HashEntry * hashEntries){
  //get position of beginning of hashed bucket
  size_t insertCurrentGlobalIndex = hashedBucketIndex * HASH_ENTRIES_PER_BUCKET;
  //loop through bucket and insert if there is a free space
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    if(hashEntries[insertCurrentGlobalIndex+i].isFree()){ 
      //get next free position in block heap
      int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
      VoxelBlock * allocBlock = new VoxelBlock();
      HashEntry * allocBlockHashEntry = new HashEntry(voxelBlockCoordinates, blockHeapFreeIndex);
      blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
      hashEntries[insertCurrentGlobalIndex+i] = *allocBlockHashEntry;
      cudaFree(allocBlock);
      cudaFree(allocBlockHashEntry);
      return true;
    }
  }
  //return false if no free position in hashed bucket
  return false;
} 

/*
* Attempt to allocated a new block into the linked list of the bucket it hashed to if there is no space in the bucket
*/
__device__
inline bool attemptLinkedListVoxelBlockCreation(size_t & hashedBucketIndex, BlockHeap * blockHeap_d, HashTable * hashTable_d, size_t & insertBucketIndex, size_t & endLinkedListPointer, Vector3f & voxelBlockCoordinates, HashEntry * hashEntries){
  size_t insertCurrentGlobalIndex;
  //only try to insert into other buckets until we get the block's hashed bucket which has already been tried
  while(insertBucketIndex!=hashedBucketIndex){
    //check that can get lock of the bucket attempting to insert the block
    if(!atomicCAS(&hashTable_d->mutex[insertBucketIndex], 0, 1)){
      insertCurrentGlobalIndex = insertBucketIndex * HASH_ENTRIES_PER_BUCKET;
      //loop through the insert bucket (not including the last slot which is reserved for linked list start of that bucket) checking for a free space
      for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET-1; ++i){
        if(hashEntries[insertCurrentGlobalIndex+i].isFree() ){ 
            int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
            VoxelBlock * allocBlock = new VoxelBlock();
            HashEntry * allocBlockHashEntry = new HashEntry(voxelBlockCoordinates, blockHeapFreeIndex);
            blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
            size_t insertPos = insertCurrentGlobalIndex + i;
            hashEntries[insertPos] = *allocBlockHashEntry;
            cudaFree(allocBlock);
            cudaFree(allocBlockHashEntry);
            //set offset value for last hashEntry in the linked list
            if(insertPos > endLinkedListPointer){
              hashEntries[endLinkedListPointer].offset = insertPos - endLinkedListPointer;
            }
            else{
              hashEntries[endLinkedListPointer].offset = HASH_TABLE_SIZE - endLinkedListPointer + insertPos;
            }
            return true;
        }
      }
      //if no free space in insert bucket then release lock
      atomicExch(&hashTable_d->mutex[insertBucketIndex], 0);
    }
    //move to next bucket to check for space to insert block
    insertBucketIndex++;
    //loop back to beginning of hash table if overflowed size
    if(insertBucketIndex == NUM_BUCKETS){
      insertBucketIndex = 0;
    }
  }
  //if found no position for block in current frame
  return false;
}

/*
* Given list of voxelBlocks check that the block is allocated in the hashtable and if not do so. Unallocated points in the current frame due to not reveiving lock or lack of
* space in hashTable then return to be processed in next frame
*/
__global__
void allocateVoxelBlocks(Vector3f * voxelBlocks_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, bool * unallocatedPoints_d, int * size_d, int * unallocatedPointsCount_d)
{
  int threadIndex = (blockIdx.x*threadsPerCudaBlock + threadIdx.x);
  //check that the thread is a valid index in voxelBlocks_d and that the block is still unallocated
  if(threadIndex>=*size_d || (unallocatedPoints_d[threadIndex]==0)){
    return;
  }
  Vector3f voxelBlock = voxelBlocks_d[threadIndex];

  size_t hashedBucketIndex = retrieveHashIndexFromPoint(voxelBlock);
  //beginning of the hashedBucket in hashTable
  size_t currentGlobalIndex = hashedBucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hashEntries = hashTable_d->hashEntries;

  int block_position = getBlockPositionForBlockCoordinates(voxelBlock, hashedBucketIndex, currentGlobalIndex, hashEntries);

  //block is already allocated then return
  if(block_position!=-1){
    unallocatedPoints_d[threadIndex] = 0;
    atomicSub(unallocatedPointsCount_d, 1);
    return;
  }

  //attempt to get lock for hashed bucket for potential insert
  if(!atomicCAS(&hashTable_d->mutex[hashedBucketIndex], 0, 1)){
    //try to insert block into the bucket it hashed to and return if possible
    if(attemptHashedBucketVoxelBlockCreation(hashedBucketIndex, blockHeap_d, voxelBlock, hashEntries)) {
      //the block is allocated so set it to allocated so it is not processed again in another frame
      unallocatedPoints_d[threadIndex] = 0;
      //subtract 1 from count of unallocated blocks
      atomicSub(unallocatedPointsCount_d, 1);
      //free lock for hashed bucket
      atomicExch(&hashTable_d->mutex[hashedBucketIndex], 0);
      return;
    }

    //start searching for a free position in the next bucket
    size_t insertBucketIndex = hashedBucketIndex + 1;
    //set insertBucket to first bucket if overflow the hash table size
    if(insertBucketIndex == NUM_BUCKETS){
      insertBucketIndex = 0;
    }

    //Note: current global index will point to end of linked list which includes hashed bucket if no linked list
    //index to the bucket which contains the end of the linked list for the hashed bucket of the block
    size_t endLinkedListBucket = currentGlobalIndex / HASH_ENTRIES_PER_BUCKET;

    bool haveEndLinkedListBucketLock = true;

    //if end of linked list is in different bucket than hashed bucket try to get the lock for the end of the linked list
    if(endLinkedListBucket!=hashedBucketIndex){
      //release lock of the hashedBucket
      atomicExch(&hashTable_d->mutex[hashedBucketIndex], 0);
      //attempt to get lock of bucket with end of linked list
      haveEndLinkedListBucketLock = !atomicCAS(&hashTable_d->mutex[endLinkedListBucket], 0, 1);
    }

    if(haveEndLinkedListBucketLock){
      //try to insert block into the linked list for it's hashed bucket
      if(attemptLinkedListVoxelBlockCreation(hashedBucketIndex, blockHeap_d, hashTable_d, insertBucketIndex, currentGlobalIndex, voxelBlock, hashEntries)){
        //the block is allocated so set it to allocated so it is not processed again in another frame
        unallocatedPoints_d[threadIndex] = 0;
        //subtract 1 from count of unallocated blocks
        atomicSub(unallocatedPointsCount_d, 1); 
        //free the lock of the end of linked list bucket and insert bucket
        atomicExch(&hashTable_d->mutex[endLinkedListBucket], 0);
        atomicExch(&hashTable_d->mutex[insertBucketIndex], 0);
      }
      else{
        //free the lock of the end of linked list bucket
        atomicExch(&hashTable_d->mutex[endLinkedListBucket], 0);
      }
      return;
    }
  }
}

/*
* given diff between a voxel coordinate and the bottom left block of the block it is in return the position of the voxel in the blocks voxel array
*/
__device__
size_t getLocalVoxelIndex(Vector3f diff){
  diff /= VOXEL_SIZE;
  return floor(diff(0)) + (floor(diff(1)) * VOXEL_PER_SIDE) + (floor(diff(2)) * VOXEL_PER_SIDE * VOXEL_PER_SIDE);
}


/*
* return magnitude of vector
*/
__device__
inline float getMagnitude(Vector3f vector){
  return sqrt(pow(vector(0),2) + pow(vector(1),2) + pow(vector(2),2));
}

/*
* return dot product of two vectors
*/
__device__
inline float dotProduct(Vector3f a, Vector3f b){
  return a(0)*b(0) + a(1)*b(1) + a(2)*b(2);
}

/*
* given voxel coordinates, position of lidar points and sensor origin calculate the distance update for the voxel
* http://helenol.github.io/publications/iros_2017_voxblox.pdf for more info
*/
__device__
inline float getDistanceUpdate(Vector3f voxelCoordinates, Vector3f point_d, Vector3f origin){
  //x:center of current voxel   p:position of lidar point   s:sensor origin
  Vector3f lidarOriginDiff = point_d - origin; //p-s
  Vector3f lidarVoxelDiff = point_d - voxelCoordinates; //p-x
  float magnitudeLidarVoxelDiff = getMagnitude(lidarVoxelDiff);
  float dotProd = dotProduct(lidarOriginDiff, lidarVoxelDiff);
  if(dotProd < 0){
    magnitudeLidarVoxelDiff *=-1;
  }

  return magnitudeLidarVoxelDiff;
}

__device__
inline float getWeightUpdate(float distance){
  return 1/(fabs(distance) + 1);
}

/*
* update voxels sdf and weight that are within truncation distance of the line between origin and point_d
*/
__global__
void updateVoxels(Vector3f * voxels, HashTable * hashTable_d, BlockHeap * blockHeap_d, Vector3f * point_d, Vector3f * origin, int * size){
  int threadIndex = blockIdx.x*threadsPerCudaBlock + threadIdx.x;
  if(threadIndex >= * size){
    return;
  }
  Vector3f voxelCoordinates = voxels[threadIndex];
  Vector3f voxelBlockCoordinates = getVolumeCenterFromPoint(voxelCoordinates, VOXEL_BLOCK_SIZE);

  size_t bucketIndex = retrieveHashIndexFromPoint(voxelBlockCoordinates);
  size_t currentGlobalIndex = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hashEntries = hashTable_d->hashEntries;

  //todo: make a global vector with half voxel block size values
  int voxelBlockHeapPosition = getBlockPositionForBlockCoordinates(voxelBlockCoordinates, bucketIndex, currentGlobalIndex, hashEntries);
  Vector3f voxelBlockBottomLeftCoordinates;
  voxelBlockBottomLeftCoordinates(0) = voxelBlockCoordinates(0)-HALF_VOXEL_BLOCK_SIZE;
  voxelBlockBottomLeftCoordinates(1) = voxelBlockCoordinates(1)-HALF_VOXEL_BLOCK_SIZE;
  voxelBlockBottomLeftCoordinates(2) = voxelBlockCoordinates(2)-HALF_VOXEL_BLOCK_SIZE;
  //get local positin of voxel in its block
  size_t localVoxelIndex = getLocalVoxelIndex(voxelCoordinates - voxelBlockBottomLeftCoordinates);

  VoxelBlock * block = &(blockHeap_d->blocks[voxelBlockHeapPosition]);
  Voxel* voxel = &(block->voxels[localVoxelIndex]);
  int * mutex = &(block->mutex[localVoxelIndex]);

  float distance = getDistanceUpdate(voxelCoordinates, *point_d, *origin);
  float weight = getWeightUpdate(distance);
  float weightTimesDistance = weight * distance;

  //get lock for voxel
  // TODO: Remove infinite loop hack
  int i = 0;
  bool updatedVoxel = false;
  while(!updatedVoxel){
    //get lock for voxel
    if(!atomicCAS(mutex, 0, 1)){
      updatedVoxel = true;
      //update sdf and weight
      float oldWeight = voxel->weight;
      float oldSdf = voxel->sdf;
      float newWeight = oldWeight + weight;
      float newDistance = (oldWeight * oldSdf + weightTimesDistance) / newWeight;
      voxel->sdf = newDistance;
      newWeight = min(newWeight, MAX_WEIGHT);
      voxel->weight = newWeight;
      //free voxel
      atomicExch(mutex, 0);
    }
    else{
      i++;
      if(i>500){
        printf("stuck\n");
      }
    }
  }
}

/*
* Get voxels traversed on ray between origin and a lidar point within truncation distance of lidar point
*/
__global__
void getVoxelsForPoint(pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, int * size_d){
  int threadIndex = (blockIdx.x*threadsPerCudaBlock + threadIdx.x);
  // if(threadIndex>0){
  //   return;
  // }
  if(threadIndex>=*size_d){
    return;
  }
  pcl::PointXYZ point_d = points_d[threadIndex];
  Vector3f truncation_start;
  Vector3f truncation_end;


  getTruncationLineEndPoints(point_d, origin_transformed_d, truncation_start, truncation_end);
  //equation of line is u+tv
  // truncation_start(0) = 127.499992;
  // truncation_start(1) = 8.746421;
  // truncation_start(2) = -2.210701;
  // truncation_end(0) = 129.193176;
  // truncation_end(1) = 15.148551;
  // truncation_end(2) = 2.277709;
  Vector3f u = truncation_start;
  Vector3f v = truncation_end - truncation_start;

  //get list of voxels traversed 
  Vector3f * voxels = new Vector3f[100]; //todo: hardcoded -> set in terms of truncation distance and voxel size
  int * size = new int(0);

  traverseVolume(truncation_start, truncation_end, VOXEL_SIZE, u, v, voxels, size);
  

  Vector3f * lidarPoint = new Vector3f(point_d.x, point_d.y, point_d.z);

  //update the voxels sdf and weight values
  int numCudaBlocks = *size/threadsPerCudaBlock + 1;
  updateVoxels<<<numCudaBlocks, threadsPerCudaBlock>>>(voxels, hashTable_d, blockHeap_d, lidarPoint, origin_transformed_d, size);
  cdpErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  cudaFree(lidarPoint);
  cudaFree(voxels);
  cudaFree(size);
  return;
}

/*
* determines if voxel is within publishing distance of drone
*/
__device__
inline bool withinDistanceSquared(Vector3f & a, Vector3f & b, float thresholdDistanceSquared){
  Vector3f diff = b - a;
  float distanceSquared  = pow(diff(0), 2) + pow(diff(1), 2) + pow(diff(2), 2);
  if(distanceSquared <= thresholdDistanceSquared){
    return true;
  }
  return false;
}

/*
* Get voxels that are occupied in a voxel block
*/
__global__
void processOccupiedVoxelBlock(Vector3f * occupiedVoxels, int * index, Voxel * sdfWeightVoxelVals_d, Vector3f * position, VoxelBlock * block){
  int threadIndex = blockIdx.x*threadsPerCudaBlock + threadIdx.x;
  if(threadIndex >= VOXEL_PER_SIDE * VOXEL_PER_SIDE * VOXEL_PER_SIDE){
    return;
  }

  int voxelIndex = threadIndex;
  Voxel voxel = block->voxels[threadIndex];
  //if voxel is occupied
  if(voxel.weight!=0){
    //get coordinates of voxel
    float z = voxelIndex / (VOXEL_PER_SIDE * VOXEL_PER_SIDE);
    voxelIndex -= z*VOXEL_PER_SIDE*VOXEL_PER_SIDE;
    float y = voxelIndex / VOXEL_PER_SIDE;
    voxelIndex -= y*VOXEL_PER_SIDE;
    float x = voxelIndex;

    Vector3f positionVec = * position;
    float xCoord = x * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(0);
    float yCoord = y * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(1);
    float zCoord = z * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(2);
  
    Vector3f v(xCoord, yCoord, zCoord);
    //if within publish distance add voxel to list of occupied voxel and its position
    int occupiedVoxelIndex = atomicAdd(&(*index), 1);
    if(occupiedVoxelIndex<OCCUPIED_VOXELS_SIZE){
      occupiedVoxels[occupiedVoxelIndex] = v;
      sdfWeightVoxelVals_d[occupiedVoxelIndex] = voxel;
    }
  }
}

/*
* check hashtable in parallel if there is an allocated block at the thread index and if so process the block to retrieve occupied voxels
*/
__global__
void publishOccupiedVoxelsCuda(Vector3f * origin_transformed_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, Vector3f * occupiedVoxels, int * index, Voxel * sdfWeightVoxelVals_d){
  int threadIndex = blockIdx.x*threadsPerCudaBlock +threadIdx.x;
  if(threadIndex >= HASH_TABLE_SIZE) return;
  HashEntry hashEntry = hashTable_d->hashEntries[threadIndex];
  if(hashEntry.isFree()){
    return;
  }
  Vector3f blockPos = hashEntry.position;
  if(withinDistanceSquared(blockPos, *origin_transformed_d, PUBLISH_DISTANCE_SQUARED)){
    int pointer = hashEntry.pointer;
    Vector3f * bottomLeftBlockPos = new Vector3f(blockPos(0) - HALF_VOXEL_BLOCK_SIZE, 
    blockPos(1)- HALF_VOXEL_BLOCK_SIZE,
    blockPos(2)- HALF_VOXEL_BLOCK_SIZE);
  
    VoxelBlock * block = &(blockHeap_d->blocks[pointer]);
    int size = VOXEL_PER_SIDE * VOXEL_PER_SIDE * VOXEL_PER_SIDE;
    int numBlocks = size/threadsPerCudaBlock + 1;
    processOccupiedVoxelBlock<<<numBlocks,threadsPerCudaBlock>>>(occupiedVoxels, index, sdfWeightVoxelVals_d, bottomLeftBlockPos, block);
    cdpErrchk(cudaPeekAtLastError());
    cudaFree(bottomLeftBlockPos);
  }
}

/*
* Remove blocks further than garbage collection distance away from lidar point
*/
__global__
void garbageCollect(Vector3f * origin_transformed_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, int * removedBlocksCounts, Vector3f * linkedListDistantBlocks, int * linkedListDistantBlocksCount){
  int threadIndex = blockIdx.x*threadsPerCudaBlock +threadIdx.x;
  if(threadIndex >= HASH_TABLE_SIZE) return;
  HashEntry * hashEntry = &hashTable_d->hashEntries[threadIndex];
  if(hashEntry->isFree()){
    return;
  }
  Vector3f blockPosition = hashEntry->position;
  //check if point is far away from lidar to remove
  if(!withinDistanceSquared(blockPosition, *origin_transformed_d, GARBAGE_COLLECT_DISTANCE_SQUARED))
  {
    size_t hashedBucketIndex = retrieveHashIndexFromPoint(blockPosition);
    size_t threadBucketIndex = threadIndex / HASH_ENTRIES_PER_BUCKET;
    if((hashedBucketIndex!=threadBucketIndex) || (hashEntry->offset > 0)){ // if hash entry is in linked list process later
      int index = atomicAdd(&(*linkedListDistantBlocksCount), 1);
      if(index<MAX_LINKED_LIST_BLOCKS){
        linkedListDistantBlocks[index] = blockPosition;
      }
    }
    else{
      hashEntry->setFree();
      int blockHeapPosition = hashEntry->pointer;
      int freeBlocksInsertIndex = atomicSub(&(blockHeap_d->currentIndex), 1);
      blockHeap_d->freeBlocks[freeBlocksInsertIndex-1] = blockHeapPosition;
      atomicAdd(&(*removedBlocksCounts), 1);
    }
  }
  hashEntry = NULL;
  delete hashEntry;
}

/*
* Remove blocks further than garbage collection distance away from lidar point whose hash entries are a part of a linked list
*/
__global__
void linkedListGarbageCollect(HashTable * hashTable_d, BlockHeap * blockHeap_d, Vector3f * linkedListDistantBlocks, int * linkedListDistantBlocksCount){

  HashEntry * hashEntries = hashTable_d->hashEntries;
  HashEntry * currHashEntry = NULL;
  HashEntry * prevHashEntry = NULL;
  HashEntry * nextHashEntry = NULL;

  for(int i=0;i<*linkedListDistantBlocksCount; ++i){ //loop through linked list points and process sequentially so no issues with having to lock other buckets
    Vector3f removeBlockPosition = linkedListDistantBlocks[i];
    int hashedBucketIndex = retrieveHashIndexFromPoint(removeBlockPosition);
    //initialize to head of linked list
    int currIndex = (hashedBucketIndex + 1) * HASH_ENTRIES_PER_BUCKET - 1;
    currHashEntry = &hashEntries[currIndex];
    prevHashEntry = NULL;

    if(checkFloatingPointVectorsEqual(currHashEntry->position, removeBlockPosition, BLOCK_EPSILON)){ //if the hash entry is the head of the linked list
      int prevHeadOffset = currHashEntry->offset;
      int blockHeapPosition = currHashEntry->pointer;
      int nextIndex = currIndex + prevHeadOffset;
      if(nextIndex >= HASH_TABLE_SIZE){
        nextIndex %= HASH_TABLE_SIZE;
      }
      nextHashEntry = &hashEntries[nextIndex];

      Vector3f nextHashEntryPosition = nextHashEntry->position;

      int nextHashEntryOffset = nextHashEntry->offset;
      int nextHashEntryPointer = nextHashEntry->pointer;
      currHashEntry->position = nextHashEntryPosition;
      currHashEntry->pointer = nextHashEntryPointer;
      if(nextHashEntryOffset!=0){
        int newOffset = prevHeadOffset + nextHashEntryOffset;
        currHashEntry->offset = newOffset;
      }
      else{
        currHashEntry->offset = 0;
      }
      nextHashEntry->setFree();
      int freeBlocksInsertIndex = atomicSub(&(blockHeap_d->currentIndex), 1);
      blockHeap_d->freeBlocks[freeBlocksInsertIndex-1] = blockHeapPosition;
      continue;
    }

    while(!checkFloatingPointVectorsEqual(currHashEntry->position, removeBlockPosition, BLOCK_EPSILON)){ //hash entry is a middle or end element of linked list
      currIndex += (int) currHashEntry->offset;
      if(currIndex >= HASH_TABLE_SIZE){
        currIndex %= HASH_TABLE_SIZE;
      }
      prevHashEntry = currHashEntry;
      currHashEntry = &hashEntries[currIndex];
    }

    int currOffset = currHashEntry->offset;
    int prevHashOffset = prevHashEntry->offset;
    int newOffset = prevHashOffset + currOffset;

    if(currOffset > 0){
      prevHashEntry->offset = newOffset;
      currHashEntry->setFree();
      int blockHeapPosition = currHashEntry->pointer;
      int freeBlocksInsertIndex = atomicSub(&(blockHeap_d->currentIndex), 1);
      blockHeap_d->freeBlocks[freeBlocksInsertIndex-1] = blockHeapPosition;
    }
    else{
      prevHashEntry->offset = 0;
      currHashEntry->setFree();
      int blockHeapPosition = currHashEntry->pointer;
      int freeBlocksInsertIndex = atomicSub(&(blockHeap_d->currentIndex), 1);
      blockHeap_d->freeBlocks[freeBlocksInsertIndex-1] = blockHeapPosition;
    }
  }
  currHashEntry = prevHashEntry = nextHashEntry = hashEntries = NULL;
  delete(currHashEntry);
  delete(prevHashEntry);
  delete(nextHashEntry);
  delete(hashEntries);
}

TSDFHandler::TSDFHandler(){
  tsdfContainer = new TSDFContainer();
}

TSDFHandler::~TSDFHandler(){
  free(tsdfContainer);
}

/*
* given a lidar pointcloud xyz and origin allocate voxel blocks and update voxels within truncation distance of lidar points and return occupied voxels for visualization/publishing
*/
void TSDFHandler::processPointCloudAndUpdateVoxels(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, Vector3f * origin_transformed_h, Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h)
{ 
  std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points = pointcloud->points;

  pcl::PointXYZ * points_h = &points[0];
  pcl::PointXYZ * points_d;
  int pointcloud_size = pointcloud->size();
  int * pointcloud_size_d;

  cudaMalloc(&points_d, sizeof(*points_h)*pointcloud_size);
  cudaMemcpy(points_d, points_h, sizeof(*points_h)*pointcloud_size, cudaMemcpyHostToDevice);
  cudaMalloc(&pointcloud_size_d, sizeof(int));
  cudaMemcpy(pointcloud_size_d, &pointcloud_size, sizeof(int), cudaMemcpyHostToDevice);

  Vector3f * origin_transformed_d;
  cudaMalloc(&origin_transformed_d, sizeof(*origin_transformed_h));
  cudaMemcpy(origin_transformed_d, origin_transformed_h,sizeof(*origin_transformed_h),cudaMemcpyHostToDevice);

  HashTable * hash_table_d = tsdfContainer->getCudaHashTable();
  BlockHeap * block_heap_d = tsdfContainer->getCudaBlockHeap();

  allocateVoxelBlocksAndUpdateVoxels(points_d, origin_transformed_d, pointcloud_size_d, pointcloud_size, hash_table_d, block_heap_d);

  publishOccupiedVoxels(origin_transformed_d, occupied_voxels_h, occupied_voxels_index, sdfWeightVoxelVals_h, hash_table_d, block_heap_d);

  garbageCollectDistantBlocks(origin_transformed_d, hash_table_d, block_heap_d);

  cudaFree(pointcloud_size_d);
  cudaFree(points_d);
  cudaFree(origin_transformed_d);

}

/*
* allocate voxel blocks and voxels within truncation distance of lidar points and update the voxels sdf and weight
*/
void TSDFHandler::allocateVoxelBlocksAndUpdateVoxels(pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, int * pointcloud_size_d, int pointcloud_size, HashTable * hash_table_d, BlockHeap * block_heap_d){
    //TODO: FIX
  // int maxBlocksPerPoint = ceil(pow(TRUNCATION_DISTANCE,3) / pow(VOXEL_BLOCK_SIZE, 3));
  //the max number of voxel blocks that are allocated per point cloud frame
  int maxBlocks = 100 * pointcloud_size; //todo: hardcoded
  Vector3f pointcloud_voxel_blocks_h[maxBlocks];
  Vector3f * pointcloud_voxel_blocks_d;
  int * pointcloud_voxel_blocks_h_index = new int(0); //keep track of number of voxel blocks allocated
  int * pointcloud_voxel_blocks_d_index;
  cudaMalloc(&pointcloud_voxel_blocks_d, sizeof(*pointcloud_voxel_blocks_h)*maxBlocks);
  cudaMemcpy(pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_h, sizeof(*pointcloud_voxel_blocks_h)*maxBlocks,cudaMemcpyHostToDevice); //need to memcpy?
  cudaMalloc(&pointcloud_voxel_blocks_d_index, sizeof(*pointcloud_voxel_blocks_h_index));
  cudaMemcpy(pointcloud_voxel_blocks_d_index, pointcloud_voxel_blocks_h_index, sizeof(*pointcloud_voxel_blocks_h_index), cudaMemcpyHostToDevice);

  int num_cuda_blocks = pointcloud_size / threadsPerCudaBlock + 1;

  getVoxelBlocks(num_cuda_blocks, points_d, pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, origin_transformed_d, pointcloud_size_d);

  integrateVoxelBlockPointsIntoHashTable(pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, hash_table_d, block_heap_d);

  updateVoxels(num_cuda_blocks, points_d, origin_transformed_d, pointcloud_size_d, hash_table_d, block_heap_d);

  cudaFree(pointcloud_voxel_blocks_d);
  cudaFree(pointcloud_voxel_blocks_d_index);
  free(pointcloud_voxel_blocks_h_index);

}

/*
* Get voxel blocks in truncation distance of lidar points on ray between origin and the lidar points
*/
void TSDFHandler::getVoxelBlocks(int num_cuda_blocks, pcl::PointXYZ * points_d, Vector3f * pointcloud_voxel_blocks_d, int * pointcloud_voxel_blocks_d_index, Vector3f * origin_transformed_d, int * pointcloud_size_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  getVoxelBlocksForPoint<<<num_cuda_blocks,threadsPerCudaBlock>>>(points_d, pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, origin_transformed_d, pointcloud_size_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Get Voxel Block Duration: %f\n", milliseconds);
}

/*
* Allocated necessary voxel blocks to the hash table
*/
void TSDFHandler::integrateVoxelBlockPointsIntoHashTable(Vector3f * points_d, int * pointcloud_voxel_blocks_d_index, HashTable * hash_table_d, BlockHeap * block_heap_d){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int * size_h = new int(0);
  cudaMemcpy(size_h, pointcloud_voxel_blocks_d_index, sizeof(int), cudaMemcpyDeviceToHost);
  int size = * size_h;

  //array to keep track of unallocated blocks so in multiple kernel calls further down can keep track of blocks that still need to be allocated
  bool * unallocatedPoints_h = new bool[size];
  for(int i=0;i<size;++i)
  {
    unallocatedPoints_h[i] = 1;
  }

  bool * unallocatedPoints_d;
  cudaMalloc(&unallocatedPoints_d, sizeof(*unallocatedPoints_h)*size);
  cudaMemcpy(unallocatedPoints_d, unallocatedPoints_h, sizeof(*unallocatedPoints_h)*size, cudaMemcpyHostToDevice);

  //keep track of number of unallocated blocks still left to be allocated
  int * unallocatedPointsCount_h = new int(size);
  int * unallocatedPointsCount_d;
  cudaMalloc(&unallocatedPointsCount_d, sizeof(*unallocatedPointsCount_h));
  cudaMemcpy(unallocatedPointsCount_d, unallocatedPointsCount_h, sizeof(*unallocatedPointsCount_h), cudaMemcpyHostToDevice);

  int num_cuda_blocks = size / threadsPerCudaBlock + 1;
  //call kernel till all blocks are allocated
  while(*unallocatedPointsCount_h > 0){ //POSSIBILITY OF INFINITE LOOP if no applicable space is left for an unallocated block even if there is still space left in hash table
    allocateVoxelBlocks<<<num_cuda_blocks,threadsPerCudaBlock>>>(points_d, hash_table_d, block_heap_d, unallocatedPoints_d, pointcloud_voxel_blocks_d_index, unallocatedPointsCount_d);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    cudaMemcpy(unallocatedPointsCount_h, unallocatedPointsCount_d, sizeof(*unallocatedPointsCount_h), cudaMemcpyDeviceToHost);
  }

  //print total num blocks allocated so far
  printHashTableAndBlockHeap<<<1,1>>>(hash_table_d, block_heap_d);
  cudaDeviceSynchronize();

  cudaFree(unallocatedPoints_d);
  cudaFree(unallocatedPointsCount_d);
  delete size_h;
  delete unallocatedPoints_h;
  delete unallocatedPointsCount_h;

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Integrate Voxel Block Duration: %f\n", milliseconds);
}

/*
* update voxels sdf and weight values
*/
void TSDFHandler::updateVoxels(int & num_cuda_blocks, pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, int * pointcloud_size_d, HashTable * hash_table_d, BlockHeap * block_heap_d){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  getVoxelsForPoint<<<num_cuda_blocks,threadsPerCudaBlock>>>(points_d, origin_transformed_d, hash_table_d, block_heap_d, pointcloud_size_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Update Voxels Duration: %f\n", milliseconds);
}

/*
* get occupied voxels for visualization and publishing 
*/
void TSDFHandler::publishOccupiedVoxels(Vector3f * origin_transformed_d, Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h, HashTable * hash_table_d, BlockHeap * block_heap_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  Vector3f * occupied_voxels_d;
  int * occupied_voxels_index_d;
  Voxel * sdfWeightVoxelVals_d;
  int occupiedVoxelsSize = OCCUPIED_VOXELS_SIZE;
  cudaMalloc(&occupied_voxels_d, sizeof(*occupied_voxels_h)*occupiedVoxelsSize);
  cudaMemcpy(occupied_voxels_d, occupied_voxels_h, sizeof(*occupied_voxels_h)*occupiedVoxelsSize,cudaMemcpyHostToDevice);
  cudaMalloc(&occupied_voxels_index_d, sizeof(*occupied_voxels_index));
  cudaMemcpy(occupied_voxels_index_d, occupied_voxels_index, sizeof(*occupied_voxels_index), cudaMemcpyHostToDevice);
  cudaMalloc(&sdfWeightVoxelVals_d, sizeof(*sdfWeightVoxelVals_h)*occupiedVoxelsSize);
  cudaMemcpy(sdfWeightVoxelVals_d, sdfWeightVoxelVals_h, sizeof(*sdfWeightVoxelVals_h)*occupiedVoxelsSize, cudaMemcpyHostToDevice);

  int numCudaBlocks = HASH_TABLE_SIZE / threadsPerCudaBlock + 1;
  publishOccupiedVoxelsCuda<<<numCudaBlocks,threadsPerCudaBlock>>>(origin_transformed_d, hash_table_d, block_heap_d, occupied_voxels_d, occupied_voxels_index_d, sdfWeightVoxelVals_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  cudaMemcpy(occupied_voxels_h, occupied_voxels_d, sizeof(*occupied_voxels_h)*occupiedVoxelsSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(occupied_voxels_index, occupied_voxels_index_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sdfWeightVoxelVals_h, sdfWeightVoxelVals_d, sizeof(*sdfWeightVoxelVals_h)*occupiedVoxelsSize, cudaMemcpyDeviceToHost);

  cudaFree(occupied_voxels_d); //instead of allocating and freeing over and over just add to tsdfhandler
  cudaFree(occupied_voxels_index_d);
  cudaFree(sdfWeightVoxelVals_d);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Publish Voxels Duration: %f\n", milliseconds);

}

/*
* Remove blocks further than garbage collection distance so not keeping distant block data
*/
void TSDFHandler::garbageCollectDistantBlocks(Vector3f * origin_transformed_d, HashTable * hash_table_d, BlockHeap * block_heap_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int * removedBlocksCount_h = new int(0);
  int * removedBlocksCount_d;
  cudaMalloc(&removedBlocksCount_d, sizeof(int));
  cudaMemcpy(removedBlocksCount_d, removedBlocksCount_h, sizeof(* removedBlocksCount_h), cudaMemcpyHostToDevice);

  int * linkedListDistantBlocksCount_h = new int(0);
  int * linkedListDistantBlocksCount_d;
  cudaMalloc(&linkedListDistantBlocksCount_d, sizeof(int));
  cudaMemcpy(linkedListDistantBlocksCount_d, linkedListDistantBlocksCount_h, sizeof(* linkedListDistantBlocksCount_h), cudaMemcpyHostToDevice);

  Vector3f linkedListDistantBlocks_h[MAX_LINKED_LIST_BLOCKS];
  Vector3f * linkedListDistantBlocks_d;
  cudaMalloc(&linkedListDistantBlocks_d, sizeof(*linkedListDistantBlocks_h)*MAX_LINKED_LIST_BLOCKS);
  cudaMemcpy(linkedListDistantBlocks_d, linkedListDistantBlocks_h, sizeof(*linkedListDistantBlocks_h)*MAX_LINKED_LIST_BLOCKS, cudaMemcpyHostToDevice);

  int numCudaBlocks = HASH_TABLE_SIZE / threadsPerCudaBlock + 1;
  garbageCollect<<<numCudaBlocks, threadsPerCudaBlock>>>(origin_transformed_d, hash_table_d, block_heap_d, removedBlocksCount_d, linkedListDistantBlocks_d, linkedListDistantBlocksCount_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(removedBlocksCount_h, removedBlocksCount_d, sizeof(* removedBlocksCount_h), cudaMemcpyDeviceToHost);
  cudaMemcpy(linkedListDistantBlocksCount_h, linkedListDistantBlocksCount_d, sizeof(* linkedListDistantBlocksCount_h), cudaMemcpyDeviceToHost);

  linkedListGarbageCollect<<<1,1>>>(hash_table_d, block_heap_d, linkedListDistantBlocks_d, linkedListDistantBlocksCount_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(linkedListDistantBlocksCount_h, linkedListDistantBlocksCount_d, sizeof(* linkedListDistantBlocksCount_h), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Garbage Collection Duration: %f\n", milliseconds);
  // printf("Linked List Blocks Removed: %d\n", *linkedListDistantBlocksCount_h);
  printf("Total Blocks Removed: %d\n", *removedBlocksCount_h + *linkedListDistantBlocksCount_h);

  delete removedBlocksCount_h;
  delete linkedListDistantBlocksCount_h;
  cudaFree(removedBlocksCount_d);
  cudaFree(linkedListDistantBlocksCount_d);
  cudaFree(linkedListDistantBlocks_d);
}

/*
* Initialize device global variables
*/
__global__
void initializeGlobalVarsCuda(float * voxel_size_d, float * truncation_distance_d, float * max_weight_d, float * publish_distance_squared_d, float * garbage_collect_distance_squared_d){
  VOXEL_SIZE = * voxel_size_d;
  HALF_VOXEL_SIZE = VOXEL_SIZE / 2;
  VOXEL_BLOCK_SIZE = VOXEL_SIZE * VOXEL_PER_SIDE;
  HALF_VOXEL_BLOCK_SIZE = VOXEL_BLOCK_SIZE / 2;
  BLOCK_EPSILON = VOXEL_BLOCK_SIZE / 4;
  VOXEL_EPSILON = VOXEL_SIZE / 4; 
  TRUNCATION_DISTANCE = * truncation_distance_d;
  MAX_WEIGHT = * max_weight_d;
  PUBLISH_DISTANCE_SQUARED = * publish_distance_squared_d;
  GARBAGE_COLLECT_DISTANCE_SQUARED = * garbage_collect_distance_squared_d;
}

/*
* Initialize device global variables
*/
void initializeGlobalVars(Params params){
  initializeGlobalVarsCuda<<<1,1>>>(params.voxel_size_param_d, params.truncation_distance_param_d, params.max_weight_param_d, 
    params.publish_distance_squared_param_d, params.garbage_collect_distance_squared_param_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
}