#include "cuda/tsdf_handler.cuh"

const int threadsPerCudaBlock = 128;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

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
  printf("Current Index: %d\n", blockHeap_d->currentIndex);
}

__device__
size_t retrieveHashIndexFromPoint(Vector3f point){ //tested using int can get negatives
  return abs((((int)point(0)*PRIME_ONE) ^ ((int)point(1)*PRIME_TWO) ^ ((int)point(2)*PRIME_THREE)) % NUM_BUCKETS);
}

__device__ 
float FloorFun(float x, float scale){
  return floor(x*scale) / scale;
}

__device__
Vector3f GetVoxelBlockCenterFromPoint(Vector3f point){
  float scale = 1 / VOXEL_BLOCK_SIZE;
  Vector3f blockCenter;
  blockCenter(0) = FloorFun(point(0), scale) + HALF_VOXEL_BLOCK_SIZE;
  blockCenter(1) = FloorFun(point(1), scale) + HALF_VOXEL_BLOCK_SIZE;
  blockCenter(2) = FloorFun(point(2), scale) + HALF_VOXEL_BLOCK_SIZE;
  return blockCenter;
}

__device__
Vector3f GetVoxelCenterFromPoint(Vector3f point){
  float scale = 1 / VOXEL_SIZE;
  Vector3f voxelCenter;
  voxelCenter(0) = FloorFun(point(0), scale) + HALF_VOXEL_SIZE;
  voxelCenter(1) = FloorFun(point(1), scale) + HALF_VOXEL_SIZE;
  voxelCenter(2) = FloorFun(point(2), scale) + HALF_VOXEL_SIZE;
  return voxelCenter;
}

__device__
bool checkFloatingPointVectorsEqual(Vector3f A, Vector3f B, float epsilon){
  Vector3f diff = A-B;
  //have to use an epsilon value due to floating point precision errors
  if((fabs(diff(0)) < epsilon) && (fabs(diff(1)) < epsilon) && (fabs(diff(2)) < epsilon))
    return true;

  return false;
}

__device__
inline void getTruncationLineEndPoints(pcl::PointXYZ & point_d, Vector3f * origin_transformed_d, Vector3f & truncation_start, Vector3f & truncation_end, Vector3f & u, Vector3f & v){
  u = *origin_transformed_d;
  Vector3f point_d_vector(point_d.x, point_d.y, point_d.z);
  v = point_d_vector - u; //direction
  //equation of line is u+tv
  float vMag = sqrt(pow(v(0), 2) + pow(v(1),2) + pow(v(2), 2));
  Vector3f v_normalized = v / vMag;
  truncation_start = point_d_vector - truncation_distance*v_normalized;
  
  truncation_end = point_d_vector + truncation_distance*v_normalized;

  float distance_tStart_origin = pow(truncation_start(0) - u(0), 2) + pow(truncation_start(1) - u(1),2) + pow(truncation_start(2) - u(2), 2);
  float distance_tEnd_origin = pow(truncation_end(0) - u(0), 2) + pow(truncation_end(1) - u(1),2) + pow(truncation_end(2) - u(2), 2);

  if(distance_tEnd_origin < distance_tStart_origin){
    Vector3f temp = truncation_start;
    truncation_start = truncation_end;
    truncation_end = temp;
  }
}

__device__
inline void traverseVolume(Vector3f & truncation_start_vol, Vector3f & truncation_end_vol, const float & volume_size, Vector3f & u, Vector3f & v, Vector3f * traversed_vols, int * traversed_vols_size){
  float half_volume_size = volume_size / 2;
  const float epsilon = volume_size / 4;
  float stepX = v(0) > 0 ? volume_size : -1 * volume_size;
  float stepY = v(1) > 0 ? volume_size : -1 * volume_size;
  float stepZ = v(2) > 0 ? volume_size : -1 * volume_size;
  float tMaxX = fabs(v(0) < 0 ? (truncation_start_vol(0) - half_volume_size - u(0)) / v(0) : (truncation_start_vol(0) + half_volume_size - u(0)) / v(0));
  float tMaxY = fabs(v(1) < 0 ? (truncation_start_vol(1) - half_volume_size - u(1)) / v(1) : (truncation_start_vol(1) + half_volume_size - u(1)) / v(1));
  float tMaxZ = fabs(v(2) < 0 ? (truncation_start_vol(2) - half_volume_size - u(2)) / v(2) : (truncation_start_vol(2) + half_volume_size - u(2)) / v(2));
  float tDeltaX = fabs(volume_size / v(0));
  float tDeltaY = fabs(volume_size / v(1));
  float tDeltaZ = fabs(volume_size / v(2));
  Vector3f current_vol(truncation_start_vol(0), truncation_start_vol(1), truncation_start_vol(2));

  int insert_index;
  while(!checkFloatingPointVectorsEqual(current_vol, truncation_end_vol, epsilon)){
    insert_index = atomicAdd(&(*traversed_vols_size), 1);
    traversed_vols[insert_index] = current_vol;
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
  }      

  insert_index = atomicAdd(&(*traversed_vols_size), 1);
  traversed_vols[insert_index] = current_vol;
}

__global__
void getVoxelBlocksForPoint(pcl::PointXYZ * points_d, Vector3f * pointCloudVoxelBlocks_d, int * pointer_d, Vector3f * origin_transformed_d, int * size_d){ //refactor
  int threadIndex = (blockIdx.x*threadsPerCudaBlock + threadIdx.x);
  if(threadIndex>=*size_d){
    return;
  }
  pcl::PointXYZ point_d = points_d[threadIndex];
  Vector3f truncation_start;
  Vector3f truncation_end;
  Vector3f u;
  Vector3f v;

  getTruncationLineEndPoints(point_d, origin_transformed_d, truncation_start, truncation_end, u, v);

  Vector3f truncation_start_block = GetVoxelBlockCenterFromPoint(truncation_start);
  Vector3f truncation_end_block = GetVoxelBlockCenterFromPoint(truncation_end);

  traverseVolume(truncation_start_block, truncation_end_block, VOXEL_BLOCK_SIZE, u, v, pointCloudVoxelBlocks_d, pointer_d);
  return;
}

__device__ 
int getBlockPositionForBlockCoordinates(Vector3f & voxelBlockCoordinates, size_t & bucket_index, size_t & currentGlobalIndex, HashEntry * hashEntries){

  HashEntry hashEntry;

  //check the hashed bucket for the block
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hashEntry = hashEntries[currentGlobalIndex+i];
    if(hashEntry.checkIsPositionEqual(voxelBlockCoordinates)){
      return hashEntry.pointer;
    }
  }

  currentGlobalIndex+=HASH_ENTRIES_PER_BUCKET-1;

  //check the linked list if necessary
  while(hashEntry.offset!=0){
    short offset = hashEntry.offset;
    currentGlobalIndex+=offset;
    if(currentGlobalIndex>=HASH_TABLE_SIZE){
      currentGlobalIndex %= HASH_TABLE_SIZE;
    }
    hashEntry = hashEntries[currentGlobalIndex];
    if(hashEntry.checkIsPositionEqual(voxelBlockCoordinates)){ 
      return hashEntry.pointer;
    }
  }

  //block not allocated in hashTable
  return -1;
}

__device__
inline bool attemptHashedBucketVoxelBlockCreation(size_t & hashedBucketIndex, BlockHeap * blockHeap_d, Vector3f & point_d, HashEntry * hashEntries){
  size_t insertCurrentGlobalIndex = hashedBucketIndex * HASH_ENTRIES_PER_BUCKET;
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    if(hashEntries[insertCurrentGlobalIndex+i].isFree()){ 
      int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
      VoxelBlock * allocBlock = new VoxelBlock();
      HashEntry * allocBlockHashEntry = new HashEntry(point_d, blockHeapFreeIndex);
      blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
      hashEntries[insertCurrentGlobalIndex+i] = *allocBlockHashEntry;
      cudaFree(allocBlock);
      cudaFree(allocBlockHashEntry);
      return true;
    }
  }
  return false;
} 

__device__
inline bool attemptLinkedListVoxelBlockCreation(size_t & hashedBucketIndex, BlockHeap * blockHeap_d, HashTable * hashTable_d, size_t & insertBucketIndex, size_t & endLinkedListPointer, Vector3f & point_d, HashEntry * hashEntries){
  size_t insertCurrentGlobalIndex;
  while(insertBucketIndex!=hashedBucketIndex){
    if(!atomicCAS(&hashTable_d->mutex[insertBucketIndex], 0, 1)){
      insertCurrentGlobalIndex = insertBucketIndex * HASH_ENTRIES_PER_BUCKET;
      for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET-1; ++i){
        if(hashEntries[insertCurrentGlobalIndex+i].isFree() ){ 
            int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
            VoxelBlock * allocBlock = new VoxelBlock();
            HashEntry * allocBlockHashEntry = new HashEntry(point_d, blockHeapFreeIndex);
            blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
            size_t insertPos = insertCurrentGlobalIndex + i;
            hashEntries[insertPos] = *allocBlockHashEntry;
            cudaFree(allocBlock);
            cudaFree(allocBlockHashEntry);
            if(insertPos > endLinkedListPointer){
              hashEntries[endLinkedListPointer].offset = insertPos - endLinkedListPointer;
            }
            else{
              hashEntries[endLinkedListPointer].offset = HASH_TABLE_SIZE - endLinkedListPointer + insertPos;
            }
            return true;
        }
      }
      atomicExch(&hashTable_d->mutex[insertBucketIndex], 0);
    }
    insertBucketIndex++;
    if(insertBucketIndex == NUM_BUCKETS){
      insertBucketIndex = 0;
    }
  }
  return false;
}
__global__
void allocateVoxelBlocks(Vector3f * points_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, bool * unallocatedPoints_d, int * size_d, int * unallocatedPointsCount_d) //clean up
{
  int threadIndex = (blockIdx.x*threadsPerCudaBlock + threadIdx.x);
  if(threadIndex>=*size_d || (unallocatedPoints_d[threadIndex]==0)){
    return;
  }
  Vector3f point_d = points_d[threadIndex];

  size_t hashedBucketIndex = retrieveHashIndexFromPoint(point_d);
  size_t currentGlobalIndex = hashedBucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hashEntries = hashTable_d->hashEntries;

  int block_position = getBlockPositionForBlockCoordinates(point_d, hashedBucketIndex, currentGlobalIndex, hashEntries);

  //block is already allocated
  if(block_position!=-1){
    unallocatedPoints_d[threadIndex] = 0;
    atomicSub(unallocatedPointsCount_d, 1);
    return;
  }

  //attempt to get lock for hashed bucket
  if(!atomicCAS(&hashTable_d->mutex[hashedBucketIndex], 0, 1)){

    if(attemptHashedBucketVoxelBlockCreation(hashedBucketIndex, blockHeap_d, point_d, hashEntries)) {
      unallocatedPoints_d[threadIndex] = 0;
      atomicSub(unallocatedPointsCount_d, 1);
      atomicExch(&hashTable_d->mutex[hashedBucketIndex], 0);
      return;
    }

    size_t insertBucketIndex = hashedBucketIndex + 1;
    if(insertBucketIndex == NUM_BUCKETS){
      insertBucketIndex = 0;
    }

    //current global index will point to end of linked list which includes hashed bucket if no linked list
    size_t endLinkedListBucket = currentGlobalIndex / HASH_ENTRIES_PER_BUCKET;

    bool haveEndLinkedListBucketLock = true;

    if(endLinkedListBucket!=hashedBucketIndex){
      atomicExch(&hashTable_d->mutex[hashedBucketIndex], 0);
      haveEndLinkedListBucketLock = !atomicCAS(&hashTable_d->mutex[endLinkedListBucket], 0, 1);
    }

    if(haveEndLinkedListBucketLock){
      if(attemptLinkedListVoxelBlockCreation(hashedBucketIndex, blockHeap_d, hashTable_d, insertBucketIndex, currentGlobalIndex, point_d, hashEntries)){
        unallocatedPoints_d[threadIndex] = 0;
        atomicSub(unallocatedPointsCount_d, 1); 
        atomicExch(&hashTable_d->mutex[endLinkedListBucket], 0);
        atomicExch(&hashTable_d->mutex[insertBucketIndex], 0);
      }
      else{
        atomicExch(&hashTable_d->mutex[endLinkedListBucket], 0);
      }
      return;
    }
  }
}

__device__
size_t getLocalVoxelIndex(Vector3f diff){
  diff /= VOXEL_SIZE;
  return floor(diff(0)) + (floor(diff(1)) * VOXEL_PER_BLOCK) + (floor(diff(2)) * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK);
}

__device__
float getMagnitude(Vector3f vector){
  return sqrt(pow(vector(0),2) + pow(vector(1),2) + pow(vector(2),2));
}

__device__
float dotProduct(Vector3f a, Vector3f b){
  return a(0)*b(0) + a(1)*b(1) + a(2)*b(2);
}


__device__
float getDistanceUpdate(Vector3f voxelCoordinates, Vector3f point_d, Vector3f origin){
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

__global__
void updateVoxels(Vector3f * voxels, HashTable * hashTable_d, BlockHeap * blockHeap_d, Vector3f * point_d, Vector3f * origin, int * size){
  int threadIndex = blockIdx.x*threadsPerCudaBlock + threadIdx.x;
  if(threadIndex >= * size){
    return;
  }
  Vector3f voxelCoordinates = voxels[threadIndex];
  Vector3f voxelBlockCoordinates = GetVoxelBlockCenterFromPoint(voxelCoordinates);

  size_t bucketIndex = retrieveHashIndexFromPoint(voxelBlockCoordinates);
  size_t currentGlobalIndex = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hashEntries = hashTable_d->hashEntries;

  //make a global vector with half voxel block size values
  int voxelBlockHeapPosition = getBlockPositionForBlockCoordinates(voxelBlockCoordinates, bucketIndex, currentGlobalIndex, hashEntries);
  Vector3f voxelBlockBottomLeftCoordinates;
  voxelBlockBottomLeftCoordinates(0) = voxelBlockCoordinates(0)-HALF_VOXEL_BLOCK_SIZE;
  voxelBlockBottomLeftCoordinates(1) = voxelBlockCoordinates(1)-HALF_VOXEL_BLOCK_SIZE;
  voxelBlockBottomLeftCoordinates(2) = voxelBlockCoordinates(2)-HALF_VOXEL_BLOCK_SIZE;
  size_t localVoxelIndex = getLocalVoxelIndex(voxelCoordinates - voxelBlockBottomLeftCoordinates);
  Vector3f point = * point_d;
  VoxelBlock * block = &(blockHeap_d->blocks[voxelBlockHeapPosition]);
  Voxel* voxel = &(block->voxels[localVoxelIndex]);
  int * mutex = &(block->mutex[localVoxelIndex]);

  float weight = 1;
  float distance = getDistanceUpdate(voxelCoordinates, *point_d, *origin);
  float weightTimesDistance = weight * distance;

  //get lock for voxel
  bool updatedVoxel = false;
  while(!updatedVoxel){
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
      // printf("voxel coords: (%f, %f, %f) with sdf: %f with weight: %f\n", voxelCoordinates(0),voxelCoordinates(1), voxelCoordinates(2), voxel->sdf, voxel->weight);
      atomicExch(mutex, 0);
    }
  }
}

__global__
void getVoxelsForPoint(pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, int * size_d){
  int threadIndex = (blockIdx.x*threadsPerCudaBlock + threadIdx.x);
  if(threadIndex>=*size_d){
  return;
  }
  pcl::PointXYZ point_d = points_d[threadIndex];
  Vector3f truncation_start;
  Vector3f truncation_end;
  Vector3f u;
  Vector3f v;

  getTruncationLineEndPoints(point_d, origin_transformed_d, truncation_start, truncation_end, u, v);

  Vector3f truncation_start_voxel = GetVoxelCenterFromPoint(truncation_start);
  Vector3f truncation_end_voxel = GetVoxelCenterFromPoint(truncation_end);

  Vector3f * voxels = new Vector3f[200]; //set in terms of truncation distance and voxel size
  int * size = new int(0);

  traverseVolume(truncation_start_voxel, truncation_end_voxel, VOXEL_SIZE, u, v, voxels, size);

  Vector3f * lidarPoint = new Vector3f(point_d.x, point_d.y, point_d.z);

  int numCudaBlocks = *size/threadsPerCudaBlock + 1;
  updateVoxels<<<numCudaBlocks, threadsPerCudaBlock>>>(voxels, hashTable_d, blockHeap_d, lidarPoint, origin_transformed_d, size);
  cdpErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  cudaFree(lidarPoint);
  cudaFree(voxels);
  cudaFree(size);
  return;
}

__global__
void processOccupiedVoxelBlock(Vector3f * occupiedVoxels, int * index, Voxel * sdfWeightVoxelVals_d, Vector3f * position, VoxelBlock * block){
  int threadIndex = blockIdx.x*threadsPerCudaBlock + threadIdx.x;
  if(threadIndex >= VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK){
    return;
  }

  int voxelIndex = threadIndex;
  Voxel voxel = block->voxels[threadIndex];
  if(voxel.weight!=0){
    float z = voxelIndex / (VOXEL_PER_BLOCK * VOXEL_PER_BLOCK);
    voxelIndex -= z*VOXEL_PER_BLOCK*VOXEL_PER_BLOCK;
    float y = voxelIndex / VOXEL_PER_BLOCK;
    voxelIndex -= y*VOXEL_PER_BLOCK;
    float x = voxelIndex;

    Vector3f positionVec = * position;
    float xCoord = x * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(0);
    float yCoord = y * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(1);
    float zCoord = z * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(2);
  
    Vector3f v(xCoord, yCoord, zCoord);
    int occupiedVoxelIndex = atomicAdd(&(*index), 1);
    if(occupiedVoxelIndex<OCCUPIED_VOXELS_SIZE){
      occupiedVoxels[occupiedVoxelIndex] = v;
      sdfWeightVoxelVals_d[occupiedVoxelIndex] = voxel;
    }
  }
}

__global__
void visualizeOccupiedVoxels(HashTable * hashTable_d, BlockHeap * blockHeap_d, Vector3f * occupiedVoxels, int * index, Voxel * sdfWeightVoxelVals_d){
  int threadIndex = blockIdx.x*threadsPerCudaBlock +threadIdx.x;
  if(threadIndex >= HASH_TABLE_SIZE) return;
  HashEntry hashEntry = hashTable_d->hashEntries[threadIndex];
  if(hashEntry.isFree()){
    return;
  }
  int pointer = hashEntry.pointer;
  Vector3f * position = new Vector3f(hashEntry.position(0) - HALF_VOXEL_BLOCK_SIZE, 
  hashEntry.position(1)- HALF_VOXEL_BLOCK_SIZE,
  hashEntry.position(2)- HALF_VOXEL_BLOCK_SIZE);

  VoxelBlock * block = &(blockHeap_d->blocks[pointer]);
  int size = VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK;
  int numBlocks = size/threadsPerCudaBlock + 1;
  processOccupiedVoxelBlock<<<numBlocks,threadsPerCudaBlock>>>(occupiedVoxels, index, sdfWeightVoxelVals_d, position, block);
  cdpErrchk(cudaPeekAtLastError());
  cudaFree(position);
}

TSDFHandler::TSDFHandler(){
  tsdfContainer = new TSDFContainer();
}

TSDFHandler::~TSDFHandler(){
  free(tsdfContainer);
}

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

  visualize(occupied_voxels_h, occupied_voxels_index, sdfWeightVoxelVals_h, hash_table_d, block_heap_d);

  cudaFree(pointcloud_size_d);
  cudaFree(points_d);
  cudaFree(origin_transformed_d);

}

void TSDFHandler::allocateVoxelBlocksAndUpdateVoxels(pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, int * pointcloud_size_d, int pointcloud_size, HashTable * hash_table_d, BlockHeap * block_heap_d){
    //TODO: FIX
  // int maxBlocksPerPoint = ceil(pow(truncation_distance,3) / pow(VOXEL_BLOCK_SIZE, 3));
  int maxBlocks = 10 * pointcloud_size;
  Vector3f pointcloud_voxel_blocks_h[maxBlocks];
  Vector3f * pointcloud_voxel_blocks_d;
  int * pointcloud_voxel_blocks_h_index = new int(0);
  int * pointcloud_voxel_blocks_d_index;
  cudaMalloc(&pointcloud_voxel_blocks_d, sizeof(*pointcloud_voxel_blocks_h)*maxBlocks);
  cudaMemcpy(pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_h, sizeof(*pointcloud_voxel_blocks_h)*maxBlocks,cudaMemcpyHostToDevice); //do I even need to memcpy
  cudaMalloc(&pointcloud_voxel_blocks_d_index, sizeof(*pointcloud_voxel_blocks_h_index));
  cudaMemcpy(pointcloud_voxel_blocks_d_index, pointcloud_voxel_blocks_h_index, sizeof(*pointcloud_voxel_blocks_h_index), cudaMemcpyHostToDevice);

    //since size can go over threads per block allocate this properly to include all data
  int num_cuda_blocks = pointcloud_size / threadsPerCudaBlock + 1;

  getVoxelBlocks(num_cuda_blocks, points_d, pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, origin_transformed_d, pointcloud_size_d);

  integrateVoxelBlockPointsIntoHashTable(pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, hash_table_d, block_heap_d);

  updateVoxels(num_cuda_blocks, points_d, origin_transformed_d, pointcloud_size_d, hash_table_d, block_heap_d);

  cudaFree(pointcloud_voxel_blocks_d);
  cudaFree(pointcloud_voxel_blocks_d_index);
  free(pointcloud_voxel_blocks_h_index);

}

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

void TSDFHandler::integrateVoxelBlockPointsIntoHashTable(Vector3f * points_d, int * pointcloud_voxel_blocks_d_index, HashTable * hash_table_d, BlockHeap * block_heap_d){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int * size_h = new int(0);
  cudaMemcpy(size_h, pointcloud_voxel_blocks_d_index, sizeof(int), cudaMemcpyDeviceToHost);
  int size = * size_h;

  bool * unallocatedPoints_h = new bool[size];
  for(int i=0;i<size;++i)
  {
    unallocatedPoints_h[i] = 1;
  }
  bool * unallocatedPoints_d;
  cudaMalloc(&unallocatedPoints_d, sizeof(*unallocatedPoints_h)*size);
  cudaMemcpy(unallocatedPoints_d, unallocatedPoints_h, sizeof(*unallocatedPoints_h)*size, cudaMemcpyHostToDevice);

  int * unallocatedPointsCount_h = new int(size);
  int * unallocatedPointsCount_d;
  cudaMalloc(&unallocatedPointsCount_d, sizeof(*unallocatedPointsCount_h));
  cudaMemcpy(unallocatedPointsCount_d, unallocatedPointsCount_h, sizeof(*unallocatedPointsCount_h), cudaMemcpyHostToDevice);

  int num_cuda_blocks = size / threadsPerCudaBlock + 1;
  while(*unallocatedPointsCount_h > 0){ //POSSIBILITY OF INFINITE LOOP if no applicable space is left for an unallocated block even if there is still space left in hash table
    allocateVoxelBlocks<<<num_cuda_blocks,threadsPerCudaBlock>>>(points_d, hash_table_d, block_heap_d, unallocatedPoints_d, pointcloud_voxel_blocks_d_index, unallocatedPointsCount_d);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    cudaMemcpy(unallocatedPointsCount_h, unallocatedPointsCount_d, sizeof(*unallocatedPointsCount_h), cudaMemcpyDeviceToHost);
  }

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

void TSDFHandler::visualize(Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h, HashTable * hash_table_d, BlockHeap * block_heap_d){
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

  int numVisVoxBlocks = HASH_TABLE_SIZE / threadsPerCudaBlock + 1;
  visualizeOccupiedVoxels<<<numVisVoxBlocks,threadsPerCudaBlock>>>(hash_table_d, block_heap_d, occupied_voxels_d, occupied_voxels_index_d, sdfWeightVoxelVals_d);
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
  printf("Visualize Voxels Duration: %f\n", milliseconds);

}