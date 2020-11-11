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
  printf("\nCurrent Index: %d\n", blockHeap_d->currentIndex);
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

 __global__
 void getVoxelBlocksForPoint(pcl::PointXYZ * points_d, Vector3f * pointCloudVoxelBlocks_d, int * pointer_d, Vector3f * origin_transformed_d, int * size_d){
  int threadIndex = (blockIdx.x*128 + threadIdx.x);
  if(threadIndex>=*size_d){
    return;
  }
  pcl::PointXYZ point_d = points_d[threadIndex];
  Vector3f u = *origin_transformed_d;
  Vector3f point_d_vector(point_d.x, point_d.y, point_d.z);
  Vector3f v = point_d_vector - u; //direction
  //equation of line is u+tv
  float vMag = sqrt(pow(v(0), 2) + pow(v(1),2) + pow(v(2), 2));
  Vector3f v_normalized = v / vMag;
  Vector3f truncation_start = point_d_vector - truncation_distance*v_normalized;
  
  Vector3f truncation_end = point_d_vector + truncation_distance*v_normalized;

  float distance_tStart_origin = pow(truncation_start(0) - u(0), 2) + pow(truncation_start(1) - u(1),2) + pow(truncation_start(2) - u(2), 2);
  float distance_tEnd_origin = pow(truncation_end(0) - u(0), 2) + pow(truncation_end(1) - u(1),2) + pow(truncation_end(2) - u(2), 2);

  if(distance_tEnd_origin < distance_tStart_origin){
    Vector3f temp = truncation_start;
    truncation_start = truncation_end;
    truncation_end = temp;
  }

  Vector3f truncation_start_block = GetVoxelBlockCenterFromPoint(truncation_start);
  Vector3f truncation_end_block = GetVoxelBlockCenterFromPoint(truncation_end);
  float stepX = v(0) > 0 ? VOXEL_BLOCK_SIZE : -1 * VOXEL_BLOCK_SIZE;
  float stepY = v(1) > 0 ? VOXEL_BLOCK_SIZE : -1 * VOXEL_BLOCK_SIZE;
  float stepZ = v(2) > 0 ? VOXEL_BLOCK_SIZE : -1 * VOXEL_BLOCK_SIZE;
  float tMaxX = fabs(v(0) < 0 ? (truncation_start_block(0) - HALF_VOXEL_BLOCK_SIZE - u(0)) / v(0) : (truncation_start_block(0) + HALF_VOXEL_BLOCK_SIZE - u(0)) / v(0));
  float tMaxY = fabs(v(1) < 0 ? (truncation_start_block(1) - HALF_VOXEL_BLOCK_SIZE - u(1)) / v(1) : (truncation_start_block(1) + HALF_VOXEL_BLOCK_SIZE - u(1)) / v(1));
  float tMaxZ = fabs(v(2) < 0 ? (truncation_start_block(2) - HALF_VOXEL_BLOCK_SIZE - u(2)) / v(2) : (truncation_start_block(2) + HALF_VOXEL_BLOCK_SIZE - u(2)) / v(2));
  float tDeltaX = fabs(VOXEL_BLOCK_SIZE / v(0));
  float tDeltaY = fabs(VOXEL_BLOCK_SIZE / v(1));
  float tDeltaZ = fabs(VOXEL_BLOCK_SIZE / v(2));
  Vector3f currentBlock(truncation_start_block(0), truncation_start_block(1), truncation_start_block(2));

  while(!checkFloatingPointVectorsEqual(currentBlock, truncation_end_block, EPSILON)){
    int pointCloudVoxelBlocksIndex = atomicAdd(&(*pointer_d), 1);
    pointCloudVoxelBlocks_d[pointCloudVoxelBlocksIndex] = currentBlock;
    if(tMaxX < tMaxY){
      if(tMaxX < tMaxZ)
      {
        currentBlock(0) += stepX;
        tMaxX += tDeltaX;
      }
      else if(tMaxX > tMaxZ){
        currentBlock(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else{
        currentBlock(0) += stepX;
        currentBlock(2) += stepZ;
        tMaxX += tDeltaX;
        tMaxZ += tDeltaZ;
      }
    }
    else if(tMaxX > tMaxY){
      if(tMaxY < tMaxZ){
        currentBlock(1) += stepY;
        tMaxY += tDeltaY;
      }
      else if(tMaxY > tMaxZ){
        currentBlock(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else{
        currentBlock(1) += stepY;
        currentBlock(2) += stepZ;
        tMaxY += tDeltaY;
        tMaxZ += tDeltaZ;
      }
    }
    else{
      if(tMaxZ < tMaxX){
        currentBlock(2) += stepZ;
        tMaxZ += tDeltaZ;
      }
      else if(tMaxZ > tMaxX){
        currentBlock(0) += stepX;
        currentBlock(1) += stepY;
        tMaxX += tDeltaX;
        tMaxY += tDeltaY;
      }
      else{ 
        currentBlock(0) += stepX;
        currentBlock(1) += stepY;
        currentBlock(2) += stepZ;
        tMaxX += tDeltaX;
        tMaxY += tDeltaY;
        tMaxZ += tDeltaZ;
      }
    } 
  }      
  
  int pointCloudVoxelBlocksIndex = atomicAdd(&(*pointer_d), 1);
  pointCloudVoxelBlocks_d[pointCloudVoxelBlocksIndex] = currentBlock;
  return;
 }

 __global__
void allocateVoxelBlocks(Vector3f * points_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, bool * unallocatedPoints_d, int * size_d, int * unallocatedPointsCount_d)
{
  
  int threadIndex = (blockIdx.x*threadsPerCudaBlock + threadIdx.x);
  if(threadIndex>=*size_d || (unallocatedPoints_d[threadIndex]==0)){
    return;
  }
  Vector3f point_d = points_d[threadIndex];
  size_t bucketIndex = retrieveHashIndexFromPoint(point_d);
  size_t currentGlobalIndex = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hashEntries = hashTable_d->hashEntries;
  bool blockNotAllocated = true;
  HashEntry hashEntry;
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hashEntry = hashEntries[currentGlobalIndex+i];
    if(hashEntry.checkIsPositionEqual(point_d)){
      unallocatedPoints_d[threadIndex] = 0;
      atomicSub(unallocatedPointsCount_d, 1);
      blockNotAllocated = false;
      return;
    }
  }

  //set currentGlobalIndex to last position in bucket to check linked list
  currentGlobalIndex+=HASH_ENTRIES_PER_BUCKET-1;

  //check linked list
  while(hashEntry.offset!=0){
    short offset = hashEntry.offset;
    currentGlobalIndex+=offset;
    if(currentGlobalIndex>=HASH_TABLE_SIZE){
      currentGlobalIndex %= HASH_TABLE_SIZE;
    }
    hashEntry = hashEntries[currentGlobalIndex];
    if(hashEntry.checkIsPositionEqual(point_d)){
      unallocatedPoints_d[threadIndex] = 0;
      atomicSub(unallocatedPointsCount_d, 1);
      blockNotAllocated = false;
      return;
    }
  }

  //can have a boolean checking if bucket is completely full and avoid rechecking bucket entries

  size_t insertCurrentGlobalIndex = bucketIndex * HASH_ENTRIES_PER_BUCKET;

  //allocate block
  if(blockNotAllocated){
    if(!atomicCAS(&hashTable_d->mutex[bucketIndex], 0, 1)){
        VoxelBlock * allocBlock = new VoxelBlock();
        bool notInserted = true;
        for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
          HashEntry entry = hashEntries[insertCurrentGlobalIndex+i];
          if(entry.isFree()){ 
            int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
            blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
            cudaFree(allocBlock);
            HashEntry * allocBlockHashEntry = new HashEntry(point_d, blockHeapFreeIndex);
            hashEntries[insertCurrentGlobalIndex+i] = *allocBlockHashEntry;
            cudaFree(allocBlockHashEntry);
            notInserted = false;
            unallocatedPoints_d[threadIndex] = 0;
            atomicSub(unallocatedPointsCount_d, 1);
            atomicExch(&hashTable_d->mutex[bucketIndex], 0);
            return;
          }
        }

        size_t insertBucketIndex = bucketIndex + 1;
        if(insertBucketIndex == NUM_BUCKETS){
          insertBucketIndex = 0;
        }

        bool haveLinkedListBucketLock = true;

        //check bucket of linked list end if different release hashbucket lock
        size_t endLinkedListBucket = currentGlobalIndex / HASH_ENTRIES_PER_BUCKET;
        if(endLinkedListBucket!=bucketIndex){
          atomicExch(&hashTable_d->mutex[bucketIndex], 0);
          haveLinkedListBucketLock = !atomicCAS(&hashTable_d->mutex[endLinkedListBucket], 0, 1);
        }

        if(haveLinkedListBucketLock){
                  //find position outside of current bucket
            while(notInserted){ //grab atomicCAS of linked list before looping for free spot
              //check offset of head linked list pointer
              if(!atomicCAS(&hashTable_d->mutex[insertBucketIndex], 0, 1)){
                insertCurrentGlobalIndex = insertBucketIndex * HASH_ENTRIES_PER_BUCKET;
                for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET-1; ++i){
                  HashEntry entry = hashEntries[insertCurrentGlobalIndex+i];
                  //make this a method like entry.checkFree super unclean currently
                  if(entry.isFree() ){ //what to do if positions are 0,0,0 then every initial block will map to the point - set initial position to null in constructor
                    //set offset of last linked list node
                      int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
                      blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
                      HashEntry * allocBlockHashEntry = new HashEntry(point_d, blockHeapFreeIndex);
                      size_t insertPos = insertCurrentGlobalIndex + i;
                      hashEntries[insertPos] = *allocBlockHashEntry;
                      cudaFree(allocBlock);
                      cudaFree(allocBlockHashEntry);
                      if(insertPos > currentGlobalIndex){
                        hashEntries[currentGlobalIndex].offset = insertPos - currentGlobalIndex;
                      }
                      else{
                        hashEntries[currentGlobalIndex].offset = HASH_TABLE_SIZE - currentGlobalIndex + insertPos;
                      }
                      notInserted = false;
                      unallocatedPoints_d[threadIndex] = 0;
                      atomicSub(unallocatedPointsCount_d, 1);
                    
                    break;
                  }
                }
                atomicExch(&hashTable_d->mutex[insertBucketIndex], 0);
              }
              insertBucketIndex++;
              if(insertBucketIndex == NUM_BUCKETS){
                insertBucketIndex = 0;
              }
              if(insertBucketIndex == bucketIndex){
                // unallocatedPoints_d[threadIndex] = 1;
                return;
              }
              //check if equals hashedbucket then break, only loop through table once then have to return point for next frame
            }
            atomicExch(&hashTable_d->mutex[endLinkedListBucket], 0);
      }
      else{
        // unallocatedPoints_d[threadIndex] = 1;
        return;
      }

      //free block here or we have another kernel in parallel reset all mutex
    }
    //printf("thread id: %d, mutex: %d\n", threadIdx.x, mutex);
    //determine which blocks are not inserted
    else{
      // unallocatedPoints_d[threadIndex] = 1;
    }
  }
  return;
}

__device__
size_t getLocalVoxelIndex(Vector3f diff){
  diff /= VOXEL_SIZE;
  return floor(diff(0)) + (floor(diff(1)) * VOXEL_PER_BLOCK) + (floor(diff(2)) * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK);
}

__device__ 
int getBlockPositionForBlockCoordinates(Vector3f voxelBlockCoordinates, HashTable * hashTable_d){
  size_t bucketIndex = retrieveHashIndexFromPoint(voxelBlockCoordinates);
  size_t currentGlobalIndex = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hashEntries = hashTable_d->hashEntries;
  HashEntry hashEntry;
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hashEntry = hashEntries[currentGlobalIndex+i];
    if(hashEntry.checkIsPositionEqual(voxelBlockCoordinates)){
      return hashEntry.pointer;
    }
  }

  currentGlobalIndex+=HASH_ENTRIES_PER_BUCKET-1;

  //check linked list
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
void updateVoxels(Vector3f * voxels, HashTable * hashTable_d, BlockHeap * blockHeap_d, Vector3f * point_d, Vector3f * origin){
  int threadIndex = threadIdx.x;
  Vector3f voxelCoordinates = voxels[threadIndex];
  Vector3f voxelBlockCoordinates = GetVoxelBlockCenterFromPoint(voxelCoordinates);
  //make a global vector with half voxel block size values
  int voxelBlockHeapPosition = getBlockPositionForBlockCoordinates(voxelBlockCoordinates, hashTable_d);
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
 int threadIndex = (blockIdx.x*128 + threadIdx.x);
 if(threadIndex>=*size_d){
  return;
}
 pcl::PointXYZ point_d = points_d[threadIndex];
 Vector3f u = *origin_transformed_d;
 Vector3f point_d_vector(point_d.x, point_d.y, point_d.z);
 Vector3f v = point_d_vector - u; //direction
 //equation of line is u+tv
 float vMag = sqrt(pow(v(0), 2) + pow(v(1),2) + pow(v(2), 2));
 Vector3f v_normalized = v / vMag;
 Vector3f truncation_start = point_d_vector - truncation_distance*v_normalized;
 
 Vector3f truncation_end = point_d_vector + truncation_distance*v_normalized;

 float distance_tStart_origin = pow(truncation_start(0) - u(0), 2) + pow(truncation_start(1) - u(1),2) + pow(truncation_start(2) - u(2), 2);
 float distance_tEnd_origin = pow(truncation_end(0) - u(0), 2) + pow(truncation_end(1) - u(1),2) + pow(truncation_end(2) - u(2), 2);

 if(distance_tEnd_origin < distance_tStart_origin){
   Vector3f temp = truncation_start;
   truncation_start = truncation_end;
   truncation_end = temp;
 }

 Vector3f truncation_start_voxel = GetVoxelCenterFromPoint(truncation_start);
 Vector3f truncation_end_voxel = GetVoxelCenterFromPoint(truncation_end);
 float stepX = v(0) > 0 ? VOXEL_SIZE : -1 * VOXEL_SIZE;
 float stepY = v(1) > 0 ? VOXEL_SIZE : -1 * VOXEL_SIZE;
 float stepZ = v(2) > 0 ? VOXEL_SIZE : -1 * VOXEL_SIZE;
 float tMaxX = fabs(v(0) < 0 ? (truncation_start_voxel(0) - HALF_VOXEL_SIZE - u(0)) / v(0) : (truncation_start_voxel(0) + HALF_VOXEL_SIZE - u(0)) / v(0));
 float tMaxY = fabs(v(1) < 0 ? (truncation_start_voxel(1) - HALF_VOXEL_SIZE - u(1)) / v(1) : (truncation_start_voxel(1) + HALF_VOXEL_SIZE - u(1)) / v(1));
 float tMaxZ = fabs(v(2) < 0 ? (truncation_start_voxel(2) - HALF_VOXEL_SIZE - u(2)) / v(2) : (truncation_start_voxel(2) + HALF_VOXEL_SIZE - u(2)) / v(2));
 float tDeltaX = fabs(VOXEL_SIZE / v(0));
 float tDeltaY = fabs(VOXEL_SIZE / v(1));
 float tDeltaZ = fabs(VOXEL_SIZE / v(2));
 Vector3f currentBlock(truncation_start_voxel(0), truncation_start_voxel(1), truncation_start_voxel(2));

 //overkill - how big should this be?
 Vector3f * voxels = new Vector3f[200]; //set in terms of truncation distance and voxel size
 int size = 0;
 while(!checkFloatingPointVectorsEqual(currentBlock, truncation_end_voxel, VOXEL_EPSILON)){
   voxels[size] = currentBlock;
   size++;
  if(tMaxX < tMaxY){
    if(tMaxX < tMaxZ)
    {
      currentBlock(0) += stepX;
      tMaxX += tDeltaX;
    }
    else if(tMaxX > tMaxZ){
      currentBlock(2) += stepZ;
      tMaxZ += tDeltaZ;
    }
    else{
      currentBlock(0) += stepX;
      currentBlock(2) += stepZ;
      tMaxX += tDeltaX;
      tMaxZ += tDeltaZ;
    }
  }
  else if(tMaxX > tMaxY){
    if(tMaxY < tMaxZ){
      currentBlock(1) += stepY;
      tMaxY += tDeltaY;
    }
    else if(tMaxY > tMaxZ){
      currentBlock(2) += stepZ;
      tMaxZ += tDeltaZ;
    }
    else{
      currentBlock(1) += stepY;
      currentBlock(2) += stepZ;
      tMaxY += tDeltaY;
      tMaxZ += tDeltaZ;
    }
  }
  else{
    if(tMaxZ < tMaxX){
      currentBlock(2) += stepZ;
      tMaxZ += tDeltaZ;
    }
    else if(tMaxZ > tMaxX){
      currentBlock(0) += stepX;
      currentBlock(1) += stepY;
      tMaxX += tDeltaX;
      tMaxY += tDeltaY;
    }
    else{
      currentBlock(0) += stepX;
      currentBlock(1) += stepY;
      currentBlock(2) += stepZ;
      tMaxX += tDeltaX;
      tMaxY += tDeltaY;
      tMaxZ += tDeltaZ;
    }
  }   
 }   

 voxels[size] = currentBlock;
 size++;

 Vector3f * lidarPoint = new Vector3f(point_d.x, point_d.y, point_d.z);
 //update to check if size is greater than threads per block
 updateVoxels<<<1, size>>>(voxels, hashTable_d, blockHeap_d, lidarPoint, origin_transformed_d);
 cdpErrchk(cudaPeekAtLastError());
 cudaDeviceSynchronize();
 cudaFree(lidarPoint);
  cudaFree(voxels);
  return;
}

__global__
void processOccupiedVoxelBlock(Vector3f * occupiedVoxels, int * index, Voxel * sdfWeightVoxelVals_d, Vector3f * position, VoxelBlock * block){
  int threadIndex = blockIdx.x*128 + threadIdx.x;
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
    occupiedVoxels[occupiedVoxelIndex] = v;
    sdfWeightVoxelVals_d[occupiedVoxelIndex] = voxel;
  }
}

__global__
void visualizeOccupiedVoxels(HashTable * hashTable_d, BlockHeap * blockHeap_d, Vector3f * occupiedVoxels, int * index, Voxel * sdfWeightVoxelVals_d){
  int threadIndex = blockIdx.x*128 +threadIdx.x;
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
  int numBlocks = size/128 + 1;
  processOccupiedVoxelBlock<<<numBlocks,128>>>(occupiedVoxels, index, sdfWeightVoxelVals_d, position, block);
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
  free(size_h);
  free(unallocatedPoints_h);
  free(unallocatedPointsCount_h);

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
  int occupiedVoxelsSize = 200000;
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

// __global__
// void printVoxelBlocksFromPoint(Vector3f * pointCloudVoxelBlocks_d, int * pointer_d){
//   printf("List of Points: \n");
//   for(int i=0;i<*pointer_d;++i){
//     Vector3f point = pointCloudVoxelBlocks_d[i];
//     printf("(%f, %f, %f)\n", point(0), point(1), point(2));
//   }
// }

// void testVoxelBlockTraversal(TSDFContainer * tsdfContainer, Vector3f * occupiedVoxels_h, int * index_h){
//   // float f = 10.23423;
//   int size = 5;
//   pcl::PointXYZ * point1 = new pcl::PointXYZ(.75,.75, 0.75);
//   pcl::PointXYZ * point2 = new pcl::PointXYZ(1.5,1.5, 1.5);
//   pcl::PointXYZ * point3 = new pcl::PointXYZ(-.85,-.85, -.85);
//   pcl::PointXYZ * point4 = new pcl::PointXYZ(.90,.90, .90);
//   pcl::PointXYZ * point5 = new pcl::PointXYZ(.65,.56, .65);
  
//   pcl::PointXYZ * points_h = new pcl::PointXYZ[size];
//   points_h[0] = *point1;
//   points_h[1] = *point2;
//   points_h[2] = *point3;
//   points_h[3] = *point4;
//   points_h[4] = *point5;
//   pcl::PointXYZ * points_d;
//   cudaMalloc(&points_d, sizeof(*points_h)*size);
//   cudaMemcpy(points_d, points_h, sizeof(*points_h)*size, cudaMemcpyHostToDevice);

//   int maxBlocks = 1000;
//   Vector3f pointcloud_voxel_blocks_h[maxBlocks]; //make these member functions of tsdf_handler if cant pass device reference on host code
//   Vector3f * pointCloudVoxelBlocks_d;
//   int * pointer_h = new int(0);
//   int * pointer_d;
//   cudaMalloc(&pointCloudVoxelBlocks_d, sizeof(*pointcloud_voxel_blocks_h)*maxBlocks);
//   cudaMemcpy(pointCloudVoxelBlocks_d, pointcloud_voxel_blocks_h, sizeof(*pointcloud_voxel_blocks_h)*maxBlocks,cudaMemcpyHostToDevice); //do I even need to memcpy
//   cudaMalloc(&pointer_d, sizeof(*pointer_h));
//   cudaMemcpy(pointer_d, pointer_h, sizeof(*pointer_h), cudaMemcpyHostToDevice);

//   Vector3f * origin_transformed_h = new Vector3f(0,0,0);
//   Vector3f * origin_transformed_d;
//   cudaMalloc(&origin_transformed_d, sizeof(*origin_transformed_h));
//   cudaMemcpy(origin_transformed_d, origin_transformed_h,sizeof(*origin_transformed_h),cudaMemcpyHostToDevice);

//   int numCudaBlocks = size / threadsPerCudaBlock + 1;

//   int * size_d;
//   cudaMalloc(&size_d, sizeof(int));
//   cudaMemcpy(size_d, &size, sizeof(int), cudaMemcpyHostToDevice);

//   getVoxelBlocksForPoint<<<numCudaBlocks,threadsPerCudaBlock>>>(points_d, pointCloudVoxelBlocks_d, pointer_d, origin_transformed_d, size_d);

//   // printVoxelBlocksFromPoint<<<1,1>>>(pointCloudVoxelBlocks_d, pointer_d);

//   cudaDeviceSynchronize();

//   cudaMemcpy(pointcloud_voxel_blocks_h, pointCloudVoxelBlocks_d, sizeof(*pointcloud_voxel_blocks_h)*maxBlocks,cudaMemcpyDeviceToHost);
//   cudaMemcpy(pointer_h, pointer_d, sizeof(*pointer_h), cudaMemcpyDeviceToHost);

//   printf("num voxel blocks: %d\n", *pointer_h);

//   // tsdfContainer->integrateVoxelBlockPointsIntoHashTable(pointCloudVoxelBlocks_h, *pointer_h);

//   HashTable * hashTable_d = tsdfContainer->getCudaHashTable();

//   BlockHeap * blockHeap_d = tsdfContainer->getCudaBlockHeap();
//   getVoxelsForPoint<<<numCudaBlocks,threadsPerCudaBlock>>>(points_d, origin_transformed_d, hashTable_d, blockHeap_d, size_d);

//   cudaDeviceSynchronize();

//   Vector3f * occupiedVoxels_d;
//   int * index_d;
//   cudaMalloc(&occupiedVoxels_d, sizeof(*occupiedVoxels_h)*100);
//   cudaMemcpy(occupiedVoxels_d, occupiedVoxels_h, sizeof(*occupiedVoxels_h)*100,cudaMemcpyHostToDevice);
//   cudaMalloc(&index_d, sizeof(*index_h));
//   cudaMemcpy(index_d, index_h, sizeof(*index_h), cudaMemcpyHostToDevice);

//   int numVisVoxBlocks = HASH_TABLE_SIZE / 128 + 1;
//   // printf("hash table size: %d\n", HASH_TABLE_SIZE);
//   visualizeOccupiedVoxels<<<numVisVoxBlocks,128>>>(hashTable_d, blockHeap_d, occupiedVoxels_d, index_d);

//   cudaDeviceSynchronize();

//   cudaMemcpy(occupiedVoxels_h, occupiedVoxels_d, sizeof(*occupiedVoxels_h)*100, cudaMemcpyDeviceToHost);
//   cudaMemcpy(index_h, index_d, sizeof(int), cudaMemcpyDeviceToHost);

//   // for(int i=0; i < *index_h; ++i){
//   //   printf("occupied voxel: (%f, %f, %f)\n", occupiedVoxels_h[i](0), occupiedVoxels_h[i](1), occupiedVoxels_h[i](2));
//   // }

//   // printf("occupied voxels: %d\n", *index_h);

// }

// void testVoxelTraversal(){
//   int size = 2;
//   pcl::PointXYZ * point1 = new pcl::PointXYZ(-73.4567,33.576, 632.8910);
//   pcl::PointXYZ * point2 = new pcl::PointXYZ(-7.23421,-278, 576.2342);
//   pcl::PointXYZ * points_h = new pcl::PointXYZ[size];
//   points_h[0] = *point1;
//   points_h[1] = *point2;
//   pcl::PointXYZ * points_d;
//   cudaMalloc(&points_d, sizeof(*points_h)*size);
//   cudaMemcpy(points_d, points_h, sizeof(*points_h)*size, cudaMemcpyHostToDevice);

  

//   // Vector3f * voxels_h = new Vector3f[size];
//   // Vector3f * voxels_d;
//   // Vector3f A;
//   // A(0) = -1*(VOXEL_SIZE/2);
//   // A(1) = -1*(VOXEL_SIZE/2);
//   // A(2) = -1*(VOXEL_SIZE/2);
//   // voxels_h[0] = A;
//   // // Vector3f B;
//   // // B(0) = 10/26;
//   // // B(1) = 5/26;
//   // // B(2) = 5/26;
//   // // voxels_h[1] = B;
//   // // Vector3f C;
//   // // C(0) = 5/26;
//   // // C(1) = 10/26;
//   // // C(2) = 5/26;
//   // // voxels_h[2] = C;
//   // // Vector3f D;
//   // // D(0) = -0.75;
//   // // D(1) = -0.75;
//   // // D(2) = -0.25;
//   // // voxels_h[3] = D;
//   // // Vector3f E;
//   // // E(0) = -0.25;
//   // // E(1) = -0.25;
//   // // E(2) = -0.75;
//   // // voxels_h[4] = E;
//   // // Vector3f F;
//   // // F(0) = -0.75;
//   // // F(1) = -0.25;
//   // // F(2) = -0.75;
//   // // voxels_h[5] = F;
//   // // Vector3f G;
//   // // G(0) = -0.25;
//   // // G(1) = -0.75;
//   // // G(2) = -0.75;
//   // // voxels_h[6] = G;
//   // // Vector3f H;
//   // // H(0) = -0.05;
//   // // H(1) = -0.05;
//   // // H(2) = -0.05;
//   // // voxels_h[7] = H;
  

//   // cudaMalloc(&voxels_d, sizeof(*voxels_h)*size);
//   // cudaMemcpy(voxels_d,voxels_h, sizeof(*voxels_h)*size, cudaMemcpyHostToDevice);

//   // updateVoxels<<<1,size>>>(voxels_d);

//   // cudaDeviceSynchronize();

// }