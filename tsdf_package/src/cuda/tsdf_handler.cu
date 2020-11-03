#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "tsdf_handler.cuh"

#define PRIME_ONE 73856093
#define PRIME_TWO 19349669
#define PRIME_THREE 83492791

typedef Eigen::Matrix<float, 3, 1> Vector3f;

//rename to voxelBlock_handler
__global__
void printHashTableAndBlockHeap(HashTable * hashTable_d, BlockHeap * blockHeap_d){
  HashEntry * hashEntries = hashTable_d->hashEntries;
  for(size_t i=0;i<NUM_BUCKETS; ++i){
    printf("Bucket: %lu\n", (unsigned long)i);
    for(size_t it = 0; it<HASH_ENTRIES_PER_BUCKET; ++it){
      HashEntry hashEntry = hashEntries[it+i*HASH_ENTRIES_PER_BUCKET];
      Vector3f position = hashEntry.position;
      if (hashEntry.isFree()){
        printf("  Hash Entry with   Position: (N,N,N)   Offset: %d   Pointer: %d\n", hashEntry.offset, hashEntry.pointer);
      }
      else{
        printf("  Hash Entry with   Position: (%f,%f,%f)   Offset: %d   Pointer: %d\n", position(0), position(1), position(2), hashEntry.offset, hashEntry.pointer);
      }
    }
    printf("%s\n", "--------------------------------------------------------");
  }

  printf("Block Heap Free List: ");
  int * freeBlocks = blockHeap_d->freeBlocks;
  for(size_t i = 0; i<NUM_HEAP_BLOCKS; ++i){
    printf("%d  ", freeBlocks[i]);
  }
  printf("\nCurrent Index: %d\n", blockHeap_d->currentIndex);
}

__device__
size_t retrieveHashIndexFromPoint(Vector3f point){ //tested using int can get negatives
  return abs((((int)point(0)*PRIME_ONE) ^ ((int)point(1)*PRIME_TWO) ^ ((int)point(2)*PRIME_THREE)) % NUM_BUCKETS);
}

__global__
void allocateVoxelBlocks(Vector3f * points_d, HashTable * hashTable_d, BlockHeap * blockHeap_d, bool * unallocatedPoints_d, int * size_d, int * unallocatedPointsCount_d)
{
  
  //REMEMBER THREAD SYNCHRONIZATION
  //CAN COPY GLOBAL MEMORY LOCALLY ANYWHERE AND THEN RESYNC WITH GLOBAL MEM?

  //check for block in table
  int threadIndex = blockIdx.x*1024 + threadIdx.x;
  if(threadIndex>=*size_d || (unallocatedPoints_d[threadIndex]==0)){
    return;
  }
  Vector3f point_d = points_d[threadIndex];
  size_t bucketIndex = retrieveHashIndexFromPoint(point_d);
  size_t currentGlobalIndex = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  printf("Point: (%f, %f, %f), Index: %lu\n", point_d(0), point_d(1), point_d(2), bucketIndex);
  HashEntry * hashEntries = hashTable_d->hashEntries;
  bool blockNotAllocated = true;
  HashEntry hashEntry;
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hashEntry = hashEntries[currentGlobalIndex+i];
    if(hashEntry.checkIsPositionEqual(point_d)){
      unallocatedPoints_d[threadIndex] = 0;
      atomicSub(unallocatedPointsCount_d, 1);
      blockNotAllocated = false;
      break; //todo: return reference to block
      //update this to just return
      //return
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
    if(hashEntry.checkIsPositionEqual(point_d)){ //what to do if positions are 0,0,0 then every initial block will map to the point
      unallocatedPoints_d[threadIndex] = 0;
      atomicSub(unallocatedPointsCount_d, 1);
      blockNotAllocated = false; //update this to just return
      // printf("%s", "block allocated");
      break; //todo: return reference to block
      //return
    }
  }

  //can have a full checker boolean if true then skip checking the hashed bucket for writing

  //leads to divergent threads so break these up

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
            HashEntry * allocBlockHashEntry = new HashEntry(point_d, blockHeapFreeIndex);
            hashEntries[insertCurrentGlobalIndex+i] = *allocBlockHashEntry;
            notInserted = false;
            unallocatedPoints_d[threadIndex] = 0;
            atomicSub(unallocatedPointsCount_d, 1);
            break;
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


TsdfHandler::TsdfHandler(){
  hashTable_h = new HashTable();
  blockHeap_h = new BlockHeap();
  cudaMalloc(&hashTable_d, sizeof(*hashTable_h));
  cudaMemcpy(hashTable_d, hashTable_h, sizeof(*hashTable_h), cudaMemcpyHostToDevice);
  cudaMalloc(&blockHeap_d, sizeof(*blockHeap_h));
  cudaMemcpy(blockHeap_d, blockHeap_h, sizeof(*blockHeap_h), cudaMemcpyHostToDevice);

  // clock_ = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  // tfBuffer = new tf2_ros::Buffer(clock_);
  // tfListener = new tf2_ros::TransformListener(*tfBuffer);
}

size_t hashMe(Vector3f point){ //tested using int can get negatives
  return abs((((int)point(0)*PRIME_ONE) ^ ((int)point(1)*PRIME_TWO) ^ ((int)point(2)*PRIME_THREE)) % NUM_BUCKETS);
}

void TsdfHandler::integrateVoxelBlockPointsIntoHashTable(Vector3f points_h[], int size)
{
  int * size_h = &size;
  int * size_d;
  cudaMalloc(&size_d, sizeof(*size_h));
  cudaMemcpy(size_d, size_h, sizeof(*size_h), cudaMemcpyHostToDevice);

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

  Vector3f * points_d;
  cudaMalloc(&points_d, sizeof(*points_h)*size);
  cudaMemcpy(points_d, points_h, sizeof(*points_h)*size, cudaMemcpyHostToDevice);

  int threadsPerCudaBlock = 1024;
  int numCudaBlocks = size / threadsPerCudaBlock + 1;
  while(*unallocatedPointsCount_h > 0){ //FIX THIS SO THERE IS NO POSSIBILITY OF INFINITE LOOP WHEN INSERTING INTO THE HASH TABLE IS NOT POSSIBLE - check size of block heap pointer or whether hash table is full in available entries for inserting a point
    allocateVoxelBlocks<<<numCudaBlocks,threadsPerCudaBlock>>>(points_d, hashTable_d, blockHeap_d, unallocatedPoints_d, size_d, unallocatedPointsCount_d);
    printHashTableAndBlockHeap<<<1,1>>>(hashTable_d, blockHeap_d);
    cudaDeviceSynchronize();
    cudaMemcpy(unallocatedPointsCount_h, unallocatedPointsCount_d, sizeof(*unallocatedPointsCount_h), cudaMemcpyDeviceToHost);
  }



  // for(int i = 0; i<size; ++i){
  //   unallocatedPointsVector.push_back(points_h[i]); //don't repeat points might get really crowded with repeat points since seeing same blocks over and over
  // }

  // size = unallocatedPointsVector.size();

  // // printf("size before: %d\n", size);

  // Vector3f * point_h = &unallocatedPointsVector[0];

  // // printf("point1: (%f,%f,%f)\n", unallocatedPointsVector[0](0), unallocatedPointsVector[0](1), unallocatedPointsVector[0](2));

  // int * size_h = &size;
  // int * size_d;
  // cudaMalloc(&size_d, sizeof(*size_h));
  // cudaMemcpy(size_d, size_h, sizeof(*size_h), cudaMemcpyHostToDevice);
  // bool * unallocatedPoints_h = new bool[size];
  // for(int i=0;i<size;++i)
  // {
  //   unallocatedPoints_h[i] = 0;
  // }
  // bool * unallocatedPoints_d;
  // cudaMalloc(&unallocatedPoints_d, sizeof(*unallocatedPoints_h)*size);
  // cudaMemcpy(unallocatedPoints_d, unallocatedPoints_h, sizeof(*unallocatedPoints_h)*size, cudaMemcpyHostToDevice);
  
  // Vector3f * point_d;
  // cudaMalloc(&point_d, sizeof(*point_h)*size);
  // cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
  // int threadsPerCudaBlock = 1024;
  // int numCudaBlocks = size / threadsPerCudaBlock + 1;
  // allocateVoxelBlocks<<<numCudaBlocks,threadsPerCudaBlock>>>(point_d, hashTable_d, blockHeap_d, unallocatedPoints_d, size_d);
  // printHashTableAndBlockHeap<<<1,1>>>(hashTable_d, blockHeap_d);

  // cudaDeviceSynchronize();

  // cudaMemcpy(unallocatedPoints_h, unallocatedPoints_d, sizeof(*unallocatedPoints_h)*size, cudaMemcpyDeviceToHost);

  // std::vector<Vector3f> tempUnallocatedPointsVector; 
  // for(int i = 0; i<size; ++i){
  //   if(unallocatedPoints_h[i]){
  //     tempUnallocatedPointsVector.push_back(point_h[i]);
  //   }
  // }

  // unallocatedPointsVector = tempUnallocatedPointsVector;

  // // printf("size after: %lu\n", unallocatedPointsVector.size());

  // for (int it = 0; it < unallocatedPointsVector.size(); ++it)
  // {
  //   printf("%d: Unallocated Point: (%f, %f, %f) at index: %lu\n", it+1, unallocatedPointsVector[it](0), 
  //   unallocatedPointsVector[it](1), unallocatedPointsVector[it](2), hashMe(unallocatedPointsVector[it]));
  // }
  
  //process Points : points -> voxels -> voxel Blocks

  //For each block insert to table either getting reference or inserting - lock

  //Create compact hashtable

  //For each block update voxels sdf and weight

  //Remove unneeded blocks
}

//can make this its own function but need to make ssure buffer works then in main
// Vector3f TsdfHandler::getOriginInPointCloudFrame(const std::string & target_frame, const sensor_msgs::msg::PointCloud2 & in){
//   // Get the TF transform
//   geometry_msgs::msg::TransformStamped transform;
//   try { //wait for a duration
//     transform =
//       tf_buffer.lookupTransform(
//       target_frame, in.header.frame_id, tf2_ros::fromMsg(in.header.stamp));
//       //use transform to transform 0,0,0 and return
//   } catch (tf2::LookupException & e) {
//     RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
//   } catch (tf2::ExtrapolationException & e) {
//     RCLCPP_ERROR(rclcpp::get_logger("pcl_ros"), "%s", e.what());
//   }
  
//   return;
// }
