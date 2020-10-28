#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "tsdf_handler.cuh"

#define PRIME_ONE 73856093
#define PRIME_TWO 19349669
#define PRIME_THREE 83492791

__global__
void printHashTableAndBlockHeap(HashTable * hashTable_d, BlockHeap * blockHeap_d){
  HashEntry * hashEntries = hashTable_d->hashEntries;
  for(size_t i=0;i<NUM_BUCKETS; ++i){
    printf("Bucket: %lu\n", (unsigned long)i);
    for(size_t it = 0; it<HASH_ENTRIES_PER_BUCKET; ++it){
      HashEntry hashEntry = hashEntries[it+i*HASH_ENTRIES_PER_BUCKET];
      Point position = hashEntry.position;
      if (position.x == 0 && position.y == 0 && position.z == 0){
        printf("  Hash Entry with   Position: (N,N,N)   Offset: %d   Pointer: %d\n", hashEntry.offset, hashEntry.pointer);
      }
      else{
        printf("  Hash Entry with   Position: (%d,%d,%d)   Offset: %d   Pointer: %d\n", position.x, position.y, position.z, hashEntry.offset, hashEntry.pointer);
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
size_t retrieveHashIndexFromPoint(Point point){ //tested using int can get negatives
  return (((size_t)point.x * PRIME_ONE) ^
  ((size_t)point.y * PRIME_TWO) ^
  ((size_t)point.z * PRIME_THREE)) % NUM_BUCKETS;
}

__global__
void allocateVoxelBlocks(Point * points_d, HashTable * hashTable_d, BlockHeap * blockHeap_d)
{
  
  //REMEMBER THREAD SYNCHRONIZATION
  //CAN COPY GLOBAL MEMORY LOCALLY ANYWHERE AND THEN RESYNC WITH GLOBAL MEM?

  //check for block in table
  Point point_d = points_d[threadIdx.x];
  size_t bucketIndex = retrieveHashIndexFromPoint(point_d);
  size_t currentGlobalIndex = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  printf("Point: (%d, %d, %d), Index: %lu\n", point_d.x, point_d.y, point_d.z, (unsigned long)bucketIndex);
  HashEntry * hashEntries = hashTable_d->hashEntries;
  bool blockNotAllocated = true;
  HashEntry hashEntry;
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hashEntry = hashEntries[currentGlobalIndex+i];
    if(hashEntry.position == point_d){
      break; //todo: return reference to block
      blockNotAllocated = false; //update this to just return
      //return
    }
  }

  //set currentGlobalIndex to last position in bucket to check linked list
  currentGlobalIndex+=HASH_ENTRIES_PER_BUCKET-1;

  //check linked list
  while(hashEntry.offset!=0){
    short offset = hashEntry.offset;
    currentGlobalIndex+=offset;
    hashEntry = hashEntries[currentGlobalIndex];
    if(hashEntry.position == point_d){ //what to do if positions are 0,0,0 then every initial block will map to the point
      blockNotAllocated = false; //update this to just return
      printf("%s", "block allocated");
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
              // printf("here");
              insertCurrentGlobalIndex = insertBucketIndex * HASH_ENTRIES_PER_BUCKET;
              for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET-1; ++i){
                HashEntry entry = hashEntries[insertCurrentGlobalIndex+i];
                //make this a method like entry.checkFree super unclean currently
                if(entry.isFree() ){ //what to do if positions are 0,0,0 then every initial block will map to the point - set initial position to null in constructor
                  //set offset of last linked list node
                  // printf("here");
                    int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
                    blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
                    HashEntry * allocBlockHashEntry = new HashEntry(point_d, blockHeapFreeIndex);
                    hashEntries[insertCurrentGlobalIndex+i] = *allocBlockHashEntry;
                    hashEntries[currentGlobalIndex].offset = insertCurrentGlobalIndex + i - currentGlobalIndex;
                    notInserted = false;
                  
                  break;
                }
              }
              atomicExch(&hashTable_d->mutex[insertBucketIndex], 0);
            }
            insertBucketIndex++;
            if(insertBucketIndex > NUM_BUCKETS){
              insertBucketIndex = 0;
            }
            if(insertBucketIndex == bucketIndex){
              //return the point or add to temp list
              break;
            }
            //check if equals hashedbucket then break, only loop through table once then have to return point for next frame
          }
          atomicExch(&hashTable_d->mutex[endLinkedListBucket], 0);
    }

      //free block here or we have another kernel in parallel reset all mutex
    }
    //printf("thread id: %d, mutex: %d\n", threadIdx.x, mutex);
    //determine which blocks are not inserted
    else{
      //return the point or add to temp list
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
}

// size_t hashMe(Point point){ //tested using int can get negatives
//   return (((size_t)point.x * PRIME_ONE) ^
//   ((size_t)point.y * PRIME_TWO) ^
//   ((size_t)point.z * PRIME_THREE)) % NUM_BUCKETS;
// }

void TsdfHandler::integrateVoxelBlockPointsIntoHashTable()
{
    int size= 4;
    Point point_h[size];
    // Point * A = new Point(1,1,1);
    // Point * B = new Point(5,5,5);
    // Point * C = new Point(9,9,9);
    // Point * D = new Point(13,13,13);
    // Point * E = new Point(4,4,4);
    // Point * F = new Point(12,12,12);
    // point_h[0] = *A;
    // point_h[1] = *D;

    for(int i=1; i<=size; ++i){
      Point * p = new Point(i,i,i);
      point_h[i-1] = *p;
    }

    // Point * point_h = new Point(1,1,1);
    Point * point_d;
    cudaMalloc(&point_d, sizeof(*point_h)*size);
    cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
    allocateVoxelBlocks<<<1,size>>>(point_d, hashTable_d, blockHeap_d);
    printHashTableAndBlockHeap<<<1,1>>>(hashTable_d, blockHeap_d);

    cudaDeviceSynchronize();

    for(int i=1; i<=size; ++i){
      Point * p = new Point(i+4,i+4,i+4);
      point_h[i-1] = *p;
    }

    // point_h[0] = *B;
    // point_h[1] = *E;
    cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
    allocateVoxelBlocks<<<1,size>>>(point_d, hashTable_d, blockHeap_d);

    printHashTableAndBlockHeap<<<1,1>>>(hashTable_d, blockHeap_d);
    cudaDeviceSynchronize();

    // point_h[0] = *C;
    // // point_h[1] = *F;
    // cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
    // pointIntegration<<<1,size>>>(point_d, table_d, blockHeap_d);

    // printHashTable<<<1,1>>>(table_d, blockHeap_d);

    // point_h[0] = *D;
    // // point_h[1] = *F;
    // cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
    // pointIntegration<<<1,size>>>(point_d, table_d, blockHeap_d);

    // printHashTable<<<1,1>>>(table_d, blockHeap_d);

    // point_h[0] = *E;
    // // point_h[1] = *F;
    // cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
    // pointIntegration<<<1,size>>>(point_d, table_d, blockHeap_d);

    // printHashTable<<<1,1>>>(table_d, blockHeap_d);
  
  //process Points : points -> voxels -> voxel Blocks

  //For each block insert to table either getting reference or inserting - lock

  //Create compact hashtable

  //For each block update voxels sdf and weight

  //Remove unneeded blocks
}
