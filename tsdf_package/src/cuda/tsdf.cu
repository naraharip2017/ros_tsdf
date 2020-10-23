#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "tsdf_node.cuh"

#define PRIME_ONE 73856093
#define PRIME_TWO 19349669
#define PRIME_THREE 83492791

// __global__
// void updateTableSDF(HashTable * table_d)
// {
//   table_d->buckets[0].hashEntries[0].offset = 1;
//   return;
// }

// __global__
// void allocatedHeap(BlockHeap * blockHeap_d)
// {
//   blockHeap_d->blocks[0].voxels[0].sdf = 2;
//   return;
// }

__global__
void printHashTable(HashTable * table_d, BlockHeap * blockHeap_d){
  Bucket * buckets = table_d->buckets;
  for(size_t i=0;i<NUM_BUCKETS; ++i){
    printf("Bucket: %d\n", i);
    HashEntry * hashEntries = buckets[i].hashEntries;
    for(size_t it = 0; it<HASH_ENTRIES_PER_BUCKET; ++it){
      HashEntry hashEntry = hashEntries[it];
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
size_t hash(Point point){ //tested using int can get negatives
  return (((size_t)point.x * PRIME_ONE) ^
  ((size_t)point.y * PRIME_TWO) ^
  ((size_t)point.z * PRIME_THREE)) % NUM_BUCKETS;
}

__global__
void pointIntegration(Point * points_d, HashTable * table_d, BlockHeap* blockHeap_d)
{
  
  //REMEMBER THREAD SYNCHRONIZATION
  //CAN COPY GLOBAL MEMORY LOCALLY ANYWHERE AND THEN RESYNC WITH GLOBAL MEM?

  //check for block in table
  Point point_d = points_d[threadIdx.x];
  printf("Point: (%d, %d, %d)\n", point_d.x, point_d.y, point_d.z);
  size_t index = hash(point_d);
  Bucket * buckets = table_d->buckets;
  HashEntry * hashEntries = buckets[index].hashEntries;
  bool blockNotAllocated = true;
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    HashEntry hashEntry = hashEntries[i];
    if(hashEntry.position == point_d){ //what to do if positions are 0,0,0 then every initial block will map to the point
      break; //todo: return reference to block
      blockNotAllocated = false; //update this to just return
      //return
    }
  }
  //todo: check linked list in hashEntries

  //leads to divergent threads so break these up

  //allocate block
  if(blockNotAllocated){
    //locks?
    //lock bucket
    if(!atomicCAS(&table_d->mutex[index], 0, 1)){
      // if(threadIdx.x>3){
      //   hashEntries[1] = *allocBlockHashEntry;
      // }
      // else{
      //   hashEntries[0] = *allocBlockHashEntry;
      // }
      VoxelBlock * allocBlock = new VoxelBlock();
      int blockHeapFreeIndex = atomicAdd(&(blockHeap_d->currentIndex), 1);
      blockHeap_d->blocks[blockHeapFreeIndex] = *allocBlock;
      
      
      HashEntry * allocBlockHashEntry = new HashEntry(point_d, blockHeapFreeIndex);
      Point *p = new Point(0,0,0);
      for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
        HashEntry hashEntry = hashEntries[i];
        if(hashEntry.position == *p ){ //what to do if positions are 0,0,0 then every initial block will map to the point - set initial position to null in constructor
          hashEntries[i] = *allocBlockHashEntry;
          break;
        }
      }
      //free block here or we have another kernel in parellel reset all mutex
      atomicExch(&table_d->mutex[index], 0);
    }
    //printf("thread id: %d, mutex: %d\n", threadIdx.x, mutex);
  }
  return;
}

// void initTableAndBlockHeap(){

// }

__global__
void cudaTest(int * arr){
  printf("%d", arr[threadIdx.x]);
}

size_t hashMe(Point point){ //tested using int can get negatives
  return (((size_t)point.x * PRIME_ONE) ^
  ((size_t)point.y * PRIME_TWO) ^
  ((size_t)point.z * PRIME_THREE)) % NUM_BUCKETS;
}

int tsdfmain()
{
  
    HashTable * table_h = new HashTable();
    HashTable * table_d;

    cudaMalloc(&table_d, sizeof(*table_h));
    cudaMemcpy(table_d, table_h, sizeof(*table_h), cudaMemcpyHostToDevice);
    // updateTableSDF<<<1,1>>>(table_d);
    // cudaMemcpy(table_h, table_d, sizeof(*table_h), cudaMemcpyDeviceToHost);

    BlockHeap * blockHeap_h = new BlockHeap();
    BlockHeap * blockHeap_d;

    cudaMalloc(&blockHeap_d, sizeof(*blockHeap_h));
    cudaMemcpy(blockHeap_d, blockHeap_h, sizeof(*blockHeap_h), cudaMemcpyHostToDevice);
    // allocatedHeap<<<1,1>>>(blockHeap_d);
    // cudaMemcpy(blockHeap_h, blockHeap_d, sizeof(*blockHeap_h), cudaMemcpyDeviceToHost);
    // printf("%f\n", blockHeap_h->blocks[0].voxels[0].sdf);

    // Point points[5];
    // for(int i=0;i<)


    int size= 4;
    Point point_h[size];
    // Point * A = new Point(1,1,1);
    // Point * B = new Point(5,5,5);
    // Point * C = new Point(9,9,9);
    // point_h[0] = *A;
    // point_h[1] = *B;
    // point_h[2] = *C;

    for(int i=1; i<=size; ++i){
      Point * p = new Point(i,i,i);
      point_h[i-1] = *p;
    }

    // Point * point_h = new Point(1,1,1);
    Point * point_d;
    cudaMalloc(&point_d, sizeof(*point_h)*size);
    cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
    pointIntegration<<<1,size>>>(point_d, table_d, blockHeap_d);

    printHashTable<<<1,1>>>(table_d, blockHeap_d);

    for(int i=1; i<=size; ++i){
      Point * p = new Point(i+4,i+4,i+4);
      point_h[i-1] = *p;
    }

    cudaMemcpy(point_d, point_h, sizeof(*point_h)*size, cudaMemcpyHostToDevice);
    pointIntegration<<<1,size>>>(point_d, table_d, blockHeap_d);

    printHashTable<<<1,1>>>(table_d, blockHeap_d);

    cudaFree(point_d);
    //free(point_h);
    cudaFree(table_d);
    free(table_h);
    cudaFree(blockHeap_d);
    free(blockHeap_h);
  
  //process Points : points -> voxels -> voxel Blocks

  //For each block insert to table either getting reference or inserting - lock

  //Create compact hashtable

  //For each block update voxels sdf and weight

  //Remove unneeded blocks

  return EXIT_SUCCESS;
}