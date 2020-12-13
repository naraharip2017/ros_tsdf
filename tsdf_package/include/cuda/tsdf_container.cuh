#ifndef _TSDF_CONTAINER_CUH
#define _TSDF_CONTAINER_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>
#include <math.h>
#include <stdio.h>

typedef Eigen::Matrix<float, 3, 1> Vector3f;

//todo: make params - need to figure out how to allocate array of objects with dynamic pointers
__constant__
const int VOXEL_PER_SIDE = 8;
__constant__
const int HASH_ENTRIES_PER_BUCKET = 2;
__constant__
const int NUM_BUCKETS = 1000000;
__constant__
const int HASH_TABLE_SIZE = HASH_ENTRIES_PER_BUCKET * NUM_BUCKETS;
__constant__
const int NUM_HEAP_BLOCKS = 500000;
__constant__
const int PRIME_ONE = 73856093;
__constant__
const int PRIME_TWO = 19349669;
__constant__
const int PRIME_THREE = 83492791;

struct Voxel{
    float sdf;
    float weight;

    __device__ __host__
    Voxel():sdf(0), weight(0) {}
};

struct VoxelBlock{
    Voxel voxels[VOXEL_PER_SIDE * VOXEL_PER_SIDE * VOXEL_PER_SIDE];
    int mutex[VOXEL_PER_SIDE * VOXEL_PER_SIDE * VOXEL_PER_SIDE]; //change to bool
    __device__ __host__
    VoxelBlock(){
        for(int i=0;i<VOXEL_PER_SIDE * VOXEL_PER_SIDE * VOXEL_PER_SIDE; ++i){
            mutex[i] = 0;
        }
    }
};

struct HashEntry{
    Vector3f position; 
    int offset;
    int pointer;
    __device__ __host__
    HashEntry():position(0,0,0),offset(0),pointer(0){
    }
    __device__ __host__ 
    HashEntry(Vector3f position, int pointer):offset(0){
        this->position = position;
        this->pointer = pointer;
    }
    __device__ __host__
    bool isFree(){ //when deleting entries make sure the positions do not get set to -0 otherwise change this to an epsilon or fabs
        if((position(0) == 0) && (position(1) == 0) && (position(2) == 0))
            return true;
            
        return false;
    }

    __device__ __host__
    void setFree(){ //when deleting entries make sure the positions do not get set to -0 otherwise change this to an epsilon or fabs
        position(0) = position(1) = position(2) = 0;
    }
};

struct HashTable{
    HashEntry hashEntries[HASH_ENTRIES_PER_BUCKET * NUM_BUCKETS];
    int mutex[NUM_BUCKETS]; //make bool
    __device__ __host__
    HashTable(){
        for(int i=0; i<NUM_BUCKETS;++i){
            mutex[i] = 0;
        }
    }
};

struct BlockHeap{
    VoxelBlock blocks[NUM_HEAP_BLOCKS]; 
    int freeBlocks[NUM_HEAP_BLOCKS];
    int currentIndex;
    BlockHeap() {   
        for(int i=0; i<NUM_HEAP_BLOCKS; ++i){
            freeBlocks[i] = i;
        }
        currentIndex = 0;
    }
};

class TSDFContainer{

public:
    __host__
    TSDFContainer();

    __host__
    ~TSDFContainer();

    __host__
    HashTable * getCudaHashTable();

    __host__
    BlockHeap * getCudaBlockHeap();

private:
    HashTable * hashTable_h;
    HashTable * hashTable_d;

    BlockHeap * blockHeap_h;
    BlockHeap * blockHeap_d;
};

#endif