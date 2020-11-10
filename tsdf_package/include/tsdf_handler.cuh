#ifndef _TSDF_HANDLER_CUH
#define _TSDF_HANDLER_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>
#include <math.h>
// #include <tf2_ros/transform_listener.h>
// #include <tf2_ros/buffer.h>

#define VOXEL_PER_BLOCK 10
#define HASH_ENTRIES_PER_BUCKET 2
#define NUM_BUCKETS 1000000
#define HASH_TABLE_SIZE HASH_ENTRIES_PER_BUCKET * NUM_BUCKETS
#define NUM_HEAP_BLOCKS 200000
#define VOXEL_SIZE .1
#define HALF_VOXEL_SIZE VOXEL_SIZE / 2

#define PRIME_ONE 73856093
#define PRIME_TWO 19349669
#define PRIME_THREE 83492791

// typedef Eigen::Matrix<float, 3, 1> Vector3f;
const float VOXEL_BLOCK_SIZE = VOXEL_SIZE * VOXEL_PER_BLOCK;
const float HALF_VOXEL_BLOCK_SIZE = VOXEL_BLOCK_SIZE / 2;
const float EPSILON = VOXEL_BLOCK_SIZE / 4;
const float VOXEL_EPSILON = VOXEL_SIZE / 4;

struct Voxel{
    float sdf;
    float weight;

    __device__ __host__
    Voxel():sdf(0), weight(0) {}
};

struct VoxelBlock{
    Voxel voxels[VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK];
    int mutex[VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK];
    //int mutex[VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK]; //change to bool
    __device__ __host__
    VoxelBlock(){
        for(int i=0;i<VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK; ++i){
            mutex[i] = 0;
        }
    }
};

struct HashEntry{
    Eigen::Matrix<float, 3, 1> position; 
    short offset;
    int pointer;
    __device__ __host__
    HashEntry():position(0,0,0),offset(0),pointer(0){
    }
    __device__ __host__
    HashEntry(Eigen::Matrix<float, 3, 1> position, int pointer):offset(0){
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
    bool checkIsPositionEqual(Eigen::Matrix<float, 3, 1> B){
        Eigen::Matrix<float, 3, 1> diff = this->position-B;
        if((fabs(diff(0)) < EPSILON) && (fabs(diff(1)) < EPSILON) && (fabs(diff(2)) < EPSILON))
            return true;
            
        return false;
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

class TsdfHandler{

public:
    __host__
    TsdfHandler();

    //implement deconstructor for cudaFree and free of

    __host__
    void integrateVoxelBlockPointsIntoHashTable(Eigen::Matrix<float, 3, 1> point_h[], int size);

    __host__
    HashTable * getCudaHashTable(){
        return hashTable_d;
    }

    BlockHeap * getCudaBlockHeap(){
        return blockHeap_d;
    }

private:
    HashTable * hashTable_h;
    HashTable * hashTable_d;

    BlockHeap * blockHeap_h;
    BlockHeap * blockHeap_d;
};

#endif