#ifndef _TSDF_HANDLER_CUH
#define _TSDF_HANDLER_CUH
#include <cuda.h>
#include <cuda_runtime.h>

#define VOXEL_PER_BLOCK 5
#define HASH_ENTRIES_PER_BUCKET 2
#define NUM_BUCKETS 4
#define NUM_HEAP_BLOCKS 8

#define PRIME_ONE 73856093
#define PRIME_TWO 19349669
#define PRIME_THREE 83492791

struct Point{ //make own header file
    short x;
    short y;
    short z;
    __device__ __host__
    Point(short x, short y, short z){
        this->x = x;
        this->y = y;
        this->z = z;
    }
    __device__ __host__
    Point():x(0),y(0),z(0){}
    __device__ __host__
    bool operator==(Point& B){
        return (this->x==B.x) && (this->y == B.y) && (this->z == B.z);
    }
};

struct Voxel{
    float sdf;
    //unsigned char sdf_color[3];
    float weight;

    __device__ __host__
    Voxel():sdf(0), weight(0) {}
    // {
    //     sdf_color[0] = sdf_color[1] = sdf_color[2] = 0;
    // }
};

struct VoxelBlock{
    Voxel voxels[VOXEL_PER_BLOCK * VOXEL_PER_BLOCK * VOXEL_PER_BLOCK];
    __device__ __host__
    VoxelBlock(){}
};

struct HashEntry{
    Point position; //if using center position of voxels then can set this to 0 0 0 and avoid using a pointer
    short offset;
    int pointer;
    __device__ __host__
    HashEntry():offset(0),pointer(0){}
    __device__ __host__
    HashEntry(Point position, int pointer):offset(0){
        this->position = position;
        this->pointer = pointer;
    }
    __device__ __host__
    bool isFree(){
        if(position.x == 0 && position.y == 0 && position.z == 0){
            return true;
        }
        return false;
    }
};

struct HashTable{
    HashEntry hashEntries[HASH_ENTRIES_PER_BUCKET * NUM_BUCKETS];
    int mutex[NUM_BUCKETS];
    __device__ __host__
    HashTable(){}
};

struct BlockHeap{
    VoxelBlock blocks[NUM_HEAP_BLOCKS]; //how big do we want the environment 2 for now
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
    void integrateVoxelBlockPointsIntoHashTable();

private:
    HashTable * hashTable_h;
    HashTable * hashTable_d;

    BlockHeap * blockHeap_h;
    BlockHeap * blockHeap_d;

};

#endif