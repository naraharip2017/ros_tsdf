#ifndef _TSDF_CUH_
#define _TSDF_CUH_
#include <cuda.h>
#include <cuda_runtime.h>

#define VOXEL_PER_BLOCK 5
#define HASH_ENTRIES_PER_BUCKET 2
#define NUM_BUCKETS 5
#define NUM_HEAP_BLOCKS 10

#define PRIME_ONE 73856093
#define PRIME_TWO 19349669
#define PRIME_THREE 83492791

struct Point{
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
    Point(){}
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
    int pointer; //potentially be a short depending on how many indices in block heap
    __device__ __host__
    HashEntry(){}
    __device__ __host__
    HashEntry(Point position, int pointer):offset(0){
        this->position = position;
        this->pointer = pointer;
    }
};

struct Bucket{
    HashEntry hashEntries[HASH_ENTRIES_PER_BUCKET]; //2
    __device__ __host__
    Bucket(){}
};

struct HashTable{
    Bucket buckets[NUM_BUCKETS]; //4
    int mutex[NUM_BUCKETS];
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

// static struct BlockHasher {
//     __device__ __host__
//     size_t hash(Point point){ //tested using int can get negatives
//         return (((size_t)point.x * PRIME_ONE) ^
//         ((size_t)point.y * PRIME_TWO) ^
//         ((size_t)point.z * PRIME_THREE)) % NUM_BUCKETS;
//     }

// };

#endif