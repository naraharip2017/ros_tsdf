#ifndef _TSDF_CONTAINER_CUH
#define _TSDF_CONTAINER_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>
#include <math.h>
#include <stdio.h>

typedef Eigen::Matrix<float, 3, 1> Vector3f;

#define X_MASK(x) x && 0x3F
#define Y_MASK(x) (x && 0x3F) << 6
#define Z_MASK(x) (x && 0x1F) << 12

__constant__
const int VOXELS_PER_SIDE = 8; //the number of voxels per side in a voxel block. See note on NUM_HEAP_BLOCKS constant if changing this.
__constant__
const int VOXELS_PER_BLOCK = VOXELS_PER_SIDE * VOXELS_PER_SIDE * VOXELS_PER_SIDE; //num voxels in a voxel block
__constant__
const int HASH_ENTRIES_PER_BUCKET = 4; //the amount of slots available in a hash table bucket
__constant__
const int NUM_BUCKETS = 131072; //number of buckets in the hash table
__constant__
const int HASH_TABLE_SIZE = HASH_ENTRIES_PER_BUCKET * NUM_BUCKETS; //number of hash entries in hash table
__constant__
//if changing voxel size, truncation distance, garbage collect distance parameters or voxels per side constant may have to adjust this value
//if the number of blocks overflows this value an error msg is printed
const int NUM_HEAP_BLOCKS = 524288; //number of blocks that can be stored in the block heap
//primes for hashing function
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
    // used to store data for the voxels inside a block
    Voxel voxels[VOXELS_PER_BLOCK];

    //locks for each voxel so that during surface update two lidar points that will update the same voxel do not cause a data race
    int mutex[VOXELS_PER_BLOCK]; //change to bool
    __device__ __host__
    VoxelBlock(){
        for(int i=0;i<VOXELS_PER_BLOCK; ++i){
            mutex[i] = 0;
        }
    }
};

struct HashEntry{
    Vector3f position; //position of the voxel block in the world
    int offset; //used for linked list for a bucket if it overflows
    int block_heap_pos; //position of voxel block in block heap 
    __device__ __host__
    HashEntry():position(0,0,0),offset(0),block_heap_pos(0){
    }
    __device__ __host__ 
    HashEntry(Vector3f position, int block_heap_pos):offset(0){
        this->position = position;
        this->block_heap_pos = block_heap_pos;
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
    HashEntry hash_entries[HASH_ENTRIES_PER_BUCKET * NUM_BUCKETS]; //list of hash entries
    //locks for each hash table bucket for insert/delete of hash entries
    int mutex[NUM_BUCKETS]; //make bool 
    __device__ __host__
    HashTable(){
        for(int i=0; i<NUM_BUCKETS;++i){
            mutex[i] = 0;
        }
    }
};

struct BlockHeap{
    VoxelBlock blocks[NUM_HEAP_BLOCKS]; //block of memory for voxel blocks to be stored
    int free_blocks[NUM_HEAP_BLOCKS]; //list of indices for keeping track of the next free position in the blocks array to insert to 
    int block_count; //the current number of allocated blocks and where to search next in the free_blocks array for inserting a new block
    BlockHeap() {   
        for(int i=0; i<NUM_HEAP_BLOCKS; ++i){
            free_blocks[i] = i;
        }
        block_count = 0;
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
    HashTable * hash_table_h;
    HashTable * hash_table_d;

    BlockHeap * block_heap_h;
    BlockHeap * block_heap_d;
};

#endif