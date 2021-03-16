#include "cuda/tsdf_handler.cuh"

//used to determine num blocks when executing cuda kernel
const int threads_per_cuda_block = 128;

//used to set the size of the array used to store voxel blocks traversed by a lidar point cloud
int max_voxel_blocks_traversed_per_lidar_point;

//error function for cpu called after kernel calls. 
//Source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//error function for dynamic parallelism
//Source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

/**
* prints the number of blocks in the block heap
* @param block_heap the block heap
*/
__global__
void printNumAllocatedVoxelBlocksCuda(BlockHeap * block_heap){
  printf("Current Block Heap Block Count: %d\n", block_heap->block_count);
  if(block_heap->block_count > NUM_HEAP_BLOCKS){
    printf("ERROR: block heap overflow");
  }
}

/**
* given voxel block center point returns hashed bucket index for the block
* @param point the world position of a voxel block to retrieve the hashed bucket in the hash table
* @return the index of the bucket the point hashes to in the hash table
*/
__device__
size_t retrieveHashIndexFromPoint(Vector3f point){
  return abs((((int)point(0)*PRIME_ONE) ^ ((int)point(1)*PRIME_TWO) ^ ((int)point(2)*PRIME_THREE)) % NUM_BUCKETS);
}

/**
* calculates floor in scale given
* @param x the value to get the floor value of in the scale provided
* @param scale the scale to use for the floor function. For example floorFun(1.7, .5) = 1.5
* @return the floor value of x in the scale provided
*/
__device__ 
float floorFun(float x, float scale){
  return floor(x*scale) / scale;
}

/**
* Given world point return center of volume given by volume_size
* @param point the world point to retrieve the center of a grid broken up into blocks of volume_size
* @param volume_size the size of a side of a cube the world is broken up into
* @return the center of the cube the world point resides in
*/
__device__
Vector3f getVolumeCenterFromPoint(Vector3f point, float volume_size){
  float scale = 1/volume_size;
  float half_volume_size = volume_size / 2;
  Vector3f volume_center;
  volume_center(0) = floorFun(point(0), scale) + half_volume_size;
  volume_center(1) = floorFun(point(1), scale) + half_volume_size;
  volume_center(2) = floorFun(point(2), scale) + half_volume_size;
  return volume_center;
}

/**
* Check if two Vector3f are equal
* @param A vector 1
* @param B vector 2
* @param epsilon the threshold to check if A and B are equal since they have floating point data
* @return whether the vectors are equal (their diff is less than epsilon)
*/
__device__
bool checkFloatingPointVectorsEqual(Vector3f A, Vector3f B, float epsilon){
  Vector3f diff = A-B;
  //have to use an epsilon value due to floating point precision errors
  if((fabs(diff(0)) < epsilon) && (fabs(diff(1)) < epsilon) && (fabs(diff(2)) < epsilon))
    return true;

  return false;
}

/**
* calculate the line between lidar point and lidar_position then get points truncation distance away from lidar point on the line. Set truncation_start and truncation_end to those points
* truncation_start will be closer to sensor position
* @param lidar_point the lidar point 
* @param lidar_position the sensor position of the lidar in the same frame as lidar point
* @param truncation_start stores the start point of the final line segment
* @param truncation_end stores the end point of the final line segment 
*/
__device__
inline void getTruncationLineEndPoints(pcl::PointXYZ & lidar_point, Vector3f * lidar_position, Vector3f & truncation_start, Vector3f & truncation_end){
  Vector3f u = *lidar_position;
  Vector3f point_d_vector(lidar_point.x, lidar_point.y, lidar_point.z);
  Vector3f v = point_d_vector - u; //direction
  //equation of line is u+tv
  float v_mag = sqrt(pow(v(0), 2) + pow(v(1),2) + pow(v(2), 2));
  Vector3f v_normalized = v / v_mag;
  truncation_start = point_d_vector - TRUNCATION_DISTANCE*v_normalized;
  
  truncation_end = point_d_vector + TRUNCATION_DISTANCE*v_normalized;

  //set truncation_start to whichever point is closer to the sensor position
  float distance_t_start_origin = pow(truncation_start(0) - u(0), 2) + pow(truncation_start(1) - u(1),2) + pow(truncation_start(2) - u(2), 2);
  float distance_t_end_origin = pow(truncation_end(0) - u(0), 2) + pow(truncation_end(1) - u(1),2) + pow(truncation_end(2) - u(2), 2);

  if(distance_t_end_origin < distance_t_start_origin){
    Vector3f temp = truncation_start;
    truncation_start = truncation_end;
    truncation_end = temp;
  }
}

/**
* Given the truncation start volume, volume size, and ray defined by u and t traverse the world till reaching truncation_end_vol
* Read for more information: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
* @param truncation_start_vol the start point of the line segment to retrieve traversed voxels/voxel blocks
* @param truncation_end_vol the end point of the line segment to retrieve traversed voxels/voxel blocks
* @param volume_size the side length of the voxels the world is divided into
* @param u the equation of the line between the two truncation points is represented as u+t*v
* @param v the v part of the above equation
* @param traversed_vols stores the position voxels/voxel blocks that are traversed between the truncation end points
* @param traversed_vols_size the count of voxels/voxel blocks that are traversed between the truncation end points
*/
__device__
inline void traverseVolume(Vector3f & truncation_start_vol, Vector3f & truncation_end_vol, const float & volume_size, Vector3f & u, Vector3f & v, 
Vector3f * traversed_vols, int * traversed_vols_size){
  float half_volume_size = volume_size / 2;
  const float epsilon = volume_size / 4;
  const float volume_size_plus_epsilon = volume_size + epsilon;
  const float volume_size_minus_epsilon = volume_size - epsilon;
  float step_x = v(0) > 0 ? volume_size : -1 * volume_size;
  float step_y = v(1) > 0 ? volume_size : -1 * volume_size;
  float step_z = v(2) > 0 ? volume_size : -1 * volume_size;
  Vector3f truncation_vol_start_center = getVolumeCenterFromPoint(truncation_start_vol, volume_size);
  float tMax_x = fabs(v(0) < 0 ? (truncation_vol_start_center(0) - half_volume_size - u(0)) / v(0) : (truncation_vol_start_center(0) + half_volume_size - u(0)) / v(0));
  float tMax_y = fabs(v(1) < 0 ? (truncation_vol_start_center(1) - half_volume_size - u(1)) / v(1) : (truncation_vol_start_center(1) + half_volume_size - u(1)) / v(1));
  float tMax_z = fabs(v(2) < 0 ? (truncation_vol_start_center(2) - half_volume_size - u(2)) / v(2) : (truncation_vol_start_center(2) + half_volume_size - u(2)) / v(2));
  float tDelta_x = fabs(volume_size / v(0));
  float tDelta_y = fabs(volume_size / v(1));
  float tDelta_z = fabs(volume_size / v(2));
  Vector3f current_vol(truncation_start_vol(0), truncation_start_vol(1), truncation_start_vol(2));
  Vector3f current_vol_center = getVolumeCenterFromPoint(current_vol, volume_size);
  Vector3f truncation_vol_end_center = getVolumeCenterFromPoint(truncation_end_vol, volume_size);

  int insert_index;
  while(!checkFloatingPointVectorsEqual(current_vol_center, truncation_vol_end_center, epsilon)){
    //add traversed volume to list of traversed volume and update size atomically
    insert_index = atomicAdd(&(*traversed_vols_size), 1);
    traversed_vols[insert_index] = current_vol_center;

    if(tMax_x < tMax_y){
      if(tMax_x < tMax_z)
      {
        current_vol(0) += step_x;
        tMax_x += tDelta_x;
      }
      else if(tMax_x > tMax_z){
        current_vol(2) += step_z;
        tMax_z += tDelta_z;
      }
      else{
        current_vol(0) += step_x;
        current_vol(2) += step_z;
        tMax_x += tDelta_x;
        tMax_z += tDelta_z;
      }
    }
    else if(tMax_x > tMax_y){
      if(tMax_y < tMax_z){
        current_vol(1) += step_y;
        tMax_y += tDelta_y;
      }
      else if(tMax_y > tMax_z){
        current_vol(2) += step_z;
        tMax_z += tDelta_z;
      }
      else{
        current_vol(1) += step_y;
        current_vol(2) += step_z;
        tMax_y += tDelta_y;
        tMax_z += tDelta_z;
      }
    }
    else{
      if(tMax_z < tMax_x){
        current_vol(2) += step_z;
        tMax_z += tDelta_z;
      }
      else if(tMax_z > tMax_x){
        current_vol(0) += step_x;
        current_vol(1) += step_y;
        tMax_x += tDelta_x;
        tMax_y += tDelta_y;
      }
      else{ 
        current_vol(0) += step_x;
        current_vol(1) += step_y;
        current_vol(2) += step_z;
        tMax_x += tDelta_x;
        tMax_y += tDelta_y;
        tMax_z += tDelta_z;
      }
    } 
    Vector3f temp_current_vol_center = current_vol_center;
    current_vol_center = getVolumeCenterFromPoint(current_vol, volume_size);
    Vector3f diff;
    diff(0) = fabs(temp_current_vol_center(0) - current_vol_center(0));
    diff(1) = fabs(temp_current_vol_center(1) - current_vol_center(1));
    diff(2) = fabs(temp_current_vol_center(2) - current_vol_center(2));
    //check if floating point precision error with traversal. If so, return.
    if((diff(0) < volume_size_minus_epsilon && diff(1) < volume_size_minus_epsilon && diff(2) < volume_size_minus_epsilon) 
    || (diff(0) > volume_size_plus_epsilon || diff(1) > volume_size_plus_epsilon || diff(2) > volume_size_plus_epsilon)){
      return;
    }
  }      

  //add traversed volume to list of traversed volume and update size atomically
  insert_index = atomicAdd(&(*traversed_vols_size), 1);
  traversed_vols[insert_index] = current_vol_center;

}

/**
* Kernel will get voxel blocks along ray between sensor position and a point in lidar cloud within truncation distance of point
* @param lidar_points the lidar points
* @param point_cloud_voxel_blocks store the voxel blocks traversed
* @param point_cloud_voxel_blocks_size store the count of voxel blocks traversed
* @param lidar_position the sensor position of the lidar in the same frame as the lidar points
* @param lidar_points_size the amount of points in lidar_points
*/
__global__
void getVoxelBlocksCuda(pcl::PointXYZ * lidar_points, Vector3f * point_cloud_voxel_blocks, int * point_cloud_voxel_blocks_size, Vector3f * lidar_position, int * lidar_points_size){
  int thread_index = (blockIdx.x*threads_per_cuda_block + threadIdx.x);
  if(thread_index>=*lidar_points_size){
    return;
  }
  pcl::PointXYZ lidar_point = lidar_points[thread_index];
  Vector3f truncation_start;
  Vector3f truncation_end;

  getTruncationLineEndPoints(lidar_point, lidar_position, truncation_start, truncation_end);
  //equation of line is u+tv
  Vector3f u = truncation_start;
  Vector3f v = truncation_end - truncation_start;

  traverseVolume(truncation_start, truncation_end, VOXEL_BLOCK_SIZE, u, v, point_cloud_voxel_blocks, point_cloud_voxel_blocks_size);
  return;
}

/**
* Given block coordinates return the position of the block in the block heap or return -1 if not allocated
* @param voxel_block_coordinates the block coordinates to search for
* @param current_global_index keep track of the hash entry position in the hash table to see if it equals the voxel_block_coordinates. 
*                             It is intially set to the first hash entry in the bucket the block with voxel_block_coordinates would hash to
* @param hash_entries the hash entries to search for a block with voxel_block_coordinates
* @return the index of the voxel block at world position voxel_block_coordinates in the block heap or -1 if it is not allocated
*/
__device__ 
int getBlockPositionForBlockCoordinates(Vector3f & voxel_block_coordinates, size_t & current_global_index, HashEntry * hash_entries){

  HashEntry hash_entry;

  //check the hashed bucket for the block
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hash_entry = hash_entries[current_global_index+i];
    if(checkFloatingPointVectorsEqual(hash_entry.position, voxel_block_coordinates, BLOCK_EPSILON)){
      return hash_entry.block_heap_pos;
    }
  }

  //set the current_global_index to the last hash entry in the bucket voxel_block_coordinates hashes to in order to check the linked list for the bucket
  current_global_index+=HASH_ENTRIES_PER_BUCKET-1;

  //check the linked list if necessary
  while(hash_entry.offset!=0){
    int offset = hash_entry.offset;
    current_global_index+=offset;
    if(current_global_index>=HASH_TABLE_SIZE){ //if the offset overflows loop back to the beginning of the hash_entries
      current_global_index %= HASH_TABLE_SIZE;
    }
    hash_entry = hash_entries[current_global_index];
    if(checkFloatingPointVectorsEqual(hash_entry.position, voxel_block_coordinates, BLOCK_EPSILON)){
      return hash_entry.block_heap_pos;
    }
  }

  //block not allocated in hashTable
  return -1;
}

/**
* If allocating a new block then first attempt to insert in the bucket it hashed to
* @param hashed_bucket_index the bucket the block with voxel_block_coordinates hashes to in the hash table
* @param block_heap reference to block heap to allocate storage for a new block
* @param voxel_block_coordinates coordinates of block to insert to hash table and block heap
* @param hash_entries the hash entries in a hash table
* @return  true if able to insert a block into the bucket it hashes to, false otherwise
*/
__device__
inline bool attemptHashedBucketVoxelBlockCreation(size_t & hashed_bucket_index, BlockHeap * block_heap, Vector3f & voxel_block_coordinates, HashEntry * hash_entries){
  //get position of beginning of hashed bucket
  size_t insert_current_global_index = hashed_bucket_index * HASH_ENTRIES_PER_BUCKET;
  //loop through bucket and insert if there is a free space
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    if(hash_entries[insert_current_global_index+i].isFree()){ 
      //get next free position in block heap
      int block_heap_free_index = atomicAdd(&(block_heap->block_count), 1);
      VoxelBlock * alloc_block = new VoxelBlock();
      HashEntry * alloc_block_hash_entry = new HashEntry(voxel_block_coordinates, block_heap_free_index);
      block_heap->blocks[block_heap_free_index] = *alloc_block;
      hash_entries[insert_current_global_index+i] = *alloc_block_hash_entry;
      cudaFree(alloc_block);
      cudaFree(alloc_block_hash_entry);
      return true;
    }
  }
  //return false if no free position in hashed bucket
  return false;
} 

/**
* Attempt to allocated a new block into the linked list of the bucket it hashed to if there is no space in the bucket
* @param hashed_bucket_index the bucket the block with voxel_block_coordinates hashed to in the hash table
* @param block_heap reference to block heap to allocate storage for a new block
* @param hash_table reference to the hash table to allocate a hash entry for a new block
* @param insert_bucket_index the bucket to first start searching for a spot to insert a hash entry 
* @param end_linked_list_pointer index of end of linked list hash entry for the bucket that block with voxel_block_coordinates hashes to
* @param voxel_block_coordinates coordinates of block to insert to hash table and block heap
* @param hash_entries the hash entries in a hash table
* @return true if found a position for allocating a new block in the hash table, false otherwise
*/
__device__
inline bool attemptLinkedListVoxelBlockCreation(size_t & hashed_bucket_index, BlockHeap * block_heap, HashTable * hash_table, size_t & insert_bucket_index, 
size_t & end_linked_list_pointer, Vector3f & voxel_block_coordinates, HashEntry * hash_entries){
  size_t insert_current_global_index;
  //only try to insert into other buckets until we get the block's hashed bucket which has already been tried
  while(insert_bucket_index!=hashed_bucket_index){
    //check that can get lock of the bucket attempting to insert the block
    if(!atomicCAS(&hash_table->mutex[insert_bucket_index], 0, 1)){
      insert_current_global_index = insert_bucket_index * HASH_ENTRIES_PER_BUCKET;
      //loop through the insert bucket (not including the last slot which is reserved for linked list start of that bucket) checking for a free space
      for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET-1; ++i){
        if(hash_entries[insert_current_global_index+i].isFree() ){ 
            int block_heap_free_index = atomicAdd(&(block_heap->block_count), 1);
            VoxelBlock * alloc_block = new VoxelBlock();
            HashEntry * alloc_block_hash_entry = new HashEntry(voxel_block_coordinates, block_heap_free_index);
            block_heap->blocks[block_heap_free_index] = *alloc_block;
            size_t insert_pos = insert_current_global_index + i;
            hash_entries[insert_pos] = *alloc_block_hash_entry;
            cudaFree(alloc_block);
            cudaFree(alloc_block_hash_entry);
            //set offset value for last hash_entry in the linked list
            if(insert_pos > end_linked_list_pointer){
              hash_entries[end_linked_list_pointer].offset = insert_pos - end_linked_list_pointer;
            }
            else{
              hash_entries[end_linked_list_pointer].offset = HASH_TABLE_SIZE - end_linked_list_pointer + insert_pos;
            }
            return true;
        }
      }
      //if no free space in insert bucket then release lock
      atomicExch(&hash_table->mutex[insert_bucket_index], 0);
    }
    //move to next bucket to check for space to insert block
    insert_bucket_index++;
    //loop back to beginning of hash table if overflowed size
    if(insert_bucket_index == NUM_BUCKETS){
      insert_bucket_index = 0;
    }
  }
  //if found no position for block in current frame
  return false;
}

/**
* Given list of voxel blocks check that the block is allocated in the hashtable and if not do so. Unallocated points in the current frame due to not receiving lock or lack of
* space in hash table are saved to be processed in next frame
* @param voxel_blocks list of blocks to allocate
* @param hash_table hash table to allocated hash entries for blocks
* @param block_heap block heap to allocate space for blocks
* @param unallocated_blocks bool list which is same size as voxel_blocks. If a block is allocated its corresponding bool is set to 0
* @param voxel_blocks_size the number of blocks in voxel_blocks
* @param unallocated_blocks_size keeps a count of the remaining number of blocks that need to be allocated in the hash table and block heap
*/
__global__
void allocateVoxelBlocksCuda(Vector3f * voxel_blocks, HashTable * hash_table, BlockHeap * block_heap, bool * unallocated_blocks, int * voxel_blocks_size, int * unallocated_blocks_size)
{
  int thread_index = (blockIdx.x*threads_per_cuda_block + threadIdx.x);
  //check that the thread is a valid index in voxel_blocks and that the block is still unallocated
  if(thread_index>=*voxel_blocks_size || (unallocated_blocks[thread_index]==0)){
    return;
  }
  Vector3f voxel_block = voxel_blocks[thread_index];

  size_t hashed_bucket_index = retrieveHashIndexFromPoint(voxel_block);
  //beginning of the hashed bucket in hash table
  size_t current_global_index = hashed_bucket_index * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hash_entries = hash_table->hash_entries;

  int block_position = getBlockPositionForBlockCoordinates(voxel_block, current_global_index, hash_entries);

  //block is already allocated then return
  if(block_position!=-1){
    unallocated_blocks[thread_index] = 0; //set the blocks corresponding bool to 0 so it won't be processed again
    atomicSub(unallocated_blocks_size, 1); //subtract from the count of blocks that need to be allocated still
    return;
  }

  //attempt to get lock for hashed bucket for potential insert
  if(!atomicCAS(&hash_table->mutex[hashed_bucket_index], 0, 1)){
    //try to insert block into the bucket it hashed to and return if possible
    if(attemptHashedBucketVoxelBlockCreation(hashed_bucket_index, block_heap, voxel_block, hash_entries)) {
      //the block is allocated so set it to allocated so it is not processed again in another frame
      unallocated_blocks[thread_index] = 0;
      //subtract 1 from count of unallocated blocks
      atomicSub(unallocated_blocks_size, 1);
      //free lock for hashed bucket
      atomicExch(&hash_table->mutex[hashed_bucket_index], 0);
      return;
    }

    //start searching for a free position in the next bucket
    size_t insert_bucket_index = hashed_bucket_index + 1;
    //set insert_bucket_index to first bucket if overflow the hash table size
    if(insert_bucket_index == NUM_BUCKETS){
      insert_bucket_index = 0;
    }

    //Note: current_global_index will point to end of linked list which includes hashed bucket if no linked list
    //index to the bucket which contains the end of the linked list for the hashed bucket of the block
    size_t end_linked_list_bucket = current_global_index / HASH_ENTRIES_PER_BUCKET;

    bool have_end_linked_list_bucket_lock = true;

    //if end of linked list is in different bucket than hashed bucket try to get the lock for the end of the linked list
    if(end_linked_list_bucket!=hashed_bucket_index){
      //release lock of the hashed bucket
      atomicExch(&hash_table->mutex[hashed_bucket_index], 0);
      //attempt to get lock of bucket with end of linked list
      have_end_linked_list_bucket_lock = !atomicCAS(&hash_table->mutex[end_linked_list_bucket], 0, 1);
    }

    if(have_end_linked_list_bucket_lock){
      //try to insert block into the linked list for it's hashed bucket
      if(attemptLinkedListVoxelBlockCreation(hashed_bucket_index, block_heap, hash_table, insert_bucket_index, current_global_index, voxel_block, hash_entries)){
        //the block is allocated so set it to allocated so it is not processed again in another frame
        unallocated_blocks[thread_index] = 0;
        //subtract 1 from count of unallocated blocks
        atomicSub(unallocated_blocks_size, 1); 
        //free the lock of the end of linked list bucket and insert bucket
        atomicExch(&hash_table->mutex[end_linked_list_bucket], 0);
        atomicExch(&hash_table->mutex[insert_bucket_index], 0);
      }
      else{
        //free the lock of the end of linked list bucket
        atomicExch(&hash_table->mutex[end_linked_list_bucket], 0);
      }
      return;
    }
  }
}

/**
* given diff between a voxel coordinate and the bottom left block of the block it is in return the position of the voxel in the blocks voxel array
* @param diff voxel coordinates - bottom left position of its block
* @return the position for that voxel in the voxels array for its block
*/
__device__
size_t getLocalVoxelIndex(Vector3f diff){
  diff /= VOXEL_SIZE;
  return floor(diff(0)) + (floor(diff(1)) * VOXELS_PER_SIDE) + (floor(diff(2)) * VOXELS_PER_SIDE * VOXELS_PER_SIDE);
}

/**
* given a voxels index in its block's voxel array and the bottom left coordinates of the block calculate the voxel's coordinates
* @param voxel_index the voxel's index in its block's voxel array
* @param bottom_left_block_pos the bottom left coordinate of the voxel's block
* @return the coordinates of the voxel
*/
__device__
Vector3f getVoxelCoordinatesFromLocalVoxelIndex(int voxel_index, Vector3f * bottom_left_block_pos){
  float z = voxel_index / (VOXELS_PER_SIDE * VOXELS_PER_SIDE);
  voxel_index -= z*VOXELS_PER_SIDE*VOXELS_PER_SIDE;
  float y = voxel_index / VOXELS_PER_SIDE;
  voxel_index -= y*VOXELS_PER_SIDE;
  float x = voxel_index;

  Vector3f v;
  v(0) = x * VOXEL_SIZE + HALF_VOXEL_SIZE + (* bottom_left_block_pos)(0);
  v(1) = y * VOXEL_SIZE + HALF_VOXEL_SIZE + (* bottom_left_block_pos)(1);
  v(2) = z * VOXEL_SIZE + HALF_VOXEL_SIZE + (* bottom_left_block_pos)(2);

  return v;
}

/**
* @param vector get magnitude of this vector
* @return magnitude of the vector
*/
__device__
inline float getMagnitude(Vector3f vector){
  return sqrt(pow(vector(0),2) + pow(vector(1),2) + pow(vector(2),2));
}

/**
* @param a vector 1 @param b vector 2
* @return dot product of two vectors 
*/
__device__
inline float dotProduct(Vector3f a, Vector3f b){
  return a(0)*b(0) + a(1)*b(1) + a(2)*b(2);
}

/**
* given voxel coordinates, position of lidar points and sensor position calculate the distance update for the voxel
* http://helenol.github.io/publications/iros_2017_voxblox.pdf for more info
* @param voxel_coordinates center of current voxel to update. This is x in the above paper.
* @param lidar_point lidar point. This is p in the above paper.
* @param lidar_position sensor position of lidar sensor. This is s in the above paper
* @return distance update for the voxel
*/
__device__
inline float getDistanceUpdate(Vector3f voxel_coordinates, Vector3f lidar_point, Vector3f lidar_position){
  Vector3f lidar_position_diff = lidar_point - lidar_position; //p-s
  Vector3f lidar_voxel_diff = lidar_point - voxel_coordinates; //p-x
  float lidar_voxel_diff_mag = getMagnitude(lidar_voxel_diff);
  float dot_prod = dotProduct(lidar_position_diff, lidar_voxel_diff);
  if(dot_prod < 0){
    lidar_voxel_diff_mag *=-1;
  }

  return lidar_voxel_diff_mag;
}

/**
* @param distance the distance to retrieve a weight value for
* @return the weight to attribute to the distance
*/
__device__
inline float getWeightUpdate(float distance){
  return 1/(fabs(distance) + 1);

  /* alternative weigting scheme below to explore in the future. May help in environments with many thin objects.

  if(distance >= 0){
    return 1/(distance + 1);
  }
  else{
    return (1/pow(fabs(distance) + 1, 2));
  }

  */
}

/**
* update voxels sdf and weight that are within truncation distance of the line between sensor position and lidar point
* @param voxels list of voxels to update for the lidar point
* @param hash_table hash table with hash entries pointing to voxel blocks
* @param block_heap block heap storing voxel blocks
* @param lidar_point lidar point 
* @param lidar_position lidar sensor position
* @param voxels_size num voxels to update
*/
__global__
void updateVoxelsCuda(Vector3f * voxels, HashTable * hash_table, BlockHeap * block_heap, Vector3f * lidar_point, Vector3f * lidar_position, int * voxels_size){
  int thread_index = blockIdx.x*threads_per_cuda_block + threadIdx.x;
  if(thread_index >= * voxels_size){
    return;
  }
  Vector3f voxel_coordinates = voxels[thread_index];
  //get the voxel block position the voxel is within
  Vector3f voxel_block_coordinates = getVolumeCenterFromPoint(voxel_coordinates, VOXEL_BLOCK_SIZE);

  //get the voxel blocks bucket in hash table
  size_t bucketIndex = retrieveHashIndexFromPoint(voxel_block_coordinates);
  size_t current_global_index = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hash_entries = hash_table->hash_entries;

  //find block heap position for block
  int voxel_block_heap_position = getBlockPositionForBlockCoordinates(voxel_block_coordinates, current_global_index, hash_entries);
  Vector3f voxel_block_bottom_left_coordinates;
  voxel_block_bottom_left_coordinates(0) = voxel_block_coordinates(0)-HALF_VOXEL_BLOCK_SIZE;
  voxel_block_bottom_left_coordinates(1) = voxel_block_coordinates(1)-HALF_VOXEL_BLOCK_SIZE;
  voxel_block_bottom_left_coordinates(2) = voxel_block_coordinates(2)-HALF_VOXEL_BLOCK_SIZE;
  //get local position of voxel in its block
  size_t local_voxel_index = getLocalVoxelIndex(voxel_coordinates - voxel_block_bottom_left_coordinates);

  //get a reference to the block from the block heap
  VoxelBlock * block = &(block_heap->blocks[voxel_block_heap_position]);
  //get a refernce to the voxel in the voxel block
  Voxel * voxel = &(block->voxels[local_voxel_index]);
  //get a reference to the lock for the voxel in the voxel block
  int * mutex = &(block->mutex[local_voxel_index]);

  //get distance and weight updates for the voxel
  float distance = getDistanceUpdate(voxel_coordinates, *lidar_point, *lidar_position);
  float weight = getWeightUpdate(distance);
  float weight_times_distance = weight * distance;

  // TODO: Remove infinite loop hack
  int i = 0;
  bool updated_voxel = false;
  while(!updated_voxel){
    //get lock for voxel
    if(!atomicCAS(mutex, 0, 1)){
      updated_voxel = true;
      //update sdf and weight
      float old_weight = voxel->weight;
      float old_sdf = voxel->sdf;
      float new_weight = old_weight + weight;
      float new_distance = (old_weight * old_sdf + weight_times_distance) / new_weight;
      voxel->sdf = new_distance;
      new_weight = min(new_weight, MAX_WEIGHT);
      voxel->weight = new_weight;
      //free the lock for the voxel
      atomicExch(mutex, 0);
    }
    else{
      i++;
      if(i>500){
        printf("stuck\n");
      }
    }
  }
}

/**
* Get voxels traversed on ray between sensor position and a lidar point within truncation distance of lidar point
* @param lidar_points lidar points for the scan
* @param lidar_position position of lidar
* @param hash_table hash table storing hash entries of voxel blocks
* @param block_heap block heap storing voxel blocks
* @param lidar_points_size num of lidar points
*/
__global__
void getVoxelsAndUpdateCuda(pcl::PointXYZ * lidar_points, Vector3f * lidar_position, HashTable * hash_table, BlockHeap * block_heap, int * lidar_points_size){
  int thread_index = (blockIdx.x*threads_per_cuda_block + threadIdx.x);
  if(thread_index>=*lidar_points_size){
    return;
  }
  pcl::PointXYZ lidar_point = lidar_points[thread_index];
  Vector3f truncation_start;
  Vector3f truncation_end;

  getTruncationLineEndPoints(lidar_point, lidar_position, truncation_start, truncation_end);
  //equation of line is u+tv
  Vector3f u = truncation_start;
  Vector3f v = truncation_end - truncation_start;

  //get list of voxels traversed
  Vector3f * voxels_traversed = new Vector3f[MAX_VOXELS_TRAVERSED_PER_LIDAR_POINT];
  int * voxels_traversed_size = new int(0); //keep track of number of voxels traversed

  //get voxels traversed on the line segment between truncation_start and truncation_end
  traverseVolume(truncation_start, truncation_end, VOXEL_SIZE, u, v, voxels_traversed, voxels_traversed_size);
  
  //create a pointer with the lidar points data so it can be passed via dynamic parallelism
  Vector3f * lidar_point_p = new Vector3f(lidar_point.x, lidar_point.y, lidar_point.z);

  //update the voxels sdf and weight values processing traversed voxels in parallel
  int num_cuda_blocks = *voxels_traversed_size/threads_per_cuda_block + 1;
  updateVoxelsCuda<<<num_cuda_blocks, threads_per_cuda_block>>>(voxels_traversed, hash_table, block_heap, lidar_point_p, lidar_position, voxels_traversed_size);
  cdpErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  cudaFree(lidar_point_p);
  cudaFree(voxels_traversed);
  cudaFree(voxels_traversed_size);
  return;
}

/**
* determines if the distance between two vectors is within a certain threshold
* @param a vector 1 @param b vector 2
* @param threshold_distance_squared check distance between a and b is less than or equal to this 
* @return true if distance between a and b is less that or equal to the threshold, false otherwise
*/
__device__
inline bool withinDistanceSquared(Vector3f & a, Vector3f & b, float threshold_distance_squared){
  Vector3f diff = b - a;
  float distance_squared  = pow(diff(0), 2) + pow(diff(1), 2) + pow(diff(2), 2);
  if(distance_squared <= threshold_distance_squared){
    return true;
  }
  return false;
}

/**
* Get voxels that are to be published in a voxel block
* @param publish_voxels_pos array to store voxel positions with data (weight > 0)
* @param publish_voxels_size number of voxels added to the publish_voxels_pos array
* @param publish_voxels_data array to store voxel sdf and weight values for voxels with weight > 0
* @param bottom_left_block_pos the coordinates of the bottom left of the block
* @param block block to process voxels for
*/
__global__
void processPublishableVoxelBlockCuda(Vector3f * publish_voxels_pos, int * publish_voxels_size, Voxel * publish_voxels_data, Vector3f * bottom_left_block_pos, VoxelBlock * block){
  int thread_index = blockIdx.x*threads_per_cuda_block + threadIdx.x;
  if(thread_index >= VOXELS_PER_SIDE * VOXELS_PER_SIDE * VOXELS_PER_SIDE){ //check that the thread index is not larger than the num of voxels in a block
    return;
  }

  Voxel voxel = block->voxels[thread_index];
  //if voxel has data to publish
  if(voxel.weight!=0){
    //get coordinates of voxel
    Vector3f v = getVoxelCoordinatesFromLocalVoxelIndex(thread_index, bottom_left_block_pos);
    // add voxel data to publish_voxels_pos and sdf_weight_voxels_vals_d for publishing
    int publish_voxel_index = atomicAdd(&(*publish_voxels_size), 1);
    //check that the arrays are not going to be overflowed by adding the voxel's data
    if(publish_voxel_index<PUBLISH_VOXELS_MAX_SIZE){
      publish_voxels_pos[publish_voxel_index] = v;
      publish_voxels_data[publish_voxel_index] = voxel;
    }
    else{
      printf("ERROR: publish voxels array overflowing\n");
    }
  }
}

/**
* check hashtable in parallel if there is an allocated block at the thread index and if so process the block to retrieve to be published voxels
* @param lidar_position position of lidar in world
* @param hash_table hash table with hash entries for voxel blocks
* @param block_heap block heap storing voxel blocks
* @param publish_voxels_pos array to store voxel positions with weight > 0 within a publishing distance of the lidar position
* @param publish_voxels_size count of voxels to publish. Used for error checking as well to make sure publish_voxels_pos and publish_voxels_data does not overflow
* @param publish_voxels_data array to store voxel sdf and weight values for voxels to be published
*/
__global__
void retrievePublishableVoxelsCuda(Vector3f * lidar_position, HashTable * hash_table, BlockHeap * block_heap, Vector3f * publish_voxels_pos, 
int * publish_voxels_size, Voxel * publish_voxels_data){
  int thread_index = blockIdx.x*threads_per_cuda_block +threadIdx.x;
  if(thread_index >= HASH_TABLE_SIZE) return;
  HashEntry hash_entry = hash_table->hash_entries[thread_index]; //check a hash entry per thread
  if(hash_entry.isFree()){ //continue if the hash entry points to a block in the block heap
    return;
  }
  Vector3f block_pos = hash_entry.position;
  //check the block position is within publishing distance of the lidar position
  if(withinDistanceSquared(block_pos, *lidar_position, PUBLISH_DISTANCE_SQUARED)){
    int block_heap_pos = hash_entry.block_heap_pos;
    Vector3f * bottom_left_block_pos = new Vector3f(block_pos(0) - HALF_VOXEL_BLOCK_SIZE, 
    block_pos(1)- HALF_VOXEL_BLOCK_SIZE,
    block_pos(2)- HALF_VOXEL_BLOCK_SIZE);
  
    VoxelBlock * block = &(block_heap->blocks[block_heap_pos]); //get reference to block
    int num_blocks = VOXELS_PER_BLOCK/threads_per_cuda_block + 1;
    //check voxels in parallel in blocks for whether they have data to publish
    processPublishableVoxelBlockCuda<<<num_blocks,threads_per_cuda_block>>>(publish_voxels_pos, publish_voxels_size, publish_voxels_data, bottom_left_block_pos, block);
    cdpErrchk(cudaPeekAtLastError());
    cudaFree(bottom_left_block_pos);
  }
}

/**
* Remove blocks further than garbage collection distance away from lidar point. Save blocks to be removed that are part of linked lists for after
* @param lidar_position position of lidar
* @param hash_table hash table with hash entries pointing to blocks in block heap
* @param block_heap block heap storing voxel blocks
* @param garbage_blocks_size keep track of num of voxel blocks removed
* @param linked_list_garbage_blocks position of voxel blocks to remove that are part of linked lists
* @param linked_list_garbage_blocks_size keep track of num of voxel blocks to remove that are part of linked lists
*/
__global__
void garbageCollectDistantBlocksCuda(Vector3f * lidar_position, HashTable * hash_table, BlockHeap * block_heap, int * garbage_blocks_size, 
Vector3f * linked_list_garbage_blocks, int * linked_list_garbage_blocks_size){
  int thread_index = blockIdx.x*threads_per_cuda_block +threadIdx.x;
  if(thread_index >= HASH_TABLE_SIZE) return;
  HashEntry * hash_entry = &hash_table->hash_entries[thread_index];
  if(hash_entry->isFree()){
    return;
  }
  Vector3f block_pos = hash_entry->position;
  //check if point is far away enough from lidar to remove
  if(!withinDistanceSquared(block_pos, *lidar_position, GARBAGE_COLLECT_DISTANCE_SQUARED))
  {
    size_t hashed_bucket_index = retrieveHashIndexFromPoint(block_pos); //get the bucket the current hash entry hashes to
    size_t thread_bucket_index = thread_index / HASH_ENTRIES_PER_BUCKET; //get the bucket the current hash entry resides in
    if((hashed_bucket_index!=thread_bucket_index) || (hash_entry->offset > 0)){ // if hash entry is in linked list process later
      int index = atomicAdd(&(*linked_list_garbage_blocks_size), 1); //add the block to the linked_list_distant_blocks array
      if(index<GARBAGE_LINKED_LIST_BLOCKS_MAX_SIZE){ //if the linked list array overflows then process additional linked list garbage blocks in another frame
        linked_list_garbage_blocks[index] = block_pos;
      }
      else{
        printf("WARNING: Garbage collect linked list array size overflow\n");
      }
    }
    else{ //if the block is not part of a linked list delete it from the block heap and hash table
      hash_entry->setFree();
      int block_heap_pos = hash_entry->block_heap_pos;
      int free_blocks_insert_index = atomicSub(&(block_heap->block_count), 1); //update the count of blocks in the block heap
      block_heap->free_blocks[free_blocks_insert_index-1] = block_heap_pos; //update the free_blocks list to include the index of the block just removed
      atomicAdd(&(*garbage_blocks_size), 1);
    }
  }
}

/**
* Remove blocks further than garbage collection distance away from lidar point whose hash entries are a part of a linked list sequentially 
* @param hash_table hash table with hash entries pointing to block in the block heap
* @param block_heap block heap storing voxel blocks
* @param linked_list_garbage_blocks list of positions of blocks to be garbage collected that are part of linked list in the hash table
* @param linked_list_garbage_blocks_size the number of blocks in linked_list_garbage_blocks
*/
__global__
void linkedListGarbageCollectCuda(HashTable * hash_table, BlockHeap * block_heap, Vector3f * linked_list_garbage_blocks, int * linked_list_garbage_blocks_size){

  HashEntry * hash_entries = hash_table->hash_entries;
  HashEntry * curr_hash_entry = NULL;
  HashEntry * prev_hash_entry = NULL;
  HashEntry * next_hash_entry = NULL;

  for(int i=0;i<*linked_list_garbage_blocks_size; ++i){ //loop through linked list points and process sequentially so no issues with having to lock other buckets
    Vector3f garbage_block_pos = linked_list_garbage_blocks[i];
    int hashed_bucket_index = retrieveHashIndexFromPoint(garbage_block_pos);
    //initialize to head of linked list
    int curr_index = (hashed_bucket_index + 1) * HASH_ENTRIES_PER_BUCKET - 1;
    curr_hash_entry = &hash_entries[curr_index];
    prev_hash_entry = NULL;

    if(checkFloatingPointVectorsEqual(curr_hash_entry->position, garbage_block_pos, BLOCK_EPSILON)){ //if the garbage_block_pos' hash entry is the head of the linked list
      //update the head of the linked list to the next block
      int prev_head_offset = curr_hash_entry->offset;
      int block_heap_pos = curr_hash_entry->block_heap_pos;
      int next_index = curr_index + prev_head_offset;
      if(next_index >= HASH_TABLE_SIZE){ //if the index would overflow the table size loop to beginning
        next_index %= HASH_TABLE_SIZE;
      }
      next_hash_entry = &hash_entries[next_index];

      Vector3f next_hash_entry_pos = next_hash_entry->position;

      //copy the next hash entry in the linked list to the head of the linked list (last hash entry of the bucket it hashes to) 
      int next_hash_entry_offset = next_hash_entry->offset;
      int next_hash_entry_block_heap_pos = next_hash_entry->block_heap_pos;
      curr_hash_entry->position = next_hash_entry_pos;
      curr_hash_entry->block_heap_pos = next_hash_entry_block_heap_pos;
      if(next_hash_entry_offset!=0){ //check if the next block has additional blocks after it in the linked list. If so, update its offset
        int new_offset = prev_head_offset + next_hash_entry_offset;
        curr_hash_entry->offset = new_offset;
      }
      else{
        curr_hash_entry->offset = 0;
      }
      next_hash_entry->setFree(); //set the hash entry that was pointing to the next hash entry to free since it has been copied to the head of the linked list
      int free_blocks_insert_index = atomicSub(&(block_heap->block_count), 1); //update count of blocks in the block heap
      block_heap->free_blocks[free_blocks_insert_index-1] = block_heap_pos; //update the free_blocks list to include the index of the block just removed
      continue; //move onto the next linked list block to remove
    }

    //move through the linked list of the bucket the garbage_block_pos hashes to till we find its hash entry
    while(!checkFloatingPointVectorsEqual(curr_hash_entry->position, garbage_block_pos, BLOCK_EPSILON)){ //hash entry is a middle or end element of linked list
      curr_index += (int) curr_hash_entry->offset;
      if(curr_index >= HASH_TABLE_SIZE){
        curr_index %= HASH_TABLE_SIZE;
      }
      prev_hash_entry = curr_hash_entry;
      curr_hash_entry = &hash_entries[curr_index];
    }

    int curr_offset = curr_hash_entry->offset;
    int prev_hash_offset = prev_hash_entry->offset;
    int new_offset = prev_hash_offset + curr_offset;

    if(curr_offset > 0){ //if the garbage_block_pos hash entry is in the middle of a linked list
      prev_hash_entry->offset = new_offset; //update the offset of the prev hash entry
      curr_hash_entry->setFree();
      int block_heap_pos = curr_hash_entry->block_heap_pos;
      int free_blocks_insert_index = atomicSub(&(block_heap->block_count), 1); //update count of blocks in the block heap
      block_heap->free_blocks[free_blocks_insert_index-1] = block_heap_pos; //update the free_blocks list to include the index of the block just removed
    }
    else{ //if the garbage_block_pos hash entry is the end of a linked list
      prev_hash_entry->offset = 0;
      curr_hash_entry->setFree();
      int block_heap_pos = curr_hash_entry->block_heap_pos;
      int free_blocks_insert_index = atomicSub(&(block_heap->block_count), 1);
      block_heap->free_blocks[free_blocks_insert_index-1] = block_heap_pos;
    }
  }
}

TSDFHandler::TSDFHandler(){
  tsdf_container = new TSDFContainer();
  cudaMalloc(&publish_voxels_pos_d, sizeof(Vector3f)*PUBLISH_VOXELS_MAX_SIZE);
  cudaMalloc(&publish_voxels_data_d, sizeof(Voxel)*PUBLISH_VOXELS_MAX_SIZE);
}

TSDFHandler::~TSDFHandler(){
  free(tsdf_container);
  cudaFree(publish_voxels_pos_d);
  cudaFree(publish_voxels_data_d);
}

/**
* given a lidar point cloud xyz and sensor position allocate voxel blocks and update voxels within truncation distance of lidar points and return voxels for publishing
* @param point_cloud point cloud to process
* @param lidar_position_h position of lidar
* @param publish_voxels_pos_h array to copy back over voxel positions to be published from GPU
* @param publish_voxels_size_h count of voxels in publish_voxels_pos_h and publish_voxels_data_h
* @param publish_voxels_data_h array to copy back over voxel data to be published from GPU
*/
void TSDFHandler::processPointCloudAndUpdateVoxels(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, Vector3f * lidar_position_h, Vector3f * publish_voxels_pos_h, 
int * publish_voxels_size_h, Voxel * publish_voxels_data_h)
{ 
  std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> lidar_points = point_cloud->points;

  pcl::PointXYZ * lidar_points_h = &lidar_points[0]; //get array reference from vector
  pcl::PointXYZ * lidar_points_d;
  int point_cloud_size_h = point_cloud->size();
  int * lidar_points_size_d;

  cudaMalloc(&lidar_points_d, sizeof(*lidar_points_h)*point_cloud_size_h); //copy lidar points to gpu
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(lidar_points_d, lidar_points_h, sizeof(*lidar_points_h)*point_cloud_size_h, cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  cudaMalloc(&lidar_points_size_d, sizeof(int)); //copy number of lidar points to gpu
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(lidar_points_size_d, &point_cloud_size_h, sizeof(int), cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  Vector3f * lidar_position_d;
  cudaMalloc(&lidar_position_d, sizeof(*lidar_position_h)); //copy lidar position to GPU
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(lidar_position_d, lidar_position_h,sizeof(*lidar_position_h),cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  HashTable * hash_table_d = tsdf_container->getCudaHashTable(); //get reference to gpu side hash table 
  BlockHeap * block_heap_d = tsdf_container->getCudaBlockHeap(); //get reference to gpu side block heap

  allocateVoxelBlocksAndUpdateVoxels(lidar_points_d, lidar_position_d, lidar_points_size_d, point_cloud_size_h, hash_table_d, block_heap_d);

  retrievePublishableVoxels(lidar_position_d, publish_voxels_pos_h, publish_voxels_size_h, publish_voxels_data_h, hash_table_d, block_heap_d);

  garbageCollectDistantBlocks(lidar_position_d, hash_table_d, block_heap_d);

  cudaFree(lidar_points_size_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaFree(lidar_points_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaFree(lidar_position_d);
  gpuErrchk(cudaPeekAtLastError());

}

/**
* allocate voxel blocks and voxels within truncation distance of lidar points and update the voxels sdf and weight
* @param lidar_points_d reference to lidar points on gpu
* @param lidar_position_d reference to lidar position on gpu
* @param lidar_points_size_d reference to num lidar points var on gpu
* @param point_cloud_size_h num lidar points var on cpu
* @param hash_table_d gpu side hash table
* @param block_heap_d gpu side block heap
*/
void TSDFHandler::allocateVoxelBlocksAndUpdateVoxels(pcl::PointXYZ * lidar_points_d, Vector3f * lidar_position_d, int * lidar_points_size_d, 
int point_cloud_size_h, HashTable * hash_table_d, BlockHeap * block_heap_d){
  //the max number of voxel blocks that can be allocated during this point cloud frame
  int max_blocks = max_voxel_blocks_traversed_per_lidar_point * point_cloud_size_h;

  Vector3f point_cloud_voxel_blocks_h[max_blocks]; //array to keep track of positions of voxel blocks that are traversed by the lidar points
  Vector3f * point_cloud_voxel_blocks_d;

  cudaMalloc(&point_cloud_voxel_blocks_d, sizeof(*point_cloud_voxel_blocks_h)*max_blocks);
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(point_cloud_voxel_blocks_d, point_cloud_voxel_blocks_h, sizeof(*point_cloud_voxel_blocks_h)*max_blocks,cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  int point_cloud_voxel_blocks_size_h = 0;
  int * point_cloud_voxel_blocks_size_d;

  cudaMalloc(&point_cloud_voxel_blocks_size_d, sizeof(point_cloud_voxel_blocks_size_h));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(point_cloud_voxel_blocks_size_d, &point_cloud_voxel_blocks_size_h, sizeof(point_cloud_voxel_blocks_size_h), cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  int num_cuda_blocks = point_cloud_size_h / threads_per_cuda_block + 1;

  getVoxelBlocks(num_cuda_blocks, lidar_points_d, point_cloud_voxel_blocks_d, point_cloud_voxel_blocks_size_d, lidar_position_d, lidar_points_size_d);

  allocateVoxelBlocks(point_cloud_voxel_blocks_d, point_cloud_voxel_blocks_size_d, hash_table_d, block_heap_d);

  getVoxelsAndUpdate(num_cuda_blocks, lidar_points_d, lidar_position_d, lidar_points_size_d, hash_table_d, block_heap_d);

  cudaFree(point_cloud_voxel_blocks_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaFree(point_cloud_voxel_blocks_size_d);
  gpuErrchk(cudaPeekAtLastError());
}

/**
* Get voxel blocks in truncation distance of lidar points on ray between sensor position and the lidar points
* @param num_cuda_blocks number of cuda blocks to allocate during kernel call
* @param lidar_points_d lidar points on gpu
* @param point_cloud_voxel_blocks_d variable to keep track of voxel blocks traversed on gpu
* @param point_cloud_voxel_blocks_size_d keep track of num voxel blocks traversed on gpu
* @param lidar_position_d lidar position var on gpu
* @param lidar_points_size_d num of lidar points on gpu
*/
void TSDFHandler::getVoxelBlocks(int num_cuda_blocks, pcl::PointXYZ * lidar_points_d, Vector3f * point_cloud_voxel_blocks_d, int * point_cloud_voxel_blocks_size_d, 
Vector3f * lidar_position_d, int * lidar_points_size_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  getVoxelBlocksCuda<<<num_cuda_blocks,threads_per_cuda_block>>>(lidar_points_d, point_cloud_voxel_blocks_d, point_cloud_voxel_blocks_size_d, 
    lidar_position_d, lidar_points_size_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Get Voxel Block Duration: %f\n", milliseconds);
}

/**
* Allocated necessary voxel blocks to the hash table
* @param lidar_points_d lidar points on gpu
* @param point_cloud_voxel_blocks_size_d variable to keep track of voxel blocks traversed on gpu
* @param hash_table_d hash table on gpu
* @param block_heap_d block heap on gpu
*/
void TSDFHandler::allocateVoxelBlocks(Vector3f * lidar_points_d, int * point_cloud_voxel_blocks_size_d, HashTable * hash_table_d, BlockHeap * block_heap_d){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int point_cloud_voxel_blocks_size_h;
  //copy number of voxel blocks to allocate
  cudaMemcpy(&point_cloud_voxel_blocks_size_h, point_cloud_voxel_blocks_size_d, sizeof(point_cloud_voxel_blocks_size_h), cudaMemcpyDeviceToHost);

  //array to keep track of unallocated blocks so in multiple kernel calls further down can keep track of blocks that still need to be allocated
  bool unallocated_blocks_h[point_cloud_voxel_blocks_size_h];
  for(int i=0;i<point_cloud_voxel_blocks_size_h;++i)
  {
    unallocated_blocks_h[i] = 1;
  }

  bool * unallocated_blocks_d;
  cudaMalloc(&unallocated_blocks_d, sizeof(*unallocated_blocks_h)*point_cloud_voxel_blocks_size_h);
  cudaMemcpy(unallocated_blocks_d, unallocated_blocks_h, sizeof(*unallocated_blocks_h)*point_cloud_voxel_blocks_size_h, cudaMemcpyHostToDevice);

  //keep track of number of unallocated blocks still left to be allocated
  int unallocated_blocks_size_h = point_cloud_voxel_blocks_size_h;
  int * unallocated_blocks_size_d;
  cudaMalloc(&unallocated_blocks_size_d, sizeof(unallocated_blocks_size_h));
  cudaMemcpy(unallocated_blocks_size_d, &unallocated_blocks_size_h, sizeof(unallocated_blocks_size_h), cudaMemcpyHostToDevice);

  int num_cuda_blocks = point_cloud_voxel_blocks_size_h / threads_per_cuda_block + 1;
  //call kernel till all blocks are allocated
  while(unallocated_blocks_size_h > 0){ //POSSIBILITY OF INFINITE LOOP if no applicable space is left for an unallocated block even if there is still space left in hash table
    allocateVoxelBlocksCuda<<<num_cuda_blocks,threads_per_cuda_block>>>(lidar_points_d, hash_table_d, block_heap_d, unallocated_blocks_d, 
      point_cloud_voxel_blocks_size_d, unallocated_blocks_size_d);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    cudaMemcpy(&unallocated_blocks_size_h, unallocated_blocks_size_d, sizeof(unallocated_blocks_size_h), cudaMemcpyDeviceToHost);
  }

  //print total num blocks allocated so far
  printNumAllocatedVoxelBlocksCuda<<<1,1>>>(block_heap_d);
  cudaDeviceSynchronize();

  cudaFree(unallocated_blocks_d);
  cudaFree(unallocated_blocks_size_d);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Integrate Voxel Block Duration: %f\n", milliseconds);
}

/**
* update voxels sdf and weight values
* @param num_cuda_blocks number of cuda blocks to allocate during kernel call
* @param lidar_points_d lidar points on gpu
* @param lidar_position_d lidar position on gpu
* @param lidar_points_side_d num lidar points on gpu
* @param hash_table_d hash table on gpu
* @param block_heap_d block heap on gpu
*/
void TSDFHandler::getVoxelsAndUpdate(int & num_cuda_blocks, pcl::PointXYZ * lidar_points_d, Vector3f * lidar_position_d, int * lidar_points_size_d, 
HashTable * hash_table_d, BlockHeap * block_heap_d){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  getVoxelsAndUpdateCuda<<<num_cuda_blocks,threads_per_cuda_block>>>(lidar_points_d, lidar_position_d, hash_table_d, block_heap_d, lidar_points_size_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Update Voxels Duration: %f\n", milliseconds);
}

/**
* get voxels for publishing 
* @param lidar_position_d lidar position on gpu
* @param publish_voxels_pos_h array to copy back voxel positions from gpu for publishing
* @param publish_voxels_size_h number of voxels to be published
* @param publish_voxels_data_h array to copy back voxel sdf and weight values from gpu for publishing
* @param hash_table_d hash table on gpu
* @param block_heap_d block heap on gpu
*/
void TSDFHandler::retrievePublishableVoxels(Vector3f * lidar_position_d, Vector3f * publish_voxels_pos_h, int * publish_voxels_size_h, Voxel * publish_voxels_data_h, 
HashTable * hash_table_d, BlockHeap * block_heap_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int * publish_voxels_size_d; //keep track of # of voxels to publish
  cudaMalloc(&publish_voxels_size_d, sizeof(*publish_voxels_size_h));
  cudaMemcpy(publish_voxels_size_d, publish_voxels_size_h, sizeof(*publish_voxels_size_h), cudaMemcpyHostToDevice);
  int num_cuda_blocks = HASH_TABLE_SIZE / threads_per_cuda_block + 1;
  retrievePublishableVoxelsCuda<<<num_cuda_blocks,threads_per_cuda_block>>>(lidar_position_d, hash_table_d, block_heap_d, 
    publish_voxels_pos_d, publish_voxels_size_d, publish_voxels_data_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  //copy voxel positions to cpu to publish
  cudaMemcpy(publish_voxels_pos_h, publish_voxels_pos_d, sizeof(*publish_voxels_pos_h)*PUBLISH_VOXELS_MAX_SIZE, cudaMemcpyDeviceToHost);
  //copy number of voxels to cpu to publish
  cudaMemcpy(publish_voxels_size_h, publish_voxels_size_d, sizeof(*publish_voxels_size_h), cudaMemcpyDeviceToHost);
  //copy voxel sdf + weight data to cpu to publish
  cudaMemcpy(publish_voxels_data_h, publish_voxels_data_d, sizeof(*publish_voxels_data_h)*PUBLISH_VOXELS_MAX_SIZE, cudaMemcpyDeviceToHost);

  cudaFree(publish_voxels_size_d);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Publish Voxels Duration: %f\n", milliseconds);

}

/**
* Remove blocks further than garbage collection distance so not keeping distant block data
* @param lidar_position_d lidar position on gpu
* @param hash_table_d hash table on gpu
* @param block_heap_d block heap on gpu
*/
void TSDFHandler::garbageCollectDistantBlocks(Vector3f * lidar_position_d, HashTable * hash_table_d, BlockHeap * block_heap_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int garbage_collected_blocks_size_h = 0; //keep track of number of blocks garbage collected
  int * garbage_collected_blocks_size_d;
  cudaMalloc(&garbage_collected_blocks_size_d, sizeof(int));
  cudaMemcpy(garbage_collected_blocks_size_d, &garbage_collected_blocks_size_h, sizeof(garbage_collected_blocks_size_h), cudaMemcpyHostToDevice);

  //keep track of number of blocks to be garbage collected whose hash entries are part of a linked list in the hash table
  int linked_list_garbage_blocks_size_h = 0;
  int * linked_list_garbage_blocks_size_d;
  cudaMalloc(&linked_list_garbage_blocks_size_d, sizeof(int));
  cudaMemcpy(linked_list_garbage_blocks_size_d, &linked_list_garbage_blocks_size_h, sizeof(linked_list_garbage_blocks_size_h), cudaMemcpyHostToDevice);

  //array to keep track of voxel blocks positions to be garbage collected whose hash entries are part of a linked list in the hash table
  Vector3f linked_list_garbage_blocks_h[GARBAGE_LINKED_LIST_BLOCKS_MAX_SIZE];
  Vector3f * linked_list_garbage_blocks_d;
  cudaMalloc(&linked_list_garbage_blocks_d, sizeof(*linked_list_garbage_blocks_h)*GARBAGE_LINKED_LIST_BLOCKS_MAX_SIZE);
  cudaMemcpy(linked_list_garbage_blocks_d, linked_list_garbage_blocks_h, sizeof(*linked_list_garbage_blocks_h)*GARBAGE_LINKED_LIST_BLOCKS_MAX_SIZE, cudaMemcpyHostToDevice);

  int num_cuda_blocks = HASH_TABLE_SIZE / threads_per_cuda_block + 1;
  garbageCollectDistantBlocksCuda<<<num_cuda_blocks, threads_per_cuda_block>>>(lidar_position_d, hash_table_d, block_heap_d, garbage_collected_blocks_size_d, 
    linked_list_garbage_blocks_d, linked_list_garbage_blocks_size_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(&garbage_collected_blocks_size_h, garbage_collected_blocks_size_d, sizeof(garbage_collected_blocks_size_h), cudaMemcpyDeviceToHost);
  cudaMemcpy(&linked_list_garbage_blocks_size_h, linked_list_garbage_blocks_size_d, sizeof(linked_list_garbage_blocks_size_h), cudaMemcpyDeviceToHost);

  //remove blocks to be garbage collected whose hash entries are part of a linked list in the hash table sequentially
  linkedListGarbageCollectCuda<<<1,1>>>(hash_table_d, block_heap_d, linked_list_garbage_blocks_d, linked_list_garbage_blocks_size_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(&linked_list_garbage_blocks_size_h, linked_list_garbage_blocks_size_d, sizeof(linked_list_garbage_blocks_size_h), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Garbage Collection Duration: %f\n", milliseconds);
  printf("Total Blocks Garbage Collected: %d\n", garbage_collected_blocks_size_h + linked_list_garbage_blocks_size_h);

  cudaFree(garbage_collected_blocks_size_d);
  cudaFree(linked_list_garbage_blocks_size_d);
  cudaFree(linked_list_garbage_blocks_d);
}

/**
* set gpu global vars and set max_voxel_blocks_traversed_per_lidar_point
* @param voxel_size_h user specified voxel size
* @param truncation_distance_h user specified truncation distance
* @param max_weight_h user specified max weight
* @param publish_distance_squared_h user specified publish distance squared
* @param garbage_collect_distance_squared_h user specified garbage collect distance squared
*/
void initGlobalVars(float voxel_size_h, float truncation_distance_h, float max_weight_h, float publish_distance_squared_h, float garbage_collect_distance_squared_h){
  cudaMemcpyToSymbol(VOXEL_SIZE, &voxel_size_h, sizeof(float));
  cudaMemcpyToSymbol(TRUNCATION_DISTANCE, &truncation_distance_h, sizeof(float));
  cudaMemcpyToSymbol(MAX_WEIGHT, &max_weight_h, sizeof(float));
  cudaMemcpyToSymbol(PUBLISH_DISTANCE_SQUARED, &publish_distance_squared_h, sizeof(float));
  cudaMemcpyToSymbol(GARBAGE_COLLECT_DISTANCE_SQUARED, &garbage_collect_distance_squared_h, sizeof(float));

  float half_voxel_size = voxel_size_h / 2;
  cudaMemcpyToSymbol(HALF_VOXEL_SIZE, &half_voxel_size, sizeof(float));

  float voxel_block_size = voxel_size_h * VOXELS_PER_SIDE;
  cudaMemcpyToSymbol(VOXEL_BLOCK_SIZE, &voxel_block_size, sizeof(float));

  float half_voxel_block_size = voxel_block_size / 2;
  cudaMemcpyToSymbol(HALF_VOXEL_BLOCK_SIZE, &half_voxel_block_size, sizeof(float));

  float block_epsilon = voxel_block_size / 4;
  cudaMemcpyToSymbol(BLOCK_EPSILON, &block_epsilon, sizeof(float));

  float voxel_epsilon = voxel_size_h / 4;
  cudaMemcpyToSymbol(VOXEL_EPSILON, &voxel_epsilon, sizeof(float));

  int max_voxels_traversed_per_lidar_point = ceil(truncation_distance_h * 2 / voxel_size_h) * 4;
  cudaMemcpyToSymbol(MAX_VOXELS_TRAVERSED_PER_LIDAR_POINT, &max_voxels_traversed_per_lidar_point, sizeof(int));

  //set max_voxel_blocks_traversed_per_lidar_point in relation to truncation distance and voxel_block_size
  max_voxel_blocks_traversed_per_lidar_point = ceil(truncation_distance_h * 2 / voxel_block_size) * 4;

}