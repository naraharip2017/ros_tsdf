#include "cuda/tsdf_handler.cuh"

//used to determine num blocks when executing cuda kernel
const int threads_per_cuda_block = 128;

//error function for cpu called after kernel calls
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
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

//used for debugging, current index represents num of heap blocks in heap
__global__
void printHashTableAndBlockHeap(HashTable * hash_table_d, BlockHeap * block_heap_d){
  // HashEntry * hash_entries = hash_table_d->hash_entries;
  // for(size_t i=0;i<NUM_BUCKETS; ++i){
  //   printf("Bucket: %lu\n", (unsigned long)i);
  //   for(size_t it = 0; it<HASH_ENTRIES_PER_BUCKET; ++it){
  //     HashEntry hash_entry = hash_entries[it+i*HASH_ENTRIES_PER_BUCKET];
  //     Vector3f position = hash_entry.position;
  //     if (hash_entry.isFree()){
  //       printf("  Hash Entry with   Position: (N,N,N)   Offset: %d   Pointer: %d\n", hash_entry.offset, hash_entry.pointer);
  //     }
  //     else{
  //       printf("  Hash Entry with   Position: (%f,%f,%f)   Offset: %d   Pointer: %d\n", position(0), position(1), position(2), hash_entry.offset, hash_entry.pointer);
  //     }
  //   }
  //   printf("%s\n", "--------------------------------------------------------");
  // }

  // printf("Block Heap Free List: ");
  // int * freeBlocks = block_heap_d->free_blocks;
  // for(size_t i = 0; i<NUM_HEAP_BLOCKS; ++i){
  //   printf("%d  ", freeBlocks[i]);
  // }
  // printf("\n");
  printf("Current Block Heap Free Index: %d\n", block_heap_d->current_index);
}

/*
* given voxel block center point returns hashed bucket index for the block
*/
__device__
size_t retrieveHashIndexFromPoint(Vector3f point){
  return abs((((int)point(0)*PRIME_ONE) ^ ((int)point(1)*PRIME_TWO) ^ ((int)point(2)*PRIME_THREE)) % NUM_BUCKETS);
}

/*
* calculates floor in scale given
*/
__device__ 
float floorFun(float x, float scale){
  return floor(x*scale) / scale;
}

/*
* Given world point return center of volume given by volume_size
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

/*
* Check if two Vector3f are equal
*/
__device__
bool checkFloatingPointVectorsEqual(Vector3f A, Vector3f B, float epsilon){
  Vector3f diff = A-B;
  //have to use an epsilon value due to floating point precision errors
  if((fabs(diff(0)) < epsilon) && (fabs(diff(1)) < epsilon) && (fabs(diff(2)) < epsilon))
    return true;

  return false;
}

/*
* calculate the line between point_d and origin_transformed_d then get points truncation distance away from point_d on the line. Set truncation_start and truncation_end to those points
* truncation_start will be closer to origin
*/
__device__
inline void getTruncationLineEndPoints(pcl::PointXYZ & point_d, Vector3f * origin_transformed_d, Vector3f & truncation_start, Vector3f & truncation_end){
  Vector3f u = *origin_transformed_d;
  Vector3f point_d_vector(point_d.x, point_d.y, point_d.z);
  Vector3f v = point_d_vector - u; //direction
  //equation of line is u+tv
  float v_mag = sqrt(pow(v(0), 2) + pow(v(1),2) + pow(v(2), 2));
  Vector3f v_normalized = v / v_mag;
  truncation_start = point_d_vector - TRUNCATION_DISTANCE*v_normalized;
  
  truncation_end = point_d_vector + TRUNCATION_DISTANCE*v_normalized;

  //set truncation_start to whichever point is closer to the origin
  float distance_t_start_origin = pow(truncation_start(0) - u(0), 2) + pow(truncation_start(1) - u(1),2) + pow(truncation_start(2) - u(2), 2);
  float distance_t_end_origin = pow(truncation_end(0) - u(0), 2) + pow(truncation_end(1) - u(1),2) + pow(truncation_end(2) - u(2), 2);

  if(distance_t_end_origin < distance_t_start_origin){
    Vector3f temp = truncation_start;
    truncation_start = truncation_end;
    truncation_end = temp;
  }
}

/*
* Given the truncation start volume, volume size, and ray defined by u and t traverse the world till reaching truncation_end_vol
* Read for more information: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
*/
__device__
inline void traverseVolume(Vector3f & truncation_start_vol, Vector3f & truncation_end_vol, const float & volume_size, Vector3f & u, Vector3f & v, Vector3f * traversed_vols, int * traversed_vols_size){
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

  // printf("-----------------------------------------------------\n");
  // printf("truncation_vol_start_center: (%f,%f,%f)\n", truncation_vol_start_center(0), truncation_vol_start_center(1), truncation_vol_start_center(2));
  // printf("truncation_vol_end_center: (%f,%f,%f)\n", truncation_vol_end_center(0), truncation_vol_end_center(1), truncation_vol_end_center(2));
  // printf("-----------------------------------------------------\n");
  int insert_index;
  int current_index = 0;
  while(!checkFloatingPointVectorsEqual(current_vol_center, truncation_vol_end_center, epsilon)){
    //add traversed volume to list of traversed volume and update size atomically
    insert_index = atomicAdd(&(*traversed_vols_size), 1);
    traversed_vols[insert_index] = current_vol_center;
    // printf("current_vol: (%f,%f,%f)\n", current_vol(0), current_vol(1), current_vol(2));
    // printf("current_vol_center: (%f,%f,%f)\n", current_vol_center(0), current_vol_center(1), current_vol_center(2));
    // printf("tMax_x: %f, tMax_y: %f, tMax_z:%f\n", tMax_x, tMax_y, tMax_z);
    // printf("deltaX: %f, deltaY: %f, deltaZ:%f\n", tDelta_x, tDelta_y, tDelta_z);
    current_index ++;
    if(current_index > 30){
      printf("current_vol_center: (%f,%f,%f), end_center: (%f,%f,%f), start_center: (%f,%f,%f), end: (%f,%f,%f), start: (%f,%f,%f)\n", current_vol_center(0), current_vol_center(1), current_vol_center(2), truncation_vol_end_center(0), truncation_vol_end_center(1), truncation_vol_end_center(2), truncation_vol_start_center(0), truncation_vol_start_center(1), truncation_vol_start_center(2), truncation_end_vol(0), truncation_end_vol(1), truncation_end_vol(2), truncation_start_vol(0), truncation_start_vol(1), truncation_start_vol(2));

      break;
    }

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
    // printf("diff: (%f,%f,%f)\n",diff(0), diff(1), diff(2));
    if((diff(0) < volume_size_minus_epsilon && diff(1) < volume_size_minus_epsilon && diff(2) < volume_size_minus_epsilon) || (diff(0) > volume_size_plus_epsilon || diff(1) > volume_size_plus_epsilon || diff(2) > volume_size_plus_epsilon)){
      return;
    }
    // printf("-----------------------------------------------------\n");
  }      

  //add traversed volume to list of traversed volume and update size atomically
  insert_index = atomicAdd(&(*traversed_vols_size), 1);
  traversed_vols[insert_index] = current_vol_center;
  // printf("current_vol: (%f,%f,%f)\n", current_vol(0), current_vol(1), current_vol(2));
  // printf("current_vol_center: (%f,%f,%f)\n", current_vol_center(0), current_vol_center(1), current_vol_center(2));

}

/*
* Kernel will get voxel blocks along ray between origin and a point in lidar cloud within truncation distance of point
*/
__global__
void getVoxelBlocksForPoint(pcl::PointXYZ * points_d, Vector3f * point_cloud_voxel_blocks_d, int * pointer_d, Vector3f * origin_transformed_d, int * size_d){
  int thread_index = (blockIdx.x*threads_per_cuda_block + threadIdx.x);
  if(thread_index>=*size_d){
    return;
  }
  pcl::PointXYZ point_d = points_d[thread_index];
  Vector3f truncation_start;
  Vector3f truncation_end;

  getTruncationLineEndPoints(point_d, origin_transformed_d, truncation_start, truncation_end);
  //equation of line is u+tv
  Vector3f u = truncation_start;
  Vector3f v = truncation_end - truncation_start;

  traverseVolume(truncation_start, truncation_end, VOXEL_BLOCK_SIZE, u, v, point_cloud_voxel_blocks_d, pointer_d);
  return;
}

/*
* Given block coordinates return the position of the block in the block heap or return -1 if not allocated
*/
__device__ 
int getBlockPositionForBlockCoordinates(Vector3f & voxel_block_coordinates, size_t & bucket_index, size_t & current_global_index, HashEntry * hash_entries){

  HashEntry hash_entry;

  //check the hashed bucket for the block
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    hash_entry = hash_entries[current_global_index+i];
    if(checkFloatingPointVectorsEqual(hash_entry.position, voxel_block_coordinates, BLOCK_EPSILON)){
      return hash_entry.pointer;
    }
  }

  current_global_index+=HASH_ENTRIES_PER_BUCKET-1;

  //check the linked list if necessary
  while(hash_entry.offset!=0){
    int offset = hash_entry.offset;
    current_global_index+=offset;
    if(current_global_index>=HASH_TABLE_SIZE){
      current_global_index %= HASH_TABLE_SIZE;
    }
    hash_entry = hash_entries[current_global_index];
    if(checkFloatingPointVectorsEqual(hash_entry.position, voxel_block_coordinates, BLOCK_EPSILON)){
      return hash_entry.pointer;
    }
  }

  //block not allocated in hashTable
  return -1;
}

/*
* If allocating a new block then first attempt to insert in the bucket it hashed to
*/
__device__
inline bool attempHashedBucketVoxelBlockCreation(size_t & hashed_bucket_index, BlockHeap * block_heap_d, Vector3f & voxel_block_coordinates, HashEntry * hash_entries){
  //get position of beginning of hashed bucket
  size_t insert_current_global_index = hashed_bucket_index * HASH_ENTRIES_PER_BUCKET;
  //loop through bucket and insert if there is a free space
  for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET; ++i){
    if(hash_entries[insert_current_global_index+i].isFree()){ 
      //get next free position in block heap
      int block_heap_free_index = atomicAdd(&(block_heap_d->current_index), 1);
      VoxelBlock * alloc_block = new VoxelBlock();
      HashEntry * alloc_block_hash_entry = new HashEntry(voxel_block_coordinates, block_heap_free_index);
      block_heap_d->blocks[block_heap_free_index] = *alloc_block;
      hash_entries[insert_current_global_index+i] = *alloc_block_hash_entry;
      cudaFree(alloc_block);
      cudaFree(alloc_block_hash_entry);
      return true;
    }
  }
  //return false if no free position in hashed bucket
  return false;
} 

/*
* Attempt to allocated a new block into the linked list of the bucket it hashed to if there is no space in the bucket
*/
__device__
inline bool attemptLinkedListVoxelBlockCreation(size_t & hashed_bucket_index, BlockHeap * block_heap_d, HashTable * hash_table_d, size_t & insert_bucket_index, size_t & end_linked_list_pointer, Vector3f & voxel_block_coordinates, HashEntry * hash_entries){
  size_t insert_current_global_index;
  //only try to insert into other buckets until we get the block's hashed bucket which has already been tried
  while(insert_bucket_index!=hashed_bucket_index){
    //check that can get lock of the bucket attempting to insert the block
    if(!atomicCAS(&hash_table_d->mutex[insert_bucket_index], 0, 1)){
      insert_current_global_index = insert_bucket_index * HASH_ENTRIES_PER_BUCKET;
      //loop through the insert bucket (not including the last slot which is reserved for linked list start of that bucket) checking for a free space
      for(size_t i=0; i<HASH_ENTRIES_PER_BUCKET-1; ++i){
        if(hash_entries[insert_current_global_index+i].isFree() ){ 
            int block_heap_free_index = atomicAdd(&(block_heap_d->current_index), 1);
            VoxelBlock * alloc_block = new VoxelBlock();
            HashEntry * alloc_block_hash_entry = new HashEntry(voxel_block_coordinates, block_heap_free_index);
            block_heap_d->blocks[block_heap_free_index] = *alloc_block;
            size_t insertPos = insert_current_global_index + i;
            hash_entries[insertPos] = *alloc_block_hash_entry;
            cudaFree(alloc_block);
            cudaFree(alloc_block_hash_entry);
            //set offset value for last hash_entry in the linked list
            if(insertPos > end_linked_list_pointer){
              hash_entries[end_linked_list_pointer].offset = insertPos - end_linked_list_pointer;
            }
            else{
              hash_entries[end_linked_list_pointer].offset = HASH_TABLE_SIZE - end_linked_list_pointer + insertPos;
            }
            return true;
        }
      }
      //if no free space in insert bucket then release lock
      atomicExch(&hash_table_d->mutex[insert_bucket_index], 0);
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

/*
* Given list of voxelBlocks check that the block is allocated in the hashtable and if not do so. Unallocated points in the current frame due to not reveiving lock or lack of
* space in hashTable then return to be processed in next frame
*/
__global__
void allocateVoxelBlocks(Vector3f * voxel_blocks_d, HashTable * hash_table_d, BlockHeap * block_heap_d, bool * unallocated_points_d, int * size_d, int * unallocated_points_count_d)
{
  int thread_index = (blockIdx.x*threads_per_cuda_block + threadIdx.x);
  //check that the thread is a valid index in voxel_blocks_d and that the block is still unallocated
  if(thread_index>=*size_d || (unallocated_points_d[thread_index]==0)){
    return;
  }
  Vector3f voxel_block = voxel_blocks_d[thread_index];

  size_t hashed_bucket_index = retrieveHashIndexFromPoint(voxel_block);
  //beginning of the hashedBucket in hashTable
  size_t current_global_index = hashed_bucket_index * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hash_entries = hash_table_d->hash_entries;

  int block_position = getBlockPositionForBlockCoordinates(voxel_block, hashed_bucket_index, current_global_index, hash_entries);

  //block is already allocated then return
  if(block_position!=-1){
    unallocated_points_d[thread_index] = 0;
    atomicSub(unallocated_points_count_d, 1);
    return;
  }

  //attempt to get lock for hashed bucket for potential insert
  if(!atomicCAS(&hash_table_d->mutex[hashed_bucket_index], 0, 1)){
    //try to insert block into the bucket it hashed to and return if possible
    if(attempHashedBucketVoxelBlockCreation(hashed_bucket_index, block_heap_d, voxel_block, hash_entries)) {
      //the block is allocated so set it to allocated so it is not processed again in another frame
      unallocated_points_d[thread_index] = 0;
      //subtract 1 from count of unallocated blocks
      atomicSub(unallocated_points_count_d, 1);
      //free lock for hashed bucket
      atomicExch(&hash_table_d->mutex[hashed_bucket_index], 0);
      return;
    }

    //start searching for a free position in the next bucket
    size_t insert_bucket_index = hashed_bucket_index + 1;
    //set insertBucket to first bucket if overflow the hash table size
    if(insert_bucket_index == NUM_BUCKETS){
      insert_bucket_index = 0;
    }

    //Note: current global index will point to end of linked list which includes hashed bucket if no linked list
    //index to the bucket which contains the end of the linked list for the hashed bucket of the block
    size_t end_linked_list_bucket = current_global_index / HASH_ENTRIES_PER_BUCKET;

    bool have_end_linked_list_bucket_lock = true;

    //if end of linked list is in different bucket than hashed bucket try to get the lock for the end of the linked list
    if(end_linked_list_bucket!=hashed_bucket_index){
      //release lock of the hashedBucket
      atomicExch(&hash_table_d->mutex[hashed_bucket_index], 0);
      //attempt to get lock of bucket with end of linked list
      have_end_linked_list_bucket_lock = !atomicCAS(&hash_table_d->mutex[end_linked_list_bucket], 0, 1);
    }

    if(have_end_linked_list_bucket_lock){
      //try to insert block into the linked list for it's hashed bucket
      if(attemptLinkedListVoxelBlockCreation(hashed_bucket_index, block_heap_d, hash_table_d, insert_bucket_index, current_global_index, voxel_block, hash_entries)){
        //the block is allocated so set it to allocated so it is not processed again in another frame
        unallocated_points_d[thread_index] = 0;
        //subtract 1 from count of unallocated blocks
        atomicSub(unallocated_points_count_d, 1); 
        //free the lock of the end of linked list bucket and insert bucket
        atomicExch(&hash_table_d->mutex[end_linked_list_bucket], 0);
        atomicExch(&hash_table_d->mutex[insert_bucket_index], 0);
      }
      else{
        //free the lock of the end of linked list bucket
        atomicExch(&hash_table_d->mutex[end_linked_list_bucket], 0);
      }
      return;
    }
  }
}

/*
* given diff between a voxel coordinate and the bottom left block of the block it is in return the position of the voxel in the blocks voxel array
*/
__device__
size_t getLocalVoxelIndex(Vector3f diff){
  diff /= VOXEL_SIZE;
  return floor(diff(0)) + (floor(diff(1)) * VOXEL_PER_SIDE) + (floor(diff(2)) * VOXEL_PER_SIDE * VOXEL_PER_SIDE);
}


/*
* return magnitude of vector
*/
__device__
inline float getMagnitude(Vector3f vector){
  return sqrt(pow(vector(0),2) + pow(vector(1),2) + pow(vector(2),2));
}

/*
* return dot product of two vectors
*/
__device__
inline float dotProduct(Vector3f a, Vector3f b){
  return a(0)*b(0) + a(1)*b(1) + a(2)*b(2);
}

/*
* given voxel coordinates, position of lidar points and sensor origin calculate the distance update for the voxel
* http://helenol.github.io/publications/iros_2017_voxblox.pdf for more info
*/
__device__
inline float getDistanceUpdate(Vector3f voxel_coordinates, Vector3f point_d, Vector3f origin){
  //x:center of current voxel   p:position of lidar point   s:sensor origin
  Vector3f lidarOriginDiff = point_d - origin; //p-s
  Vector3f lidarVoxelDiff = point_d - voxel_coordinates; //p-x
  float magnitudeLidarVoxelDiff = getMagnitude(lidarVoxelDiff);
  float dotProd = dotProduct(lidarOriginDiff, lidarVoxelDiff);
  if(dotProd < 0){
    magnitudeLidarVoxelDiff *=-1;
  }

  return magnitudeLidarVoxelDiff;
}

__device__
inline float getWeightUpdate(float distance){
  return 1/(fabs(distance) + 1);
  // if(distance >= 0){
  //   return 1/(distance + 1);
  // }
  // else{
  //   return (1/pow(fabs(distance) + 1, 2));
  // }
}

/*
* update voxels sdf and weight that are within truncation distance of the line between origin and point_d
*/
__global__
void updateVoxels(Vector3f * voxels, HashTable * hash_table_d, BlockHeap * block_heap_d, Vector3f * point_d, Vector3f * origin, int * size){
  int thread_index = blockIdx.x*threads_per_cuda_block + threadIdx.x;
  if(thread_index >= * size){
    return;
  }
  Vector3f voxel_coordinates = voxels[thread_index];
  Vector3f voxel_block_coordinates = getVolumeCenterFromPoint(voxel_coordinates, VOXEL_BLOCK_SIZE);

  size_t bucketIndex = retrieveHashIndexFromPoint(voxel_block_coordinates);
  size_t current_global_index = bucketIndex * HASH_ENTRIES_PER_BUCKET;
  HashEntry * hash_entries = hash_table_d->hash_entries;

  //todo: make a global vector with half voxel block size values
  int voxel_block_heap_position = getBlockPositionForBlockCoordinates(voxel_block_coordinates, bucketIndex, current_global_index, hash_entries);
  Vector3f voxel_block_bottom_left_coordinates;
  voxel_block_bottom_left_coordinates(0) = voxel_block_coordinates(0)-HALF_VOXEL_BLOCK_SIZE;
  voxel_block_bottom_left_coordinates(1) = voxel_block_coordinates(1)-HALF_VOXEL_BLOCK_SIZE;
  voxel_block_bottom_left_coordinates(2) = voxel_block_coordinates(2)-HALF_VOXEL_BLOCK_SIZE;
  //get local positin of voxel in its block
  size_t local_voxel_index = getLocalVoxelIndex(voxel_coordinates - voxel_block_bottom_left_coordinates);

  VoxelBlock * block = &(block_heap_d->blocks[voxel_block_heap_position]);
  Voxel* voxel = &(block->voxels[local_voxel_index]);
  int * mutex = &(block->mutex[local_voxel_index]);

  float distance = getDistanceUpdate(voxel_coordinates, *point_d, *origin);
  float weight = getWeightUpdate(distance);
  float weight_times_distance = weight * distance;

  //get lock for voxel
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
      //free voxel
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

/*
* Get voxels traversed on ray between origin and a lidar point within truncation distance of lidar point
*/
__global__
void getVoxelsForPoint(pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, HashTable * hash_table_d, BlockHeap * block_heap_d, int * size_d){
  int thread_index = (blockIdx.x*threads_per_cuda_block + threadIdx.x);
  // if(thread_index>0){
  //   return;
  // }
  if(thread_index>=*size_d){
    return;
  }
  pcl::PointXYZ point_d = points_d[thread_index];
  Vector3f truncation_start;
  Vector3f truncation_end;


  getTruncationLineEndPoints(point_d, origin_transformed_d, truncation_start, truncation_end);
  //equation of line is u+tv
  // truncation_start(0) = 127.499992;
  // truncation_start(1) = 8.746421;
  // truncation_start(2) = -2.210701;
  // truncation_end(0) = 129.193176;
  // truncation_end(1) = 15.148551;
  // truncation_end(2) = 2.277709;
  Vector3f u = truncation_start;
  Vector3f v = truncation_end - truncation_start;

  //get list of voxels traversed 
  Vector3f * voxels = new Vector3f[100]; //todo: hardcoded -> set in terms of truncation distance and voxel size
  int * size = new int(0);

  traverseVolume(truncation_start, truncation_end, VOXEL_SIZE, u, v, voxels, size);
  

  Vector3f * lidarPoint = new Vector3f(point_d.x, point_d.y, point_d.z);

  //update the voxels sdf and weight values
  int num_cuda_blocks = *size/threads_per_cuda_block + 1;
  updateVoxels<<<num_cuda_blocks, threads_per_cuda_block>>>(voxels, hash_table_d, block_heap_d, lidarPoint, origin_transformed_d, size);
  cdpErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  cudaFree(lidarPoint);
  cudaFree(voxels);
  cudaFree(size);
  return;
}

/*
* determines if voxel is within publishing distance of drone
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

/*
* Get voxels that are occupied in a voxel block
*/
__global__
void processOccupiedVoxelBlock(Vector3f * occupied_voxels, int * index, Voxel * sdf_weight_voxel_vals_d, Vector3f * position, VoxelBlock * block){
  int thread_index = blockIdx.x*threads_per_cuda_block + threadIdx.x;
  if(thread_index >= VOXEL_PER_SIDE * VOXEL_PER_SIDE * VOXEL_PER_SIDE){
    return;
  }

  int voxelIndex = thread_index;
  Voxel voxel = block->voxels[thread_index];
  //if voxel is occupied
  if(voxel.weight!=0){
    //get coordinates of voxel
    float z = voxelIndex / (VOXEL_PER_SIDE * VOXEL_PER_SIDE);
    voxelIndex -= z*VOXEL_PER_SIDE*VOXEL_PER_SIDE;
    float y = voxelIndex / VOXEL_PER_SIDE;
    voxelIndex -= y*VOXEL_PER_SIDE;
    float x = voxelIndex;

    Vector3f positionVec = * position;
    float x_coord = x * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(0);
    float y_coord = y * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(1);
    float z_coord = z * VOXEL_SIZE + HALF_VOXEL_SIZE + positionVec(2);
  
    Vector3f v(x_coord, y_coord, z_coord);
    //if within publish distance add voxel to list of occupied voxel and its position
    int occupied_voxel_index = atomicAdd(&(*index), 1);
    if(occupied_voxel_index<OCCUPIED_VOXELS_SIZE){
      occupied_voxels[occupied_voxel_index] = v;
      sdf_weight_voxel_vals_d[occupied_voxel_index] = voxel;
    }
  }
}

/*
* check hashtable in parallel if there is an allocated block at the thread index and if so process the block to retrieve occupied voxels
*/
__global__
void publish_occupied_voxelsCuda(Vector3f * origin_transformed_d, HashTable * hash_table_d, BlockHeap * block_heap_d, Vector3f * occupied_voxels, int * index, Voxel * sdf_weight_voxel_vals_d){
  int thread_index = blockIdx.x*threads_per_cuda_block +threadIdx.x;
  if(thread_index >= HASH_TABLE_SIZE) return;
  HashEntry hash_entry = hash_table_d->hash_entries[thread_index];
  if(hash_entry.isFree()){
    return;
  }
  Vector3f block_pos = hash_entry.position;
  if(withinDistanceSquared(block_pos, *origin_transformed_d, PUBLISH_DISTANCE_SQUARED)){
    int pointer = hash_entry.pointer;
    Vector3f * bottom_left_block_pos = new Vector3f(block_pos(0) - HALF_VOXEL_BLOCK_SIZE, 
    block_pos(1)- HALF_VOXEL_BLOCK_SIZE,
    block_pos(2)- HALF_VOXEL_BLOCK_SIZE);
  
    VoxelBlock * block = &(block_heap_d->blocks[pointer]);
    int size = VOXEL_PER_SIDE * VOXEL_PER_SIDE * VOXEL_PER_SIDE;
    int num_blocks = size/threads_per_cuda_block + 1;
    processOccupiedVoxelBlock<<<num_blocks,threads_per_cuda_block>>>(occupied_voxels, index, sdf_weight_voxel_vals_d, bottom_left_block_pos, block);
    cdpErrchk(cudaPeekAtLastError());
    cudaFree(bottom_left_block_pos);
  }
}

/*
* Remove blocks further than garbage collection distance away from lidar point
*/
__global__
void garbageCollect(Vector3f * origin_transformed_d, HashTable * hash_table_d, BlockHeap * block_heap_d, int * removed_blocks_count, Vector3f * linked_list_distanct_blocks, int * linked_list_distance_block_count){
  int thread_index = blockIdx.x*threads_per_cuda_block +threadIdx.x;
  if(thread_index >= HASH_TABLE_SIZE) return;
  HashEntry * hash_entry = &hash_table_d->hash_entries[thread_index];
  if(hash_entry->isFree()){
    return;
  }
  Vector3f block_pos = hash_entry->position;
  //check if point is far away from lidar to remove
  if(!withinDistanceSquared(block_pos, *origin_transformed_d, GARBAGE_COLLECT_DISTANCE_SQUARED))
  {
    size_t hashed_bucket_index = retrieveHashIndexFromPoint(block_pos);
    size_t thread_bucket_index = thread_index / HASH_ENTRIES_PER_BUCKET;
    if((hashed_bucket_index!=thread_bucket_index) || (hash_entry->offset > 0)){ // if hash entry is in linked list process later
      int index = atomicAdd(&(*linked_list_distance_block_count), 1);
      if(index<MAX_LINKED_LIST_BLOCKS){
        linked_list_distanct_blocks[index] = block_pos;
      }
    }
    else{
      hash_entry->setFree();
      int block_heap_pos = hash_entry->pointer;
      int free_blocks_insert_index = atomicSub(&(block_heap_d->current_index), 1);
      block_heap_d->free_blocks[free_blocks_insert_index-1] = block_heap_pos;
      atomicAdd(&(*removed_blocks_count), 1);
    }
  }
  hash_entry = NULL;
  delete hash_entry;
}

/*
* Remove blocks further than garbage collection distance away from lidar point whose hash entries are a part of a linked list
*/
__global__
void linkedListGarbageCollect(HashTable * hash_table_d, BlockHeap * block_heap_d, Vector3f * linked_list_distanct_blocks, int * linked_list_distance_block_count){

  HashEntry * hash_entries = hash_table_d->hash_entries;
  HashEntry * curr_hash_entry = NULL;
  HashEntry * prev_hash_entry = NULL;
  HashEntry * next_hash_entry = NULL;

  for(int i=0;i<*linked_list_distance_block_count; ++i){ //loop through linked list points and process sequentially so no issues with having to lock other buckets
    Vector3f remove_block_pos = linked_list_distanct_blocks[i];
    int hashed_bucket_index = retrieveHashIndexFromPoint(remove_block_pos);
    //initialize to head of linked list
    int curr_index = (hashed_bucket_index + 1) * HASH_ENTRIES_PER_BUCKET - 1;
    curr_hash_entry = &hash_entries[curr_index];
    prev_hash_entry = NULL;

    if(checkFloatingPointVectorsEqual(curr_hash_entry->position, remove_block_pos, BLOCK_EPSILON)){ //if the hash entry is the head of the linked list
      int prev_head_offset = curr_hash_entry->offset;
      int block_heap_pos = curr_hash_entry->pointer;
      int next_index = curr_index + prev_head_offset;
      if(next_index >= HASH_TABLE_SIZE){
        next_index %= HASH_TABLE_SIZE;
      }
      next_hash_entry = &hash_entries[next_index];

      Vector3f next_hash_entry_pos = next_hash_entry->position;

      int next_hash_entry_offset = next_hash_entry->offset;
      int next_hash_entry_pointer = next_hash_entry->pointer;
      curr_hash_entry->position = next_hash_entry_pos;
      curr_hash_entry->pointer = next_hash_entry_pointer;
      if(next_hash_entry_offset!=0){
        int new_offset = prev_head_offset + next_hash_entry_offset;
        curr_hash_entry->offset = new_offset;
      }
      else{
        curr_hash_entry->offset = 0;
      }
      next_hash_entry->setFree();
      int free_blocks_insert_index = atomicSub(&(block_heap_d->current_index), 1);
      block_heap_d->free_blocks[free_blocks_insert_index-1] = block_heap_pos;
      continue;
    }

    while(!checkFloatingPointVectorsEqual(curr_hash_entry->position, remove_block_pos, BLOCK_EPSILON)){ //hash entry is a middle or end element of linked list
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

    if(curr_offset > 0){
      prev_hash_entry->offset = new_offset;
      curr_hash_entry->setFree();
      int block_heap_pos = curr_hash_entry->pointer;
      int free_blocks_insert_index = atomicSub(&(block_heap_d->current_index), 1);
      block_heap_d->free_blocks[free_blocks_insert_index-1] = block_heap_pos;
    }
    else{
      prev_hash_entry->offset = 0;
      curr_hash_entry->setFree();
      int block_heap_pos = curr_hash_entry->pointer;
      int free_blocks_insert_index = atomicSub(&(block_heap_d->current_index), 1);
      block_heap_d->free_blocks[free_blocks_insert_index-1] = block_heap_pos;
    }
  }
  curr_hash_entry = prev_hash_entry = next_hash_entry = hash_entries = NULL;
  delete(curr_hash_entry);
  delete(prev_hash_entry);
  delete(next_hash_entry);
  delete(hash_entries);
}

TSDFHandler::TSDFHandler(){
  tsdf_container = new TSDFContainer();
  cudaMalloc(&occupied_voxels_d, sizeof(Vector3f)*OCCUPIED_VOXELS_SIZE);
  cudaMalloc(&sdf_weight_voxel_vals_d, sizeof(Voxel)*OCCUPIED_VOXELS_SIZE);
}

TSDFHandler::~TSDFHandler(){
  free(tsdf_container);
  cudaFree(occupied_voxels_d);
  cudaFree(sdf_weight_voxel_vals_d);
}

/*
* given a lidar pointcloud xyz and origin allocate voxel blocks and update voxels within truncation distance of lidar points and return occupied voxels for visualization/publishing
*/
void TSDFHandler::processPointCloudAndUpdateVoxels(pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud, Vector3f * origin_transformed_h, Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h)
{ 
  std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points = pointcloud->points;

  pcl::PointXYZ * points_h = &points[0];
  pcl::PointXYZ * points_d;
  int pointcloud_size = pointcloud->size();
  int * pointcloud_size_d;

  cudaMalloc(&points_d, sizeof(*points_h)*pointcloud_size);
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(points_d, points_h, sizeof(*points_h)*pointcloud_size, cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&pointcloud_size_d, sizeof(int));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(pointcloud_size_d, &pointcloud_size, sizeof(int), cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  Vector3f * origin_transformed_d;
  cudaMalloc(&origin_transformed_d, sizeof(*origin_transformed_h));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(origin_transformed_d, origin_transformed_h,sizeof(*origin_transformed_h),cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  HashTable * hash_table_d = tsdf_container->getCudaHashTable();
  BlockHeap * block_heap_d = tsdf_container->getCudaBlockHeap();

  allocateVoxelBlocksAndUpdateVoxels(points_d, origin_transformed_d, pointcloud_size_d, pointcloud_size, hash_table_d, block_heap_d);

  publishOccupiedVoxels(origin_transformed_d, occupied_voxels_h, occupied_voxels_index, sdfWeightVoxelVals_h, hash_table_d, block_heap_d);

  garbageCollectDistantBlocks(origin_transformed_d, hash_table_d, block_heap_d);

  cudaFree(pointcloud_size_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaFree(points_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaFree(origin_transformed_d);
  gpuErrchk(cudaPeekAtLastError());

}

/*
* allocate voxel blocks and voxels within truncation distance of lidar points and update the voxels sdf and weight
*/
void TSDFHandler::allocateVoxelBlocksAndUpdateVoxels(pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, int * pointcloud_size_d, int pointcloud_size, HashTable * hash_table_d, BlockHeap * block_heap_d){
    //TODO: FIX
  // int maxBlocksPerPoint = ceil(pow(TRUNCATION_DISTANCE,3) / pow(VOXEL_BLOCK_SIZE, 3));
  //the max number of voxel blocks that are allocated per point cloud frame
  int max_blocks = 100 * pointcloud_size; //todo: hardcoded
  Vector3f pointcloud_voxel_blocks_h[max_blocks];
  Vector3f * pointcloud_voxel_blocks_d;
  int * pointcloud_voxel_blocks_h_index = new int(0); //keep track of number of voxel blocks allocated
  int * pointcloud_voxel_blocks_d_index;
  cudaMalloc(&pointcloud_voxel_blocks_d, sizeof(*pointcloud_voxel_blocks_h)*max_blocks);
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_h, sizeof(*pointcloud_voxel_blocks_h)*max_blocks,cudaMemcpyHostToDevice); //need to memcpy?
  gpuErrchk(cudaPeekAtLastError());
  cudaMalloc(&pointcloud_voxel_blocks_d_index, sizeof(*pointcloud_voxel_blocks_h_index));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(pointcloud_voxel_blocks_d_index, pointcloud_voxel_blocks_h_index, sizeof(*pointcloud_voxel_blocks_h_index), cudaMemcpyHostToDevice);

  int num_cuda_blocks = pointcloud_size / threads_per_cuda_block + 1;

  getVoxelBlocks(num_cuda_blocks, points_d, pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, origin_transformed_d, pointcloud_size_d);

  integrateVoxelBlockPointsIntoHashTable(pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, hash_table_d, block_heap_d);

  updateVoxels(num_cuda_blocks, points_d, origin_transformed_d, pointcloud_size_d, hash_table_d, block_heap_d);

  cudaFree(pointcloud_voxel_blocks_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaFree(pointcloud_voxel_blocks_d_index);
  gpuErrchk(cudaPeekAtLastError());
  free(pointcloud_voxel_blocks_h_index);
  gpuErrchk(cudaPeekAtLastError());
}

/*
* Get voxel blocks in truncation distance of lidar points on ray between origin and the lidar points
*/
void TSDFHandler::getVoxelBlocks(int num_cuda_blocks, pcl::PointXYZ * points_d, Vector3f * pointcloud_voxel_blocks_d, int * pointcloud_voxel_blocks_d_index, Vector3f * origin_transformed_d, int * pointcloud_size_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  getVoxelBlocksForPoint<<<num_cuda_blocks,threads_per_cuda_block>>>(points_d, pointcloud_voxel_blocks_d, pointcloud_voxel_blocks_d_index, origin_transformed_d, pointcloud_size_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Get Voxel Block Duration: %f\n", milliseconds);
}

/*
* Allocated necessary voxel blocks to the hash table
*/
void TSDFHandler::integrateVoxelBlockPointsIntoHashTable(Vector3f * points_d, int * pointcloud_voxel_blocks_d_index, HashTable * hash_table_d, BlockHeap * block_heap_d){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int * size_h = new int(0);
  cudaMemcpy(size_h, pointcloud_voxel_blocks_d_index, sizeof(int), cudaMemcpyDeviceToHost);
  int size = * size_h;

  //array to keep track of unallocated blocks so in multiple kernel calls further down can keep track of blocks that still need to be allocated
  bool * unallocated_points_h = new bool[size];
  for(int i=0;i<size;++i)
  {
    unallocated_points_h[i] = 1;
  }

  bool * unallocated_points_d;
  cudaMalloc(&unallocated_points_d, sizeof(*unallocated_points_h)*size);
  cudaMemcpy(unallocated_points_d, unallocated_points_h, sizeof(*unallocated_points_h)*size, cudaMemcpyHostToDevice);

  //keep track of number of unallocated blocks still left to be allocated
  int * unallocated_points_count_h = new int(size);
  int * unallocated_points_count_d;
  cudaMalloc(&unallocated_points_count_d, sizeof(*unallocated_points_count_h));
  cudaMemcpy(unallocated_points_count_d, unallocated_points_count_h, sizeof(*unallocated_points_count_h), cudaMemcpyHostToDevice);

  int num_cuda_blocks = size / threads_per_cuda_block + 1;
  //call kernel till all blocks are allocated
  while(*unallocated_points_count_h > 0){ //POSSIBILITY OF INFINITE LOOP if no applicable space is left for an unallocated block even if there is still space left in hash table
    allocateVoxelBlocks<<<num_cuda_blocks,threads_per_cuda_block>>>(points_d, hash_table_d, block_heap_d, unallocated_points_d, pointcloud_voxel_blocks_d_index, unallocated_points_count_d);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    cudaMemcpy(unallocated_points_count_h, unallocated_points_count_d, sizeof(*unallocated_points_count_h), cudaMemcpyDeviceToHost);
  }

  //print total num blocks allocated so far
  printHashTableAndBlockHeap<<<1,1>>>(hash_table_d, block_heap_d);
  cudaDeviceSynchronize();

  cudaFree(unallocated_points_d);
  cudaFree(unallocated_points_count_d);
  delete size_h;
  delete unallocated_points_h;
  delete unallocated_points_count_h;

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Integrate Voxel Block Duration: %f\n", milliseconds);
}

/*
* update voxels sdf and weight values
*/
void TSDFHandler::updateVoxels(int & num_cuda_blocks, pcl::PointXYZ * points_d, Vector3f * origin_transformed_d, int * pointcloud_size_d, HashTable * hash_table_d, BlockHeap * block_heap_d){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  getVoxelsForPoint<<<num_cuda_blocks,threads_per_cuda_block>>>(points_d, origin_transformed_d, hash_table_d, block_heap_d, pointcloud_size_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Update Voxels Duration: %f\n", milliseconds);
}

/*
* get occupied voxels for visualization and publishing 
*/
void TSDFHandler::publishOccupiedVoxels(Vector3f * origin_transformed_d, Vector3f * occupied_voxels_h, int * occupied_voxels_index, Voxel * sdfWeightVoxelVals_h, HashTable * hash_table_d, BlockHeap * block_heap_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int * occupied_voxels_index_d; //keep track of # of occupied voxels
  int occupiedVoxelsSize = OCCUPIED_VOXELS_SIZE;
  cudaMalloc(&occupied_voxels_index_d, sizeof(*occupied_voxels_index));
  cudaMemcpy(occupied_voxels_index_d, occupied_voxels_index, sizeof(*occupied_voxels_index), cudaMemcpyHostToDevice);
  int num_cuda_blocks = HASH_TABLE_SIZE / threads_per_cuda_block + 1;
  publish_occupied_voxelsCuda<<<num_cuda_blocks,threads_per_cuda_block>>>(origin_transformed_d, hash_table_d, block_heap_d, occupied_voxels_d, occupied_voxels_index_d, sdf_weight_voxel_vals_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();

  cudaMemcpy(occupied_voxels_h, occupied_voxels_d, sizeof(*occupied_voxels_h)*occupiedVoxelsSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(occupied_voxels_index, occupied_voxels_index_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sdfWeightVoxelVals_h, sdf_weight_voxel_vals_d, sizeof(*sdfWeightVoxelVals_h)*occupiedVoxelsSize, cudaMemcpyDeviceToHost);

  cudaFree(occupied_voxels_index_d);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Publish Voxels Duration: %f\n", milliseconds);

}

/*
* Remove blocks further than garbage collection distance so not keeping distant block data
*/
void TSDFHandler::garbageCollectDistantBlocks(Vector3f * origin_transformed_d, HashTable * hash_table_d, BlockHeap * block_heap_d){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int * removed_blocks_count_h = new int(0);
  int * removed_blocks_count_d;
  cudaMalloc(&removed_blocks_count_d, sizeof(int));
  cudaMemcpy(removed_blocks_count_d, removed_blocks_count_h, sizeof(* removed_blocks_count_h), cudaMemcpyHostToDevice);

  int * linked_list_distant_blocks_count_h = new int(0);
  int * linked_list_distant_blocks_count_d;
  cudaMalloc(&linked_list_distant_blocks_count_d, sizeof(int));
  cudaMemcpy(linked_list_distant_blocks_count_d, linked_list_distant_blocks_count_h, sizeof(* linked_list_distant_blocks_count_h), cudaMemcpyHostToDevice);

  Vector3f linked_list_distance_blocks_h[MAX_LINKED_LIST_BLOCKS];
  Vector3f * linked_list_distance_blocks_d;
  cudaMalloc(&linked_list_distance_blocks_d, sizeof(*linked_list_distance_blocks_h)*MAX_LINKED_LIST_BLOCKS);
  cudaMemcpy(linked_list_distance_blocks_d, linked_list_distance_blocks_h, sizeof(*linked_list_distance_blocks_h)*MAX_LINKED_LIST_BLOCKS, cudaMemcpyHostToDevice);

  int num_cuda_blocks = HASH_TABLE_SIZE / threads_per_cuda_block + 1;
  garbageCollect<<<num_cuda_blocks, threads_per_cuda_block>>>(origin_transformed_d, hash_table_d, block_heap_d, removed_blocks_count_d, linked_list_distance_blocks_d, linked_list_distant_blocks_count_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(removed_blocks_count_h, removed_blocks_count_d, sizeof(* removed_blocks_count_h), cudaMemcpyDeviceToHost);
  cudaMemcpy(linked_list_distant_blocks_count_h, linked_list_distant_blocks_count_d, sizeof(* linked_list_distant_blocks_count_h), cudaMemcpyDeviceToHost);

  linkedListGarbageCollect<<<1,1>>>(hash_table_d, block_heap_d, linked_list_distance_blocks_d, linked_list_distant_blocks_count_d);
  gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(linked_list_distant_blocks_count_h, linked_list_distant_blocks_count_d, sizeof(* linked_list_distant_blocks_count_h), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Garbage Collection Duration: %f\n", milliseconds);
  // printf("Linked List Blocks Removed: %d\n", *linked_list_distant_blocks_count_h);
  printf("Total Blocks Removed: %d\n", *removed_blocks_count_h + *linked_list_distant_blocks_count_h);

  delete removed_blocks_count_h;
  delete linked_list_distant_blocks_count_h;
  cudaFree(removed_blocks_count_d);
  cudaFree(linked_list_distant_blocks_count_d);
  cudaFree(linked_list_distance_blocks_d);
}

/*
* Initialize device global variables
*/
__global__
void initGlobalVarsCuda(float * voxel_size_d, float * truncation_distance_d, float * max_weight_d, float * publish_distance_squared_d, float * garbage_collect_distance_squared_d){
  VOXEL_SIZE = * voxel_size_d;
  HALF_VOXEL_SIZE = VOXEL_SIZE / 2;
  VOXEL_BLOCK_SIZE = VOXEL_SIZE * VOXEL_PER_SIDE;
  HALF_VOXEL_BLOCK_SIZE = VOXEL_BLOCK_SIZE / 2;
  BLOCK_EPSILON = VOXEL_BLOCK_SIZE / 4;
  VOXEL_EPSILON = VOXEL_SIZE / 4; 
  TRUNCATION_DISTANCE = * truncation_distance_d;
  MAX_WEIGHT = * max_weight_d;
  PUBLISH_DISTANCE_SQUARED = * publish_distance_squared_d;
  GARBAGE_COLLECT_DISTANCE_SQUARED = * garbage_collect_distance_squared_d;
}

/*
* Initialize device global variables
*/
void initGlobalVars(Params params){
  initGlobalVarsCuda<<<1,1>>>(params.voxel_size_param_d, params.truncation_distance_param_d, params.max_weight_param_d, 
    params.publish_distance_squared_param_d, params.garbage_collect_distance_squared_param_d);
  gpuErrchk( cudaPeekAtLastError() );
  cudaDeviceSynchronize();
}