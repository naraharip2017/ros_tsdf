#include <stdio.h>

#include "cuda/tsdf_container.cuh"

/*
* Definitions for tsdf_container class declared in tsdf_container.cuh
*/

namespace tsdf {

TSDFContainer::TSDFContainer(){
    //allocate hash table and block heap on host and device
    hash_table_h = new HashTable();
    block_heap_h = new BlockHeap();
    cudaMalloc(&hash_table_d, sizeof(*hash_table_h));
    cudaMemcpy(hash_table_d, hash_table_h, sizeof(*hash_table_h), cudaMemcpyHostToDevice);
    cudaMalloc(&block_heap_d, sizeof(*block_heap_h));
    cudaMemcpy(block_heap_d, block_heap_h, sizeof(*block_heap_h), cudaMemcpyHostToDevice);
}

TSDFContainer::~TSDFContainer(){
    cudaFree(hash_table_d);
    cudaFree(block_heap_d);
    delete hash_table_h;
    delete block_heap_h;
}

/**
* @return return the GPU side hash table
*/
HashTable * TSDFContainer::getCudaHashTable(){
    return hash_table_d;
}

/**
* @return return the GPU side block heap
*/
BlockHeap * TSDFContainer::getCudaBlockHeap(){
    return block_heap_d;
}

}