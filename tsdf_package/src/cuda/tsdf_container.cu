#include <stdio.h>

#include "cuda/tsdf_container.cuh"

TSDFContainer::TSDFContainer(){
  hashTable_h = new HashTable();
  blockHeap_h = new BlockHeap();
  cudaMalloc(&hashTable_d, sizeof(*hashTable_h));
  cudaMemcpy(hashTable_d, hashTable_h, sizeof(*hashTable_h), cudaMemcpyHostToDevice);
  cudaMalloc(&blockHeap_d, sizeof(*blockHeap_h));
  cudaMemcpy(blockHeap_d, blockHeap_h, sizeof(*blockHeap_h), cudaMemcpyHostToDevice);
}

TSDFContainer::~TSDFContainer(){
    cudaFree(hashTable_d);
    cudaFree(blockHeap_d);
    free(hashTable_h);
    free(blockHeap_h);
}

HashTable * TSDFContainer::getCudaHashTable(){
    return hashTable_d;
}

BlockHeap * TSDFContainer::getCudaBlockHeap(){
    return blockHeap_d;
}