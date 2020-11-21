#ifndef _PARAMS_HPP
#define _PARAMS_HPP

struct Params{
    float * voxel_size_param_d;
    Params(float voxel_size_input){
        float * voxel_size_param_h = new float(voxel_size_input);
        cudaMalloc(&voxel_size_param_d, sizeof(*voxel_size_param_h));
        cudaMemcpy(voxel_size_param_d, voxel_size_param_h, sizeof(*voxel_size_param_h), cudaMemcpyHostToDevice);
        delete voxel_size_param_h;
    }
    ~Params(){
        cudaFree(voxel_size_param_d);
    }
};

#endif