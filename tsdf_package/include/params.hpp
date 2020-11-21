#ifndef _PARAMS_HPP
#define _PARAMS_HPP

struct Params{
    float * voxel_size_param_d;
    float * truncation_distance_param_d;
    float * max_weight_param_d;
    float * publish_distance_squared_param_d;
    Params(float voxel_size_input, float truncation_distance_input, float max_weight_input, float publish_distance_squared_input){
        float * voxel_size_param_h = new float(voxel_size_input);
        cudaMalloc(&voxel_size_param_d, sizeof(*voxel_size_param_h));
        cudaMemcpy(voxel_size_param_d, voxel_size_param_h, sizeof(*voxel_size_param_h), cudaMemcpyHostToDevice);
        delete voxel_size_param_h;

        float * truncation_distance_param_h = new float(truncation_distance_input);
        cudaMalloc(&truncation_distance_param_d, sizeof(*truncation_distance_param_h));
        cudaMemcpy(truncation_distance_param_d, truncation_distance_param_h, sizeof(*truncation_distance_param_h), cudaMemcpyHostToDevice);
        delete truncation_distance_param_h;

        float * max_weight_param_h = new float(max_weight_input);
        cudaMalloc(&max_weight_param_d, sizeof(*max_weight_param_h));
        cudaMemcpy(max_weight_param_d, max_weight_param_h, sizeof(*max_weight_param_h), cudaMemcpyHostToDevice);
        delete max_weight_param_h;

        float * publish_distance_squared_param_h = new float(publish_distance_squared_input);
        cudaMalloc(&publish_distance_squared_param_d, sizeof(*publish_distance_squared_param_h));
        cudaMemcpy(publish_distance_squared_param_d, publish_distance_squared_param_h, sizeof(*publish_distance_squared_param_h), cudaMemcpyHostToDevice);
        delete publish_distance_squared_param_h;
    }
    ~Params(){
        cudaFree(voxel_size_param_d);
        cudaFree(truncation_distance_param_d);
        cudaFree(max_weight_param_d);
        cudaFree(publish_distance_squared_param_d);
    }
};

#endif