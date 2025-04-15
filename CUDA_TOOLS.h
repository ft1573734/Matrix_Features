#pragma once
#ifndef DEBUG
#define DEBUG false
#endif

#include "cuda_runtime.h"
#include <stdlib.h> 

template<typename T>
T* debug_array(size_t size, T* device_arr) {
	T* host_arr = (T*)malloc(size * sizeof(T));
	cudaMemcpy(host_arr, device_arr, size * sizeof(T), cudaMemcpyDeviceToHost);
	return host_arr;
}


template<typename T>
T* _device_array_to_host(size_t size, T* device_arr) {
	T* host_arr = (T*)malloc(size * sizeof(T));
	cudaMemcpy(host_arr, device_arr, size * sizeof(T), cudaMemcpyDeviceToHost);
	return host_arr;
}
