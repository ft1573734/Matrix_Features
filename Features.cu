#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"
#include "Features.cuh"
#include <climits>
#include <stdlib.h> 
#include "math.h"
#include <cublas_v2.h>
#include <stdio.h>
#include <iostream>
#include "CUDA_TOOLS.h"

// Define the maximum number of threads per block for your GPU
const int MAX_THREADS_PER_BLOCK = 1024;

// Define the warp size for your GPU
const int WARP_SIZE = 32;

const unsigned int CONST_NUM_BLOCKS = 24;


#define CUSPARSE_CHECK(err) { \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE error: " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void CUDA_check_identical(int row, int* A_indptr, int* A_indices, double *A_data, int* B_indptr, int* B_indices, double* B_data, bool* result) {
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < row) {
		// For each row:
		int A_row_start = A_indptr[t_idx];
		int A_row_end = A_indptr[t_idx + 1];
		int B_row_start = B_indptr[t_idx];
		int B_row_end = B_indptr[t_idx + 1];

		// Check length
		if (A_row_end - A_row_start != B_row_end - B_row_start) {
			result[t_idx] = false;
			goto RETURN;
		}
		else {
			//Check values
			bool is_row_identical = true;
			for (int i = A_row_start; i < A_row_end; i++) {
				if (A_indices[i] != B_indices[i]) {
					is_row_identical = false;
					goto RETURN;
				}
				if (A_data[i] != B_data[i]) {
					goto RETURN;
					is_row_identical = false;
				}
			}
			RETURN:
			result[t_idx] = is_row_identical;
		}
	}
}

//__global__ void CUDA_Test(int row, int nnz, int* indices, int* indptr, double* data, int* positive_diagonal_count_per_row) {
//	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
//	if (t_idx < row) {
//		int row_start = indptr[t_idx];
//		int row_end = indptr[t_idx + 1];
//		for (int i = row_start; i < row_end; i++) {
//			int tmp_col_index = indices[i];
//			// Update diagonal arrays:
//			if (tmp_col_index == t_idx) {
//				if (data[i] < 0) {
//
//					if (t_idx < 10) {
//						printf("Setting positive_diagonal_count_per_row[%d] to %d...\n", t_idx, 1);
//					}
//					positive_diagonal_count_per_row[t_idx] += 1;
//				}
//			}
//		}
//	}
//}

__global__ void CUDA_update_diagonal_statistics(int64_t row, int* indices, int* indptr, double* data, int* positive_diagonal_count_per_row, int* negative_diagonal_count_per_row, double* diagonal_abs_per_row, int* row_nnz_count) {
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < row) {
		int row_start = indptr[t_idx];
		int row_end = indptr[t_idx + 1];
		row_nnz_count[t_idx] = row_end - row_start;
;		for (int i = row_start; i < row_end; i++) {
			// Update diagonal arrays:
			if (indices[i] == t_idx) {
				if (data[i] > 0) {
					positive_diagonal_count_per_row[t_idx] += 1;
				}
				else {
					negative_diagonal_count_per_row[t_idx] += 1;
				}
				diagonal_abs_per_row[t_idx] = abs(data[i]);
			}
		}
	}
}

__global__ void CUDA_update_nondiagonal_statistics(int64_t row, int* indices, int* indptr, double* data, int* positive_nondiagonal_count_per_row, int* negative_nondiagonal_count_per_row, double* nondiagonal_abs_sum_per_row, double* min_nondiagonal_abs_per_row, double* max_nondiagonal_abs_per_row) {
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < row) {
		int row_start = indptr[t_idx];
		int row_end = indptr[t_idx + 1];
		for (int i = row_start; i < row_end; i++) {
			if (indices[i] != t_idx) {
				if (data[i] > 0) {
					positive_nondiagonal_count_per_row[t_idx] += 1;
				}
				else {
					negative_nondiagonal_count_per_row[t_idx] += 1;
				}
				nondiagonal_abs_sum_per_row[t_idx] += abs(data[i]);
				if (abs(data[i]) < min_nondiagonal_abs_per_row[t_idx]) {
					min_nondiagonal_abs_per_row[t_idx] = abs(data[i]);
				}
				if (abs(data[i]) > max_nondiagonal_abs_per_row[t_idx]) {
					max_nondiagonal_abs_per_row[t_idx] = abs(data[i]);
				}
			}
		}
	}
}

// Update Gershgorin Circle Radius
__global__ void CUDA_estimate_Gershgorin_Circles(int64_t n, double* centroids, double* radius, double* out_lowerbound, double* out_upperbound) {
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < n) {
		out_lowerbound[t_idx] = centroids[t_idx] - radius[t_idx];
		out_upperbound[t_idx] = centroids[t_idx] + radius[t_idx];
	}
}


__global__ void CUDA_compute_basic_features_kernel(int64_t row, int64_t nnz, int* indices, int* indptr, double* data, int* positive_diagonal_count_per_row, int* positive_nondiagonal_count_per_row, int* negative_diagonal_count_per_row, int* negative_nondiagonal_count_per_row, double* diagonal_abs_per_row, double* nondiagonal_abs_sum_per_row, double* min_nondiagonal_abs_per_row, double* max_nondiagonal_abs_per_row, double* Greshgorin_radius_max, double* Greshgorin_radius_min) {
	int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (t_idx < 10) {
		printf("Processing row %d\n", t_idx);
	}
	if (t_idx < row) {
		int row_start = indptr[t_idx];
		int row_end = indptr[t_idx + 1];
		int positive_nondiagonal_count = 0;
		int negative_nondiagonal_count = 0;
		double nondiagonal_abs_sum = 0.0;
		for (int i = row_start; i < row_end; i++) {
			// Update diagonal arrays:
			if (indices[i] == t_idx) {
				if (data[i] > 0) {
					positive_diagonal_count_per_row[t_idx] = 1;
				}
				else {
					negative_diagonal_count_per_row[t_idx] = 1;
				}
				diagonal_abs_per_row[t_idx] = abs(data[i]);
			}
			// Update nondiagonal arrays:
			//if (indices[i] != t_idx) {
			//	if (data[i] > 0) {
			//		positive_nondiagonal_count += 1;
			//		//positive_nondiagonal_count_per_row[t_idx] += 1;
			//	}
			//	else {
			//		negative_nondiagonal_count += 1;
			//		//negative_nondiagonal_count_per_row[t_idx] += 1;
			//	}
			//	double abs_data = abs(data[i]);
			//	nondiagonal_abs_sum += abs_data;
			//	if (abs_data < min_nondiagonal_abs_per_row[t_idx]) {
			//		min_nondiagonal_abs_per_row[t_idx] = abs_data;
			//	}
			//	if (abs_data > max_nondiagonal_abs_per_row[t_idx]) {
			//		max_nondiagonal_abs_per_row[t_idx] = abs_data;
			//	}
			//}
		}
		positive_nondiagonal_count_per_row[t_idx] = positive_nondiagonal_count;
		negative_nondiagonal_count_per_row[t_idx] = negative_nondiagonal_count;
		nondiagonal_abs_sum_per_row[t_idx] = nondiagonal_abs_sum;

		// Update Gershgorin Circle Radius
		Greshgorin_radius_max[t_idx] = diagonal_abs_per_row[t_idx] + max_nondiagonal_abs_per_row[t_idx];
		Greshgorin_radius_min[t_idx] = diagonal_abs_per_row[t_idx] - max_nondiagonal_abs_per_row[t_idx];
	}
}

template <typename T>
__global__ void CUDA_initialize_array_kernel(T* array, T value, size_t N) {
	size_t t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (t_idx < N) {
		array[t_idx] = value;
	}
}

MatrixFeatures::MatrixFeatures() {
	this->row = -1;
	this->col = -1;
	this->nnz = -1;

	this->positive_diagonal_count = 0;
	this->negative_diagonal_count = 0;
	this->positive_nondiagonal_count = 0;
	this->negative_nondiagonal_count = 0;
	this->diagonally_dominant_row_count = 0;
	this->non_diagonally_dominant_row_count = 0;
	this->diagonally_dominant_row_percentage = 0.0;
	this->non_diagonally_dominant_row_percentage = 0.0;
	this->is_symmetric = false;

	this->min_row_nnz_count = std::numeric_limits<int64_t>::max();
	this->max_row_nnz_count = 0;
	this->min_abs_diagonal = std::numeric_limits<double>::max();
	this->max_abs_diagonal = 0.0;
	this->min_abs_non_diagonal = std::numeric_limits<double>::max();
	this->max_abs_non_diagonal = 0.0;
	this->min_frac_nondiag_max_vs_min = std::numeric_limits<double>::max();
	this->max_frac_nondiag_max_vs_min = 0.0;

	this->Gershgorin_radius_max = 0.0;
	this->Gershgorin_radius_min = 0.0;

}


void MatrixFeatures::Compute_Features(cusparseHandle_t cusparseHandle, cusparseSpMatDescr_t csr_matrix) {
	int NUM_THREADS;
	int NUM_BLOCKS;

	int64_t row;
	int64_t col;
	int64_t nnz;
	void* indptr_receiver;
	void* indices_receiver;
	void* data_receiver;
	cusparseIndexType_t indptr_type;
	cusparseIndexType_t indexes_type;
	cusparseIndexBase_t idx_base;
	cudaDataType dataType;
	cusparseCsrGet(csr_matrix, &row, &col, &nnz, &indptr_receiver, &indices_receiver, &data_receiver, &indptr_type, &indexes_type, &idx_base, 
		&dataType);

	int* indptr = (int*)indptr_receiver;
	int* indices = (int*)indices_receiver;
	double* data = (double*)data_receiver;

	if (DEBUG) {
		int* indptr_debug = (int*)malloc((row + 1) * sizeof(int));
		int* indices_debug = (int*)malloc(nnz * sizeof(int));
		double* data_debug = (double*)malloc(nnz * sizeof(double));
		cudaMemcpy(indptr_debug, indptr, (row + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(indices_debug, indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(data_debug, data, nnz * sizeof(double), cudaMemcpyDeviceToHost);
		std::cout << "Debugging..." << std::endl;
	}

	cudaError_t cudaStatus;

	int* positive_diagonal_count_per_row;
	int* positive_nondiagonal_count_per_row;
	int* negative_diagonal_count_per_row;
	int* negative_nondiagonal_count_per_row;
	double* diagonal_abs_per_row;
	double* nondiagonal_abs_sum_per_row;
	double* min_nondiagonal_abs_per_row;
	double* max_nondiagonal_abs_per_row;
	int* row_nnz_count;

	double* Greshgorin_Radius_max;
	double* Greshgorin_Radius_min;

	cudaStatus = cudaMalloc(&positive_diagonal_count_per_row, row * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&positive_nondiagonal_count_per_row, row * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&negative_diagonal_count_per_row, row * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&negative_nondiagonal_count_per_row, row * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&diagonal_abs_per_row, row * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&nondiagonal_abs_sum_per_row, row * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&min_nondiagonal_abs_per_row, row * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&max_nondiagonal_abs_per_row, row * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&Greshgorin_Radius_max, row * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&Greshgorin_Radius_min, row * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&row_nnz_count, row * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	NUM_THREADS = calculate_optimal_NUM_THREADS(row, CONST_NUM_BLOCKS);
	NUM_BLOCKS = row / NUM_THREADS + 1;

	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (positive_diagonal_count_per_row, 0, row);
	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (positive_nondiagonal_count_per_row, 0, row);
	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (negative_diagonal_count_per_row, 0, row);
	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (negative_nondiagonal_count_per_row, 0, row);
	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (diagonal_abs_per_row, 0.0, row);
	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (nondiagonal_abs_sum_per_row, 0.0, row);
	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (min_nondiagonal_abs_per_row, DBL_MAX, row);
	CUDA_initialize_array_kernel <<<NUM_BLOCKS, NUM_THREADS >>> (max_nondiagonal_abs_per_row, DBL_MIN, row);

	cudaDeviceSynchronize();

	if (DEBUG) {
		int* positive_diagonal_count_per_row_debug = debug_array(row, positive_diagonal_count_per_row);
		int* positive_nondiagonal_count_per_row_debug = debug_array(row, positive_nondiagonal_count_per_row);
		int* negative_diagonal_count_per_row_debug = debug_array(row, negative_diagonal_count_per_row);
		int* negative_nondiagonal_count_per_row_debug = debug_array(row, negative_nondiagonal_count_per_row);
		double* diagonal_abs_per_row_debug = debug_array(row, diagonal_abs_per_row);
		double* nondiagonal_abs_sum_per_row_debug = debug_array(row, nondiagonal_abs_sum_per_row);
		double* min_nondiagonal_abs_per_row_debug = debug_array(row, min_nondiagonal_abs_per_row);
		double* max_nondiagonal_abs_per_row_debug = debug_array(row, max_nondiagonal_abs_per_row);
		free(positive_diagonal_count_per_row_debug);
		free(positive_nondiagonal_count_per_row_debug);
		free(negative_diagonal_count_per_row_debug);
		free(negative_nondiagonal_count_per_row_debug);
		free(diagonal_abs_per_row_debug);
		free(nondiagonal_abs_sum_per_row_debug);
		free(min_nondiagonal_abs_per_row_debug);
		free(max_nondiagonal_abs_per_row_debug);
	}

	NUM_THREADS = calculate_optimal_NUM_THREADS(row, CONST_NUM_BLOCKS);
	NUM_BLOCKS = row / NUM_THREADS + 1;

	//CUDA_Test <<<NUM_BLOCKS, NUM_THREADS >>> (row, nnz, indices, indptr, data, positive_diagonal_count_per_row);

	CUDA_update_diagonal_statistics <<<NUM_BLOCKS, NUM_THREADS>>> (row, indices, indptr, data, positive_diagonal_count_per_row, negative_diagonal_count_per_row, diagonal_abs_per_row, row_nnz_count);

	cudaDeviceSynchronize();

	CUDA_update_nondiagonal_statistics <<<NUM_BLOCKS, NUM_THREADS>>> (row, indices, indptr, data, positive_nondiagonal_count_per_row, negative_nondiagonal_count_per_row, nondiagonal_abs_sum_per_row, min_nondiagonal_abs_per_row, max_nondiagonal_abs_per_row);

	cudaDeviceSynchronize();

	CUDA_estimate_Gershgorin_Circles <<<NUM_BLOCKS, NUM_THREADS >>> (row, diagonal_abs_per_row, max_nondiagonal_abs_per_row, Greshgorin_Radius_min, Greshgorin_Radius_max);

	cudaDeviceSynchronize();

	if (DEBUG) {
		int* positive_diagonal_count_per_row_debug = debug_array(row, positive_diagonal_count_per_row);
		int* positive_nondiagonal_count_per_row_debug = debug_array(row, positive_nondiagonal_count_per_row);
		int* negative_diagonal_count_per_row_debug = debug_array(row, negative_diagonal_count_per_row);
		int* negative_nondiagonal_count_per_row_debug = debug_array(row, negative_nondiagonal_count_per_row);
		double* diagonal_abs_per_row_debug = debug_array(row, diagonal_abs_per_row);
		double* nondiagonal_abs_sum_per_row_debug = debug_array(row, nondiagonal_abs_sum_per_row);
		double* min_nondiagonal_abs_per_row_debug = debug_array(row, min_nondiagonal_abs_per_row);
		double* max_nondiagonal_abs_per_row_debug = debug_array(row, max_nondiagonal_abs_per_row);
		free(positive_diagonal_count_per_row_debug);
		free(positive_nondiagonal_count_per_row_debug);
		free(negative_diagonal_count_per_row_debug);
		free(negative_nondiagonal_count_per_row_debug);
		free(diagonal_abs_per_row_debug);
		free(nondiagonal_abs_sum_per_row_debug);
		free(min_nondiagonal_abs_per_row_debug);
		free(max_nondiagonal_abs_per_row_debug);
	}

	int* positive_diagonal_count_per_row_host = _device_array_to_host(row, positive_diagonal_count_per_row);
	int* positive_nondiagonal_count_per_row_host = _device_array_to_host(row, positive_nondiagonal_count_per_row);
	int* negative_diagonal_count_per_row_host = _device_array_to_host(row, negative_diagonal_count_per_row);
	int* negative_nondiagonal_count_per_row_host = _device_array_to_host(row, negative_nondiagonal_count_per_row);
	double* diagonal_abs_per_row_host = _device_array_to_host(row, diagonal_abs_per_row);
	double* nondiagonal_abs_sum_per_row_host = _device_array_to_host(row, nondiagonal_abs_sum_per_row);
	double* min_nondiagonal_abs_per_row_host = _device_array_to_host(row, min_nondiagonal_abs_per_row);
	double* max_nondiagonal_abs_per_row_host = _device_array_to_host(row, max_nondiagonal_abs_per_row);
	double* Greshgorin_Radius_max_host = _device_array_to_host(row, Greshgorin_Radius_max);
	double* Greshgorin_Radius_min_host = _device_array_to_host(row, Greshgorin_Radius_min);
	int* row_nnz_count_host = (int*)_device_array_to_host(row, row_nnz_count);


	this->row = row;
	this->col = col;
	this->nnz = nnz;

	for (int i = 0; i < row; i++) {
		this->positive_diagonal_count += positive_diagonal_count_per_row_host[i];
		this->positive_nondiagonal_count += positive_nondiagonal_count_per_row_host[i];
		this->negative_diagonal_count += negative_diagonal_count_per_row_host[i];
		this->negative_nondiagonal_count += negative_nondiagonal_count_per_row_host[i];

		if (diagonal_abs_per_row_host[i] >= nondiagonal_abs_sum_per_row_host[i]) {
			this->diagonally_dominant_row_count += 1;
		}

		if (row_nnz_count_host[i] > this->max_row_nnz_count) {
			this->max_row_nnz_count = row_nnz_count_host[i];
		}
		if (row_nnz_count_host[i] < this->min_row_nnz_count) {
			this->min_row_nnz_count = row_nnz_count_host[i];
		}

		if (diagonal_abs_per_row_host[i] < this->min_abs_diagonal) {
			this->min_abs_diagonal = diagonal_abs_per_row_host[i];
		}
		if (diagonal_abs_per_row_host[i] > this->max_abs_diagonal) {
			this->max_abs_diagonal = diagonal_abs_per_row_host[i];
		}
		if (min_nondiagonal_abs_per_row_host[i] < this->min_abs_non_diagonal){
			this->min_abs_non_diagonal = min_nondiagonal_abs_per_row_host[i];
		}
		if (max_nondiagonal_abs_per_row_host[i] > this->max_abs_non_diagonal) {
			this->max_abs_non_diagonal = max_nondiagonal_abs_per_row_host[i];
		}

		double max_min_frac = max_nondiagonal_abs_per_row_host[i] / min_nondiagonal_abs_per_row_host[i];
		if (max_min_frac < this->min_frac_nondiag_max_vs_min) {
			this->min_frac_nondiag_max_vs_min = max_min_frac;
		}
		if (max_min_frac > this->max_frac_nondiag_max_vs_min) {
			this->max_frac_nondiag_max_vs_min = max_min_frac;
		}

	}
	this->non_diagonally_dominant_row_count = this->row - this->diagonally_dominant_row_count;
	
	this->diagonally_dominant_row_percentage = (1.0 * this->diagonally_dominant_row_count) / (1.0 * this->row);
	this->non_diagonally_dominant_row_percentage = (1.0 * this->non_diagonally_dominant_row_count) / (1.0 * this->row);

	// The following features are estimated based on computing eigenvalues using Gershgorin Cicle Theorem
	double max_estimated_eigenvalue = DBL_MIN;
	double min_estimated_eigenvalue = DBL_MAX;
	for (int i = 0; i < row; i++) {
		// Compute estimated eigenvalue limits
		if (Greshgorin_Radius_max_host[i] > max_estimated_eigenvalue) {
			max_estimated_eigenvalue = Greshgorin_Radius_max_host[i];
		}
		if (Greshgorin_Radius_min_host[i] < min_estimated_eigenvalue) {
			min_estimated_eigenvalue = Greshgorin_Radius_min_host[i];
		}
		
	}

	this->Gershgorin_radius_max = max_estimated_eigenvalue;
	this->Gershgorin_radius_min = min_estimated_eigenvalue;

	this->condition_number = abs(max_estimated_eigenvalue) / abs(min_estimated_eigenvalue);
	this->spectral_radius = max_estimated_eigenvalue; 
	this->positive_definiteness = MatrixFeatures::is_positive_definite();

	free(positive_diagonal_count_per_row_host);
	free(positive_nondiagonal_count_per_row_host);
	free(negative_diagonal_count_per_row_host);
	free(negative_nondiagonal_count_per_row_host);
	free(diagonal_abs_per_row_host);
	free(nondiagonal_abs_sum_per_row_host);
	free(min_nondiagonal_abs_per_row_host);
	free(max_nondiagonal_abs_per_row_host);
	free(Greshgorin_Radius_max_host);
	free(Greshgorin_Radius_min_host);
	free(row_nnz_count_host);

	Error:

	return;
}

void MatrixFeatures::check_is_symmetric(cusparseHandle_t handle, cusparseSpMatDescr_t mat_A, cusparseSpMatDescr_t mat_B) {
	// get the first matrix
	bool _is_symmetric = false;

	int64_t A_row;
	int64_t A_col;
	int64_t A_nnz;
	void* A_indptr;
	void* A_indexes;
	void* A_data;
	cusparseIndexType_t A_offset_type;
	cusparseIndexType_t A_columns_type;
	cusparseIndexBase_t A_index_base;
	cudaDataType A_values_type;

	cusparseCsrGet(
		mat_A,
		&A_row, &A_col, &A_nnz,
		&A_indptr, &A_indexes, &A_data,
		&A_offset_type, &A_columns_type, &A_index_base , &A_values_type);

	// get the second matrix
	int64_t B_row;
	int64_t B_col;
	int64_t B_nnz;
	void* B_indptr;
	void* B_indexes;
	void* B_data;
	cusparseIndexType_t B_offset_type;
	cusparseIndexType_t B_columns_type;
	cusparseIndexBase_t B_index_base;
	cudaDataType B_values_type;

	cusparseCsrGet(
		mat_B,
		&B_row, &B_col, &B_nnz,
		&B_indptr, &B_indexes, &B_data,
		&B_offset_type, &B_columns_type, &B_index_base, &B_values_type);

	if (A_row != B_row || A_col != B_col) {
		_is_symmetric = false;
		goto RETURN;
	}

	bool* result_device;
	cudaMalloc(&result_device, A_row * sizeof(bool));

	int NUM_THREADS = calculate_optimal_NUM_THREADS(A_row, CONST_NUM_BLOCKS);
	int NUM_BLOCKS = A_row / NUM_THREADS + 1;
	CUDA_check_identical <<<NUM_BLOCKS, NUM_THREADS>>> (A_row, (int*)A_indptr, (int*)A_indexes, (double*)A_data, (int*)B_indptr, (int*)B_indexes, (double*)B_data, result_device);

	cudaDeviceSynchronize();

	bool* result_host = (bool*)malloc(A_row * sizeof(bool));
	cudaMemcpy(result_host, result_device, A_row * sizeof(bool), cudaMemcpyDeviceToHost);

	for (int i = 0; i < A_row; i++) {
		if (result_host[i] == false) {
			_is_symmetric = false;
			goto RETURN;
		}
	}
	_is_symmetric = true;

RETURN:
	this->is_symmetric = _is_symmetric;
}

int MatrixFeatures::calculate_optimal_NUM_THREADS(int N, int NUM_BLOCKS) {
	// Calculate the initial number of threads per block
	int num_threads = static_cast<int>(ceil(static_cast<double>(N) / NUM_BLOCKS));

	// Round to the nearest multiple of the warp size
	num_threads = ((num_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

	// Ensure that the number of threads per block does not exceed the GPU's limit
	num_threads = std::min(num_threads, MAX_THREADS_PER_BLOCK);

	return num_threads;
}



bool MatrixFeatures::is_positive_definite() {
	bool result = true;

	// step 1. Check symmetry
	if (this->is_symmetric == false) {
		result = false;
		goto RETURN;
	}
	
	// step 2. Check if diagonal entries are all positive
	if (this->negative_diagonal_count > 0) {
		result = false;
		goto RETURN;
	}

	// step 3. Eigenvalue check (estimated)
	if (this->Gershgorin_radius_min < 0) {
		result = false;
		goto RETURN;
	}
	
RETURN:
	return result;
}


