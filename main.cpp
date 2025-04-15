#include "mm.hpp"
#include "cusparse.h"
#include "cusparse_v2.h"
#include "cuda_runtime.h"
#include "Features.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_TOOLS.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <windows.h> // Windows API
#include <direct.h> // For _mkdir


struct COOComparator {
    __host__ __device__
        bool operator()(const thrust::tuple<int, int, double>& a, const thrust::tuple<int, int, double>& b) const {  // <-- `const` is critical
        if (thrust::get<0>(a) < thrust::get<0>(b)) return true;
        if (thrust::get<0>(a) > thrust::get<0>(b)) return false;
        return thrust::get<1>(a) < thrust::get<1>(b);
    }
};

void sortCOO(int nnz, int* rows_device, int* cols_device, double* data_device) {
    // Sort coo matrix for csr conversion
    thrust::device_ptr<int> rows_device_thrust(rows_device);
    thrust::device_ptr<int> cols_device_thrust(cols_device);
    thrust::device_ptr<double> data_device_thrust(data_device);

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(rows_device_thrust, cols_device_thrust, data_device_thrust));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(rows_device_thrust + nnz, cols_device_thrust + nnz, data_device_thrust + nnz));


    // Sort the COO matrix using the custom comparator
    thrust::sort(zip_begin, zip_end, COOComparator());

    if (DEBUG) {
        int* rows_host = (int*)malloc(nnz * sizeof(int));
        int* cols_host = (int*)malloc(nnz * sizeof(int));
        int* data_host = (int*)malloc(nnz * sizeof(int));

        cudaMemcpy(rows_host, rows_device, nnz * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cols_host, cols_device, nnz * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(data_host, data_device, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    }
    return;
}

//std::vector<std::string> list_files(const std::string& dir_path) {
//    std::vector<std::string> files;
//    try {
//        for (const auto& entry : fs::directory_iterator(dir_path)) {
//            if (entry.is_regular_file()) { // Skip subdirectories/symlinks
//                files.push_back(entry.path().string());
//            }
//        }
//    }
//    catch (const fs::filesystem_error& e) {
//        std::cerr << "Error accessing directory: " << e.what() << std::endl;
//    }
//    return files;
//}

// List all files in a directory
std::vector<std::string> list_files(const std::string& directory) {
    std::vector<std::string> files;
    WIN32_FIND_DATAA find_data;
    HANDLE h_find;

    // Search pattern: "directory/*"
    std::string search_path = directory + "\\*";
    h_find = FindFirstFileA(search_path.c_str(), &find_data);

    if (h_find == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Failed to open directory (" << GetLastError() << ")" << std::endl;
        return files;
    }

    do {
        std::string file_name = find_data.cFileName;

        // Skip "." (current directory) and ".." (parent directory)
        if (file_name == "." || file_name == "..") continue;

        // Check if it's a file (not a directory)
        if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            files.push_back(file_name);
        }
    } while (FindNextFileA(h_find, &find_data));

    FindClose(h_find);
    return files;
}


int main(int argc, char** argv)
{
    // std::filesystem::path cwd = std::filesystem::current_path();
    // std::string current_path = cwd.string();
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    std::string path;

    // Load .mtx data into raw COO matrix

    if (argc < 2) {
        char buffer[_MAX_PATH]; // _MAX_PATH is a Windows constant

        // Get current directory path
        if (_getcwd(buffer, sizeof(buffer))) {
            std::cout << "Current Directory: " << buffer << std::endl;
        }
        else {
            std::cerr << "Error: Failed to get current directory!" << std::endl;
        }
        std::string path(buffer);
    }
    else {
        path = argv[1];
    }
    std::vector<std::string> files = list_files(path);

    std::string features_dir = "Extracted_Features";
    std::string output_dir = path + "\\" + features_dir;
    // _mkdir returns 0 on success, -1 on failure
    if (_mkdir(output_dir.c_str()) == 0) {
        std::cout << "Directory created: " << features_dir << std::endl;
    }
    else {
        std::cerr << "Directory already exists: "<< features_dir << std::endl;
    }

    for (auto file : files) {
        std::string matrix_name = file;
        std::string suffix = ".features";
        std::string result_name = matrix_name + suffix;
        std::cout << "with Ordinal=int64, Scalar=double, Offset=size_t" << std::endl;
        typedef int64_t Ordinal;
        typedef double Scalar;
        typedef MtxReader<Ordinal, Scalar> reader_t;
        typedef typename reader_t::coo_type coo_t;
        typedef typename coo_t::entry_type entry_t;

        /*
        * Step 1. Read .mtx file as COO matrix
        */

        std::cout << "Reading file " << matrix_name << std::endl;
        reader_t reader(path +"\\" + file);
        coo_t coo = reader.read_coo();

        //// non-zeros, rows, cols
        //std::cout << coo.nnz() << std::endl;                               // size_t
        //std::cout << coo.num_rows() << "," << coo.num_cols() << std::endl; // int

        //// first entry
        //entry_t e = coo.entries[0];
        //std::cout << e.i << "," << e.j << std::endl; // int, int
        //std::cout << e.e << std::endl;               // float

        int* rows_host = (int*)malloc(coo.nnz() * sizeof(int));
        int* cols_host = (int*)malloc(coo.nnz() * sizeof(int));
        double* data_host = (double*)malloc(coo.nnz() * sizeof(double));

        // dereferencing to eliminate C++ warnings
        if (rows_host == nullptr) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return 1; // Indicate an error
        }

        if (cols_host == nullptr) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return 1; // Indicate an error
        }

        if (data_host == nullptr) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return 1; // Indicate an error
        }

        for (int i = 0; i < coo.nnz(); i++) {
            entry_t e = coo.entries[i];
            rows_host[i] = e.i;
            cols_host[i] = e.j;
            data_host[i] = e.e;
        }

        /*
        * Step 2. Convert COO matrix into CUDA csr format.
        */
        std::cout << "Converting coo matrix to csr..." << std::endl;
        cusparseHandle_t cusparseHandle;
        cusparseCreate(&cusparseHandle);
        cudaError_t cudaStatus;

        cusparseSpMatDescr_t matrix_coo;
        int64_t row_count = coo.num_rows();
        int64_t col_count = coo.num_cols();
        int64_t nnz_count = coo.nnz();

        int* rows_device;
        int* cols_device;
        double* data_device;

        cudaStatus = cudaMalloc(&rows_device, nnz_count * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        cudaStatus = cudaMalloc(&cols_device, nnz_count * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        cudaStatus = cudaMalloc(&data_device, nnz_count * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaMemcpy(rows_device, rows_host, nnz_count * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cols_device, cols_host, nnz_count * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(data_device, data_host, nnz_count * sizeof(double), cudaMemcpyHostToDevice);

        // Additionally, we create a transposed csr version for checking symmetry
        // Performing transpose is basically switching the row/col columns for a COO format matrix
        int* transposed_row_device;
        int* transposed_col_device;
        double* transposed_data_device;

        cudaStatus = cudaMalloc(&transposed_row_device, nnz_count * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        cudaStatus = cudaMalloc(&transposed_col_device, nnz_count * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        cudaStatus = cudaMalloc(&transposed_data_device, nnz_count * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaMemcpy(transposed_row_device, cols_device, nnz_count * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(transposed_col_device, rows_device, nnz_count * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(transposed_data_device, data_device, nnz_count * sizeof(double), cudaMemcpyDeviceToDevice);


        cusparseStatus_t cusparseStatus;
        //cusparseStatus = cusparseCreateCoo(&matrix_coo, row_count, col_count, nnz_count, rows_device, cols_device, data_device, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        //if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
        //    fprintf(stderr, "Creating COO matrix failed!");
        //    goto Error;
        //}

        // sort COO arrays for csr construction
        if (DEBUG) {
            int* rows_debug = debug_array(nnz_count, rows_device);
            int* cols_debug = debug_array(nnz_count, cols_device);
            double* data_debug = debug_array(nnz_count, data_device);

            std::cout << "Debugging..." << std::endl;

            free(rows_debug);
            free(cols_debug);
            free(data_debug);
        }

        sortCOO(nnz_count, rows_device, cols_device, data_device);

        if (DEBUG) {
            int* rows_debug = debug_array(nnz_count, rows_device);
            int* cols_debug = debug_array(nnz_count, cols_device);
            double* data_debug = debug_array(nnz_count, data_device);

            std::cout << "Debugging..." << std::endl;

            free(rows_debug);
            free(cols_debug);
            free(data_debug);
        }

        cusparseSpMatDescr_t matrix_csr;
        int* indptr_device;

        cudaStatus = cudaMalloc(&indptr_device, (row_count + 1) * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cusparseStatus = cusparseXcoo2csr(cusparseHandle, rows_device, nnz_count, row_count, indptr_device, CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "COO to CSR transformation failed!");
            goto Error;
        }

        if (DEBUG) {
            int* indptr_debug = debug_array(row_count + 1, indptr_device);

            std::cout << "Debugging..." << std::endl;

            free(indptr_debug);
        }

        cusparseStatus = cusparseCreateCsr(&matrix_csr, row_count, col_count, nnz_count, indptr_device, cols_device, data_device, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "Creating csr matrix failed!");
            goto Error;
        }

        // Sort transposed coo matrix for csr conversion
        cusparseSpMatDescr_t transposed_matrix_csr;
        int* transposed_indptr_device;
        cudaStatus = cudaMalloc(&transposed_indptr_device, (col_count + 1) * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        sortCOO(nnz_count, transposed_row_device, transposed_col_device, transposed_data_device);

        cusparseStatus = cusparseXcoo2csr(cusparseHandle, transposed_row_device, nnz_count, col_count, transposed_indptr_device, CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "COO to CSR transformation failed!");
            goto Error;
        }

        if (DEBUG) {
            int* indptr_debug = debug_array(row_count + 1, transposed_indptr_device);

            std::cout << "Debugging..." << std::endl;

            free(indptr_debug);
        }

        cusparseStatus = cusparseCreateCsr(&transposed_matrix_csr, col_count, row_count, nnz_count, transposed_indptr_device, transposed_col_device, data_device, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
            fprintf(stderr, "Creating csr matrix failed!");
            goto Error;
        }

        /*
        * Step 3. Compute Matrix Features
        */
        std::cout << "Computing matrix features..." << std::endl;
        size_t pBufferSizeInBytes = 0;
        void* pBuffer = NULL;
        int* P = NULL;

        MatrixFeatures features;
        features.check_is_symmetric(cusparseHandle, matrix_csr, transposed_matrix_csr);
        features.Compute_Features(cusparseHandle, matrix_csr);

        /*
        * Step 4. Print results
        */
        std::cout << "Printing results..." << std::endl;
        std::string output_dir = path + "\\" + features_dir + "\\" + result_name;
        std::ofstream outFile(output_dir);
        if (outFile.is_open()) {
            outFile << "# Matrix Features" << "\n"
                << "Row count = " << features.row << "\n"
                << "Column count = " << features.col << "\n"
                << "nnz count = " << features.nnz << "\n"
                << "is_symmetric = " << features.is_symmetric << "\n"
                << "positive_diagonal_count = " << features.positive_diagonal_count << "\n"
                << "negative_diagonal_count = " << features.negative_diagonal_count << "\n"
                << "positive_nondiagonal_count = " << features.positive_nondiagonal_count << "\n"
                << "negative_nondiagonal_count = " << features.negative_nondiagonal_count << "\n"
                << "diagonally_dominant_row_count = " << features.diagonally_dominant_row_count << "\n"
                << "non_diagonally_dominant_row_count = " << features.non_diagonally_dominant_row_count << "\n"
                << "diagonally_dominant_row_percentage = " << features.diagonally_dominant_row_percentage << "\n"
                << "non_diagonally_dominant_row_percentage = " << features.non_diagonally_dominant_row_percentage << "\n"
                << "min_row_nnz_count = " << features.min_row_nnz_count << "\n"
                << "max_row_nnz_count = " << features.max_row_nnz_count << "\n"
                << "min_abs_diagonal = " << features.min_abs_diagonal << "\n"
                << "max_abs_diagonal = " << features.max_abs_diagonal << "\n"
                << "min_abs_non_diagonal = " << features.min_abs_non_diagonal << "\n"
                << "max_abs_non_diagonal = " << features.max_abs_non_diagonal << "\n"
                << "min_frac_nondiag_max_vs_min = " << features.min_frac_nondiag_max_vs_min << "\n"
                << "max_frac_nondiag_max_vs_min = " << features.max_frac_nondiag_max_vs_min << "\n"
                << "is_positive_definite = " << features.positive_definiteness << "\n"
                << "condition_number = " << features.condition_number << "\n"
                << "spectral_radius = " << features.spectral_radius << "\n"
                << "EOF";
        }
        std::cout << "Recycling..." << std::endl;
        /*
        * Recycling
        */
        cusparseDestroySpMat(matrix_csr);
        cusparseDestroySpMat(transposed_matrix_csr);

        cudaFree(rows_device);
        cudaFree(cols_device);
        cudaFree(data_device);

        cudaFree(indptr_device);
        cudaFree(transposed_indptr_device);
    }
    return 0;

    Error:

    return 0;
}

