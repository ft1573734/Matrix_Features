#include <limits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include "cusparse_v2.h"


class MatrixFeatures {
    /*
     * Basic Features
     */ 

public:


    // The shape of the matrix
    int64_t row;
    int64_t col;
    // NNZ count of the matrix
    int64_t nnz;
    // The number of positive diagonal elements
    int64_t positive_diagonal_count = 0;
    // The number of negative diagonal elements
    int64_t negative_diagonal_count = 0;
    // The number of positive non - diagonal elements
    int64_t positive_nondiagonal_count = 0;
    // The number of negative non - diagonal elements
    int64_t negative_nondiagonal_count = 0;
    // The number of diangonally dominant rows
    // Definition of diagonally dominant : | diagonal element| >= sum(| non - diagonal elements | )
    int64_t diagonally_dominant_row_count = 0;
    // The number of non - diagonally dominant rows
    // Definition of diagonally dominant : | diagonal element| >= sum(| non - diagonal elements | )
    int64_t non_diagonally_dominant_row_count = 0;
    // (diagonally dominant row count)/(total row count)
    double diagonally_dominant_row_percentage = 0.0;
    // (non-diagonally dominant row count)/(total row count)
    double non_diagonally_dominant_row_percentage = 0.0;

    /*
     * Multiscale Features
     */

    // minimum row nnz count of the matrix
    int64_t min_row_nnz_count = std::numeric_limits<int64_t>::max();
    // maximum row nnz count of the matrix
    int64_t max_row_nnz_count = 0;
    // minimum | diagonal element |
    double min_abs_diagonal = std::numeric_limits<double>::max();
    // maximum | diagonal element |
    double max_abs_diagonal = 0.0;
    // minimum | non - diagonal element |
    double min_abs_non_diagonal = std::numeric_limits<double>::max();
    // maximum | non - diagonal element |
    double max_abs_non_diagonal = 0.0;
    // min(max(| non - diagonal element | ) / min(| non - diagonal element | ))
    double min_frac_nondiag_max_vs_min = std::numeric_limits<double>::max();
    // max(max(| non - diagonal element | ) / min(| non - diagonal element | ))
    double max_frac_nondiag_max_vs_min = 0.0;

    /*
     * Complicated Features
     */

    // These features requires much more time to compute, more efficient solution is needed

    bool is_symmetric = false;
    int64_t rank = 0;
    double condition_number = 0.0;
    double spectral_radius = 0.0;
    bool positive_definiteness = false;

    MatrixFeatures();

    void Compute_Features(cusparseHandle_t cusparseHandle, cusparseSpMatDescr_t csr_matrix);

    void check_is_symmetric(cusparseHandle_t handle, cusparseSpMatDescr_t original_matrix_csr, cusparseSpMatDescr_t transposed_matrix_csr);

private:

    double Gershgorin_radius_max;
    
    double Gershgorin_radius_min;

    int calculate_optimal_NUM_THREADS(int N, int NUM_BLOCKS);

    bool is_positive_definite();

    void transposeCSR(
        cusparseHandle_t handle,
        int64_t n,                      // Matrix dimension (n x n)
        int64_t nnz,                    // Number of non-zeros in output
        const int* d_rowPtr,        // Input CSR row pointers (device)
        const int* d_colInd,        // Input CSR column indices (device)
        const double* d_values,      // Input CSR values (device)
        int* d_transRowPtr,        // Output CSR row pointers (device)
        int* d_transColInd,        // Output CSR column indices (device)
        double* d_transValues      // Output CSR values (device)
    );
};

