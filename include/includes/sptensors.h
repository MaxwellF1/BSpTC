/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PARTI_SPTENSORS_H
#define PARTI_SPTENSORS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>

/* Sparse tensor */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]);
int sptNewSparseTensorNuma(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[], int numa_node);
int sptNewSparseTensorWithSizeNuma(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[], int numa_node, unsigned long long size);
int sptNewSparseTensorWithSize(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[], unsigned long long size);
int sptCopySparseTensor(sptSparseTensor *dest, const sptSparseTensor *src, int const nt);
void sptFreeSparseTensor(sptSparseTensor *tsr);
double SparseTensorFrobeniusNormSquared(sptSparseTensor const * const spten);
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname);
int sptLoadSparseTensorNuma(sptSparseTensor *tsr, sptIndex start_index, char const * const fname, int numa_node);
int sptDumpSparseTensor(const sptSparseTensor *tsr, sptIndex start_index, FILE *fp);
int sptMatricize(sptSparseTensor const * const X,
    sptIndex const m,
    sptSparseMatrix * const A,
    int const transpose);
void sptGetBestModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes);
void sptGetWorstModeOrder(
    sptIndex * mode_order,
    sptIndex const mode,
    sptIndex const * ndims,
    sptIndex const nmodes);
void sptGetRandomShuffleElements(sptSparseTensor *tsr);
void sptGetRandomShuffledIndices(sptSparseTensor *tsr, sptIndex ** map_inds);
void sptSparseTensorShuffleIndices(sptSparseTensor *tsr, sptIndex ** map_inds);
void sptSparseTensorInvMap(sptSparseTensor *tsr, sptIndex ** in_map_inds);
void sptSparseTensorShuffleModes(sptSparseTensor *tsr, sptIndex * mode_order);
int sptSparseTensorSetIndices(
    sptSparseTensor *ref,
    sptIndex * mode_order,
    sptIndex num_cmodes,
    sptNnzIndexVector *fiberidx);
int sptSparseTensorHiCOOSetIndices(
    sptSparseTensorHiCOO *ref,
    sptIndex * mode_order,
    sptIndex num_cmodes,
    sptNnzIndexVector *fiberidx); 
int sptSparseTensorSetIndicesIndex(
    sptSparseTensor *ref,
    sptIndex * mode_order,
    sptIndex num_cmodes,
    sptNnzIndex begin,
    sptNnzIndex end,
    sptNnzIndexVector *fiberidx); 
void sptSparseTensorSortIndex(sptSparseTensor *tsr, int force, int tk);
void sptSparseTensorSortIndexCmode(sptSparseTensor *tsr, int force, int tk, int cmode_stsrt, int num_cmode);

void sptSparseTensorSortIndexAtMode(sptSparseTensor *tsr, sptIndex const mode, int force);
void sptSparseTensorSortIndexCustomOrder(sptSparseTensor *tsr, sptIndex const *  mode_order, int force, int tk);
void sptSparseTensorSortIndexMorton(
    sptSparseTensor *tsr, 
    int force,
    sptNnzIndex begin,
    sptNnzIndex end,
    sptElementIndex sb_bits,
    int tk);
void sptSparseTensorSortIndexExceptSingleModeRowBlock(
    sptSparseTensor *tsr, 
    int force,
    sptNnzIndex begin,
    sptNnzIndex end,
    sptIndex * const mode_order,
    sptElementIndex sk_bits,
    int tk);
void sptSparseTensorSortIndexRowBlock(
    sptSparseTensor *tsr, 
    int force,
    sptNnzIndex begin,
    sptNnzIndex end,
    sptElementIndex sk_bits,
    int tk);
void sptSparseTensorSortIndexSingleMode(sptSparseTensor *tsr, int force, sptIndex mode, int tk);
void sptSparseTensorSortIndexExceptSingleMode(sptSparseTensor *tsr, int force, sptIndex * mode_order, int tk);
void sptSparseTensorSortIndexExceptSingleModeMorton(sptSparseTensor *tsr, int force, sptIndex * mode_order, sptElementIndex sb_bits, int tk);
int sptSparseTensorMixedOrder(
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    int const tk);
int sptSparseTensorSortPartialIndex(
    sptSparseTensor *tsr, 
    sptIndex const *  mode_order,
    const sptElementIndex sb_bits,
    int const tk);
void sptSparseTensorCalcIndexBounds(sptIndex inds_low[], sptIndex inds_high[], const sptSparseTensor *tsr);
int spt_ComputeSliceSizes(
    sptNnzIndex * slice_nnzs, 
    sptSparseTensor * const tsr,
    sptIndex const mode);
void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp);
double sptSparseTensorDensity(sptSparseTensor const * const tsr);

/* Renumbering */
void sptIndexRenumber(sptSparseTensor * tsr, sptIndex ** newIndices, int renumber, sptIndex iterations, sptElementIndex sb_bits, int tk, int impl_num);
void orderit(sptSparseTensor *tsr, sptIndex **newIndices, int const renumber, sptIndex const iterations);

/* Sparse tensor HiCOO */
int sptNewSparseTensorHiCOO(
    sptSparseTensorHiCOO *hitsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptNnzIndex nnz,
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits);
int sptNewSparseTensorHiCOO_NoNnz(
    sptSparseTensorHiCOO *hitsr, 
    const sptIndex nmodes, 
    const sptIndex ndims[],
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits);
void sptFreeSparseTensorHiCOO(sptSparseTensorHiCOO *hitsr);
int sptSparseTensorToHiCOO(
    sptSparseTensorHiCOO *hitsr, 
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits,
    int const tk);
int sptSparseTensorToHiCOOFidx(
    sptSparseTensorHiCOO *hitsr,
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr,
    sptIndex num_cmodes, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits,
    int const tk);
int sptSparseTensorToBCOO(
    sptSparseTensorHiCOO *hitsr,
    sptNnzIndex *max_nnzb,
    sptSparseTensor *tsr, 
    const sptElementIndex sb_bits,
    const sptElementIndex sk_bits,
    const sptElementIndex sc_bits,
    int const tk);
int sptDumpSparseTensorHiCOO(sptSparseTensorHiCOO * const hitsr, FILE *fp);
void sptLoadShuffleFile(sptSparseTensor *tsr, FILE *fs, sptIndex ** map_inds);
void sptSparseTensorStatusHiCOO(sptSparseTensorHiCOO *hitsr, FILE *fp);
double SparseTensorFrobeniusNormSquaredHiCOO(sptSparseTensorHiCOO const * const hitsr);
int sptSetKernelPointers(
    sptNnzIndexVector *kptr,
    sptNnzIndexVector *knnzs,
    sptSparseTensor *tsr, 
    const sptElementIndex sk_bits);


/* Sparse tensor unary operations */
int sptSparseTensorMulScalar(sptSparseTensor *X, sptValue const a);
int sptSparseTensorDivScalar(sptSparseTensor *X, sptValue const a);

/* Sparse tensor binary operations */
int sptSparseTensorAdd(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor * const Y);
int sptSparseTensorSub(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor * const Y);
int sptSparseTensorAddOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads);
int sptSparseTensorSubOMP(sptSparseTensor *Y, sptSparseTensor *X, int const nthreads);

int sptSparseTensorDotMul(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor * const Y);
int sptSparseTensorDotMulEq(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor * const Y);
int sptOmpSparseTensorDotMulEq(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor * const Y);
int sptCudaSparseTensorDotMulEq(sptSparseTensor *Z, const sptSparseTensor *X, const sptSparseTensor *Y);
int sptSparseTensorDotDiv(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor * const Y);

int sptSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor * const X, sptMatrix *const U, sptIndex mode);
int sptOmpSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor * const X, sptMatrix * const U, sptIndex mode);
int sptCudaSparseTensorMulMatrix(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, sptIndex const mode);
int sptCudaSparseTensorMulMatrixOneKernel(sptSemiSparseTensor *Y, sptSparseTensor *X, const sptMatrix *U, sptIndex const mode, sptIndex const impl_num, sptNnzIndex const smen_size);

int sptSparseTensorMulVector(sptSemiSparseTensor *Y, sptSparseTensor * const X, sptValueVector * const V, sptIndex mode);

int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int nt, int output_sorting, int placement);
int sptBlockSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement, int mode);
int sptBlockSparseTensorMulTensorCOO(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement, int mode);

/**
 * Kronecker product
 */
int sptSparseTensorKroneckerMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);

/**
 * Khatri-Rao product
 */
int sptSparseTensorKhatriRaoMul(sptSparseTensor *Y, const sptSparseTensor *A, const sptSparseTensor *B);


/**
 * Matricized tensor times Khatri-Rao product.
 */
int sptMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
int sptOmpMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRP_Reduce(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptMatrix * copy_mats[],    // temporary matrices for reduction
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk);
int sptOmpMTTKRP_Lock(sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    sptMutexPool * lock_pool);
int sptCudaMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex * const mats_order,    // Correspond to the mode order of X.
    sptIndex const mode,
    sptIndex const impl_num);
int sptCudaMTTKRPOneKernel(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    sptIndex * const mats_order,    // Correspond to the mode order of X.
    sptIndex const mode,
    sptIndex const impl_num);
int sptCudaMTTKRPSM(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    sptIndex * const mats_order,    // Correspond to the mode order of X.
    sptIndex const mode,
    sptIndex const impl_num);
int sptCudaMTTKRPDevice(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptIndex rank,
    const sptIndex stride,
    const sptIndex * Xndims,
    sptIndex ** const Xinds,
    const sptValue * Xvals,
    const sptIndex * dev_mats_order,
    sptValue ** dev_mats,
    sptValue * dev_scratch);
int sptSplittedMTTKRP(
    sptSparseTensor const *const X,
    sptMatrix *mats[],
    sptIndex const mats_order[],
    sptIndex const mode,
    sptValueVector *scratch,
    sptIndex const split_count[]
);

/* Coarse GPU */
int sptCudaCoarseMTTKRP(
    sptSparseTensor const * const X,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    sptIndexVector const * const mats_order,    // Correspond to the mode order of X.
    sptIndex const mode);

/**
 * Matricized tensor times Khatri-Rao product for HiCOO tensors
 */
int sptMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
int sptMTTKRPHiCOO_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode);
int sptOmpMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOO_MatrixTiling(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb,
    int balanced);
int sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptRankMatrix * copy_mats[],    // temporary matrices for reduction
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb,
    int balanced);
int sptOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce_Two(
    sptSparseTensorHiCOO const * const hitsr,
    sptRankMatrix * mats[],     // mats[nmodes] as temporary space.
    sptRankMatrix * copy_mats[],    // temporary matrices for reduction
    sptIndex const mats_order[],    // Correspond to the mode order of X.
    sptIndex const mode,
    const int tk,
    const int tb);
int sptCudaMTTKRPHiCOO(
    sptSparseTensorHiCOO const * const hitsr,
    sptMatrix ** const mats,     // mats[nmodes] as temporary space.
    sptIndex * const mats_order,    // Correspond to the mode order of X.
    sptIndex const mode,
    sptNnzIndex const max_nnzb,
    int const impl_num);
int sptMTTKRPKernelHiCOO(
    const sptIndex mode,
    const sptIndex nmodes,
    const sptNnzIndex nnz,
    const sptNnzIndex max_nnzb,
    const sptIndex R,
    const sptIndex stride,
    const sptElementIndex sb_bits,
    const sptElementIndex sc_bits,
    const sptIndex blength,
    const int impl_num,
    const sptNnzIndex kptr_begin,
    const sptNnzIndex kptr_end,
    sptIndex * const dev_ndims,
    sptNnzIndex * const dev_cptr,
    sptNnzIndex * const dev_bptr,
    sptBlockIndex ** const dev_binds,
    sptElementIndex ** const dev_einds,
    sptValue * const dev_values,
    sptIndex * const dev_mats_order,
    sptValue ** const dev_mats);
int sptTTMHiCOO_MatrixTiling(
    sptSparseTensorHiCOO * const Y,
    sptSparseTensorHiCOO const * const X,
    sptRankMatrix * U,     // mats[nmodes] as temporary space.
    sptIndex const mode);

/// Binary search
sptNnzIndex sptBinarySearch(sptIndex *array, int arrayStart, int arrayEnd, sptIndex target);

// Hash table for SPA: Cuckcoo
typedef struct CK_table{
    unsigned int size; // realsize = 2^size
    unsigned int shift; // for hash function
    sptNnzIndex len;
    // LP_node_t *item;
    // int* stat;
    char* stat[2];
    unsigned long long* key[2];
    sptValue* val[2];
}CK_table_t;
typedef struct pos_key_pair{
    unsigned int pos;
    unsigned long long key;
    sptValue val;
}pkpair_t;

CK_table_t* CK_htCreate(unsigned int ht_size);
void CK_htUpdate(CK_table_t *CK_t, unsigned int pos, unsigned long long key, sptValue val);
unsigned int CK_htInsert(CK_table_t *CK_t, unsigned int tableidx, unsigned long long key, sptValue val);
unsigned long long CK_htGet(CK_table_t *CK_t, unsigned long long key);
void CK_htFree( CK_table_t *CK_t);
// Hash table for SPA: Linear probing

typedef struct LP_node{
    unsigned long long key;
    sptValue val;
    char stat; // 0 for empty, 1 for valid, 2 for delete/
}LP_node_t; // An item of LP hash table

// typedef size_t (*HashFunc)(unsigned long long key)
typedef struct LP_table{
    unsigned int size; // realsize = 2^size
    unsigned int shift; // for hash function
    sptNnzIndex len;
    // LP_node_t *item;
    char* stat;
    unsigned long long* key;
    sptValue *val;
}LP_table_t;

// Hash Functions 
unsigned int MultShift_htHashCode(unsigned long long key, unsigned int shift);
unsigned int MultShift_htHashCode1(unsigned long long key, unsigned int shift);
unsigned int MultAddShift_htHashcode(unsigned long long key, unsigned int shift);
uint32_t Murmur3_32_htHashCode(const void *key, size_t len, uint32_t seed, unsigned int shift);

LP_table_t* LP_htCreate(unsigned int ht_size);
void LP_htUpdate(LP_table_t *LP_t, unsigned long long key, sptValue val);
unsigned int LP_htInsert(LP_table_t *LP_t, unsigned long long key, sptValue val);
sptValue LP_htGet(LP_table_t *LP_t, unsigned long long key);
void LP_htFree( LP_table_t *LP_t);

void QP_htUpdate(LP_table_t *LP_t, unsigned long long key, sptValue val);
unsigned int QP_htInsert(LP_table_t *LP_t, unsigned long long key, sptValue val);
sptValue QP_htGet(LP_table_t *LP_t, unsigned long long key);

void HS_htUpdate(LP_table_t *LP_t, unsigned long long key, sptValue val);
unsigned int HS_htInsert(LP_table_t *LP_t, unsigned long long key, sptValue val);
sptValue HS_htGet(LP_table_t *LP_t, unsigned long long key);

void PHF_htUpdate(LP_table_t *LP_t, unsigned long long key, sptValue val, int min_key, int sub_ht_size, int key_split_size);
unsigned int PHF_htInsert(LP_table_t *LP_t, unsigned long long key, sptValue val, int min_key, int sub_ht_size, int key_split_size);
sptValue PHF_htGet(LP_table_t *LP_t, unsigned long long key, int min_key, int sub_ht_size, int key_split_size);

// void PHF_htUpdate(LP_table_t *LP_t, unsigned long long key, sptValue val);
// unsigned int PHF_htInsert(LP_table_t *LP_t, unsigned long long key, sptValue val);
// sptValue PHF_htGet(LP_table_t *LP_t, unsigned long long key);

void SEG_htUpdate(LP_table_t *LP_t, unsigned long long key, sptValue val, unsigned int key_idx, int nnz_each_seg_shift, int sub_ht_size);
unsigned int SEG_htInsert(LP_table_t *LP_t, unsigned long long key, sptValue val, unsigned int key_idx, int nnz_each_seg_shift, int sub_ht_size);
sptValue SEG_htGet(LP_table_t *LP_t, unsigned long long key, unsigned int key_idx, int nnz_each_seg_shift, int sub_ht_size);



void RH_htUpdate(LP_table_t *LP_t, unsigned long long key, sptValue val);
unsigned int RH_htInsert(LP_table_t *LP_t, unsigned long long key, sptValue val);
sptValue RH_htGet(LP_table_t *LP_t, unsigned long long key);

/// Hash table for SPA
typedef struct node{
    unsigned long long key;
    sptValue val;
    struct node *next;
}node_t;

typedef struct{
    int size;
    unsigned int shift;
    node_t **list;
}table_t;

table_t *htCreate(const unsigned int size);
unsigned int htHashCode(unsigned long long key);
unsigned int htHashCode_size(unsigned long long key, unsigned int ht_size);
void htUpdate( table_t *t, unsigned long long key, sptValue val);
// void htInsert( table_t *t, unsigned long long key, sptValue val);
unsigned int htInsert( table_t *t, unsigned long long key, sptValue val);
sptValue htGet( table_t *t,unsigned long long key);
void htFree( table_t *t);


/// Hash table for the second tensor
typedef struct {
    unsigned int    len;        /// length
    unsigned int    cap;        /// capacity
    unsigned long long* key_FM; /// key with free modes of the second tensor
    unsigned int* key_FM_idx;            /// Guofeng Add: record the idx of each key FM
    sptValue* val;              /// data
} tensor_value;

typedef struct t_node{
    unsigned long long key;
    tensor_value val;
    struct t_node *next;
}tensor_node_t;

typedef struct{
    int size;
    int shift;
    tensor_node_t **list;
}tensor_table_t;

typedef struct LP_t_node{
    char stat;
    unsigned long long key;
    tensor_value val_ptr;
}LP_tensor_node_t;
typedef struct LP_tensor_table{
    unsigned int size; // realsize = 2^size
    unsigned int shift; // for hash function
    sptNnzIndex len;
    LP_tensor_node_t* LP_node_list;
}LP_tensor_table_t;
// typedef struct LP_tensor_table{
//     unsigned int size; // realsize = 2^size
//     unsigned int shift; // for hash function
//     sptNnzIndex len;
//     // LP_node_t *item;
//     char* stat;
//     unsigned long long* key;
//     tensor_value* val_ptr;
// }LP_tensor_table_t;
typedef struct CK_t_node{
    char stat;
    unsigned long long key;
    tensor_value val_ptr;
}CK_tensor_node_t;
typedef struct CK_tensor_table{
    unsigned int size;
    unsigned int shift;
    sptNnzIndex len;
    CK_tensor_node_t* CK_node_list[2];
}CK_tensor_table_t;

typedef struct{
    unsigned int    len;        /// length
    unsigned int    cap;        /// capacity
    unsigned long long* key_FM; /// key with free modes of the second tensor
    tensor_table_t** b_ht;       /// data
}tensor_block_entry;

typedef struct t_b_node{
    unsigned long long key;
    tensor_block_entry entry;
    struct t_b_node *next;
}tensor_block_node_t;

typedef struct{
    int size;
    tensor_block_node_t **list;
}tensor_block_table_t;

int tensor_htNewValueVector(tensor_value *vec, unsigned int len, unsigned int cap);
int tensor_htAppendValueVector(tensor_value *vec, unsigned long long key_FM, sptValue val);
void tensor_htFreeValueVector(tensor_value *vec);

tensor_table_t *tensor_htCreate(const unsigned int size);
unsigned int tensor_htHashCode(unsigned long long key);
unsigned int tensor_htHashCode_size(unsigned long long key, unsigned int ht_size);
void tensor_htUpdate( tensor_table_t *t, unsigned long long key_cmodes, unsigned long long key_fmodes, sptValue value);
void tensor_htInsert( tensor_table_t *t, unsigned long long key_cmodes, unsigned long long key_fmodes, sptValue value);
tensor_value tensor_htGet( tensor_table_t *t,unsigned long long key);
void tensor_htFree( tensor_table_t *t);

/*LP_tensor_table_t*/
LP_tensor_table_t* LP_tensor_htCreate(unsigned int ht_size);
void LP_tensor_htUpdate(LP_tensor_table_t *LP_t, unsigned long long key_cmodes,  unsigned long long key_fmodes, sptValue value);
unsigned int LP_tensor_htInsert(LP_tensor_table_t *LP_t, unsigned long long key_cmodes,  unsigned long long key_fmodes, sptValue value);
tensor_value LP_tensor_htGet(LP_tensor_table_t *LP_t, unsigned long long key);
void LP_tensor_htFree(LP_tensor_table_t *LP_t);

/*CK_tensor_table_t*/
CK_tensor_table_t* CK_tensor_htCreate(unsigned int ht_size);
unsigned int CK_htHashCode(int func, unsigned long long key, unsigned int shift);
void CK_tensor_htUpdate(CK_tensor_table_t *CK_t, unsigned long long key_cmodes,  unsigned long long key_fmodes, sptValue value);
unsigned int CK_tensor_htInsert(CK_tensor_table_t *CK_t, unsigned long long key_cmodes,  unsigned long long key_fmodes, sptValue value);
tensor_value CK_tensor_htGet(CK_tensor_table_t *CK_t, unsigned long long key);
void CK_tensor_htFree(CK_tensor_table_t *CK_t);

tensor_block_table_t* tensor_b_htCreate(const unsigned int size);
unsigned int tensor_b_htHashCode(unsigned long long key);
unsigned int tensor_b_htHashCode_size(unsigned long long key, unsigned int b_ht_size);
void tensor_b_htUpdate(tensor_block_table_t *t, unsigned long long key_cmode_b, unsigned long long key_fmode_b, tensor_table_t* b_ht);
void tensor_b_htInsert(tensor_block_table_t *t, unsigned long long key_cmode_b, unsigned long long key_fmode_b, tensor_table_t* b_ht);
tensor_block_entry tensor_b_htGet(tensor_block_table_t* t, unsigned long long key_b);
void tensor_b_htFree(tensor_block_table_t *t);
int tensor_b_htNewBlockEntry(tensor_block_entry *entry, unsigned int len, unsigned int cap);
int tensor_b_htAppendBlockEntry(tensor_block_entry *entry, unsigned long long key_FM_b, tensor_table_t* b_ht);
void tensor_b_htFreeBlockEntry(tensor_block_entry *entry);

#ifdef __cplusplus
}
#endif


#endif
