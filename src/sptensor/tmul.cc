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
#include <iostream>
#include <ParTI.h>
#include "sptensor.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <numa.h>
#include <sys/time.h>
#include "malloc.h"
#include <string>
#include <vector>
#include "absl/strings/str_join.h"
#include "absl/container/flat_hash_map.h"
#include <cmath>
#include <algorithm>
// #include<boost/sort/spreadsort/spreadsort.hpp>
// #include "mkl.h"
#include <hashmap.h>
#include <cuckoohash_map.hh>
#include <unordered_map>
#include <unordered_set>
#include <set>
using namespace std;


void printVector(const std::vector<std::vector<int>>& vec) {
    for (const auto& innerVec : vec) {
        for (int val : innerVec) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// used for unordered set
struct VectorHash {
    std::size_t operator()(const std::vector<int>& vec) const {
        std::size_t seed = 0;
        for (const auto& elem : vec) {
            seed ^= std::hash<int>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct VectorEqual {
    bool operator()(const std::vector<int>& lhs, const std::vector<int>& rhs) const {
        return lhs == rhs;
    }
};

// 合并两个 std::unordered_set<std::vector<int>> 的函数
void merge_sets(std::unordered_set<std::vector<int>, VectorHash, VectorEqual>& set1, 
                const std::unordered_set<std::vector<int>, VectorHash, VectorEqual>& set2) {
    for (const auto& vec : set2) {
        set1.insert(vec);
    }
}


int find_position(const std::unordered_set<std::vector<int>, VectorHash, VectorEqual>& mySet, const std::vector<int>& target) {
    auto it = mySet.find(target);
    if (it != mySet.end()) {
        return std::distance(mySet.begin(), it) + 1;
    }
    return -1; // 如果未找到，返回 -1
}


struct VectorLess {
    bool operator()(const std::vector<int>& lhs, const std::vector<int>& rhs) const {
        return lhs < rhs;
    }
};

int find_position(const std::set<std::vector<int>, VectorLess>& mySet, const std::vector<int>& target) {
    auto it = mySet.find(target);
    if (it != mySet.end()) {
        return std::distance(mySet.begin(), it) + 1;
    }
    return -1; // 如果未找到，返回 -1
}

void printSet(const std::set<std::vector<int>, VectorLess>& mySet) {
    for (const auto& vec : mySet) {
        for (int val : vec) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

bool my_cmp(pkpair_t x, pkpair_t y) {
    return x.key < y.key;
}

sptValue standard_deviation(sptValue arr[], int n) {
    double mean = std::accumulate(arr, arr + n, 0.0) / n;
    double sum_sq_diff = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq_diff += pow((arr[i] - mean), 2);
    }
    return sqrt(sum_sq_diff / n);
}

void rotate(sptIndex arr[], sptIndex n, sptIndex k) {

    // 对k进行取模，防止旋转次数大于n
    k = k % n;
    
    // 将前k个元素翻转
    for (int i = 0; i < k / 2; i++) {
        int temp = arr[i];
        arr[i] = arr[k - i - 1];
        arr[k - i - 1] = temp;
    }
    
    // 将后n-k个元素翻转
    for (int i = k; i < (n + k) / 2; i++) {
        int temp = arr[i];
        arr[i] = arr[n - i + k - 1];
        arr[n - i + k - 1] = temp;
    }
    
    // 将整个数组翻转
    for (int i = 0; i < n / 2; i++) {
        int temp = arr[i];
        arr[i] = arr[n - i - 1];
        arr[n - i - 1] = temp;
    }
}


extern "C" int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement);
/** All combined:
 * 0: COOY + SPA
 * 1: COOY + HTA
 * 2: HTY + SPA
 * 3: HTY + HTA
 * 4: HTY + HTA on HM
 **/
int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement)
{
    // Experiment modes
    int experiment_modes;
    sscanf(getenv("EXPERIMENT_MODES"), "%d", &experiment_modes);

//0: COOY + SPA
  if(experiment_modes == 0){
    int result;
    /// The number of threads
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    /// Shuffle tensor indices according to mode_order_X
    sptSparseTensorShuffleModes(X, mode_order_X);
    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    /// Shuffle Y indices and sort Y as the order of free modes -> contract modes
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_Y);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }
    sptAssert(ci == num_cmodes);
    /// Shuffle tensor indices according to mode_order_Y
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    // printf("Permuted Y:\n");
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);  
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time );

    //printf("Sorted X:\n");
    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    //printf("Sorted Y:\n");
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

    /// Set fidx_X: indexing the combined free indices and fidx_Y: indexing the combined contract indices
    sptNnzIndexVector fidx_X, fidx_Y;
    //sptStartTimer(timer);
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    /// Set indices for contract modes, use Y
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);
    //sptStopTimer(timer);
    //sptPrintElapsedTime(timer, "Set fidx X,Y");
    //sptPrintElapsedTime(timer, "Set fidx X");
    // printf("fidx_X: \n");
    // sptDumpNnzIndexVector(&fidx_X, stdout);
    // printf("fidx_Y: \n");
    // sptDumpNnzIndexVector(&fidx_Y, stdout);
    free(mode_order_X);
    free(mode_order_Y);
    // printf("fidx_X.len: %ld\n",fidx_X.len);
    // printf("fidx_Y.len: %ld\n", fidx_Y.len);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *)malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
    }   
    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *)malloc(tk * sizeof (sptSparseTensor));

    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;

    sptNewTimer(&timer_SPA, 0);

    sptStartTimer(timer);

    // For the progress
    int fx_counter = fidx_X.len;

    #pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, fidx_Y, nmodes_X, nmodes_Y, num_cmodes, Z_tmp, fx_counter)   
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        //Print the progress 
        fx_counter--;
        //if (fx_counter % 1 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        /// Allocate the SPA buffer
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        sptIndexVector * spa_inds = (sptIndexVector*)malloc(nmodes_spa * sizeof(sptIndexVector));
        sptValueVector spa_vals;
        for(sptIndex m = 0; m < nmodes_spa; ++m)
            sptNewIndexVector(&spa_inds[m], 0, 0);
        sptNewValueVector(&spa_vals, 0, 0);

        /// Allocate a small index buffer
        sptIndexVector inds_buf;
        sptNewIndexVector(&inds_buf, (nmodes_Y - num_cmodes), (nmodes_Y - num_cmodes));
        //printf("\nzX: [%lu, %lu]\n", fx_begin, fx_end);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
            sptStartTimer(timer_SPA);
        }
        sptIndex fy_ptr = 0;
        sptIndex fy_ptr_prev = 0;
        /// zX has common free indices
        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0) {
                sptStartTimer(timer_SPA); 
            }
            sptValue valX = X->values.data[zX];
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                 cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                 //printf("\ncmode_index_X[%lu]: %lu", i, cmode_index_X[i]);
             }

            sptNnzIndex fy_begin = -1;
            sptNnzIndex fy_end = -1;
            unsigned int current_idx = 0;
            
            while(fy_ptr < fidx_Y.len - 1){
                for(sptIndex i = 0; i < num_cmodes; i++){
                    if(cmode_index_X.data[i] != Y->inds[i].data[fidx_Y.data[fy_ptr]])
                        break;
                    if(i == (num_cmodes - 1)){
                        fy_begin = fidx_Y.data[fy_ptr];
                        fy_end = fidx_Y.data[fy_ptr+1];
                        break;
                    }
                }
                if(fy_begin != -1 || fy_end != -1){
                    fy_ptr_prev = fy_ptr; 
                    break;
                }
                fy_ptr++;
            }   
            if(fy_ptr == fidx_Y.len -1) fy_ptr = fy_ptr_prev;
            
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_free_mode += sptElapsedTime(timer_SPA);
            }
        
            if (fy_begin == -1 || fy_end == -1) continue;
            //printf("zX: %lu, valX: %.2f, cmode_index_X[0]: %u, zY: [%lu, %lu]\n", zX, valX, cmode_index_X.data[0], fy_begin, fy_end);

            if (tid == 0){
                sptStartTimer(timer_SPA);               
            }

            /// zY has common contraction indices
            char tmp[32];
            char index_str[128]; 
            long int tmp_key;
            for(sptNnzIndex zY = fy_begin; zY < fy_end; ++ zY) {    // Loop nnzs inside a Y fiber
                for(sptIndex m = 0; m < nmodes_spa; ++m)
                    inds_buf.data[m] = Y->inds[m + num_cmodes].data[zY];
                //printf("inds_buf:\n");
                //sptDumpIndexVector(&inds_buf, stdout);
                long int found = sptInIndexVector(spa_inds, nmodes_spa, spa_inds[0].len, &inds_buf);
                if( found == -1) {
                    for(sptIndex m = 0; m < nmodes_spa; ++m)
                        sptAppendIndexVector(&spa_inds[m], Y->inds[m + num_cmodes].data[zY]);
                    sptAppendValueVector(&spa_vals, Y->values.data[zY] * valX);
                } else {
                    spa_vals.data[found] += Y->values.data[zY] * valX;
                }
            }   // End Loop nnzs inside a Y fiber
            //printf("spa_inds:\n");
            //for(sptIndex m = 0; m < nmodes_spa; ++m) {
            //    printf("[m%u]:\n", m);
            //    sptDumpIndexVector(&spa_inds[m], stdout);
            //}
            //printf("spa_vals:\n");
            //sptDumpValueVector(&spa_vals, stdout);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }

        }   // End Loop nnzs inside a X fiber
    
        if (tid == 0){
            sptStartTimer(timer_SPA);   
        }

        /// Write back to Z
        Z_tmp[tid].nnz += spa_vals.len;
     
        for(sptIndex i = 0; i < spa_vals.len; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }
        for(sptIndex m = 0; m < nmodes_spa; ++m) 
            sptAppendIndexVectorWithVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], &spa_inds[m]);
        sptAppendValueVectorWithVector(&Z_tmp[tid].values, &spa_vals);  
             
        //printf("Z:\n");
        //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        /// Free SPA buffer
        for(sptIndex m = 0; m < nmodes_spa; ++m){
            sptFreeIndexVector(&(spa_inds[m]));   
         }     
         sptFreeValueVector(&spa_vals);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
    }   // End Loop fiber pointers of X

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);

    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
    }
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
        }
    } 

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");
  } 

//1: COOY + HTA
  if(experiment_modes == 1){
    int result;
    /// The number of threads
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    /// Shuffle tensor indices according to mode_order_X
    sptSparseTensorShuffleModes(X, mode_order_X);
    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    /// Shuffle Y indices and sort Y as the order of free modes -> contract modes
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_Y);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }
    sptAssert(ci == num_cmodes);
    /// Shuffle tensor indices according to mode_order_Y
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    // printf("Permuted Y:\n");
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Processing]: %.6f s\n", X_time + sptElapsedTime(timer));

    //printf("Sorted X:\n");
    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    //printf("Sorted Y:\n");
    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

    /// Set fidx_X: indexing the combined free indices and fidx_Y: indexing the combined contract indices
    sptNnzIndexVector fidx_X, fidx_Y;
    //sptStartTimer(timer);
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    /// Set indices for contract modes, use Y
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);
    //sptStopTimer(timer);
    //sptPrintElapsedTime(timer, "Set fidx X,Y");
    //sptPrintElapsedTime(timer, "Set fidx X");
    // printf("fidx_X: \n");
    // sptDumpNnzIndexVector(&fidx_X, stdout);
    // printf("fidx_Y: \n");
    // sptDumpNnzIndexVector(&fidx_Y, stdout);
    free(mode_order_X);
    free(mode_order_Y);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *)malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
    }   
    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *)malloc(tk * sizeof (sptSparseTensor));

    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;

    sptNewTimer(&timer_SPA, 0);

    sptStartTimer(timer);

    // For the progress
    int fx_counter = fidx_X.len;

    #pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, fidx_Y, nmodes_X, nmodes_Y, num_cmodes, Z_tmp, fx_counter)   
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        //Print the progress 
        fx_counter--;
        //if (fx_counter % 1 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }

        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        long int nnz_counter = 0;
        
        /// Calculate key range for hashtable
        sptIndex* inds_buf = (sptIndex*)malloc((nmodes_spa + 1) * sizeof(sptIndex));
        sptIndex current_idx = 0;
        for(sptIndex i = 0; i < nmodes_spa + 1; i++) inds_buf[i] = 1;
        for(sptIndex i = 0; i < nmodes_spa;i++){
            for(sptIndex j = i; j < nmodes_spa;j++)
                inds_buf[i] = inds_buf[i] * Y->ndims[j + num_cmodes];
        }

        /// Create a hashtable for SPAs
        table_t *ht;
        const unsigned int ht_size = 10000;
        ht = htCreate(ht_size);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
        }

        /// zX has common free indices
        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }

            sptValue valX = X->values.data[zX];
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                 cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                 //printf("\ncmode_index_X[%lu]: %lu", i, cmode_index_X[i]);
             }

            sptNnzIndex fy_begin = -1;
            sptNnzIndex fy_end = -1;
            
            for(sptIndex j = 0; j < fidx_Y.len; j++){
                for(sptIndex i = 0; i< num_cmodes; i++){
                    if(cmode_index_X.data[i] != Y->inds[i].data[fidx_Y.data[j]]) break;
                    if(i == (num_cmodes - 1)){
                        fy_begin = fidx_Y.data[j];
                        fy_end = fidx_Y.data[j+1];
                        break;
                    }
                    //printf("\ni: %lu, current_idx: %lu, Y->inds[i].data[fidx_Y.data[current_idx]]: %lu\n", i, current_idx, Y->inds[i].data[fidx_Y.data[current_idx]]);
                }
                if (fy_begin != -1 || fy_end != -1) break;
            }
            
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_free_mode += sptElapsedTime(timer_SPA);
            }
        
            if (fy_begin == -1 || fy_end == -1) continue;
            //printf("zX: %lu, valX: %.2f, cmode_index_X[0]: %u, zY: [%lu, %lu]\n", zX, valX, cmode_index_X.data[0], fy_begin, fy_end);

            if (tid == 0) sptStartTimer(timer_SPA);               

            /// zY has common contraction indices
            for(sptNnzIndex zY = fy_begin; zY < fy_end; ++ zY) {    // Loop nnzs inside a Y fiber
                long int tmp_key = 0;    
                for(sptIndex m = 0; m < nmodes_spa; ++m)
                    tmp_key += Y->inds[m + num_cmodes].data[zY] * inds_buf[m + 1];
                sptValue val = htGet(ht, tmp_key);
                if(val == LONG_MIN) 
                    htInsert(ht, tmp_key, Y->values.data[zY] * valX);
                else    
                    htUpdate(ht, tmp_key, val + (Y->values.data[zY] * valX));
                //printf("val: %f\n", val);
            }
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }

        }   // End Loop nnzs inside a X fiber
    
        if (tid == 0){
            sptStartTimer(timer_SPA);   
        }

        /// Write back to Z
        for(int i = 0; i < ht->size; i++){
            node_t *temp = ht->list[i];
            while(temp){
                long int idx_tmp = temp->key;
                nnz_counter++;
                for(sptIndex m = 0; m < nmodes_spa; ++m) {
                    //printf("idx_tmp: %lu, m: %d, (idx_tmp inds_buf[m])/inds_buf[m+1]): %d\n", idx_tmp, m, (idx_tmp%inds_buf[m])/inds_buf[m+1]);
                    sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%inds_buf[m])/inds_buf[m+1]);
                }
                //printf("val: %f\n", temp->val);
                sptAppendValueVector(&Z_tmp[tid].values, temp->val);
                node_t* pre = temp;
                temp = temp->next;
                free(pre);
            }
        }

        Z_tmp[tid].nnz += nnz_counter;
        for(sptIndex i = 0; i < nnz_counter; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }

        // release spa hashtable
        htFree(ht);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
    }   // End Loop fiber pointers of X

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[Index Search]: %.2f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.2f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);

    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
    }
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
        }
    } 

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");
  }

//2: HTY + SPA
  if(experiment_modes == 2){
    int result;
    /// The number of threads
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    /// Shuffle tensor indices according to mode_order_X
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    /// Convert Y into a hashtable
    /// Create a hashtable 
    tensor_table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    // omp lock
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    /// Calculate key range for Y hashtable
    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;
    printf("Y_nnz: %ld\n", Y_nnz);
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        /// Contract modes of Y
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        /// Free modes of Y
        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        unsigned pos = tensor_htHashCode_size(key_cmodes, Y_nnz);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    // Release omp lock
    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);

    sptStopTimer(timer);   
    total_time += sptElapsedTime(timer);  
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time );
    
    /// Set fidx_X: indexing the combined free indices
    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    sptStartTimer(timer);

    // For the progress
    int fx_counter = fidx_X.len;

    #pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, fx_counter)       
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        //Print the progress 
        fx_counter--;
        //if (fx_counter % 100 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        /// Allocate the SPA buffer
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        sptIndexVector * spa_inds = (sptIndexVector*)malloc(nmodes_spa * sizeof(sptIndexVector));
        sptValueVector spa_vals;
        for(sptIndex m = 0; m < nmodes_spa; ++m)
            sptNewIndexVector(&spa_inds[m], 0, 0);
        sptNewValueVector(&spa_vals, 0, 0);

        /// Allocate a small index buffer
        sptIndexVector inds_buf;
        sptNewIndexVector(&inds_buf, (nmodes_Y - num_cmodes), (nmodes_Y - num_cmodes));
        //printf("\nzX: [%lu, %lu]\n", fx_begin, fx_end);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
        }

        /// zX has common free indices
        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
            if (tid == 0) {
                sptStartTimer(timer_SPA);  
            }             
            sptValue valX = X->values.data[zX];
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
            }

            unsigned long long key_cmodes = 0;    
            for(sptIndex m = 0; m < num_cmodes; ++m)
                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];
            //printf("key_cmodes: %d\n", key_cmodes);    

            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);  
            //printf("Y_val.len: %d\n", Y_val.len);
            unsigned int my_len = Y_val.len;
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_free_mode += sptElapsedTime(timer_SPA);
            }
            if(my_len == 0) continue;

            if (tid == 0) sptStartTimer(timer_SPA);               

            for(int i = 0; i < my_len; i++){
                unsigned long long fmode =  Y_val.key_FM[i];
                float result = Y_val.val[i] * valX;

                for(sptIndex m = 0; m < nmodes_spa; ++m)
                    inds_buf.data[m] =  (fmode%Y_fmode_inds[m])/Y_fmode_inds[m+1];
                //printf("inds_buf:\n");
                //sptDumpIndexVector(&inds_buf, stdout);
                long int found = sptInIndexVector(spa_inds, nmodes_spa, spa_inds[0].len, &inds_buf);
                if( found == -1) {
                    for(sptIndex m = 0; m < nmodes_spa; ++m)
                        sptAppendIndexVector(&spa_inds[m], (fmode%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                    sptAppendValueVector(&spa_vals, result);
                } else {
                    spa_vals.data[found] += result;
                }
            }

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }
            
        }   // End Loop nnzs inside a X fiber

        if (tid == 0) sptStartTimer(timer_SPA);    

        /// Write back to Z
        Z_tmp[tid].nnz += spa_vals.len;
     
        for(sptIndex i = 0; i < spa_vals.len; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }
        for(sptIndex m = 0; m < nmodes_spa; ++m) 
            sptAppendIndexVectorWithVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], &spa_inds[m]);
        sptAppendValueVectorWithVector(&Z_tmp[tid].values, &spa_vals);  
             
        //printf("Z:\n");
        //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        /// Free SPA buffer
        for(sptIndex m = 0; m < nmodes_spa; ++m){
            sptFreeIndexVector(&(spa_inds[m]));   
         }     
         sptFreeValueVector(&spa_vals);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
    }

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
            //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        }
    } 

    //  for(int i = 0; i < tk; i++)
    //      sptFreeSparseTensor(&Z_tmp[i]);
    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Writeback");
    sptStartTimer(timer);

    sptSparseTensorSortIndex(Z, 1, tk);

    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");

    //sptFreeTimer(timer);
    //sptFreeNnzIndexVector(&fidx_X);

    FILE *p_tz1 = fopen("mode2.txt", "w");
    sptDumpSparseTensor(Z, 0, p_tz1);
    fclose(p_tz1);
  }  

//3: HTY(CH) + HTA(CH)
  if(experiment_modes == 3){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    /// Shuffle tensor indices according to mode_order_X
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);
    
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    sptNnzIndexVector fidx_Y;
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);

    printf("[Num of subtensorY]:%lu\n", fidx_Y.len - 1);
    tensor_table_t *Y_ht;
    unsigned int Y_ht_size_log2 = ceil(log2(2*(fidx_Y.len-1)));
    Y_ht = tensor_htCreate(Y_ht_size_log2);    
    unsigned int Y_ht_size = Y_ht->size;

    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptTimer timer_hty;
    double time_hty_calckey = 0;
    sptNewTimer(&timer_hty, 0);

    sptNnzIndex Y_nnz = Y->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        //if (Y->values.data[i] <0.00000001) continue;
        int tid = omp_get_thread_num();
        if(tid == 0){
            sptStartTimer(timer_hty);
        }
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        
        if(tid == 0){
            sptStopTimer(timer_hty);
            time_hty_calckey += sptElapsedTime(timer_hty);
        }

        // unsigned pos = tensor_htHashCode_size(key_cmodes, Y_nnz);
        unsigned pos = MultShift_htHashCode(key_cmodes, Y_ht->shift);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);
    // int min_nnz_y = 10000, max_nnz_y = 0;
    sptNnzIndex empty_buckets = 0;
    sptIndex max_bucket_length = 0;
    sptNnzIndex num_collisions = 0;
    sptNnzIndex num_elements = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtY += sizeof(int) + Y_ht_size * sizeof(tensor_node_t*);
    for(sptNnzIndex i = 0; i < Y_ht_size; i++){
        tensor_node_t *list = Y_ht->list[i];
        tensor_node_t *temp = list;
        sptIndex count = -1;
        if(temp){
            while(temp){
                // bytes_HtY += sizeof(unsigned long long) + sizeof(tensor_value) + sizeof(tensor_node_t*);
                // max_nnz_y = std::max((int)temp->val.len, max_nnz_y);
                // min_nnz_y = std::min((int)temp->val.len, min_nnz_y);
                bytes_HtY += sizeof(tensor_node_t);
                bytes_HtY += (temp->val.len)*(sizeof(unsigned long long) + sizeof(sptValue));
                count++;
                temp = temp->next;
            }
            num_collisions += count;
            num_elements += (count + 1);
            max_bucket_length = max_bucket_length >= count ? max_bucket_length : count;
        }else{
            empty_buckets++;
        }
    }
    bytes_HtY = 2*sizeof(int) + (Y_ht->size) * sizeof(tensor_node_t*) + (fidx_Y.len-1)*sizeof(tensor_node_t) + Y->nnz*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Whole hashbucket]: %lu \n", Y_ht_size);
    printf("[Empty hashbucket]: %ld \n", empty_buckets);
    printf("[Wasted hash rate]: %%%.7lf\n", 100.0*(1.0*empty_buckets)/(1.0*Y_ht_size));
    printf("[Hash  collisions]: %ld \n", num_collisions);
    printf("[Max   collision]: %ld \n", max_bucket_length);
    printf("[Hash  elements]: %ld \n", num_elements);
    printf("[Loading Factor]: %.7lf \n", num_elements*1.0/Y_nnz);
    sptNnzIndex nnz_bucket = Y_ht_size - empty_buckets;
    printf("[Avg   collision]: %.7lf \n", 1.0*num_collisions/(1.0*nnz_bucket));
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Calc htKey]: %.6f s\n", time_hty_calckey);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[m];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    double time_calckey = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    
    // Some parameters to capture the HtA characteristics
    sptValue *collision_rate_HtA = (sptValue*)malloc((fidx_X.len - 1)*sizeof(sptValue));
    sptIndex *num_collison_HtA = (sptIndex*)malloc((fidx_X.len - 1)*sizeof(sptIndex));
    sptIndex *max_num_collison_HtA = (sptIndex*)malloc((fidx_X.len - 1)*sizeof(sptIndex));
    sptValue *loading_factor_HtA = (sptValue*)malloc((fidx_X.len - 1)*sizeof(sptValue));

    sptValue max_collision_rate_HtA = 0.0, max_loading_factor_HtA = 0.0;
    sptValue avg_collision_rate_HtA = 0.0, avg_loading_factor_HtA = 0.0;
    sptValue mid_collision_rate_HtA = 0.0, mid_loading_factor_HtA = 0.0;
    sptValue min_collision_rate_HtA = 10.0, min_loading_factor_HtA = 10.0;
    sptIndex avg_num_collison_HtA = 0, max_num_collison_all_HtA = 0;

    sptNnzIndex *bytes_HtA = (sptNnzIndex*)calloc((fidx_X.len - 1), sizeof(sptNnzIndex));
    sptNnzIndex total_bytes_HtA = 0, max_bytes_HtA = 0;
    // int min_nnz_x = 10000, max_nnz_x = 0;
    // int min_nnz_z = 10000, max_nnz_z = 0;
    printf("[Num of subtensorX]: %lu\n", fidx_X.len - 1);
    sptStartTimer(timer);
    // For the progress
    int fx_counter = fidx_X.len;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
        int tid = omp_get_thread_num();
        // double start_time = omp_get_wtime();
        fx_counter--;
        //if (fx_counter % 100 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
        if (tid == 0){
            sptStartTimer(timer_SPA);
        }
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];

        // max_nnz_x = std::max(max_nnz_x, (int)(fx_end-fx_begin));
        // min_nnz_x = std::min(min_nnz_x, (int)(fx_end-fx_begin));
        /// hashtable size
        sptNnzIndex nnz_upper_bound = 0;
        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++zX){
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
            }
            unsigned long long key_cmodes = 0;    
            for(sptIndex m = 0; m < num_cmodes; ++m)
                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1]; 
            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes); 
            nnz_upper_bound += Y_val.len;
        }
        unsigned int ht_size = std::min(16, std::max((int)ceil(log2(4*nnz_upper_bound)),2));
        /// hashtable size
        // unsigned int shift = 16;
        // unsigned int ht_size = 1 << shift;
        // unsigned int ht_size = 10000;
        // unsigned int ht_size = 8192; // 2^13
        // const unsigned int ht_size = 100;
        hashtable_a_size = 1 << ht_size;
        // const unsigned int ht_size = 9973;
        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
        long int nnz_counter = 0;
        sptIndex current_idx = 0;

        table_t *ht;
        ht = htCreate(ht_size);

        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_prep += sptElapsedTime(timer_SPA);
        }

        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {   
            sptValue valX = X->values.data[zX];  
            if (tid == 0) {
                sptStartTimer(timer_SPA);
            }       
            sptIndexVector cmode_index_X; 
            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
            for(sptIndex i = 0; i < num_cmodes; ++i){
                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
                //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
            }

            if (tid == 0) {
                sptStartTimer(timer_hash);
            }  
            unsigned long long key_cmodes = 0;    
            for(sptIndex m = 0; m < num_cmodes; ++m)
                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];  

            if (tid == 0) {
                sptStopTimer(timer_hash);
                time_calckey += sptElapsedTime(timer_hash);
            }  

            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);  
            //printf("Y_val.len: %d\n", Y_val.len);
            unsigned int my_len = Y_val.len;
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_free_mode += sptElapsedTime(timer_SPA);
            }
            if(my_len == 0) continue;

            if (tid == 0) {
                sptStartTimer(timer_SPA);       
            }        

            for(int i = 0; i < my_len; i++){
                unsigned long long fmode =  Y_val.key_FM[i];
                //printf("i: %d, Y_val.key_FM[i]: %lu, Y_val.val[i]: %f\n", i, Y_val.key_FM[i], Y_val.val[i]);
                sptValue spa_val = htGet(ht, fmode);
                float result = Y_val.val[i] * valX;
                if(spa_val == LONG_MIN) {
                    htInsert(ht, fmode, result);
                    nnz_counter++;
                }
                else    
                    htUpdate(ht, fmode, spa_val + result);
            }

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_spa += sptElapsedTime(timer_SPA);
            }
            
        }   // End Loop nnzs inside a X fiber

        if (tid == 0) {
            sptStartTimer(timer_SPA);    
        }

        
        collision_rate_HtA[fx_ptr] = 0.0;
        num_collison_HtA[fx_ptr] = 0;
        max_num_collison_HtA[fx_ptr] = 0;        
        bytes_HtA[fx_ptr] += (sizeof(int) + ht->size*(sizeof(node_t*)));

        for(int i = 0; i < ht->size; i++){
            node_t *temp = ht->list[i];
            sptIndex count_collison = 0;
            while(temp){
                // bytes_HtA[fx_ptr] += sizeof(unsigned long long) + sizeof(sptValue) + sizeof(node_t*);
                bytes_HtA[fx_ptr] += sizeof(node_t);
                count_collison++;
                unsigned long long idx_tmp = temp->key;
                //nnz_counter++;
                if (tid == 0) {
                    sptStartTimer(timer_hash);    
                }       
                for(sptIndex m = 0; m < nmodes_spa; ++m) {
                    //printf("idx_tmp: %lu, m: %d, (idx_tmp inds_buf[m])/inds_buf[m+1]): %d\n", idx_tmp, m, (idx_tmp%inds_buf[m])/inds_buf[m+1]);                   
                    sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
                }
                if (tid == 0) {
                    sptStopTimer(timer_hash);   
                    time_reversehash += sptElapsedTime(timer_hash);
                } 

                //printf("val: %f\n", temp->val);
                sptAppendValueVector(&Z_tmp[tid].values, temp->val);
                node_t* pre = temp;
                temp = temp->next;
                free(pre);
            }
            if(count_collison >= 1){
                num_collison_HtA[fx_ptr] += count_collison - 1;
                max_num_collison_HtA[fx_ptr] = std::max(max_num_collison_HtA[fx_ptr], count_collison - 1);
            }
        }
        // max_nnz_z = std::max((int)nnz_counter, max_nnz_z);
        // min_nnz_z = std::min((int)nnz_counter, min_nnz_z);
        // printf("bytes_HtA:%lu", bytes_HtA);
        Z_tmp[tid].nnz += nnz_counter;
        for(sptIndex i = 0; i < nnz_counter; ++i) {
            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {               
                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
            }
        }

        collision_rate_HtA[fx_ptr] = 1.0 * num_collison_HtA[fx_ptr] / nnz_counter;
        loading_factor_HtA[fx_ptr] = 1.0 * nnz_counter / ht->size;

        htFree(ht);
        if (tid == 0){
            sptStopTimer(timer_SPA);
            time_accumulate_z += sptElapsedTime(timer_SPA);
        }
        // double end_time = omp_get_wtime();
        // elapsed_time[tid] += end_time - start_time;
        // printf("Thread %d took %f seconds\n", omp_get_thread_num(), elapsed_time);
    }

    // for(sptIndex i = 0; i < tk; i++){
    //     printf("Thread %d took %f seconds\n", i, elapsed_time[i]);
    // }
    // printf("[Stddeviation]: %.6f \n", standard_deviation(elapsed_time, tk));
    // printf("[Coefficient of Variation]: %.6f\% \n", 100.0*standard_deviation(elapsed_time, tk)/(std::accumulate(elapsed_time, elapsed_time + tk, 0.0) / tk));
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Difference  ]: %.6f\%\n", 100*(main_computation - spa_total)/main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    // printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    printf("[Calculatekey]: %.6f s\n", time_calckey);
    printf("[Reverse Hash]: %.6f s\n", time_reversehash);
    printf("[Hasht_A Size]: %u\n", hashtable_a_size);
    // printf("[NNZ_X  range]: [%llu, %llu]\n", min_nnz_x, max_nnz_x);
    // printf("[NNZ_Y  range]: [%llu, %llu]\n", min_nnz_y, max_nnz_y);
    // printf("[NNZ_Z  range]: [%llu, %llu]\n", min_nnz_z, max_nnz_z);

    sptValue sum_collision_rate_HtA = 0.0;
    sptIndex sum_num_collison_HtA = 0;
    sptValue sum_loading_factor_HtA = 0.0;
    for(sptIndex i = 0; i < fidx_X.len - 1; i++){
        sum_collision_rate_HtA += collision_rate_HtA[i];
        sum_loading_factor_HtA += loading_factor_HtA[i];
        sum_num_collison_HtA += num_collison_HtA[i];

        max_collision_rate_HtA = std::max(collision_rate_HtA[i], max_collision_rate_HtA);
        min_collision_rate_HtA = std::min(collision_rate_HtA[i], min_collision_rate_HtA);
        max_loading_factor_HtA = std::max(loading_factor_HtA[i], max_loading_factor_HtA);
        min_loading_factor_HtA = std::min(loading_factor_HtA[i], min_loading_factor_HtA);
        max_num_collison_all_HtA = std::max(max_num_collison_HtA[i], max_num_collison_all_HtA);
    }
    avg_collision_rate_HtA = sum_collision_rate_HtA / (fidx_X.len - 1);
    avg_loading_factor_HtA = sum_loading_factor_HtA / (fidx_X.len - 1);
    avg_num_collison_HtA = sum_num_collison_HtA / (fidx_X.len - 1);
    mid_collision_rate_HtA = collision_rate_HtA[(fidx_X.len - 1)/2];
    mid_loading_factor_HtA = loading_factor_HtA[(fidx_X.len - 1)/2];

    printf("[Collision Rate HtA]: [%.6f, %.6f] with Avg: %.6f Mid: %.6f\n", min_collision_rate_HtA, max_collision_rate_HtA, avg_collision_rate_HtA, mid_collision_rate_HtA);
    printf("[Loading Factor HtA]: [%.6f, %.6f] with Avg: %.6f Mid: %.6f\n", min_loading_factor_HtA, max_loading_factor_HtA, avg_loading_factor_HtA, mid_loading_factor_HtA);
    printf("[All collisions HtA]: %lu\n", sum_num_collison_HtA);
    printf("[Max collisions HtA]: %lu\n", max_num_collison_all_HtA);

    for(int i = 0; i < fidx_X.len - 1; i++){
        total_bytes_HtA += bytes_HtA[i];
        max_bytes_HtA = bytes_HtA[i] > max_bytes_HtA ? bytes_HtA[i] : max_bytes_HtA;
    }
    // Sum of all HtA: only if each HtA corresponds to a thread, considering omp shared memory model
    // printf("[Memory cost of HtA]: %lu Bytes\n", total_bytes_HtA);
    printf("[Memory cost of HtA]: %10.2f KiB %10.2f MiB %10.2f GiB\n", total_bytes_HtA / 1024.0, total_bytes_HtA / 1048576.0, total_bytes_HtA / 1073741824.0); 
    // Min cost: sequence algorithm,sequence malloc & free. 
    // printf("[Max memcost of HtA]: %lu Bytes\n", max_bytes_HtA);
    printf("[Max memcost of HtA]: %10.2f KiB %10.2f MiB %10.2f GiB\n", max_bytes_HtA / 1024.0, max_bytes_HtA / 1048576.0, max_bytes_HtA / 1073741824.0);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 
    Z->stidx = (int*)malloc(sizeof(int)*Z->nnz);

    #pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
    for(int i = 0; i < tk; i++){
        int tid = omp_get_thread_num();
        if(Z_tmp[tid].nnz > 0){
            for(sptIndex m = 0; m < nmodes_Z; ++m) 
                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);        
            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);  
            //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
        }
    } 

    //  for(int i = 0; i < tk; i++)
    //      sptFreeSparseTensor(&Z_tmp[i]);
    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Writeback");

    sptStartTimer(timer);
    if(output_sorting == 1){
        sptSparseTensorSortIndex(Z, 1, tk);
    }
    sptStopTimer(timer);
    total_time += sptPrintElapsedTime(timer, "Output Sorting");
    printf("[Total time]: %.6f s\n", total_time);
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode3.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  }  

// 4: Bullseye Hash
  if(experiment_modes == 4){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    sptTimer timer_writehty;
    sptNewTimer(&timer_writehty, 0);
    double time_writehty = 0;

    sptTimer timer_writeht;
    sptNewTimer(&timer_writeht, 0);
    double time_writeht = 0;

    sptTimer timer_inputsort;
    sptNewTimer(&timer_inputsort, 0);
    double time_inputsort = 0;

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);

    sptSparseTensorSortIndex(X, 1, tk);


    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    // printf("[Num of subtensorX]:%lu\n", fidx_X.len - 1);
    LP_tensor_table_t *X_ht;
    unsigned int X_ht_size = ceil(log2(2*(fidx_X.len-1)));
    // printf("X_ht_size: %u\n", X_ht_size);
    X_ht = LP_tensor_htCreate(X_ht_size);

    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }
    sptNnzIndex X_nnz = X->nnz;

    unsigned long long* fkey_Y = (unsigned long long*)malloc((fidx_X.len-1)*sizeof(unsigned long long));

    // omp_lock_t *locks_x = (omp_lock_t *)malloc(X_ht->size*sizeof(omp_lock_t));
    // for(size_t i = 0; i < X_ht->size; i++) omp_init_lock(&locks_x[i]);
    #pragma omp parallel for schedule(static) num_threads(tk) shared(X_ht, X_num_fmodes, mode_order_X, num_cmodes, X_cmode_inds, X_fmode_inds)
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len-1; fx_ptr++){
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptNnzIndex nnzs_subtensor = fx_end - fx_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_cmodes += X->inds[mode_order_X[m]].data[fx_begin] * X_fmode_inds[m + 1]; 
        fkey_Y[fx_ptr] = key_cmodes; /*record the nnz ckey_x/fkey_y*/
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, X_ht->shift);
        // omp_set_lock(&locks_x[pos]);
        int size = X_ht->size;
        while(size--){
            // int prev_stat;
            // #pragma omp atomic capture
            // {
            //     prev_stat = X_ht->LP_node_list[pos].stat;
            //     if(prev_stat == 0) X_ht->LP_node_list[pos].stat = 1;
            // }
            int prev_stat;
            #pragma omp atomic capture
            {
                prev_stat = X_ht->LP_node_list[pos].stat;
                X_ht->LP_node_list[pos].stat = 1;
            }
            if(prev_stat == 0){
                // X_ht->LP_node_list[pos].stat = 1;
                X_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(X_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zX = fx_begin; zX < fx_end; zX++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < num_cmodes; ++m)
                        key_fmodes += X->inds[mode_order_X[m+X_num_fmodes]].data[zX] * X_cmode_inds[m + 1];
                    tmp->key_FM[zX-fx_begin] = key_fmodes;
                    tmp->val[zX-fx_begin] = X->values.data[zX];
                }
                X_ht->len++;
                break;
            }
            else{
                // omp_unset_lock(&locks_x[pos]);
                // printf("Collisions occur\n");
                pos++;
                if(pos >= X_ht->size) pos = 0;
                // omp_set_lock(&locks_x[pos]);
            }
        }
        // omp_unset_lock(&locks_x[pos]);
    }

    // sort(fkey_Y, fkey_Y + fidx_X.len-1);
    // for(size_t i = 0; i < X_ht->size; i++) omp_destroy_lock(&locks_x[i]);
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    sptNnzIndexVector fidx_Y;
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order


    sptSparseTensorSortIndex(Y, 1, tk);
    
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);

    // printf("[Num of subtensorY]:%lu\n", fidx_Y.len - 1);
    LP_tensor_table_t *Y_ht;
    unsigned int Y_ht_size = ceil(log2(2*(fidx_Y.len-1)));
    // printf("Y_ht_size: %u\n", Y_ht_size);
    Y_ht = LP_tensor_htCreate(Y_ht_size);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;

    // omp_lock_t *locks_y = (omp_lock_t *)malloc(Y_ht->size*sizeof(omp_lock_t));
    // for(size_t i = 0; i < Y_ht->size; i++) omp_init_lock(&locks_y[i]);
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex fy_ptr = 0; fy_ptr < fidx_Y.len-1; fy_ptr++){
        sptNnzIndex fy_begin = fidx_Y.data[fy_ptr];
        sptNnzIndex fy_end = fidx_Y.data[fy_ptr+1];
        sptNnzIndex nnzs_subtensor = fy_end - fy_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[fy_begin] * Y_cmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, Y_ht->shift);
        // omp_set_lock(&locks_y[pos]);
        int size = Y_ht->size;
        while(size--){
            // int prev_stat;
            // #pragma omp atomic capture
            // {
            //     prev_stat = Y_ht->LP_node_list[pos].stat;
            //     if(prev_stat == 0) Y_ht->LP_node_list[pos].stat = 1;
            // }
            int prev_stat;
            #pragma omp atomic capture
            {
                prev_stat = Y_ht->LP_node_list[pos].stat;
                Y_ht->LP_node_list[pos].stat = 1;
            }
            if(prev_stat == 0){
                // Y_ht->LP_node_list[pos].stat = 1;
                Y_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(Y_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                tmp->key_FM_idx = (unsigned int*)malloc(nnzs_subtensor*sizeof(unsigned int)); /*record the idx of each key_FM*/
                for(sptNnzIndex zY = fy_begin; zY < fy_end; zY++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < Y_num_fmodes; ++m)
                        key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[zY] * Y_fmode_inds[m + 1];
                    tmp->key_FM[zY-fy_begin] = key_fmodes;
                    tmp->val[zY-fy_begin] = Y->values.data[zY];

                    tmp->key_FM_idx[zY-fy_begin] = lower_bound(fkey_Y, fkey_Y + fidx_X.len-1,key_fmodes) - fkey_Y;

                    // cout << "Key:" << tmp->key_FM[zY-fy_begin] << "Key_idx:" << tmp->key_FM_idx[zY-fy_begin] << endl;
                    // cout << tmp->key_FM_idx[zY-fy_begin] << endl;
                }
                Y_ht->len++;
                break;
            }
            pos++;
            if(pos >= Y_ht->size) pos = 0;
        }
        // omp_unset_lock(&locks_y[pos]);
    }
    // for(size_t i = 0; i < Y_ht->size; i++) omp_destroy_lock(&locks_y[i]);
    free(fkey_Y); /*release*/

    LP_tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = LP_tensor_htCreate(Z_ht_size);
    for(sptNnzIndex i = 0; i < X_ht->size; i++){
        if(X_ht->LP_node_list[i].stat == 1){
            Z_ht->LP_node_list[i].stat = 1;
            Z_ht->LP_node_list[i].key = X_ht->LP_node_list[i].key;
        }
    }

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    // sptFreeSparseTensor(Y);
    sptNnzIndex bytes_HtX = 0;
    sptNnzIndex bytes_HtY = 0;
    // bytes_HtX = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + X_ht->size*sizeof(LP_tensor_node_t) + (X->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    // bytes_HtY = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Y_ht->size*sizeof(LP_tensor_node_t) + (Y->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    // printf("[Memorycost of HtX]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtX / 1024.0, bytes_HtX / 1048576.0, bytes_HtX/ 1073741824.0);
    // printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);
    // printf("[Input Write_hash]: %.6f s\n", time_writeht);
    // printf("[Input Sorting   ]: %.6f s\n", time_inputsort);
    // printf("[Input ProcessingY]: %.6f s\n", time_writehty);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    // unsigned int hashtable_a_size = 0;
    // unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // for(sptIndex i = 0; i < tk ; i++){
    //     min_hta_size[i] = 64;
    //     max_hta_size[i] = 0;
    // }    
    sptNnzIndex max_nnz_range = 0;
    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht->size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        if(X_ht->LP_node_list[fx_ptr].stat == 1){
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = X_ht->LP_node_list[fx_ptr].val_ptr;
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            unsigned int min_key_idx = UINT_MAX, max_key_idx = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
                if(Y_val.len > 0){
                    // cout << Y_len << ":" << Y_val.key_FM[0] << "~" << Y_val.key_FM[Y_len-1] << endl;
                    min_key_idx = std::min(min_key_idx, Y_val.key_FM_idx[0]);
                    max_key_idx = std::max(max_key_idx, Y_val.key_FM_idx[Y_len-1]);
                }
            } 
            if(nnz_upper_bound ==0 ) continue;
            sptNnzIndex tot_nnz_range = max_key_idx - min_key_idx + 1;

            max_nnz_range = max(tot_nnz_range, max_nnz_range);
                       
            unsigned int* pos_nnz = (unsigned int *)malloc((tot_nnz_range)*sizeof(unsigned int)); /*record pos*/
            unsigned long long* fkey_array = (unsigned long long*)calloc(tot_nnz_range, sizeof(unsigned long long));
            sptValue* spa_val_array = (sptValue*)calloc(tot_nnz_range, sizeof(sptValue));
            
            // cout << "tot_nnz_range" << tot_nnz_range << endl;
            
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X  
                if (tid == 0) {
                    sptStartTimer(timer_SPA);
                }       
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX]; 

                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);  

                unsigned int my_len = Y_val.len;
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                if(my_len == 0) continue;

                if (tid == 0) {
                    sptStartTimer(timer_SPA);       
                }        
                
                // memory access error in this part
                for(int i = 0; i < my_len; i++){
                    unsigned long long fmode =  Y_val.key_FM[i];
                    unsigned int fmode_idx = Y_val.key_FM_idx[i] - min_key_idx;
                    
                    float result = Y_val.val[i] * valX;
                    // if (abs(spa_val_array[fmode_idx]-0.0) < numeric_limits<double>::epsilon()){
                    if (abs(spa_val_array[fmode_idx]-0.0) < 4.94e-324){
                        fkey_array[fmode_idx] = fmode;
                        spa_val_array[fmode_idx] = result;
                        pos_nnz[nnz_counter] = fmode_idx;
                        nnz_counter++;
                    }else{
                        spa_val_array[fmode_idx] += result;
                    }
                }

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                }
                
            }   // End Loop nnzs inside a X fiber

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.len = nnz_counter;
            Z_ht->LP_node_list[fx_ptr].val_ptr.cap = nnz_counter;

            for(int z = 0; z < nnz_counter; z++){
                unsigned int pos = pos_nnz[z];
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = fkey_array[pos];
                Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = spa_val_array[pos];
            }
            // printf("bytes_HtA:%lu", bytes_HtA);
            Z_tmp[tid].nnz += nnz_counter;

            free(fkey_array);
            free(spa_val_array);
            free(pos_nnz);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        }
    } // End loop of Buckts of HtX
    // unsigned int Max_hta_size = 0, Min_hta_size = 64;
    // for(sptIndex i = 0; i < tk; i++){
    //     Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
    //     Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    // }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    // printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    // printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    // printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    // printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    // printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    // printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    // printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);
    printf("[Max nnz range]:%d\n", max_nnz_range);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 
    Z_ht->len = Z_total_size;
    sptNnzIndex bytes_HtZ;
    bytes_HtZ = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Z_ht->size*sizeof(LP_tensor_node_t) + (Z_total_size)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtZ]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtZ / 1024.0, bytes_HtZ / 1048576.0, bytes_HtZ / 1073741824.0);
    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 



// 21: All Hash table formats: use sparta chaining methods
  if(experiment_modes == 11){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
 
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    tensor_table_t *X_ht;
    unsigned int X_ht_size = X->nnz;
    X_ht = tensor_htCreate(X_ht_size);    
    
    omp_lock_t *locks_x = (omp_lock_t *)malloc(X_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < X_ht_size; i++) omp_init_lock(&locks_x[i]);
    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }


    sptNnzIndex X_nnz = X->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(X_ht, X_num_fmodes, mode_order_X, num_cmodes, X_cmode_inds, X_fmode_inds)
    for(sptNnzIndex i = 0; i < X_nnz; i++){
        int tid = omp_get_thread_num();

        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += X->inds[mode_order_X[X_num_fmodes+m]].data[i] * X_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_fmodes += X->inds[mode_order_X[m]].data[i] * X_fmode_inds[m + 1];
        

        unsigned pos = tensor_htHashCode_size(key_fmodes, X_nnz);
        omp_set_lock(&locks_x[pos]);    
        tensor_value X_val = tensor_htGet(X_ht, key_fmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(X_val.len == 0) {
            tensor_htInsert(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
        }
        else{
            tensor_htUpdate(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks_x[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < X_ht_size; i++) omp_destroy_lock(&locks_x[i]);


    tensor_table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = (tensor_table_t*)malloc(sizeof(tensor_table_t));
    Z_ht->size = X_ht_size;
    Z_ht->list = (tensor_node_t**)malloc(sizeof(tensor_node_t*)*Z_ht->size);
    for(sptNnzIndex i = 0; i < X_ht_size; i++){
        Z_ht->list[i] = NULL;
        tensor_node_t* temp = X_ht->list[i];
        while(temp){
            tensor_node_t *newNode = (tensor_node_t*) malloc(sizeof(tensor_node_t));
            newNode->key = temp->key;
            newNode->next = Z_ht->list[i];
            Z_ht->list[i] = newNode;
            temp = temp->next;
        }
    }
    sptTimer timer_hty;
    double time_hty_calckey = 0;
    sptNewTimer(&timer_hty, 0);

    sptNnzIndex Y_nnz = Y->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        //if (Y->values.data[i] <0.00000001) continue;
        int tid = omp_get_thread_num();
        if(tid == 0){
            sptStartTimer(timer_hty);
        }
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        
        if(tid == 0){
            sptStopTimer(timer_hty);
            time_hty_calckey += sptElapsedTime(timer_hty);
        }

        unsigned pos = tensor_htHashCode_size(key_cmodes, Y_nnz);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);
    // sptFreeSparseTensor(Y);
    sptNnzIndex empty_buckets = 0;
    sptIndex max_bucket_length = 0;
    sptNnzIndex num_collisions = 0;
    sptNnzIndex num_elements = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtY += sizeof(int) + Y_ht_size * sizeof(tensor_node_t*);
    for(sptNnzIndex i = 0; i < Y_ht_size; i++){
        tensor_node_t *list = Y_ht->list[i];
        tensor_node_t *temp = list;
        sptIndex count = -1;
        if(temp){
            while(temp){
                // bytes_HtY += sizeof(unsigned long long) + sizeof(tensor_value) + sizeof(tensor_node_t*);
                bytes_HtY += sizeof(tensor_node_t);
                count++;
                temp = temp->next;
            }
            num_collisions += count;
            num_elements += (count + 1);
            max_bucket_length = max_bucket_length >= count ? max_bucket_length : count;
        }else{
            empty_buckets++;
        }
    }
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Whole hashbucket]: %lu \n", Y_ht_size);
    printf("[Empty hashbucket]: %ld \n", empty_buckets);
    printf("[Wasted hash rate]: %%%.7lf\n", 100.0*(1.0*empty_buckets)/(1.0*Y_ht_size));
    printf("[Hash  collisions]: %ld \n", num_collisions);
    printf("[Max   collision]: %ld \n", max_bucket_length);
    printf("[Hash  elements]: %ld \n", num_elements);
    printf("[Loading Factor]: %.7lf \n", num_elements*1.0/Y_nnz);
    sptNnzIndex nnz_bucket = Y_ht_size - empty_buckets;
    printf("[Avg   collision]: %.7lf \n", 1.0*num_collisions/(1.0*nnz_bucket));
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Calc htKey]: %.6f s\n", time_hty_calckey);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht_size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        tensor_node_t* temp = X_ht->list[fx_ptr];
        tensor_node_t* node_Z = Z_ht->list[fx_ptr];
        while(temp){ // If collisions occur
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = temp->val;    
            unsigned int X_len = X_val.len;
            
            unsigned int ht_size = 16;
            hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;
            CK_table_t *CK_ht;
            CK_ht = CK_htCreate(ht_size);

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];

                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                unsigned int Y_len = Y_val.len;
                if(Y_len == 0) continue;
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                for(unsigned int zY = 0; zY < Y_len; zY++){
                    unsigned long long fmode = Y_val.key_FM[zY];
                    unsigned long long pos = CK_htGet(CK_ht, fmode);
                    float result = Y_val.val[zY]*valX;

                    if(pos == ULLONG_MAX){
                        pos_nnz[CK_ht->len] = CK_htInsert(CK_ht, 0, fmode, result);
                        nnz_counter++;
                    }else{
                        CK_htUpdate(CK_ht, pos, fmode, result);
                    }
                } 
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                } 
            } // End Loop nnzs inside a X subtensor

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            node_Z->val.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            node_Z->val.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            node_Z->val.len = nnz_counter;
            for(unsigned int z = 0; z < CK_ht->len; z++){
                unsigned int pos = pos_nnz[z];
                unsigned int tableidx = (pos & 0x8000000) >> 31;
                unsigned int real_pos = pos & 0x7fffffff;
                node_Z->val.key_FM[z] = CK_ht->key[tableidx][real_pos];
                node_Z->val.val[z] = CK_ht->val[tableidx][real_pos];
            }

            Z_tmp[tid].nnz += nnz_counter;
            free(pos_nnz);
            CK_htFree(CK_ht);
            // tensor_node_t* pre = temp;
            temp = temp->next;
            // free(pre);
            node_Z = node_Z->next;

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        } // end loop of subtensors X
    } // End loop of Buckts of HtX

    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 

// 23: All Hash table formats, upper bound to predict the nnzs of C, hta LP
  if(experiment_modes == 12){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
 
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    tensor_table_t *X_ht;
    unsigned int X_ht_size = X->nnz;
    X_ht = tensor_htCreate(X_ht_size);    
    
    omp_lock_t *locks_x = (omp_lock_t *)malloc(X_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < X_ht_size; i++) omp_init_lock(&locks_x[i]);
    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }


    sptNnzIndex X_nnz = X->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(X_ht, X_num_fmodes, mode_order_X, num_cmodes, X_cmode_inds, X_fmode_inds)
    for(sptNnzIndex i = 0; i < X_nnz; i++){
        int tid = omp_get_thread_num();

        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += X->inds[mode_order_X[X_num_fmodes+m]].data[i] * X_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_fmodes += X->inds[mode_order_X[m]].data[i] * X_fmode_inds[m + 1];
        

        unsigned pos = tensor_htHashCode_size(key_fmodes, X_nnz);
        omp_set_lock(&locks_x[pos]);    
        tensor_value X_val = tensor_htGet(X_ht, key_fmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(X_val.len == 0) {
            tensor_htInsert(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
        }
        else{
            tensor_htUpdate(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks_x[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < X_ht_size; i++) omp_destroy_lock(&locks_x[i]);


    tensor_table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = (tensor_table_t*)malloc(sizeof(tensor_table_t));
    Z_ht->size = X_ht_size;
    Z_ht->list = (tensor_node_t**)malloc(sizeof(tensor_node_t*)*Z_ht->size);
    for(sptNnzIndex i = 0; i < X_ht_size; i++){
        Z_ht->list[i] = NULL;
        tensor_node_t* temp = X_ht->list[i];
        while(temp){
            tensor_node_t *newNode = (tensor_node_t*) malloc(sizeof(tensor_node_t));
            newNode->key = temp->key;
            newNode->next = Z_ht->list[i];
            Z_ht->list[i] = newNode;
            temp = temp->next;
        }
    }
    sptTimer timer_hty;
    double time_hty_calckey = 0;
    sptNewTimer(&timer_hty, 0);

    sptNnzIndex Y_nnz = Y->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        //if (Y->values.data[i] <0.00000001) continue;
        int tid = omp_get_thread_num();
        if(tid == 0){
            sptStartTimer(timer_hty);
        }
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        
        if(tid == 0){
            sptStopTimer(timer_hty);
            time_hty_calckey += sptElapsedTime(timer_hty);
        }

        unsigned pos = tensor_htHashCode_size(key_cmodes, Y_nnz);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);
    // sptFreeSparseTensor(Y);
    sptNnzIndex empty_buckets = 0;
    sptIndex max_bucket_length = 0;
    sptNnzIndex num_collisions = 0;
    sptNnzIndex num_elements = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtY += sizeof(int) + Y_ht_size * sizeof(tensor_node_t*);
    for(sptNnzIndex i = 0; i < Y_ht_size; i++){
        tensor_node_t *list = Y_ht->list[i];
        tensor_node_t *temp = list;
        sptIndex count = -1;
        if(temp){
            while(temp){
                // bytes_HtY += sizeof(unsigned long long) + sizeof(tensor_value) + sizeof(tensor_node_t*);
                bytes_HtY += sizeof(tensor_node_t);
                count++;
                temp = temp->next;
            }
            num_collisions += count;
            num_elements += (count + 1);
            max_bucket_length = max_bucket_length >= count ? max_bucket_length : count;
        }else{
            empty_buckets++;
        }
    }
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Whole hashbucket]: %lu \n", Y_ht_size);
    printf("[Empty hashbucket]: %ld \n", empty_buckets);
    printf("[Wasted hash rate]: %%%.7lf\n", 100.0*(1.0*empty_buckets)/(1.0*Y_ht_size));
    printf("[Hash  collisions]: %ld \n", num_collisions);
    printf("[Max   collision]: %ld \n", max_bucket_length);
    printf("[Hash  elements]: %ld \n", num_elements);
    printf("[Loading Factor]: %.7lf \n", num_elements*1.0/Y_nnz);
    sptNnzIndex nnz_bucket = Y_ht_size - empty_buckets;
    printf("[Avg   collision]: %.7lf \n", 1.0*num_collisions/(1.0*nnz_bucket));
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Calc htKey]: %.6f s\n", time_hty_calckey);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    for(sptIndex i = 0; i < tk ; i++){
        min_hta_size[i] = 64;
        max_hta_size[i] = 0;
    }

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht_size; fx_ptr++){// Loop all buckets to loop the subtensor X
        // printf("%llu\n", fx_ptr);
        int tid = omp_get_thread_num();
        tensor_node_t* temp = X_ht->list[fx_ptr];
        tensor_node_t* node_Z = Z_ht->list[fx_ptr];
        while(temp){ // If collisions occur
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = temp->val;    
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            // unsigned int ht_size = std::min(std::max((int)ceil(log2(nnz_upper_bound)),2), 10);
            unsigned int ht_size = std::max((int)ceil(log2(4*nnz_upper_bound)),2);
            // while(pow(2, ht_size) < 32*nnz_upper_bound && ht_size < 20) ht_size++;
            // while(pow(2, ht_size) < 2*nnz_upper_bound) ht_size++;
            // while(pow(2, ht_size) < 32*nnz_upper_bound && ht_size < 16) ht_size++;
            min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
            max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);
            // printf("ht_size: %u\n", ht_size);
            // unsigned int ht_size = 16;
            unsigned int hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;
            LP_table_t *LP_ht;
            LP_ht = LP_htCreate(ht_size);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];

                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                unsigned int Y_len = Y_val.len;
                if(Y_len == 0) continue;
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                for(unsigned int zY = 0; zY < Y_len; zY++){
                    unsigned long long fmode = Y_val.key_FM[zY];
                    sptValue spa_val = LP_htGet(LP_ht, fmode);
                    float result = Y_val.val[zY]*valX;

                    if(spa_val == LONG_MIN){
                        pos_nnz[LP_ht->len] = LP_htInsert(LP_ht, fmode, result);
                        nnz_counter++;
                    }else{
                        LP_htUpdate(LP_ht, fmode, spa_val + result);
                    }
                } 
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                } 
            } // End Loop nnzs inside a X subtensor

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            node_Z->val.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            node_Z->val.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            node_Z->val.len = nnz_counter;
            for(unsigned int z = 0; z < LP_ht->len; z++){
                unsigned int i = pos_nnz[z];
                node_Z->val.key_FM[z] = LP_ht->key[i];
                node_Z->val.val[z] = LP_ht->val[i];
            }

            Z_tmp[tid].nnz += nnz_counter;
            free(pos_nnz);
            LP_htFree(LP_ht);
            // tensor_node_t* pre = temp;
            temp = temp->next;
            // free(pre);
            node_Z = node_Z->next;

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        } // end loop of subtensors X
    } // End loop of Buckts of HtX
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    unsigned int Max_hta_size = 0, Min_hta_size = 64;
    for(sptIndex i = 0; i < tk; i++){
        Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
        Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    }
    printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 

// 24: All Hash table formats, upper bound to predict the nnzs of C, hta CK
  if(experiment_modes == 13){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
 
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    tensor_table_t *X_ht;
    unsigned int X_ht_size = X->nnz;
    X_ht = tensor_htCreate(X_ht_size);    
    
    omp_lock_t *locks_x = (omp_lock_t *)malloc(X_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < X_ht_size; i++) omp_init_lock(&locks_x[i]);
    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }


    sptNnzIndex X_nnz = X->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(X_ht, X_num_fmodes, mode_order_X, num_cmodes, X_cmode_inds, X_fmode_inds)
    for(sptNnzIndex i = 0; i < X_nnz; i++){
        int tid = omp_get_thread_num();

        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += X->inds[mode_order_X[X_num_fmodes+m]].data[i] * X_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_fmodes += X->inds[mode_order_X[m]].data[i] * X_fmode_inds[m + 1];
        

        unsigned pos = tensor_htHashCode_size(key_fmodes, X_nnz);
        omp_set_lock(&locks_x[pos]);    
        tensor_value X_val = tensor_htGet(X_ht, key_fmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(X_val.len == 0) {
            tensor_htInsert(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
        }
        else{
            tensor_htUpdate(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks_x[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < X_ht_size; i++) omp_destroy_lock(&locks_x[i]);


    tensor_table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = (tensor_table_t*)malloc(sizeof(tensor_table_t));
    Z_ht->size = X_ht_size;
    Z_ht->list = (tensor_node_t**)malloc(sizeof(tensor_node_t*)*Z_ht->size);
    for(sptNnzIndex i = 0; i < X_ht_size; i++){
        Z_ht->list[i] = NULL;
        tensor_node_t* temp = X_ht->list[i];
        while(temp){
            tensor_node_t *newNode = (tensor_node_t*) malloc(sizeof(tensor_node_t));
            newNode->key = temp->key;
            newNode->next = Z_ht->list[i];
            Z_ht->list[i] = newNode;
            temp = temp->next;
        }
    }
    sptTimer timer_hty;
    double time_hty_calckey = 0;
    sptNewTimer(&timer_hty, 0);

    sptNnzIndex Y_nnz = Y->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        //if (Y->values.data[i] <0.00000001) continue;
        int tid = omp_get_thread_num();
        if(tid == 0){
            sptStartTimer(timer_hty);
        }
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        
        if(tid == 0){
            sptStopTimer(timer_hty);
            time_hty_calckey += sptElapsedTime(timer_hty);
        }

        unsigned pos = tensor_htHashCode_size(key_cmodes, Y_nnz);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);
    // sptFreeSparseTensor(Y);
    sptNnzIndex empty_buckets = 0;
    sptIndex max_bucket_length = 0;
    sptNnzIndex num_collisions = 0;
    sptNnzIndex num_elements = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtY += sizeof(int) + Y_ht_size * sizeof(tensor_node_t*);
    for(sptNnzIndex i = 0; i < Y_ht_size; i++){
        tensor_node_t *list = Y_ht->list[i];
        tensor_node_t *temp = list;
        sptIndex count = -1;
        if(temp){
            while(temp){
                // bytes_HtY += sizeof(unsigned long long) + sizeof(tensor_value) + sizeof(tensor_node_t*);
                bytes_HtY += sizeof(tensor_node_t);
                count++;
                temp = temp->next;
            }
            num_collisions += count;
            num_elements += (count + 1);
            max_bucket_length = max_bucket_length >= count ? max_bucket_length : count;
        }else{
            empty_buckets++;
        }
    }
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Whole hashbucket]: %lu \n", Y_ht_size);
    printf("[Empty hashbucket]: %ld \n", empty_buckets);
    printf("[Wasted hash rate]: %%%.7lf\n", 100.0*(1.0*empty_buckets)/(1.0*Y_ht_size));
    printf("[Hash  collisions]: %ld \n", num_collisions);
    printf("[Max   collision]: %ld \n", max_bucket_length);
    printf("[Hash  elements]: %ld \n", num_elements);
    printf("[Loading Factor]: %.7lf \n", num_elements*1.0/Y_nnz);
    sptNnzIndex nnz_bucket = Y_ht_size - empty_buckets;
    printf("[Avg   collision]: %.7lf \n", 1.0*num_collisions/(1.0*nnz_bucket));
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Calc htKey]: %.6f s\n", time_hty_calckey);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    for(sptIndex i = 0; i < tk ; i++){
        min_hta_size[i] = 64;
        max_hta_size[i] = 0;
    }    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht_size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        tensor_node_t* temp = X_ht->list[fx_ptr];
        tensor_node_t* node_Z = Z_ht->list[fx_ptr];
        while(temp){ // If collisions occur
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = temp->val;    
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            // unsigned int ht_size = std::min(std::max((int)ceil(log2(16*nnz_upper_bound)),2), 25);
            unsigned int ht_size = std::max((int)ceil(log2(16*nnz_upper_bound)),2);
            // while(pow(2, ht_size) < 32*nnz_upper_bound && ht_size < 20) ht_size++;
            // while(pow(2, ht_size) < 32*nnz_upper_bound && ht_size < 16) ht_size++;
            min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
            max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);
            // printf("ht_size: %u\n", ht_size);
            // unsigned int ht_size = 16;
            unsigned int hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;
            CK_table_t *CK_ht;
            CK_ht = CK_htCreate(ht_size);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];

                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                unsigned int Y_len = Y_val.len;
                if(Y_len == 0) continue;
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                for(unsigned int zY = 0; zY < Y_len; zY++){
                    unsigned long long fmode = Y_val.key_FM[zY];
                    unsigned long long pos = CK_htGet(CK_ht, fmode);
                    float result = Y_val.val[zY]*valX;

                    if(pos == ULLONG_MAX){
                        pos_nnz[CK_ht->len] = CK_htInsert(CK_ht, 0, fmode, result);
                        nnz_counter++;
                    }else{
                        CK_htUpdate(CK_ht, pos, fmode, result);
                    }
                } 
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                } 
            } // End Loop nnzs inside a X subtensor

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            node_Z->val.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            node_Z->val.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            node_Z->val.len = nnz_counter;
            for(unsigned int z = 0; z < CK_ht->len; z++){
                unsigned int pos = pos_nnz[z];
                unsigned int tableidx = (pos & 0x8000000) >> 31;
                unsigned int real_pos = pos & 0x7fffffff;
                node_Z->val.key_FM[z] = CK_ht->key[tableidx][real_pos];
                node_Z->val.val[z] = CK_ht->val[tableidx][real_pos];
            }

            Z_tmp[tid].nnz += nnz_counter;
            free(pos_nnz);
            CK_htFree(CK_ht);
            // tensor_node_t* pre = temp;
            temp = temp->next;
            // free(pre);
            node_Z = node_Z->next;

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        } // end loop of subtensors X
    } // End loop of Buckts of HtX
    unsigned int Max_hta_size = 0, Min_hta_size = 64;
    for(sptIndex i = 0; i < tk; i++){
        Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
        Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 

// 25: All Hash table formats, upper bound to predict the nnzs of C, hta Chaining
  if(experiment_modes == 14){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
 
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    tensor_table_t *X_ht;
    unsigned int X_ht_size = X->nnz;
    X_ht = tensor_htCreate(X_ht_size);    
    
    omp_lock_t *locks_x = (omp_lock_t *)malloc(X_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < X_ht_size; i++) omp_init_lock(&locks_x[i]);
    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }


    sptNnzIndex X_nnz = X->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(X_ht, X_num_fmodes, mode_order_X, num_cmodes, X_cmode_inds, X_fmode_inds)
    for(sptNnzIndex i = 0; i < X_nnz; i++){
        int tid = omp_get_thread_num();

        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += X->inds[mode_order_X[X_num_fmodes+m]].data[i] * X_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_fmodes += X->inds[mode_order_X[m]].data[i] * X_fmode_inds[m + 1];
        

        unsigned pos = tensor_htHashCode_size(key_fmodes, X_nnz);
        omp_set_lock(&locks_x[pos]);    
        tensor_value X_val = tensor_htGet(X_ht, key_fmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(X_val.len == 0) {
            tensor_htInsert(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
        }
        else{
            tensor_htUpdate(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks_x[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < X_ht_size; i++) omp_destroy_lock(&locks_x[i]);


    tensor_table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = (tensor_table_t*)malloc(sizeof(tensor_table_t));
    Z_ht->size = X_ht_size;
    Z_ht->list = (tensor_node_t**)malloc(sizeof(tensor_node_t*)*Z_ht->size);
    for(sptNnzIndex i = 0; i < X_ht_size; i++){
        Z_ht->list[i] = NULL;
        tensor_node_t* temp = X_ht->list[i];
        while(temp){
            tensor_node_t *newNode = (tensor_node_t*) malloc(sizeof(tensor_node_t));
            newNode->key = temp->key;
            newNode->next = Z_ht->list[i];
            Z_ht->list[i] = newNode;
            temp = temp->next;
        }
    }
    sptTimer timer_hty;
    double time_hty_calckey = 0;
    sptNewTimer(&timer_hty, 0);

    sptNnzIndex Y_nnz = Y->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        //if (Y->values.data[i] <0.00000001) continue;
        int tid = omp_get_thread_num();
        if(tid == 0){
            sptStartTimer(timer_hty);
        }
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        
        if(tid == 0){
            sptStopTimer(timer_hty);
            time_hty_calckey += sptElapsedTime(timer_hty);
        }

        unsigned pos = tensor_htHashCode_size(key_cmodes, Y_nnz);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);
    // sptFreeSparseTensor(Y);
    sptNnzIndex empty_buckets = 0;
    sptIndex max_bucket_length = 0;
    sptNnzIndex num_collisions = 0;
    sptNnzIndex num_elements = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtY += sizeof(int) + Y_ht_size * sizeof(tensor_node_t*);
    for(sptNnzIndex i = 0; i < Y_ht_size; i++){
        tensor_node_t *list = Y_ht->list[i];
        tensor_node_t *temp = list;
        sptIndex count = -1;
        if(temp){
            while(temp){
                // bytes_HtY += sizeof(unsigned long long) + sizeof(tensor_value) + sizeof(tensor_node_t*);
                bytes_HtY += sizeof(tensor_node_t);
                count++;
                temp = temp->next;
            }
            num_collisions += count;
            num_elements += (count + 1);
            max_bucket_length = max_bucket_length >= count ? max_bucket_length : count;
        }else{
            empty_buckets++;
        }
    }
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Whole hashbucket]: %lu \n", Y_ht_size);
    printf("[Empty hashbucket]: %ld \n", empty_buckets);
    printf("[Wasted hash rate]: %%%.7lf\n", 100.0*(1.0*empty_buckets)/(1.0*Y_ht_size));
    printf("[Hash  collisions]: %ld \n", num_collisions);
    printf("[Max   collision]: %ld \n", max_bucket_length);
    printf("[Hash  elements]: %ld \n", num_elements);
    printf("[Loading Factor]: %.7lf \n", num_elements*1.0/Y_nnz);
    sptNnzIndex nnz_bucket = Y_ht_size - empty_buckets;
    printf("[Avg   collision]: %.7lf \n", 1.0*num_collisions/(1.0*nnz_bucket));
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Calc htKey]: %.6f s\n", time_hty_calckey);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    for(sptIndex i = 0; i < tk ; i++){
        min_hta_size[i] = 64;
        max_hta_size[i] = 0;
    }    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht_size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        tensor_node_t* temp = X_ht->list[fx_ptr];
        tensor_node_t* node_Z = Z_ht->list[fx_ptr];
        while(temp){ // If collisions occur
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = temp->val;    
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            unsigned int ht_size = std::min(std::max((int)ceil(log2(2*nnz_upper_bound)),2), 30);
            // while(pow(2, ht_size) < 32*nnz_upper_bound && ht_size < 20) ht_size++;
            // while(pow(2, ht_size) < 32*nnz_upper_bound && ht_size < 16) ht_size++;
            min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
            max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);
            // printf("ht_size: %u\n", ht_size);
            // unsigned int ht_size = 16;
            unsigned int hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            unsigned int bucked_counter = 0;
            sptIndex current_idx = 0;
            table_t *ht;
            ht = htCreate(ht_size);

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];

                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                unsigned int Y_len = Y_val.len;
                if(Y_len == 0) continue;
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                for(unsigned int zY = 0; zY < Y_len; zY++){
                    unsigned long long fmode = Y_val.key_FM[zY];
                    sptValue spa_val = htGet(ht, fmode);
                    float result = Y_val.val[zY]*valX;

                    if(spa_val == LONG_MIN) {
                        int check_pos = htInsert(ht, fmode, result);
                        nnz_counter++;
                        if(check_pos != UINT_MAX){
                            pos_nnz[bucked_counter] = check_pos;
                            bucked_counter++;
                        }
                    }
                    else    
                        htUpdate(ht, fmode, spa_val + result);
                } 
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                } 
            } // End Loop nnzs inside a X subtensor

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            node_Z->val.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            node_Z->val.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            node_Z->val.len = nnz_counter;
            int nnz_idx = 0;
            // for(unsigned int i = 0; i < ht->size; i++){
            for(unsigned int i = 0; i < bucked_counter; i++){
                node_t *temp = ht->list[pos_nnz[i]];
                while(temp){
                    node_Z->val.key_FM[nnz_idx] = temp->key;
                    node_Z->val.val[nnz_idx] = temp->val;
                    node_t* pre = temp;
                    temp = temp->next;
                    nnz_idx++;
                    // free(pre);
                }
            }

            Z_tmp[tid].nnz += nnz_counter;
            free(pos_nnz);
            htFree(ht);
            // tensor_node_t* pre = temp;
            temp = temp->next;
            // free(pre);
            node_Z = node_Z->next;

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        } // end loop of subtensors X
    } // End loop of Buckts of HtX
    unsigned int Max_hta_size = 0, Min_hta_size = 64;
    for(sptIndex i = 0; i < tk; i++){
        Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
        Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 

// 26: All Hash table formats, upper bound to predict the nnzs of C, hta select
  if(experiment_modes == 15){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
 
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    tensor_table_t *X_ht;
    unsigned int X_ht_size = X->nnz;
    X_ht = tensor_htCreate(X_ht_size);    
    
    omp_lock_t *locks_x = (omp_lock_t *)malloc(X_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < X_ht_size; i++) omp_init_lock(&locks_x[i]);
    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }


    sptNnzIndex X_nnz = X->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(X_ht, X_num_fmodes, mode_order_X, num_cmodes, X_cmode_inds, X_fmode_inds)
    for(sptNnzIndex i = 0; i < X_nnz; i++){
        int tid = omp_get_thread_num();

        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += X->inds[mode_order_X[X_num_fmodes+m]].data[i] * X_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_fmodes += X->inds[mode_order_X[m]].data[i] * X_fmode_inds[m + 1];
        

        unsigned pos = tensor_htHashCode_size(key_fmodes, X_nnz);
        omp_set_lock(&locks_x[pos]);    
        tensor_value X_val = tensor_htGet(X_ht, key_fmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(X_val.len == 0) {
            tensor_htInsert(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
        }
        else{
            tensor_htUpdate(X_ht, key_fmodes, key_cmodes, X->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks_x[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < X_ht_size; i++) omp_destroy_lock(&locks_x[i]);


    tensor_table_t *Y_ht;
    unsigned int Y_ht_size = Y->nnz;
    Y_ht = tensor_htCreate(Y_ht_size);    
    
    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = (tensor_table_t*)malloc(sizeof(tensor_table_t));
    Z_ht->size = X_ht_size;
    Z_ht->list = (tensor_node_t**)malloc(sizeof(tensor_node_t*)*Z_ht->size);
    for(sptNnzIndex i = 0; i < X_ht_size; i++){
        Z_ht->list[i] = NULL;
        tensor_node_t* temp = X_ht->list[i];
        while(temp){
            tensor_node_t *newNode = (tensor_node_t*) malloc(sizeof(tensor_node_t));
            newNode->key = temp->key;
            newNode->next = Z_ht->list[i];
            Z_ht->list[i] = newNode;
            temp = temp->next;
        }
    }
    sptTimer timer_hty;
    double time_hty_calckey = 0;
    sptNewTimer(&timer_hty, 0);

    sptNnzIndex Y_nnz = Y->nnz;
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex i = 0; i < Y_nnz; i++){
        //if (Y->values.data[i] <0.00000001) continue;
        int tid = omp_get_thread_num();
        if(tid == 0){
            sptStartTimer(timer_hty);
        }
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];    

        unsigned long long key_fmodes = 0;    
        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
        
        if(tid == 0){
            sptStopTimer(timer_hty);
            time_hty_calckey += sptElapsedTime(timer_hty);
        }

        unsigned pos = tensor_htHashCode_size(key_cmodes, Y_nnz);
        omp_set_lock(&locks[pos]);    
        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
        //printf("Y_val.len: %d\n", Y_val.len); 
        if(Y_val.len == 0) {
            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
        }
        else  {
            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
            //for(int i = 0; i < Y_val.len; i++)
            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]); 
        }
        omp_unset_lock(&locks[pos]);    
        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
    }

    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);
    // sptFreeSparseTensor(Y);
    sptNnzIndex empty_buckets = 0;
    sptIndex max_bucket_length = 0;
    sptNnzIndex num_collisions = 0;
    sptNnzIndex num_elements = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtY += sizeof(int) + Y_ht_size * sizeof(tensor_node_t*);
    for(sptNnzIndex i = 0; i < Y_ht_size; i++){
        tensor_node_t *list = Y_ht->list[i];
        tensor_node_t *temp = list;
        sptIndex count = -1;
        if(temp){
            while(temp){
                // bytes_HtY += sizeof(unsigned long long) + sizeof(tensor_value) + sizeof(tensor_node_t*);
                bytes_HtY += sizeof(tensor_node_t);
                count++;
                temp = temp->next;
            }
            num_collisions += count;
            num_elements += (count + 1);
            max_bucket_length = max_bucket_length >= count ? max_bucket_length : count;
        }else{
            empty_buckets++;
        }
    }
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Whole hashbucket]: %lu \n", Y_ht_size);
    printf("[Empty hashbucket]: %ld \n", empty_buckets);
    printf("[Wasted hash rate]: %%%.7lf\n", 100.0*(1.0*empty_buckets)/(1.0*Y_ht_size));
    printf("[Hash  collisions]: %ld \n", num_collisions);
    printf("[Max   collision]: %ld \n", max_bucket_length);
    printf("[Hash  elements]: %ld \n", num_elements);
    printf("[Loading Factor]: %.7lf \n", num_elements*1.0/Y_nnz);
    sptNnzIndex nnz_bucket = Y_ht_size - empty_buckets;
    printf("[Avg   collision]: %.7lf \n", 1.0*num_collisions/(1.0*nnz_bucket));
    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    printf("[Input Calc htKey]: %.6f s\n", time_hty_calckey);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
    //printf("fidx_X: \n");
    //sptDumpNnzIndexVector(&fidx_X, stdout);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }

    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }

    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    unsigned int min_hta_size = 20, max_hta_size = 0;

    sptStartTimer(timer);
    omp_lock_t locks_htsz;
    omp_init_lock(&locks_htsz);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht_size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        tensor_node_t* temp = X_ht->list[fx_ptr];
        tensor_node_t* node_Z = Z_ht->list[fx_ptr];
        while(temp){ // If collisions occur
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = temp->val;    
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            // unsigned int ht_size = std::min(std::max((int)ceil(log2(16*nnz_upper_bound)),2), 25);
            if(ceil(log2(16*nnz_upper_bound)) <= 16){ //CK_mult
                unsigned int ht_size = std::min(std::max((int)ceil(log2(16*nnz_upper_bound)),2), 16);

                omp_set_lock(&locks_htsz);
                min_hta_size = std::min(ht_size, min_hta_size);
                max_hta_size = std::max(ht_size, max_hta_size);
                omp_unset_lock(&locks_htsz);
                // printf("ht_size: %u\n", ht_size);
                // unsigned int ht_size = 16;
                unsigned int hashtable_a_size = (1 << ht_size);
                unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
                sptIndex nmodes_spa = nmodes_Y - num_cmodes;
                long int nnz_counter = 0;
                sptIndex current_idx = 0;
                CK_table_t *CK_ht;
                CK_ht = CK_htCreate(ht_size);
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_prep += sptElapsedTime(timer_SPA);
                }

                for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                    if (tid == 0){
                        sptStartTimer(timer_SPA);
                    }
                    sptValue valX = X_val.val[zX];
                    unsigned long long key_cmodes = X_val.key_FM[zX];

                    tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);

                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_free_mode += sptElapsedTime(timer_SPA);
                    }
                    unsigned int Y_len = Y_val.len;
                    if(Y_len == 0) continue;
                    if (tid == 0){
                        sptStartTimer(timer_SPA);
                    }
                    for(unsigned int zY = 0; zY < Y_len; zY++){
                        unsigned long long fmode = Y_val.key_FM[zY];
                        unsigned long long pos = CK_htGet(CK_ht, fmode);
                        float result = Y_val.val[zY]*valX;

                        if(pos == ULLONG_MAX){
                            pos_nnz[CK_ht->len] = CK_htInsert(CK_ht, 0, fmode, result);
                            nnz_counter++;
                        }else{
                            CK_htUpdate(CK_ht, pos, fmode, result);
                        }
                    } 
                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_spa += sptElapsedTime(timer_SPA);
                    } 
                } // End Loop nnzs inside a X subtensor

                if (tid == 0) {
                    sptStartTimer(timer_SPA);    
                }
                node_Z->val.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
                node_Z->val.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
                node_Z->val.len = nnz_counter;
                for(unsigned int z = 0; z < CK_ht->len; z++){
                    unsigned int pos = pos_nnz[z];
                    unsigned int tableidx = (pos & 0x8000000) >> 31;
                    unsigned int real_pos = pos & 0x7fffffff;
                    node_Z->val.key_FM[z] = CK_ht->key[tableidx][real_pos];
                    node_Z->val.val[z] = CK_ht->val[tableidx][real_pos];
                }

                Z_tmp[tid].nnz += nnz_counter;
                free(pos_nnz);
                CK_htFree(CK_ht);
                // tensor_node_t* pre = temp;
                temp = temp->next;
                // free(pre);
                node_Z = node_Z->next;

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_accumulate_z += sptElapsedTime(timer_SPA);
                }
            }else{ // Use LP
                unsigned int ht_size = std::max((int)ceil(log2(4*nnz_upper_bound)),2);
                omp_set_lock(&locks_htsz);
                min_hta_size = std::min(ht_size, min_hta_size);
                max_hta_size = std::max(ht_size, max_hta_size);
                omp_unset_lock(&locks_htsz);
                // printf("ht_size: %u\n", ht_size);
                // unsigned int ht_size = 16;
                unsigned int hashtable_a_size = (1 << ht_size);
                unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
                sptIndex nmodes_spa = nmodes_Y - num_cmodes;
                long int nnz_counter = 0;
                sptIndex current_idx = 0;
                LP_table_t *LP_ht;
                LP_ht = LP_htCreate(ht_size);
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_prep += sptElapsedTime(timer_SPA);
                }

                for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                    if (tid == 0){
                        sptStartTimer(timer_SPA);
                    }
                    sptValue valX = X_val.val[zX];
                    unsigned long long key_cmodes = X_val.key_FM[zX];

                    tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);

                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_free_mode += sptElapsedTime(timer_SPA);
                    }
                    unsigned int Y_len = Y_val.len;
                    if(Y_len == 0) continue;
                    if (tid == 0){
                        sptStartTimer(timer_SPA);
                    }
                    for(unsigned int zY = 0; zY < Y_len; zY++){
                        unsigned long long fmode = Y_val.key_FM[zY];
                        sptValue spa_val = LP_htGet(LP_ht, fmode);
                        float result = Y_val.val[zY]*valX;

                        if(spa_val == LONG_MIN){
                            pos_nnz[LP_ht->len] = LP_htInsert(LP_ht, fmode, result);
                            nnz_counter++;
                        }else{
                            LP_htUpdate(LP_ht, fmode, spa_val + result);
                        }
                    } 
                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_spa += sptElapsedTime(timer_SPA);
                    } 
                } // End Loop nnzs inside a X subtensor

                if (tid == 0) {
                    sptStartTimer(timer_SPA);    
                }
                node_Z->val.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
                node_Z->val.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
                node_Z->val.len = nnz_counter;
                for(unsigned int z = 0; z < LP_ht->len; z++){
                    unsigned int i = pos_nnz[z];
                    node_Z->val.key_FM[z] = LP_ht->key[i];
                    node_Z->val.val[z] = LP_ht->val[i];
                }

                Z_tmp[tid].nnz += nnz_counter;
                free(pos_nnz);
                LP_htFree(LP_ht);
                // tensor_node_t* pre = temp;
                temp = temp->next;
                // free(pre);
                node_Z = node_Z->next;

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_accumulate_z += sptElapsedTime(timer_SPA);
                }                
            }

        } // end loop of subtensors X
    } // End loop of Buckts of HtX
    omp_destroy_lock(&locks_htsz);
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    printf("[HtA     size]: [%u, %u]\n", min_hta_size, max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 

    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 




// 27: All Hash table formats, LP+CK
  if(experiment_modes == 21){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    sptTimer timer_writehty;
    sptNewTimer(&timer_writehty, 0);
    double time_writehty = 0;

    sptTimer timer_writeht;
    sptNewTimer(&timer_writeht, 0);
    double time_writeht = 0;

    sptTimer timer_inputsort;
    sptNewTimer(&timer_inputsort, 0);
    double time_inputsort = 0;

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptStartTimer(timer_inputsort);
    sptSparseTensorSortIndex(X, 1, tk);
    sptStopTimer(timer_inputsort);
    time_inputsort += sptElapsedTime(timer_inputsort);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    printf("[Num of subtensorX]:%lu\n", fidx_X.len - 1);
    LP_tensor_table_t *X_ht;
    unsigned int X_ht_size = ceil(log2(2*(fidx_X.len-1)));
    printf("X_ht_size: %u\n", X_ht_size);
    X_ht = LP_tensor_htCreate(X_ht_size);

    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }
    sptNnzIndex X_nnz = X->nnz;

    // omp_lock_t *locks_x = (omp_lock_t *)malloc(X_ht->size*sizeof(omp_lock_t));
    // for(size_t i = 0; i < X_ht->size; i++) omp_init_lock(&locks_x[i]);
    sptStartTimer(timer_writeht);
    #pragma omp parallel for schedule(static) num_threads(tk) shared(X_ht, X_num_fmodes, mode_order_X, num_cmodes, X_cmode_inds, X_fmode_inds)
    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len-1; fx_ptr++){
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptNnzIndex nnzs_subtensor = fx_end - fx_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_cmodes += X->inds[mode_order_X[m]].data[fx_begin] * X_fmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, X_ht->shift);
        // omp_set_lock(&locks_x[pos]);
        int size = X_ht->size;
        while(size--){
            // int prev_stat;
            // #pragma omp atomic capture
            // {
            //     prev_stat = X_ht->LP_node_list[pos].stat;
            //     if(prev_stat == 0) X_ht->LP_node_list[pos].stat = 1;
            // }
            int prev_stat;
            #pragma omp atomic capture
            {
                prev_stat = X_ht->LP_node_list[pos].stat;
                X_ht->LP_node_list[pos].stat = 1;
            }
            if(prev_stat == 0){
                // X_ht->LP_node_list[pos].stat = 1;
                X_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(X_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zX = fx_begin; zX < fx_end; zX++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < num_cmodes; ++m)
                        key_fmodes += X->inds[mode_order_X[m+X_num_fmodes]].data[zX] * X_cmode_inds[m + 1];
                    tmp->key_FM[zX-fx_begin] = key_fmodes;
                    tmp->val[zX-fx_begin] = X->values.data[zX];
                }
                X_ht->len++;
                break;
            }
            else{
                // omp_unset_lock(&locks_x[pos]);
                // printf("Collisions occur\n");
                pos++;
                if(pos >= X_ht->size) pos = 0;
                // omp_set_lock(&locks_x[pos]);
            }
        }
        // omp_unset_lock(&locks_x[pos]);
    }
    sptStopTimer(timer_writeht);
    time_writeht += sptElapsedTime(timer_writeht);
    // for(size_t i = 0; i < X_ht->size; i++) omp_destroy_lock(&locks_x[i]);
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    sptNnzIndexVector fidx_Y;
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order


    sptStartTimer(timer_inputsort);
    sptSparseTensorSortIndex(Y, 1, tk);
    sptStopTimer(timer_inputsort);
    time_inputsort += sptElapsedTime(timer_inputsort);
    
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);

    printf("[Num of subtensorY]:%lu\n", fidx_Y.len - 1);
    LP_tensor_table_t *Y_ht;
    unsigned int Y_ht_size = ceil(log2(2*(fidx_Y.len-1)));
    printf("Y_ht_size: %u\n", Y_ht_size);
    Y_ht = LP_tensor_htCreate(Y_ht_size);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;

    sptStartTimer(timer_writeht);
    sptStartTimer(timer_writehty);
    // omp_lock_t *locks_y = (omp_lock_t *)malloc(Y_ht->size*sizeof(omp_lock_t));
    // for(size_t i = 0; i < Y_ht->size; i++) omp_init_lock(&locks_y[i]);
    #pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
    for(sptNnzIndex fy_ptr = 0; fy_ptr < fidx_Y.len-1; fy_ptr++){
        sptNnzIndex fy_begin = fidx_Y.data[fy_ptr];
        sptNnzIndex fy_end = fidx_Y.data[fy_ptr+1];
        sptNnzIndex nnzs_subtensor = fy_end - fy_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[fy_begin] * Y_cmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, Y_ht->shift);
        // omp_set_lock(&locks_y[pos]);
        int size = Y_ht->size;
        while(size--){
            // int prev_stat;
            // #pragma omp atomic capture
            // {
            //     prev_stat = Y_ht->LP_node_list[pos].stat;
            //     if(prev_stat == 0) Y_ht->LP_node_list[pos].stat = 1;
            // }
            int prev_stat;
            #pragma omp atomic capture
            {
                prev_stat = Y_ht->LP_node_list[pos].stat;
                Y_ht->LP_node_list[pos].stat = 1;
            }
            if(prev_stat == 0){
                // Y_ht->LP_node_list[pos].stat = 1;
                Y_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(Y_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zY = fy_begin; zY < fy_end; zY++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < Y_num_fmodes; ++m)
                        key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[zY] * Y_fmode_inds[m + 1];
                    tmp->key_FM[zY-fy_begin] = key_fmodes;
                    tmp->val[zY-fy_begin] = Y->values.data[zY];
                }
                Y_ht->len++;
                break;
            }
            pos++;
            if(pos >= Y_ht->size) pos = 0;
        }
        // omp_unset_lock(&locks_y[pos]);
    }
    // for(size_t i = 0; i < Y_ht->size; i++) omp_destroy_lock(&locks_y[i]);
    sptStopTimer(timer_writehty);
    time_writehty += sptElapsedTime(timer_writehty);

    LP_tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = LP_tensor_htCreate(Z_ht_size);
    for(sptNnzIndex i = 0; i < X_ht->size; i++){
        if(X_ht->LP_node_list[i].stat == 1){
            Z_ht->LP_node_list[i].stat = 1;
            Z_ht->LP_node_list[i].key = X_ht->LP_node_list[i].key;
        }
    }
    sptStopTimer(timer_writeht);
    time_writeht += sptElapsedTime(timer_writeht);

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    // sptFreeSparseTensor(Y);
    sptNnzIndex bytes_HtX = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtX = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + X_ht->size*sizeof(LP_tensor_node_t) + (X->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    bytes_HtY = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Y_ht->size*sizeof(LP_tensor_node_t) + (Y->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtX]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtX / 1024.0, bytes_HtX / 1048576.0, bytes_HtX/ 1073741824.0);
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);
    printf("[Input Write_hash]: %.6f s\n", time_writeht);
    printf("[Input Sorting   ]: %.6f s\n", time_inputsort);
    printf("[Input ProcessingY]: %.6f s\n", time_writehty);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    // unsigned int hashtable_a_size = 0;
    // unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // for(sptIndex i = 0; i < tk ; i++){
    //     min_hta_size[i] = 64;
    //     max_hta_size[i] = 0;
    // }    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht->size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        if(X_ht->LP_node_list[fx_ptr].stat == 1){
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = X_ht->LP_node_list[fx_ptr].val_ptr;
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            nnz_upper_bound = std::min(nnz_upper_bound, fidx_X.len-1); // for A*A tensor contraction, the "num_cols_Y = num_rows_X
            unsigned int ht_size = std::max((int)ceil(log2(16*nnz_upper_bound)),10);
            // unsigned int ht_size = std::min(19, std::max((int)ceil(log2(16*nnz_upper_bound)),10));

            // min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
            // max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);

            unsigned int hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;
            CK_table_t *CK_ht;
            CK_ht = CK_htCreate(ht_size);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];

                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                unsigned int Y_len = Y_val.len;
                if(Y_len == 0) continue;

                // printf("Not empty!");
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                for(unsigned int zY = 0; zY < Y_len; zY++){
                    unsigned long long fmode = Y_val.key_FM[zY];
                    unsigned long long pos = CK_htGet(CK_ht, fmode);
                    float result = Y_val.val[zY]*valX;

                    if(pos == ULLONG_MAX){
                        pos_nnz[CK_ht->len] = CK_htInsert(CK_ht, 0, fmode, result);
                        nnz_counter++;
                        // printf("insert!");
                    }else{
                        CK_htUpdate(CK_ht, pos, fmode, result);
                    }
                } 
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                } 
            } // End Loop nnzs inside a X subtensor

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.len = nnz_counter;
            Z_ht->LP_node_list[fx_ptr].val_ptr.cap = nnz_counter;
            for(unsigned int z = 0; z < CK_ht->len; z++){
                unsigned int pos = pos_nnz[z];
                unsigned int tableidx = (pos & 0x8000000) >> 31;
                unsigned int real_pos = pos & 0x7fffffff;
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = CK_ht->key[tableidx][real_pos];
                Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = CK_ht->val[tableidx][real_pos];
            }

            free(pos_nnz);
            CK_htFree(CK_ht);
            Z_tmp[tid].nnz += nnz_counter;
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        }
    } // End loop of Buckts of HtX
    // unsigned int Max_hta_size = 0, Min_hta_size = 64;
    // for(sptIndex i = 0; i < tk; i++){
    //     Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
    //     Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    // }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    // printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 
    Z_ht->len = Z_total_size;
    sptNnzIndex bytes_HtZ;
    bytes_HtZ = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Z_ht->size*sizeof(LP_tensor_node_t) + (Z_total_size)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtZ]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtZ / 1024.0, bytes_HtZ / 1048576.0, bytes_HtZ / 1073741824.0);
    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 

// 28: All Hash table formats, LP+LP
  if(experiment_modes == 22){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    printf("[Num of subtensorX]:%lu\n", fidx_X.len - 1);
    LP_tensor_table_t *X_ht;
    unsigned int X_ht_size = ceil(log2(2*(fidx_X.len-1)));
    printf("X_ht_size: %u\n", X_ht_size);
    X_ht = LP_tensor_htCreate(X_ht_size);

    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }
    sptNnzIndex X_nnz = X->nnz;

    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len-1; fx_ptr++){
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptNnzIndex nnzs_subtensor = fx_end - fx_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_cmodes += X->inds[mode_order_X[m]].data[fx_begin] * X_fmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, X_ht->shift);
        int size = X_ht->size;
        while(size--){
            if(X_ht->LP_node_list[pos].stat == 0){
                X_ht->LP_node_list[pos].stat = 1;
                X_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(X_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zX = fx_begin; zX < fx_end; zX++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < num_cmodes; ++m)
                        key_fmodes += X->inds[mode_order_X[m+X_num_fmodes]].data[zX] * X_cmode_inds[m + 1];
                    tmp->key_FM[zX-fx_begin] = key_fmodes;
                    tmp->val[zX-fx_begin] = X->values.data[zX];
                }
                X_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= X_ht->size) pos = 0;
            }
        }
    }
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    sptNnzIndexVector fidx_Y;
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);

    printf("[Num of subtensorY]:%lu\n", fidx_Y.len - 1);
    LP_tensor_table_t *Y_ht;
    unsigned int Y_ht_size = ceil(log2(2*(fidx_Y.len-1)));
    printf("Y_ht_size: %u\n", Y_ht_size);
    Y_ht = LP_tensor_htCreate(Y_ht_size);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;
    for(sptNnzIndex fy_ptr = 0; fy_ptr < fidx_Y.len-1; fy_ptr++){
        sptNnzIndex fy_begin = fidx_Y.data[fy_ptr];
        sptNnzIndex fy_end = fidx_Y.data[fy_ptr+1];
        sptNnzIndex nnzs_subtensor = fy_end - fy_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[fy_begin] * Y_cmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, Y_ht->shift);
        int size = Y_ht->size;
        while(size--){
            if(Y_ht->LP_node_list[pos].stat == 0){
                Y_ht->LP_node_list[pos].stat = 1;
                Y_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(Y_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zY = fy_begin; zY < fy_end; zY++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < Y_num_fmodes; ++m)
                        key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[zY] * Y_fmode_inds[m + 1];
                    tmp->key_FM[zY-fy_begin] = key_fmodes;
                    tmp->val[zY-fy_begin] = Y->values.data[zY];
                }
                Y_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= Y_ht->size) pos = 0;
            }
        }
    }
    LP_tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = LP_tensor_htCreate(Z_ht_size);
    for(sptNnzIndex i = 0; i < X_ht->size; i++){
        if(X_ht->LP_node_list[i].stat == 1){
            Z_ht->LP_node_list[i].stat = 1;
            Z_ht->LP_node_list[i].key = X_ht->LP_node_list[i].key;
        }
    }

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    // sptFreeSparseTensor(Y);
    sptNnzIndex bytes_HtX = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtX = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + X_ht->size*sizeof(LP_tensor_node_t) + (X->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    bytes_HtY = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Y_ht->size*sizeof(LP_tensor_node_t) + (Y->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtX]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtX / 1024.0, bytes_HtX / 1048576.0, bytes_HtX/ 1073741824.0);
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    // unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // for(sptIndex i = 0; i < tk ; i++){
    //     min_hta_size[i] = 64;
    //     max_hta_size[i] = 0;
    // }    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht->size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        if(X_ht->LP_node_list[fx_ptr].stat == 1){
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = X_ht->LP_node_list[fx_ptr].val_ptr;
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            nnz_upper_bound = std::min(nnz_upper_bound, fidx_X.len-1); // for A*A tensor contraction, the "num_cols_Y = num_rows_X
            unsigned int ht_size = std::max((int)ceil(log2(16*nnz_upper_bound)),10);
            // unsigned int ht_size = std::min(19, std::max((int)ceil(log2(16*nnz_upper_bound)),10));
            // min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
            // max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);
            // sptNnzIndex nnz_upper_bound = 0;
            // unsigned int ht_size = 15;
            // unsigned int ht_size = ceil(log2((fidx_X.len-1)*(fx_end-fx_begin)))-2; // 2^13 = 8192
            // const unsigned int ht_size = 100;
            // hashtable_a_size = ht_size;
            unsigned int hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            // const unsigned int ht_size = 9973;
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;
            // table_t *ht;
            // ht = htCreate(ht_size);
            LP_table_t *LP_ht;
            LP_ht = LP_htCreate(ht_size);

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X  
                if (tid == 0) {
                    sptStartTimer(timer_SPA);
                }       
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX]; 

                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);  

                unsigned int my_len = Y_val.len;
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                if(my_len == 0) continue;

                if (tid == 0) {
                    sptStartTimer(timer_SPA);       
                }        
                
                for(int i = 0; i < my_len; i++){
                    unsigned long long fmode =  Y_val.key_FM[i];
                    sptValue spa_val = LP_htGet(LP_ht, fmode);
                    float result = Y_val.val[i] * valX;

                    if(spa_val == LONG_MIN) {
                        pos_nnz[LP_ht->len] = LP_htInsert(LP_ht, fmode, result);
                        nnz_counter++;
                    }
                    else{
                        LP_htUpdate(LP_ht, fmode, spa_val + result);
                    }
                }

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                }
                
            }   // End Loop nnzs inside a X fiber

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.len = nnz_counter;
            Z_ht->LP_node_list[fx_ptr].val_ptr.cap = nnz_counter;
            for(int z = 0; z < LP_ht->len; z++){
                unsigned int pos = pos_nnz[z];
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = LP_ht->key[pos];
                Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = LP_ht->val[pos];
            }
            // printf("bytes_HtA:%lu", bytes_HtA);
            Z_tmp[tid].nnz += nnz_counter;

            LP_htFree(LP_ht);
            free(pos_nnz);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }    
        }
    } // End loop of Buckts of HtX
    // unsigned int Max_hta_size = 0, Min_hta_size = 64;
    // for(sptIndex i = 0; i < tk; i++){
    //     Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
    //     Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    // }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    // printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 
    Z_ht->len = Z_total_size;
    sptNnzIndex bytes_HtZ;
    bytes_HtZ = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Z_ht->size*sizeof(LP_tensor_node_t) + (Z_total_size)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtZ]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtZ / 1024.0, bytes_HtZ / 1048576.0, bytes_HtZ / 1073741824.0);
    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 

// 271: All Hash table formats, LP+CK
  if(experiment_modes == 23){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    printf("[Num of subtensorX]:%lu\n", fidx_X.len - 1);
    LP_tensor_table_t *X_ht;
    unsigned int X_ht_size = ceil(log2(2*(fidx_X.len-1)));
    printf("X_ht_size: %u\n", X_ht_size);
    X_ht = LP_tensor_htCreate(X_ht_size);

    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }
    sptNnzIndex X_nnz = X->nnz;

    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len-1; fx_ptr++){
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptNnzIndex nnzs_subtensor = fx_end - fx_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_cmodes += X->inds[mode_order_X[m]].data[fx_begin] * X_fmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, X_ht->shift);
        int size = X_ht->size;
        while(size--){
            if(X_ht->LP_node_list[pos].stat == 0){
                X_ht->LP_node_list[pos].stat = 1;
                X_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(X_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zX = fx_begin; zX < fx_end; zX++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < num_cmodes; ++m)
                        key_fmodes += X->inds[mode_order_X[m+X_num_fmodes]].data[zX] * X_cmode_inds[m + 1];
                    tmp->key_FM[zX-fx_begin] = key_fmodes;
                    tmp->val[zX-fx_begin] = X->values.data[zX];
                }
                X_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= X_ht->size) pos = 0;
            }
        }
    }
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    sptNnzIndexVector fidx_Y;
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);

    printf("[Num of subtensorY]:%lu\n", fidx_Y.len - 1);
    LP_tensor_table_t *Y_ht;
    unsigned int Y_ht_size = ceil(log2(2*(fidx_Y.len-1)));
    printf("Y_ht_size: %u\n", Y_ht_size);
    Y_ht = LP_tensor_htCreate(Y_ht_size);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;
    for(sptNnzIndex fy_ptr = 0; fy_ptr < fidx_Y.len-1; fy_ptr++){
        sptNnzIndex fy_begin = fidx_Y.data[fy_ptr];
        sptNnzIndex fy_end = fidx_Y.data[fy_ptr+1];
        sptNnzIndex nnzs_subtensor = fy_end - fy_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[fy_begin] * Y_cmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, Y_ht->shift);
        int size = Y_ht->size;
        while(size--){
            if(Y_ht->LP_node_list[pos].stat == 0){
                Y_ht->LP_node_list[pos].stat = 1;
                Y_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(Y_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zY = fy_begin; zY < fy_end; zY++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < Y_num_fmodes; ++m)
                        key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[zY] * Y_fmode_inds[m + 1];
                    tmp->key_FM[zY-fy_begin] = key_fmodes;
                    tmp->val[zY-fy_begin] = Y->values.data[zY];
                }
                Y_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= Y_ht->size) pos = 0;
            }
        }
    }
    LP_tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = LP_tensor_htCreate(Z_ht_size);
    for(sptNnzIndex i = 0; i < X_ht->size; i++){
        if(X_ht->LP_node_list[i].stat == 1){
            Z_ht->LP_node_list[i].stat = 1;
            Z_ht->LP_node_list[i].key = X_ht->LP_node_list[i].key;
        }
    }

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    // sptFreeSparseTensor(Y);
    sptNnzIndex bytes_HtX = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtX = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + X_ht->size*sizeof(LP_tensor_node_t) + (X->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    bytes_HtY = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Y_ht->size*sizeof(LP_tensor_node_t) + (Y->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtX]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtX / 1024.0, bytes_HtX / 1048576.0, bytes_HtX/ 1073741824.0);
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    // unsigned int hashtable_a_size = 0;
    // unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // for(sptIndex i = 0; i < tk ; i++){
    //     min_hta_size[i] = 64;
    //     max_hta_size[i] = 0;
    // }    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht->size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        if(X_ht->LP_node_list[fx_ptr].stat == 1){
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = X_ht->LP_node_list[fx_ptr].val_ptr;
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            nnz_upper_bound = std::min(nnz_upper_bound, fidx_X.len-1); // for A*A tensor contraction, the "num_cols_Y = num_rows_X
            unsigned int ht_size = std::max((int)ceil(log2(16*nnz_upper_bound)),10);
            // unsigned int ht_size = std::min(19, std::max((int)ceil(log2(16*nnz_upper_bound)),10));

            // min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
            // max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);

            unsigned int hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;
            CK_table_t *CK_ht;
            CK_ht = CK_htCreate(ht_size);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];

                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                unsigned int Y_len = Y_val.len;
                if(Y_len == 0) continue;

                // printf("Not empty!");
                if (tid == 0){
                    sptStartTimer(timer_SPA);
                }
                for(unsigned int zY = 0; zY < Y_len; zY++){
                    unsigned long long fmode = Y_val.key_FM[zY];
                    unsigned long long pos = CK_htGet(CK_ht, fmode);
                    float result = Y_val.val[zY]*valX;

                    if(pos == ULLONG_MAX){
                        pos_nnz[CK_ht->len] = CK_htInsert(CK_ht, 0, fmode, result);
                        nnz_counter++;
                        // printf("insert!");
                    }else{
                        CK_htUpdate(CK_ht, pos, fmode, result);
                    }
                } 
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                } 
            } // End Loop nnzs inside a X subtensor

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            // Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            // Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.len = nnz_counter;
            Z_ht->LP_node_list[fx_ptr].val_ptr.cap = nnz_counter;
            // process the first one
            unsigned int pos = pos_nnz[0];
            unsigned int tableidx = (pos & 0x8000000) >> 31;
            unsigned int real_pos = pos & 0x7fffffff;
            Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long));
            Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue));
            Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[0] = CK_ht->key[tableidx][real_pos];
            Z_ht->LP_node_list[fx_ptr].val_ptr.val[0] = CK_ht->val[tableidx][real_pos];

            for(unsigned int z = 1; z < CK_ht->len; z++){
                unsigned int pos = pos_nnz[z];
                unsigned int tableidx = (pos & 0x8000000) >> 31;
                unsigned int real_pos = pos & 0x7fffffff;
                // Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = CK_ht->key[tableidx][real_pos];
                // Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = CK_ht->val[tableidx][real_pos];
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*) realloc(Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM, (z+1)*sizeof(unsigned long long));
                Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*) realloc(Z_ht->LP_node_list[fx_ptr].val_ptr.val, (z+1)*sizeof(sptValue));
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = CK_ht->key[tableidx][real_pos];
                Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = CK_ht->val[tableidx][real_pos];
            }

            free(pos_nnz);
            CK_htFree(CK_ht);
            Z_tmp[tid].nnz += nnz_counter;
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }
        }
    } // End loop of Buckts of HtX
    // unsigned int Max_hta_size = 0, Min_hta_size = 64;
    // for(sptIndex i = 0; i < tk; i++){
    //     Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
    //     Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    // }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    // printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 
    Z_ht->len = Z_total_size;
    sptNnzIndex bytes_HtZ;
    bytes_HtZ = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Z_ht->size*sizeof(LP_tensor_node_t) + (Z_total_size)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtZ]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtZ / 1024.0, bytes_HtZ / 1048576.0, bytes_HtZ / 1073741824.0);
    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 

// 281: All Hash table formats, LP+LP
  if(experiment_modes == 24){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    printf("[Num of subtensorX]:%lu\n", fidx_X.len - 1);
    LP_tensor_table_t *X_ht;
    unsigned int X_ht_size = ceil(log2(2*(fidx_X.len-1)));
    printf("X_ht_size: %u\n", X_ht_size);
    X_ht = LP_tensor_htCreate(X_ht_size);

    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }
    sptNnzIndex X_nnz = X->nnz;

    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len-1; fx_ptr++){
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptNnzIndex nnzs_subtensor = fx_end - fx_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_cmodes += X->inds[mode_order_X[m]].data[fx_begin] * X_fmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, X_ht->shift);
        int size = X_ht->size;
        while(size--){
            if(X_ht->LP_node_list[pos].stat == 0){
                X_ht->LP_node_list[pos].stat = 1;
                X_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(X_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zX = fx_begin; zX < fx_end; zX++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < num_cmodes; ++m)
                        key_fmodes += X->inds[mode_order_X[m+X_num_fmodes]].data[zX] * X_cmode_inds[m + 1];
                    tmp->key_FM[zX-fx_begin] = key_fmodes;
                    tmp->val[zX-fx_begin] = X->values.data[zX];
                }
                X_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= X_ht->size) pos = 0;
            }
        }
    }
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    sptNnzIndexVector fidx_Y;
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);

    printf("[Num of subtensorY]:%lu\n", fidx_Y.len - 1);
    LP_tensor_table_t *Y_ht;
    unsigned int Y_ht_size = ceil(log2(2*(fidx_Y.len-1)));
    printf("Y_ht_size: %u\n", Y_ht_size);
    Y_ht = LP_tensor_htCreate(Y_ht_size);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;
    for(sptNnzIndex fy_ptr = 0; fy_ptr < fidx_Y.len-1; fy_ptr++){
        sptNnzIndex fy_begin = fidx_Y.data[fy_ptr];
        sptNnzIndex fy_end = fidx_Y.data[fy_ptr+1];
        sptNnzIndex nnzs_subtensor = fy_end - fy_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[fy_begin] * Y_cmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, Y_ht->shift);
        int size = Y_ht->size;
        while(size--){
            if(Y_ht->LP_node_list[pos].stat == 0){
                Y_ht->LP_node_list[pos].stat = 1;
                Y_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(Y_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zY = fy_begin; zY < fy_end; zY++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < Y_num_fmodes; ++m)
                        key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[zY] * Y_fmode_inds[m + 1];
                    tmp->key_FM[zY-fy_begin] = key_fmodes;
                    tmp->val[zY-fy_begin] = Y->values.data[zY];
                }
                Y_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= Y_ht->size) pos = 0;
            }
        }
    }
    LP_tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = LP_tensor_htCreate(Z_ht_size);
    for(sptNnzIndex i = 0; i < X_ht->size; i++){
        if(X_ht->LP_node_list[i].stat == 1){
            Z_ht->LP_node_list[i].stat = 1;
            Z_ht->LP_node_list[i].key = X_ht->LP_node_list[i].key;
        }
    }

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    // sptFreeSparseTensor(Y);
    sptNnzIndex bytes_HtX = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtX = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + X_ht->size*sizeof(LP_tensor_node_t) + (X->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    bytes_HtY = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Y_ht->size*sizeof(LP_tensor_node_t) + (Y->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtX]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtX / 1024.0, bytes_HtX / 1048576.0, bytes_HtX/ 1073741824.0);
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    // unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    // for(sptIndex i = 0; i < tk ; i++){
    //     min_hta_size[i] = 64;
    //     max_hta_size[i] = 0;
    // }    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht->size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        if(X_ht->LP_node_list[fx_ptr].stat == 1){
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = X_ht->LP_node_list[fx_ptr].val_ptr;
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            nnz_upper_bound = std::min(nnz_upper_bound, fidx_X.len-1); // for A*A tensor contraction, the "num_cols_Y = num_rows_X
            unsigned int ht_size = std::max((int)ceil(log2(16*nnz_upper_bound)),10);
            // unsigned int ht_size = std::min(19, std::max((int)ceil(log2(16*nnz_upper_bound)),10));
            // min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
            // max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);
            // sptNnzIndex nnz_upper_bound = 0;
            // unsigned int ht_size = 15;
            // unsigned int ht_size = ceil(log2((fidx_X.len-1)*(fx_end-fx_begin)))-2; // 2^13 = 8192
            // const unsigned int ht_size = 100;
            // hashtable_a_size = ht_size;
            unsigned int hashtable_a_size = (1 << ht_size);
            unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
            // const unsigned int ht_size = 9973;
            sptIndex nmodes_spa = nmodes_Y - num_cmodes;
            long int nnz_counter = 0;
            sptIndex current_idx = 0;
            // table_t *ht;
            // ht = htCreate(ht_size);
            LP_table_t *LP_ht;
            LP_ht = LP_htCreate(ht_size);

            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_prep += sptElapsedTime(timer_SPA);
            }

            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X  
                if (tid == 0) {
                    sptStartTimer(timer_SPA);
                }       
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX]; 

                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);  

                unsigned int my_len = Y_val.len;
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_free_mode += sptElapsedTime(timer_SPA);
                }
                if(my_len == 0) continue;

                if (tid == 0) {
                    sptStartTimer(timer_SPA);       
                }        
                
                for(int i = 0; i < my_len; i++){
                    unsigned long long fmode =  Y_val.key_FM[i];
                    sptValue spa_val = LP_htGet(LP_ht, fmode);
                    float result = Y_val.val[i] * valX;

                    if(spa_val == LONG_MIN) {
                        pos_nnz[LP_ht->len] = LP_htInsert(LP_ht, fmode, result);
                        nnz_counter++;
                    }
                    else{
                        LP_htUpdate(LP_ht, fmode, spa_val + result);
                    }
                }

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_spa += sptElapsedTime(timer_SPA);
                }
                
            }   // End Loop nnzs inside a X fiber

            if (tid == 0) {
                sptStartTimer(timer_SPA);    
            }
            // Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
            // Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
            Z_ht->LP_node_list[fx_ptr].val_ptr.len = nnz_counter;
            Z_ht->LP_node_list[fx_ptr].val_ptr.cap = nnz_counter;
            // process the first one
            Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long));
            Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue));
            unsigned int pos = pos_nnz[0];
            Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[0] = LP_ht->key[pos];
            Z_ht->LP_node_list[fx_ptr].val_ptr.val[0] = LP_ht->val[pos];

            for(int z = 1; z < LP_ht->len; z++){
                unsigned int pos = pos_nnz[z];
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*) realloc(Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM, (z+1)*sizeof(unsigned long long));
                Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*) realloc(Z_ht->LP_node_list[fx_ptr].val_ptr.val, (z+1)*sizeof(sptValue));
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = LP_ht->key[pos];
                Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = LP_ht->val[pos];
            }
            // printf("bytes_HtA:%lu", bytes_HtA);
            Z_tmp[tid].nnz += nnz_counter;

            LP_htFree(LP_ht);
            free(pos_nnz);
            if (tid == 0){
                sptStopTimer(timer_SPA);
                time_accumulate_z += sptElapsedTime(timer_SPA);
            }    
        }
    } // End loop of Buckts of HtX
    // unsigned int Max_hta_size = 0, Min_hta_size = 64;
    // for(sptIndex i = 0; i < tk; i++){
    //     Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
    //     Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    // }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    // printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 
    Z_ht->len = Z_total_size;
    sptNnzIndex bytes_HtZ;
    bytes_HtZ = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Z_ht->size*sizeof(LP_tensor_node_t) + (Z_total_size)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtZ]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtZ / 1024.0, bytes_HtZ / 1048576.0, bytes_HtZ / 1073741824.0);
    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 


// 29: All Hash table formats, LP+LP/CK
  if(experiment_modes == 25){
    int result;
    sptIndex nmodes_X = X->nmodes;
    sptIndex nmodes_Y = Y->nmodes;
    sptTimer timer;
    double total_time = 0;
    sptNewTimer(&timer, 0);

    if(num_cmodes >= X->nmodes) {
        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
    }
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
        }
    }

    sptStartTimer(timer);
    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
    for(sptIndex m = 0; m < nmodes_X; ++m) {
        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
            mode_order_X[fi] = m;
            ++ fi;
        }
    }
    sptAssert(fi == nmodes_X - num_cmodes);
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_X[ci] = cmodes_X[m];
        ++ ci;
    }
    sptAssert(ci == nmodes_X);
    
    sptSparseTensorShuffleModes(X, mode_order_X);

    // printf("Permuted X:\n");
    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
    sptSparseTensorSortIndex(X, 1, tk);

    sptNnzIndexVector fidx_X;
    /// Set indices for free modes, use X
    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

    printf("[Num of subtensorX]:%lu\n", fidx_X.len - 1);
    LP_tensor_table_t *X_ht;
    unsigned int X_ht_size = ceil(log2(2*(fidx_X.len-1)));
    printf("X_ht_size: %u\n", X_ht_size);
    X_ht = LP_tensor_htCreate(X_ht_size);

    sptIndex X_num_fmodes = nmodes_X - num_cmodes;
    sptIndex* X_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) X_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            X_cmode_inds[i] = X_cmode_inds[i] * X->ndims[mode_order_X[X_num_fmodes+j]];    
    }

    
    sptIndex* X_fmode_inds = (sptIndex*)malloc((X_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < X_num_fmodes + 1; i++) X_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < X_num_fmodes;i++){
        for(sptIndex j = i; j < X_num_fmodes;j++)
            X_fmode_inds[i] = X_fmode_inds[i] * X->ndims[mode_order_X[j]]; 
    }
    sptNnzIndex X_nnz = X->nnz;

    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len-1; fx_ptr++){
        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
        sptNnzIndex nnzs_subtensor = fx_end - fx_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < X_num_fmodes; ++m)
            key_cmodes += X->inds[mode_order_X[m]].data[fx_begin] * X_fmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, X_ht->shift);
        int size = X_ht->size;
        while(size--){
            if(X_ht->LP_node_list[pos].stat == 0){
                X_ht->LP_node_list[pos].stat = 1;
                X_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(X_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zX = fx_begin; zX < fx_end; zX++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < num_cmodes; ++m)
                        key_fmodes += X->inds[mode_order_X[m+X_num_fmodes]].data[zX] * X_cmode_inds[m + 1];
                    tmp->key_FM[zX-fx_begin] = key_fmodes;
                    tmp->val[zX-fx_begin] = X->values.data[zX];
                }
                X_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= X_ht->size) pos = 0;
            }
        }
    }
    sptStopTimer(timer);
    //total_time += sptPrintElapsedTime(timer, "Sort X");
    double X_time = sptElapsedTime(timer);
    total_time += X_time;
    sptStartTimer(timer);

    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
    ci = 0;
    fi = num_cmodes;
    for(sptIndex m = 0; m < nmodes_Y; ++m) {
        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
            mode_order_Y[fi] = m;
            ++ fi;
        }
    }
    /// Copy the contract modes while keeping the contraction mode order
    for(sptIndex m = 0; m < num_cmodes; ++m) {
        mode_order_Y[ci] = cmodes_Y[m];
        ++ ci;
    }

    sptNnzIndexVector fidx_Y;
    sptSparseTensorShuffleModes(Y, mode_order_Y);
    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
    sptSparseTensorSortIndex(Y, 1, tk);
    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);

    printf("[Num of subtensorY]:%lu\n", fidx_Y.len - 1);
    LP_tensor_table_t *Y_ht;
    unsigned int Y_ht_size = ceil(log2(2*(fidx_Y.len-1)));
    printf("Y_ht_size: %u\n", Y_ht_size);
    Y_ht = LP_tensor_htCreate(Y_ht_size);

    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
    for(sptIndex i = 0; i < num_cmodes;i++){
        for(sptIndex j = i; j < num_cmodes;j++)
            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];    
    }

    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
    for(sptIndex i = 0; i < Y_num_fmodes;i++){
        for(sptIndex j = i; j < Y_num_fmodes;j++)
            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]]; 
    }

    sptNnzIndex Y_nnz = Y->nnz;
    for(sptNnzIndex fy_ptr = 0; fy_ptr < fidx_Y.len-1; fy_ptr++){
        sptNnzIndex fy_begin = fidx_Y.data[fy_ptr];
        sptNnzIndex fy_end = fidx_Y.data[fy_ptr+1];
        sptNnzIndex nnzs_subtensor = fy_end - fy_begin;
        unsigned long long key_cmodes = 0;    
        for(sptIndex m = 0; m < num_cmodes; ++m)
            key_cmodes += Y->inds[mode_order_Y[m]].data[fy_begin] * Y_cmode_inds[m + 1];  
        // find a pos to insert this subtensor
        unsigned int pos = MultShift_htHashCode(key_cmodes, Y_ht->shift);
        int size = Y_ht->size;
        while(size--){
            if(Y_ht->LP_node_list[pos].stat == 0){
                Y_ht->LP_node_list[pos].stat = 1;
                Y_ht->LP_node_list[pos].key = key_cmodes;
                // tensor_htNewValueVector(&(Y_ht->LP_node_list[pos].val_ptr), nnzs_subtensor, nnzs_subtensor);
                tensor_value *tmp = &(Y_ht->LP_node_list[pos].val_ptr);
                tmp->len = nnzs_subtensor;
                tmp->cap = nnzs_subtensor;
                tmp->key_FM = (unsigned long long *)malloc(nnzs_subtensor*sizeof(unsigned long long));
                tmp->val = (sptValue *)malloc(nnzs_subtensor*sizeof(sptValue));
                for(sptNnzIndex zY = fy_begin; zY < fy_end; zY++){
                    unsigned long long key_fmodes = 0;    
                    for(sptIndex m = 0; m < Y_num_fmodes; ++m)
                        key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[zY] * Y_fmode_inds[m + 1];
                    tmp->key_FM[zY-fy_begin] = key_fmodes;
                    tmp->val[zY-fy_begin] = Y->values.data[zY];
                }
                Y_ht->len++;
                break;
            }
            else{
                // printf("Collisions occur\n");
                pos++;
                if(pos >= Y_ht->size) pos = 0;
            }
        }
    }
    LP_tensor_table_t *Z_ht;
    unsigned int Z_ht_size = X_ht_size;
    Z_ht = LP_tensor_htCreate(Z_ht_size);
    for(sptNnzIndex i = 0; i < X_ht->size; i++){
        if(X_ht->LP_node_list[i].stat == 1){
            Z_ht->LP_node_list[i].stat = 1;
            Z_ht->LP_node_list[i].key = X_ht->LP_node_list[i].key;
        }
    }

    sptStopTimer(timer);     
    total_time += sptElapsedTime(timer);
    // sptFreeSparseTensor(Y);
    sptNnzIndex bytes_HtX = 0;
    sptNnzIndex bytes_HtY = 0;
    bytes_HtX = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + X_ht->size*sizeof(LP_tensor_node_t) + (X->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    bytes_HtY = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Y_ht->size*sizeof(LP_tensor_node_t) + (Y->nnz)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtX]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtX / 1024.0, bytes_HtX / 1048576.0, bytes_HtX/ 1073741824.0);
    printf("[Memorycost of HtY]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtY / 1024.0, bytes_HtY / 1048576.0, bytes_HtY/ 1073741824.0);
    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

    /// Allocate the output tensor
    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
    sptIndex *ndims_buf = (sptIndex *) malloc(nmodes_Z * sizeof *ndims_buf);
    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
        ndims_buf[m] = X->ndims[mode_order_X[m]];
    }
    /// For non-sorted Y 
    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
    }
    free(mode_order_X);
    free(mode_order_Y);
    sptFreeSparseTensor(X);
    sptFreeSparseTensor(Y);

    /// Each thread with a local Z_tmp
    sptSparseTensor *Z_tmp = (sptSparseTensor *) malloc(tk * sizeof (sptSparseTensor));
    for (int i = 0; i < tk; i++){
        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
    }

    //free(ndims_buf);
    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);
    
    sptTimer timer_SPA;
    double time_prep = 0;
    double time_free_mode = 0;
    double time_spa = 0;
    double time_accumulate_z = 0;
    sptNewTimer(&timer_SPA, 0);
    double time_reversehash = 0;
    // double time_spa_search = 0;
    // double time_spa_compute = 0;
    double time_calckey = 0;
    double time_outsort = 0;
    sptTimer timer_hash;
    sptNewTimer(&timer_hash, 0);
    unsigned int hashtable_a_size = 0;
    unsigned int* min_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    unsigned int* max_hta_size = (unsigned int*)malloc(tk*sizeof(unsigned int));
    for(sptIndex i = 0; i < tk ; i++){
        min_hta_size[i] = 64;
        max_hta_size[i] = 0;
    }    

    sptStartTimer(timer);
    // For the progress
    #pragma omp parallel for schedule(dynamic) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, X_ht, Z_ht)     
    for(sptNnzIndex fx_ptr = 0; fx_ptr < X_ht->size; fx_ptr++){// Loop all buckets to loop the subtensor X
        int tid = omp_get_thread_num();
        if(X_ht->LP_node_list[fx_ptr].stat == 1){
            if (tid == 0){
                sptStartTimer(timer_SPA);
            }
            tensor_value X_val = X_ht->LP_node_list[fx_ptr].val_ptr;
            unsigned int X_len = X_val.len;

            sptNnzIndex nnz_upper_bound = 0;
            for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                sptValue valX = X_val.val[zX];
                unsigned long long key_cmodes = X_val.key_FM[zX];
                tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);
                unsigned int Y_len = Y_val.len;
                nnz_upper_bound += Y_val.len;
            } 
            nnz_upper_bound = std::min(nnz_upper_bound, fidx_X.len-1); // for A*A tensor contraction, the "num_cols_Y = num_rows_X
            if(ceil(log2(16*nnz_upper_bound)) <= 18){ // use CK
                unsigned int ht_size = std::max((int)ceil(log2(16*nnz_upper_bound)),11);
                min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
                max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);

                unsigned int hashtable_a_size = (1 << ht_size);
                unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
                sptIndex nmodes_spa = nmodes_Y - num_cmodes;
                long int nnz_counter = 0;
                sptIndex current_idx = 0;
                CK_table_t *CK_ht;
                CK_ht = CK_htCreate(ht_size);
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_prep += sptElapsedTime(timer_SPA);
                }

                for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X
                    if (tid == 0){
                        sptStartTimer(timer_SPA);
                    }
                    sptValue valX = X_val.val[zX];
                    unsigned long long key_cmodes = X_val.key_FM[zX];

                    tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);

                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_free_mode += sptElapsedTime(timer_SPA);
                    }
                    unsigned int Y_len = Y_val.len;
                    if(Y_len == 0) continue;

                    // printf("Not empty!");
                    if (tid == 0){
                        sptStartTimer(timer_SPA);
                    }
                    for(unsigned int zY = 0; zY < Y_len; zY++){
                        unsigned long long fmode = Y_val.key_FM[zY];
                        unsigned long long pos = CK_htGet(CK_ht, fmode);
                        float result = Y_val.val[zY]*valX;

                        if(pos == ULLONG_MAX){
                            pos_nnz[CK_ht->len] = CK_htInsert(CK_ht, 0, fmode, result);
                            nnz_counter++;
                            // printf("insert!");
                        }else{
                            CK_htUpdate(CK_ht, pos, fmode, result);
                        }
                    } 
                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_spa += sptElapsedTime(timer_SPA);
                    } 
                } // End Loop nnzs inside a X subtensor

                if (tid == 0) {
                    sptStartTimer(timer_SPA);    
                }
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
                Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
                Z_ht->LP_node_list[fx_ptr].val_ptr.len = nnz_counter;
                Z_ht->LP_node_list[fx_ptr].val_ptr.cap = nnz_counter;

                for(unsigned int z = 0; z < CK_ht->len; z++){
                    unsigned int pos = pos_nnz[z];
                    unsigned int tableidx = (pos & 0x8000000) >> 31;
                    unsigned int real_pos = pos & 0x7fffffff;
                    Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = CK_ht->key[tableidx][real_pos];
                    Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = CK_ht->val[tableidx][real_pos];
                }

                free(pos_nnz);
                CK_htFree(CK_ht);
                Z_tmp[tid].nnz += nnz_counter;
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_accumulate_z += sptElapsedTime(timer_SPA);
                }
            }else{// use LP
                unsigned int ht_size = std::min(std::max((int)ceil(log2(4*nnz_upper_bound)),2), 21);
                min_hta_size[tid] = std::min(ht_size, min_hta_size[tid]);
                max_hta_size[tid] = std::max(ht_size, max_hta_size[tid]);
                // sptNnzIndex nnz_upper_bound = 0;
                // unsigned int ht_size = 15;
                // unsigned int ht_size = ceil(log2((fidx_X.len-1)*(fx_end-fx_begin)))-2; // 2^13 = 8192
                // const unsigned int ht_size = 100;
                // hashtable_a_size = ht_size;
                unsigned int hashtable_a_size = (1 << ht_size);
                unsigned int* pos_nnz = (unsigned int *)malloc((hashtable_a_size)*sizeof(unsigned int));
                // const unsigned int ht_size = 9973;
                sptIndex nmodes_spa = nmodes_Y - num_cmodes;
                long int nnz_counter = 0;
                sptIndex current_idx = 0;
                // table_t *ht;
                // ht = htCreate(ht_size);
                LP_table_t *LP_ht;
                LP_ht = LP_htCreate(ht_size);

                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_prep += sptElapsedTime(timer_SPA);
                }

                for(unsigned int zX = 0; zX < X_len; zX++){ // loop nnzs in a subtensor X  
                    if (tid == 0) {
                        sptStartTimer(timer_SPA);
                    }       
                    sptValue valX = X_val.val[zX];
                    unsigned long long key_cmodes = X_val.key_FM[zX]; 

                    tensor_value Y_val = LP_tensor_htGet(Y_ht, key_cmodes);  

                    unsigned int my_len = Y_val.len;
                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_free_mode += sptElapsedTime(timer_SPA);
                    }
                    if(my_len == 0) continue;

                    if (tid == 0) {
                        sptStartTimer(timer_SPA);       
                    }        
                    
                    for(int i = 0; i < my_len; i++){
                        unsigned long long fmode =  Y_val.key_FM[i];
                        sptValue spa_val = LP_htGet(LP_ht, fmode);
                        float result = Y_val.val[i] * valX;

                        if(spa_val == LONG_MIN) {
                            pos_nnz[LP_ht->len] = LP_htInsert(LP_ht, fmode, result);
                            nnz_counter++;
                        }
                        else{
                            LP_htUpdate(LP_ht, fmode, spa_val + result);
                        }
                    }

                    if (tid == 0){
                        sptStopTimer(timer_SPA);
                        time_spa += sptElapsedTime(timer_SPA);
                    }
                    
                }   // End Loop nnzs inside a X fiber

                if (tid == 0) {
                    sptStartTimer(timer_SPA);    
                }
                Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM = (unsigned long long*)malloc(sizeof(unsigned long long)*nnz_counter);
                Z_ht->LP_node_list[fx_ptr].val_ptr.val = (sptValue*)malloc(sizeof(sptValue)*nnz_counter);
                Z_ht->LP_node_list[fx_ptr].val_ptr.len = nnz_counter;
                Z_ht->LP_node_list[fx_ptr].val_ptr.cap = nnz_counter;

                for(int z = 0; z < LP_ht->len; z++){
                    unsigned int pos = pos_nnz[z];
                    Z_ht->LP_node_list[fx_ptr].val_ptr.key_FM[z] = LP_ht->key[pos];
                    Z_ht->LP_node_list[fx_ptr].val_ptr.val[z] = LP_ht->val[pos];
                }
                // printf("bytes_HtA:%lu", bytes_HtA);
                Z_tmp[tid].nnz += nnz_counter;

                LP_htFree(LP_ht);
                free(pos_nnz);
                if (tid == 0){
                    sptStopTimer(timer_SPA);
                    time_accumulate_z += sptElapsedTime(timer_SPA);
                }    
            }
        }
    } // End loop of Buckts of HtX
    unsigned int Max_hta_size = 0, Min_hta_size = 64;
    for(sptIndex i = 0; i < tk; i++){
        Max_hta_size = std::max(Max_hta_size, max_hta_size[i]);
        Min_hta_size = std::min(Min_hta_size, min_hta_size[i]);
    }
    sptStopTimer(timer);
    double main_computation = sptElapsedTime(timer);
    total_time += main_computation;
    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
    printf("[SPATOTALTIME]: %.6f s\n", spa_total);
    printf("[MainComputat]: %.6f s\n", main_computation);
    printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
    printf("[TimeFreemode]: %.6f s\n", (time_free_mode)/spa_total * main_computation);
    printf("[Time    Prep]: %.6f s\n", (time_prep)/spa_total * main_computation);
    printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);
    printf("[Acc: SPApart]: %.6f s\n", (time_spa)/spa_total * main_computation);
    printf("[Acc: AccoutZ]: %.6f s\n", (time_accumulate_z)/spa_total * main_computation);
    printf("[HtA     size]: [%u, %u]\n", Min_hta_size, Max_hta_size);

    sptStartTimer(timer);
    /// Append Z_tmp to Z
    //Calculate the indecies of Z
    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
    unsigned long long Z_total_size = 0;

    unsigned long long Z_ht_total_size = 0;
    Z_tmp_start[0] = 0;
    for(int i = 0; i < tk; i++){
        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
        Z_total_size +=  Z_tmp[i].nnz;
        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
    }
    //printf("%d\n", Z_total_size);
    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size); 
    Z_ht->len = Z_total_size;
    sptNnzIndex bytes_HtZ;
    bytes_HtZ = 2*sizeof(unsigned int) + sizeof(sptNnzIndex) + Z_ht->size*sizeof(LP_tensor_node_t) + (Z_total_size)*(sizeof(unsigned long long) + sizeof(sptValue));
    printf("[Memorycost of HtZ]: %10.2f KiB %10.2f MiB %10.2f GiB\n", bytes_HtZ / 1024.0, bytes_HtZ / 1048576.0, bytes_HtZ / 1073741824.0);
    // sptNnzIndex pos_z = 0; 
    // for(sptNnzIndex i = 0; i < Z_ht_size; i++){
    //     tensor_node_t* temp = Z_ht->list[i];
    //     while(temp){
    //         Z_ht_total_size += temp->val.len;
    //         unsigned long long fmode_x = temp->key;
    //         tensor_value Z_Val = temp->val;
    //         for(unsigned int j = 0; j < Z_Val.len; j++){
    //             unsigned long long fmode_y = Z_Val.key_FM[j];
    //             sptValue ValZ = Z_Val.val[j];

    //             for(sptIndex m = 0; m < X_num_fmodes; m++){
    //                 Z->inds[m].data[pos_z] = (fmode_x % X_fmode_inds[m])/X_fmode_inds[m+1];
    //             }
    //             for(sptIndex m = 0; m < Y_num_fmodes; m++){
    //                 Z->inds[X_num_fmodes + m].data[pos_z] = (fmode_y % Y_fmode_inds[m])/Y_fmode_inds[m+1];
    //             }
    //             Z->values.data[pos_z] = ValZ;
    //             pos_z ++;
    //         }
    //         temp = temp->next;
    //     }
    // }
    // printf("[Z_ht total nnz]: %llu \n", Z_ht_total_size);

    // //  for(int i = 0; i < tk; i++)
    // //      sptFreeSparseTensor(&Z_tmp[i]);
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Writeback");

    // sptStartTimer(timer);
    // if(output_sorting == 1){
    //     sptSparseTensorSortIndex(Z, 1, tk);
    //     // sptSparseTensorSortIndexCmode(Z, 1, tk, nmodes_X - num_cmodes - 1, nmodes_Y - num_cmodes);
    // }
    // sptStopTimer(timer);
    // total_time += sptPrintElapsedTime(timer, "Output Sorting");
    // printf("[Output Sorting]: %.6f s\n", time_outsort);

    printf("[Total time]: %.6f s\n", total_time);
    // printf("[Num subtensors per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_subtensor_id[i]);
    // printf("\n");
    // printf("[Num of nonzero per thread]:");
    // for(sptIndex i = 0; i < tk; i++)
    //     printf("%10d", num_nnz_id[i]);
    printf("\n");
    printf("\n");
    
    // FILE *p_tz = fopen("tensor_tz_mode21.txt", "w");
    // sptDumpSparseTensor(Z, 0, p_tz);
    // fclose(p_tz);
  } 



  return 0;
}
