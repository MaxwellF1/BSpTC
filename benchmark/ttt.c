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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ParTI.h>
#include <assert.h>

void print_usage(char ** argv) {
    printf("Usage: %s [options] \n\n", argv[0]);
    printf("Options: -X FIRST INPUT TENSOR\n");
    printf("         -Y FIRST INPUT TENSOR\n");
    printf("         -Z OUTPUT TENSOR (Optinal)\n");
    printf("         -m NUMBER OF CONTRACT MODES\n");
    printf("         -x CONTRACT MODES FOR TENSOR X (0-based)\n");
    printf("         -y CONTRACT MODES FOR TENSOR Y (0-based)\n");
    printf("         -t NTHREADS, --nt=NT (Optinal)\n");
    printf("         --help\n");
    printf("\n");
}

int main(int argc, char *argv[]) {

    char Xfname[1000], Yfname[1000];
    FILE *fZ = NULL;
    sptSparseTensor X, Y, Z;
    sptIndex * cmodes_X = NULL, * cmodes_Y = NULL;
    sptIndex num_cmodes = 1;
    int cuda_dev_id = -2;
    int output_sorting=1;
    int niters = 5;
    int placement = 0;
    int nt = 1;

    if(argc < 3) {
        print_usage(argv);
        exit(-1);
    }

    static struct option long_options[] = {
        {"X", required_argument, 0, 'X'},
        {"Y", required_argument, 0, 'Y'},
        {"mode", required_argument, 0, 'm'},
        {"x", required_argument, 0, 'x'},
        {"y", required_argument, 0, 'y'},
        {"Z", optional_argument, 0, 'Z'},
        {"o", optional_argument, 0, 'o'},
        {"p", optional_argument, 0, 'p'},
        {"cuda-dev-id", optional_argument, 0, 'd'},
        {"nt", optional_argument, 0, 't'},
        {"help", no_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int c;
    for(;;) {
        int option_index = 0;
        c = getopt_long(argc, argv, "X:Y:m:x:y:o:p:Z:d:t:", long_options, &option_index);
        if(c == -1) {
            break;
        }
        switch(c) {
        case 'X':
            strcpy(Xfname, optarg);
            printf("1st tensor file: %s\n", Xfname);
            break;
        case 'Y':
            strcpy(Yfname, optarg);
            printf("2nd tensor file: %s\n", Yfname);
            break;
        case 'Z':
            fZ = fopen(optarg, "w");
            sptAssert(fZ != NULL);
            printf("output tensor file: %s\n", optarg);
            break;
        case 'm':
            sscanf(optarg, "%"PARTI_SCN_INDEX, &num_cmodes);
            cmodes_X = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
            cmodes_Y = (sptIndex*)malloc(num_cmodes * sizeof(sptIndex));
            sptAssert(cmodes_X != NULL && cmodes_Y != NULL);
            printf("%s\n", optarg);
            break;
        case 'x':
            if(PARTI_INDEX_TYPEWIDTH == 32){
                for(sptIndex i = 0; i < num_cmodes; ++ i) {
                    // Only can input an array directly from argv not optarg.
                    sscanf(argv[optind - 1], "%u", &(cmodes_X[i])); 
                    ++ optind;
                }
                optind -= num_cmodes;
                break; 
            }
            if(PARTI_INDEX_TYPEWIDTH == 64){
                for(sptIndex i = 0; i < num_cmodes; ++ i) {
                    // Only can input an array directly from argv not optarg.
                    sscanf(argv[optind - 1], "%lu", &(cmodes_X[i])); 
                    ++ optind;
                }
                optind -= num_cmodes;
                break; 
            }
            printf("ERROR: Faild to read tensor X\n");
            break;
        case 'y':
            if(PARTI_INDEX_TYPEWIDTH == 32){
                for(sptIndex i = 0; i < num_cmodes; ++ i) {
                    // Only can input an array directly from argv not optarg.
                    sscanf(argv[optind - 1], "%u", &(cmodes_Y[i])); 
                    ++ optind;
                }
                optind -= num_cmodes;
                break; 
            }
            if(PARTI_INDEX_TYPEWIDTH == 64){
                for(sptIndex i = 0; i < num_cmodes; ++ i) {
                    // Only can input an array directly from argv not optarg.
                    sscanf(argv[optind - 1], "%lu", &(cmodes_Y[i])); 
                    ++ optind;
                }
                optind -= num_cmodes;
                break; 
            }
            printf("ERROR: Faild to read tensor Y\n");
            break;
        case 'o':
            sscanf(optarg, "%d", &output_sorting);
            break;    
        case 'p':
            sscanf(optarg, "%d", &placement);
            break;      
        case 'd':
            sscanf(optarg, "%d", &cuda_dev_id);
            break;
        case 't':
            sscanf(optarg, "%d", &nt);
            break;
        case '?':   /* invalid option */
        case 'h':
        default:
            print_usage(argv);
            exit(1);
        }
    }
    omp_set_num_threads(nt);
    printf("#Contraction modes: %"PARTI_PRI_INDEX"\n", num_cmodes);
    //sptDumpIndexArray(cmodes_X, num_cmodes, stdout);
    //sptDumpIndexArray(cmodes_Y, num_cmodes, stdout);

    int experiment_modes;
    sscanf(getenv("EXPERIMENT_MODES"), "%d", &experiment_modes);
    if(experiment_modes == 999){
        int dram_node;
        sscanf(getenv("DRAM_NODE"), "%d", &dram_node);
        int optane_node;
        sscanf(getenv("OPTANE_NODE"), "%d", &optane_node);
        int numa_node = dram_node;
        if(placement==1){
            sptAssert(sptLoadSparseTensorNuma(&X, 1, Xfname, optane_node) == 0);
            sptAssert(sptLoadSparseTensorNuma(&Y, 1, Yfname, dram_node) == 0);
        }
        else if (placement==2){
            sptAssert(sptLoadSparseTensorNuma(&X, 1, Xfname, dram_node) == 0);
            sptAssert(sptLoadSparseTensorNuma(&Y, 1, Yfname, optane_node) == 0); 
        }
        else{
            sptAssert(sptLoadSparseTensorNuma(&X, 1, Xfname, optane_node) == 0);
            sptAssert(sptLoadSparseTensorNuma(&Y, 1, Yfname, optane_node) == 0);
        }
        //sptAssert(sptLoadSparseTensor(&X, 1, Xfname) == 0);
        //sptAssert(sptLoadSparseTensor(&Y, 1, Yfname) == 0);
        sptSparseTensorStatus(&X, stdout);
        sptSparseTensorStatus(&Y, stdout);   
    }
    else{
    sptAssert(sptLoadSparseTensor(&X, 1, Xfname) == 0);
    sptSparseTensorStatus(&X, stdout);

    // FILE *p_tz = fopen("nips.txt", "w");
    // sptDumpSparseTensor(&X, 0, p_tz);
    // fclose(p_tz);
    
    sptAssert(sptLoadSparseTensor(&Y, 1, Yfname) == 0);
    sptSparseTensorStatus(&Y, stdout);   
    }
    //printf("Original Tensors: \n"); 
    //sptAssert(sptDumpSparseTensor(&X, 0, stdout) == 0);
    //sptAssert(sptDumpSparseTensor(&Y, 0, stdout) == 0);  

    X.stidx = (int*)malloc(sizeof(int)*X.nnz); 
    Y.stidx = (int*)malloc(sizeof(int)*Y.nnz); 

    /* For warm-up caches, timing not included */
    if(cuda_dev_id == -2) {     
            sptAssert(sptSparseTensorMulTensor(&Z, &X, &Y, num_cmodes, cmodes_X, cmodes_Y, nt, output_sorting, placement) == 0);
    } else if(cuda_dev_id == -1) {
        // sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    }

    // for(int it=0; it<niters; ++it) {
    //     sptFreeSparseTensor(&Z);
    //     if(cuda_dev_id == -2) {
    //         // sptAssert(sptSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    //     } else if(cuda_dev_id == -1) {
    //         // sptAssert(sptOmpSparseTensorMulMatrix(&Y, &X, &U, mode) == 0);
    //     }
    // }

    sptSparseTensorStatus(&Z, stdout);
    //sptAssert(sptDumpSparseTensor(&Z, 0, stdout) == 0);

    if(fZ != NULL) {
        sptSparseTensorSortIndex(&Z, 1, 1);
        //sptAssert(sptDumpSparseTensor(&Z, 0, fZ) == 0);
        fclose(fZ);
    }

    //sptFreeSparseTensor(&Y);
    //sptFreeSparseTensor(&X);
    //sptFreeSparseTensor(&Z);
    // test
    // sptSparseTensor X_tmp, Y_tmp, Z_tmp;

    // X_tmp.nmodes = 4;
    // Y_tmp.nmodes = 4;
    // Z_tmp.nmodes = 4;

    // X_tmp.sortorder = malloc(4*sizeof(sptIndex));
    // Y_tmp.sortorder = malloc(4*sizeof(sptIndex));
    // Z_tmp.sortorder = malloc(4*sizeof(sptIndex));

    // X_tmp.ndims = malloc(4*sizeof(sptIndex));
    // Y_tmp.ndims = malloc(4*sizeof(sptIndex));
    // Z_tmp.ndims = malloc(4*sizeof(sptIndex));
    // for(sptIndex dim = 0; dim < 4; dim++){
    //     X_tmp.sortorder[dim] = dim;
    //     X_tmp.ndims[dim] = 6;
    // }
    // for(sptIndex dim = 0; dim < 4; dim++){
    //     Y_tmp.sortorder[dim] = dim;
    //     Y_tmp.ndims[dim] = 6;
    // }
    // for(sptIndex dim = 0; dim < 4; dim++){
    //     Z_tmp.sortorder[dim] = dim;
    //     Z_tmp.ndims[dim] = 6;
    // }

    // X_tmp.nnz = 3;
    // Y_tmp.nnz = 4;

    // X_tmp.inds = malloc(4 * sizeof(sptIndexVector));
    // for(sptIndex m = 0; m < 4; ++m)
    //     sptNewIndexVector(&X_tmp.inds[m], X_tmp.nnz, X_tmp.nnz);
    // X_tmp.inds[0].data[0] = 0;
    // X_tmp.inds[1].data[0] = 0;
    // X_tmp.inds[2].data[0] = 0;
    // X_tmp.inds[3].data[0] = 1;
    // X_tmp.inds[0].data[1] = 0;
    // X_tmp.inds[1].data[1] = 0;
    // X_tmp.inds[2].data[1] = 0;
    // X_tmp.inds[3].data[1] = 2;
    // X_tmp.inds[0].data[2] = 0;
    // X_tmp.inds[1].data[2] = 1;
    // X_tmp.inds[2].data[2] = 0;
    // X_tmp.inds[3].data[2] = 0;
    // sptNewValueVector(&X_tmp.values, X_tmp.nnz, X_tmp.nnz);
    // X_tmp.values.data[0] = 1;
    // X_tmp.values.data[1] = 2;
    // X_tmp.values.data[2] = 3;
    

    // Y_tmp.inds = malloc(4 * sizeof(sptIndexVector));
    // for(sptIndex m = 0; m < 4; ++m)
    //     sptNewIndexVector(&Y_tmp.inds[m], Y_tmp.nnz, Y_tmp.nnz);
    // Y_tmp.inds[0].data[0] = 0;
    // Y_tmp.inds[1].data[0] = 3;
    // Y_tmp.inds[2].data[0] = 0;
    // Y_tmp.inds[3].data[0] = 0;
    // Y_tmp.inds[0].data[1] = 0;
    // Y_tmp.inds[1].data[1] = 4;
    // Y_tmp.inds[2].data[1] = 0;
    // Y_tmp.inds[3].data[1] = 1;
    // Y_tmp.inds[0].data[2] = 0;
    // Y_tmp.inds[1].data[2] = 4;
    // Y_tmp.inds[2].data[2] = 0;
    // Y_tmp.inds[3].data[2] = 2;
    // Y_tmp.inds[0].data[3] = 0;
    // Y_tmp.inds[1].data[3] = 5;
    // Y_tmp.inds[2].data[3] = 0;
    // Y_tmp.inds[3].data[3] = 1;
    // sptNewValueVector(&Y_tmp.values, Y_tmp.nnz, Y_tmp.nnz);
    // Y_tmp.values.data[0] = 4;
    // Y_tmp.values.data[1] = 5;
    // Y_tmp.values.data[2] = 7;
    // Y_tmp.values.data[3] = 6;
    // sptIndex my_cmodes_X[2]={2,3};
    // sptIndex my_cmodes_Y[2]={2,3};
    // sptIndex my_num_cmodes = 2;

    // sscanf(getenv("EXPERIMENT_MODES"), "%d", &experiment_modes);
    // sptSparseTensorStatus(&X_tmp, stdout);
    // sptSparseTensorStatus(&Y_tmp, stdout);
    // // sptBlockSparseTensorMulTensor(&Z_tmp, &X_tmp, &Y_tmp, my_num_cmodes, my_cmodes_X, my_cmodes_Y, nt, output_sorting, placement, 3);
    // if(experiment_modes == 6)
    //     sptSparseTensorMulTensor(&Z_tmp, &X_tmp, &Y_tmp, my_num_cmodes, my_cmodes_X, my_cmodes_Y, nt, output_sorting, placement);
    // sptSparseTensorStatus(&Z_tmp, stdout);

    // FILE *py = fopen("my_testsor.txt", "w");
    // sptDumpSparseTensor(&Z_tmp, 0, py);
    // fclose(py);    
    // return 0;
}
