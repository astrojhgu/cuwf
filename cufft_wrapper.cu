#include "cufft_wrapper.h"
#include <cstdio>
#include <cstdlib>
#include <cufft.h>


struct FftResources
{
    int npt;
    int nbatch;
    struct fcomplex *d_input;
    struct fcomplex *d_output;
    cufftHandle plan;
};

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// cuFFT 错误检查宏
#define CHECK_CUFFT(call)                                                        \
    do {                                                                         \
        cufftResult err = (call);                                                \
        if (err != CUFFT_SUCCESS) {                                              \
            fprintf(stderr, "cuFFT Error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cufftGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)


const char* cufftGetErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "Unknown CUFFT error";
    }
}

extern "C" FftResources* init_resources(int npt, int nbatch) {
    FftResources *resources=new FftResources;
    resources->npt = npt;
    resources->nbatch = nbatch;
    CHECK_CUFFT(cufftPlan1d(&resources->plan, npt, CUFFT_C2C, nbatch));
    CHECK_CUDA(cudaMalloc((void**)&resources->d_input, sizeof(fcomplex) * npt*nbatch));
    CHECK_CUDA(cudaMalloc((void**)&resources->d_output, sizeof(fcomplex) * npt*nbatch));
    return resources;
}

extern "C" void destroy_resources(FftResources *resources) {
    CHECK_CUFFT(cufftDestroy(resources->plan));
    CHECK_CUDA(cudaFree(resources->d_input));
    CHECK_CUDA(cudaFree(resources->d_output));
    delete resources;
}

extern "C" void fft_execute(FftResources *resources, fcomplex *h_input, fcomplex *h_output) {
    CHECK_CUDA(cudaMemcpy(resources->d_input, h_input, sizeof(fcomplex) * resources->npt*resources->nbatch, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecC2C(resources->plan, (cufftComplex*)resources->d_input, (cufftComplex*)resources->d_output, CUFFT_FORWARD));
    CHECK_CUDA(cudaMemcpy(h_output, resources->d_output, sizeof(fcomplex) * resources->npt*resources->nbatch, cudaMemcpyDeviceToHost));
}

extern "C" void fft_execute_inverse(FftResources *resources, fcomplex *h_input, fcomplex *h_output) {
    CHECK_CUDA(cudaMemcpy(resources->d_input, h_input, sizeof(cufftComplex) * resources->npt*resources->nbatch, cudaMemcpyHostToDevice));
    CHECK_CUFFT(cufftExecC2C(resources->plan, (cufftComplex*)resources->d_input, (cufftComplex*)resources->d_output, CUFFT_INVERSE));
    CHECK_CUDA(cudaMemcpy(h_output, resources->d_output, sizeof(cufftComplex) * resources->npt*resources->nbatch, cudaMemcpyDeviceToHost));
}
