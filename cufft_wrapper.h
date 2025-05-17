#ifndef DDC_H_
#define DDC_H_


typedef short int int16_t;

struct fcomplex
{
    float real;
    float imag;
};


struct FftResources;


#ifdef __cplusplus
extern "C"
{
#endif
struct FftResources * init_resources(int npt, int nbatch);

void destroy_resources(struct FftResources *resources);

void fft_execute(struct FftResources *resources, struct fcomplex *h_input, struct fcomplex *h_output);

void fft_execute_inverse(struct FftResources *resources, struct fcomplex *h_input, struct fcomplex *h_output);

#ifdef __cplusplus
}
#endif


#endif
// DDC_H_