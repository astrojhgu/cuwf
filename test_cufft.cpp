#include "cufft_wrapper.h"
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    int nch = 8192;
    int npt = (480000000 / nch) * nch;
    int nbatch = npt / nch / 2;
    std::vector<fcomplex> h_input(nch * 2 * nbatch);

    for (int i = 0; i < h_input.size(); i++)
    {
        h_input[i].real = i % 32 > 16 ? -1 : 1;
        h_input[i].imag = 0;
    }

    FftResources *resources = init_resources(nch * 2, nbatch);

    for (int i = 0; i < 10; ++i)
    {
        std::cout<<i<<std::endl;
        fft_execute(resources, h_input.data(), h_input.data());
    }

    destroy_resources(resources);
}
