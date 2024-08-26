#ifndef DataFormats_CaloRecHit_interface_maxDef_h
#define DataFormats_CaloRecHit_interface_maxDef_h

namespace calo {
    namespace multifit {
#if __CUDA_ARCH__

        __device__ float atomicMaxFloat(float *addr, const float value) {
                    int *intAddr = (int *)addr;
                    int old = *intAddr, assumed;

                        do {
                            assumed = old;

                            old = atomicCAS(intAddr, assumed, __float_as_int(fmaxf(__int_as_float(assumed), value)));

                        } while (assumed != old);

                    return __int_as_float(old);
                    }
#endif

    } //multifit
} //calo

#endif