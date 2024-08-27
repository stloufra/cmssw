#include <cmath>
#include <limits>

#include <cuda.h>

#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"
#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include <cooperative_groups.h>

#include "AmplitudeComputationCommonKernels.h"
#include "AmplitudeComputationKernels.h"
#include "KernelHelpers.h"

namespace ecal {
    namespace multifit {
        namespace cg = cooperative_groups;

#define __SIZE_OF_TILE_MULTIFIT__ 4

        template<typename MatrixType, unsigned int TileSize>
        __device__ __forceinline__ void update_covariance_coop(EcalPulseCovariance const &pulse_covariance, //pulse cov
                                                               MatrixType &inverse_cov, // covMatrix/matrixLforfnnls
                                                               SampleVector const &amplitudes,
                                                               cg::thread_block_tile <TileSize> &tile) { //result amplitudes

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();


            constexpr int nsamples = SampleVector::RowsAtCompileTime; //10
            constexpr int npulses = BXVectorType::RowsAtCompileTime; //12 !!!

            //CMS_UNROLL_LOOP
            for (unsigned int ipulse = thrdIdx; ipulse < npulses; ipulse += numThrd) {
                auto const amplitude = amplitudes.coeff(ipulse);
                if (amplitude == 0)
                    continue;

                // FIXME: ipulse - 5 -> ipulse - firstOffset
                int bx = ipulse - 5;
                int first_sample_t = std::max(0, bx + 3);
                int offset = -3 - bx;

                auto const value_sq = amplitude * amplitude;

                for (int col = first_sample_t; col < nsamples; col++) {
                    for (int row = col; row < nsamples; row++) {
                        auto tmp = value_sq * __ldg(&pulse_covariance.covval[row + offset][col + offset]);
                        atomicAdd(&inverse_cov(row, col), tmp);
                        //printf("I am thread %d \n", thrdIdx); // all threads are here
                    }
                }
            }

            tile.sync();
        }

        template<typename MatrixType>
        __device__ __forceinline__ bool update_covariance(EcalPulseCovariance const &pulse_covariance,
                                                          MatrixType &inverse_cov,
                                                          SampleVector const &amplitudes) {
            constexpr int nsamples = SampleVector::RowsAtCompileTime;
            constexpr int npulses = BXVectorType::RowsAtCompileTime;

            CMS_UNROLL_LOOP
            for (unsigned int ipulse = 0; ipulse < npulses; ipulse++) {
                auto const amplitude = amplitudes.coeff(ipulse);
                if (amplitude == 0)
                    continue;

                // FIXME: ipulse - 5 -> ipulse - firstOffset
                int bx = ipulse - 5;
                int first_sample_t = std::max(0, bx + 3);
                int offset = -3 - bx;

                auto const value_sq = amplitude * amplitude;

                for (int col = first_sample_t; col < nsamples; col++) {
                    for (int row = col; row < nsamples; row++) {
                        inverse_cov(row, col) += value_sq * __ldg(&pulse_covariance.covval[row + offset][col + offset]);
                    }
                }
            }

            return true;
        }

        ///
        /// launch ctx parameters are (nchannels / block, blocks)
        /// TODO: trivial impl for now, there must be a way to improve
        ///
        /// Conventions:
        ///   - amplitudes -> solution vector, what we are fitting for
        ///   - samples -> raw detector responses
        ///   - passive constraint - satisfied constraint
        ///   - active constraint - unsatisfied (yet) constraint
        ///
        __global__ void kernel_minimize(uint32_t const *dids_eb,
                                        uint32_t const *dids_ee,
                                        SampleMatrix const *__restrict__ noisecov,
                                        EcalPulseCovariance const *__restrict__ pulse_covariance,
                                        BXVectorType *bxs,
                                        SampleVector const *__restrict__ samples,
                                        SampleVector *amplitudesEB,
                                        SampleVector *amplitudesEE,
                                        PulseMatrixType const *__restrict__ pulse_matrix,
                                        ::ecal::reco::StorageScalarType *chi2sEB,
                                        ::ecal::reco::StorageScalarType *chi2sEE,
                                        ::ecal::reco::StorageScalarType *energiesEB,
                                        ::ecal::reco::StorageScalarType *energiesEE,
                                        char *acState,
                                        int nchannels,
                                        int max_iterations,
                                        uint32_t const offsetForHashes,
                                        uint32_t const offsetForInputs) {
            // FIXME: ecal has 10 samples and 10 pulses....
            // but this needs to be properly treated and renamed everywhere
            constexpr auto NSAMPLES = SampleMatrix::RowsAtCompileTime;
            constexpr auto NPULSES = SampleMatrix::ColsAtCompileTime;
            static_assert(NSAMPLES == NPULSES);

            using DataType = SampleVector::Scalar;

            //cooperative group magic
            cg::thread_block block = cg::this_thread_block();
            cg::thread_block_tile<__SIZE_OF_TILE_MULTIFIT__> tile = cg::tiled_partition<__SIZE_OF_TILE_MULTIFIT__>(
                    block);

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();

            auto const tileIdx = tile.meta_group_rank();
            auto const numTile = tile.meta_group_size();

            int idx = tileIdx + numTile * block.group_index().x;


            //------------------SHARED MEMORY OPERATIONS --------------------------

            extern __shared__ char shrmem[];
            char *myPlace = shrmem;

            DataType *shrMatrixLForFnnlsStorage =
                    reinterpret_cast<DataType *>(myPlace) + calo::multifit::MapSymM<DataType, NPULSES>::total * tileIdx;
            myPlace += calo::multifit::MapSymM<DataType, NPULSES>::total * sizeof(DataType) * numTile;

            DataType *shrMatrixLStorage =
                    reinterpret_cast<DataType *>(myPlace) + calo::multifit::MapSymM<DataType, NPULSES>::total * tileIdx;
            myPlace += calo::multifit::MapSymM<DataType, NPULSES>::total * sizeof(DataType) * numTile;

            DataType *shrAtAStorage =
                    reinterpret_cast<DataType *>(myPlace) + calo::multifit::MapSymM<DataType, NPULSES>::total * tileIdx;
            myPlace += calo::multifit::MapSymM<DataType, NPULSES>::total * sizeof(DataType) * numTile;

            int *shrpulseOffsetsStorage = reinterpret_cast<int *>(myPlace) + NPULSES * tileIdx;
            myPlace += NPULSES * sizeof(int) * numTile;

            DataType *shrresultAmplitudesStorage = reinterpret_cast<DataType *>(myPlace) + NPULSES * tileIdx;
            myPlace += NPULSES * sizeof(DataType) * numTile;

            float *shrchi2Storage = reinterpret_cast<float *>(myPlace) + tileIdx;
            myPlace += sizeof(float) * numTile;

            float *shrchi2_nowStorage = reinterpret_cast<float *>(myPlace) + tileIdx;
            myPlace += sizeof(float) * numTile;

            float *shrsumsq2Storage = reinterpret_cast<float *>(myPlace) + tileIdx;
            myPlace += sizeof(float) * numTile;

            float *shrAStorage = reinterpret_cast<float *>(myPlace) + NPULSES * NSAMPLES * tileIdx;
            myPlace += NPULSES * NSAMPLES * sizeof(float) * numTile;

            float *shrreg_bStorage = reinterpret_cast<float *>(myPlace) + tileIdx * NSAMPLES;
            myPlace += NSAMPLES * sizeof(float) * numTile;

            float *shrreg_b_tmpStorage = reinterpret_cast<float *>(myPlace) + tileIdx * NSAMPLES;
            myPlace += NSAMPLES * sizeof(float) * numTile; //foward sub vector

            float *shrreg_LStorage = reinterpret_cast<float *>(myPlace) + tileIdx * NSAMPLES;
            myPlace += NSAMPLES * sizeof(float) * numTile; //foward sub vector

            DataType *shrAtbStorage = reinterpret_cast<DataType *>(myPlace) + NPULSES * tileIdx;
            myPlace += NPULSES * sizeof(DataType) * numTile;

            Eigen::Index* shrw_max_idxStorage = reinterpret_cast<Eigen::Index *>(myPlace) + tileIdx;
            myPlace += sizeof(Eigen::Index) * numTile;

            Eigen::Index* shrw_max_idx_prevStorage = reinterpret_cast<Eigen::Index *>(myPlace) + tileIdx;
            myPlace += sizeof(Eigen::Index) * numTile;

            float *shrw_maxStorage = reinterpret_cast<float *>(myPlace) + tileIdx;
            myPlace += sizeof(float) * numTile;

            float *shrw_max_prevStorage = reinterpret_cast<float *>(myPlace) + tileIdx;
            myPlace += sizeof(float) * numTile;

            bool *shrrecomputeStorage = reinterpret_cast<bool *>(myPlace) + tileIdx;
            myPlace += sizeof(bool) * numTile;

            int *shrnpassiveStorage = reinterpret_cast<int *>(myPlace) + tileIdx;
            myPlace +=  sizeof(int) * numTile;

            bool *shrhasNegativeStorage = reinterpret_cast<bool *>(myPlace) + tileIdx;
            myPlace += sizeof(bool) * numTile;

            bool *shrhasNansStorage = reinterpret_cast<bool *>(myPlace) + tileIdx;
            myPlace += sizeof(bool) * numTile;

            float *shrsFnnlsStorage = reinterpret_cast<float *>(myPlace) + tileIdx * NSAMPLES;
            myPlace += NSAMPLES * sizeof(float) * numTile;

            double *shrEpsStorage = reinterpret_cast<double *>(myPlace) + tileIdx;
            myPlace += sizeof(double) * numTile;

            //---------------VARIABLES DECLARATION------------------------
            float &chi2 = *shrchi2Storage;
            float &chi2_now = *shrchi2_nowStorage;
            float &sumsq2 = *shrsumsq2Storage;

            float *reg_b = shrreg_bStorage;
            float *reg_b_tmp = shrreg_b_tmpStorage;
            float *reg_L = shrreg_LStorage;

            float &w_max = *shrw_maxStorage;
            float &w_max_prev = *shrw_max_prevStorage;

            Eigen::Index& w_max_idx = *shrw_max_idxStorage;
            Eigen::Index& w_max_idx_prev = *shrw_max_idx_prevStorage;

            bool &recompute = *shrrecomputeStorage;
            bool &hasNegative = *shrhasNegativeStorage;
            bool &hasNans = *shrhasNansStorage;

            int &npassive = *shrnpassiveStorage;

            double& eps = *shrEpsStorage;

            Eigen::Map <calo::multifit::ColumnVector<NPULSES, int>> pulseOffsets(shrpulseOffsetsStorage);
            Eigen::Map <calo::multifit::ColumnVector<NPULSES, DataType>> resultAmplitudes(shrresultAmplitudesStorage);
            Eigen::Map <calo::multifit::ColMajorMatrix<NSAMPLES, NPULSES>> A(shrAStorage);
            Eigen::Map <ecal::multifit::SampleVector> Atb(shrAtbStorage);
            Eigen::Map <calo::multifit::ColumnVector<NPULSES, float>> sFnnls(shrsFnnlsStorage);


            DataType *covMatrixStorage = shrMatrixLForFnnlsStorage;
            calo::multifit::MapSymM <DataType, NSAMPLES> covMatrix{covMatrixStorage};
            calo::multifit::MapSymM <DataType, NPULSES> matrixLForFnnls{
                    shrMatrixLForFnnlsStorage}; //use same memory space to save memory
            calo::multifit::MapSymM <DataType, NSAMPLES> matrixL{shrMatrixLStorage};
            calo::multifit::MapSymM <DataType, NPULSES> AtA{shrAtAStorage};

            //SampleVector Atb;



// ref the right ptr
#define ARRANGE(var) auto* var = idx >= offsetForInputs ? var##EE : var##EB
            ARRANGE(amplitudes);
            ARRANGE(chi2s);
            ARRANGE(energies);
#undef ARRANGE

            if (idx < nchannels) {
                if (static_cast<MinimizationState>(acState[idx]) == MinimizationState::Precomputed)
                    return;

                // get the hash
                int const inputCh = idx >= offsetForInputs ? idx - offsetForInputs : idx;
                auto const *dids = idx >= offsetForInputs ? dids_ee : dids_eb;
                auto const did = DetId{dids[inputCh]};
                auto const isBarrel = did.subdetId() == EcalBarrel;
                auto const hashedId = isBarrel ? ecal::reconstruction::hashedIndexEB(did.rawId())
                                               : offsetForHashes + ecal::reconstruction::hashedIndexEE(did.rawId());

                npassive = 0;

                CMS_UNROLL_LOOP

                for (int i = thrdIdx; i < NPULSES; i += numThrd) {
                    pulseOffsets(i) = i;
                    resultAmplitudes(i) = 0;
                }

                tile.sync();

                for (int iter = 0; iter < max_iterations; iter++) {

                    //tile.sync();

                    CMS_UNROLL_LOOP
                    for (int col = thrdIdx; col < NSAMPLES; col += numThrd) {
                        CMS_UNROLL_LOOP
                        for (int row = col; row < NSAMPLES; row++) {
                            covMatrix(row, col) = __ldg(&noisecov[idx].coeffRef(row, col));
                        }
                    }

                    tile.sync();

                    update_covariance_coop(pulse_covariance[hashedId], covMatrix, resultAmplitudes, tile);


                    calo::multifit::compute_decomposition_unrolled_coop(matrixL, covMatrix, sumsq2, tile);

                    // L * A = P
                    calo::multifit::solve_forward_subst_matrix(A, pulse_matrix[idx], matrixL, tile);

                    //TODO: felt slower with coop intead of one - check
                    // L b = s
                    calo::multifit::solve_forward_subst_vector_coop(reg_b, reg_b_tmp, reg_L, samples[idx], matrixL,
                                                                    tile);


                    CMS_UNROLL_LOOP
                    for (int icol = thrdIdx; icol < NPULSES; icol += numThrd) {
                    //for (int icol = 0; icol < NPULSES; icol++) {

                        float reg_ai[NSAMPLES];

                        // load column icol
                        CMS_UNROLL_LOOP
                        for (int counter = 0; counter < NSAMPLES; counter++) {
                            reg_ai[counter] = A(counter, icol);
                        }

                        // compute diagonal
                        float sum = 0.f;
                        CMS_UNROLL_LOOP
                        for (int counter = 0; counter < NSAMPLES; counter++) {
                            sum += reg_ai[counter] * reg_ai[counter];
                        }

                        // store
                        AtA(icol, icol) = sum;

                        // go thru the other columns
                        CMS_UNROLL_LOOP
                        for (int j = icol + 1; j < NPULSES; j++) {
                            // load column j
                            float reg_aj[NSAMPLES];
                            CMS_UNROLL_LOOP
                            for (int counter = 0; counter < NSAMPLES; counter++) {
                                reg_aj[counter] = A(counter, j);
                            }

                            // accum
                            float sum = 0.f;
                            CMS_UNROLL_LOOP
                            for (int counter = 0; counter < NSAMPLES; counter++) {
                                sum += reg_aj[counter] * reg_ai[counter];
                            }

                            // store
                            //AtA(icol, j) = sum;
                            AtA(j, icol) = sum;
                        }


                        // Atb accum
                        float sum_atb = 0.f;
                        CMS_UNROLL_LOOP
                        for (int counter = 0; counter < NSAMPLES; counter++) {
                            sum_atb += reg_ai[counter] * reg_b[counter];
                        }

                        // store atb
                        Atb(icol) = sum_atb;
                    }

                    eps = 1e-11;

                    tile.sync();



                        calo::multifit::fnnls_coop(AtA,
                                                  Atb,
                                                  resultAmplitudes,
                                                  npassive,
                                                  pulseOffsets,
                                                  sFnnls,
                                                  matrixLForFnnls,
                                                  eps, //eps
                                                  500, //max iterations
                                                  16,  //relaxperiod
                                                  2,   //relax factor
                                                  w_max,
                                                  w_max_prev,
                                                  w_max_idx,
                                                  w_max_idx_prev,
                                                  recompute,
                                                  sumsq2,
                                                  hasNegative,
                                                  hasNans,
                                                  reg_b_tmp,
                                                  tile);

                    if (thrdIdx == 0) { //TODO: this is not done
                        calo::multifit::calculateChiSq(matrixL, pulse_matrix[idx], resultAmplitudes, samples[idx],
                                                       chi2_now);
                    }

                    tile.sync();

                    if (std::abs(chi2_now - chi2) < 1e-3) {
                        chi2 = chi2_now;
                        break;
                    }

                    chi2 = chi2_now;


                }

                tile.sync();

                // store to global output values
                // FIXME: amplitudes are used in global directly //only first thread
                chi2s[inputCh] = chi2; //same for the whole tile
                energies[inputCh] = resultAmplitudes(5);

                CMS_UNROLL_LOOP
                for (int i = thrdIdx; i < NPULSES; i += numThrd) {
                    amplitudes[inputCh](i) = resultAmplitudes(i);
                }

            }
        }

        namespace v1 {

            void minimization_procedure(EventInputDataGPU const &eventInputGPU,
                                        EventOutputDataGPU &eventOutputGPU,
                                        EventDataForScratchGPU &scratch,
                                        ConditionsProducts const &conditions,
                                        ConfigurationParameters const &configParameters,
                                        cudaStream_t cudaStream) {
                using DataType = SampleVector::Scalar;
                unsigned int totalChannels = eventInputGPU.ebDigis.size + eventInputGPU.eeDigis.size;
                //    unsigned int threads_min = conf.threads.x;
                // TODO: configure from python
                unsigned int threads_min = configParameters.kernelMinimizeThreads[0]; //should be 32

                //static_assert(threads_min == 32);

                unsigned int blocks_min =
                        threads_min > totalChannels ? __SIZE_OF_TILE_MULTIFIT__ :
                        (totalChannels + threads_min - 1) /
                        threads_min *
                        __SIZE_OF_TILE_MULTIFIT__;
                uint32_t const offsetForHashes = conditions.offsetForHashes;
                uint32_t const offsetForInputs = eventInputGPU.ebDigis.size;

                //      constexpr auto NSAMPLES = SampleMatrix::RowsAtCompileTime;
                //      constexpr auto NPULSES = SampleMatrix::ColsAtCompileTime;

                auto const nbytesShared = (threads_min *
                                           (calo::multifit::MapSymM<DataType, SampleVector::RowsAtCompileTime>::total *
                                            sizeof(DataType) //matrixLforFnnl
                                            +
                                            calo::multifit::MapSymM<DataType, SampleVector::RowsAtCompileTime>::total *
                                            sizeof(DataType) //matrixL
                                            +
                                            calo::multifit::MapSymM<DataType, SampleVector::RowsAtCompileTime>::total *
                                            sizeof(DataType) //AtA
                                            + SampleMatrix::ColsAtCompileTime * sizeof(int) //pulseOffset
                                            + SampleMatrix::ColsAtCompileTime * sizeof(DataType)//resultAmplitudes
                                            + sizeof(float) //chi2
                                            + sizeof(float) //chi2_now
                                            + sizeof(float) //sumsq2 - decompositon chol.
                                            + SampleMatrix::RowsAtCompileTime * SampleMatrix::ColsAtCompileTime *
                                              sizeof(float) //A
                                            + sizeof(float) * SampleMatrix::RowsAtCompileTime //reg_b
                                            + sizeof(float) * SampleMatrix::RowsAtCompileTime //reg_b_tmp
                                            + sizeof(float) * SampleMatrix::RowsAtCompileTime //reg_b_L
                                            + SampleVector::RowsAtCompileTime * sizeof(DataType) // Atb
                                            + sizeof(Eigen::Index) //w_max_idx
                                            + sizeof(Eigen::Index) //w_max_idx_prev
                                            + sizeof(float) //w_max
                                            + sizeof(float) //w_max_prev
                                            + sizeof(bool)  //recompute
                                            + sizeof(int)   //npassive
                                            + sizeof(float) * SampleMatrix::RowsAtCompileTime //s
                                            + sizeof(double) //eps
                                           )
                                           / __SIZE_OF_TILE_MULTIFIT__);

                kernel_minimize<<<blocks_min, threads_min, nbytesShared, cudaStream>>>(
                        eventInputGPU.ebDigis.ids.get(),
                        eventInputGPU.eeDigis.ids.get(),
                        (SampleMatrix *) scratch.noisecov.get(),
                        conditions.pulseCovariances.values,
                        (BXVectorType *) scratch.activeBXs.get(),
                        (SampleVector *) scratch.samples.get(),
                        (SampleVector *) eventOutputGPU.recHitsEB.amplitudesAll.get(),
                        (SampleVector *) eventOutputGPU.recHitsEE.amplitudesAll.get(),
                        (PulseMatrixType *) scratch.pulse_matrix.get(),
                        eventOutputGPU.recHitsEB.chi2.get(),
                        eventOutputGPU.recHitsEE.chi2.get(),
                        eventOutputGPU.recHitsEB.amplitude.get(),
                        eventOutputGPU.recHitsEE.amplitude.get(),
                        scratch.acState.get(),
                        totalChannels,
                        50,
                        offsetForHashes,
                        offsetForInputs);
                cudaCheck(cudaGetLastError());
            }

        }  // namespace v1
#undef __SIZE_OF_TILE_MULTIFIT__
    }  // namespace multifit
}  // namespace ecal
