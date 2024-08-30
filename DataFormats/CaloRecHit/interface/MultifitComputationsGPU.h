#ifndef DataFormats_CaloRecHit_interface_MultifitComputations_h
#define DataFormats_CaloRecHit_interface_MultifitComputations_h

#include <cmath>
#include <limits>
#include <type_traits>
#include <cooperative_groups.h>


#include <Eigen/Dense>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"


namespace calo {
    namespace multifit {
        namespace cg = cooperative_groups;

        template<int NROWS, int NCOLS>
        using ColMajorMatrix = Eigen::Matrix<float, NROWS, NCOLS, Eigen::ColMajor>;

        template<int NROWS, int NCOLS>
        using RowMajorMatrix = Eigen::Matrix<float, NROWS, NCOLS, Eigen::RowMajor>;

        template<int SIZE, typename T = float>
        using ColumnVector = Eigen::Matrix<T, SIZE, 1>;

        template<int SIZE, typename T = float>
        using RowVector = Eigen::Matrix<T, 1, SIZE>;

        // FIXME: provide specialization for Row Major layout
        // triangular layout
        template<typename T, int Stride, int Order = Eigen::ColMajor>
        struct MapSymM {
            using type = T;
            using base_type = typename std::remove_const<type>::type;

            static constexpr int total = Stride * (Stride + 1) / 2;
            static constexpr int stride = Stride;
            T *_data;

            EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

            MapSymM(T *data) : _data{data} {}

            EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

            T const &operator()(int const row, int const col) const {
                auto const tmp = (Stride - col) * (Stride - col + 1) / 2;
                auto const index = total - tmp + row - col;
                return _data[index];
            }

            template<typename U = T>
            EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

            typename std::enable_if<std::is_same<base_type, U>::value, base_type>::type &
            operator()(int const row, int const col) {
                auto const tmp = (Stride - col) * (Stride - col + 1) / 2;
                auto const index = total - tmp + row - col;
                return _data[index];
            }
        };

        // FIXME: either use/modify/improve eigen or make this more generic
        // this is a map for a pulse matrix to building a 2d matrix for each channel
        // and hide indexing
        template<typename T>
        struct MapMForPM {
            using type = T;
            using base_type = typename std::remove_cv<type>::type;

            type *data;
            EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

            MapMForPM(type *data) : data{data} {}

            EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

            base_type operator()(int const row, int const col) const {
                auto const index = 2 - col + row;
                return index >= 0 ? data[index] : 0;
            }
        };

        // simple/trivial cholesky decomposition impl
        template<typename MatrixType1, typename MatrixType2, typename SUM, unsigned int TileSize>
        EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

        void compute_decomposition_unrolled_coop(MatrixType1 &L,
                                                 MatrixType2 const &M,
                                                 SUM &sumsq2,
                                                 cg::thread_block_tile <TileSize> &tile) {
            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();


            auto const sqrtm_0_0 = std::sqrt(M(0, 0));
            L(0, 0) = sqrtm_0_0;
            using T = typename MatrixType1::base_type;

            CMS_UNROLL_LOOP
            for (int i = 1; i < MatrixType1::stride; i++) {
                T sumsq{0};
                for (int j = 0; j < i; j++) {
                    sumsq2 = 0;
                    for (int k = thrdIdx; k < j; k += numThrd) {
                        atomicAdd(&sumsq2, L(i, k) * L(j, k));
                    }

                    tile.sync();

                    auto const m_i_j = M(i, j);
                    auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
                    L(i, j) = value_i_j;

                    if (thrdIdx == 0) {
                        sumsq += value_i_j * value_i_j;
                    }
                }

                if (thrdIdx == 0) {
                    auto const l_i_i = std::sqrt(M(i, i) - sumsq);
                    L(i, i) = l_i_i;
                }
            }

            tile.sync();
        }

        // simple/trivial cholesky decomposition impl
        template<typename MatrixType1, typename MatrixType2>
        EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

        void compute_decomposition_unrolled_legacy(MatrixType1 &L,
                                                   MatrixType2 const &M) {

            auto const sqrtm_0_0 = std::sqrt(M(0, 0));
            L(0, 0) = sqrtm_0_0;
            using T = typename MatrixType1::base_type;

            CMS_UNROLL_LOOP
            for (int i = 1; i < MatrixType1::stride; i++) {
                T sumsq{0};
                for (int j = 0; j < i; j++) {
                    T sumsq2{0};
                    auto const m_i_j = M(i, j);
                    for (int k = 0; k < j; ++k)
                        sumsq2 += L(i, k) * L(j, k);

                    auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
                    L(i, j) = value_i_j;

                    sumsq += value_i_j * value_i_j;
                }

                auto const l_i_i = std::sqrt(M(i, i) - sumsq);
                L(i, i) = l_i_i;
            }


        }

        template<typename MatrixType1, typename MatrixType2, unsigned int TileSize>
        EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

        void compute_decomposition_coop(MatrixType1 &L,
                                        MatrixType2 const &M,
                                        cg::thread_block_tile <TileSize> &tile) {

            if (tile.thread_rank() == 0) {

                auto const sqrtm_0_0 = std::sqrt(M(0, 0));
                L(0, 0) = sqrtm_0_0;
                using T = typename MatrixType1::base_type;

                for (int i = 1; i < MatrixType1::stride; i++) {
                    T sumsq{0};
                    for (int j = 0; j < i; j++) {
                        T sumsq2{0};
                        auto const m_i_j = M(i, j);
                        for (int k = 0; k < j; ++k)
                            sumsq2 += L(i, k) * L(j, k);

                        auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
                        L(i, j) = value_i_j;

                        sumsq += value_i_j * value_i_j;
                    }

                    auto const l_i_i = std::sqrt(M(i, i) - sumsq);
                    L(i, i) = l_i_i;
                }
            }
        }

        template<typename MatrixType1, typename MatrixType2, typename VectorType, unsigned int TileSize>//, unsigned int TileSize>
        EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

        void compute_decomposition_forwardsubst_with_offsets(
                MatrixType1 &L, //matrixL
                MatrixType2 const &M, //AtA
                float b[MatrixType1::stride], //reg_b
                VectorType const &Atb, //Atb
                int const N, //npasssive
                ColumnVector<MatrixType1::stride, int> const &pulseOffsets,
                float &sumsq2,
                cg::thread_block_tile <TileSize> &tile) {

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();

            sumsq2 = 0;

            auto const real_0 = pulseOffsets(0);
            auto const sqrtm_0_0 = std::sqrt(M(real_0, real_0));
            L(0, 0) = sqrtm_0_0;
            using T = typename MatrixType1::base_type;
            b[0] = Atb(real_0) / sqrtm_0_0;


            for (int i = 1; i < N; i++) { //for (int i = idx + 1; i < N; i+=tile.num_threads()) {
                auto const i_real = pulseOffsets(i);
                T sumsq{0};
                T total = 0;
                auto const atb = Atb(i_real);
                for (int j = 0; j < i; j++) {

                    auto const j_real = pulseOffsets(j);
                    auto const m_i_j = M(std::max(i_real, j_real), std::min(i_real, j_real));

                    for (int k = thrdIdx; k < j; k += numThrd) {
                        atomicAdd(&sumsq2, L(i, k) * L(j, k));
                    }

                    tile.sync();


                    auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
                    L(i, j) = value_i_j;

                    sumsq += value_i_j * value_i_j;
                    total += value_i_j * b[j];

                    tile.sync();

                    sumsq2 = 0;
                }

                tile.sync();


                auto const l_i_i = std::sqrt(M(i_real, i_real) - sumsq);
                L(i, i) = l_i_i;
                b[i] = (atb - total) / l_i_i;
            }


        }

        template<typename MatrixType1, typename MatrixType2, typename VectorType, unsigned int TileSize>
        EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC

        void update_decomposition_forwardsubst_with_offsets(
                MatrixType1 &L,
                MatrixType2 const &M,
                float b[MatrixType1::stride],
                VectorType const &Atb,
                int const N,
                ColumnVector<MatrixType1::stride, int> const &pulseOffsets,
                float &sumsq2,
                cg::thread_block_tile <TileSize> &tile) {

            sumsq2 = 0;

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();

            using T = typename MatrixType1::base_type;
            auto const i = N - 1;
            auto const i_real = pulseOffsets(i);
            T sumsq{0};
            T total = 0;
            for (int j = 0; j < i; j++) {
                auto const j_real = pulseOffsets(j);
                auto const m_i_j = M(std::max(i_real, j_real), std::min(i_real, j_real));
                for (int k = thrdIdx; k < j; k += numThrd) {
                    atomicAdd(&sumsq2, L(i, k) * L(j, k));
                }

                tile.sync();

                auto const value_i_j = (m_i_j - sumsq2) / L(j, j);
                L(i, j) = value_i_j;
                sumsq += value_i_j * value_i_j;

                total += value_i_j * b[j];

                tile.sync();

                sumsq2 = 0;

            }

            tile.sync();

            auto const l_i_i = std::sqrt(M(i_real, i_real) - sumsq);
            L(i, i) = l_i_i;
            b[i] = (Atb(i_real) - total) / l_i_i;

            tile.sync();
        }

        template<typename MatrixType1, typename MatrixType2, typename MatrixType3, unsigned int TileSize>
        EIGEN_DEVICE_FUNC void solve_forward_subst_matrix(MatrixType1 &A,
                                                          MatrixType2 const &pulseMatrixView,
                                                          MatrixType3 const &matrixL,
                                                          cg::thread_block_tile <TileSize> &tile) {
            // FIXME: this assumes pulses are on columns and samples on rows

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();

            // TODO: TRY different approaches
            constexpr auto NPULSES = MatrixType2::ColsAtCompileTime;
            constexpr auto NSAMPLES = MatrixType2::RowsAtCompileTime;

            CMS_UNROLL_LOOP
            for (int icol = thrdIdx; icol < NPULSES; icol += numThrd) {
                float reg_b[NSAMPLES];
                float reg_L[NSAMPLES];

                // preload a column and load column 0 of cholesky
                CMS_UNROLL_LOOP
                for (int i = 0; i < NSAMPLES; i++) {
#ifdef __CUDA_ARCH__
                    // load through the read-only cache
          reg_b[i] = __ldg(&pulseMatrixView.coeffRef(i, icol));
#else
                    reg_b[i] = pulseMatrixView.coeffRef(i, icol);
#endif  // __CUDA_ARCH__
                    reg_L[i] = matrixL(i, 0);
                }

                // compute x0 and store it
                auto x_prev = reg_b[0] / reg_L[0];
                A(0, icol) = x_prev;

                // iterate
                CMS_UNROLL_LOOP
                for (int iL = 1; iL < NSAMPLES; iL++) {
                    // update accum
                    CMS_UNROLL_LOOP
                    for (int counter = iL; counter < NSAMPLES; counter++)
                        reg_b[counter] -= x_prev * reg_L[counter];

                    // load the next column of cholesky
                    CMS_UNROLL_LOOP
                    for (int counter = iL; counter < NSAMPLES; counter++)
                        reg_L[counter] = matrixL(counter, iL);

                    // compute the next x for M(iL, icol)
                    x_prev = reg_b[iL] / reg_L[iL];

                    // store the result value
                    A(iL, icol) = x_prev;
                }
            }
        }

        template<typename MatrixType1, typename MatrixType2, unsigned int TileSize>
        EIGEN_DEVICE_FUNC __device__

        void solve_forward_subst_vector_coop(float *reg_b,
                                             float *reg_b_tmp,
                                             float *reg_L,
                                             MatrixType1 inputAmplitudesView,
                                             MatrixType2 matrixL,
                                             cg::thread_block_tile <TileSize> &tile) {
            constexpr auto NSAMPLES = MatrixType1::RowsAtCompileTime; //10

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();

            // preload a column and load column 0 of cholesky
            CMS_UNROLL_LOOP
            for (int i = thrdIdx; i < NSAMPLES; i += numThrd) {
                reg_b_tmp[i] = inputAmplitudesView(i);
                reg_L[i] = matrixL(i, 0);
            }

            tile.sync();

            // compute x0 and store it
            auto x_prev = reg_b_tmp[0] / reg_L[0];
            reg_b[0] = x_prev;

            // iterate
            CMS_UNROLL_LOOP
            for (int iL = 1; iL < NSAMPLES; iL++) {
                // update accum
                CMS_UNROLL_LOOP
                for (int counter = iL + thrdIdx; counter < NSAMPLES; counter += numThrd) {
                    auto tmp = -1 * x_prev * reg_L[counter];
#ifdef __CUDA_ARCH__
                    atomicAdd(&reg_b_tmp[counter],tmp);
#endif
                    reg_L[counter] = matrixL(counter, iL);
                }

                tile.sync();


                // compute the next x for M(iL, icol)
                x_prev = reg_b_tmp[iL] / reg_L[iL];

                // store the result value
                reg_b[iL] = x_prev;

                tile.sync();
            }
        }

        template<typename MatrixType1, typename MatrixType2>
        EIGEN_DEVICE_FUNC void solve_forward_subst_vector(float reg_b[MatrixType1::RowsAtCompileTime],
                                                          MatrixType1 inputAmplitudesView,
                                                          MatrixType2 matrixL) {
            constexpr auto NSAMPLES = MatrixType1::RowsAtCompileTime;

            float reg_b_tmp[NSAMPLES];
            float reg_L[NSAMPLES];

            // preload a column and load column 0 of cholesky
            CMS_UNROLL_LOOP
            for (int i = 0; i < NSAMPLES; i++) {
                reg_b_tmp[i] = inputAmplitudesView(i);
                reg_L[i] = matrixL(i, 0);
            }

            // compute x0 and store it
            auto x_prev = reg_b_tmp[0] / reg_L[0];
            reg_b[0] = x_prev;

            // iterate
            CMS_UNROLL_LOOP
            for (int iL = 1; iL < NSAMPLES; iL++) {
                // update accum
                CMS_UNROLL_LOOP
                for (int counter = iL; counter < NSAMPLES; counter++)
                    reg_b_tmp[counter] -= x_prev * reg_L[counter];

                // load the next column of cholesky
                CMS_UNROLL_LOOP
                for (int counter = iL; counter < NSAMPLES; counter++)
                    reg_L[counter] = matrixL(counter, iL);

                // compute the next x for M(iL, icol)
                x_prev = reg_b_tmp[iL] / reg_L[iL];

                // store the result value
                reg_b[iL] = x_prev;
            }
        }


        template<typename MatrixType1, typename MatrixType2, typename MatrixType3, typename MatrixType4, unsigned int TileSize>
        EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC
        void calculateChiSq(MatrixType1 const &matrixL, //matrixL -shared
                            MatrixType2 const &pulseMatrixView, //pulseMatrix
                            MatrixType3 const &resultAmplitudesVector, //resultAmp -shared
                            MatrixType4 const &inputAmplitudesView, //samples
                            float &chi2, //shared
                            float *accum, //shared
                            float *results, //shared
                            float *reg_L, //shared
                            cg::thread_block_tile <TileSize> &tile
        ) {
            // FIXME: this assumes pulses are on columns and samples on rows
            constexpr auto NPULSES = MatrixType2::ColsAtCompileTime; //10
            constexpr auto NSAMPLES = MatrixType2::RowsAtCompileTime; //10

            static_assert(NPULSES == 10, "NPULSES must be 10. Not intended use");
            static_assert(NSAMPLES == 10, "NSAMPLES must be 10. Not intended use");

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();



            // replace pulseMatrixView * resultAmplitudesVector - inputAmplitudesView
            // NOTE:
            //float accum[NSAMPLES];
            {
                //float results[NPULSES];

                // preload results and permute according to the pulse offsets /////////////// ??? this is not done in ECAL
                CMS_UNROLL_LOOP
                for (int counter = thrdIdx; counter < NPULSES; counter += numThrd) {
                    results[counter] = resultAmplitudesVector[counter];
                }

                // load accum
                CMS_UNROLL_LOOP
                for (int counter = thrdIdx; counter < NPULSES; counter += numThrd) {
                    accum[counter] = -inputAmplitudesView(counter);
                }



                // iterate
                for (int icol = thrdIdx; icol < NPULSES; icol += numThrd) {
                    float pm_col[NSAMPLES];

                    // preload a column of pulse matrix
                    CMS_UNROLL_LOOP
                    for (int counter = 0; counter < NSAMPLES; counter++)
#ifdef __CUDA_ARCH__
                        pm_col[counter] = __ldg(&pulseMatrixView.coeffRef(counter, icol));
#else
                            pm_col[counter] = pulseMatrixView.coeffRef(counter, icol);
#endif

                    // accum
                    CMS_UNROLL_LOOP
                    for (int counter = 0; counter < NSAMPLES; counter++)
                        accum[counter] += results[icol] * pm_col[counter];
                }

            }

            // compute chi2 and check that there is no rotation
            // chi2 = matrixDecomposition
            //    .matrixL()
            //    . solve(mapAccum)
            //            .solve(pulseMatrixView * resultAmplitudesVector - inputAmplitudesView)
            //    .squaredNorm();

                //float reg_L[NSAMPLES];
                float accumSum = 0;

                // preload a column and load column 0 of cholesky
                CMS_UNROLL_LOOP
                for (int i = thrdIdx; i < NSAMPLES; i+=numThrd) {
                    reg_L[i] = matrixL(i, 0);
                }

                // compute x0 and store it
                auto x_prev = accum[0] / reg_L[0];
                accumSum += x_prev * x_prev;

                // iterate
                CMS_UNROLL_LOOP
                for (int iL = 1; iL < NSAMPLES; iL++) {

                    // update accum and load new column
                    CMS_UNROLL_LOOP
                    for (int counter = iL + thrdIdx; counter < NSAMPLES; counter+=numThrd){
                        accum[counter] -= x_prev * reg_L[counter];
                        reg_L[counter] = matrixL(counter, iL);}


                    tile.sync();
                    // compute the next x for M(iL, icol)
                    x_prev = accum[iL] / reg_L[iL];

                    // store the result value
                    accumSum += x_prev * x_prev;
                }


                chi2 = accumSum;

            tile.sync();

        }




       __device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
            float old;
            old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
                  __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

            return old;
        }


        __device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
            float old;
            old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
                  __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

            return old;
        }

        // TODO: add active bxs
        template<typename MatrixType, typename MapType, typename DataType, unsigned int TileSize>
        //, unsigned int TileSize>
        EIGEN_DEVICE_FUNC void fnnls_coop(MatrixType const &AtA,
                                          MapType const &Atb,
                                          Eigen::Map <calo::multifit::ColumnVector<MapType::RowsAtCompileTime, DataType>> &solution, //resultAmplitudes
                                          int &npassive,
                                          Eigen::Map <calo::multifit::ColumnVector<MapType::RowsAtCompileTime, int>> &pulseOffsets, //pulseOffsets
                                          Eigen::Map <calo::multifit::ColumnVector<MapType::RowsAtCompileTime, float>> &s,
                                          MapSymM<float, MapType::RowsAtCompileTime> &matrixL, //matrixLForFnnls
                                          double &eps,                    // convergence condition
                                          const int maxIterations,       // maximum number of iterations
                                          const int relaxationPeriod,    // every "relaxationPeriod" iterations
                                          const int relaxationFactor,
                                          float &w_max,
                                          float &w_max_prev,
                                          Eigen::Index &w_max_idx,
                                          Eigen::Index &w_max_idx_prev,
                                          bool &recompute,
                                          float &sumsq2,
                                          int &hasNegative,
                                          int &hasNans,
                                          float *reg_b,
                                          cg::thread_block_tile <TileSize> &tile) {

            auto const thrdIdx = tile.thread_rank();
            auto const numThrd = tile.num_threads();

            typedef Eigen::Matrix<DataType, 10, 1> VectorType;
            constexpr auto NPULSES = 10;

            w_max_idx_prev = 0;
            w_max_prev = 0;
            recompute = false;

            for (int iter = 0; iter < maxIterations; iter++) {

                if (iter > 0 || npassive == 0) {


                    auto const nactive = NPULSES - npassive;
                    // exit if there are no more pulses to constrain
                    if (nactive == 0)
                        break;

                    w_max_idx = 0;
                    w_max = -std::numeric_limits<float>::max();

                    //if (thrdIdx == 0) {  ///TODO!!
                        //for (int icol = npassive; icol <NPULSES; icol++) {
                            for (int icol = thrdIdx + npassive; icol < NPULSES; icol+= numThrd) {
                            auto const icol_real = pulseOffsets(icol);
                            auto const atb = Atb(icol_real);
                            float sum = 0;
                            CMS_UNROLL_LOOP
                            for (int counter = 0; counter < NPULSES; counter++)
                                sum += counter > icol_real ? AtA(counter, icol_real) * solution(counter)
                                                           : AtA(icol_real, counter) * solution(counter);

                            auto const w = atb - sum;

                            auto value = atomicMaxFloat(&w_max, w);
                            value = atomicMaxFloat(&w_max, w);

                            if(value == w){
                                w_max_idx = icol - npassive;
                            }


                        }
                    //}

                    tile.sync();
                    // check for convergence
                    if (w_max < eps || (w_max_idx == w_max_idx_prev && w_max == w_max_prev))
                        break;

                    tile.sync();

                    w_max_prev = w_max;
                    w_max_idx_prev = w_max_idx;

                    tile.sync();

                    // move index to the right part of the vector
                    if (thrdIdx == 0) {
                        w_max_idx += npassive;

                        Eigen::numext::swap(pulseOffsets.coeffRef(npassive),
                                            pulseOffsets.coeffRef(w_max_idx)); // coefRef is O(log) binary search
                        ++npassive;
                    }

                    tile.sync();
                }


                // inner loop
                for (int HMT = 0; HMT < maxIterations; HMT++) {


                    if (npassive == 0)
                        break;


                    if (recompute || iter == 0)
                        compute_decomposition_forwardsubst_with_offsets(matrixL, AtA, reg_b, Atb, npassive,
                                                                        pulseOffsets, sumsq2, tile);
                    else
                        update_decomposition_forwardsubst_with_offsets(matrixL, AtA, reg_b, Atb, npassive,
                                                                       pulseOffsets, sumsq2, tile);

                    // run backward substituion
                    s(npassive - 1) = reg_b[npassive - 1] / matrixL(npassive - 1, npassive - 1);


                    sumsq2 = 0;
                    tile.sync();

                    for (int i = npassive - 2; i >= 0; i--) {
                        for (int j = i + 1 + thrdIdx; j < npassive; j += numThrd) {
                            atomicAdd(&sumsq2, matrixL(j, i) * s(j));
                        }

                        tile.sync();

                        if (thrdIdx == 0) {
                            s(i) = (reg_b[i] - sumsq2) / matrixL(i, i);
                            sumsq2 = 0;
                        }
                    }


                    // done if solution values are all positive
                    hasNegative = 0;
                    hasNans = 0;

                    tile.sync();

                    for (int counter = thrdIdx; counter < npassive; counter += numThrd) {
                        auto const s_ii = s(counter);
                        atomicOr(&hasNegative, s_ii <= 0);
                        atomicOr(&hasNans, std::isnan(s_ii));
                    }


                    tile.sync();

                    // FIXME: temporary solution. my cholesky impl is unstable yielding nans
                    // this check removes nans - do not accept solution unless all values
                    // are stable
                    if (hasNans)
                        break;


                    if (!hasNegative) {
                        for (int i = thrdIdx; i < npassive; i += numThrd) {
                            auto const i_real = pulseOffsets(i);
                            solution(i_real) = s(i);
                        }

                        //tile.sync();
                        //solution.head(npassive) = s.head(npassive);
                        recompute = false;
                        break;
                    }

                    tile.sync();

                    // there were negative values -> have to recompute the whole decomp
                    recompute = true;

                    sumsq2 = std::numeric_limits<float>::max(); //alpha

                    Eigen::Index alpha_idx = 0, alpha_idx_real = 0;

                    if (thrdIdx == 0) { ///TODO!!
                        for (int i = 0; i < npassive; i++) {
                            if (s[i] <= 0.) {
                                auto const i_real = pulseOffsets(i);
                                auto const ratio = solution[i_real] / (solution[i_real] - s[i]);


                                auto value = atomicMinFloat(&sumsq2, ratio);
                                value = atomicMinFloat(&sumsq2, ratio);

                                if(value == ratio){
                                    alpha_idx = i;
                                    alpha_idx_real = i_real; //atomicMin
                                }


                                /*if (ratio < sumsq2) {
                                    sumsq2 = ratio;
                                    alpha_idx = i;
                                    alpha_idx_real = i_real; //atomicMax
                                }*/
                            }
                        }
                    }

                    tile.sync();

                    // upadte solution
                    for (int i = 0; i < npassive; i++) {
                        auto const i_real = pulseOffsets(i);
                        solution(i_real) += sumsq2 * (s(i) - solution(i_real));
                    }


                    if (thrdIdx == 0) {
                        solution[alpha_idx_real] = 0;
                        --npassive;
                        Eigen::numext::swap(pulseOffsets.coeffRef(npassive), pulseOffsets.coeffRef(alpha_idx));
                    }
                }


                if (thrdIdx == 0) {
                    // as in cpu
                    if (iter % relaxationPeriod == 0)
                        eps *= relaxationFactor;
                }

                tile.sync();

            }

            tile.sync();
        }

    }  // namespace multifit
}  // namespace calo

#endif  // DataFormats_CaloRecHit_interface_MultifitComputations_h
