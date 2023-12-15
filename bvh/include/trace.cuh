#ifndef BVH_TRACE_CUH
#define BVH_TRACE_CUH
#include <cstdint>
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include "utility.cuh"

template <typename T, int MAX_SIZE=32>
class IndexStack {
public:
    __host__ __device__ void push(T val) {
        if (m_count >= MAX_SIZE - 1) {
            printf("WARNING TOO BIG\n");
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
};

std::tuple<int32_t, thrust::device_vector<int32_t>, thrust::device_vector<float3>, thrust::device_vector<int32_t>>
trace_bvh_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, int32_t* contributes);

void trace_bvh_opacity_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, float3* normals,
                    int32_t* contributes,
                    float* rendered_opacity);

#endif //BVH_TRACE_CUH