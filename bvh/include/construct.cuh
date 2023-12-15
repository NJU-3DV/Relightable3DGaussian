#ifndef BVH_CONSTRUCT_CUH
#define BVH_CONSTRUCT_CUH
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

void construct_bvh(
        int32_t num_objects,
        const float* means3D,
        const float* scales,
        const float* rotations,
        int32_t* nodes,
        float* aabbs,
        uint64_t* morton);

#endif //BVH_CONSTRUCT_CUH