#ifndef BVH_BVH_H
#define BVH_BVH_H
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
create_bvh(const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& nodes, const torch::Tensor& aabbs);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities);

std::tuple<torch::Tensor, torch::Tensor>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& normals);
#endif //BVH_BVH_H
