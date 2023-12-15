#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderEquationForwardCUDA_complex(
	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents_shs,
	const torch::Tensor& direct_shs,
	const torch::Tensor& visibility_shs,
	const int sample_num);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RenderEquationForwardCUDA(
	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents_shs,
	const torch::Tensor& direct_shs,
	const torch::Tensor& visibility_shs,
	const int sample_num,
	const bool is_training,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RenderEquationBackwardCUDA(
 	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents,
	const torch::Tensor& direct_shs,
	const torch::Tensor& visibility_shs,
	const int sample_num,
	const torch::Tensor& incident_dirs,
    const torch::Tensor& dL_drgb,
    const torch::Tensor& dL_ddiffuse_light,
	const bool debug);