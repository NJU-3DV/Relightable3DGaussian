#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include "cuda_rasterizer/auxiliary.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <glm/glm.hpp>

__device__ void computeSHcoef(int deg, const glm::vec3 dir, float* coef)
{
	coef[0] = SH_C0;
	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
        coef[1] = - SH_C1 * y;
        coef[2] = SH_C1 * z;
        coef[3] = - SH_C1 * x;

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
            coef[4] = SH_C2[0] * xy;
            coef[5] = SH_C2[1] * yz;
            coef[6] = SH_C2[2] * (2.0f * zz - xx - yy);
            coef[7] = SH_C2[3] * xz;
            coef[8] = SH_C2[4] * (xx - yy);
			if (deg > 2)
			{
				coef[9] = SH_C3[0] * y * (3.0f * xx - yy);
                coef[10] = SH_C3[1] * xy * z;
                coef[11] = SH_C3[2] * y * (4.0f * zz - xx - yy);
                coef[12] = SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
                coef[13] = SH_C3[4] * x * (4.0f * zz - xx - yy);
                coef[14] = SH_C3[5] * z * (xx - yy);
                coef[15] = SH_C3[6] * x * (xx - 3.0f * yy);
			}
		}
	}
}

__global__ void render_equation_forward_complex_kernel(
    const int P, const int S_incident, const int S_direct, const int S_vis,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents_shs,
    const glm::vec3* direct_shs,
    const float* visibility_shs,
    const int sample_num,
    glm::vec3* incident_dirs,
    glm::vec3* out_pbr,
    glm::vec3* out_incident_lights,
    glm::vec3* out_local_incident_lights,
    glm::vec3* out_global_incident_lights,
    float* out_incident_visibility,
    glm::vec3* out_diffuse_light,
    glm::vec3* out_local_diffuse_light,
    float* out_accum,
    glm::vec3* out_rgb_d,
    glm::vec3* out_rgb_s)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
    glm::vec3 rgb_d = {0.0f, 0.0f, 0.0f};
    glm::vec3 rgb_s = {0.0f, 0.0f, 0.0f};
    glm::vec3 diffuse_light = {0.0f, 0.0f, 0.0f};
    glm::vec3 local_diffuse_light = {0.0f, 0.0f, 0.0f};
    const glm::vec3 normal = normals[idx];
    const glm::vec3 viewdir = viewdirs[idx];
    const glm::vec3 base = base_color[idx];
    const float metal = metallic[idx];
    const float rough = roughness[idx];
	for (int ray_id=0;ray_id<sample_num;ray_id++){
        const int whole_idx = idx*sample_num + ray_id;
        const float delta = 3.14159f * (3.0f - sqrtf(5.0f));
        const float z = 1 - 2 * (float)ray_id / (2 * (float)sample_num - 1);
        const float rad = sqrtf(1 - z * z);
        float theta = delta * ray_id;
        const float y = cosf(theta) * rad;
        const float x = sinf(theta) * rad;
        float3 z_s = {x, y, z};

        const float v1 = -normal.y;
        const float v2 = normal.x;
        const float v3 = 0.f;
        const float v11 = v1 * v1;
        const float v22 = v2 * v2;
        const float v33 = v3 * v3;
        const float v12 = v1 * v2;
        const float v13 = v1 * v3;
        const float v23 = v2 * v3;
        const float cos_p_1 = fmaxf(normal.z + 1, 0.0000001f);
        z_s = {
            (1 + (-v33 - v22) / cos_p_1) * z_s.x + (-v3 + v12 / cos_p_1) * z_s.y + (v2 + v13 / cos_p_1) * z_s.z,
            (v3 + v12 / cos_p_1) * z_s.x + (1 + (-v33 - v11) / cos_p_1) * z_s.y + (-v1 + v23 / cos_p_1) * z_s.z,
            (-v2 + v13 / cos_p_1) * z_s.x + (v1 + v23 / cos_p_1) * z_s.y + (1 + (-v22 - v11) / cos_p_1) * z_s.z
        };
        const float norm = sqrtf(fmaxf(0.0000001f, z_s.x*z_s.x+z_s.y*z_s.y+z_s.z*z_s.z));
        glm::vec3 incident_dir = {z_s.x / norm, z_s.y / norm, z_s.z / norm};
        float coef[16] = {0.0f};
        computeSHcoef(3, incident_dir, coef);

        glm::vec3 local_incident_light = {0.f, 0.f, 0.f};
        for (int i=0;i<S_incident;i++){
            local_incident_light+=incidents_shs[idx * S_incident + i] * coef[i];
        }
        local_incident_light = glm::max(local_incident_light, 0.0f);

        glm::vec3 global_incident_light = {0.5f, 0.5f, 0.5f};
        for (int i=0;i<S_direct;i++){
            global_incident_light+=direct_shs[i] * coef[i];
        }
        global_incident_light = glm::max(global_incident_light, 0.0f);

        float incident_visibility = 0.5f;
        for (int i=0;i<S_vis;i++){
            incident_visibility+=visibility_shs[idx * S_vis + i] * coef[i];
        }
        incident_visibility = fmaxf(0.0f, fminf(incident_visibility, 1.0f));

        global_incident_light = incident_visibility*global_incident_light;
        glm::vec3 incident_light=global_incident_light+local_incident_light;

        glm::vec3 half_dir = incident_dir + viewdir;
        half_dir = half_dir / fmaxf(glm::length(half_dir), 0.0000001f);
        float h_d_n = fmaxf(glm::dot(half_dir, normal), 0.0f);
        float h_d_o = fmaxf(glm::dot(half_dir, viewdir), 0.0f);
        float n_d_i = fmaxf(glm::dot(normal, incident_dir), 0.0f);
        float n_d_o = fmaxf(glm::dot(normal, viewdir), 0.0f);

        glm::vec3 f_d = (1 - metal) * base / 3.14159f;

        // D
        float r2 = fmaxf(rough * rough, 0.0000001f);
        float amp = 1.0f / (r2 * 3.14159f);
        float sharp = 2.0f / r2;
        float D = amp * expf(sharp * (h_d_n - 1.0f));

        // F
        glm::vec3 F_0 = 0.04f * (1.0f - metal) + base * metal;
        glm::vec3 F = F_0 + (1.0f - F_0) * powf(1.0f - h_d_o, 5.0f);

        // V
        r2 = __powf(1.0f + rough, 2.0f) / 8.0f;
        float V = (0.5f / fmaxf(n_d_i * (1 - r2) + r2, 0.0000001f)) * (0.5f / fmaxf(n_d_o * (1 - r2) + r2, 0.0000001f));

        glm::vec3 f_s = D*F*V;
        float tmp = 2.0f * 3.14159f * n_d_i / (float)sample_num;
        glm::vec3 transport = incident_light * tmp;
        glm::vec3 local_transport = local_incident_light * tmp;
        diffuse_light += transport;
        local_diffuse_light += local_transport;
        rgb_d += f_d * transport;
        rgb_s += f_s * transport;

        incident_dirs[whole_idx] = incident_dir;
        out_incident_lights[whole_idx] = incident_light;
        out_local_incident_lights[whole_idx] = local_incident_light;
        out_global_incident_lights[whole_idx] = global_incident_light;
        out_incident_lights[whole_idx] = incident_light;
        out_incident_visibility[whole_idx] = incident_visibility;
	}

    // accum
    glm::vec3 accum_vec = diffuse_light / 3.14159f + rgb_s;
	out_accum[idx]=(accum_vec.x + accum_vec.y + accum_vec.z)/3;

	out_pbr[idx]=rgb_d+rgb_s;
	out_rgb_d[idx]=rgb_d;
	out_rgb_s[idx]=rgb_s;
	out_diffuse_light[idx]=diffuse_light;
	out_local_diffuse_light[idx]=local_diffuse_light;
}

void render_equation_forward_complex_cuda(
    const int P, const int S_incident, const int S_direct, const int S_vis,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents_shs,
    const glm::vec3* direct_shs,
    const float* visibility_shs,
    const int sample_num,
    glm::vec3* incident_dirs,
    glm::vec3* out_pbr,
    glm::vec3* incident_lights,
    glm::vec3* local_incident_lights,
    glm::vec3* global_incident_lights,
    float* incident_visibility,
    glm::vec3* diffuse_light,
    glm::vec3* local_diffuse_light,
    float* accum,
    glm::vec3* rgb_d,
    glm::vec3* rgb_s
){
    render_equation_forward_complex_kernel << <(P + 255) / 256, 256 >> > (
    P, S_incident, S_direct, S_vis, base_color, roughness, metallic, normals, viewdirs, incidents_shs, direct_shs, visibility_shs,
	sample_num, incident_dirs, out_pbr,
	incident_lights, local_incident_lights, global_incident_lights, incident_visibility,
	diffuse_light, local_diffuse_light, accum, rgb_d, rgb_s);
}


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
	const int sample_num){
	const int P = base_color.size(0);
	const int S_incident = incidents_shs.size(1);
	const int S_direct = direct_shs.size(1);
	const int S_vis = visibility_shs.size(1);
    auto float_opts = base_color.options().dtype(torch::kFloat32);
    torch::Tensor pbr = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor incident_dirs = torch::full({P, sample_num, 3}, 0.0, float_opts);

    torch::Tensor incident_lights = torch::full({P, sample_num, 3}, 0.0, float_opts);
    torch::Tensor local_incident_lights = torch::full({P, sample_num, 3}, 0.0, float_opts);
    torch::Tensor global_incident_lights = torch::full({P, sample_num, 3}, 0.0, float_opts);
    torch::Tensor incident_visibility = torch::full({P, sample_num, 1}, 0.0, float_opts);
    torch::Tensor diffuse_light = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor local_diffuse_light = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor accum = torch::full({P, 1}, 0.0, float_opts);
    torch::Tensor rgb_d = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor rgb_s = torch::full({P, 3}, 0.0, float_opts);
    render_equation_forward_complex_cuda(
        P,S_incident,S_direct,S_vis,
        (glm::vec3*)base_color.contiguous().data_ptr<float>(),
        roughness.contiguous().data_ptr<float>(),
        metallic.contiguous().data_ptr<float>(),
        (glm::vec3*)normals.contiguous().data_ptr<float>(),
        (glm::vec3*)viewdirs.contiguous().data_ptr<float>(),
        (glm::vec3*)incidents_shs.contiguous().data_ptr<float>(),
        (glm::vec3*)direct_shs.contiguous().data_ptr<float>(),
        visibility_shs.contiguous().data_ptr<float>(),
        sample_num,
        (glm::vec3*)incident_dirs.contiguous().data_ptr<float>(),
        (glm::vec3*)pbr.contiguous().data_ptr<float>(),
        (glm::vec3*)incident_lights.contiguous().data_ptr<float>(),
        (glm::vec3*)local_incident_lights.contiguous().data_ptr<float>(),
        (glm::vec3*)global_incident_lights.contiguous().data_ptr<float>(),
        incident_visibility.contiguous().data_ptr<float>(),
        (glm::vec3*)diffuse_light.contiguous().data_ptr<float>(),
        (glm::vec3*)local_diffuse_light.contiguous().data_ptr<float>(),
        accum.contiguous().data_ptr<float>(),
        (glm::vec3*)rgb_d.contiguous().data_ptr<float>(),
        (glm::vec3*)rgb_s.contiguous().data_ptr<float>()
    );
    return std::make_tuple(pbr, incident_dirs, incident_lights, local_incident_lights,
    global_incident_lights, incident_visibility, diffuse_light, local_diffuse_light,
    accum, rgb_d, rgb_s);
}


__global__ void render_equation_backward_kernel(
    const int P, const int S_incident, const int S_direct, const int S_vis,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents_shs,
    const glm::vec3* direct_shs,
    const float* visibility_shs,
    const int sample_num,
    const glm::vec3* incident_dirs,
    const glm::vec3* dL_dpbrs,
    const glm::vec3* dL_ddiffuse_lights,
    glm::vec3* dL_dbase_color,
    float* dL_droughness,
    float* dL_dmetallic,
    glm::vec3* dL_dnormals,
    glm::vec3* dL_dviewdirs,
    glm::vec3* dL_dincidents_shs,
    glm::vec3* dL_ddirect_shs,
    float* dL_dvisibility_shs)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	const glm::vec3 normal = normals[idx];
    const glm::vec3 viewdir = viewdirs[idx];
    const glm::vec3 base = base_color[idx];
    const glm::vec3 dL_dpbr = dL_dpbrs[idx];
    const glm::vec3 dL_ddiffuse_light = dL_ddiffuse_lights[idx];
    const float metal = metallic[idx];
    const float rough = roughness[idx];
	for (int ray_id=0;ray_id<sample_num;ray_id++){
        const int whole_idx = idx*sample_num + ray_id;
        glm::vec3 incident_dir = incident_dirs[whole_idx];
        //const float* coefs= incident_dirs_coefs + whole_idx * 16;
        float coefs[16] = {0.0f};
	    computeSHcoef(3, incident_dir, coefs);

        glm::vec3 local_incident_light = {0.f, 0.f, 0.f};
        for (int i=0;i<S_incident;i++){
            local_incident_light+=incidents_shs[idx * S_incident + i] * coefs[i];
        }
        local_incident_light = glm::max(local_incident_light, 0.0f);

        glm::vec3 global_incident_light = {0.5f, 0.5f, 0.5f};
        for (int i=0;i<S_direct;i++){
            global_incident_light+=direct_shs[i] * coefs[i];
        }
        global_incident_light = glm::max(global_incident_light, 0.0f);

        float incident_visibility = 0.5f;
        for (int i=0;i<S_vis;i++){
            incident_visibility+=visibility_shs[idx * S_vis + i] * coefs[i];
        }
        incident_visibility = fmaxf(0.0f, fminf(incident_visibility, 1.0f));

        glm::vec3 incident_light=incident_visibility*global_incident_light+local_incident_light;

        glm::vec3 half_dir = incident_dir + viewdir;
        float half_norm = fmaxf(glm::length(half_dir), 0.0000001f);
        glm::vec3 half_dir_normalize = half_dir / half_norm;
        float h_d_n = fmaxf(glm::dot(half_dir_normalize, normal), 0.0f);
        float h_d_o = fmaxf(glm::dot(half_dir_normalize, viewdir), 0.0f);
        float n_d_i = fmaxf(glm::dot(normal, incident_dir), 0.0f);
        float n_d_o = fmaxf(glm::dot(normal, viewdir), 0.0f);

        glm::vec3 f_d = (1 - metal) * base / 3.14159f;

        // D
        float r2 = fmaxf(rough * rough, 0.0000001f);
        float amp = 1.0f / (r2 * 3.14159f);
        float sharp = 2.0f / r2;
        float expf_amp = expf(sharp * (h_d_n - 1.0f));
        float D = amp * expf_amp;

        // F
        glm::vec3 F_0 = 0.04f * (1.0f - metal) + base * metal;
        glm::vec3 F = F_0 + (1.0f - F_0) * powf(1.0f - h_d_o, 5.0f);

        // V
        float r2v = powf(1.0f + rough, 2.0f) / 8.0f;
        float denom1 = fmaxf(n_d_i * (1 - r2v) + r2v, 0.0000001f);
        float denom2 = fmaxf(n_d_o * (1 - r2v) + r2v, 0.0000001f);
        float v_schlick_ggx1 = 0.5f / denom1;
        float v_schlick_ggx2 = 0.5f / denom2;
        float V = v_schlick_ggx1 * v_schlick_ggx2;

        glm::vec3 f_s = D*F*V;
        //pbr += (f_d+f_s) * incident_light * (2.0f * 3.14159f * n_d_i / (float)sample_num);

        glm::vec3 dL_dfd = dL_dpbr * incident_light * (2.0f * 3.14159f * n_d_i / (float)sample_num);
        glm::vec3 dL_dfs = dL_dpbr * incident_light * (2.0f * 3.14159f * n_d_i / (float)sample_num);
        glm::vec3 dL_dincident_light = dL_dpbr * (f_d+f_s) * (2.0f * 3.14159f * n_d_i / (float)sample_num);
        float dL_dn_d_i = glm::dot(dL_dpbr, (f_d+f_s) * incident_light) * (2.0f * 3.14159f / (float)sample_num);

        // from dL_ddiffuse_light
        dL_dincident_light += dL_ddiffuse_light * (2.0f * 3.14159f * n_d_i / (float)sample_num);
        dL_dn_d_i += glm::dot(dL_ddiffuse_light, incident_light * (2.0f * 3.14159f / (float)sample_num));

        // from dL_dfd
        glm::vec3 dL_dbase = dL_dfd * (1 - metal) / 3.14159f;
        float dL_dmetal = -glm::dot(dL_dfd, base) / 3.14159f;

        // from dL_dfs
        float dL_dD = glm::dot(dL_dfs*V, F);
        glm::vec3 dL_dF = dL_dfs*D*V;
        float dL_dV = glm::dot(dL_dfs*D,F);
        // from dL_dD
        float dL_damp = dL_dD * expf_amp;
        float dL_dexpf_amp = dL_dD * amp;
        float dL_dsharp = (h_d_n - 1.0f) * expf_amp * dL_dexpf_amp;
        float dL_dh_d_n = sharp * expf_amp * dL_dexpf_amp;
        float dL_dr2 = -2.0f/(r2*r2) * dL_dsharp - 1.0f / (r2 * r2 * 3.14159f) * dL_damp;
        float dL_drough = dL_dr2 * 2.0f * rough;
        // from dL_dF
        glm::vec3 dL_dF0 = (1.0f - powf(1.0f - h_d_o, 5.0f)) * dL_dF;
        float dL_dh_d_o = glm::dot((1.0f - F_0), dL_dF) * -5.0f * powf(1.0f - h_d_o, 4.0f);
        dL_dbase += metal * dL_dF0;
        dL_dmetal += glm::dot(base-0.04f, dL_dF0);
        // from dL_dV
        float dL_dv_schlick_ggx1 = dL_dV * v_schlick_ggx2;
        float dL_dv_schlick_ggx2 = dL_dV * v_schlick_ggx1;
        float dL_ddenom1 = -0.5f / (denom1 * denom1) * dL_dv_schlick_ggx1;
        float dL_ddenom2 = -0.5f / (denom2 * denom2) * dL_dv_schlick_ggx2;
        dL_dn_d_i = dL_ddenom1 * (1 - r2v);
        float dL_dn_d_o = dL_ddenom2 * (1 - r2v);
        float dL_dr2v = (1.0f - n_d_i) * dL_ddenom1 + (1.0f - n_d_o) * dL_ddenom2;
        dL_drough += (1.0f + rough) / 4.0f * dL_dr2v;

        glm::vec3 dL_dhalf_dir_normalize={0.0f,0.0f,0.0f};
        glm::vec3 dL_dnormal={0.0f,0.0f,0.0f};
        glm::vec3 dL_dviewdir={0.0f,0.0f,0.0f};
        if (h_d_n>0.0f){
            dL_dhalf_dir_normalize += normal * dL_dh_d_n;
            dL_dnormal += half_dir_normalize * dL_dh_d_n;
        }
        if (h_d_o>0.0f){
            dL_dhalf_dir_normalize += viewdir * dL_dh_d_o;
            dL_dviewdir += half_dir_normalize * dL_dh_d_o;
        }
        if (n_d_i>0.0f){
            dL_dnormal += incident_dir * dL_dn_d_i;
        }

        if (n_d_o>0.0f){
            dL_dnormal += viewdir * dL_dn_d_o;
            dL_dviewdir += normal * dL_dn_d_o;
        }

        // half_dir TODO:consider norm
        dL_dviewdir += dL_dhalf_dir_normalize / half_norm;

        // shs
        glm::vec3 dL_dlocal_incident_light = dL_dincident_light;
        glm::vec3 dL_dglobal_incident_light = dL_dincident_light * incident_visibility;
        float dL_dincident_visibility = glm::dot(dL_dincident_light, global_incident_light);
        if (incident_visibility<=1.0f && incident_visibility>=0.0f){
            for (int i=0;i<S_vis;i++){
                dL_dvisibility_shs[idx * S_vis + i]+=dL_dincident_visibility * coefs[i];
            }
        }
        if (global_incident_light.x<0.0f) dL_dglobal_incident_light.x = 0.0f;
        if (global_incident_light.y<0.0f) dL_dglobal_incident_light.y = 0.0f;
        if (global_incident_light.z<0.0f) dL_dglobal_incident_light.z = 0.0f;
        for (int i=0;i<S_direct;i++){
            dL_ddirect_shs[i]+=dL_dglobal_incident_light * coefs[i];
        }

        if (local_incident_light.x<0.0f) dL_dlocal_incident_light.x = 0.0f;
        if (local_incident_light.y<0.0f) dL_dlocal_incident_light.y = 0.0f;
        if (local_incident_light.z<0.0f) dL_dlocal_incident_light.z = 0.0f;
        for (int i=0;i<S_direct;i++){
            dL_dincidents_shs[idx * S_incident + i]+=dL_dlocal_incident_light * coefs[i];
        }

        dL_dviewdirs[idx]+=dL_dviewdir;
        dL_dnormals[idx]+=dL_dnormal;
        dL_dbase_color[idx]+=dL_dbase;
        dL_dmetallic[idx]+=dL_dmetal;
        dL_droughness[idx]+=dL_drough;
	}
}


void render_equation_backward_cuda(
    const int P, const int S_incident, const int S_direct, const int S_vis,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents_shs,
    const glm::vec3* direct_shs,
    const float* visibility_shs,
    const int sample_num,
    const glm::vec3* incident_dirs,
    const glm::vec3* dL_dpbrs,
    const glm::vec3* dL_ddiffuse_light,
    glm::vec3* dL_dbase_color,
    float* dL_droughness,
    float* dL_dmetallic,
    glm::vec3* dL_dnormals,
    glm::vec3* dL_dviewdirs,
    glm::vec3* dL_dincidents_shs,
    glm::vec3* dL_ddirect_shs,
    float* dL_dvisibility_shs
){
    render_equation_backward_kernel << <(P + 255) / 256, 256 >> > (
    P, S_incident, S_direct, S_vis, base_color, roughness, metallic,
    normals, viewdirs, incidents_shs, direct_shs, visibility_shs,
	sample_num, incident_dirs,
	dL_dpbrs, dL_ddiffuse_light, dL_dbase_color, dL_droughness, dL_dmetallic,
	dL_dnormals, dL_dviewdirs, dL_dincidents_shs, dL_ddirect_shs, dL_dvisibility_shs);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RenderEquationBackwardCUDA(
 	const torch::Tensor& base_color,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
    const torch::Tensor& normals,
    const torch::Tensor& viewdirs,
	const torch::Tensor& incidents_shs,
	const torch::Tensor& direct_shs,
	const torch::Tensor& visibility_shs,
	const int sample_num,
	const torch::Tensor& incident_dirs,
    const torch::Tensor& dL_dpbrs,
    const torch::Tensor& dL_ddiffuse_light,
	const bool debug){
    const int P = base_color.size(0);
	const int S_incident = incidents_shs.size(1);
	const int S_direct = direct_shs.size(1);
	const int S_vis = visibility_shs.size(1);
    auto float_opts = base_color.options().dtype(torch::kFloat32);
	torch::Tensor dL_dbase_color = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_droughness = torch::zeros({P, 1}, float_opts);
    torch::Tensor dL_dmetallic = torch::zeros({P, 1}, float_opts);
    torch::Tensor dL_dnormals = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dviewdirs = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dincidents_shs = torch::zeros({P, S_incident, 3}, float_opts);
    torch::Tensor dL_ddirect_shs = torch::zeros({1, S_direct, 3}, float_opts);
    torch::Tensor dL_dvisibility_shs = torch::zeros({P, S_vis, 1}, float_opts);
    render_equation_backward_cuda(
            P,S_incident,S_direct,S_vis,
            (glm::vec3*)base_color.contiguous().data_ptr<float>(),
            roughness.contiguous().data_ptr<float>(),
            metallic.contiguous().data_ptr<float>(),
            (glm::vec3*)normals.contiguous().data_ptr<float>(),
            (glm::vec3*)viewdirs.contiguous().data_ptr<float>(),
            (glm::vec3*)incidents_shs.contiguous().data_ptr<float>(),
            (glm::vec3*)direct_shs.contiguous().data_ptr<float>(),
            visibility_shs.contiguous().data_ptr<float>(),
            sample_num,
            (glm::vec3*)incident_dirs.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dpbrs.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_ddiffuse_light.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dbase_color.contiguous().data_ptr<float>(),
            dL_droughness.contiguous().data_ptr<float>(),
            dL_dmetallic.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dnormals.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dviewdirs.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_dincidents_shs.contiguous().data_ptr<float>(),
            (glm::vec3*)dL_ddirect_shs.contiguous().data_ptr<float>(),
            dL_dvisibility_shs.contiguous().data_ptr<float>()
        );
    return std::make_tuple(dL_dbase_color, dL_droughness, dL_dmetallic, dL_dnormals,
        dL_dviewdirs, dL_dincidents_shs, dL_ddirect_shs, dL_dvisibility_shs);
}




__global__ void render_equation_forward_kernel(
    const int P, const int S_incident, const int S_direct, const int S_vis,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents_shs,
    const glm::vec3* direct_shs,
    const float* visibility_shs,
    const int sample_num,
    const bool is_training,
    const float* rand_float,
    glm::vec3* incident_dirs,
    glm::vec3* out_pbr,
    glm::vec3* out_diffuse_light)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
    glm::vec3 diffuse_light = {0.0f, 0.0f, 0.0f};
    glm::vec3 pbr = {0.0f, 0.0f, 0.0f};
    const glm::vec3 normal = normals[idx];
    const glm::vec3 viewdir = viewdirs[idx];
    const glm::vec3 base = base_color[idx];
    const float metal = metallic[idx];
    const float rough = roughness[idx];
	for (int ray_id=0;ray_id<sample_num;ray_id++){
        const int whole_idx = idx*sample_num + ray_id;
        const float delta = 3.14159f * (3.0f - sqrtf(5.0f));
        const float z = 1 - 2 * (float)ray_id / (2 * (float)sample_num - 1);
        const float rad = sqrtf(1 - z * z);
        float theta = delta * ray_id;
        if (is_training) theta = rand_float[whole_idx] * 2 * 3.14159f + theta;
        const float y = cosf(theta) * rad;
        const float x = sinf(theta) * rad;
        float3 z_s = {x, y, z};

        const float v1 = -normal.y;
        const float v2 = normal.x;
        const float v3 = 0.f;
        const float v11 = v1 * v1;
        const float v22 = v2 * v2;
        const float v33 = v3 * v3;
        const float v12 = v1 * v2;
        const float v13 = v1 * v3;
        const float v23 = v2 * v3;
        const float cos_p_1 = fmaxf(normal.z + 1, 0.0000001f);
        z_s = {
            (1 + (-v33 - v22) / cos_p_1) * z_s.x + (-v3 + v12 / cos_p_1) * z_s.y + (v2 + v13 / cos_p_1) * z_s.z,
            (v3 + v12 / cos_p_1) * z_s.x + (1 + (-v33 - v11) / cos_p_1) * z_s.y + (-v1 + v23 / cos_p_1) * z_s.z,
            (-v2 + v13 / cos_p_1) * z_s.x + (v1 + v23 / cos_p_1) * z_s.y + (1 + (-v22 - v11) / cos_p_1) * z_s.z
        };
        const float norm = sqrtf(fmaxf(0.0000001f, z_s.x*z_s.x+z_s.y*z_s.y+z_s.z*z_s.z));
        glm::vec3 incident_dir = {z_s.x / norm, z_s.y / norm, z_s.z / norm};
        float coef[16] = {0.0f};
        computeSHcoef(3, incident_dir, coef);

        glm::vec3 local_incident_light = {0.f, 0.f, 0.f};
        for (int i=0;i<S_incident;i++){
            local_incident_light+=incidents_shs[idx * S_incident + i] * coef[i];
        }
        local_incident_light = glm::max(local_incident_light, 0.0f);

        glm::vec3 global_incident_light = {0.5f, 0.5f, 0.5f};
        for (int i=0;i<S_direct;i++){
            global_incident_light+=direct_shs[i] * coef[i];
        }
        global_incident_light = glm::max(global_incident_light, 0.0f);

        float incident_visibility = 0.5f;
        for (int i=0;i<S_vis;i++){
            incident_visibility+=visibility_shs[idx * S_vis + i] * coef[i];
        }
        incident_visibility = fmaxf(0.0f, fminf(incident_visibility, 1.0f));

        global_incident_light = incident_visibility*global_incident_light;
        glm::vec3 incident_light=global_incident_light+local_incident_light;

        glm::vec3 half_dir = incident_dir + viewdir;
        half_dir = half_dir / fmaxf(glm::length(half_dir), 0.0000001f);
        float h_d_n = fmaxf(glm::dot(half_dir, normal), 0.0f);
        float h_d_o = fmaxf(glm::dot(half_dir, viewdir), 0.0f);
        float n_d_i = fmaxf(glm::dot(normal, incident_dir), 0.0f);
        float n_d_o = fmaxf(glm::dot(normal, viewdir), 0.0f);

        glm::vec3 f_d = (1 - metal) * base / 3.14159f;

        // D
        float r2 = fmaxf(rough * rough, 0.0000001f);
        float amp = 1.0f / (r2 * 3.14159f);
        float sharp = 2.0f / r2;
        float D = amp * expf(sharp * (h_d_n - 1.0f));

        // F
        glm::vec3 F_0 = 0.04f * (1.0f - metal) + base * metal;
        glm::vec3 F = F_0 + (1.0f - F_0) * powf(1.0f - h_d_o, 5.0f);

        // V
        r2 = __powf(1.0f + rough, 2.0f) / 8.0f;
        float V = (0.5f / fmaxf(n_d_i * (1 - r2) + r2, 0.0000001f)) * (0.5f / fmaxf(n_d_o * (1 - r2) + r2, 0.0000001f));

        glm::vec3 f_s = D*F*V;
        glm::vec3 transport = incident_light * (2.0f * 3.14159f * n_d_i / (float)sample_num);
        pbr += (f_d+f_s) * transport;
        diffuse_light += transport;
        incident_dirs[whole_idx] = incident_dir;

	}
	out_pbr[idx]=pbr;
	out_diffuse_light[idx]=diffuse_light;
}

void render_equation_forward_cuda(
    const int P, const int S_incident, const int S_direct, const int S_vis,
    const glm::vec3* base_color,
    const float* roughness,
    const float* metallic,
    const glm::vec3* normals,
    const glm::vec3* viewdirs,
    const glm::vec3* incidents_shs,
    const glm::vec3* direct_shs,
    const float* visibility_shs,
    const int sample_num,
    const bool is_training,
    const float* rand_float,
    glm::vec3* incident_dirs,
    glm::vec3* out_pbr,
    glm::vec3* out_diffuse_light
){
    render_equation_forward_kernel << <(P + 255) / 256, 256 >> > (
    P, S_incident, S_direct, S_vis, base_color, roughness, metallic, normals, viewdirs, incidents_shs, direct_shs, visibility_shs,
	sample_num, is_training, rand_float, incident_dirs, out_pbr, out_diffuse_light);
}


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
	const bool debug){
	const int P = base_color.size(0);
	const int S_incident = incidents_shs.size(1);
	const int S_direct = direct_shs.size(1);
	const int S_vis = visibility_shs.size(1);
    auto float_opts = base_color.options().dtype(torch::kFloat32);
    torch::Tensor pbr = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor incident_dirs = torch::full({P, sample_num, 3}, 0.0, float_opts);
    torch::Tensor rand_float = torch::rand({P, sample_num, 1}, float_opts);
    torch::Tensor diffuse_light = torch::full({P, 3}, 0.0, float_opts);
    render_equation_forward_cuda(
        P,S_incident,S_direct,S_vis,
        (glm::vec3*)base_color.contiguous().data_ptr<float>(),
        roughness.contiguous().data_ptr<float>(),
        metallic.contiguous().data_ptr<float>(),
        (glm::vec3*)normals.contiguous().data_ptr<float>(),
        (glm::vec3*)viewdirs.contiguous().data_ptr<float>(),
        (glm::vec3*)incidents_shs.contiguous().data_ptr<float>(),
        (glm::vec3*)direct_shs.contiguous().data_ptr<float>(),
        visibility_shs.contiguous().data_ptr<float>(),
        sample_num, is_training,
        rand_float.contiguous().data_ptr<float>(),
        (glm::vec3*)incident_dirs.contiguous().data_ptr<float>(),
        (glm::vec3*)pbr.contiguous().data_ptr<float>(),
        (glm::vec3*)diffuse_light.contiguous().data_ptr<float>()
    );
    return std::make_tuple(pbr, incident_dirs, diffuse_light);
}