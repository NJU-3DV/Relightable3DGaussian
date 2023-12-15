#include "trace.cuh"

struct id_intersection{
    int32_t id;
    float2 intersection;
};

std::tuple<int32_t, thrust::device_vector<int32_t>, thrust::device_vector<float3>, thrust::device_vector<int32_t>>
trace_bvh_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, int32_t* num_contributes){
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        float milliseconds = 0;


        auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);

        thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_rays),
                     [nodes, aabbs_internal, rays_o, rays_d, num_contributes] __device__ (int32_t idx)
                     {
                         IndexStack<int32_t> stack_device;
                         stack_device.push(0);
                         int32_t count = 0;
                         float3 ray_o=rays_o[idx],ray_d=rays_d[idx];
                         while (!stack_device.empty()) {
                             int32_t node_id = stack_device.pop();
                             int32_t* node = nodes + node_id*5;
                             if (node[4] <= 4){
                                 count += node[4];
                             }else{
                                 int32_t lid = node[1], rid = node[2];
                                 float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                                 float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                                 if (interection_l.y > interection_r.y){
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                 }else{
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                 }
                             }
                         }
                         num_contributes[idx] = count;
                     });

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "num_contributes time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);

    thrust::device_vector<int32_t> ray_offsets_vec(num_rays);
    thrust::device_ptr<int32_t> num_contributes_ptr = thrust::device_pointer_cast(num_contributes);

    thrust::inclusive_scan(num_contributes_ptr, num_contributes_ptr+num_rays, ray_offsets_vec.begin());
    int32_t* ray_offsets = thrust::raw_pointer_cast(ray_offsets_vec.data());
    int32_t num_rendered;
    cudaMemcpy(&num_rendered, ray_offsets + num_rays - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    // printf("num_rendered: %d\n",num_rendered);

    thrust::device_vector<uint64_t> ray_list_key_vec(num_rendered);
    uint64_t* ray_list_key = thrust::raw_pointer_cast(ray_list_key_vec.data());

    thrust::device_vector<int32_t> point_list_vec(num_rendered);
    int32_t* point_list = thrust::raw_pointer_cast(point_list_vec.data());
    thrust::device_vector<float3> position_list_vec(num_rendered);
    float3* position_list = thrust::raw_pointer_cast(position_list_vec.data());
    thrust::device_vector<int32_t> ray_id_list_vec(num_rendered);
    int32_t* ray_id_list = thrust::raw_pointer_cast(ray_id_list_vec.data());

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "inclusive_scan time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(num_rays),
                     [nodes, aabbs_internal, rays_o, rays_d, num_contributes,
                      ray_offsets, ray_list_key, point_list, ray_id_list,
                             means3D, covs3D, position_list
                     ] __device__ (uint32_t idx)
                     {
                         if (num_contributes[idx]==0) return;
                         uint32_t offset = (idx == 0) ? 0 : ray_offsets[idx - 1];
                         uint64_t* ray_key_ptr=ray_list_key+offset;
                         int32_t* point_ptr=point_list+offset;
                         float3* position_ptr=position_list+offset;
                         int32_t* ray_id_ptr=ray_id_list+offset;

                         IndexStack<id_intersection> stack_device;
                         stack_device.push({0, {-1000, 1000}});
                         int32_t count = 0;
                         float3 ray_o=rays_o[idx],ray_d=rays_d[idx];
                         while (!stack_device.empty()) {
                             id_intersection pop_result = stack_device.pop();
                             int32_t node_id = pop_result.id;
                             float2 intersection = pop_result.intersection;
                             int32_t* node = nodes + node_id*5;
                             if (node[4] <= 4){
                                 IndexStack<int32_t> stack2;
                                 stack2.push(node_id);
                                 int32_t count2 = 0;
                                 int32_t count2_total = node[4];
                                 while (!stack2.empty()) {
                                     int32_t node_id = stack2.pop();
                                     int32_t* node = nodes + node_id*5;
                                     if (node[3] >= 0){

                                         int32_t object_id = node[3];
                                         float t = ray_intersects(means3D[object_id], ray_o, ray_d);
                                         // float t = ray_intersects(means3D[object_id], covs3D + object_id*6, ray_o, ray_d);
                                         if (t<0.01 || t < intersection.x || t > intersection.y){
                                            t = 1000000.f;
                                            object_id = -1;
                                         }
                                         uint64_t key = idx;
                                         key <<= 32;
                                         key |= *(uint32_t*)&t;
                                         ray_key_ptr[count] = key;
                                         point_ptr[count] = object_id;
                                         ray_id_ptr[count] = idx;
                                         position_ptr[count] = {
                                                 ray_o.x + t * ray_d.x,
                                                 ray_o.y + t * ray_d.y,
                                                 ray_o.z + t * ray_d.z,
                                         };
                                         ++count2;
                                         ++count;
                                     }else{
                                         stack2.push(node[1]);
                                         stack2.push(node[2]);
                                     }
                                 }
                                 assert(count2 == count2_total);



                             }else{
                                 int32_t lid = node[1], rid = node[2];
                                 float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                                 float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                                 if (interection_l.y > interection_r.y){
                                     if (interection_l.y>0){
                                         stack_device.push({lid, interection_l});
                                     }
                                     if (interection_r.y>0){
                                         stack_device.push({rid, interection_r});
                                     }
                                 }else{
                                     if (interection_r.y>0){
                                         stack_device.push({rid, interection_r});
                                     }
                                     if (interection_l.y>0){
                                         stack_device.push({lid, interection_l});
                                     }
                                 }
                             }
                         }
                     });
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "trace time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);
    thrust::stable_sort_by_key(ray_list_key_vec.begin(),
                               ray_list_key_vec.end(),
                               thrust::make_zip_iterator(
                                       thrust::make_tuple(point_list_vec.begin(),
                                                          position_list_vec.begin())));
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "sort time: " << milliseconds << " ms" << std::endl;
//     cudaEventRecord(start);
    return std::make_tuple(num_rendered, point_list_vec, position_list_vec, ray_id_list_vec);
}



void trace_bvh_opacity_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, float3* normals,
                    int32_t* num_contributes,
                    float* rendered_opacity){
//         cudaEvent_t start, stop;
//         cudaEventCreate(&start);
//         cudaEventCreate(&stop);
//         cudaEventRecord(start);
//         float milliseconds = 0;
        auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);
        thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_rays),
                     [nodes, aabbs_internal, rays_o, rays_d, num_contributes,
                     means3D, covs3D, opacities, normals, rendered_opacity] __device__ (int32_t idx)
                     {
                         IndexStack<int32_t> stack_device;
                         stack_device.push(0);
                         int32_t count = 0;
                         float3 ray_o=rays_o[idx],ray_d=rays_d[idx];
                         float ray_opacity = 1.0f;
                         while (!stack_device.empty()) {
                             int32_t node_id = stack_device.pop();
                             int32_t* node = nodes + node_id*5;
                             if (node[4] <= 1){  // 8
                                 IndexStack<int32_t> stack_device2;
                                 stack_device2.push(node_id);
                                 while (!stack_device2.empty()) {
                                    int32_t node_id2 = stack_device2.pop();
                                    int32_t* node2 = nodes + node_id2*5;
                                    if (node2[4] > 1){
                                        stack_device2.push(node2[1]);
                                        stack_device2.push(node2[2]);
                                    }else{
                                        int32_t object_id2 = node2[3];
                                        if (opacities[object_id2]<1.f/255.f) continue;
                                        float3 normal = normals[object_id2];
                                        if (normal.x*ray_d.x+normal.y*ray_d.y+normal.z*ray_d.z>0) continue;

                                        float t = ray_intersects(means3D[object_id2], covs3D + object_id2*6, ray_o, ray_d);
                                         if (t<0.01){
                                            continue;
                                         }
                                         float3 pos = {
                                                         ray_o.x + t * ray_d.x,
                                                         ray_o.y + t * ray_d.y,
                                                         ray_o.z + t * ray_d.z,
                                         };
                                         float power = gaussian_fn(means3D[object_id2], pos, covs3D + object_id2*6);
                                         if (power>0) continue;
                                         count += 1;
                                         float alpha = opacities[object_id2] * __expf(power);
                                         ray_opacity *= 1-alpha;
                                         if (ray_opacity<0.9) {
                                            rendered_opacity[idx] = 0.0f;
                                            return;
                                         }
                                    }
                                 }
                             }else{
                                 int32_t lid = node[1], rid = node[2];
                                 float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                                 float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                                 if (interection_l.y > interection_r.y){
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                 }else{
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                 }
                             }
                         }
                         num_contributes[idx] = count;
                         rendered_opacity[idx] = ray_opacity;
                     });

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "tracing time: " << milliseconds << " ms" << std::endl;
//     cudaEventRecord(start);
}