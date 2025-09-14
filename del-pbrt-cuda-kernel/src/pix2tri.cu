#include <cfloat>
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include "tri3.h"
#include "aabb.h"
#include "ray_for_pixel.h"


extern "C" {
__global__
void fwd_pix2tri(
  uint32_t *pix2tri,
  const uint32_t num_tri,
  const uint32_t *tri2vtx,
  const float *vtx2xyz,
  const uint32_t img_w,
  const uint32_t img_h,
  const float *transform_ndc2world,
  const uint32_t *bvhnodes,
  const float *aabbs)
{
    int i_pix = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_pix >= img_w * img_h ){ return; }
    //
    auto ray = ray_for_pixel(i_pix, img_w, img_h, transform_ndc2world);
    float hit_depth = FLT_MAX;
    uint32_t hit_idxtri = UINT32_MAX;

/*
    for(int i_tri=0;i_tri<num_tri;++i_tri){
        const float* p0 = vtx2xyz + tri2vtx[i_tri*3+0]*3;
        const float* p1 = vtx2xyz + tri2vtx[i_tri*3+1]*3;
        const float* p2 = vtx2xyz + tri2vtx[i_tri*3+2]*3;
        const auto res = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
        if(!res){ continue; }
        float depth = res.value();
        if( depth < hit_depth ){
          hit_depth = depth;
          hit_idxtri = i_tri;
        }
    }
    pix2tri[i_pix] = hit_idxtri;
    return;
*/

    constexpr int STACK_SIZE = 512;
    uint32_t stack[STACK_SIZE];
    int32_t i_stack = 1;
    stack[0] = 0;
    while( i_stack > 0 ){
        uint32_t i_bvhnode = stack[i_stack-1];
        assert(i_bvhnode < num_tri*2-1);
        --i_stack;
        if( !aabb::is_intersect_ray<3>(aabbs + i_bvhnode*6, ray.first.data(), ray.second.data() ) ){
              continue;
        }
        if( bvhnodes[i_bvhnode * 3 + 2] == UINT32_MAX ){
            const uint32_t i_tri = bvhnodes[i_bvhnode * 3 + 1];
            assert(i_tri<num_tri);
            const float* p0 = vtx2xyz + tri2vtx[i_tri*3+0]*3;
            const float* p1 = vtx2xyz + tri2vtx[i_tri*3+1]*3;
            const float* p2 = vtx2xyz + tri2vtx[i_tri*3+2]*3;
            const auto r = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
            if(r) { // hit triangle
                float depth = r.value();
                if( depth < hit_depth ){
                    hit_depth = depth;
                    hit_idxtri = i_tri;
                }
            }
            continue;
        }
        stack[i_stack] = bvhnodes[i_bvhnode * 3 + 1];
        ++i_stack;
        stack[i_stack] = bvhnodes[i_bvhnode * 3 + 2];
        ++i_stack;
        assert(i_stack<STACK_SIZE);
    }
    pix2tri[i_pix] = hit_idxtri;
}

}