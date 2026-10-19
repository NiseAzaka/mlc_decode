#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void main_kernel(const half_t* __restrict__ KV, const half_t* __restrict__ K_pe, const half_t* __restrict__ Q, const half_t* __restrict__ Q_pe, half_t* __restrict__ glse, half_t* __restrict__ output_partial);
extern "C" __global__ void main_kernel_1(half_t* __restrict__ Output, const half_t* __restrict__ glse, const half_t* __restrict__ output_partial);
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(const half_t* __restrict__ KV, const half_t* __restrict__ K_pe, const half_t* __restrict__ Q, const half_t* __restrict__ Q_pe, half_t* __restrict__ glse, half_t* __restrict__ output_partial) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_pe_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* K_pe_shared = ((void*)((char*)buf_dyn_shmem + 18432));
  void* KV_shared = ((void*)((char*)buf_dyn_shmem + 20480));
  void* S_shared = ((void*)((char*)buf_dyn_shmem + 36864));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 37376));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 37376));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 37376));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 37632));
  float acc_o[128];
  float logsum[1];
  float scores_max[1];
  float acc_s[4];
  float scores_max_prev[1];
  float scores_scale[1];
  float scores_sum[1];
  float scores_max_clear[1];
  half_t S_shared_local_cast[4];
  float scores_max_clear_1[1];
  half_t S_shared_local_cast_1[4];
  float inv_logsum[1];
  half_t output_partial_local_cast_2[4];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  logsum[0] = 0x0p+0f/*0.000000e+00*/;
  scores_max[0] = -MACART_INF_F;
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    *(uint4*)(((half_t*)Q_shared) + ((((((((int)threadIdx.x) >> 3) * 1024) + (i_1 * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + ((i_1 & 3) >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) & 1) + (i_1 & 1)) & 1) * 8))) = *(uint4*)(Q + ((i_1 * 512) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    *(uint4*)(((half_t*)Q_pe_shared) + (((((i_2 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q_pe + ((i_2 * 512) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(uint4*)(((half_t*)K_pe_shared) + (((((i_3 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K_pe + (((((int)blockIdx.z) * 8192) + (i_3 * 512)) + (((int)threadIdx.x) * 8)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    *(uint4*)(((half_t*)KV_shared) + ((((((((int)threadIdx.x) >> 3) * 1024) + (i_4 * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + ((i_4 & 3) >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) & 1) + (i_4 & 1)) & 1) * 8))) = *(uint4*)(KV + (((((int)blockIdx.z) * 65536) + (i_4 * 512)) + (((int)threadIdx.x) * 8)));
  }
  for (int k = 0; k < 7; ++k) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    half_t A_local[4];
    half_t B_local[4];
    __syncthreads();
    for (int ki = 0; ki < 32; ++ki) {
      *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)Q_shared) + (((((((ki >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      *(uint2*)(B_local + 0) = *(uint2*)(((half_t*)KV_shared) + (((((((ki >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + 0),
                    *(((float16x4*)A_local) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    half_t A_local_1[4];
    half_t B_local_1[4];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      *(uint2*)(A_local_1 + 0) = *(uint2*)(((half_t*)Q_pe_shared) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      *(uint2*)(B_local_1 + 0) = *(uint2*)(((half_t*)K_pe_shared) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + 0),
                    *(((float16x4*)A_local_1) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      *(uint4*)(((half_t*)K_pe_shared) + (((((i_5 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K_pe + (((((((int)blockIdx.z) * 8192) + (k * 1024)) + (i_5 * 512)) + (((int)threadIdx.x) * 8)) + 1024));
    }
    scores_max_prev[0] = scores_max[0];
    scores_max_clear[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      scores_max_clear[0] = max(scores_max_clear[0], acc_s[rv]);
    }
    __syncthreads();
    scores_max_clear[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(scores_max_clear[0], (&(((float*)workspace_3)[0])));
    scores_max[0] = max(scores_max[0], scores_max_clear[0]);
    scores_scale[0] = exp2f(((scores_max_prev[0] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[0] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      acc_s[i_6] = exp2f(((acc_s[i_6] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[0] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
    }
    scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      scores_sum[0] = (scores_sum[0] + acc_s[rv_1]);
    }
    scores_sum[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(scores_sum[0], (&(((float*)workspace_2)[0])));
    uint2 __1;
    float4 v_ = *(float4*)(acc_s + 0);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(S_shared_local_cast + 0) = __1;
    __syncthreads();
    *(uint2*)(((half_t*)S_shared) + ((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(S_shared_local_cast + 0);
    logsum[0] = ((logsum[0] * scores_scale[0]) + scores_sum[0]);
    #pragma unroll
    for (int i_7 = 0; i_7 < 128; ++i_7) {
      acc_o[i_7] = (acc_o[i_7] * scores_scale[0]);
    }
    half_t A_local_2[4];
    half_t B_local_2[128];
    *(uint2*)(A_local_2 + 0) = *(uint2*)(((half_t*)S_shared) + ((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    for (int j = 0; j < 32; ++j) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        B_local_2[((j * 4) + local_id)] = ((half_t*)KV_shared)[((((((((j >> 2) * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j & 3) >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
      }
    }
    for (int j_1 = 0; j_1 < 32; ++j_1) {
      {
      *(((float32x4*)acc_o) + j_1) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_2) + j_1),
                    *(((float16x4*)A_local_2) + 0),
                    *(((float32x4*)acc_o) + j_1));
    };
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      *(uint4*)(((half_t*)KV_shared) + ((((((((int)threadIdx.x) >> 3) * 1024) + (i_8 * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_8 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + ((i_8 & 3) >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) & 1) + (i_8 & 1)) & 1) * 8))) = *(uint4*)(KV + (((((((int)blockIdx.z) * 65536) + (k * 8192)) + (i_8 * 512)) + (((int)threadIdx.x) * 8)) + 8192));
    }
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(acc_s + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  half_t A_local_3[4];
  half_t B_local_3[4];
  __syncthreads();
  for (int ki_2 = 0; ki_2 < 32; ++ki_2) {
    *(uint2*)(A_local_3 + 0) = *(uint2*)(((half_t*)Q_shared) + (((((((ki_2 >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    *(uint2*)(B_local_3 + 0) = *(uint2*)(((half_t*)KV_shared) + (((((((ki_2 >> 2) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_3) + 0),
                    *(((float16x4*)A_local_3) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
  }
  half_t A_local_4[4];
  half_t B_local_4[4];
  for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
    *(uint2*)(A_local_4 + 0) = *(uint2*)(((half_t*)Q_pe_shared) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    *(uint2*)(B_local_4 + 0) = *(uint2*)(((half_t*)K_pe_shared) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_4) + 0),
                    *(((float16x4*)A_local_4) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
  }
  scores_max_prev[0] = scores_max[0];
  scores_max_clear_1[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
    scores_max_clear_1[0] = max(scores_max_clear_1[0], acc_s[rv_2]);
  }
  __syncthreads();
  scores_max_clear_1[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(scores_max_clear_1[0], (&(((float*)workspace_1)[0])));
  scores_max[0] = max(scores_max[0], scores_max_clear_1[0]);
  scores_scale[0] = exp2f(((scores_max_prev[0] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[0] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    acc_s[i_9] = exp2f(((acc_s[i_9] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (scores_max[0] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  }
  scores_sum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
    scores_sum[0] = (scores_sum[0] + acc_s[rv_3]);
  }
  __syncthreads();
  scores_sum[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(scores_sum[0], (&(((float*)workspace)[0])));
  uint2 __2;
  float4 v__1 = *(float4*)(acc_s + 0);
  ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
  ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
  *(uint2*)(S_shared_local_cast_1 + 0) = __2;
  __syncthreads();
  *(uint2*)(((half_t*)S_shared) + ((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(S_shared_local_cast_1 + 0);
  logsum[0] = ((logsum[0] * scores_scale[0]) + scores_sum[0]);
  #pragma unroll
  for (int i_10 = 0; i_10 < 128; ++i_10) {
    acc_o[i_10] = (acc_o[i_10] * scores_scale[0]);
  }
  half_t A_local_5[4];
  half_t B_local_5[128];
  *(uint2*)(A_local_5 + 0) = *(uint2*)(((half_t*)S_shared) + ((((((int)threadIdx.x) & 15) * 16) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
  for (int j_2 = 0; j_2 < 32; ++j_2) {
    for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
      B_local_5[((j_2 * 4) + local_id_1)] = ((half_t*)KV_shared)[((((((((j_2 >> 2) * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + (local_id_1 * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j_2 & 3) >> 1)) & 1) * 32)) + ((((local_id_1 >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_1 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
    }
  }
  for (int j_3 = 0; j_3 < 32; ++j_3) {
    {
      *(((float32x4*)acc_o) + j_3) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_5) + j_3),
                    *(((float16x4*)A_local_5) + 0),
                    *(((float32x4*)acc_o) + j_3));
    };
  }
  inv_logsum[0] = (0x1p+0f/*1.000000e+00*/ / logsum[0]);
  #pragma unroll
  for (int i_11 = 0; i_11 < 128; ++i_11) {
    acc_o[i_11] = (acc_o[i_11] * inv_logsum[0]);
  }
  logsum[0] = (log2f(logsum[0]) + (scores_max[0] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/));
  if ((((int)threadIdx.x) >> 4) == 0) {
    glse[(((((int)threadIdx.x) & 15) * 128) + ((int)blockIdx.z))] = ((half_t)logsum[0]);
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 32; ++i_12) {
    uint2 __3;
    float4 v__2 = *(float4*)(acc_o + (i_12 * 4));
    ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
    ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
    *(uint2*)(output_partial_local_cast_2 + 0) = __3;
    *(uint2*)(output_partial + (((((((int)threadIdx.x) & 15) * 36864) + (((int)blockIdx.z) * 576)) + (i_12 * 16)) + ((((int)threadIdx.x) >> 4) * 4))) = *(uint2*)(output_partial_local_cast_2 + 0);
  }
}

extern "C" __global__ void __launch_bounds__(64, 1) main_kernel_1(half_t* __restrict__ Output, const half_t* __restrict__ glse, const half_t* __restrict__ output_partial) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* glse_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 128));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 128));
  float o_accum_local[1];
  float lse_frag[1];
  float lse_max_local[1];
  float lse_sum_local[1];
  float lse_logsum_local = 0x0p+0f/*0.000000e+00*/;
  float scale_local = 0x0p+0f/*0.000000e+00*/;
  half_t po_local[1];
  o_accum_local[0] = 0x0p+0f/*0.000000e+00*/;
  ((half_t*)glse_shared)[((int)threadIdx.x)] = glse[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))];
  lse_frag[0] = ((float)((half_t*)glse_shared)[((int)threadIdx.x)]);
  lse_max_local[0] = -MACART_INF_F;
  lse_max_local[0] = max(lse_max_local[0], lse_frag[0]);
  __syncthreads();
  lse_max_local[0] = tl::AllReduce<tl::MaxOp, 64, 1, 0>::run(lse_max_local[0], (&(((float*)workspace_1)[0])));
  lse_frag[0] = exp2f((lse_frag[0] - lse_max_local[0]));
  lse_sum_local[0] = 0x0p+0f/*0.000000e+00*/;
  lse_sum_local[0] = (lse_sum_local[0] + lse_frag[0]);
  __syncthreads();
  lse_sum_local[0] = tl::AllReduce<tl::SumOp, 64, 1, 0>::run(lse_sum_local[0], (&(((float*)workspace)[0])));
  lse_logsum_local = (log2f(lse_sum_local[0]) + lse_max_local[0]);
  __syncthreads();
  for (int k = 0; k < 64; ++k) {
    scale_local = exp2f((((float)((half_t*)glse_shared)[k]) - lse_logsum_local));
    po_local[0] = output_partial[((((((int)blockIdx.x) * 36864) + (k * 576)) + (((int)blockIdx.z) * 64)) + ((int)threadIdx.x))];
    o_accum_local[0] = (o_accum_local[0] + (((float)po_local[0]) * scale_local));
  }
  Output[(((((int)blockIdx.x) * 512) + (((int)blockIdx.z) * 64)) + ((int)threadIdx.x))] = ((half_t)o_accum_local[0]);
}

