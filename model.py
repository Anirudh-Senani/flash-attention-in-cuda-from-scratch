"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // TODO: implement elementwise c[i] = a[i] + b[i]
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    // TODO: multiply each element of a by scalar in place
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) a[i] = a[i] * scalar;
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) a[i] = expf(a[i]);
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= rows) return;

    float max_val = -INFINITY;
    for (int c=0; c<cols; c+=1){
            max_val = fmaxf(matrix[r * cols + c], max_val);
    }

    // extern __shared__ float shared_data[];
    // shared_data[threadIdx.x] = max_val;
    // __syncthreads();

    // for (int offset=blockDim.x/2; offset>0; offset /=2){
    //     if (threadIdx.x < offset){
    //         shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + offset]);
    //     }
    //     __syncthreads();
    // }

    // if (threadIdx.x == 0){
    //     out[r] = shared_data[0];
    // }

    out[r] = max_val;
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r
    // int r = blockIdx.x * blockDim.x + threadIdx.x;

    // if (r >= rows) return;

    // float row_sum = 0.0;

    // for (int c=0; c < cols; c++){
    //     row_sum += matrix[r * cols + c];
    // }

    // out[r] = row_sum;

    int r = blockIdx.x;

    if (r >= rows) return;

    float row_sum = 0.0;
    for (int c = threadIdx.x; c<cols; c+=blockDim.x){
        row_sum += matrix[r * cols + c];
    }

    extern __shared__ float shared_data[];
    shared_data[threadIdx.x] = row_sum;
    __syncthreads();

    for (int offset=blockDim.x/2; offset > 0; offset/=2){
        if (threadIdx.x < offset) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        out[r] = shared_data[0];
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    // TODO: return the dot product of a and b
    // int i = threadIdx.x;
    
    // if (i >= n) return 0.0f;

    float dp = 0.0f;
    for (int i=0; i<n; i++){
        dp += a[i]*b[i];
    }
    return dp;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) return;

    float cij = 0.0f;
    for (int j=0; j<k; j++){
        cij += a[row*k + j] * b[j*n + col];
    }

    c[row*n + col] = cij;

    // int col = blockIdx.x;
    // int row = blockIdx.y;

    // if (row >= m || col >= n) return;

    // float cij = 0.0f;
    // for (int i=threadIdx.y,j=threadIdx.x; i<k&&j<k; i+=blockDim.y,j+=blockDim.x){
    //     cij += a[row*k + i] * b[j*n + col];
    // }

    // extern __shared__ float shared_data[];
    // shared_data[row*n + col] = cij;
    // __syncthreads();

    // for (int offset=(blockDim.x+blockDim.y)/2; offset>0; offset /= 2){
    //     if (threadIdx.x + threadIdx.y)
    // }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // TODO: write out[c*rows + r] = in[r*cols + c]
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    out[col * rows + row] = in[row * cols + col];
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    // TODO: compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= seq_len || j >= seq_len) return;

    float scale = sqrtf((float)head_dim);

    // float sij = 0.0f;
    // for (int h=0; h<head_dim; h++){
    //     sij += q[i*head_dim + h] * k[j*head_dim + h];
    // }

    scores[i*seq_len + j] = dot_product(&q[i*head_dim], &k[j*head_dim], head_dim)/scale;
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    // TODO: implement numerically stable row-wise softmax in place
    int i = blockIdx.x;

    if (i >= rows) return;

    float rmax = -INFINITY;
    for (int j=threadIdx.x; j<cols; j+=blockDim.x){
        rmax = fmaxf(matrix[i*cols + j], rmax);
    }

    extern __shared__ float shared_data[];
    shared_data[threadIdx.x] = rmax;
    __syncthreads();

    for (int offset=blockDim.x/2; offset > 0; offset /= 2){
        if (threadIdx.x < offset){
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    float rsum = 0;
    for (int j=threadIdx.x; j<cols; j+=blockDim.x){
        matrix[i*cols + j] = __expf(matrix[i*cols + j] - shared_data[0]);
        rsum += matrix[i*cols + j];
    }

    // extern __shared__ float shared_sum[];
    shared_data[threadIdx.x] = rsum;
    __syncthreads();

    for (int offset=blockDim.x/2; offset > 0; offset /= 2){
        if (threadIdx.x < offset){
            shared_data[threadIdx.x] += shared_data[threadIdx.x + offset];
        }
        __syncthreads();
    }

    for (int j=threadIdx.x; j<cols; j+=blockDim.x){
        matrix[i*cols + j] /= shared_data[0];
    }

}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    // TODO: compute out[i, d] = sum_j p[i, j] * v[j, d]
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= seq_len || d >= head_dim) return;

    float oid = 0.0f;
    for (int j=0; j<seq_len; j++){
        oid += p[i*seq_len + j] * v[j*head_dim + d];
    }

    out[i*head_dim + d] = oid;
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    // TODO: allocate scratch, launch qk_scores -> softmax_rows -> pv_matmul, free scratch
    size_t qkv_size = seq_len * head_dim * sizeof(float);
    size_t attn_size = seq_len * seq_len * sizeof(float);
    size_t shared_size = 128 * sizeof(float);

    float *q, *k, *v, *p, *out;
    cudaMalloc((void**)&q, qkv_size);
    cudaMalloc((void**)&k, qkv_size);
    cudaMalloc((void**)&v, qkv_size);
    cudaMalloc((void**)&p, attn_size);
    cudaMalloc((void**)&out, qkv_size);

    cudaMemcpy(q, d_q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(k, d_k, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(v, d_v, qkv_size, cudaMemcpyHostToDevice);

    dim3 qkThreadsPerBlock(16, 16);
    dim3 qkNumBlocks((seq_len + qkThreadsPerBlock.x - 1)/qkThreadsPerBlock.x, 
                     (seq_len + qkThreadsPerBlock.y - 1)/qkThreadsPerBlock.y);
    
    qk_scores<<<qkNumBlocks, qkThreadsPerBlock>>>(q, k, p, seq_len, head_dim);
    softmax_rows<<<seq_len, 128, shared_size>>>(p, seq_len, seq_len);

    dim3 pvThreadsPerBlock(16, 16);
    dim3 pvNumBlocks((head_dim + pvThreadsPerBlock.x - 1)/pvThreadsPerBlock.x,
                     (seq_len + pvThreadsPerBlock.y - 1)/pvThreadsPerBlock.y);
    
    pv_matmul<<<pvNumBlocks, pvThreadsPerBlock>>>(p, v, out, seq_len, head_dim);

    cudaDeviceSynchronize();

    cudaMemcpy(d_out, out, qkv_size, cudaMemcpyDeviceToHost);

    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(p);
    cudaFree(out);

}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    // TODO: return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}

# Step 14 - correction_factor
__device__ float correction_factor(float old_max, float new_max) {
    // TODO: return the scalar used to rescale running statistics
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    // TODO: combine the rescaled old sum with the new block sum
    return old_sum * correction + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    // TODO: multiply each of the head_dim entries of out_row by correction in place
    for (int i=0; i<head_dim; ++i){
        out_row[i] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    // TODO: cooperatively copy the tile into shared_dst, zero-filling out-of-bounds positions.
    int total = tile_rows * tile_cols;
    for (int idx=thread_id; idx<total; idx += num_threads){
        int local_row = idx/tile_cols;
        int local_col = idx%tile_cols;
        int global_row = local_row + src_row_start;
        int global_col = local_col + src_col_start;
        if (global_row < src_rows && global_col < src_cols){
            shared_dst[idx] = src[global_row*src_cols + global_col];
        } else{
            shared_dst[idx] = 0.0f;
        }
    }
}

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill s_tile[i, j] = scale * dot(q_tile[i, :], k_tile[j, :])
    int total = tile_q * tile_k;
    for (int idx=thread_id; idx<total; idx+=num_threads){
        int row = idx/tile_k;
        int col = idx%tile_k;
        s_tile[idx] = scale * dot_product(&q_tile[row*head_dim], &k_tile[col*head_dim], head_dim);
    }
}

# Step 19 - tile_rowmax
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    // TODO: write row_max_out[r] = max over c of s_tile[r, c]
    for (int row=thread_id; row<tile_q; row+=num_threads){
        float m = -INFINITY;
        for (int col=0; col<tile_k; ++col){
            m = fmaxf(m, s_tile[row*tile_k + col]);
        }
        row_max_out[row] = m;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // TODO: for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])
    for (int row=thread_id; row<tile_q; row += num_threads){
        for (int col=0; col<tile_k; ++col){
            s_tile[row*tile_k + col] = expf(s_tile[row*tile_k + col] - row_max[row]);
        }
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    // TODO: cooperatively fill row_sum_out[r] with the sum of p_tile row r
    for (int row=thread_id; row<tile_q; row += num_threads){
        float row_sum = 0.0f;
        for (int col=0; col<tile_k; ++col){
            row_sum += p_tile[row*tile_k + col];
        }
        row_sum_out[row] = row_sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    // TODO: cooperatively add P_tile * V_tile into out_acc
    int total = tile_q * head_dim;
    for (int idx=thread_id; idx<total; idx+=num_threads){
        int row = idx/head_dim;
        int col = idx%head_dim;
        float acc = 0.0f;
        for (int k=0; k<tile_k; ++k){
            acc += p_tile[row*tile_k + k] * v_tile[k*head_dim + col];
        }
        out_acc[row*head_dim + col] += acc;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nt  = blockDim.x * blockDim.y;
    int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
    int num_k_tiles = (seq_len + tile_k - 1) / tile_k;

    extern __shared__ float smem[];
    float* s_q       = smem;                          // tile_q * head_dim
    float* s_kv      = s_q + tile_q * head_dim;       // tile_k * head_dim (K/V share)
    float* s_scores  = s_kv      + tile_k * head_dim; // tile_q * tile_k
    float* s_tilemax = s_scores  + tile_q * tile_k;   // tile_q
    float* s_tilesum = s_tilemax + tile_q;            // tile_q
    float* s_max     = s_tilesum + tile_q;            // tile_q
    float* s_sum     = s_max     + tile_q;            // tile_q
    float* s_out     = s_sum     + tile_q;            // tile_q * head_dim

    for (int row = blockIdx.x; row < num_q_tiles; row += gridDim.x) {
        for (int i = tid; i < tile_q; i += nt) {
            s_max[i] = -INFINITY;
            s_sum[i] = 0.0f;
        }
        for (int i = tid; i < tile_q * head_dim; i += nt)
            s_out[i] = 0.0f;
        __syncthreads();

        load_tile(q, s_q, row * tile_q, 0, seq_len, head_dim, tile_q, head_dim, tid, nt);
        __syncthreads();

        // FIX: reduce over ALL k-tiles within this block (do not split by gridDim.y).
        for (int col = 0; col < num_k_tiles; col++) {
            load_tile(k, s_kv, col * tile_k, 0, seq_len, head_dim, tile_k, head_dim, tid, nt);
            __syncthreads();

            tile_scores(s_q, s_kv, s_scores, tile_q, tile_k, head_dim, scale, tid, nt);
            __syncthreads();

            for (int idx = tid; idx < tile_q * tile_k; idx += nt) {
                int kj = (idx % tile_k) + col * tile_k;   // global K index
                if (kj >= seq_len)
                    s_scores[idx] = -INFINITY;
            }
            __syncthreads();

            tile_rowmax(s_scores, s_tilemax, tile_q, tile_k, tid, nt);
            __syncthreads();

            for (int i = tid; i < tile_q; i += nt) {
                float m_old = s_max[i];
                float m_new = online_max(m_old, s_tilemax[i]);
                float correction = correction_factor(m_old, m_new);
                s_max[i] = m_new;
                s_tilemax[i] = correction;   // stash correction
            }
            __syncthreads();

            tile_exp(s_scores, s_max, tile_q, tile_k, tid, nt);
            __syncthreads();

            tile_rowsum(s_scores, s_tilesum, tile_q, tile_k, tid, nt);
            __syncthreads();

            for (int i = tid; i < tile_q; i += nt) {
                float c = s_tilemax[i];
                s_sum[i] = update_running_sum(s_sum[i], c, s_tilesum[i]);
                rescale_output(&s_out[i * head_dim], head_dim, c);
            }
            __syncthreads();

            load_tile(v, s_kv, col * tile_k, 0, seq_len, head_dim, tile_k, head_dim, tid, nt);
            __syncthreads();

            accumulate_pv(s_scores, s_kv, s_out, tile_q, tile_k, head_dim, tid, nt);
            __syncthreads();
        }

        // FIX: normalize by the running denominator before writing out.
        for (int i = tid; i < tile_q * head_dim; i += nt) {
            int qi = i / head_dim;
            int d  = i % head_dim;
            int gi = row * tile_q + qi;
            if (gi < seq_len) {
                float denom = s_sum[qi];
                float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
                out[gi * head_dim + d] = s_out[i] * inv;
            }
        }
        __syncthreads();
    }
}

# Step 24 - flash_attention_launcher
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    // TODO: configure grid/block/shared memory and launch flash_attention_kernel
    int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
    int num_k_tiles = (seq_len + tile_k - 1) / tile_k;

    size_t qkv_size = seq_len * head_dim * sizeof(float);
    size_t q_tile = tile_q * head_dim * sizeof(float);
    size_t kv_tile = tile_k * head_dim * sizeof(float);
    size_t qk_tile = tile_k * tile_q * sizeof(float);
    size_t acc = 4 * tile_q * sizeof(float);
    size_t shared = 2*q_tile + kv_tile + qk_tile + acc;

    float scale = 1.0f/sqrt(head_dim);
    // float *q, *k, *v, *out;
    // cudaMalloc((void**)&q, qkv_size);
    // cudaMalloc((void**)&k, qkv_size);
    // cudaMalloc((void**)&v, qkv_size);
    // cudaMalloc((void**)&out, qkv_size);

    // cudaMemcpy(q, d_q, qkv_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(k, d_k, qkv_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(v, d_v, qkv_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(128);
    dim3 numBlocks(num_q_tiles);

    // flash_attention_kernel<<<numBlocks, threadsPerBlock, shared>>>(q, k, v, out, seq_len, head_dim, tile_q, tile_k, scale);
    flash_attention_kernel<<<numBlocks, threadsPerBlock, shared>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale);

    cudaDeviceSynchronize();

    // cudaMemcpy(d_out, out, qkv_size, cudaMemcpyDeviceToHost);

    // cudaFree(q);
    // cudaFree(k);
    // cudaFree(v);
    // cudaFree(out);

}

# Step 25 - causal_mask
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    // TODO: write -INFINITY into entries where the global key index exceeds the global query index.
    for (int row = thread_id; row<tile_q; row += num_threads){
        for (int col=0; col<tile_k; col++){
            if (q_row_start + row < k_col_start + col){
                s_tile[row*tile_k + col] = -INFINITY;
            }
        }
    }
}

# Step 26 - flash_attention_causal_kernel
__global__ void flash_attention_causal_kernel(const float* q, const float* k, const float* v,
                                                float* out, int seq_len, int head_dim,
                                                int tile_q, int tile_k, float scale) {
    // TODO: tiled causal flash attention using shared memory and online softmax
    extern __shared__ float smem[];

    float* s_q = smem;
    float* s_kv = s_q + (tile_q * head_dim);
    float* s_score = s_kv + (tile_k * head_dim);
    float* s_out = s_score + (tile_q * tile_k);
    float* s_max = s_out + (tile_q * head_dim);
    float* s_sum = s_max + tile_q;
    float* s_tilemax = s_sum + tile_q;
    float* s_tilesum = s_tilemax + tile_q;

    int tid = threadIdx.x;
    int row = blockIdx.x * tile_q;
    int nt = blockDim.x;

    if (row >= seq_len) return;

    load_tile(q, s_q, row, 0, seq_len, head_dim, tile_q, head_dim, tid, nt);
    __syncthreads();

    for(int i=tid; i<tile_q; i+=nt){
        s_max[i] = -INFINITY;
        s_sum[i] = 0.0f;
    }
    for (int i=tid; i<tile_q*head_dim; i+=nt)
        s_out[i] = 0.0f;
    __syncthreads();

    for (int col=0; col<seq_len; col+=tile_k){
        load_tile(k, s_kv, col, 0, seq_len, head_dim, tile_k, head_dim, tid, nt);
        __syncthreads();

        tile_scores(s_q, s_kv, s_score, tile_q, tile_k, head_dim, scale, tid, nt);
        __syncthreads();

        causal_mask(s_score, row, col, tile_q, tile_k, tid, nt);
        __syncthreads();

        tile_rowmax(s_score, s_tilemax, tile_q, tile_k, tid, nt);
        __syncthreads();

        for (int i=tid; i<tile_q; i+=nt){
            float old_max = s_max[i];
            float new_max = online_max(old_max, s_tilemax[i]);
            float correction = correction_factor(old_max, new_max);
            s_max[i] = new_max;
            s_tilemax[i] = correction;
        }
        __syncthreads();

        tile_exp(s_score, s_max, tile_q, tile_k, tid, nt);
        __syncthreads();

        tile_rowsum(s_score, s_tilesum, tile_q, tile_k, tid, nt);
        __syncthreads();

        for (int i=tid; i<tile_q; i+=nt){
            float c = s_tilemax[i];
            s_sum[i] = update_running_sum(s_sum[i], c, s_tilesum[i]);
            rescale_output(&s_out[i*head_dim], head_dim, c);
        }
        __syncthreads();

        load_tile(v, s_kv, col, 0, seq_len, head_dim, tile_k, head_dim, tid, nt);
        __syncthreads();

        accumulate_pv(s_score, s_kv, s_out, tile_q, tile_k, head_dim, tid, nt);
        __syncthreads();

    }

    for (int i=tid; i<tile_q*head_dim; i+=nt){
        int ri = i/head_dim;
        int ci = i%head_dim;
        int gi = row + ri;
        if (gi < seq_len){
            float denom = s_sum[ri];
            float inv = (denom > 0.0f) ? 1.0f/denom : 0.0f;
            out[gi * head_dim + ci] = s_out[i] * inv;
        }
    }
    __syncthreads();

}

