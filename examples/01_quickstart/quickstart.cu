// iket-cuda quickstart: a small tiled-sum kernel with every kind of IKET marker.
//
// Each block sums its tiles of `input` into `output[blockIdx.x]`:
//   load_tile    push/pop range with a payload (the tile index), one per tile
//   reduce_tile  push/pop range nested after it, one per tile
//   tile_done    point marker with a payload (the tile index)
//   block_life   start/end range around the whole block
//
// Build with -DIKET_ENABLE to trace, without it for the plain kernel (the
// markers then compile to nothing). See docs/getting-started.md.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include <iket/iket.cuh>

// Event table: one declaration per event, ids unique in this file (1..30).
IKET_DECLARE_RANGE(load_tile, 1, "load_tile", ::iket::Payload::U32);
IKET_DECLARE_RANGE(reduce_tile, 2, "reduce_tile", ::iket::Payload::None);
IKET_DECLARE_START_END(block_life, 3, "block_life", ::iket::Payload::None);
IKET_DECLARE_MARK(tile_done, 4, "tile_done", ::iket::Payload::U32);

constexpr int kTileElems = 4096;
constexpr int kThreads = 128;
constexpr int kBlocks = 64;
constexpr int kTilesPerBlock = 16;

__global__ void tile_sum(const float* input, float* output) {
    __shared__ float tile[kTileElems];
    __shared__ float warp_sums[kThreads / 32];

    IKET_RANGE_START(block_life);
    float block_sum = 0.0f;
    for (int t = 0; t < kTilesPerBlock; ++ t) {
        const unsigned tile_idx = blockIdx.x * kTilesPerBlock + t;

        IKET_RANGE_PUSH(load_tile, tile_idx);
        for (int i = threadIdx.x; i < kTileElems; i += kThreads)
            tile[i] = input[static_cast<size_t>(tile_idx) * kTileElems + i];
        __syncthreads();
        IKET_RANGE_POP();

        IKET_RANGE_PUSH(reduce_tile);
        float thread_sum = 0.0f;
        for (int i = threadIdx.x; i < kTileElems; i += kThreads)
            thread_sum += tile[i];
        for (int offset = 16; offset > 0; offset /= 2)
            thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
        if (threadIdx.x % 32 == 0)
            warp_sums[threadIdx.x / 32] = thread_sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < kThreads / 32; ++ w)
                block_sum += warp_sums[w];
        }
        __syncthreads();
        IKET_RANGE_POP();

        IKET_MARK(tile_done, tile_idx);
    }
    if (threadIdx.x == 0)
        output[blockIdx.x] = block_sum;
    IKET_RANGE_END(block_life);
}

int main() {
    const size_t num_elems = static_cast<size_t>(kBlocks) * kTilesPerBlock * kTileElems;
    std::vector<float> host_input(num_elems, 1.0f);
    float* input = nullptr;
    float* output = nullptr;
    cudaMalloc(&input, num_elems * sizeof(float));
    cudaMalloc(&output, kBlocks * sizeof(float));
    cudaMemcpy(input, host_input.data(), num_elems * sizeof(float), cudaMemcpyHostToDevice);

    // A warmup launch and a measured one; run-iket traces both.
    for (int launch = 0; launch < 2; ++ launch)
        tile_sum<<<kBlocks, kThreads>>>(input, output);
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(status));
        return EXIT_FAILURE;
    }

    std::vector<float> host_output(kBlocks);
    cudaMemcpy(host_output.data(), output, kBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    const float expected = static_cast<float>(kTilesPerBlock) * kTileElems;
    for (int b = 0; b < kBlocks; ++ b) {
        if (host_output[b] != expected) {
            std::fprintf(stderr, "block %d: got %f, expected %f\n", b, host_output[b], expected);
            return EXIT_FAILURE;
        }
    }
    std::printf("tile_sum OK: %d blocks x %d tiles\n", kBlocks, kTilesPerBlock);
    cudaFree(input);
    cudaFree(output);
    return EXIT_SUCCESS;
}
