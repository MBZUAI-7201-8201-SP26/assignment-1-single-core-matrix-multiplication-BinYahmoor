#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

using std::size_t;

// Row-major index helper
inline size_t idx(size_t r, size_t c, size_t cols) {
    return r * cols + c;
}

// Reference naive matmul: C = A * B
void ref_matmul(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                size_t M, size_t K, size_t N) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            float a = A[idx(i, k, K)];
            for (size_t j = 0; j < N; j++) {
                C[idx(i, j, N)] += a * B[idx(k, j, N)];
            }
        }
    }
}

// Single tile matrix multiplication: multiply (TH x TK) from A with (TK x TW) from B,
// accumulate into (TH x TW) tile in C.
void tile_matmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    size_t M, size_t K, size_t N,
    size_t row_offset,
    size_t col_offset,
    size_t k_offset,
    size_t TH,
    size_t TW,
    size_t TK
) {
    const size_t i_end = std::min(row_offset + TH, M);
    const size_t j_end = std::min(col_offset + TW, N);
    const size_t k_end = std::min(k_offset + TK, K);

    for (size_t i = row_offset; i < i_end; i++) {
        for (size_t k = k_offset; k < k_end; k++) {
            float a = A[idx(i, k, K)];
            for (size_t j = col_offset; j < j_end; j++) {
                C[idx(i, j, N)] += a * B[idx(k, j, N)];
            }
        }
    }
}

// Tiled matrix multiplication: C = A * B
std::vector<float> tiled_matrix_multiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    size_t M, size_t K, size_t N,
    size_t TH, size_t TW, size_t TK
) {
    std::vector<float> C(M * N, 0.0f);

    for (size_t row = 0; row < M; row += TH) {
        for (size_t col = 0; col < N; col += TW) {
            for (size_t kk = 0; kk < K; kk += TK) {
                tile_matmul(A, B, C, M, K, N, row, col, kk, TH, TW, TK);
            }
        }
    }

    return C;
}

float max_abs_diff(const std::vector<float>& X, const std::vector<float>& Y) {
    float m = 0.0f;
    for (size_t i = 0; i < X.size(); i++) {
        m = std::max(m, std::fabs(X[i] - Y[i]));
    }
    return m;
}

int main() {
    // A: 640x320, B: 320x640, C: 640x640
    const size_t M = 640, K = 320, N = 640;

    std::vector<float> A(M * K), B(K * N), Cref(M * N);

    // Fill A and B with random values
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Reference timing
    auto t0 = std::chrono::high_resolution_clock::now();
    ref_matmul(A, B, Cref, M, K, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Tiled timing (required: 32x32 tiles, and TK=32)
    const size_t tile = 32;
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<float> Ctiled = tiled_matrix_multiply(A, B, M, K, N, tile, tile, tile);
    auto t3 = std::chrono::high_resolution_clock::now();
    double tiled_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Correctness check
    float diff = max_abs_diff(Cref, Ctiled);

    std::cout << "Reference time: " << ref_ms << " ms\n";
    std::cout << "Tiled time (tile=" << tile << "): " << tiled_ms << " ms\n";
    std::cout << "Max abs diff: " << diff << "\n";

    // Try a few other tile sizes
    for (size_t ts : {8u, 16u, 32u, 64u}) {
        auto s0 = std::chrono::high_resolution_clock::now();
        auto Ctest = tiled_matrix_multiply(A, B, M, K, N, ts, ts, ts);
        auto s1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
        float d = max_abs_diff(Cref, Ctest);
        std::cout << "Tile " << ts << ": " << ms << " ms, diff=" << d << "\n";
    }

    return 0;
}
