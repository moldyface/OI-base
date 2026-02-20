#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

typedef long long ll;
typedef vector<vector<ll>> matrix;
mt19937_64 mt1(time(nullptr));

// CUDA kernel for naive matrix multiplication
__global__ void naive_mult_kernel(ll* a, ll* b, ll* res, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        ll sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        res[row * n + col] = sum;
    }
}

matrix randommatrix(ll a) {
    matrix k(a, vector<ll>(a));
    for (ll i = 0; i < a; ++i)
        for (ll j = 0; j < a; ++j)
            k[i][j] = mt1() % 100;
    return k;
}

ll nextPowerOfTwo(ll n) {
    if (n <= 1) return 1;
    ll power = 1;
    while (power < n) power <<= 1;
    return power;
}

matrix resizeMatrix(const matrix& mat, ll newrow, ll newcolumns) {
    matrix res(newrow, vector<ll>(newcolumns, 0));
    ll rows = min((ll)mat.size(), newrow);
    ll cols = min((ll)mat[0].size(), newcolumns);
    for (ll i = 0; i < rows; ++i)
        for (ll j = 0; j < cols; ++j)
            res[i][j] = mat[i][j];
    return res;
}

matrix naive_mult_cuda(const matrix& a, const matrix& b) {
    int n = a.size();
    size_t bytes = n * n * sizeof(ll);

    ll* h_a = new ll[n * n];
    ll* h_b = new ll[n * n];
    ll* h_res = new ll[n * n];

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = a[i][j];
            h_b[i * n + j] = b[i][j];
        }

    ll *d_a, *d_b, *d_res;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_res, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    naive_mult_kernel<<<blocks, threads>>>(d_a, d_b, d_res, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_res, bytes, cudaMemcpyDeviceToHost);

    matrix res(n, vector<ll>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            res[i][j] = h_res[i * n + j];

    delete[] h_a;
    delete[] h_b;
    delete[] h_res;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return res;
}

void outputmatrix(const matrix& a) {
    for (const auto& row : a) {
        for (auto val : row)
            cout << val << " ";
        cout << "\n";
    }
}

int main() {
    int n = 512;
    matrix A = randommatrix(n);
    matrix B = randommatrix(n);

    cout << "Starting CUDA naive multiplication...\n";
    clock_t start = clock();
    matrix C = naive_mult_cuda(A, B);
    clock_t end = clock();
    cout << "Time taken: " << (end - start) / 1000.0 << " seconds\n";

    // outputmatrix(C); // Uncomment to print result

    return 0;
}