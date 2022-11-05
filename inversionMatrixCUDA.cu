#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#define MAX_DIM_ROW 1024
#define REPS 40

using namespace std;

int blocks;
int threads;
int n;

__global__ void inverseMatrixP1(double *M, double *X, double *R1, double *R2, int DIM)
{

    __shared__ double row[MAX_DIM_ROW];
    double sum;
    // Bucle que saltará numBlocks en cada iteración
    for (int d = blockIdx.x; d < DIM; d += gridDim.x)
    {

        // Bucle que con cada bloque usará hilos
        for (int i = threadIdx.x; i < DIM; i += blockDim.x)
        {

            // row será igual al valor de X en cierta posición.
            row[i] = *(X + i + DIM * d);
        }
        // Se sincronizan los hilos
        __syncthreads();

        // Se realizar el bucle para avanzar de acuero al id del hilo en el bloque por cada bloque
        for (int col = threadIdx.x; col < DIM; col += blockDim.x)
        {
            sum = 0;

            // Se recorre
            for (int pos = 0; pos < DIM; pos++)
            {

                // Se realiza la multiplicacion de las matrices teniendo en cuenta que la matriz X estará en row
                sum += row[pos] * *(M + col + DIM * pos);
            }

            // Se asigna a la matriz resultante el valor de la multiplicación
            *(R1 + col + DIM * d) = sum;
        }
        __syncthreads();
    }
}

__global__ void inverseMatrixP2(double *M, double *X, double *R1, double *R2, int DIM)
{

    __shared__ double row[MAX_DIM_ROW];
    double sum;
    for (int d = blockIdx.x; d < DIM; d += gridDim.x)
    {

        // Bucle que con cada bloque usará hilos
        for (int i = threadIdx.x; i < DIM; i += blockDim.x)
        {

            // row será igual al valor de A en cierta posición.
            row[i] = *(R1 + i + DIM * d);
        }
        // Se sincronizan los hilos
        __syncthreads();

        // Se realizar el bucle para avanzar de acuero al id del hilo en el bloque por cada bloque
        for (int col = threadIdx.x; col < DIM; col += blockDim.x)
        {
            sum = 0;

            // Se recorre
            for (int pos = 0; pos < DIM; pos++)
            {

                // Se realiza la multiplicacion de las matrices teniendo en cuenta que la matriz A estará en row
                sum += row[pos] * *(X + col + DIM * pos);
            }

            // Se asigna a la matriz resultante el valor de la multiplicación
            *(R2 + col + DIM * d) = 2 * *(X + col + DIM * d) - sum;
        }
        __syncthreads();
    }
}

__global__ void inverseMatrixP3(double *M, double *X, double *R1, double *R2, int DIM)
{
    for (int d = blockIdx.x; d < DIM; d += gridDim.x)
    {

        // Bucle que con cada bloque usará hilos
        for (int col = threadIdx.x; col < DIM; col += blockDim.x)
        {

            // row será igual al valor de A en cierta posición.
            *(X + col + DIM * d) = *(R2 + col + DIM * d);
        }
    }
    __syncthreads();
}

double *init_x(double *m)
{
    double max_column = -INT_MAX;
    double max_fila = -INT_MAX;
    for (int i = 0; i < n; i++)
    {
        double column = 0;
        double fila = 0;
        for (int j = 0; j < n; j++)
        {
            fila += abs(m[i * n + j]);
            column += abs(m[j * n + i]);
        }
        max_column = max(max_column, column);
        max_fila = max(max_fila, fila);
    }
    double *x = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            x[i * n + j] = m[j * n + i] / (max_column * max_fila);
        }
    }
    return x;
}
void print_matrix(double *a)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << a[i * n + j] << " ";
        }
        cout << endl;
    }
}

double matrix_diff(double *a, double *b)
{
    double ans = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            ans += abs(a[i * n + j] - b[i * n + j]);
        }
    }
    return ans;
}

void multiplication_matrix(double *a, double *b, double *resultado)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            resultado[i * n + j] = 0;
            for (int k = 0; k < n; k++)
            {
                resultado[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

/*****************************************************************************/
int calcInverse()
{

    cudaError_t err = cudaSuccess;
    cin >> n;
    double *matrix = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cin >> matrix[i * n + j];
        }
    }
    double *x = init_x(matrix);
    int matrix_size = (n * n) * sizeof(double);
    double *resultado = (double *)malloc(n * n * sizeof(double));

    double *cudaM;
    err = cudaMalloc((void **)&cudaM, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double *cudaR1;
    err = cudaMalloc((void **)&cudaR1, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix R1(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double *cudaR2;
    err = cudaMalloc((void **)&cudaR2, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix R2(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double *cudaX;
    err = cudaMalloc((void **)&cudaX, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix X(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(cudaM, matrix, matrix_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy to device matrix A(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(cudaX, x, matrix_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy to device matrix A(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    const int cu_n = n;
    printf("Launching kernel %d, %d \n", blocks, threads);

    auto start = chrono::high_resolution_clock::now();

    for (int rep = 0; rep < REPS; rep++)
    {
        inverseMatrixP1<<<blocks, threads>>>(cudaM, cudaX, cudaR1, cudaR2, cu_n);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
        inverseMatrixP2<<<blocks, threads>>>(cudaM, cudaX, cudaR1, cudaR2, cu_n);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
        inverseMatrixP3<<<blocks, threads>>>(cudaM, cudaX, cudaR1, cudaR2, cu_n);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
    }

    inverseMatrixP1<<<blocks, threads>>>(cudaM, cudaX, cudaR1, cudaR2, cu_n);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    auto int_s = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "MatrixMult elapsed time is " << int_s.count() / (float)1000000 << " seconds " << endl;

    printf("Kernel finalizado\n");
    err = cudaMemcpy(x, cudaX, matrix_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy to device solution matrix(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(resultado, cudaR1, matrix_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy to device solution matrix(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(cudaM);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device original  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(cudaX);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device solution (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(cudaR1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device solution (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(cudaR2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device original  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cout << endl;
    print_matrix(x);
    cout << endl;
    print_matrix(resultado);
    free(resultado);
    free(matrix);
    free(x);

    err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

int main(int argc, char *argv[])
{
    freopen("input 1000.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    if (argc != 3)
    {
        cout << "Usage: " << argv[0] << " <bloques> <hilos>" << endl;
        return 1;
    }
    blocks = atoi(argv[1]);
    threads = atoi(argv[2]);
    calcInverse();
}