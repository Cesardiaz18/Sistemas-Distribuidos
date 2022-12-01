#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#define MAX_DIM_ROW 1024
#define REPS 40

using namespace std;

int blocks;
int threads;
int n;
double reps;
double *matrix;
double *x;
double *resultado;
double *ident;
cudaError_t err = cudaSuccess;
double *cudaM;
double *cudaR1;
double *cudaR2;
double *cudaX;

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

void initI()
{
    ident = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                *(ident + i + j * n) = 1;
            }
            else
            {
                *(ident + i + j * n) = 0;
            }
        }
    }
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

void calccc()
{
    auto start = chrono::high_resolution_clock::now();
    x = init_x(matrix);
    int matrix_size = (n * n) * sizeof(double);
    err = cudaMemcpy(cudaM, matrix, matrix_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy to device matrix A(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // print_matrix(ident);

    err = cudaMemcpy(cudaX, x, matrix_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy to device matrix A(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    const int cu_n = n;
    for (int i = 0; i < REPS; i++)
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
    auto end = chrono::high_resolution_clock::now();
    auto int_s = chrono::duration_cast<chrono::microseconds>(end - start);
    reps = int_s.count() / 1000000.0;
}

void createMatrix()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            *(matrix + i * n + j) = rand() % 1000;
        }
    }
}

int calcInverse()
{

    n = 1000;
    matrix = (double *)malloc(n * n * sizeof(double));
    int matrix_size = (n * n) * sizeof(double);
    resultado = (double *)malloc(n * n * sizeof(double));

    err = cudaMalloc((void **)&cudaM, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&cudaR1, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix R1(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&cudaR2, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix R2(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&cudaX, matrix_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix X(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int a[] = {5, 10, 50, 100, 200, 500, 1000};
    for (int i = 0; i < 7; i++)
    {
        double sum = 0;
        n = a[i];
        initI();
        for (int j = 0; j < 10; j++)
        {
            createMatrix();
            calccc();
            sum += reps;
            reps = 0;
        }
        cout << a[i] << ";" << (double)sum / 10 << endl;
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