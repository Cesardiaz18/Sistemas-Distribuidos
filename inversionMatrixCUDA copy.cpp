#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define DIM 1024
#define XDIM DIM
#define YDIM DIM
#define MATRIXSIZE XDIM *YDIM
#define TILE_SIZE 16
#define NUMTHREADS DIM

/*****************************************************************************/
__global__ void multMatrix_globalMemory(double *A, double *B, double *C, int numElements)
{
    int yOffset;
    int i, x;
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    yOffset = y * XDIM;

    if (y < numElements)
    {
        for (x = 0; x < XDIM; x++)
        {
            *(C + yOffset + x) = 0;
            for (i = 0; i < XDIM; i++)
            {
                //*(C + yOffset + x) = y;
                *(C + yOffset + x) = *(C + yOffset + x) + (*(A + yOffset + i) * (*(B + (i * YDIM) + x)));
            }
        }
    }
}

/*****************************************************************************/
__global__ void multMatrix_sharedMemory(double *A, double *B, double *C, int numElements)
{
    int x, y, i, j, k;
    double p;
    // int id = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sharedA[TILE_SIZE][TILE_SIZE], sharedB[TILE_SIZE][TILE_SIZE];

    int tiles_per_row = DIM / TILE_SIZE;
    int globalRowTileOffset = (blockIdx.x * TILE_SIZE * TILE_SIZE * tiles_per_row);
    int offsetA, offsetB, offset_x_c, offset_y_c;

    y = threadIdx.x / TILE_SIZE;
    x = threadIdx.x % TILE_SIZE;

    // go through left to right
    // other rows of tiles are run by other blocks

    for (i = 0; i < tiles_per_row; i++)
    {
        for (k = 0; k < tiles_per_row; k++)
        {
            offsetA = globalRowTileOffset + (k * TILE_SIZE) + (y * tiles_per_row * TILE_SIZE) + x;
            sharedA[x][y] = *(A + offsetA);
            offsetB = (i * TILE_SIZE) + (k * TILE_SIZE * TILE_SIZE * tiles_per_row) + (y * TILE_SIZE * tiles_per_row) + x;
            sharedB[x][y] = *(B + offsetB);
            offset_y_c = offsetA / DIM; // offset to write C
            offset_x_c = offsetB % DIM;
            __syncthreads();

            p = 0;
            for (j = 0; j < TILE_SIZE; j++)
            {
                p = p + sharedA[j][y] * sharedB[x][j];
            }

            __syncthreads();
            *(C + (offset_y_c * DIM) + offset_x_c) += p;
        }
    }
}

/*****************************************************************************/

int multMatrix_cpu(double *A, double *B, double *C)
{
    int i, x, y;
    int yOffset;

    for (y = 0; y < YDIM; y++)
    {
        yOffset = y * XDIM;
        for (x = 0; x < XDIM; x++)
        {
            for (i = 0; i < XDIM; i++)
            {
                *(C + yOffset + x) = *(C + yOffset + x) + (*(A + yOffset + i) * (*(B + (i * YDIM) + x)));
            }
        }
    }
    return 0;
}

/*****************************************************************************/

int printMatrix(double *ap)
{
    int x, y;
    for (y = 0; y < YDIM; y++)
    {
        printf("\n");
        for (x = 0; x < XDIM; x++)
        {
            printf("%.1f \t", *(ap + (y * XDIM) + x));
        }
    }
    printf("\n");
    return 0;
}

/*****************************************************************************/

int matrixCompare(double *C, double *C2)
{
    int x, y;
    double diff;
    for (y = 0; y < YDIM; y++)
    {
        for (x = 0; x < XDIM; x++)
        {
            diff = fabs(*(C + (y * YDIM) + x) - *(C2 + (y * YDIM) + x));
            if (diff > 1e-5)
                return -1;
        }
    }
    printf("\nVerification OK! \n");
    return 0;
}

/******************************************************************************
 * Host main routine
 */
int main(int argc, char *argv[])
{
    int i, v = 1;
    int threadsNum, blocksPerGrid, threadsPerBlock;
    long elapsed;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    struct timeval tv1, tv2;

    // Print the vector length to be used, and compute its size
    int numElements = MATRIXSIZE;
    size_t size = MATRIXSIZE * sizeof(double);
    if (v == 1)
        printf("[Matrix mult of %d elements]    %d x %d \n", numElements, DIM, DIM);

    // Allocate the host input vector A
    double *h_A = (double *)malloc(size);

    // Allocate the host input vector B
    double *h_B = (double *)malloc(size);

    // Allocate the host output vector C
    double *h_C = (double *)malloc(size);
    double *h_CPU = (double *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors

    for (i = 0; i < MATRIXSIZE; i++)
    {
        *(h_A + i) = rand() & 0xF;
        *(h_B + i) = rand() & 0xF;
        *(h_C + i) = 0;
    }
    // printMatrix(h_A);
    // printMatrix(h_B);
    printf("\nStarting CPU multiplication ");
    fflush(stdout);
    gettimeofday(&tv1, NULL);
    multMatrix_cpu(h_A, h_B, h_CPU);
    gettimeofday(&tv2, NULL);
    elapsed = ((tv2.tv_sec - tv1.tv_sec) * 1000000) + (tv2.tv_usec - tv1.tv_usec);
    printf("\nCPU multiplication finished in %ld microseconds\n", elapsed);
    fflush(stdout);

    // Allocate the device input vector A
    double *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    double *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    double *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /****** Launch the Kernel *******/
    threadsNum = DIM;
    threadsPerBlock = 64;
    blocksPerGrid = threadsNum / threadsPerBlock;
    printf("\nCUDA kernel launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);
    fflush(stdout);
    gettimeofday(&tv1, NULL);
    multMatrix_globalMemory<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    gettimeofday(&tv2, NULL);
    elapsed = ((tv2.tv_sec - tv1.tv_sec) * 1000000) + (tv2.tv_usec - tv1.tv_usec);
    printf("\nGPU multiplication executed in %ld microseconds", elapsed);
    fflush(stdout);

    if (matrixCompare(h_C, h_CPU) == -1)
        printf("\nNot equal");
    for (i = 0; i < MATRIXSIZE; i++)
    {
        *(h_C + i) = 0; // clean C
    }
    err = cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /****** Launch the tiled Kernel *******/
    threadsPerBlock = TILE_SIZE * TILE_SIZE;
    blocksPerGrid = (DIM / TILE_SIZE);
    printf("\nCUDA kernel launch with %d blocks of %d threads", blocksPerGrid, threadsPerBlock);
    fflush(stdout);
    gettimeofday(&tv1, NULL);
    multMatrix_sharedMemory<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    gettimeofday(&tv2, NULL);
    elapsed = ((tv2.tv_sec - tv1.tv_sec) * 1000000) + (tv2.tv_usec - tv1.tv_usec);
    printf("\nGPU tiled multiplication executed in %ld  microseconds", elapsed);
    fflush(stdout);

    if (matrixCompare(h_C, h_CPU) == -1)
        printf("\nNot equal");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // printMatrix(h_C);
    // printMatrix(h_CPU);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CPU);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}
