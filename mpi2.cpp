#include <bits/stdc++.h>
#include <mpi.h>
#include <chrono>

#define MAX_PROCESS 32

using namespace std;

double *matriz_a;
double *x;

int tag = 1, tasks, iam, n, root = 0;

void readMatrix()
{

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cin >> matriz_a[i * n + j];
        }
    }
}

void init_x()
{
    double max_column = -INT_MAX;
    double max_fila = -INT_MAX;
    for (int i = 0; i < n; i++)
    {
        double column = 0;
        double fila = 0;
        for (int j = 0; j < n; j++)
        {
            fila += abs(matriz_a[i * n + j]);
            column += abs(matriz_a[j * n + i]);
        }
        max_column = max(max_column, column);
        max_fila = max(max_fila, fila);
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            x[i * n + j] = matriz_a[j * n + i] / (max_column * max_fila);
        }
    }
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

void calculate(int hilo, double *resultado, double *resultado1)
{
    int m = n % tasks;
    int start = (n / tasks) * hilo + min(hilo, m);
    int size = (n / tasks) + (hilo < m);
    for (int i = start; i < start + size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            resultado1[i * n + j] = 0;
            for (int k = 0; k < n; k++)
            {
                resultado1[i * n + j] += (x[i * n + k] * matriz_a[k * n + j]);
            }
        }
        for (int j = 0; j < n; j++)
        {
            resultado[i * n + j] = 2 * x[i * n + j];
            for (int k = 0; k < n; k++)
            {
                resultado[i * n + j] = resultado[i * n + j] - (resultado1[i * n + k] * x[k * n + j]);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < tasks; i++)
    {
        start = (n / tasks) * i + min(i, m);
        size = (n / tasks) + (i < m);
        // MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(resultado + start * n, n * size, MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
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

int main(int argc, char *argv[])
{

    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    MPI_Status status;

    MPI_Request req[MAX_PROCESS];

    MPI_Init(&argc, &argv);

    // Se obtiene la cantidad de procesos
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);

    // Se accede al id del proceso
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    if (iam == root)
    {
        cin >> n;
    }
    MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
    matriz_a = (double *)malloc(n * n * sizeof(double));
    x = (double *)malloc(n * n * sizeof(double));

    if (iam == root)
    {
        readMatrix();
    }
    auto start = chrono::high_resolution_clock::now();
    if (iam == root)
    {
        init_x();
    }

    MPI_Bcast(x, n * n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(matriz_a, n * n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    double *resultado;

    resultado = (double *)malloc(n * n * sizeof(double));

    double *resultado1;

    resultado1 = (double *)malloc(n * n * sizeof(double));
    for (int cont = 0; cont < 40; cont++)
    {
        calculate(iam, resultado, resultado1);
        memcpy(x, resultado, n * n * sizeof(double));
    }

    if (iam == root)
    {

        auto end = chrono::high_resolution_clock::now();

        auto int_s = chrono::duration_cast<chrono::microseconds>(end - start);

        cerr << int_s.count() / (float)1e6 << endl;

        cout << endl;
        print_matrix(x);
        cout << endl;
        multiplication_matrix(x, matriz_a, resultado);
        print_matrix(resultado);
        cout << endl;
    }
    free(x);
    free(matriz_a);
    free(resultado);
    free(resultado1);
    MPI_Finalize();
    return 0;
}
