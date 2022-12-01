#include <bits/stdc++.h>
#include <omp.h>

#define REPS 40

using namespace std;

int n;
double *matriz_a;
double *x;
int hilos;
double reps = 0;

void readMatrix()
{
    cin >> n;
    matriz_a = (double *)malloc(n * n * sizeof(double));
    x = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cin >> matriz_a[i * n + j];
        }
    }
}

void createMatrix()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            *(matriz_a + i * n + j) = rand() % 1000;
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

void calculate(int hilo, double *resultado, double *resultado1)
{
    for (int i = hilo; i < n; i += hilos)
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

void calculation_inverse()
{
    auto start = chrono::high_resolution_clock::now();

    init_x();

    double *resultado;
    resultado = (double *)malloc(n * n * sizeof(double));
    double *resultado1;
    resultado1 = (double *)malloc(n * n * sizeof(double));
    double ans = +INT_MAX;

    for (int k = 0; k < REPS; k++)
    {
#pragma omp parallel num_threads(hilos)
        {
            int id = omp_get_thread_num();
            calculate(id, resultado, resultado1);
        }
        memcpy(x, resultado, n * n * sizeof(double));
    }

    // cout << endl;
    // print_matrix ( x ) ;
    // cout << endl;
    // multiplication_matrix ( x , matriz_a , resultado );
    // print_matrix ( resultado ) ;
    free(resultado);
    free(resultado1);
    auto end = chrono::high_resolution_clock::now();
    auto int_s = chrono::duration_cast<chrono::microseconds>(end - start);
    reps = int_s.count() / 1000000.0;
}

int main(int argc, char *argv[])
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    if (argc != 2)
    {
        cout << "Entrada erronea\n";
        return 1;
    }
    hilos = atoi(argv[1]);
    n = 1000;
    matriz_a = (double *)malloc(n * n * sizeof(double));
    x = (double *)malloc(n * n * sizeof(double));
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    int a[] = {5, 10, 50, 100, 200, 500, 1000};
    for (int i = 0; i < 7; i++)
    {
        double sum = 0;
        n = a[i];
        int r = n > 500 ? 0 : 10;
        for (int j = 0; j < r; j++)
        {
            createMatrix();
            calculation_inverse();
            sum += reps;
            reps = 0;
        }
        cout << setprecision(20) << fixed << a[i] << ";" << sum / r << endl;
        fprintf(stderr, "%d\n", a[i]);
    }
    free(x);
    free(matriz_a);
    return 0;
}
