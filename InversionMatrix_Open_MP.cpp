#include <bits/stdc++.h>
#include <omp.h>

#define MAX_THREADS 8
using namespace std ;

int n ;
double *matriz_a;
double *x ;

void readMatrix(){
    cin >> n;
    matriz_a = (double *)malloc(n * n * sizeof(double));
    x = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            cin >> matriz_a[i * n + j];
        }
    }

}

void init_x ( ){
    double max_column = -INT_MAX ;
    double max_fila = -INT_MAX ;
    for ( int i = 0 ; i < n ; i++ ){
        double column = 0 ;
        double fila = 0 ;
        for ( int j = 0 ; j < n ; j++ ){
            fila += abs(matriz_a[i*n+j]) ;
            column += abs(matriz_a[j*n+i]) ;
        }
        max_column = max ( max_column , column ) ;
        max_fila = max ( max_fila , fila ) ;
    }
    for ( int i = 0 ; i < n ; i++ ){
        for ( int j = 0 ; j < n ; j++ ){
            x[i*n+j] = matriz_a[j*n+i] / ( max_column * max_fila ) ;
        }

    }

}

void calculate ( int hilo , double *resultado , double *resultado1 ){
    for ( int i = hilo ; i < n ; i+=MAX_THREADS ){
        for ( int j = 0 ; j < n ; j++ ){
            resultado1[ i * n + j ] = 0 ;
            for ( int k = 0 ; k < n ; k++ ){
                resultado1[ i * n + j ] += ( x [ i * n + k ] * matriz_a [ k * n + j ] )  ;
            }
        }
        for ( int j = 0 ; j < n ; j++ ){
            resultado [ i * n + j ] = 2 * x [ i * n + j ] ;
            for ( int k = 0 ; k < n ; k++ ){
                resultado [ i * n + j ] = resultado [ i * n + j ] - ( resultado1[ i * n + k ] * x [ k * n + j ] )  ;
            }
        }
    }
}

void print_matrix ( double *a ){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j ++ ){
            cout<<a[i * n + j]<<" ";
        }
        cout<<endl ;
    }
}

double matrix_diff ( double *a , double *b ){
    double ans = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j ++ ){
            ans += abs(a[i * n + j] - b[i * n + j]);
        }
    }
    return ans;

}

void multiplication_matrix ( double *a , double *b , double *resultado){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j ++ ){
            resultado [ i * n + j ] = 0 ;
            for (int k = 0; k < n; k++){
                resultado [ i * n + j ] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void calculation_inverse ( ){

    readMatrix ( );
    init_x ( ) ;

    double *resultado ;
    resultado = (double *)malloc(n * n * sizeof(double));
    double *resultado1 ;
    resultado1 = (double *)malloc(n * n * sizeof(double));
    double *resultado2 ;
    resultado2 = (double *)malloc(n * n * sizeof(double));
    memset(resultado2, 0, n * n * sizeof(double));
    double ans = +INT_MAX;
    int i = 0;

    while (true){
        i++;
         #pragma omp parallel num_threads(MAX_THREADS)
         {
             int id = omp_get_thread_num();
            calculate ( id , resultado , resultado1 ) ;
         }
        memcpy(x, resultado, n * n * sizeof(double));

        if(matrix_diff(x, resultado2)<1e-10){
            break;
        }
        memcpy(resultado2, x, n * n * sizeof(double));
    }

    cout << endl;
    print_matrix ( x ) ;
    cout << endl;
    multiplication_matrix ( x , matriz_a , resultado );
    print_matrix ( resultado ) ;
    cout << "i = " << i << endl;
    free(x);
    free(matriz_a);
    free(resultado);
    free(resultado1);
    free(resultado2);
}




int main ( ){
    freopen ( "input.txt" , "r" , stdin ) ; 
    freopen ( "output.txt" , "w" , stdout ) ;  
    calculation_inverse ( ) ;
    return 0 ;

}
