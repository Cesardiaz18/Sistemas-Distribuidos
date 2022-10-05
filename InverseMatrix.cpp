#include <bits/stdc++.h>

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

void res_matrix ( double *a , double *resultado ){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j ++ ){
            a[i * n + j] -= resultado[i * n + j];
        }
    }
}

void multiplication_value ( double *a , double p ){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j ++ ){

            a[i * n + j] = p * a[i * n + j];

        }
    }
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
        multiplication_matrix ( x , matriz_a , resultado ) ;
        multiplication_matrix ( resultado , x , resultado1 ) ;
        multiplication_value ( x , 2.0 ) ;
        res_matrix ( x , resultado1 ) ;
//        multiplication_matrix ( x , matriz_a , resultado );
//        print_matrix ( resultado ) ;

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
    calculation_inverse ( ) ;
    return 0 ;

}
