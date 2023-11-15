#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <string.h>
#include "symnmf.h"


/* Global variables */
int N;
int vectorDim;
const int MAX_ITER = 300;
const double BETA = 0.5;
const double EPSILON =  0.0001;

/* Function declarations */
static void matrix_product(double **m1, double **m2);
static double euclidian_distance(double *vector1, double *vector2);
static void error_func(int input);
double average_of_matrix(double** matrix);
double frobenius_norm_squared(int k,double** kW, double** H);
int update_H_until_convergence(int cols, double** H, double** W);
double** allocate_matrix(int rows, int cols);
static void print_output(double **matrix);
static void free_matrix(double **matrix, int rows);
void build_similarity_matrix(double **data_points, double **A);
void build_degree_matrix(double **A_ptr, double **D_ptr);
void build_norm(double **W_ptr, double **D_ptr);
static double** read_input_file(char *fileName);

int main(int argc, char *argv[]) {
    char *goal, *input_file_name;
    double **data_points, **W, **D;

    if (argc != 3) {
        error_func(1);
    }
    else {
        goal = argv[1];
        input_file_name = argv[2];
        data_points = read_input_file(input_file_name);

         if (strcmp(goal, "sym") == 0) {
            W = allocate_matrix(N, N);
            build_similarity_matrix(data_points, W);
            print_output(W);
            free_matrix(data_points, N);
            free_matrix(W, N);
        }

        else if (strcmp(goal, "ddg") == 0) {
            W = allocate_matrix(N, N);
            build_similarity_matrix(data_points, W);
            D = allocate_matrix(N, N);
            build_degree_matrix(W, D);
            print_output(D);
            free_matrix(data_points, N);
            free_matrix(W, N);
            free_matrix(D, N);
        }

        else if (strcmp(goal, "norm") == 0) {
            W = allocate_matrix(N, N);
            build_similarity_matrix(data_points, W);
            D = allocate_matrix(N, N);
            build_degree_matrix(W, D);
            build_norm(W, D);
            print_output(D);
            free_matrix(data_points, N);
            free_matrix(W, N);
            free_matrix(D, N);
        }
        else {
            error_func(1);
        }
    }
    exit(0);
}

static double** read_input_file(char *fileName) {
    FILE *file = NULL;
    int numOfLines = 0;
    int numOfPoints = 0;
    double num;
    double **dataPoints;
    int i,j;
    double v;
    char c;
    int b =0;

    file = fopen(fileName, "r");
    while ((b = fgetc(file)) != EOF){
    while ((fscanf(file, "%lf%c", &v, &c) == 2) || (c = fgetc(file)) == ',') {
        numOfPoints++;
        if (c == '\n') {
            numOfLines++;
            break;
        }
    }
    }
    rewind(file);
    fclose(file);


    if (numOfLines == 0)
        error_func(0);

    N = numOfLines;
    vectorDim = numOfPoints / numOfLines;
    dataPoints = calloc(numOfLines, sizeof(double*));
    if (dataPoints == NULL)
        error_func(0);

    file = fopen(fileName, "r");
    if (file != NULL) {
        for (i = 0; i < numOfLines; i++) {
            dataPoints[i] = calloc(vectorDim, sizeof(double));
            if (dataPoints[i] == NULL)
                error_func(0);
            for (j = 0; j < vectorDim; j++) {
                if (fscanf(file, "%lf", &num) != 1) {
                error_func(1); 
            }
            
            if (fscanf(file, ",") != 0) {
                error_func(1); 
            }
            dataPoints[i][j] = num;
            }   
        }  
    }
    else {
        error_func(0);
    }
    fclose(file);

    return dataPoints;
}

/* Calculates the squared Euclidian distance between two vectors */
static double euclidian_distance(double *vector1, double *vector2){
    double sum = 0.0;
    int i;

    for (i = 0; i < vectorDim; ++i){
        sum += pow(vector1[i] - vector2[i], 2);
    }

    return sum;
}

/* Build the weighted similarity matrix of the graph */
void build_similarity_matrix(double **data_points, double **A) {
    int i, j;
    double l2_norm, weight;

    for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) {
            
            if (i == j) {
                A[i][j] = 0;
            }   
            else {
                l2_norm =euclidian_distance(data_points[i], data_points[j]);
                weight = exp(-l2_norm/2);
                A[i][j] = A[j][i] = weight;
            }
        }
    } 
}

/* Build the Diagonal degree matrix of the graph */
void build_degree_matrix(double **A_ptr, double **D_ptr) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
                D_ptr[i][i] += A_ptr[i][j];
        }
    }
}


/* Multiply m1 by m2 using inner product and write result to m2 */
static void matrix_product(double **m1, double **m2) {
    int i, j, k;
    double *C;

    /* Allocate auxiliary array */
    C = calloc(N, sizeof(double));
    if (C == NULL) 
        error_func(0);

    /* Perform mulitiplication inplace */
    for (j = 0; j < N; ++j) {
        for (i = 0; i < N; ++i) {
            C[i] = m2[i][j];
        }
        for (i = 0; i < N; ++i) {
            m2[i][j] = 0;
            for (k = 0; k < N; ++k) {
                m2[i][j] += m1[i][k] * C[k];
            }
        }
    }

    /* Free auxiliary array */
    free(C);
}

/* Build the Normalized Graph Laplacian */
/* At exit result is in D_ptr */
void build_norm(double **W_ptr, double **D_ptr) {
    int i;

    /* Calculate D^(-1/2) */
    for (i = 0; i < N; i++) {
        D_ptr[i][i] = pow(D_ptr[i][i], -0.5);
    }

    /* Calculate Normalized Laplacian */
    matrix_product(D_ptr, W_ptr);
    matrix_product(W_ptr, D_ptr);
}

/* Function to calculate the average of all entries in a matrix W */
double average_of_matrix(double** matrix) {
    double sum = 0.0;
    int i,j;
    for ( i = 0; i < N; i++) {
        for ( j = 0; j < N; j++) {
            sum += matrix[i][j];
        }
    }
    return sum / (N * N);
}

/* Function to compute the Frobenius norm squared of the difference between two matrices */
double frobenius_norm_squared(int k, double** current, double** prev) {

    double norm = 0.0;
    int i,j;

    for ( i = 0; i < N; i++) {
        for ( j = 0; j < k; j++) {
            double element = current[i][j]-prev[i][j];
            norm += element * element;
        }
    }

    return norm;
}

/* Function to update H until convergence or maximum iterations */
int update_H_until_convergence(int cols, double** H, double** W) {

    int i, j, k,p, exit_code;
    double norm_squared;
    double** WH = (double**)malloc(N * sizeof(double*));
    double** Ht = (double**)malloc(cols * sizeof(double*));
    double** HHt = (double**)malloc(N * sizeof(double*));
    double** HHtH = (double**)malloc(N * sizeof(double*));
    double** prev_H = (double**)malloc(N * sizeof(double*));
    if (WH == NULL || Ht == NULL || HHt == NULL || HHtH == NULL || prev_H == NULL ) {
    printf("An Error Has Occurred");
    free(WH);
    free(Ht);
    free(HHt);
    free(HHtH);
    free(prev_H);
    exit(1);
    }

    for ( i = 0; i < N; i++) {
        WH[i] = (double*)malloc(cols * sizeof(double));
        HHt[i] = (double*)malloc(N * sizeof(double));
        HHtH[i] = (double*)malloc(cols * sizeof(double));
        prev_H[i] = (double*)malloc(cols * sizeof(double));

    }

    for ( i = 0; i < cols; i++) {
        Ht[i] = (double*)malloc(N * sizeof(double));
    }
    
    exit_code = 0;
    k = 0;
 
    for ( i = 0; i < N; i++) {
        for ( j = 0; j < cols; j++) {
            prev_H[i][j] = 0.0;
                prev_H[i][j] =H[i][j];
            
        }
    }

    while (k < MAX_ITER) {
          
     /*Compute WH = W * H*/ 
    for ( i = 0; i < N; i++) {
        for ( j = 0; j < cols; j++) {
            WH[i][j] = 0.0;
            for ( p = 0; p < N; p++) {
                WH[i][j] += W[i][p] * H[p][j];
            }
        }
    }

    /*get transpose matrix*/ 
    for (i = 0; i <N; i++) {
        for (j = 0; j <cols; j++) {
           Ht[j][i]=0.0;
           Ht[j][i] =H[i][j];
        }
    }  

   /*Compute HHt = H * H^T*/ 
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
        HHt[i][j] = 0.0;
        for (p = 0; p < cols; p++) {  
            HHt[i][j] += H[i][p] * Ht[p][j];
        }
    }
    }

    /*Compute result = HHt * H*/
    for ( i = 0; i < N; i++) {
        for ( j = 0; j < cols; j++) {
            HHtH[i][j] = 0.0;
            for ( p = 0; p < N; p++) {
                HHtH[i][j] += HHt[i][p] * H[p][j];
            }
        }
    }
 
    if (WH == NULL || HHt == NULL || HHtH == NULL || Ht == NULL)  {
        exit_code = 1;
        for ( i = 0; i < N; i++) {
            free(WH[i]);
            free(HHt[i]);
            free(HHtH[i]);
            free(prev_H[i]);

            }
        for( i=0; i<cols; i++){
                free(Ht[i]);
            }
        free(WH);
        free(Ht);
        free(HHt);
        free(HHtH);
        free(prev_H);
        if (exit_code == 1) {
                break;
        }
    }

    for ( i = 0; i < N; i++) {
        for ( j = 0; j < cols; j++) {
            double update_term = 1.0 - BETA + BETA * (WH[i][j] / HHtH[i][j]);
            H[i][j] *= update_term;
 
            /*Ensure H remains non-negative*/
            if (H[i][j] < 0.0) {
                H[i][j] = 0.0;
            }
        }
    }
  
    norm_squared = frobenius_norm_squared(cols, H, prev_H);
    /*Check for convergence*/
    if (norm_squared < EPSILON) {
            break;  
    }
 
    for ( i = 0; i < N; i++) {
        for ( j = 0; j < cols; j++) {
            prev_H[i][j] = 0.0;
            for ( p = 0; p < N; p++) {
                prev_H[i][j] =H[i][j];
            }
        }
    }
    
        k++;
    }
    
    for ( i = 0; i < N; i++) {
        free(WH[i]);
        free(HHt[i]);
        free(HHtH[i]);
        free(prev_H[i]);

    }
    for ( i = 0; i < cols; i++){
      free(Ht[i]);
    }

    free(WH);
    free(Ht);
    free(HHt);
    free(HHtH);
    free(prev_H);
    return exit_code;
}

/* Allocate matrix of size rowsXcols */
double** allocate_matrix(int rows, int cols) {
    int i;
    double **matrix;

    matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL)
        return NULL;
    for (i = 0; i < rows; i++){
        matrix[i] = calloc(cols, sizeof(double));
        if (matrix[i] == NULL)
            return NULL;
    }

    return matrix;
}

/* Print output to console */
static void print_output(double **matrix) {
    int i, j;

    /* Print matrix */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (j != (N - 1)) {
                printf("%.4f,", matrix[i][j]); 
            }
            else {
                printf("%.4f\n", matrix[i][j]);
            }
        }
    } 
}

/* Free matrix of size rows */
static void free_matrix(double **matrix, int rows) {
    int i;

    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/* Default Error print */
static void error_func(int input) {
    if (input == 1)
        printf("Invalid Input!\n");
    else
        printf("An Error Has Occured!\n");
    exit(1);
}