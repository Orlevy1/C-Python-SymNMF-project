#ifndef SYMNMF_H
#define SYMNMF_H 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* Global variables */
extern int N;
extern int vectorDim;
extern const int MAX_ITER;
extern const double BETA;
extern const double EPSILON;

/* Function declarations */
void build_similarity_matrix(double **data_points, double **W);
void build_degree_matrix(double **W_ptr, double **D_ptr);
void build_norm(double **W_ptr, double **D_ptr);
double** allocate_matrix(int rows, int cols);
int update_H_until_convergence(int cols, double** H, double** W);
#endif 
