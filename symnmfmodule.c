#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Build the PyObject of the result and free the C matrix that was used */
static PyObject* create_res_for_py(double **matrix, int rows, int cols) {
    int i, j;
    PyObject *sublist, *result;

    result = PyList_New(rows);
    if (!result)
        return NULL;

    for (i = 0; i < rows; i++){
        sublist = PyList_New(cols);
        if (!sublist)
            return NULL;
        for (j = 0; j < cols; j++){
            PyList_SET_ITEM(sublist, j, Py_BuildValue("d", matrix[i][j]));
        }
        PyList_SET_ITEM(result, i, Py_BuildValue("O", sublist));
        free(matrix[i]);
    }
    free(matrix);
  
    return result;
}

/* Build the datapoints C matrix from python object */
static double** build_data_points(PyObject* datapoints_from_py, int rows, int cols) { 
    int i, j;
    PyObject *sublist;
    double **data_points;

    data_points = (double**)malloc(N * sizeof(double*));
    if (data_points == NULL)
        return NULL;
    for (i = 0; i < rows; i++){
        sublist = PyList_GetItem(datapoints_from_py, i);
        data_points[i] = calloc(cols, sizeof(double));
        if (data_points[i] == NULL)
            return NULL;
        for (j = 0; j < cols; j++){
            data_points[i][j] = PyFloat_AsDouble(PyList_GetItem(sublist, j));
        }
    }

    return data_points;
}

/* Goal == sym */
static PyObject* sym(PyObject *self, PyObject *args) {
    int i;
    PyObject *datapoints_from_py;
    double **data_points, **W; 
     
    if(!PyArg_ParseTuple(args, "O", &datapoints_from_py)) {
        return NULL;
    } 

    /* Set globals and read datapoints*/
    vectorDim = PyList_Size(PyList_GetItem(datapoints_from_py, 0));
    N = PyList_Size(datapoints_from_py);
    data_points = build_data_points(datapoints_from_py, N, vectorDim);
    
    /* Allocate and calculate Weigthed similarity Matrix */
    W = allocate_matrix(N, N);
    build_similarity_matrix(data_points, W);

    /* Free datapoints matrix */
    for (i = 0; i < N; i++){
        free(data_points[i]);
    } 
    free(data_points);

    return create_res_for_py(W, N, N);
}

/* Goal == ddg */
static PyObject* ddg(PyObject *self, PyObject *args) {
    int i;
    PyObject *datapoints_from_py;
    double **data_points, **W, **D;
     
    if(!PyArg_ParseTuple(args, "O", &datapoints_from_py)) {
        return NULL;
    } 

    /* Set globals and read datapoints*/
    vectorDim = PyList_Size(PyList_GetItem(datapoints_from_py, 0));
    N = PyList_Size(datapoints_from_py);
    data_points = build_data_points(datapoints_from_py, N, vectorDim);

    /* Allocate and calculate Weigthed similarity Matrix */
    W = allocate_matrix(N, N);
    build_similarity_matrix(data_points, W);

    /* Allocate and calculate Diagonal Degree Matrix */
    D = allocate_matrix(N, N);
    build_degree_matrix(W, D);

    /* Free datapoints and adjacency matrices */
    for (i = 0; i < N; i++){
        free(data_points[i]);
        free(W[i]);
    } 
    free(data_points);
    free(W);

    return create_res_for_py(D, N, N);
}

/* Goal == norm */
static PyObject* norm(PyObject *self, PyObject *args) {
    int i;
    PyObject *datapoints_from_py;
    double **data_points, **W, **D;
     
    if(!PyArg_ParseTuple(args, "O", &datapoints_from_py)) {
        return NULL;
    } 

    /* Set globals and read datapoints*/
    vectorDim = PyList_Size(PyList_GetItem(datapoints_from_py, 0));
    N = PyList_Size(datapoints_from_py);
    data_points = build_data_points(datapoints_from_py, N, vectorDim);

    /* Allocate and calculate Weigthed Adjacency Matrix */
    W = allocate_matrix(N, N);
    build_similarity_matrix(data_points, W);

    /* Allocate and calculate Diagonal Degree Matrix */
    D = allocate_matrix(N, N);
    build_degree_matrix(W, D);

    /* Calculate Normalized */
    build_norm(W, D);

    /* Free datapoints and adjacency matrices */
    for (i = 0; i < N; i++){
        free(data_points[i]);
        free(W[i]);
    } 
    free(data_points);
    free(W);

    return create_res_for_py(D, N, N);
}

/* Goal == symnmf */
static PyObject* symnmf(PyObject *self, PyObject *args) {
    PyObject *h_from_py;  
    PyObject *w_from_py; 
    double **H, **W;
    int i, rows,cols;

    if (!PyArg_ParseTuple(args, "OO", &h_from_py, &w_from_py)) {
        return NULL;
    }

    if (w_from_py == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "invalid input!");
        return NULL;
    }

    if (!PyList_Check(h_from_py)) {
        PyErr_SetString(PyExc_TypeError, "invalid input!");
        return NULL;
    }

    rows = PyList_Size(h_from_py);
    if (rows <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "invalid input!");
        return NULL;
    }
    cols = PyList_Size(PyList_GetItem(h_from_py, 0));
    if (cols <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "invalid input!");
        return NULL;
    }
    H = build_data_points(h_from_py, rows,cols);
    W = build_data_points(w_from_py, rows,rows);

    vectorDim = PyList_Size(PyList_GetItem(h_from_py, 0));

    update_H_until_convergence(cols,H,W);
    
    /* Free  W matrices */
    for (i = 0; i < N; i++){
        free(W[i]);
    } 
    free(W);

    return create_res_for_py(H, N, cols); 

}


static PyMethodDef symnmfmoduleMethods[] = {
    {"sym",
        (PyCFunction) sym,
        METH_VARARGS,
        PyDoc_STR("Calculate Weigthed similarity Matrix")},
    {"ddg",
        (PyCFunction) ddg,
        METH_VARARGS,
        PyDoc_STR("Calculate Diagonal Degree Matrix")},
    {"norm",
        (PyCFunction) norm,
        METH_VARARGS,
        PyDoc_STR("Calculate Normalized")},
    {"symnmf",
        (PyCFunction) symnmf,
        METH_VARARGS,
        PyDoc_STR("Perform full symNMF")},
   
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    symnmfmoduleMethods
};


PyMODINIT_FUNC
PyInit_symnmfmodule(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if(!m){
        return NULL;
    }
    return m;
}
