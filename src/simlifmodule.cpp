#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <math.h>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "utils.hpp"


static PyObject* sim_lif(PyObject *self, PyObject *args);

static PyMethodDef simlif_methods[] = {
    { "simu", sim_lif, METH_VARARGS, "LIF network Brunel"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef simlifmodule =
{
    PyModuleDef_HEAD_INIT,
    "brunel",
    "", /* module documentation*/
    -1,
    simlif_methods
};

// PyMODINIT_FUNC PyInit_stdp(void) {
//     import_array();
//     return PyModule_Create(&simlifmodule);
// }

PyMODINIT_FUNC PyInit_brunel(void) {
    import_array();
    return PyModule_Create(&simlifmodule);
}



static PyObject* sim_lif(PyObject *dummy, PyObject *args)
{
  /* arguments */
  // PyObject* pars = NULL;
  // PyArrayObject* Varg= NULL;
  // PyArrayObject* Sarg= NULL;
  // PyArrayObject* Sincrarg= NULL;

  read_params();

  std::cout << " -- -- --- setup --- -- -- "  << "\n";
  init_state_vectors();

  gen_conmat();

  
  integrate_brunel();

  delete_state_vectors();

  Py_RETURN_NONE;
  // return NULL;
}
