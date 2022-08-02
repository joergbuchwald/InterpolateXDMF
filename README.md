# InterpolateXDMF

InterpolateXDMF is a python package for easy accessing XDMF/HDF5 files as outputed by Finite Element software like OpenGeoSys. It uses the meshio and the VTK python wrapper and linear interpolation between time steps and grid points to access any points in space and time within the simulation domain.

## 0. Installation

Note: InterpolateXDMF requires the vtk wrapper for python.
clone the repository and use pip to install the package

```shell
# pip install [--user] https://github.com/joergbuchwald/InterpolateXDMF/archive/refs/heads/master.zip
```

## 1. Documentation for InterpolateXDMF

You can find the documentation under [https://joergbuchwald.github.io/InterpolateXDMF-doc/](https://joergbuchwald.github.io/VTUinterface-doc/)



## 2. Quick start


Unittests can be run via

```shell
# python tests/test_interpolate.py
```
from the project root directory.

## 3. FAQ/Troubleshooting

Installation:
- If the vtk whell can't be found on PyPI you can checkout https://github.com/pyvista/pyvista/discussions/2064 for unofficial wheels.

As the input data is triangulated with QHull for the linear interpolation it might fail at boundaries or if a wrong input dimension is given.
Possible solutions:

- In order for interpolation to work correctly providing the correct dimension (set via `dim` keyword) of the problem is crucial.
- As the `dim` keyword specifies also the coordinates to use, InterpolateXDMF assumes that `dim=1` refers to the x coordinate and `dim=2` implies that the problem lies in the xy-plane by default. This can be changed by specifying `one_d_axis` for one dimension or `two_d_planenormal` for two dimensions.
- For some meshes it might help to adjust the number of points taken into account by the triangulation, which can be done using the `nneighbors` keyword. Default value is 20.
- Especially along boundaries, linear interpolation with the QHULL method often fails, this can be resolved by using nearest neighbor interpolation.
- Alternatively, you can change now the `interpolation_backend` from scipy to vtk and try out different interpolation kernels.

