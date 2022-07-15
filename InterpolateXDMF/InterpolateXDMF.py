# -*- coding: utf-8 -*-
"""
NetCDFInterpolate is a python package for easy accessing VTU/PVD files as
outputed by Finite Element software like OpenGeoSys. It uses the VTK python
wrapper and linear interpolation between time steps and grid points access
any points in and and time within the simulation domain.

Copyright (c) 2012-2022, OpenGeoSys Community (http://www.opengeosys.org)
            Distributed under a Modified BSD License.
              See accompanying file LICENSE or
              http://www.opengeosys.org/project/license

"""

# pylint: disable=C0103, R0902, R0914, R0913
import numpy as np
import pandas as pd
from lxml import etree as ET
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import meshio
import vtuIO

class XDMFreader:
    """
    Interface for XDMF data

    """
    def __init__(self, filename, nneighbors=20, dim=3, one_d_axis=0, two_d_planenormal=2,
                                                            interpolation_backend="scipy"):
        self.filename = filename
        self._fix_xdmf_file()
        self.dim = dim
        self.one_d_axis = one_d_axis
        self.two_d_planenormal = two_d_planenormal
        self.interpolation_backend = interpolation_backend
        with meshio.xdmf.TimeSeriesReader(filename) as reader:
            self.points, cells = reader.read_points_cells()
            self.h5_data = {}
            self.timesteps = range(reader.num_steps)
            for t in self.timesteps:
                self.h5_data[t]  = reader.read_data(t)
        # Points
        if self.points.shape[1] == 2:
            self.points = np.hstack([self.points, np.zeros((len(self.points), 1))])
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(self.points))
        self.output = vtk.vtkUnstructuredGrid()
        self.output.SetPoints(vtk_points)
        cell_types = np.array([], dtype=np.ubyte)
        cell_offsets = np.array([], dtype=int)
        cell_conn = np.array([], dtype=int)
        for cell in cells:
            meshio_type = cell.type
            data = cell.data
            vtk_type = meshio._vtk_common.meshio_to_vtk_type[meshio_type]
            ncells, npoints = data.shape
            cell_types = np.hstack([cell_types, np.full(ncells, vtk_type, dtype=np.ubyte)])
            offsets = len(cell_conn) + (1 + npoints) * np.arange(ncells, dtype=int)
            cell_offsets = np.hstack([cell_offsets, offsets])
            conn = np.hstack([npoints * np.ones((ncells, 1), dtype=int), data]).flatten()
            cell_conn = np.hstack([cell_conn, conn])
        connectivity = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(
                cell_conn.astype(np.int64), deep=1)
        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(cell_types), connectivity)
        self.output.SetCells(vtk.util.numpy_support.numpy_to_vtk(
                cell_types, deep=1, array_type=vtk.vtkUnsignedCharArray().GetDataType() ),
                vtk.util.numpy_support.numpy_to_vtk(cell_offsets, deep=1,
                array_type=vtk.vtkIdTypeArray().GetDataType()), vtk_cells, )
        self.data_objects = []
        for t in self.timesteps:
            self.data_objects.append(vtuIOobject(self.output,self.points, self.h5_data[t], nneighbors=nneighbors,
                dim=dim, one_d_axis=one_d_axis, two_d_planenormal=two_d_planenormal,
                interpolation_backend=interpolation_backend))
    def _fix_xdmf_file(self):
        self.tree = ET.parse(self.filename)
        grid_zero = self.tree.find("./Domain/Grid/Grid")
        grid_zero.attrib["GridType"] = "Uniform"
        data_items = self.tree.findall("./Domain/Grid/Grid/Attribute/DataItem")
        data_items.append(self.tree.find("./Domain/Grid/Grid/Geometry/DataItem"))
        data_items.append(self.tree.find("./Domain/Grid/Grid/Topology/DataItem"))
        for item in data_items:
            item.text = item.text.replace(":meshes",":/meshes")
        ET.indent(self.tree, space="    ")
        self.tree.write(self.filename,
                            encoding="ISO-8859-1",
                            xml_declaration=True,
                            pretty_print=True)
    @property
    def header(self):
        header_dict = {"N Cells": [str(len(self.cell_center_points))], "N Points": [str(len(self.points))],
                "N Cell Arrays": [len(self.get_cell_field_names())],
                "N Point Arrays": [len(self.get_point_field_names())],
                "X Bounds": [str(np.min(self.points[:,0])) + ", " + str(np.max(self.points[:,0]))]}
        if self.dim > 1:
            header_dict["Y Bounds"] = [str(np.min(self.points[:,1]))+" "+str(np.max(self.points[:,1]))]
        if self.dim > 2:
            header_dict["Z Bounds"] = [str(np.min(self.points[:,2]))+" "+str(np.max(self.points[:,2]))]
        df = pd.DataFrame(header_dict).T
        return df.rename({0:"Information"}, axis='columns')

    @property
    def data_arrays(self):
        pf_names = self.get_point_field_names()
        cf_names = self.get_cell_field_names()
        data_array_dict = {}
        for name in pf_names:
            field = self.get_point_field(name)
            if field.ndim == 1:
                components = 1
            else:
                components = field.shape[1]
            data_array_dict[name] = ["point", components, np.min(field), np.mean(field), np.max(field)]
        for name in cf_names:
            field = self.get_cell_field(name)
            data_array_dict[name] = ["cell", components, np.min(field), np.mean(field), np.max(field)]
        df = pd.DataFrame(data_array_dict).T
        return df.rename({0:"type", 1: "components", 2: "Min", 3: "Mean", 4: "Max"}, axis='columns')

    @property
    def cell_center_points(self):
        """
        Method for obtaining cell center points
        """
        ccf = vtk.vtkCellCenters()
        ccf.SetInputData(self.output)
        ccf.VertexCellsOn()
        ccf.Update()
        cell_center_points = vtk_to_numpy(ccf.GetOutput().GetPoints().GetData())
        if self.dim == 1:
            self.one_d_axis = self.one_d_axis
            cell_center_points = cell_center_points[:, self.one_d_axis]
        if self.dim == 2:
            self.plane = [0, 1, 2]
            self.plane.pop(self.two_d_planenormal)
            cell_center_points = np.delete(cell_center_points, self.two_d_planenormal, 1)
        return cell_center_points

    def get_point_field(self, fieldname, timestep=0):
        return self.h5_data[timestep][1][fieldname]
    def get_cell_field(self, fieldname, timestep=0):
        return self.h5_data[timestep][2][fieldname]
    def get_point_field_names(self, timestep=0):
        return list(self.h5_data[timestep][1].keys())
    def get_cell_field_names(self, timestep=0):
        return list(self.h5_data[timestep][2].keys())
    def read_time_slice(self, fieldname, time, interpolation_method="linear"):
        pass
    def read_data(self, fieldname, time,pts=None):
        pass
    def read_set_data(self, fieldname, time, pointsetarray = None, interpolation_method="linear"):
        pass
    def read_time_series(self, fieldname, pts=None, interpolation_method="linear"):
        pass
    def read_aggregate(self):
        pass



class vtuIOobject(vtuIO.VTUIO):
    def __init__(self, obj, points, h5_data, nneighbors=20, dim=3, one_d_axis=0, two_d_planenormal=2,
                                                        interpolation_backend="scipy"):
        self.output = obj
        #self.pdata = self.output.GetPointData()
        #self.cdata = self.output.GetCellData()
        self.points = points
        self._cell_center_points = None
        self.dim = dim
        self.nneighbors = nneighbors
        self.one_d_axis=one_d_axis
        self.two_d_planenormal = two_d_planenormal
        if self.dim == 1:
            self.one_d_axis = one_d_axis
            self.points = self.points[:,one_d_axis]
        if self.dim == 2:
            self.plane = [0, 1, 2]
            self.plane.pop(two_d_planenormal)
            self.points = np.delete(self.points, two_d_planenormal, 1)
        self.interpolation_backend = interpolation_backend
        self.h5_data = h5_data

    def get_point_field(self, fieldname):
        """
        Return vtu cell field as numpy array.

        Parameters
        ----------
        fieldname : `str`
        """
        field = self.h5_data[1][fieldname]
        return field

    def get_cell_field(self, fieldname):
        """
        Return vtu point field as numpy array.

        Parameters
        ----------
        fieldname : `str`
        """
        field = self.h5_data[2][fieldname]
        return field

    def get_cell_field_names(self):
        """
        Get names of all cell fields in the vtu file.
        """
        return list(self.h5_data[2].keys())

    def get_point_field_names(self):
        """
        Get names of all point fields in the vtu file.
        """
        return list(self.h5_data[1].keys())

    def func_to_field(self):
        raise NotImplementedError
    def func_to_m_dim_field(self):
        raise NotImplementedError
    def point_data_to_cell_data(self):
        raise NotImplementedError
    def delete_point_field(self):
        raise NotImplementedError
    def delete_cell_field(self):
        raise NotImplementedError
    def write_point_field(self):
        raise NotImplementedError
    def write_cell_field(self):
        raise NotImplementedError
    def write(self):
        raise NotImplementedError
