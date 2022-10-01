# -*- coding: utf-8 -*-
"""
InterpolateXDMF is a python package for easy accessing XDMF/HDF5 files as
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
import time

class XDMFReader:
    """
    Interface for XDMF data

    Parameters
    ----------
    filename : `str`
    nneighbors : `int`, optional
                 default: 20
    dim : `int`, optional
          default: 3
    one_d_axis : `int`
                 between 0 and 2, default: 0
    two_d_planenormal : `int`
                 between 0 and 2, default: 2
    interpolation_backend : `str`
                 scipy or vtk
    """
    def __init__(self, filename, nneighbors=20, dim=3, one_d_axis=0, two_d_planenormal=2,
                                                            interpolation_backend="scipy"):
        time0 = time.time()
        self.filename = filename
        self._fix_xdmf_file()
        self.dim = dim
        self.one_d_axis = one_d_axis
        self.two_d_planenormal = two_d_planenormal
        self.interpolation_backend = interpolation_backend
        self.timesteps = []
        with meshio.xdmf.TimeSeriesReader(filename) as reader:
            self.points, self.cells = reader.read_points_cells()
            self.h5_data = {}
            timesteps = range(reader.num_steps)
            for t in timesteps:
                self.h5_data[t]  = reader.read_data(t)
                self.timesteps.append(self.h5_data[t][0])
        self.timesteps = np.array(self.timesteps)
        if self.points.shape[1] == 2:
            self.points = np.hstack([self.points, np.zeros((len(self.points), 1))])
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(self.points))
        self.output = vtk.vtkUnstructuredGrid()
        self.output.SetPoints(vtk_points)
        cell_types = np.array([], dtype=np.ubyte)
        cell_offsets = np.array([], dtype=int)
        cell_conn = np.array([], dtype=int)
        for cell in self.cells:
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
        for t in timesteps:
            self.data_objects.append(VTUIOObject(self.output, self.points, self.h5_data[t], nneighbors=nneighbors,
                    dim=dim, one_d_axis=one_d_axis, two_d_planenormal=two_d_planenormal,
                    interpolation_backend=interpolation_backend))
        #time1 = time.time()
        #print(f"time constructor: {time1-time0}")

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
        """
        Return vtu point field as numpy array.

        Parameters
        ----------
        fieldname : `str`
        timestep : `int`
        """
        return self.h5_data[timestep][1][fieldname]
    def get_cell_field(self, fieldname, timestep=0):
        """
        Return vtu cell field as numpy array.

        Parameters
        ----------
        fieldname : `str`
        timestep : `int`
        """
        return self.h5_data[timestep][2][fieldname]
    def get_point_field_names(self, timestep=0):
        """
        Get names of all point fields in the vtu file.

        Parameters
        ----------
        timestep : `int`, optional
                    default: 0
        """
        return list(self.h5_data[timestep][1].keys())
    def get_cell_field_names(self, timestep=0):
        """
        Get names of all cell fields in the vtu file.
        Parameters
        ----------
        timestep : `int`, optional
                    default: 0
        """
        return list(self.h5_data[timestep][2].keys())
    def read_time_slice(self, time, fieldname):
        """
        Print field "fieldname" at time "time".

        Parameters
        ----------
        time : `float`
        fieldname : `str`
        """
        for i, ts in enumerate(self.timesteps):
            if time == ts:
                field = self.data_objects[i].get_point_field(fieldname)
        else:
            time1 = 0.0
            time2 = 0.0
            timestep = 0
            for i, ts in enumerate(self.timesteps):
                try:
                    if ts < time < self.timesteps[i+1]:
                        time1 = ts
                        time2 = self.timesteps[i+1]
                        timestep = i
                except IndexError:
                    print("time is out of range")
            if (time1 == 0.0) and (time2 == 0.0):
                print("time is out of range")
            else:
                vtu1 = self.data_objects[timestep]
                vtu2 = self.data_objects[timestep+1]
                field1 = vtu1.get_point_field(fieldname)
                field2 = vtu2.get_point_field(fieldname)
                fieldslope = (field2-field1)/(time2-time1)
                field = field1 + fieldslope * (time-time1)
        return field
    def read_set_data(self, time, fieldname, pointsetarray = None, data_type="point", interpolation_method="linear"):
        """
        Get data of field "fieldname" at time "time" alon a given "pointsetarray".

        Parameters
        ----------
        time : `float`
        fieldname : `str`
        pointsetarray : `list`, `numpy.ndarray` or `str`
                        containing a list of 3-tuples
        interpolation_method : `str`
                               default: 'linear'
        """
        if pointsetarray is None:
            raise RuntimeError("No pointsetarray given.")
        for i, ts in enumerate(self.timesteps):
            if time == ts:
                field = self.data_objects[i].get_set_data(fieldname, pointsetarray, data_type=data_type, interpolation_method=interpolation_method)
        else:
            time1 = 0.0
            time2 = 0.0
            timestep = 0
            for i, ts in enumerate(self.timesteps):
                try:
                    if ts < time < self.timesteps[i+1]:
                        time1 = ts
                        time2 = self.timesteps[i+1]
                        timestep = i
                except IndexError:
                    print("time is out of range")
            if (time1 == 0.0) and (time2 == 0.0):
                print("time is out of range")
            else:
                vtu1 = self.data_objects[timestep]
                vtu2 = self.data_objects[timestep+1]
                field1 = vtu1.get_set_data(fieldname, pointsetarray, data_type=data_type, interpolation_method=interpolation_method)
                field2 = vtu2.get_set_data(fieldname, pointsetarray, data_type=data_type, interpolation_method=interpolation_method)
                fieldslope = (field2-field1)/(time2-time1)
                field = field1 + fieldslope * (time-time1)
        return field
    def read_time_series(self, fieldname, pts=None, data_type="point", interpolation_method="linear"):
        """
        Return time series data of field "fieldname" at points pts.
        Also a list of fieldnames can be provided as "fieldname"

        Parameters
        ----------
        fieldname : `str`
        pts : `dict`, optional
        data_type : `str` optional
              "point" or "cell"
        interpolation_method : `str`, optional
                               default: 'linear
        """
        if pts is None:
            raise RuntimeError("No points given")
        resp_t = {}
        for pt in pts:
            if isinstance(fieldname, str):
                resp_t[pt] = []
            elif isinstance(fieldname, list):
                resp_t[pt] = {}
                for field in fieldname:
                    resp_t[pt][field] = []
        for i, _ in enumerate(self.timesteps):
            vtu = self.data_objects[i]
            if self.interpolation_backend == "scipy":
                if i == 0:
                    nb = vtu.get_neighbors(pts, data_type=data_type)
                if isinstance(fieldname, str):
                    data = vtu.get_data_scipy(nb, pts, fieldname, data_type=data_type,
                            interpolation_method=interpolation_method)
                    for pt in pts:
                        resp_t[pt].append(data[pt])
                elif isinstance(fieldname, list):
                    data = {}
                    for field in fieldname:
                        data[field] = vtu.get_data_scipy(nb, pts, field, data_type=data_type,
                                interpolation_method=interpolation_method)
                    for pt in pts:
                        for field in fieldname:
                            resp_t[pt][field].append(data[field][pt])
            elif self.interpolation_backend == "vtk":
                if data_type != "point":
                    raise RuntimeError("reading cell data is not working with vtk backend yet")
                if isinstance(fieldname, str):
                    data = vtk_to_numpy(
                        vtu.get_data_vtk(pts, interpolation_method=interpolation_method).GetArray(fieldname))
                    for j, pt in enumerate(pts):
                        resp_t[pt].append(data[j])
                elif isinstance(fieldname, list):
                    data = {}
                    vtkdata = vtu.get_data_vtk(pts, interpolation_method=interpolation_method)
                    for field in fieldname:
                        data[field] = vtk_to_numpy(vtkdata.GetArray(fieldname))
                    for j, pt in enumerate(pts):
                        for field in fieldname:
                            resp_t[pt][field].append(data[field][j])
        resp_t_array = {}
        for pt, field in resp_t.items():
            if isinstance(fieldname, str):
                resp_t_array[pt] = np.array(field)
            elif isinstance(fieldname, list):
                resp_t_array[pt] = {}
                for field_, fieldarray in resp_t[pt].items():
                    resp_t_array[pt][field_] = np.array(fieldarray)
        return resp_t_array

    def read_aggregate(self, fieldname, agg_fct, data_type="point", pointsetarray=None):
        """
        Return time series data of an aggregate function for field "fieldname".

        Parameters
        ----------
        fieldname : `str` or `list`
        agg_fct : `str`,
              can be: "min", "max" or "mean"
        data_type : `str` optional
              "point" or "cell"
        pointsetarray : `str`, `list` or `numpy.ndarray`
                        defines a submesh
                        if `str` pointsetarray is construed as filename containing the mesh
        """
        agg_fcts = {"min": np.min,
                    "max": np.max,
                    "mean": np.mean}
        resp_t = {}
        if isinstance(fieldname, str):
            resp_t = []
        elif isinstance(fieldname, list):
            resp_t = {}
            for field in fieldname:
                resp_t[field] = []
        if not pointsetarray is None:
            pointsetarray = np.array(pointsetarray)
        submeshindices = None
        for i, _ in enumerate(self.timesteps):
            vtu = self.data_objects[i]
            if (i == 0) and (not pointsetarray is None):
                submeshindices = vtu.get_nearest_indices(pointsetarray)
            if isinstance(fieldname, str):
                if data_type == "point":
                    data = agg_fcts[agg_fct](vtu.get_point_field(fieldname)[submeshindices])
                elif data_type == "cell":
                    data = agg_fcts[agg_fct](vtu.get_cell_field(fieldname)[submeshindices])
                resp_t.append(data)
            elif isinstance(fieldname, list):
                for field in fieldname:
                    if data_type == "point":
                        data = agg_fcts[agg_fct](vtu.get_point_field(field)[submeshindices])
                    elif data_type == "cell":
                        data = agg_fcts[agg_fct](vtu.get_cell_field(field)[submeshindices])
                    resp_t[field].append(data)
        return resp_t
    def write_pvd(self, filename):
        """
        Writes data to PVD/VTU

        Parameters
        ----------
        filename : `str`
        """
        root = ET.Element("VTKFile")
        root.attrib["type"] = "Collection"
        root.attrib["version"] = "0.1"
        root.attrib["byte_order"] = "LittleEndian"
        root.attrib["compressor"] = "vtkZLibDataCompressor"
        collection = ET.SubElement(root,"Collection")
        timestepselements = []
        #pvdwriter
        for i, timestep in enumerate(self.timesteps):
            timestepselements.append(ET.SubElement(collection, "DataSet"))
            timestepselements[-1].attrib["timestep"] = str(timestep)
            timestepselements[-1].attrib["group"] = ""
            timestepselements[-1].attrib["part"] = "0"
            timestepselements[-1].attrib["file"] = f"{filename.split('.')[0]}_{i}_{timestep}.vtu"
            mesh = meshio.Mesh(self.points, self.cells,
                        point_data=self.h5_data[i][1],
                        cell_data=self.h5_data[i][2])
            mesh.write(timestepselements[-1].attrib["file"])
        tree = ET.ElementTree(root)
        tree.write(filename, encoding="ISO-8859-1", xml_declaration=True, pretty_print=True)



class VTUIOObject(vtuIO.VTUIO):
    def __init__(self, obj, points, h5_data, nneighbors=20, dim=3, one_d_axis=0, two_d_planenormal=2,
                                                        interpolation_backend="scipy"):
        if interpolation_backend == "vtk":
            self.output = vtk.vtkUnstructuredGrid()
            self.output.ShallowCopy(obj)
        else:
            self.output = obj
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
        if interpolation_backend == "vtk":
            self.pdata = self.output.GetPointData()
            self.cdata = self.output.GetCellData()
            cellfieldnames = self.get_cell_field_names()
            for cellfield in cellfieldnames:
                q = self.cdata.AddArray(numpy_to_vtk(self.get_cell_field(cellfield)))
                self.cdata.GetArray(q).SetName(cellfield)
            pointfieldnames = self.get_point_field_names()
            for pointfield in pointfieldnames:
                q = self.pdata.AddArray(numpy_to_vtk(self.get_point_field(pointfield)))
                self.pdata.GetArray(q).SetName(pointfield)
        # interpolation settings
        self.interpolation_backend = interpolation_backend
        self.vtk_gaussian_sharpness = 4.0
        self.vtk_gaussian_radius = 0.5
        self.vtk_gaussian_footprint_to_n_closest = False
        self.vtk_shepard_power_parameter = 2.0
        self.vtk_shepard_radius = 0.5

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
