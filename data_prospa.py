# -*- coding: utf-8 -*-
# data import/export for prospa file formats
# Achim Gaedke, Sept 2009

# From Documentation: Prospa3.1Documentation/General Information/Data structures/Data File structure.htm

import numpy
import re

def read_nd_file(filename, expected_dimension=None):
        """
        reads data from file and returns numpy array with x,y,z index order
        """

        f=file(filename, "rb")
        # read header
        header=numpy.dtype([('identifiers', "S4", 3), ('datatype', numpy.int32, 1)])
        headerdata=numpy.fromstring(f.read(4*4), dtype=header)

        # check identifiers
        identifiers=headerdata["identifiers"][0]
        if str(identifiers[0])[::-1]!="PROS" or str(identifiers[1])[::-1]!="DATA":
                raise Exception("%s: invalid prospa data file identifier"%filename)
        
        if  str(identifiers[2])[3]!="V" or str(identifiers[2])[1]!=".":
                raise Exception("%s: invalid prospa data file version tag"%filename)

        datafileversion=float(identifiers[2][2])+float(identifiers[2][0])/10.0
        if datafileversion not in [1.0, 1.1]:
                raise Exception("%s: prospa data file version %s not supported"%(filename, identifiers[::-1]))

        if datafileversion==1.0:
                shapeheader=numpy.fromstring(f.read(3*4), dtype=numpy.int32)
        elif datafileversion>=1.1:
                shapeheader=numpy.fromstring(f.read(4*4), dtype=numpy.int32)

        if expected_dimension is not None:
                if expected_dimension==1 and shapeheader[1]!=1 and shapeheader[2]!=1 and shapeheader[3]!=1:
                        raise Exception("%s no 1d file"%filename)
                elif expected_dimension==2 and shapeheader[2]!=1 and shapeheader[3]!=1:
                        raise Exception("%s: no 2d file"%filename)
                elif expected_dimension==3 and shapeheader[3]!=1:
                        raise Exception("%s: no 3d file"%filename)
                elif expected_dimension>3:
                        raise Exception("higher dimensions than 3 not supported (by now...)")

        # read data
        datatype=headerdata["datatype"][0]
        if datatype in [500, 503, 504]:
                # real float
                # real float/complex: (x(,y(,z,)) value)
                data=numpy.fromstring(f.read(), dtype="<f4")
        elif datatype==501:
                # complex float
                data=numpy.fromstring(f.read(), dtype="<c8")
        elif datatype==502:
                # real double
                data=numpy.fromstring(f.read(), dtype="<f8")
        else:
                f.close()
                raise Exception("%s: prospa data type %d unknonwn"%(filename, headerdata["datatype"]))
        f.close()


        if datatype in [500, 501, 502]:
                # reshape for z,y,x dimensions 
                if expected_dimension is not None:
                        data=data.reshape(*(shapeheader[expected_dimension-1::-1]))
                else:
                        data=data.reshape(*(shapeheader[::-1]))
        
                ## and transpose axes to yield x,y,z order
                return data.transpose()

        if datatype in [503, 504]:
                dimensions=len(filter(lambda x:x!=1, shapeheader))
                expected_size=reduce(lambda x,y:x*y, shapeheader)
                datacolums=[]
                if datatype==503:
                        datasize=1 # real only
                if datatype==504:
                        datasize=2 # real and imag
                for d in xrange(dimensions):
                        datacolums.append(data[expected_size*d:expected_size*(d+1)])
                if datatype==503:
                        datacolums.append(data[expected_size*dimensions:])
                elif datatype==504:
                        datacolums.append(data[expected_size*dimensions::2]+
                                          numpy.complex64(1j)*data[expected_size*dimensions+1::2])
                return datacolums

def read_3d_file(filename):
        """
        reads data from file and returns numpy array with x,y,z index order
        """
        return read_nd_file(filename, 3)

def read_2d_file(filename):
        """
        reads data from file and returns numpy array with x,y index order
        """
        return read_nd_file(filename, 2)

def read_1d_file(filename):
        """
        reads data from file and returns numpy array with x,y index order
        """
        return read_nd_file(filename, 1)
        
def read_par_file(filename):
        """
        reads parameter file and returns contents as dictionary
        recognized data types were converted
        ToDo: Complex Numbers
        """
        parameter_file=file(filename, "r")
        parameters={}
        parameter_match=re.compile(r"(\S+) += +(.*)")
        fnumber_match=re.compile(r"^[+-]?\d*(\.\d*)?([eE][+-]?\d+)?$")
        dnumber_match=re.compile(r"^[+-]?\d+$")
        for l in parameter_file:
                param_line=re.match(parameter_match, l.strip())
                param_name=param_line.group(1)
                param_value=param_line.group(2)
                if param_value[0]=='"' and param_value[-1]=='"':
                        param_value=param_value[1:-1]
                elif param_value[0]=='[' and param_value[-1]==']':
                        param_value=map(float,param_value[1:-1].split(','))
                elif re.match(dnumber_match, param_value):
                        param_value=int(param_value)
                elif re.match(fnumber_match, param_value):
                        param_value=float(param_value)
                parameters[param_name]=param_value
        parameter_file.close()
        return parameters
