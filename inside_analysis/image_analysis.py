# -*- coding: utf-8 -*-
#! /Users/gretadelnista/opt/miniconda3/bin/python
import sys
import numpy as np
from numpy import ma
import logging
from skimage import measure
from scipy import interpolate, stats
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn import metrics
logging.basicConfig(level='INFO')

if sys.flags.interactive:
    plt.ion()

def binary_surface(volume_mask):
    """
        Creates a (M, N, P) array like ``volume_mask``
        where the voxels belonging to the 2D surface mesh
        are set to 1. The 2D surface mesh is extracted using
        the marching cubes lewiner algorithm.
        
        Parameters
        ----------
        volume_mask: (M, N, P) array
            Binary mask of a 3D volume, where the voxels belonging
            to it have value **True** (or 1).
            
        Returns
        -------
        _binary_surface : (M, N, P) array
            Voxels belonging to the 2D surface mesh have value 1.
    """
    verts, _, _, _ = measure.marching_cubes_lewiner(volume_mask, level=0)
    _binary_surface = np.zeros_like(volume_mask)
    for i, j, k in verts:
        _binary_surface[int(i), int(j), int(k)] = 1
    return _binary_surface

def range_map(binary_surface):
    """
        Given a binary surface, computes the range
        (i.e. the distance in voxel between surface
        margin) along the axis 0 and return a 2D map.
        The range unit is voxel.
        
        Parameters
        ----------
        binary_surface : (M, N, P) array
            The voxels external to the surface have
            to be set to zero (or **False** if bool).
            
        Returns
        -------
        _range_map : (N, P) array
            The value of each pixel corresponds
            to the range computed in that position.
    """
    _shape = (binary_surface.shape[1], binary_surface.shape[2])
    _range_map = np.zeros(shape=_shape, dtype=np.float32)
    for i in range(_shape[0]):
        for j in range(_shape[1]):
            if binary_surface[:, i, j].any():
                points = np.nonzero(binary_surface[:, i, j])
                _max = np.amax(points)
                _min = np.amin(points)
                if _min != _max: _range_map[i, j] = _max - _min
                else: _range_map[i, j] = 1
    return _range_map

def end_point(binary_surface):
    """
        Given a binary surface, finds the end points
        along the axis 0 and returns a 2D map.
        End point unit is voxel.
        
        Parameters
        ----------
        binary_surface : (M, N, P) array
            The voxels external to the surface have
            to be set to zero (or **False** if bool).
            
        Returns
        -------
        _end_point_map : (N, P) array
            The value of each pixel corresponds
            to the end point computed in that position.
    """
    _shape = (binary_surface.shape[1], binary_surface.shape[2])
    _end_point_map = np.zeros(shape=_shape, dtype=np.float32)
    for i in range(_shape[0]):
        for j in range(_shape[1]):
            if binary_surface[:, i, j].any():
                points = np.nonzero(binary_surface[:, i, j])
                _end_point_map[i, j] = np.amax(points)
    return _end_point_map


def BEV(mask1, mask2, voxel_dim=1):
    """
        Creates a 2D range difference maps
        comparing two 3D binary masks of a volume
        of interest: the range is computed along the axis 0.
        ``voxel_dim`` is the voxel dimension:
        if provided, the range is computed according
        to it, so the returned map is in physical dimension,
        otherwise it's in voxel unit (by default, 'voxel_dim' = 1 ).
        
        Parameters
        ----------
        mask1 : (M, N, P) array
        mask2 : (M, N, P) array
            Binary masks of a volume of interest.
        voxel_dim : float, optional
            Physical voxel dimension.
            
        Returns
        -------
        range_difference_map :  (N, P) numpy.ma.core.MaskedArray
            Range difference map between the two given masks.
    """
    #   Extracting a 2D binary surface from the 3D volume.
    binary_surface_1 = binary_surface(mask1)
    binary_surface_2 = binary_surface(mask2)
    plt.figure('Binary surface')
    plt.subplot(1, 2, 1)
    x = int(mask1.shape[2] / 2)
    plt.imshow(binary_surface_1[:, :, x], origin='lower')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(binary_surface_2[:, :, x], origin='lower')
    plt.axis('off')
    #   Computing range_map and converting it to physical dimension.
    range_map_1 = range_map(binary_surface_1) * voxel_dim
    range_map_2 = range_map(binary_surface_2) * voxel_dim
    plt.figure('Range map')
    plt.subplot(1, 2, 1)
    plt.imshow(range_map_1)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(range_map_2)
    plt.axis('off')
    plt.colorbar()
    #   Searching common pairs of non null points
    common_points = np.logical_and((range_map_1 > 0), (range_map_2 > 0))
    uncommon_points = np.logical_not(common_points)
    plt.figure('Uncommon points')
    plt.imshow(uncommon_points)
    #   Masking range map, so that just common pairs of points
    #   are taken into account.
    range_map_1_masked = ma.masked_array(range_map_1, \
                                         mask=uncommon_points,\
                                         fill_value=-999
                                         )
    range_map_1_masked.filled()
    range_map_2_masked = ma.masked_array(range_map_2, \
                                         mask=uncommon_points,\
                                         fill_value=-999
                                         )
    range_map_2_masked.filled()
    #   Computing range difference map
    range_difference_map = np.subtract(range_map_1_masked, range_map_2_masked)
    
    return range_difference_map

def MP(image1, image2, mask=None, voxel_dim=1):
    """
        Computes a range difference map between
        ``image1`` and ``image2`` according to
        the *Middle Point* algorithm.
        If a ``mask`` is given,
        only the pixels that are set to **True**
        are considered valid.
        In addition, only the profiles with an integrated
        activity above the 20% of the maximum
        integrated activity are taken into account.
        
        Parameters
        ----------
        image1 : SimpleITK Image Object or filepath to images.
        image2 : SimpleITK Image Object or filepath to images.
            Images to be compared.
        mask: (M, N) array, bool
            By default is **None**.
            Its shape has to match the images'
            axial dimensions.
        voxel_dim: int or float
            Voxel physical dimension.
            By default is set to 1.
            
        Returns
        -------
        mp: (M, N) numpy.ma.core.MaskedArray
            Range difference map between the two given images.
    """
    
    if isinstance(image1, sitk.SimpleITK.Image):    pass
    else:   image1 = sitk.ReadImage(image1)
    if isinstance(image2, sitk.SimpleITK.Image):    pass
    else:   image2 = sitk.ReadImage(image2)
    
    data1 = sitk.GetArrayFromImage(image1)
    data2 = sitk.GetArrayFromImage(image2)
    _shape = data1.shape
    #   Computing the integrated activity along each profile
    #   and creating a binary mask of those above the 20%
    #   of the maximum.
    int_activity = np.sum(data1, axis=0)
    _max = np.amax(int_activity)
    mask1 = (int_activity > .2 * _max)
    int_activity = np.sum(data2, axis=0)
    _max = np.amax(int_activity)
    mask2 = (int_activity > .2 * _max)
    #   Computing a unique common mask.
    common_points = np.logical_and(mask1, mask2)
    #   Taking into account for 'mask' if given.
    if mask is not None:
        common_points = np.logical_and(common_points, mask)
    #   Defining a 3D mask for masking operations.
    _unvalid = np.logical_not(common_points)
    _unvalid_3D = np.repeat(_unvalid[np.newaxis, :, :],\
                            _shape[0], axis=0)
    #   Masking data array, otherwise invalid divisions
    #   are encountered where the datas are all null.
    data_1_ma = ma.masked_array(data1, _unvalid_3D, fill_value=0)
    data_1_ma_norm = data_1_ma / np.amax(data_1_ma, axis=0)
    data_2_ma = ma.masked_array(data2, _unvalid_3D, fill_value=0)
    data_2_ma_norm = data_2_ma / np.amax(data_2_ma, axis=0)
    #   Points above 25% and 50% in each profile.
    data_1_25 = ((data_1_ma_norm >= .25) * data_1_ma_norm).filled()
    data_1_50 = ((data_1_ma_norm >= .50) * data_1_ma_norm).filled()
    data_2_25 = ((data_2_ma_norm >= .25) * data_2_ma_norm).filled()
    data_2_50 = ((data_2_ma_norm >= .50) * data_2_ma_norm).filled()
    #   (M, N) array where storing end points' position
    index_1_25 = np.zeros(shape=(_shape[1], _shape[2]))
    index_1_50 = np.zeros(shape=(_shape[1], _shape[2]))
    index_2_25 = np.zeros(shape=(_shape[1], _shape[2]))
    index_2_50 = np.zeros(shape=(_shape[1], _shape[2]))

    for i in range(_shape[1]):
        for j in range(_shape[2]):
            try:
                index_1_25[i, j] = np.amax(np.nonzero(data_1_25[:, i, j]))
                index_1_50[i, j] = np.amax(np.nonzero(data_1_50[:, i, j]))
                index_2_25[i, j] = np.amax(np.nonzero(data_2_25[:, i, j]))
                index_2_50[i, j] = np.amax(np.nonzero(data_2_50[:, i, j]))
            except ValueError:
                #   this exception occurs if all values
                #   along the longitudinal direction
                #   in the pixel (i, j) are null.
                pass

    mp_1 = (index_1_25 + index_1_50) * .5 * voxel_dim
    mp_2 = (index_2_25 + index_2_50) * .5 * voxel_dim
    mp = ma.masked_array((mp_1 - mp_2), mask=_unvalid)
    return mp

def shift_method(image1, image2, mask=None, voxel_dim=1.6):
    """
        Computes a range difference map between
        ``image1`` and ``image2`` by means of
        `Most likely shift` approach.
        If no ``mask`` is given,
        the differences are computed over the
        whole axial plane (with reference to the beam direction),
        otherwise the region of interest is restricted
        to that defined by 'mask'.
        In any case, only the profile with a maximum activity
        above the 20% of the absolute maximum activity in the image
        are considered.
        
        Parameters
        ----------
        image1 : SimpleITK Image Object or filepath to images.
        image2 : SimpleITK Image Object or filepath to images.
            Images to compare.
        mask : (M, N) array, bool
            By default is **None**.
            Its shape has to match the images'
            axial dimensions.
        voxel_dim : float
            Voxel physical dimension in mm.
            By default is set to 1.6 mm.
            
        Returns
        -------
        map_ma : (M, N) numpy.ma.core.MaskedArray
            Map of most likely shift between the
            two compared images.
    """
    
    if isinstance(image1, sitk.SimpleITK.Image):    pass
    else:   image1 = sitk.ReadImage(image1)
    if isinstance(image2, sitk.SimpleITK.Image):    pass
    else:   image2 = sitk.ReadImage(image2)
    
    data1 = sitk.GetArrayFromImage(image1)
    data2 = sitk.GetArrayFromImage(image2)
    _shape = data1.shape
    delta = np.linspace(-10, 10, endpoint=True, num=320) #   [voxel] => 320 steps of 0.1 mm
    id_20 = np.greater_equal(data1, np.amax(data1, axis=0)*0.2)
    id_3 = np.less_equal(data1, np.amax(data1, axis=0)*0.03)
    map = np.zeros((_shape[1], _shape[2]))
    for y in range(1, _shape[1] - 1):
        for x in range (1, _shape[2] - 1):
            if not id_20[:, y, x].all() and np.amax(data1[:, y, x]) > 0.2*np.amax(data1):
                d2 = np.mean(data2[:, y-1:y+2, x-1:x+2], axis=(1,2))
                d2 = d2/np.amax(d2)
                d1 = np.mean(data1[:, y-1:y+2, x-1:x+2], axis=(1,2))
                d1 = d1/np.amax(d1)
                interp_function_data2 = interpolate.interp1d(np.arange(0, _shape[0], 1), d2)
                interp_function_data1 = interpolate.interp1d(np.arange(0, _shape[0], 1), d1)
                tmp = np.amax(np.nonzero(id_20[:, y, x]))
                start_point = tmp - int(10/voxel_dim)   #   start_point is located 1cm before 20% activity.
                tmp_ = np.nonzero(id_3[:, y, x])[0]
                end_point = _shape[0] - max(delta) - 1
                zz = np.arange(start_point, end_point, 0.5)
                data1_compared = np.repeat(interp_function_data1(zz)[np.newaxis, :],\
                                  len(delta), axis=0)
                zz = np.repeat(zz[np.newaxis, :],\
                                   len(delta), axis=0) - np.vstack(delta)
                data2_compared =  interp_function_data2(zz)
                diff = np.sum((data1_compared - data2_compared)**2, axis=1)
                map[y, x] = delta[np.argmin(diff)]*voxel_dim
    if mask is not None:
        mask_ = np.logical_not(mask)
    else: mask_ = None
    map_ma = ma.masked_array(map, mask_, fill_value=0)
    logging.info('Basic map describing statistics: {}'.format(stats.describe(map_ma.compressed())))
    return map_ma



def RMSE(image1, image2, mask=None, voxel_dim=1.6):
    """
        Given two images,computes Root Mean Squared
        Error between each
        profile along beam directions.
        The comparision is made from 1 cm before
        the 20 % profile's activity to the point
        located at the 3 % activity.
        If no ``mask`` is given,
        the RMSE is computed over the
        whole axial plane (with reference to the beam direction),
        otherwise the region of interest is restricted
        to that defined by 'mask'.
        In any case, only the profile with a maximum activity
        above the 20% of the absolute maximum activity in the image
        are considered.
        
        Parameters
        ----------
        image1 : SimpleITK Image Object or filepath to images.
        image2 : SimpleITK Image Object or filepath to images.
            Images to compare.
        mask : (M, N) array, bool
            By default is **None**.
            Its shape has to match the images'
            axial dimensions.
        voxel_dim : float
            Voxel physical dimension in mm.
            By default is set to 1.6 mm.

    """
    
    if isinstance(image1, sitk.SimpleITK.Image):    pass
    else:   image1 = sitk.ReadImage(image1)
    if isinstance(image2, sitk.SimpleITK.Image):    pass
    else:   image2 = sitk.ReadImage(image2)
    
    data1 = sitk.GetArrayFromImage(image1)
    data2 = sitk.GetArrayFromImage(image2)
    _shape = data1.shape
    id_20 = np.greater_equal(data1, np.amax(data1, axis=0)*0.2)
    id_3 = np.less_equal(data1, np.amax(data1, axis=0)*0.03)
    map = np.zeros((_shape[1], _shape[2]))
    for y in range(1, _shape[1] - 1):
        for x in range (1, _shape[2] - 1):
            if not id_20[:, y, x].all() and np.amax(data1[:, y, x]) > 0.2*np.amax(data1):
                d2 = np.mean(data2[:, y-1:y+2, x-1:x+2], axis=(1,2))
                d2 = d2/np.amax(d2)
                d1 = np.mean(data1[:, y-1:y+2, x-1:x+2], axis=(1,2))
                d1 = d1/np.amax(d1)
                tmp = np.amax(np.nonzero(id_20[:, y, x]))
                start_point = tmp - int(10/voxel_dim)   #   start_point is located 1cm before 20% activity.
                tmp_ = np.nonzero(id_3[:, y, x])[0]
                end_point = np.amin(tmp_[tmp_>tmp])     #   end_point is located at 3% activity (if possible
                map[y, x] = metrics.mean_squared_error(d1[start_point:end_point + 1], d2[start_point:end_point + 1], squared=False)
    map_ma = ma.masked_array(map, np.logical_not(mask), fill_value=0)
    return map_ma


if __name__ == '__main__':
    pass
