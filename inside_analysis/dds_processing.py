import os
import argparse
import numpy as np
from numpy import ma
import scipy.ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk


_DESCRIPTION = """ Processing Dose Delivey System output file. """

def decoding(input_file, output_file=None):
    """
    Processes the :file:`Treatment{seriesNumber}.txt` Dose Delivery System\'s (DDS)
    output file and returns (x, y) coordinates
    for each irradiated point in unit of mm.
    
    x-, y- axis refer to the DDS system of reference.
    
    Parameters
    ----------
    input_file : file, str
        Path to the DDS output file (.txt).
    output_file: filename, str, optional
        Absolute path where to save
        :func:`decoding`\'s output.
        By default, `x` and `y` arrays are not
        saved.
    
    Returns
    -------
    x: 1-D array, int
        x coordinate  of each irradiated point.
    y: 1-D array, int
        y coordinate of each irradiated point.
    """
    
    with open(input_file) as _input:
        list_file = list(_input)
    _idx = list_file.index('[SLICE_1]\n')   #   identifying header's end.
    list_file = [line.strip().split('\\') for line in list_file[_idx: ]]    #   skipping header
     #   removing overlength and slice identification's line
    _ma = np.asarray([len(line) == 5 for line in list_file])
    list_file = np.asarray(list_file)[_ma]
    x = np.stack(list_file, axis=-1)[1]
    y = np.stack(list_file, axis=-1)[2]
    #   cast firstly to type float and then to int
    #   is necessary, because float value in str format
    #   can't be directly casted to int value
    point_xy = np.column_stack((x, y)).astype(float).astype(int)
    if output_file is not None:
        _header = 'x[mm]    y[mm]'
        np.savetxt(output_file, point_xy,\
                   fmt='%d', delimiter='\t',\
                   header=_header)
    return point_xy.transpose()


def DDS_mask(x, y, x_FOV_dim=224, y_FOV_dim=112, voxel_dim=1.6, x_shift=0, y_shift=0):
    """
        Creates a binary mask from the pairs of coordinates
        (x, y) of the irradiated points returned by the
        Dose Delivery System: it's compatible with the PET scanner
        system of reference.
        Morphological operations, i.e. closing and dilation,
        are applied to the original mask in order to obtain
        a less fragmented region of interest.
        
        Parameters
        ----------
        x: 1D - array
            x coordinates of the irradiated points in [mm]
        y: 1D - array
            y coordinates of the irradiated points in [mm]
        x_FOV_dim: int or float
            PET scanner's FOV dimensions in [mm] along x-axis
            (default = 224, i.e. actual dimension)
        y_FOV_dim: int or float
            PET scanner's FOV dimensions in [mm] along y-axis
            (default = 112, i.e. actual dimension)
        voxel_dim: int or float
            Voxel dimension in [mm]. (default 1.6)
        x_shift: int or float
            Shift along x-axis in [mm]:
            this value has to be changed if the DDS isocenter
            and the PET scanner isocenter are not aligned.
            (default 0)
        y_shift: int or float
            Shift along y-axis in [mm]:
            this value has to be changed if the DDS isocenter
            and the PET scanner isocenter are not aligned.
            (default 0)
        
        Returns
        -------
        _mask : (M, N) - array
            Bidimensional binary mask where the irradiated points
            are set to **True**.
            M x N is the dimension of the PET scanner FOV
            in the axial plane (with reference to the beam direction).
    """
    _shape = (int(y_FOV_dim/voxel_dim), int(x_FOV_dim/voxel_dim))
    x = x + x_shift + x_FOV_dim * .5
    y = y + y_shift + y_FOV_dim * .5
    _unvalid = np.logical_not(np.logical_and(x >= 0, y >= 0))
    x = ma.masked_array(x, _unvalid).compressed()
    y = ma.masked_array(y, _unvalid).compressed()
    x = np.around(x/voxel_dim).astype(int)
    y = np.around(y/voxel_dim).astype(int)
    _mask = np.zeros(shape=_shape)
    for i, j in zip(x, y):
        _mask[j, i] = 1
    struct = scipy.ndimage.generate_binary_structure(2, 1)
    struct = scipy.ndimage.iterate_structure(struct, 2)
    _mask = scipy.ndimage.binary_closing(_mask, struct)
    struct = np.ones((3, 3), dtype = bool)
    _mask = scipy.ndimage.binary_dilation(_mask, struct)
    return np.flip(_mask, axis=0)

def mask_to_image(_mask, output_filename,
                  input_header='PET_header.txt',
                  input_mask_reference='mask_reference.nii'):
    """
        Saves Dose Delivery System\'s mask as image
        adapting its format to `filename`\'s specification.
        A 3D image is created with the same
        header as PET image acquired with INSIDE PET scanner.
        
        Parameters
        ----------
        _mask: (M, N) - array, int or bool
            Mask to save,
            as returned by :func:`DDS_mask`.
        output_filename: str
            Filename where to save 3D-mask. If no image extension is
            provided or is not supported,
            by default the image is saved in NIfTI format `.nii`.
    """
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), input_header), 'br') as header_file:
        _header = nib.nifti1.Nifti1Header.from_fileobj(header_file)
    _affine = _header.get_base_affine()
    _data_shape = _header.get_data_shape()
    mask_3d = np.repeat(_mask[np.newaxis, :, :],\
                        _data_shape[2], axis=0)
    mask_3d = mask_3d.transpose(2, 1, 0).astype(int)
    mask_3d_nii = nib.Nifti1Image(mask_3d,
                                  _affine,
                                  _header)
    try:
        nib.save(mask_3d_nii, output_filename)
    except nib.filebasedimages.ImageFileError:
        _name, _ext = os.path.splitext(output_filename)
        output_filename = output_filename.replace(_ext, '.nii')
        nib.save(mask_3d_nii, output_filename)
    
    mask_reference = sitk.ReadImage(
                                    os.path.join(
                                                 os.path.dirname(os.path.realpath(__file__)),\
                                                 input_mask_reference)
                                    )

    mask_3d = sitk.ReadImage(output_filename)
    mask_3d.CopyInformation(mask_reference)
    sitk.WriteImage(mask_3d, output_filename)
    return


parser = argparse.ArgumentParser(description=_DESCRIPTION)
parser.add_argument('file', help='DDS file output (.txt)')
parser.add_argument('-o', '--outputFile_path_coords',
                    default = None,
                    help='Coordinates\' output file (absolute path).')
parser.add_argument('-m', '--mask_filename', help='Mask image output file.')

if __name__ == '__main__':

    args = parser.parse_args()
    x, y = decoding(args.file, args.outputFile_path_coords)
    mask_ = DDS_mask(x, y)
    plt.figure('mask')
    plt.imshow(mask_)
    mask_to_image(mask_, args.mask_filename)
