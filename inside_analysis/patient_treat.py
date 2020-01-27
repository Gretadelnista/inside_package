# -*- coding: utf-8 -*-
#! /Users/gretadelnista/opt/miniconda3/bin/python
import os
import sys
import glob
import SimpleITK as sitk
import numpy as np
import logging
from numpy import ma
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from skimage import filters, util, measure, morphology, feature
logging.basicConfig(level='INFO')
if sys.flags.interactive:
    plt.ion()

VOXEL_DIM = 1.6 # [mm] Voxel dimensions (isotropic).
FOV_Z_DIM = 165 # Number of pixels along beam direction.
FOV_Y_DIM = 70
FOV_X_DIM = 140

class Patient():
    """
        Class rapresenting a patient's profile.
        
        The main folder must be structered as:
            - main_folder
                - PET_measurements
                - DDS
                - other subfolders, i.e. (DICOM)
        
        Parameters
        -----------
        id_ : str
            Patient ID in the format *001P* or *001C* for proton
            or carbon treatment, respectively.
        main_folder : str
            Path to the main folder where all the objects related
            to the patient are stored (i.e. PET images, DDS mask, etc...).
        beam_dds : :class:`patient_treat.Fraction` arguments, dict
            Beam id and related dds mask that are passed to the
            class:class:`patient_treat.Fraction` for further analysis.
            They must be passed as ``beamID = dds_mask``,
            where ``dds_mask`` is a (M, N) array-like or the path to a Nifti
            image.
    """
    def __init__(self, id_, main_folder, beam_dds={}):
        """
            Constructor.
        """
        self.id_ = id_
        self.main_folder = main_folder
        self.pet_main_folder = os.path.join(self.main_folder, 'PET_measurements')
        self.beam_dds = beam_dds
        self.fraction_id, self.fraction_id_2T = self.fraction_id_init()
        self.fraction = {}
        self.fraction_2T = {}
        self.fraction_init()
    
    
    def __str__(self):
        """
            Classmethod.
        """
        info = 'Patient {}:\n\
                Beams: {}\n\
                Total number of fractions: {} + {}'\
                .format(self.id_, list(self.beam_dds.keys()),\
                        len(self.fraction_id), len(self.fraction_id_2T))
        return info
    
    def __repr__(self):
        """
            Classmethod.
        """
        warning = 'Only accessible fractions are displayed.'
        fractions_info = '\n \n'.join((self.fraction[i].__str__() for i in self.fraction))
        fractions2T_info = '\n \n'.join((self.fraction_2T[i].__str__() for i in self.fraction_2T))
        return '\n{}\n\n{}\n\n{}\n{}' .format(self.__str__(), warning, fractions_info, fractions2T_info)
    
    
    def fraction_id_init(self):
        """
            Checks which treatment's fractions have been acquired
            with PET scanner and return them as a list of string ID
            in the format ``001`` or ``001_2T`` in the case of first
            or second treatment's time.
            If folders :file:`{PET}fraction{number}` are not present,
            it searches for :file:`PET_measurements/median` folder.
            It fails if both of them are not present.
            
            Returns
            -------
            id_ : array - like
                ID representing fraction's number, i.e. ``001``
            id_2T : array - like
                |ID representing fraction's number
                |for the second treatment's time, i.e. ``001_2T``
                It's not empty if a second time has been irradiated.
            """
        try:
            id_ = glob.glob(os.path.join(self.pet_main_folder, '*fraction*'))
            p = 'fraction'
            id_ = np.asarray(sorted([i.replace(i, i[i.find(p)+len(p):]) for i in id_]))
            id_2T_mask = np.asarray([i.endswith('_2T') for i in id_])
            id_2T = id_[id_2T_mask]
            id_ = id_[np.logical_not(id_2T_mask)]
        except IndexError:
            id_ = glob.glob(os.path.join(self.pet_main_folder, 'median', '*.nii'))
            id_ = [os.path.split(i)[1] for i in id_]
            id_ = np.asarray(sorted([i.replace(i, i[:-4]) for i in id_]))
            id_2T_mask = np.asarray([i.endswith('_2T') for i in id_])
            id_2T = id_[id_2T_mask]
            id_ = id_[np.logical_not(id_2T_mask)]
        return id_, id_2T
    
    def fraction_init(self):
        """
            Initializates :attr:`fraction` and :attr:`fraction_2T`.
        """
        if len(self.fraction_id) == 0:
            pass
        else:
            self.loading(self.fraction_id, 'fraction')
        if len(self.fraction_id_2T) == 0:
            logging.info(repr(self))
            return
        else:
            self.loading(self.fraction_id_2T, 'fraction_2T')
            logging.info(repr(self))
        return

    def loading(self, id_ref, ref):
        """
            Creates :class:`Fraction()` instances and adds them to
            :attr:`Patient.fraction` or :attr:`Patient.fraction_2T` dictionary.
            Both raw input PET image and median image are added
            if they exist or can be created. In the last case,
            the median image is also save in
            :file:`{Patient.main_folder}/PET_measurents/median/` as :file:`{fraction_id}.nii`,
            i.e. :file:`{001}.nii`.
            
            Parameters
            ----------
            id_ref: array-like, str
                Array of fractions' IDs.
            ref: str
                Class attribute: :attr:`fraction` or :attr:`fraction_2T`.
                If `fraction`, :class:`Fraction()` instance is added to :attr:`Patient.fraction`
                dictionary, otherwise to :attr:`Patient.fraction_2T` dictionary.
                
        """
        logging.info('Loading fractions...')
        _ref = vars(self)[ref]
        for _id in id_ref:
            folder = '*fraction{}'.format(_id)
            img = '*{}*_iter5subset1.gipl.gz'.format(_id)
            try:
                raw_img = glob.glob(os.path.join(self.pet_main_folder,\
                                                folder,\
                                                img))[0]
            except IndexError:
                raw_img = None
            try:
                img = '{}.nii'.format(_id)
                med_img = glob.glob(os.path.join(self.pet_main_folder,\
                                                 'median',\
                                                 img))[0]
                med_img = sitk.ReadImage(med_img)
            except IndexError:
                if raw_img is not None:
                    output_dir = os.path.join(self.pet_main_folder, 'median')
                    if not os.access(output_dir, os.F_OK):
                        os.mkdir(output_dir)
                    med_img = get_median_image(raw_img, save=True,\
                                            output_dir=output_dir, output_filename=img)
                else: med_img = None
            if med_img is not None:
                _ref[_id] = Fraction(_id, raw_img_path=raw_img, med_img=med_img)
        logging.info('End.')



class Fraction():
    """
        Class rapresenting a treatment's fraction.
        
        It stores all the information for further
        analysis: acquired images, beam ID, DDS mask, etc.
        
        In addition, it allows to create and save median images
        computed with different kernel size.
        Moreover, you can compute binary masks giving
        threshold values with respect to the maximum
        intensity in the median image.
        
    """
    def __init__(self, fraction_number, beam=None, raw_img_path=None, med_img=None, dds_mask=None):
        """
            Constructor.
            
            Parameters
            ----------
            fraction_number : str
                It must be a string in the format *001*, *011*, *111*, etc.
            beam : str
                String that identifies the beam, i.e. *B1*, *B2*, etc.
            raw_img_path : str
                File path to raw image. (default = **None**)
            med_img : str or SimpleITK Image Object.
                Valid input values can be the file path to the median image or
                a SimpleITK Image Object. (default = **None**)
                If ``med_img`` is **None**, but the reference to the raw input image has been given,
                (i.e. ``raw_img_path`` is not None), the :attr:`median_image` attribute is set with a default image computed using a radius of 5mm. in the three direction.
            dds_mask : str or 2D - array.
                Valid input values can be the file path to the DDS mask saved as Nifti
                or a 2D - array.
        """
        self.fraction_number = fraction_number
        self.beam = beam
        self.__raw_img_path = raw_img_path
        self.__med_img = med_img
        self.__dds_mask = dds_mask
        self.__thr = {}
    
    def __str__(self):
        if self.beam is None: _beam = 'Unknown'
        else: _beam = self.beam
        if self.dds is None: _dds = 'Unknown'
        else: _dds = 'Given as {}'.format(type(self.dds).__name__)

        if self.raw_image is None: _raw = 'No'
        else: _raw = 'Yes'
        if self.median_image is None: _med = 'No'
        else: _med = 'Yes'
        _thr = [t for t in self.threshold_image.keys()]
        info = 'Fraction number: {}\n Beam: {}\t DoseDeliverySystem Mask: {}\
                \n Raw Image: {}\t Median Image: {}\
                \n Threshold computed: {} %'\
                .format(self.fraction_number, _beam, _dds, _raw, _med, _thr)
        return info

    @property
    def raw_image(self):
        """
            File path to original raw image.
            It's used to access to the original image
            and to perform various image processing,
            for instance Image Median Filter can be applied
            with a different kernel radius.
        """
        if self.__raw_img_path is None:
            print('No raw image has been provided.')
        else:
            return self.__raw_img_path

    @raw_image.setter
    def raw_image(self, file_path):
        self.__raw_img_path = file_path

    @raw_image.deleter
    def raw_image(self):
        del self.__raw_img_path

    @property
    def median_image(self):
        """
            SimpleITK ImageObject:
            if not provided, a median image is computed from the
            raw image specified by its file path.
        """
        if self.__med_img is None:
            self.median_image = [5, 5, 5]
        elif not isinstance(self.__med_img, sitk.SimpleITK.Image):
            self.__med_img = sitk.ReadImage(self.__med_img)
        return self.__med_img

    @median_image.setter
    def median_image(self, radius):
        """
            Returns a SimpleITK ImageObject:
            a median filter is applied to the input raw image.
            The kernel dimension is specified by the ``radiu``.
            
            Parameters
            ----------
            radius : uint or array_like uint
                Median filter kernel dimension.
        """
        
        try:
            self.__med_img = get_median_image(self.__raw_img_path, radius)
        except ValueError:
            print('No input image has been provided.')
            return
        except NotImplementedError:
            print('Invalid radius: it must be an unsigned integer or an array of unsigned integer.')
            return
    
    @property
    def dds(self):
        """
            Returns Dose Delivery Mask as a 2D - array.
        """
        if isinstance(self.__dds_mask, str):
            self.dds = self.__dds_mask
        return self.__dds_mask

    @dds.setter
    def dds(self, value):
        if isinstance(value, str):
            tmp = sitk.ReadImage(value)
            tmp = sitk.GetArrayFromImage(tmp)
            if len(tmp.shape) > 2:
                tmp = tmp[0, :, :]
            self.__dds_mask = tmp
        else:
            self.__dds_mask = value

    @property
    def threshold_image(self):
        """
            Returns volume binary mask
            corresponding to computed threshold
            as dictionary.
        """
        return self.__thr
    
    @threshold_image.setter
    def threshold_image(self, thr):
        """
            Given the median image returns a boolean mask
            such as all the voxels with value *True* have an intensity
            above the threshold.
            The threshold value is computed as a percent of the maximum
            intensity in the median image.
            If the class property 'dds' is not **None**,
            a volume restriction is applied.
            
            Parameters
            ----------
            thr : int or float
                Threshold value: i.e. 'thr' = 40 means 40%
                of the maximum intensity.
        """
        img = sitk.GetArrayFromImage(self.median_image)
        thr_value = np.amax(img) * thr * 0.01
        mask = img > thr_value
        mask = connected_regions(mask)
        mask = opening(mask)
        img = img * mask
        mask = img > thr_value
        try:
            mask = mask * self.dds
        except TypeError:
            pass
        mask = connected_regions(mask)
        self.__thr[thr] = mask



def opening(mask, er_radius=1, dil_radius=2):
    """
        Performs the opening morphological operation
        of a binary image.
        
        Parameters
        ----------
        mask : ndarray, bool
            Input binary image.
        er_radius : int
            Erosion kernel radius. Its value in specified as pixel dimension,
            i.e. `er_radius` = 1 corresponds to 1.6 mm since
            the pixel dimensions is 1.6.
            (default 1)
        dil_radius : int
            Dilation kernel radius.
            Its value in specified as pixel dimension,
            i.e. `dil_radius` = 1 corresponds to 1.6 mm
            since the pixel dimensions is 1.6.
            (default 2)
       
       Returns
       -------
       mask : ndarray, bool
            Mask after performing opening operation.
    """
    struct_er = morphology.ball(er_radius)
    struct_dil = morphology.ball(dil_radius)
    mask = scipy.ndimage.binary_erosion(mask, struct_er)
    mask = scipy.ndimage.binary_dilation(mask, struct_dil)
    return mask

def connected_regions(mask):
    """
        Finds the most connected region
        of a binary mask.
        
        Parameters
        ----------
        mask : ndarray, bool
            Binary mask.
        
        Returns
        -------
        mask : ndarray, bool
            Binary mask where the most connected region
            is set to **True**.
    """
    label, _ = scipy.ndimage.measurements.label(mask)
    m = np.argmax(np.delete(np.bincount(label.flat), 0)) + 1
    mask = label == m
    return mask

def get_median_image(input_, radius=[5, 5, 5],
                     save=False, output_dir=None,
                     output_filename=None
                     ):
    """
        Applies a 3D median filter to the original input image
        with the specified ``radius`` in mm.
        Returns the final image as SimpleITK ImageObject.
        
        Parameters
        ----------
        input : str
            Original image file path.
        radius: uint,  array_like
            Radius of the filter's kernel
            (default 5 mm).
        save : bool
            Option to save final image.
            Default value is **False**.
        output_dir : str, optional
            Path to the ouput directory where the output image is saved.
            If ``output_dir`` is not given, the final image is saved in
            the input file directory .
        output_filename : str, optional
            Output image name with extension, i.e. `medianImage.nii`.
            If the output file name and extension are not specified,
            the output file name will be 'originalName_median' and
            the extension will be the same as the original
            (not compressed).
            
        Returns
        -------
        img_med : SimpleITK ImageObject
            The output median image is stored in a SimpleITK ImageObject.
    """
    
    median = sitk.MedianImageFilter()
    median.SetRadius(radius)
    img = sitk.ReadImage(input_)
    img_med = median.Execute(img)
    if save:
        if ((output_dir is not None) and (os.path.isdir(output_dir) == False)) or output_dir is None:
            [output_dir, input_name] = os.path.split(input_)
            print(' Output directory not provided or doesn\'t exit. The ouput image will be saved in the input image directory:             {}.'.format(output_dir))
        if output_filename == None:
            output_filename = os.path.splitext(os.path.splitext(input_name)[0])[0] +'_median' +  os.path.splitext(os.path.splitext(input_name)[0])[1]
        output = os.path.join(output_dir, output_filename)
        sitk.WriteImage(img_med, output)
    return img_med


def plot_profile(imgs, x_center=70, y_center=35, radius=0):
    """
        Computes intensity profiles along beam direction for each
        given images:
        the position in the axial plane is specified by
        ``x_center`` and ``y_center``.
        The ``radius`` is specified as number of pixels along the axial dimensions: it defines the neighborhood size.
        The returned profiles are normalized with respect to
        the maximum intensity in the image.
        
        Parameters
        ----------
        imgs : list or 3D - array
            List of input images or unique image as array.
        x_center : uint
            Pixel identification in the axial plane, along x-axis:
            x_center can be varied in the range [0, 139].
        y_center : uint
            Pixel identification in the axial plane, along y-axis:
            y_center can be varied in the range [0, 70].
        radius : uint
            The mean profile is computed from (2*radius) * (2*radius) - 1
            pixels around the central pixel [y_center, x_center].
            i.e., if ``radius = 0`` (default) no mean is computed,
            otherwise if ``radius = 2``, the mean profile is computed
            taking into account for the eight pixels around the central one.
            
        Returns
        -------
        zz : 1D - array
            Position along beam direction in mm.
        p : list of 1D - array or 1D - array
            List of intensity profiles computed
            for each image.
    """
    
    zz = VOXEL_DIM * np.arange(FOV_Z_DIM)
    x1 = x_center - radius
    x2 = x_center + radius + 1
    y1 = y_center - radius
    y2 = y_center + radius + 1
    if x1 < 0: x1 = x_center
    if x2 > (FOV_X_DIM - 1): x2 = x_center
    if y1 < 0: y1 = y_center
    if y2 > (FOV_Y_DIM - 1): y2 = y_center
    try:
        p = []
        for  img in imgs:
            img = img / np.amax(img)
            p.append(np.mean(img[:, y1:y2, x1:x2], axis=(1, 2)))
        p = np.transpose(p)
    except IndexError:
        imgs = imgs / np.amax(imgs)
        p = np.mean(imgs[:, y1:y2, x1:x2], axis=(1, 2))
    return zz, p




if __name__ == '__main__':
    pass

