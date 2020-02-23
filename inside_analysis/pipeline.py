import os, sys, glob
import image_analysis
import patient_treat
import dds_processing
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import SimpleITK as sitk

def threshold(img_path, thr, dds_mask=None):
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    thr_value = np.amax(img) * thr * 0.01
    mask = img > thr_value
    mask = patient_treat.connected_regions(mask)
    mask = patient_treat.opening(mask)
    img = img * mask
    mask = img > thr_value
    try:
        mask = mask * dds_mask
    except TypeError:
        pass
    mask = patient_treat.connected_regions(mask)
    return mask


if __name__=='__main__':
    images_folder = sys.argv[1]
    dds_mask = sys.argv[2]
    dds_mask = sitk.GetArrayFromImage(sitk.ReadImage(dds_mask))[0, :, :]
    images = glob.glob(os.path.join(images_folder, '*.nii'))
    images = sorted(images, key=lambda x: os.path.split(x)[1])
    img_ = [sitk.ReadImage(img) for img in images]
    img_array = [sitk.GetArrayFromImage(img) for img in img_]
    name_img_ = [os.path.splitext(os.path.split(img)[1])[0] for img in images]
    
    ### plot profili
    plt.figure()
    color=cm.rainbow(np.linspace(0, 1, len(img_)))
    for n, img in enumerate(img_array):
        yy = img[:, 17, 35]/img[:, 17, 35].max()
        xx = np.arange(len(yy))*3.2
        plt.plot(xx, yy,
                 c=color[n], alpha = .8,
                 drawstyle='steps', label=name_img_[n])
    plt.xlabel('[mm]')
    plt.legend()

    ### SHIFT ANALYSIS
    '''
    for img in images:
        name_img_0 = os.path.splitext(os.path.split(images[0])[1])[0]
        name_img_1 = os.path.splitext(os.path.split(img)[1])[0]
        name_map = '{}vs{}.nii'.format(name_img_0, name_img_1)
        output_name = os.path.join(sys.argv[3], name_map)
        map = (image_analysis.shift_method(images[0], img, voxel_dim=3.2, mask=dds_mask)).filled()
        dds_processing.mask_to_image(map, output_name,
                                     input_header='./PET_header_reb2x2x2.bin',
                                     input_mask_reference='./DDS_mask_reb2x2_reference.nii')
    '''
    ### BEV ANALYSIS
    '''
    for img in images[1:]:
        name_img_0 = os.path.splitext(os.path.split(images[0])[1])[0]
        name_img_1 = os.path.splitext(os.path.split(img)[1])[0]
        name_map = '{}vs{}.nii'.format(name_img_0, name_img_1)
        output_name = os.path.join(sys.argv[3], name_map)
        mask1 = threshold(images[0], thr=10, dds_mask=dds_mask)
        mask1[:30, :, :]=0 # elimino Range shift
        mask2 = threshold(img, thr=10, dds_mask=dds_mask)
        mask2[:30, :, :]=0 # elimino Range shift
        map = image_analysis.BEV(mask1, mask2, voxel_dim=3.2)
        dds_processing.mask_to_image(map, output_name,
                                     input_header='./PET_header_reb2x2x2.bin',
                                     input_mask_reference='./DDS_mask_reb2x2_reference.nii')
    
    
    '''
    plt.show()
