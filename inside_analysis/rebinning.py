import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def rebinning(input_img, size=2):
    filter = sitk.BinShrinkImageFilter()
    filter.SetShrinkFactors((size, size, size))
    output_img = filter.Execute(input_img)
    
    '''
    plt.figure()
    plt.subplot(1, 2, 1)
    array_input = sitk.GetArrayFromImage(input_img)
    plt.imshow(array_input[:, :, int(array_input.shape[2]/2)], vmax=900)
    plt.subplot(1, 2, 2)
    array_output = sitk.GetArrayFromImage(output_img)
    plt.imshow(array_output[:, :, int(array_output.shape[2]/2)])
    '''
    return output_img

def dds_rebinning(input_file):
    from inside_analysis import dds_processing as dds
    x, y = dds.decoding(input_file)
    mask = dds.DDS_mask(x, y, voxel_dim = 3.2)
    dds.mask_to_image(mask, '_DDS_mask_reb2x2.nii',
                     input_header='PET_header_reb2x2x2.bin',
                     input_mask_reference='DDS_mask_reb2x2_reference.nii')
    return mask


def median(input_image, size=2):
    filter = sitk.MedianImageFilter()
    filter.SetRadius(size)
    output_img = filter.Execute(input_image)
    return output_img

def process(input_file):
    input_img = sitk.ReadImage(input_file)
    output_img = rebinning(input_img)
    median_img = median(output_img)
    med_array = sitk.GetArrayFromImage(median_img)
    return med_array, median_img



if __name__ == '__main__':
    import glob
    import os, sys
    from matplotlib.pyplot import cm
    '''
    list_input = glob.glob('/Users/gretadelnista/Desktop/INSIDE/Data/Trial/006P/rebinning/*')
    list_input = [(os.path.split(input)[1][0:3], input) for input in list_input]
    list_input.sort(key=lambda x: x[0])
    plt.figure()
    color=cm.rainbow(np.linspace(0, 1, len(list_input)))
    
    for n, input in enumerate(list_input):
        img = sitk.ReadImage(input[1])
        array = sitk.GetArrayFromImage(img)
        yy = array[:, 17, 35]/array[:, 17, 35].max()
        xx = np.arange(len(yy))*3.2
        plt.plot(xx, yy,
                 c=color[n], alpha = .8,
                 drawstyle='steps', label=input[0])
    plt.xlabel('[mm]')
    plt.legend()
    '''
    list_input = glob.glob('/Users/gretadelnista/paziente_test/PETfraction*/*_iter5subset1.gipl.gz')
    for input in list_input:
        name = os.path.split(input)[1][:3]
        img = sitk.ReadImage(input)
        reb_img = rebinning(img)
        med_img = median(reb_img)
        sitk.WriteImage(reb_img, '/Users/gretadelnista/paziente_test/rebinning/raw/' + name + '.nii' )
        sitk.WriteImage(med_img, '/Users/gretadelnista/paziente_test/rebinning/median_global/' + name + '.nii' )
    plt.show()

