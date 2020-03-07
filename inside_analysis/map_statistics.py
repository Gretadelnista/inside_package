import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import SimpleITK as sitk
from scipy.stats import pearsonr
import pandas

def map_analysis(map_folder, dds_path):
    dds_mask = sitk.GetArrayFromImage(sitk.ReadImage(dds_path))[0, :, :]
    maps = sorted(os.listdir(map_folder))
    maps = [map for map in maps if map.endswith('.nii')]
    maps = [map for map in maps if not map.endswith('CT.nii')]
    with open(os.path.join(map_folder, 'Result.txt'), 'a') as result_file:
        result_file.write('Fraction \t mean [mm] \t FWHM[mm] \n')
    plt.figure()
    color=cm.rainbow(np.linspace(0, 1, len(maps)))
    bins = np.arange(-16, 18, 2)
    for n, map in enumerate(maps):
        try:
            map_np = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(map_folder,map)))[0, :, :]
            name = os.path.splitext(map)[0]
            map_np_valid = np.extract(dds_mask>0, map_np)
            plt.hist(map_np_valid, bins=bins,
                    histtype='step', color=color[n], density=True, label=name)
            with open(os.path.join(map_folder, 'Result.txt'), 'a') as result_file:
                result_file.write('{}\t{:.2f}\t{:.2f}\n'.format(name, map_np_valid.mean(), 2.35*map_np_valid.std()))
        except:
            pass
    plt.xlabel('[mm]')
    plt.title('Range difference')
    plt.legend()
    plt.show()

def image_correlation(image_folder, dds_path):
    dds_mask = sitk.GetArrayFromImage(sitk.ReadImage(dds_path))
    images = sorted(os.listdir(image_folder))
    images = [img for img in images if img.endswith('.nii')]
    img_init_np = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_folder, images[0])))
    img_init_np = img_init_np
    for img in images[:]:
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_folder, img)))
        img_np = img_np
        print(pearsonr(np.extract(dds_mask>0, img_init_np), np.extract(dds_mask>0, img_np)))

def map_correlation(map_folder, dds_path):
    dds_mask = sitk.GetArrayFromImage(sitk.ReadImage(dds_path))[0, :, :]
    maps = sorted(os.listdir(map_folder))
    maps = [map for map in maps if map.endswith('.nii')]
    names = [os.path.splitext(map)[0] for map in maps]
    map_np = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(map_folder,map)))[0, :, :]
              for map in maps]
    data = np.zeros((len(maps), len(maps)))
    for i in range(len(maps)):
        with open(os.path.join(map_folder, 'map_correlation.txt'), 'a') as corr:
            corr.write('\n{}\n'.format(names[i]))
        for j, map in enumerate(map_np):
            PCC = pearsonr(
                            np.extract(dds_mask>0, map_np[i]),
                            np.extract(dds_mask>0, map)
                            )[0]
            data[j, i] = PCC

    np.savetxt(os.path.join(map_folder, 'map_correlation.txt'), data,
               fmt = '%.2f', delimiter='\t', header='PCC maps')
    plt.figure('PCC')
    plt.imshow(data, vmin=0, vmax=+1)

    xx = np.arange(len(names)+1)
    xx = -1 + (xx[1:]+xx[:-1])*.5
    plt.xticks(xx, names, rotation=45)
    plt.yticks([])
    plt.colorbar()
    plt.show()



if __name__=='__main__':
    input_folder = sys.argv[1]
    dds_path = sys.argv[2]
#image_correlation(input_folder, dds_path)
    map_analysis(sys.argv[1], sys.argv[2])
#map_correlation(sys.argv[1], sys.argv[2])
