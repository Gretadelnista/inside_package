import unittest
import os
import SimpleITK as sitk
import numpy as np
from inside_analysis import dds_processing

TEST_FOLDER = os.path.dirname(os.path.realpath(__file__))

class TestFunction(unittest.TestCase):
    """
        Unit test for dds_processing module.
    """
    input_dir = os.path.join(TEST_FOLDER, 'test_data/dds_test')
    input_file = os.path.join(input_dir, 'Treatment_X.txt')
    check_file = os.path.join(input_dir, 'coordinate.txt')
    ref_mask_img = sitk.ReadImage(
                                  os.path.join(input_dir,
                                               'check_mask.nii')
                                  )
    
    def test_decoding(self):
        output_file = os.path.join(self.input_dir, 'TestDecoding.txt')
        x, y = dds_processing.decoding(self.input_file, output_file)
        x_check, y_check = np.loadtxt(self.check_file, unpack=True)
        self.assertListEqual(list(x), list(x_check))
        self.assertListEqual(list(y), list(y_check))
        x_write, y_write = np.loadtxt(output_file, unpack=True)
        self.assertListEqual(list(x), list(x_write))
        self.assertListEqual(list(y), list(y_write))
        os.remove(output_file)

    def test_DDS_mask(self):
        x, y = np.loadtxt(self.check_file, unpack=True)
        ref_mask = sitk.GetArrayFromImage(self.ref_mask_img)
        mask = dds_processing.DDS_mask(x, y)
        self.assertTrue((ref_mask == mask).all())

    def test_mask_to_image(self):
        x, y = np.loadtxt(self.check_file, unpack=True)
        mask = dds_processing.DDS_mask(x, y)
        dds_processing.mask_to_image(mask, os.path.join(TEST_FOLDER, 'Test.nii'))
        ref_mask = sitk.GetArrayFromImage(self.ref_mask_img)
        mask_test = sitk.GetArrayFromImage(
                                           sitk.ReadImage(os.path.join(TEST_FOLDER, 'Test.nii'))
                                           )
        self.assertListEqual(list(ref_mask.flatten()), list(mask_test.flatten()))
        os.remove(os.path.join(TEST_FOLDER, 'Test.nii'))


if __name__ == '__main__':
    unittest.main()
