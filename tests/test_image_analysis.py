import unittest
import os
import SimpleITK as sitk
import numpy as np
from skimage import morphology
from inside_analysis import image_analysis
import matplotlib.pyplot as plt


TEST_FOLDER = os.path.dirname(os.path.realpath(__file__))


class TestFunction(unittest.TestCase):
    """
        Unit test for functions in
        image_analysis module.
    """
    s = np.zeros((10, 10, 10))
    s[3:7, 3:7, 2] = s[3:7, 3:7, 7] = 1
    s[3:7, 2, 3:7] = s[3:7, 7, 3:7] = 1
    s[2, 3:7, 3:7] = s[7, 3:7, 3:7] = 1
    bin_surface = s
    
    def test_binary_surface(self):
        """
            Checking binary surface extraction.
        """
        #   data
        cube = np.zeros((10, 10, 10))
        cube[3:7, 3:7, 3:7] = 1
        #   sample surface
        surface = image_analysis.binary_surface(cube)
        self.assertTrue((surface == self.bin_surface).all())


    def test_range_map(self):
        """
            Checking range map computing.
            The check is performed comparing the range of
            a known cube with the range of a rectangular.
        """
        rect = np.zeros((10, 10, 10))
        rect[3:8, 3:7, 3:7] = 1
        surface = image_analysis.binary_surface(rect)
        range_map_1 = image_analysis.range_map(self.bin_surface)
        self.assertTrue(range_map_1.max() == 5)
        range_map_2 = image_analysis.range_map(surface)
        self.assertTrue((range_map_2 >= range_map_1).all())
        check_diff = range_map_2 - range_map_1
        self.assertTrue((check_diff[range_map_2 != range_map_1] == 1).all())

    def test_end_point(self):
        """
            Checking end_point function on the known
            binary surface, 'self.bin_surface'.
        """
        map = image_analysis.end_point(self.bin_surface)
        self.assertTrue(map.max() == 7)
        self.assertTrue((map[map!=0]).min() == 6)

    def test_BEV(self):
        """
            Testing BEV function.
        """
        #   three cubes with different range as
        #   (mask1, mask2)
        #   or (x, y) extension as (mask2, mask3)
        mask1 = np.zeros((20, 10, 10))
        mask2 = np.zeros((20, 10, 10))
        mask3 = mask1.copy()
        mask1[3:8, 2:8, 2:8] = 1
        mask2[3:15, 2:8, 2:8] = 1
        mask3[3:15, 4:8, 2:8] = 1
        range_diff = image_analysis.BEV(mask1, mask2)
        self.assertTrue(abs(range_diff).max() == 7 )
        range_diff_3 = image_analysis.BEV(mask1, mask3)
        #   the region of common points in the (x, y) plane
        #   must be different
        self.assertFalse((range_diff.mask == range_diff_3.mask).all())
        self.assertGreater(range_diff_3.mask.sum(), range_diff.mask.sum())
        self.assertEqual(abs(range_diff_3).max(), abs(range_diff).max())
        #   mask1 is made of two spheres with different radius
        #   one centered in (z1, y1, x1) and the other one centered
        #   in (z2, y1, x1) with z2 > z1
        #   mask2 has only the sphere centered in (z1, y1, x1)
        mask1 = np.zeros((30, 30, 30))
        mask2 = np.zeros_like(mask1)
        mask1[5:20,5:20, 5:20] += morphology.ball(7)
        mask1[15:26,7:18, 7:18] += morphology.ball(5)
        mask1[mask1 != 0] = 1
        mask2[5:20, 5:20, 5:20] += morphology.ball(7)
        range_map = image_analysis.BEV(mask1, mask2)
        self.assertTrue(range_map.max() == 6)

    def test_MP(self):
        """
            Testing MP function.
        """
        #   defining a useful function
        #   for getting 'sample image'
        #   to compare
        def f(x, y, z, a=0):
            if z<3:
                return .1*z
            else:
                return np.exp(-x**2-y**2) + .1*(a + 19-z)
        #   defining firt image array
        rect = np.reshape([f(x, y, z) for x in range(-5, 6) for y in range(-5, 6) for z in range(0, 20)], (11, 11, 20))
        rect = rect.transpose(2, 1, 0)
        #   defining second image array:
        #   its intensity reaches zero after
        #   than rect's intensity (except for z=0)
        rect_2 = np.reshape([f(x, y, z, a=5) for x in range(-5, 6) for y in range(-5, 6) for z in range(0, 20)], (11, 11, 20))
        rect_2 = rect_2.transpose(2, 1, 0)
        #   MP function requires SimpleITK image
        rect = sitk.GetImageFromArray(rect)
        rect_2 = sitk.GetImageFromArray(rect_2)
        map = image_analysis.MP(rect, rect_2)
        self.assertTrue(map.mean()<0)
        #   in this case, rect_20s intensity
        #   reaches zero before than rect's intensity
        rect_2 = np.reshape([f(x, y, z, a=-2) for x in range(-5, 6) for y in range(-5, 6) for z in range(0, 20)], (11, 11, 20))
        rect_2 = rect_2.transpose(2, 1, 0)/rect_2.max()
        rect_2 = sitk.GetImageFromArray(rect_2)
        map = image_analysis.MP(rect, rect_2)
        self.assertTrue(map.mean()>0)
        
        #   defining a useful function
        def gauss(x, y):
            return np.exp(-(x**2+y**2)/(2*36))
        
        A = [gauss(x, y) for x in range(-15, 16) for y in range(-15, 16)]
        A = np.reshape(A, (31, 31))
        #   building two different images:
        #   one is made by two intersecting spheres
        #   with different radius ( 7 and 5)
        #   while the other is made only by the sphere
        #   with radius 7.
        data1 = np.zeros((50, 31, 31))
        data2 = np.zeros_like(data1)
        data1[5:20,5:20, 5:20] += morphology.ball(7)
        data1[15:26,7:18, 7:18] += morphology.ball(5)
        data1[data1 != 0] = 1
        data2[5:20, 5:20, 5:20] += morphology.ball(7)
        #   data1 and data2 are multiplied with A
        #   in order to obtain images with varying
        #   intensity.
        data1 = data1.astype(int)*A
        data2 = data2.astype(int)*A
        map = image_analysis.MP(sitk.GetImageFromArray(data1), sitk.GetImageFromArray(data2))
        #   At the centre of the map the MP
        #   value must be as the maximum.
        self.assertLessEqual(map.max(), 6)
        self.assertGreaterEqual(map.min(), 0)
        self.assertTrue(map[12, 12] == map.max())
        #   testing on two cubes made by cubes
        #   one inside an other with incresing intensity
        #   towards the center: along axis 0
        #   cube_2 presents an asimmetry.
        cube_1 = np.zeros((21, 21, 21)) - 10
        cube_2 = np.zeros((21, 21, 21)) - 10
        for i in range(21):
            j = i+2
            cube_1[i:21-i, i:21-i, i:21-i] += 10
            cube_2[i:21-j, i:21-i, i:21-i] += 10
        map = image_analysis.MP(sitk.GetImageFromArray(cube_1), sitk.GetImageFromArray(cube_2))
        #   by construction, the difference must be
        #   equal to 2
        self.assertTrue((map.compressed()==2).all())
        #   testing that MP works indipendently
        #   from image's intensity.
        cube_2 = cube_2 * .6
        map = image_analysis.MP(sitk.GetImageFromArray(cube_1), sitk.GetImageFromArray(cube_2))
        self.assertTrue((map.compressed()==2).all())


if __name__ == '__main__':
    unittest.main()
