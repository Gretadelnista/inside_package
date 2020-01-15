import unittest
import os
import SimpleITK as sitk
import numpy as np
from skimage import morphology
from inside_analysis import patient_treat
from inside_analysis.patient_treat import Patient, Fraction

TEST_FOLDER = os.path.dirname(os.path.realpath(__file__))


class TestPatientClass(unittest.TestCase):
    """
        Unit test for class Patient in
        patient_treat module.
    """
    def test_setUp(self):
        '''
            Checking requested arguments.
        '''
        self.assertRaises(TypeError, Patient)
        self.assertRaises(TypeError, Patient, '001P')
    
    def setUp(self):
        self.patient = Patient('001P', os.path.join(TEST_FOLDER, 'test_data/patient_test'))
        self._beam = 'B2'
        self._dds = os.path.join(TEST_FOLDER, 'test_data/patient_test/DDS_mask.nii')
    
    def test_init(self):
        '''
            Testing constructor method.
        '''
        ID = ['013', '016', '017', '022', '023', '024', '025', '027']
        _lenID = len(self.patient.fraction_id)
        _lenID_2T = len(self.patient.fraction_id_2T)
        self.assertEqual(_lenID, len(ID))
        self.assertCountEqual(self.patient.fraction_id, ID)
        self.assertEqual(_lenID_2T, 0)
        beam_list = list(self.patient.beam_dds.keys())
        self.assertEqual(len(beam_list), 0)
        self.patient.beam_dds[self._beam] = self._dds
        beam_list = list(self.patient.beam_dds.keys())
        self.assertEqual(len(beam_list), 1)
        self.assertNotEqual(list(self.patient.fraction.keys()),
                                 ID)
        ID.remove('013')
        self.assertListEqual(list(self.patient.fraction.keys()),
                                                      ID)
        for i in list(self.patient.fraction.values()):
            self.assertIsInstance(i, Fraction)

    def test_exception(self):
        '''
            Testing exception.
        '''
        def get_first_element(list):
            return list[0]
        self.assertRaises(IndexError, get_first_element, [])
        def dot_operation(data, mask):
            return data * mask
        self.assertRaises(TypeError, dot_operation,\
                          np.ones((10, 10, 10)), None)


class TestFractionClass(unittest.TestCase):
    """
        Unit test for class Fraction in
        patient_treat module.
    """
    def test_setUp(self):
        self.assertRaises(TypeError, Fraction)
    
    def setUp(self):
        self._id = '016'
        self._beam = 'B2'
        self._dds = os.path.join(TEST_FOLDER, 'test_data/patient_test/DDS_mask.nii')
        self.fraction = Fraction(self._id)

    def test_init(self):
        self.assertIsNone(self.fraction.beam)
        self.assertIsNone(self.fraction.raw_image)
        self.assertIsNone(self.fraction.median_image)
        self.fraction.beam = self._beam
        self.fraction.dds = self._dds
        self.assertCountEqual(
                              [patient_treat.FOV_Y_DIM,
                               patient_treat.FOV_X_DIM],
                              self.fraction.dds.shape
                              )
                              
class TestFunction(unittest.TestCase):
    """
        Unit test for functions in
        patient_treat module.
    """
    
    def test_get_median_image(self):
        """
            Testing get_median_image's
            functionality.
        """
        raw = os.path.join(TEST_FOLDER, 'test_data/patient_test/PET_measurements/fraction016/016_iter5subset1.gipl.gz')
        res = patient_treat.get_median_image(raw)
        self.assertIsInstance(res, sitk.SimpleITK.Image)
        res = patient_treat.get_median_image(raw, save=True)
        res_file = os.path.join(TEST_FOLDER, 'test_data/patient_test/PET_measurements/fraction016/016_iter5subset1_median.gipl')
        _exist = os.path.exists(res_file)
        self.assertTrue(_exist)
        os.remove(res_file)
        res = patient_treat.get_median_image(raw, save=True, output_filename='test.nii')
        res_file = os.path.join(TEST_FOLDER, 'test_data/patient_test/PET_measurements/fraction016/test.nii')
        _exist = os.path.exists(res_file)
        self.assertTrue(_exist)
        os.remove(res_file)
    
    def test_connected_regions(self):
        """
            Testing connected region's extraction.
        """
        #   base cube (reference mask)
        cube = np.zeros((10, 10, 10))
        cube[0:3, 0:3, 0:3] = 1
        #   sample with more non-null elements than cube
        data = cube.copy()
        data[1, 2:4, 5:6] = 1
        res = patient_treat.connected_regions(data)
        self.assertTrue((res == cube).all())
        data[7, 0:1, 9:] = 1
        res = patient_treat.connected_regions(data)
        self.assertTrue((res == cube).all())
        #   sample with a cube greater than base cube
        data = cube.copy()
        data[5:, 5:, 5:] = 1
        res = patient_treat.connected_regions(data)
        cube_added = data - cube
        self.assertGreater(cube_added.sum(), cube.sum())
        res = patient_treat.connected_regions(data)
        self.assertTrue((res == cube_added).all())
    
    def test_opening(self):
        """
            Testing opening function.
        """
        cube = np.zeros((10, 10, 10))
        cube[0:3, 0:3, 0:3] = 1
        res = patient_treat.opening(cube, er_radius=1, dil_radius=2)
        data = cube.copy()
        data[5:6, 6, 6] = 1
        res_2 = patient_treat.opening(data, er_radius=1, dil_radius=2)
        self.assertTrue(res.sum() == res_2.sum())
        #   checking ideal behaviour on a sphere
        ball = morphology.ball(5)
        res = patient_treat.opening(ball, 1, 1)
        self.assertTrue(res.sum(), ball.sum())
        res = patient_treat.opening(ball, 3, 1)
        self.assertTrue(res.sum() < ball.sum())
        res = patient_treat.opening(ball, 2, 4)
        self.assertTrue(res.sum() > ball.sum())

    def test_threshold_image(self):
        """
            Testing thresholding function,
            simulating the same operations
            on two different cubes.
            One has dimensions (21, 21, 21)
            and it's built by stacked layers
            in the range
            [100, 90,..., 10, 0, 10, ..., 90, 100].
            The other is made by cube with decreasing
            intensity from the center towards the outside.
            
        """
        cube_1 = np.ones((21, 21, 21))
        tmp = np.arange(0, 110, 10)
        tmp = np.concatenate((np.flip(tmp), tmp[1:]))
        cube_1 = (np.add(cube_1, tmp)) - 1
        cube_2 = np.zeros((21, 21, 21)) - 10
        for i in range(21):
            cube_2[i:21-i, i:21-i, i:21-i] += 10
        self.assertEqual(cube_1.max(), cube_2.max())
        thr = np.arange(10, 100, 1)*cube_1.max()/100
        for thr_value in thr:
            #   testing on cube_1
            mask = cube_1 > thr_value
            mask = patient_treat.connected_regions(mask)
            mask = patient_treat.opening(mask)
            img = cube_1 * mask
            mask = cube_1 > thr_value
            mask = patient_treat.connected_regions(mask)
            #   'connected_regions' selects only an half
            #   of the total number of voxels with a value
            #   above 'thr_value'
            n_layer = 10 - (thr_value - thr_value%10)/10
            sum_check = 21*21*n_layer
            self.assertTrue(mask.sum() == sum_check)
            #   testing on cube_2
            mask = cube_2 > thr_value
            mask = patient_treat.connected_regions(mask)
            mask = patient_treat.opening(mask)
            img = cube_2 * mask
            mask = cube_2 > thr_value
            mask = patient_treat.connected_regions(mask)
            #   in this case, 'connected_regions' really
            #   selects the total number of voxels with
            #   a value above 'thr_value', being the region
            #   connected by construction.
            sum_check = (cube_2 > thr_value).sum()
            self.assertTrue(mask.sum() == sum_check)

if __name__ == '__main__':
    unittest.main()
