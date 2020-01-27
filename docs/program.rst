Programs
========

dds_processing
---------------

Processing the Dose Delivery System output file, the binary mask of the irradiated points is computed: note that only the points inside the PET system's field of view are set to **True**, therefore the mask could be cut.
FOV dimensions are specified as :func:`DDS_mask` arguments, so if necessary you can change them by :program:`dds_processing.py` source code: at the moment, they're set according to the actual INSIDE PET system's FOV dimension.
In addition, the shape of the returned mask is set according to the shape of PET images, which is established by the algorithm of reconstruction: 165 x 70 x 140 voxels.
In any case,  you can change it from :mod:`patient_treat` source code, where they're specified as global variable (``FOV_Z_DIM``,  ``FOV_Y_DIM``  and  ``FOV_X_DIM``,  respectively).

Moreover, you can also save the original coordinated in mm of the irradiated points as Text File specifying the output file path.

See also :mod:`dds_processing`.

.. autoprogram:: dds_processing:parser
   :prog: dds_processing.py
   :groups:

main_process
------------

This program allows you to create a :class:`patient_treat.Patient` instance providing
patient's ID and main folder, which must contain *PET_measurements* subfolder.
If not already present, median images are computed with a kernel set by default and saved in *. ./PET_measurements/median* subfolder in order to make the following analysis faster.
For further  details, see :mod:`patient_treat` documentation.
Beam's ID and Dose Delivery Mask's specification are required so that each fraction can be handled in the correct way.

By default, a user-interface is opened and allows you to select the beam of interested, if more than one is given, and to plot the profiles of all the fractions belonging to it.
You can set the point of interest in the axial plane and the neighborhood that you want to consider.
Additional informations are given in :mod:`main_process`.

You can perform both *Middle Point* (MP) and *Beam Eye's View* (BEV) analyses: 
the resulting maps are saved in the main folder. 
In the case of BEV analysis, you must specified the thresholds that you want to use to extract the binary surface from the median images. The program will ask you for that.
In addition, you can perform *Shift Method* analysis: for each point in the axial plane, the profile of the compared image is shifted with respect to the profile of the first image in order to minimize the mean squared difference between them.
See also :mod:`image_analysis` for details about the methods' implementation.

Note that if more than one time is acquired, i.e. T1 and T2, the program has to be run at least twice with the option ``--T2`` when you want to analyse the second time.

.. autoprogram:: main_process:parser
   :prog: main_process.py
