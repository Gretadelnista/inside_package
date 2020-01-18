import os, sys
import argparse
import patient_treat, dds_processing
import image_analysis
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import SimpleITK as sitk

_DESCRIPTION = 'Treatment analysis.'

def fraction_specification(T2):
    """
        Return the rigth attribute
        relative to fraction for Patient class
        instance.
        
        Parameters
        ----------
        T2 : bool
            It specifies if the observed fractions
            belong to the second treatment time.
            **True** for T2 fractions.
            
        Returns
        -------
        attribute : str
            *fraction* if T2 is **False**,
            *fraction_2T* if T2 is **True**.
        attribute_id : str
            *fraction_id* if T2 is **False**,
            *fraction_id_2T* if T2 is **True**.
    """
    if T2 is False:
        attribute = 'fraction'
        attribute_id = 'fraction_id'
    else:
        attribute = 'fraction_2T'
        attribute_id = 'fraction_id_2T'
    return attribute, attribute_id

def fraction_set(patient_dict, fraction_attribute, beam_list, mask_list):
    """
        Sets *beam* and *mask*
        for patient's fractions:
        if more than one beam ID are provided
        by the user, it asks which fractions
        belong to each beam.
        
        Parameters
        ----------
        patient_dict : dict
            Dictionary of :class:`patient_treat.Patient` instance.
        fraction_attribute : str
            **fraction** or **fraction_2T**,
            as returned by :func:`fraction_specification`.
        beam_list : list
            List of beams' ID.
        mask_list : list
            List of DDS masks: it must be
            in the same order as *beam_list*.
            
        
    """
    f = fraction_attribute
    if len(beam_list) == 1:
        for i in patient_dict[f]:
            patient_dict[f][i].beam = beam_list[0]
            patient_dict[f][i].dds = mask_list[0]
    else:
        beam_fraction_id = {}
        beam_mask = {beam: mask for beam, mask in zip(beam_list, mask_list)}
        fraction_id_beam = {}
        for i in beam_list:
            beam_fraction_id[i] = input("Insert fraction number_ID for beam {},then press 'Return': \n ".format(i)).split()
            values = beam_fraction_id[i]
            for value in values:
                fraction_id_beam[value] = i

        for n in patient_dict[f]:
            id_ = patient_dict[f][n].fraction_number
            patient_dict[f][n].beam = fraction_id_beam[id_]
            patient_dict[f][n].dds = beam_mask[patient_dict[f][n].beam]
    return

def plotting_profile(patient_dict, fraction_attribute, x_center, y_center, radius, beam):
    """
        Selecting images belonging to a specific time ( T2 or not)
        and to a specific beam ('B1', 'B2', etc.),
        returns a list of computed profiles for the selected point
        *(x_center, y_center)*, considering a neighborhood set with *radius*.
        
        Parameters
        ----------
        patient_dict: dict
            Dictionary of :class:`patient_treat.Patient` instance.
        fraction_attribute: str
            **fraction** or **fraction_2T** as returned by
            :func:`fraction_specification`.
        x_center: int
            x-coordinate of the point of interest.
        y_center: int
            y-coordinate of the point of interest.
        radius: int
            Number of pixel to take into account around
            point specified by *(x_center, y_center)*.
        beam: str
            Beam ID.
        
        Returns
        -------
        zz: 1D-array
            Position along beam direction in mm.
            See also :func:`patient_treat.plot_profile`.
        P: list of 1D-array
            List of intensity profiles computed for each
            image that matches both *fraction_attribute* and
            *beam* parameters.
            See also :func:`patient_treat.plot_profile`.
        labels: list
            List of label in the same order as P.
    """
    f = fraction_attribute
    imgs = []
    labels = []
    for n in patient_dict[f]:
        if patient_dict[f][n].beam == beam:
            imgs.append(sitk.GetArrayFromImage(patient_dict[f][n].median_image))
            labels.append(patient_dict[f][n].fraction_number)
    zz, P = patient_treat.plot_profile(imgs, x_center=x_center, y_center=y_center, radius=radius)
    return zz, P, labels

def update(val):
    """
        Updating callback for interactive
        mode profile plot.
    """
    global patient_
    global f
    x = int(s_x.val)
    y = int(s_y.val)
    r = int(s_r.val)
    b = radio.value_selected
    zz, P, labels = plotting_profile(vars(patient_), f, x_center=x, y_center=y, radius=r, beam=b)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels))]
    ax.legend(labels)
    ax.set_title('Profile along beam direction. Neighborhood radius = {} pixels.'.format(r))
    try:
        for n, line in enumerate(ax.lines):
            line.set_ydata(P[:, n])
            line.set_color(colors[n])
    except IndexError:
        ax.lines.set_ydata(P)
        ax.lines.set_color(colors[0])
    fig.canvas.draw_idle()

def reset(event):
    """
        Callback function for interactive
        mode profile plot: reset to initial
        value.
    """
    s_x.reset()
    s_y.reset()
    s_r.reset()


def update_beam(label):
    """
        Changes the observed beam:
        callback function for interactive mode
        profile plot.
    """
    global patient_
    global f
    x = int(s_x.val)
    y = int(s_y.val)
    r = int(s_r.val)
    zz, P, labels = plotting_profile(vars(patient_), f, x_center=x, y_center=y, radius=r, beam=label)
    ax.lines = ax.plot(zz, P, drawstyle='steps')
    ax.set_title('Neighborhood radius R = {} pixels.'.format(r))
    ax.legend(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels))]
    for n, line in enumerate(ax.lines):
        line.set_color(colors[n])
    ax1.imshow(vars(patient_)[f][labels[0]].dds)
    fig.canvas.draw_idle()

parser = argparse.ArgumentParser(description=_DESCRIPTION)
parser.add_argument('patient_ID', help='Patient ID, i.e. 001P, 001C.')
parser.add_argument('patient_dir', help='Patient main folder.')
parser.add_argument('--T2', action='store_true',\
                    help='Perform analysis on T2 fractions.')
parser.add_argument('--beam', action='append',\
                    nargs='+', type=str, required=True,\
                    help='Beams ID, i.e. B1, B2, etc.')
parser.add_argument('--mask', action='append',\
                    nargs='+', type=str, required=True,\
                    help='DDS mask filepath (.nii). Provide them in the same order as beam.')
parser.add_argument('--mp', action='store_true',\
                    help='Perform middle point analysis. Default is False.')
parser.add_argument('--bev', action='store_true',\
                    help='Perform BEV analysis. Default is False.')
parser.add_argument('--shift_method', action='store_true',\
                    help='Perform Shift Method analysis. Default is False.')

if __name__ == '__main__':
    args = parser.parse_args()
    
    beam_ = args.beam[0]
    mask_ = args.mask[0]
    
    assert os.path.exists(args.patient_dir), "Patient main folder doesn't exist."
    assert len(beam_) == len(mask_), "Beams ID and masks doesn't match."
    
    patient_ = patient_treat.Patient(args.patient_ID, args.patient_dir)
    f, f_id = fraction_specification(args.T2)
    fraction_set(vars(patient_), f, beam_, mask_)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    ax.set_xlabel('Position along beam direction [mm]')
    ax.set_ylabel('Normalized intensity [U.A.]')
    x0 = int(patient_treat.FOV_X_DIM/2)
    y0 = int(patient_treat.FOV_Y_DIM/2)
    beam0 = beam_[0]
    r0 = 0
    delta_x = 1
    delta_y = 1
    delta_r = 1
    zz, P, labels = plotting_profile(vars(patient_), f, x_center=x0, y_center=y0, radius=r0, beam=beam0)
    ax.set_position([0.4, 0.3, 0.5, 0.5])
    ax.set_title('Neighborhood radius R = {} pixels.'.format(r0))
    ax.plot(zz, P, drawstyle='steps')
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(labels))]
    for n, line in enumerate(ax.lines):
        line.set_color(colors[n])
    ax.legend(labels)
    ax1 = fig.add_axes([0.01, 0.6, 0.3, 0.2])
    ax1.set_title('DDS mask')
    ax1.set_xlabel('x [voxel]')
    ax1.set_ylabel('y [voxel]')
    ax1.imshow(vars(patient_)[f][labels[0]].dds)
    axcolor = 'lightgoldenrodyellow'
    ax_x = plt.axes([0.4, 0.1, 0.5, 0.02], facecolor=axcolor)
    ax_y = plt.axes([0.4, 0.15, 0.5, 0.02], facecolor=axcolor)
    ax_r = plt.axes([0.4, 0.2, 0.5, 0.02], facecolor=axcolor)
    s_x = Slider(ax_x, 'x [voxel]', 0, int(patient_treat.FOV_X_DIM), valinit=x0, valstep=delta_x, valfmt='%d')
    s_y = Slider(ax_y, 'y [voxel]', 0, int(patient_treat.FOV_Y_DIM), valinit=y0, valstep=delta_y, valfmt='%d')
    s_r = Slider(ax_r, 'R [pixel]', 0, int(10), valinit=r0, valstep=delta_r, valfmt='%d')
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_r.on_changed(update)
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    button.on_clicked(reset)
    rax = plt.axes([0.025, 0.3, 0.1, 0.2], facecolor=axcolor)
    radio = RadioButtons(rax, tuple(beam_), active=0)
    radio.on_clicked(update_beam)
    plt.show()
    
    
    
    if args.mp:
        patient_d = vars(patient_)
        fraction_ = patient_d[f]
        f_separated = {beam_key:[] for beam_key in beam_}
        for i in fraction_:
            f_separated[fraction_[i].beam].append(fraction_[i])
        output_folder ='map_MP'
        output_folder = os.path.join(patient_.mainFolder, output_folder)
        result_file = os.path.join(output_folder, 'MP_result.txt')
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        with open(result_file, 'a') as _f:
            _f.write('Fraction \t mean[mm] \t std[mm]\n')
        for beam_key in f_separated.keys():
            _1 = f_separated[beam_key][0]
            _compared = f_separated[beam_key][1:]
            for i in _compared:
                mp_map = image_analysis.MP(_1.median_image, i.median_image, \
                                           mask=_1.dds, voxel_dim=1.6)
                plt.figure(i.fraction_number)
                plt.imshow(mp_map, cmap='RdBu_r', vmin=-16, vmax=16)
                plt.colorbar()
                plt.axis()
                output_filename = '{}_{}vs{}.nii'.format(_1.beam, _1.fraction_number, i.fraction_number)
                output = os.path.join(output_folder, output_filename)
                dds_processing.mask_to_image(mp_map, output)
                with open(result_file, 'a') as _f:
                    _result = '{}vs{}\t{:.3f}\t{:.3f}\n'.format(
                                                                _1.fraction_number,
                                                                i.fraction_number,
                                                                mp_map.mean(),
                                                                mp_map.std())
                    _f.write(_result)
    
    
    if args.bev:
        patient_d = vars(patient_)
        fraction_ = patient_d[f]
        f_separated = {beam_key:[] for beam_key in beam_}
        thr_BEV = {beam_key: input("Insert BEV threshold for beam {},then press 'Return': \n ".format(beam_key)).split() for beam_key in beam_}
        for i in fraction_:
            f_separated[fraction_[i].beam].append(fraction_[i])
        for beam_key in f_separated.keys():
            _1 = f_separated[beam_key][0]
            _compared = f_separated[beam_key][1:]
            for thr in thr_BEV[beam_key]:
                thr = int(thr)
                output_folder ='map_BEV_{}_perc'.format(thr)
                output_folder = os.path.join(patient_.mainFolder, output_folder)
                result_file = os.path.join(output_folder, 'BEV_result.txt')
                try:
                    os.mkdir(output_folder)
                except FileExistsError:
                    pass
                with open(result_file, 'a') as _f:
                    _f.write('Fraction \t mean[mm] \t std[mm]\n')
                _1.threshold_image = thr
                for j in _compared:
                    j.threshold_image = thr
                    range_map = image_analysis.BEV(_1.threshold_image[thr], j.threshold_image[thr], voxel_dim=1.6)
                    plt.figure(j.fraction_number)
                    plt.imshow(range_map, cmap='RdBu_r', vmin=-16, vmax=16)
                    plt.colorbar()
                    plt.axis()
                    plt.show()
                    output_filename = '{}vs{}.nii'.format(_1.fraction_number, j.fraction_number)
                    output = os.path.join(output_folder, output_filename)
                    dds_processing.mask_to_image(range_map, output)
                    with open(result_file, 'a') as _f:
                        _result = '{}vs{}\t{:.3f}\t{:.3f}\n'.format(
                                                                    _1.fraction_number,
                                                                    j.fraction_number,
                                                                    range_map.mean(),
                                                                    range_map.std())
                        _f.write(_result)

    if args.shift_method:
        patient_d = vars(patient_)
        fraction_ = patient_d[f]
        f_separated = {beam_key:[] for beam_key in beam_}
        for i in fraction_:
            f_separated[fraction_[i].beam].append(fraction_[i])
            output_folder ='map_SHIFT'
            output_folder = os.path.join(patient_.mainFolder, output_folder)
            result_file = os.path.join(output_folder, 'SHIFT_result.txt')
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            pass
        with open(result_file, 'a') as _f:
            _f.write('Fraction \t mean[mm] \t std[mm]\n')
        for beam_key in f_separated.keys():
            _1 = f_separated[beam_key][0]
            _compared = f_separated[beam_key][1:]            
            for i in _compared:
                shift_map = image_analysis.shift_method(_1.median_image, i.median_image,\
                                                        mask=_1.dds, voxel_dim=1.6)
                plt.figure(i.fraction_number)
                plt.imshow(shift_map, cmap='RdBu_r', vmin=-16, vmax=16)
                plt.colorbar()
                plt.axis()
                output_filename = '{}_{}vs{}.nii'.format(_1.beam, _1.fraction_number, i.fraction_number)
                output = os.path.join(output_folder, output_filename)
                dds_processing.mask_to_image(shift_map.filled(), output)
                with open(result_file, 'a') as _f:
                    _result = '{}vs{}\t{:.3f}\t{:.3f}\n'.format(
                                                                _1.fraction_number,
                                                                i.fraction_number,
                                                                shift_map.mean(),
                                                                shift_map.std())
                    _f.write(_result)
            
    plt.show()
