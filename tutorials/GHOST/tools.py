# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:39:52 2026

@author: cheritier
"""

import matplotlib.pyplot as plt
import numpy as np
from OOPAO.MisRegistration import MisRegistration
from OOPAO.tools.displayTools import makeSquareAxes, display_wfs_signals
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from OOPAO.OPD_map import OPD_map
from OOPAO.tools.tools import OopaoError
from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration

def check_wfs_pupils(valid_pixel_map, wfs, n_it=3, correct=False):
    from OOPAO.tools.tools import centroid
    plt.close("all")
    xs = wfs.sx
    ys = wfs.sy
    if correct is False:
        for i in range(4):
            I = wfs.grabFullQuadrant(i + 1, valid_pixel_map)
            [x, y] = np.asarray(centroid(I, threshold=0.3))
            I_ = np.abs(wfs.grabFullQuadrant(i + 1))
            I_ /= I_.max()
            [x_, y_] = np.asarray(centroid(I_, threshold=0.3))

            plt.figure(1)
            plt.subplot(2, 2, i + 1)
            plt.imshow(I - I_)
            plt.plot(x, y, "+", markersize=20)
            plt.plot(x_, y_, "+", markersize=20)
            plt.axis("off")
            xs[i] += x - x_
            ys[i] += y_ - y
            plt.title("Quadrant #" + str(i))
            plt.draw()

    else:
        for i_it in range(n_it):
            wfs.apply_shift_wfs(sx=xs, sy=ys)

            for i in range(4):
                I = wfs.grabFullQuadrant(i + 1, valid_pixel_map)
                [x, y] = np.asarray(centroid(I, threshold=0.3))
                I_ = np.abs(wfs.grabFullQuadrant(i + 1))
                I_ /= I_.max()
                [x_, y_] = np.asarray(centroid(I_, threshold=0.3))
                plt.figure(1)
                plt.subplot(n_it, 4, 4 * i_it + i + 1)
                plt.imshow(I - I_)
                plt.plot(x, y, "+", markersize=20)
                plt.plot(x_, y_, "+", markersize=20)
                plt.title(
                    "Step "
                    + str(i_it)
                    + " -- ["
                    + str(np.round(x - x_, 1))
                    + ","
                    + str(np.round(y - y_, 1))
                    + "]"
                )
                plt.axis("off")
                xs[i] += x - x_
                ys[i] += y_ - y
                plt.draw()
                plt.pause(0.0001)
        return


def mini_sprint(
    ngs,
    tel,
    wfs,
    index_sprint,  # modes index considered for sprint (the more the better... but the slower)
    int_mat_exp,  # reference experimental interaction matrix
    dm=None,  # input deformable mirror
    modal_basis=None,  # if dm is None : 3D cube of the modal basis, n_modes * n_res*n_res to be interpolated, otherwise M2C matrix
    arg_mis_reg=None,
    starting_point=None,
    n_iteration=5,
    single_pass=True,
    do_plot=True
    ):
    
    if dm is None and modal_basis is None:
        OopaoError('At least a dm or a modal basis is required')
    if dm is not None and modal_basis is not None:
        OopaoError('Both a dm and a modal basis are provided. Only one can be input to the function.')
    
    # mis-reg parameters to be considered (the order matters)
    if arg_mis_reg is None:
        arg_mis_reg = [
            "rotationAngle",
            "shiftX",
            "shiftY",
            "magnification",
            "radialScaling",
            "tangentialScaling",
        ]
    if np.isscalar(index_sprint):
        index_sprint = [index_sprint]
    mis_registration_dict = dict()
    mis_registration_dict["rotationAngle"] = dict()
    mis_registration_dict["rotationAngle"]["delta"] = 0.01
    mis_registration_dict["rotationAngle"]["units"] = "[deg]"

    mis_registration_dict["shiftX"] = dict()
    mis_registration_dict["shiftX"]["delta"] = 0.01 * tel.D / wfs.nSubap
    mis_registration_dict["shiftX"]["units"] = "[m]"

    mis_registration_dict["shiftY"] = dict()
    mis_registration_dict["shiftY"]["delta"] = 0.01 * tel.D / wfs.nSubap
    mis_registration_dict["shiftY"]["units"] = "[m]"

    mis_registration_dict["magnification"] = dict()
    mis_registration_dict["magnification"]["delta"] = 0.01
    mis_registration_dict["magnification"]["units"] = "[%]"

    mis_registration_dict["radialScaling"] = dict()
    mis_registration_dict["radialScaling"]["delta"] = 0.01
    mis_registration_dict["radialScaling"]["units"] = "[%]"

    mis_registration_dict["tangentialScaling"] = dict()
    mis_registration_dict["tangentialScaling"]["delta"] = 0.01
    mis_registration_dict["tangentialScaling"]["units"] = "[%]"

    int_mat_exp = np.squeeze(int_mat_exp[:, index_sprint])
    # considered modal basis and reference signal
    if starting_point is None:
        starting_point = MisRegistration()  # reference mis-registration (model)
    # % start mini-sprint
    misreg_id = MisRegistration()
    mis_reg_estimate_buffer = []
    for i_it in range(n_iteration):
        meta_mat = []  # list to stack all the sensitivity matrices (re-initialized at each iteration)
        if dm is None:
            basis = modal_basis[index_sprint, :, :]
        else:
            basis = modal_basis[:,index_sprint]
            
  
        #apply a mis-registration (both for DM and Modal basis interpolation)
        dm_tmp, basis_tmp = set_mis_registration(tel = tel,
                                      mis_registration=starting_point,
                                      dm = dm,
                                      basis = modal_basis)
        amp = 1e-10
        signal = []
        for i_mode in range(len(index_sprint)):
            opd_ = get_opd_mis_registered(modal_basis=basis_tmp,
                                          index=index_sprint[i_mode],
                                          dm=dm_tmp)
            opd = OPD_map(opd_ * tel.pupil * amp)
            ngs**tel * opd * wfs
            signal.append(wfs.signal / amp)
        ref_wfs_signal = np.squeeze(np.asarray(signal).T)
        if i_it == 0:
            ref_wfs_signal_0 = ref_wfs_signal.copy()

        for i in range(len(arg_mis_reg)):
            # initialize the delta-mis-reg corresponding to each mis-registration type
            delta_misreg = MisRegistration()
            if arg_mis_reg[i] == "magnification":
                setattr(delta_misreg,"tangentialScaling",mis_registration_dict[arg_mis_reg[i]]["delta"])
                setattr(delta_misreg,"radialScaling",mis_registration_dict[arg_mis_reg[i]]["delta"])
            else:
                setattr(delta_misreg,arg_mis_reg[i],mis_registration_dict[arg_mis_reg[i]]["delta"])
                
            #apply a mis-registration (both for DM and Modal basis interpolation)
            dm_tmp, basis_tmp = set_mis_registration(tel = tel,
                                      mis_registration=misreg_id + delta_misreg + starting_point,
                                      dm = dm,
                                      basis = basis)
            signal = []
            for i_mode in range(len(index_sprint)):
                opd_ = get_opd_mis_registered(modal_basis = modal_basis,
                                              index = index_sprint[i_mode],
                                              dm = dm_tmp)
                ngs**tel * opd * wfs
                signal.append(wfs.signal / amp)
            signal = np.hstack(signal)
            push = (signal) / mis_registration_dict[arg_mis_reg[i]]["delta"]

            if single_pass:
                pull = -push
            else:
                #apply a mis-registration (both for DM and Modal basis interpolation)
                dm_tmp, basis_tmp = set_mis_registration(tel = tel,
                                          mis_registration=misreg_id - delta_misreg + starting_point,
                                          dm = dm,
                                          basis = basis)
                signal = []
                for i_mode in range(len(index_sprint)):
                    opd_ = get_opd_mis_registered(modal_basis = modal_basis,
                                                  index = index_sprint[i_mode],
                                                  dm = dm_tmp)
                    opd = OPD_map(opd_ * tel.pupil * amp)
                    ngs**tel * opd * wfs
                    signal.append(wfs.signal / amp)
                signal = np.hstack(signal)
                pull = (signal) / mis_registration_dict[arg_mis_reg[i]]["delta"]
            meta_mat.append(0.5 * (push - pull))
        # compute mis-reg reconstructor
        meta_mat = np.asarray(meta_mat).T

        meta_rec = np.linalg.pinv(meta_mat)

        # scaling_factor
        if len(index_sprint) == 1:
            scaling = np.sum(
                np.squeeze(ref_wfs_signal) * np.squeeze(int_mat_exp)
            ) / np.sum(np.squeeze(ref_wfs_signal) * np.squeeze(ref_wfs_signal))
            estimated_mis_reg = meta_rec @ (
                (int_mat_exp * (1 / scaling) - ref_wfs_signal)
            )
        else:
            scaling = np.diag(ref_wfs_signal.T @ int_mat_exp) / np.diag(
                ref_wfs_signal.T @ ref_wfs_signal
            )
            estimated_mis_reg = meta_rec @ (
                (int_mat_exp @ np.diag(1 / scaling) - ref_wfs_signal).flatten()
            )

        # overwrite the mis-reg
        for i in range(len(arg_mis_reg)):
            if arg_mis_reg[i] == "magnification":
                setattr(misreg_id, "tangentialScaling", estimated_mis_reg[i])
                setattr(misreg_id, "radialScaling", estimated_mis_reg[i])
            else:
                setattr(misreg_id, arg_mis_reg[i], estimated_mis_reg[i])
        # update working point
        starting_point = starting_point + misreg_id
        mis_reg_estimate_buffer.append(np.asarray(estimated_mis_reg))

    mis_reg_estimate_buffer = np.asarray(mis_reg_estimate_buffer)

    misreg_out = misreg_id + starting_point
    print(misreg_out)
    if do_plot:
        for i in range(len(arg_mis_reg)):
            plt.figure()
            plt.plot(mis_reg_estimate_buffer[:, i])
            plt.xlabel("Iteration #")
            plt.title(arg_mis_reg[i])
            plt.ylabel(mis_registration_dict[arg_mis_reg[i]]["units"])
            makeSquareAxes()

        a = display_wfs_signals(wfs, ref_wfs_signal_0, norma=True, returnOutput=True)
        b = display_wfs_signals(wfs, int_mat_exp, norma=True, returnOutput=True)
        c = display_wfs_signals(wfs, ref_wfs_signal, norma=True, returnOutput=True)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(a)
        plt.title("Starting Point")
        plt.subplot(1, 3, 2)
        plt.imshow(b)
        plt.title("Target")
        plt.subplot(1, 3, 3)
        plt.imshow(c)
        plt.title("Convergence")

    return misreg_out, mis_reg_estimate_buffer


def set_mis_registration(tel,mis_registration,basis, dm = None):
    if dm is None:
        basis_tmp = interpolate_cube(
            basis,
            pixel_size_in=tel.D / tel.resolution,
            pixel_size_out=tel.D / tel.resolution,
            resolution_out=tel.resolution,
            shape_out=[tel.resolution, tel.resolution],
            mis_registration=mis_registration,
            fliplr=False,
            flipud=False)
        dm_tmp = None
    else:
        dm_tmp = applyMisRegistration(tel, mis_registration, dm_input=dm,print_dm_properties=False)
        basis_tmp = basis
    return dm_tmp,basis_tmp
    
def get_opd_mis_registered(modal_basis, index,dm = None):
    if dm is None:
        opd_ = np.squeeze(modal_basis[index, :, :]) 
    else:
        dm.coefs = modal_basis[:,index]
        opd_ = dm.OPD.copy()
    return opd_

