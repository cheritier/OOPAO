# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:47:10 2025

@author: cheritier
"""

from OOPAO.tools.tools import read_fits
from astropy.io import fits as pfits
import time
import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec
from OOPAO.tools.displayTools import makeSquareAxes,display_wfs_signals
from lbt_tools import *
from joblib import Parallel,delayed

# current function to decode the LBT signals into OOPAO logic. 
# Not necessary if we tune OOPAO into LBT logic but here for reference and logic of the code

def get_on_sky_modulated_signal(obj,on_sky_slopes_raw,phi_raw):
    # ordered slopes from LBT to comply with OOPAO logic
    on_sky_slopes_ordered = get_int_mat_from_lbt(obj.param,obj.wfs, IM_from_LBT = on_sky_slopes_raw)
    phi_ordered = get_int_mat_from_lbt(obj.param,obj.wfs, IM_from_LBT = phi_raw)

    
    reference_imat      = get_int_mat_from_lbt(obj.param,obj.wfs)
    reference_slope     = reference_imat[:,30]

    # reconstruct the slopes from modulation parameter
    on_sky_slopes = (np.sin(phi_ordered))*on_sky_slopes_ordered/10e-9

    n = len(on_sky_slopes)
    on_sky_slopes[:n//2]*=-1
    on_sky_slopes*= np.sign(reference_slope.T@on_sky_slopes/reference_slope.T@reference_slope)
    
    # create 2D slopes map using 2D map of valid pixels
    valid_pix_map = np.concatenate((obj.wfs.userValidSignal,obj.wfs.userValidSignal))
    valid_pix = np.reshape(valid_pix_map,valid_pix_map.shape[0]*valid_pix_map.shape[1]).astype('float64')
    # replace all the ones in the valid pixel array with the on_sky_slopes values
    valid_pix[valid_pix == 1] = on_sky_slopes
    obj.slopes_2D = np.reshape(valid_pix,[valid_pix_map.shape[0],valid_pix_map.shape[1]])
    
    obj.on_sky_slopes = on_sky_slopes
    return on_sky_slopes
# function to interpolate the intermediate values when frames are skipped
import numpy as np

def fill_with_linear_ramps(data):
    data = np.array(data, dtype=float)
    filled = data.copy()
    if filled.ndim == 1:
        filled = filled[:, None]

    n_rows, n_cols = filled.shape

    for col in range(n_cols):
        col_data = filled[:, col]
        isnan = np.isnan(col_data)

        if not np.any(isnan):
            continue

        # Indices of valid (non-NaN) and missing points
        idx = np.arange(n_rows)
        valid = ~isnan
        valid_idx = idx[valid]
        valid_vals = col_data[valid]

        # --- Left and right nearest valid indices ---
        # Forward fill: each NaN gets left neighbor index
        left_idx = np.maximum.accumulate(np.where(valid, idx, -1))
        # Backward fill: each NaN gets right neighbor index
        right_idx = np.flip(np.maximum.accumulate(np.where(valid[::-1], idx[::-1], -1)))
        right_idx = np.where(right_idx == -1, -1, right_idx)

        # Replace -1 (no valid neighbor) with nearest available
        left_val = np.where(left_idx != -1, col_data[left_idx], np.nan)
        right_val = np.where(right_idx != -1, col_data[right_idx], np.nan)

        # Distance between left/right neighbors
        left_dist = idx - left_idx
        right_dist = right_idx - idx

        # Compute linear interpolation weights
        total_dist = left_dist + right_dist
        w_right = left_dist / total_dist
        w_left = 1 - w_right

        # When missing one side, copy the other
        w_left[np.isnan(right_val)] = 1.0
        w_right[np.isnan(left_val)] = 0.0
        w_right[np.isnan(right_val)] = 0.0

        # Interpolated result
        interp_vals = w_left * left_val + w_right * right_val

        # Fill missing points only
        col_data[isnan] = interp_vals[isnan]
        filled[:, col] = col_data

    return filled.squeeze()



plt.close('all')

list_tn = ['20250422_194704']


mreg_est = []
smap_out = []
slopes_out = []

loc  = 'C:/Users/cheritier/input_data_lbt_sprint/'
n_test = 20

for TN in list_tn:
    # read the data from somewhere
    loc = 'C:/Users/cheritier/Documents/LBT/input_data_lbt_sprint_22042025/'
# TODO extract the files and relevant values from the LBT telemetry
    modes_ = read_fits(loc+'Data_'+(TN)+'/Modes_'+(TN)+'.fits')
    slopes_ = read_fits(loc+'Data_'+(TN)+'/Slopes_'+(TN)+'.fits')
    frame_counter = read_fits(loc+'Data_'+(TN)+'/FramesCounter_'+(TN)+'.fits')
    decimation = 2
    loop_frequency = 1700
    samplingTime = (1/loop_frequency)*decimation
    frame_counter = (frame_counter-frame_counter[0])//decimation
    n_skipped = frame_counter[-1]-len(frame_counter)
    print(str(n_skipped)+ ' frames skipped')
    
    # flags for the different pptions
    interpolate = True
    # interpolate with neighbour frames
    if interpolate: 
        modes                   = np.zeros((frame_counter[-1]+1,modes_.shape[1]))+np.nan
        slopes = np.zeros((frame_counter[-1]+1,slopes_.shape[1]))+np.nan
        # fill the data where the frames where not skippef
        modes[frame_counter, :] = modes_
        slopes[frame_counter, :] = slopes_
        # interpolate when required between skipped frames ( robust to more than 1 frame skipped)
        modes = fill_with_linear_ramps(modes)
        slopes = fill_with_linear_ramps(slopes)
    else:
        modes = modes_
        slopes = slopes_
    n_in = []
    mod_f = []
    buffer_smap = []
    buffer_slopes = []
    buffer_modes_demo = []
    for i in range(n_test):
        N = modes.shape[0] - i*2
        fft_modal_coefs = 2/N * np.fft.fft(modes[:N, 30])
        u = np.fft.fftfreq(N, d=samplingTime)
        ind = np.argmin(np.abs(u - 80))
        modulation_frequency = u[ind]

        # Crop or pad
        if N < modes.shape[0]:
            modes_ = modes[:N, :]
            slopes_ = slopes[:N, :]
        else:
            modes_ = np.pad(modes, ((0, N - modes.shape[0]), (0, 0)))
            slopes_ = np.pad(slopes, ((0, N - slopes.shape[0]), (0, 0)))

        # --- FFT over all slopes (2D FFT: one FFT per slope signal)
        # Each column in slopes_ is a signal
        Y_f = np.fft.fft(slopes_, axis=0)  # shape (N, n_slopes)
        Y_f_mag = np.abs(Y_f)
    
        # Extract amplitude and phase at modulation frequency index
        slopes_phase = np.arctan2(np.real(Y_f[ind, :]), np.imag(Y_f[ind, :]))
        slopes_module = 2.0 * Y_f_mag[ind, :] / N

        # put the last value to 0 (don't remember why, but this is necessary)
        slopes_phase[-1]=0
        slopes_module[-1]=0
        
        try:
            slopes_dem = get_on_sky_modulated_signal(LBT,on_sky_slopes_raw=slopes_module, phi_raw=slopes_phase)
            tmp = display_wfs_signals(LBT.wfs, signals = slopes_dem , returnOutput=True,norma=True)
            plt.title(N)
            buffer_slopes.append(slopes_dem)
            buffer_smap.append(tmp)
            mod_f.append(modulation_frequency)

        except:
            print('skipping '+str(i))   
    
    buffer_smap = np.asarray(buffer_smap)
    buffer_smap_std = np.reshape(buffer_smap,[buffer_smap.shape[0],buffer_smap.shape[1]*buffer_smap.shape[2]])
    buffer_smap_std[np.isinf(buffer_smap_std)]=0
    from OOPAO.tools.displayTools import interactive_plot
    
    loc = 'C:/Users/cheritier/Documents/LBT/data_lbt_sprint_22042025/'+TN+'/'
    slopes = loc + 'demodulated_slopes_'+str(TN)+'.fits '
    phi = loc + '/phi_'+str(TN)+'.fits'
    LBT.get_on_sky_modulated_signal(slopes=slopes, phi=phi)
    
    ref = display_wfs_signals(LBT.wfs, signals = LBT.on_sky_slopes,returnOutput=True,norma=True)
    ref[np.isinf(ref)]=0
    
    ref = np.moveaxis(np.tile(ref[:,:,None],buffer_smap.shape[0]),2,0)
    
    mod_f = np.asarray(mod_f)
    ind_f = np.argsort(mod_f)    
    
    var_sign = np.var(buffer_smap_std,axis=1)
    # interactive plot, not necessary   
    interactive_plot(mod_f[ind_f],var_sign[ind_f], im_array=buffer_smap[ind_f,:,:], im_array_ref=ref,zoom=3,n_fig=1000)
    plt.plot(mod_f[ind_f],var_sign[ind_f],'-s',markersize = 10,label = TN,alpha=0.3)
    ind_m = np.argmax(var_sign)
    slopes_out.append(buffer_slopes[ind_m])
plt.legend()