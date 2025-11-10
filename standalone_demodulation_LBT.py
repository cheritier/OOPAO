# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 22:27:35 2025

@author: cheritier
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

#================================ functions required for the demodulation and to decode the data ============================================================
    
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


def get_int_mat_from_lbt(wfs_pup,binning, filename_int_mat,filename_slopex,IM_from_LBT = None):
    "Decode the LBT interaction matrix to match the OOPAO logic"
    with open(filename_slopex) as f:
        mylist = (f.read().splitlines())
    slopes_frame = (np.asarray(mylist).astype(int))
    
    #!!! ordering of the slopes is [sx,sy,sx,sy,...]
    pix2slopes = [np.where(slopes_frame!=-1)]
    if IM_from_LBT is None:
        IM_from_LBT = fits.getdata(filename_int_mat)        
        
    if np.ndim(IM_from_LBT)==1: 
        valid_pix       = (np.where(abs(IM_from_LBT)>0))      

        IM_for_model_0  = np.squeeze(IM_from_LBT[valid_pix])
        N1 = 120
        N2 = int(N1*2/binning)
        A= np.zeros([N2*N1])
        A[pix2slopes]= IM_for_model_0[slopes_frame[pix2slopes]]
    
        valid_signals_T = np.concatenate((wfs_pup,wfs_pup)).T    
        valid_signals = np.concatenate((wfs_pup,wfs_pup))

        IM_for_model = IM_for_model_0.copy()
            
        tmp = np.copy(valid_signals_T)
        tmp[np.where(valid_signals_T>0)] = A[pix2slopes]
        tmp_T = tmp.T
        IM_for_model[:] = tmp_T[np.where(valid_signals>0)]
        nSlopes = IM_for_model_0.shape[0]
        IM_for_model[1+nSlopes//2 : -1] = -IM_for_model[1+nSlopes//2 : -1] 

    else:
        valid_pix       = (np.where(abs(IM_from_LBT[:,10])>0))    
        IM_for_model_0  = np.squeeze(IM_from_LBT[valid_pix,:])
        N1 = 120
        N2 = int(N1*2/binning)
        
        A= np.zeros([N2*N1,IM_for_model_0.shape[1]])
    
        A[pix2slopes,:]= IM_for_model_0[slopes_frame[pix2slopes],:]
    
       
        valid_signals_T = np.concatenate((wfs_pup,wfs_pup)).T
        #print(valid_signals_T)
        
        valid_signals = np.concatenate((wfs_pup,wfs_pup))
    
        IM_for_model = IM_for_model_0.copy()
            
        for i in range(IM_for_model_0.shape[1]):
            tmp = np.copy(valid_signals_T)
           # test = tmp[np.where(valid_signals_T>0)]
            #print(test.shape)
            tmp[np.where(valid_signals_T>0)] = A[pix2slopes,i]
            tmp_T = tmp.T
            IM_for_model[:,i] = tmp_T[np.where(valid_signals>0)]
            
        nSlopes = IM_for_model_0.shape[0]
        
        IM_for_model[1+nSlopes//2 : -1,:] = -IM_for_model[1+nSlopes//2 : -1,:]

    return IM_for_model


def get_wfs_pupil(filename,N0 = 120,binning =1):
    "Load the WFS valid pixel map for the requested binning"
    N1 = int(N0)
    N2 = int(N1*2/binning) 
    A= np.zeros(N2*N1)
    index_pup = fits.getdata(filename)
    index_pup=index_pup.astype(int)
    A[index_pup]=1
    wfsPup_ = np.reshape(A,[N1,N2])
    lin,col = np.where(wfsPup_>0)
    wfsPup=wfsPup_[min(lin):max(lin)+1,min(col):max(col)+1]
    wfsPup = np.flip(np.fliplr(np.rot90(((wfsPup)))))
    return wfsPup

def demodulate_on_sky_signal(TN,n_test):
    """Extract the data from the tracking number and perform the demodulation.
    n_test corresponds to the number of demodulation frequency to try"""
    
    # TODO extract the files and relevant values from the TN directly
    # --
    loc_tn = 'C:/Users/cheritier/Documents/LBT/input_data_lbt_sprint_22042025/'
    modes_ = fits.getdata(loc_tn+'Data_'+(TN)+'/Modes_'+(TN)+'.fits')
    slopes_ = fits.getdata(loc_tn+'Data_'+(TN)+'/Slopes_'+(TN)+'.fits')
    frame_counter = fits.getdata(loc_tn+'Data_'+(TN)+'/FramesCounter_'+(TN)+'.fits')
    decimation = 2
    loop_frequency = 1700
    binning = 1
    loc = 'C:/Users/cheritier/Documents/LBT/lbt_data_model/old_data_from_LBT/SX/'
    wfs_pup = get_wfs_pupil(filename = loc+'pupils/mode2/20190909_203854//pup1.fits')
    filename_slopex = loc+'/pupils/mode2/20190909_203854//slopex'
    filename_int_mat = loc+'/KL_v20/RECs/Intmat_20181215_201757.fits'
    # --
    
    samplingTime = (1/loop_frequency)*decimation
    frame_counter = (frame_counter-frame_counter[0])//decimation
    n_skipped = frame_counter[-1]-len(frame_counter)
    
    print(str(n_skipped) + ' frames skipped in ' + TN)
    # flags for the different pptions
    interpolate = True
    # interpolate with neighbour frames
    if interpolate:
        modes = np.zeros((frame_counter[-1]+1, modes_.shape[1]))+np.nan
        slopes = np.zeros((frame_counter[-1]+1, slopes_.shape[1]))+np.nan
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
            # ordered slopes from LBT to comply with OOPAO logic
            on_sky_slopes_ordered = get_int_mat_from_lbt(wfs_pup = wfs_pup,
                                                         binning = binning,
                                                         filename_int_mat = filename_int_mat,
                                                         filename_slopex=filename_slopex,
                                                         IM_from_LBT = slopes_module)
            
            phi_ordered = get_int_mat_from_lbt(wfs_pup = wfs_pup,
                                                         binning = binning,
                                                         filename_int_mat = filename_int_mat,
                                                         filename_slopex=filename_slopex,
                                                         IM_from_LBT = slopes_phase)
        
            reference_imat      = get_int_mat_from_lbt(wfs_pup = wfs_pup,
                                                         binning = binning,
                                                         filename_int_mat = filename_int_mat,
                                                         filename_slopex=filename_slopex,
                                                         IM_from_LBT = None)
            reference_slope     = reference_imat[:,30]
        
            # reconstruct the slopes from modulation parameter (amplitude hardcoded for now)
            on_sky_slopes = (np.sin(phi_ordered))*on_sky_slopes_ordered/10e-9
        
            n = len(on_sky_slopes)
            on_sky_slopes[:n//2]*=-1
            on_sky_slopes*= np.sign(reference_slope.T@on_sky_slopes/reference_slope.T@reference_slope)
            
            # create 2D slopes map using 2D map of valid pixels
            valid_pix_map = np.concatenate((wfs_pup,wfs_pup))
            valid_pix = np.reshape(valid_pix_map,valid_pix_map.shape[0]*valid_pix_map.shape[1]).astype('float64')
            # replace all the ones in the valid pixel array with the on_sky_slopes values
            valid_pix[valid_pix == 1] = on_sky_slopes
            slopes_2D = np.reshape(valid_pix,[valid_pix_map.shape[0],valid_pix_map.shape[1]])
            
            buffer_slopes.append(on_sky_slopes)
            buffer_smap.append(slopes_2D)
            mod_f.append(modulation_frequency)

        except:
            print('skipping '+str(i))   
    
    buffer_smap = np.asarray(buffer_smap)
    buffer_smap_std = np.reshape(buffer_smap,[buffer_smap.shape[0],buffer_smap.shape[1]*buffer_smap.shape[2]])
    buffer_smap_std[np.isinf(buffer_smap_std)]=0
    
    mod_f = np.asarray(mod_f)
    ind_f = np.argsort(mod_f)    
    
    var_sign = np.var(buffer_smap_std,axis=1)
    # interactive plot, not necessary   
    # from OOPAO.tools.displayTools import interactive_plot
    # interactive_plot(mod_f[ind_f],var_sign[ind_f], im_array=buffer_smap[ind_f,:,:], im_array_ref=buffer_smap[ind_f,:,:],zoom=3,n_fig=1000)
    # plt.plot(mod_f[ind_f],var_sign[ind_f],'-s',markersize = 10,label = TN,alpha=0.3)
    ind_m = np.argmax(var_sign)
    slopes_out = buffer_slopes[ind_m]
    slopes_out_2D = buffer_smap[ind_m]
    
    return slopes_out,slopes_out_2D



#%%


slopes, slopes_2D = demodulate_on_sky_signal(TN = '20250422_194704',n_test = 10)

plt.figure()
plt.imshow(slopes_2D)
