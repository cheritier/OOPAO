# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:34:13 2025

@author: cheritier
"""
from pymatreader import read_mat 
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap, display_wfs_signals
from OOPAO.tools.tools import OopaoError
from compute_ekatwin import compute_ekarus_model
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from OOPAO.MisRegistration import MisRegistration
from OOPAO.tools.tools import OopaoError

class Ekarus:

    def __init__(self,param):
        self.param = param

        # BE SUR TO SET CONSOLE TO WORKING DIRECTORY BEFORE RUNNING
        directory = os.getcwd().replace("\\", "/")
    
        # location of the relevant data (WFS pupil mask, KL basis, measured iMat, T152 pupil, ... )
        loc = directory + '/ekarus_inputs/'
    
        # tel_calib,_,dm_calib,_,_ = compute_papyrus_model(param = param, loc = loc, source=False, IFreal=IFreal)
        self.tel,self.ngs,self.src,self.dm,self.dm_synthetic,self.wfs,self.atm,self.tt,self.perfet_OCAM,self.OCAM = compute_ekarus_model(param = self.param, loc = loc, source=True, IFreal=False)
        
        
    def set_pupil(self,calibration=True,sky_offset = [0,0],spiders=False):
        if calibration:
            self.tel.pupil = np.pad(self.tel.pupil_calib,self.tel.n_extra_pix//2)
            self.tel.is_calibration_pupil = True
            print('Switched to Calibration pupil')
        else:
            self.tel.pupil[:]=0
            
            if max(sky_offset)>self.tel.n_extra_pix:
                raise OopaoError('The sky offset of the pupil position is too large and must be smaller than '+str(self.tel.n_extra_pix)+' pix')
            if spiders:
                    self.tel.pupil[self.tel.n_extra_pix//2-sky_offset[0] : -self.tel.n_extra_pix//2-sky_offset[0],
                                   self.tel.n_extra_pix//2-sky_offset[1] : -self.tel.n_extra_pix//2-sky_offset[1]] = self.tel.pupil_sky_spiders            
            else:
                    self.tel.pupil[self.tel.n_extra_pix//2-sky_offset[0] : -self.tel.n_extra_pix//2-sky_offset[0],
                                   self.tel.n_extra_pix//2-sky_offset[1] : -self.tel.n_extra_pix//2-sky_offset[1]] = self.tel.pupil_sky 
            self.tel.pupil = self.tel.pupil
            self.tel.is_calibration_pupil = False

            print('Switched to Sky pupil')
        return

    def check_pwfs_pupils(self,valid_pixel_map,n_it=3, correct = False):
        
        self.wfs.modulation = 10
        
        from OOPAO.tools.tools import centroid
        
        plt.close('all')
        xs = self.wfs.sx
        ys = self.wfs.sy
        if correct is False:
            for i in range(4):
                I = self.wfs.grabFullQuadrant(i+1,valid_pixel_map)
                xc = I.shape[0]//2
                
                [x,y] = np.asarray(centroid(I,threshold=0.3))
                
                I_ = np.abs(self.wfs.grabFullQuadrant(i+1))
                I_ /= I_.max()
                
                [x_,y_] = np.asarray(centroid(I_,threshold=0.3))
        
                
                plt.figure(1)
                plt.subplot(2,2,i+1)
                plt.imshow(I-I_)
                plt.plot(x,y,'+',markersize = 20)
                plt.plot(x_,y_,'+',markersize = 20)
                plt.axis('off')
                xs[i] += ((x-x_))
                ys[i] += ((y_-y))
                plt.title('PAPYRUS/PAPYTWIN Q'+str(i))
                plt.draw()
    
        else:
            # if self.wfs.telescope.is_calibration_pupil is False:
            #     raise OopaoError('The calibration of the pupil positions must be achieved using the calibration pupil.\n'+
            #                      'Switch to the calibration pupil using the Papyrus.set_pupil() method')
            for i_it in range(n_it):
                self.wfs.apply_shift_wfs(sx =xs,sy = ys)
            
                for i in range(4):
                    I = self.wfs.grabFullQuadrant(i+1,valid_pixel_map)
                    xc = I.shape[0]//2
                    
                    [x,y] = np.asarray(centroid(I,threshold=0.3))
                    
                    I_ = np.abs(self.wfs.grabFullQuadrant(i+1))
                    I_ /= I_.max()
                    
                    [x_,y_] = np.asarray(centroid(I_,threshold=0.3))
            
                    
                    plt.figure(1)
                    plt.subplot(n_it,4,4*i_it + i+1)
                    plt.imshow(I-I_)
                    plt.plot(x,y,'+',markersize = 20)
                    plt.plot(x_,y_,'+',markersize = 20)
                    plt.title('Step '+str(i_it)+' -- ['+str(np.round(x-x_,1))+','+str(np.round(y-y_,1))+']')
                    plt.axis('off')
                    xs[i] += ((x-x_))
                    ys[i] += ((y_-y))
                    plt.draw()
                    plt.pause(0.2)
        self.wfs.modulation = 5
                
                
    def bin_bench_data(self,valid_pixel,full_int_mat, ratio):
        if ratio !=1:
            mis_reg = MisRegistration()
            pixel_size_in = 1
            pixel_size_out = ratio
            resolution_out = 240//ratio
            cube_in = np.float32(np.atleast_3d(valid_pixel)).T
            valid_pixel = np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out,mis_registration=mis_reg)).astype(int).T
            cube_in = (((full_int_mat.reshape(240,240,full_int_mat.shape[1]))).T).astype(float)
            full_int_mat_ = (interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out,mis_registration=mis_reg).T).reshape(resolution_out*resolution_out,full_int_mat.shape[1])
            full_int_mat_ *= ratio*ratio
            
        else:
            valid_pixel   = valid_pixel.astype(bool)
            full_int_mat_ = (full_int_mat).astype(float)
    
        
        return valid_pixel, full_int_mat_
    
    
    
    def calibrate_mis_registration(self,dm,M2C,input_im, index_modes = np.arange(10,150,10)):
        if self.wfs.telescope.is_calibration_pupil is False:
            raise OopaoError('The calibration of the mis-registrations must be achieved using the calibration pupil.\n'+
                             'Switch to the calibration pupil using the Papyrus.set_pupil() method')
        from OOPAO.SPRINT import SPRINT
        from OOPAO.tools.tools import emptyClass
        
        # modal basis considered
        
        basis =  emptyClass()
        basis.modes         = M2C[:,index_modes]
        basis.extra         = 'PAP_full_KL'              # EXTRA NAME TO DISTINGUISH DIFFERENT SENSITIVITY MATRICES, BE CAREFUL WITH THIS!     
        

            
        Sprint = SPRINT(self, basis,dm_input=dm,n_mis_reg=3,recompute_sensitivity=True )
        Sprint.estimate(self, on_sky_slopes = input_im[:,index_modes],n_iteration=1,n_update_zero_point=2,tolerance=100)
        from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
        dm_tmp = dm.apply_mis_registration(Sprint.mis_registration_out)
        return  dm_tmp
    
    def from_im_fit_ellipse(self,loc, threshold, n):
        """
        This functions generate the modulation coordinates from 
        a snapshot of the modulated diffraction limit PSF
        The type file should .npy, if needed use function
        im2npy
        No need to use this function if modulation is perfectly circular
    
        Parameters
        ----------
        loc : Location of the input data()
        threshold : threshold to keep only the pixels of interest for 
                    fitting 
        n : number of modulation points wanted
        
        Returns
        -------
        x_t : x coordinates of the n modulation points
        y_t : y coordinates of the n modulation points
        a : minor radius of the fitted ellipse 
        b : major radius of the fitted ellipse
        e : eccentricity of the fitted ellipse
    
        """
        #Import modulation frame and applying treshold
        Im_mod = np.load(loc+'modulation_frame.npy')
        Im_mod = Im_mod/np.max(Im_mod)
        Im_mod[Im_mod <= threshold] = 0
        Im_mod[Im_mod > threshold] = 1
        
        # Ellipse coordinates points
        ellipse = np.array(np.where(Im_mod==1)).T
        
        X = ellipse[:,0:1] 
        Y = ellipse[:,1:]
        
        # Least squares problem ||Ax - b ||^2
        Afit = np.hstack([X**2, 2 * X * Y, Y**2, 2 * X, 2 * Y])
        Bfit = np.ones_like(X)
        A, B, C, D, E = np.linalg.lstsq(Afit, Bfit, rcond=None)[0].squeeze()
        F = -1
        
        num = 2 * (A*E**2 + C*D**2 + F*B**2 - 2*B*D*E - A*C*F)
        den1 = B**2 - A*C
        den2a = np.sqrt((A-C)**2 + 4*B**2) - (A+C)
        den2b = -np.sqrt((A-C)**2 + 4*B**2) - (A+C)
        
        a = np.sqrt(num / (den1 * den2a))
        b = np.sqrt(num / (den1 * den2b))
        phi =  np.arctan(2*B / (A-C))/2 + np.pi / 4
        e = np.sqrt(1 - a**2/b**2)
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        x_t = a * np.cos(phi) * np.cos(t) + b * np.sin(phi) * np.sin(t)
        y_t = a * np.sin(phi) * np.cos(t) - b * np.cos(phi) * np.sin(t)
        
        return x_t, y_t, a, b, e
    
def compress_ekarus_data(frame_,n_pix,n_pix_crop = None,offset_X = 0, offset_Y = 0):
    
    frame = np.zeros(frame_.shape)    
    
    if offset_X!=0:
        if np.sign(offset_X)==1:
            frame[offset_X:,:] = frame_[:-offset_X,:]
        else:
            frame[:offset_X,:] = frame_[-offset_X:,:]
            
    if offset_Y!=0:
        if np.sign(offset_Y)==1:
            frame[:,offset_Y:] = frame[:,:-offset_Y]        
        else:
            frame[:,:offset_Y] = frame[:,-offset_Y:]        
    sum_input = frame.sum()
    
    Q1 = frame[:n_pix,:n_pix]
    Q2 = frame[:n_pix,-n_pix:]
    Q3 = frame[-n_pix:,:n_pix]
    Q4 = frame[-n_pix:,-n_pix:]
    output_frame =  np.vstack((np.hstack((Q1,Q2)),np.hstack((Q3,Q4))))
    if n_pix_crop is not None:
        output_frame = output_frame[n_pix_crop:-n_pix_crop,n_pix_crop:-n_pix_crop]

    # if output_frame.sum()!=sum_input:
    #     print(output_frame.sum())
    #     print(sum_input)        
    #     raise OopaoError('You are missing valid pixel! select a larger n_pix_crop or n_pix value')
    return output_frame


def compute_ekarus_frame(signal,valid_signal):
    data = np.zeros(valid_signal.shape)
    data[valid_signal] = signal
    return data
