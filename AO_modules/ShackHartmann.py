# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie
"""

import numpy as np
import scipy.ndimage as sp
import sys
import inspect
import time
import matplotlib.pyplot as plt
import multiprocessing
from AO_modules.Detector import Detector
from AO_modules.tools.tools import bin_ndarray
import scipy.ndimage as ndimage

try:
    from joblib import Parallel, delayed
except:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('WARNING: The joblib module is not installed. This would speed up considerably the operations.')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

import ctypes
try : 
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_set_num_threads(6)
except:
    try:
        mkl_rt = ctypes.CDLL('./mkl_rt.dll')
        mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
        mkl_set_num_threads(6)
    except:
        print('Could not optimize the parallelisation of the code ')



class ShackHartmann:
    def __init__(self,nSubap,telescope,lightRatio,threshold_cog = 0,is_geometric = False ):
        self.tag            = 'shackHartmann'
        self.telescope      = telescope
        self.is_geometric   = is_geometric
        self.nSubap         = nSubap
        self.lightRatio     = lightRatio
        self.pupil          = telescope.pupil.astype(float)     
        self.zero_padding   = 2
        self.n_pix_subap    = self.telescope.resolution// self.nSubap 
        self.n_pix_lenslet  = self.n_pix_subap*self.zero_padding 
        self.center         = self.n_pix_lenslet//2 
        self.threshold_cog  = threshold_cog
        
        self.cam                = Detector(round(nSubap*self.n_pix_subap))                     # WFS detector object
        self.cam.photonNoise    = 0
        self.cam.readoutNoise   = 0        # single lenslet 
        self.lenslet_frame  = np.zeros([self.n_pix_subap*self.zero_padding,self.n_pix_subap*self.zero_padding], dtype =complex)
        
        # camera frame 
        self.camera_frame   = np.zeros([self.n_pix_subap*(self.nSubap),self.n_pix_subap*(self.nSubap)], dtype =float)
        
        # cube of lenslet zero padded
        self.cube           = np.zeros([self.nSubap**2,self.n_pix_lenslet,self.n_pix_lenslet])
        
        self.index_x                = []
        self.index_y                = []
        # reference signal
        self.sx0                    = np.zeros([self.nSubap,self.nSubap])
        self.sy0                    = np.zeros([self.nSubap,self.nSubap])
        # signal vector
        self.sx                     = np.zeros([self.nSubap,self.nSubap])
        self.sy                     = np.zeros([self.nSubap,self.nSubap])
        # signal map
        self.SX                     = np.zeros([self.nSubap,self.nSubap])
        self.SY                     = np.zeros([self.nSubap,self.nSubap])
        # flux per subaperture
        self.photon_per_subaperture = np.zeros(self.nSubap**2)
        self.reference_slopes_maps = np.zeros([self.nSubap*2,self.nSubap])
        self.get_camera_frame_multi = False
        count=0
        print('Selecting valid subapertures based on flux considerations..')

        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)
                
                mask_amp_SH = np.sqrt(self.telescope.src.fluxMap[i*self.n_pix_subap:(i+1)*self.n_pix_subap,j*self.n_pix_subap:(j+1)*self.n_pix_subap]).astype(float)
                # define the cube of lenslet arrays
                self.cube[count,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = mask_amp_SH
                self.photon_per_subaperture[count] = mask_amp_SH.sum()
                count+=1
        [xx,yy]                    = np.meshgrid(np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet),np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet))
        self.phasor                = np.exp(-(1j*np.pi*(self.n_pix_lenslet+1)/self.n_pix_lenslet)*(xx+yy))
        self.index_x = np.asarray(self.index_x)
        self.index_y = np.asarray(self.index_y)
        
        
        self.photon_per_subaperture_2D = np.reshape(self.photon_per_subaperture,[self.nSubap,self.nSubap])
                
        
        self.valid_subapertures = np.reshape(self.photon_per_subaperture >= self.lightRatio*np.max(self.photon_per_subaperture), [self.nSubap,self.nSubap])
        
        self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])

        [self.validLenslets_x , self.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(self.photon_per_subaperture))
        
        # index of valid slopes X and Y
        self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
        # number of valid lenslet
        self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
        self.nSignal = 2*self.nValidSubaperture
        
        self.isInitialized = False
        print('Acquiring reference slopes..')        
        self.telescope.resetOPD()    
        self.sh_measure()        
        self.reference_slopes_maps[self.valid_slopes_maps] = np.concatenate((self.sx0,self.sy0))[self.valid_slopes_maps]
        self.isInitialized = True
        print('Done!')

    def centroid(self,im, threshold = 0):
        im[im<threshold*im.max()]=0
        [x,y] = ndimage.center_of_mass(im.T)
        return x,y
#%% DIFFRACTIVE 

# single measurement 
    def lenslet_propagation_diffractive(self,mask,ind_x,ind_y):
        
        support = np.copy(self.lenslet_frame)
        
        lenslet_phase = self.telescope.src.phase[ind_x*self.n_pix_subap:(ind_x+1)*self.n_pix_subap,ind_y*self.n_pix_subap:(ind_y+1)*self.n_pix_subap]
        
        support[self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = lenslet_phase
        
        norma = mask.shape[0]
        
        ft_em_field_lenslet = (np.fft.fft2((mask*np.exp(1j*support)) * self.phasor))/norma
        
        I = np.abs(ft_em_field_lenslet)**2
        
        I = bin_ndarray(I, [self.n_pix_subap,self.n_pix_subap], operation='sum')
        
        if self.cam.photonNoise!=0:
            rs = np.random.RandomState(seed=int(time.time()))
            I  = rs.poisson(I)
                
        if self.cam.readoutNoise!=0:
            I += np.int64(np.round(np.random.randn(I.shape[0],I.shape[1])*self.cam.readoutNoise))
        
        self.camera_frame[ind_x*self.n_pix_subap:(ind_x+1)*self.n_pix_subap,ind_y*self.n_pix_subap:(ind_y+1)*self.n_pix_subap] = I

        if self.isInitialized:
            [x,y] = self.centroid(I,threshold=self.threshold_cog)
            self.sx[ind_x,ind_y] = (x-self.sx0[ind_x,ind_y] )
            self.sy[ind_x,ind_y] = (y-self.sy0[ind_x,ind_y] )
        else:
            [x,y] = self.centroid(I,threshold=self.threshold_cog)
            self.sx0[ind_x,ind_y] = x
            self.sy0[ind_x,ind_y] = y
            
        return I
    
# multiple measurements
    def get_phase_buffer(self,amp,ind_x,ind_y):
        support = np.copy(self.lenslet_frame)
                
        support[self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = np.exp(1j*self.telescope.src.phase[ind_x*self.n_pix_subap:(ind_x+1)*self.n_pix_subap,ind_y*self.n_pix_subap:(ind_y+1)*self.n_pix_subap])
        
        return support*amp*self.phasor    
    
    def get_lenslet_phase_buffer(self,phase_in):
        self.telescope.src.phase = np.squeeze(phase_in) 
        
        def joblib_get_phase_buffer():
            Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.get_phase_buffer)(i,j,k) for i,j,k in zip(self.cube[self.valid_subapertures_1D,:,:],self.index_x[self.valid_subapertures_1D],self.index_y[self.valid_subapertures_1D]))
            return Q
        
        out = np.asarray(joblib_get_phase_buffer())
        
        return out
        
    def fill_camera_frame(self,ind_x,ind_y,index_frame,I):
        self.camera_frame[index_frame,ind_x*self.n_pix_subap:(ind_x+1)*self.n_pix_subap,ind_y*self.n_pix_subap:(ind_y+1)*self.n_pix_subap] = I

    def compute_camera_frame_multi(self,maps_intensisty):   
        self.ind_frame =np.zeros(maps_intensisty.shape[0],dtype=(int))
        self.maps_intensisty = maps_intensisty
        index_x = np.tile(self.index_x[self.valid_subapertures_1D],self.phase_buffer.shape[0])
        index_y = np.tile(self.index_y[self.valid_subapertures_1D],self.phase_buffer.shape[0])
        
        for i in range(self.phase_buffer.shape[0]):
            self.ind_frame[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture]=i
        
        def joblib_fill_camera_frame():
            Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.fill_camera_frame)(i,j,k,l) for i,j,k,l in zip(index_x,index_y,self.ind_frame,self.maps_intensisty))
            return Q
        
        joblib_fill_camera_frame()
        return
        
    #%% GEOMETRIC self   
         
    def gradient_2D(self,arr):
        res_x = (np.gradient(arr,axis=1)/self.telescope.pixelSize)*self.telescope.pupil
        res_y = (np.gradient(arr,axis=0)/self.telescope.pixelSize)*self.telescope.pupil
        return res_x,res_y
        
    def lenslet_propagation_geometric(self,arr):
        
        [SLx,SLy]  = self.gradient_2D(arr)
        
        sx = (bin_ndarray(SLx, [self.nSubap,self.nSubap], operation='mean'))
        sy = (bin_ndarray(SLy, [self.nSubap,self.nSubap], operation='mean'))
        
        return np.concatenate((sx,sy))
            
            
#%% self Measurement 
    def sh_measure(self,phase_in = None):
        if phase_in is not None:
            self.telescope.src.phase = phase_in
            
        if self.is_geometric is False:
            
            if np.ndim(self.telescope.src.phase)==2:
                # reset camera frame
                self.camera_frame   = np.zeros([self.n_pix_subap*(self.nSubap),self.n_pix_subap*(self.nSubap)], dtype =float)

                def compute_diffractive_signals():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.lenslet_propagation_diffractive)(i,j,k) for i,j,k in zip(self.cube,self.index_x,self.index_y))
                    return Q
                
                self.maps_intensity = np.asarray(compute_diffractive_signals())
                self.SX[self.validLenslets_x,self.validLenslets_y] = self.sx[self.validLenslets_x,self.validLenslets_y]
                self.SY[self.validLenslets_x,self.validLenslets_y] = self.sy[self.validLenslets_x,self.validLenslets_y]
                        
                self.signal = np.concatenate([self.sx[self.validLenslets_x,self.validLenslets_y],self.sy[self.validLenslets_x,self.validLenslets_y]])
                        
                self.signal_2D = np.concatenate([self.SX,self.SY])
                        
                self.signal_2D[np.isnan(self.signal_2D)] = 0
                
                self*self.cam
                        
            else:
                # set phase buffer
                self.phase_buffer = np.moveaxis(self.telescope.src.phase_no_pupil,-1,0)
                # reset camera frame
                self.camera_frame   = np.zeros([self.phase_buffer.shape[0],self.n_pix_subap*(self.nSubap),self.n_pix_subap*(self.nSubap)], dtype =float)

                def compute_diffractive_signals_multi():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.get_lenslet_phase_buffer)(i) for i in self.phase_buffer)
                    return Q 
                self.maps_intensity = np.reshape(np.asarray(compute_diffractive_signals_multi()),[self.phase_buffer.shape[0]*np.sum(self.valid_subapertures_1D),16,16])
                norma = self.maps_intensity.shape[1]
                
                F = np.abs(np.fft.fft2(self.maps_intensity)/norma)**2
                
                F_binned = (bin_ndarray(F, [F.shape[0],self.n_pix_subap,self.n_pix_subap], operation='sum'))

                if self.cam.photonNoise!=0:
                    rs = np.random.RandomState(seed=int(time.time()))
                    F_binned  = rs.poisson(F_binned)
                        
                if self.cam.readoutNoise!=0:
                    F_binned += np.int64(np.round(np.random.randn(F_binned.shape[0],F_binned.shape[1],F_binned.shape[2])*self.cam.readoutNoise))
                
                def joblib_centroid():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.centroid)(i) for i in F_binned)
                    return Q
                
                if self.get_camera_frame_multi is True:
                    self.compute_camera_frame_multi(F_binned)
                
                self.centroid_multi = np.asarray(joblib_centroid())
                
                self.signal_2D = np.zeros([self.phase_buffer.shape[0],self.nSubap*2,self.nSubap])

                for i in range(self.phase_buffer.shape[0]):
                    self.SX[self.validLenslets_x,self.validLenslets_y] = self.centroid_multi[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture,0]
                    self.SY[self.validLenslets_x,self.validLenslets_y] = self.centroid_multi[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture,1]
                    signal_2D = np.concatenate((self.SX,self.SY)) - self.reference_slopes_maps
                    signal_2D[~self.valid_slopes_maps] = 0
                    self.signal_2D[i,:,:] = signal_2D
                    
                self.signal = self.signal_2D[:,self.valid_slopes_maps].T
                self*self.cam

        else:
            if np.ndim(self.telescope.src.phase)==2:
                self.signal_2D = self.lenslet_propagation_geometric(self.telescope.src.phase_no_pupil)*self.valid_slopes_maps
                    
                self.signal = self.signal_2D[self.valid_slopes_maps]
                
            else:
                self.phase_buffer = np.moveaxis(self.telescope.src.phase_no_pupil,-1,0)

                def compute_diffractive_signals():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.lenslet_propagation_geometric)(i) for i in self.phase_buffer)
                    return Q
                maps = compute_diffractive_signals()
                self.signal_2D = np.asarray(maps)

        
    def __mul__(self,obj): 
        if obj.tag=='detector':
            obj.frame = self.camera_frame
        else:
            print('Error light propagated to the wrong type of object')
        return -1# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
    def show(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        print(self.tag+':')
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                if not(a[0].startswith('_')):
                    if not np.shape(a[1]):
                        tmp=a[1]
                        try:
                            print('          '+str(a[0])+': '+str(tmp.tag)+' object') 
                        except:
                            print('          '+str(a[0])+': '+str(a[1])) 
                    else:
                        if np.ndim(a[1])>1:
                            print('          '+str(a[0])+': '+str(np.shape(a[1])))   
    
            
