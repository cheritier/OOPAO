# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:35:18 2020

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
try:
    # error
    import cupy as np_cp

    
# print(cupy.get_default_memory_pool().get_limit()/1024/1024/1024) 
    

except:
    import numpy as np_cp
    # print('NO GPU available!')
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
        import mkl
        mkl_set_num_threads = mkl.set_num_threads



class Pyramid:
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def __init__(self,nSubap,telescope,modulation,lightRatio,pupilSeparationRatio=1.2,calibModulation=50,extraModulationFactor=0,zeroPadding=0,psfCentering=True,edgePixel=2,unitCalibration=0,binning =1,nTheta_user_defined=None, postProcessing='slopesMaps',userValidSignal=None,old_mask=True,rooftop = None):
        try:
            # error
            import cupy as np_cp
            self.gpu_available = True
            self.convert_for_gpu = np_cp.asarray
            self.convert_for_numpy = np_cp.asnumpy
            self.nJobs = 1
            self.mempool = np_cp.get_default_memory_pool()
            from AO_modules.tools.tools import get_gpu_memory
            self.mem_gpu = get_gpu_memory()
            
            print('GPU available!')    
            for i in range(len(self.mem_gpu)):   
                print('GPU device '+str(i)+' : '+str(self.mem_gpu[i]/1024)+ 'GB memory')
        
        except:
            import numpy as np_cp
            def no_function(input_matrix):
                return input_matrix
            self.gpu_available = False
            self.convert_for_gpu = no_function
            self.convert_for_numpy = no_function
        # initialize the Pyramid Object 
        self.telescope                  = telescope
        self.nTheta_user_defined        = nTheta_user_defined
        self.extraModulationFactor      = extraModulationFactor                             # Extra Factor to increase/reduce the number of modulation point
        self.nSubap                     = nSubap                                            # Number of subaperture
        self.telRes                     = self.telescope.resolution                         # Resolution of the telescope pupil
        self.edgePixel                  = 2* edgePixel                                      # Number of pixel on the edges of the PWFS pupils
        self.centerPixel                = 0                                                 # Value used for the centering for the slopes-maps computation
        self.unitCalibration            = unitCalibration                                   # Calibration of the WFS units using a Tip Tilt signal
        self.postProcessing             = postProcessing                                    # type of processing
        self.userValidSignal            = userValidSignal                                   # user valid mask for the valid pixel selection
        self.psfCentering               = psfCentering                                      # tag for the PSF centering
        self.backgroundNoise            = False                                             # background noise in photon 
        self.binning                    = binning                                           # binning factor for the detector
        self.old_mask                   = old_mask
        if self.gpu_available:
            self.joblib_setting             = 'processes'
        else:
            self.joblib_setting             = 'threads'
        self.rooftop                    = rooftop      
        # Case where the zero-padding is not specificed => taking the smallest value ensuring to get edgePixel space from the edge.
        count =0
        if zeroPadding==0:
            extraPix = self.edgePixel*(self.telRes//nSubap)
            self.zeroPadding = int((round(np.max(pupilSeparationRatio)*self.telescope.resolution)+extraPix)//2)     # Number of pixel for the zero-padding
            self.nRes = int(2*(self.zeroPadding)+self.telRes)                                                  # Resolution of the zero-padded images
            self.zeroPaddingFactor=self.nRes/self.telRes
            tmp=self.nRes%(self.telRes//self.nSubap)             # change slightly the zero-padding factor to get the right number of pixel for the detector binning
            while tmp!=0:
                count+=1
                extraPix+=(self.telRes//nSubap)
                self.zeroPadding = int((round(np.max(pupilSeparationRatio)*self.telescope.resolution)+extraPix)//2)     # Number of pixel for the zero-padding
                self.nRes = int(2*(self.zeroPadding)+self.telRes)                                                  # Resolution of the zero-padded images
                self.zeroPaddingFactor=self.nRes/self.telRes
                tmp=self.nRes%(self.telRes//self.nSubap)                                       # Making sure the zeropadding is a multiple of the number of pixel per subap.
                if count ==100:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ERROR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Error: The zero-padding value cannot be set for this pupil separation ratio! Try using a user defined zero-padding')
                    print('Aborting...')
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    sys.exit(0)
            self.edgePixel = int(extraPix/(self.telRes//nSubap))
            if self.edgePixel != 2*edgePixel:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WARNING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Warning: The number of pixel on each edges of the pupils has been changed to '+str(self.edgePixel))
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # Case where the zero-padding is user-defined (in number of pixel)
        else:
             self.zeroPadding = zeroPadding
             
             if np.max(pupilSeparationRatio)<=2*self.zeroPadding/self.telRes:    
                self.nRes = int(2*(self.zeroPadding)+self.telRes)                                 # Resolution of the zero-padded images
                self.zeroPaddingFactor = self.nRes/self.telRes                                    # zero-Padding Factor
        
             else:
                 print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ERROR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                 print('Error: The Separation of the pupils is too large for this value of zeroPadding!')
                 print('Aborting...')
                 print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                 sys.exit(0)
        #update value of edge pixel:
        
        self.pupilSeparationRatio       = pupilSeparationRatio                                               # Separation ratio of the PWFS pupils (Diameter/Distance Center to Center)
        self.tag                        = 'pyramid'                                                          # Tag of the object 
        self.cam                        = Detector(round(nSubap*self.zeroPaddingFactor))                     # WFS detector object
        self.lightRatio                 = lightRatio                                                         # Light ratio for the valid pixels selection
        self.calibModulation            = calibModulation                                                    # Modulation used for the valid pixel selection
        self.isInitialized              = False                                                              # Flag for the initialization of the WFS
        self.isCalibrated               = False                                                              # Flag for the initialization of the WFS
        self.center                     = self.nRes//2                                                       # Center of the zero-Padded array
        n_cpu = multiprocessing.cpu_count()
        if self.gpu_available is False:
            if n_cpu > 16:
                self.nJobs                      = 32                                                                 # number of jobs for the joblib package
            else:
                self.nJobs                      = 8
        A = np.ones([self.nRes,self.nRes]) + 1j*np.ones([self.nRes,self.nRes])
        self.n_max = int(0.75*(np.min(self.mem_gpu)/1024)/(A.nbytes/1024/1024/1024))
        print(self.n_max)
        self.spatialFilter              = None
        self.supportPadded              = self.convert_for_gpu(np.pad(self.telescope.pupil.astype(complex),((self.zeroPadding,self.zeroPadding),(self.zeroPadding,self.zeroPadding)),'constant'))

        
        # Prepare the Tip Tilt for the modulation
        tmp                             = np.ones([self.telRes,self.telRes])
        tmp[:,0]                        = 0
        Tip                             = (sp.morphology.distance_transform_edt(tmp))*self.telescope.pupil
        Tilt                            = (sp.morphology.distance_transform_edt(np.transpose(tmp)))*self.telescope.pupil
        
        # normalize the TT to apply the modulation in terms of lambda/D
        self.Tip                        = (((Tip/Tip.max())-0.5)*2*np.pi)*self.telescope.pupil
        self.Tilt                       = (((Tilt/Tilt.max())-0.5)*2*np.pi)*self.telescope.pupil


        # compute the phasor to center the PSF on 4 pixels
        [xx,yy]                         = np.meshgrid(np.linspace(0,self.nRes-1,self.nRes),np.linspace(0,self.nRes-1,self.nRes))
        self.phasor                     = self.convert_for_gpu(np.exp(-(1j*np.pi*(self.nRes+1)/self.nRes)*(xx+yy)))
        
       
        
        #%% MASK GENERATION
        # Creating the PWFS mask by adding two rooftops.
        # Each rooftop is created by computing the distance to the central pixel line
        self.mask_computation()
        self.user_mask = np.ones([self.nRes,self.nRes])

        # initialize the reference slopes and units 
        self.slopesUnits                = 1     
        self.referenceSignal            = 0
        self.referenceSignal_2D         = 0
        self.referencePyramidFrame      = 0 
        self.modulation                 = modulation                                        # Modulation radius (in lambda/D)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS VALID PIXEL SELECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
        # Select the valid pixels
        print('Selection of the valid pixels...')
        self.initialization(self.telescope)

        print('Done!')        
        print('Acquisition of the reference slopes and units calibration...')
        # set the modulation radius and propagate light
        self.modulation = modulation
        
        self.wfs_calibration(self.telescope)

        self.telescope.resetOPD()
        self.pyramid_propagation(telescope)
        print('Done!')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PYRAMID WFS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Pupils Diameter \t'+str(self.nSubap) + ' \t [pixels]')
        print('Pupils Separation \t'+str(((np.max(self.pupilSeparationRatio)-1)*self.nSubap)) + ' \t [pixels]')
        print('Pixel Size \t\t' + str(np.round(self.telescope.D/self.nSubap,2)) + str('\t [m]'))
        print('TT Modulation \t\t'+str(self.modulation) + ' \t [lamda/D]')
        print('PSF Core Sampling \t'+str(1+self.psfCentering*3) + ' \t [pixel(s)]')
        print('Signal Post-Processing \t' + self.postProcessing)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MASK WFS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def mask_computation(self):
        print('Pyramid Mask initialization...')
        if self.old_mask is True:
            M = np.ones([self.nRes,self.nRes])
    
            if self.psfCentering:
                # mask centered on 4 pixels
                M[:,-1+self.nRes//2] = 0
                M[:,self.nRes//2] = 0
                m = sp.morphology.distance_transform_edt(M) 
                if self.rooftop is None:
                    m += np.transpose(m)
                else:
                    if self.rooftop == 'H':
                        m = np.transpose(m)
                    
                
                # normalization of the mask 
                m[-1+self.nRes//2:self.nRes//2,-1+self.nRes//2:self.nRes//2] = m[-1+self.nRes//2:self.nRes//2,-1+self.nRes//2:self.nRes//2]/2   
                m /= np.sqrt(2) 
                m -= m.mean()
                # if self.psfCentering is False:                       
                    # m*=(1+self.nSubap)
             
    #        # create an amplitude mask for the debugging
    #            self.debugMask = np.ones([self.nRes,self.nRes])
    #            self.debugMask[:,-1+self.nRes//2:]=0
    #            self.debugMask[-1+self.nRes//2:,:]=0
                
            else:
                M[:,self.nRes//2] = 0
                m = sp.morphology.distance_transform_edt(M) 
                if self.rooftop is None:
                    m += np.transpose(m)
                else:
                    if self.rooftop == 'H':
                        m = np.transpose(m)
                # m += np.transpose(m)
                # normalization of the mask 
                m[self.nRes//2,self.nRes//2] = m[self.nRes//2,self.nRes//2]/2   
                m /= np.sqrt(2) 
    
            # apply the right angle to the faces
            if np.isscalar(self.pupilSeparationRatio):
                # case with a single value for each face (perfect pyramid)
                m*=((np.pi/(self.nRes/self.telRes))*self.pupilSeparationRatio)/np.sin(np.pi/4)
            else:
                # Prepare the Tip Tilt for the modulation
                tmp                             = np.ones([self.nRes//2,self.nRes//2])
                tmp[:,0]                        = 0
                Tip                             = (sp.morphology.distance_transform_edt(tmp))
                Tilt                            = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
                
                self.Tip_mask                   = (((Tip/np.max(np.abs(Tip))-0.5))*self.telRes)
                self.Tilt_mask                  = (((Tilt/np.max(np.abs(Tilt))-0.5))*self.telRes)           
    
                theta = np.arctan(self.pupilSeparationRatio[:,0]/(self.pupilSeparationRatio[:,1]))
                m*=((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4)
    
                norma = 2
                m[0:self.nRes//2,0:self.nRes//2]                     += norma*(((self.pupilSeparationRatio[0,0]-1)*(-self.Tip_mask)*np.cos(theta[0])) + ((self.pupilSeparationRatio[0,1]-1)*(-self.Tilt_mask)*np.sin(theta[0])))
                m[0+self.nRes//2:self.nRes,0:self.nRes//2]           += norma*(((self.pupilSeparationRatio[1,0]-1)*(-self.Tip_mask)*np.cos(theta[0])) + ((self.pupilSeparationRatio[1,1]-1)*self.Tilt_mask*np.sin(theta[0])))
                m[0:self.nRes//2,0+self.nRes//2:self.nRes]           += norma*(((self.pupilSeparationRatio[2,0]-1)*self.Tip_mask*np.cos(theta[0])) + ((self.pupilSeparationRatio[2,1]-1)*(-self.Tilt_mask)*np.sin(theta[0])))
                m[0+self.nRes//2:self.nRes,0+self.nRes//2:self.nRes] += norma*(((self.pupilSeparationRatio[3,0]-1)*self.Tip_mask*np.cos(theta[0])) + ((self.pupilSeparationRatio[3,1]-1)*(self.Tilt_mask)*np.sin(theta[0])))
                m -= m.min()          
            m -=m.mean()
            self.m = m   
            self.initial_m = m.copy()   
    
            self.mask = self.convert_for_gpu(np.complex64(np.exp(1j*m)))                                    # compute the PWFS mask)
            self.initial_mask = np.copy(self.mask)                      # Save a copy of the initial mask
    #        self.mask = self.debugMask*np.exp(1j*m)                 # compute the PWFS mask + the debug amplitude mask
    
            print('Done!')
        else:
            M = np.ones([self.nRes,self.nRes])
            if self.psfCentering:
   
                # mask centered on 4 pixels
                M[:,-1+self.nRes//2] = 0
                M[:,self.nRes//2] = 0
                m = sp.morphology.distance_transform_edt(M) 
                m += np.transpose(m)
                # normalization of the mask 
                
                m[-1+self.nRes//2:self.nRes//2,-1+self.nRes//2:self.nRes//2] = m[-1+self.nRes//2:self.nRes//2,-1+self.nRes//2:self.nRes//2]/2   
                m /= np.sqrt(2) 
                # Prepare the Tip Tilt for the faces
                tmp                                     = np.ones([self.nRes//2,self.nRes//2])
                tmp[:,0]                                = 0
                Tip                                     = (sp.morphology.distance_transform_edt(tmp))
                Tilt                                    = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
                
                Tip_mask                                = (((Tip-Tip.mean())))*((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4)
                Tilt_mask                               = (((Tilt-Tilt.mean())))*((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4)     
        
            else:
                M[:,self.nRes//2] = 0
                m = sp.morphology.distance_transform_edt(M) 
                m += np.transpose(m)
                # normalization of the mask 
                m[self.nRes//2,self.nRes//2] = m[self.nRes//2,self.nRes//2]/2   
                m /= np.sqrt(2) 
                
                # Prepare the Tip Tilt for the faces
                #1) case with the quadarant with 1 extra pixel
                tmp                                     = np.ones([self.nRes//2,self.nRes//2])
                tmp[:,0]                                = 0
                Tip                                     = (sp.morphology.distance_transform_edt(tmp))
                Tilt                                    = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
                
                Tip_mask                                = (((Tip-Tip.mean())))*((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4)
                Tilt_mask                               = (((Tilt-Tilt.mean())))*((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4)     
            
            
                Tip_mask_QS                             = np.copy(Tip_mask[:,:])
                Tilt_mask_QS                            = np.copy(Tilt_mask[:,:])
                Tip_mask_QS                             -= Tip_mask_QS.mean() 
                Tilt_mask_QS                            -= Tilt_mask_QS.mean() 
                
                #2) case with the quadarants with 1 pixel less

                tmp                                     = np.ones([1+self.nRes//2,1+self.nRes//2])
                tmp[:,0]                                = 0
                Tip                                     = (sp.morphology.distance_transform_edt(tmp))
                Tilt                                    = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
                
                Tip_mask                                = (((Tip-Tip.mean())))*((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4)
                Tilt_mask                               = (((Tilt-Tilt.mean())))*((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4)     
            
                
                Tip_mask_Q1                             = Tip_mask[:-1,:-1]
                Tilt_mask_Q1                            = Tilt_mask[:-1,:-1]
            
            m*=((np.pi/(self.nRes/self.telRes)))/np.sin(np.pi/4) 

            if np.isscalar(self.pupilSeparationRatio):
                m*=self.pupilSeparationRatio
                m -= m.mean()

            else:
                m = m.copy()*0
                theta= np.pi/4
                norma = 1
                if self.psfCentering:
                    m[0:+self.nRes//2,0:+self.nRes//2]                  += 1 *norma*(((self.pupilSeparationRatio[0,0]) * (-Tip_mask) * np.cos(theta)) + ((self.pupilSeparationRatio[0,1]) * (-Tilt_mask) * np.sin(theta)))
                    m[0+self.nRes//2:self.nRes,0:(self.nRes//2)]        += 1 *norma*(((self.pupilSeparationRatio[1,0]) * (-Tip_mask) * np.cos(theta)) + ((self.pupilSeparationRatio[1,1]) * (Tilt_mask)  * np.sin(theta)))
                    m[0:+self.nRes//2,self.nRes//2:self.nRes]           += 1 *norma*(((self.pupilSeparationRatio[2,0]) * (Tip_mask)  * np.cos(theta)) + ((self.pupilSeparationRatio[2,1]) * (-Tilt_mask) * np.sin(theta)))
                    m[0+self.nRes//2:self.nRes,self.nRes//2:self.nRes]  += 1 *norma*(((self.pupilSeparationRatio[3,0]) * (Tip_mask)  * np.cos(theta)) + ((self.pupilSeparationRatio[3,1]) * (Tilt_mask)  * np.sin(theta)))
                else:
                    m[0:+self.nRes//2,0:+self.nRes//2]                  += 1 *norma*(((self.pupilSeparationRatio[0,0]) * (-Tip_mask_Q1) * np.cos(theta)) + ((self.pupilSeparationRatio[0,1]) * (-Tilt_mask_Q1) * np.sin(theta)))
                    m[0+self.nRes//2:self.nRes,0:(self.nRes//2)]        += 1 *norma*(((self.pupilSeparationRatio[1,0]) * (-Tip_mask_QS) * np.cos(theta)) + ((self.pupilSeparationRatio[1,1]) * (Tilt_mask_QS)  * np.sin(theta)))
                    m[0:+self.nRes//2,self.nRes//2:self.nRes]           += 1 *norma*(((self.pupilSeparationRatio[2,0]) * (Tip_mask_Q1)  * np.cos(theta)) + ((self.pupilSeparationRatio[2,1]) * (-Tilt_mask_Q1) * np.sin(theta)))
                    m[0+self.nRes//2:self.nRes,self.nRes//2:self.nRes]  += 1 *norma*(((self.pupilSeparationRatio[3,0]) * (Tip_mask_Q1)  * np.cos(theta)) + ((self.pupilSeparationRatio[3,1]) * (Tilt_mask_Q1)  * np.sin(theta)))
                    
#                    m[0:+self.nRes//2,0:+self.nRes//2]                        -= np.mean(m[0:+self.nRes//2,0:+self.nRes//2])
#                    m[0+self.nRes//2:self.nRes,0:(self.nRes//2)]               -= np.mean(m[0+self.nRes//2:self.nRes,0:(self.nRes//2)])
#                    m[0:+self.nRes//2,self.nRes//2:self.nRes]                  -= np.mean(m[0:+self.nRes//2,self.nRes//2:self.nRes])
#                    m[0+self.nRes//2:self.nRes,self.nRes//2:self.nRes]          -= np.mean(m[0+self.nRes//2:self.nRes,self.nRes//2:self.nRes])

            self.m = m   
            self.mask = self.convert_for_gpu(np.exp(1j*m))                                    # compute the PWFS mask

            self.initial_m = m.copy()   
            self.initial_mask = np.copy(self.mask)     
            
            print('Done!')

                

    def apply_shift_wfs(self,sx,sy):
        # apply a TIP/TILT of the PWFS mask to shift the pupils
        # sx and sy are the units of displacements in pixels
        
        tmp                             = np.ones([self.nRes,self.nRes])
        tmp[:,0]                        = 0
        Tip                             = (sp.morphology.distance_transform_edt(tmp))
        Tilt                            = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
        
        # normalize the TT to apply the modulation in terms of lambda/D
        Tip                        = (self.telRes/self.nSubap)*(((Tip/Tip.max())-0.5)*2*np.pi)
        Tilt                       = (self.telRes/self.nSubap)*(((Tilt/Tilt.max())-0.5)*2*np.pi)
        
        self.mask = self.convert_for_gpu(np.exp(1j*(self.initial_m+sx*Tip+sy*Tilt)))
           
            


        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     
    def initialization(self,telescope):
        telescope.resetOPD()

        if self.userValidSignal is None:
            print('The valid pixel are selected on flux considerations')
            self.modulation = self.calibModulation                  # set the modulation to a large value   
            self.pyramid_propagation(telescope)                      # propagate 
            # save initialization frame
            self.initFrame = self.cam.frame
            
            # save the number of signals depending on the case    
            if self.postProcessing == 'slopesMaps' or self.postProcessing == 'slopesMaps_incidence_flux':
                # select the valid pixels of the detector according to the flux (case slopes-maps)
                I1 = self.grabQuadrant(1)
                I2 = self.grabQuadrant(2)
                I3 = self.grabQuadrant(3)
                I4 = self.grabQuadrant(4)
                
                # sum of the 4 quadrants
                self.I4Q            = I1+I2+I3+I4
                # valid pixels to consider for the slopes-maps computation
                self.validI4Q       = (self.I4Q>=self.lightRatio*self.I4Q.max()) 
                self.validSignal    = np.concatenate((self.validI4Q,self.validI4Q))
                self.nSignal        = int(np.sum(self.validSignal))
                
            if self.postProcessing == 'fullFrame':
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validSignal = (self.initFrame>=self.lightRatio*self.initFrame.max())   
                self.nSignal        = int(np.sum(self.validSignal))
        else:
            print('You are using a user-defined mask for the selection of the valid pixel')
            if self.postProcessing == 'slopesMaps':
                
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validI4Q       =  self.userValidSignal
                self.validSignal    = np.concatenate((self.validI4Q,self.validI4Q))
                self.nSignal        = int(np.sum(self.validSignal))
                
            if self.postProcessing == 'fullFrame':            
                self.validSignal    = self.userValidSignal  
                self.nSignal        = int(np.sum(self.validSignal))
                    
        # Tag to indicate that the wfs is initialized
        self.isInitialized = True
            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS CALIBRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
    def wfs_calibration(self,telescope):
        # reference slopes acquisition 
        telescope.OPD = telescope.pupil.astype(float)
        # compute the refrence slopes
        self.pyramid_propagation(telescope)
        self.referenceSignal_2D,self.referenceSignal = self.signalProcessing()
      
        # 2D reference Frame before binning with detector
        self.referencePyramidFrame         = np.copy(self.pyramidFrame)

        print('WFS calibrated!')
        self.isCalibrated= True
        telescope.OPD = telescope.pupil.astype(float)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PYRAMID TRANSFORM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def pyramid_transform(self,phase_in):
        # copy of the support for the zero-padding
        support = self.supportPadded.copy()
        # em field corresponding to phase_in
        
        if np.ndim(self.telescope.OPD)==2:  
            if self.modulation==0:
                em_field     = self.maskAmplitude*np.exp(1j*(phase_in))
            else:
                em_field     = self.maskAmplitude*np.exp(1j*(self.convert_for_gpu(self.telescope.src.phase)+phase_in))
        else:
            em_field     = self.maskAmplitude*np.exp(1j*phase_in)
            
        # zero-padding for the FFT computation
        support[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field
        del em_field
        # case with mask centered on 4 pixels
        if self.psfCentering:
            em_field_ft     = np_cp.fft.fft2(support*self.phasor)
            em_field_pwfs   = np_cp.fft.ifft2(em_field_ft*self.mask)
            I               = np_cp.abs(em_field_pwfs)**2
       
        # case with mask centered on 1 pixel
        else:
            if self.spatialFilter is not None:
                em_field_ft     = np_cp.fft.fftshift(np_cp.fft.fft2(support))*self.spatialFilter 
            else:
                em_field_ft     = np_cp.fft.fftshift(np_cp.fft.fft2(support)) 

            em_field_pwfs   = np_cp.fft.ifft2(em_field_ft*self.mask)
            I               = np_cp.abs(em_field_pwfs)**2
        del support
        del em_field_pwfs
        # self.modulation_camera_frame.append(em_field_ft)
        del em_field_ft
        del phase_in
        

        return I    
    
    
    
#        def pyramid_transform_spatial_filter(self,phase_in):
#            # copy of the support for the zero-padding
#            support = np.copy(self.supportPadded)
#            # em field corresponding to phase_in
#            if np.ndim(self.telescope.OPD)==2:  
#                if self.modulation==0:
#                    em_field     = self.maskAmplitude*np.exp(1j*(phase_in))
#                else:
#                    em_field     = self.maskAmplitude*np.exp(1j*(self.telescope.src.phase+phase_in))
#            else:
#                em_field     = self.maskAmplitude*np.exp(1j*phase_in)
#                
#            # zero-padding for the FFT computation
#            support[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field
#            
#            # case with mask centered on 4 pixels
#            if self.psfCentering:
#                em_field_ft     = np.fft.fft2(support*self.phasor)
#                em_field_pwfs   = np.fft.ifft2(em_field_ft*self.mask)
#                I               = np.abs(em_field_pwfs)**2
#           
#            # case with mask centered on 1 pixel
#            else:
#                em_field_ft     = np.fft.fftshift(np.fft.fft2(support)) 
#                em_field_pwfs   = np.fft.ifft2(em_field_ft*self.mask)
#                I               = np.abs(em_field_pwfs)**2
#            
#            self.modulation_camera_frame.append(em_field_ft)
#    
#            return I  
        
        
    def setPhaseBuffer(self,phaseIn):
        B=self.phaseBuffModulationLowres+self.convert_for_gpu(phaseIn)
        return B    
    
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PYRAMID PROPAGATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
    def pyramid_propagation(self,telescope):
        # mask amplitude for the light propagation
        self.maskAmplitude = self.convert_for_gpu(self.telescope.pupilReflectivity)
        
        if self.spatialFilter is not None:
            if np.ndim(telescope.OPD)==2:
                support_spatial_filter = np.copy(self.supportPadded)   
                em_field = self.maskAmplitude*np.exp(1j*(self.telescope.src.phase))
                support_spatial_filter[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field
                self.em_field_spatial_filter       = (np.fft.fft2(support_spatial_filter*self.phasor))
                self.pupil_plane_spatial_filter    = (np.fft.ifft2(self.em_field_spatial_filter*self.spatialFilter))
                

        
        # modulation camera
        self.modulation_camera_frame=[]

        if self.modulation==0:
            if np.ndim(telescope.OPD)==2:
                self.pyramidFrame = self.convert_for_numpy(self.pyramid_transform(self.convert_for_gpu(telescope.src.phase)))              
                
    #            self.pyramidFrame*=(self.telescope.src.fluxMap.sum())/self.pyramidFrame.sum()
                self*self.cam
                if self.isInitialized and self.isCalibrated:
                    self.pyramidSignal_2D,self.pyramidSignal=self.signalProcessing()
            else:
                nModes = telescope.OPD.shape[2]
                # move axis to get the number of modes first
                self.phase_buffer = self.convert_for_gpu(np.moveaxis(telescope.src.phase,-1,0))
                
                #define the parallel jobs
                def job_loop_multiple_modes_non_modulated():
                    Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.phase_buffer)
                    return Q 
                # applt the pyramid transform in parallel
                maps = job_loop_multiple_modes_non_modulated()
                
                self.pyramidSignal_2D    = np.zeros([self.validSignal.shape[0],self.validSignal.shape[1],nModes])
                self.pyramidSignal       = np.zeros([self.nSignal,nModes])
                
                for i in range(nModes):
                    self.pyramidFrame = self.convert_for_numpy(maps[i])
                    self*self.cam
                    if self.isInitialized:
                        self.pyramidSignal_2D[:,:,i],self.pyramidSignal[:,i] = self.signalProcessing()             
                del maps

        else:
            if np.ndim(telescope.OPD)==2:       
                # print(self.phaseBuffModulationLowres.shape)
                n_max_ = self.n_max
                if self.nTheta>n_max_:
                    # break problem in pieces: 
                    # buffer_map = self.phaseBuffModulationLowres.copy()
                    
                    nCycle = int(np.ceil(self.nTheta/n_max_))
                    # print(self.nTheta)
                    maps = self.convert_for_numpy(np_cp.zeros([self.nRes,self.nRes]))
                    for i in range(nCycle):
                        
                        if self.gpu_available:
                            try:
                                self.mempool = np_cp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                print('could not free the memory')
                            
                        if i<nCycle-1:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(self.phaseBuffModulationLowres[i*n_max_:(i+1)*n_max_,:,:]))
                                return Q 
                            maps+=self.convert_for_numpy(np_cp.sum(np_cp.asarray(job_loop_single_mode_modulated()),axis=0))

                        else:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(self.phaseBuffModulationLowres[i*n_max_:,:,:]))
                                return Q 
                            maps+=self.convert_for_numpy(np_cp.sum(np_cp.asarray(job_loop_single_mode_modulated()),axis=0))

                    self.pyramidFrame=maps/self.nTheta
                    del maps

                else:
                    #define the parallel jobs
                    def job_loop_single_mode_modulated():
                        Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.phaseBuffModulationLowres)
                        return Q 
        
                        # applt the pyramid transform in parallel
                    maps=np_cp.asarray(job_loop_single_mode_modulated())
                
                    # compute the sum of the pyramid frames for each modulation points
                    self.pyramidFrame=self.convert_for_numpy(np_cp.sum((maps),axis=0))/self.nTheta
                    del maps
                #propagate to the detector
                self*self.cam
                
                if self.isInitialized and self.isCalibrated:
                    self.pyramidSignal_2D,self.pyramidSignal=self.signalProcessing()
                
                # case with multiple modes simultaneously

            else:
                if np.ndim(telescope.OPD)==3:

                    nModes = telescope.OPD.shape[2]
                    # move axis to get the number of modes first
                    self.phase_buffer = np.moveaxis(telescope.src.phase,-1,0)

                    def jobLoop_setPhaseBuffer():
                        Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.setPhaseBuffer)(i) for i in self.phase_buffer)
                        return Q                   
                    
                    self.phaseBuffer=self.convert_for_gpu(np.reshape(np_cp.asarray(jobLoop_setPhaseBuffer()),[nModes*self.nTheta,self.telRes,self.telRes]))
                    
                    n_measurements = nModes*self.nTheta
                    n_max = self.n_max
                    
                    n_measurement_max = int(np.floor(n_max/self.nTheta))

                    maps = (np_cp.zeros([n_measurements,self.nRes,self.nRes]))
                    if n_measurements >n_max:
                        nCycle = int(np.ceil(nModes/n_measurement_max))
                        for i in range(nCycle):
                        
                            if self.gpu_available:
                                try:
                                    self.mempool = np_cp.get_default_memory_pool()
                                    self.mempool.free_all_blocks()
                                except:
                                    print('could not free the memory')
                            
                            if i<nCycle-1:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(self.phaseBuffer[i*n_measurement_max*self.nTheta:(i+1)*n_measurement_max*self.nTheta,:,:]))
                                    return Q 
                                maps[i*n_measurement_max*self.nTheta:(i+1)*n_measurement_max*self.nTheta,:,:]=(np_cp.asarray(job_loop_multiple_mode_modulated()))
    
                            else:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.convert_for_gpu(self.phaseBuffer[i*n_measurement_max*self.nTheta:,:,:]))
                                    return Q 
                                maps[i*n_measurement_max*self.nTheta:,:,:]=(np_cp.asarray(job_loop_multiple_mode_modulated()))
                        self.bufferPyramidFrames = self.convert_for_numpy(maps)
                        del self.phaseBuffer
                        del maps
                        if self.gpu_available:
                            try:
                                self.mempool = np_cp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                print('could not free the memory')

                    else:
                        def job_loop_multiple_mode_modulated():
                            Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.pyramid_transform)(i) for i in self.phaseBuffer)
                            return Q 
                        
                        self.bufferPyramidFrames  = self.convert_for_numpy(np_cp.asarray(job_loop_multiple_mode_modulated()))
                        
                    self.pyramidSignal_2D     = np.zeros([self.validSignal.shape[0],self.validSignal.shape[1],nModes])
                    self.pyramidSignal        = np.zeros([self.nSignal,nModes])
                    
                    for i in range(nModes):
                        self.pyramidFrame = np_cp.sum(self.bufferPyramidFrames[i*(self.nTheta):(self.nTheta)+i*(self.nTheta)],axis=0)/self.nTheta
                        self*self.cam
                        if self.isInitialized:
                            self.pyramidSignal_2D[:,:,i],self.pyramidSignal[:,i] = self.signalProcessing()                   
                    del self.bufferPyramidFrames
      

                else:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Error - Wrong dimension for the input phase. Aborting....')
                    print('Aborting...')
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    sys.exit(0)
                if self.gpu_available:
                    try:
                        self.mempool = np_cp.get_default_memory_pool()
                        self.mempool.free_all_blocks()
                    except:
                        print('could not free the memory')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS SIGNAL PROCESSING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
    def signalProcessing(self,cameraFrame=0):
        if cameraFrame==0:
            cameraFrame=self.cam.frame
#            self.cam.frame = self.cam.frame *(self.telescope.src.fluxMap.sum())/self.cam.frame.sum()

            
            if self.postProcessing == 'slopesMaps':
                # slopes-maps computation
                I1              = self.grabQuadrant(1,cameraFrame=0)*self.validI4Q
                I2              = self.grabQuadrant(2,cameraFrame=0)*self.validI4Q
                I3              = self.grabQuadrant(3,cameraFrame=0)*self.validI4Q
                I4              = self.grabQuadrant(4,cameraFrame=0)*self.validI4Q
                # global normalisation
                I4Q        = I1+I2+I3+I4
                norma      = np.mean(I4Q[self.validI4Q])
    #            norma = np.mean(I4Q)
                # slopesMaps computation cropped to the valid pixels
                Sx         = (I1-I2+I4-I3)            
                Sy         = (I1-I4+I2-I3)         
                # 2D slopes maps      
                slopesMaps = (np.concatenate((Sx,Sy)/norma) - self.referenceSignal_2D) *self.slopesUnits
                
                # slopes vector
                slopes     = slopesMaps[np.where(self.validSignal==1)]
                return slopesMaps,slopes
        
            if self.postProcessing == 'slopesMaps_incidence_flux':
                # slopes-maps computation
                I1              = self.grabQuadrant(1,cameraFrame=0)*self.validI4Q
                I2              = self.grabQuadrant(2,cameraFrame=0)*self.validI4Q
                I3              = self.grabQuadrant(3,cameraFrame=0)*self.validI4Q
                I4              = self.grabQuadrant(4,cameraFrame=0)*self.validI4Q
                # global normalisation
                I4Q        = I1+I2+I3+I4
                #norma      = np.mean(I4Q[self.validI4Q])
                #pdb.set_trace()
                subArea  = (self.telescope.D / self.nSubap)**2
                norma = np.float64(self.telescope.src.nPhoton*self.telescope.samplingTime*subArea)
    #            norma = np.mean(I4Q)
                # slopesMaps computation cropped to the valid pixels
                Sx         = (I1-I2+I4-I3)            
                Sy         = (I1-I4+I2-I3)         
                # 2D slopes maps      
                slopesMaps = (np.concatenate((Sx,Sy)/norma) - self.referenceSignal_2D) *self.slopesUnits
                
                # slopes vector
                slopes     = slopesMaps[np.where(self.validSignal==1)]
                return slopesMaps,slopes
        
        if self.postProcessing == 'fullFrame':
            # global normalization
            norma = np.sum(cameraFrame[self.validSignal])
            # 2D full-frame
            fullFrameMaps  = (cameraFrame / norma )  - self.referenceSignal_2D
            # full-frame vector
            fullFrame  = fullFrameMaps[np.where(self.validSignal==1)]
            
            return fullFrameMaps,fullFrame
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GRAB QUADRANTS FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def grabQuadrant(self,n,cameraFrame=0):
        
        nExtraPix   = int(np.round((np.max(self.pupilSeparationRatio)-1)*self.telescope.resolution/(self.telescope.resolution/self.nSubap)/2/self.binning))
        centerPixel = int(np.round((self.cam.resolution/self.binning)/2))
        n_pixels    = int(np.ceil(self.nSubap/self.binning))
        if cameraFrame==0:
            cameraFrame=self.cam.frame
            
        if self.rooftop is None:
            if n==3:
                I=cameraFrame[nExtraPix+centerPixel:(nExtraPix+centerPixel+n_pixels),nExtraPix+centerPixel:(nExtraPix+centerPixel+n_pixels)]
            if n==4:
                I=cameraFrame[nExtraPix+centerPixel:(nExtraPix+centerPixel+n_pixels),-nExtraPix+centerPixel-n_pixels:(-nExtraPix+centerPixel)]
            if n==1:
                I=cameraFrame[-nExtraPix+centerPixel-n_pixels:(-nExtraPix+centerPixel),-nExtraPix+centerPixel-n_pixels:(-nExtraPix+centerPixel)]
            if n==2:
                I=cameraFrame[-nExtraPix+centerPixel-n_pixels:(-nExtraPix+centerPixel),nExtraPix+centerPixel:(nExtraPix+centerPixel+n_pixels)]
        else:
            if self.rooftop == 'V':
                if n==1:
                    I=cameraFrame[centerPixel-n_pixels//2:(centerPixel)+n_pixels//2,(self.edgePixel//2):(self.edgePixel//2 +n_pixels)]
                if n==2:
                    I=cameraFrame[centerPixel-n_pixels//2:(centerPixel)+n_pixels//2,(self.edgePixel//2 +n_pixels+nExtraPix*2):(self.edgePixel//2+nExtraPix*2+2*n_pixels)]
                if n==4:
                    I=cameraFrame[centerPixel-n_pixels//2:(centerPixel)+n_pixels//2,(self.edgePixel//2):(self.edgePixel//2 +n_pixels)]
                if n==3:
                    I=cameraFrame[centerPixel-n_pixels//2:(centerPixel)+n_pixels//2,(self.edgePixel//2 +n_pixels+nExtraPix*2):(self.edgePixel//2+nExtraPix*2+2*n_pixels)]                    
            else:
                if n==1:
                    I=cameraFrame[(self.edgePixel//2):(self.edgePixel//2 +n_pixels),centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
                if n==2:
                    I=cameraFrame[(self.edgePixel//2 +n_pixels+nExtraPix*2):(self.edgePixel//2+nExtraPix*2+2*n_pixels),centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
                if n==4:
                    I=cameraFrame[(self.edgePixel//2):(self.edgePixel//2 +n_pixels),centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
                if n==3:
                    I=cameraFrame[(self.edgePixel//2 +n_pixels+nExtraPix*2):(self.edgePixel//2+nExtraPix*2+2*n_pixels),centerPixel-n_pixels//2:(centerPixel)+n_pixels//2]
  
        return I
        

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #properties required for backward compatibility (20/10/2020)
    @property
    def pyramidSignal(self):
        return self._pyramidSignal
    
    @pyramidSignal.setter
    def pyramidSignal(self,val):
        self._pyramidSignal = val
        self.signal = val
    
    @property
    def pyramidSignal_2D(self):
        return self._pyramidSignal_2D
    
    @pyramidSignal_2D.setter
    def pyramidSignal_2D(self,val):
        self._pyramidSignal_2D = val
        self.signal_2D = val
        
    @property
    def lightRatio(self):
        return self._lightRatio
    
    @lightRatio.setter
    def lightRatio(self,val):
        self._lightRatio = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Updating the map if valid pixels ...')
                self.validI4Q           = (self.I4Q>=self._lightRatio*self.I4Q.max())   
                self.validSignal        = np.concatenate((self.validI4Q,self.validI4Q))
                self.validPix           = (self.initFrame>=self.lightRatio*self.initFrame.max())   
                
                # save the number of signals depending on the case    
                if self.postProcessing == 'slopesMaps':
                    self.nSignal        = np.sum(self.validSignal)
                    # display
                    xPix,yPix = np.where(self.validI4Q==1)
                    plt.figure()
                    plt.imshow(self.I4Q.T)
                    plt.plot(xPix,yPix,'+')
                if self.postProcessing == 'fullFrame':
                    self.nSignal        = np.sum(self.validPix)  
                print('Done!')
 

    @property
    def spatialFilter(self):
        return self._spatialFilter
    
    @spatialFilter.setter
    def spatialFilter(self,val):
        self._spatialFilter = val
        if self.isInitialized:
            if val is None:         
                print('No spatial filter considered')
                self.mask = self.initial_mask
                if self.isCalibrated:
                    print('Updating the reference slopes and Wavelength Calibration for the new modulation...')
                    self.slopesUnits                = 1     
                    self.referenceSignal            = 0
                    self.referenceSignal_2D         = 0
                    self.wfs_calibration(self.telescope)
                    print('Done!')
            else:
                tmp                             = np.ones([self.nRes,self.nRes])
                tmp[:,0]                        = 0
                Tip                             = (sp.morphology.distance_transform_edt(tmp))
                Tilt                            = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
                
                # normalize the TT to apply the modulation in terms of lambda/D
                self.Tip_spatial_filter                        = (((Tip/Tip.max())-0.5)*2*np.pi)
                self.Tilt_spatial_filter                       = (((Tilt/Tilt.max())-0.5)*2*np.pi)
                if val.shape == self.mask.shape:
                    print('A spatial filter is now considered')
                    self.mask = self.initial_mask * val
                    plt.figure()
                    plt.imshow(np.real(self.mask))
                    plt.title('Spatial Filter considered')
                    if self.isCalibrated:
                        print('Updating the reference slopes and Wavelength Calibration for the new modulation...')
                        self.slopesUnits                = 1     
                        self.referenceSignal            = 0
                        self.referenceSignal_2D         = 0
                        self.wfs_calibration(self.telescope)
                        print('Done!')
                else:
                    print('ERROR: wrong shape for the spatial filter. No spatial filter attached to the mask')
                    self.mask = self.initial_mask
                
    @property
    def modulation(self):
        return self._modulation
    
    @modulation.setter
    def modulation(self,val):
        self._modulation = val
        if val !=0:
            # define the modulation point
            perimeter                       = np.pi*2*self._modulation    
            if self.nTheta_user_defined is None:
                self.nTheta                     = 4*int((self.extraModulationFactor+np.ceil(perimeter/4)))
            else:
                self.nTheta = self.nTheta_user_defined   
                
            self.thetaModulation            = np.linspace(0,2*np.pi,self.nTheta,endpoint=False)
            self.phaseBuffModulation        = np.zeros([self.nTheta,self.nRes,self.nRes]).astype(np_cp.float32)    
            self.phaseBuffModulationLowres  = np.zeros([self.nTheta,self.telRes,self.telRes]).astype(np_cp.float32)          
            
            for i in range(self.nTheta):
                dTheta                                  = self.thetaModulation[i]                
                self.TT                                 = (self.modulation*(np.cos(dTheta)*self.Tip+np.sin(dTheta)*self.Tilt))*self.telescope.pupil
                self.phaseBuffModulation[i,self.center-self.telRes//2:self.center+self.telRes//2,self.center-self.telRes//2:self.center+self.telRes//2] = self.TT
                self.phaseBuffModulationLowres[i,:,:]   = self.TT
            if self.gpu_available:
                if self.nTheta<=self.n_max:                        
                    self.phaseBuffModulationLowres = self.convert_for_gpu(self.phaseBuffModulationLowres)
        else:
            self.nTheta = 1

        if hasattr(self,'isCalibrated'):
            if self.isCalibrated:
                print('Updating the reference slopes and Wavelength Calibration for the new modulation...')
                self.slopesUnits                = 1     
                self.referenceSignal            = 0
                self.referenceSignal_2D         = 0
                self.wfs_calibration(self.telescope)
                print('Done!')
            

    @property
    def backgroundNoise(self):
        return self._backgroundNoise
    
    @backgroundNoise.setter
    def backgroundNoise(self,val):
        self._backgroundNoise = val
        if val == True:
            self.backgroundNoiseMap = []
            
 
    
    def __mul__(self,obj): 
        if obj.tag=='detector':
            I = self.pyramidFrame
            obj.frame = (obj.rebin(I,(obj.resolution,obj.resolution)))
            if self.binning != 1:
                try:
                    obj.frame = (obj.rebin(obj.frame,(obj.resolution//self.binning,obj.resolution//self.binning)))    
                except:
                    print('ERROR: the shape of the detector ('+str(obj.frame.shape)+') is not valid with the binning value requested:'+str(self.binning)+'!')
            obj.frame = obj.frame *(self.telescope.src.fluxMap.sum())/obj.frame.sum()
            
            if obj.photonNoise!=0:
                rs=np.random.RandomState(seed=int(time.time()))
                obj.frame = rs.poisson(obj.frame)
                
            if obj.readoutNoise!=0:
                obj.frame += np.int64(np.round(np.random.randn(obj.resolution,obj.resolution)*obj.readoutNoise))
#                obj.frame = np.round(obj.frame)
                
            if self.backgroundNoise is True:    
                rs=np.random.RandomState(seed=int(time.time()))
                self.backgroundNoiseAdded = rs.poisson(self.backgroundNoiseMap)
                obj.frame +=self.backgroundNoiseAdded
        else:
            print('Error light propagated to the wrong type of object')
        return -1
    class setDataModulation():
        pass
        
                                        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
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



                                   
        
