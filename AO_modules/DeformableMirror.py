# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie
"""
import numpy as np
import inspect
import sys
from AO_modules.MisRegistration import MisRegistration
from joblib import Parallel, delayed

import skimage.transform as sk
import ctypes
import time

from astropy.io import fits as pfits
from AO_modules.M4_model.make_M4_influenceFunctions import makeM4influenceFunctions
from AO_modules.tools.tools import print_
#from AO_modules.tools.interpolateGeometricalTransformation import rotateImageMatrix,rotation,translationImageMatrix,translation,anamorphosis,anamorphosisImageMatrix

try : 
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_set_num_threads(8)
except:
    try:
        mkl_rt = ctypes.CDLL('./mkl_rt.dll')
        mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
        mkl_set_num_threads(8)
    except:
        import mkl
        mkl_set_num_threads = mkl.set_num_threads

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class DeformableMirror:
    def __init__(self,telescope,nSubap,mechCoupling = 0.35, coordinates=0, pitch=0, modes=0, misReg=0, M4_param = [], nJobs = 30, nThreads = 20,print_dm_properties = True,floating_precision = 64 ):
        self.print_dm_properties = print_dm_properties
        self.floating_precision = floating_precision
        self.M4_param = M4_param
        if M4_param:
            if M4_param['isM4']:
#                from AO_modules.M4_model.make_M4_influenceFunctions import makeM4influenceFunctions
                print_('Building the set of influence functions of M4...',print_dm_properties)
                # generate the M4 influence functions            

                pup = telescope.pupil
                filename = M4_param['m4_filename']
                nAct = M4_param['nActuator']
                
                a = time.time()
                # compute M4 influence functions
                try:
                    coordinates_M4 = makeM4influenceFunctions(pup                   = pup,\
                                                              filename              = filename,\
                                                              misReg                = misReg,\
                                                              dm                    = self,\
                                                              nAct                  = nAct,\
                                                              nJobs                 = nJobs,\
                                                              nThreads              = nThreads,\
                                                              order                 = M4_param['order_m4_interpolation'],\
                                                              floating_precision    = floating_precision)
                except:
                    coordinates_M4 = makeM4influenceFunctions(pup                   = pup,\
                                                              filename              = filename,\
                                                              misReg                = misReg,\
                                                              dm                    = self,\
                                                              nAct                  = nAct,\
                                                              nJobs                 = nJobs,\
                                                              nThreads              = nThreads,\
                                                              floating_precision    = floating_precision)

    #            selection of the valid M4 actuators
                if M4_param['validActCriteria']!=0:
                    IF_STD = np.std(np.squeeze(self.modes[telescope.pupilLogical,:]), axis=0)
                    ACTXPC=np.where(IF_STD >= np.mean(IF_STD)*M4_param['validActCriteria'])
                    self.modes         = self.modes[:,ACTXPC[0]]
                
                    coordinates = coordinates_M4[ACTXPC[0],:]
                else:
                    coordinates = coordinates_M4

                self.M4_param = M4_param
                self.isM4 = True
                print_ ('Done!',print_dm_properties)
                b = time.time()

                print_('Done! M4 influence functions computed in ' + str(b-a) + ' s!',print_dm_properties)
            else:
                self.isM4 = False
        else:
            self.isM4 = False   
        self.resolution            = telescope.resolution      # Resolution of the DM influence Functions 
        self.mechCoupling          = mechCoupling
        self.tag                   = 'deformableMirror'
        self.D                     = telescope.D
        
        # case with no pitch specified (Cartesian geometry)
        if pitch==0:
            self.pitch             = self.D/(nSubap)                 # size of a subaperture
        else:
            self.pitch = pitch
        
        if misReg==0:
            # create a MisReg object to store the different mis-registration
            self.misReg = MisRegistration()
        else:
            self.misReg=misReg
        
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# If no coordinates are given, the DM is in a Cartesian Geometry
        
        if np.ndim(coordinates)==0:  
            print_('No coordinates loaded.. taking the cartesian geometry as a default',print_dm_properties)
            self.nAct                               = nSubap+1                            # In that case corresponds to the number of actuator along the diameter            
            self.nActAlongDiameter                  = self.nAct-1
            
            # set the coordinates of the DM object to produce a cartesian geometry
            x = np.linspace(-(self.D)/2,(self.D)/2,self.nAct)
            X,Y=np.meshgrid(x,x)            
            
            # compute the initial set of coordinates
            self.xIF0 = np.reshape(X,[self.nAct**2])
            self.yIF0 = np.reshape(Y,[self.nAct**2])
            
            # select valid actuators (central and outer obstruction)
            r = np.sqrt(self.xIF0**2 + self.yIF0**2)
            validActInner = r>(telescope.centralObstruction*self.D/2-0.5*self.pitch)
            validActOuter = r<=(self.D/2+0.7533*self.pitch)
    
            self.validAct = validActInner*validActOuter
            self.nValidAct = sum(self.validAct) 
            
        # If the coordinates are specified
            
        else:
            print_('Coordinates loaded...',print_dm_properties)

            self.xIF0 = coordinates[:,0]
            self.yIF0 = coordinates[:,1]
            self.nAct = len(self.xIF0)                            # In that case corresponds to the total number of actuators
            self.nActAlongDiameter = (self.D)/self.pitch
            
            validAct=(np.arange(0,self.nAct))                     # In that case assumed that all the Influence Functions provided are controlled actuators
            
            self.validAct = validAct.astype(int)         
            self.nValidAct = self.nAct 
            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INFLUENCE FUNCTIONS COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        #  initial coordinates
        xIF0 = self.xIF0[self.validAct]
        yIF0 = self.yIF0[self.validAct]

        # anamorphosis
        xIF3,yIF3 = self.anamorphosis(xIF0,yIF0,self.misReg.anamorphosisAngle*np.pi/180,self.misReg.tangentialScaling,self.misReg.radialScaling)
        
        # rotation
        xIF4,yIF4 = self.rotateDM(xIF3,yIF3,self.misReg.rotationAngle*np.pi/180)
        
        # shifts
        xIF = xIF4-self.misReg.shiftX
        yIF = yIF4-self.misReg.shiftY
        
        self.xIF = xIF
        self.yIF = yIF

        
        # corresponding coordinates on the pixel grid
        u0x      = self.resolution/2+xIF*self.resolution/self.D
        u0y      = self.resolution/2+yIF*self.resolution/self.D      
        self.nIF = len(xIF)
        # store the coordinates
        self.coordinates        = np.zeros([self.nIF,2])
        self.coordinates[:,0]   = xIF
        self.coordinates[:,1]   = yIF
        
        if self.isM4==False:
            print_('Generating a Deformable Mirror: ',print_dm_properties)
            if np.ndim(modes)==0:
                print_('Computing the 2D zonal modes...',print_dm_properties)
    #                FWHM of the gaussian depends on the anamorphosis
                def joblib_construction():
                    Q=Parallel(n_jobs=8,prefer='threads')(delayed(self.modesComputation)(i,j) for i,j in zip(u0x,u0y))
                    return Q 
                self.modes=np.squeeze(np.moveaxis(np.asarray(joblib_construction()),0,-1))
                    
            else:
                print_('Loading the 2D zonal modes...',print_dm_properties)
                self.modes = modes
                print_('Done!',print_dm_properties)

        else:
            print_('Using M4 Influence Functions',print_dm_properties)
        if floating_precision==32:            
            self.coefs = np.zeros(self.nValidAct,dtype=np.float32)
        else:
            self.coefs = np.zeros(self.nValidAct,dtype=np.float64)
            
        if self.print_dm_properties:
            self.print_properties()
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GEOMETRICAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def rotateDM(self,x,y,angle):
        xOut =   x*np.cos(angle)-y*np.sin(angle)
        yOut =   y*np.cos(angle)+x*np.sin(angle)
        return xOut,yOut

    def anamorphosis(self,x,y,angle,mRad,mNorm):
    
        mRad  += 1
        mNorm += 1
        xOut   = x * (mRad*np.cos(angle)**2  + mNorm* np.sin(angle)**2)  +  y * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
        yOut   = y * (mRad*np.cos(angle)**2  + mNorm* np.sin(angle)**2)  +  x * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
    
        return xOut,yOut
        
    def modesComputation(self,i,j):
        x0 = i
        y0 = j
        cx = (1+self.misReg.radialScaling)*(self.resolution/self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))
        cy = (1+self.misReg.tangentialScaling)*(self.resolution/self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))

#                    Radial direction of the anamorphosis
        theta  = self.misReg.anamorphosisAngle*np.pi/180
        x      = np.linspace(0,1,self.resolution)*self.resolution
        X,Y    = np.meshgrid(x,x)
    
#                Compute the 2D Gaussian coefficients
        a = np.cos(theta)**2/(2*cx**2)  +  np.sin(theta)**2/(2*cy**2)
        b = -np.sin(2*theta)/(4*cx**2)   +  np.sin(2*theta)/(4*cy**2)
        c = np.sin(theta)**2/(2*cx**2)  +  np.cos(theta)**2/(2*cy**2)
    
        G=np.exp(-(a*(X-x0)**2 +2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))    
        output = np.reshape(G,[1,self.resolution**2])
        return output
    
    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFORMABLE MIRROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Controlled Actuators \t '+str(self.nIF))
        if self.isM4:
            print('M4 influence functions \t Yes')
            print('Pixel Size \t\t'+str(np.round(self.D/self.resolution,2)) + ' \t [m]')
        else:
            print('M4 influence functions \t No')
            print('Pixel Size \t\t'+str(np.round(self.D/self.resolution,2)) + ' \t [m]')
            print('Pitch \t\t\t '+str(self.pitch) + ' \t [m]')
            print('Mechanical Coupling \t '+str(self.mechCoupling) + ' \t [m]')        
        print('Rotation: ' +str(np.round(self.misReg.rotationAngle,2)) + ' deg -- shift X: ' +str(np.round(self.misReg.shiftX,2)) +' m -- shift Y: ' +str(np.round(self.misReg.shiftY,2)) +' m -- Anamorphosis Angle: ' +str(np.round(self.misReg.anamorphosisAngle,2)) +' deg -- Radial Scaling: ' +str(np.round(self.misReg.radialScaling,2)) + ' -- Tangential Scaling: ' +str(np.round(self.misReg.tangentialScaling,2)))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

#        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    @property
    def coefs(self):
        return self._coefs
    
    @coefs.setter
    def coefs(self,val):
        if np.isscalar(val):
            if val==0:
                if self.floating_precision==32:            
                    self._coefs = np.zeros(self.nValidAct,dtype=np.float32)
                else:
                    self._coefs = np.zeros(self.nValidAct,dtype=np.float64)
                    
                # self._coefs=np.arange(0,self.nValidAct)*val
                
                try:
                    self.OPD =  np.float64(np.reshape(np.matmul(self.modes,self._coefs),[self.resolution,self.resolution]))
                except:
                    self.OPD= np.float64(np.reshape(self.modes@self._coefs,[self.resolution,self.resolution]))

            else:
                print('Error: wrong value for the coefficients')    
        else:
            self._coefs=val
            if len(val)==self.nValidAct:
#                case of a single mode at a time
                if np.ndim(val)==1:
                    try:
                        self.OPD = np.float64(np.reshape(np.matmul(self.modes,self._coefs),[self.resolution,self.resolution]))
                    except:
                        self.OPD = np.float64(np.reshape(self.modes@self._coefs,[self.resolution,self.resolution]))

#                case of multiple modes at a time
                else:
                    try:
                        self.OPD =  np.float64(np.reshape(np.matmul(self.modes,self._coefs),[self.resolution,self.resolution,val.shape[1]]))
                    except:
                        self.OPD =  np.float64(np.reshape(self.modes@self._coefs,[self.resolution,self.resolution,val.shape[1]]))

            else:
                print('Error: wrong value for the coefficients')    
                sys.exit(0)


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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       