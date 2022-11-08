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

import ctypes
import time

from AO_modules.M4_model.make_M4_influenceFunctions import makeM4influenceFunctions
from AO_modules.tools.tools import print_, pol2cart, emptyClass
from AO_modules.tools.interpolateGeometricalTransformation import interpolate_cube

# try : 
#     mkl_rt = ctypes.CDLL('libmkl_rt.so')
#     mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
#     mkl_set_num_threads(8)
# except:
#     try:
#         mkl_rt = ctypes.CDLL('./mkl_rt.dll')
#         mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
#         mkl_set_num_threads(8)
#     except:
#         try:
#             import mkl
#             mkl_set_num_threads = mkl.set_num_threads
#         except:
#             mkl_set_num_threads = None

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class DeformableMirror:
    def __init__(self,telescope,nSubap,mechCoupling = 0.35, coordinates=0, pitch=0, modes=0, misReg=0, M4_param = [], nJobs = 30, nThreads = 20,print_dm_properties = True,floating_precision = 64, altitude = None ):
        self.print_dm_properties = print_dm_properties
        self.floating_precision = floating_precision
        self.M4_param = M4_param
        if M4_param:
            if M4_param['isM4']:
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
                # normalize coordinates 
                coordinates   = (coordinates/telescope.resolution - 0.5)*40
                self.M4_param = M4_param
                self.isM4 = True
                print_ ('Done!',print_dm_properties)
                b = time.time()

                print_('Done! M4 influence functions computed in ' + str(b-a) + ' s!',print_dm_properties)
            else:
                self.isM4 = False
        else:
            self.isM4 = False
        self.telescope             = telescope
        self.altitude = altitude
        if altitude is None:
            self.resolution            = telescope.resolution      # Resolution of the DM influence Functions 
            self.mechCoupling          = mechCoupling
            self.tag                   = 'deformableMirror'
            self.D                     = telescope.D
        else:
            if telescope.src.tag == 'asterism':
                self.oversampling_factor    = np.max((np.asarray(self.telescope.src.coordinates)[:,0]/(self.telescope.resolution/2)))
            else:
                self.oversampling_factor = self.telescope.src.coordinates[0]/(self.telescope.resolution/2)
            self.altitude_layer        = self.buildLayer(self.telescope,altitude)
            self.resolution            = self.altitude_layer.resolution      # Resolution of the DM influence Functions 
            self.mechCoupling          = mechCoupling
            self.tag                   = 'deformableMirror'
            self.D                     = self.altitude_layer.D


        
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
        self.current_coefs = self.coefs.copy()
        if self.print_dm_properties:
            self.print_properties()
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GEOMETRICAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def buildLayer(self,telescope,altitude):
        # initialize layer object
        layer               = emptyClass()
       
        # gather properties of the atmosphere
        layer.altitude      = altitude       
                
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D             = telescope.D+2*np.tan(telescope.fov/2)*layer.altitude*self.oversampling_factor
        layer.resolution    = int(np.ceil((telescope.resolution/telescope.D)*layer.D))
        layer.D_fov             = telescope.D+2*np.tan(telescope.fov/2)*layer.altitude
        layer.resolution_fov    = int(np.ceil((telescope.resolution/telescope.D)*layer.D))
        layer.center = layer.resolution//2
        
        if telescope.src.tag =='source':
            [x_z,y_z] = pol2cart(telescope.src.coordinates[0]*(layer.D_fov-telescope.D)/telescope.D,np.deg2rad(telescope.src.coordinates[1]))
     
            center_x = int(y_z)+layer.resolution//2
            center_y = int(x_z)+layer.resolution//2
        
            layer.pupil_footprint = np.zeros([layer.resolution,layer.resolution])
            layer.pupil_footprint[center_x-telescope.resolution//2:center_x+telescope.resolution//2,center_y-telescope.resolution//2:center_y+telescope.resolution//2 ] = 1
        else:
            layer.pupil_footprint= []
            layer.center_x= []
            layer.center_y= []

            for i in range(telescope.src.n_source):
                 [x_z,y_z] = pol2cart(telescope.src.coordinates[i][0]*(layer.D_fov-telescope.D)/telescope.D,np.deg2rad(telescope.src.coordinates[i][1]))
     
                 center_x = int(y_z)+layer.resolution//2
                 center_y = int(x_z)+layer.resolution//2
            
                 pupil_footprint = np.zeros([layer.resolution,layer.resolution])
                 pupil_footprint[center_x-telescope.resolution//2:center_x+telescope.resolution//2,center_y-telescope.resolution//2:center_y+telescope.resolution//2 ] = 1
                 layer.pupil_footprint.append(pupil_footprint)   
                 layer.center_x.append(center_x)   
                 layer.center_y.append(center_y)   

        return layer
    def get_OPD_altitude(self,i_source):
        
        if np.ndim(self.OPD)==2:                    
            OPD = np.reshape(self.OPD[np.where(self.altitude_layer.pupil_footprint[i_source]==1)],[self.telescope.resolution,self.telescope.resolution])
        else:
            OPD = np.reshape(self.OPD[self.altitude_layer.center_x[i_source]-self.telescope.resolution//2:self.altitude_layer.center_x[i_source]+self.telescope.resolution//2,self.altitude_layer.center_y[i_source]-self.telescope.resolution//2:self.altitude_layer.center_y[i_source]+self.telescope.resolution//2,:],[self.telescope.resolution,self.telescope.resolution,self.OPD.shape[2]])
    
        if self.telescope.src.src[i_source].type == 'LGS':
                    if np.ndim(self.OPD)==2:  
                        sub_im = np.atleast_3d(OPD)
                    else:
                        sub_im = np.moveaxis(OPD,2,0)
                        
                    alpha_cone = np.arctan(self.telescope.D/2/self.telescope.src.altitude[i_source])
                    h = self.telescope.src.altitude[i_source]-self.altitude_layer.altitude
                    if np.isinf(h):
                        r =self.telescope.D/2
                    else:
                        r = h*np.tan(alpha_cone)
                    ratio = self.telescope.D/r/2
                    cube_in = sub_im.T
                    pixel_size_in   = self.altitude_layer.D/self.altitude_layer.resolution
                    pixel_size_out  = pixel_size_in/ratio
                    resolution_out  = self.telescope.resolution

                    OPD = np.asarray(np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out)).T)
        
        return OPD



        
    def dm_propagation(self,telescope,OPD_in = None, i_source = None):
        if self.coefs.all() == self.current_coefs.all():
           self.coefs = self.coefs  
        if OPD_in is None:
            OPD_in = telescope.OPD_no_pupil
        
        if i_source is not None:
            dm_OPD = self.get_OPD_altitude(i_source)
        else:
            dm_OPD = self.OPD

        # case where the telescope is paired to an atmosphere
        if telescope.isPaired:
            if telescope.isPetalFree:
                telescope.removePetalling()       
            # case with single OPD
            if np.ndim(self.OPD)==2:                    
                OPD_out_no_pupil    = OPD_in + dm_OPD
            # case with multiple OPD
            else:
                OPD_out_no_pupil    = np.tile(OPD_in[...,None],(1,1,self.OPD.shape[2]))+dm_OPD
                    
        # case where the telescope is separated from a telescope object
        else:
                OPD_out_no_pupil    = dm_OPD

        return OPD_out_no_pupil
        
    def rotateDM(self,x,y,angle):
        xOut =   x*np.cos(angle)-y*np.sin(angle)
        yOut =   y*np.cos(angle)+x*np.sin(angle)
        return xOut,yOut

    def anamorphosis(self,x,y,angle,mRad,mNorm):
    
        mRad  += 1
        mNorm += 1
        xOut   = x * (mRad*np.cos(angle)**2  + mNorm* np.sin(angle)**2)  +  y * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
        yOut   = y * (mRad*np.sin(angle)**2  + mNorm* np.cos(angle)**2)  +  x * (mNorm*np.sin(2*angle)/2  - mRad*np.sin(2*angle)/2)
    
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
        print('{: ^21s}'.format('Controlled Actuators')                     + '{: ^18s}'.format(str(self.nValidAct)))
        print('{: ^21s}'.format('M4')                   + '{: ^18s}'.format(str(self.isM4)))
        print('{: ^21s}'.format('Pitch')                                    + '{: ^18s}'.format(str(self.pitch))                    +'{: ^18s}'.format('[m]'))
        print('{: ^21s}'.format('Mechanical Coupling')                      + '{: ^18s}'.format(str(self.mechCoupling))             +'{: ^18s}'.format('[%]' ))
        print('-------------------------------------------------------------------------------')
        print('Mis-registration:')
        self.misReg.print_()
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

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
            self.current_coefs = self.coefs.copy()

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