# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie
"""

import inspect
import sys
import time

import numpy as np
from joblib import Parallel, delayed

from .MisRegistration import MisRegistration
from .tools.interpolateGeometricalTransformation import interpolate_cube
from .tools.tools import emptyClass, pol2cart, print_


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
    def __init__(self,
                 telescope,
                 nSubap:float,
                 mechCoupling:float = 0.35,
                 coordinates:np.ndarray = None,
                 pitch:float = None,
                 modes:np.ndarray = None,
                 misReg = None,
                 M4_param = None,
                 nJobs:int = 30,
                 nThreads:int = 20,
                 print_dm_properties:bool = True,
                 floating_precision:int = 64,
                 altitude:float = None):
        """DEFORMABLE MIRROR
        A Deformable Mirror object consists in defining the 2D maps of influence functions of the actuators. 
        By default, the actuator grid is cartesian in a Fried Geometry with respect to the nSubap parameter. 
        The Deformable Mirror is considered to to be in a pupil plane.
        By default, the influence functions are 2D Gaussian functions normalized to 1 [m]. 
        IMPORTANT: The deformable mirror is considered to be transmissive instead of reflective. 
        This is to prevent any confusion with an eventual factor 2 in OPD due to the reflection.

        Parameters
        ----------
        telescope : Telescope
            the telescope object associated. 
            In case no coordinates are provided, the selection of the valid actuator is based on the radius
            of the telescope (assumed to be circular) and the central obstruction value (assumed to be circular).
            The telescope spiders are not considered in the selection of the valid actuators. 
            For more complex selection of actuators, specify the coordinates of the actuators using
            the optional parameter "coordinates" (see below).
        nSubap : float
            This parameter is used when no user-defined coordinates / modes are specified. This is used to compute
            the DM actuator influence functions in a fried geometry with respect to nSubap subapertures along the telescope 
            diameter. 
            If the optional parameter "pitch" is not specified, the Deformable Mirror pitch property is computed 
            as the ratio of the Telescope Diameter with the number of subaperture nSubap. 
            This impacts how the DM influence functions mechanical coupling is computed. .
        mechCoupling : float, optional
            This parameter defines the mechanical coupling between the influence functions. 
            A value of 0.35 which means that if an actuator is pushed to an arbitrary value 1, 
            the mechanical deformation at a circular distance of "pitch" from this actuator is equal to 0.35. 
            By default, "pitch" is the inter-actuator distance when the Fried Geometry is considered.   
            If the parameter "modes" is used, this parameter is ignored. The default is 0.35.
        coordinates : np.ndarray, optional
            User defined coordinates for the DM actuators. Be careful to specify the pitch parameter associated, 
            otherwise the pitch is computed using its default value (see pitch parameter). 
            If this parameter is specified, all the actuators are considered to be valid 
            (no selection based on the telescope pupil).
            The default is None.
        pitch : float, optional
            pitch considered to compute the Gaussian Influence Functions, associated to the mechanical coupling. 
            If no pitch is specified, the pitch is computed to match a Fried geometry according to the nSubap parameter.  
            The default is None.
        modes : np.ndarray, optional
            User defined influence functions or modes (modal DM) can be input to the Deformable Mirror. 
            They must match the telescope resolution and be input as a 2D matrix, where the 2D maps are 
            reshaped as a 1D vector of size n_pix*n_pix : size = [n_pix**2,n_modes].
            The default is None.
        misReg : TYPE, optional
            A Mis-Registration object (See the Mis-Registration class) can be input to apply some geometrical transformations
            to the Deformable Mirror. When using user-defined influence functions, this parameter is ignored.
            Consider to use the function applyMisRegistration in OOPAO/mis_registration_identification_algorithm/ to perform interpolations.
            The default is None.
        M4_param : Parameter File, optional
            Parameter File for M4 computation. The default is None.
        nJobs : int, optional
            Number of jobs for the joblib multi-threading. The default is 30.
        nThreads : int, optional
            Number of threads for the joblib multi-threading. The default is 20.
        print_dm_properties : bool, optional
            Boolean to print the dm properties. The default is True.
        floating_precision : int, optional
            If set to 32, uses float32 precision to save memory. The default is 64.
        altitude : float, optional
            Altitude to which the DM is conjugated. The default is None and corresponds to a DM conjugated to the ground.

        Returns
        -------
        None.

        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Deformable Mirror object are listed here: 
        _ dm.coefs             : dm coefficients in units of dm.modes, if using the defauly gaussian influence functions, in [m].
        _ dm.OPD               : the 2D map of the optical path difference in [m]
        _ dm.modes             : matrix of size: [n_pix**2,n_modes]. 2D maps of the dm influence functions (or modes for a modal dm) where the 2D maps are reshaped as a 1D vector of size n_pix*n_pix.
        _ dm.nValidAct         : number of valid actuators
        _ dm.nAct              : Total number of actuator along the diameter (valid only for the default case using cartesian fried geometry).
                                 Otherwise nAct = dm.nValidAct.
        _ dm.coordinates       : coordinates in [m] of the dm actuators (should be input as a 2D array of dimension [nAct,2])
        _ dm.pitch             : pitch used to compute the gaussian influence functions
        _ dm.misReg            : MisRegistration object associated to the dm object
        
        The main properties of the object can be displayed using :
            dm.print_properties()

        ************************** PROPAGATING THE LIGHT THROUGH THE DEFORMABLE MIRROR **************************
        The light can be propagated from a telescope tel through the Deformable Mirror dm using: 
            tel*dm
        Two situations are possible:
            * Free-space propagation: The telescope is not paired to an atmosphere object (tel.isPaired = False). 
                In that case tel.OPD is overwritten by dm.OPD: tel.OPD = dm.OPD
            
            * Propagation through the atmosphere: The telescope is paired to an atmosphere object (tel.isPaired = True). 
                In that case tel.OPD is summed with dm.OPD: tel.OPD = tel.OPD + dm.OPD

        ************************** CHANGING THE OPD OF THE MIRROR **************************
        
        * The dm.OPD can be reseted to 0 by setting the dm.coefs property to 0:
            dm.coefs = 0

        * The dm.OPD can be updated by setting the dm.coefs property using a 1D vector vector_command of length dm.nValidAct: 
            
            dm.coefs = vector_command
        
        The resulting OPD is a 2D map obtained computing the matricial product dm.modes@dm.coefs and reshaped in 2D. 
        
        * It is possible to compute a cube of 2D OPD using a 2D matrix, matrix_command of size [dm.nValidAct, n_opd]: 
            
            dm.coefs = matrix_command
        
        The resulting OPD is a 3D map [n_pix,n_pix,n_opd] obtained computing the matricial product dm.modes@dm.coefs and reshaped in 2D. 
        This can be useful to parallelize the measurements, typically when measuring interaction matrices. This is compatible with tel*dm operation. 
        
        WARNING: At the moment, setting the value of a single (or subset) actuator will not update the dm.OPD property if done like this: 
            dm.coefs[given_index] = value
        It requires to re-assign dm.coefs to itself so the change can be detected using:
            dm.coefs = dm.coefs
        
                 
        ************************** EXEMPLE **************************
        
        1) Create an 8-m diameter circular telescope with a central obstruction of 15% and the pupil sampled with 100 pixels along the diameter. 
        tel = Telescope(resolution = 100, diameter = 8, centralObstruction = 0.15)
        
        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8) 
        
        3) Create a Deformable Mirror object with 21 actuators along the diameters (20 in the pupil) and influence functions with a coupling of 45 %.
        dm = DeformableMirror(telescope = tel, nSubap = 20, mechCoupling = 0.45)
        
        4) Assign a random vector for the coefficients and propagate the light
        dm. coefs = numpy.random.randn(dm.nValidAct)
        src*tel*dm
        
        5) To visualize the influence function as seen by the telescope: 
        dm. coefs = numpy.eye(dm.nValidAct)
        src*tel*dm
        
        tel.OPD contains a cube of 2D maps for each actuator

        """
        self.print_dm_properties = print_dm_properties
        self.floating_precision = floating_precision
        self.M4_param = M4_param
        if M4_param is not None:
            if M4_param['isM4']:
                from .M4_model.make_M4_influenceFunctions import makeM4influenceFunctions

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
        if mechCoupling <=0:
            raise ValueError('The value of mechanical coupling should be positive.')
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
        if pitch is None:
            self.pitch             = self.D/(nSubap)                 # size of a subaperture
        else:
            self.pitch = pitch
        
        if misReg is None:
            # create a MisReg object to store the different mis-registration
            self.misReg = MisRegistration()
        else:
            self.misReg=misReg
        
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

# If no coordinates are given, the DM is in a Cartesian Geometry
        
        if coordinates is None:  
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
            if np.shape(coordinates)[1] !=2:
                raise AttributeError('Wrong size for the DM coordinates, the (x,y) coordinates should be input as a 2D array of dimension [nAct,2]')
                
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
                self.nValidAct = self.modes.shape[1]
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
        if self.floating_precision == 32:
            output = np.float32(output)
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
        if self.floating_precision==32:            
            self._coefs=np.float32(val)
        else:
            self._coefs=val

        if np.isscalar(val):
            if val==0:
                self._coefs = np.zeros(self.nValidAct,dtype=np.float64)
                try:
                    self.OPD = np.float64(np.reshape(np.matmul(self.modes,self._coefs),[self.resolution,self.resolution]))
                except:
                    self.OPD = np.float64(np.reshape(self.modes@self._coefs,[self.resolution,self.resolution]))

            else:
                print('Error: wrong value for the coefficients')    
        else:                
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
    def __repr__(self):
        self.print_properties()
        return ' '
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       