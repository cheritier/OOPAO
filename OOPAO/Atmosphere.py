# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:59:02 2020

@author: cheritie
"""

import inspect
import json
import time

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import matplotlib.gridspec as gridspec

from .phaseStats import ft_phase_screen, ft_sh_phase_screen, makeCovarianceMatrix
from .tools.displayTools import getColorOrder,makeSquareAxes
from .tools.interpolateGeometricalTransformation import interpolate_cube, interpolate_image
from .tools.tools import createFolder, emptyClass, globalTransformation, pol2cart, translationImageMatrix


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Atmosphere:
    def __init__(self,
                 telescope,
                 r0:float,
                 L0:float,
                 windSpeed:list,
                 fractionalR0:list,
                 windDirection:list,
                 altitude:list,
                 mode:float=2,
                 param = None,
                 asterism = None):
        """ ATMOSPHERE.
        An Atmosphere is made of one or several layer of turbulence that follow the Van Karmann statistics. 
        Each layer is considered to be independant to the other ones and has its own properties (direction, speed, etc.)
        The Atmosphere object can be defined for a single Source object (default) or multi Source Object. 
        The Source coordinates allow to span different areas in the field (defined as well by the tel.fov).
        If the source type is an LGS the cone effect is considered using an interpolation. 
        NGS and LGS can be combined together in the Asterism object. 
        The convention chosen is that all the wavelength-dependant atmosphere parameters are expressed at 500 nm. 

        Parameters
        ----------
        telescope : Telescope
            The telescope object to which the Atmosphere is associated. 
            This object carries the phase, flux, pupil information and sampling time as well as the type of source (NGS/LGS, source/asterism).
        r0 : float
            the Fried Parameter in m, at 500 nm.
        L0 : float
            Outer scale parameter.
        windSpeed : list
            List of wind-speed for each layer in [m/s].
        fractionalR0 : list
            Cn2 profile of the turbulence. This should be a list of values for each layer.
        windDirection : list
            List of wind-direction for each layer in [deg].
        altitude : list
            List of altitude for each layer in [m].
        mode : float, optional
            Method to compute the atmospheric spectrum from which are computed the atmospheric phase screens. 
            1 : using aotools dependency
            2 : using OOPAO dependancy
            The default is 2.
        param : Parameter File Object, optional
            Parameter file of the system. Once computed, the covariance matrices are saved in the calibration data folder and loaded instead of re-computed evry time.
            The default is None.
        asterism : Asterism, optional
            If the system contains multiple source, an astrism should be input to the atmosphere object.
            The default is None.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        ************************** COUPLING A TELESCOPE AND AN ATMOSPHERE OBJECT **************************
        A Telescope object "tel" can be coupled to an Atmosphere object "atm" using: 
            _ tel + atm
        This means that a bridge is created between atm and tel: everytime that atm.OPD is updated, the tel.OPD property is automatically set to atm.OPD to reproduce the effect of the turbulence.
        
        A Telescope object "tel" can be separated of an Atmosphere object "atm" using: 
            _ tel - atm
        This corresponds to a diffraction limited case (no turbulence)

    
        ************************** PROPERTIES **************************
        
        The main properties of the Atmosphere object are listed here: 
    
        _ atm.OPD : Optical Path Difference in [m] truncated by the telescope pupil. If the atmosphere has multiple sources, the OPD is a list of OPD for each source
        _ atm.OPD_no_pupil : Optical Path Difference in [m]. If the atmosphere has multiple sources, the OPD is a list of OPD for each source
        _ atm.r0                   
        _ atm.L0
        _ atm.nLayer                                : number of turbulence layers
        _ atm.seeingArcsec                          : seeing in arcsec at 500 nm
        _ atm.layer_X                               : access the child object corresponding to the layer X where X starts at 0   
        
        The main properties of the object can be displayed using :
            atm.print_properties()
            
        the following properties can be updated on the fly:
            _ atm.r0            
            _ atm.windSpeed      
            _ atm.windDirection 
        ************************** FUNCTIONS **************************

        _ atm.update()                              : update the OPD of the atmosphere for each layer according to the time step defined by tel.samplingTime  
        _ atm.update(OPD)                           : update the OPD of the atmosphere using a user defined OPD                   
        _ atm.generateNewPhaseScreen(seed)          : generate a new phase screen for the atmosphere OPD
        _ atm.print_atm_at_wavelength(wavelength)   : prompt seeing and r0 at specified wavelength
        _ atm.print_atm()                           : prompt the main properties of the atm object
        _ display_atm_layers(layer_index)           : imshow the OPD of each layer with the intersection beam for each source
        
        """        
        self.hasNotBeenInitialized  = True
        self.r0_def                 = 0.15              # Fried Parameter in m 
        self.r0                     = r0                # Fried Parameter in m 
        self.fractionalR0           = fractionalR0      # Cn2 square profile
        self.L0                     = L0                # Outer Scale in m
        self.altitude               = altitude          # altitude of the layers
        self.nLayer                 = len(fractionalR0)     # number of layer
        self.windSpeed              = windSpeed         # wind speed of the layers in m/s
        self.windDirection          = windDirection     # wind direction in degrees
        self.tag                    = 'atmosphere'      # Tag of the object
        self.nExtra                 = 2                 # number of extra pixel to generate the phase screens
        self.wavelength             = 500*1e-9          # Wavelengt used to define the properties of the atmosphere
        self.telescope              = telescope         # associated telescope object
        self.user_defined_opd       = False             # default value to update phase screens at each iteration 
        if self.telescope.src is None:
            raise AttributeError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        self.mode                   = mode              # DEBUG -> first phase screen generation mode
        self.seeingArcsec           = 206265*(self.wavelength/self.r0)
        self.asterism               = asterism          # case when multiple sources are considered (LGS and NGS)
        self.param                  = param
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATM INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def initializeAtmosphere(self,telescope):
        phase_support = self.initialize_phase_support()
        self.fov      = telescope.fov
        self.fov_rad  = telescope.fov_rad
        if self.hasNotBeenInitialized:
            self.initial_r0 = self.r0
            for i_layer in range(self.nLayer):       
                # create the layer
                print('Creation of layer' + str(i_layer+1) + '/' + str(self.nLayer) + ' ...' )
                tmpLayer=self.buildLayer(telescope,self.r0_def,self.L0,i_layer = i_layer)
                setattr(self,'layer_'+str(i_layer+1),tmpLayer)
                
                phase_support = self.fill_phase_support(tmpLayer,phase_support,i_layer)
                tmpLayer.phase_support = phase_support
                tmpLayer.phase *= self.wavelength/2/np.pi

        else:
            print('Re-setting the atmosphere to its initial state...' )
            self.r0 = self.initial_r0
            for i_layer in range(self.nLayer):       
                print('Updating layer' + str(i_layer+1) + '/' + str(self.nLayer) + ' ...' )
                tmpLayer = getattr(self,'layer_'+str(i_layer+1))
                tmpLayer.phase          = tmpLayer.initialPhase/self.wavelength*2*np.pi
                tmpLayer.randomState    = RandomState(42+i_layer*1000)
               
                Z = tmpLayer.phase[tmpLayer.innerMask[1:-1,1:-1]!=0]
                X = np.matmul(tmpLayer.A,Z) + np.matmul(tmpLayer.B,tmpLayer.randomState.normal( size=tmpLayer.B.shape[1]))
                
                tmpLayer.mapShift[tmpLayer.outerMask!=0] = X
                tmpLayer.mapShift[tmpLayer.outerMask==0] = np.reshape(tmpLayer.phase,tmpLayer.resolution*tmpLayer.resolution)
                
                tmpLayer.notDoneOnce = True                

                setattr(self,'layer_'+str(i_layer+1),tmpLayer) 
                
                phase_support = self.fill_phase_support(tmpLayer,phase_support,i_layer)

                # wavelenfth scaling
                tmpLayer.phase *= self.wavelength/2/np.pi
        self.generateNewPhaseScreen(seed=0)             
        self.hasNotBeenInitialized  = False        
        # move of one time step to create the atm variables 
        self.update()   
        # save the resulting phase screen in OPD  
        self.set_OPD(phase_support)
        self.print_properties()
            
    def buildLayer(self,telescope,r0,L0,i_layer,compute_turbulence = True):
        """
            Generation of phase screens using the method introduced in Assemat et al (2006)
        """

    
        # initialize layer object
        layer               = emptyClass()
        # create a random state to allow reproductible sequences of phase screens
        layer.randomState   = RandomState(42+i_layer*1000)
        
        # gather properties of the atmosphere
        layer.altitude      = self.altitude[i_layer]       
        layer.windSpeed     = self.windSpeed[i_layer]
        layer.direction     = self.windDirection[i_layer]
        
        # compute the X and Y wind speed
        layer.vY            = layer.windSpeed*np.cos(np.deg2rad(layer.direction))
        layer.vX            = layer.windSpeed*np.sin(np.deg2rad(layer.direction))      
        layer.extra_sx = 0
        layer.extra_sy = 0
        
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        
        layer.D_fov             = self.telescope.D+2*np.tan(self.fov_rad/2)*layer.altitude
        layer.resolution_fov    = int(np.ceil((self.telescope.resolution/self.telescope.D)*layer.D_fov))
        layer.resolution        = layer.resolution_fov + 4 #4 pixels are added as a margin for the edges
        layer.D                 = layer.resolution *self.telescope.D/ self.telescope.resolution


        
        layer.center = layer.resolution//2
        
        if self.asterism is None:
            [x_z,y_z] = pol2cart(layer.altitude*np.tan(self.telescope.src.coordinates[0]/206265) * layer.resolution / layer.D,np.deg2rad(self.telescope.src.coordinates[1]))

            center_x = int(y_z)+layer.resolution//2
            center_y = int(x_z)+layer.resolution//2
        
            layer.pupil_footprint = np.zeros([layer.resolution,layer.resolution])
            layer.pupil_footprint[center_x-self.telescope.resolution//2:center_x+self.telescope.resolution//2,center_y-self.telescope.resolution//2:center_y+self.telescope.resolution//2 ] = 1
        else:
            layer.pupil_footprint= []
            for i in range(self.asterism.n_source):
                 [x_z,y_z] = pol2cart(self.asterism.coordinates[i][0]**np.tan(self.telescope.src.coordinates[0]/206265) * layer.resolution / layer.D,np.deg2rad(self.asterism.coordinates[i][1]))
     
                 center_x = int(y_z)+layer.resolution//2
                 center_y = int(x_z)+layer.resolution//2
            
                 pupil_footprint = np.zeros([layer.resolution,layer.resolution])
                 pupil_footprint[center_x-self.telescope.resolution//2:center_x+self.telescope.resolution//2,center_y-self.telescope.resolution//2:center_y+self.telescope.resolution//2 ] = 1
                 layer.pupil_footprint.append(pupil_footprint)   
        # layer pixel size
        layer.d0            = layer.D/layer.resolution
        
        # number of pixel for the phase screens computation
        layer.nExtra        = self.nExtra
        layer.nPixel        = int(1+np.round(layer.D/layer.d0))
        if compute_turbulence:
            print('-> Computing the initial phase screen...')  
            a=time.time()
            if self.mode == 2:
                layer.phase        = ft_sh_phase_screen(self,layer.resolution,layer.D/layer.resolution,seed=i_layer)
            else: 
                    layer.phase         = ft_phase_screen(self,layer.resolution,layer.D/layer.resolution,seed=i_layer)                    
            layer.initialPhase = layer.phase.copy()
            layer.seed = i_layer
            b=time.time()
            print('initial phase screen : ' +str(b-a) +' s')
            
            # Outer ring of pixel for the phase screens update 
            layer.outerMask             = np.ones([layer.resolution+layer.nExtra,layer.resolution+layer.nExtra])
            layer.outerMask[1:-1,1:-1]  = 0
            
            # inner pixels that contains the phase screens
            layer.innerMask             = np.ones([layer.resolution+layer.nExtra,layer.resolution+layer.nExtra])
            layer.innerMask -= layer.outerMask
            layer.innerMask[1+layer.nExtra:-1-layer.nExtra,1+layer.nExtra:-1-layer.nExtra] = 0
            
            l = np.linspace(0,layer.resolution+1,layer.resolution+2) * layer.D/(layer.resolution-1)
            u,v = np.meshgrid(l,l)
            
            layer.innerZ = u[layer.innerMask!=0] + 1j*v[layer.innerMask!=0]
            layer.outerZ = u[layer.outerMask!=0] + 1j*v[layer.outerMask!=0]
            
            layer.ZZt, layer.ZXt, layer.XXt, layer.ZZt_inv =  self.get_covariance_matrices(layer)                            

            layer.ZZt_r0 = self.ZZt_r0.copy()     
            layer.ZXt_r0 = self.ZXt_r0.copy()     
            layer.XXt_r0 = self.XXt_r0.copy()     
            layer.ZZt_inv_r0 =  self.ZZt_inv_r0.copy()                           
            
            layer.A         = np.matmul(layer.ZXt_r0.T,layer.ZZt_inv_r0)    
            layer.BBt             = layer.XXt_r0 -  np.matmul(layer.A,layer.ZXt_r0)
            layer.B         = np.linalg.cholesky(layer.BBt)
            layer.mapShift  = np.zeros([layer.nPixel+1,layer.nPixel+1])        
            Z               = layer.phase[layer.innerMask[1:-1,1:-1]!=0]
            X               = np.matmul(layer.A,Z) + np.matmul(layer.B,layer.randomState.normal(size=layer.B.shape[1]))
            
            layer.mapShift[layer.outerMask!=0] = X
            layer.mapShift[layer.outerMask==0] = np.reshape(layer.phase,layer.resolution*layer.resolution)
            layer.notDoneOnce                  = True

        print('Done!')
        
    
        return layer
    
    
    def add_row(self,layer,stepInPixel,map_full = None):
        if map_full is None:
            map_full = layer.mapShift
        shiftMatrix                         = translationImageMatrix(map_full,[stepInPixel[0],stepInPixel[1]]) #units are in pixel of the M1            
        tmp                                 = globalTransformation(map_full,shiftMatrix)
        onePixelShiftedPhaseScreen          = tmp[1:-1,1:-1]        
        Z                                   = onePixelShiftedPhaseScreen[layer.innerMask[1:-1,1:-1]!=0]
        X                                   = layer.A@Z + layer.B@layer.randomState.normal(size=layer.B.shape[1])
        map_full[layer.outerMask!=0]  = X
        map_full[layer.outerMask==0]  = np.reshape(onePixelShiftedPhaseScreen,layer.resolution*layer.resolution)
        return onePixelShiftedPhaseScreen

    def set_pupil_footprint(self):
        
        for i_layer in range(self.nLayer):
            layer = getattr(self,'layer_'+str(i_layer+1)) 
            if self.asterism is None:
                if self.telescope.src.chromatic_shift is not None:                    
                    if len(self.telescope.src.chromatic_shift) == self.nLayer:
                        chromatic_shift = self.telescope.src.chromatic_shift[i_layer]
                    else:
                        raise ValueError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')    
                    
                else:
                    chromatic_shift = 0
                [x_z,y_z] = pol2cart(layer.altitude*np.tan((self.telescope.src.coordinates[0]+chromatic_shift)/206265) * layer.resolution / layer.D,np.deg2rad(self.telescope.src.coordinates[1]))
                layer.extra_sx = int(x_z)-x_z
                layer.extra_sy = int(y_z)-y_z
                
                center_x = int(y_z)+layer.resolution//2
                center_y = int(x_z)+layer.resolution//2
        
                layer.pupil_footprint = np.zeros([layer.resolution,layer.resolution])
                layer.pupil_footprint[center_x-self.telescope.resolution//2:center_x+self.telescope.resolution//2,center_y-self.telescope.resolution//2:center_y+self.telescope.resolution//2 ] = 1
            else:
                layer.pupil_footprint= []
                for i in range(self.asterism.n_source):
                    [x_z,y_z] = pol2cart(self.asterism.coordinates[i][0]*np.tan(self.telescope.src.coordinates[0]/206265) * layer.resolution / layer.D,np.deg2rad(self.asterism.coordinates[i][1]))
                    
                    center_x = int(y_z)+layer.resolution//2
                    center_y = int(x_z)+layer.resolution//2
                    
                    pupil_footprint = np.zeros([layer.resolution,layer.resolution])
                    pupil_footprint[center_x-self.telescope.resolution//2:center_x+self.telescope.resolution//2,center_y-self.telescope.resolution//2:center_y+self.telescope.resolution//2 ] = 1
                    layer.pupil_footprint.append(pupil_footprint)   
            
        
        
        
    def updateLayer(self,layer,shift = None):
        self.ps_loop    = layer.D / (layer.resolution)
        ps_turb_x       = layer.vX*self.telescope.samplingTime 
        ps_turb_y       = layer.vY*self.telescope.samplingTime 
        
        if layer.vX==0 and layer.vY==0 and shift is None:
            layer.phase = layer.phase
            
        else:
            if layer.notDoneOnce:
                layer.notDoneOnce = False
                layer.ratio = np.zeros(2)
                layer.ratio[0] = ps_turb_x/self.ps_loop  
                layer.ratio[1] = ps_turb_y/self.ps_loop
                layer.buff = np.zeros(2)
            
            if shift is None:
                ratio = layer.ratio
            else:
                 ratio = shift    # shift in pixels
            tmpRatio = np.abs(ratio)
            tmpRatio[np.isinf(tmpRatio)]=0
            nScreens = (tmpRatio)
            nScreens = nScreens.astype('int')
            
            stepInPixel =np.zeros(2)
            stepInSubPixel =np.zeros(2)
            
            for i in range(nScreens.min()):
                stepInPixel[0]=1
                stepInPixel[1]=1
                stepInPixel=stepInPixel*np.sign(ratio)
                layer.phase = self.add_row(layer,stepInPixel)
                
            for j in range(nScreens.max()-nScreens.min()):   
                stepInPixel[0]=1
                stepInPixel[1]=1
                stepInPixel=stepInPixel*np.sign(ratio)
                stepInPixel[np.where(nScreens==nScreens.min())]=0
                layer.phase = self.add_row(layer,stepInPixel)
            
            
            stepInSubPixel[0] =  (np.abs(ratio[0])%1)*np.sign(ratio[0])
            stepInSubPixel[1] =  (np.abs(ratio[1])%1)*np.sign(ratio[1])
            
            layer.buff += stepInSubPixel
            if np.abs(layer.buff[0])>=1 or np.abs(layer.buff[1])>=1:   
                stepInPixel[0] = 1*np.sign(layer.buff[0])
                stepInPixel[1] = 1*np.sign(layer.buff[1])
                stepInPixel[np.where(np.abs(layer.buff)<1)]=0    
                
                layer.phase = self.add_row(layer,stepInPixel)
    
            layer.buff[0]   =  (np.abs(layer.buff[0])%1)*np.sign(layer.buff[0])
            layer.buff[1]   =  (np.abs(layer.buff[1])%1)*np.sign(layer.buff[1])
                
            shiftMatrix     = translationImageMatrix(layer.mapShift,[layer.buff[0],layer.buff[1]]) #units are in pixel of the M1            
            layer.phase     = globalTransformation(layer.mapShift,shiftMatrix)[1:-1,1:-1]

    def update(self,OPD=None):
        if OPD is None:
            self.user_defined_opd = False

            phase_support = self.initialize_phase_support()
            for i_layer in range(self.nLayer):
                tmpLayer=getattr(self,'layer_'+str(i_layer+1))
                self.updateLayer(tmpLayer)
                # tmpLayer.phase *= self.wavelength/2/np.pi
    
                phase_support = self.fill_phase_support(tmpLayer,phase_support,i_layer)
            self.set_OPD(phase_support)
        else:
            self.user_defined_opd = True
            # case where the OPD is input
            self.OPD_no_pupil   = OPD
            self.OPD            = OPD*self.telescope.pupil
            
        if self.telescope.isPaired:
            self*self.telescope
            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATM METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def initialize_phase_support(self):
        if self.asterism is None:
            phase_support = np.zeros([self.telescope.resolution,self.telescope.resolution])
        else:
            phase_support = []
            for i in range(self.asterism.n_source):
                phase_support.append(np.zeros([self.telescope.resolution,self.telescope.resolution]))
        return phase_support
    def fill_phase_support(self,tmpLayer,phase_support,i_layer):
        if self.asterism is None:
            _im = tmpLayer.phase.copy()

            if tmpLayer.extra_sx !=0 or tmpLayer.extra_sy != 0:
                                
                pixel_size_in   = 1
                pixel_size_out  = 1
                resolution_out  = _im.shape[0]
                _im = np.squeeze(interpolate_image(_im, pixel_size_in, pixel_size_out, resolution_out,shift_x =tmpLayer.extra_sx, shift_y= tmpLayer.extra_sy  ))
                    
            phase_support+= np.reshape(_im[np.where(tmpLayer.pupil_footprint==1)],[self.telescope.resolution,self.telescope.resolution])* np.sqrt(self.fractionalR0[i_layer])
        else:
            for i in range(self.asterism.n_source):
                if self.asterism.src[i].type == 'LGS':
                    sub_im = np.reshape(tmpLayer.phase[np.where(tmpLayer.pupil_footprint[i]==1)],[self.telescope.resolution,self.telescope.resolution])
                    alpha_cone = np.arctan(self.telescope.D/2/self.asterism.altitude[i])
                    h = self.asterism.altitude[i]-tmpLayer.altitude
                    if np.isinf(h):
                        r =self.telescope.D/2
                    else:
                        r = h*np.tan(alpha_cone)
                    ratio = self.telescope.D/r/2
                    cube_in = np.atleast_3d(sub_im).T
                    
                    pixel_size_in   = tmpLayer.D/tmpLayer.resolution
                    pixel_size_out  = pixel_size_in/ratio
                    resolution_out  = self.telescope.resolution

                    phase_support[i]+= np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out)).T* np.sqrt(self.fractionalR0[i_layer])

                else:
                    phase_support[i]+= np.reshape(tmpLayer.phase[np.where(tmpLayer.pupil_footprint[i]==1)],[self.telescope.resolution,self.telescope.resolution])* np.sqrt(self.fractionalR0[i_layer])
        return phase_support
    
    def set_OPD(self,phase_support):
        if self.asterism is None:
            self.OPD_no_pupil   = phase_support*self.wavelength/2/np.pi
            self.OPD            = self.OPD_no_pupil*self.telescope.pupil
        else:
            self.OPD =[]
            self.OPD_no_pupil = []
            for i in range(self.asterism.n_source):
                self.OPD_no_pupil.append(phase_support[i]*self.wavelength/2/np.pi)
                self.OPD.append(self.OPD_no_pupil[i]*self.telescope.pupil)
        return
    def get_covariance_matrices(self,layer):
        # Compute the covariance matrices
        compute_covariance_matrices = True

        if self.fov_rad == 0: 
            try:
                c=time.time()        
                self.ZZt_r0 = self.ZZt_r0
                d=time.time()
                print('ZZt.. : ' +str(d-c) +' s')
                self.ZXt_r0 = self.ZXt_r0
                e=time.time()
                print('ZXt.. : ' +str(e-d) +' s')
                self.XXt_r0 = self.XXt_r0
                f=time.time()
                print('XXt.. : ' +str(f-e) +' s')
                self.ZZt_inv_r0 = self.ZZt_inv_r0
    
                print('SCAO system considered: covariance matrices were already computed!')
                compute_covariance_matrices = False
            except:                        
                compute_covariance_matrices = True
        if compute_covariance_matrices:
            c=time.time()        
            self.ZZt = makeCovarianceMatrix(layer.innerZ,layer.innerZ,self)
            
            if self.param is None:
                self.ZZt_inv = np.linalg.pinv(self.ZZt)
            else:
                try:
                    print('Loading pre-computed data...')            
                    name_data       = 'ZZt_inv_spider_L0_'+str(self.L0)+'_m_r0_'+str(self.r0_def)+'_shape_'+str(self.ZZt.shape[0])+'x'+str(self.ZZt.shape[1])+'.json'
                    location_data   = self.param['pathInput'] + self.param['name'] + '/sk_v/'
                    try:
                        with open(location_data+name_data ) as f:
                            C = json.load(f)
                        data_loaded = jsonpickle.decode(C)               
                    except:
                        createFolder(location_data)
                        with open(location_data+name_data ) as f:
                            C = json.load(f)
                        data_loaded = jsonpickle.decode(C)                    
                    self.ZZt_inv = data_loaded['ZZt_inv']
                    
                except: 
                    print('Something went wrong.. re-computing ZZt_inv ...')
                    name_data       = 'ZZt_inv_spider_L0_'+str(self.L0)+'_m_r0_'+str(self.r0_def)+'_shape_'+str(self.ZZt.shape[0])+'x'+str(self.ZZt.shape[1])+'.json'
                    location_data   = self.param['pathInput'] + self.param['name'] + '/sk_v/'
                    createFolder(location_data)
                    
                    self.ZZt_inv = np.linalg.pinv(self.ZZt)
                
                    print('saving for future...')
                    data = dict()
                    data['pupil'] = self.telescope.pupil
                    data['ZZt_inv'] = self.ZZt_inv
                            
                    data_encoded  = jsonpickle.encode(data)
                    with open(location_data+name_data, 'w') as f:
                        json.dump(data_encoded, f)
            d=time.time()
            print('ZZt.. : ' +str(d-c) +' s')
            self.ZXt = makeCovarianceMatrix(layer.innerZ,layer.outerZ,self)
            e=time.time()
            print('ZXt.. : ' +str(e-d) +' s')
            self.XXt = makeCovarianceMatrix(layer.outerZ,layer.outerZ,self)
            f=time.time()
            print('XXt.. : ' +str(f-e) +' s')
            
            self.ZZt_r0     = self.ZZt*(self.r0_def/self.r0)**(5/3)
            self.ZXt_r0     = self.ZXt*(self.r0_def/self.r0)**(5/3)
            self.XXt_r0     = self.XXt*(self.r0_def/self.r0)**(5/3)
            self.ZZt_inv_r0 = self.ZZt_inv/((self.r0_def/self.r0)**(5/3))
        return self.ZZt, self.ZXt, self.XXt, self.ZZt_inv
        
    def generateNewPhaseScreen(self,seed = None):
        if seed is None:
            t = time.localtime()
            seed = t.tm_hour*3600 + t.tm_min*60 + t.tm_sec
        phase_support = self.initialize_phase_support()
        for i_layer in range(self.nLayer):
            tmpLayer=getattr(self,'layer_'+str(i_layer+1))
            
            if self.mode ==1:
                import aotools as ao
                phaseScreen   = ao.turbulence.infinitephasescreen.PhaseScreenVonKarman(tmpLayer.resolution,tmpLayer.D/(tmpLayer.resolution),self.r0,self.L0,random_seed=seed+i_layer)
                phase         = phaseScreen.scrn
            else:
                if self.mode == 2:
                    phase = ft_sh_phase_screen(self,tmpLayer.resolution,tmpLayer.D/tmpLayer.resolution,seed=seed+i_layer)
                else: 
                    phase         = ft_phase_screen(self,tmpLayer.resolution,tmpLayer.D/tmpLayer.resolution,seed=seed+i_layer)
            
            tmpLayer.phase = phase
            tmpLayer.randomState    = RandomState(seed+i_layer*1000)
            
            Z = tmpLayer.phase[tmpLayer.innerMask[1:-1,1:-1]!=0]
            X = np.matmul(tmpLayer.A,Z) + np.matmul(tmpLayer.B,tmpLayer.randomState.normal( size=tmpLayer.B.shape[1]))
            
            tmpLayer.mapShift[tmpLayer.outerMask!=0] = X
            tmpLayer.mapShift[tmpLayer.outerMask==0] = np.reshape(tmpLayer.phase,tmpLayer.resolution*tmpLayer.resolution)
            tmpLayer.notDoneOnce = True

            setattr(self,'layer_'+str(i_layer+1),tmpLayer )
            phase_support = self.fill_phase_support(tmpLayer,phase_support,i_layer)
        self.set_OPD(phase_support)
        if self.telescope.isPaired:
            self*self.telescope


    def print_atm_at_wavelength(self,wavelength):

        r0_wvl              = self.r0*((wavelength/self.wavelength)**(5/3))
        seeingArcsec_wvl    = 206265*(wavelength/r0_wvl)

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE AT '+str(wavelength)+' nm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('r0 \t\t'+str(r0_wvl) + ' \t [m]') 
        print('Seeing \t' + str(np.round(seeingArcsec_wvl,2)) + str('\t ["]'))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return
    
    def print_atm(self):
             # Keep this ancient function name for compatibility
             self.print_properties()
             
       
    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^12s}'.format('Layer') + '{: ^12s}'.format('Direction')+ '{: ^12s}'.format('Speed')+ '{: ^12s}'.format('Altitude')+ '{: ^12s}'.format('Cn2')+ '{: ^12s}'.format('Diameter') )
        print('{: ^12s}'.format('') + '{: ^12s}'.format('[deg]')+ '{: ^12s}'.format('[m/s]')+ '{: ^12s}'.format('[m]')+ '{: ^12s}'.format('[m-2/3]') + '{: ^12s}'.format('[m]'))

        print('======================================================================')
        
        for i_layer in range(self.nLayer):
            print('{: ^12s}'.format(str(i_layer+1)) + '{: ^12s}'.format(str(self.windDirection[i_layer]))+ '{: ^12s}'.format(str(self.windSpeed[i_layer]))+ '{: ^12s}'.format(str(self.altitude[i_layer]))+ '{: ^12s}'.format(str(self.fractionalR0[i_layer]) ) + '{: ^12s}'.format(str(getattr(self,'layer_'+str(i_layer+1)).D )))
            if i_layer<self.nLayer-1:
                print('----------------------------------------------------------------------')

        print('======================================================================')


        print('{: ^18s}'.format('r0 @500 nm') + '{: ^18s}'.format(str(self.r0)+' [m]' ))
        print('{: ^18s}'.format('L0') + '{: ^18s}'.format(str(self.L0)+' [m]' ))
        print('{: ^18s}'.format('Seeing @500nm') + '{: ^18s}'.format(str(np.round(self.seeingArcsec,2))+' ["]'))
        print('{: ^18s}'.format('Frequency') + '{: ^18s}'.format(str(np.round(1/self.telescope.samplingTime,2))+' [Hz]' ))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
    def __mul__(self,obj):
        if obj.tag == 'telescope' or obj.tag == 'source':
            if obj.tag == 'telescope':
                if self.fov == obj.fov:   
                    self.telescope = obj
                else:          
                    print('Re-initializing the atmosphere to match the new telescope fov')
                    self.hasNotBeenInitialized = True
                    self.initializeAtmosphere(obj)

            else:
                if obj.coordinates[0] <= self.fov/2:
                    self.telescope.src = obj
                    obj = self.telescope
                else:
                    raise ValueError('The source object zenith ('+str(obj.coordinates[0])+'") is outside of the telescope fov ('+str(self.fov//2)+'")! You can:\n - Reduce the zenith of the source \n - Re-initialize the atmosphere object using a telescope with a larger fov')
            
            if self.user_defined_opd is False:
                self.set_pupil_footprint()    
                phase_support = self.initialize_phase_support()
                for i_layer in range(self.nLayer):
                    tmpLayer = getattr(self,'layer_'+str(i_layer+1))
                    # self.updateLayer(tmpLayer,shift = [tmpLayer.extra_sx,tmpLayer.extra_sy])
                    phase_support = self.fill_phase_support(tmpLayer, phase_support, i_layer)
                self.set_OPD(phase_support)
            
            if obj.src.tag == 'source':
                obj.optical_path =[[obj.src.type + '('+obj.src.optBand+')',id(obj.src)]]
            else:
                obj.optical_path =[[obj.src.type,id(obj.src)]]
                
            obj.optical_path.append([self.tag,id(self)])
            obj.optical_path.append([obj.tag,id(obj)])
            obj.OPD          = self.OPD.copy()
            obj.OPD_no_pupil = self.OPD_no_pupil.copy()
            obj.isPaired     = True
            return obj
        else:
            raise AttributeError('The atmosphere can be multiplied only with a Telescope or a Source object!')
    
    
    def display_atm_layers(self,layer_index= None,fig_index = None,list_src = None):
        display_cn2 = False
        
        if layer_index is None:
            layer_index = list(np.arange(self.nLayer))
            n_sp        = len(layer_index) 
            display_cn2 = True
            
        if type(layer_index) is not list:
            raise TypeError(' layer_index should be a list') 
        normalized_speed = np.asarray(self.windSpeed)/max(self.windSpeed)

        if self.telescope.src.tag =='asterism':
            list_src = self.telescope.src.src
        
        plt.figure(fig_index,figsize = [n_sp*4,3*(1+display_cn2)], edgecolor = None)
        if display_cn2:
            gs = gridspec.GridSpec(1,n_sp+1, height_ratios=[1], width_ratios=np.ones(n_sp+1), hspace=0.5, wspace=0.5)
        else:
            gs = gridspec.GridSpec(1,n_sp, height_ratios=np.ones(1), width_ratios=np.ones(n_sp), hspace=0.25, wspace=0.25)
            
        axis_list = []
        for i in range(len(layer_index)):
            axis_list.append(plt.subplot(gs[0,i]))  
                      
        if display_cn2:
            # axCn2 = f.add_subplot(gs[1, :])
            ax = plt.subplot(gs[0,-1])         
            plt.imshow(np.tile(np.asarray(self.fractionalR0)[:,None],self.nLayer),origin='lower',interpolation='gaussian',extent=[0,1,self.altitude[0],self.altitude[-1]+5000],cmap='jet'),plt.clim([0,np.max(self.fractionalR0)])
            
                
            for i_layer in range(self.nLayer):
                plt.text(0.5,self.altitude[i_layer],str(self.fractionalR0[i_layer]*100)+'%',color = 'w',fontweight='bold')
            plt.ylabel('Altitude [m]')
            plt.title('Cn2 Profile')
            ax.set_xticks([])
            makeSquareAxes(plt.gca())
            
        for i_l,ax in enumerate(axis_list):
            
            tmpLayer = getattr(self, 'layer_'+str(layer_index[i_l]+1))
            ax.imshow(tmpLayer.phase,extent = [-tmpLayer.D/2,tmpLayer.D/2,-tmpLayer.D/2,tmpLayer.D/2])
            center = tmpLayer.D/2
            [x_tel,y_tel] = pol2cart(tmpLayer.D_fov/2, np.linspace(0,2*np.pi,100,endpoint=True))  
            if list_src is not None:
                cm = plt.get_cmap('gist_rainbow')
                col = []
                for i_source in range(len(list_src)):                   
                    col.append(cm(1.*i_source/len(list_src))) 
                    
                    [x_c,y_c] = pol2cart(self.telescope.D/2, np.linspace(0,2*np.pi,100,endpoint=True))
                    alpha_cone = np.arctan(self.telescope.D/2/list_src[i_source].altitude)
                    
                    h = list_src[i_source].altitude-tmpLayer.altitude
                    if np.isinf(h):
                        r =self.telescope.D/2
                    else:
                        r = h*np.tan(alpha_cone)
                    [x_cone,y_cone] = pol2cart(r, np.linspace(0,2*np.pi,100,endpoint=True))
                    if list_src[i_source].chromatic_shift is not None:
                        if len(list_src[i_source].chromatic_shift) == self.nLayer:
                            chromatic_shift = list_src[i_source].chromatic_shift[i_l]
                        else:
                            raise ValueError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')                            
                    else:
                        chromatic_shift = 0
                    
                    [x_z,y_z] = pol2cart(tmpLayer.altitude*np.tan((list_src[i_source].coordinates[0]+chromatic_shift)/206265) ,np.deg2rad(list_src[i_source].coordinates[1]))
                    center = 0
                    [x_c,y_c] = pol2cart(tmpLayer.D_fov/2, np.linspace(0,2*np.pi,100,endpoint=True))  
                    nm = (list_src[i_source].type) +'@'+str(list_src[i_source].coordinates[0])+'"'
    
                    ax.plot(x_cone+x_z+center,y_cone+y_z+center,'-', color = col [i_source],label=nm)        
                    ax.fill(x_cone+x_z+center,y_cone+y_z+center,y_z+center, alpha = 0.25, color = col[i_source])
            else:
                [x_c,y_c] = pol2cart(self.telescope.D/2, np.linspace(0,2*np.pi,100,endpoint=True))
                alpha_cone = np.arctan(self.telescope.D/2/self.telescope.src.altitude)
                
                h = self.telescope.src.altitude-tmpLayer.altitude
                if np.isinf(h):
                    r =self.telescope.D/2
                else:
                    r = h*np.tan(alpha_cone)
                [x_cone,y_cone] = pol2cart(r, np.linspace(0,2*np.pi,100,endpoint=True))
                
                if self.telescope.src.chromatic_shift is not None:
                    if len(self.telescope.src.chromatic_shift) == self.nLayer:
                        chromatic_shift = self.telescope.src.chromatic_shift[i_l]
                    else:
                        raise ValueError('The chromatic_shift property is expected to be the same length as the number of atmospheric layer. ')        
                else:
                    chromatic_shift = 0
                    
                    
                [x_z,y_z] = pol2cart((self.telescope.src.coordinates[0]+chromatic_shift)*np.tan((self.telescope.src.coordinates[0]+chromatic_shift)/206265) * tmpLayer.resolution / tmpLayer.D,np.deg2rad(self.telescope.src.coordinates[1]))
            
                center = 0
                [x_c,y_c] = pol2cart(tmpLayer.D_fov/2, np.linspace(0,2*np.pi,100,endpoint=True))  
            
                ax.plot(x_cone+x_z+center,y_cone+y_z+center,'-')        
                ax.fill(x_cone+x_z+center,y_cone+y_z+center,y_z+center, alpha = 0.6)     
            
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.set_title('Altitude '+str(tmpLayer.altitude)+' m')
            ax.plot(x_tel+center,y_tel+center,'--',color = 'k')
            ax.legend(loc ='upper left')

            makeSquareAxes(plt.gca())

            ax.arrow(center, center, center+normalized_speed[i_l]*(tmpLayer.D_fov/2)*np.cos(np.deg2rad(tmpLayer.direction)),center+normalized_speed[i_l]*(tmpLayer.D_fov/2)*np.sin(np.deg2rad(tmpLayer.direction)),length_includes_head=True,width=0.25, facecolor = [0,0,0],alpha=0.3,edgecolor= None)
            ax.text(center+tmpLayer.D_fov/8*np.cos(np.deg2rad(tmpLayer.direction)), center+tmpLayer.D_fov/8*np.sin(np.deg2rad(tmpLayer.direction)),str(self.windSpeed[i_l])+' m/s', fontweight=100,color=[1,1,1],fontsize = 18)
 # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    
    @property
    def r0(self):
         return self._r0
    
    @r0.setter
    def r0(self,val):
         self._r0 = val

         if self.hasNotBeenInitialized is False:
             print('Updating the Atmosphere covariance matrices...')             
             self.seeingArcsec           = 206265*(self.wavelength/val)             
             for i_layer in range(self.nLayer):
                 tmpLayer = getattr(self,'layer_'+str(i_layer+1))

                 tmpLayer.ZZt_r0        = tmpLayer.ZZt*(self.r0_def/self.r0)**(5/3)
                 tmpLayer.ZXt_r0        = tmpLayer.ZXt*(self.r0_def/self.r0)**(5/3)
                 tmpLayer.XXt_r0        = tmpLayer.XXt*(self.r0_def/self.r0)**(5/3)
                 tmpLayer.ZZt_inv_r0    = tmpLayer.ZZt_inv/((self.r0_def/self.r0)**(5/3))
                 BBt                    = tmpLayer.XXt_r0 -  np.matmul(tmpLayer.A,tmpLayer.ZXt_r0)
                 tmpLayer.B             = np.linalg.cholesky(BBt)

    @property
    def L0(self):
         return self._L0
    
    @L0.setter
    def L0(self,val):
         self._L0 = val
         if self.hasNotBeenInitialized is False:
             print('Updating the Atmosphere covariance matrices...')

             self.hasNotBeenInitialized = True
             del self.ZZt
             del self.XXt
             del self.ZXt
             del self.ZZt_inv
             self.initializeAtmosphere(self.telescope)
    @property
    def windSpeed(self):
         return self._windSpeed
    
    @windSpeed.setter
    def windSpeed(self,val):
        self._windSpeed = val

        if self.hasNotBeenInitialized is False:
            if len(val)!= self.nLayer:
                print('Error! Wrong value for the wind-speed! Make sure that you inpute a wind-speed for each layer')
            else:
                print('Updating the wind speed...')
                for i_layer in range(self.nLayer):
                    tmpLayer = getattr(self,'layer_'+str(i_layer+1))
                    tmpLayer.windSpeed = val[i_layer]
                    tmpLayer.vY            = tmpLayer.windSpeed*np.cos(np.deg2rad(tmpLayer.direction))                    
                    tmpLayer.vX            = tmpLayer.windSpeed*np.sin(np.deg2rad(tmpLayer.direction))
                    ps_turb_x = tmpLayer.vX*self.telescope.samplingTime
                    ps_turb_y = tmpLayer.vY*self.telescope.samplingTime
                    tmpLayer.ratio[0] = ps_turb_x/self.ps_loop
                    tmpLayer.ratio[1] = ps_turb_y/self.ps_loop
                    setattr(self,'layer_'+str(i_layer+1),tmpLayer )
                # self.print_atm()
                
    @property
    def windDirection(self):
         return self._windDirection
    
    @windDirection.setter
    def windDirection(self,val):
        self._windDirection = val

        if self.hasNotBeenInitialized is False:
            if len(val)!= self.nLayer:
                print('Error! Wrong value for the wind-speed! Make sure that you inpute a wind-speed for each layer')
            else:
                print('Updating the wind direction...')
                for i_layer in range(self.nLayer):
                    tmpLayer = getattr(self,'layer_'+str(i_layer+1))
                    # tmpLayer.notDoneOnce = True
                    tmpLayer.direction = val[i_layer]
                    tmpLayer.vY            = tmpLayer.windSpeed*np.cos(np.deg2rad(tmpLayer.direction))                    
                    tmpLayer.vX            = tmpLayer.windSpeed*np.sin(np.deg2rad(tmpLayer.direction))
                    ps_turb_x = tmpLayer.vX*self.telescope.samplingTime
                    ps_turb_y = tmpLayer.vY*self.telescope.samplingTime
                    tmpLayer.ratio[0] = ps_turb_x/self.ps_loop
                    tmpLayer.ratio[1] = ps_turb_y/self.ps_loop
                    setattr(self,'layer_'+str(i_layer+1),tmpLayer )
                # self.print_atm()


                          
             
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
                            
class Layer:
    pass


        
        
            
            
            
            
    
        