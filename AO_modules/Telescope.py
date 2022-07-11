# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:23:18 2020

@author: cheritie
"""
import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne
import inspect

from AO_modules.Source import Source

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class Telescope:
    
    def __init__(self,resolution, diameter,samplingTime=0.001,centralObstruction = 0,fov = 0,pupil=None,pupilReflectivity=1):
        """
        ************************** REQUIRED PARAMETERS **************************
        
        A Telescope object consists in defining the 2D mask of the entrance pupil. It is mainly characterised by two parameters 
        _ resolution            : the resolution of the pupil mask
        _ diameter              : The physical diameter of the telescope in [m]
        
        If no pupil mask is input, the default pupil geometry is circular.
        
        ************************** OPTIONAL PARAMETERS **************************
        
        _ samplingTime          : Defines the frequency of the AO loop. It is used in the Atmosphere object to update the turbulence phase screens according to the wind speed. 
        _ centralObstruction    : Adds a central obstruction in percentage of diameter. 
        _ fov                   : Defines the Field of View of the Telescope object. This will be useful for off-axis targets but it hasn't been properly implemented yet.
        _ pupil                 : A user-defined pupil mask can be input to the Telescope object. It should consist of a binary array. 
        _ pupilReflectivcty     : Defines the reflectivity of the Telescope object. If not set to 1, it can be input as a 2D map of uneven reflectivy correspondong to the pupil mask. 
                                  This property can be set after the initialization of the Telescope object.
            
        ************************** COUPLING A SOURCE OBJECT **************************
        
        Once generated, the telescope should be coupled with a Source object "src" that contains the wavelength and flux properties of a target. 
        _ This is achieved using the * operator     : src*tel
        _ It can be accessed using                  : tel.src
        _ By default, a Source object in the visible with a magnitude 0 is coupled to the telescope object
        
        ************************** COUPLING WITH AN ATMOSPHERE OBJECT **************************
        
        The telescope can be coupled to an Atmosphere object. In that case, the OPD of the atmosphere is automatically added to the telescope object.
        _ Coupling an Atmosphere and telescope Object   : tel+atm
        _ Separating an Atmosphere and telescope Object : tel-atm
        
        ************************** COMPUTING THE PSF **************************
        
        1) PSF computation
        tel.computePSF(zeroPaddingFactor)  : computes the square module of the Fourier transform of the tel.src.phase using the zeropadding factor for the FFT
        
    
        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ tel.OPD       : the optical path difference
        _ tel.src.phase : 2D map of the phase scaled to the src wavelength corresponding to tel.OPD
        _ tel.PSF       : Point Spread Function corresponding to to the tel.src.phase. This is not automatically set, it requires to run tel.computePSF().
        
                 
        ************************** EXEMPLE **************************
        
        1) Create an 8-m diameter circular telescope with a central obstruction of 15% and the pupil sampled with 100 pixels along the diameter. 
        tel = Telescope(resolution = 100, diameter = 8, centralObstruction = 0.15)
                
        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8) 
        src*tel
        
        3) Compute the PSF with a zero-padding factor of 2.
        tel.computePSF(zeroPaddingFactor = 2)
        
        """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
        self.resolution                  = resolution                # Resolution of the telescope
        self.D                           = diameter                  # Diameter in m
        self.pixelSize                   = self.D/self.resolution    # size of the pixels in m
        self.centralObstruction          = centralObstruction        # central obstruction
        self.fov                         = fov                       # Field of View 
        self.samplingTime                = samplingTime              # AO loop speed
        self.isPetalFree                 = False                     # Flag to remove the petalling effect with ane ELT system. 
        self.index_pixel_petals          = None                      # indexes of the pixels corresponfong to the M1 petals. They need to be set externally
#        Case where the pupil is not input: circular pupil with central obstruction    
        if pupil is None:
            D           = self.resolution+1
            x           = np.linspace(-self.resolution/2,self.resolution/2,self.resolution)
            xx,yy       = np.meshgrid(x,x)
            circle      = xx**2+yy**2
            obs         = circle>=(self.centralObstruction*D/2)**2
            self.pupil  = circle<(D/2)**2 
            self.pupil  = self.pupil*obs
        else:
            print('User-defined pupil, the central obstruction will not be taken into account...')
            self.pupil  = pupil        
            
        self.pupilReflectivity           = self.pupil.astype(float)*pupilReflectivity                   # A non uniform reflectivity can be input by the user
        self.pixelArea                   = np.sum(self.pupil)                                           # Total number of pixels in the pupil area
        self.pupilLogical                = np.where(np.reshape(self.pupil,resolution*resolution)>0)     # index of valid pixels in the pupil
        self.src                         = Source(optBand = 'V', magnitude = 0)                                                # temporary source object associated to the telescope object
        self.OPD                         = self.pupil.astype(float)                                     # set the initial OPD
        self.OPD_no_pupil                = 1+self.pupil.astype(float)*0                                     # set the initial OPD
        self.em_field                    = self.pupilReflectivity*np.exp(1j*self.src.phase)
        self.tag                         = 'telescope'                                                  # tag of the object
        self.isPaired                    = False                                                        # indicate if telescope object is paired with an atmosphere object
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Diameter \t\t\t'+str(self.D) + ' \t [m]') 
        print('Resolution \t\t\t'+str(self.resolution) + ' \t [pix]') 
        print('Pixel Size \t\t\t' + str(np.round(self.pixelSize,2)) + str('\t [m]'))
        print('Surface \t\t\t'+ str(np.round(self.pixelArea*self.pixelSize**2)) + str('\t [m2]'))
        print('Central Obstruction \t\t'+str(100*self.centralObstruction)+str('\t [% of diameter]'))
        print('Number of pixel in the pupil \t'+str(self.pixelArea)+' \t [pix]') 
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PSF COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def computePSF(self,zeroPaddingFactor):
        if hasattr(self,'src'): 
            # number of pixel considered 
            N       = int(zeroPaddingFactor * self.resolution)        
            center = N//2           
            norma   = N
            
            # zeroPadded support for the FFT
            supportPadded = np.zeros([N,N],dtype='complex')
            supportPadded [center-self.resolution//2:center+self.resolution//2,center-self.resolution//2:center+self.resolution//2] = self.pupil*self.pupilReflectivity*np.sqrt(self.src.fluxMap)*np.exp(1j*self.src.phase)
            # phasor to center the FFT on the 4 central pixels
            [xx,yy]                         = np.meshgrid(np.linspace(0,N-1,N),np.linspace(0,N-1,N))
            self.phasor                     = np.exp(-(1j*np.pi*(N+1)/N)*(xx+yy))
            
            # axis in arcsec
            self.xPSF_arcsec       = [-206265*(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N),206265*(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N)]
            self.yPSF_arcsec       = [-206265*(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N),206265*(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N)]
            
            # axis in radians
            self.xPSF_rad   = [-(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N),(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N)]
            self.yPSF_rad   = [-(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N),(np.fix(N/2))*(self.src.wavelength/self.D) * (self.resolution/N)]
            
            # PSF computation
            self.PSF        = (np.abs(np.fft.fft2(supportPadded*self.phasor)/norma)**2)
            # PSF normalization
            # self.PSF  = self.PSF
            
            # zoom on the core of the PSF 
            self.indPSF     = [N//2-self.resolution//2-1,N//2+self.resolution//2-1]
            self.PSF_trunc  = self.PSF[self.indPSF[0]:self.indPSF[-1],self.indPSF[0]:self.indPSF[-1]]
            self.xPSF_trunc = [-206265*(np.fix(max(self.indPSF)/2))*(self.src.wavelength/self.D) * (self.resolution/N),206265*(np.fix(max(self.indPSF)/2))*(self.src.wavelength/self.D) * (self.resolution/N)]
            self.yPSF_trunc = [-206265*(np.fix(max(self.indPSF)/2))*(self.src.wavelength/self.D) * (self.resolution/N),206265*(np.fix(max(self.indPSF)/2))*(self.src.wavelength/self.D) * (self.resolution/N)]
        else:
            print('Error: no NGS associated to the Telescope. Combine a tel object with an ngs using ngs*tel')
            return -1
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PSF DISPLAY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def showPSF(self,zoom = 1):
        # display the full PSF or zoom on the core of the PSF
        if hasattr(self, 'PSF'): 
            print('Displaying the PSF...')
        else:
            self.computePSF(6)
            print('Displaying the PSF...')

        if zoom:
            plt.imshow(self.PSF_trunc,extent = [self.xPSF_trunc[0],self.xPSF_trunc[1],self.xPSF_trunc[0],self.xPSF_trunc[1]])
        else:
            plt.imshow(self.PSF,extent = [self.xPSF[0],self.xPSF[1],self.xPSF[0],self.xPSF[1]])
        plt.xlabel('[arcsec]')
        plt.ylabel('[arcsec]')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    @property
    def pupil(self):
        return self._pupil
    
    @pupil.setter
    def pupil(self,val):
        self._pupil = val.astype(int)
        self.pixelArea = np.sum(self._pupil)
        tmp = np.reshape(self._pupil,self.resolution**2)
        self.pupilLogical = np.where(tmp>0)
    
    @property        
    def OPD(self):
        return self._OPD
    
    @OPD.setter
    def OPD(self,val):
        self._OPD = val
        self.src.phase = self._OPD*2*np.pi/self.src.wavelength
        
    @property        
    def OPD_no_pupil(self):
        return self._OPD_no_pupil
    
    @OPD_no_pupil.setter
    def OPD_no_pupil(self,val):
        self._OPD_no_pupil = val
        self.src.phase_no_pupil = self._OPD_no_pupil*2*np.pi/self.src.wavelength
        
        # if np.ndim(self.src.phase)==2: 
        #     self.em_field  = self.pupilReflectivity*np.exp(1j*self.src.phase)
        # else:
        #     self.em_field  = np.tile(self.pupilReflectivity[...,None],(1,1,self.src.phase.shape[2]))*np.exp(1j*self.src.phase)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE INTERACTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def __mul__(self,obj): 
        # interaction with pyramid object: Propagation of the phase screen
        if obj.tag=='pyramid':
            obj.telescope=self              # assign the telescope object to the pyramid-telescope object
            obj.pyramid_propagation(self)    # propagation of the telescope-source phase screen to the pyramid-detector
        
        if  obj.tag == 'double_wfs':
            obj.telescope = self              # assign the telescope object to the pyramid-telescope object
            obj.wfs_measure(self)    # propagation of the telescope-source phase screen to the pyramid-detector

        if obj.tag=='shackHartmann':
            obj.telescope=self              # assign the telescope object to the pyramid-telescope object
            obj.sh_measure()

        # interaction with detector: computation of the PSF
        if obj.tag=='detector':
            self.computePSF()
            obj.frame = obj.rebin(self.PSF,(obj.resolution,obj.resolution))
            
        if obj.tag=='spatialFilter':
            N = obj.resolution
            EF_in = np.zeros([N,N],dtype='complex')
            
            EF_in [obj.center-self.resolution//2:obj.center+self.resolution//2,obj.center-self.resolution//2:obj.center+self.resolution//2] = self.pupil*self.pupilReflectivity*np.sqrt(self.src.fluxMap)* np.exp(1j*self.src.phase)
            
            FP_in = np.fft.fft2(EF_in)/N
            
            FP_filtered = FP_in*obj.mask
            
            em_field = np.fft.ifft2(FP_filtered)
            self.em_field = em_field[obj.center-self.resolution//2:obj.center+self.resolution//2,obj.center-self.resolution//2:obj.center+self.resolution//2]
            return self
        
        # interaction with deformable mirror object: update of the of the phase screen
        if obj.tag=='deformableMirror':
            # local variable to use the numexpr module
            
            # case where the telescope is paired to an atmosphere
            if self.isPaired:
                if self.isPetalFree:
                    self.removePetalling()       
                telOPD  = self.OPD
                dmOPD   = obj.OPD
                telPupil  = self.pupil                      
                if np.ndim(obj.OPD)==2:                    
                    self.OPD = ne.evaluate('telOPD+dmOPD*telPupil')
                    self.OPD_no_pupil = self.OPD_no_pupil+obj.OPD

                else:
                    self.OPD = np.tile(self.OPD[...,None],(1,1,obj.OPD.shape[2]))+obj.OPD*np.tile(self.pupil[...,None],(1,1,obj.OPD.shape[2]))
                    self.OPD_no_pupil = np.tile(self.OPD_no_pupil[...,None],(1,1,obj.OPD.shape[2])) + obj.OPD
 
            # case where the telescope is separated from a telescope object
            else:
                dmOPD =obj.OPD
                telPupil  = self.pupil  
                if np.ndim(obj.OPD)==2:
                    self.OPD = ne.evaluate('dmOPD*telPupil')
                else:
                    self.OPD = obj.OPD*np.tile(self.pupil[...,None],(1,1,obj.OPD.shape[2]))
                self.OPD_no_pupil = obj.OPD
        return self
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def resetOPD(self):
        # re-initialize the telescope OPD to a flat wavefront
        self.OPD = self.pupil.astype(float)
        self.OPD_no_pupil = 1+0*self.pupil.astype(float)
    
    def removePetalling(self,image = None):
        try:
            # remove the petalling from image if the index_pixel_petals have been computed externally
            if image is None:
                petalFreeOPD = np.copy(self.OPD)
                if self.index_pixel_petals is None:
                    print('ERROR : the indexes of the petals have not been set yet!')
                    print('Returning the current OPD..')
                    return self.OPD
                else:
                    try:
                        N = self.index_pixel_petals.shape[2]
                        # Case with N petals
                        for i in range(N):
                            meanValue = np.mean(petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals[:,:,i])==1)])
    #                        residualValue = meanValue % (self.src.wavelength )
                            residualValue = 0
                            petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals[:,:,i])==1)]    += residualValue- meanValue
                    except:
                        # Case with 1 petal
                        petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals)==1)]               -= np.mean(petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals)==1)])
                # update the OPD with the petal-free OPD
                self.OPD = petalFreeOPD
            else:
                petalFreeOPD = image
                try:
                    N = self.index_pixel_petals.shape[2]
                    for i in range(N):
                        petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals[:,:,i])==1)]-=np.mean(petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals[:,:,i])==1)])
                except:
                    petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals)==1)]-=np.mean(petalFreeOPD[np.where(np.squeeze(self.index_pixel_petals)==1)])
            return petalFreeOPD
        except:
            print('ERROR : the indexes of the petals have not been properly set yet!')
            return self.OPD
            
    def pad(self,resolution_padded):
        if np.ndim(self.OPD) == 2:
            em_field_padded = np.zeros([resolution_padded,resolution_padded],dtype = complex)
            OPD_padded = np.zeros([resolution_padded,resolution_padded],dtype = float)

            center = resolution_padded//2
            
            em_field_padded[center-self.resolution//2:center+self.resolution//2,center-self.resolution//2:center+self.resolution//2] =  self.em_field
            OPD_padded[center-self.resolution//2:center+self.resolution//2,center-self.resolution//2:center+self.resolution//2]      =  self.OPD
        else:
            em_field_padded = np.zeros([resolution_padded,resolution_padded, self.OPD.shape[2]],dtype = complex)
            OPD_padded = np.zeros([resolution_padded,resolution_padded,self.OPD.shape[2]],dtype = float)

            center = resolution_padded//2
            
            em_field_padded[center-self.resolution//2:center+self.resolution//2,center-self.resolution//2:center+self.resolution//2,:] =  self.em_field
            OPD_padded[center-self.resolution//2:center+self.resolution//2,center-self.resolution//2:center+self.resolution//2,:]      =  self.OPD
        
        
        return OPD_padded, em_field_padded
        
        
    
    def getPetalOPD(self,petalIndex,image = None):
        if image is None:
            petal_OPD = np.copy(self.OPD)
        else:
            petal_OPD = image.copy()
        if self.index_pixel_petals is None:
            print('ERROR : the indexes of the petals have not been set yet!')
            print('Returning the current OPD..')
            return self.OPD
        else:
             self.petalMask =  np.zeros(self.pupil.shape)
             try:
                 self.petalMask [np.where(np.squeeze(self.index_pixel_petals[:,:,petalIndex])==1)] = 1
             except:
                 self.petalMask [np.where(np.squeeze(self.index_pixel_petals)==1)] = 1
         
        petal_OPD = petal_OPD * self.petalMask
        return petal_OPD



    # Combining with an atmosphere object
    def __add__(self,atmObject):
        self.isPaired   = True
        self.OPD  = atmObject.OPD
        self.OPD_no_pupil  = atmObject.OPD_no_pupil

        if self.isPetalFree:
                    self.removePetalling()  
        print('Telescope and Atmosphere combined!')
        
    # Separating from an atmosphere object
    def __sub__(self,atmObject):
        self.isPaired   = False  
        self.OPD  = self.pupil.astype(float)
        self.OPD_no_pupil  =1+ self.pupil.astype(float)*0    

        print('Telescope and Atmosphere separated!')

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






