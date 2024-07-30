# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:23:18 2020

@author: cheritie
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy import signal as csg

    global_gpu_flag = True

except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np

from OOPAO.tools.tools import set_binning

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Telescope:
    
    def __init__(self,resolution:float,
                 diameter:float,
                 samplingTime:float=0.001,
                 centralObstruction:float = 0,
                 fov:float = 0,
                 pupil:bool=None,
                 pupilReflectivity:float=1,
                 display_optical_path:bool = False):
        """TELESCOPE
        A Telescope object consists in defining the 2D mask of the entrance pupil.
        The Telescope is a central object in OOPAO:
            A source object is associated to the Telescope that carries the flux and wavelength information. 
            An Atmosphere object can be paired to the Telescope to propagate the light through turbulent phase screens.
            The Telescope is required to initialize many of the OOPAO classes as it carries the pupil definition and pixel size. 

        Parameters
        ----------
        resolution : float
            The resolution of the pupil mask.
        diameter : float
            The physical diameter of the telescope in [m].
        samplingTime : float, optional
            Defines the frequency of the AO loop. It is used in the Atmosphere object 
            to update the turbulence phase screens according to the wind speed.
            The default is 0.001.
        centralObstruction : float, optional
            Adds a central obstruction in percentage of diameter.
            The default is 0.
        fov : float, optional
            Defines the Field of View of the Telescope object. 
            This is useful for off-axis targets but it hasn't been properly implemented yet.
            The default is 0.
        pupil : bool, optional
            A user-defined pupil mask can be input to the Telescope object. It should consist of a binary array. 
            The default is None.
        pupilReflectivity : float, optional
            Defines the reflectivity of the Telescope object. 
            If not set to 1, it can be input as a 2D map of uneven reflectivy correspondong to the pupil mask.
            The default is 1.
        display_optical_path : bool, optional
            If desired, the optical path can be printed at each time the light is propagated to a WFS object 
            setting the display_optical_path property to True.
            The default is False.

        Returns
        -------
        None.
   
        ************************** ADDING SPIDERS *******************************
        It is possible to add spiders to the telescope pupil using the following property: 
            
            tel.apply_spiders(angle,thickness_spider,offset_X = None, offset_Y=None)
    
        where : 
            - angle is a list of angle in [degrees]. The length of angle defines the number of spider. 
            - thickness is the width of the spider in [m]
            - offset_X is a list (same lenght as angle) of shift X to apply to the individual spider  in [m]
            - offset_Y is a list (same lenght as angle) of shift Y to apply to the individual spider  in [m]
   
        ************************** PRINTING THE OPTICAL PATH *******************************
        It is possible to print the current optical path to verify through which object the light went through using the print_optical_path method:
                
                tel.print_optical_path()
        
        If desired, the optical path can be printed at each time the light is propagated to a WFS object setting the display_optical_path property to True:
            tel.display_optical_path = True
        
        
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
        self.isInitialized               = False                # Resolution of the telescope
        
        self.resolution                  = resolution                # Resolution of the telescope
        self.D                           = diameter                  # Diameter in m
        self.pixelSize                   = self.D/self.resolution    # size of the pixels in m
        self.centralObstruction          = centralObstruction        # central obstruction
        self.fov                         = fov      # Field of View in arcsec converted in radian
        self.fov_rad                     = fov/206265      # Field of View in arcsec converted in radian
        self.samplingTime                = samplingTime              # AO loop speed
        self.isPetalFree                 = False                     # Flag to remove the petalling effect with ane ELT system. 
        self.index_pixel_petals          = None                      # indexes of the pixels corresponfong to the M1 petals. They need to be set externally
        self.optical_path                = None                      # indexes of the pixels corresponfong to the M1 petals. They need to be set externally
        self.user_defined_pupil          = pupil
        self.pupilReflectivity           = pupilReflectivity
        self.set_pupil()
        self.src                         = None                                               # temporary source object associated to the telescope object
        self.OPD                         = self.pupil.astype(float)                                     # set the initial OPD
        self.OPD_no_pupil                = 1+self.pupil.astype(float)*0                                     # set the initial OPD
        # self.em_field                    = self.pupilReflectivity*np.exp(1j*self.src.phase)
        self.tag                         = 'telescope'                                                  # tag of the object
        self.isPaired                    = False                                                        # indicate if telescope object is paired with an atmosphere object
        self.spatialFilter               = None
        self.display_optical_path        = display_optical_path
        self.print_properties()

        self.isInitialized= True
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PSF COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def set_pupil(self):
    #        Case where the pupil is not input: circular pupil with central obstruction    
        if self.user_defined_pupil is None:
            D           = self.resolution+1
            x           = np.linspace(-self.resolution/2,self.resolution/2,self.resolution)
            xx,yy       = np.meshgrid(x,x)
            circle      = xx**2+yy**2
            obs         = circle>=(self.centralObstruction*D/2)**2
            self.pupil  = circle<(D/2)**2 
            self.pupil  = self.pupil*obs
        else:
            print('User-defined pupil, the central obstruction will not be taken into account...')
            self.pupil  = self.user_defined_pupil.copy()        
            
        self.pupilReflectivity           = self.pupil.astype(float)*self.pupilReflectivity                   # A non uniform reflectivity can be input by the user
        self.pixelArea                   = np.sum(self.pupil)                                           # Total number of pixels in the pupil area
        self.pupilLogical                = np.where(np.reshape(self.pupil,self.resolution*self.resolution)>0)     # index of valid pixels in the pupil
    def computeCoronoPSF(self,zeroPaddingFactor=2, display = False, coronagraphDiameter = 4.5):

        # coronagraphDiameter is the FPM diameter in L/D of imaging wavelength

        if self.src is None:
            raise AttributeError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')            # number of pixel considered 
        N       = int(zeroPaddingFactor * self.resolution)        
        center  = N//2           

        [xx,yy] = np.meshgrid(np.linspace(0,N-1,N),np.linspace(0,N-1,N))
        xxc = xx - (N-1)/2
        yyc = yy - (N-1)/2

        self.pupilPadded = np.sqrt(xxc**2 + yyc**2) < self.resolution/2
        self.pupilSpiderPadded = np.zeros((N,N))
        self.pupilSpiderPadded[center-self.resolution//2:center+self.resolution//2,center-self.resolution//2:center+self.resolution//2] = self.pupil
        self.focalMask = np.sqrt(xxc**2 + yyc**2) > coronagraphDiameter/2 * zeroPaddingFactor
        self.apodizer  = self.pupilPadded
        self.lyotStop  = (np.sqrt((xxc-1.0)**2 + (yyc-1.0)**2) < self.resolution/2 * 0.9) * self.pupilSpiderPadded
        self.diffraction2meterGEO = self.src.wavelength/self.D * 36e6 # assumes GEO orbit 36 000 km

        phase = self.src.phase
        amp_mask = 1

        # axis limits in meters at GEO orbit
        self.xPSF_mGEO         = [-self.resolution /2 * self.diffraction2meterGEO, self.resolution/2 * self.diffraction2meterGEO]
        self.yPSF_mGEO         = [-self.resolution /2 * self.diffraction2meterGEO, self.resolution/2 * self.diffraction2meterGEO]

        # axis in arcsec => BUG ? Assumes Shannon sampling only ?
        self.xPSF_arcsec       = [-206265*(self.src.wavelength/self.D) * (self.resolution/2), 206265*(self.src.wavelength/self.D) * (self.resolution/2)]
        self.yPSF_arcsec       = [-206265*(self.src.wavelength/self.D) * (self.resolution/2), 206265*(self.src.wavelength/self.D) * (self.resolution/2)]
        
        # axis in radians
        self.xPSF_rad   = [-(self.src.wavelength/self.D) * (self.resolution/2),(self.src.wavelength/self.D) * (self.resolution/2)]
        self.yPSF_rad   = [-(self.src.wavelength/self.D) * (self.resolution/2),(self.src.wavelength/self.D) * (self.resolution/2)]
        
        # zero-padded support for electric field
        supportPadded = np.zeros([N,N],dtype='complex')
        supportPadded [center-self.resolution//2:center+self.resolution//2,center-self.resolution//2:center+self.resolution//2] = amp_mask*self.pupil*self.pupilReflectivity*np.sqrt(self.src.fluxMap)*np.exp(1j*phase)
        self.phasor                     = np.exp(-(1j*np.pi*(N+1)/N)*(xx+yy))
        

        # Fields computation in A B C D planes
        A = self.phasor * supportPadded * self.apodizer
        B = np.fft.fft2(A) * self.focalMask
        C = np.fft.fft2(B) * self.lyotStop
        D = np.fft.fft2(C)

        self.PSFc        = (np.abs(D)**2) / N**6            
        self.PSFc_norma  = self.PSFc/self.PSFc.max()   
        N_trunc = int(np.floor(2*N/6))
        self.PSFc_norma_zoom  = self.PSFc_norma[N_trunc:-N_trunc,N_trunc:-N_trunc]

        if display is True:
            #subplot(r,c) provide the no. of rows and columns
            fig1 = plt.figure(num=1, figsize=(16, 6), dpi=80)
            fig1.suptitle('Pupil & FP electric fields', fontsize=16)
            axarr = fig1.subplots(1,4) 
            axarr[0].imshow(np.log(np.abs(A)))
            axarr[0].title.set_text('Entrance pupil plane A')
            axarr[1].imshow(np.log(np.abs(B)))
            axarr[1].title.set_text('After focal plane B')
            axarr[2].imshow(np.log(np.abs(C)))
            axarr[2].title.set_text('After Lyot stop plane C')
            axarr[3].imshow(np.log(np.abs(D)))
            axarr[3].title.set_text('Detector plane D')
    
            #subplot(r,c) provide the no. of rows and columns
            fig2 = plt.figure(num=2, figsize=(12, 6), dpi=80)
            fig2.suptitle('Pupil & FP masks')
            axarr = fig2.subplots(1,3) 
            axarr[0].imshow(self.apodizer)
            axarr[1].imshow(self.focalMask)
            axarr[2].imshow(self.lyotStop)
            

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PSF COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


    def computePSF(self,zeroPaddingFactor=2,detector = None,img_resolution=None):
        # kept for backward compatibility
        conversion_constant = (180/np.pi)*3600
        if detector is not None:
            zeroPaddingFactor = detector.psf_sampling
            img_resolution    = detector.resolution
        if img_resolution is None:
            img_resolution = zeroPaddingFactor*self.resolution
        if self.src is None:
            raise AttributeError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')   
        
        if self.spatialFilter is None:            
            amp_mask = 1
            phase    = self.src.phase

        else:
            amp_mask = self.amplitude_filtered               
            phase    = self.phase_filtered

        amp      = amp_mask*self.pupil*self.pupilReflectivity*np.sqrt(self.src.fluxMap)
        
        # function to compute the em-field and PSF        
        self.PropagateField(amplitude = amp , phase = phase, zeroPaddingFactor = zeroPaddingFactor,img_resolution=img_resolution)

        # axis in arcsec
        self.xPSF_arcsec       = [-conversion_constant*(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor), conversion_constant*(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor)]
        self.yPSF_arcsec       = [-conversion_constant*(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor), conversion_constant*(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor)]
        
        # axis in radians
        self.xPSF_rad   = [-(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor),(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor)]
        self.yPSF_rad   = [-(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor),(self.src.wavelength/self.D) * (img_resolution/2/zeroPaddingFactor)]
        
        # normalized PSF           
        self.PSF_norma  = self.PSF/self.PSF.max()  
 
    
    def PropagateField(self, amplitude, phase, zeroPaddingFactor, img_resolution = None):

        xp                  = np
        oversampling        = 1
        resolution          = self.pupil.shape[0]

        if oversampling is not None: oversampling = oversampling

        if img_resolution is not None:
            if img_resolution > zeroPaddingFactor * resolution:
                raise ValueError('Error: image has too many pixels for this pupil sampling. Try using a pupil mask with more pixels')
        else:
                img_resolution = zeroPaddingFactor * resolution

        # If PSF is undersampled apply the integer oversampling
        if zeroPaddingFactor * oversampling < 2:
            oversampling = (np.ceil(2.0 / zeroPaddingFactor)).astype('int')

        # This is to ensure that PSF will be binned properly if number of pixels is odd
        if img_resolution is not None:
            if oversampling % 2 != img_resolution % 2:
                oversampling += 1

        img_size = np.ceil(img_resolution * oversampling).astype('int')
        N = np.fix(zeroPaddingFactor * oversampling * resolution).astype('int')
        pad_width = np.ceil((N - resolution) / 2).astype('int')

        supportPadded = xp.pad(amplitude * xp.exp(1j * phase),
                               pad_width=((pad_width, pad_width), (pad_width, pad_width)), constant_values=0)
        N = supportPadded.shape[0]  # make sure the number of pxels is correct after the padding
    
        # PSF computation
        [xx, yy] = xp.meshgrid(xp.linspace(0, N - 1, N), xp.linspace(0, N - 1, N), copy=False)
        phasor = xp.exp(-1j * xp.pi / N * (xx + yy) * (1 - img_resolution % 2)).astype(xp.complex64)
        #                                                        ^--- this is to account odd/even number of pixels
        # Propagate with Fourier shifting
        EMF = xp.fft.fftshift(1 / N * xp.fft.fft2(xp.fft.ifftshift(supportPadded * phasor)))
    
        # Again, this is to properly crop a PSF with the odd/even number of pixels
        if N % 2 == img_size % 2:
            shift_pix = 0
        else:
            if N % 2 == 0:
                shift_pix = 1
            else:
                shift_pix = -1

        # self.em_field_padded = EMF

        # Support only rectangular PSFs
        ids = xp.array(
            [np.ceil(N / 2) - img_size // 2 + (1 - N % 2) - 1, np.ceil(N / 2) + img_size // 2 + shift_pix]).astype(
            xp.int32)
        EMF = EMF[ids[0]:ids[1], ids[0]:ids[1]]

        self.focal_EMF = EMF
    
        if oversampling !=1:
            self.PSF = set_binning(xp.abs(EMF) ** 2, oversampling)
            # self.PSF = xp.abs(EMF) ** 2

        else:
            self.PSF = xp.abs(EMF) ** 2
                        
        return oversampling
                       
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PSF DISPLAY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def showPSF(self,zoom = 1, GEO = False):
        # display the full PSF or zoom on the core of the PSF
        if hasattr(self, 'PSF'): 
            print('Displaying the PSF...')
        else:
            self.computePSF(6)
            print('Displaying the PSF...')
        if zoom:
            plt.imshow(self.PSF_trunc,extent = [self.xPSF_trunc[0],self.xPSF_trunc[1],self.xPSF_trunc[0],self.xPSF_trunc[1]])
            plt.xlabel('[arcsec]')
            plt.ylabel('[arcsec]')
        elif GEO == True: # works only with zoom = False
            plt.imshow(np.log(self.PSFc),extent = [self.xPSF_mGEO[0],self.xPSF_mGEO[1],self.xPSF_mGEO[0],self.xPSF_mGEO[1]])
            plt.xlabel('[meters in GEO orbit]')
            plt.ylabel('[meters in GEO orbit]')
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
        self.pupilReflectivity = self.pupil.astype(float)
        if self.isInitialized:
            print('Warning!: A new pupil is now considered, its reflectivity is considered to be uniform. Assign the proper reflectivity map to tel.pupilReflectivity if required.')
    
    @property        
    def OPD(self):
        return self._OPD
    
    @OPD.setter
    def OPD(self,val):
        self._OPD = val
        if self.src is not None:
            if type(val) is not list:
                if self.src.tag == 'source':
                    self.src.phase = self._OPD*2*np.pi/self.src.wavelength
                    if np.ndim(self.OPD)==2:
                        self.mean_removed_OPD = (self.OPD - np.mean(self.OPD[np.where(self.pupil ==1)]))*self.pupil
                else:
                    if self.src.tag == 'asterism':
                        for i in range(self.src.n_source):
                            self.src.src[i].phase = self._OPD*2*np.pi/self.src.src[i].wavelength
                    else:
                        raise TypeError('The wrong object was attached to the telescope')                                      
            else:
                if self.src.tag == 'asterism':
                    if len(self._OPD)==self.src.n_source:
                        for i in range(self.src.n_source):
                            self.src.src[i].phase = self._OPD[i]*2*np.pi/self.src.src[i].wavelength
                    else:
                        raise TypeError('A list of OPD cannnot be propagated to a single source')
                    
                    
    @property        
    def OPD_no_pupil(self):
        return self._OPD_no_pupil 
    
    @OPD_no_pupil.setter
    def OPD_no_pupil(self,val):
        self._OPD_no_pupil = val
        if self.src is not None:

            if type(val) is not list:
                if self.src.tag == 'source':
                    self.src.phase_no_pupil = self._OPD_no_pupil*2*np.pi/self.src.wavelength
                else:
                    if self.src.tag == 'asterism':
                        for i in range(self.src.n_source):
                            self.src.src[i].phase_no_pupil = self._OPD_no_pupil*2*np.pi/self.src.src[i].wavelength
                    else:
                        raise TypeError('The wrong object was attached to the telescope')                                      
            else:
                if self.src.tag == 'asterism':
                    if len(self._OPD_no_pupil)==self.src.n_source:
                        for i in range(self.src.n_source):
                            self.src.src[i].phase_no_pupil = self._OPD_no_pupil[i]*2*np.pi/self.src.src[i].wavelength
                    else:
                        raise TypeError('The lenght of the OPD list ('+str(len(self._OPD_no_pupil))+') does not match the number of sources ('+str(self.src.n_source)+')')
                        

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE INTERACTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    
    def __mul__(self,obj): 
        # case where multiple objects are considered
        if type(obj) is list:
            if type(self.OPD) is list:
                if len(self.OPD) == len(obj):
                    for i_obj in range(len(self.OPD)):
                        tel_tmp = getattr(obj[i_obj], 'telescope')
                        self.src.src[i_obj]*tel_tmp
                        tel_tmp.OPD = self.OPD[i_obj]
                        tel_tmp.OPD_no_pupil = self.OPD_no_pupil[i_obj]
                        setattr(obj[i_obj],'telescope',tel_tmp)
                        obj[i_obj].wfs_measure(phase_in = tel_tmp.src.phase)               # propagation of the telescope-source phase screen to the pyramid-detector
                else:
                    raise ValueError('Error! There is a mis-match between the number of Sources ('+str(len(self.OPD))+') and the number of WFS ('+str(len(obj))+')')
            else:
                for i_obj in range(len(obj)):
                    self*obj[i_obj]
        else:    
              # interaction with WFS object: Propagation of the phase screen
            if obj.tag=='pyramid' or obj.tag == 'double_wfs' or obj.tag=='shackHartmann' or obj.tag == 'bioEdge':
                self.optical_path.append([obj.tag,id(obj)])
                if self.display_optical_path is True:
                    self.print_optical_path()
                self.optical_path = self.optical_path[:-1]
                if type(self.OPD) is list:
                    raise ValueError('Error! There is a mis-match between the number of Sources ('+str(len(self.OPD))+') and the number of WFS (1)')
                else:
                    obj.telescope=self              # assign the telescope object to the pyramid-telescope object
                    obj.wfs_measure(phase_in = self.src.phase)               # propagation of the telescope-source phase screen to the pyramid-detector

            if obj.tag=='detector':
                if self.optical_path[-1] != obj.tag: 
                    self.optical_path.append([obj.tag,id(obj)])

                self.computePSF(detector = obj)
                obj.fov_arcsec = self.xPSF_arcsec[1] -self.xPSF_arcsec[0] 
                obj.fov_rad = self.xPSF_rad[1] - self.xPSF_rad[0]
                if obj.integrationTime is not None:
                    if obj.integrationTime < self.samplingTime:
                        raise ValueError('The Detector integration time is smaller than the AO loop sampling Time. ')
                obj._integrated_time += self.samplingTime 
                obj.integrate(self.PSF)
                
                self.PSF = obj.frame
         
            if obj.tag=='OPD_map':
                self.optical_path.append([obj.tag,id(obj)])

                self.OPD += obj.OPD
                self.OPD_no_pupil += obj.OPD
                
            if obj.tag=='NCPA':
                self.optical_path.append([obj.tag,id(obj)])

                self.OPD += obj.OPD
                self.OPD_no_pupil += obj.OPD

                    
            if obj.tag=='spatialFilter':
                self.optical_path.append([obj.tag,id(obj)])


                self.spatialFilter  = obj
                N                   = obj.resolution
                EF_in               = np.zeros([N,N],dtype='complex')
                EF_in [obj.center-self.resolution//2:obj.center+self.resolution//2,obj.center-self.resolution//2:obj.center+self.resolution//2] =  self.pupilReflectivity*np.exp(1j*(self.OPD_no_pupil*2*np.pi/self.src.wavelength))
                FP_in               = np.fft.fft2(EF_in)
                FP_filtered         = FP_in*np.fft.fftshift(obj.mask)
                em_field            = np.fft.ifft2(FP_filtered)
                self.em_field_filtered  = em_field[obj.center-self.resolution//2:obj.center+self.resolution//2,obj.center-self.resolution//2:obj.center+self.resolution//2]
                # self.phase_filtered = np.arctan2(np.imag(self.em_field_filtered),np.real(self.em_field_filtered))*self.pupil
                self.phase_filtered = ((np.angle(self.em_field_filtered)))*self.pupil
                
                self.amplitude_filtered  = np.abs(self.em_field_filtered)
                return self
            
            if obj.tag=='deformableMirror':  
                if  self.optical_path[-1][1]!=id(obj):
                    self.optical_path.append([obj.tag,id(obj)])

                pupil = np.atleast_3d(self.pupil)
                
                if self.src.tag == 'source':
                    self.OPD_no_pupil = obj.dm_propagation(self)
                    if np.ndim(self.OPD_no_pupil) == 2:
                        self.OPD = self.OPD_no_pupil*self.pupil
                    else:
                        self.OPD =self.OPD_no_pupil*pupil 
                else:           
                     if self.src.tag == 'asterism':
                         if len(self.OPD) == self.src.n_source:
                             for i in range(self.src.n_source):
                                 if obj.altitude is not None:
                                     self.OPD_no_pupil[i]  = obj.dm_propagation(self,OPD_in = self.OPD_no_pupil[i], i_source = i  )
                                 else:
                                     self.OPD_no_pupil[i]  = obj.dm_propagation(self,OPD_in = self.OPD_no_pupil[i] )
                                 if np.ndim(self.OPD_no_pupil[i]) == 2:
                                     self.OPD[i] = self.OPD_no_pupil[i]*self.pupil
                                 else:
                                     self.OPD[i] =self.OPD_no_pupil[i]*pupil 
                             self.OPD = self.OPD
                             self.OPD_no_pupil = self.OPD_no_pupil
                                 
                         else:
                             raise TypeError('The lenght of the OPD list ('+str(len(self._OPD_no_pupil))+') does not match the number of sources ('+str(self.src.n_source)+')')
                     else:
                         raise TypeError('The wrong object was attached to the telescope')                                      
        return self
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE METHODS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def resetOPD(self):
        # re-initialize the telescope OPD to a flat wavefront
        if self.src is not None:

            if self.src.tag == 'asterism':
                self.optical_path = [[self.src.type, id(self.src)]]
                self.optical_path.append([self.tag,id(self)])
                self.OPD = [self.pupil.astype(float) for i in range(self.src.n_source)]
                self.OPD_no_pupil = [self.pupil.astype(float)*0 +1 for i in range(self.src.n_source)]
            else:
                self.optical_path = [[self.src.type + '('+self.src.optBand+')', id(self.src)]]
                self.optical_path.append([self.tag,id(self)])
                self.OPD = 0*self.pupil.astype(float)
                self.OPD_no_pupil = 0*self.pupil.astype(float)
                
    def print_optical_path(self):
        if self.optical_path is not None:
            tmp_path = ''
            for i in range(len(self.optical_path)):
                tmp_path += self.optical_path[i][0]
                if i <len(self.optical_path)-1:
                    tmp_path += ' ~~> '
            print(tmp_path)
        else:
            print('No light propagated through the telescope')      
                
    def apply_spiders(self,angle,thickness_spider,offset_X = None, offset_Y=None):
        self.isInitialized = False
        if thickness_spider >0:
            self.set_pupil()
            pup = np.copy(self.pupil)
            max_offset = self.centralObstruction*self.D/2 - thickness_spider/2
            if offset_X is None:
                offset_X = np.zeros(len(angle))
                
            if offset_Y is None:
                offset_Y = np.zeros(len(angle))
                        
            if np.max(np.abs(offset_X))>=max_offset or np.max(np.abs(offset_Y))>max_offset:
                print('WARNING ! The spider offsets are too large! Weird things could happen!')
            for i in range(len(angle)):
                angle_val = (angle[i]+90)%360
                x = np.linspace(-self.D/2,self.D/2,self.resolution)
                [X,Y] = np.meshgrid(x,x)
                X+=offset_X[i]
                Y+=offset_Y[i]
    
                map_dist = np.abs(X*np.cos(np.deg2rad(angle_val)) + Y*np.sin(np.deg2rad(-angle_val)))
        
                if 0<=angle_val<90:
                    map_dist[:self.resolution//2,:] = thickness_spider
                if 90<=angle_val<180:
                    map_dist[:,:self.resolution//2] = thickness_spider
                if 180<=angle_val<270:
                    map_dist[self.resolution//2:,:] = thickness_spider
                if 270<=angle_val<360:
                    map_dist[:,self.resolution//2:] = thickness_spider                
                pup*= map_dist>thickness_spider/2
            self.isInitialized = True

            self.pupil = pup.copy()
            
        else:
            print('Thickness is <=0, returning default pupil')
            self.set_pupil()

        return 
    
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
    def __add__(self,obj):
        if obj.tag == 'atmosphere':
            # obj.set_pupil_footprint()
            # self.optical_path =[[self.src.type + '('+self.src.optBand+')',id(self.src)]]
            # self.optical_path.append([obj.tag,id(obj)])
            # self.optical_path.append([self.tag,id(self)])
            # self.isPaired   = True
            obj*self

            # self.OPD  = obj.OPD.copy()
            # self.OPD_no_pupil  = obj.OPD_no_pupil.copy()

            if self.isPetalFree:
                    self.removePetalling()  
            print('Telescope and Atmosphere combined!')
        if obj.tag == 'spatialFilter':
            self.spatialFilter   = obj
            self*obj
            print('Telescope and Spatial Filter combined!')
            
        
    # Separating from an atmosphere object
    def __sub__(self,obj):
        if obj.tag == 'atmosphere':
            self.optical_path =[[self.src.type + '('+self.src.optBand+')',id(self.src)]]
            self.optical_path.append([self.tag,id(self)])
            self.isPaired   = False  
            self.resetOPD()   
            print('Telescope and Atmosphere separated!')
            
        if obj.tag == 'spatialFilter':
            self.spatialFilter   = None
            print('Telescope and Spatial Filter separated!')
            
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


    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^18s}'.format('Diameter')                     + '{: ^18s}'.format(str(self.D))                                        +'{: ^18s}'.format('[m]'   ))
        print('{: ^18s}'.format('Resolution')                   + '{: ^18s}'.format(str(self.resolution))                               +'{: ^18s}'.format('[pixels]'   ))
        print('{: ^18s}'.format('Pixel Size')                   + '{: ^18s}'.format(str(np.round(self.pixelSize,2)))                    +'{: ^18s}'.format('[m]'   ))
        print('{: ^18s}'.format('Surface')                      + '{: ^18s}'.format(str(np.round(self.pixelArea*self.pixelSize**2)))    +'{: ^18s}'.format('[m2]'  ))
        print('{: ^18s}'.format('Central Obstruction')          + '{: ^18s}'.format(str(100*self.centralObstruction))                   +'{: ^18s}'.format('[% of diameter]' ))
        print('{: ^18s}'.format('Pixels in the pupil')          + '{: ^18s}'.format(str(self.pixelArea))                                +'{: ^18s}'.format('[pixels]' ))
        print('{: ^18s}'.format('Field of View')                + '{: ^18s}'.format(str(self.fov))                                      +'{: ^18s}'.format('[arcsec]' ))
        if self.src:
            if self.src.type == 'asterism':
                for i_src in range(len(self.src.src)):
                    print('{: ^18s}'.format('Source '+self.src.src[i_src].type)            + '{: ^18s}'.format(str(np.round(1e9*self.src.src[i_src].wavelength,2)))            +'{: ^18s}'.format('[nm]' ))
            else:
                print('{: ^18s}'.format('Source '+self.src.type)            + '{: ^18s}'.format(str(np.round(1e9*self.src.wavelength,2)))            +'{: ^18s}'.format('[nm]' ))
        else:
            print('{: ^18s}'.format('Source')            + '{: ^18s}'.format('None')                                             +'{: ^18s}'.format('' ))

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        self.print_optical_path()
            


    def __repr__(self):
        self.print_properties()
        return ' '

