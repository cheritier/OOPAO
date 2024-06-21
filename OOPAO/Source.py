# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:32:15 2020

@author: cheritie
"""
import inspect

import numpy as np


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

class Source:    
    def __init__(self,
                 optBand:str,
                 magnitude:float,
                 coordinates:list = [0,0],
                 altitude:float = np.inf, 
                 laser_coordinates:list = [0,0],
                 Na_profile:float = None,
                 FWHM_spot_up:float = None,
                 display_properties:bool=True,
                 chromatic_shift:list = None):
        """SOURCE
        A source object can be defined as a point source at infinite distance (NGS) or as a extended object

        Parameters
        ----------
        optBand : str
            The optical band of the source (see the method photometry)
            ex, 'V' corresponds to a wavelength of 500 nm
.
        magnitude : float
            The magnitude of the star.
        coordinates : list, optional
            DESCRIPTION. The default is [0,0].
        altitude : float, optional
            DESCRIPTION. The default is np.inf.
        laser_coordinates : list, optional
            DESCRIPTION. The default is [0,0].
        Na_profile : float, optional
            DESCRIPTION. The default is None.
        FWHM_spot_up : float, optional
            DESCRIPTION. The default is None.
        display_properties : bool, optional
            DESCRIPTION. The default is True.
        chromatic_shift : list, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        """
        ************************** REQUIRED PARAMETERS **************************
        
        A Source object is characterised by two parameter:
        _ optBand               : the optical band of the source (see the method photometry)
        _ magnitude             : The magnitude of the star
                            
        ************************** COUPLING A SOURCE OBJECT **************************
        
        Once generated, a Source object "src" can be coupled to a Telescope "tel" that contains the OPD.
        _ This is achieved using the * operator     : src*tel
        _ It can be accessed using                  : tel.src       

    
        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Source object are listed here: 
        _ src.phase     : 2D map of the phase scaled to the src wavelength corresponding to tel.OPD
        _ src.type      : Ngs or LGS  

        _ src.nPhoton   : number of photons per m2 per s. if this property is changed after the initialization, the magnitude is automatically updated to the right value. 
        _ src.fluxMap   : 2D map of the number of photons per pixel per frame (depends on the loop frequency defined by tel.samplingTime)  
        _ src.display_properties : display the properties of the src object
        _ src.chromatic_shift : list of shift in arcesc to be applied to the pupil footprint at each layer of the atmosphere object. 
        
        The main properties of the object can be displayed using :
            src.print_properties()
            
        ************************** OPTIONAL PROPERTIES **************************
        _ altitude              : altitude of the source. Default is inf (NGS) 
        _ laser_coordinates     : The coordinates in [m] of the laser launch telescope
        _ Na_profile            : An array of 2 dimensions and n sampling points for the Sodium profile. The first dimension corresponds to the altitude and the second dimention to the sodium profile value.
        _ FWHM_spot_up          : FWHM of the LGS spot in [arcsec]
        ************************** EXEMPLE **************************

        Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(opticalBand = 'H', magnitude = 8) 
        src*tel

        
        """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        self.is_initialized = False
        self.display_properties = display_properties
        tmp             = self.photometry(optBand)              # get the photometry properties
        self.optBand    = optBand                               # optical band
        self.wavelength = tmp[0]                                # wavelength in m
        self.bandwidth  = tmp[1]                                # optical bandwidth
        self.zeroPoint  = tmp[2]/368                            # zero point
        self.magnitude  = magnitude                             # magnitude
        self.phase      = []                                    # phase of the source 
        self.phase_no_pupil      = []                           # phase of the source (no pupil)
        self.fluxMap    = []                                    # 2D flux map of the source
        self.nPhoton    = self.zeroPoint*10**(-0.4*magnitude)   # number of photon per m2 per s
        self.tag        = 'source'                              # tag of the object
        self.altitude = altitude                                # altitude of the source object in m    
        self.coordinates = coordinates                          # polar coordinates [r,theta] 
        self.laser_coordinates = laser_coordinates              # Laser Launch Telescope coordinates in [m] 
        self.chromatic_shift = chromatic_shift                             # shift in arcsec to be applied to the atmospheric phase screens (one value for each layer) to simulate a chromatic effect
        if Na_profile is not None and FWHM_spot_up is not None:
            self.Na_profile = Na_profile
            self.FWHM_spot_up = FWHM_spot_up
            
            # consider the altitude weigthed by Na profile
            self.altitude = np.sum(Na_profile[0,:]*Na_profile[1,:])
            self.type     = 'LGS'
        else:
            
            self.type     = 'NGS'

        if self.display_properties:
            self.print_properties()
            
        self.is_initialized = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE INTERACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
    def __mul__(self,telescope):
        telescope.src   = self
        if type(telescope.OPD) is list:
            telescope.resetOPD()
        
        if np.ndim(telescope.OPD) ==3:
            telescope.resetOPD()

        telescope.OPD = telescope.OPD*telescope.pupil # here to ensure that a new pupil is taken into account

        # update the phase of the source
        self.phase      = telescope.OPD*2*np.pi/self.wavelength
        self.phase_no_pupil      = telescope.OPD_no_pupil*2*np.pi/self.wavelength

        # compute the variance in the pupil
        self.var        = np.var(self.phase[np.where(telescope.pupil==1)])
        # assign the source object to the telescope object

        self.fluxMap    = telescope.pupilReflectivity*self.nPhoton*telescope.samplingTime*(telescope.D/telescope.resolution)**2
        if telescope.optical_path is None:
            telescope.optical_path = []
            telescope.optical_path.append([self.type + '('+self.optBand+')',id(self)])
            telescope.optical_path.append([telescope.tag,id(telescope)])
        else:
            telescope.optical_path[0] =[self.type + '('+self.optBand+')',id(self)]
            
        return telescope
     
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE PHOTOMETRY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def photometry(self,arg):
        # photometry object [wavelength, bandwidth, zeroPoint]
        class phot:
            pass
        
        ## New entries with corrected zero point flux values 
        phot.U      = [ 0.360e-6 , 0.070e-6 , 1.96e12 ]
        phot.B      = [ 0.440e-6 , 0.100e-6 , 5.38e12 ]
        phot.V0     = [ 0.500e-6 , 0.090e-6 , 3.64e12 ]
        phot.V      = [ 0.550e-6 , 0.090e-6 , 3.31e12 ]
        phot.R      = [ 0.640e-6 , 0.150e-6 , 4.01e12 ]
        phot.R2     = [ 0.650e-6 , 0.300e-6 , 7.9e12  ] 
        phot.R3     = [ 0.600e-6 , 0.300e-6 , 8.56e12 ] # Fixed (entries with big difference)
        phot.R4     = [ 0.670e-6 , 0.300e-6 , 7.66e12 ] # Fixed (entries with big difference)
        phot.I      = [ 0.790e-6 , 0.150e-6 , 2.69e12 ]
        phot.I1     = [ 0.700e-6 , 0.033e-6 , 0.67e12 ] # Fixed (entries with big difference)
        phot.I2     = [ 0.750e-6 , 0.033e-6 , 0.62e12 ] # Fixed (entries with big difference)
        phot.I3     = [ 0.800e-6 , 0.033e-6 , 0.58e12 ] # Fixed (entries with big difference)
        phot.I4     = [ 0.700e-6 , 0.100e-6 , 2.02e12 ] # Fixed (entries with big difference)
        phot.I5     = [ 0.850e-6 , 0.100e-6 , 1.67e12 ] # Fixed (entries with big difference)
        phot.I6     = [ 1.000e-6 , 0.100e-6 , 1.42e12 ] # Fixed (entries with big difference)
        phot.I7     = [ 0.850e-6 , 0.300e-6 , 5.00e12 ] # Fixed (entries with big difference)
        phot.I8     = [ 0.750e-6 , 0.100e-6 , 1.89e12 ] # Fixed (entries with big difference)
        phot.I9     = [ 0.850e-6 , 0.300e-6 , 5.00e12 ] # Fixed (entries with big difference)
        phot.I10    = [ 0.900e-6 , 0.300e-6 , 4.72e12 ] # Fixed (entries with big difference)
        phot.J      = [ 1.215e-6 , 0.260e-6 , 1.90e12 ]
        phot.J2     = [ 1.550e-6 , 0.260e-6 , 1.49e12 ] # Fixed (entries with big difference)
        phot.H      = [ 1.654e-6 , 0.290e-6 , 1.05e12 ]
        phot.Kp     = [ 2.1245e-6 , 0.351e-6 , 0.62e12 ]
        phot.Ks     = [ 2.157e-6 , 0.320e-6 , 0.55e12 ]
        phot.K      = [ 2.179e-6 , 0.410e-6 , 0.70e12 ]
        phot.K0     = [ 2.000e-6 , 0.410e-6 , 0.76e12 ]
        phot.K1     = [ 2.400e-6 , 0.410e-6 , 0.64e12 ]
        '''
        #  Old entries
        phot.U      = [ 0.360e-6 , 0.070e-6 , 2.0e12 ]
        phot.B      = [ 0.440e-6 , 0.100e-6 , 5.4e12 ]
        phot.V0     = [ 0.500e-6 , 0.090e-6 , 3.3e12 ]
        phot.V      = [ 0.550e-6 , 0.090e-6 , 3.3e12 ]
        phot.R      = [ 0.640e-6 , 0.150e-6 , 4.0e12 ]
        phot.I      = [ 0.790e-6 , 0.150e-6 , 2.7e12 ]
        phot.I1     = [ 0.700e-6 , 0.033e-6 , 2.7e12 ]
        phot.I2     = [ 0.750e-6 , 0.033e-6 , 2.7e12 ]
        phot.I3     = [ 0.800e-6 , 0.033e-6 , 2.7e12 ]
        phot.I4     = [ 0.700e-6 , 0.100e-6 , 2.7e12 ]
        phot.I5     = [ 0.850e-6 , 0.100e-6 , 2.7e12 ]
        phot.I6     = [ 1.000e-6 , 0.100e-6 , 2.7e12 ]
        phot.I7     = [ 0.850e-6 , 0.300e-6 , 2.7e12 ]
        phot.R2     = [ 0.650e-6 , 0.300e-6 , 7.92e12 ]
        phot.R3     = [ 0.600e-6 , 0.300e-6 , 7.92e12 ]
        phot.R4     = [ 0.670e-6 , 0.300e-6 , 7.92e12 ]
        phot.I8     = [ 0.750e-6 , 0.100e-6 , 2.7e12 ]
        phot.I9     = [ 0.850e-6 , 0.300e-6 , 7.36e12 ]
        phot.I10    = [ 0.900e-6 , 0.300e-6 , 2.7e12 ]
        phot.J      = [ 1.215e-6 , 0.260e-6 , 1.9e12 ]
        phot.J2     = [ 1.550e-6 , 0.260e-6 , 1.9e12 ]
        phot.H      = [ 1.654e-6 , 0.290e-6 , 1.1e12 ]
        phot.Kp     = [ 2.1245e-6 , 0.351e-6 , 6e11 ]
        phot.Ks     = [ 2.157e-6 , 0.320e-6 , 5.5e11 ]
        phot.K      = [ 2.179e-6 , 0.410e-6 , 7.0e11 ]
        phot.K0     = [ 2.000e-6  , 0.410e-6 , 7.0e11 ]
        phot.K1     = [ 2.400e-6 , 0.410e-6 , 7.0e11 ]
        phot.IR1310 = [ 1.310e-6 , 0        , 2e12 ]   # bandwidth is zero?
        '''
        phot.L      = [ 3.547e-6 , 0.570e-6 , 2.5e11 ]
        phot.M      = [ 4.769e-6 , 0.450e-6 , 8.4e10 ]
        phot.Na     = [ 0.589e-6 , 0        , 3.3e12 ]  # bandwidth is zero?
        phot.EOS    = [ 1.064e-6 , 0        , 3.3e12 ]  # bandwidth is zero?
        phot.IR1310 = [ 1.310e-6 , 0        , 2e12 ]   # bandwidth is zero?
        
        if isinstance(arg,str):
            if hasattr(phot,arg):
                return getattr(phot,arg)
            else:
                print('Error: Wrong name for the photometry object')
                return -1
        else:
            print('Error: The photometry object takes a scalar as an input')
            return -1   
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    @property
    def nPhoton(self):
        return self._nPhoton
    
    @nPhoton.setter
    def nPhoton(self,val):
        self._nPhoton  = val
        self.magnitude = -2.5*np.log10(val/self.zeroPoint)
        if self.is_initialized:

            print('NGS flux updated!')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Wavelength \t'+str(round(self.wavelength*1e6,3)) + ' \t [microns]') 
            print('Optical Band \t'+self.optBand) 
            print('Magnitude \t' + str(self.magnitude))
            print('Flux \t\t'+ str(np.round(self.nPhoton)) + str('\t [photons/m2/s]'))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
#    @property
#    def magnitude(self):
#        return self._magnitude
#    
#    @magnitude.setter
#    def magnitude(self,val):
#        self._magnitude  = val
#        self.nPhoton     = self.zeroPoint*10**(-0.4*val)
#            
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
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^8s}'.format('Source') +'{: ^10s}'.format('Wavelength')+ '{: ^8s}'.format('Zenith')+ '{: ^10s}'.format('Azimuth')+ '{: ^10s}'.format('Altitude')+ '{: ^10s}'.format('Magnitude') + '{: ^10s}'.format('Flux') )
        print('{: ^8s}'.format('') +'{: ^10s}'.format('[m]')+ '{: ^8s}'.format('[arcsec]')+ '{: ^10s}'.format('[deg]')+ '{: ^10s}'.format('[m]')+ '{: ^10s}'.format('') + '{: ^10s}'.format('[phot/m2/s]') )

        print('-------------------------------------------------------------------')        
        print('{: ^8s}'.format(self.type) +'{: ^10s}'.format(str(self.wavelength))+ '{: ^8s}'.format(str(self.coordinates[0]))+ '{: ^10s}'.format(str(self.coordinates[1]))+'{: ^10s}'.format(str(np.round(self.altitude,2)))+ '{: ^10s}'.format(str(self.magnitude))+'{: ^10s}'.format(str(np.round(self.nPhoton,1))) )
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    def __repr__(self):
        self.print_properties()
        return ' '