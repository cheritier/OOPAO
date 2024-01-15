# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:22:10 2022

@author: cheritier
"""
import numpy as np
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class NCPA:
    
    def __init__(self,tel,dm,atm,modal_basis='KL',zernike_coefficients = None,f2=None,seed=5):
        """
        ************************** REQUIRED PARAMETERS **************************
        
        An NCPA object consists in defining the 2D map that acts as a static OPD offset. It requires the following parameters
        _ telescope              : the telescope object that contains all the informations (diameter, pupil, etc)
                
        ************************** OPTIONAL PARAMETERS **************************
        
        _ zernike_coefficients  : a list of coefficients of Zernike Polynomials (see Zernike class). The coefficients are normalized to 1 m. 
        _ f2                    : a list of 3 elements [amplitude, start mode, end mode] which will follow 1/f2 law
        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ NCPA.OPD       : the optical path difference

        ************************** EXEMPLE **************************
        
        1) Create a blank NCPA object corresponding to a Telescope object
        ncpa = NCPA(telescope = tel)
        
        2) Update the OPD of the NCPA object using a given OPD_map
        ncpa.OPD = OPD_map
        
        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8) 
        src*tel
        
        3) Create an NCPA object corresponding based on a linear combinaison of Zernike coefficients 
        list_coefficients = [0,0,10e-9,20e-9]
        ncpa = NCPA(telescope = tel, zernike_coefficients = list_coefficients)
        
        """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        self.isInitialized = False                # Resolution of the telescope
        self.basis = modal_basis
        self.tel   = tel
        self.atm   = atm
        self.dm    = dm
        self.seed  = seed 
        if f2 is None:
            if zernike_coefficients is None:
                self.OPD = self.pupil.astype(float)
            
            if zernike_coefficients is not None:
                if type(zernike_coefficients) is list:
                    n_max = len(zernike_coefficients)
                    from OOPAO.Zernike import Zernike
                    self.Z = Zernike(self.tel,J=n_max)
                    self.Z.computeZernike(self.tel)
                    
                else:
                    raise TypeError('The zernike coefficients should be input as a list.')        
                    
            self.OPD = np.matmul(self.Z.modesFullRes,np.asarray(zernike_coefficients))
            self.tag = 'NCPA'
        
        else:
            self.NCPA_f2_law(self.tel,f2)
            self.tag = 'NCPA'
            
    def NCPA_f2_law(self,telescope,f2):
        if type(f2) is list and len(f2)==4:
            if self.basis=='KL':
                from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
                M2C_KL = compute_KL_basis(self.tel, self.atm, self.dm,lim=1e-2)
                self.dm.coefs = M2C_KL
                self.tel*self.dm
                basis = self.tel.OPD#.reshape((self.tel.resolution**2,M2C_KL.shape[1]))
                phase = np.sum([np.random.RandomState(i*self.seed).randn()/np.sqrt(i+f2[3])*basis[:,:,i] for i in range(f2[1],f2[2])],axis=0)
                self.OPD = phase / np.std(phase[np.where(telescope.pupil==1)]) * f2[0]
                
            if self.basis=='Zernike':
                from OOPAO.Zernike import Zernike
                self.Z = Zernike(telescope,J=f2[2])
                self.Z.computeZernike(telescope)          
                phase = np.sum([np.random.RandomState(i*self.seed).randn()/np.sqrt(i+f2[3])*self.Z.modesFullRes[:,:,i] for i in range(f2[1],f2[2])],axis=0)
                self.OPD = phase / np.std(phase[np.where(telescope.pupil==1)]) * f2[0]
        else:
            raise TypeError('f2 should be a list containing [amplitude, start_mode, end_mode, cutoff]')
        pass
        
        