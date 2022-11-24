# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:15:06 2020

@author: cheritie
"""

import numpy as np
import matplotlib.pyplot as plt


def getFittingError(OPD, proj, basis, display = True):
#    compute projector
    phi_turb = np.reshape(OPD,OPD.shape[0]*OPD.shape[1])
    phi_corr = np.matmul(basis, np.matmul(proj,phi_turb))
    
    OPD_fitting_2D = np.reshape(phi_turb-phi_corr,[OPD.shape[0],OPD.shape[1]])
    OPD_corr_2D = np.reshape(phi_corr,[OPD.shape[0],OPD.shape[1]])
    OPD_turb_2D = np.reshape(phi_turb,[OPD.shape[0],OPD.shape[1]])
    
    if display:
        
        plt.figure()
        plt.subplot(131)
        plt.imshow(1e9*OPD_fitting_2D)
        plt.title('Fitting Error OPD [nm]')

        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(1e9*OPD_turb_2D)
        plt.colorbar()
        plt.title('Turbulence OPD [nm]')
        
        plt.subplot(133)
        plt.imshow(1e9*OPD_corr_2D)
        plt.colorbar()
        plt.title('DM OPD [nm]')
    
        plt.show()

    return OPD_fitting_2D,OPD_corr_2D,OPD_turb_2D

def getFittingError_dm(OPD, proj, tel,dm, M2C, display = True):
    tel.resetOPD()
    #    compute projector
    OPD_turb = np.reshape(OPD,OPD.shape[0]*OPD.shape[1])
    coefs_dm =M2C@np.matmul(proj,OPD_turb)
    dm.coefs = -coefs_dm
    tel.isPaired = True
    tel.OPD = OPD
    
    tel*dm
    
    OPD_fitting_2D = tel.OPD
    OPD_corr_2D = dm.OPD*tel.pupil
    OPD_turb_2D = np.reshape(OPD_turb,[OPD.shape[0],OPD.shape[1]])
    
    if display:
        
        plt.figure()
        plt.subplot(131)
        plt.imshow(1e9*OPD_fitting_2D)
        plt.title('Fitting Error OPD [nm]')

        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(1e9*OPD_turb_2D)
        plt.colorbar()
        plt.title('Turbulence OPD [nm]')
        
        plt.subplot(133)
        plt.imshow(1e9*OPD_corr_2D)
        plt.colorbar()
        plt.title('DM OPD [nm]')
    
        plt.show()

    return OPD_fitting_2D,OPD_corr_2D,OPD_turb_2D, coefs_dm