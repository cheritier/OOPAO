# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:03:22 2024

@author: astriffl
"""
import numpy as np

def papyrus2oopaoUnits(mode, papyrus_amp, M2C_papyrus, dm, tel):
    """
    
    Parameters
    ----------
    mode : int
        Index of the mode to measure
    papyrus_amp : float
        Amplitude of the mode to measure in units used on the PAPYRTC
    M2C_papyrus : array
        M2C used on the PAPYRTC
    dm : OOPAO object
        Deformable mirror with 16 Subap and 0.45 of mechanical coupling
    tel : OOPAO object
        Telescope of 1.52 m without central obstruction

    Returns
    -------
    oopao_nm_rms : float
        Amplitude of the OPD created by applying the mode on the DM [nm RMS]

    """
    alpao_unit = 7591.024876
    C2P = dm.modes * alpao_unit * 1e-9
    vect = np.zeros(M2C_papyrus.shape[1])
    vect[mode] = papyrus_amp
    OPD = C2P @ (M2C_papyrus @ vect)
    OPD = OPD.reshape(tel.resolution,tel.resolution)
    oopao_nm_rms = np.std(OPD[np.where(tel.pupil==1)])
    return oopao_nm_rms


def actuators_position(param):
    """
    
    Parameters
    ----------
    param : Dictionnary 
        Comes from the parameter file of the Papytwin.

    Returns
    -------
    coordinates : array
        Locations of actuators with respect to the pupil of the DM.
    pitch : float
        Interactuator distance.

    """
    nAct = param['nActuator']
    pitch = 2.5e-3 #m
    DM_diameter  = nAct * pitch # mm
    pupil_calib_diameter = 37.5e-3 # mm
    scale_factor = DM_diameter / pupil_calib_diameter
    x = np.linspace(-scale_factor * param['diameter']/2, scale_factor * param['diameter']/2, param['nActuator'])
    [X,Y] = np.meshgrid(x,x)
    coordinates = np.asarray([X.reshape(nAct**2),Y.reshape(nAct**2)]).T
    dist = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
    coordinates = coordinates[dist <= param['diameter']/2 + 2 *pitch * param['diameter'] / pupil_calib_diameter, :]
    pitch = pitch * param['diameter'] / pupil_calib_diameter

    return coordinates, pitch