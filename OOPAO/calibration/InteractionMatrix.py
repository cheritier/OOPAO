# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:26:22 2020

@author: cheritie
"""
import numpy as np
import time
import tqdm
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.tools.tools import OopaoError
from OOPAO.OPD_map import OPD_map


def _as_opd_map(opd_offset):
    """
    Normalise the opd_offset input into an OPD_map object (or None if there is no offset).

    Accepted inputs:
        - None, or a scalar equal to 0 -> no offset (returns None)
        - a 2D array (same size as the telescope resolution), in meters -> wrapped in an OPD_map
        - an OPD_map object -> used as is
    """
    if opd_offset is None:
        return None
    # OPD_map object (checked through its tag to avoid depending on the exact class identity)
    if isinstance(opd_offset, OPD_map) or getattr(opd_offset, 'tag', None) == 'OPD_map':
        return opd_offset
    # scalar zero => no offset
    if np.ndim(opd_offset) == 0:
        if opd_offset == 0:
            return None
        raise OopaoError('opd_offset must be a 2D array or an OPD_map object (a non-zero scalar is not supported).')
    if np.ndim(opd_offset) == 2:
        return OPD_map(opd_offset)
    raise OopaoError('opd_offset must be a 2D array or an OPD_map object, got an array with ndim = '
                     + str(np.ndim(opd_offset)) + '.')


def InteractionMatrix(ngs,
                      tel,
                      dm,
                      wfs,
                      M2C,
                      stroke,
                      atm = None,
                      opd_offset = None,
                      nMeasurements = 50,
                      noise = 'off',
                      invert = True,
                      nTrunc = 0,
                      print_time = False,
                      display = False,
                      single_pass = True):

    """

    This function allows to build the Interaction Matrix (or its inverse if invert == True) of an AO system.
    IntMat is a 2D matrix with DM modes columns and wfs.nSignal lines. Each column of the IntMat is the measured signal
    by the wfs when applying a certain mode in the DM.

    :param ngs: Source object
    :param atm: Atmosphere object
    :param tel: Telescope object
    :param dm: Deformable mirror object
    :param wfs: Wavefront sensor object
    :param M2C: Mode to command matrix: can be either zonal or modal
    :param stroke: stroke for the push/pull in M2C units
    :param opd_offset: static OPD offset (in meters) considered when computing the Interaction matrix. Can be useful to take into
                       account possible non-common path aberrations (NCPAs) for instance. It can be:
                           - None (default) or 0: no offset
                           - a 2D array of the size of the telescope resolution
                           - an OPD_map object
                       The offset is added on top of the DM shape, right before the wfs.
    :param nMeasurements: number of simultaneous wfs measurements. Instead of building the interaction matrix column by column, we do it in chunks
                          of nMeasurements columns.
    :param noise: if noise = "off" (default), we do not keep the noise when doing the wfs measurements
    :param invert: boolean variable. If True, we invert the Interaction matrix. Otherwise, the output is the Interaction matrix itself
    :param print_time: boolean variable: if True, we print the elapsed time in seconds per iteration when building the IntMat. Default is False
    :param display: boolean variable: if True (default), we display time using tqdm module
    :param single_pass: boolean variable: if True (default), we only perform push. If False, we perform push and pull

    :return: InteractionMatrix (intMat) if invert == False or the inverse of the InteractionMatrix if invert == True

    """
    
    if display is False:
        def iterate(x):
            return x
    else:
        def iterate(x):
            return tqdm.tqdm(x)

    # disabled noise functionality from WFS
    if noise =='off':  
        wfs.cam.photonNoise     = 0
        wfs.cam.readoutNoise    = 0
        wfs.cam.backgroundNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')    
    
    # separate tel from ATM
    tel.isPaired = False
    ngs**tel

    # static OPD offset (e.g. NCPA), applied through an OPD_map
    opd_map = _as_opd_map(opd_offset)

    try: 
        nModes = M2C.shape[1]
    except:
        nModes = 1

    intMat = np.zeros([wfs.nSignal,nModes])
    nCycle = int(np.ceil(nModes/nMeasurements))
    nExtra = int(nModes%nMeasurements)

    if nMeasurements>nModes:
        nMeasurements = nModes

    # reference signal of the offset alone (flat DM). In single pass, this is what is subtracted
    # from the push signal, so that the offset itself does not leak into the interaction matrix.
    sref = None
    if single_pass and opd_map is not None:
        ngs**tel*opd_map*wfs
        sref = np.array(wfs.signal, copy=True)

    for i in iterate(range(nCycle)):  
        
        if nModes>1:
            if i==nCycle-1:
                if nExtra != 0:
                    intMatCommands = np.squeeze(M2C[:,-nExtra:])
                else:
                    intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])
            else:
                intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])
        else:
            intMatCommands = np.squeeze(M2C) 
            
        a= time.time()
        # push
        dm.coefs = intMatCommands*stroke
        if opd_map is None:
            ngs**tel*dm*wfs
        else:
            ngs**tel*dm*opd_map*wfs
        sp = wfs.signal

        # pull
        if single_pass:
            if sref is None:
                sm = 0*wfs.signal
            else:
                # broadcast the reference over the columns when measuring several modes at once
                sm = sref[:, None] if np.ndim(sp) == 2 else sref
            factor = 2
        else:
            dm.coefs = -intMatCommands*stroke
            if opd_map is None:
                ngs**tel*dm*wfs
            else:
                ngs**tel*dm*opd_map*wfs
            sm = wfs.signal
            factor = 1

        if i==nCycle-1:
            if nExtra !=0:
                if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)                
                else:
                    if nExtra ==1:
                        intMat[:, -nExtra] = np.squeeze(0.5 * (sp - sm) / stroke)                        
                    else:                        
                        intMat[:, -nExtra:] = np.squeeze(0.5 * (sp - sm) / stroke)
            else:
                 if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)      
                 else:
                    intMat[:,-nMeasurements:] =  np.squeeze(0.5*(sp-sm)/stroke)
        else:
            if nMeasurements==1:
                intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)                
            else:
                intMat[:,i*nMeasurements:((i+1)*nMeasurements)] = np.squeeze(0.5*(sp-sm)/stroke)

        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i+1)*nMeasurements)+'/'+str(nModes))
            b=time.time()
            print('Time elapsed: '+str(b-a)+' s' )
    
    out = CalibrationVault(factor*intMat, invert=invert, nTrunc=nTrunc)
       
    return out