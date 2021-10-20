# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:26:22 2020

@author: cheritie
"""
import numpy as np
import time
from AO_modules.calibration.CalibrationVault import calibrationVault

def interactionMatrix(ngs,atm,tel,dm,wfs,M2C,stroke,phaseOffset=0,nMeasurements=50,noise='off',invert=True):
#    disabled noise functionality from WFS
    if noise =='off':  
        wfs.cam.photonNoise  = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')    
    
    tel-atm
    ngs*tel
    nModes = M2C.shape[1]
    intMat = np.zeros([wfs.nSignal,nModes])
    nCycle = int(np.ceil(nModes/nMeasurements))
    nExtra = int(nModes%nMeasurements)
    if nMeasurements>nModes:
        nMeasurements = nModes
    
    if np.ndim(phaseOffset)==2:
        if nMeasurements !=1:      
            phaseBuffer = np.tile(phaseOffset[...,None],(1,1,nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

        
    for i in range(nCycle):  
        if i==nCycle-1:
            if nExtra != 0:
                intMatCommands  = np.squeeze(M2C[:,-nExtra:])                
                try:               
                    phaseBuffer     = np.tile(phaseOffset[...,None],(1,1,intMatCommands.shape[-1]))
                except:
                    phaseBuffer     = phaseOffset
            else:
                intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])

        else:
            intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])
            
        a= time.time()
#        push
        dm.coefs = intMatCommands*stroke
        tel*dm
        tel.src.phase+=phaseBuffer
        tel*wfs
        sp = wfs.signal
            

#       pull
        dm.coefs=-intMatCommands*stroke
        tel*dm
        tel.src.phase+=phaseBuffer
        tel*wfs
        sm = wfs.signal
        if i==nCycle-1:
            if nExtra !=0:
                if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)                
                else:
                    if nExtra ==1:
                        intMat[:,-nExtra] =  np.squeeze(0.5*(sp-sm)/stroke)
                    else:
                        intMat[:,-nExtra:] =  np.squeeze(0.5*(sp-sm)/stroke)
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

        print(str((i+1)*nMeasurements)+'/'+str(nModes))
        b=time.time()
        print('Time elapsed: '+str(b-a)+' s' )
    
    if invert:
        out=calibrationVault(intMat)
    else:
        from AO_modules.tools.tools  import emptyClass
        out = emptyClass        
        out.D = intMat
        
    return out


def interactionMatrixFromPhaseScreen(ngs,atm,tel,wfs,phasScreens,stroke,phaseOffset=0,nMeasurements=50,noise='off',invert=True):
    
    #    disabled noise functionality from WFS
    if noise =='off':  
        wfs.cam.photonNoise  = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')    
    
    tel-atm
    ngs*tel
    nModes=phasScreens.shape[2]
    intMat = np.zeros([wfs.nSignal,nModes])
    nCycle = int(np.ceil(nModes/nMeasurements))
    nExtra = int(nModes%nMeasurements)
    if nMeasurements>nModes:
        nMeasurements = nModes
    
    if np.ndim(phaseOffset)==2:
        if nMeasurements !=1:      
            phaseBuffer = np.tile(phaseOffset[...,None],(1,1,nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

        
    for i in range(nCycle):  
        if i==nCycle-1:
            if nExtra != 0:
                modes_in  = np.squeeze(phasScreens[:,:,-nExtra:])                
                try:               
                    phaseBuffer     = np.tile(phaseOffset[...,None],(1,1,modes_in.shape[-1]))
                except:
                    phaseBuffer     = phaseOffset
            else:
                modes_in = np.squeeze(phasScreens[:,:,i*nMeasurements:((i+1)*nMeasurements)])

        else:
            modes_in = np.squeeze(phasScreens[:,:,i*nMeasurements:((i+1)*nMeasurements)])
            
        a= time.time()
#        push
        tel.OPD = modes_in*stroke
        tel.src.phase+=phaseBuffer
        tel*wfs
        sp = wfs.signal
#       pull
        tel.OPD=-modes_in*stroke
        tel.src.phase+=phaseBuffer
        tel*wfs
        sm = wfs.signal        
        if i==nCycle-1:
            if nExtra !=0:
                if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp-sm)/stroke)                
                else:
                    if nExtra ==1:
                        intMat[:,-nExtra] =  np.squeeze(0.5*(sp-sm)/stroke)
                    else:
                        intMat[:,-nExtra:] =  np.squeeze(0.5*(sp-sm)/stroke)
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

        print(str((i+1)*nMeasurements)+'/'+str(nModes))
        b=time.time()
        print('Time elapsed: '+str(b-a)+' s' )
    
    if invert:
        out=calibrationVault(intMat)
    else:
        from AO_modules.tools.tools  import emptyClass
        out = emptyClass        
        out.D = intMat
        
    return out



def interactionMatrixOnePass(ngs,atm,tel,dm,wfs,M2C,stroke,phaseOffset=0,nMeasurements=50,noise='off'):
#    disabled noise functionality from WFS
    if noise =='off':  
        wfs.cam.photonNoise  = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')    
    
    tel-atm
    ngs*tel
    nModes = M2C.shape[1]
    intMat = np.zeros([wfs.nSignal,nModes])
    nCycle = int(np.ceil(nModes/nMeasurements))
    nExtra = int(nModes%nMeasurements)
    if nMeasurements>nModes:
        nMeasurements = nModes
    
    if np.ndim(phaseOffset)==2:
        if nMeasurements !=1:      
            phaseBuffer = np.tile(phaseOffset[...,None],(1,1,nMeasurements))
        else:
            phaseBuffer = phaseOffset
    else:
        phaseBuffer = phaseOffset

        
    for i in range(nCycle):  
        if i==nCycle-1:
            if nExtra != 0:
                intMatCommands  = np.squeeze(M2C[:,-nExtra:])                
                try:               
                    phaseBuffer     = np.tile(phaseOffset[...,None],(1,1,intMatCommands.shape[-1]))
                except:
                    phaseBuffer     = phaseOffset
            else:
                intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])

        else:
            intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])
            
        a= time.time()
#        push
        dm.coefs = intMatCommands*stroke
        tel*dm
        tel.src.phase+=phaseBuffer
        tel*wfs
        sp = wfs.signal        
        if i==nCycle-1:
            if nExtra !=0:
                if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp)/stroke)                
                else:
                    intMat[:,-nExtra:] =  np.squeeze(0.5*(sp)/stroke)
            else:
                 if nMeasurements==1:
                    intMat[:,i] = np.squeeze(0.5*(sp)/stroke)      
                 else:
                    intMat[:,-nMeasurements:] =  np.squeeze(0.5*(sp)/stroke)


        else:
            if nMeasurements==1:
                intMat[:,i] = np.squeeze(0.5*(sp)/stroke)                
            else:
                intMat[:,i*nMeasurements:((i+1)*nMeasurements)] = np.squeeze(0.5*(sp)/stroke)

        print(str((i+1)*nMeasurements)+'/'+str(nModes))
        b=time.time()
        print('Time elapsed: '+str(b-a)+' s' )

    out=calibrationVault(intMat)
    return out      
        
        
