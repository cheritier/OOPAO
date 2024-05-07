# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:26:22 2020

@author: cheritie
"""
import numpy as np
import time
import tqdm
from .CalibrationVault import CalibrationVault


def InteractionMatrix(ngs,
                      atm,
                      tel,
                      dm,
                      wfs,
                      M2C,
                      stroke,
                      phaseOffset=0,
                      nMeasurements=50,
                      noise='off',
                      invert=True,
                      print_time=False,
                      display = False,
                      single_pass = True):
    
    if display is False:
        def iterate(x):
            return x
    else:
        def iterate(x):
            return tqdm.tqdm(x) 
    if wfs.tag=='pyramid' and wfs.gpu_available:
        nMeasurements = 1
        print('Pyramid with GPU detected => using single mode measurement to increase speed.')
#    disabled noise functionality from WFS
    if noise =='off':  
        wfs.cam.photonNoise     = 0
        wfs.cam.readoutNoise    = 0
        wfs.cam.backgroundNoise = 0

    else:
        print('Warning: Keeping the noise configuration for the WFS')    
    
    # separate tel from ATM
    tel.isPaired = False
    ngs*tel
    try: 
        nModes = M2C.shape[1]
    except:
        nModes = 1
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

        # for i in tqdm.tqdm(range(nCycle)):  
    for i in iterate(range(nCycle)):  
        
        if nModes>1:
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
        else:
            intMatCommands = np.squeeze(M2C) 
            
        a= time.time()
#        push
        dm.coefs = intMatCommands*stroke
        tel*dm
        tel.src.phase+=phaseBuffer
        tel*wfs
        sp = wfs.signal
            

#       pull
        if single_pass:
            sm = 0*wfs.signal
            factor = 2
        else:
            dm.coefs=-intMatCommands*stroke
            tel*dm
            tel.src.phase+=phaseBuffer
            tel*wfs
            sm = wfs.signal
            factor = 1
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
        intMat = np.squeeze(intMat)

        if print_time:
            print(str((i+1)*nMeasurements)+'/'+str(nModes))
            b=time.time()
            print('Time elapsed: '+str(b-a)+' s' )
    
    out=CalibrationVault(factor*intMat,invert=invert)
       
    return out


def InteractionMatrixFromPhaseScreen(ngs,atm,tel,wfs,phasScreens,stroke,phaseOffset=0,nMeasurements=50,noise='off',invert=True,print_time=True):
    
    #    disabled noise functionality from WFS
    if noise =='off':  
        wfs.cam.photonNoise  = 0
        wfs.cam.readoutNoise = 0
    else:
        print('Warning: Keeping the noise configuration for the WFS')    
    
    tel.isPaired = False
    ngs*tel
    try: 
        nModes = phasScreens.shape[2]
    except:
        nModes = 1
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
        if nModes>1:
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
        else:
            modes_in = np.squeeze(phasScreens)

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
        intMat = np.squeeze(intMat)
        if print_time:
            print(str((i+1)*nMeasurements)+'/'+str(nModes))
            b=time.time()
            print('Time elapsed: '+str(b-a)+' s' )
    
    out=CalibrationVault(intMat,invert=invert)
  
    return out



# def InteractionMatrixOnePass(ngs,atm,tel,dm,wfs,M2C,stroke,phaseOffset=0,nMeasurements=50,noise='off'):
# #    disabled noise functionality from WFS
#     if noise =='off':  
#         wfs.cam.photonNoise  = 0
#         wfs.cam.readoutNoise = 0
#     else:
#         print('Warning: Keeping the noise configuration for the WFS')    
    
#     tel-atm
#     ngs*tel
#     nModes = M2C.shape[1]
#     intMat = np.zeros([wfs.nSignal,nModes])
#     nCycle = int(np.ceil(nModes/nMeasurements))
#     nExtra = int(nModes%nMeasurements)
#     if nMeasurements>nModes:
#         nMeasurements = nModes
    
#     if np.ndim(phaseOffset)==2:
#         if nMeasurements !=1:      
#             phaseBuffer = np.tile(phaseOffset[...,None],(1,1,nMeasurements))
#         else:
#             phaseBuffer = phaseOffset
#     else:
#         phaseBuffer = phaseOffset

        
#     for i in range(nCycle):  
#         if i==nCycle-1:
#             if nExtra != 0:
#                 intMatCommands  = np.squeeze(M2C[:,-nExtra:])                
#                 try:               
#                     phaseBuffer     = np.tile(phaseOffset[...,None],(1,1,intMatCommands.shape[-1]))
#                 except:
#                     phaseBuffer     = phaseOffset
#             else:
#                 intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])

#         else:
#             intMatCommands = np.squeeze(M2C[:,i*nMeasurements:((i+1)*nMeasurements)])
            
#         a= time.time()
# #        push
#         dm.coefs = intMatCommands*stroke
#         tel*dm
#         tel.src.phase+=phaseBuffer
#         tel*wfs
#         sp = wfs.signal        
#         if i==nCycle-1:
#             if nExtra !=0:
#                 if nMeasurements==1:
#                     intMat[:,i] = np.squeeze(0.5*(sp)/stroke)                
#                 else:
#                     intMat[:,-nExtra:] =  np.squeeze(0.5*(sp)/stroke)
#             else:
#                  if nMeasurements==1:
#                     intMat[:,i] = np.squeeze(0.5*(sp)/stroke)      
#                  else:
#                     intMat[:,-nMeasurements:] =  np.squeeze(0.5*(sp)/stroke)


#         else:
#             if nMeasurements==1:
#                 intMat[:,i] = np.squeeze(0.5*(sp)/stroke)                
#             else:
#                 intMat[:,i*nMeasurements:((i+1)*nMeasurements)] = np.squeeze(0.5*(sp)/stroke)

#         print(str((i+1)*nMeasurements)+'/'+str(nModes))
#         b=time.time()
#         print('Time elapsed: '+str(b-a)+' s' )

#     out=CalibrationVault(intMat)
#     return out      
        
        