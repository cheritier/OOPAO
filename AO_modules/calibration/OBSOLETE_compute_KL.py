# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:57:29 2020

@author: cheritie
"""

# modules for the KL basis computation:
import numpy as np
from astropy.io                                     import fits as pfits
from AO_modules.tools.tools                         import createFolder
import AO_modules.calibration.ao_cockpit_utils_v0p6 as aou

def compute_M2C(telescope, atmosphere, deformableMirror, param, nameFolder = None, nameFile = None, nPlan = 2,remove_piston = True):
    
    
    if deformableMirror.isM4:
        initName = 'M2C_M4_'
    else:
        initName = 'M2C_'
        
    if nameFolder is None:
        nameFolder = param['pathInput']
    createFolder(nameFolder)
    if nameFile is None:
        try:
            nameFile = initName + str(param['resolution'])+'_res'+param['extra']
        except:
            nameFile = initName + str(param['resolution'])+'_res'
            
    
    
    
    
    # the function takes as an input an object with obj.tel, obj.atm,obj.
    npt         = telescope.resolution
    diameter    = telescope.D
    siz         = telescope.resolution
    dim         = telescope.resolution
    
#    # consider both cases with ans without sparse matrices
#    try:
#        IF_2D=np.reshape(np.moveaxis(deformableMirror.modes.A,-1,0),[deformableMirror.nValidAct,npt,npt])
#    except:
#        IF_2D=np.reshape(np.moveaxis(deformableMirror.modes,-1,0),[deformableMirror.nValidAct,npt,npt])

    telescope.isPaired = False # separate from eventual atmosphere
    
    deformableMirror.coefs = np.eye(deformableMirror.nValidAct) # assign dm coefs to get the cube of IF in OPD
    
    telescope*deformableMirror    # propagate to get the OPD of the IFS after reflection
    
    IF_2D = np.moveaxis(telescope.OPD,-1,0)
    
    
    print('Computing Forced modes PTT...')
    GEO = aou.mkp(siz/dim*diameter,siz,diameter,0.)
    PTT=aou.give_zernike(GEO, diameter, 3)
    
    NDIVL       = nPlan                                     # FASTEST WAY OF DOUBLE DIAG: Geometrical covariance matrix is computed
    SZ = int(npt*(2))

    needGramm   = 1                                     # Forced modes PTT is given in input inphase [siz,siz,3]
    lim         = 1.e-3                                 # DM EIGE
    r0          = atmosphere.r0
    L0          = atmosphere.L0
    TYPE_MO     = 'DM_EIGEN_DOUBLE'
    PSD_atm     = 0                                     # ALWAYS 0 for now
    faster      = 1                                    # with FFT size SZ ~ 10% larger than input, leave it to 1.
    DELTApre    = np.zeros(deformableMirror.nValidAct)        #DELTA.copy() If DELTA has been pre-computed you can put it in input there . Otherwise np.zeros(deformableMirror.nValidAct) (cf. dimension is not the final one) and DELTA is recomputed
    
    inputBASIS  = np.eye(deformableMirror.nValidAct)          # In case the input basis in not the IFms one has to put here the modal basis so that it is used to compute the right Covariance Matrix.
    
     # Influence Functions (or modal basis) in 2D phase space on a small array (here [deformableMirror.nValidAct,siz,siz] ) and in np.float32
    inputBASIS_2Dphase = IF_2D.copy() # Note to save memory one can avoid the copy() but then IF_2D is multiplied by the pupil inside aou.build_modes (THIS MAY BE CHANGED LATER ON, BUT TO SAVE MEMORY AND KEEP COMPUTATION SPEED, IT IS LIKE THAT)
    
    M2C , C_D_TAI0_f4,verif_SIGMA2_TAI0_f4,SIGMA_TAI0_f4,PSDu,dfu,HiHju = aou.build_modes(inputBASIS_2Dphase,deformableMirror.coordinates.T,telescope.pupil,PTT,r0,L0,diameter,DELTApre,dim,PSD_atm,TYPE_MO,faster, NDIVL,needGramm,lim,inputBASIS,SZ)
    
    aou.save(nameFolder+ 'COV_AND_DSP.pkl',[HiHju, PSDu, dfu])

    # remove piston
    if remove_piston:
        M2C = np.asarray(M2C[:,1:])
        print('Piston removed from the modal basis!' )
    else:
        M2C = np.asarray(M2C[:,:])
        print('Piston not removed from the modal basis!' )
       

 # 29/01/2021 -- cheritier :removed, now done directly on the input Inf Funct  
#    # normalize to be used by DM
#    M2C *=0.5
    
    # save output in fits file
    hdr=pfits.Header()
    hdr['TITLE'] = 'KL_M2C'
    empty_primary = pfits.PrimaryHDU(header=hdr)
    ## CAREFUL THE CUBE IS SAVED AS A NON SPARSE MATRIX
    primary_hdu = pfits.ImageHDU(M2C)
    hdu = pfits.HDUList([empty_primary, primary_hdu])
    hdu.writeto(nameFolder+nameFile+'.fits',overwrite=True)
    
    print('M2C matrix saved here:' +nameFolder )
    
    return M2C
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# same function using an ao object as an input

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def compute_M2C_from_ao_obj(ao_obj,nameFolder=None, nameFile=None,remove_piston = True,nPlan = 4):
    
    
    # check if the deformable mirror is M4
    if ao_obj.dm.isM4:
        initName = 'M2C_M4_'
    else:
        initName = 'M2C_'
    
    # check if a name for the folder destination is specified
    if nameFolder is None:
        nameFolder = ao_obj.param['pathInput']
    createFolder(nameFolder)
    # check if the name for the destination file is specifiec
    if nameFile is None:
        try:
            nameFile = initName+str(ao_obj.param['resolution'])+'_res'+ao_obj.param['extra']
        except:
            nameFile = initName+str(ao_obj.param['resolution'])+'_res'
            
    
    # the function takes as an input an object with obj.tel, obj.atm,...
    npt         = ao_obj.tel.resolution
    diameter    = ao_obj.tel.D
    siz         = ao_obj.tel.resolution
    dim         = ao_obj.tel.resolution
    
#    # consider both cases with ans without sparse matrices
#    try:
#        IF_2D = np.reshape(np.moveaxis(ao_obj.dm.modes.A,-1,0),[ao_obj.dm.nValidAct,npt,npt])
#    except:
#        IF_2D = np.reshape(np.moveaxis(ao_obj.dm.modes,-1,0),[ao_obj.dm.nValidAct,npt,npt])

    # get OPD of the IFs after reflection
    
    ao_obj.tel.isPaired = False # separate from eventual atmosphere
    
    ao_obj.dm.coefs = np.eye(ao_obj.dm.nValidAct) # assign dm coefs to get the cube of IF in OPD
    
    ao_obj.tel*ao_obj.dm    # propagate to get the OPD of the IFS after reflection
    
    IF_2D = np.moveaxis(ao_obj.tel.OPD,-1,0)
    print(IF_2D.shape)

    print('Computing Forced modes PTT...')
    GEO = aou.mkp(siz/dim*diameter,siz,diameter,0.)
    PTT=aou.give_zernike(GEO, diameter, 3)
    
    print(PTT.shape)
    NDIVL       = nPlan                                     # FASTEST WAY OF DOUBLE DIAG: Geometrical covariance matrix is computed
    SZ = int(npt*(2))

    needGramm   = 1                                     # Forced modes PTT is given in input inphase [siz,siz,3]
    lim         = 1.e-3                                 # DM EIGE
    r0          = ao_obj.atm.r0
    L0          = ao_obj.atm.L0
    TYPE_MO     = 'DM_EIGEN_DOUBLE'
    PSD_atm     = 0                                     # ALWAYS 0 for now
    faster      = 1                                     # with FFT size SZ ~ 10% larger than input, leave it to 1.
    DELTApre    = np.zeros(ao_obj.dm.nValidAct)        # DELTA.copy() If DELTA has been pre-computed you can put it in input there . Otherwise np.zeros(ao_obj.dm.nValidAct) (cf. dimension is not the final one) and DELTA is recomputed
    
    inputBASIS  = np.eye(ao_obj.dm.nValidAct)          # In case the input basis in not the IFms one has to put here the modal basis so that it is used to compute the right Covariance Matrix.
    
     # Influence Functions (or modal basis) in 2D phase space on a small array (here [ao_obj.dm.nValidAct,siz,siz] ) and in np.float32
    inputBASIS_2Dphase = IF_2D.copy() # Note to save memory one can avoid the copy() but then IF_2D is multiplied by the pupil inside aou.build_modes (THIS MAY BE CHANGED LATER ON, BUT TO SAVE MEMORY AND KEEP COMPUTATION SPEED, IT IS LIKE THAT)
    
    M2C , C_D_TAI0_f4,verif_SIGMA2_TAI0_f4,SIGMA_TAI0_f4,PSDu,dfu,HiHju = aou.build_modes(inputBASIS_2Dphase,ao_obj.dm.coordinates.T,ao_obj.tel.pupil,PTT,r0,L0,diameter,DELTApre,dim,PSD_atm,TYPE_MO,faster, NDIVL,needGramm,lim,inputBASIS,SZ)
    
    aou.save(nameFolder+ 'COV_AND_DSP.pkl',[HiHju, PSDu, dfu])
    
    
    # remove piston
    if remove_piston:
        M2C = np.asarray(M2C[:,1:])
        print('Piston removed from the modal basis!' )
    else:
        M2C = np.asarray(M2C[:,:])
        print('Piston not removed from the modal basis!' )
       
 # 29/01/2021 -- cheritier :removed, now done directly on the input Inf Funct  
#    # normalize to be used by DM
#    M2C *=0.5
    
    # save output in fits file
    hdr=pfits.Header()
    hdr['TITLE'] = 'M4_M2C'
    empty_primary = pfits.PrimaryHDU(header=hdr)
    ## CAREFUL THE CUBE IS SAVED AS A NON SPARSE MATRIX
    primary_hdu = pfits.ImageHDU(M2C)
    hdu = pfits.HDUList([empty_primary, primary_hdu])
    hdu.writeto(nameFolder+nameFile+'.fits',overwrite=True)
    

    return M2C
        
               