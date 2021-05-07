# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:31:57 2020

@author: cheritie
"""

import AO_modules.calibration.ao_cockpit_utils_v0p6 as aou
import numpy as np
from astropy.io import fits as pfits




def compute_M2C_minimal_saturation_from_ao_obj(ao_obj, nameFolder = None, nameFile = None, nPetal = 6,nModes = 4300, remove_piston = True):
    
    
 # check if the deformable mirror is M4
    if ao_obj.dm.isM4:
        initName = 'M2C_M4_'
    else:
        initName = 'M2C_'
    
    # check if a name for the folder destination is specified
    if nameFolder is None:
        nameFolder = ao_obj.param['pathInput']
    
    # check if the name for the destination file is specifiec
    if nameFile is None:
        try:
            nameFile = initName+str(ao_obj.param['resolution'])+'_res'+ao_obj.param['extra']
        except:
            nameFile = initName+str(ao_obj.param['resolution'])+'_res'
            
    ### KL computation with extrapolation (Force freindly)

    
    ## RELOADING SOME PARAMETERS
    nact        = ao_obj.dm.nAct
    siz         = ao_obj.tel.resolution
    diameter    = ao_obj.tel.D
    dim         = ao_obj.tel.resolution
    
    GEO = aou.mkp(siz/dim*diameter,siz,diameter,0.)
    PTT = aou.give_zernike(GEO, diameter, 3)
    
    [HiHju, PSDu, dfu] = aou.load(nameFolder+ 'COV_AND_DSP.pkl')
    
    try:
        P2F=np.float64(pfits.getdata('/diskb/cverinau/DATA/Stiffness_Matrix.fits'))

    except:
        P2F=np.float64(pfits.getdata('/Disk3/cheritier/psim/data_calibration/Stiffness_Matrix.fits'))

    nap=P2F.shape[0]
    
    print('DATA LOADED ! ')
    
    P2Ff=np.zeros([nact,nact],dtype=np.float64) 
    
    for k in range(0,nPetal):
        P2Ff[k*nap:(k+1)*nap,k*nap:(k+1)*nap] = P2F.copy()
       
    # get OPD of the IFs after reflection
    
    ao_obj.tel.isPaired = False # separate from eventual atmosphere
    
    ao_obj.dm.coefs = np.eye(ao_obj.dm.nValidAct) # assign dm coefs to get the cube of IF in OPD
    
    ao_obj.tel*ao_obj.dm    # propagate to get the OPD of the IFS after reflection
    
    IF_2D = np.moveaxis(ao_obj.tel.OPD,-1,0)
        
    idxpup=np.where(ao_obj.tel.pupil==1)
    tpup=len(idxpup[0])
    
    IFma=np.matrix(aou.vectorifyb(IF_2D,idxpup))
    DELTA=IFma.T @ IFma
    
    #orthonormalization of PTT
    PTTn=PTT.copy()
    nptt=PTT.shape[2]
    for k in range(1,nptt):
        PTTn[:,:,k] = PTTn[:,:,k]/ np.std(PTT[:,:,k][idxpup])
    
    PTTn_phase=aou.vectorify(PTTn,idxpup)
    
    
    Qn,Rn = np.linalg.qr(PTTn_phase)
    
    Rn=np.asmatrix(Rn)
    Rnm1=Rn.I
    
    PTTn_O =PTTn_phase @ Rnm1*np.sqrt(tpup)
    
    ## A FIT WITH REGULARIZATION In FORCE ADAPTED TO LOW SPATIAL FREQUENCY
    alpha=1.0e-12 # note with uncropped IFs this is alpha=1.e-15
    
    REC = (DELTA + alpha*P2Ff.T @ P2Ff).I @ IFma.T
    
    ## COMMANDS FOR SPCIAL WITH RIGHT EXTRAPOLATION
    CMD_TIK_ptt = REC @ PTTn_O #_phase
    
    ## CHECK FITTING ERROR AND FORCES (should be small)
    PTT_R_phase = IFma @ CMD_TIK_ptt
    
    probe5mum =  CMD_TIK_ptt * 5.e-6  ## TT of 20 mic PtV
    F_probe5mum = P2Ff @ probe5mum
    
    
    print('RMS phase error ='+ str( np.std(PTT_R_phase-PTTn_O,axis=0)))
    print('RMS Force ='+ str(np.std(F_probe5mum,axis=0)))
    print('MAX Force ='+ str(np.max(np.abs(F_probe5mum),axis=0)))
    
    
    P2Ff=np.asmatrix(P2Ff)
    iP2Ff=P2Ff.I
    
    
    ## Compute change of Base from Positions to Forces
    
    Id=np.eye(nact)
    alpha_i=0.   ## This is for an extra damping of large positions if needed
    
    CB=(alpha_i*Id+iP2Ff)
    
    ## IFs in Force
    IF_F = IFma @ CB
    
    DELTA_IF_F = IF_F.T @ IF_F
    
    
    ## PTT in Forces
    TAU=P2Ff @ CMD_TIK_ptt.copy()
    
    
    TAU_phi = IF_F @ TAU
    DELTA_TAU = TAU_phi.T @ TAU_phi
    
    
    G=np.eye(nact)-TAU @ (DELTA_TAU.I @ TAU.T @ DELTA_IF_F )
    
    ## COMPUTE EIGEN-MODES OF MODIFIED IFs
    IF_F_m = IF_F @ G
    
    U_Fm,S_Fm,V_FmT = np.linalg.svd(IF_F_m,full_matrices=False)
    
    V_Fm = V_FmT.T
    
    ## SEED BASIS MINIMIZING FORCE
    BB=np.zeros([nact,nact],dtype=np.float64)
    BB[:,0:nptt] = TAU
    BB[:,nptt:] = G @ V_Fm[:,0:nact-nptt] @ np.diag(1./S_Fm[0:nact-nptt])*np.sqrt(tpup)
    
    ## SEED BASIS MINIMIZING FORCE BACK IN POSITION SPACE
    BBinP = CB @ BB
    
    DELTA_BBinP = BBinP.T @ DELTA @ BBinP
    ## Number of modes conserved: These modes need to be orthogonal
    nmof = nModes
    np.max(np.abs(DELTA_BBinP[0:nmof,0:nmof]/tpup-np.eye(nmof)))
    
    nmax=nmof
    OB = BBinP[:,0:nmax+nptt]
    DELTA_OB = DELTA_BBinP[0:nmax+nptt,0:nmax+nptt]
    
    OBsptt = OB[:,nptt:]
    DELTA_OBsptt = DELTA_OB[nptt:,nptt:]
    
    ## COVARIANCE MATRIX IN SEED BASIS
    Cp = DELTA_OBsptt.I @ OBsptt.T @ HiHju @ OBsptt @ DELTA_OBsptt.I.T
    
    ## COMPUTASTION OF KLs
    Uc,Sc,VcT = np.linalg.svd(Cp)
    Vc=VcT.T
    KL_F = np.zeros([nact,nmax+nptt],dtype=np.float64)
    KL_F[:,0:nptt] = OB[:,0:nptt]
    KL_F[:,nptt:] = OBsptt @ Vc
    
    verif_SIGMA2 = KL_F[:,nptt:].T @ HiHju  @ KL_F[:,nptt:]/tpup**2
    
    print('DM EIGEN MODES WITH DOUBLE DIAGONALISATION: COVARIANCE ERROR = ',np.max(np.abs(verif_SIGMA2-np.diag(Sc))))
   
    
    # remove piston
    if remove_piston:
        M2C = np.asarray(KL_F[:,1:])
        print('Piston removed from the modal basis!' )
    else:
        M2C = np.asarray(KL_F[:,:])
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
    
    import os
    os.remove('COV_AND_DSP.pkl') 
    
    return M2C




def compute_M2C_minimal_saturation(tel, dm, param, nameFolder = None, nameFile = None, nPetal = 6,nModes = 4300, remove_piston = True):
    
    
 # check if the deformable mirror is M4
    if dm.isM4:
        initName = 'M2C_M4_'
    else:
        initName = 'M2C_'
    
    # check if a name for the folder destination is specified
    if nameFolder is None:
        nameFolder = param['pathInput']
    
    # check if the name for the destination file is specifiec
    if nameFile is None:
        try:
            nameFile = initName+str(param['resolution'])+'_res'+param['extra']
        except:
            nameFile = initName+str(param['resolution'])+'_res'
            
    ### KL computation with extrapolation (Force freindly)

    
    ## RELOADING SOME PARAMETERS
    nact        = dm.nAct
    siz         = tel.resolution
    diameter    = tel.D
    dim         = tel.resolution
    
    GEO = aou.mkp(siz/dim*diameter,siz,diameter,0.)
    PTT = aou.give_zernike(GEO, diameter, 3)
    
    [HiHju, PSDu, dfu] = aou.load(nameFolder+ 'COV_AND_DSP.pkl')
    
    try:
        P2F=np.float64(pfits.getdata('/diskb/cverinau/DATA/Stiffness_Matrix.fits'))

    except:
        P2F=np.float64(pfits.getdata('/Disk3/cheritier/psim/data_calibration/Stiffness_Matrix.fits'))

    nap=P2F.shape[0]
    
    print('DATA LOADED ! ')
    
    P2Ff=np.zeros([nact,nact],dtype=np.float64) 
    
    for k in range(0,nPetal):
        P2Ff[k*nap:(k+1)*nap,k*nap:(k+1)*nap] = P2F.copy()
    
    # get OPD of the IFs after reflection
    tel.isPaired = False
    
    dm.coefs = np.eye(dm.nValidAct)
    
    tel*dm
    
    IF_2D = np.moveaxis(tel.OPD,-1,0)
    
    idxpup=np.where(tel.pupil==1)
    tpup=len(idxpup[0])
    
    IFma=np.matrix(aou.vectorifyb(IF_2D,idxpup))
    DELTA=IFma.T @ IFma
    
    #orthonormalization of PTT
    PTTn=PTT.copy()
    nptt=PTT.shape[2]
    for k in range(1,nptt):
        PTTn[:,:,k] = PTTn[:,:,k]/ np.std(PTT[:,:,k][idxpup])
    
    PTTn_phase=aou.vectorify(PTTn,idxpup)
    
    
    Qn,Rn = np.linalg.qr(PTTn_phase)
    
    Rn=np.asmatrix(Rn)
    Rnm1=Rn.I
    
    PTTn_O =PTTn_phase @ Rnm1*np.sqrt(tpup)
    
    ## A FIT WITH REGULARIZATION In FORCE ADAPTED TO LOW SPATIAL FREQUENCY
    alpha=1.0e-12 # note with uncropped IFs this is alpha=1.e-15
    
    REC = (DELTA + alpha*P2Ff.T @ P2Ff).I @ IFma.T
    
    ## COMMANDS FOR SPCIAL WITH RIGHT EXTRAPOLATION
    CMD_TIK_ptt = REC @ PTTn_O #_phase
    
    ## CHECK FITTING ERROR AND FORCES (should be small)
    PTT_R_phase = IFma @ CMD_TIK_ptt
    
    probe5mum =  CMD_TIK_ptt * 5.e-6  ## TT of 20 mic PtV
    F_probe5mum = P2Ff @ probe5mum
    
    
    print('RMS phase error ='+ str( np.std(PTT_R_phase-PTTn_O,axis=0)))
    print('RMS Force ='+ str(np.std(F_probe5mum,axis=0)))
    print('MAX Force ='+ str(np.max(np.abs(F_probe5mum),axis=0)))
    
    
    P2Ff=np.asmatrix(P2Ff)
    iP2Ff=P2Ff.I
    
    
    ## Compute change of Base from Positions to Forces
    
    Id=np.eye(nact)
    alpha_i=0.   ## This is for an extra damping of large positions if needed
    
    CB=(alpha_i*Id+iP2Ff)
    
    ## IFs in Force
    IF_F = IFma @ CB
    
    DELTA_IF_F = IF_F.T @ IF_F
    
    
    ## PTT in Forces
    TAU=P2Ff @ CMD_TIK_ptt.copy()
    
    
    TAU_phi = IF_F @ TAU
    DELTA_TAU = TAU_phi.T @ TAU_phi
    
    
    G=np.eye(nact)-TAU @ (DELTA_TAU.I @ TAU.T @ DELTA_IF_F )
    
    ## COMPUTE EIGEN-MODES OF MODIFIED IFs
    IF_F_m = IF_F @ G
    
    U_Fm,S_Fm,V_FmT = np.linalg.svd(IF_F_m,full_matrices=False)
    
    V_Fm = V_FmT.T
    
    ## SEED BASIS MINIMIZING FORCE
    BB=np.zeros([nact,nact],dtype=np.float64)
    BB[:,0:nptt] = TAU
    BB[:,nptt:] = G @ V_Fm[:,0:nact-nptt] @ np.diag(1./S_Fm[0:nact-nptt])*np.sqrt(tpup)
    
    ## SEED BASIS MINIMIZING FORCE BACK IN POSITION SPACE
    BBinP = CB @ BB
    
    DELTA_BBinP = BBinP.T @ DELTA @ BBinP
    ## Number of modes conserved: These modes need to be orthogonal
    nmof = nModes
    np.max(np.abs(DELTA_BBinP[0:nmof,0:nmof]/tpup-np.eye(nmof)))
    
    nmax=nmof
    OB = BBinP[:,0:nmax+nptt]
    DELTA_OB = DELTA_BBinP[0:nmax+nptt,0:nmax+nptt]
    
    OBsptt = OB[:,nptt:]
    DELTA_OBsptt = DELTA_OB[nptt:,nptt:]
    
    ## COVARIANCE MATRIX IN SEED BASIS
    Cp = DELTA_OBsptt.I @ OBsptt.T @ HiHju @ OBsptt @ DELTA_OBsptt.I.T
    
    ## COMPUTASTION OF KLs
    Uc,Sc,VcT = np.linalg.svd(Cp)
    Vc=VcT.T
    KL_F = np.zeros([nact,nmax+nptt],dtype=np.float64)
    KL_F[:,0:nptt] = OB[:,0:nptt]
    KL_F[:,nptt:] = OBsptt @ Vc
    
    verif_SIGMA2 = KL_F[:,nptt:].T @ HiHju  @ KL_F[:,nptt:]/tpup**2
    
    print('DM EIGEN MODES WITH DOUBLE DIAGONALISATION: COVARIANCE ERROR = ',np.max(np.abs(verif_SIGMA2-np.diag(Sc))))
   
    
    # remove piston
    if remove_piston:
        M2C = np.asarray(KL_F[:,1:])
        print('Piston removed from the modal basis!' )
    else:
        M2C = np.asarray(KL_F[:,:])
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
    
    import os
    os.remove('COV_AND_DSP.pkl') 
    
    return M2C
