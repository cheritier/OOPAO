# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:57:29 2020

@author: cheritie
"""

import numpy as np
from astropy.io import fits as pfits

import OOPAO.calibration.ao_cockpit_psim as aou
from ..tools.tools import createFolder

def compute_KL_basis(tel,atm,dm,lim = 1e-3,remove_piston = True):
    
    M2C_KL = compute_M2C(telescope            = tel,\
                        atmosphere         = atm,\
                        deformableMirror   = dm,\
                        param              = None,\
                        nameFolder         = None,\
                        nameFile           = None,\
                        remove_piston      = remove_piston,\
                        HHtName            = None,\
                        baseName           = None ,\
                        mem_available      = None,\
                        minimF             = False,\
                        nmo                = None,\
                        ortho_spm          = True,\
                        SZ                 = int(2*tel.OPD.shape[0]),\
                        nZer               = 3,\
                        NDIVL              = 1,\
                        recompute_cov=True,\
                        save_output= False, lim_inversion=lim,display=False)
        
    return M2C_KL

def compute_M2C(telescope, atmosphere, deformableMirror, param = None, nameFolder = None, nameFile = None,remove_piston = False,HHtName = None, baseName = None, SpM_2D = None, nZer = 3, SZ=None, mem_available = None, NDIVL = None, computeSpM = True, ortho_spm = True, computeSB = True, computeKL = True, minimF = False, P2F = None, alpha = None, beta = None, lim_SpM = None, lim_SB = None, nmo = None, IF_2D = None, IFma = None, returnSB = False, returnHHt_PSD_df = False, recompute_cov = False,extra_name = '', save_output = True,lim_inversion=1e-3,display=True):

    """
    - HHtName       = None      extension for the HHt Covariance file
    - baseName      = None      extension to the filename for basis saving
    - SpM_2D        = None      2D Specific modes [dim,dim,nspm], if None then automatic
    - nZer          = 3         number of zernike (PTT,...) for automatic computation
    - SZ            = None      telescope.resolutione of FFts for HHt (By default SZ=2*dim)
    - mem_available = None      Memory allocated for HHt computation (default is 50GB)
    - NDIVL         = None      Subdiv. of HHt task in ~NDIVL**2. None:-> mem_available
    - computeSpM    = True      Flag to compute Specific modes 
    - ortho_spm     = True      Flag to orthonormalize specific modes (QR decomposition)
    - computeSB     = True      Flag to compute the Seed Basis
    - computeKL     = True      Flag to compute the KL basis
    - minimF        = False     Flag to minimize Forces
    - P2F           = None      Stiffness matrix (loaded by default)
    - alpha         = None      Force regularization parameter (expert)
    - beta          = None      Position damping parameter (expert)
    - nmo           = None      Number of modes to compute
    - IF_2D         = None      2D Influence Functions (only for speeding up)
    - IFma          = None      Serial Influence Functions (only for speeding up)
    - returnSB      = False     Flag to return also the Seed Basis (w/ or w/o KL)
    
    """

    if nmo is None:
        nmo = deformableMirror.nValidAct
        
    if deformableMirror.isM4:
        initName = 'M2C_M4_'
    else:
        initName = 'M2C_'
    if baseName is not None:
        initName = initName + baseName+'_'
    if nameFolder is None:
        if param:
            nameFolder = param['pathInput']
        else:
            nameFolder = ''
    if save_output:
        createFolder(nameFolder)
    
    if nameFile is None:
        try:
            nameFile = initName + str(telescope.resolution)+'_res'+param['extra']+extra_name
        except:
            nameFile = initName + str(telescope.resolution)+'_res'+extra_name
            
   
    # the function takes as an input an object with obj.tel, obj.atm,obj.
    diameter    = telescope.D
    r0          = atmosphere.r0
    L0          = atmosphere.L0
    pupil       = telescope.pupil

    telescope.isPaired = False # separate from eventual atmosphere
    
    if IF_2D is None:
        deformableMirror.coefs = np.eye(deformableMirror.nValidAct) # assign dm coefs to get the cube of IF in OPD
        if display:
            print('COMPUTING TEL*DM...')
            print(' ')
        telescope*deformableMirror    # propagate to get the OPD of the IFS after reflection
        if display:
            print('PREPARING IF_2D...')
            print(' ')
        IF_2D = np.moveaxis(telescope.OPD,-1,0)
    
    nact = IF_2D.shape[0]
    if display:
        print('Computing Specific Modes ...')
        print(' ')
    GEO = aou.mkp(telescope.resolution/telescope.resolution*diameter,telescope.resolution,diameter,0.)
    
    if nZer is not None and SpM_2D is None:
        SpM_2D = aou.give_zernike(GEO, diameter, nZer)
        nspm = nZer
    if SpM_2D is not None:
        nspm=SpM_2D.shape[2]
    
    if SZ is None:
        SZ = int(2*telescope.resolution) ## SZ=1110 for dxo=0.06944 and SZ=1542 for dxo=0.05
    if display:
        print('COMPUTING VON KARMAN 2D PSD...')
        print(' ')
    PSD_atm , df, pterm = aou.VK_DSP_up(diameter,r0,L0,SZ,telescope.resolution,1,pupil)

#%% ---------- EVALUATE SPLIT OF WORK UPON MEMORY AVAILABLE ----------
    

#%% ----------COMPUTE HHt COVARIANCE MATRIX (OR LOAD EXISTING ONE) ----------    
        
    if recompute_cov is False:
        try:
            #HHt, PSD_atm, df = aou.load(nameFolder+'HHt_PSD_df_'+HHtName+'_r'+str(r0)+'_SZ'+str(SZ)+'.pkl')
            HHt, PSD_atm, df = aou.load(nameFolder+'HHt_PSD_df_'+HHtName+'.pkl')
            if display:
                print('LOADED COV MAT HHt...')
                print(' ')
        except:
            recompute_cov = True
    
    if recompute_cov is True:
        if display:
            print('COMPUTING COV MAT HHt...')
            print(' ')
        
        if mem_available is None:
            mem_available=100.e9   
        if NDIVL is None:
            mem,NDIVL=aou.estimate_ndivl(SZ,telescope.resolution,nact,mem_available)
        if NDIVL == 0:
            NDIVL = 1
        BLOCKL=nact//NDIVL
        REST=nact-BLOCKL*NDIVL
        HHt = aou.DO_HHt(IF_2D,PSD_atm,df,pupil,BLOCKL,REST,SZ,0)
        if save_output:
            try:
                aou.save(nameFolder+'HHt_PSD_df_'+HHtName+'.pkl',[HHt, PSD_atm, df])
            except:    
                aou.save(nameFolder+'HHt_PSD_df_'+initName+'r'+str(r0)+'_SZ'+str(SZ)+'.pkl',[HHt, PSD_atm, df])

    
#%% ----------PRECOMPUTE MOST USED QUANTITIES   ----------    
    if computeSpM == True or computeSB == True or computeKL == True:
        
    ## VALID OPD POINTS IN PUPIL
        idxpup=np.where(pupil==1)
        tpup=len(idxpup[0])

    ## Matrix of serialized IFs
        if IFma is None:
            if display:
                print('SERIALIZING IFs...')
                print(' ')
            IFma=np.matrix(aou.vectorifyb(IF_2D,idxpup))

    ## Matrix of serialized Special modes
        if display:
            print('SERIALIZING Specific Modes...')
            print(' ')
        Tspm=np.matrix(aou.vectorify(SpM_2D,idxpup))

    ## CROSS-PRODUCT OF IFs
        if display:
            print('COMPUTING IFs CROSS PRODUCT...')
            print(' ')
        DELTA=IFma.T @ IFma
    
#%% ----------COMPUTE SPECIFIC MODES BASIS    ----------  
    if minimF == True:
        if P2F is None:
            try:
                P2F=np.float64(pfits.getdata(param['pathInput']+'P2F.fits'))*1.e6 #( in N/m)
                
                P2Ff=np.zeros([nact,nact],dtype=np.float64)
                if nact>892:
                    nap=nact//6
                    for k in range(0,6):
                        P2Ff[k*nap:(k+1)*nap,k*nap:(k+1)*nap] = P2F.copy()
                        
                    K=np.asmatrix(P2Ff)
                    del P2Ff
                else:
                    K = np.asmatrix(P2F) 
            except:
                print('Could not find the P2F matrix.. ignoring the force minimization')
                minimF = False
                        

        if alpha is None:
            alpha = 1.e-9
        if beta is None:
            beta=1.e-5
    
    if computeSpM == True and minimF == True:
        if display:
            print('BUILDING FORCE-OPTIMIZED SPECIFIC MODES...')
            print(' ')
        check=1
        amp_check=1.e-6
            
        SpM = aou.build_SpecificBasis_F(Tspm,IFma,DELTA,K,alpha,ortho_spm,check,amp_check)
#        SpM_opd = IFma @ SpM
        if display:
            print('CHECKING ORTHONORMALITY OF SPECIFIC MODES...')
            print(' ')
        
        DELTA_SpM_opd = SpM.T @ DELTA @ SpM
        if display:
            print('Orthonormality error for SpM = ', np.max(np.abs(DELTA_SpM_opd/tpup-np.eye(nspm))))


    if computeSpM == True and minimF == False:
        if lim_SpM is None:
            lim_SpM = lim_inversion #1.e-3
        check=1
        amp_check=1.e-6
        lim=lim_SpM #lim_inversion
        SpM = aou.build_SpecificBasis_C(Tspm,IFma,DELTA,lim,ortho_spm,check,amp_check)                                   
        if display:
            print('CHECKING ORTHONORMALITY OF SPECIFIC MODES...')
            print(' ')
        DELTA_SpM_opd = SpM.T @ DELTA @ SpM
        if display:
            print('Orthonormality error for SpM = ', np.max(np.abs(DELTA_SpM_opd/tpup-np.eye(nspm))))

        
#%% ----------COMPUTE SEED BASIS    ----------
    if computeKL == True:
        computeSB = True
        
    if computeSB == True:
        
        if minimF == False:
            if display:
                print('BUILDING SEED BASIS ...')
                print(' ')
            lim = lim_inversion
            SB = aou.build_SeedBasis_C(IFma, SpM,DELTA,lim)
            nSB=SB.shape[1]
            DELTA_SB = SB.T @ DELTA @ SB
            if display:
                print('Orthonormality error for '+str(nSB)+' modes of the Seed Basis = ',np.max(np.abs(DELTA_SB[0:nSB,0:nSB]/tpup-np.eye(nSB))))
            
        if minimF == True:
            if display:
                print('BUILDING FORCE OPTIMIZED SEED BASIS ...')
                print(' ')
            SB = aou.build_SeedBasis_F(IFma, SpM, K, beta)
            nSB=SB.shape[1]
            DELTA_SB = SB.T @ DELTA @ SB
            if nmo>SB.shape[1]:
                print('WARNING: Number of modes requested too high, taking the maximum value possible!')
                nmo = SB.shape[1]
            if display:
                print('Orthonormality error for '+str(nmo)+' modes of the Seed Basis = ',np.max(np.abs(DELTA_SB[0:nmo,0:nmo]/tpup-np.eye(nmo))))

    if computeKL == False:
        BASIS=np.asmatrix(np.zeros([nact,nspm+nSB],dtype=np.float64))
        BASIS[:,0:nspm] = SpM
        BASIS[:,nspm:] = SB
        if remove_piston == True:
            BASIS = np.asarray(BASIS[:,1:])
            print('Piston removed from the modal basis!' )
            
#%% ----------COMPUTE KL BASIS    ----------

    if computeKL == True:
        check=1
        if nmo>SB.shape[1]:
            print('WARNING: Number of modes requested too high, taking the maximum value possible!')
            nmoKL = SB.shape[1]
        else:
            nmoKL = nmo
        KL,Sc=aou.build_KLBasis(HHt,SB,DELTA,nmoKL,check)
        
        DELTA_KL = KL.T @ DELTA @ KL
        if display:
            print('Orthonormality error for '+str(nmoKL)+' modes of the KL Basis = ',np.max(np.abs(DELTA_KL[0:nmoKL,0:nmoKL]/tpup-np.eye(nmoKL))))
        

        BASIS=np.asmatrix(np.zeros([nact,nspm+nmoKL],dtype=np.float64))
        BASIS[:,0:nspm] = SpM
        BASIS[:,nspm:] = KL
        if remove_piston == True:
            BASIS = np.asarray(BASIS[:,1:])
            if display:
                print('Piston removed from the modal basis!' )
# save output in fits file
        if save_output:

            hdr=pfits.Header()
            hdr['TITLE'] = initName+'_KL' #'M4_KL'
            empty_primary = pfits.PrimaryHDU(header=hdr)
        ## CAREFUL THE CUBE IS SAVED AS A NON SPARSE MATRIX
            primary_hdu = pfits.ImageHDU(BASIS)
            hdu = pfits.HDUList([empty_primary, primary_hdu])
            hdu.writeto(nameFolder+nameFile+'.fits',overwrite=True)
        if returnHHt_PSD_df == True:
            return np.asarray(BASIS), HHt,PSD_atm, df
        else:
            return np.asarray(BASIS)
        
    if returnSB == True:
        if save_output:
    
            hdr=pfits.Header()
            hdr['TITLE'] = initName+'_SB' #'M4_KL'
            empty_primary = pfits.PrimaryHDU(header=hdr)
        ## CAREFUL THE CUBE IS SAVED AS A NON SPARSE MATRIX
            primary_hdu = pfits.ImageHDU(BASIS)
            hdu = pfits.HDUList([empty_primary, primary_hdu])
            hdu.writeto(nameFolder+nameFile+'.fits',overwrite=True)
     
        return np.asarray(BASIS),SB
 
