#MY MODULE

import numpy as np
#import proper as prp
import math
import mpmath
import numba
from numba import jit
from math import factorial
import os
import psutil
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import pyfftw
import numexpr as ne
from astropy.io import fits

import pdb

import pickle

def run(file2exec):
    exec(open(file2exec).read())

    
def save(filename,variables_list):

    f=open(filename,'wb')
    pickle.dump(variables_list,f)
    f.close()
    
def load(filename):
    f=open(filename,'rb')
    variables_list=pickle.load(f)
    return variables_list


#import aotools as aot

#import concurrent.futures

def aid(x):
    #This fucntion returns the memory block adress of an array
    return x.__array_interface__['data'][0]



def vectorify(array,idxpup):
    if array.ndim==3:
        array_=array[idxpup[0],idxpup[1],0:array.shape[2]]
    if array.ndim==2:
        array_=array[idxpup[0],idxpup[1]]
    return array_

def vectorifyb(array,idxpup):
    if array.ndim==3:
        nact=array.shape[0]
        array_=np.zeros([len(idxpup[0]),array.shape[0]],dtype=np.float64)
        for k in range(0,nact):
            print(k, ' ', end='\r', flush=True)
            array_[:,k] = array[k,idxpup[0],idxpup[1]]
    return array_


#def triple_d_modes((IFma,posxy,pupil,PTT,r0,L0,diameter,DELTA,dim,PSD_atm,TYPE_MO, faster, NDIVL, needGramm, lim, P2Ff):
                   

def build_modes(IF_2D,posxy,pupil,PTT,r0,L0,diameter,DELTA,dim,PSD_atm,TYPE_MO, faster, NDIVL, needGramm, lim,EG_P2Ff,SZ):
    # RULE FOR MATRIX REPRESENTATION: 1ST INDEX = DATA. 2ND INDEX = RANK
    nact=IF_2D.shape[0]
    size=pupil.shape[0]
    for k in range(0,nact):
        IF_2D[k,:,:]=IF_2D[k,:,:]*pupil
    idxpup=np.where(pupil==1)
    tpup=len(idxpup[0]) # NUMBER OF POINTS INSIDE THE PUPIL
    fmax=dim/diameter*0.5
    PSD_atm=np.asarray(PSD_atm)
    PTT=np.asarray(PTT)
    ### FOR GENDRON COVARIANCE MATRIX CREATE CRUDE PTT COMMANDS
    if PTT.ndim < 2 and TYPE_MO != 'GENDRON_STF' and TYPE_MO != 'GENDRON_COV':
        nptt=PTT
        COBS=0.28
        geo=mkp(size/dim*diameter,size,diameter,COBS)
        pupil=geo.pupil
        tpup=np.sum(pupil)
        idxpup=np.where(pupil == 1)
        
        # GIVES Piston, Tip and Tilt or whatever other modes needed
        if TYPE_MO == 'GENDRON_STF' or TYPE_MO == 'GENDRON_COV' :
            CMD_PTT_geo = np.zeros([nact,nptt],dtype='float32')
            #print('SHAPE : ',posxy[0,:].shape)
            CMD_PTT_geo[:,0]=1.0
            CMD_PTT_geo[:,1]=posxy[0,:]/(diameter/2)*2. # should be Zernike normalization
            CMD_PTT_geo[:,2]=posxy[1,:]/(diameter/2)*2.
            CMD_PTT_geo=np.asmatrix(CMD_PTT_geo)
            
        else:
            PTT=aou.give_zernike(geo, diameter, nptt)
       
    else:
        if PTT.ndim==2:
            nptt=PTT.shape[1]
        if PTT.ndim==3:    
            nptt=PTT.shape[2]

    
    if PSD_atm.ndim < 2 and TYPE_MO == 'DM_EIGEN_DOUBLE':
        #PSD_atm , df = VK_DSP(diameter,r0,L0,SZ,dim)

        PSD_atm , df, pterm = VK_DSP_up(diameter,r0,L0,SZ,dim,1,pupil)
        
    else:
        df=1./(2.*fmax)
    
    # COMPUTING ZONAL BASIS CROSS-PRODUCT: needed almost always
    print('VECTORIFYING INFLUENCE FUNCTIONS...')
    IFma=np.matrix(vectorifyb(IF_2D,idxpup))
    print('COMPUTING DELTA AND ITS INVERSE...')
    if DELTA.ndim < 2:
        DELTA = IFma.T @ IFma
    if lim ==0:
        iDELTA = DELTA.I
    if lim !=0:
        ud,sd,vdh=np.linalg.svd(DELTA)
        if lim <=1.:
            nmax=np.max(np.where(sd/np.max(sd) > lim))
        if lim > 1:
            nmax=lim
        MEV=np.zeros([nact,nact],dtype=np.float64)
        MEV[0:nmax,0:nmax] = np.diag(1./sd[0:nmax])
                            
        iDELTA = vdh.T @ MEV @ ud.T
    
    print(' DELTA DONE...')
    
    if TYPE_MO == 'GENDRON_STF' or TYPE_MO == 'GENDRON_COV' :
        
        iPTT = np.linalg.pinv(CMD_PTT_geo) # not so sure about the transpose
        Gg = np.eye(nact) - CMD_PTT_geo@iPTT
        
        print('BUILDING COVARIANCE MATRIX / ACT FOR GENDRON MODES...')
        if TYPE_MO == 'GENDRON_STF':
            #geocovmat=geo_covmat(8.,0.1,posxy[0,:],posxy[1,:])
            geocovmat, DELTA2=gendron_covmat(pupil,diameter,r0,L0,size,dim,posxy[0,:],posxy[1,:],'light')
                                     
        if TYPE_MO == 'GENDRON_COV':
            geocovmat, DELTA2=gendron_covmat(pupil,diameter,r0,L0,size,dim,posxy[0,:],posxy[1,:],'heavy')        
                                     
        geocovmat_rptt = Gg.T@geocovmat@Gg
        print('DIAGONALIZING COVARIANCE MATRIX IN PTT-FREE ACTUATOR SPACE...')
        ug,sg,vgh = np.linalg.svd(geocovmat_rptt)

        cmd_gendron = np.zeros([nact,nact],dtype='float64')
        cmd_gendron[:,0:3] = CMD_PTT_geo@np.sqrt(np.diag(1./np.diag(CMD_PTT_geo.T@CMD_PTT_geo)))
        cmd_gendron[:,3:] = ug[:,0:nact-nptt]
        
        
        GENDRON=cmd_gendron.copy()
        GENDRON[:,3:]=Gg@cmd_gendron[:,3:]
        
        MMG=GENDRON.T@GENDRON
        print('MODES IN COMMAND SPACE ORTHONORMALITY ERROR = ', np.max(np.abs(MMG-np.eye(nact))))
    
    if TYPE_MO == 'DM_EIGEN_FORCED' or TYPE_MO == 'DM_EIGEN_DOUBLE':
        if needGramm==1:
            print('GRAMM SCHMIDT ON FORCED MODES...')
            # BUILDING THE FIRST  imposed ZERNIKE MODES
            nptt=PTT.shape[2] # PTT stands for Piston Tip Tilt that are usually forced but it may be whatever

            # MAKING THE 2D ARRAYS AS 1D VECTOR (SPEED UP BY MATRIX OPERATIONS)
            print('VECTORIZING PTT PHASE SHAPES...')
            PHA_ptt=np.matrix(vectorify(PTT,idxpup))

            # COMPUTING ZONAL COMMANDS CORRESPONDING TO FORCED MODES
            print('COMPUTING COMMANDS FOR PTT...')
            projPTT = IFma.T @ PHA_ptt
            CMD_PTT_=iDELTA @projPTT
            del projPTT
            #CMD_PTT_=DELTA.I @ IFma.T @ PHA_ptt # TEMPORARY BECAUSE NON ORTHOGONAL

            # USING GRAMM SCHMIDT  TO ORTHOGONALIZE (AND NORMALIZE) FORCED MODES
            print('EXECUTING GRAMM-SCMIDT ORTHOGONALIZATION...')
            CMD_PTT=CMD_PTT_.copy()*0.
            modzin=IFma@CMD_PTT_
            modzout=PHA_ptt.copy()*0.

            proj=np.zeros([nptt,nptt],dtype=np.float64)

            for k in range(0,nptt):
                if k==0 :
                    CMD_PTT[:,0]=CMD_PTT_[:,0]
                    modzout[:,0]=modzin[:,0]

                if k > 0:
                    modzout[:,k]=modzin[:,k]
                    CMD_PTT[:,k]=CMD_PTT_[:,k]
                    proj=np.zeros(k,dtype=np.float64)
                    for j in range(0,k):
                        proj[j]=modzin[:,k].T @ modzout[:,j] / tpup
                    for j in range(0,k):
                        modzout[:,k] = modzout[:,k]-proj[j]*modzout[:,j]
                        CMD_PTT[:,k]=CMD_PTT[:,k]-proj[j]*CMD_PTT[:,j]

            # FINAL NORMALIZATION OF FORCED MODES
            print('NORMALIZING MODES...')
            newMM=modzout.T@modzout
            cor_factor=1./np.sqrt(np.diag(newMM/tpup))

            for k in range(0,nptt):
                CMD_PTT[:,k]=CMD_PTT[:,k]*cor_factor[k]
            print('GRAMM SCHMIDT ON FORCED MODES: DONE')
        if needGramm==0:
            CMD_PTT=PTT
        finalPTT=IFma@CMD_PTT

        finalPPT_MM=finalPTT.T@finalPTT
        print(' ')
        
        print('FORCED MODES MAX ORTHONORMALITY ERROR = ', np.max(np.abs(finalPPT_MM/tpup-np.eye(nptt))))
    
    # COMPUTING DM EIGEN MODES
    if TYPE_MO == 'DM_EIGEN_PURE':
        print(' ')
        print('STARTING PURE DM EIGENMODES COMPUTATION...')
        s_time = time.time()
        M0,s0,M0t=np.linalg.svd(DELTA)
        e_time = time.time() - s_time
        M0O=M0@np.diag(1./np.sqrt(s0))*np.sqrt(tpup)
        
        print('DONE. DM_EIGEN MODES TOOK: ', e_time)
        #EIGEN0=IFma@M0
    
    if TYPE_MO == 'DM_EIGEN_FORCED' or TYPE_MO == 'DM_EIGEN_DOUBLE':
        print(' ')
        print('STARTING FORCED DM EIGENMODES COMPUTATION...')
        if TYPE_MO == 'DM_EIGEN_DOUBLE':
            print('1ST DIAGONALIZATION...')
        s_time = time.time()
        TAU=CMD_PTT
        TAU_T=np.transpose(TAU)
        G=np.eye(nact)-(TAU @ (np.linalg.inv(TAU_T @ DELTA @ TAU)) @ TAU_T @ DELTA )
        Gt=np.transpose(G)
        Mp,D2,Mpt=np.linalg.svd(Gt@DELTA@G)
        if lim > 1:
            nmax=lim
        if lim <= 1:
            nmax=np.max(np.where( D2/np.max(D2) > lim))
        if lim == 0.:
            M=Mp[:,0:nact-nptt]@np.diag(np.sqrt(1./D2[0:nact-nptt]))*np.sqrt(tpup)
            EIGENF=np.zeros([nact,nact],dtype=np.float64)
            EIGENF[:,0:nptt]=CMD_PTT
            EIGENF[:,nptt:] = G@M
            
        if lim !=0.:
            M=Mp[:,0:nmax]@np.diag(np.sqrt(1./D2[0:nmax]))*np.sqrt(tpup)
            EIGENF=np.zeros([nact,nmax+nptt],dtype=np.float64)
            EIGENF[:,0:nptt]=CMD_PTT
            EIGENF[:,nptt:] = G@M

        e_time = time.time() - s_time
        print('DONE. FORCED DM_EIGEN MODES TOOK: ', e_time)
        print(' ')
        print('VERIFYING FORCED DM EIGENMODES ...')
        #EIGENF_phase=IFma@EIGENF
        MM_EIGENF = EIGENF.T @ DELTA @ EIGENF  # EIGENF_phase.T@EIGENF_phase
        if lim ==0.:
            print('MAX FORCED DM EIGENMODES ORTHONORMALITY ERROR = ', np.max(np.abs(MM_EIGENF/tpup-np.eye(nact))))
        if lim != 0.:
            print('MAX FORCED DM EIGENMODES ORTHONORMALITY ERROR = ', np.max(np.abs(MM_EIGENF/tpup-np.eye(nmax+nptt))))
        #iMM_EIGEN3=np.linalg.inv(MM_EIGEN3)
        #iEIGEN3=np.linalg.inv(EIGEN3)
     
    if TYPE_MO == 'DM_EIGEN_DOUBLE':
        if NDIVL != 0. :
            BLOCKL=nact//NDIVL
            REST=nact-BLOCKL*NDIVL
            print('BLOCKL / REST = ',BLOCKL, REST)
            print(' ')
        STOT=time.time()
       
        #print('SHIFTING INPUT IFMs ARRAYS ...')
        # IIs=np.zeros([nact,size,size],dtype=np.float64)

#         IIs=np.zeros([nact,size,size],dtype=np.float64)
        
#         for k in range(0,nact):
#             IIs[k,:,:]=np.fft.fftshift(IFMlarge[:,:,k]*pupil)
        print(' ')
        #print('COMPUTING VAN KARMAN POWER SPECTRUM ...')
        #PSD_ATM=VK_DSP(diameter,r0,L0,size,dim)
        size2=SZ//faster
        print(' ')
        print('STARTING STATISTICAL DM/ATM COVARIANCE MATRIX COMPUTATION...')
        s_time = time.time()
        #C = DO_COVAR_IF_ATM_MAT(IFma,PSD_atm,df,size2,idxpup,BLOCKL,REST)
        if NDIVL > 0:
            print('COMPUTING HiHj...')
            #print('WARNING altering input IFs (xpupil ans /SZ/df)')
            #for k in range(0,nact):
            #    IF_2D[k,:,:]=IF_2D[k,:,:]*pupil/SZ/df/np.sqrt(tpup)

            HiHj = DO_COVAR_IF_ATM_MAT(IF_2D,PSD_atm,df,size2,idxpup,BLOCKL,REST,SZ)
            #HiHj = 0
            #C=HiHj

            C = iDELTA @ HiHj @ iDELTA.T
            
            #C = load('tryFC_Z.pkl')

            #if lim==0:
            #    C = DELTA.I @ HiHj @ DELTA.T.I
            #if lim !=0:
            #    C = iDELTA @ HiHj @ iDELTA.T
        if NDIVL==0:
            C_interm=geo_covmat_fast(diameter, r0, posxy[0,:],posxy[1,:])
            EG_P2Ff=np.asmatrix(EG_P2Ff)
            C = EG_P2Ff.I @ C_interm @ EG_P2Ff.T.I
            #C, DELTA2=gendron_covmat(pupil,diameter,r0,L0,size,dim,posxy[0,:],posxy[1,:],'light')
        
        e_time = time.time() - s_time
        print(' ')
        print('DONE. DM/ATM COVARIANCE MATRIX TOOK: ', e_time)
        print(' ')
        print('2ND DIAGONALIZATION...')
        s_time = time.time()
        
        #Cp=M.I@C@M.I.T

        EGFred = np.asmatrix(EIGENF[:,nptt:])
        
        Cp = EGFred.I @ C @ EGFred.I.T
        
        A,SIGMA,At=np.linalg.svd(Cp)

        #B=G@M@A

        B = EGFred @ A
        
        e_time = time.time() - s_time
        print('DONE. 2ND DIAGONALIZATION TOOK: ',e_time)
        print(' ')
        print('VERIFICATION...')
        #Bo=M@A
        if lim == 0.:
            theBASIS = np.asmatrix(np.zeros([nact,nact],dtype=np.float64))
        if lim != 0.:
            theBASIS = np.asmatrix(np.zeros([nact,nmax+nptt],dtype=np.float64))
        theBASIS[:,0:nptt]=CMD_PTT #/np.sqrt(tpup)
        theBASIS[:,nptt:]=B

        #theBASIS_phase=IFma @ theBASIS
        MMthe=theBASIS.T@DELTA@theBASIS
        if lim ==0.:
            print('DM EIGEN MODES WITH DOUBLE DIAGONALISATION: ORTHONORMALITY ERROR = ', np.max(np.abs(MMthe/tpup-np.eye(nact))))
        if lim !=0.:
            print('DM EIGEN MODES WITH DOUBLE DIAGONALISATION: ORTHONORMALITY ERROR = ', np.max(np.abs(MMthe/tpup-np.eye(nmax+nptt))))
            
        #verif_SIGMA2 = Bo.I @ C @ Bo.T.I
        
        verif_SIGMA2 = B.I @ C @ B.T.I

        #np.max(np.abs(verif_SIGMA2-np.diag(SIGMA)))

        print('DM EIGEN MODES WITH DOUBLE DIAGONALISATION: COVARIANCE ERROR = ',np.max(np.abs(verif_SIGMA2-np.diag(SIGMA))))

        #C_real = theBASIS.I @ C @ theBASIS.T.I

        ETOT=time.time()-STOT
        print('KL WITH DOUBLE DIAGONALISATION TOTAL TIME :',ETOT)
                
    
    if TYPE_MO == 'FULL_KL':
        geo=mkp(size/dim*diameter,size,diameter,0.)
        pupil=geo.pupil
        idxpup=np.where(pupil==1.0)
        posxy=np.zeros([2,len(idxpup[0])],dtype=np.complex128)
        posxy[0,:]=geo.xx[idxpup[0],idxpup[1]]
        posxy[1,:]=geo.yy[idxpup[0],idxpup[1]]

        nact=len(idxpup[0])
        IFMlarge=np.zeros([size,size,nact],dtype=np.float64)
        posxy=np.zeros([2,len(idxpup[0])],dtype=np.complex128)
        posxy[0,:]=geo.xx[idxpup[0],idxpup[1]]
        posxy[1,:]=geo.yy[idxpup[0],idxpup[1]]

        nact=len(idxpup[0])
    
        IFMlarge=np.zeros([size,size,nact],dtype=np.float64)
        for k in range(0,nact):
            IFMlarge[(idxpup[0])[k],(idxpup[1])[k],k]=1.0

        IFma=vectorify(IFMlarge,idxpup)
        #IFmaT=np.transpose(IFma)

        PHA_ptt=vectorify(PTT,idxpup)
        #PHA_ptt_T=np.transpose(PHA_ptt)

        #DELTA=np.eye(nact) #IFmaT@IFma
        #iDELTA=np.eye(nact)     #np.linalg.inv(DELTA)



        #CMD_PTT_=iDELTA@IFmaT@PHA_ptt

        CMD_PTT_= IFma.T @ PHA_ptt
        CMD_PTT=CMD_PTT_.copy()*0.


        # GRAMM SCMIDT ORTHO
        modzin=IFma@CMD_PTT_
        modzout=PHA_ptt.copy()*0.

        proj=np.zeros([nptt,nptt],dtype=np.float64)

        for k in range(0,nptt):
            print(k, ' ', end='\r', flush=True)
            if k==0 :
                CMD_PTT[:,0]=CMD_PTT_[:,0]
                modzout[:,0]=modzin[:,0]

            if k > 0:
                modzout[:,k]=modzin[:,k]
                CMD_PTT[:,k]=CMD_PTT_[:,k]
                proj=np.zeros(k,dtype=np.float64)
                for j in range(0,k):
                    proj[j]=np.transpose(modzin[:,k])@modzout[:,j]/tpup
                for j in range(0,k):
                    modzout[:,k] = modzout[:,k]-proj[j]*modzout[:,j]
                    CMD_PTT[:,k]=CMD_PTT[:,k]-proj[j]*CMD_PTT[:,j]

        newMM=np.transpose(modzout)@modzout

        cor_factor=1./np.sqrt(np.diag(newMM/tpup))

        for k in range(0,nptt):
            CMD_PTT[:,k]=CMD_PTT[:,k]*cor_factor[k]
        
        verifo=CMD_PTT.T@CMD_PTT
        
        print('PTT for full KL: COVARIANCE ERROR = ',np.max(np.abs(verifo/tpup-np.eye(nptt))))   
        ### 1st DIAGONALISATION
        Id=np.asmatrix(np.eye(nact))
        TAU=np.asmatrix(CMD_PTT)
        TAU_T=np.transpose(TAU)
        DELTA=Id #np.eye(nact)
        G=np.eye(nact)-(TAU @ (np.linalg.inv(TAU_T @ DELTA @ TAU)) @ TAU_T @ DELTA )
        #G=Id-(TAU @ ((TAU.T @ TAU).I) @ TAU.T  )
        Gt=np.transpose(G)
        print,('STARTING 1ST DIAGONALIZATION...')
        s_time = time.time()
        Mp,D2,Mpt=np.linalg.svd(Gt@DELTA@G) #svd(Gt@DELTA@G) # could be reduecd to svd(Gt@G)
        #Mp,D2,Mpt=np.linalg.svd(G.T@G)
        e_time = time.time() - s_time
        print('DONE. 1st DIAGONALIZATION TOOK: ',e_time)
        
        M=Mp[:,0:nact-nptt]@np.diag(np.sqrt(1./D2[0:nact-nptt]))*np.sqrt(nact)
        
        print('BUILDING COVARIANCE MATRIX...: ')
        C=-1.*np.asmatrix(geo_covmat(diameter,r0,posxy[0,:],posxy[1,:]))
        #C=np.asmatrix()
        Cp=M.I@C@M.I.T
        A,SIGMA,At=np.linalg.svd(Cp)

        B=G@M@A
        Bo=M@A
        KL = np.asmatrix(np.zeros([nact,nact],dtype=np.float64))
        KL[:,0:nptt]=CMD_PTT #/np.sqrt(nact)
        KL[:,nptt:]=B #*np.sqrt(nact)
        
        MM_KL=KL.T@KL
        
        print('KL WITH DOUBLE DIAGONALISATION: ORTHONORMALITY ERROR = ', np.max(np.abs(MM_KL/nact-np.eye(nact))))
        
        verif_SIGMA2 = Bo.I @ C @ Bo.T.I #KL[:,nptt:].I @ C @ KL[:,nptt:].T.I
        
        C_real = KL.I @ C @ KL.T.I

        #np.max(np.abs(verif_SIGMA2-np.diag(SIGMA)))

        print('KL WITH DOUBLE DIAGONALISATION: COVARIANCE ERROR = ',np.max(np.abs(verif_SIGMA2-np.diag(SIGMA))))
        print(' ')
        #ETOT=time.time()-STOT
        #print('KL WITH DOUBLE DIAGONALISATION TOTAL TIME :',ETOT)

    
    if TYPE_MO == 'DM_EIGEN_PURE':
        return M0O
    
    if TYPE_MO == 'DM_EIGEN_FORCED':
        return EIGENF #, MM_EIGENF, DELTA
    
    if TYPE_MO == 'DM_EIGEN_DOUBLE':
        if NDIVL==0:
            HiHj=0
        return theBASIS,C,verif_SIGMA2,SIGMA, PSD_atm, df, HiHj
        #return theBASIS,C, C_real
        #return theBASIS,C, C_real,Bo , G, M, A,SIGMA #EIGEN_DOUBLE
    
    if TYPE_MO == 'FULL_KL':
        
        return KL, SIGMA, C, C_real, idxpup, Bo  #C, C_real, idxpup
    
    if TYPE_MO == 'GENDRON_STF' or TYPE_MO == 'GENDRON_COV' :
        
        return GENDRON, cmd_gendron, Gg # CMD_PTT_geo #GENDRON
    
    #return CMD_PTT #CMD_PTT #, finalPPT_MM, EIGENF
    
#def project_on_basis():
    

# @numba.jit(nopython = False, parallel=True)
# def do_covar_IF_ATM(IFMs,PSD,pupil,df):
#     #IFMs,PSD,pupil must be size/2 shifted already
#     nact=IFMs.shape[2]
#     size=IFMs.shape[0]
#     #pupc=np.zeros([size,size],dtype='complex128')
#     pupc=pyfftw.empty_aligned((size,size), dtype='complex128')
#     test=np.fft.fft2(pupc)
#     PSDs=PSD#myshift2D(PSD,size//2,size//2)
    
#     #COMPUTE FT of IFMs
#     FT_IFMs=np.zeros([size,size,nact],dtype='complex128')
#     #print('Doing FFTs of IFMs')
#     for k in range(0,nact):
#         #print(k, ' ', end='\r', flush=True)
#         pupc=IFMs[:,:,k]*pupil #myshift2D(IFMs[:,:,k]*pupil,size//2,size//2)
#         FT_IFMs[:,:,k]=np.fft.fft2(pupc)/size
    
#     #COMPUTE INTEGRAL OF CONVOLUTION IN FOURIER SPACE
#     MM_stat=np.zeros([nact,nact],dtype=np.float64)
#     #print('Summing convolution')
#     for k in range(0,nact):
#         #print(k, ' ', end='\r', flush=True)
#         for j in numba.prange(k, nact):
#         #for j in range(k, nact):
#             tosum=(FT_IFMs[:,:,k]*np.conj(FT_IFMs[:,:,j])*df**2)*PSDs
#             MM_stat[k,j]=np.sum(tosum.real)
#             MM_stat[j,k]=MM_stat[k,j]
            
#     return  MM_stat       
 
#@numba.jit(nopython = False, parallel=True)
def do_covar_IF_ATM_(IFMs,PSD,pupil,df):
    #IFMs,PSD,pupil must be size/2 shifted already
    nact=IFMs.shape[2]
    size=IFMs.shape[0]
    #pupc=np.zeros([size,size],dtype='complex128')
    pupc=pyfftw.empty_aligned((size,size), dtype='complex128')
    test=np.fft.fft2(pupc)
    PSDs=PSD#myshift2D(PSD,size//2,size//2)
    
    #COMPUTE FT of IFMs
    FT_IFMs=np.zeros([size,size,nact],dtype='complex128')
    print('Doing FFTs of IFMs')
    for k in range(0,nact):
        print(k, ' ', end='\r', flush=True)
        pupc=IFMs[:,:,k]*pupil #myshift2D(IFMs[:,:,k]*pupil,size//2,size//2)
        FT_IFMs[:,:,k]=np.fft.fft2(pupc)/size
    
    #COMPUTE INTEGRAL OF CONVOLUTION IN FOURIER SPACE
    MM_stat=np.zeros([nact,nact],dtype=np.float64)
    print('Shifting')
    PSDs_=(myshift2D(PSDs,size//2,size//2))[size//2-size//4:size//2+size//4,size//2-size//4:size//2+size//4]
    FT_IFMs_=np.zeros([size//2,size//2,nact],dtype='complex128')
    for k in range(0,nact):
        print(k, ' ', end='\r', flush=True)
        FT_IFMs_[:,:,k]=myshift2D(FT_IFMs[:,:,k],size//2,size//2)[size//2-size//4:size//2+size//4,size//2-size//4:size//2+size//4]
    print('Summing convolution')
    for k in range(0,nact):
        print(k, ' ', end='\r', flush=True)
        #for j in numba.prange(k, nact):
        for j in range(k, nact):
            tosum=(FT_IFMs_[:,:,k]*np.conj(FT_IFMs_[:,:,j])*df**2)*PSDs_
            MM_stat[k,j]=np.sum(tosum.real)
            MM_stat[j,k]=MM_stat[k,j]
            
    return  MM_stat       
 
#def DO_COVAR_IF_ATM_MAT(IFMs,PSD,df,size2): #,restr_idx)

def DO_COVAR_IF_ATM_MAT(ilal,PSD,df,size2,idxpup,BLOCKL,REST,SZ): #,restr_idx)

#IFMs,PSD,pupil must be size/2 shifted already
    #nact=IFma.shape[1]
    nact=ilal.shape[0]
    size=ilal.shape[1] #PSD.shape[0]
    #SZ=PSD.shape[0]
    NCL=nact//BLOCKL
    if REST!=0:
        NCL=NCL+1
        
    print(' ')
    print('CREATING FFTW PLANS...')

    aa = pyfftw.empty_aligned((BLOCKL,SZ, SZ), dtype='complex128')
    #bb = pyfftw.empty_aligned((BLOCKL,SZ, SZ), dtype='complex128')

    cc = pyfftw.empty_aligned((REST,SZ, SZ), dtype='complex128')
    #dd = pyfftw.empty_aligned((REST,SZ, SZ), dtype='complex128')

    #fft_object_ab = pyfftw.FFTW(aa,bb, axes=(1,2),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=mp.cpu_count())

    #fft_object_cd = pyfftw.FFTW(cc,dd, axes=(1,2),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=mp.cpu_count())


    fft_object_aa = pyfftw.FFTW(aa,aa, axes=(1,2),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=mp.cpu_count())
  
    fft_object_cc = pyfftw.FFTW(cc,cc, axes=(1,2),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=mp.cpu_count())

 

    MM_stat=np.zeros([nact,nact],dtype=np.float64)
    stot_time = time.time()
    print('NCL = ', NCL)
    #temp=np.zeros([size2,size2],dtype=np.float32)
    #idfx=np.where(temp == 0.)
    #del temp
    
    #PSD_=PSD[size//2-size2//2:size//2+size2//2,size//2-size2//2:size//2+size2//2]
    PSD_=PSD[SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2]
    #MAT_PSD = PSD_[idfx[0],idfx[1]]
    MAT_PSD = np.reshape(PSD_,size2*size2)

    #pdb.set_trace()
    #for kA in range(0,1):
     #   for kB in range(0,1):
            
    for kA in range(0,NCL):
#        print(kA, ' ', end='\r', flush=True)
        
        for kB in range(kA,NCL):

            print(' ')
            print('Expanding IFMs...', kA, kB)
            

            if REST==0:
                
                IFMs_A=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)
                IFMs_A[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kA*BLOCKL:(kA+1)*BLOCKL,:,:]
                
                if kB != kA:
                    IFMs_B=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)    
                    IFMs_B[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kB*BLOCKL:(kB+1)*BLOCKL,:,:]
                
                if kB == kA:
                    IFMs_B = IFMs_A

            #pdb.set_trace()

            if REST!=0:
                
                if kA < NCL-1:
                    IFMs_A=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)
                    IFMs_A[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kA*BLOCKL:(kA+1)*BLOCKL,:,:]
                    
                if kB < NCL-1 and kB != kA:
                    IFMs_B=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)
                    IFMs_B[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kB*BLOCKL:(kB+1)*BLOCKL,:,:]
                   
                if kA == NCL-1:
                    IFMs_A=np.zeros([REST,SZ,SZ],dtype=np.float64)
                    IFMs_A[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[-REST:,:,:]
                    
                if kB == NCL-1 and kB != kA:
                    IFMs_B=np.zeros([REST,SZ,SZ],dtype=np.float64)
                    IFMs_B[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[-REST:,:,:]
                    
                if kB == kA:
                   IFMs_B = IFMs_A
            
            #fits.writeto('IFMs_A_'+str(kA)+'_'+str(kB)+'.fits',IFMs_A,overwrite=True)
            #fits.writeto('IFMs_B_'+str(kA)+'_'+str(kB)+'.fits',IFMs_B,overwrite=True)

            print(' ')
            print('EXECUTING ...', kA, kB)
            
            if REST != 0:
                if kA < NCL-1:
                    print('STARTED NE WORK  A ...', kA, kB)
                    aa[:,:,:]=ne.evaluate("complex(IFMs_A,0.)")
                    print('FINISHED NE WORK A ...', kA, kB)
                    del IFMs_A
                    
                    print('STARTED SHIFTING  A ...', kA, kB)
                    for k in range(0,aa.shape[0]):
                        aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                    print('FINISHED SHIFTING  A ...', kA, kB)
                    
                    print('STARTED FFT  A ...', kA, kB)
                    res=fft_object_aa()
                    print('FINISHED FFT  A ...', kA, kB)
                    
                    print('STARTED SHIFTING#2  A ...', kA, kB)
                    for k in range(0,aa.shape[0]):
                        aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                    print('FINISHED SHIFTING#2  A ...', kA, kB)

                    print('STARTED FILLING  A ...', kA, kB)
                    MAT_A = np.zeros([aa.shape[0],size2*size2],dtype=np.complex128)
                    for ki in range(0,aa.shape[0]):
                        print(ki, ' ', end='\r', flush=True)
                        MAT_A[ki,:] = np.reshape(aa[ki,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2],size2*size2)*MAT_PSD*df**2
                     
                    #for k in range(0,aa.shape[0]):
                     #   MAT_A[k,:]=(aa[k,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2])[idfx[0],idfx[1]]*MAT_PSD*df**2
                    print('FINISHED FILLING  A ...', kA, kB)     
###
                                   
                if kB < NCL-1:
                    print('STARTED NE WORK  B ...', kA, kB)
                    aa[:,:,:]=ne.evaluate("complex(IFMs_B,0.)")
                    del IFMs_B
                    print('FINISHED NE WORK  B ...', kA, kB)
                    
                    print('STARTED SHIFTING  B ...', kA, kB)
                    for k in range(0,aa.shape[0]):
                        aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                    print('FINISHED SHIFTING  B ...', kA, kB)
                    
                    print('STARTED FFT  B ...', kA, kB)
                    res=fft_object_aa()
                    print('FINISHED FFT  B ...', kA, kB)
                    
                    print('STARTED SHIFTING#2  B ...', kA, kB)
                    for k in range(0,aa.shape[0]):
                        aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                    print('FINISHED SHIFTING#2  B ...', kA, kB)

                    print('STARTED FILLING  B ...', kA, kB)
                    MAT_B = np.zeros([size2*size2,aa.shape[0]],dtype=np.complex128)
                    for ki in range(0,aa.shape[0]):
                        print(ki, ' ', end='\r', flush=True)
                        tmp = np.reshape(aa[ki,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2],size2*size2)
                        cj_tmp=ne.evaluate("conj(tmp)")
                        MAT_B[:,ki]=cj_tmp
                    #for k in range(0,aa.shape[0]):
                     #   MAT_B[:,k]=np.conj(aa[k,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2])[idfx[0],idfx[1]]
                    print('FINISHED FILLING  B ...', kA, kB)



                if kA==NCL-1:
                    print('STARTED NE WORK  A ...', kA, kB)
                    cc[:,:,:]=ne.evaluate("complex(IFMs_A,0.)")
                    print('FINISHED NE WORK A ...', kA, kB)
                    del IFMs_A

                    print('STARTED SHIFTING  A ...', kA, kB)
                    for k in range(0,cc.shape[0]):
                        cc[k,:,:] = np.fft.fftshift(cc[k,:,:])
                    print('FINISHED SHIFTING  A ...', kA, kB)

                    print('STARTED FFT  A ...', kA, kB)
                    res=fft_object_cc()
                    print('FINISHED FFT  A ...', kA, kB)

                    print('STARTED SHIFTING#2  A ...', kA, kB  )    
                    for k in range(0,cc.shape[0]):
                        cc[k,:,:] = np.fft.fftshift(cc[k,:,:])
                    print('FINISHED SHIFTING#2  A ...', kA, kB)

                    print('STARTED FILLING  A ...', kA, kB)      
                    MAT_A = np.zeros([cc.shape[0],size2*size2],dtype=np.complex128)
                    for ki in range(0,cc.shape[0]):
                        print(ki, ' ', end='\r', flush=True)
                        MAT_A[ki,:] = np.reshape(cc[ki,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2],size2*size2)*MAT_PSD*df**2
                     
                    #for k in range(0,cc.shape[0]):
                     #   MAT_A[k,:]=(cc[k,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2])[idfx[0],idfx[1]]*MAT_PSD*df**2
                    print('FINISHED FILLING  A ...', kA, kB)


###

                    
                if kB==NCL-1:
                    print('STARTED NE WORK  B ...', kA, kB)
                    cc[:,:,:]=ne.evaluate("complex(IFMs_B,0.)")
                    print('FINISHED NE WORK B ...', kA, kB)
                    del IFMs_B

                    print('STARTED SHIFTING  B ...', kA, kB)
                    for k in range(0,cc.shape[0]):
                        cc[k,:,:] = np.fft.fftshift(cc[k,:,:])
                    print('FINISHED SHIFTING  A ...', kA, kB)     

                    print('STARTED FFT  B ...', kA, kB)    
                    res=fft_object_cc()
                    print('FINISHED FFT  B ...', kA, kB)

                    print('STARTED SHIFTING#2  B ...', kA, kB )     
                    for k in range(0,cc.shape[0]):
                        cc[k,:,:] = np.fft.fftshift(cc[k,:,:])
                    print('FINISHED SHIFTING#2  B ...', kA, kB)
                          
                    print('STARTED FILLING  B ...', kA, kB)      
                    MAT_B = np.zeros([size2*size2,cc.shape[0]],dtype=np.complex128)
                    for ki in range(0,cc.shape[0]):
                        print(ki, ' ', end='\r', flush=True)
                        tmp = np.reshape(cc[ki,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2],size2*size2)
                        cj_tmp=ne.evaluate("conj(tmp)")
                        MAT_B[:,ki]=cj_tmp
                    #for k in range(0,cc.shape[0]):
                     #   MAT_B[:,k]=np.conj(cc[k,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2])[idfx[0],idfx[1]]
                    print('FINISHED FILLING  B ...', kA, kB)  
                                                       

            if REST == 0:
                print('STARTED NE WORK  A ...', kA, kB)
                aa[:,:,:]=ne.evaluate("complex(IFMs_A,0.)")
                print('FINISHED NE WORK A ...', kA, kB)
                del IFMs_A

                print('STARTED SHIFTING  A ...', kA, kB)
                for k in range(0,aa.shape[0]):
                    aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                print('FINISHED SHIFTING  A ...', kA, kB)

                print('STARTED FFT  A ...', kA, kB) 
                res=fft_object_aa()
                print('FINISHED FFT  A ...', kA, kB)

                print('STARTED SHIFTING#2  A ...', kA, kB)
                for k in range(0,aa.shape[0]):
                    aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                print('FINISHED SHIFTING#2  A ...', kA, kB)
                      
                print('STARTED FILLING  A...', kA, kB) 
                MAT_A = np.zeros([aa.shape[0],size2*size2],dtype=np.complex128)

                for ki in range(0,aa.shape[0]):
                    print(ki, ' ', end='\r', flush=True)
                    MAT_A[ki,:] = np.reshape(aa[ki,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2],size2*size2)*MAT_PSD*df**2
                
                #for k in range(0,aa.shape[0]):
                 #   MAT_A[k,:]=(aa[k,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2])[idfx[0],idfx[1]]*MAT_PSD*df**2
                print('FINISHED FILLING  A...', kA, kB)       

###

                print('STARTED NE WORK  B ...', kA, kB)
                aa[:,:,:]=ne.evaluate("complex(IFMs_B,0.)")
                print('FINISHED NE WORK B ...', kA, kB)
                del IFMs_B

                print('STARTED SHIFTING  B ...', kA, kB)
                for k in range(0,aa.shape[0]):
                    aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                print('FINISHED SHIFTING  B ...', kA, kB)

                print('STARTED FFT  B ...', kA, kB) 
                res=fft_object_aa()
                print('FINISHED FFT  B ...', kA, kB)

                print('STARTED SHIFTING#2  B ...', kA, kB )
                for k in range(0,aa.shape[0]):
                    aa[k,:,:] = np.fft.fftshift(aa[k,:,:])
                print('FINISHED SHIFTING#2  B ...', kA, kB )

                print('STARTED FILLING  B ...', kA, kB)
                MAT_B = np.zeros([size2*size2,aa.shape[0]],dtype=np.complex128)

                for ki in range(0,aa.shape[0]):
                    print(ki, ' ', end='\r', flush=True)
                    tmp = np.reshape(aa[ki,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2],size2*size2)
                    cj_tmp=ne.evaluate("conj(tmp)")
                    MAT_B[:,ki]=cj_tmp

                #for k in range(0,aa.shape[0]):
                 #   MAT_B[:,k]=np.conj(aa[k,SZ//2-size2//2:SZ//2+size2//2,SZ//2-size2//2:SZ//2+size2//2])[idfx[0],idfx[1]]

                print('FINISHED FILLING  B ...', kA, kB)

            # fits.writeto('MAT_PSD.fits',MAT_PSD,overwrite=True)
#             #fits.writeto('DF.fits',df,overwrite=True)
#             fits.writeto('AAr_'+str(kA)+'_'+str(kB)+'.fits',aa.real,overwrite=True)
#             fits.writeto('AAi_'+str(kA)+'_'+str(kB)+'.fits',aa.imag,overwrite=True)
              
           #  fits.writeto('MATr_A_'+str(kA)+'_'+str(kB)+'.fits',MAT_A.real,overwrite=True)
#             fits.writeto('MATr_B_'+str(kA)+'_'+str(kB)+'.fits',MAT_B.real,overwrite=True)
#             fits.writeto('MATi_A_'+str(kA)+'_'+str(kB)+'.fits',MAT_A.imag,overwrite=True)
#             fits.writeto('MATi_B_'+str(kA)+'_'+str(kB)+'.fits',MAT_B.imag,overwrite=True)
            
            print(' ')      
            print('SUMMING CONVOLUTION BY MATRIX MULTIPLICATION...', kA, kB)
            s_time = time.time()
            if REST==0:
                MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL] =  np.ascontiguousarray((MAT_A @ MAT_B).real)
                if kB != kA:
                     MM_stat[kB*BLOCKL:(kB+1)*BLOCKL,kA*BLOCKL:(kA+1)*BLOCKL] = MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL].T
                    
                
            if REST!=0:
                
                if kA<NCL-1 and kB< NCL-1:
                    MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL] =  np.ascontiguousarray((MAT_A @ MAT_B).real)
                    
                    
                    
                if kA==NCL-1 and kB==NCL-1:
                    MM_stat[-REST:,-REST:] =  np.ascontiguousarray((MAT_A @ MAT_B).real)

                    
                if kA==NCL-1 and kB<NCL-1:
                    MM_stat[-REST:,kB*BLOCKL:(kB+1)*BLOCKL] =  np.ascontiguousarray((MAT_A @ MAT_B).real)
                    
                if kA<NCL-1 and kB==NCL-1:
                    
                    MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,-REST:] =  np.ascontiguousarray((MAT_A @ MAT_B).real)
                    
                if kB != kA:
                    MM_stat[kB*BLOCKL:(kB+1)*BLOCKL,kA*BLOCKL:(kA+1)*BLOCKL] = MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL].T
                    
            e_time = time.time() - s_time
            print(' ')
            print('DONE. MATRIX MULTIPLICATION TOOK: ', e_time, kA, kB)

            #fits.writeto('MM_stat_'+str(kA)+'_'+str(kB)+'.fits',MM_stat,overwrite=True)


    etot_time = time.time() - stot_time
    print('DONE. DOUBLE DIAGONALIZATION TOOK: ', etot_time)
    print('DF IS:', df)         
    return  MM_stat #/(SZ**2)


def do_covar_IF_ATM_MAT(IFMs,PSDs,pupil,df,size2): #,restr_idx)
    #IFMs,PSD,pupil must be size/2 shifted already
    nact=IFMs.shape[2]
    print('NACT = ',nact)
    size=IFMs.shape[0]
    #pupc=np.zeros([size,size],dtype='complex128')
    pupc=pyfftw.empty_aligned((size,size), dtype=np.complex128)
    test=np.fft.fft2(pupc)
    #PSDs=PSD#myshift2D(PSD,size//2,size//2)
    
    #COMPUTE FT of IFMs
    FT_IFMs=np.zeros([size,size,nact],dtype=np.complex128)
    print('Doing FFTs of IFMs')
    for k in range(0,nact):
        print(k, ' ', end='\r', flush=True)
        pupc=IFMs[:,:,k]*pupil #myshift2D(IFMs[:,:,k]*pupil,size//2,size//2)
        FT_IFMs[:,:,k]=np.fft.fft2(pupc)/size
    
    #COMPUTE INTEGRAL OF CONVOLUTION IN FOURIER SPACE
    #MM_stat=np.zeros([nact,nact],dtype=np.float64)
    print('Shifting')
    #PSDs_=(myshift2D(PSDs,size//2,size//2))
    PSDs_=np.fft.fftshift(PSDs)
    #PSDs_=(myshift2D(PSDs,size//2,size//2))[size//2-size//4:size//2+size//4,size//2-size//4:size//2+size//4]
    #FT_IFMs_=np.zeros([size//2,size//2,nact],dtype='complex128')
    FT_IFMs_=np.zeros([size,size,nact],dtype='complex128')
    for k in range(0,nact):
        print(k, ' ', end='\r', flush=True)
        #FT_IFMs_[:,:,k]=myshift2D(FT_IFMs[:,:,k],size//2,size//2)
        FT_IFMs_[:,:,k]=np.fft.fftshift(FT_IFMs[:,:,k])

#[size//2-size//4:size//2+size//4,size//2-size//4:size//2+size//4]
    print('Building matrices by re-dimensioning')
    temp=np.zeros([size,size],dtype=np.float32)
    idfx=np.where(temp == 0.)
    MAT_PSD = PSDs_[idfx[0],idfx[1]]
    MAT_A = np.zeros([nact,size*size],dtype=np.complex128)
    MAT_B = np.zeros([size*size,nact],dtype=np.complex128) 
    for k in range(0,nact):
        print(k, ' ', end='\r', flush=True)
        MAT_A[k,:]=FT_IFMs_[idfx[0],idfx[1],k]*MAT_PSD*df**2
    MAT_B=np.conj(FT_IFMs_[idfx[0],idfx[1],:])
        
    print('Summing convolution by Matrix multiplcation')
    MM_stat =  np.ascontiguousarray((MAT_A @ MAT_B).real)
    
    return  MM_stat
 

    
def do_covar_IF_ATM_MAT_evenfaster(IFMs,PSDs,pupil,df,size2): #,restr_idx):
    #IFMs,PSD,pupil must be size/2 shifted already
    nact=IFMs.shape[0]
    size=IFMs.shape[2]
    #pupc=np.zeros([size,size],dtype='complex128')
    #pupc=pyfftw.empty_aligned((size,size), dtype=np.complex128)
    #test=np.fft.fft2(pupc)
    #PSDs=PSD#myshift2D(PSD,size//2,size//2)
    
    #COMPUTE FT of IFMs
    #FT_IFMs=np.zeros([size,size,nact],dtype=np.complex128)
    print('Doing FFTs of IFMs')
    FT_IFMs=np.transpose(np.fft.fft2(IFMs)/size)
    FT_IFMs.shape
#     for k in range(0,nact):
#         print(k, ' ', end='\r', flush=True)
#         pupc=IFMs[:,:,k]*pupil #myshift2D(IFMs[:,:,k]*pupil,size//2,size//2)
#         FT_IFMs[:,:,k]=np.fft.fft2(pupc)/size
    
    #COMPUTE INTEGRAL OF CONVOLUTION IN FOURIER SPACE
    MM_stat=np.zeros([nact,nact],dtype=np.float64)
    print('Shifting')
    
    #PSDs_=(myshift2D(PSDs,size//2,size//2))[size//2-size//4:size//2+size//4,size//2-size//4:size//2+size//4]
    #FT_IFMs_=np.zeros([size//2,size//2,nact],dtype='complex128')
#     if size==size2 :
#         FT_IFMs_=np.zeros([size,size,nact],dtype=np.complex128)
#         for k in range(0,nact):
#             print(k, ' ', end='\r', flush=True)
#             FT_IFMs_[:,:,k]=myshift2D(FT_IFMs[:,:,k],size//2,size//2) #[size//2-size//4:size//2+size//4,size//2-size//4:size//2+size//4]
    if size >= size2 :
        PSDs_=(myshift2D(PSDs,size//2,size//2))[size//2-size2//2:size//2+size2//2,size//2-size2//2:size//2+size2//2] 
        FT_IFMs_=np.zeros([size2,size2,nact],dtype=np.complex128)
        for k in range(0,nact):
            print(k, ' ', end='\r', flush=True)
            FT_IFMs_[:,:,k]=myshift2D(FT_IFMs[:,:,k],size//2,size//2)[size//2-size2//2:size//2+size2//2,size//2-size2//2:size//2+size2//2] 
            #[size//2-size//4:size//2+size//4,size//2-size//4:size//2+size//4]
        
    print('Building matrices by re-dimensioning')
    temp=np.zeros([size2,size2],dtype=np.float32)
    idfx=np.where(temp == 0.)
    MAT_PSD = PSDs_[idfx[0],idfx[1]]
    MAT_A = np.zeros([nact,size2*size2],dtype=np.complex128)
    #MAT_B = np.zeros([size*size,nact],dtype=np.complex128) 
    for k in range(0,nact):
        print(k, ' ', end='\r', flush=True)
        MAT_A[k,:]=FT_IFMs_[idfx[0],idfx[1],k]*MAT_PSD*df**2
    MAT_B=np.conj(FT_IFMs_[idfx[0],idfx[1],:])
        
    print('Summing convolution by Matrix multiplcation')
    MM_stat = MAT_A @ MAT_B
    
    return  MM_stat.real       
    
    
    
    
    
    
# def do_covar_IF_ATM_byMAT(IIs,PSDs,pupils,df):
#     nact=IFMs.shape[2]
#     size=IFMs.shape[0]
#     idx=np.where(pupils!=0)
#     #pupc=np.zeros([size,size],dtype='complex128')
#     pupc=pyfftw.empty_aligned((size,size), dtype='complex128')
#     test=np.fft.fft2(pupc)
#     FT_IFMs=np.zeros([size,size,nact],dtype='complex128')
#     #print('Doing FFTs of IFMs')
#     for k in range(0,nact):
#         #print(k, ' ', end='\r', flush=True)
#         pupc=IFMs[:,:,k]*pupil #myshift2D(IFMs[:,:,k]*pupil,size//2,size//2)
#         FT_IFMs[:,:,k]=np.fft.fft2(pupc)/size
#     #COMPUTE INTEGRAL OF CONVOLUTION IN FOURIER SPACE
#     #MM_stat=np.zeros([nact,nact],dtype=np.float64)
#     MAT_A =    

def plt_plot(a):
    plt.figure()
    plt.plot(a)
    plt.show(block=False)

def plt_imshow(a):
    plt.figure()
    plt.imshow(a,cmap='gray', origin='lower')
    plt.show(block=False)

def expand_a(a,size,idxpup,more,norm):
    if more==0:
        sha=np.zeros([size,size],dtype=np.float64)
        sha[idxpup[0],idxpup[1]]=a/norm
    if more !=0:
        sha=np.zeros([size,size,more],dtype=np.float64)
        sha[idxpup[0],idxpup[1],:]=a/norm
    return sha


def cmplx_mult(a,b,c,d):
    return a*b-c*d,a*d-b*c
   
   # RES_PROD.real=a*b-c*d
   # RES_PROD.imag=a*d-b*c



def expand_a_c(a,size,idxpup,more,norm):
    if more==0:
        sha=np.zeros([size,size],dtype=np.complex128)
        sha[idxpup[0],idxpup[1]]=a/norm
    if more !=0:
        sha=np.zeros([size,size,more],dtype=np.complex128)
        sha[idxpup[0],idxpup[1],:]=a/norm
    return sha


def expand_4fft_c(a,size,idxpup,more,norm,sha):
    if more==0:
        sha[idxpup[0],idxpup[1]]=a/norm
    if more !=0:
        sha[:,idxpup[0],idxpup[1]]=a/norm
   # return sha

   
@numba.jit(nopython = True, parallel=True)
def expand_4fft_numba(a,size,idxpup,more,norm,sha):
    for k in numba.prange(0,more):
        sha[k,idxpup[0],idxpup[1]]=a[k,:]/norm

# def expand_4fft_numba(a,size,idxpup,more,norm,sha):
#     for k in range(0,more):
#         sha[k,idxpup[0],idxpup[1]]=a[k,:]/norm

def expand_4fft_c_para(a,size,idxpup,more,norm,sha,deb,fin):
    if more==0:
        sha[idxpup[0],idxpup[1]]=a/norm
    if more !=0:
        sha[deb:fin,idxpup[0],idxpup[1]]=a[deb:fin,:]/norm
   # return sha



def plt_imshow_expa(a,size,idxpup):
    more=0
    plt.figure()
    plt_imshow(expand_a(a,size,idxpup,more))
    plt.show(block=False)
 

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    print(os.getpid(),'>',process.memory_info().rss/1.e9,'G')

def mytest():
    print("HELLO")


def VK_DSP_up(diameter,r0,L0,size,dim,Pcor,pupil_):
    dom=pupil_.shape[0]
    pupil=np.zeros([size,size],dtype=np.float64)
    pupil[ size//2-dom//2:size//2+dom//2 , size//2-dom//2:size//2+dom//2 ] = pupil_
    fmax=dim/diameter*0.5
    if L0==0:
        L0=1.e6
    fx=np.linspace(-size/2, size/2-1, size)/(size/2) * fmax
    fy=np.linspace(-size/2, size/2-1, size)/(size/2) * fmax
    ffx,ffy=np.meshgrid(fy,fx)
    ffr=np.sqrt(ffx**2+ffy**2)
    cst = (math.gamma(11./6.)**2/(2.*np.pi**(11./3.)))*(24.*math.gamma(6./5.)/5.)**(5./6.)
    PSD_atm=np.zeros([size,size],dtype=np.float64)
    if Pcor == 0:
        PSD_atm [:,:] = cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-11/6)
    pupils=myshift2D(pupil,size//2,size//2)
    ft_pupils=myshift2D(np.fft.fft2(pupils),size//2,size//2)
    pterm = np.abs(ft_pupils)**2
    pterm0 = pterm/np.max(pterm)
    if Pcor == 1:
        PSD_atm [:,:] = (cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-11/6))*(1.-pterm0)
        
    #PSD_atm [:,:] = cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-6)#**(-11/6)
    #PSD_atm[size//2,size//2] = 0.
    df=abs(fx[size//2]-fx[size//2-1])
    
    return PSD_atm, df , pterm





def VK_DSP(diameter,r0,L0,size,dim):
    fmax=dim/diameter*0.5
    if L0==0:
        L0=1.e6
    fx=np.linspace(-size/2, size/2-1, size)/(size/2) * fmax
    fy=np.linspace(-size/2, size/2-1, size)/(size/2) * fmax
    ffx,ffy=np.meshgrid(fy,fx)
    ffr=np.sqrt(ffx**2+ffy**2)
    cst = (math.gamma(11./6.)**2/(2.*np.pi**(11./3.)))*(24.*math.gamma(6./5.)/5.)**(5./6.)
    PSD_atm=np.zeros([size,size],dtype=np.float64)
    PSD_atm [:,:] = cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-11/6)
    #PSD_atm [:,:] = cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-6)#**(-11/6)
    PSD_atm[size//2,size//2] = 0.
    df=abs(fx[size//2]-fx[size//2-1])
    
    return PSD_atm, df

def gendron_covmat(pupil,diameter,r0,L0,size,dim,xact,yact,flag):
    nact=xact.shape[0]
   

    if flag!='light':
        surface = np.pi*(diameter/2.)**2.
        Dphi, rr = VK_STRUCT_F(diameter,r0,L0,size,dim,'2D')
        x=np.linspace(-size/2, size/2-1, size)*(diameter/dim)
        y=x
        xx,yy=np.meshgrid(y,x)
        rr=np.sqrt(xx**2.+yy**2.)
        r=rr[size//2:,size//2]
        dr=np.abs(r[1]-r[0])
        print('DR=',dr)

        FTpupils=np.fft.fft2(np.fft.fftshift(pupil))
        FTDphis=np.fft.fft2(np.fft.fftshift(Dphi))

        convo=np.fft.fftshift((np.fft.fft2(FTpupils*FTDphis)).real)

        F0=np.sum(Dphi*pupil*dr**2.)/surface 
        Fx=convo/convo[size//2,size//2]*F0

        idxpup=np.where(pupil == 1.0)
        tpup=np.sum(pupil)
        nxp=len(idxpup[0])

        DELTA2 = 1./(2.*surface)*np.sum(Fx*pupil*dr**2)
        
        #COV_r_s=np.zeros([nxp,nxp],dtype=np.float64)
        Fx_Fy_term = np.zeros([nact,nact],dtype=np.float64)
    
        for k in range(0,nact):
            print(k, ' ', end='\r', flush=True)
            for j in range(k,nact):
                arr2m=(xx-xact[k])**2.+(yy-yact[k])**2.
                idxk=np.unravel_index(arr2m.argmin(), arr2m.shape)
                arr2m=(xx-xact[j])**2.+(yy-yact[j])**2.
                idxj=np.unravel_index(arr2m.argmin(), arr2m.shape)
                Fx_Fy_term[k,j]=Fx[idxk[0],idxk[1]]+Fx[idxj[0],idxj[1]]
                Fx_Fy_term[j,k]=Fx_Fy_term[k,j]
        
        Dphi_term=np.zeros([nact,nact],dtype=np.float64)

        if L0 >= 1.e4 or L0==0.:
            for k in range(0,nact):
                print(k, ' ', end='\r', flush=True)
                for j in range(k,nact):
                    rs=np.sqrt( ( xact[j] - xact[k]  )**2 + (yact[j] -yact[k]   )**2 )     
                    Dphi_term [k,j] = 6.88*( abs(  rs  )/r0   )**(5./3.)
                    Dphi_term [j,k] = Dphi_term [k,j] 

            
    
        if L0 < 1.e4 and L0!=0.:
            
            Dphir,r=VK_STRUCT_F(diameter,r0,L0,size,dim,'1D')
            
            for k in range(0,nact):
                print(k, ' ', end='\r', flush=True)
                for j in range(k,nact):
                    rs=np.sqrt( ( xact[j] - xact[k]  )**2 + (yact[j] -yact[k]   )**2 )
                    arr2m=(np.abs(r-rs))
                    #rw=np.unravel_index(arr2m.argmin(), arr2m.shape)
                    rw=arr2m.argmin()
                    Dphi_term [k,j] = Dphir[rw]
                    Dphi_term [j,k] = Dphi_term [k,j]
                    
    
    
    
    if flag=='heavy':
        covmat = 0.5* (  Fx_Fy_term -  Dphi_term - DELTA2   )
        
    
    if flag=='light':
        covmat=geo_covmat(diameter, r0, xact,yact)
        DELTA2=0.
        
        
    return covmat, DELTA2 #Fx_Fy_term #covmat


def geo_covmat(Diameter, R0, x,y):
    nact=x.shape[0]
    covmat=np.zeros([nact,nact],dtype='float64')
    dist_X=np.zeros([nact,nact],dtype='float64')
    dist_Y=np.zeros([nact,nact],dtype='float64')
    
    for k in range(0, nact):
        print(k, ' ', end='\r', flush=True)
        for j in range(k, nact):
            dist_X[k,j] = np.abs(x[k] - x[j]) 
            dist_Y[k,j] = np.abs(y[k] - y[j])
            dist_X[j,k] = dist_X[k,j]
            dist_Y[j,k] = dist_Y[k,j]
            
    rr=np.sqrt(dist_X**2+dist_Y**2)
    dvec=rr/R0
    covmat = np.ascontiguousarray(6.88*dvec**(5./3.))
    
    #fits.writeto('covmat.fits',covmat,overwrite=True)
    
    return covmat

def geo_covmat_fast(Diameter, R0, x,y):
    nact=x.shape[0]
    covmat=np.zeros([nact,nact],dtype='float64')
    
    xx=(np.tile(x,[nact,1]))
    yy=(np.tile(y,[nact,1]))
    dist_X =np.abs( xx - xx.T)
    dist_Y =np.abs( yy - yy.T)        
    rr=np.sqrt(dist_X**2+dist_Y**2)
    dvec=rr/R0
    sig2=1.03*(Diameter/R0)**(5./3.)
    dphi=6.88*dvec**(5./3.)
    covmat = np.ascontiguousarray(sig2-0.5*dphi)
    return covmat
    #covmat = np.ascontiguousarray(np.mean(dphi)-dphi)



def VK_STRUCT_F(diameter,r0,L0,size,dim,forma):
    if L0 >= 1.e4 or L0==0.:
        Dphi, rr = KOL_STRUCT_F(diameter,r0,L0,size,dim,'2D')
    if L0<1.e4 and L0!=0.:
        #geo=mkp(size/dim*diameter,size,diameter,0.)
        x=np.linspace(-size/2, size/2-1, size)*(diameter/dim)
        y=x
        xx,yy=np.meshgrid(y,x)
        rr=np.sqrt(xx**2.+yy**2.)
        #rr[size//2,size//2]=np.min(rr)
        
        
        if forma=='2D':
            fbesselk=np.zeros([size,size],dtype=np.float64)
            for k in range(0,size):
                print(k, ' ', end='\r', flush=True)
                for j in range(k,size):
                    fbesselk[k,j]=mpmath.besselk(5./6.,2*np.pi*rr[k,j]/L0)
                    fbesselk[j,k] = fbesselk[k,j]
            fbesselk[size//2,size//2]=0.
            rr[size//2,size//2]=0.    
            
            Dphi= 2*math.gamma(11./6.)/2**(5./6.)/np.pi**(8./3.) * \
            (24./5.*math.gamma(6./5.))**(5./6.) * (r0/L0)**(-5./3.)* \
            (math.gamma(5./6.)/2**(1./6.)  - (2*np.pi*rr/L0)**(5./6.) * fbesselk)
            Dphi[size//2,size//2]=0.
            
        if forma=='1D':
            r=rr[size//2:,size//2]
            fbesselk=np.zeros([size//2],dtype=np.float64)
            for k in range(0,size//2):
                 fbesselk[k]=mpmath.besselk(5./6.,2*np.pi*r[k]/L0)   
            Dphi= 2*math.gamma(11./6.)/2**(5./6.)/np.pi**(8./3.) * \
            (24./5.*math.gamma(6./5.))**(5./6.) * (r0/L0)**(-5./3.)* \
            (math.gamma(5./6.)/2**(1./6.)  - (2*np.pi*r/L0)**(5./6.) * fbesselk)
            Dphi[0]=0.
    
    if forma=='2D':
        return Dphi, rr
    if forma=='1D': 
        return Dphi, r
    
def KOL_STRUCT_F(diameter,r0,L0,size,dim,forma):
    x=np.linspace(-size/2, size/2-1, size)*(diameter/dim)
    y=x
    xx,yy=np.meshgrid(y,x)    
    rr=np.sqrt(xx**2.+yy**2.)
    Dphi=6.88*(rr/r0)**(5./3.)
    rr.shape
    if forma=='2D':
        return Dphi, rr
    if forma=='1D': 
        return Dphi[size/2:,size/2], rr[size/2:,size/2]
    
    

def COVAR_FS(diameter,r0,L0,size,dim,pupil):
    
    surface = np.pi*(diameter/2.)**2.
    Dphi, rr = VK_STRUCT_F(diameter,r0,L0,size,dim,'2D')
    
    x=np.linspace(-size/2, size/2-1, size)*(diameter/dim)
    y=x
    xx,yy=np.meshgrid(y,x)
    rr=np.sqrt(xx**2.+yy**2.)
    r=rr[size//2:,size//2]
    dr=np.abs(r[1]-r[0])
    print('DR=',dr)
    
    FTpupils=np.fft.fft2(np.fft.fftshift(pupil))
    FTDphis=np.fft.fft2(np.fft.fftshift(Dphi))
    
    convo=np.fft.fftshift((np.fft.fft2(FTpupils*FTDphis)).real)
    
    F0=np.sum(Dphi*pupil*dr**2.)/surface 
    Fx=convo/convo[size//2,size//2]*F0
    
    idxpup=np.where(pupil == 1.0)
    tpup=np.sum(pupil)
    nxp=len(idxpup[0])
    
    DELTA2 = 1./(2.*surface)*np.sum(Fx*pupil*dr**2)
    
    COV_r_s=np.zeros([nxp,nxp],dtype=np.float64)
    Dphi_term=np.zeros([nxp,nxp],dtype=np.float64)
    
    if L0 >= 1.e4 or L0==0.:
        for k in range(0,nxp):
            print(k, ' ', end='\r', flush=True)
            for j in range(k,nxp):
                rs = np.sqrt( ( xx[idxpup[0][j],idxpup[1][j]]  - xx[idxpup[0][k],idxpup[1][k]])**2+( yy[idxpup[0][j],idxpup[1][j]]  - yy[idxpup[0][k],idxpup[1][k]])**2  )          
                Dphi_term [k,j] = 6.88*( abs(  rs  )/r0   )**(5./3.)
                Dphi_term [j,k] = Dphi_term [k,j] 
    
    
    
    if L0<1.e4 and L0!=0:
        for k in range(0,nxp):
            print(k, ' ', end='\r', flush=True)
            for j in range(k,nxp):
                rs = np.sqrt( ( xx[idxpup[0][j],idxpup[1][j]]  - xx[idxpup[0][k],idxpup[1][k]])**2+( yy[idxpup[0][j],idxpup[1][j]]  - yy[idxpup[0][k],idxpup[1][k]])**2  )
                fbesselk=mpmath.besselk(5./6.,2*np.pi*rs/L0)
                Dphi_term [k,j] =  2*math.gamma(11./6.)/2**(5./6.)/np.pi**(8./3.) * \
        (24./5.*math.gamma(6./5.))**(5./6.) * (r0/L0)**(-5./3.)* \
        (math.gamma(5./6.)/2**(1./6.)  - (2*np.pi*rs/L0)**(5./6.) * fbesselk)    
                Dphi_term [j,k] = Dphi_term [k,j]   
                
    Fx_Fy_term = np.zeros([nxp,nxp],dtype=np.float64)
    
    
    for k in range(0,nxp):
        print(k, ' ', end='\r', flush=True)
        for j in range(k,nxp):
            Fx_Fy_term[k,j]=Fx[idxpup[0][k],idxpup[1][k]] + Fx[idxpup[0][j],idxpup[1][k]]
            Fx_Fy_term[j,k]=Fx_Fy_term[k,j]
    
    COV_r_s = 0.5* (  Fx_Fy_term -  Dphi_term - DELTA2   )
            
    return  COV_r_s, DELTA2 #COV_MAT


# def full_covmat(Diameter, R0, , L0, x,y):
#     nact=x.shape[0]
#     covmat=np.zeros([nact,nact],dtype='float64')
#     dist_X=np.zeros([nact,nact],dtype='float64')
#     dist_Y=np.zeros([nact,nact],dtype='float64')
    
#     for k in range(0, nact):
#         for j in range(0, nact):
#             dist_X[k,j] = x[k] - x[j] 
#             dist_Y[k,j] = y[k] - y[j] 
            
#     rr=np.sqrt(dist_X**2+dist_Y**2)
#     if L0==-1: # KOLMOGOROV
#         dvec=rr/R0
#         Dphi = 6.88*dvec**(5./3.)
#     else :
#         Dphi=0.023*R0**(-5./3.)*()**()
        
    
#     covmat=Dphi    
#     return covmat

#Shift 2D arrays
def myshift1D(vector,xshift):
    return np.roll(vector,int(xshift))
                   
def myshift2D(image,xshift,yshift):
    return np.roll(np.roll(image,int(xshift),0),int(yshift),1)

#Create binary pupil with central obscuration
def makepupil(dim_x, diam, eps, xc=None, yc=None, dim_y=None) :#, YC=yc, DIM_Y=dim_y):
    # by default the pupil is centered on the center of the array
    # (i.e. NOT on a pixel for a square array…)
    if (xc is None)   : xc    = (dim_x-1)/2.
    if (dim_y is None): dim_y = dim_x
    if (yc is None)   : yc    = (dim_y-1)/2.
    
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    
    dummy = np.sqrt(xx**2.+yy**2.)/(diam/2.)
    #return dummy

    #pupil=dummy.copy()*0.
    
    # pupil array is 1 under the pupil & 0 elsewhere
    #pupil[dummy < 1.]=1.0
    
    pup1 = 1*(dummy < 1 ) * 1*(dummy >= eps) 
    
    return pup1

def toto():
    print("TOTO")

    
def mkp_(dim_x, diam, eps, xc=None, yc=None, dim_y=None): #, YC=yc, DIM_Y=dim_y):
    
    # by default the pupil is centered on the center of the array
    # (i.e. NOT on a pixel for a square array…)
    if (xc is None)   : xc    = (dim_x-1)/2.
    if (dim_y is None): dim_y = dim_x
    if (yc is None)   : yc    = (dim_y-1)/2.
    
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    
    dummy = np.sqrt(xx**2.+yy**2.)/(diam/2.)
    
    pupil=dummy.copy()*0.
    # pupil array is 1 under the pupil & 0 elsewhere
    idx1=np.where(dummy < 1.)
    idx2=np.where(dummy < eps)
    pupil[idx1]=1.0
    pupil[idx2]=0.
    return np.intc(pupil)

def mkp2(real_sz, dim_x, diam, eps, xc=None, yc=None, dim_y=None): #, YC=yc, DIM_Y=dim_y):
    class geom:
        pass
    
    # by default the pupil is centered on the center of the array
    # (i.e. NOT on a pixel for a square array…)
    if (xc is None)   : xc    = (dim_x-1)/2.
    if (dim_y is None): dim_y = dim_x
    if (yc is None)   : yc    = (dim_y-1)/2.
    
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    
    dummy = np.sqrt(xx**2.+yy**2.)/(diam/2.)
    
    pupil=dummy.copy()*0.
    # pupil array is 1 under the pupil & 0 elsewhere
    idx1=np.where(dummy < 1.)
    idx2=np.where(dummy < eps)
    pupil[idx1]=1.0
    pupil[idx2]=0.
    geomr=geom()
    geomr.pupil=np.intc(pupil)
    geomr.xx=xx
    geomr.yy=yy
    return geomr
    #return np.intc(pupil)
    
    
def mkp(real_sz,dim_x, diam, eps, xc=None, yc=None, dim_y=None): #, YC=yc, DIM_Y=dim_y):
    class geom:
        pass
    
    # by default the pupil is centered on the center of the array
    # (i.e. NOT on a pixel for a square array…)
    if (xc is None)   : xc    = (dim_x-1)/2. #*1.*real_sz/dim_x 
    if (dim_y is None): dim_y = dim_x
    if (yc is None)   : yc    = (dim_y-1)/2. #*1.*real_sz/dim_x 
    
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    
    xx=xx*real_sz/(dim_x-1)
    yy=yy*real_sz/(dim_x-1)
    
    dummy = np.sqrt(xx**2.+yy**2.) #/(diam/2.)
    
    pupil=dummy.copy()*0.
    # pupil array is 1 under the pupil & 0 elsewhere
    idx1=np.where(dummy < diam/2.)
    idx2=np.where(dummy < eps*diam/2)
    pupil[idx1]=1.0
    pupil[idx2]=0.
    geomr=geom()
    geomr.pupil=np.intc(pupil)
    geomr.xx=xx
    geomr.yy=yy
    return geomr
   
    
def give_zernike(geom, diameter, nmodes, norming=None, meaning=None):
    
    max_z=nmodes
    
    n=(geom.xx).shape[0]
    
    
    modz = np.zeros([n,n,nmodes], dtype = np.float64)
    
    x=geom.xx[n//2,:]/(diameter/2)
    y_=geom.yy[:,n//2]/(diameter/2)
    
    x_pow_2 = x**2
    
    ########### PART COPIED FROM prop_zernike.py
    
    zlist, maxrp, maxtc = prop_noll_zernikes(max_z, COMPACT = True, EXTRA_VALUES = True)
    
    for numz in range(1,max_z+1):
        dmap = np.zeros([n,n], dtype = np.float64)
        
        for j in range(n):
            ab = np.zeros(n, dtype = np.float64)
            y = y_[j]   #(j - n//2) * proper.prop_get_sampling(a) / beam_radius
            r = np.sqrt(x_pow_2 + y**2)
            t = np.arctan2(y,x)
        
            # predefine r**power, cos(const*theta), sin(const*theta) vectors
            for i in range(2, maxrp+1):
                rps = str(i).strip()
                cmd = "r_pow_" + rps + " = r**i"
                exec(cmd) 
                
            for i in range(1, maxtc+1):
                tcs = str(i).strip()
                cmd = "cos" + tcs + "t = np.cos(i*t)"
                exec(cmd)
                cmd = "sin" + tcs + "t = np.sin(i*t)"
                exec(cmd)
        
            tmp = eval(zlist[numz])        
            dmap[j,:] = tmp
            
        modz[:,:,numz-1] = dmap
    return modz
    
    
#computes statistics on arrays like WFs
def mean(input,idx):
    meann=np.mean(input[idx])
    return meann
        
def ptv(input,idx):
    meann=np.max(input[idx]) - np.min(input[idx])
    return meann
        
def std(input,idx):
    meann=np.std(input[idx])
    return meann

def st2(input,idx):
    meann=np.mean(input[idx]**2)
    return meann

def analyse(input, idx):
    class result:
        pass
    
    n=1
    
    if input.ndim == 3: 
        n=input.shape[2]

    if n == 1:

        moyy=mean(input,idx)
        stdd=std(input,idx)
        ptvv=ptv(input,idx)
        st22=np.zeros(n)
        
        print('MEAN = ', moyy)
        print('STANDART DEVIATION = ', stdd)
        print('PtV = ', ptvv)

    if n >= 1:
        moyy=np.zeros(n)
        stdd=np.zeros(n)
        ptvv=np.zeros(n)
        st22=np.zeros(n)
        
#         if n ==1:
#             inpu=input.deepcopy
#             input=np.zeros([dim,dim,1])
#             input[:,:,0]=inpu
  
        for k in range(0,n):
            print(k, ' ', end='\r', flush=True)
            moyy[k]=mean(input[:,:,k],idx)  
            stdd[k]=std(input[:,:,k],idx)  
            ptvv[k]=ptv(input[:,:,k],idx)  
            st22[k]=st2(input[:,:,k],idx)  

        print('MEAN OVER ',n,' = ',np.mean(moyy))        
        print('STANDART DEVIATION OVER ', n, ' = ',np.sqrt(np.mean(stdd**2.))) 
        print('PtV OVER ', n, np.mean(ptvv)) 

    res=result()
    res.mean=moyy
    res.std=stdd
    res.ptv=ptvv
    res.st2=ptvv
    
    return res

    
def analyse_z(input, modz, idx, IMETHOD):
    class result:
        pass
    
    n=1
    
    if input.ndim == 3: 
        n=input.shape[2]
    
    dim=input.shape[0]   
    
    npup=np.zeros([dim,dim])
    npup[idx]=1.0
    
    if modz.ndim == 3:
        nmodes=modz.shape[2]
        
    if np.size(IMETHOD) > 2:
        IMM = IMETHOD
        IMETHOD = 0
    
    else:
        
        if IMETHOD == 1: 
            IMM = np.eye(nmodes)
        
        #if IMETHOD == 0:
            #IMM = invert(build_MM(modz,npup,nmodes))
    
    if n == 1:

        moyy=mean(input,idx)
        stdd=std(input,idx)
        ptvv=ptv(input,idx)
     
        print('MEAN = ', moyy)
        print('STANDART DEVIATION = ', stdd)
        print('PtV = ', ptvv)
        input
        
    if n >= 1:
        moyy=np.zeros(n)
        stdd=np.zeros(n)
        ptvv=np.zeros(n)
        C_z=np.zeros([nmodes,n])
        ay=np.zeros(nmodes)
        
        if n ==1:
            inpu=input.copy()
            input=np.zeros([dim,dim,2],dtype='float64')
            input[:,:,0]=inpu

        for k in range(0,n):
            print(k, ' ', end='\r', flush=True)
            moyy[k]=mean(input[:,:,k],idx)  
            stdd[k]=std(input[:,:,k],idx)  
            ptvv[k]=ptv(input[:,:,k],idx)
            
            for nn in range(0,nmodes):
                ay[nn] = np.sum(input[:,:,k]*modz[:,:,nn]*npup)/np.sum(npup)
            
            C_z[:,k]=IMM@ay
        
        print('MEAN OVER ',n,' = ',np.mean(moyy))        
        print('STANDART DEVIATION OVER ', n, ' = ',np.sqrt(np.mean(stdd**2.))) 
        print('PtV OVER ', n, np.mean(ptvv)) 

    res=result()
    res.mean=moyy
    res.std=stdd
    res.ptv=ptvv
    res.C_z=C_z
    
    return res

def scalar_prod(input, modz, npup):
    # npup is the pupil
    # projects on the number of modes in modz
    # the output vec is created
    nmodes=modz.shape[2]
    vec=np.zeros(nmodes,dtype='float32')
    tpup=np.sum(npup)
    for nn in range(0,nmodes):
       # vec[nn] = np.sum(input[:,:,k]*modz[:,:,nn]*npup)/np.sum(npup)
        vec[nn] = np.sum(input*modz[:,:,nn]*npup)/tpup
    return vec


def scalar_prod_4P0(input, modz, npup, vec, rank):
    # npup is the pupil
    # projects on the number of modes in modz (in // each process sees a part of modz)
    # the output vec is created outside
    # the rank gives the stating position in vec where to store the projection
    # supposes that the size of vec is dividable by the number of processes
    #nmodes=modz.shape[2]
    #vec=np.zeros(nmodes,dtype='float32')
    tpup=np.sum(npup)
    #for nn in range(0,nmodes):
        #vec[nn] = np.sum(input[:,:,k]*modz[:,:,nn]*npup)/np.sum(npup)
    #    vec[rank*nmodes+nn] = np.sum(input*modz[:,:,nn]*npup)/tpup
    memory_usage_psutil()
    time.sleep(10)
    #return vec




def scalar_prod_4P(input, modz, npup, vec, rank):
    # npup is the pupil
    # projects on the number of modes in modz (in // each process sees a part of modz)
    # the output vec is created outside
    # the rank gives the stating position in vec where to store the projection
    # supposes that the size of vec is dividable by the number of processes
    nmodes=modz.shape[2]
    #vec=np.zeros(nmodes,dtype='float32')
    tpup=np.sum(npup)
    for nn in range(0,nmodes):
        #vec[nn] = np.sum(input[:,:,k]*modz[:,:,nn]*npup)/np.sum(npup)
        vec[rank*nmodes+nn] = np.sum(input*modz[:,:,nn]*npup)/tpup
    #memory_usage_psutil()
    #time.sleep(20)
    #return vec

def scalar_prod_4Pidx(input, modz, idx, vec, rank):
    nmodes=modz.shape[2]
    tpup=len(idx[0])
    for nn in range(0,nmodes):
        a=modz[idx[0],idx[1],nn]
        b=input[idx[0],idx[1]]
        vec[rank*nmodes+nn] = np.sum(a*b)/tpup
    #memory_usage_psutil()
  


def scalar_prod_4P2(input, modz, npup, vec,idxm):
    # npup is the pupil
    # modz is passed entirely
    # projects on the modes listed in idxm and stored accordingly in vec
    # the output vec is created outside
    tpup=np.sum(npup)
    for nn in idxm:
        vec[nn] = np.sum(input*modz[:,:,nn]*npup)/tpup 
        #vec[nn] = np.sum(input*modz[:,:,nn]*npup)/tpup    
        #vec[nn] = np.sum(modz[:,:,nn].dot(npup.dot(input)))/tpup
    #memory_usage_psutil()

    
def scalar_prod_4P3(input, modz, npup,idxm):
    ## npup is the pupil
    # modz is passed entirely
    # projects on the modes listed in idxm and stored accordingly in vec
    # difference with 4P2 is that vec is created in the function and returned (to be used with pool)
    # in that case the sum of all returns is the final
    nmo=modz.shape[2]
    vec=np.zeros([nmo])
    tpup=np.sum(npup)
    for nn in idxm:
        vec[nn] = np.sum(input*modz[:,:,nn]*npup)/tpup    
    memory_usage_psutil()
    return vec

def scalar_prod_4P4(input, modz, npup,i):
    ## npup is the pupil
    # modz is passed entirely
    # projects on the modes listed in idxm and stored accordingly in vec
    # difference with 4P2 is that vec is created in the function and returned (to be used with pool)
    # in that case the sum of all returns is the final
    nmo=modz.shape[2]
    tpup=np.sum(npup)
    #print(i)
    vec = np.sum(input*modz[:,:,i]*npup)/tpup    
    #memory_usage_psutil()
    return vec


###### PARALLEL CODE

def scalar_prod_4Pid(input_,modz_, vec, rank, nmoEXT,rest):
    
    if nmoEXT==0:
        nmodes=modz_.shape[1]
        tpup=input_.shape[0]
        for nn in range(0,nmodes):
        #print(nn,rank,nmo4rank,nn)
            vec[rank*nmodes+nn] = np.sum(input_*modz_[:,nn])/tpup
    if nmoEXT != 0:
        tpup=input_.shape[0]
        for nn in range(0,rest):
            vec[nmoEXT-rest+nn] = np.sum(input_*modz_[:,nn])/tpup
            
    #if nmodes==1:
    #    print('nmodes is 1 and value is:',vec[rank*nmodes+nn])
    #memory_usage_psutil()

def proj_asb_para(a,b,nprocs):
    nmodes=b.shape[1]
    nper=nmodes//nprocs
    rest=nmodes-nper*nprocs
    #print('REST = ',rest)
    jobs = []
    vecPid=mp.Array('d', nmodes)
    for kkk in range(0,nprocs):
        rank=kkk
        p = mp.Process(target=scalar_prod_4Pid, args=(a,b[:,rank*nper:(rank+1)*nper], vecPid,rank,0,0))
        jobs.append(p)
        p.start()
    if rest > 0:
        #print('JY SUIS!')
        rank=nprocs
        p = mp.Process(target=scalar_prod_4Pid, args=(a,b[:,rank*nper:rank*nper+rest], vecPid,rank,nmodes,rest))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
        
    return vecPid

# def proj_asb_para_fidx(a,b,idxk,nprocs):
#     nmodes=b.shape[1]
#     nper=nmodes//nprocs
#     rest=nmodes-nper*nprocs
#     #print('REST = ',rest)
#     jobs = []
#     vecPid=mp.Array('d', nmodes)
#     for kkk in range(0,nprocs):
#         rank=kkk
#         p = mp.Process(target=scalar_prod_4Pid, args=(a,b[:,rank*nper:(rank+1)*nper], vecPid,rank,0,0))
#         jobs.append(p)
#         p.start()
#     if rest > 0:
#         #print('JY SUIS!')
#         rank=nprocs
#         p = mp.Process(target=scalar_prod_4Pid, args=(a,b[:,rank*nper:rank*nper+rest], vecPid,rank,nmodes,rest))
#         jobs.append(p)
#         p.start()
#     for proc in jobs:
#         proc.join()
        
#     return vecPid




def analyse_z_para(inpu, modz, idx, IMETHOD, nprocs):
    class result:
        pass
    
#############INIT############
    n=1
    
    if inpu.ndim == 3: 
        n=inpu.shape[2]
    
    dim=inpu.shape[0]   
    
#     npup=np.zeros([dim,dim],dtype=np.float64)
#     npup[idx]=1.0
    
    if modz.ndim == 3:
        nmodes=modz.shape[2]
        
    if np.size(IMETHOD) > 2:
        IMM = IMETHOD
        IMETHOD = 0
    
    else:
        if IMETHOD == 1: 
            IMM = np.eye(nmodes)


##PROJECT ON ALL MODES FIRST
    if nmodes >= n:
        C_z=np.zeros([nmodes,n],dtype=np.float64)
        #ay=np.zeros(nmodes)
        a=inpu[idx[0],idx[1],:]
        b=modz[idx[0],idx[1],:]
        
        for k in range(0,n):
            print(k, ' ', end='\r', flush=True)
            ay=proj_asb_para(a[:,k],b,nprocs)
            C_z[:,k]=IMM@ay


##PROJECT ON ALL ITERATIONS  FIRST
    if nmodes < n:
        C_z=np.zeros([nmodes,n],dtype=np.float64)
        ya=np.zeros([n,nmodes],dtype=np.float64)
        a=inpu[idx[0],idx[1],:]
        b=modz[idx[0],idx[1],:]
        
        for k in range(0,nmodes):
            ya[:,k]=proj_asb_para(b[:,k],a,nprocs)
           
        #print('Matrix-Vectoring')
        for ll in range(0,n):
            C_z[:,ll]=IMM@ya[ll,:]
#       print('not yet available')
        #print('DONE')
    
    res=result()
    res.C_z=C_z
    return res





def analyse_z_MAT(inpu, modz, idx, IMETHOD, nprocs):

#############INIT############
    n=1
    
    if inpu.ndim == 3: 
        n=inpu.shape[2]
    
    dim=inpu.shape[0]   
  
    if modz.ndim == 3:
        nmodes=modz.shape[2]
        
    if np.size(IMETHOD) > 2:
        IMM = IMETHOD
        IMETHOD = 0
    
    else:
        if IMETHOD == 1: 
            IMM = np.eye(nmodes)
    
    # PREPARE MATRICES AND COMPUTE
    tpup=len(idx[0])
    if inpu.ndim == 3: 
        MAT_INP=inpu[idx[0],idx[1],:]/tpup
    
    if inpu.ndim == 2:
        MAT_INP=np.zeros([len(idx[0]),1])
        MAT_INP[:,0]=inpu[idx[0],idx[1]]/tpup
    
    MAT_MODZ_T=np.transpose(modz[idx[0],idx[1],:])
    
    C_z = IMM @ MAT_MODZ_T @ MAT_INP
    
    return C_z

# ##PROJECT ON ALL MODES FIRST
#     if nmodes >= n:
#         C_z=np.zeros([nmodes,n],dtype=np.float64)
#         #ay=np.zeros(nmodes)
#         a=inpu[idx[0],idx[1],:]
#         b=modz[idx[0],idx[1],:]
        
#         for k in range(0,n):
#             print(k, ' ', end='\r', flush=True)
#             ay=proj_asb_para(a[:,k],b,nprocs)
#             C_z[:,k]=IMM@ay


# ##PROJECT ON ALL ITERATIONS  FIRST
#     if nmodes < n:
#         C_z=np.zeros([nmodes,n],dtype=np.float64)
#         ya=np.zeros([n,nmodes],dtype=np.float64)
#         a=inpu[idx[0],idx[1],:]
#         b=modz[idx[0],idx[1],:]
        
#         for k in range(0,nmodes):
#             ya[:,k]=proj_asb_para(b[:,k],a,nprocs)
           
#         #print('Matrix-Vectoring')
#         for ll in range(0,n):
#             C_z[:,ll]=IMM@ya[ll,:]
# #       print('not yet available')
#         #print('DONE')
    
#     res=result()
#     res.C_z=C_z
#     return res







# def analyse_z_para_fidx(inpu, modz, idxk, IMETHOD, nprocs):
#     class result:
#         pass
    
# #############INIT############
#     n=1
    
#     if inpu.ndim == 3: 
#         n=inpu.shape[2]
    
#     dim=inpu.shape[0]   
    
# #     npup=np.zeros([dim,dim],dtype=np.float64)
# #     npup[idx]=1.0
    
#     if modz.ndim == 3:
#         nmodes=modz.shape[2]
        
#     if np.size(IMETHOD) > 2:
#         IMM = IMETHOD
#         IMETHOD = 0
    
#     else:
#         if IMETHOD == 1: 
#             IMM = np.eye(nmodes)


# ##PROJECT ON ALL MODES FIRST
#     if nmodes >= n:
#         C_z=np.zeros([nmodes,n],dtype=np.float64)
#         #ay=np.zeros(nmodes)
#         a=inpu[idx[0],idx[1],:]
#         b=modz[idx[0],idx[1],:]
        
#         for k in range(0,n):
#             ay=proj_asb_para(a[:,k],b,nprocs)
#             C_z[:,k]=IMM@ay


# ##PROJECT ON ALL ITERATIONS  FIRST
#     if nmodes < n:
#         C_z=np.zeros([nmodes,n],dtype=np.float64)
#         ya=np.zeros([n,nmodes],dtype=np.float64)
#         a=inpu[idx[0],idx[1],:]
#         b=modz[idx[0],idx[1],:]
        
#         for k in range(0,nmodes):
#             ya[:,k]=proj_asb_para(b[:,k],a,nprocs)
           
#         #print('Matrix-Vectoring')
#         for ll in range(0,n):
#             C_z[:,ll]=IMM@ya[ll,:]
# #       print('not yet available')
#         #print('DONE')
    
#     res=result()
#     res.C_z=C_z
#     return res




# def scalar_prod_4P(input, modz, npup):
    
#     nmodes=modz.shape[2]
#     vec=np.zeros(nmodes,dtype=np.float32)
#     for nn in range(0,nmodes):
#         vec[nn] = np.sum(input[:,:,k]*modz[:,:,nn]*npup)/np.sum(npup)
    
#     return vec



def build_mm(defo,idx):
    nact=defo.shape[2]
    tpup=len(idx[0])
    MM=np.zeros([nact,nact],dtype='float64')
    
    for k in range(0, nact):
        #print(k,' ', end = '')
        print(k, ' ', end='\r', flush=True)
        for j in range(0, nact):
            a=defo[idx[0],idx[1],k]
            b=defo[idx[0],idx[1],j]
            MM[k,j] = np.sum(a*b)/tpup
            #MM[j,k] = MM[k,j]
            
#             a=defo[:,:,k]
#             b=defo[:,:,j]
#             MM[k,j] = np.sum(a[idx]*b[idx])/tpup
        
    
    #print('DONE')
    
    return MM

def build_mm2(defo,idx):
    nact=defo.shape[2]
    tpup=len(idx[0])
    MM=np.zeros([nact,nact],dtype='float64')
    defo_=defo[idx[0],idx[1],:]
    
    for k in range(0, nact):
        #print(k,' ', end = '')
        print(k, ' ', end='\r', flush=True)
        for j in range(k, nact):
            #a=defo_[:,k]
            #b=defo_[:,j]
            #MM[k,j] = np.sum(a*b)/tpup
            MM[k,j] = np.sum(defo_[:,k]*defo_[:,j])/tpup
            MM[j,k] = MM[k,j]
            
#             a=defo[:,:,k]
#             b=defo[:,:,j]
#             MM[k,j] = np.sum(a[idx]*b[idx])/tpup
        
    
    #print('DONE')
    
    return MM

#IFlin2shape(IFM, posi, posj, CMDs, size)

def build_modes_cmd(MM):
    u, s, vh = np.linalg.svd(MM)
    nmo=MM.shape[0]
    cmd_modes=vh.copy()
    for k in range(0,nmo):
        cmd_modes[k,:]=cmd_modes[k,:]/np.sqrt(s[k])
    # vh are the modes in command space with 1st index = rank of the mode
    return cmd_modes

def build_modes_phase(defo, cmd_modes,posi,posj,size):
    nact=defo.shape[2]
    dimact=defo.shape[0]
    nmo=cmd_modes.shape[0]
    phase_modes=np.zeros([size,size,nmo],dtype='float32')
    
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        phase_modes[:,:,k]=IFlin2shape(defo, posi, posj, cmd_modes[k,:], size)
        
    return phase_modes


# def build_mm_(defo,idx,klist,jlist)
#     tpup=len(idx[0])
#     MM=np.zeros([nact,nact])




# @jit(nopython=True)
# def build_mm_para(defo,idx):
#     nact = 100
#     return nact
 
# @jit(nopython=True) 
# def build_mm_para(defo,idx):
#     nact = 100
#     return nact

#nact,tpup,defo, idxpup, MM

#(nopython=True)

#ONE CASE THAT WORKS BUT IS SLOW
# @jit(nopython=True)
# def build_mm_para(nact,tpup,defo,pup,MM):
#     val=nact
#     for k in range(0, nact):         
#         print(k)
#         for j in range(0, nact):
            
#             #a=defo[idx[0],idx[1],k]
#             #b=defo[idx[0],idx[1],j]
#             a=defo[:,:,k]*pup
#             b=defo[:,:,j]*pup
#             MM[k,j] = np.sum(a*b)/tpup
            
#     return MM

@jit(nopython=True)
def build_mm_para(nact,tpup,defo_,MM):

    for k in range(0, nact):         
#        print(k)
        #print(k, ' ', end='\r', flush=True)
        for j in range(k, nact):
            a=defo_[:,k]
            b=defo_[:,j]
            MM[k,j] = np.sum(a*b)/tpup
            MM[j,k] = MM[k,j]
    return MM



# @jit(nopython=True) 
# def build_mm_para(nact,tpup,defo,idx,MM):
    
#     for k in range(0, 10):
#         k
#     return MM        
        
        #MM=k
        #print(k,' ', end = '')
        #print(k, ' ', end='\r', flush=True)
#         for j in range(0, nact):
#             a=defo[:,:,k]
#             b=defo[:,:,j]
#             MM[k,j] = np.sum(a[idx]*b[idx])/tpup
        
    #print('DONE')

    
def IFlin2shape(IFM, posi, posj, CMDs, size):
    
    shape=np.zeros([size,size])
    nact=posi.shape[0]
    small=IFM.shape[0]
    
    for k in range(0,nact):
        shape[posi[k] - small//2 : posi[k] + small//2, posj[k] - small//2 : posj[k] + small//2] += CMDs[k]*IFM[:,:,k]
        
    return shape



#def IF_MM(IFM,posx,posy,pupil,dmax):
#    return MM

def geo_KL(posx,posy):
    nact=posx.shape[1]
    dist_matrix = np.zeros([nact, nact])
    
    
def prop_noll_zernikes(maxz, **kwargs):
    """Return a string array in which each element contains the Zernike polynomial
    equation corresponding to the index of that element.

    The polynomials are orthonormal for an unobscured circular aperture. They
    follow the ordering convention of Noll (J. Opt. Soc. America, 66, 207 (1976)).
    The first element (0) is always blank. The equations contain the variables "r"
    (normalized radius) and "t" (azimuth angle in radians). The polynomials have
    an RMS of 1.0 relative to a mean of 0.0.

    Parameters
    ----------
    maxz : int
        Maximum number of zernike polynomials to return. The returned string
        array will have max_z+1 elements, the first being blank.


    Returns
    -------
    z_list : numpy ndarray
        Returns a string array with each element containing z zernike polynomial
        (the first element is blank).

    max_r_power : float, optional
        The maximum radial exponent.

    max_theta_multiplier : float, optional
        Maximum multiplier of the angle.


    Other Parameters
    ----------------
    COMPACT : bool
       If set, the equations are returned using the naming convention for terms
       assumed by PROP_ZERNIKES.

    EXTRA_VALUES : bool
        If true, return maximum radial power and maximum theta multiplier in
        addition to equation strings

    Notes
    -----
    For example:
        zlist = prop_noll_zernikes(5)
	for i in range(1, 6):
            print(i, '   ', zlist[i])

	will display:
      		1   1
      		2   2 * (r)  * cos(t)
      		3   2 * (r)  * sin(t)
      		4   sqrt(3) * (2*r^2 - 1)
      		5   sqrt(6) * (r^2)  * sin(2*t)
    Note that PROP_PRINT_ZERNIKES can also be used to print a table of Zernikes.
    """
    if ("COMPACT" in kwargs and kwargs["COMPACT"]):
        rop = "_pow_"
    else:
        rop = "**"

    max_r_power = 0
    max_theta_multiplier = 0

    z_list = np.zeros(maxz+1, dtype = "S250")    # z_list[0] is always blank
    iz = 1
    n = 0

    while (iz <= maxz):
        for m  in range(np.mod(n,2), n+1, 2):
            for p in range(0, (m != 0) + 1):
                if n != 0:
                    if m != 0:
                        val = 2 * (n+1)
                    else:
                        val = n + 1

                    sqrt_val = int(np.sqrt(val))
                    if val == sqrt_val**2:
                        t = str(sqrt_val).strip() + " * ("
                    else:
                        t = "math.sqrt(" + str(val).strip() + ") * ("
                else:
                    z_list[iz] = "1"
                    iz += 1
                    continue

                for s in range(0, (n-m)//2 + 1):
                    term_top = int((-1)**s) * int(factorial(n-s))
                    term_bottom = int(factorial(s)) * int(factorial((n+m)/2-s)) * int(factorial((n-m)/2 - s))
                    term_val = int(term_top / term_bottom)
                    term = str(np.abs(term_val)).strip() + ".0"
                    term_r = int(n - 2*s)
                    rpower = str(term_r).strip()

                    if max_r_power < term_r:
                        max_r_power = term_r

                    if term_top != 0:
                        if s == 0:
                            if term_val < 0:
                                sign = "-"
                            else:
                                sign = ""
                        else:
                            if term_val < 0:
                                sign = " - "
                            else:
                                sign = " + "

                        if rpower == "0":
                            t += sign + term
                        elif term_r == 1:
                            if term_val != 1:
                                t += sign + term + "*r"
                            else:
                                t += sign + "r"
                        else:
                            if term_val != 1:
                                t += sign + term + "*r" + rop + rpower
                            else:
                                t += sign + "r" + rop + rpower

                if m > max_theta_multiplier:
                    max_theta_multiplier = m

                if m == 0:
                    cterm = ""
                else:
                    if (m != 1):
                        term_m = str(int(m)).strip() + "*t"
                    else:
                        term_m = "t"
                    if np.mod(iz,2) == 0:
                        if ("COMPACT" in kwargs and kwargs["COMPACT"]):
                            cterm = " * cos" + str(int(m)).strip() + "t"
                        else:
                            cterm = " * cos(" + term_m + ")"
                    else:
                        if ("COMPACT" in kwargs and kwargs["COMPACT"]):
                            cterm = " * sin" + str(int(m)).strip() + "t"
                        else:
                            cterm = " * sin(" + term_m + ")"

                if cterm != "":
                    z_list[iz] = t + ")" + cterm
                else:
                    t += ")"
                    z_list[iz] = t

                iz += 1
                if iz > maxz:
                    break

            if iz > maxz:
                break

        n += 1

    if ("EXTRA_VALUES" in kwargs and kwargs["EXTRA_VALUES"]):
        return (z_list, max_r_power, max_theta_multiplier)
    else:
        return z_list

    
# def rebin(a, new_shape):
#     """
#     Resizes a 2d array by averaging or repeating elements, 
#     new dimensions must be integral factors of original dimensions
#     Parameters
#     ----------
#     a : array_like
#         Input array.
#     new_shape : tuple of int
#         Shape of the output array
#     Returns
#     -------
#     rebinned_array : ndarray
#         If the new shape is smaller of the input array, the data are averaged, 
#         if the new shape is bigger array elements are repeated
#     See Also
#     --------
#     resize : Return a new array with the specified shape.
#     Examples
#     --------
#     >>> a = np.array([[0, 1], [2, 3]])
#     >>> b = rebin(a, (4, 6)) #upsize
#     >>> b
#     array([[0, 0, 0, 1, 1, 1],
#            [0, 0, 0, 1, 1, 1],
#            [2, 2, 2, 3, 3, 3],
#            [2, 2, 2, 3, 3, 3]])
#     >>> c = rebin(b, (2, 3)) #downsize
#     >>> c
#     array([[ 0. ,  0.5,  1. ],
#            [ 2. ,  2.5,  3. ]])
#     """
#     M, N = a.shape
#     m, n = new_shape
#     if m<M:
#         return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
#     else:
#         return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)

# if __name__ == '__main__':
#     import doctest
#     doctest.testmod()

