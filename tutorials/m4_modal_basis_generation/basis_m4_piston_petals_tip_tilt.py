# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:10:32 2022

@author: cheritie
"""

## Ref.: https://www.aanda.org/articles/aa/full_html/2024/01/aa46902-23/aa46902-23.html
############################################
PLOT_M4_RMS_POS_FOR = True 

if PLOT_M4_RMS_POS_FOR == True:
    nmoa=4000 #NUMBER OF MODES RECONSTRUCTED FOR PLOT (4000 like in paper)
    r0a = 0.1 ## SEEING FOR ESTIMATION FOR PLOT (r0a = 0.1 is like in paper)

L0a = 50. ## like in paper
############################################

import pdb
import numpy             as np 
#import __load__oopao
#__load__oopao.load_oopao()

# local modules 
from OOPAO.Telescope         import Telescope
from OOPAO.Source            import Source
from OOPAO.Atmosphere        import Atmosphere
from OOPAO.DeformableMirror  import DeformableMirror
from OOPAO.Pyramid           import Pyramid

from OOPAO.calibration.compute_KL_modal_basis       import compute_M2C

import OOPAO.calibration.ao_cockpit_psim as aou
from importlib import reload
reload(aou)
#%%

from OOPAO.MisRegistration   import MisRegistration

# ELT modules
from OOPAO.M1_model.make_ELT_pupil             import generateEeltPupilReflectivity
from OOPAO.M4_model.make_M4_influenceFunctions import getPetalModes


from parameterFile_m4_384x384 import initializeParameterFile

from astropy.io import fits

param = initializeParameterFile()

from OOPAO.calibration.ao_calibration import ao_calibration

import matplotlib.pyplot as plt

from OOPAO.calibration.ao_cockpit_psim import plt_plot
from OOPAO.calibration.ao_cockpit_psim import plt_imshow
from OOPAO.calibration.ao_cockpit_psim import plt_imshow_expa


M1_pupil_reflectivity = generateEeltPupilReflectivity(refl = param['m1_reflectivityy'],\
                                          npt       = param['resolution'],\
                                          dspider   = param['spiderDiameter'],\
                                          i0        = param['m1_center'][0]+param['m1_shiftX'] ,\
                                          j0        = param['m1_center'][1]+param['m1_shiftY'] ,\
                                          pixscale  = param['pixelSize'],\
                                          gap       = param['gapSize'],\
                                          rotdegree = param['m1_rotationAngle'],\
                                          softGap   = True)
dim = M1_pupil_reflectivity.shape[0]

M1_pupil = M1_pupil_reflectivity>0

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'],\
                pupilReflectivity   = M1_pupil_reflectivity,\
                pupil               = M1_pupil)

#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

#%% -----------------------     ATMOSPHERE   ----------------------------------

R1=1.0
# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = R1,\
               L0            = L0a,\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])

# initialize atmosphere
atm.initializeAtmosphere(tel)

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object reading the param
misReg = MisRegistration(param)



# M4 is genererated already projected in the M1 space
dm = DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    misReg       = misReg,\
                    M4_param     = param)
  
######
import ctypes
import multiprocessing as mp

mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads

mkl_set_num_threads(mp.cpu_count())

### BUILD PTT
diameter = tel.D
pupil = tel.pupil
idxpup = np.where(pupil == 1)
tpup = np.sum(pupil)

GEO = aou.mkp(tel.resolution/tel.resolution*diameter,tel.resolution,diameter,0.)
nZer = 3
SpM_2D = aou.give_zernike(GEO, diameter, nZer)

basis_ptt_2D = SpM_2D*np.reshape(np.repeat(pupil[:,:,np.newaxis],nZer),[dim,dim,nZer])

### BUILD PETALS: FORCE PISTON AND NUCLEAR MODE
petals,petals_float = getPetalModes(tel,dm,[1,2,3,4,5,6])
PM=petals.copy()
PM[:,:,0] = np.sum(petals,axis=2)
PM[:,:,1]=PM[:,:,1]*0.
for k in range(0,6):
    PM[:,:,1] = PM[:,:,1]+(-1.)**k *  petals[:,:,k]

## BUILD 6 ORTHOGONAL PETAL MODES 
PM_s = np.reshape(PM,[dim*dim,6])
Q,R = np.linalg.qr(PM_s)
Q_2D = np.reshape(Q,[dim,dim,6])

## ADD TT
SpM_2D_6PO_PTT = np.zeros([dim,dim,6+2])
SpM_2D_6PO_PTT[:,:,0:6] = PM.copy()
SpM_2D_6PO_PTT[:,:,6] = basis_ptt_2D[:,:,1]*pupil
SpM_2D_6PO_PTT[:,:,7] = basis_ptt_2D[:,:,2]*pupil

## ORTHONORMALIZE THE BASIS WITH QR FACTORIZATION
SpM_s = np.reshape(SpM_2D_6PO_PTT,[dim*dim,8])
Q_spm,R_spm = np.linalg.qr(SpM_s)
Q_spm_2D = np.reshape(Q_spm*np.sqrt(tpup),[dim,dim,8])

nSpM = Q_spm_2D.shape[2]

#%%
## COMPUTE HHt and KL basis with PTT as specific basis
## ESTIMATE HHt division
#siz             = tel.OPD.shape[0]
#SZ              = siz*2 #np.int(siz*1.1)
nact            = dm.modes.shape[1]
#mem_available   = 200.e9

#mem,NDIVL       = aou.estimate_ndivl(SZ,siz,nact,mem_available)
#nameFolder = '/diskb/cverinau/oopao_data/data_calibration/TOYs/'

NDIVL = 2

KL_6POTT_F, HHt, PSD_atm,df = compute_M2C(telescope          = tel,\
                                                  atmosphere         = atm,\
                                                  deformableMirror   = dm,\
                                                  param              = param,\
                                                  nameFolder         = None,\
                                                  nameFile           = None,\
                                                  remove_piston      = False,\
                                                  HHtName            = None,\
                                                  baseName           = None ,\
                                                  mem_available      = None,\
                                                  minimF             = True,\
                                                  nmo                = 4000,\
                                                  ortho_spm          = True,\
                                                  IF_2D              = None,\
                                                  IFma               = None,\
                                                  P2F                = None,\
                                                  alpha              = None,\
                                                  beta               = None,\
                                                  SpM_2D             = Q_spm_2D.copy(),\
                                                  NDIVL              = NDIVL,\
                                                  save_output        = False,\
                                                  returnHHt_PSD_df = True) 


if PLOT_M4_RMS_POS_FOR:


    P2F=np.float64(fits.getdata(param['pathInput']+'P2F.fits'))*1.e6


    P2Ff=np.zeros([nact,nact],dtype=np.float64)

    nap=nact//6
    for k in range(0,6):
        P2Ff[k*nap:(k+1)*nap,k*nap:(k+1)*nap] = P2F.copy()

    K=np.asmatrix(P2Ff)

    IF_2D = np.reshape(dm.modes,[dim,dim,nact])
    pupil = tel.pupil
    tpup = np.sum(pupil)
    idxpup = np.where(pupil == 1.)

    IFma = aou.vectorify(IF_2D,idxpup)

    IFma = np.asmatrix(IFma)
    DELTA = IFma.T @ IFma

    ## COMPUTING FITTING ERROR VS NUMBER OF MODES (for ORTHONORMAL BASIS ONLY)
    print('ESTIMATING FITTING EFFICIENCY OF FULL BASIS')
    print(' ')

    BASIS_full = KL_6POTT_F

    print('ESTIMATING DISTRIBUTION OF POSITIONS AND FORCES FOR NB OF MODES =',nmoa)
    print(' ')

    FIT_BASIS = aou.FIT_ONB(BASIS_full,DELTA,HHt*r0a**(-5./3.),PSD_atm*r0a**(-5./3.),df,tpup)

    RMS_POS_BASIS = aou.POS_ONB(BASIS_full,DELTA,HHt*r0a**(-5./3.),nmoa)
    RMS_FOR_BASIS = aou.FOR_ONB(BASIS_full,DELTA,HHt*r0a**(-5./3.),nmoa,K)

    plt.figure()
    plt.title('Fitting error in function of modes corrected',fontsize=15)
    plt.xlabel('number of modes',fontsize=14)
    plt.ylabel('micrometers',fontsize=14)
    plt.xlim(1,nmoa)
    plt.loglog(FIT_BASIS*1.e6,label='Minimum Force KL modes')
    plt.legend()

    plt.show(block=False)

    plt.figure()
    plt.title('Positions RMS distributions',fontsize=15)
    plt.xlabel('actuator index',fontsize=14)
    plt.ylabel('micrometers',fontsize=14)
    plt.plot(RMS_POS_BASIS*1.e6)

    plt.legend()

    plt.show(block=False)



    plt.figure()
    plt.title('Forces RMS distributions',fontsize=15)
    plt.xlabel('actuator index',fontsize=14)
    plt.ylabel('Newtons',fontsize=14)
    plt.plot(RMS_FOR_BASIS)
    plt.legend()

    plt.show(block=False)

    BASIS_full_s = IFma @ BASIS_full
    BASIS_full_2D = aou.expand_a(BASIS_full_s,dim,idxpup,BASIS_full_s.shape[1],1.0)
    affi = np.zeros([dim,dim*nSpM])
    for k in range(nSpM):
        affi[:,k*dim:(k+1)*dim] = BASIS_full_2D[:,:,k]

    plt_imshow(affi)
