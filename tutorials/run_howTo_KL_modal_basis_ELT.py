import numpy             as np 
import __load__psim
__load__psim.load_psim()

# local modules 
from AO_modules.Telescope         import Telescope
from AO_modules.Source            import Source
from AO_modules.Atmosphere        import Atmosphere
from AO_modules.Pyramid           import Pyramid
from AO_modules.DeformableMirror  import DeformableMirror
from AO_modules.calibration.compute_KL_modal_basis        import compute_M2C
import AO_modules.calibration.ao_cockpit_psim as aou

#%%
from AO_modules.MisRegistration   import MisRegistration

# ELT modules
from AO_modules.M1_model.make_ELT_pupil             import generateEeltPupilReflectivity
from AO_modules.M4_model.make_M4_influenceFunctions import getPetalModes

from parameterFile_ELT_SCAO_I_Band_3000_KL   import initializeParameterFile

param = initializeParameterFile()


M1_pupil_reflectivity = generateEeltPupilReflectivity(refl = param['m1_reflectivityy'],\
                                          npt       = param['resolution'],\
                                          dspider   = param['spiderDiameter'],\
                                          i0        = param['m1_center'][0]+param['m1_shiftX'] ,\
                                          j0        = param['m1_center'][1]+param['m1_shiftY'] ,\
                                          pixscale  = param['pixelSize'],\
                                          gap       = param['gapSize'],\
                                          rotdegree = param['m1_rotationAngle'],\
                                          softGap   = True)

M1_pupil = M1_pupil_reflectivity>0

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'],\
                pupilReflectivity   = M1_pupil_reflectivity,\
                pupil               = M1_pupil)


#%%
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(tel.pupil)
#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

#%% -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])

# initialize atmosphere
#atm.initializeAtmosphere(tel)

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object reading the param
misReg = MisRegistration(param)

# M4 is genererated already projected in the M1 space
dm = DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    misReg       = misReg,\
                    M4_param     = param)
                    

##%% -----------------------     WFS   ----------------------------------
#
#tel-atm
#
## create the Pyramid Object
#wfs = Pyramid(nSubap                = param['nSubaperture'],\
#              telescope             = tel,\
#              modulation            = param['modulation'],\
#              lightRatio            = param['lightThreshold'],\
#              pupilSeparationRatio  = param['pupilSeparationRatio'],\
#              calibModulation       = param['calibrationModulation'],\
#              psfCentering          = param['psfCentering'],\
#              edgePixel             = param['edgePixel'],\
#              unitCalibration       = param['unitCalibration'],\
#              extraModulationFactor = param['extraModulationFactor'],\
#              postProcessing        = param['postProcessing'])


#%% -----------------------    MODAL BASIS   ----------------------------------
### CVE: THIS IS TO SPEED UP/DEBUGGING (OR MULTIPLE BASIS NEED TO BE COMPUED)
print('Pre-computing the influence functions truncated by the pupil...')
pupil_1D = np.reshape(tel.pupil, tel.resolution**2)

# get influence functions truncated by the pupil mask in 2D
IF_2D  = np.reshape(np.moveaxis(dm.modes*np.tile(pupil_1D[:,None],dm.nAct),-1,0), [dm.nAct,tel.resolution,tel.resolution])

IF_1D_pup = np.squeeze(dm.modes[tel.pupilLogical,:])
print('Done!')

#%%
## COMPUTE HHt and KL basis with PTT as specific basis
## ESTIMATE HHt division
resolution_IF   = tel.resolution #np.int(siz*1.1)
resolution_FFT  = resolution_IF*1.1 #np.int(siz*1.1)
nact            = 5352
mem_available   = 50.e9

# estimate how to break up the computation of the covariance matrix in NDIVL pieces 
mem,NDIVL       = aou.estimate_ndivl(SZ = resolution_FFT,\
                                     sz = resolution_IF,\
                                     nact = dm.nAct,\
                                     MEMmax = mem_available)

#%%
"""
 ---------------------- DOUBLE DIAGON:IZATION KL IN DM SPACE FORCING PISTON, TIP & TILT ----------------------
 computes KL modes forcing to include piston and tit-tilt modes (default value of nZer = 3)
 
 
    - HHtName       = None      extension for the HHt Covariance file
    - baseName      = None      extension to the filename for basis saving
    - SpM_2D        = None      2D Specific modes [dim,dim,nspm], if None then automatic
    - nZer          = 3         number of zernike (Piston,Tip,Tilt...) for automatic computation of specific modes
    - SZ            = None      resolution of FFts for HHt (By default SZ=2*tel.resolution)
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
    - IF_2D         = None      2D Influence Functions in OPD (for speeding up)
    - IFma          = None      Serial Influence Functions (only for speeding up)
    - returnSB      = False     Flag to return also the Seed Basis (w/ or w/o KL)
    
"""
KL_piston_tip_tilt = compute_M2C(telescope          = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = False,\
                                 HHtName            = 'covariance_matrix_HHt_tutorial_KL',\
                                 baseName           = 'KL_piston_tip_tilt_tutorial_new' ,\
                                 mem_available      = mem_available,\
                                 minimF             = False,\
                                 nmo                = 4300,\
                                 ortho_spm          = True,\
                                 IF_2D              = IF_2D,\
                                 IFma               = IF_1D_pup,\
                                 nZer               = 3,\
                                 NDIVL              = NDIVL) 

#%%
"""
 ---------------------- DOUBLE DIAGON:IZATION KL IN DM SPACE FORCING 9 ZERNIKES ----------------------
 computes KL modes forcing to include the 9 first zernike polynomials)
"""
KL_9_zernike = compute_M2C(telescope                = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = False,\
                                 HHtName            = 'covariance_matrix_HHt_tutorial',\
                                 baseName           = 'KL_9_zernike_tutorial' ,\
                                 mem_available      = mem_available,\
                                 minimF             = False,\
                                 nmo                = 4300,\
                                 ortho_spm          = True,\
                                 IF_2D              = IF_2D,\
                                 IFma               = IF_1D_pup,\
                                 nZer               = 9,\
                                 NDIVL              = NDIVL) 

#%% 
"""
---------------------- DOUBLE DIAGON:IZATION KL IN DM SPACE WITH SMALLER ZEROPADDING TO REDUCE COV. MAT. SIZE ----------------------

computes KL modes considering a smaller factor for cov.mat computation (here factor 1.1) 
Allows to reduce NDIVL value, typically to 1
"""
KL_small_padding = compute_M2C(telescope                = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = False,\
                                 HHtName            = 'covariance_matrix_HHt_tutorial_padding_factor_1p1',\
                                 baseName           = 'KL_small_padding_tutorial' ,\
                                 mem_available      = mem_available,\
                                 minimF             = False,\
                                 nmo                = 4300,\
                                 ortho_spm          = True,\
                                 SZ                 = np.int(1.1*tel.OPD.shape[0]),\
                                 IF_2D              = IF_2D,\
                                 IFma               = IF_1D_pup,\
                                 nZer               = 3,\
                                 NDIVL              = 1) 

#%%
"""
 ---------------------- DOUBLE DIAGON:IZATION KL IN DM SPACE FORCING PISTON, TIP & TILT + FORCES MINIMIZATION ----------------------

 computes KL modes forcing to include piston and tit-tilt modes (default value of nZer = 3)
 Plus minimize the forces using P2F matrix and minimF = True. 
 if P2F = None, the matrix is loaded from param['pathInput']
"""
KL_piston_tip_tilt_minimized_forces = compute_M2C(telescope          = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = False,\
                                 HHtName            = 'covariance_matrix_HHt_tutorial',\
                                 baseName           = 'KL_piston_tip_tilt_minimized_forces_tutorial' ,\
                                 mem_available      = mem_available,\
                                 minimF             = True,\
                                 nmo                = 4300,\
                                 ortho_spm          = True,\
                                 IF_2D              = IF_2D,\
                                 IFma               = IF_1D_pup,\
                                 P2F                = None,\
                                 nZer               = 3,\
                                 NDIVL              = NDIVL) 

#%% 
"""
---------------------- DOUBLE DIAGON:IZATION KL IN DM SPACE FORCING PURE PETAL MODES + FORCES MINIMIZATION  ----------------------

computes KL modes forcing to include 6 pure petal modes (overwrite nZer).
 
The corresponding petal modes are kept as such and not orthogonalized (ortho_spm  = False)

In addition, it minimizes the forces using P2F matrix and minimF = True. 

if P2F = None, the matrix is loaded from param['pathInput']

"""
try:
    petals,petals_float = getPetalModes(tel,dm,[1,2,3,4,5,6])
except:
    petals,petals_float = getPetalModes(tel,dm,[1])

KL_pure_petals_minimized_forces = compute_M2C(telescope          = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = False,\
                                 HHtName            = 'covariance_matrix_HHt_tutorial',\
                                 baseName           = 'KL_pure_petals_minimized_forces_tutorial' ,\
                                 mem_available      = mem_available,\
                                 minimF             = True,\
                                 nmo                = 4300,\
                                 ortho_spm          = False,\
                                 IF_2D              = IF_2D,\
                                 IFma               = IF_1D_pup,\
                                 P2F                = None,\
                                 SpM_2D             = petals.copy(),\
                                 NDIVL              = NDIVL) 

#%% 
"""
---------------------- DOUBLE DIAGONALIZATION KL IN DM SPACE FORCING PURE PISTON + 5 PETAL MODES + FORCES MINIMIZATION  ----------------------
computes KL modes forcing to include :
    _ pure piston mode
    _ "nuclear petal mode"
    _ remaining petal modes

The corresponding petal modes are then orthogonalized (ortho_spm  = True) to provide an orthogonal petal basis

"""
PM=petals.copy()

PM[:,:,0] = np.sum(petals,axis=2)
PM[:,:,1]=PM[:,:,1]*0.
for k in range(0,6):
    PM[:,:,1] = PM[:,:,1]+(-1.)**k *  petals[:,:,k]

KL_orthogonal_petals_minimized_forces = compute_M2C(telescope          = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = False,\
                                 HHtName            = 'covariance_matrix_HHt_tutorial',\
                                 baseName           = 'KL_orthogonal_petals_minimized_forces_tutorial' ,\
                                 mem_available      = mem_available,\
                                 minimF             = True,\
                                 nmo                = 4300,\
                                 ortho_spm          = True,\
                                 IF_2D              = IF_2D,\
                                 IFma               = IF_1D_pup,\
                                 P2F                = None,\
                                 SpM_2D             = PM.copy(),\
                                 NDIVL              = NDIVL) 