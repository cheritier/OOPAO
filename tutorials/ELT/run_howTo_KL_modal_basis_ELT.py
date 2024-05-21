import numpy             as np 
import __load__oopao
__load__oopao.load_oopao()

# local modules 
from OOPAO.Telescope         import Telescope
from OOPAO.Source            import Source
from OOPAO.Atmosphere        import Atmosphere
from OOPAO.Pyramid           import Pyramid
from OOPAO.DeformableMirror  import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis        import compute_M2C
import OOPAO.calibration.ao_cockpit_psim as aou

#%%
from OOPAO.MisRegistration   import MisRegistration

# ELT modules
from OOPAO.M1_model.make_ELT_pupil             import generateEeltPupilReflectivity
from OOPAO.M4_model.make_M4_influenceFunctions import getPetalModes

from parameterFile_ELT_SCAO_K_Band_3000_KL   import initializeParameterFile

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
atm.initializeAtmosphere(tel)

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object reading the param
misReg = MisRegistration(param)

# M4 is genererated already projected in the M1 space
dm = DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    misReg       = misReg,\
                    M4_param     = param,floating_precision=32)
                    


#%%
## COMPUTE HHt and KL basis with PTT as specific basis
## ESTIMATE HHt division
siz             = tel.OPD.shape[0]
SZ              = siz*2 #np.int(siz*1.1)
nact            = 892
mem_available   = 8.e9

# estimate how to break up the computation of the covariance matrix in NDIVL pieces 
mem,NDIVL       = aou.estimate_ndivl(SZ,siz,nact,mem_available)


#%%
tel-atm
dm.coefs = np.eye(dm.nValidAct)

tel*dm

IF_2D = tel.OPD.copy()
IFma = np.squeeze(np.reshape(IF_2D,[tel.resolution**2,dm.nValidAct])[tel.pupilLogical,:])
IF_2D = IF_2D.T

tel.resetOPD()
dm.coefs = 0

tel*dm
#%%
"""
 ---------------------- DOUBLE DIAGON:IZATION KL IN DM SPACE FORCING PISTON, TIP & TILT ----------------------
 computes KL modes forcing to include piston and tit-tilt modes (default value of nZer = 3)

"""
KL_piston_tip_tilt = compute_M2C(telescope          = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = False,\
                                 HHtName            = 'covariance_matrix_HHt_tutorial',\
                                 baseName           = 'KL_piston_tip_tilt_tutorial' ,\
                                 mem_available      = mem_available,\
                                 minimF             = False,\
                                 nmo                = 4300,\
                                 ortho_spm          = True,\
                                 IF_2D              = IF_2D,\
                                 IFma               = IFma,\
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
                                 IFma               = IFma,\
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
                                 IFma               = IFma,\
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
                                 IF_2D              = None,\
                                 IFma               = None,\
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
                                 IF_2D              = None,\
                                 IFma               = None,\
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
                                 IF_2D              = None,\
                                 IFma               = None,\
                                 P2F                = None,\
                                 SpM_2D             = PM.copy(),\
                                 NDIVL              = NDIVL) 