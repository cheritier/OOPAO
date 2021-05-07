# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:34:45 2020

@author: cheritie
"""
# commom modules
import jsonpickle
import json
import __load__psim
__load__psim.load_psim()
# AO modules 
from AO_modules.calibration.ao_calibration          import ao_calibration_from_ao_obj
from AO_modules.calibration.CalibrationVault        import calibrationVault
from AO_modules.calibration.initialization_ELT_SCAO import run_initialization_ELT_SCAO
from AO_modules.closed_loop.run_cl                  import run_cl


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
from parameterFile_ELT_SCAO_R_Band_3000_KL  import initializeParameterFile

# read parameter file
param = initializeParameterFile()

param['getProjector'] = False # if False do not compute the pseudo-inverse of the modal basis to get the projector phase to modal coefficients

# create ELT Object
obj = run_initialization_ELT_SCAO(param)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALIBRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create a calibration object that contains the modal basis, the projector and the interaction matrix and reconstructor


calib_obj  = ao_calibration_from_ao_obj(obj,nameFolderIntMat = None, nameIntMat = None,nameFolderBasis=None,  nameBasis = None)
        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RUN CLOSE LOOP %%%%%%%%%%%%%%%%%%%%%%%%%%%%

try:
    obj.projector   = calib_obj.projector
    print('Projector Loaded!')
except:
    print('No Projector Loaded')
 
    
obj.calib       = calib_obj.calib
obj.M2C_cl      = calib_obj.M2C
obj.basis       = calib_obj.basis
obj.gOpt        = calib_obj.gOpt

# disable display of the CL phase screens
obj.display         = True
obj.displayPetals   = False
obj.printPetals     = True

# disable photon noise
param['photonNoise'] = False
param['nLoop'] = 500

# disable the petral free modes
obj.tel.isPetalFree = False

# run the close loop
output          = run_cl(param,obj)
# store output as a JSON file
nameOutput = 'output_CL_'+str(param['resolution'])+'_res_'+'_r0_'+str(param['r0'])+'_'+str(param['modulation'])+'_mod_'+str(param['nModes'])+'_modes_petals.fits'
output_encoded  = jsonpickle.encode(output)
with open(output['destinationFolder']+nameOutput+'.json', 'w') as f:
    json.dump(output_encoded, f)
