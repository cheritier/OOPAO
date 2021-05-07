# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:00:39 2021

@author: cheritie
"""
# commom modules
import jsonpickle
import json
import numpy as np
import copy

# AO modules 
from AO_modules.mis_registration_identification_algorithm.estimateMisRegistration import estimateMisRegistration
from AO_modules.mis_registration_identification_algorithm.computeMetaSensitivyMatrix import computeMetaSensitivityMatrix

from AO_modules.calibration.CalibrationVault import calibrationVault
from AO_modules.MisRegistration import MisRegistration


def batch_estimation_mis_registration(obj, basis, mis_registration_tests, flux_values, amplitude_values, r0_values,name_output, mis_registration_zero = None):
  
    # prepare the fields
    flux_fields = []
    flux_names  = []
    for i in range(len(flux_values)):
        flux_fields.append('nPhotonperSubap_'+str(flux_values[i]))
        flux_names.append(' '+str(flux_values[i])+' phot/subap')
    
    amplitude_fields = []
    amplitude_names  = []
    for i in range(len(amplitude_values)):
        amplitude_fields.append('amplitude_'+str(amplitude_values[i]))
        amplitude_names.append(' Amplitude '+str(amplitude_values[i])+' nm')
    
    r0_fields = []
    r0_names  = []
    for i in range(len(r0_values)):
        r0_fields.append('r0_'+str(r0_values[i]))
        r0_names.append(' r0 '+str(r0_values[i])+' m')
        
    # zero point and epsilon mis-registration
    if mis_registration_zero is None:
        misRegistrationZeroPoint                = MisRegistration()
    else:
        misRegistrationZeroPoint = mis_registration_zero
    
    epsilonMisRegistration                  = MisRegistration()
    epsilonMisRegistration.shiftX           = np.round(obj.dm.pitch /100,4)
    epsilonMisRegistration.shiftY           = np.round(obj.dm.pitch /100,4)
    epsilonMisRegistration.rotationAngle    = np.round(np.rad2deg(np.arctan(epsilonMisRegistration.shiftX)/(obj.tel.D/2)),4)
    
    # location of the sensitivity matrices
    nameFolder_sensitivity_matrice = obj.generate_name_location_sensitivity_matrices(obj)
    
    # load sensitiviy matrices (to do it only once)
    [metaMatrix,calib_0] = computeMetaSensitivityMatrix(nameFolder                 = nameFolder_sensitivity_matrice,\
                                 nameSystem                 = '',\
                                 tel                        = obj.tel,\
                                 atm                        = obj.atm,\
                                 ngs                        = obj.ngs,\
                                 dm_0                       = obj.dm,\
                                 pitch                      = obj.dm.pitch,\
                                 wfs                        = obj.wfs,\
                                 basis                      = basis,\
                                 misRegistrationZeroPoint   = misRegistrationZeroPoint,\
                                 epsilonMisRegistration     = epsilonMisRegistration,\
                                 param                      = obj.param)

    # dictionnaries to store the output
    tmp_data_amp    = dict()
    tmp_data_flux   = dict()
    tmp_data_r0     = dict()
    data_out        = dict()
    
    # start
    for i_r0 in range(len(r0_values)):
        
        for i_flux in range(len(flux_values)):
            
            for i_amp in range(len(amplitude_values)):
                
                mis_reg_out = np.zeros([3,len(mis_registration_tests.misRegValuesX)])
                gamma_out   = np.zeros([basis.modes.shape[1],len(mis_registration_tests.misRegValuesX)])
        
                for i_misReg in range(len(mis_registration_tests.misRegValuesX)):
                    
                    misRegistration_cl = MisRegistration(obj.param)
                    misRegistration_cl.shiftX = mis_registration_tests.misRegValuesX[i_misReg]*obj.dm.pitch/100
                    misRegistration_cl.shiftY = mis_registration_tests.misRegValuesY[i_misReg]*obj.dm.pitch/100
                    misRegistration_cl.rotationAngle = mis_registration_tests.misRegValuesRot[i_misReg]
                    
                    # location of the data
                    nameFolder_data= obj.generate_name_location_data(obj,misRegistration_cl)
                    
                    # name of the files
                    nameFile = obj.generate_name_data(obj,r0_values[i_r0],flux_values[i_flux],amplitude_values[i_amp])
                    
                    # read the file
                    try:
                        with open(nameFolder_data+nameFile+'.json') as f:
                            C = json.load(f)
                            output_decoded = jsonpickle.decode(C)  
                            print('file succesfully open!')
                        # extract the on-sky interaction matrix
                        try:
                            print('considering only ' +str(obj.nMeasurements)+' measurements')
                            calib_misReg_in = calibrationVault(np.mean(output_decoded['push_pull_buffer'][:obj.nMeasurements,:,:],axis = 0))
                        
                        except:
                            print('considering the ' +str(obj.number_push_pull)+' measurements')
                            calib_misReg_in = output_decoded['interactionMatrix']
                        
                        # reduce it to the considered modes
                        calib_misReg = calibrationVault(calib_misReg_in.D[:,obj.ind_modes])
                        
                        # estimation script
                        [mis_reg, gamma, alpha ] = estimateMisRegistration(nameFolder               = nameFolder_sensitivity_matrice,\
                                                                 nameSystem                 = '',\
                                                                 tel                        = obj.tel,\
                                                                 atm                        = obj.atm,\
                                                                 ngs                        = obj.ngs,\
                                                                 dm_0                       = obj.dm,\
                                                                 calib_in                   = calib_misReg,\
                                                                 wfs                        = obj.wfs,\
                                                                 basis                      = basis,\
                                                                 misRegistrationZeroPoint   = misRegistrationZeroPoint,\
                                                                 epsilonMisRegistration     = epsilonMisRegistration,\
                                                                 param                      = obj.param,\
                                                                 precision                  = 5,\
                                                                 sensitivity_matrices       = metaMatrix,\
                                                                 return_all                 = True,\
                                                                 fast                       = False )
                        gamma = np.asarray(gamma)   
                        alpha = np.asarray(alpha)   
                        
                        # values output
                        values_out                              = dict()
                        values_out['optical_gains_full'    ]    = gamma
                        values_out['mis_registrations_full']    = alpha
                        
                        # storing the convergence value
                        mis_reg_out[:,i_misReg]                 = alpha[-1,:]
                        gamma_out[:,i_misReg]                   = gamma[-1,:]
                        
                        # data out
                        data_out[misRegistration_cl.misRegName] = values_out
                        
                        
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                        print('Mis-Registrations identified:')
                        print(r0_names[i_r0]+'--'+flux_names[i_flux]+'--'+amplitude_names[i_amp])
                        print('Rotation [deg] \t Shift X [m] \t Shift Y [m]')
                        print(str(mis_reg_out[0,i_misReg])   + '\t\t' +str(100*(mis_reg_out[1,i_misReg]/obj.dm.pitch))+'\t\t' + str(100*(mis_reg_out[2,i_misReg]/obj.dm.pitch))) 
                        print('Mis-Registrations True:')
                        print('Rotation [deg] \t Shift X [m] \t Shift Y [m]')
                        print(str(mis_registration_tests.misRegValuesRot[i_misReg]) + '\t\t'+str(mis_registration_tests.misRegValuesX[i_misReg])+'\t\t'  + str(mis_registration_tests.misRegValuesY[i_misReg]))             
                    except:
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                        print('ERROR! NO FILE FOUND FOR '+ nameFolder_data+nameFile)
                        print('skipping...')
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                
                data_out['mis_registrations'] = mis_reg_out
                data_out['optical_gains    '] = gamma_out
                
                data_out['misReg_values_rot_out']   = mis_reg_out[0,:]
                data_out['misReg_values_x_out']     = 100*(mis_reg_out[1,:]/obj.dm.pitch)
                data_out['misReg_values_y_out']     = 100*(mis_reg_out[2,:]/obj.dm.pitch)
                    
        
                tmp_data_amp[amplitude_fields[i_amp]]       = copy.deepcopy(data_out)
            tmp_data_flux[flux_fields[i_flux]] = copy.deepcopy(tmp_data_amp)
        tmp_data_r0[r0_fields[i_r0]] = copy.deepcopy(tmp_data_flux)
    
            
            
    out=dict()        
    out['data_out']         = tmp_data_r0
    
    out['r0_names']         = r0_names
    out['amp_names']        = amplitude_names
    out['flux_names']       = flux_names
    
    out['r0_fields']         = r0_fields
    out['amp_fields']        = amplitude_fields
    out['flux_fields']       = flux_fields
    
    out['r0_values']         = r0_values
    out['amp_values']        = amplitude_values
    out['flux_values']       = flux_values
    
    out['misReg_values_x'  ]  = mis_registration_tests.misRegValuesX
    out['misReg_values_y'  ]  = mis_registration_tests.misRegValuesY
    out['misReg_values_rot']  = mis_registration_tests.misRegValuesRot
    
    output_encoded  = jsonpickle.encode(out)
    
    with open(name_output+'.json', 'w') as f:
        json.dump(output_encoded, f)
