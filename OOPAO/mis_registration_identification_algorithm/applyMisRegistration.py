# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:03:13 2020

@author: cheritie
"""

import numpy as np

from ..DeformableMirror import DeformableMirror
from ..MisRegistration import MisRegistration


def applyMisRegistration(tel,
                         misRegistration_tmp,
                         param=None,
                         wfs=None,
                         extra_dm_mis_registration=None,
                         print_dm_properties=True,
                         floating_precision=64,
                         dm_input=None):

    # used when the wfs is provided to the function to apply a shift to both WFS and DM
    if extra_dm_mis_registration is None:
        extra_dm_mis_registration = MisRegistration()
    try:
        if param['pitch'] is None:
            pitch = None
        else:
            pitch = param['pitch']
    except:
        pitch=None
    if wfs is None:
        try:
            if param['dm_coordinates'] is None:
                coordinates = None
            else:
                coordinates = param['dm_coordinates'] 
        except:
            coordinates = None
            
        if dm_input is None:
            # case synthetic DM - with user-defined coordinates                
            # try:
                if param['isM4'] is True:
                    # case with M4
                    dm_tmp = DeformableMirror(telescope    = tel,\
                                        nSubap       = param['nSubaperture'],\
                                        mechCoupling = param['mechanicalCoupling'],\
                                        coordinates  = coordinates,\
                                        pitch        = None,\
                                        misReg       = misRegistration_tmp + extra_dm_mis_registration,\
                                        M4_param     = param,\
                                        print_dm_properties = print_dm_properties,\
                                        floating_precision =floating_precision)
                    if print_dm_properties:
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                        print('Mis-Registrations Applied on M4!')
            
                elif param['isLBT'] is True:
                    # case with LBT asm model
                    if param['new_IF']:
                        from lbt_tools import get_influence_functions_new as get_influence_functions
                    else:
                        from lbt_tools import get_influence_functions as get_influence_functions

                    modes, coord, M2C, validAct =  get_influence_functions(telescope             = tel,\
                                                                            misReg               = misRegistration_tmp + extra_dm_mis_registration,\
                                                                            filename_IF          = param['filename_if'],\
                                                                            filename_mir_modes   = param['filename_mir_modes'],\
                                                                            filename_coordinates = param['filename_coord'],\
                                                                            filename_M2C         = param['filename_m2c'])
                    param['isM4'] = False
                    # create a deformable mirror with input influence functions interpolated
                    dm_tmp = DeformableMirror(telescope    = tel,\
                        nSubap       = param['nSubaperture'],\
                        mechCoupling = param['mechanicalCoupling'],\
                        coordinates  = coord,\
                        pitch        = pitch,\
                        misReg       = misRegistration_tmp + extra_dm_mis_registration,\
                        modes        = np.reshape(modes,[tel.resolution**2,modes.shape[2]]),\
                        M4_param     = param,\
                        print_dm_properties = print_dm_properties)
                    if print_dm_properties:
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                        print('Mis-Registrations Applied on user-defined DM!')
                        
                    
                else:
                        # default case => use of param['dm_ccordinates']
                        dm_tmp = DeformableMirror(telescope    = tel,\
                                                  nSubap       = param['nSubaperture'],\
                                                  mechCoupling = param['mechanicalCoupling'],\
                                                  coordinates  = coordinates,\
                                                  pitch        = pitch,\
                                                  misReg       = misRegistration_tmp + extra_dm_mis_registration,\
                                                  print_dm_properties = print_dm_properties)
                if print_dm_properties:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Mis-Registrations Applied on Synthetic DM!')
        else:
            # case when no parameter file is provided => copy the properties of the input DM
              dm_tmp = DeformableMirror(telescope    = tel,
                      nSubap       = dm_input.nAct-1,
                      mechCoupling = dm_input.mechCoupling,
                      coordinates  = dm_input.coordinates,
                      pitch        = dm_input.pitch,
                      misReg       = misRegistration_tmp + extra_dm_mis_registration,
                      flip         = dm_input.flip_,
                      flip_lr      = dm_input.flip_lr,
                      sign         = dm_input.sign, 
                      print_dm_properties = print_dm_properties)
              if print_dm_properties:
                  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                  print('Mis-Registrations Applied to input DM!')
                        
    else:
        if wfs.tag == 'pyramid':
            # mis-registration for the WFS
            misRegistration_wfs                 = MisRegistration()
            # sign change to match Pyramid pupil shift vs DM shift
            misRegistration_wfs.shiftX          = -misRegistration_tmp.shiftX
            misRegistration_wfs.shiftY          = misRegistration_tmp.shiftY
            
            wfs.apply_shift_wfs( misRegistration_wfs.shiftX, misRegistration_wfs.shiftY,units = 'm') # units in m to be consistent with dm shift
            
            # mis-registration for the DM
            misRegistration_dm                   = MisRegistration()
            misRegistration_dm.rotationAngle     = misRegistration_tmp.rotationAngle
            misRegistration_dm.tangentialScaling = misRegistration_tmp.tangentialScaling
            misRegistration_dm.radialScaling     = misRegistration_tmp.radialScaling
            
            dm_tmp = applyMisRegistration(tel,
                                          misRegistration_dm + extra_dm_mis_registration,
                                          param,
                                          wfs = None,
                                          print_dm_properties = print_dm_properties,
                                          dm_input  = dm_input)
            if print_dm_properties:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Mis-Registrations Applied on both DM and WFS!')
        else:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Wrong object passed as a wfs.. aplying the mis-registrations to the DM only')
            dm_tmp = applyMisRegistration(tel,
                                          misRegistration_tmp + extra_dm_mis_registration,
                                          param,
                                          wfs=None,
                                          dm_input=dm_input)
    _ = dm_tmp.misReg.properties()
    return dm_tmp