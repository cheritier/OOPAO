# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:03:13 2020

@author: cheritie
"""

import numpy as np

from ..DeformableMirror import DeformableMirror
from ..MisRegistration import MisRegistration


def applyMisRegistration(tel,misRegistration_tmp,param, wfs = None, extra_dm_mis_registration = None,print_dm_properties=True,floating_precision=64):
        if extra_dm_mis_registration is None:
            extra_dm_mis_registration = MisRegistration()
        try:
                if param['pitch'] is None:
                    pitch = None
                else:
                    pitch = param['pitch'] 
        except:
                pitch = None
        if wfs is None:
            
            # case synthetic DM - with user-defined coordinates
            try:
                if param['dm_coordinates'] is None:
                    coordinates = None
                else:
                    coordinates = param['dm_coordinates'] 
            except:
                coordinates = None
                
            
                
            try:
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
            
                else:
                    from users.cheritie.model_LBT.lbt_tools import get_influence_functions

                    modes, coord, M2C, validAct =  get_influence_functions(telescope             = tel,\
                                                                            misReg               = misRegistration_tmp + extra_dm_mis_registration,\
                                                                            filename_IF          = param['filename_if'],\
                                                                            filename_mir_modes   = param['filename_mir_modes'],\
                                                                            filename_coordinates = param['filename_coord'],\
                                                                            filename_M2C         = param['filename_m2c'])
                    param['isM4'] = False
                    
                    dm_tmp = DeformableMirror(telescope    = tel,\
                        nSubap       = param['nSubaperture'],\
                        mechCoupling = param['mechanicalCoupling'],\
                        coordinates  = coord,\
                        pitch        = None,\
                        misReg       = misRegistration_tmp + extra_dm_mis_registration,\
                        modes        = np.reshape(modes,[tel.resolution**2,modes.shape[2]]),\
                        M4_param     = param,\
                        print_dm_properties = print_dm_properties)
                    if print_dm_properties:
                        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                        print('Mis-Registrations Applied on user-defined DM!')
                        
                    
            except:
                    # default case
                        
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
            if wfs.tag == 'pyramid':
                misRegistration_wfs                 = MisRegistration()
                misRegistration_wfs.shiftX          = misRegistration_tmp.shiftX
                misRegistration_wfs.shiftY          = misRegistration_tmp.shiftY
                
                wfs.apply_shift_wfs( misRegistration_wfs.shiftX, misRegistration_wfs.shiftY)

                
                misRegistration_dm                   = MisRegistration()
                misRegistration_dm.rotationAngle     = misRegistration_tmp.rotationAngle
                misRegistration_dm.tangentialScaling = misRegistration_tmp.tangentialScaling
                misRegistration_dm.radialScaling     = misRegistration_tmp.radialScaling
                
                dm_tmp = applyMisRegistration(tel,misRegistration_dm + extra_dm_mis_registration, param, wfs = None, print_dm_properties = print_dm_properties)
                if print_dm_properties:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Mis-Registrations Applied on both DM and WFS!')
            else:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Wrong object passed as a wfs.. aplying the mis-registrations to the DM only')
                dm_tmp = applyMisRegistration(tel,misRegistration_tmp + extra_dm_mis_registration, param, wfs = None)
                             
        return dm_tmp
    
    
    
    
    

# def apply_shift_wfs(wfs,sx,sy):
#     if wfs.tag =='pyramid':
#         sx *= 1/(wfs.telescope.pixelSize*(wfs.telescope.resolution/wfs.nSubap))
#         sy *= 1/(wfs.telescope.pixelSize*(wfs.telescope.resolution/wfs.nSubap))
#         tmp                             = np.ones([wfs.nRes,wfs.nRes])
#         tmp[:,0]                        = 0
#         Tip                             = (sp.morphology.distance_transform_edt(tmp))
#         Tilt                            = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
        
#         # normalize the TT to apply the modulation in terms of lambda/D
#         Tip                        = (wfs.telRes/wfs.nSubap)*(((Tip/Tip.max())-0.5)*2*np.pi)
#         Tilt                       = (wfs.telRes/wfs.nSubap)*(((Tilt/Tilt.max())-0.5)*2*np.pi)
        
#         wfs.mask = wfs.convert_for_gpu(np.exp(1j*(wfs.initial_m+sx*Tip+sy*Tilt)))