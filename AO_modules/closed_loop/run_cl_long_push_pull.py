# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:32:08 2021

@author: cheritie
"""

import matplotlib.pyplot as plt
import numpy             as np 
import time

# local modules 
from AO_modules.calibration.CalibrationVault  import calibrationVault
from AO_modules.MisRegistration   import MisRegistration
from AO_modules.tools.tools                  import createFolder, emptyClass
from AO_modules.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration


#####################################################################################################################################################
#     # #   #         #     # # #   # # #           #         #       #     # # #
#   #       #       #   #   #       #               #       #   #   #   #   #    #
#   #       #       #   #   # # #   # #     # # #   #       #   #   #   #   # # #
#   #       #       #   #       #   #               #       #   #   #   #   #
#     # #   # # #     #     # # #   # # #           # # #     #       #     #
#####################################################################################################################################################

def run_cl_long_push_pull(param,obj):
    

    gain_cl = param['gainCL']
    
    #% ------------------------------------ Noise Update ------------------------------------

    # set magnitude of the NGS
    obj.ngs.nPhoton = param['nPhotonPerSubaperture'] * (1/obj.tel.samplingTime)/((obj.tel.D/param['nSubaperture'] )**2)
    
    extraName =''
    # photon Noise
    obj.wfs.cam.photonNoise = param['photonNoise']
    if obj.wfs.cam.photonNoise:
        extraName+=('photonNoise_')
    
    # RON
    obj.wfs.cam.readoutNoise = param['readoutNoise']
    
    if obj.wfs.cam.readoutNoise:
        extraName+=('readoutNoise_')
    
    
    #% ------------------------------------ Calibration ------------------------------------
    
    calib_cl    = calibrationVault(obj.calib.D[:,:param['nModes']])
    M2C_cl      = obj.M2C_cl
    M2C_cl_inv  = np.linalg.pinv(M2C_cl)
    #% ------------------------------------ M4 Saturations ------------------------------------
    
    obj.P2F_full = np.zeros([6*892,6*892])
    try: 
        for i in range(6):
            obj.P2F_full[i*892:(i+1)*892,i*892:(i+1)*892] = obj.P2F    	
        obj.get_forces = True
        print('Stiffness matrix properly attached to the ao-object!')

    except:            
        print('No stiffness matrix attached to the ao-object.. computation of the M4 forces not considered')
        obj.get_forces = False
    #%% ------------------------------------ mis-registrations  ------------------------------------
    
    misRegistration_cl = MisRegistration(param)
    
    misRegistration_cl.show()
    obj.dm.misReg.show()
    
#    case with no mis-registration
    if misRegistration_cl == obj.dm.misReg:
        dm_cl = obj.dm
    else:
        dm_cl = applyMisRegistration(obj.tel,misRegistration_cl,param)
  
    
    
    #%%
    def push_pull(obj, dm_cl, wfsSignal, i_loop,cl_data):
        
        sp = np.zeros(obj.wfs.nSignal)
        sm = np.zeros(obj.wfs.nSignal)
        
        # push
        for i_length in range(obj.push_pull_duration):
            # update of the atmosphere
            obj.atm.update()
            cl_data.ao_turbulence[i_loop+i_length ]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9

            # get modal coefficients from DM        
            modal_coeffs = M2C_cl_inv @ dm_cl.coefs
            # force a push/pull on the desired mode
            modal_coeffs[obj.push_pull_index_mode[0]] = obj.push_pull_amplitude[0]
            print('Push '+str(i_length))
            # apply it on the dm
            dm_cl.coefs = M2C_cl@modal_coeffs
            # propagate
            obj.tel*dm_cl*obj.wfs
            
            if obj.get_forces:
                cl_data.dm_forces[i_loop,:] = obj.P2F_full@dm_cl.coefs
            
            cl_data.ao_residuals[i_loop+i_length]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9
            # control law
            dm_cl.coefs -= np.matmul(M2C_cl,obj.push_pull_gains@np.matmul(reconstructor,wfsSignal))  
            # buffer of the wfs signal to simulate delay
            wfsSignal = obj.wfs.signal
            if i_length>=obj.start_pp:
                print('saving wfs signal')
                sp+=obj.wfs.signal
            print('Loop '+str(i_loop+i_length)+'/'+str(cl_data.nLoop)+' Turbulence: '+str(cl_data.ao_turbulence[i_loop+i_length])+' -- Residual:' +str(cl_data.ao_residuals[i_loop+i_length])+ '\n')
            print('Elapsed Time: ' + str(b-a) + ' s ')
            if obj.display==True:
                cl_plot([obj.atm.OPD, obj.tel.OPD,  [np.arange(param['nModes']),np.diag(obj.push_pull_gains)],[np.arange(i_loop+i_length),cl_data.ao_residuals[:i_loop+i_length]]], plt_obj = plt_obj)
                if plt_obj.keep_going is False:
                    print('Loop stopped by the user')
                    break
                plt.pause(0.01)
                
        for i_length in range(obj.push_pull_duration):
            # update of the atmosphere
            obj.atm.update()
            # get modal coefficients from DM        
            modal_coeffs = M2C_cl_inv @ dm_cl.coefs
            # force a push/pull on the desired mode
            modal_coeffs[obj.push_pull_index_mode[0]] = -obj.push_pull_amplitude[0]
            print('Pull '+str(i_length))
            # apply it on the dm
            dm_cl.coefs = M2C_cl@modal_coeffs
            # propagate
            obj.tel*dm_cl*obj.wfs
            cl_data.ao_residuals[i_loop+i_length+obj.push_pull_duration]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9
            # control law
            dm_cl.coefs -= np.matmul(M2C_cl,obj.push_pull_gains@np.matmul(reconstructor,wfsSignal))  
            
            if obj.get_forces:
                cl_data.dm_forces[i_loop,:] = obj.P2F_full@dm_cl.coefs
                
            # buffer of the wfs signal to simulate delay
            wfsSignal = obj.wfs.signal
            if i_length>=obj.start_pp:
                print('saving wfs signal')
                sm+=obj.wfs.signal
            print('Loop '+str(i_loop+i_length+obj.push_pull_duration)+'/'+str(cl_data.nLoop)+' Turbulence: '+str(cl_data.ao_turbulence[i_loop+i_length+obj.push_pull_duration])+' -- Residual:' +str(cl_data.ao_residuals[i_loop+i_length+obj.push_pull_duration])+ '\n')
            print('Elapsed Time: ' + str(b-a) + ' s ')
            if obj.display==True:
                cl_plot([obj.atm.OPD, obj.tel.OPD,  [np.arange(param['nModes']),np.diag(obj.push_pull_gains)],[np.arange(i_loop+i_length+obj.push_pull_duration),cl_data.ao_residuals[:i_loop+i_length+obj.push_pull_duration]]], plt_obj = plt_obj)
                if plt_obj.keep_going is False:
                    print('Loop stopped by the user')
                    break
                plt.pause(0.01)
        sp /= (obj.push_pull_duration-obj.start_pp)
        sm /= (obj.push_pull_duration-obj.start_pp)
        push_pull_signal = 0.5*(sp-sm)/obj.push_pull_amplitude[0]
        
        return push_pull_signal
            
            
    cl_data                         = emptyClass()
    cl_data.nLoop                   = obj.n_bootstrap + obj.n_push_pull * (2*obj.push_pull_duration + obj.n_closed_loop)
    cl_data.ao_residuals            = np.zeros(cl_data.nLoop )
    cl_data.ao_turbulence           = np.zeros(cl_data.nLoop )
    cl_data.dm_forces               = np.zeros([cl_data.nLoop,obj.dm.nValidAct])
    
    #% ------------------------------------ destination folder  ------------------------------------
    misregistrationFolder = misRegistration_cl.misRegName
    destinationFolder =  param['pathOutput']+ '/'+param['name'] +'/'+ misregistrationFolder + '/' + extraName + '/'
    createFolder(destinationFolder)
    
    #% ------------------------------------ initialization  ------------------------------------
    
    # combine with atmosphere
    obj.tel+obj.atm
    plt.close('all')
    
    # initialize DM commands and closed loop gain
    dm_cl.coefs = 0
    gain_cl = param['gainCL']
    
    # propagate through th whole system
    obj.ngs*obj.tel*dm_cl*obj.wfs
    
    #% ------------------------------------ variable to save data  ------------------------------------
    wfsSignal   = np.zeros(obj.wfs.nSignal)
    pp_index    = np.linspace(obj.n_bootstrap,cl_data.nLoop-obj.n_closed_loop,obj.n_push_pull).astype(int)
    pp_signal   = []
    
    #% ------------------------------------ Modal Gains  ------------------------------------
    ModalGainsMatrix = obj.gOpt
    # Loop initialization
    reconstructor = np.matmul(ModalGainsMatrix,calib_cl.M)
    

    #% ------------------------------------ setu the display  ------------------------------------
    if obj.display:
            from AO_modules.tools.displayTools import cl_plot
            plt.close('all')
            list_fig = [obj.atm.OPD, obj.tel.OPD, [np.arange(param['nModes']),np.diag(obj.gOpt)],[np.arange(cl_data.nLoop),cl_data.ao_residuals]]
            
            type_fig = ['imshow','imshow','plot','plot']
    
            list_title = ['atm OPD [m]','tel OPD [m]','Optical Gains','WFE [nm]']
            
            plt_obj = cl_plot(list_fig,plt_obj = None, type_fig = type_fig, fig_number = 20, list_ratio = [[1,1,1],[1,1]], list_title = list_title)    
    
    #%%
    
    keep_going = True
    i_loop=-1
    while keep_going:
        i_loop+=1
        a= time.time()
        # update atmospheric phase screen
        obj.atm.update()
        
        # save phase variance
        cl_data.ao_turbulence[i_loop ]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9
            
        # propagate to the WFS with the CL commands applied
        obj.tel*dm_cl*obj.wfs
            
        dm_cl.coefs=dm_cl.coefs-gain_cl*np.matmul(M2C_cl,np.matmul(reconstructor,wfsSignal))  
    
        cl_data.ao_residuals[i_loop ]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9
    
        if obj.get_forces:
            cl_data.dm_forces[i_loop,:] = obj.P2F_full@dm_cl.coefs
            
        # store the slopes after computing the commands => 2 frames delay
        wfsSignal=obj.wfs.signal
        b= time.time()
          
        if obj.display==True:
            if i_loop>1:
                if i_loop >2500:
                    cl_plot([obj.atm.OPD, obj.tel.OPD,  [np.arange(param['nModes']),np.diag(obj.gOpt)],[np.arange(25,i_loop,1),cl_data.ao_residuals[25:i_loop ]]], plt_obj = plt_obj)
                else:
                    cl_plot([obj.atm.OPD, obj.tel.OPD,  [np.arange(param['nModes']),np.diag(obj.gOpt)],[np.arange(i_loop),cl_data.ao_residuals[:i_loop ]]], plt_obj = plt_obj)
                    
                if plt_obj.keep_going is False:
                    print('Loop stopped by the user')
                    break
            plt.pause(0.001)
            
        print('Loop '+str(i_loop)+'/'+str(cl_data.nLoop)+' Turbulence: '+str(cl_data.ao_turbulence[i_loop ])+' -- Residual:' +str(cl_data.ao_residuals[i_loop ])+ '\n')
        print('Elapsed Time: ' + str(b-a) + ' s ')
        if i_loop in pp_index:
            print('Applying the push-pull sequence!')
            tmp = push_pull(obj, dm_cl, wfsSignal,i_loop,cl_data)
            pp_signal.append(tmp)
            
            i_loop +=obj.push_pull_duration*2 -1
            print(i_loop)
            
        if i_loop == cl_data.nLoop-1:
            keep_going = False
            
    push_pull_signal = np.mean(np.asarray(pp_signal), axis = 0 )
    res_OPD = np.reshape(obj.tel.OPD,obj.tel.resolution**2)

#%%

    dataCL = dict()
    dataCL['ao_residual'    ]       = cl_data.ao_residuals
    dataCL['ao_turbulence'  ]       = cl_data.ao_turbulence
    dataCL['dm_forces'      ]       = cl_data.dm_forces
    dataCL['on_sky_signal'  ]       = push_pull_signal
    dataCL['on_sky_signal_list'  ]  = np.asarray(pp_signal)

    dataCL['destinationFolder']     = destinationFolder
    dataCL['last_residual_OPD']     = np.reshape(res_OPD,[obj.tel.resolution,obj.tel.resolution])


    return dataCL



