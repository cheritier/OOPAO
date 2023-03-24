# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:18:31 2020

@author: cheritie
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from ..MisRegistration import MisRegistration
from ..calibration.CalibrationVault import CalibrationVault
from ..mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
from ..tools.tools import createFolder


#####################################################################################################################################################
#     # #   #         #     # # #   # # #           #         #       #     # # #
#   #       #       #   #   #       #               #       #   #   #   #   #    #
#   #       #       #   #   # # #   # #     # # #   #       #   #   #   #   # # #
#   #       #       #   #       #   #               #       #   #   #   #   #
#     # #   # # #     #     # # #   # # #           # # #     #       #     #
#####################################################################################################################################################

def run_cl_first_stage(param,obj,speed_factor = 2,filename_phase_screen = None, extra_name = '', destination_folder = None, get_le_psf = False):
        
    nLoop = param['nLoop']
    gain_cl = param['gainCL']
    
    #% ------------------------------------ Noise Update ------------------------------------
    # set magnitude of the NGS
    # obj.ngs.nPhoton = param['nPhotonPerSubaperture'] * (1/obj.tel.samplingTime)/((obj.tel.D/param['nSubaperture'] )**2)
    obj.ngs.magnitude = param['magnitude']
    obj.ngs.nPhoton = obj.ngs.zeroPoint*10**(-0.4*obj.ngs.magnitude)

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
    if obj.calib.D.shape[1] != param['nModes']:
        calib_cl    = CalibrationVault(obj.calib.D[:,:param['nModes']])
    else:
        calib_cl = obj.calib
    M2C_cl      = obj.M2C
    
    
    #%%
    # def interpolation_function(image):
    #     image = np.reshape(image,[1,image.shape[0],image.shape[1]])
    #     misReg = MisRegistration()
    #     # parameters for the interpolation
    #     nRes = image.shape[1]
    #     pixel_size_in       = obj.tel.D/nRes                # pixel size in [m] of the input influence function
    #     resolution_out      = 1152                          # resolution in pixels of the input influence function
    #     pixel_size_out      = obj.tel.D/resolution_out      # resolution in pixels of the output influence function
    #     mis_registration    = misReg                        # mis-registration object to apply with respect to the input influence function
    #     coordinates_in      = np.zeros([100,2])             # coordinated in [m] of the input influence function
    
    #     P_out,coord_out = interpolate_influence_functions(image,pixel_size_in,pixel_size_out, resolution_out, mis_registration,coordinates_in)
        
    #     return (np.squeeze(P_out.astype(int)))
    #%% ------------------------------------ mis-registrations  ------------------------------------
    
    misRegistration_cl = MisRegistration(param)
    
    # misRegistration_cl.show()
    # obj.dm.misReg.show()
    
#    case with no mis-registration
    if misRegistration_cl == obj.dm.misReg:
        dm_cl = obj.dm
    else:
        dm_cl = applyMisRegistration(obj.tel,misRegistration_cl,param)

    #%% ------------------------------------ closed loop data to be saved  ------------------------------------
    
    wfs_signals             = np.zeros([nLoop,obj.wfs.nSignal])
    dm_commands             = np.zeros([nLoop,obj.dm.nValidAct])
    ao_residuals            = np.zeros(nLoop)
    ao_turbulence           = np.zeros(nLoop)
    residual_phase_screen   = np.zeros([nLoop,obj.tel.resolution,obj.tel.resolution],dtype=np.float32)

    #%% ------------------------------------ destination folder  ------------------------------------
    
    misregistrationFolder = misRegistration_cl.misRegName
    if destination_folder is None:
        destination_folder =  param['pathOutput']+ '/'+param['name'] +'/'+obj.wfs.tag + '/'
    createFolder(destination_folder)
    
    gain_cl = param['gainCL']
    
    # combine with atmosphere
    obj.tel_fast+obj.atm_fast
    plt.close('all')
    
    # initialize DM commands
    dm_cl.coefs=0
    
    #%% ------------------------------------ Modal Gains  ------------------------------------

    ModalGainsMatrix = obj.gOpt
    
    # Loop initialization
    reconstructor = np.matmul(M2C_cl,np.matmul(ModalGainsMatrix,calib_cl.M))
    

    #%%
    # propagate through th whole system
    obj.ngs*obj.tel*dm_cl*obj.wfs
   
    obj.ngs*obj.tel_fast*dm_cl
    
    if obj.display:
        from OOPAO.tools.displayTools import cl_plot
    
        plot_obj = cl_plot(list_fig          = [obj.tel_fast.OPD,obj.tel_fast.OPD],\
                           type_fig          = ['imshow','imshow'],\
                           list_title        = ['Input OPD [m]','Residual OPD [m]'],\
                           list_lim          = [None,None],\
                           list_label        = [None,None],\
                           n_subplot         = [2,1],\
                           list_display_axis = [None,None],\
                           list_ratio        = [[1,1],[1,1]])
#%%   
    wfsSignal = np.zeros(obj.wfs.nSignal)
    
    OPD_buffer              = []
    obj.tel_fast.isPaired   = True
    obj.tel.isPaired        = True
    OPD_first_stage_res     = obj.atm_fast.OPD

    PSF_LE              = []

    for i_loop in range(nLoop):
        
        a= time.time()
        # update atmospheric phase screen
        obj.atm_fast.update()
        
        # skipping the first phase screen
        if i_loop>0:
            	OPD_buffer.append(obj.atm_fast.OPD_no_pupil)
        
        # save phase variance
        ao_turbulence[i_loop]=np.std(obj.atm_fast.OPD[np.where(obj.tel.pupil>0)])*1e9
        
        # save turbulent phase
        OPD_turb = obj.tel_fast.OPD.copy()

        if len(OPD_buffer)==speed_factor:    
            OPD_first_stage = np.mean(OPD_buffer,axis=0)
            obj.tel.OPD_no_pupil     = OPD_first_stage.copy()
            obj.tel.OPD     = OPD_first_stage.copy()*obj.tel.pupil            

            # propagate to the WFS with the CL commands applied
            obj.tel*dm_cl*obj.wfs
            
            OPD_first_stage_res = obj.tel.OPD.copy()
            # reinitialize phase buffer
            OPD_buffer = []
        
            dm_cl.coefs    = dm_cl.coefs-gain_cl*np.matmul(reconstructor,wfsSignal)
            wfsSignal      = obj.wfs.signal
            if get_le_psf:
                if i_loop >100:
                    obj.tel.computePSF(zeroPaddingFactor = 4)
                    PSF_LE.append(obj.tel.PSF)
        
        # else:
        obj.tel_fast*dm_cl
        
        ao_residuals[i_loop]=np.std(obj.tel_fast.OPD[np.where(obj.tel.pupil>0)])*1e9

        b= time.time()
        
        mean_rem_OPD = obj.tel_fast.OPD.copy()
        mean_rem_OPD[np.where(obj.tel.pupil>0)] -= np.mean(mean_rem_OPD[np.where(obj.tel.pupil>0)])
                
        residual_phase_screen[i_loop,:,:] = mean_rem_OPD
        dm_commands[i_loop,:] = dm_cl.coefs.copy()
        wfs_signals[i_loop,:] = wfsSignal.copy()
                
        if obj.display==True:
            cl_plot(list_fig   = [obj.atm_fast.OPD,OPD_first_stage_res],plt_obj = plot_obj)
            plt.pause(0.01)
            if plot_obj.keep_going is False:
                break

        print('Loop'+str(i_loop)+'/'+str(nLoop)+'Turbulence: '+str(ao_turbulence[i_loop])+' -- Residual:' +str(ao_residuals[i_loop])+ '\n')
        print('Elapsed Time: ' + str(b-a) + ' s ')

    dataCL = dict()
    dataCL['ao_residual']          = ao_residuals
    dataCL['ao_turbulence']        = ao_turbulence
    dataCL['parameterFile']        = param
    dataCL['destination_folder']   = destination_folder
    dataCL['long_exposure_psf']    = np.mean(np.asarray(PSF_LE),axis =0)
    
    if filename_phase_screen is None:
        filename_phase_screen           = 'seeing_'+str(np.round(obj.atm.seeingArcsec,2))+'_magnitude_'+str(np.round(obj.ngs.magnitude))+\
        '_nKL_'+str(param['nModes'])+'_nLoop_'+str(param['nLoop'])+'_windspeed_'+str(np.round(np.mean(obj.atm_fast.windSpeed)))+\
        '_stg1_'+str(np.round(0.001/obj.tel.samplingTime,2))+'_kHz_stg2_'+str(np.round(0.001/obj.tel_fast.samplingTime,2))+'_kHz'
    
    dataCL['filename']              = filename_phase_screen+extra_name
    dataCL['dm_commands']           = dm_commands
    dataCL['residual_phase_screen'] = residual_phase_screen
    dataCL['wfs_signals']           = wfs_signals
    
    return dataCL



