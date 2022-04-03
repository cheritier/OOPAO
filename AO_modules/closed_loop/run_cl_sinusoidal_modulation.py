# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:18:31 2020

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

def run_cl_sinusoidal_modulation(param,obj):
    

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


    M2C_cl      = obj.M2C_cl
    
    calib_cl    = calibrationVault(obj.calib.D[:,:param['nModes']])


    #% ------------------------------------ M4 Saturations ------------------------------------
    
    P2F_full = np.zeros([6*892,6*892])
    try: 
        for i in range(6):
            P2F_full[i*892:(i+1)*892,i*892:(i+1)*892] = obj.P2F    	
        get_forces = True
        print('Stiffness matrix properly attached to the ao-object!')

    except:            
        print('No stiffness matrix attached to the ao-object.. computation of the M4 forces not considered')
        get_forces = False
    #%% ------------------------------------ mis-registrations  ------------------------------------
    
    misRegistration_cl = MisRegistration(param)
    
    misRegistration_cl.show()
    obj.dm.misReg.show()
    
#    case with no mis-registration
    if misRegistration_cl == obj.dm.misReg:
        print('No need to update M4')
    else:
        obj.dm = applyMisRegistration(obj.tel,misRegistration_cl,param)
        
    #%%  modulation signal
    
    obj.n_iteration_per_mode =  np.zeros(obj.n_multiplexed_modulated_modes, dtype = int)
    obj.n_iteration_per_period =  np.zeros(obj.n_multiplexed_modulated_modes, dtype = int)
    obj.modulation_index_frequency =  np.zeros(obj.n_multiplexed_modulated_modes, dtype = int)
    
    for i_mode in range(obj.n_multiplexed_modulated_modes):
        period              = 1/obj.modulation_frequency[i_mode]
        n_period            = obj.n_modulation_period 
    
    
        criterion = period/obj.tel.samplingTime
        target = int(np.round(criterion))
        period = target*obj.tel.samplingTime
        print('Mode '+str(obj.modulation_index_mode[i_mode])+' -- Requested frequency: '+str(obj.modulation_frequency[i_mode]) + ' Hz' )
        obj.modulation_frequency[i_mode]                = 1/period
        print('Mode '+str(obj.modulation_index_mode[i_mode])+' -- Updated frequency: '+str(obj.modulation_frequency[i_mode]) + ' Hz' )
    
        criterion = period/obj.tel.samplingTime
        
        N_tmp           = int(np.ceil(n_period*criterion))
        print(N_tmp)
        criteria = True
    
        while criteria:
            u = (np.fft.fftfreq(N_tmp, d =obj.tel.samplingTime))
            if obj.modulation_frequency[i_mode] in u:
                criteria = False
                index_freq = list(u).index(obj.modulation_frequency[i_mode])
            else:
                N_tmp+=1
        print(N_tmp)
    
        print('Mode '+str(obj.modulation_index_mode[i_mode])+' -- Sinusoidal modulation sampled with '+str(criterion) + ' points per period -- total of iterations required: '+str(N_tmp))
        obj.n_iteration_per_period[i_mode]      = int(criterion)
        obj.n_iteration_per_mode[i_mode]        = int(N_tmp)
        obj.modulation_index_frequency[i_mode]  = int(index_freq )
    
    # TAKING THE MAXIMUM NUMBER OF ITERATION
    N = int(np.max(obj.n_iteration_per_mode))
    N_real = N+obj.n_iteration_per_period[i_mode]
    obj.modulation_signal =  np.zeros([obj.n_multiplexed_modulated_modes, N_real])

    
    t                                   = np.linspace(0,N_real*obj.tel.samplingTime,N_real, endpoint = True)

    t_measurement                                   = np.linspace(0,N*obj.tel.samplingTime,N_real, endpoint = True)
    
    obj.sin = np.sin(t_measurement*obj.modulation_frequency[i_mode]*2.0*np.pi) 
    obj.cos = np.cos(t_measurement*obj.modulation_frequency[i_mode]*2.0*np.pi) 
    
    for i_mode in range(obj.n_multiplexed_modulated_modes):
        obj.modulation_signal[i_mode,:] =  (obj.modulation_amplitude[i_mode])*np.sin(t*obj.modulation_frequency[i_mode]*2.0*np.pi + obj.modulation_phase_shift[i_mode]) 
        
    nLoop = N_real+param['nLoop']
    #%% ------------------------------------ closed loop data to be saved  ------------------------------------
    
    dm_forces               = np.zeros([nLoop,obj.dm.nValidAct])
    modal_coefficients_res  = np.zeros([nLoop,obj.M2C_cl.shape[1]])
    modal_coefficients_turb = np.zeros([nLoop,obj.M2C_cl.shape[1]])
    ao_residuals            = np.zeros(nLoop)
    ao_turbulence           = np.zeros(nLoop)
    petalling_nm            = np.zeros([6,nLoop])
    res_OPD_buff            = np.zeros([nLoop,obj.tel.resolution,obj.tel.resolution])
    # buffer for demodulations
    buffer_dm               = np.zeros([obj.dm.nValidAct,N])
    buffer_wfs              = np.zeros([obj.wfs.nSignal,N])

    #%% ------------------------------------ destination folder  ------------------------------------
    
    misregistrationFolder = misRegistration_cl.misRegName
    destinationFolder =  param['pathOutput']+ '/'+param['name'] +'/'+ misregistrationFolder + '/' + extraName + '/'
    createFolder(destinationFolder)
    
    gain_cl = param['gainCL']
    
    # combine with atmosphere
    obj.tel+obj.atm
    plt.close('all')
    
    # initialize DM commands
    obj.dm.coefs=0
    
    #%% ------------------------------------ Modal Gains  ------------------------------------


    ModalGainsMatrix = obj.gOpt
    
    # Loop initialization
#    reconstructor = np.matmul(M2C_cl,np.matmul(ModalGainsMatrix,calib_cl.M))
    reconstructor = np.matmul(ModalGainsMatrix,calib_cl.M)
    

    #%%
    # propagate through th whole system
    obj.ngs*obj.tel*obj.dm*obj.wfs
    
    # setup the display
    if obj.display:
            from AO_modules.tools.displayTools import cl_plot
            plt.close('all')
            list_fig = [obj.atm.OPD, obj.tel.OPD, [np.arange(param['nModes']),np.diag(obj.gOpt)],[np.arange(nLoop),ao_residuals]]
            
            type_fig = ['imshow','imshow','plot','plot']

            list_title = ['atm OPD [m]','tel OPD [m]','Optical Gains','WFE [nm]']
            
            plt_obj = cl_plot(list_fig,plt_obj = None, type_fig = type_fig, fig_number = 20, list_ratio = [[1,1,1],[1,1]], list_title = list_title)    

        
    if obj.displayPetals:
        if obj.dm.isM4:
            plt.figure(80)
            ax_p0   = plt.subplot(2,3,1)
            p       = obj.tel.getPetalOPD(0)
            pet_0   = ax_p0.imshow(p)
            plt.colorbar(pet_0)
            plt.title('Petal mean value:' + str(1e9*np.mean(p[np.where(obj.tel.petalMask == 1)])) + ' nm')
    
            ax_p1   = plt.subplot(2,3,2)
            p       = obj.tel.getPetalOPD(1)
            pet_1   = ax_p1.imshow(p)
            plt.colorbar(pet_1)
            plt.title('Petal mean value:' + str(1e9*np.mean(p[np.where(obj.tel.petalMask == 1)])) + ' nm')
            
            ax_p2   = plt.subplot(2,3,3)
            p       = obj.tel.getPetalOPD(2)
            pet_2   = ax_p2.imshow(p)
            plt.colorbar(pet_2)
            plt.title('Petal mean value:' + str(1e9*np.mean(p[np.where(obj.tel.petalMask == 1)])) + ' nm')
            
            ax_p3   = plt.subplot(2,3,4)
            p       = obj.tel.getPetalOPD(3)
            pet_3   = ax_p3.imshow(p)
            plt.colorbar(pet_3)
            plt.title('Petal mean value:' + str(1e9*np.mean(p[np.where(obj.tel.petalMask == 1)])) + ' nm')
            
            ax_p4   = plt.subplot(2,3,5)
            p       = obj.tel.getPetalOPD(4)
            pet_4   = ax_p4.imshow(p)
            plt.colorbar(pet_4)
            plt.title('Petal mean value:' + str(1e9*np.mean(p[np.where(obj.tel.petalMask == 1)])) + ' nm')
            
            ax_p5   = plt.subplot(2,3,6)
            p       = obj.tel.getPetalOPD(5)
            pet_5   = ax_p5.imshow(p)
            plt.colorbar(pet_5)
            plt.title('Petal mean value:' + str(1e9*np.mean(p[np.where(obj.tel.petalMask == 1)])) + ' nm')
        
        
#%%   
    wfsSignal = np.zeros(obj.wfs.nSignal)
    count = -1
    count_bootstrap = -1


    for i_loop in range(nLoop):
        a= time.time()
        # update atmospheric phase screen
        obj.atm.update()
        
        # save phase variance
        ao_turbulence[i_loop]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9
        
        # save turbulent phase
        turbPhase=obj.tel.src.phase
        turb_OPD = np.reshape(obj.tel.OPD,obj.tel.resolution**2)
        
        try:
            modal_coefficients_turb[i_loop,:] = obj.projector @ turb_OPD
            save_modal_coef = True
        except:
            save_modal_coef = False
            if i_loop == 0:
                print('Error - no projector for the modal basis..')
                
                
        # apply the modulation of the modes
        if i_loop>=param['nLoop']:
            count_bootstrap+=1
            print('Applying perturbation')

            if count_bootstrap>= obj.n_iteration_per_period[0]:
                count+=1
            dm_perturbation_in = np.zeros(obj.dm.nValidAct)
            if count_bootstrap==0:
                dm_perturbation_out = np.zeros(obj.dm.nValidAct)
                
            for i_mode in range(obj.n_multiplexed_modulated_modes):
                    dm_perturbation_in += obj.modulation_M2C[:,obj.modulation_index_mode[i_mode]]*obj.modulation_signal[i_mode,count_bootstrap]
                    
            obj.dm.coefs+= dm_perturbation_in 
        # propagate to the WFS with the CL commands applied
        obj.tel*obj.dm*obj.wfs
        
        if i_loop == param['nLoop']-1:
                res_OPD_to_save = obj.tel.OPD 
            
        if i_loop>=param['nLoop']:

            if count_bootstrap>= obj.n_iteration_per_period[0]:
                	buffer_dm[:,count]  = obj.dm.coefs
                    
        res_OPD_buff[i_loop,:,:] = obj.tel.OPD.copy()                 
        obj.dm.coefs=obj.dm.coefs-gain_cl*np.matmul(M2C_cl,np.matmul(reconstructor,wfsSignal))
        
        if get_forces:
            dm_forces[i_loop,:] = P2F_full@obj.dm.coefs
            

        if i_loop>=param['nLoop']:
            if count_bootstrap>= obj.n_iteration_per_period[0]:
                print('Acquirring WFS signal')
                buffer_wfs[:,count] = obj.wfs.signal
            dm_perturbation_out = np.zeros(obj.dm.nValidAct)

            for i_mode in range(obj.n_multiplexed_modulated_modes):
                dm_perturbation_out -= obj.modulation_M2C[:,obj.modulation_index_mode[i_mode]]*obj.modulation_signal[i_mode,count_bootstrap]*obj.modulation_OG[i_mode]
            obj.dm.coefs+= dm_perturbation_out

        if obj.tel.isPetalFree:
            obj.dm.OPD = obj.tel.removePetalling(obj.dm.OPD)
        ao_residuals[i_loop]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9

        # store the slopes after computing the commands => 2 frames delay
        wfsSignal=obj.wfs.signal
        b= time.time()
        
        res_OPD = np.reshape(obj.tel.OPD,obj.tel.resolution**2)
        try:
            modal_coefficients_res[i_loop,:] = obj.projector @ res_OPD
        except:
            if i_loop==0:
                print('Error - no projector for the modal basis..')

        if obj.display==True:
            if i_loop>1:
                if i_loop >25000:
                    cl_plot([obj.atm.OPD, obj.tel.OPD,  [np.arange(param['nModes']),np.diag(obj.gOpt)],[np.arange(25,i_loop,1),ao_residuals[25:i_loop]]], plt_obj = plt_obj)
                else:
                    cl_plot([obj.atm.OPD, obj.tel.OPD,  [np.arange(param['nModes']),np.diag(obj.gOpt)],[np.arange(i_loop),ao_residuals[:i_loop]]], plt_obj = plt_obj)
                    
                if plt_obj.keep_going is False:
                    print('Loop stopped by the user')
                    break
            plt.pause(0.001)
            
        if obj.displayPetals:
            if obj.dm.isM4:
                p       = obj.tel.getPetalOPD(0)*1e9            
                pet_0.set_data(p)
                pet_0.set_clim(vmin=p.min(),vmax=p.max())
                ax_p0.set_title('Petal mean value:' + str(np.round(np.mean(p[np.where(obj.tel.petalMask == 1)]),5)) + ' nm')
                
                p       = obj.tel.getPetalOPD(1)*1e9            
                pet_1.set_data(p)
                pet_1.set_clim(vmin=p.min(),vmax=p.max())
                ax_p1.set_title('Petal mean value:' + str(np.round(np.mean(p[np.where(obj.tel.petalMask == 1)]),5)) + ' nm')
    
                p       = obj.tel.getPetalOPD(2)*1e9             
                pet_2.set_data(p)
                pet_2.set_clim(vmin=p.min(),vmax=p.max())
                ax_p2.set_title('Petal mean value:' + str(np.round(np.mean(p[np.where(obj.tel.petalMask == 1)]),5)) + ' nm')
    
                p       = obj.tel.getPetalOPD(3)*1e9             
                pet_3.set_data(p)
                pet_3.set_clim(vmin=p.min(),vmax=p.max())
                ax_p3.set_title('Petal mean value:' + str(np.round(np.mean(p[np.where(obj.tel.petalMask == 1)]),5)) + ' nm')
    
                p       = obj.tel.getPetalOPD(4)*1e9             
                pet_4.set_data(p)
                pet_4.set_clim(vmin=p.min(),vmax=p.max())
                ax_p4.set_title('Petal mean value:' + str(np.round(np.mean(p[np.where(obj.tel.petalMask == 1)]),5)) + ' nm')
    
                
                p       = obj.tel.getPetalOPD(5)*1e9             
                pet_5.set_data(p)
                pet_5.set_clim(vmin=p.min(),vmax=p.max())            
                ax_p5.set_title('Petal mean value:' + str(np.round(np.mean(p[np.where(obj.tel.petalMask == 1)]),5)) + ' nm')
    
                plt.draw()
                plt.show()
                plt.pause(0.001)
        if obj.printPetals:
            if obj.dm.isM4:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                # remove mean OPD
                obj.tel.OPD[np.where(obj.tel.pupil == 1)] -= np.mean(obj.tel.OPD[np.where(obj.tel.pupil == 1)])
                for i_petal in range(6):
                    petal       = obj.tel.getPetalOPD(i_petal)*1e9             
                    petal_nm    = np.round(np.mean(petal[np.where(obj.tel.petalMask == 1)]),5)
                    print('Petal '+str(i_petal+1) +': ' + str(petal_nm))
                    petalling_nm[i_petal,i_loop] = petal_nm
                    
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            
        print('Loop '+str(i_loop)+'/'+str(nLoop)+' Turbulence: '+str(ao_turbulence[i_loop])+' -- Residual:' +str(ao_residuals[i_loop])+ '\n')
        print('Elapsed Time: ' + str(b-a) + ' s ')

    


#%% DEMODULATION AMPLITUDE
        
    modal_coefs = np.linalg.pinv(obj.modulation_M2C)@buffer_dm
    

   
#%% SLOPESMAPS DEMODULATION
    from joblib import Parallel,delayed


    
    def demodulate_function(obj, buffer_wfs, modal_coefs, i_mode, n_period):        
        # test that the n_period can be requested
        N_requested = int((n_period) *obj.n_iteration_per_period[i_mode])
        
        criteria = True
    
        while criteria:
            u = (np.fft.fftfreq(N_requested, d =obj.tel.samplingTime))
            if obj.modulation_frequency[i_mode] in u:
                criteria = False
                index_freq_temp = list(u).index(obj.modulation_frequency[i_mode])
            else:
                N_requested+=1
        
        print('number of period condsidered: ' + str(N_requested/obj.n_iteration_per_period[i_mode]))
        
        
        def demodulate(temporal_series_wfs):   
            
            y_f = np.fft.fft(temporal_series_wfs[:N_requested])
            
            y_f_mod = np.abs(y_f)
            
            y_f_phi = np.arctan2(np.real(y_f[index_freq_temp]),np.imag(y_f[index_freq_temp])) 
        
            amp = 2.*y_f_mod[index_freq_temp]/N_requested
        
            return amp, y_f_phi
    
        fft_modal_coefs         = np.fft.fft(modal_coefs[obj.modulation_index_mode[i_mode],:N_requested])
        
        fft_frequency_vector = np.fft.fftfreq(len(fft_modal_coefs), d =obj.tel.samplingTime)
    
        tmp_demodulated_amplitude   = 2.0/N_requested * np.abs(fft_modal_coefs[index_freq_temp])
        
        
        def job_loop_demodulate():
            Q = Parallel(n_jobs=4,prefer='threads')(delayed(demodulate)(i) for i in (buffer_wfs[:,:N_requested]))
            return Q 
    
        maps = np.asarray(job_loop_demodulate())
    
        tmp_demodulated_phase       = maps[:,1]
        tmp_demodulated_wfs_signal  = (maps[:,0]*(np.sign(np.sin(tmp_demodulated_phase))))/tmp_demodulated_amplitude
        
        return tmp_demodulated_phase,tmp_demodulated_wfs_signal, tmp_demodulated_amplitude, N_requested, fft_modal_coefs,fft_frequency_vector
    
    demodulated_phase      = np.zeros([obj.n_multiplexed_modulated_modes,obj.wfs.nSignal])
    demodulated_wfs_signal = np.zeros([obj.n_multiplexed_modulated_modes,obj.wfs.nSignal])
    demodulated_amplitude  = np.zeros([obj.n_multiplexed_modulated_modes])
    
    fft_frequency_vector = []
    fft_modal_coefs      = []
    for i_mode in range(obj.n_multiplexed_modulated_modes):
        tmp_demodulated_phase,tmp_demodulated_wfs_signal, tmp_demodulated_amplitude,N,tmp_fft_modal_coefs,tmp_fft_frequency_vector= demodulate_function(obj, buffer_wfs, modal_coefs,i_mode, n_period = obj.n_modulation_period)
        
        demodulated_phase[i_mode,:]      = tmp_demodulated_phase
        demodulated_wfs_signal[i_mode,:] = tmp_demodulated_wfs_signal
        demodulated_amplitude[i_mode]    = tmp_demodulated_amplitude
        fft_frequency_vector.append(tmp_fft_frequency_vector)
        fft_modal_coefs.append(tmp_fft_modal_coefs)
        


#%%

    dataCL = dict()
    dataCL['ao_residual'            ] = ao_residuals
    dataCL['ao_turbulence'          ] = ao_turbulence
    dataCL['demodulate_function'    ] = demodulate_function
    dataCL['modulation_signal'      ] = obj.modulation_signal
    dataCL['modulation_frequency'   ] = obj.modulation_frequency
    
    dataCL['demodulated_phase'      ] = demodulated_phase
    dataCL['demodulated_amplitude'  ] = demodulated_amplitude
    dataCL['demodulated_wfs_signal' ] = demodulated_wfs_signal
    dataCL['fft_modal_coefs'        ] = fft_modal_coefs
    dataCL['fft_frequency_vector'   ] = fft_frequency_vector
    # dataCL['dm_forces']           = dm_forces
    # dataCL['wfs_signals']         = buffer_wfs
    dataCL['modal_coeff']         = modal_coefs
    # dataCL['dm_commands']         = buffer_dm


    try:
        if obj.save_residual_phase:
            dataCL['residual_phase']      = res_OPD_buff
    except:
            dataCL['residual_phase']      = None
        
            
        
    if save_modal_coef:
        dataCL['modal_coeff_res']      = modal_coefficients_res
        dataCL['modal_coeff_turb']     = modal_coefficients_turb
        
    dataCL['parameterFile']        = param
    if obj.printPetals:
        dataCL['petalling']            = petalling_nm
    dataCL['destinationFolder']    = destinationFolder
    dataCL['last_residual_OPD']    = res_OPD_to_save

    try:
        if obj.perf_only:
            return dataCL
        else:
            dataCL['wfs_signals']         = buffer_wfs
            dataCL['dm_commands']         = buffer_dm
            dataCL['dm_forces']           = dm_forces
            return dataCL

    except:
        dataCL['wfs_signals']          = buffer_wfs
        dataCL['dmCommands']           = buffer_dm
        #    dataCL['m4_cube_map']          = np.reshape(np.sum((obj.dm.modes)**3,axis=1),[obj.tel.resolution,obj.tel.resolution])
        return dataCL



