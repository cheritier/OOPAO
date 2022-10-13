# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:18:31 2020

@author: cheritie
"""

import matplotlib.pyplot as plt
import numpy             as np 
import time
import scipy
from astropy.io import fits as pfits
# local modules 
from AO_modules.calibration.CalibrationVault  import calibrationVault
from AO_modules.MisRegistration   import MisRegistration
from AO_modules.tools.tools       import createFolder,write_fits
from AO_modules.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
from AO_modules.tools.interpolate_influence_functions                          import interpolate_influence_functions


#####################################################################################################################################################
#     # #   #         #     # # #   # # #           #         #       #     # # #
#   #       #       #   #   #       #               #       #   #   #   #   #    #
#   #       #       #   #   # # #   # #     # # #   #       #   #   #   #   # # #
#   #       #       #   #       #   #               #       #   #   #   #   #
#     # #   # # #     #     # # #   # # #           # # #     #       #     #
#####################################################################################################################################################

def run_cl_two_stages(param,obj,speed_factor = 2,filename_phase_screen = None, extra_name = '', destination_folder = None, get_le_psf = False):
        
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
        calib_cl    = calibrationVault(obj.calib.D[:,:param['nModes']])
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
    
    # setup the display
    if obj.display:
        plt.figure(79)
        plt.ion()
        
        ax1=plt.subplot(1,3,1)
        im_atm = ax1.imshow(obj.tel.src.phase)
        plt.colorbar(im_atm,fraction=0.046, pad=0.04)
        plt.title('Turbulence OPD [nm]')
        
        
        ax2=plt.subplot(1,3,2)
        im_residual_1 = ax2.imshow(obj.tel.src.phase)
        plt.colorbar(im_residual_1,fraction=0.046, pad=0.04)
        plt.title('Residual OPD First stage [nm]')
        
        ax3=plt.subplot(1,3,3)
        im_residual_2 = ax3.imshow(obj.tel.src.phase)
        plt.colorbar(im_residual_2,fraction=0.046, pad=0.04)
        plt.title('Intermediate OPD [nm]')
        
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
            # turbulence phase
            im_atm.set_data(OPD_turb)
            im_atm.set_clim(vmin=OPD_turb.min(),vmax=OPD_turb.max())
            
            # residual phase
            tmp = obj.tel_fast.OPD.copy()
            # tmp-=np.mean(tmp[obj.tel.pupil])
            im_residual_2.set_data(tmp)
            im_residual_2.set_clim(vmin=tmp.min(),vmax=tmp.max()) 

            tmp = OPD_first_stage_res
            # tmp-=np.mean(tmp[obj.tel.pupil])
            im_residual_1.set_data(tmp)
            im_residual_1.set_clim(vmin=tmp.min(),vmax=tmp.max()) 
            plt.draw()
            
            plt.show()
            plt.pause(0.001)

        print('Loop'+str(i_loop)+'/'+str(nLoop)+'Turbulence: '+str(ao_turbulence[i_loop])+' -- Residual:' +str(ao_residuals[i_loop])+ '\n')
        print('Elapsed Time: ' + str(b-a) + ' s ')

    dataCL = dict()
    dataCL['ao_residual']          = ao_residuals
    dataCL['ao_turbulence']        = ao_turbulence
    dataCL['parameterFile']        = param
    dataCL['destination_folder']   = destination_folder
    dataCL['long_exposure_psf']    = np.mean(np.asarray(PSF_LE),axis =0)
    
    if filename_phase_screen is None:
        filename_phase_screen           = 'phi_'+'seeing_'+str(np.round(obj.atm.seeingArcsec,2))+'_magnitude_'+str(np.round(obj.ngs.magnitude))+\
        '_nKL_'+str(param['nModes'])+'_nLoop_'+str(param['nLoop'])+'_windspeed_'+str(np.round(np.mean(obj.atm_fast.windSpeed)))+\
        '_stg1_'+str(np.round(0.001/obj.tel.samplingTime,2))+'_kHz_stg2_'+str(np.round(0.001/obj.tel_fast.samplingTime,2))+'_kHz'
    
    dataCL['filename']              = filename_phase_screen+extra_name
    dataCL['dm_commands']           = dm_commands
    dataCL['residual_phase_screen'] = residual_phase_screen
    dataCL['wfs_signals']           = wfs_signals
    
    # hdr             = pfits.Header()
    # hdr['TITLE']    = 'phase screens'
    # primary_hdu   = pfits.PrimaryHDU(residual_phase_screen)
    # hdu             = pfits.HDUList([primary_hdu])
    # # save output
    # hdu.writeto(destination_folder+filename_phase_screen+'.fits',overwrite=True)
    # hdu.close()
    
    # hdr             = pfits.Header()
    # hdr['TITLE']    = 'wfs signals'
    # primary_hdu   = pfits.PrimaryHDU(wfs_signals)
    # hdu             = pfits.HDUList([primary_hdu])
    # # save output
    # hdu.writeto(destination_folder+filename_phase_screen+'_wfs_signals.fits',overwrite=True)
    # hdu.close()
    
    # hdr             = pfits.Header()
    # hdr['TITLE']    = 'dm commands'
    # primary_hdu   = pfits.PrimaryHDU(dm_commands)
    # hdu             = pfits.HDUList([primary_hdu])
    # # save output
    # hdu.writeto(destination_folder+filename_phase_screen+'_dm_commands.fits',overwrite=True)
    # hdu.close()
    
    return dataCL



