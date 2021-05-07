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
from AO_modules.tools.tools                  import createFolder
from AO_modules.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration


#####################################################################################################################################################
#     # #   #         #     # # #   # # #           #         #       #     # # #
#   #       #       #   #   #       #               #       #   #   #   #   #    #
#   #       #       #   #   # # #   # #     # # #   #       #   #   #   #   # # #
#   #       #       #   #       #   #               #       #   #   #   #   #
#     # #   # # #     #     # # #   # # #           # # #     #       #     #
#####################################################################################################################################################

def run_cl(param,obj):
        
    nLoop = param['nLoop']
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
        
    #%% ------------------------------------ mis-registrations  ------------------------------------
    
    misRegistration_cl = MisRegistration(param)
    
    misRegistration_cl.show()
    obj.dm.misReg.show()
    
#    case with no mis-registration
    if misRegistration_cl == obj.dm.misReg:
        dm_cl = obj.dm
    else:
        dm_cl = applyMisRegistration(obj.tel,misRegistration_cl,param)
  
    
    
    #%% ------------------------------------ closed loop data to be saved  ------------------------------------
    
    wfs_signals             = np.zeros([nLoop,obj.wfs.nSignal])
    dm_commands             = np.zeros([nLoop,obj.dm.nValidAct])
    modal_coefficients_res  = np.zeros([nLoop,obj.M2C_cl.shape[1]])
    modal_coefficients_turb = np.zeros([nLoop,obj.M2C_cl.shape[1]])
    ao_residuals            = np.zeros(nLoop)
    ao_turbulence           = np.zeros(nLoop)
    petalling_nm            = np.zeros([6,nLoop])

    #%% ------------------------------------ destination folder  ------------------------------------
    
    misregistrationFolder = misRegistration_cl.misRegName
    destinationFolder =  param['pathOutput']+ '/'+param['name'] +'/'+ misregistrationFolder + '/' + extraName + '/'
    createFolder(destinationFolder)
    
    gain_cl = param['gainCL']
    
    # combine with atmosphere
    obj.tel+obj.atm
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
    
    # setup the display
    if obj.display:
        plt.figure(79)
        plt.ion()
        
        ax1=plt.subplot(2,2,1)
        im_atm=ax1.imshow(obj.tel.src.phase)
        plt.colorbar(im_atm)
        plt.title('Turbulence phase')
        
        
        ax2=plt.subplot(2,2,2)
        im_residual=ax2.imshow(obj.tel.src.phase)
        plt.colorbar(im_residual)
        plt.title('Residual Phase')
        
        ax3=plt.subplot(2,2,3)
        ax3.plot(np.diag(obj.gOpt))
        plt.title('Modal Gains')
        
        ax4 = plt.subplot(2,2,4)
        obj.tel.computePSF(zeroPaddingFactor = 4)
        psf_cl=ax4.imshow(np.log(np.abs(obj.tel.PSF)))
        plt.colorbar(psf_cl)
        plt.title('CL PSF')
        
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
        # propagate to the WFS with the CL commands applied
        obj.tel*dm_cl*obj.wfs
            
        dm_cl.coefs=dm_cl.coefs-gain_cl*np.matmul(reconstructor,wfsSignal)

        if obj.tel.isPetalFree:
            dm_cl.OPD = obj.tel.removePetalling(dm_cl.OPD)
        ao_residuals[i_loop]=np.std(obj.tel.OPD[np.where(obj.tel.pupil>0)])*1e9

        # store the slopes after computing the commands => 2 frames delay
        wfsSignal=obj.wfs.pyramidSignal
        b= time.time()
        
        res_OPD = np.reshape(obj.tel.OPD,obj.tel.resolution**2)
        try:
            modal_coefficients_res[i_loop,:] = obj.projector @ res_OPD
        except:
            if i_loop==0:
                print('Error - no projector for the modal basis..')

        if obj.display==True:
            # turbulence phase
            im_atm.set_data(turbPhase)
            im_atm.set_clim(vmin=turbPhase.min(),vmax=turbPhase.max())
            
            # residual phase
            tmp = obj.tel.src.phase
            tmp-=np.mean(tmp[obj.tel.pupil])
            im_residual.set_data(tmp)
            im_residual.set_clim(vmin=tmp.min(),vmax=tmp.max()) 

            obj.tel.computePSF(zeroPaddingFactor = 6)
            tmp = obj.tel.PSF_trunc/obj.tel.PSF_trunc.max()
            psf_cl.set_data(np.log(np.abs(tmp)))
            psf_cl.set_clim(vmin=-5,vmax=0)             
            plt.draw()
            plt.show()
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
            
        print('Loop'+str(i_loop)+'/'+str(nLoop)+'Turbulence: '+str(ao_turbulence[i_loop])+' -- Residual:' +str(ao_residuals[i_loop])+ '\n')
        print('Elapsed Time: ' + str(b-a) + ' s ')

    


    dataCL = dict()
    dataCL['ao_residual']          = ao_residuals
    dataCL['ao_turbulence']        = ao_turbulence
    if save_modal_coef:
        dataCL['modal_coeff_res']      = modal_coefficients_res
        dataCL['modal_coeff_turb']     = modal_coefficients_turb
    dataCL['parameterFile']        = param
    if obj.printPetals:
        dataCL['petalling']            = petalling_nm
    dataCL['destinationFolder']    = destinationFolder
    dataCL['last_residual_OPD']    = np.reshape(res_OPD,[obj.tel.resolution,obj.tel.resolution])

    try:
        if obj.perf_only:
            return dataCL
        else:
            dataCL['wfs_signals']          = wfs_signals
            dataCL['dmCommands']           = dm_commands
    except:
        dataCL['wfs_signals']          = wfs_signals
        dataCL['dmCommands']           = dm_commands
        #    dataCL['m4_cube_map']          = np.reshape(np.sum((dm_cl.modes)**3,axis=1),[obj.tel.resolution,obj.tel.resolution])
        return dataCL



