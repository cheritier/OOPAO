# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:04:46 2021

@author: cheritie
"""

#    def interpolateInfluenceFunctions(self,influenceFunctions_in,misReg, ):
#        
#        nx, ny, nAct = influenceFunctions_in.shape
#            
#        pixelSize_DM_original = self.D/nx
#                        
#        # size of the influence functions maps
#        resolution_DM_original       = int(nx)   
#        
#        # resolution of the M1 pupil
#        resolution_M1                = tel.pupil.shape[1]
#        
#        # compute the pixel scale of the M1 pupil
#        pixelSize_M1                 = 8.25/resolution_M1
#        
#        # compute the ratio_DM_M1 between both pixel scale.
#        ratio_DM_M1                  = pixelSize_DM_original/pixelSize_M1
#        # after the interpolation the image will be shifted of a fraction of pixel extra if ratio_DM_M1 is not an integer
#        extra = (ratio_DM_M1)%1 
#        
#        # difference in pixels between both resolutions    
#        nPix = resolution_DM_original-resolution_M1
#        
#        if nPix%2==0:
#            # case nPix is even
#            # alignement of the array with respect to the interpolation 
#            # (ratio_DM_M1 is not always an integer of pixel)
#            extra_x = extra/2 -0.5
#            extra_y = extra/2 -0.5
#            
#            # crop one extra pixel on one side
#            nCrop_x = nPix//2
#            nCrop_y = nPix//2
#        else:
#            # case nPix is uneven
#            # alignement of the array with respect to the interpolation 
#            # (ratio_DM_M1 is not always an integer of pixel)
#            extra_x = extra/2 -0.5 -0.5
#            extra_y = extra/2 -0.5 -0.5
#            # crop one extra pixel on one side
#            nCrop_x = nPix//2
#            nCrop_y = (nPix//2)+1
#               
#        # allocate memory to store the influence functions
#        influMap = np.zeros([resolution_DM_original,resolution_DM_original])  
#        
#        #-------------------- The Following Transformations are applied in the following order -----------------------------------
#           
#        # 1) Down scaling to get the right pixel size according to the resolution of M1
#        downScaling     = anamorphosisImageMatrix(influMap,0,[ratio_DM_M1,ratio_DM_M1])
#        
#        # 2) transformations for the mis-registration
#        anamMatrix              = anamorphosisImageMatrix(influMap,misReg.anamorphosisAngle,[1+misReg.radialScaling,1+misReg.tangentialScaling])
#        rotMatrix               = rotateImageMatrix(influMap,misReg.rotationAngle)
#        shiftMatrix             = translationImageMatrix(influMap,[misReg.shiftY/pixelSize_M1,misReg.shiftX/pixelSize_M1]) #units are in m
#        
#        # Shift of half a pixel to center the images on an even number of pixels
#        alignmentMatrix         = translationImageMatrix(influMap,[extra_x,extra_y])
#            
#        # 3) Global transformation matrix
#        transformationMatrix    = downScaling + anamMatrix + rotMatrix + shiftMatrix + alignmentMatrix
#        
#        def globalTransformation(image):
#                output  = sk.warp(image,(transformationMatrix).inverse,order=3)
#                return output
#        
#        # definition of the function that is run in parallel for each 
#        def reconstruction_IF(influMap):
#            output = globalTransformation(influMap)  
#            return output
#        
#        
#        
#        print('Reconstructing the influence functions ... ')    
#        def joblib_reconstruction():
#            Q=Parallel(n_jobs=4,prefer='threads')(delayed(reconstruction_IF)(i) for i in influenceFunctions_in)
#            return Q 
#        
#        
#        influenceFunctions_tmp =  np.moveaxis(np.asarray(joblib_reconstruction()),0,-1)
#        influenceFunctions_tmp = influenceFunctions_tmp  [nCrop_x:-nCrop_y,nCrop_x:-nCrop_y,:]
#        
#        return np.reshape(influenceFunctions_out,[self.resolution*self.resolution,self.nValidAct])    