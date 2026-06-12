# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:43:10 2022

@author: cheritie
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# generic function to create an AO system with Pyramid WFS
from compute_ao_model_pyramid import compute_ao_model_pyramid

# display modules
from OOPAO.tools.displayTools           import displayMap, display_wfs_signals, makeSquareAxes,interactive_show

# % -----------------------     read parameter file   ----------------------------------
from parameterFile_GHOST import initializeParameterFile, get_imat_ghost, compute_imat_4_ghost
param = initializeParameterFile()

#%%
# Overrides location of the relevant data (WFS pupil mask, KL basis, measured iMat... )
# param['location_data'] = 'C:/Users/cheritier/Documents/oopao_private/ghost/ghost_simulation/'

# create the ghost AO objects
tel,ngs,dm,wfs,atm = compute_ao_model_pyramid(param = param, loc = param['location_data'])


#%% Data from the bench
plt.close('all')
# 1 --  Modal basis
# check that the KL basis did not change
M2C = np.load(param['location_data']+'KL_PTT_ifun_BMC492_psim.npy')

dm.coefs = M2C[:,10:20]*1e-9

ngs**tel*dm*wfs

# show the KL modes projected on the synthetic DM
displayMap(tel.OPD, norma = True)
plt.title('Modal Basis')

# show the corresponding synthetic WFS signals
synthetic_imat = display_wfs_signals(wfs, wfs.signal, norma=True,returnOutput=True)
synthetic_imat[np.isinf(synthetic_imat)] = 0

# 2 --  Interaction Matrix

nm_i_mat = 'hadamard_IM_slopes_15_no_residual.npy'

#% read and display the input imat
input_imat = get_imat_ghost(param['location_data']+nm_i_mat, wfs, M2C=M2C[:,:400]) # consider only 400 modes in this example


# display the modes 10 to 20
exp_imat = display_wfs_signals(wfs, input_imat[:,10:20], norma = True,returnOutput=True)
exp_imat[np.isinf(exp_imat)] = 0

synthetic_imat = display_wfs_signals(wfs, wfs.signal, norma=True,returnOutput=True)
synthetic_imat[np.isinf(synthetic_imat)] = 0


plt.figure()
plt.subplot(1,2,1)
plt.imshow(exp_imat)
plt.title('Experimental Interction Matrix')
plt.subplot(1,2,2)
plt.imshow(synthetic_imat)
plt.title('Synthetic Interction Matrix')



#%% Check for potential flips using a few actuators randomly distributed

#% read and display the input imat
zonal_input_imat = get_imat_ghost(param['location_data']+nm_i_mat, wfs)

index = [30,36,50,140,144,204,320]

# display the modes 10 to 20
exp_imat = display_wfs_signals(wfs, np.sum(zonal_input_imat[:,index],axis=1), norma = True,returnOutput=True)
exp_imat[np.isinf(exp_imat)] = 0

dm.coefs=0
dm.set_coefs_value(index, [1e-9]*len(index))

ngs**tel*dm*wfs


synthetic_imat = display_wfs_signals(wfs, wfs.signal, norma=True,returnOutput=True)
synthetic_imat[np.isinf(synthetic_imat)] = 0


plt.figure()
plt.subplot(1,2,1)
plt.imshow(exp_imat)
plt.title('Experimental Interction Matrix')
plt.subplot(1,2,2)
plt.imshow(synthetic_imat)
plt.title('Synthetic Interction Matrix')

#%% Pyramid Pupil position Calibration
from tools import check_wfs_pupils
# assuming the full interaction matrix is available with intensities with the shape [n_actuators, n_pix,n_pix]

# interaction_matrix_variance = np.var(interaction_matrix,axis=0)
# simulate it for now shifting the pyramid pupils
wfs.apply_shift_wfs(sx = [3,-2,1,1],sy = [1,0.5,1,2])
# save variance map as a binary mask
interaction_matrix_variance = wfs.cam.frame>wfs.cam.frame.max()*0.3
# recenter the PWFS
wfs.apply_shift_wfs()

check_wfs_pupils(interaction_matrix_variance, wfs,correct=False)
#%% DM/WFS registration using SPRINT

# index of the KL modes used to estimate the parameters (for the first estimation pick a few middle-order within the first 100 KL modes -- avoid high order modes)
index_modes = 80

# modal basis considered
from OOPAO.tools.tools import emptyClass
basis               = emptyClass()
basis.modes         = M2C[:,index_modes]
basis.extra         = 'ghost_smat'

# compact the AO objects into a single object obj
obj         =  emptyClass()
obj.ngs     = ngs
obj.tel     = tel
obj.wfs     = wfs
obj.dm      = dm
obj.atm     = atm
obj.param   = param


from OOPAO.SPRINT import SPRINT
tel.resetOPD()
n_mis_reg             = 5           # consider the 5 mis-reg shift X & Y, rotation, magnification X & Y
recompute_sensitivity = True        # forces to recompute the sensitivity matrices. If False, the pre-eisting ones are loaded if they exist.
mis_registration_zero_point = None  # mis-registration starting point. if None, zero-mis-registration is considered

# create the SPRINT object
Sprint = SPRINT(obj                         = obj,\
                basis                       = basis,\
                n_mis_reg                   = n_mis_reg,\
                mis_registration_zero_point = None,\
                recompute_sensitivity       = True)

    
# estimate the parameters
Sprint.estimate(obj, input_imat[:,index_modes],n_iteration=6,n_update_zero_point=0)   


plt.close('all')
from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration

dm_ghost = applyMisRegistration(tel,Sprint.mis_registration_out,dm_input = dm)

input_imat = get_imat_ghost(param['location_data']+'hadamard_IM_slopes_15_no_residual.npy', wfs, M2C=M2C[:,:400])

ind = 100
# show the comparison for a few modes
dm_ghost.coefs = M2C[:,ind]*1e-9

ngs**tel*dm_ghost*wfs

synthetic_imat = display_wfs_signals(wfs, wfs.signal, norma=True,returnOutput=True)
synthetic_imat[np.isinf(synthetic_imat)] = 0

# display the modes 10 to 20
exp_imat = display_wfs_signals(wfs, input_imat[:,ind], norma = True,returnOutput=True)
exp_imat[np.isinf(exp_imat)] = 0

interactive_show(synthetic_imat,exp_imat)