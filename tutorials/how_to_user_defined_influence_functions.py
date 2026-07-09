# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:32:34 2026

@author: cheritier
"""

import matplotlib.pyplot as plt
plt.close('all')
from OOPAO.InfluenceFunctions import InfluenceFunctions
from OOPAO.MisRegistration import MisRegistration
import numpy as np 

mis_registration = MisRegistration()
# mis_registration.rotationAngle = 30

specific_parameters =dict()
specific_parameters['n_segments'] = 6
specific_parameters['new_arrangement'] = False
specific_parameters['parallel'] = False
specific_parameters['n_jobs'] = 20


name_system = 'ELT_M4'
loc = 'C:/Users/cheritier/Documents/CREATE_M4/'
diameter = 40
resolution = 300
IF = InfluenceFunctions(name_system=name_system,
                        diameter=diameter,
                        resolution=resolution,
                        specific_parameters = specific_parameters,
                        loc = loc,
                        mis_registration=mis_registration)
plt.figure(),plt.imshow(np.sum(IF.influence_function_2D**3,axis=0))



name_system = 'RAMA_DM97'
loc = 'C:/Users/cheritier/Documents/RAMA5/'
diameter = 0.5
resolution = 100
IF = InfluenceFunctions(name_system=name_system,
                        diameter=diameter,
                        resolution=resolution,
                        specific_parameters = None,
                        loc = loc,
                        mis_registration=mis_registration)
plt.figure(),plt.imshow(np.sum(IF.influence_function_2D**3,axis=0))


name_system = 'LBT_ASM'
loc = 'C:/Users/cheritier/Documents/LBT/lbt_data_model/new_data_from_LBT/SX/KL_v20454/'
diameter = 8.5
resolution = 180
IF = InfluenceFunctions(name_system=name_system,
                        diameter=diameter,
                        resolution=resolution,
                        specific_parameters = None,
                        loc = loc,
                        mis_registration=mis_registration)
plt.figure(),plt.imshow(np.sum(IF.influence_function_2D**3,axis=0))


name_system = 'LBT_ASM'
loc = 'C:/Users/cheritier/Documents/LBT/lbt_data_model/new_data_from_LBT/DX/KL_v29/44'
diameter = 8.5
resolution = 400
IF = InfluenceFunctions(name_system=name_system,
                        diameter=diameter,
                        resolution=resolution,
                        specific_parameters = None,
                        loc = loc,
                        mis_registration=mis_registration)
plt.figure(),plt.imshow(np.sum(IF.influence_function_2D**3,axis=0))



diameter = 6.9
resolution = 100
name_system = 'GHOST_DM492'
loc = 'C:/Users/cheritier/Documents/oopao_private/ghost/ghost_simulation14/'
IF = InfluenceFunctions(name_system=name_system,
                        diameter=diameter,
                        resolution=resolution,
                        specific_parameters = None,
                        loc = loc,
                        mis_registration=mis_registration)
plt.figure(),plt.imshow(np.sum(IF.influence_function_2D**3,axis=0))


diameter = 1.52
resolution = 100
name_system = 'PAPYRUS_DM241'
loc = ''
IF = InfluenceFunctions(name_system=name_system,
                        diameter=diameter,
                        resolution=resolution,
                        specific_parameters = None,
                        loc = loc,
                        mis_registration=mis_registration)
plt.figure(),plt.imshow(np.sum(IF.influence_function_2D**3,axis=0))


diameter = 1.82
resolution = 500
name_system = 'EKARUS_DM468'
loc = 'C:/Users/cheritier/Documents/oopao_private/ekarus/EKARUS_DM468/'
IF = InfluenceFunctions(name_system=name_system,
                        diameter=diameter,
                        resolution=resolution,
                        specific_parameters = None,
                        loc = loc,
                        mis_registration=mis_registration)
plt.figure(),plt.imshow(np.sum(IF.influence_function_2D**3,axis=0))
