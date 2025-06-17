# %%

import matplotlib.pyplot as plt
import numpy as np
import OOPAO

import tomoAO

ao_mode = "MLAO"

config_dir = "/home/joaomonteiro/Desktop/OOPAO_ast/tutorials/"
config_file = "config.ini"

# Loading the config
config_vars = tomoAO.IO.load_from_ini(config_file, ao_mode=ao_mode,config_dir=config_dir)


from OOPAO.Source import Source
from OOPAO.Asterism import Asterism

optBand = config_vars["lgs_opticalBand"]
magnitude = config_vars["lgs_magnitude"]
lgs_zenith = config_vars["lgs_zenith"]
lgs_azimuth = config_vars["lgs_azimuth"]
lgs_altitude = config_vars["lgs_altitude"]

n_lgs = config_vars["n_lgs"]

lgsAst = Asterism([Source(optBand=optBand,
              magnitude=magnitude,
              coordinates=[lgs_zenith[kLgs], lgs_azimuth[kLgs]],
            altitude=lgs_altitude)
          for kLgs in range(n_lgs)])



ngs = Source(magnitude = config_vars["lgs_magnitude"],
             optBand   = config_vars["lgs_opticalBand"],
             altitude= config_vars["lgs_altitude"])



from OOPAO.Telescope import Telescope

sensing_wavelength = lgsAst.src[0].wavelength      # sensing wavelength of the WFS, read from the ngs object
n_subaperture      = config_vars["nSubaperture"]                  # number of subaperture across the diameter
diameter           = config_vars["diameter"]                   # diameter of the support of the phase screens in [m]
resolution         = config_vars["resolution"]     # resolution of the phase screens in pixels
# pixel_size         = diameter/resolution # size of the pixels in [m]
obs_ratio          = config_vars["centralObstruction"]                 # central obstruction in fraction of the telescope diameter
sampling_time      = config_vars["samplingTime"]             # sampling time of the AO loop in [s]
fieldOfViewInArcsec = config_vars["fieldOfViewInArcsec"]




# initialize the telescope object
tel_ast = Telescope(diameter          = diameter,
               resolution         = resolution,
               centralObstruction = obs_ratio,
               samplingTime       = sampling_time,
               fov                = fieldOfViewInArcsec)




sensing_wavelength = ngs.wavelength      # sensing wavelength of the WFS, read from the ngs object
n_subaperture      = config_vars["nSubaperture"]                  # number of subaperture across the diameter
diameter           = config_vars["diameter"]                   # diameter of the support of the phase screens in [m]
resolution         = config_vars["resolution"]     # resolution of the phase screens in pixels
# pixel_size         = diameter/resolution # size of the pixels in [m]
obs_ratio          = config_vars["centralObstruction"]                 # central obstruction in fraction of the telescope diameter
sampling_time      = config_vars["samplingTime"]             # sampling time of the AO loop in [s]
fieldOfViewInArcsec = config_vars["fieldOfViewInArcsec"]


# initialize the telescope object
tel_single = Telescope(diameter          = diameter,
               resolution         = resolution,
               centralObstruction = obs_ratio,
               samplingTime       = sampling_time,
               fov                = fieldOfViewInArcsec)






plt.imshow(tel_ast.pupil^tel_single.pupil)
plt.colorbar()
plt.show()


lgsAst**tel_ast
ngs**tel_single

from OOPAO.Atmosphere import Atmosphere

r0 = config_vars["r0"]
L0 = config_vars["L0"]

fractionnalR0 = config_vars["fractionnalR0"]
windSpeed = config_vars["windSpeed"]
windDirection = config_vars["windDirection"]
altitude = config_vars["altitude"]



atm_ast = Atmosphere(telescope      = tel_ast,
                 r0             = r0,
                 L0             = L0,
                 fractionalR0   = fractionnalR0,
                 altitude       = altitude,
                 windDirection  = windDirection,
                 windSpeed      = windSpeed)





atm_single = Atmosphere(telescope      = tel_single,
                 r0             = r0,
                 L0             = L0,
                 fractionalR0   = fractionnalR0,
                 altitude       = altitude,
                 windDirection  = windDirection,
                 windSpeed      = windSpeed)





# %%

atm_ast.initializeAtmosphere(telescope=tel_ast)
atm_single.initializeAtmosphere(telescope=tel_single)

atm_ast.display_atm_layers()
plt.show()
atm_single.display_atm_layers()
plt.show()

tel_ast+atm_ast
tel_single+atm_single

# %%


print(f"ngs OPD mean: {np.mean(ngs.OPD)}, lgsAst OPD mean: {np.mean(lgsAst.OPD[0])}")

fig, axes = plt.subplots(1, 3, figsize=(20, 10))

im = axes[0].imshow(ngs.OPD)
axes[0].axis('off')
axes[0].set_title(f'ngs OPD')
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

im = axes[1].imshow(lgsAst.OPD[0])
axes[1].axis('off')
axes[1].set_title(f'lgsAst OPD')
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

im = axes[2].imshow(ngs.OPD - lgsAst.OPD[0])
axes[2].axis('off')
axes[2].set_title(f'OPD Diff')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.show()






