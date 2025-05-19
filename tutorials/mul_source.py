# %%
import matplotlib.pyplot as plt
import numpy as np
import OOPAO



# %%
from OOPAO.Source import Source
from OOPAO.Asterism import Asterism
# to create a natural guide star in I band of magnitude 5


src_1 = Source(optBand = 'I', magnitude = 8, coordinates=[14.14, -135.00])
src_2 = Source(optBand = 'I', magnitude = 8, coordinates=[14.14, -45.00])
src_3 = Source(optBand = 'I', magnitude = 8, coordinates=[14.14, 135.00])
src_4 = Source(optBand = 'I', magnitude = 8, coordinates=[14.14, 45.00])


src_5 = Source(optBand = 'I', magnitude = 8, coordinates=[14.14, -135.00])
# src_5 = Source(optBand = 'I', magnitude = 8, coordinates=[1,1])

lgsAst = Asterism([src_1, src_2, src_3, src_4])




# %%

from OOPAO.Telescope import Telescope

# telescope parameters
sensing_wavelength = lgsAst.src[0].wavelength      # sensing wavelength of the WFS, read from the ngs object
n_subaperture      = 20                  # number of subaperture across the diameter
diameter           = 8                   # diameter of the support of the phase screens in [m]
resolution         = n_subaperture*8     # resolution of the phase screens in pixels
pixel_size         = diameter/n_subaperture # size of the pixels in [m]
obs_ratio          = 0.1                 # central obstruction in fraction of the telescope diameter
sampling_time      = 1/1000              # sampling time of the AO loop in [s]

# initialize the telescope object
tel = Telescope(diameter          = diameter,
               resolution         = resolution,
               centralObstruction = obs_ratio,
               samplingTime       = sampling_time,
               fov = 30)





lgsAst*tel


# %%
tel_single = Telescope(diameter          = diameter,
               resolution         = resolution,
               centralObstruction = obs_ratio,
               samplingTime       = sampling_time,
               fov=30)

src_5*tel_single







# %%

from OOPAO.Atmosphere import Atmosphere

atm = Atmosphere(telescope      = tel,
                 r0             = 0.15,
                 L0             = 25,
                 fractionalR0   = [0.7, 0.3  ],
                 altitude       = [0  , 10000],
                 windDirection  = [0  , 20   ],
                 windSpeed      = [5  , 10   ])

atm_single = Atmosphere(telescope      = tel_single,
                 r0             = 0.15,
                 L0             = 25,
                 fractionalR0   = [0.7, 0.3  ],
                 altitude       = [0  , 10000],
                 windDirection  = [0  , 20   ],
                 windSpeed      = [5  , 10   ])

# %%
atm.initializeAtmosphere(telescope=tel)

tel+atm

atm.display_atm_layers()
plt.show()



# %%

for i in range(100):
    atm.update()

# %%

atm_single.initializeAtmosphere(telescope=tel_single)

tel_single+atm_single

atm_single.display_atm_layers()
plt.show()

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i in range(4):
    axes[i].imshow(lgsAst.OPD[i])
    axes[i].axis('off')
    axes[i].set_title(f'Asterism OPD {i}')
plt.show()

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i in range(4):
    axes[i].imshow(tel.OPD[i])
    axes[i].axis('off')
    axes[i].set_title(f'Telescope OPD {i}')
plt.show()


# %%
print(f"lgsAst:")
lgsAst.print_optical_path()

print(f"\ntel:")
tel.print_optical_path()


# %%
print(f"\nsrc_5:")
src_5.print_optical_path()

print(f"\ntel_single:")
tel_single.print_optical_path()

# %%
plt.imshow(src_5.OPD)
plt.axis('off')
plt.title(f'Source OPD')
plt.show()

# %%
plt.imshow(tel_single.OPD)
plt.axis('off')
plt.title(f'Telescope OPD')
plt.show()

# %%

lgsAst**tel

# %%
print(f"lgsAst:")
lgsAst.print_optical_path()

print(f"\ntel:")
tel.print_optical_path()


# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i in range(4):
    axes[i].imshow(lgsAst.OPD[i])
    axes[i].axis('off')
    axes[i].set_title(f'Asterism OPD {i}')
plt.show()

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

for i in range(4):
    axes[i].imshow(tel.OPD[i])
    axes[i].axis('off')
    axes[i].set_title(f'Telescope OPD {i}')
plt.show()

# %%

src_5**tel_single

# %%
print(f"\nsrc_5:")
src_5.print_optical_path()

print(f"\ntel_single:")
tel_single.print_optical_path()

# %%
plt.imshow(src_5.OPD)
plt.axis('off')
plt.title(f'Source OPD')
plt.show()

# %%
plt.imshow(tel_single.OPD)
plt.axis('off')
plt.title(f'Telescope OPD')
plt.show()




# %%

from OOPAO.DeformableMirror import DeformableMirror
mechanical_coupling = 0.45

dm_fried = DeformableMirror(  telescope    = tel,
                        nSubap       = n_subaperture, # by default n_subaperture+1 actuators are considered (Fried Geometry)
                        mechCoupling = mechanical_coupling)

dm_fried.coefs = np.random.rand(dm_fried.nValidAct)


lgsAst**tel*dm_fried


fig, axes = plt.subplots(3, 5, figsize=(25, 16))

for i in range(4):
    axes[0, i].imshow(lgsAst.OPD_no_pupil[i])
    axes[0, i].axis('off')
    axes[0, i].set_title(f'Source {i} OPD no pupil')

axes[0, 4].imshow(dm_fried.OPD)
axes[0, 4].axis('off')
axes[0, 4].set_title(f'Deformable Mirror OPD')

for i in range(4):
    axes[1, i].imshow(lgsAst.OPD[i])
    axes[1, i].axis('off')
    axes[1, i].set_title(f'Source {i} OPD')

axes[1, 4].imshow(dm_fried.OPD)
axes[1, 4].axis('off')
axes[1, 4].set_title(f'Deformable Mirror OPD')


for i in range(4):
    axes[2, i].imshow(tel.OPD[i])
    axes[2, i].axis('off')
    axes[2, i].set_title(f'Telescope OPD {i}')

axes[2, 4].imshow(dm_fried.OPD)
axes[2, 4].axis('off')
axes[2, 4].set_title(f'Deformable Mirror OPD')

plt.show()



# %%

from OOPAO.DeformableMirror import DeformableMirror
mechanical_coupling = 0.45

dm_fried_single = DeformableMirror(  telescope    = tel_single,
                        nSubap       = n_subaperture, # by default n_subaperture+1 actuators are considered (Fried Geometry)
                        mechCoupling = mechanical_coupling)


dm_fried_single.coefs = np.random.rand(dm_fried_single.nValidAct)

src_5**tel_single*dm_fried_single

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].imshow(src_5.OPD)
axes[0].axis('off')
axes[0].set_title(f'Source OPD')

axes[1].imshow(tel_single.OPD)
axes[1].axis('off')
axes[1].set_title(f'Telescope OPD')

axes[2].imshow(dm_fried_single.OPD)
axes[2].axis('off')
axes[2].set_title(f'Deformable Mirror OPD')

plt.show()

# %%
print(f"lgsAst:")
lgsAst.print_optical_path()

print(f"\nsrc_5:")
src_5.print_optical_path()


# %%
from OOPAO.ShackHartmann import ShackHartmann

# %%

import copy

tel_copy = copy.deepcopy(tel)

shwfs = ShackHartmann(telescope          = tel_copy,
                      src                = lgsAst,
                      nSubap             = n_subaperture,
                      lightRatio         = 1,
                      is_geometric       = False,
                      shannon_sampling   = True,
                      threshold_cog      = 0.1)




shwfs_single = ShackHartmann(telescope          = tel_single,
                             src                = src_5,
                             nSubap             = n_subaperture,
                             lightRatio         = 1,
                             is_geometric       = False,
                             shannon_sampling   = True,
                             threshold_cog      = 0.1)


fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(4):
    im = axes[i].imshow(shwfs.signal_2D[i])
    axes[i].axis('off')
    axes[i].set_title(f'Source {i} Signal')
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

im = axes[4].imshow(shwfs_single.signal_2D)
axes[4].axis('off')
axes[4].set_title(f'Source Single Signal')
fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

plt.show()

# %%
lgsAst ** tel * dm_fried * shwfs

# %%
src_5 ** tel_single * dm_fried_single * shwfs_single

# %%

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(4):
    im = axes[i].imshow(shwfs.signal_2D[i])
    axes[i].axis('off')
    axes[i].set_title(f'Source {i} Signal')
    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

im = axes[4].imshow(shwfs_single.signal_2D)
axes[4].axis('off')
axes[4].set_title(f'Source Single Signal')
fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

plt.show()

