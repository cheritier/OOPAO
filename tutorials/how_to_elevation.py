# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
plt.ion()

# %%-----------------------     INITIALIZATION    ----------------------------------
tel = Telescope(resolution=128, diameter=0.6, samplingTime=1/1000, centralObstruction=0)

ngs = Source(optBand     = 'V', magnitude = 1, coordinates = [0,0])

ngs ** tel

atm = Atmosphere(telescope=tel,
                 r0=0.06,
                 L0=20,
                 fractionalR0=[0.45, 0.2, 0.35],
                 windSpeed=[10, 12, 11],
                 windDirection=[0, 72, 288],
                 altitude=[0, 1000, 2000])

atm.initializeAtmosphere(tel)

# %%-----------------------     STEP 1: BASELINE AT ZENITH (90°)    ----------------------------------
print("\n" + "="*50)
print("--- STEP 1: INITIAL STATE (ZENITH) ---")
print("="*50)
print(f"Current Elevation : {atm.elevation}°")
print(f"Current r0        : {atm.r0:.4f} m")
print(f"Zenith r0 memory  : {atm.r0_zenith:.4f} m")
print(f"Current Altitudes : {atm.altitude} m")
print(f"Zenith Alt memory : {atm.altitude_zenith} m")

# %%-----------------------     STEP 2: CHANGING ELEVATION    ----------------------------------
atm.elevation = 45

print("\n" + "="*50)
print("--- STEP 2: CHANGING ELEVATION TO 45° ---")
print("="*50)
print("Note: r0 decreases and altitudes increase based on Airmass.")
print("The zenith reference variables remain unchanged in memory.\n")
print(f"Current Elevation : {atm.elevation}°")
print(f"Current r0        : {atm.r0:.4f} m (Scaled)")
print(f"Zenith r0 memory  : {atm.r0_zenith:.4f} m (Unchanged)")
print(f"Current Altitudes : {[round(alt, 1) for alt in atm.altitude]} m (Scaled)")
print(f"Zenith Alt memory : {atm.altitude_zenith} m (Unchanged)")

# %%-----------------------     STEP 3: OVERRIDING R0 AT CURRENT ELEVATION    ----------------------------------
atm.r0 = 0.1

print("\n" + "="*50)
print("--- STEP 3: OVERRIDING R0 AT 45° ---")
print("="*50)
print("Note: Modifying r0 at 45° updates the global atmospheric conditions.")
print("The zenith reference r0 is recalculated backwards.\n")
print(f"Current Elevation : {atm.elevation}°")
print(f"Current r0        : {atm.r0:.4f} m (Forced value)")
print(f"Zenith r0 memory  : {atm.r0_zenith:.4f} m (Globally Updated)")
print("="*50 + "\n")

# %%-----------------------     VISUALIZATION: ELEVATION SWEEP    ----------------------------------
atm.elevation = 90
atm.r0 = 0.06

elevations_sim = np.array([15, 30, 45, 60, 75, 90])
elevations_theo = np.linspace(15, 90, 100)

simulated_r0 = []
simulated_altitudes = []

base_r0 = 0.06
base_altitudes = np.array([0, 1000, 2000])

for elev in elevations_sim:
    atm.elevation = elev
    simulated_r0.append(atm.r0)
    simulated_altitudes.append(atm.altitude)

simulated_r0 = np.array(simulated_r0)
simulated_altitudes = np.array(simulated_altitudes).T

airmass_theo = 1.0 / np.sin(np.deg2rad(elevations_theo))
theoretical_r0 = base_r0 * (airmass_theo)**(-3.0/5.0)
theoretical_altitudes = np.outer(base_altitudes, airmass_theo)

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12})
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

axs[0].plot(elevations_theo, theoretical_r0 * 100, linestyle='-', color='gray', label='Theory')
axs[0].plot(elevations_sim, simulated_r0 * 100, marker='o', linestyle='', color='black', label='OOPAO')
axs[0].set_xlabel('Elevation Angle [Degrees]')
axs[0].set_ylabel('Effective $r_0$ [cm]')
axs[0].set_title('$r_0$ vs Elevation')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(frameon=True, edgecolor='black')

for i in range(simulated_altitudes.shape[0]):
    axs[1].plot(elevations_theo, theoretical_altitudes[i] / 1000, linestyle='-', color='gray')
    axs[1].plot(elevations_sim, simulated_altitudes[i] / 1000, marker='o', linestyle='', color='black')

axs[1].plot([], [], linestyle='-', color='gray', label='Theory')
axs[1].plot([], [], marker='o', linestyle='', color='black', label='OOPAO')

axs[1].set_xlabel('Elevation Angle [Degrees]')
axs[1].set_ylabel('Effective Altitude [km]')
axs[1].set_title('Layer Altitudes vs Elevation')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(frameon=True, edgecolor='black')

plt.tight_layout()
plt.show()