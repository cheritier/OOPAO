# OOPAO / PAPYTWIN — Interactive Exercise Notebook (One-Question Reveal + Code Templates)
# --------------------------------------------------
# This version uses ipywidgets to reveal questions one by one, and includes code templates.

# %%
"""
OOPAO Adaptive Optics — PAPYRUS Exercise
Interactive version: only one question is shown at a time, with example code templates.
"""

# %%
# Setup interactive reveal system
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets

questions = [
    "1️⃣ Load the papytwin objects",
    "2️⃣ Perform a calibration of the system with `wfs.modulation = 0`",
    "3️⃣ What test can you do to test the quality of the interaction matrix?",
    "4️⃣ What test can you do to identify the maximum amplitude that can be properly reconstructed?",
    "5️⃣ Recompute a new interaction matrix with `wfs.modulation = 3`. What does this parameter represent? How do you expect the previous test to change?",
    "6️⃣ Once your calibration is satisfactory, create a closed loop with papytwin using 2.5″ seeing. Does the result look like the expected on-sky result?",
    "7️⃣ What could explain the difference in the on-sky PSF and the one obtained in simulation?",
    "8️⃣ Load the NCPA objects. Try to reproduce the expected level of NCPA playing with the modal coefficient index amplitude [clue: only one mode to apply]. Do you know why we see this NCPA?",
    "9️⃣ Tune the NCPA to reach a target Strehl in the IR for a 2.5″ seeing and a windspeed of 5 m/s.",
    "🔟 What parameter is driving the AO performance of PAPYRUS?",
    "1️⃣1️⃣ How could we improve the AO performance of PAPYRUS?",
    "1️⃣2️⃣ Anisoplanatism: How far would you need to put the source to reduce the SR by a factor 2?"
]

i = 0
out = widgets.Output()

def next_question(_):
    global i
    with out:
        clear_output()
        if i < len(questions):
            display(Markdown(f"### Question {i+1}\n{questions[i]}"))
            # show example code template
            if i == 0:
                display(Markdown("""```python
from oopao import papytwin
pt = papytwin.Papytwin()  # Load the papytwin bench
print(pt)
```"""))
            elif i == 1:
                display(Markdown("""```python
# Calibration with wfs.modulation = 0
pt.wfs.modulation = 0
pt.calibrate_system()
pt.plot_interaction_matrix()
```"""))
            elif i == 2:
                display(Markdown("""```python
# Test quality of interaction matrix (e.g., via SVD condition number)
import numpy as np
U, s, Vt = np.linalg.svd(pt.M2C, full_matrices=False)
print('Condition number:', s.max()/s.min())
```"""))
            elif i == 3:
                display(Markdown("""```python
# Linearity test — increase input amplitude and reconstruct
amplitudes = np.linspace(0, 1e-6, 10)
errors = []
for amp in amplitudes:
    pt.dm.set_phase(amp * pt.modes[:,0])
    recon = pt.wfs.reconstruct()
    errors.append(np.std(recon - amp*pt.modes[:,0]))
plt.plot(amplitudes, errors)
plt.xlabel('Amplitude [rad]')
plt.ylabel('Reconstruction error')
```"""))
            elif i == 4:
                display(Markdown("""```python
# Change modulation and recalibrate
pt.wfs.modulation = 3
pt.calibrate_system()
# The modulation increases dynamic range but reduces sensitivity.
```"""))
            elif i == 5:
                display(Markdown("""```python
# Closed loop simulation under 2.5'' seeing
pt.atmosphere.set_seeing(2.5)
pt.loop_gain = 0.5
pt.run_closed_loop(n_iter=200)
pt.display_psf()
```"""))
            elif i == 6:
                display(Markdown("""```python
# Discuss differences between simulated and on-sky PSF
# Possible factors: unmodeled turbulence layers, calibration error, telescope vibrations.
```"""))
            elif i == 7:
                display(Markdown("""```python
# Load and apply NCPA
pt.load_ncpa('path/to/ncpa/file')
pt.apply_ncpa(mode_index=5, amplitude=0.1)
pt.display_psf()
```"""))
            elif i == 8:
                display(Markdown("""```python
# Tune NCPA to reach target Strehl
from oopao.metrics import compute_strehl
for amp in np.linspace(0, 0.2, 10):
    pt.apply_ncpa(mode_index=5, amplitude=amp)
    sr = compute_strehl(pt.get_psf())
    print(f'Amplitude={amp:.2f}, SR={sr:.3f}')
```"""))
            elif i == 9:
                display(Markdown("""```python
# Explore parameter sensitivity
# Example: number of modes, loop gain, frame rate
```"""))
            elif i == 10:
                display(Markdown("""```python
# Discuss improvements: faster loop, predictive control, better calibration
```"""))
            elif i == 11:
                display(Markdown("""```python
# Anisoplanatism test: vary source position
angles = np.linspace(0, 30, 10)  # arcseconds
strehl = []
for ang in angles:
    pt.src.set_offset(ang)
    pt.run_closed_loop(100)
    strehl.append(pt.compute_strehl())
plt.plot(angles, strehl)
plt.xlabel('Off-axis angle [arcsec]')
plt.ylabel('Strehl ratio')
```"""))
            i += 1
        else:
            display(Markdown("✅ **All questions revealed!**"))

btn = widgets.Button(description="Show next question", button_style="info", icon="arrow-down-circle")
btn.on_click(next_question)

display(btn, out)

print("Interactive question reveal setup ready. Run the top cell and click the button to begin.")
