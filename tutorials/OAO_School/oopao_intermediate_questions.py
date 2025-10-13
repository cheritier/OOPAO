import streamlit as st
from PIL import Image
import os
import re

def format_question_text(text):
    """Format question text with bold title and indented grey sub-items."""
    # Split title and body
    lines = text.strip().split('\n', 1)
    title = lines[0].strip()
    body = lines[1] if len(lines) > 1 else ""

    # Convert (i), (ii), etc. to HTML-styled bullets
    body = re.sub(
        r'\s*\((?=[ivx]+\))',
        r'<br>&nbsp;&nbsp;&nbsp;<span style="color:#cccccc;">• (',
        body
    )
    # Close the <span> after each item (basic cleanup)
    body = body.replace(")", ")</span>", 1)  # Close after first bullet properly

    formatted = f"**{title}**<br><br>{body.strip()}"
    return formatted
# -------------------------------
# QUESTIONS, CODE TEMPLATES, IMAGES
# -------------------------------
questions = [
    "Objective of the tutorial: Get a realistic estimate of the performance of PAPYRUS\n",
    
    "Load the papytwin objects.\nExplore the different objects and provide an illustration for their main properties:\n"
    "    (i) Telescope pupil\n    (ii) WFS Detector\n    (iii) DM Coordinates with respect to the pupil\n"
    "    (iv) Modal basis applied on the DM\n    (v) Diffraction Limited PSF on calibration source\n"
    "    (vi) Diffraction limit PSF on-sky\n    (vii) How is the modal basis normalized?",
    
    "Compute an Interaction Matrix of the system with wfs.modulation = 0.\n"
    "    (i) What amplitude should you use for the calibration?\n"
    "    (ii) What is the conditioning number of the interaction matrix?\n"
    "    (iii) What does this number represent?\n"
    "    (iv) What test can you do to test the quality of the interaction matrix?\n"
    "    (v) What test can you do to identify the maximum amplitude that can be properly reconstructed?",
    
    "Recompute a new interaction matrix with wfs.modulation = 3\n"
    "    (i) What does this parameter represent?\n"
    "    (ii) How do you expect the previous test to change?\n"
    "    (iii) Considering that the seeing at OHP is around 2.5 arcsec, what modulation would you recommend to close the loop?",
    
    "Once your calibration is satisfactory:\n"
    "    (i) Create a closed loop with the papytwin using 2.5″ seeing and a wind-speed of 5 m/s\n"
    "    (ii) What Strehl ratio do you get for the ngs in the visible and for the src in the infrared?",
    
    "In reality the SR on the infrared path on Papyrus is lower than what you found."
    " On the right, you can find a simulated image of the infrared PSF of the bench.\n"
    "    (i) What could explain the difference in the on-sky PSF and the one obtained in simulation\n"
    "    (ii) Try to reproduce the expected PSF playing with the DM.\n"
    
    "This NCPA is annoying...We would like to get rid of it:"
    "    (i) How could we compensate for the aberration in PAPYRUS?"
    "    (ii) Do you know why we see this aberration in the Infrared Path of PAPYRUS?", 

    "PAPYRUS Performance analysis:\n "
    "    (i) What parameter is driving the AO performance of PAPYRUS? Give a plot that illustrates this answer\n"
    "    (ii) How could we improve the AO performance of PAPYRUS with the current hardware? Give a plot that illustrates this answer\n"    
    "    (iii) How could we improve the AO performance of PAPYRUS adding/replacing some hardware? Give a plot that illustrates this answer\n"    
    "    (iv) Anisoplanatism: How far would you need to put the source to reduce the SR by a factor 2 in the visible channel? Give a plot that illustrates this answer"]

code_templates = [" # You will find some hints to answer the questions in this frame.\n"
    "# For instance, make sure that the following code is put at the beginning of your notebook:\n"
    "import sys\n"
    "xs = sys.path\n"
    "matching = [s for s in xs if 'OOPAO' in s]\n"
    "sys.path.append(matching[0]+'/tutorials/ORP_AO_School/')\n"
    "loc = matching[0]+'/tutorials/PAPYRUS/'",
    
    "# Here is a summary of how to access the papytwin:\n"
    "from Papyrus import Papyrus\nPapytwin = Papyrus()\ntel = Papytwin.tel\nngs = Papytwin.ngs\nsrc = Papytwin.src\n"
    "dm = Papytwin.dm\nwfs = Papytwin.wfs\natm = Papytwin.atm\nM2C = Papytwin.M2C",
    
    "# A useful function in OOPAO:\n"
    "from OOPAO.InteractionMatrix import InteractionMatrix\ncalib_0 = InteractionMatrix(ngs,tel,dm,wfs,M2C,stroke,nMeasurements=2)\nreconstructor = calib_0.M",
    
    "# A useful function in OOPAO:\n"
    "from OOPAO.InteractionMatrix import InteractionMatrix\ncalib_0 = InteractionMatrix(ngs,tel,dm,wfs,M2C,stroke,nMeasurements=2)\nreconstructor = calib_0.M",
    
    "# here is a pseudo-code to make a close-loop with OOPAO:\n"
    "dm.coefs = 0 \n"
    "tel+atm\n"
    "for i in range(n_loop):\n"
    "   atm.update()\n"
    "   ngs*atm*tel*dm*wfs\n"
    "   dm.coefs -= g * reconstructor@wfs_signal_delayed\n"
    "   wfs_signal_delayed = wfs.signal",
    "# Here is a useful function to apply a user_input_ncpa:\n"
    "from OOPAO.NCPA import NCPA\n"
    "ncpa = NCPA(tel, dm, atm, M2C = M2C_CL, coefficients=[0,0])\n"
    "ncpa.OPD = user_defined_ncpa\n",
    "# You can change the parameters of the atmosphere directly:\n"
    "atm.windspeed = [10] # adapt to the number of layers\n"
    "atm.windDirection = [90] # adapt to the number of layers\n"
    "atm.r0 = 0.15\n"
    "# You can change the magnitude and coordinates of the stars:\n"
    "ngs.magnitude = 0\n"
    "ngs.coordinates = [10,90]  # #[zenith in arcsec, azimuth in deg] requires tel.fov to be different than 0 (see parameter file)\n"
    "'The parameter file of the papytwin is located in /parameter_files/parameterFile_papytwin/'"
]

oopao_templates = ["#You will find some generic OOPAO help in this frame\n"
"tel.OPD  # Access the telescope Optical Path Difference\n"
"src.phase  # Access the source phase at the source wavelength\n"
"tel.src  # Access the telescope source object\n"                   
"tel.resetOPD() # Reset the light propagation\n"
"ngs*tel*dm*wfs  # Propagate the light through the system\n"
"dm.coefs=M2C[:,10]  #apply the mode 10 on the dm\n"
"Papytwin.set_pupil(calibration=False/True) # switch to the calibration or on-sky pupil\n"
"wfs.modulation=val # update the modulation radius and re-compute the reference signal of the Pyramid\n"
"tel+atm # couple the atmosphere and the telescope to go through the turbulence\n"
"tel-atm # separate the atmosphere and the telescope to go in diffraction-limit case\n"
"atm.update() # update the atmospheric phase-screen by 1 time-step".format(i+1) for i in range(len(questions))]



question_images = [
    None, None, None, None,None,
    "PSF_papyrus_ncpa.png",  # Example image for question 5
    None, None, None, None, None
]

# -------------------------------
# STREAMLIT UI SETUP
# -------------------------------
st.set_page_config(page_title="OOPAO / PAPYRUS AO Exercise", layout="wide")

if "q_index" not in st.session_state:
    st.session_state.q_index = 0

# Title
st.title("OOPAO / PAPYRUS Adaptive Optics Exercise")
st.subheader(f"Question {st.session_state.q_index} of {len(questions)-1}")

# Two-column layout
col1, col2 = st.columns([3, 2])

# Question text
# col1.markdown(f"### 📘 Question\n\n{questions[st.session_state.q_index]}")
col1.markdown(format_question_text(questions[st.session_state.q_index]), unsafe_allow_html=True)
# Image (if any)
img_path = question_images[st.session_state.q_index]
if img_path and os.path.exists(img_path):
    image = Image.open(img_path)
    col2.image(image, caption="Related illustration", use_column_width=True)
# Code hint
st.markdown("### Hints")
st.code(code_templates[st.session_state.q_index], language="python")
st.markdown("### OOPAO cheat-list")
st.code(oopao_templates[st.session_state.q_index], language="python")

# Navigation buttons
col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("⬅ Previous"):
        if st.session_state.q_index > 0:
            st.session_state.q_index -= 1
            st.rerun()

with col_next:
    if st.button("Next ➡"):
        if st.session_state.q_index < len(questions) - 1:
            st.session_state.q_index += 1
            st.rerun()
