# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:37:28 2025

@author: cheritier
"""

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
    "Objective of the tutorial:\n"
    "Compute a synthetic interaction matrix that you can use on the real system\n",
    
    "Load the papytwin objects.\nExplore the different objects and provide an illustration for their main properties.\n"
    "Generate a synthetic interaction matrix and compare it to the experimental one.\n"
    "What parameters of the model would you need to adapt from this first look?\n",
    "Try to tune the parameters of the model to match the real system.\n"
    " (i) Write a simple test to test the quality of the interaction matrix you computed.\n"
    " (ii) Try to close the loop of the papytwin making use of the experimental interaction matrix"]
code_templates = [" # You will find some hints to answer the questions in this frame.\n"
    "# For instance, make sure that the following code is put at the beginning of your notebook:\n"
    "import sys\n"
    "xs = sys.path\n"
    "matching = [s for s in xs if 'OOPAO' in s]\n"
    "sys.path.append(matching[0]+'/tutorials/OAO_School/')\n"
    "loc = matching[0]+'/tutorials/PAPYRUS/'\n"
    "Some data are available here: 'https://drive.google.com/drive/folders/1OPrhTyuZHtHknvQ9StGYF9wPvsNOwtFk'\n"
    " - 'IMFull.fits' is the interaction matrix of the system\n"
    " - 'M2C.fits' is the mode-to-command matrix of PAPYRUS\n",
    "Check the 'papytwin_advanced.py' \n"
    "This script will be useful to understand how to:\n"
    " - Tune the position of the pyramid pupils\n"
    " - Tune the DM rotation\n"
    " - Tune the DM/WFS fine registration\n",
    "Check the 'papytwin_advanced.py' \n"
    "This script will be useful to understand how to:\n"
    " - Tune the position of the pyramid pupils\n"
    " - Tune the DM rotation\n"
    " - Tune the DM/WFS fine registration\n"
    
    ]

oopao_templates = ["#You will find some generic OOPAO help in this frame\n"
"tel.OPD  # Access the telescope Optical Path Difference\n"
"src.phase  # Access the source phase at the source wavelength\n"
"tel.src  # Access the telescope source object\n"                   
"tel.resetOPD() # Reset the light propagation\n"
"ngs*tel*dm*wfs  # Propagate the light through the system\n"
"dm.coefs=M2C[:,10]  #apply the mode 10 on the dm\n"
"wfs.modulation=val # update the modulation radius and re-compute the reference signal of the Pyramid\n"
"tel+atm # couple the atmosphere and the telescope to go through the turbulence\n"
"tel-atm # separate the atmosphere and the telescope to go in diffraction-limit case\n"
"atm.update() # update the atmospheric phase-screen by 1 time-step".format(i+1) for i in range(len(questions))]



question_images = [
    None, None, None, None,None,
    None,  # Example image for question 5
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
