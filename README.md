# OOPAO
Object Oriented Python Adaptive Optics (OOPAO) is a project under development to propose a python-based tool to perform end-to-end AO simulations.
This code is inspired from the OOMAO architecture: https://github.com/cmcorreia/LAM-Public developped by C. Correia and R. Conan (https://doi.org/10.1117/12.2054470). 
The project was initially intended for personal use in the frame of an ESO project. It is now open to any interested user. 

# FUNCTIONALITIES

	_ Atmosphere: 		Multi-layers with infinitely and non-stationary phase screens, conditions can be updated on the fly if required
	_ Telescope: 		Default circular pupil or user defined, with/without spiders
	_ Deformable Mirror:	Gaussian Influence Functions (default) or user defined, cartesian coordinates (default) or user defined
	_ WFS: 			Pyramid, SH-WFS (diffractive and geometric)
	_ Source: 		NGS or LGS
	_ Control Basis: 	KL modal basis, Zernike Polynomials

# LICENSE
This project is licensed under the terms of the MIT license.

# MODULES REQUIRED
The code is written for Python 3 (version 3.8.8) and requires the following modules

    joblib 1.01         => paralleling computing
    scikit-image 0.18.3 => 2D interpolations
    numexpr 2.7.3       => memory optimized simple operations
    astropy 4.2.1       => handling of fits files
    pyFFTW  0.12.0      => optimization of the FFT  
    mpmath 1.2.1        => arithmetic with arbitrary precision
    jsonpickle 1.4.1    => json files encoding
    aotools 		=> zernike modes and functionalities for atmosphere computation
    numba 0.53.1        => required in aotools

If GPU computation is available:
    cupy-cuda114  9.5.0 => GPU computation of the PWFS code (Not required)

# CODE OPTIMIZATION

OOPAO multi-threading is based on the use of the numpy package built with the mkl library, make sure that the proper numpy package is installed to make sure that the operations are multi-threaded. 
To do this you can use the numpy.show_config() function in your python session: 

import numpy
numpy.show_config()

If the wrong version is installed, the __load__oopao.py function will raise a warning.

    
# CONTRIBUTORS
C.T. Heritier, C. Vérinaud

# AKNOWLEDGEMENTS
This tool has been developped during the Engineering & Research Technology Fellowship of C. Héritier funded by ESO. 
Some functionalities of the code make use of the aotools package developped by M. J. Townson et al (2019). See https://doi.org/10.1364/OE.27.031316.