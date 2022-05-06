# OOPAO
Object Oriented Python Adaptive Optics (OOPAO) is a project under development to propose a python-based tool to perform end-to-end AO simulations. 
It is inspired from the OOMAO architecture: https://github.com/cmcorreia/LAM-Public
The project was initially intended for personal use, i tried to make it user-friendly so it can benefit to other users. 

# LICENSE
This project is licensed under the terms of the MIT license.

# MODULES REQUIRED
The code is written for Python 3 (version 3.8.8) and requires the following modules

    numba 0.53.1        => required in aotools
    joblib 1.01         => paralleling computing
    scikit-image 0.18.3 => 2D interpolations
    numexpr 2.7.3       => memory optimized simple operations
    astropy 4.2.1       => handling of fits files
    pyFFTW  0.12.0      => optimization of the FFT  
    mpmath 1.2.1        => arithmetic with arbitrary precision
    jsonpickle 1.4.1    => json files encoding
    json       0.9.5    => json files
    mkl        2022.1.0 => optimization of the parallized operations
    
# CONTRIBUTORS
C.T. Heritier
