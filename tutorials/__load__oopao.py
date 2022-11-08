# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:54:16 2020

@author: cheritie
"""
import glob
import sys
import numpy as np

def load_oopao():

    # set path properly
    name_repository = 'AO_modules' 
    counter  = 0     
    path = glob.glob(name_repository,recursive = True)
    print('\n')
    print('======================================================================================>      ')
    print('   ✸       *           °                *      *                                            ')
    print('        °   ✸           ▄███▄   ▄███▄  ▄████▄   ▄███▄ * ▄███▄    =>               ▄▄▄▄           ')
    print('  ✸            °       ██*  ██ ██   ██ ██   ██ ██   ██ ██   ██   ====>         ▄█▀▀  ▀▀█▄        ')                                       
    print('   *   °    ✸          ██   ██ ██ ° ██ ██   ██ ██ * ██ ██   ██   ==>          █▀ ▄█▀▀█▄ ▀█       ')
    print('✸    *             °   ██   ██ ██   ██ █████▀  ██▄▄▄██ ██   ██   =========>  █▀ █▀ ▄▄ ▀█ ▀█      ')
    print('           ✸   °       ██ * ██ ██   ██ ██      ██▀▀▀██ ██   ██   ========>   █▄ █▄ ▀▀ ▄█ ▄█      ')
    print(' *    ✸     °          ██   ██ ██   ██ ██  *   ██   ██ ██*  ██   =>           █▄ ▀█▄▄█▀ ▄█       ')
    print('    °        *    ✸     ▀███▀   ▀███▀  ██    ° ██   ██  ▀███▀    ==>           ▀█▄▄  ▄▄█▀        ')
    print('         ✸       *          *         *                                           ▀▀▀▀           ')
    print('======================================================================================>      ')
    print('\n')

    n_max = 40
    if path ==[]:
        while path == [] and counter!= n_max :
            # print('Looking for AO_Modules in the parent repositories...')
            name_repository = '../'+name_repository
            path = glob.glob(name_repository,recursive = True)
            counter += 1 
        nameFolder =path[0] 
        sys.path.append(nameFolder[:-10])
    if counter == n_max:
        raise RuntimeError('The AO modules repository could not be found. Make sure that your script is located in a sub-folder of the OOPAO repository!')
    # check the version of numpy libraries
    try:
        config = np.__config__.blas_mkl_info['libraries'][0]
        if config != 'mkl_rt':
            print('**************************************************************************************************************************************************************')
            print('NUMPY WARNING: OOPAO multi-threading requires to use numpy built with mkl library! If you are using AMD or Apple processors the code could be single threaded!')
            print('**************************************************************************************************************************************************************')
    except:
        print('**************************************************************************************************************************************************************')
        print('NUMPY WARNING: mkl blas not found! Multi-threading may not work as expected.')
        print('**************************************************************************************************************************************************************')
    # print('OOPAO has been properly initialized!')
    # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

