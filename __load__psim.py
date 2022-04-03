# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:54:16 2020

@author: cheritie
"""
import glob
import sys



def load_psim():

    nameTest = 'AO_modules' 
    counter  = 0     
    path = glob.glob(nameTest,recursive = True)
    if path ==[]:
            
        while path == [] and counter!= 10 :
            print('Looking for AO_Modules...')
            nameTest = '../'+nameTest
            
            path = glob.glob(nameTest,recursive = True)
            
            print(path)
            
        nameFolder =path[0] 
#        os.chdir(nameFolder[:-10])

        sys.path.append(nameFolder[:-10])
        counter += 1 
    
    print('AO_Modules found! Loading the main modules:')

