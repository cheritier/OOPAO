# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:21:48 2020

@author: cheritie
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np


class CalibrationVault():
    def __init__(self,
                 D,
                 nTrunc: int  = 0,
                 display: bool = False,
                 print_details: bool = False,
                 invert: bool = True):

        """
        This function allows to calculate the pseudo-inverse of the matrix D in case invert == True.
        The inversion is computed using the single value decomposition (SVD).

        Usually D is the Interaction matrix of our AO system. Usually not a square matrix, so pseudo-inverse is needed

        :param D: matrix we wish to invert
        :param nTrunc: Number of singular values we wish to truncate
        :param display: if True, we display a plot with the singular values from the SVD
        :param print_details: if True, we print some information during the calculations
        :param invert: if True, we invert D. Otherwise, we do not compute the pseudo-inverse

        Outputs:

        -- self.M is the pseudo-inverse of D without truncation in case invert == True. Otherwise, no inversion is done
        -- self.Mtrunc is the pseudo-inverse of D with truncation.
        -- self.cond is the conditioning number: ratio between the highest singular value and the lowest singular value.

        """

        if print_details:
            print('Computing the SVD...')

        if invert:

            U,s,V=np.linalg.svd(D,full_matrices=False)
            self.s=s
            self.S=np.diag(s)
            self.eigenValues=s
            self.D=U@self.S@V
        
            self.U=np.transpose(U)
            self.V=V
            nEigenValues=len(s)-nTrunc
            
            self.iS=np.diag(1/self.eigenValues)        
            self.M=np.transpose(self.V)@self.iS@self.U
            
            self.iStrunc=np.diag(1/self.eigenValues[:nEigenValues])  
            self.Vtrunc=self.V[:nEigenValues,:]
            self.Utrunc=self.U[:nEigenValues,:]
            
            self.VtruncT=np.transpose(self.Vtrunc)
            self.UtruncT=np.transpose(self.Utrunc)
            
            self.Mtrunc=self.VtruncT@self.iStrunc@self.Utrunc
            self.Dtrunc=self.UtruncT@np.diag(self.eigenValues[:nEigenValues])@self.Vtrunc
            
            self.cond=self.eigenValues[0]/self.eigenValues[(-nTrunc-1)]

            if print_details:
                print('Done! The conditionning number is ' + str(self.cond))

            if display:
                
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(self.D)
                plt.colorbar()
                plt.title('Cond: '+str(self.cond))
                
                plt.subplot(1,2,2)
                plt.loglog(self.eigenValues)
                plt.plot([0,nEigenValues],[self.eigenValues[nEigenValues-1],self.eigenValues[nEigenValues-1]])
        else:                
            self.D=D

    @property
    def nTrunc(self):
        return self._nTrunc
    
    @nTrunc.setter
    def nTrunc(self,val):
        self._nTrunc=val
        
        nEigenValues=len(self.s)-self._nTrunc
        
        self.iStrunc=np.diag(1/self.eigenValues[:nEigenValues])
        
        self.Vtrunc=self.V[:nEigenValues,:]
        self.Utrunc=self.U[:nEigenValues,:]
        
        self.VtruncT=np.transpose(self.Vtrunc)
        self.UtruncT=np.transpose(self.Utrunc)
        
        self.Mtrunc=self.VtruncT@self.iStrunc@self.Utrunc

        self.Dtrunc=self.UtruncT@np.diag(self.eigenValues[:nEigenValues])@self.Vtrunc
        
        self.cond=self.eigenValues[0]/self.eigenValues[(-self.nTrunc-1)]
        
        plt.close(3456789)
        plt.figure(3456789)
        plt.subplot(1,2,1)
        plt.imshow(self.D.T)
        plt.colorbar()
        plt.title('Cond: '+str(self.cond))
        
        plt.subplot(1,2,2)
        plt.loglog(self.eigenValues)
        plt.plot([0,nEigenValues],[self.eigenValues[nEigenValues-1],self.eigenValues[nEigenValues-1]])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
    def show(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        print(self.tag+':')
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                if not(a[0].startswith('_')):
                    tmp=str(type(a[1]))
                    if not np.shape(a[1]):
                        print('          '+str(a[0])+' -- '+tmp[8:-2])
                    else:
                        if np.ndim(a[1])>1:
                            print('          '+str(a[0])+' -- '+tmp[8:-2]+' : '+str(np.shape(a[1])))    
                        else:
                            print('          '+str(a[0])+' -- '+tmp[8:-2]+' : '+str(a[1]))    
        
        
        