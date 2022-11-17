# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:49:50 2020

@author: cheritie
"""

import json
import random
import time

import jsonpickle
import numpy as np
import scipy as sp
from numpy.random import RandomState

from .tools.tools import bsxfunMinus, createFolder


def gamma(x):
    return np.math.gamma(x)

def variance(atm):
    # compute the phase variance from an atmosphere object
    L0r0Ratio   = (atm.L0/atm.r0)**(5./3)

    out         = (  (24.*gamma(6./5))**(5./6)    )*\
                  (  gamma(11./6)*gamma(5./6)     )/\
                  (  2*np.pi**(8./3)              )*\
                  (  L0r0Ratio                    )      
          
    out = np.sum(atm.cn2)*out
    return out
    

def covariance(rho,atm):
    # compute the phase covariance from the baseline rho and an atmosphere object
    L0r0Ratio   = (atm.L0/atm.r0)**(5./3)

    cst         = (  (24.*gamma(6./5)/5)**(5./6)   )*\
                  (  gamma(11./6)/((2.**(5./6))    )*\
                  (  np.pi**(8./3))                )*\
                  (  L0r0Ratio                     )
                    
    out         = (  np.ones(rho.shape)            )*\
                  (  (24.*gamma(6./5)/5)**(5./6)   )*\
                  (  gamma(11./6)*gamma(5./6)      )/\
                  (  2*np.pi**(8./3)               )*\
                  (  L0r0Ratio                     ) 
    
    index = np.where(rho!=0)

    u = 2*np.pi*rho[index]/atm.L0
    out[index] = cst*u**(5./6)*sp.special.kv(5./6,u)
    return out

    

def spectrum(f,atm):
    # compute the phase power spectral density from the spatial frequency f and an atmsphere object
    out         = (  (24.*gamma(6./5)/5)**(5./6)   )*\
                  (  gamma(11./6)**2               )/\
                  (  2*np.pi**(11./3)              )*\
                  (  atm.r0**(-5/3)                )
    
    out = out * (f**2 + 1./atm.L0**2)**(-11/6)
    return out


def makeCovarianceMatrix(rho1,rho2,atm):
    
    rho = np.abs(bsxfunMinus(rho1,rho2))

    L0r0ratio   =  (atm.L0/atm.r0_def)**(5./3)
    
    cst = (24.*np.math.gamma(6./5)/5)**(5./6) * (np.math.gamma(11./6)/((2.**(5./6))*np.pi**(8./3)))*L0r0ratio

    out = np.ones(rho.shape)*((24.*np.math.gamma(6./5)/5)**(5./6))*(np.math.gamma(11./6)*np.math.gamma(5./6)/(2*np.pi**(8./3))) * L0r0ratio

    index = np.where(rho!=0)

    u = 2*np.pi*rho[index]/atm.L0

    
    if atm.param is None:
        sp_kv = sp.special.kv(5./6,u)

    else:
        try:
            print('Loading pre-computed data...')            
            name_data       = 'sp_kv_L0_'+str(atm.param['L0'])+'_m_shape_'+str(len(rho1))+'x'+str(len(rho2))+'.json'

            location_data   = atm.param['pathInput'] + atm.param['name'] + '/sk_v/'

            try:
                with open(location_data+name_data ) as f:
                    C = json.load(f)
                data_loaded = jsonpickle.decode(C)               
            except:
                createFolder(location_data)
                with open(location_data+name_data ) as f:
                    C = json.load(f)
                data_loaded = jsonpickle.decode(C)  
                
            sp_kv = data_loaded['sp_kv']
           
                        
        except: 
            print('Something went wrong.. re-computing sp_kv ...')
            name_data       = 'sp_kv_L0_'+str(atm.param['L0'])+'_m_shape_'+str(len(rho1))+'x'+str(len(rho2))+'.json'
            location_data   = atm.param['pathInput'] + atm.param['name'] + '/sk_v/'
            
            sp_kv = sp.special.kv(5./6,u)
            
            print('saving for future...')
            data = dict()
            data['sp_kv'] = sp_kv
            data_encoded  = jsonpickle.encode(data)

            try: 
                with open(location_data+name_data, 'w') as f:
                    json.dump(data_encoded, f)
            except:
                createFolder(location_data)
                with open(location_data+name_data, 'w') as f:
                    json.dump(data_encoded, f)
                
            
    
    out[index] = cst*u**(5./6)*sp_kv


    return out

#def fourierPhaseScreen(atm,D,resolution):
#    
#    N = int(8* resolution)
#    
#    L = (N-1)*D/(resolution-1)
#    
#    
#    fx = np.fft.fftshift(np.fft.fftfreq(N))
#    fy = np.fft.fftshift(np.fft.fftfreq(N))
#    
#    fx,fy = np.meshgrid(fx,fy)
#    f_r,f_phi = cart2pol(fx,fx)
#    
#    f_r = np.fft.fftshift(f_r*(N-1)/L/2)
#    f_r+=f_r.T
#    
#    psdRoot = np.sqrt(spectrum(f_r,atm))   
#    index = np.where(f_r==0)
#    psdRoot[index]=0
#    
#    # low order correc
#    
#    fourierSampling = 1./L
#    
#    u = np.arange(resolution+1)
#    WNF = np.fft.fft2(np.random.randn(N,N))/N
#    
#    outMap = psdRoot*WNF   
#    outMap = np.real(np.fft.ifft2(outMap))*fourierSampling*N**2
#    
#    out = outMap[u[0]:u[-1],u[0]:u[-1]]
#
#    
#    return out

def ift2(G, delta_f):
    """
     ------------ Function adapted from aotools ----------------------

    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """
        
    N = G.shape[0]

#    g = np.fft.fftshift( np.fft.fft2( np.fft.fftshift(G) ) ) * (N * delta_f)**2
    g = np.fft.fftshift( np.fft.fft2( np.fft.fftshift(G) ) ) 

    return g


def ft_phase_screen(atm, N, delta, l0 =1e-10,seed=None):
    '''
        ------------ Function adapted from aotools ----------------------

    Creates a random phase screen with Von Karmen statistics.
    (Schmidt 2010)
    
    Parameters:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres

    Returns:
        ndarray: np array representing phase screen
    '''
    delta = float(delta)
    r0 = float(atm.r0)
    L0 = float(atm.L0)
    R = random.SystemRandom(time.time())
    if seed is None:
        seed = int(R.random()*100000)
    randomState   = RandomState(seed)

    del_f = 1./(N*delta)

    fx = np.arange(-N/2.,N/2.) * del_f

    (fx,fy) = np.meshgrid(fx,fx)
    f = np.sqrt(fx**2 + fy**2)

    fm = 5.92/l0/(2*np.pi)
    f0 = 1./L0

    PSD_phi  = (0.023*r0**(-5./3.) * np.exp(-1*((f/fm)**2)) /
                ( ( (f**2) + (f0**2) )**(11./6) ) )

    PSD_phi[int(N/2), int(N/2)] = 0

    cn = ( (randomState.normal(size=(N,N)) + 1j* randomState.normal(size=(N,N)) )
                * np.sqrt(PSD_phi)*del_f )

    phs = ift2(cn,1).real

    return phs







def ft_sh_phase_screen(atm, N, delta, l0= 1e-10, seed=None):

    """
    ------------ Function adapted from aotools ----------------------
    
    Creates a random phase screen with Von Karmen statistics with added
    sub-harmonics to augment tip-tilt modes.
    (Schmidt 2010)
    
    Args:
        r0 (float): r0 parameter of scrn in metres
        N (int): Size of phase scrn in pxls
        delta (float): size in Metres of each pxl
        L0 (float): Size of outer-scale in metres
        l0 (float): inner scale in metres

    Returns:
        ndarray: np array representing phase screen
    """
    delta = float(delta)
    r0 = float(atm.r0)
    L0 = float(atm.L0)
    R = random.SystemRandom(time.time())
    if seed is None:
        seed = int(R.random()*100000)
    randomState   = RandomState(seed)

    D = N*delta
    # high-frequency screen from FFT method
    phs_hi = ft_phase_screen(atm, N, delta, seed=seed)

    # spatial grid [m]
    coords = np.arange(-N/2,N/2)*delta
    x, y = np.meshgrid(coords,coords)

    # initialize low-freq screen
    phs_lo = np.zeros(phs_hi.shape)

    # loop over frequency grids with spacing 1/(3^p*L)
    for p in range(1,4):
        # setup the PSD
        del_f = 1 / (3**p*D) #frequency grid spacing [1/m]
        fx = np.arange(-1,2) * del_f

        # frequency grid [1/m]
        fx, fy = np.meshgrid(fx,fx)
        f = np.sqrt(fx**2 +  fy**2) # polar grid

        fm = 5.92/l0/(2*np.pi) # inner scale frequency [1/m]
        f0 = 1./L0

        # outer scale frequency [1/m]
        # modified von Karman atmospheric phase PSD
        PSD_phi = (0.023*r0**(-5./3)
                    * np.exp(-1*(f/fm)**2) / ((f**2 + f0**2)**(11./6)) )
        PSD_phi[1,1] = 0

        # random draws of Fourier coefficients
        cn = ( (randomState.normal(size=(3,3))
            + 1j*randomState.normal(size=(3,3)) )
                        * np.sqrt(PSD_phi)*del_f )
        SH = np.zeros((N,N),dtype="complex")
        # loop over frequencies on this grid
        for i in range(0,2):
            for j in range(0,2):

                SH += cn[i,j] * np.exp(1j*2*np.pi*(fx[i,j]*x+fy[i,j]*y))

        phs_lo = phs_lo + SH
        # accumulate subharmonics

    phs_lo = phs_lo.real - phs_lo.real.mean()

    phs = phs_lo+phs_hi

    return phs