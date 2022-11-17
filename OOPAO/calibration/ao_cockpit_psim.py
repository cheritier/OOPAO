# -*- coding: utf-8 -*-
"""
Created on Fri Nov  11 10:33:10 2020

@author: cverinau
"""

import math
import multiprocessing as mp
import os
import pickle
import time
from math import factorial

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import psutil
import pyfftw


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)




def run(file2exec):
    exec(open(file2exec).read())

    
def save(filename,variables_list):

    f=open(filename,'wb')
    pickle.dump(variables_list,f)
    f.close()
    
def load(filename):
    f=open(filename,'rb')
    variables_list=pickle.load(f)
    return variables_list


def aid(x):
    #This fucntion returns the memory block adress of an array
    return x.__array_interface__['data'][0]



def vectorify(array,idxpup):
    if array.ndim==3:
        array_=array[idxpup[0],idxpup[1],0:array.shape[2]]
    if array.ndim==2:
        array_=array[idxpup[0],idxpup[1]]
    return array_

def vectorifyb(array,idxpup):
    if array.ndim==3:
        nact=array.shape[0]
        array_=np.zeros([len(idxpup[0]),array.shape[0]],dtype=np.float64)
        for k in range(0,nact):
            print(k, ' ', end='\r', flush=True)
            array_[:,k] = array[k,idxpup[0],idxpup[1]]
    return array_


                  
def plt_plot(a):
    plt.figure()
    plt.plot(a)
    plt.show(block=False)

def plt_imshow(a):
    plt.figure()
    plt.imshow(a,cmap='gray', origin='lower')
    plt.show(block=False)

def expand_a(a,size,idxpup,more,norm):
    if more==0:
        sha=np.zeros([size,size],dtype=np.float64)
        sha[idxpup[0],idxpup[1]]=a/norm
    if more !=0:
        sha=np.zeros([size,size,more],dtype=np.float64)
        sha[idxpup[0],idxpup[1],:]=a/norm
    return sha


def plt_imshow_expa(a,size,idxpup):
    more=0
    plt.figure()
    plt_imshow(expand_a(a,size,idxpup,more))
    plt.show(block=False)
 

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    print(os.getpid(),'>',process.memory_info().rss/1.e9,'G')



def VK_DSP_up(diameter,r0,L0,size,dim,Pcor,pupil_):
    """
    Computes a 2D Power Spectral Density map for Von Karman statistics
    - diameter: size of the telescope must correspond to dim*resolution
    - r0: seeing in meters
    - L0: Outer-Scale of turbulence
    - size: size of the 2D array. Must correspond to the FFT size for Covariance computation
    - dim: size of pupil in pix. With diameter used only to compute the spatial freq df
    - Pcor: if ==1 computes the Piston term
    - pupil_: pupil shape used to compute piston term
    
    """
    dom=pupil_.shape[0]
    pupil=np.zeros([size,size],dtype=np.float64)
    pupil[ size//2-dom//2:size//2+dom//2 , size//2-dom//2:size//2+dom//2 ] = pupil_
    fmax=dim/diameter*0.5
    if L0==0:
        L0=1.e6
    fx=np.linspace(-size/2, size/2-1, size)/(size/2) * fmax
    fy=np.linspace(-size/2, size/2-1, size)/(size/2) * fmax
    ffx,ffy=np.meshgrid(fy,fx)
    ffr=np.sqrt(ffx**2+ffy**2)
    cst = (math.gamma(11./6.)**2/(2.*np.pi**(11./3.)))*(24.*math.gamma(6./5.)/5.)**(5./6.)
    PSD_atm=np.zeros([size,size],dtype=np.float64)
    if Pcor == 0:
        PSD_atm [:,:] = cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-11/6)
    pupils=myshift2D(pupil,size//2,size//2)
    ft_pupils=myshift2D(np.fft.fft2(pupils),size//2,size//2)
    pterm = np.abs(ft_pupils)**2
    pterm0 = pterm/np.max(pterm)
    if Pcor == 1:
        PSD_atm [:,:] = (cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-11/6))*(1.-pterm0)
        
    #PSD_atm [:,:] = cst*r0**(-5/3) *  (ffr**2 + (1./L0)**2)**(-6)#**(-11/6)
    #PSD_atm[size//2,size//2] = 0.
    df=abs(fx[size//2]-fx[size//2-1])
    
    return PSD_atm, df , pterm

def myshift1D(vector,xshift):
    return np.roll(vector,int(xshift))
                   
def myshift2D(image,xshift,yshift):
    return np.roll(np.roll(image,int(xshift),0),int(yshift),1)



def mkp(real_sz,dim_x, diam, eps, xc=None, yc=None, dim_y=None): #, YC=yc, DIM_Y=dim_y):
    """
    Compute a pupil and x,y coordinates 2D map
    - real_sz: size in meter of overall map of size dim_x
    - dim_x: size of array containing pupil
    - diam: size in meters of circular pupil
    - eps: linear central obscuration
    - xc,yc : position of center in pixels. By default center is on 4 central pixels.
    - size of y dimension if different from x
    """
    class geom:
        pass
    
    # by default the pupil is centered on the center of the array
    # (i.e. NOT on a pixel for a square arrayâ¦)
    if (xc is None)   : xc    = (dim_x-1)/2. #*1.*real_sz/dim_x 
    if (dim_y is None): dim_y = dim_x
    if (yc is None)   : yc    = (dim_y-1)/2. #*1.*real_sz/dim_x 
    
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    
    xx=xx*real_sz/(dim_x-1)
    yy=yy*real_sz/(dim_x-1)
    
    dummy = np.sqrt(xx**2.+yy**2.) #/(diam/2.)
    
    pupil=dummy.copy()*0.
    # pupil array is 1 under the pupil & 0 elsewhere
    idx1=np.where(dummy < diam/2.)
    idx2=np.where(dummy < eps*diam/2)
    pupil[idx1]=1.0
    pupil[idx2]=0.
    geomr=geom()
    geomr.pupil=np.intc(pupil)
    geomr.xx=xx
    geomr.yy=yy
    return geomr
   

def give_zernike(geom, diameter, nmodes, norming=None, meaning=None):
    """
    Returns 2D maps of zernike modes. Used PROPER package.
    - geom: structure computed with mkp
    - diameter: diameter of pupil in meter
    - nmodes: number of Zernike modes returned
    - norming: when ==1 then normalised to one
    - meaning: when ==1 then mean is subtracted
    """
    max_z=nmodes
    
    n=(geom.xx).shape[0]
    
    
    modz = np.zeros([n,n,nmodes], dtype = np.float64)
    
    x=geom.xx[n//2,:]/(diameter/2)
    y_=geom.yy[:,n//2]/(diameter/2)
    
    x_pow_2 = x**2
    
    ########### PART COPIED FROM prop_zernike.py
    
    zlist, maxrp, maxtc = prop_noll_zernikes(max_z, COMPACT = True, EXTRA_VALUES = True)
    
    for numz in range(1,max_z+1):
        dmap = np.zeros([n,n], dtype = np.float64)
        
        for j in range(n):
            ab = np.zeros(n, dtype = np.float64)
            y = y_[j]   #(j - n//2) * proper.prop_get_sampling(a) / beam_radius
            r = np.sqrt(x_pow_2 + y**2)
            t = np.arctan2(y,x)
        
            # predefine r**power, cos(const*theta), sin(const*theta) vectors
            for i in range(2, maxrp+1):
                rps = str(i).strip()
                cmd = "r_pow_" + rps + " = r**i"
                exec(cmd) 
                
            for i in range(1, maxtc+1):
                tcs = str(i).strip()
                cmd = "cos" + tcs + "t = np.cos(i*t)"
                exec(cmd)
                cmd = "sin" + tcs + "t = np.sin(i*t)"
                exec(cmd)
        
            tmp = eval(zlist[numz])        
            dmap[j,:] = tmp
            
        modz[:,:,numz-1] = dmap
    return modz

def prop_noll_zernikes(maxz, **kwargs):
    """
    function taken from proper-library.sourceforge.net
    Krist, J., "PROPER: an optical propagation library for IDL", Proc. SPIE, 6675, 66700P (2007).

    Return a string array in which each element contains the Zernike polynomial
    equation corresponding to the index of that element.

    The polynomials are orthonormal for an unobscured circular aperture. They
    follow the ordering convention of Noll (J. Opt. Soc. America, 66, 207 (1976)).
    The first element (0) is always blank. The equations contain the variables "r"
    (normalized radius) and "t" (azimuth angle in radians). The polynomials have
    an RMS of 1.0 relative to a mean of 0.0.

    Parameters
    ----------
    maxz : int
        Maximum number of zernike polynomials to return. The returned string
        array will have max_z+1 elements, the first being blank.


    Returns
    -------
    z_list : numpy ndarray
        Returns a string array with each element containing z zernike polynomial
        (the first element is blank).

    max_r_power : float, optional
        The maximum radial exponent.

    max_theta_multiplier : float, optional
        Maximum multiplier of the angle.


    Other Parameters
    ----------------
    COMPACT : bool
       If set, the equations are returned using the naming convention for terms
       assumed by PROP_ZERNIKES.

    EXTRA_VALUES : bool
        If true, return maximum radial power and maximum theta multiplier in
        addition to equation strings

    Notes
    -----
    For example:
        zlist = prop_noll_zernikes(5)
	for i in range(1, 6):
            print(i, '   ', zlist[i])

	will display:
      		1   1
      		2   2 * (r)  * cos(t)
      		3   2 * (r)  * sin(t)
      		4   sqrt(3) * (2*r^2 - 1)
      		5   sqrt(6) * (r^2)  * sin(2*t)
    Note that PROP_PRINT_ZERNIKES can also be used to print a table of Zernikes.
    """
    if ("COMPACT" in kwargs and kwargs["COMPACT"]):
        rop = "_pow_"
    else:
        rop = "**"

    max_r_power = 0
    max_theta_multiplier = 0

    z_list = np.zeros(maxz+1, dtype = "S250")    # z_list[0] is always blank
    iz = 1
    n = 0

    while (iz <= maxz):
        for m  in range(np.mod(n,2), n+1, 2):
            for p in range(0, (m != 0) + 1):
                if n != 0:
                    if m != 0:
                        val = 2 * (n+1)
                    else:
                        val = n + 1

                    sqrt_val = int(np.sqrt(val))
                    if val == sqrt_val**2:
                        t = str(sqrt_val).strip() + " * ("
                    else:
                        t = "math.sqrt(" + str(val).strip() + ") * ("
                else:
                    z_list[iz] = "1"
                    iz += 1
                    continue

                for s in range(0, (n-m)//2 + 1):
                    term_top = int((-1)**s) * int(factorial(n-s))
                    term_bottom = int(factorial(s)) * int(factorial(int((n+m)/2-s))) * int(factorial(int((n-m)/2 - s)))
                    term_val = int(term_top / term_bottom)
                    term = str(np.abs(term_val)).strip() + ".0"
                    term_r = int(n - 2*s)
                    rpower = str(term_r).strip()

                    if max_r_power < term_r:
                        max_r_power = term_r

                    if term_top != 0:
                        if s == 0:
                            if term_val < 0:
                                sign = "-"
                            else:
                                sign = ""
                        else:
                            if term_val < 0:
                                sign = " - "
                            else:
                                sign = " + "

                        if rpower == "0":
                            t += sign + term
                        elif term_r == 1:
                            if term_val != 1:
                                t += sign + term + "*r"
                            else:
                                t += sign + "r"
                        else:
                            if term_val != 1:
                                t += sign + term + "*r" + rop + rpower
                            else:
                                t += sign + "r" + rop + rpower

                if m > max_theta_multiplier:
                    max_theta_multiplier = m

                if m == 0:
                    cterm = ""
                else:
                    if (m != 1):
                        term_m = str(int(m)).strip() + "*t"
                    else:
                        term_m = "t"
                    if np.mod(iz,2) == 0:
                        if ("COMPACT" in kwargs and kwargs["COMPACT"]):
                            cterm = " * cos" + str(int(m)).strip() + "t"
                        else:
                            cterm = " * cos(" + term_m + ")"
                    else:
                        if ("COMPACT" in kwargs and kwargs["COMPACT"]):
                            cterm = " * sin" + str(int(m)).strip() + "t"
                        else:
                            cterm = " * sin(" + term_m + ")"

                if cterm != "":
                    z_list[iz] = t + ")" + cterm
                else:
                    t += ")"
                    z_list[iz] = t

                iz += 1
                if iz > maxz:
                    break

            if iz > maxz:
                break

        n += 1

    if ("EXTRA_VALUES" in kwargs and kwargs["EXTRA_VALUES"]):
        return (z_list, max_r_power, max_theta_multiplier)
    else:
        return z_list



def count(SZ,sz,NDIVL,nact,disp):
    """
    Evaluates the amount of memory need to compute the covariance with DO_HHt
    - SZ: size of FFTs
    - sz size of input IF and pupil arrays
    - NDIVL: Linear factor of work division. NDIVL=N: COV divided by NxN blocks + rest
    - nact: number of actuators
    - disp: if ==1 then prints intermediate calculations
    """
    BLOCKL=nact//NDIVL
    REST=nact-BLOCKL*NDIVL
    SC=16
    SD=8  
    S_DmataC = SZ**2*BLOCKL*SC
    S_DmatcC = SZ**2*REST*SC
    S_IF = sz**2*nact*SD
    S_IN_A = SZ**2*BLOCKL*SD
    S_IN_B = SZ**2*BLOCKL*SD
    S_aa = SZ**2*BLOCKL*SC
    S_cc = SZ**2*REST*SC
    S_IFh = SZ**2*BLOCKL*SC
    S_DoIF = SZ**2*BLOCKL*SC
    MM_stat = nact**2*SD
    S_tmp = SZ**2*BLOCKL*SC
    S_cj_tmp = SZ**2*BLOCKL*SC
    TOTAL = S_DmataC + S_DmatcC + S_IF + S_IN_A +S_IN_B + S_aa + S_cc + S_IFh + S_DoIF + MM_stat + S_tmp + S_cj_tmp
    TOTAL_IN = S_DmataC + S_DmatcC + S_IF + S_IN_A + S_aa + S_cc + S_IFh + S_DoIF + MM_stat + S_cj_tmp
    if NDIVL==1:
        TOTAL_IN = S_DmataC  + S_IF + S_aa + S_IFh + S_DoIF + MM_stat + S_cj_tmp
    if disp==1:print('S_DmataC=',S_DmataC/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_DmatcC=',S_DmatcC/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_IF=',S_IF/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_IN_A=',S_IN_A/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_IN_B=',S_IN_B/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_aa=',S_aa/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_cc=',S_cc/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_IFh=',S_IFh/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_DoIF=',S_DoIF/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('MM_stat=',MM_stat/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_tmp=',S_tmp/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:print('S_cj_tmp=',S_cj_tmp/1.e9, 'GB')
    if disp==1:print(' ')
    if disp==1:
        print('TOTAL MEMORY FOR ALL VARIABLES=',TOTAL/1.e9, 'GB')
        print('MAX TOTAL ESTIMATED AT ONE TIME=',TOTAL_IN/1.e9, 'GB')
    return [TOTAL, TOTAL_IN]


def estimate_ndivl(SZ,sz,nact,MEMmax):
    """
    Computes an estimate of the NDIVL parameter for Covariance work splitting
    - SZ: size of FFT
    - sz: size of IFs and pupil array
    - nact: number of IFs involved in the covariance computation
    - MEMmax: Maximum amount of memory (Bytes) allowed to the COVARIANCE COMPUTATION
    Note: in case memory happens to be insufficient the computation aborts in 'Memory error'
    """
    
    nt=1000
    MEME=np.zeros(nt,dtype=np.float64)
    for k in range(1,nt+1):
        TOTAL,TOTAL_in = count(SZ,sz,k,nact,0)
        if k<=2:
            MEME[k-1] = TOTAL_in
        if k>2:
            MEME[k-1] = TOTAL
    idx=(np.where(MEME < MEMmax))
    #pdb.set_trace()
    val=idx[0][0]
    print('RECOMMENDED NDIVL = ',val)    
    return MEME, val


def build_SpecificBasis_F(Tspm,IFma,DELTA,GAMMA,alpha,ortho, check,amp_check):
    """
    Compute commands producing specific modes with force minimization
    - Tspm: Matrix of serialized Specific modes [npup,nspm], npup is number of pupil points
    - IFma:Matrix of serialized IFs
    - DELTA: Cross-product of IFs (pre-computed)
    - GAMMA: Tikhonov regularization term (= stiffness matrix for force minimization)
    - alpha: Regularization parameter: the larger alpha the smaller the force
    - ortho: if ortho==1 then a Gramm-Schmid-like orthonormalization is done (qr transform)
    - check: if check==1 then performance of mode is printed (fitting error  / Forces)
    - amp_check: amplitued applied to modes to compute the check
    """
    tpup = IFma.shape[0] # NUMBER OF VALID OPD POINTS IN THE PUPIL
    nact = IFma.shape[1]
    FIT = (DELTA + alpha*GAMMA.T @ GAMMA).I @ IFma.T
    CMD_TIK = FIT @ Tspm
    if check==1:
        CMD_TIK_check = CMD_TIK * amp_check
        CMD_TIK_check_opd = IFma @ CMD_TIK_check
    if check == 1:
        F_fit = GAMMA @ CMD_TIK_check
        print('RMS opd error =', np.std(CMD_TIK_check_opd-Tspm*amp_check,axis=0))
        print('RMS Force =', np.std(F_fit,axis=0))
        print('MAX Force =', np.max(np.abs(F_fit),axis=0))
    if ortho == False:
        CMD_FINAL = CMD_TIK.copy()
    if ortho == True: #Orthonormalization of the Specific basis
        CMD_TIK_opd = IFma @ CMD_TIK 
        Q,R = np.linalg.qr(CMD_TIK_opd)
        CMD_FINAL = CMD_TIK @ R.I * np.sqrt(tpup)
    return CMD_FINAL



def build_SpecificBasis_C(Tspm,IFma,DELTA,lim,ortho, check,amp_check):
    """
    Computes the commands producing specific modes W/O regularization, in Positions space
    - Tspm: Matrix of serialized Specific modes [npup,nspm], npup is number of pupil points
    - IFma:Matrix of serialized IFs
    - DELTA: Cross-product of IFs (pre-computed)
    - lim: lim == number of modes if large or ratio between smallest and largest eval kept
    - ortho: if ortho==1 then a Gramm-Schmid-like orthonormalization is done (qr transform)
    - check: if check==1 then performance of mode is printed (fitting error  / Positions)
    - amp_check: amplitued applied to modes to compute the check
    """
    
    tpup = IFma.shape[0] # NUMBER OF VALID OPD POINTS IN THE PUPIL
    nact = IFma.shape[1]
    if lim == 0:
        iDELTA = DELTA.I
    if lim !=0:
        ud,sd,vdh=np.linalg.svd(DELTA)
    if lim <=1.:
        nmax=np.max(np.where(sd/np.max(sd) > lim))
    if lim > 1:
        nmax=lim
    print('NMAX = ',nmax)
    MEV=np.zeros([nact,nact],dtype=np.float64)
    MEV[0:nmax,0:nmax] = np.diag(1./sd[0:nmax])
    iDELTA = vdh.T @ MEV @ ud.T
    FIT = iDELTA @ IFma.T
    CMD_TIK = FIT @ Tspm
    if check==1:
        CMD_TIK_check = CMD_TIK * amp_check
        CMD_TIK_check_opd = IFma @ CMD_TIK_check
        print('RMS opd error =', np.std(CMD_TIK_check_opd-Tspm*amp_check,axis=0))
        print('RMS Positions =', np.std(CMD_TIK_check,axis=0))
        print('MAX Positions =', np.max(np.abs(CMD_TIK_check),axis=0))
    if ortho == False:
        CMD_FINAL = CMD_TIK.copy()
    if ortho == True: #Orthonormalization of the Specific basis
        CMD_TIK_opd = IFma @ CMD_TIK 
        Q,R = np.linalg.qr(CMD_TIK_opd)
        CMD_FINAL = CMD_TIK @ R.I * np.sqrt(tpup)
    return CMD_FINAL


def build_SeedBasis_F(IFma, SpM, K, beta):
    """
    Computes the DM eigen-modes analysing OPD in function of forces. Seed basis for KLs.
    - IFma:Matrix of serialized IFs
    - SpM: Matrix of serialized Specific Modes (used to compute in space orthogonal to SpM)
    - K: Stiffness matrix. Position to Force matrix.
    - Beta: Regularization parameter for large Positions damping.
    """
    
    tpup = IFma.shape[0] # NUMBER OF VALID OPD POINTS IN THE PUPIL
    nact = IFma.shape[1]
    Id=np.eye(nact)
    nspm=SpM.shape[1]
    ## FORCE-LIKE BASIS WITH POSITIONS DAMPING PARAMETER BETA
    FB=(K.I+beta*Id)
    iFB=FB.I
    ## FORCE-LIKE IF
    IF_F = IFma @ FB
    DELTA_IF_F = IF_F.T @ IF_F
    ## Specific Modes in Force-like space
    #TAU = iCB @ SpM
    ## GENERATOR IN FORCE SPACE FOR MODIFIED IFs (ORTHOGONAL TO SpM modes)
    G=np.eye(nact) - iFB @ SpM @ (SpM.T @ iFB.T @ DELTA_IF_F @ iFB @ SpM).I @ SpM.T @ iFB.T @ DELTA_IF_F
    ## COMPUTE EIGEN MODES OF MODIFIED IFs
    U_Fm,S_Fm,V_FmT = np.linalg.svd(IF_F @ G,full_matrices=False)
    del U_Fm
    V_Fm = V_FmT.T
    del V_FmT
    #pdb.set_trace()
    SB = FB @ G @ V_Fm[:,0:nact-nspm] @ np.diag(1./S_Fm[0:nact-nspm])*np.sqrt(tpup)
    return SB



def build_SeedBasis_C(IFma, SpM, DELTA,lim):
    """
    Computes the DM eigen-modes analysing OPD in function of forces. Seed basis for KLs.
    - IFma:Matrix of serialized IFs
    - SpM: Matrix of serialized Specific Modes (used to compute in space orthogonal to SpM)
    - K: Stiffness matrix. Position to Force matrix.
    - Beta: Regularization parameter for large Positions damping.
    """
    
    tpup = IFma.shape[0] # NUMBER OF VALID OPD POINTS IN THE PUPIL
    nact = IFma.shape[1]
    Id=np.eye(nact)
    nspm=SpM.shape[1]
    
    ## GENERATOR IN FORCE SPACE FOR MODIFIED IFs (ORTHOGONAL TO SpM modes)
    G = np.eye(nact) - SpM @ (SpM.T @ DELTA @ SpM).I @ SpM.T @ DELTA
    Mp,D2,Mpt = np.linalg.svd(G.T @ DELTA @ G)
    
    if lim > 1:
        nmax=lim
    if lim <= 1:
        nmax=np.max(np.where( D2/np.max(D2) > lim))
    if lim == 0.:
        nmax=nact-nspm

    M = Mp[:,0:nmax] @ np.diag(np.sqrt(1./D2[0:nmax]))*np.sqrt(tpup)
    SB = G @ M
    
    return SB







def build_KLBasis(HHt,SBf,DELTA,nmoKL,check):
    """
    Computes the KL modes by diaginalization of the atm Cov matrix in the Seed basis
    - HHt: Covariance matrix of IFs projection on turbulent phase
    - SBf: Seed basis
    - DELTA: Cross-product of IFs (pre-computed)
    - nmoKL: number of KL modes selected (max of modes permitting to keep orthogonality)
    - check: if==1 then prints the performance in terms of verification of Covariance diag.
    """
    SB=np.asmatrix(SBf[:,0:nmoKL])
    DELTA_SB = SB.T @ DELTA @ SB
    Cp = DELTA_SB.I @ SB.T @ HHt @ SB @ DELTA_SB.I.T
    Uc,Sc,VcT = np.linalg.svd(Cp)
    Vc=VcT.T
    del Uc, VcT
    KL_F = SB @ Vc
    DELTA_KL_F = KL_F.T @ DELTA @ KL_F
    iDELTA_KL_F=DELTA_KL_F.I
    if check==1:
        verif_SIGMA2 = iDELTA_KL_F @KL_F.T @ HHt @ KL_F @ iDELTA_KL_F
        print('KL WITH DOUBLE DIAGONALISATION: COVARIANCE ERROR = ',np.max(np.abs(verif_SIGMA2-np.diag(Sc))))  
    return KL_F #, Sc



def FIT_ONB(B,DELTA,HHt,PSD_atm,df,tpup):
    """
    Computes the fitting error in function of orthonormal basis. Fitting Efficiency.
    - B: Basis [nact,nmodes] to analyse
    - DELTA: Cross-product of IFs (pre-computed)
    - HHt: Covariance matrix of IFs projection on turbulent phase
    - PSD_atm: 2D Power Spectral Density of turbulent phase used to compute HHt
    - df: resolution element in Fourier space
    - tpup: number of valid points in the pupil
    """
    DELTA_B = B.T @ DELTA @ B
    iDELTA_B = DELTA_B.I
    rmsPSD_wiP = np.sqrt(np.sum(PSD_atm*df**2))*0.5e-6/(2.*np.pi)
    Cmo_B = iDELTA_B @ B.T @ HHt @ B @ iDELTA_B.T*(0.5e-6/(2.*np.pi))**2
    rmsDM_wiP = np.sqrt(np.sum(np.asarray(Cmo_B) * np.asarray(DELTA_B))/tpup )
    rmsDM = np.sqrt(np.sum(np.diag(Cmo_B[1:,1:])))
    #pdb.set_trace()
    fitting_error=np.sqrt(rmsPSD_wiP**2-rmsDM_wiP**2)
    #fitting_error=145.e-9
    print('FITTING ERROR IS:',fitting_error)
    RESfnmo_B=np.zeros(B.shape[1],dtype=np.float64)
    for k in range(0,B.shape[1]):
        print(k, ' ', end='\r', flush=True)
        RESfnmo_B[k] = np.sqrt( (rmsDM**2+fitting_error**2)-np.sum(np.diag(Cmo_B[1:k,1:k])))
    return RESfnmo_B



def POS_ONB(B,DELTA,HHt,nmo):
    """
    Computes the statistical distribution of Positions for Basis B for HHt covariance matrix
    - B [nact,nmodes]: basis to analyse
    - DELTA: Cross-product of IFs (pre-computed)
    - HHt: Covariance matrix of IFs projection on turbulent phase
    - nmo: number of reconstruction modes
    """
    DELTA_B = B.T @ DELTA @ B
    iDELTA_B = DELTA_B.I
    Cmo_B = iDELTA_B @ B.T @ HHt @ B @ iDELTA_B.T*(0.5e-6/(2.*np.pi))**2
    PPT_B_nmo = B[:,1:nmo] @ Cmo_B[1:nmo,1:nmo] @ B[:,1:nmo].T
    RMS_PPT_B_nmo =  np.sqrt(np.diag(PPT_B_nmo))
    return RMS_PPT_B_nmo

def FOR_ONB(B,DELTA,HHt,nmo,K):
    """
    Computes the statistical distribution of Forces for Basis B for HHt covariance matrix
    - B [nact,nmodes]: basis to analyse
    - DELTA: Cross-product of IFs (pre-computed)
    - HHt: Covariance matrix of IFs projection on turbulent phase
    - nmo: number of reconstruction modes
    - K: Positions to Forces matrix. Stiffness matrix.
    """
    DELTA_B = B.T @ DELTA @ B
    iDELTA_B = DELTA_B.I
    Cmo_B = iDELTA_B @ B.T @ HHt @ B @ iDELTA_B.T*(0.5e-6/(2.*np.pi))**2
    PPT_B_nmo = B[:,1:nmo] @ Cmo_B[1:nmo,1:nmo] @ B[:,1:nmo].T
    RMS_FFT_B_nmo =  np.sqrt(np.diag(K @ PPT_B_nmo @ K.T))
    return RMS_FFT_B_nmo


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def DO_HHt(ilal,PSD,df,pupil,BLOCKL,REST,SZ,verb): 
    """
    Computes the statistical covariance of the projection of IFs on turbulent phase screens
    - ilal: 2D IFs arranged like: [n_elem,size,size]. size must be <= SZ
    - PSD: 2D Power Spectral Density of turbulent phase of size SZ
    - df: numerical increment in Fourier Spatial Frequency space
    - BLOCKL: linear size of blocks computed to optimize memory
    - REST: size of last block
    - SZ: size of FFTs of IFs and of PSD.
    
    """
    st0=time.time()
    nact=ilal.shape[0]
    size=ilal.shape[1]

    ## APPLY PUPIL ON IFs
    for k in range(0,nact):
        ilal[k,:,:]*=pupil
    
    NCL=nact//BLOCKL
    NDIVL=NCL
    tot0=0.
    if REST!=0:
        NCL=NCL+1
    if REST==0:
        nac2=NDIVL*(NDIVL+1)/2.*BLOCKL**2
    if REST!=0:
        nac2=NDIVL*(NDIVL+1)/2.*BLOCKL**2 +  NDIVL*BLOCKL*REST +  REST**2
    
    if verb==1: print(' ')
    if verb==1: print('CREATING FFTW PLANS...')

    aa = pyfftw.empty_aligned((BLOCKL,SZ, SZ), dtype='complex128')
    
    cc = pyfftw.empty_aligned((REST,SZ, SZ), dtype='complex128')

    fft_object_aa = pyfftw.FFTW(aa,aa, axes=(1,2),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=mp.cpu_count())
  
    fft_object_cc = pyfftw.FFTW(cc,cc, axes=(1,2),flags=('FFTW_MEASURE',),direction='FFTW_FORWARD',threads=mp.cpu_count())

 
    ## CREATION OF COVARIANCE MATRIX TO STORE RESULT 
    MM_stat=np.zeros([nact,nact],dtype=np.float64)
    stot_time = time.time()
    if verb==1: print('NCL = ', NCL)

    ## SHAPING PSD FOR THE ELEMENT-WISE MULTIPLICATION IN (DoIF) (eq. 29)
    PSD_=np.fft.fftshift(PSD) 
    MAT_PSD = np.reshape(PSD_,SZ*SZ)

    Dma=np.tile(MAT_PSD*df**2,BLOCKL)
    Dmata = np.asmatrix(np.reshape(Dma,[BLOCKL,SZ*SZ]))#.T
    del Dma
    DmataC=ne.evaluate("complex(Dmata,0.)") ## D for DoIF WHEN NB. ACT IS BLOCKL
    del Dmata
    if REST!=0:
        Dmc=np.tile(MAT_PSD*df**2,REST)
        Dmatc = np.asmatrix(np.reshape(Dmc,[REST,SZ*SZ]))#.T
        del Dmc
        DmatcC=ne.evaluate("complex(Dmatc,0.)") ## D for DoIF WHEN NB. ACT IS REST
        del Dmatc
    
     
    for kA in range(0,NCL):

        
        for kB in range(kA,NCL):

            if verb==1: print(' ')
            if verb==1: print('Expanding IFMs...', kA, kB)
            

            if REST==0:
                
                IFMs_A=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)
                IFMs_A[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kA*BLOCKL:(kA+1)*BLOCKL,:,:]
                
                if kB != kA:
                    IFMs_B=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)    
                    IFMs_B[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kB*BLOCKL:(kB+1)*BLOCKL,:,:]
                
                if kB == kA:
                    IFMs_B = IFMs_A

           

            if REST!=0:
                
                if kA < NCL-1:
                    IFMs_A=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)
                    IFMs_A[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kA*BLOCKL:(kA+1)*BLOCKL,:,:]
                    
                if kB < NCL-1 and kB != kA:
                    IFMs_B=np.zeros([BLOCKL,SZ,SZ],dtype=np.float64)
                    IFMs_B[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[kB*BLOCKL:(kB+1)*BLOCKL,:,:]
                   
                if kA == NCL-1:
                    IFMs_A=np.zeros([REST,SZ,SZ],dtype=np.float64)
                    IFMs_A[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[-REST:,:,:]
                    
                if kB == NCL-1 and kB != kA:
                    IFMs_B=np.zeros([REST,SZ,SZ],dtype=np.float64)
                    IFMs_B[:,SZ//2-size//2:SZ//2+size//2,SZ//2-size//2:SZ//2+size//2]=ilal[-REST:,:,:]
                    
                if kB == kA:
                   IFMs_B = IFMs_A


            if verb==1: print(' ')
            if verb==1: print('EXECUTING ...', kA, kB)
            
            if REST != 0:
                if kA < NCL-1:
                    if verb==1: print('STARTED NE WORK  A ...', kA, kB)
                    aa[:,:,:]=ne.evaluate("complex(IFMs_A,0.)")
                    if verb==1: print('FINISHED NE WORK A ...', kA, kB)
                    del IFMs_A
                    #pdb.set_trace()
                  
                    if verb==1: print('STARTED FFT  A ...', kA, kB)
                    res=fft_object_aa()
                    if verb==1: print('FINISHED FFT  A ...', kA, kB)
                    
                    if verb==1: print('STARTED FILLING  A ...', kA, kB)
                    tmp=np.reshape(aa,[BLOCKL,SZ*SZ])
                    cj_tmp=ne.evaluate("conj(tmp)")
                    del tmp
                    IFh = np.asmatrix(cj_tmp)
                    del cj_tmp

                    if verb==1: print('FINISHED FILLING  A ...', kA, kB)
                    
                                   
                if kB < NCL-1:
                    if verb==1: print('STARTED NE WORK  B ...', kA, kB)
                    aa[:,:,:]=ne.evaluate("complex(IFMs_B,0.)")
                    del IFMs_B
                    if verb==1: print('FINISHED NE WORK  B ...', kA, kB)
                    
                    if verb==1: print('STARTED FFT  B ...', kA, kB)
                    res=fft_object_aa()
                    if verb==1: print('FINISHED FFT  B ...', kA, kB)
                    
                    if verb==1: print('STARTED FILLING  B ...', kA, kB)
                   
                    tmpb = np.reshape(aa,[BLOCKL,SZ*SZ])
                    DoIF=np.asmatrix(ne.evaluate("tmpb*DmataC")).T
                    del tmpb
               

                if kA==NCL-1:
                    if verb==1: print('STARTED NE WORK  A ...', kA, kB)
                    cc[:,:,:]=ne.evaluate("complex(IFMs_A,0.)")
                    if verb==1: print('FINISHED NE WORK A ...', kA, kB)
                    del IFMs_A
                    
                    if verb==1: print('STARTED FFT  A ...', kA, kB)
                    res=fft_object_cc()
                    if verb==1: print('FINISHED FFT  A ...', kA, kB)
                    
                    if verb==1: print('STARTED FILLING  A ...', kA, kB)
                    tmp=np.reshape(cc,[REST,SZ*SZ])
                    cj_tmp=ne.evaluate("conj(tmp)")
                    del tmp
                    IFh = np.asmatrix(cj_tmp)
                    del cj_tmp
                    
                    if verb==1: print('FINISHED FILLING  A ...', kA, kB)


                if kB==NCL-1:
                    if verb==1: print('STARTED NE WORK  B ...', kA, kB)
                    cc[:,:,:]=ne.evaluate("complex(IFMs_B,0.)")
                    if verb==1: print('FINISHED NE WORK B ...', kA, kB)
                    del IFMs_B
                    
                    if verb==1: print('STARTED FFT  B ...', kA, kB)    
                    res=fft_object_cc()
                    if verb==1: print('FINISHED FFT  B ...', kA, kB)
                    
                    if verb==1: print('STARTED FILLING  B ...', kA, kB)      
                    tmpb = np.reshape(cc,[REST,SZ*SZ])
                    DoIF=np.asmatrix(ne.evaluate("tmpb*DmatcC")).T
                    del tmpb
                    if verb==1: print('FINISHED FILLING  B ...', kA, kB)  
                                                       

            if REST == 0:
                if verb==1: print('STARTED NE WORK  A ...', kA, kB)
                aa[:,:,:]=ne.evaluate("complex(IFMs_A,0.)")
                if verb==1: print('FINISHED NE WORK A ...', kA, kB)
                del IFMs_A
                
                if verb==1: print('STARTED FFT  A ...', kA, kB) 
                res=fft_object_aa()
                if verb==1: print('FINISHED FFT  A ...', kA, kB)
                
                if verb==1: print('STARTED FILLING  A...', kA, kB)
                tmp=np.reshape(aa,[BLOCKL,SZ*SZ])
                cj_tmp=ne.evaluate("conj(tmp)")
                del tmp
                #MAT_A=np.asmatrix(cj_tmp)
                IFh = np.asmatrix(cj_tmp)
                del cj_tmp


                if verb==1: print('STARTED NE WORK  B ...', kA, kB)
                aa[:,:,:]=ne.evaluate("complex(IFMs_B,0.)")
                if verb==1: print('FINISHED NE WORK B ...', kA, kB)
                del IFMs_B
                
                if verb==1: print('STARTED FFT  B ...', kA, kB) 
                res=fft_object_aa()
                if verb==1: print('FINISHED FFT  B ...', kA, kB)
                
                if verb==1: print('STARTED FILLING  B ...', kA, kB)
                #pdb.set_trace()
                
                tmpb = np.reshape(aa,[BLOCKL,SZ*SZ])
                DoIF=np.asmatrix(ne.evaluate("tmpb*DmataC")).T
                del tmpb
               
               
                if verb==1: print('FINISHED FILLING  B ...', kA, kB)
    
            if verb==1: print(' ')      
            if verb==1: print('SUMMING CONVOLUTION BY MATRIX MULTIPLICATION...', kA, kB)
            s_time = time.time()
            if REST==0:
                MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL] = np.ascontiguousarray((IFh @ DoIF).real)
                
                if kB != kA:
                    MM_stat[kB*BLOCKL:(kB+1)*BLOCKL,kA*BLOCKL:(kA+1)*BLOCKL] = MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL].T

                
            if REST!=0:
                
                if kA<NCL-1 and kB< NCL-1:
                    MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL] = np.ascontiguousarray((IFh @ DoIF).real) 
                    
                    
                    
                if kA==NCL-1 and kB==NCL-1:
                    MM_stat[-REST:,-REST:] = np.ascontiguousarray((IFh @ DoIF).real) 

                    
                if kA==NCL-1 and kB<NCL-1:
                    MM_stat[-REST:,kB*BLOCKL:(kB+1)*BLOCKL] =  np.ascontiguousarray((IFh @ DoIF).real) 
                    
                if kA<NCL-1 and kB==NCL-1:
                    
                    MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,-REST:] =  np.ascontiguousarray((IFh @ DoIF).real)
                    
                if kB != kA:
                    MM_stat[kB*BLOCKL:(kB+1)*BLOCKL,kA*BLOCKL:(kA+1)*BLOCKL] = MM_stat[kA*BLOCKL:(kA+1)*BLOCKL,kB*BLOCKL:(kB+1)*BLOCKL].T

            
                    
            e_time = time.time() - s_time
            if verb==1: print(' ')
            if verb==1: print('DONE. MATRIX MULTIPLICATION TOOK: ', e_time, kA, kB)

 
            ## INFORM TIMING AND WHAT IS LEFT
            if REST==0:
                tot0=tot0+BLOCKL**2
                et0=time.time()-st0
                PERC=tot0/nac2*100.
                print("TIME ELAPSED: %d sec. COMPLETED: %d %%" %(int(et0),int(PERC)))
            if REST!=0:
                if kA < NCL-1:
                    AS=BLOCKL
                if kB < NCL-1:
                    BS=BLOCKL
                if kA == NCL-1:
                    AS=REST
                if kB == NCL-1:
                    BS=REST
                tot0=tot0+AS*BS
                et0=time.time()-st0
                PERC=tot0/nac2*100.
                print("TIME ELAPSED: %d sec. COMPLETED: %d %%" %(int(et0),int(PERC)))


                
    etot_time = time.time() - stot_time
    if verb==1: print('DONE. COVARIANCE MATRIX TOOK: ', etot_time)
    if verb==1: print('DF IS:', df)         
    return  MM_stat #/(SZ**2)

