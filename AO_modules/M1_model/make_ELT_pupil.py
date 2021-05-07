#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:38:32 2018

@author: gendron
"""

import numpy as np
import matplotlib.pyplot as plt

plt.ion()


"""

DISCLAIMER:
This set of functions intend to produce images of the ELT pupil. 
They have been coded by E. Gendron in the hope they can be helpful and serve various purposes. 
Their validity has been checked against — ELT pupil dimensions written in the  
“Common ICD between the E-ELT Nasmyth Instruments and the Rest of the E-ELT System” ESO-253082, v 3, 
released 2017-11-07, Figure 4 — ESO data package and was deemed to be correct and accurate.
Despite some careful debugging, this software is not official, and is distributed “as is” 
without waranties of any kind concerning potential residual bugs or inaccuracies that could be found in the future.
His author would really appreciate any feedback (eric.gendron_at_obspm.fr) if a user finds any problem, inaccuracy, bug or general comment that could improve that piece of software.


"""



def fillPolygon(x, y, i0, j0, scale, gap, N, index=0):
    """
    From a list of points defined by their 2 coordinates list
    x and y, creates a filled polygon with sides joining the points.
    The polygon is created in an image of size (N, N).
    The origin (x,y)=(0,0) is mapped at pixel i0, j0 (both can be
    floating-point values).
    Arrays x and y are supposed to be in unit U, and scale is the
    pixel size in U units.

    :returns: filled polygon (N, N), boolean
    :param float x, y: list of points defining the polygon
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float scale: size of a pixel of the image, in same unit as x and y.
    :param float N: size of output image.

    :Example:
    x = np.array([1,-1,-1.5,0,1.1])
    y = np.array([1,1.5,-0.2,-2,0])
    N = 200
    i0 = N/2
    j0 = N/2
    gap = 0.
    scale = 0.03
    pol = fillPolygon(x, y, i0, j0, scale, gap, N, index=2)
    
    """
    # define coordinates map centred on (i0,j0) with same units as x,y.
    X = (np.arange(N) - i0)*scale
    Y = (np.arange(N) - j0)*scale
    X,Y = np.meshgrid(X,Y,indexing='ij') # indexage [x,y]
    
    # define centre of polygon x0, y0
    x0 = np.mean(x)
    y0 = np.mean(y)
    
    # compute angles of all pixels coordinates of the map, and all 
    # corners of the polygon
    T = (np.arctan2(Y-y0, X-x0) + 2*np.pi)%(2*np.pi)
    t = (np.arctan2(y-y0, x-x0) + 2*np.pi)%(2*np.pi)
    
    # on va voir dans quel sens ca tourne. Je rajoute ca pour que ca marche
    # quel que soit le sens de rotation des points du polygone.
    # En fait, j'aurais peut etre pu classer les points par leur angle, pour
    # etre sur que ca marche meme si les points sont donnes dans ts les cas
    sens = np.median(np.diff(t))
    if sens<0:
        x = x[::-1]
        y = y[::-1]
        t = t[::-1]
    
    # re-organise order of polygon points so that it starts from
    # angle = 0, or at least closest to 0.
    imin = t.argmin()  # position of the minimum
    if imin != 0:
        x = np.roll(x, -imin)
        y = np.roll(y, -imin)
        t = np.roll(t, -imin)
  
    # For each couple of consecutive corners A, B, of the polygon, one fills
    # the triangle AOB with True.
    # Last triangle has a special treatment because it crosses the axis
    # with theta=0=2pi
    n = x.shape[0]   # number of corners of polygon
    indx, indy = (np.array([],dtype=np.int), np.array([],dtype=np.int))
    distedge = np.array([],dtype=np.float)
    for i in range(n):
        j = i+1  # j=element next i except when i==n : then j=0 (cycling)
        if j==n:
            j = 0
            sub = np.where((T>=t[-1]) | (T<=(t[0])))
        else:
            sub = np.where((T>=t[i]) & (T<=t[j]))
        # compute unitary vector des 2 sommets 
        dy = y[j]-y[i]
        dx = x[j]-x[i]
        vnorm = np.sqrt(dx**2 + dy**2)
        dx /= vnorm
        dy /= vnorm
        # calcul du produit vectoriel
        crossprod = dx*(Y[sub]-y[i]) - dy*(X[sub]-x[i])
        tmp = crossprod > gap
        indx = np.append(indx, sub[0][tmp])
        indy = np.append(indy, sub[1][tmp])
        distedge = np.append(distedge, crossprod[tmp] )
    
    # choice of what is returned : either only the indexes, or the
    # boolean map
    if index==1:
        return (indx, indy, distedge)
    elif index==2:
        a = np.zeros((N,N))
        a[indx, indy] = distedge
        return a
    else:
        a = np.zeros((N,N), dtype=np.bool)
        a[indx, indy] = True        # convention [x,y]
   
    return a


def centrePourVidal(N, i0, j0, centerMark):
    """
    Renvoie une image de boolens (False) de taille (N,N) avec un point
    ou une croix (True) centree sur (i0, j0).
    :param int N: taille de l'image de sortie
    :param float i0, j0: position du marqueur de sortie
    :param int centerMark: 0 (pour rien), 1 (option point) ou 2 (option croix)
    """
    scale = 1.0
    res = 0
    X = (np.arange(N) - i0)*scale
    Y = (np.arange(N) - j0)*scale
    X,Y = np.meshgrid(X,Y,indexing='ij')  # convention d'appel [x,y]
    if centerMark==1:
        res = (X**2+Y**2)<1
    if centerMark==2:
        res = (np.abs(X)<0.9) | (np.abs(Y)<0.9)
    return res


def fillSpider(N, nspider, dspider, i0, j0, scale, rot):
    """
    Creates a boolean spider mask on a map of dimensions (N,N)
    The spider is centred at floating-point coords (i0,j0).
    
    :returns: spider image (boolean)
    :param int N: size of output image
    :param int nspider: number of spiders
    :param float dspider: width of spiders
    :param float i0: coord of spiders symmetry centre
    :param float j0: coord of spiders symmetry centre
    :param float scale: size of a pixel in same unit as dspider
    :param float rot: rotation angle in radians
    
    """
    a = np.ones((N,N), dtype=np.bool)
    X = (np.arange(N) - i0)*scale
    Y = (np.arange(N) - j0)*scale
    X,Y = np.meshgrid(X,Y,indexing='ij')  # convention d'appel [x,y]
    w = 2*np.pi / nspider
    # rot += np.pi/2  # parce que c'est comme ca !!
    for i in range(nspider):
        nn = (abs(X*np.cos(i*w - rot)+Y*np.sin(i*w - rot)) < dspider/2.)
        a[nn] = False
    return a



def generateEeltPupil_slow(npt, dspider, i0, j0, pixscale, rotdegree):
    """
    OBSOLETE FUNCTION --- DO NOT USE
      It is kept here just because it contains the coordinates of the edges
      of the ELT pupil, which are still valid today (march 2020).
    
    Computes the binary EELT pupil on a map of size (npt, npt).
    This is the original function, that builds the pupil shape according to
    hardcoded contours.
    This function is now obsolete, because it's been replaced by the faster
    one generateEeltPupilMask()

    :returns: pupil image (npt, npt), boolean
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float pixscale: size of a pixel of the image, in meters.
    :param float rotdegree: rotation angle of the pupil, in degrees.

    :Example:
    >>> pup = generateEeltPupil_slow(800, 0.6, 400, 400, 0.1, 3.0)

    """
    x = np.array([18.4524,18.798,18.4514,18.796,18.4484,18.7919,18.4433,18.7858,18.4363,18.7776,
         18.4273,17.7349,17.3831,17.7243,17.3717,16.6772,16.3233,16.6645,16.3099,
         15.6135,15.2579,15.5991,15.2429,14.545,14.188,14.5292,14.1718,13.4727,13.1146,
         12.4138,12.0552,11.3528,10.9939,10.2902,9.93103,9.22619,8.86699,8.16118,7.8021,
         7.09552,6.73671,6.02955,5.67117,4.96362,4.60582,3.89808,3.54699,2.83805,
         2.48141,1.77263,1.41934,0.709707,0.354564,-0.354564,-0.709707,-1.41934,
         -1.77263,-2.48141,-2.83805,-3.54699,-3.89808,-4.60582,-4.96362,-5.67117,
         -6.02955,-6.73671,-7.09552,-7.8021,-8.16118,-8.86699,-9.22619,-9.93103,
         -10.2902,-10.9939,-11.3528,-12.0552,-12.4138,-13.1146,-13.4727,-14.1718,
         -14.5292,-14.188,-14.545,-15.2429,-15.5991,-15.2579,-15.6135,-16.3099,-16.6645,
         -16.3233,-16.6772,-17.3717,-17.7243,-17.3831,-17.7349,-18.4273,-18.7776,
         -18.4363,-18.7858,-18.4433,-18.7919,-18.4484,-18.796,-18.4514,-18.798,-18.4524,
         -18.798,-18.4514,-18.796,-18.4484,-18.7919,-18.4433,-18.7858,-18.4363,-18.7776,
         -18.4273,-17.7349,-17.3831,-17.7243,-17.3717,-16.6772,-16.3233,-16.6645,
         -16.3099,-15.6135,-15.2579,-15.5991,-15.2429,-14.545,-14.188,-14.5292,-14.1718,
         -13.4727,-13.1146,-12.4138,-12.0552,-11.3528,-10.9939,-10.2902,-9.93103,
         -9.22619,-8.86699,-8.16118,-7.8021,-7.09552,-6.73671,-6.02955,-5.67117,
         -4.96362,-4.60582,-3.89808,-3.54699,-2.83805,-2.48141,-1.77263,-1.41934,
         -0.709707,-0.354564,0.354564,0.709707,1.41934,1.77263,2.48141,2.83805,3.54699,
         3.89808,4.60582,4.96362,5.67117,6.02955,6.73671,7.09552,7.8021,8.16118,8.86699,
         9.22619,9.93103,10.2902,10.9939,11.3528,12.0552,12.4138,13.1146,13.4727,
         14.1718,14.5292,14.188,14.545,15.2429,15.5991,15.2579,15.6135,16.3099,16.6645,
         16.3233,16.6772,17.3717,17.7243,17.3831,17.7349,18.4273,18.7776,18.4363,
         18.7858,18.4433,18.7919,18.4484,18.796,18.4514,18.798])

    y = np.array([0,0.614323,1.22918,1.84277,2.45796,3.07061,3.68594,4.29746,4.91271,5.5229,
         6.13789,6.14356,6.75902,7.36787,7.98272,7.98968,8.60474,9.21184,9.82596,
         9.83398,10.448,11.053,11.6658,11.6747,12.2871,12.8896,13.5004,13.51,14.1202,
         14.1294,14.7389,14.7478,15.3564,15.3648,15.9724,15.9802,16.5867,16.5939,
         17.1992,17.2057,17.8095,17.8154,18.4177,18.4227,19.0233,19.0275,18.4307,
         18.4337,19.0337,19.0357,18.4377,18.4387,19.0378,19.0378,18.4387,18.4377,
         19.0357,19.0337,18.4337,18.4307,19.0275,19.0233,18.4227,18.4177,17.8154,
         17.8095,17.2057,17.1992,16.5939,16.5867,15.9802,15.9724,15.3648,15.3564,
         14.7478,14.7389,14.1294,14.1202,13.51,13.5004,12.8896,12.2871,11.6747,11.6658,
         11.053,10.448,9.83398,9.82596,9.21184,8.60474,7.98968,7.98272,7.36787,6.75902,
         6.14356,6.13789,5.5229,4.91271,4.29746,3.68594,3.07061,2.45796,1.84277,1.22918,
         0.614323,0,-0.614323,-1.22918,-1.84277,-2.45796,-3.07061,-3.68594,
         -4.29746,-4.91271,-5.5229,-6.13789,-6.14356,-6.75902,-7.36787,-7.98272,
         -7.98968,-8.60474,-9.21184,-9.82596,-9.83398,-10.448,-11.053,-11.6658,-11.6747,
         -12.2871,-12.8896,-13.5004,-13.51,-14.1202,-14.1294,-14.7389,-14.7478,-15.3564,
         -15.3648,-15.9724,-15.9802,-16.5867,-16.5939,-17.1992,-17.2057,-17.8095,
         -17.8154,-18.4177,-18.4227,-19.0233,-19.0275,-18.4307,-18.4337,-19.0337,
         -19.0357,-18.4377,-18.4387,-19.0378,-19.0378,-18.4387,-18.4377,-19.0357,
         -19.0337,-18.4337,-18.4307,-19.0275,-19.0233,-18.4227,-18.4177,-17.8154,
         -17.8095,-17.2057,-17.1992,-16.5939,-16.5867,-15.9802,-15.9724,-15.3648,
         -15.3564,-14.7478,-14.7389,-14.1294,-14.1202,-13.51,-13.5004,-12.8896,-12.2871,
         -11.6747,-11.6658,-11.053,-10.448,-9.83398,-9.82596,-9.21184,-8.60474,-7.98968,
         -7.98272,-7.36787,-6.75902,-6.14356,-6.13789,-5.5229,-4.91271,-4.29746,
         -3.68594,-3.07061,-2.45796,-1.84277,-1.22918,-0.614323])

    # Rotation matrices
    rot = rotdegree * np.pi/180.00
    mrot = np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]])

    # rotation of coordinates of outer segments
    u = mrot.dot([x,y])
    pup = fillPolygon(u[0], u[1], i0, j0, pixscale, 0., npt)

    # INTERNAL CONTOUR OF OBSCURATION
    xx = np.array([5.02571,4.66726,5.02543,4.66673,5.02458,4.66568,3.94877,3.58959,2.87216,
        2.51286,1.7951,1.43584,0.717959,0.358899,-0.358899,-0.717959,-1.43584,-1.7951,
        -2.51286,-2.87216,-3.58959,-3.94877,-4.66568,-5.02458,-4.66673,-5.02543,
        -4.66726,-5.02571,-4.66726,-5.02543,-4.66673,-5.02458,-4.66568,-3.94877,
        -3.58959,-2.87216,-2.51286,-1.7951,-1.43584,-0.717959,-0.358899,0.358899,
        0.717959,1.43584,1.7951,2.51286,2.87216,3.58959,3.94877,4.66568,5.02458,
        4.66673,5.02543,4.66726])

    yy = np.array([0.,0.62184,1.24347,1.86531,2.48652,3.10816,3.10885,3.73041,3.73104,
        4.35239,4.35288,4.97389,4.97416,5.59468,5.59468,4.97416,4.97389,4.35288,
        4.35239,3.73104,3.73041,3.10885,3.10816,2.48652,1.86531,1.24347,0.62184,
        0.0,-0.62184,-1.24347,-1.86531,-2.48652,-3.10816,-3.10885,-3.73041,
        -3.73104,-4.35239,-4.35288,-4.97389,-4.97416,-5.59468,-5.59468,-4.97416,
        -4.97389,-4.35288,-4.35239,-3.73104,-3.73041,-3.10885,-3.10816,-2.48652,
        -1.86531,-1.24347,-0.62184])


    # rotation of coordinates of inner segments (central obs)
    u = mrot.dot([xx,yy])
    pup = pup & ~fillPolygon(u[0], u[1], i0, j0, pixscale, 0, npt)
    

    # SPIDERS ............................................
    nspider = 3  # pour le jour ou on voudra plus de spiders..
    if( dspider>0 and nspider>0 ):
        pup = pup & fillSpider(npt, nspider, dspider, i0, j0, pixscale, rot)
          
  
    return pup




def createHexaPattern(pitch, supportSize):
    """
    Cree une liste de coordonnees qui decrit un maillage hexagonal.
    Retourne un tuple (x,y).
    
    Le maillage est centre sur 0, l'un des points est (0,0).
    Une des pointes de l'hexagone est dirigee selon l'axe Y, au sens ou le
    tuple de sortie est (x,y).
    
    :param float pitch: distance between 2 neighbour points
    :param int supportSize: size of the support that need to be populated
    
    """
    V3 = np.sqrt(3)
    nx = int(np.ceil((supportSize/2.0)/pitch) + 1)
    x = pitch * (np.arange(2*nx+1)-nx)
    ny = int(np.ceil((supportSize/2.0)/pitch/V3) + 1)
    y = (V3*pitch) * (np.arange(2*ny+1)-ny)
    x, y = np.meshgrid(x, y, indexing='ij')
    x = x.flatten()
    y = y.flatten()
    peak_axis = np.append(x, x + pitch/2.)    # axe dirige selon sommet
    flat_axis = np.append(y, y + pitch*V3/2.) # axe dirige selon plat
    return flat_axis, peak_axis




def generateCoordSegments(D, rot):
    """
    Computes the coordinates of the corners of all the hexagonal
    segments of M1.
    Result is a tuple of arrays(6, 798).
    
    :param float D: D is the pupil diameter in meters, it must be set to 40.0 m
    for the nominal EELT.
    :param float rot: pupil rotation angle in radians
    
    """
    V3 = np.sqrt(3)
    pitch = 1.227314    # no correction du bol
    pitch = 1.244683637214  # diametre du cerle INSCRIT
    # diamseg = pitch*2/V3  # diametre du cercle contenant TOUT le segment
    # print("segment diameter : %.6f\n" % diamseg)
    
    # Creation d'un pattern hexa avec pointes selon la variable <ly>
    lx, ly = createHexaPattern(pitch, 35*pitch)
    ll = np.sqrt(lx**2 + ly**2)
    # Elimination des segments non valides grace a 2 nombres parfaitement
    # empiriques ajustes a-la-mano.
    inner_rad, outer_rad = 4.1, 15.4   # nominal, 798 segments 
    nn = (ll>inner_rad*pitch) & (ll<outer_rad*pitch);
    lx = lx[nn]
    ly = ly[nn]
    lx, ly = reorganizeSegmentsOrderESO(lx, ly)
    ll = np.sqrt(lx**2 + ly**2)
    
    # n = ll.shape[0]
    # print("Nbre de segments : %d\n" % n)
    # Creation d'un hexagone-segment avec pointe dirigee vers 
    # variable <hx> (d'ou le cos() sur hx)
    th = np.linspace(0, 2*np.pi, 7)[0:6]
    hx = np.cos(th)*pitch/V3
    hy = np.sin(th)*pitch/V3
    
    # Le maillage qui permet d'empiler des hexagones avec sommets 3h-9h
    # est un maillage hexagonal avec sommets 12h-6h, donc a 90°.
    # C'est pour ca qu'il a fallu croiser les choses avant.
    x = (lx[None,:] + hx[:,None])
    y = (ly[None,:] + hy[:,None])
    r = np.sqrt(x**2+y**2)
    R = 95.7853
    rrc = R / r * np.arctan(r/R)    # correction factor
    x *= rrc
    y *= rrc
    
    nominalD = 40.0   # size of the OFFICIAL E-ELT
    if D!=nominalD:
        x *= D / nominalD
        y *= D / nominalD
    
    # Rotation matrices
    mrot = np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]])

    # rotation of coordinates
    # le tableau [x,y] est de taille (2,6,798). Faut un transpose a la con
    # pour le transformer en (6,2,798) pour pouvoir faire le np.dot
    # correctement. En sortie, xrot est (2,6,798).
    xyrot = np.dot(mrot,np.transpose(np.array([x,y]),(1,0,2)))
    
    return xyrot[0], xyrot[1]


def reorganizeSegmentsOrderESO(x, y):
    """
    Reorganisation des segments facon ESO.
    Voir 
    ESO-193058 Standard Coordinate System and Basic Conventions
    
    :param float x: tableau des centres X des segments
    :param float y: idem Y
    :return tuple (x,y): meme tuple que les arguments d'entree, mais tries.
    
    """
    # pi/2, pi/6, 2.pi, ...
    pi_3 = np.pi/3
    pi_6 = np.pi/6
    pix2 = 2*np.pi
    # calcul des angles
    t = (np.arctan2(y, x) + pi_6 - 1e-3) % (pix2)
    X = np.array([])
    Y = np.array([])
    A = 100.
    for k in range(6):
        sector = (t>k*pi_3) & (t<(k+1)*pi_3)
        u = k * pi_3
        distance = (A*np.cos(u)-np.sin(u))*x[sector] + (np.cos(u)+A*np.sin(u))*y[sector]
        indsort = np.argsort(distance)
        X = np.append(X, x[sector][indsort])
        Y = np.append(Y, y[sector][indsort])
    return X, Y



def getdatatype(truc):
    """
    Returns the data type of a numpy variable, either scalar value or array.
    
    """
    if np.isscalar(truc):
        return type(truc)
    else:
        return type(truc.flatten()[0])


def generateSegmentProperties(attribute, hx, hy, i0, j0, scale, gap, N, D, softGap=0):
    """
    Builds a 2D image of the pupil with some attributes for each of the
    segments. Those segments are described from arguments hx and hy, that
    are produced by the function generateCoordSegments(D, rot).
    
    When attribute is a phase, then it must be a float array of dimension
    [3, 798] with the dimension 3 being piston, tip, and tilt.
    Units of phase is xxx rms, and the output of the procedure will be
    in units of xxx.
    

    :returns: pupil image (N, N), with the same type of input argument attribute
    
    :param float/int/bool attribute: scalar value or 1D-array of the reflectivity of
           the segments or 2D array of phase
           If attribute is scalar, the value will be replicated for all segments.
           If attribute is a 1D array, then it shall contain the reflectivities 
           of all segments.
           If attribute is a 2D array then it shall contain the piston, tip
           and tilt of the segments. The array shall be of dimension
           [3, 798] that contains [piston, tip, tilt]
           On output, the data type of the pupil map will be the same as attribute.
    :param float hx, hy: arrays [6,:] describing the segment shapes. They are
        generated using generateCoordSegments()
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float scale: size of a pixel of the image, in meters.
    :param float gap: half-space between segments in meters
    :param int N: size of the output array (N,N)
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0
    :param bool softGap: if False, the gap between segments is binary 0/1
          depending if the pixel is within the gap or not. If True, the gap
          is a smooth region of a fwhm of 2 pixels with a depth related to the
          gap width.
    

    
    """

    # number of segments
    nseg = hx.shape[-1]
    # If <attribute> is a scalar, then we make a list. It will be required
    # later on to set the attribute to each segment.
    if np.isscalar(attribute):
        attribute = np.array([attribute]*nseg)
        
    # the pupil map is created with the same data type as <attribute>
    pupil = np.zeros((N,N), dtype=getdatatype(attribute))
    
    # average coord of segments
    x0 = np.mean(hx, axis=0)
    y0 = np.mean(hy, axis=0)
    # avg coord of segments in pixel indexes
    x0 = x0/scale + i0
    y0 = y0/scale + j0
    # size of mini-support
    hexrad = 0.75 * D/40. / scale
    ix0 = np.floor(x0-hexrad).astype(int)-1
    iy0 = np.floor(y0-hexrad).astype(int)-1
    segdiam = np.ceil(hexrad*2+1).astype(int)+1
    
    n = attribute.shape[0]
    if n!=3:
        # attribute is a signel value : either reflectivity, or boolean, 
        # or just piston.
        if softGap!=0:
            # Soft gaps
            # The impact of gaps are modelled using a simple function: Lorentz, 1/(1+x**2)
            # The fwhm is always equal to 2 pixels because the gap is supposed
            # to be "small/invisible/undersampled". The only visible thing is
            # the width of the impulse response, chosen 2-pixel wide to be
            # well sampled.
            # The "depth" is related to the gap width. The integral of a Lorentzian
            # of 2 pix wide is PI. Integral of a gap of width 'gap' in pixels is 'gap'.
            # So the depth equals to gap/scale/np.pi.
            for i in range(nseg):
                indx, indy, distedge = fillPolygon(hx[:,i], hy[:,i], i0-ix0[i], j0-iy0[i], scale, gap*0., segdiam, index=1)
                pupil[indx + ix0[i], indy + iy0[i]] = attribute[i] * (1. - (gap/scale/np.pi) / (1+(distedge/scale)**2))
        else:
            # Hard gaps
            for i in range(nseg):
                indx, indy, distedge = fillPolygon(hx[:,i], hy[:,i], i0-ix0[i], j0-iy0[i], scale, gap, segdiam, index=1)
                pupil[indx + ix0[i], indy + iy0[i]] = attribute[i]
    else:
        # attribute is [piston, tip, tilt]
        minimap = np.zeros((segdiam, segdiam))
        xmap = np.arange(segdiam) - segdiam/2
        xmap, ymap = np.meshgrid(xmap,xmap,indexing='ij')     # [x,y] convention
        pitch = 1.244683637214        # diameter of inscribed circle
        diamseg = pitch*2/np.sqrt(3)  # diameter of circumscribed circle 
        diamfrizou = (pitch + diamseg)/2 * D/40.  # average diameter of the 2
        # Calcul du facteur de mise a l'echelle pour l'unite des tilts.
        # xmap et ymap sont calculees avec un increment de +1 pour deux pixels
        # voisins, donc le facteur a appliquer est tel que l'angle se conserve
        # donc factunit*1 / scale = 4*factunit
        factunit = 4*scale/diamfrizou
        for i in range(nseg):     
            indx, indy, _ = fillPolygon(hx[:,i], hy[:,i], i0-ix0[i], j0-iy0[i], scale, 0., segdiam, index=1)
            minimap = attribute[0,i] + (factunit*attribute[1,i])*xmap + (factunit*attribute[2,i])*ymap
            pupil[indx + ix0[i], indy + iy0[i]] = minimap[indx, indy] 
            
    return pupil




"""
      

██╗  ██╗██╗ ██████╗ ██╗  ██╗      ██╗     ███████╗██╗   ██╗███████╗██╗
██║  ██║██║██╔════╝ ██║  ██║      ██║     ██╔════╝██║   ██║██╔════╝██║
███████║██║██║  ███╗███████║█████╗██║     █████╗  ██║   ██║█████╗  ██║
██╔══██║██║██║   ██║██╔══██║╚════╝██║     ██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║
██║  ██║██║╚██████╔╝██║  ██║      ███████╗███████╗ ╚████╔╝ ███████╗███████╗
╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝      ╚══════╝╚══════╝  ╚═══╝  ╚══════╝╚══════╝


"""


def getEeltSegmentNumber():
    """
    Just returns the number of segments of the EELT nominal pupil, in order
    to be able to generate either reflectivities, or phase errors, or else.
    
    """
    hx, hy = generateCoordSegments( 40., 0. )
    n = hx.shape[-1]
    return n


def generateEeltPupilMask(npt, dspider, i0, j0, pixscale, gap, rotdegree, D=40.0, centerMark=0):
    """
    Generates a boolean pupil mask of the binary EELT pupil
    on a map of size (npt, npt).


    :returns: pupil image (npt, npt), boolean
    :param int npt: size of the output array
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float pixscale: size of a pixel of the image, in meters.
    :param float gap: half-space between segments in meters
    :param float rotdegree: rotation angle of the pupil, in degrees.
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0
    :param int centerMark: when centerMark!=0, a pixel is added at the centre of
        symmetry of the pupil in order to debug things using compass.
        centerMark==1 draws a point
        centerMark==2 draws 2 lines

    :Example:
    npt = 752
    i0 = npt/2+0.5
    j0 = npt/2+0.5
    rotdegree = 90.0
    pixscale = 40./npt
    dspider = 0.53
    gap = 0.02
    pup = generateEeltPupilMask(npt, dspider, i0, j0, pixscale, gap, rotdegree)
    
    """
    rot = rotdegree * np.pi/180
    
    # Generation of segments coordinates. 
    # hx and hy have a shape [6,798] describing the 6 vertex of the 798
    # hexagonal mirrors
    hx, hy = generateCoordSegments( D, rot )

    # From the data of hex mirrors, we build the pupil image using
    # boolean
    pup = generateSegmentProperties(True, hx, hy, i0, j0, pixscale, gap, npt, D)
    
    # SPIDERS ............................................
    nspider = 3  # for the day where we have more/less spiders ;-)
    if( dspider>0 and nspider>0 ):
        pup = pup & fillSpider(npt, nspider, dspider, i0, j0, pixscale, rot)
    
    # Rajout d'un pixel au centre (pour marquer le centre) ou d'une croix,
    # selon la valeur de centerMark
    if centerMark:
        pup = np.logical_xor(pup , centrePourVidal(npt, i0, j0, centerMark))

    return pup
    
   
def generateEeltPupilReflectivity(refl, npt, dspider, i0, j0, pixscale, gap, rotdegree, D=40.0, softGap=False):
    """
    Generates a map of the reflectivity of the EELT pupil, on an array
    of size (npt, npt).

    :returns: pupil image (npt, npt), with the same type of input argument refl
    :param float/int/bool refl: scalar value or 1D-array of the reflectivity of
           the segments.
           If refl is scalar, the value will be replicated for all segments.
           If refl is a 1D array, then it shall contain the reflectivities 
           of all segments.
           On output, the data type of the pupil map will be the same as refl.
    :param int npt: size of the output array
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float pixscale: size of a pixel of the image, in meters.
    :param float gap: half-space between segments in meters
    :param float rotdegree: rotation angle of the pupil, in degrees.
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0
    :param bool softGap: if False, the gap between segments is binary 0/1
          depending if the pixel is within the gap or not. If True, the gap
          is a smooth region of a fwhm of 2 pixels with a depth related to the
          gap width.
    

    :Example:
    
    # This is an example with random reflectivities of segments
    refl = np.ones(798)+np.random.randn(798)/20.
    dead = 3
    refl[(np.random.rand(dead)*797).astype(int)] = 0.
    # The generated image will be 752 pixels wide
    npt = 752
    i0 = npt/2+0.5  # the centre will be at the centre of the image
    j0 = npt/2+0.5
    rotdegree = 11.0  # pupil will be rotated by 11 degrees
    pixscale = 40./npt  # this is pixel scale
    dspider = 0.51   # spiders are 51cm wide
    gap = 0.02   # this is the gap between segments
    # GO !!!!.......
    pup = generateEeltPupilReflectivity(refl, npt, dspider, i0, j0, pixscale, gap, rotdegree, softGap=True)
    # plot things
    plt.imshow(pup.T, origin='l')
    
    """
    rot = rotdegree * np.pi/180
    
    # Generation of segments coordinates. 
    # hx and hy have a shape [6,798] describing the 6 vertex of the 798
    # hexagonal mirrors
    hx, hy = generateCoordSegments( D, rot )
    
    # From the data of hex mirrors, we build the pupil image according
    # to the properties defined by input argument <refl>
    pup = generateSegmentProperties(refl, hx, hy, i0, j0, pixscale, gap, npt, D, softGap=softGap)
    
    # SPIDERS ............................................
    nspider = 3  # for the day where we have more/less spiders ;-)
    if( dspider>0 and nspider>0 ):
        pup = pup * fillSpider(npt, nspider, dspider, i0, j0, pixscale, rot)
          
    return pup
    
    
#refl = np.ones(798)+np.random.randn(798)/20.
#dead = 0
#refl[(np.random.rand(dead)*797).astype(int)] = 0.
## The generated image will be 752 pixels wide
#npt = 752
#i0 = npt/2+0.5  # the centre will be at the centre of the image
#j0 = npt/2+0.5
#rotdegree = 0  # pupil will be rotated by 11 degrees
#pixscale = 40./npt  # this is pixel scale
#dspider = 0.51   # spiders are 51cm wide
#gap = 0.02   # this is the gap between segments
## GO !!!!.......
#pup = generateEeltPupilReflectivity(refl, npt, dspider, i0, j0, pixscale, gap, rotdegree, softGap=True)
## plot things
#plt.imshow(pup.T, origin='l')

def generateEeltPupilPhase(phase, npt, dspider, i0, j0, pixscale, rotdegree, D=40.0):
    """
    Generates a map of the segments phase errors of the EELT pupil, on an array
    of size (npt, npt).

    :returns: phase image (npt, npt), with the same type of input argument phase
    :param float phase: scalar value or 2D-array of the piston, tip
           and tilt of the segments. The array shall be of dimension
           [3, 798] that contains [piston, tip, tilt]
    :param int npt: size of the output array
    :param float dspider: width of spiders in meters
    :param float i0, j0: index of pixels where the pupil should be centred.
                         Can be floating-point indexes.
    :param float pixscale: size of a pixel of the image, in meters.
    :param float rotdegree: rotation angle of the pupil, in degrees.
    :param float D: diameter of the pupil. For the nominal EELT, D shall
                    be set to 40.0

    :Example:
        
    phase = np.random.randn(3,798)
    phase = np.zeros((3,798)); phase[1,:]=1.
    npt = 752
    i0 = npt/2+0.5
    j0 = npt/2+0.5
    rotdegree = 90.0
    pixscale = 41./npt
    dspider = 0.51
    pup = generateEeltPupilPhase(phase, npt, dspider, i0, j0, pixscale, rotdegree)
    
    """
    rot = rotdegree * np.pi/180
    
    # Generation of segments coordinates. 
    # hx and hy have a shape [6,798] describing the 6 vertex of the 798
    # hexagonal mirrors
    hx, hy = generateCoordSegments( D, rot )
    
    # From the data of hex mirrors, we build the pupil phase image according
    # to the properties defined by input argument <phase>
    pup = generateSegmentProperties(phase, hx, hy, i0, j0, pixscale, 0.0, npt, D)
    
    return pup
    
    
  