# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:31:33 2020

@author: cheritie
"""
import numpy as np


class Zernike:
    def __init__(self, telObject, J=1):
        self.resolution = telObject.resolution
        self.D = telObject.D
        self.centralObstruction = telObject.centralObstruction
        self.nModes = J

    def zernike_tel(self, tel, j):
        """
         ADAPTED FROM AOTOOLS PACKAGE:https://github.com/AOtools/aotools

         Creates the Zernike polynomial with radial index, n, and azimuthal index, m.

         Args:
            n (int): The radial order of the zernike mode
            m (int): The azimuthal order of the zernike mode
            N (int): The diameter of the zernike more in pixels
         Returns:
            ndarray: The Zernike mode
         """
        X, Y = np.where(tel.pupil > 0)

        X = (X-(tel.resolution + tel.resolution %
             2-1)/2) / tel.resolution * tel.D
        Y = (Y-(tel.resolution + tel.resolution %
             2-1)/2) / tel.resolution * tel.D
        #                                          ^- to properly allign coordinates relative to the (0,0) for even/odd telescope resolutions
        R = np.sqrt(X**2 + Y**2)
        R = R/R.max()
        theta = np.arctan2(Y, X)
        out = np.zeros([tel.pixelArea, j])
        outFullRes = np.zeros([tel.resolution**2, j])

        for i in range(1, j+1):
            n, m = self.zernIndex(i+1)
            if m == 0:
                Z = np.sqrt(n+1) * self.zernikeRadialFunc(n, 0, R)
            else:
                if m > 0:  # j is even
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n,
                                                                  m, R) * np.cos(m * theta)
                else:  # i is odd
                    m = abs(m)
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n,
                                                                  m, R) * np.sin(m * theta)

            Z -= Z.mean()
            Z *= (1/np.std(Z))

            # clip
            out[:, i-1] = Z
            outFullRes[tel.pupilLogical, i-1] = Z

        outFullRes = np.reshape(
            outFullRes, [tel.resolution, tel.resolution, j])
        return out, outFullRes

    def computeZernike(self, telObject2):
        self.modes, self.modesFullRes = self.zernike_tel(
            telObject2, self.nModes)
        # normalize modes

    def modeName(self, index):
        modes_names = [
            'Tip', 'Tilt', 'Defocus', 'Astigmatism (X-shaped)', 'Astigmatism (+-shaped)',
            'Coma vertical', 'Coma horizontal', 'Trefoil vertical', 'Trefoil horizontal',
            'Sphere', 'Secondary astigmatism (X-shaped)', 'Secondary astigmatism (+-shaped)',
            'Quadrofoil vertical', 'Quadrofoil horizontal',
            'Secondary coma horizontal', 'Secondary coma vertical',
            'Secondary trefoil horizontal', 'Secondary trefoil vertical',
            'Pentafoil horizontal', 'Pentafoil vertical'
        ]

        if index < 0:
            return ('Incorrent index!')
        elif index >= len(modes_names):
            return ('Z', index+2)
        else:
            return (modes_names[index])

    def zernikeRadialFunc(self, n, m, r):
        """
        ADAPTED FROM AOTOOLS PACKAGE:https://github.com/AOtools/aotools
        Function to calculate the Zernike radial function

        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the centre the array

        Returns:
            ndarray: The Zernike radial function
        """
        try:
            factorial = np.math.factorial
        except:
            import scipy

            factorial = scipy.special.factorial

        R = np.zeros(r.shape)
        # Can cast the below to "int", n,m are always *both* either even or odd
        for i in range(0, int((n - m) / 2) + 1):

            R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                                            factorial(n - i)) /
                          (factorial(i) *
                              factorial(int(0.5 * (n + m) - i)) *
                              factorial(int(0.5 * (n - m) - i))),
                          dtype='float')
        return R

    def zernIndex(self, j):
        """
        ADAPTED FROM AOTOOLS PACKAGE:https://github.com/AOtools/aotools

        Find the [n,m] list giving the radial order n and azimuthal order
        of the Zernike polynomial of Noll index j.

        Parameters:
            j (int): The Noll index for Zernike polynomials

        Returns:
            list: n, m values
        """
        n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n % 2
        m = int((p+k)/2.)*2 - k

        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1
            m *= s

        return [n, m]
