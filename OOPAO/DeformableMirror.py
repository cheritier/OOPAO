# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie
"""

import sys
import time
import numpy as np
try:
    import cupy as xp
    global_gpu_flag = True
    xp = np  #for now
except ImportError or ModuleNotFoundError:
    xp = np
from joblib import Parallel, delayed
from .MisRegistration import MisRegistration
from .tools.interpolateGeometricalTransformation import interpolate_cube
from .tools.tools import emptyClass, pol2cart, print_, OopaoError, warning
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .tools.displayTools import makeSquareAxes

class DeformableMirror:
    def __init__(self,
                 telescope,
                 nSubap: float,
                 mechCoupling: float = 0.35,
                 coordinates: np.ndarray = None,
                 pitch: float = None,
                 modes: np.ndarray = None,
                 misReg=None,
                 M4_param=None,
                 nJobs: int = 30,
                 nThreads: int = 20,
                 print_dm_properties: bool = True,
                 floating_precision: int = 64,
                 altitude: float = None,
                 flip=False,
                 flip_lr=False,
                 sign=1):
        """DEFORMABLE MIRROR
        A Deformable Mirror object consists in defining the 2D maps of influence functions of the actuators.
        By default, the actuator grid is cartesian in a Fried Geometry with respect to the nSubap parameter.
        The Deformable Mirror is considered to to be in a pupil plane.
        By default, the influence functions are 2D Gaussian functions normalized to 1 [m].
        IMPORTANT: The deformable mirror is considered to be transmissive instead of reflective.
        This is to prevent any confusion with an eventual factor 2 in OPD due to the reflection.

        Parameters
        ----------
        telescope : Telescope
            the telescope object associated.
            In case no coordinates are provided, the selection of the valid actuator is based on the radius
            of the telescope (assumed to be circular) and the central obstruction value (assumed to be circular).
            The telescope spiders are not considered in the selection of the valid actuators.
            For more complex selection of actuators, specify the coordinates of the actuators using
            the optional parameter "coordinates" (see below).
        nSubap : float
            This parameter is used when no user-defined coordinates / modes are specified. This is used to compute
            the DM actuator influence functions in a fried geometry with respect to nSubap subapertures along the telescope
            diameter.
            If the optional parameter "pitch" is not specified, the Deformable Mirror pitch property is computed
            as the ratio of the Telescope Diameter with the number of subaperture nSubap.
            This impacts how the DM influence functions mechanical coupling is computed. .
        mechCoupling : float, optional
            This parameter defines the mechanical coupling between the influence functions.
            A value of 0.35 which means that if an actuator is pushed to an arbitrary value 1,
            the mechanical deformation at a circular distance of "pitch" from this actuator is equal to 0.35.
            By default, "pitch" is the inter-actuator distance when the Fried Geometry is considered.
            If the parameter "modes" is used, this parameter is ignored. The default is 0.35.
        coordinates : np.ndarray, optional
            User defined coordinates for the DM actuators. Be careful to specify the pitch parameter associated,
            otherwise the pitch is computed using its default value (see pitch parameter).
            If this parameter is specified, all the actuators are considered to be valid
            (no selection based on the telescope pupil).
            The default is None.
        pitch : float, optional
            pitch considered to compute the Gaussian Influence Functions, associated to the mechanical coupling.
            If no pitch is specified, the pitch is computed to match a Fried geometry according to the nSubap parameter.
            The default is None.
        modes : np.ndarray, optional
            User defined influence functions or modes (modal DM) can be input to the Deformable Mirror.
            They must match the telescope resolution and be input as a 2D matrix, where the 2D maps are
            reshaped as a 1D vector of size n_pix*n_pix : size = [n_pix**2,n_modes].
            The default is None.
        misReg : TYPE, optional
            A Mis-Registration object (See the Mis-Registration class) can be input to apply some geometrical transformations
            to the Deformable Mirror. When using user-defined influence functions, this parameter is ignored.
            Consider to use the function applyMisRegistration in OOPAO/mis_registration_identification_algorithm/ to perform interpolations.
            The default is None.
        M4_param : Parameter File, optional
            Parameter File for M4 computation. The default is None.
        nJobs : int, optional
            Number of jobs for the joblib multi-threading. The default is 30.
        nThreads : int, optional
            Number of threads for the joblib multi-threading. The default is 20.
        print_dm_properties : bool, optional
            Boolean to print the dm properties. The default is True.
        floating_precision : int, optional
            If set to 32, uses float32 precision to save memory. The default is 64.
        altitude : float, optional
            Altitude to which the DM is conjugated. The default is None and corresponds to a DM conjugated to the ground.

        Returns
        -------
        None.

        ************************** MAIN PROPERTIES **************************

        The main properties of a Deformable Mirror object are listed here:
        _ dm.coefs             : dm coefficients in units of dm.modes, if using the defauly gaussian influence functions, in [m].
        _ dm.OPD               : the 2D map of the optical path difference in [m]
        _ dm.modes             : matrix of size: [n_pix**2,n_modes]. 2D maps of the dm influence functions (or modes for a modal dm) where the 2D maps are reshaped as a 1D vector of size n_pix*n_pix.
        _ dm.nValidAct         : number of valid actuators
        _ dm.nAct              : Total number of actuator along the diameter (valid only for the default case using cartesian fried geometry).
                                 Otherwise nAct = dm.nValidAct.
        _ dm.coordinates       : coordinates in [m] of the dm actuators (should be input as a 2D array of dimension [nAct,2])
        _ dm.pitch             : pitch used to compute the gaussian influence functions
        _ dm.misReg            : MisRegistration object associated to the dm object

        The main properties of the object can be displayed using :
            dm.print_properties()

        ************************** PROPAGATING THE LIGHT THROUGH THE DEFORMABLE MIRROR **************************
        The light can be propagated from a telescope tel through the Deformable Mirror dm using:
            tel*dm
        Two situations are possible:
            * Free-space propagation: The telescope is not paired to an atmosphere object (tel.isPaired = False).
                In that case tel.OPD is overwritten by dm.OPD: tel.OPD = dm.OPD

            * Propagation through the atmosphere: The telescope is paired to an atmosphere object (tel.isPaired = True).
                In that case tel.OPD is summed with dm.OPD: tel.OPD = tel.OPD + dm.OPD

        ************************** CHANGING THE OPD OF THE MIRROR **************************

        * The dm.OPD can be reseted to 0 by setting the dm.coefs property to 0:
            dm.coefs = 0

        * The dm.OPD can be updated by setting the dm.coefs property using a 1D vector vector_command of length dm.nValidAct:

            dm.coefs = vector_command

        The resulting OPD is a 2D map obtained computing the matricial product dm.modes@dm.coefs and reshaped in 2D.

        * It is possible to compute a cube of 2D OPD using a 2D matrix, matrix_command of size [dm.nValidAct, n_opd]:

            dm.coefs = matrix_command

        The resulting OPD is a 3D map [n_pix,n_pix,n_opd] obtained computing the matricial product dm.modes@dm.coefs and reshaped in 2D.
        This can be useful to parallelize the measurements, typically when measuring interaction matrices. This is compatible with tel*dm operation.

        WARNING: At the moment, setting the value of a single (or subset) actuator will not update the dm.OPD property if done like this:
            dm.coefs[given_index] = value
        It requires to re-assign dm.coefs to itself so the change can be detected using:
            dm.coefs = dm.coefs


        ************************** EXEMPLE **************************

        1) Create an 8-m diameter circular telescope with a central obstruction of 15% and the pupil sampled with 100 pixels along the diameter.
        tel = Telescope(resolution = 100, diameter = 8, centralObstruction = 0.15)

        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8)

        3) Create a Deformable Mirror object with 21 actuators along the diameters (20 in the pupil) and influence functions with a coupling of 45 %.
        dm = DeformableMirror(telescope = tel, nSubap = 20, mechCoupling = 0.45)

        4) Assign a random vector for the coefficients and propagate the light
        dm. coefs = numpy.random.randn(dm.nValidAct)
        src*tel*dm

        5) To visualize the influence function as seen by the telescope:
        dm. coefs = numpy.eye(dm.nValidAct)
        src*tel*dm

        tel.OPD contains a cube of 2D maps for each actuator

        """
        OOPAO_path = [s for s in sys.path if "OOPAO" in s]
        l = []
        for i in OOPAO_path:
            l.append(len(i))
        path = OOPAO_path[np.argmin(l)]
        precision = np.load(path+'/precision_oopao.npy')
        if precision ==64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128
        self.print_dm_properties = print_dm_properties
        self.floating_precision = floating_precision
        self.flip_ = flip
        self.flip_lr = flip_lr
        self.sign = sign
        self.M4_param = M4_param
        self.rad2arcsec = (180./np.pi)*3600
        if M4_param is not None:
            if M4_param['isM4']:
                from .M4_model.make_M4_influenceFunctions import makeM4influenceFunctions

                print_('Building the set of influence functions of M4...',
                       print_dm_properties)
                # generate the M4 influence functions

                pup = telescope.pupil
                filename = M4_param['m4_filename']
                nAct = M4_param['nActuator']

                a = time.time()
                # compute M4 influence functions
                try:
                    coordinates_M4 = makeM4influenceFunctions(pup=pup,
                                                              filename=filename,
                                                              misReg=misReg,
                                                              dm=self,
                                                              nAct=nAct,
                                                              nJobs=nJobs,
                                                              nThreads=nThreads,
                                                              order=M4_param['order_m4_interpolation'],
                                                              floating_precision=floating_precision)
                except:
                    coordinates_M4 = makeM4influenceFunctions(pup=pup,
                                                              filename=filename,
                                                              misReg=misReg,
                                                              dm=self,
                                                              nAct=nAct,
                                                              nJobs=nJobs,
                                                              nThreads=nThreads,
                                                              floating_precision=floating_precision)

    #            selection of the valid M4 actuators
                if M4_param['validActCriteria'] != 0:
                    IF_STD = np.std(np.squeeze(
                        self.modes[telescope.pupilLogical, :]), axis=0)
                    ACTXPC = np.where(IF_STD >= np.mean(
                        IF_STD)*M4_param['validActCriteria'])
                    self.modes = self.modes[:, ACTXPC[0]]

                    coordinates = coordinates_M4[ACTXPC[0], :]
                else:
                    coordinates = coordinates_M4
                # normalize coordinates
                coordinates = (coordinates/telescope.resolution - 0.5)*40
                self.M4_param = M4_param
                self.isM4 = True
                print_('Done!', print_dm_properties)
                b = time.time()

                print_('Done! M4 influence functions computed in ' +
                       str(b-a) + ' s!', print_dm_properties)
            else:
                self.isM4 = False
        else:
            self.isM4 = False
        self.telescope = telescope
        self.altitude = altitude
        if mechCoupling <= 0:
            raise OopaoError('The value of mechanical coupling should be positive.')
        if altitude is None:
            # Resolution of the DM influence Functions
            self.resolution = telescope.resolution
            self.mechCoupling = mechCoupling
            self.tag = 'deformableMirror'
            self.D = telescope.D
        else:
            if telescope.src.tag == 'asterism':
                self.oversampling_factor = np.max((np.asarray(self.telescope.src.coordinates)[
                                                  :, 0]/(self.telescope.resolution/2)))
            else:
                self.oversampling_factor = self.telescope.src.coordinates[0]/(
                    self.telescope.resolution/2)
            self.altitude_layer = self.buildLayer(self.telescope, altitude)
            # Resolution of the DM influence Functions
            self.resolution = self.altitude_layer.resolution
            self.mechCoupling = mechCoupling
            self.tag = 'deformableMirror'
            self.D = self.altitude_layer.D

        # case with no pitch specified (Cartesian geometry)
        if pitch is None:
            # size of a subaperture
            self.pitch = self.D/(nSubap)
        else:
            self.pitch = pitch

        if misReg is None:
            # create a MisReg object to store the different mis-registration
            self.misReg = MisRegistration()
        else:
            self.misReg = misReg

        # If no coordinates are given, the DM is in a Cartesian Geometry
        if coordinates is None:
            print_(
                'No coordinates loaded.. taking the cartesian geometry as a default', print_dm_properties)
            # In that case corresponds to the number of actuator along the diameter
            self.nAct = nSubap+1
            self.nActAlongDiameter = self.nAct-1

            # set the coordinates of the DM object to produce a cartesian geometry
            x = np.linspace(-(self.D)/2, (self.D)/2, self.nAct)
            X, Y = np.meshgrid(x, x)

            # compute the initial set of coordinates
            self.xIF0 = np.reshape(X, [self.nAct**2])
            self.yIF0 = np.reshape(Y, [self.nAct**2])

            # select valid actuators (central and outer obstruction)
            r = np.sqrt(self.xIF0**2 + self.yIF0**2)
            validActInner = r > (
                telescope.centralObstruction*self.D/2-0.5*self.pitch)
            validActOuter = r <= (self.D/2+0.7533*self.pitch)

            self.validAct = validActInner*validActOuter
            self.nValidAct = sum(self.validAct)

        # If the coordinates are specified

        else:
            if np.shape(coordinates)[1] != 2:
                raise OopaoError('Wrong size for the DM coordinates, the (x,y) coordinates should be input as a 2D array of dimension [nAct,2]')

            print_('Coordinates loaded...', print_dm_properties)

            self.xIF0 = coordinates[:, 0]
            self.yIF0 = coordinates[:, 1]
            # In that case corresponds to the total number of actuators
            self.nAct = len(self.xIF0)
            self.nActAlongDiameter = (self.D)/self.pitch

            # In that case assumed that all the Influence Functions provided are controlled actuators
            validAct = (np.arange(0, self.nAct))

            self.validAct = validAct.astype(int)
            self.nValidAct = self.nAct

        #  initial coordinates
        xIF0 = self.xIF0[self.validAct]
        yIF0 = self.yIF0[self.validAct]

        # anamorphosis
        xIF3, yIF3 = self.anamorphosis(xIF0, yIF0, self.misReg.anamorphosisAngle *
                                       np.pi/180, self.misReg.tangentialScaling, self.misReg.radialScaling)

        # rotation
        xIF4, yIF4 = self.rotateDM(
            xIF3, yIF3, self.misReg.rotationAngle*np.pi/180)

        # shifts
        xIF = xIF4-self.misReg.shiftX
        yIF = yIF4-self.misReg.shiftY

        self.xIF = xIF
        self.yIF = yIF

        # corresponding coordinates on the pixel grid
        u0x = self.resolution/2+xIF*self.resolution/self.D
        u0y = self.resolution/2+yIF*self.resolution/self.D
        self.nIF = len(xIF)
        # store the coordinates
        self.coordinates = np.zeros([self.nIF, 2])
        self.coordinates[:, 0] = xIF
        self.coordinates[:, 1] = yIF

        if self.isM4 is False:
            print_('Generating a Deformable Mirror: ', print_dm_properties)
            if np.ndim(modes) == 0:
                print_('Computing the 2D zonal modes...', print_dm_properties)

                def joblib_construction():
                    Q = Parallel(n_jobs=8, prefer='threads')(
                        delayed(self.modesComputation)(i, j) for i, j in zip(u0x, u0y))
                    return Q
                self.modes = np.squeeze(np.moveaxis(
                    np.asarray(joblib_construction()), 0, -1))

            else:
                print_('Loading the 2D zonal modes...', print_dm_properties)
                self.modes = modes
                self.nValidAct = self.modes.shape[1]
                print_('Done!', print_dm_properties)

        else:
            print_('Using M4 Influence Functions', print_dm_properties)
        if floating_precision == 32:
            self.coefs = np.zeros(self.nValidAct, dtype=np.float32)
        else:
            self.coefs = np.zeros(self.nValidAct, dtype=self.precision())
        self.current_coefs = self.coefs.copy()
        if self.print_dm_properties:
            print(self)

    def buildLayer(self, telescope, altitude):

        # initialize layer object
        layer = emptyClass()
        # create a random state to allow reproductible sequences of phase screens
        # gather properties of the atmosphere
        layer.altitude = altitude
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D_fov = telescope.D+2*xp.tan(telescope.fov_rad/2)*layer.altitude
        layer.resolution_fov = int(
            xp.ceil((telescope.resolution/telescope.D)*layer.D_fov))
        # 4 pixels are added as a margin for the edges
        layer.resolution = layer.resolution_fov + 4
        layer.D = layer.resolution * telescope.D / telescope.resolution
        layer.center = layer.resolution//2

        if telescope.src.tag == 'source':
            [x_z, y_z] = pol2cart(layer.altitude*xp.tan(telescope.src.coordinates[0]/self.rad2arcsec)
                                  * layer.resolution / layer.D, xp.deg2rad(telescope.src.coordinates[1]))
            center_x = int(y_z)+layer.resolution//2
            center_y = int(x_z)+layer.resolution//2
            layer.pupil_footprint = xp.zeros([layer.resolution, layer.resolution], dtype=self.precision())
            layer.pupil_footprint[center_x-telescope.resolution//2:center_x+telescope.resolution //
                                  2, center_y-telescope.resolution//2:center_y+telescope.resolution//2] = 1
        else:
            layer.pupil_footprint = []
            layer.extra_sx = []
            layer.extra_sy = []
            layer.center_x = []
            layer.center_y = []
            for i in range(telescope.src.n_source):
                [x_z, y_z] = pol2cart(layer.altitude*xp.tan(telescope.src.coordinates[i][0]/self.rad2arcsec)
                                      * layer.resolution / layer.D, xp.deg2rad(telescope.src.coordinates[i][1]))
                layer.extra_sx.append(int(x_z)-x_z)
                layer.extra_sy.append(int(y_z)-y_z)
                center_x = int(y_z)+layer.resolution//2
                center_y = int(x_z)+layer.resolution//2

                pupil_footprint = xp.zeros([layer.resolution, layer.resolution], dtype=self.precision())
                pupil_footprint[center_x-telescope.resolution//2:center_x+telescope.resolution //
                                2, center_y-telescope.resolution//2:center_y+telescope.resolution//2] = 1
                layer.pupil_footprint.append(pupil_footprint)
                layer.center_x.append(center_x)
                layer.center_y.append(center_y)
        return layer

    def set_pupil_footprint(self, src):
        [x_z, y_z] = pol2cart(self.altitude_layer.altitude*xp.tan((src.coordinates[0]/self.rad2arcsec))
                              * self.altitude_layer.resolution / self.altitude_layer.D, xp.deg2rad(src.coordinates[1]))
        self.altitude_layer.extra_sx = int(x_z)-x_z
        self.altitude_layer.extra_sy = int(y_z)-y_z

        center_x = int(y_z)+self.altitude_layer.resolution//2
        center_y = int(x_z)+self.altitude_layer.resolution//2

        self.altitude_layer.pupil_footprint = xp.zeros(
            [self.altitude_layer.resolution, self.altitude_layer.resolution], dtype=self.precision())
        self.altitude_layer.pupil_footprint[center_x-self.telescope.resolution//2:center_x+self.telescope.resolution //
                                            2, center_y-self.telescope.resolution//2:center_y+self.telescope.resolution//2] = 1

    def get_OPD_altitude(self, src):
        self.set_pupil_footprint(src)
        if np.ndim(self.OPD) == 2:
            OPD = np.reshape(self.OPD[np.where(self.altitude_layer.pupil_footprint == 1)], [
                             self.telescope.resolution, self.telescope.resolution])
        else:
            OPD = np.reshape(self.OPD[self.altitude_layer.center_x-self.telescope.resolution//2:self.altitude_layer.center_x+self.telescope.resolution//2, self.altitude_layer.center_y -
                             self.telescope.resolution//2:self.altitude_layer.center_y+self.telescope.resolution//2, :], [self.telescope.resolution, self.telescope.resolution, self.OPD.shape[2]])
        if np.isinf(src.altitude) is not True:
            if np.ndim(self.OPD) == 2:
                sub_im = np.atleast_3d(OPD)
            else:
                sub_im = np.moveaxis(OPD, 2, 0)
            h = src.altitude - self.altitude_layer.altitude
            if np.isinf(h):
                magnification_cone_effect = 1
            else:
                magnification_cone_effect = h/src.altitude
            cube_in = sub_im.T
            pixel_size_in = 1
            pixel_size_out = pixel_size_in*magnification_cone_effect
            resolution_out = self.telescope.resolution

            OPD = np.asarray(np.squeeze(interpolate_cube(
                cube_in, pixel_size_in, pixel_size_out, resolution_out)).T)

        return OPD

    def dm_propagation(self, telescope, OPD_in=None, src=None):
        if self.coefs.all() == self.current_coefs.all():
            self.coefs = self.coefs
        if OPD_in is None:
            OPD_in = telescope.OPD_no_pupil
        if np.ndim(OPD_in) == 3:
            telescope.resetOPD()
            OPD_in = telescope.OPD_no_pupil
            # warning('Multiple wave-front were already propagated at the telescope level. The telescope OPD is reset to a single flat wave-front.')
        if self.altitude is not None:
            dm_OPD = self.get_OPD_altitude(src)
        else:
            dm_OPD = self.OPD
        # case where the telescope is paired to an atmosphere
        if telescope.isPetalFree:
            telescope.removePetalling()
        # case with single OPD
        if np.ndim(self.OPD) == 2:
            OPD_out_no_pupil = OPD_in*telescope.isPaired + dm_OPD
        # case with multiple OPD
        if np.ndim(self.OPD) == 3:
            OPD_out_no_pupil = np.tile(
                OPD_in[..., None], (1, 1, self.OPD.shape[2]))*telescope.isPaired+dm_OPD

        return OPD_out_no_pupil

    def rotateDM(self, x, y, angle):
        xOut = x*np.cos(angle)-y*np.sin(angle)
        yOut = y*np.cos(angle)+x*np.sin(angle)
        return xOut, yOut

    def anamorphosis(self, x, y, angle, mRad, mNorm):

        mRad += 1
        mNorm += 1
        xOut = x * (mRad*np.cos(angle)**2 + mNorm * np.sin(angle)**2) + \
            y * (mNorm*np.sin(2*angle)/2 - mRad*np.sin(2*angle)/2)
        yOut = y * (mRad*np.sin(angle)**2 + mNorm * np.cos(angle)**2) + \
            x * (mNorm*np.sin(2*angle)/2 - mRad*np.sin(2*angle)/2)

        return xOut, yOut

    def modesComputation(self, i, j):
        x0 = i
        y0 = j
        cx = (1+self.misReg.radialScaling)*(self.resolution /
                                            self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))
        cy = (1+self.misReg.tangentialScaling)*(self.resolution /
                                                self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))

#                    Radial direction of the anamorphosis
        theta = self.misReg.anamorphosisAngle*np.pi/180
        x = np.linspace(0, 1, self.resolution)*self.resolution
        X, Y = np.meshgrid(x, x)

#                Compute the 2D Gaussian coefficients
        a = np.cos(theta)**2/(2*cx**2) + np.sin(theta)**2/(2*cy**2)
        b = -np.sin(2*theta)/(4*cx**2) + np.sin(2*theta)/(4*cy**2)
        c = np.sin(theta)**2/(2*cx**2) + np.cos(theta)**2/(2*cy**2)

        G = self.sign * \
            np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))

        if self.flip_lr:
            G = np.fliplr(G)

        if self.flip_:
            G = np.flip(G)

        output = np.reshape(G, [1, self.resolution**2])
        if self.floating_precision == 32:
            output = np.float32(output)

        return output

    def display_dm(self, fig_index=None, list_src=None, input_opd=None):
        if list_src is None:
            if self.telescope.src.tag == 'asterism':
                list_src = self.telescope.src.src
            else:
                list_src = [self.telescope.src]
        plt.figure(fig_index, figsize=[6, 6], edgecolor=None)
        gs = gridspec.GridSpec(1, 1,
                               height_ratios=[1],
                               width_ratios=[1],
                               hspace=0.5,
                               wspace=0.5)
        ax = plt.subplot(gs[0, 0])
        if input_opd is None:
            input_opd = np.reshape(np.sum(self.modes**5, axis=1), [self.resolution, self.resolution])
        ax.imshow(input_opd,extent=[-self.D/2, self.D/2, -self.D/2, self.D/2])
        center = self.D/2
        [x_tel, y_tel] = pol2cart(self.D/2, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
        cm = plt.get_cmap('gist_rainbow')
        col = []
        for i_source in range(len(list_src)):
            col.append(cm(1.*i_source/len(list_src)))
            [x_c, y_c] = pol2cart(self.telescope.D/2, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
            if self.altitude is None:
                h = list_src[i_source].altitude
            else:
                h = list_src[i_source].altitude-self.altitude
            if xp.isinf(h):
                r = self.telescope.D/2
            else:
                r = (h/self.telescope.src.altitude)*self.telescope.D/2
            [x_cone, y_cone] = pol2cart(r, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
            print(self.telescope.src.altitude)
            if self.altitude is None:
                [x_z, y_z] = [0, 0]
            else:
                [x_z, y_z] = pol2cart(self.altitude*xp.tan((list_src[i_source].coordinates[0])/self.rad2arcsec), xp.deg2rad(list_src[i_source].coordinates[1]))
            center = 0
            [x_c, y_c] = pol2cart(self.D/2, xp.linspace(0, 2*xp.pi, 100, endpoint=True))
            nm = (list_src[i_source].type) + '@' + \
                str(list_src[i_source].coordinates[0])+'"'
            ax.plot(x_cone+x_z+center, y_cone+y_z+center,
                    '-', color=col[i_source], label=nm)
            ax.fill(x_cone+x_z+center, y_cone+y_z+center,
                    y_z+center, alpha=0.1, color=col[i_source])
        ax.set_xlabel('[m]')
        ax.set_ylabel('[m]')
        ax.set_title('Altitude '+str(self.altitude)+' m')
        ax.plot(x_tel+center, y_tel+center, '--', color='k')
        ax.legend(loc='upper left')
        makeSquareAxes(plt.gca())
        return
    
    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, val):
        if self.floating_precision == 32:
            self._coefs = np.float32(val)
        else:
            self._coefs = val

        if np.isscalar(val):
            if val == 0:
                self._coefs = np.zeros(self.nValidAct, dtype=self.precision())
                try:
                    self.OPD = self.precision(np.reshape(np.matmul(self.modes, self._coefs), [
                                              self.resolution, self.resolution]))
                except:
                    self.OPD = self.precision(np.reshape(
                        self.modes@self._coefs, [self.resolution, self.resolution]))

            else:
                print('Error: wrong value for the coefficients')
        else:
            if len(val) == self.nValidAct:
                if np.ndim(val) == 1:  # case of a single mode at a time
                    try:
                        self.OPD = self.precision(np.reshape(np.matmul(self.modes, self._coefs), [
                                                  self.resolution, self.resolution]))
                    except:
                        self.OPD = self.precision(np.reshape(
                            self.modes@self._coefs, [self.resolution, self.resolution]))
                else:                # case of multiple modes at a time
                    try:
                        self.OPD = self.precision(np.reshape(np.matmul(self.modes, self._coefs), [
                                                  self.resolution, self.resolution, val.shape[1]]))
                    except:
                        self.OPD = self.precision(np.reshape(
                            self.modes@self._coefs, [self.resolution, self.resolution, val.shape[1]]))

            else:
                print('Error: wrong value for the coefficients')
                sys.exit(0)
            self.current_coefs = self.coefs.copy()

    # for backward compatibility
    def print_properties(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['controlled_act'] = f"{'Controlled Actuators':<25s}|{self.nValidAct:^9.0f}"
        self.prop['is_m4'] = f"{'M4':<25s}|{str(self.isM4):^9s}"
        self.prop['pitch'] = f"{'Pitch [m]':<25s}|{self.pitch:^9.2f}"
        self.prop['mechanical_coupling'] = f"{'Mechnical coupling [%]':<25s}|{self.mechCoupling*100:^9.0f}"
        self.prop['delimiter'] = ''
        self.prop.update(self.misReg.prop)
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        self.prop['delimiter'] = f'{"== Misregistration ":=<{n_char}}'
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" Deformable mirror ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table
