# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:32:10 2020

@author: cheritie
"""

import copy
import sys
import time
import numpy as np
import skimage.transform as sk
from .runtime import array_backend, gpu_resident, precision_bits
xp, global_gpu_flag = array_backend()
from .runtime import backend_of as _backend_of, to_backend as _to_backend
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
                 sign=1,
                 actuator_selection=None,
                 user_defined_influence_functions_tag=None):
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
            To generate a mis-registered copy of an existing DM (any kind of influence functions), use dm.apply_mis_registration.
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
        actuator_selection : optional
            Selection of the valid actuator:
                - if actuator_selection is a scalar => selected based on the standard deviation of their influence function within the pupil.
                - if actuator_selection is a vector of length two [r_inner,r_outer] => the actuators are selected based on their coordinates r to be part of the r_inner<r<r_outer.
                - if None (default), r_inner is computed based on the telescope central obstruction and r_outer on the telescope diameter.
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
        precision = precision_bits()
        if precision == 64:
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
        self.actuator_selection = actuator_selection
        self.defaut_configuration_flag = False
        self.user_defined_influence_functions_tag = user_defined_influence_functions_tag
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

    # selection of the valid M4 actuators
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
                self.oversampling_factor = np.max((np.asarray(self.telescope.src.coordinates)[:, 0]/(self.telescope.resolution/2)))
            else:
                self.oversampling_factor = self.telescope.src.coordinates[0]/(self.telescope.resolution/2)
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
            self.defaut_configuration_flag = True

            # set the coordinates of the DM object to produce a cartesian geometry
            x = np.linspace(-(self.D)/2, (self.D)/2, self.nAct)
            X, Y = np.meshgrid(x, x)

            # compute the initial set of coordinates
            self.xIF0 = np.reshape(X, [self.nAct**2])
            self.yIF0 = np.reshape(Y, [self.nAct**2])

            #  valid actuators selection
            r = np.sqrt(self.xIF0**2 + self.yIF0**2)

            # default actuator selection
            if self.actuator_selection is None:
                r_in = (telescope.centralObstruction*telescope.initial_D/2-0.5*self.pitch)
                r_out = (telescope.initial_D/2+0.7533*self.pitch)
                self.actuator_selection = [r_in, r_out]
            if np.isscalar(self.actuator_selection):
                # Selection based on energy of the actuators in the pupil (after IF computation)
                self.validAct = np.ones_like(r, dtype=bool)
            else:
                if len(self.actuator_selection) != 2:
                    raise OopaoError('actuator_selection must be either a scalar or a vector of length 2 to specify the inner and outer radius.')
                # Selection based on inner and outer radius
                validActInner = r >= self.actuator_selection[0]
                validActOuter = r <= self.actuator_selection[1]
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

        #  initial coordinates (no mis-registration applied)
        self.initial_coordinates = np.stack([self.xIF0[self.validAct], self.yIF0[self.validAct]], axis=1).astype(float)
        self.nIF = self.initial_coordinates.shape[0]
        # mis-registered coordinates (xIF, yIF, coordinates) and their position on the pixel grid
        u0x, u0y = self._set_mis_registered_coordinates()
        # user-defined modes (without InfluenceFunctions), used by apply_mis_registration
        self._reference_modes = None
        self._reference_mis_registration = None
        if self.isM4 is False:
            print_('Generating a Deformable Mirror: ', print_dm_properties)
            if modes is None:
                print_('Computing the 2D zonal modes...', print_dm_properties)
                self._compute_gaussian_modes(u0x, u0y)
                if np.isscalar(self.actuator_selection):
                    print_('Filtering valid actuators based on pupil influence functions std...', print_dm_properties)
                    # Calculate std only within the illuminated pupil
                    IF_STD = np.std(np.squeeze(self.modes[np.where(telescope.pupil.flatten() == 1), :]), axis=0)
                    self.validAct = np.where(IF_STD > self.actuator_selection * np.max(IF_STD))[0]

                    # Update properties
                    self.modes = self.modes[:, self.validAct]
                    self.nValidAct = len(self.validAct)
                    self.nIF = self.nValidAct

                    # Update coordinates (the initial ones too, so that the selection is kept
                    # when a mis-registration is applied)
                    self.initial_coordinates = self.initial_coordinates[self.validAct, :]
                    self.xIF = self.xIF[self.validAct]
                    self.yIF = self.yIF[self.validAct]
                    self.coordinates = self.coordinates[self.validAct, :]

            else:
                print_('Loading the 2D zonal modes...', print_dm_properties)
                if hasattr(modes, 'influence_function_2D'):
                    # InfluenceFunctions object: can be recomputed for any mis-registration
                    self.modes = modes.influence_function_2D
                    self.name_system = modes.name_system
                    self.flip_lr = modes.flip_lr
                    self.flip_ud = modes.flip_ud
                    self.loc = modes.loc
                    self.sign = modes.sign
                    self.specific_parameters = modes.specific_parameters
                else:
                    self.modes = modes * self.sign
                    # the input modes are taken as the DM at self.misReg: other mis-registrations
                    # are obtained by interpolation (see apply_mis_registration)
                    self._reference_modes = self.modes
                    self._reference_mis_registration = MisRegistration(self.misReg)
                self.nValidAct = self.modes.shape[1]
                print_('Done!', print_dm_properties)

        else:
            print_('Using M4 Influence Functions', print_dm_properties)
        self.gpu_available = global_gpu_flag
        self.gpu_resident = gpu_resident()
        self._gpu_modes = None
        self._gpu_modes_source = None
        if floating_precision == 32:
            self.coefs = np.zeros(self.nValidAct, dtype=np.float32)
        else:
            self.coefs = np.zeros(self.nValidAct, dtype=self.precision())
        self.current_coefs = self.coefs.copy()
        if self.print_dm_properties:
            print(self)

    def relay(self, src):
        if src.tag == 'source':
            self.src_list = [src]
        elif src.tag == 'asterism':
            self.src_list = src.src
        if self.altitude is not None:
            self.set_pupil_footprint()

        for src in self.src_list:
            src.optical_path.append([self.tag, self])

            if np.ndim(src.OPD_no_pupil) > 2:
                src.OPD_no_pupil = _backend_of(src.OPD_no_pupil).zeros(
                    [self.resolution, self.resolution])

            if self.altitude is not None:
                dm_OPD = self.get_OPD_altitude(src)
            else:
                dm_OPD = self.OPD

            if np.ndim(self.OPD) == 2:
                # the DM shape is added on the backend of the source (it is moved there if needed)
                src.OPD_no_pupil += _to_backend(dm_OPD, _backend_of(src.OPD_no_pupil))
            else:
                # case with multiple OPD (resets the current OPD by default); the cube stays where the DM computed it
                src.OPD_no_pupil = dm_OPD

            # the pupil mask is moved to the backend of the OPD (NumPy and CuPy arrays cannot be mixed)
            mask = _to_backend(src.mask, _backend_of(src.OPD_no_pupil))
            if len(src.OPD_no_pupil.shape) > 2:
                src.OPD = src.OPD_no_pupil * (mask[:, :, None] if np.ndim(mask) == 2 else mask)
            else:
                src.OPD = src.OPD_no_pupil * mask
    # ------------------------------------------------------------------------------------------
    # Mis-registrations
    # ------------------------------------------------------------------------------------------

    def _set_mis_registered_coordinates(self):
        """Apply self.misReg to self.initial_coordinates.

        Sets xIF, yIF and coordinates [m] and returns the positions (u0x, u0y) on the grid of modesComputation.
        """
        x0, y0 = self.initial_coordinates[:, 0], self.initial_coordinates[:, 1]
        # anamorphosis
        x, y = self.anamorphosis_coordinates(x0, y0,
                                             self.misReg.anamorphosisAngle * np.pi/180,
                                             self.misReg.radialScaling,
                                             self.misReg.tangentialScaling)
        # rotation
        x, y = self.rotate_coordinates(x, y, self.misReg.rotationAngle * np.pi/180)
        # shifts
        self.xIF = x - self.misReg.shiftX
        self.yIF = y - self.misReg.shiftY
        self.coordinates = np.stack([self.xIF, self.yIF], axis=1)
        # corresponding coordinates on the pixel grid
        u0x = self.resolution/2 + self.xIF*self.resolution/self.D
        u0y = self.resolution/2 + self.yIF*self.resolution/self.D
        return u0x, u0y

    def _compute_gaussian_modes(self, u0x, u0y):
        """Gaussian influence functions centered on (u0x, u0y), using self.misReg for their shape."""
        Q = Parallel(n_jobs=8, prefer='threads')(delayed(self.modesComputation)(i, j) for i, j in zip(u0x, u0y))
        self.modes = np.squeeze(np.moveaxis(np.asarray(Q), 0, -1))

    def _pixel_transform(self, mis_registration):
        """Mapping (column, row) of the initial geometry -> mis-registered geometry, on the grid of the modes.

        Built with anamorphosis_coordinates / rotate_coordinates so that it follows exactly the
        conventions used for the actuator coordinates.
        """
        linear = np.zeros([2, 2])
        for i_axis, (x, y) in enumerate([(1., 0.), (0., 1.)]):
            x, y = self.anamorphosis_coordinates(x, y,
                                                 mis_registration.anamorphosisAngle * np.pi/180,
                                                 mis_registration.radialScaling,
                                                 mis_registration.tangentialScaling)
            linear[:, i_axis] = self.rotate_coordinates(x, y, mis_registration.rotationAngle * np.pi/180)
        # grid of modesComputation: pixel k <-> x = (k - center) * D / (resolution - 1)
        meter_per_pixel = self.D / (self.resolution - 1)
        center = np.full(2, (self.resolution - 1) / 2)
        shift = -np.array([mis_registration.shiftX, mis_registration.shiftY]) / meter_per_pixel
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        matrix[:2, 2] = center - linear @ center + shift
        return sk.AffineTransform(matrix=matrix)

    def _interpolate_modes(self, modes, mis_registration_modes, mis_registration):
        """Interpolate modes [n_pix**2, n] given at mis_registration_modes to mis_registration."""
        # output pixel -> initial geometry -> geometry of the input modes
        inverse_map = self._pixel_transform(mis_registration).inverse + self._pixel_transform(mis_registration_modes)
        cube = np.asarray(modes).T.reshape(-1, self.resolution, self.resolution)

        def warp(image):
            return sk.warp(image, inverse_map, order=3, mode='constant', cval=0, preserve_range=True).reshape(-1)
        warped = Parallel(n_jobs=8, prefer='threads')(delayed(warp)(image) for image in cube)
        return np.stack(warped, axis=1).astype(np.asarray(modes).dtype, copy=False)

    def apply_mis_registration(self, mis_registration):
        """Return a copy of the DM with mis_registration applied.

        The mis-registration is absolute: it replaces self.misReg and is applied to the initial
        (not mis-registered) geometry. Everything else is inherited from self (pitch, mechanical
        coupling, valid actuators, floating precision, altitude, flips, sign, ...), so that
        dm.apply_mis_registration(dm.misReg) reproduces dm. The influence functions are:
            - Gaussian (default): recomputed at the new actuator positions
            - InfluenceFunctions object: recomputed by InfluenceFunctions for the mis-registration
            - user-defined modes: interpolated, the input modes being the DM at its misReg at creation
            - M4: recomputed with the M4 model

        Parameters
        ----------
        mis_registration : MisRegistration

        Returns
        -------
        DeformableMirror
        """
        mis_registration = MisRegistration(mis_registration)

        if self.isM4:
            dm = DeformableMirror(telescope=self.telescope,
                                  nSubap=self.nAct,
                                  mechCoupling=self.mechCoupling,
                                  pitch=self.pitch,
                                  misReg=mis_registration,
                                  M4_param=self.M4_param,
                                  floating_precision=self.floating_precision,
                                  altitude=self.altitude,
                                  print_dm_properties=False)
            if dm.nValidAct != self.nValidAct:
                warning('The M4 valid actuator selection changed with the mis-registration ('
                        + str(self.nValidAct) + ' -> ' + str(dm.nValidAct) + ' actuators).')
            return dm

        dm = copy.copy(self)
        dm.misReg = mis_registration
        dm.print_dm_properties = False
        if getattr(self, 'altitude_layer', None) is not None:
            dm.altitude_layer = copy.deepcopy(self.altitude_layer)

        if getattr(self, 'name_system', None) is not None:
            from OOPAO.InfluenceFunctions import InfluenceFunctions
            IF = InfluenceFunctions(name_system=self.name_system,
                                    diameter=self.D,
                                    resolution=self.resolution,
                                    specific_parameters=self.specific_parameters,
                                    loc=self.loc,
                                    mis_registration=mis_registration,
                                    flip_lr=self.flip_lr,
                                    flip_ud=self.flip_ud,
                                    sign=self.sign)
            dm.modes = IF.influence_function_2D
            dm.coordinates = np.asarray(IF.coordinates, dtype=float)
            dm.xIF, dm.yIF = dm.coordinates[:, 0], dm.coordinates[:, 1]
        elif self._reference_modes is not None:
            dm._set_mis_registered_coordinates()
            dm.modes = self._interpolate_modes(self._reference_modes, self._reference_mis_registration, mis_registration)
        else:
            dm._compute_gaussian_modes(*dm._set_mis_registered_coordinates())

        n_modes = dm.modes.shape[1] if np.ndim(dm.modes) == 2 else 1
        if n_modes != self.nValidAct:
            raise OopaoError('apply_mis_registration changed the number of actuators ('
                             + str(self.nValidAct) + ' -> ' + str(n_modes) + ').')

        # own copy of the commands / OPD (the attributes of self are shared after copy.copy)
        dm.reset_gpu_modes()
        dm.coefs = np.zeros(dm.nValidAct, dtype=np.float32 if dm.floating_precision == 32 else dm.precision())
        return dm

    def set_pupil_footprint(self):
        if len(self.src_list) == 1:
            [x_z, y_z] = pol2cart(self.altitude_layer.altitude * np.tan(self.src_list[0].coordinates[0] / self.rad2arcsec) * self.altitude_layer.resolution / self.altitude_layer.D, np.deg2rad(self.src_list[0].coordinates[1]))
            center_x = int(y_z) + self.altitude_layer.resolution // 2
            center_y = int(x_z) + self.altitude_layer.resolution // 2
            self.altitude_layer.center_x = center_x
            self.altitude_layer.center_y = center_y
            self.altitude_layer.pupil_footprint = np.zeros([self.altitude_layer.resolution, self.altitude_layer.resolution], dtype=self.precision())
            self.altitude_layer.pupil_footprint[center_x - self.telescope.resolution // 2:center_x + self.telescope.resolution // 2, center_y - self.telescope.resolution // 2:center_y + self.telescope.resolution // 2] = 1
        else:
            self.altitude_layer.pupil_footprint = []
            self.altitude_layer.extra_sx = []
            self.altitude_layer.extra_sy = []
            self.altitude_layer.center_x = []
            self.altitude_layer.center_y = []

            for src in self.src_list:
                [x_z, y_z] = pol2cart(self.altitude_layer.altitude * np.tan(src.coordinates[0] / self.rad2arcsec)
                                      * self.altitude_layer.resolution / self.altitude_layer.D, np.deg2rad(src.coordinates[1]))
                self.altitude_layer.extra_sx.append(int(x_z) - x_z)
                self.altitude_layer.extra_sy.append(int(y_z) - y_z)
                center_x = int(y_z) + self.altitude_layer.resolution // 2
                center_y = int(x_z) + self.altitude_layer.resolution // 2

                pupil_footprint = np.zeros([self.altitude_layer.resolution, self.altitude_layer.resolution], dtype=self.precision())
                pupil_footprint[center_x - self.telescope.resolution // 2:center_x + self.telescope.resolution // 2, center_y - self.telescope.resolution // 2:center_y + self.telescope.resolution // 2] = 1
                self.altitude_layer.pupil_footprint.append(pupil_footprint)
                self.altitude_layer.center_x.append(center_x)
                self.altitude_layer.center_y.append(center_y)

    def buildLayer(self, telescope, altitude):

        # initialize layer object
        layer = emptyClass()
        # create a random state to allow reproductible sequences of phase screens
        # gather properties of the atmosphere
        layer.altitude = altitude
        # Diameter and resolution of the layer including the Field Of View and the number of extra pixels
        layer.D_fov = telescope.D+2*np.tan(telescope.fov_rad/2)*layer.altitude
        layer.resolution_fov = int(np.ceil((telescope.resolution/telescope.D)*layer.D_fov))
        # 4 pixels are added as a margin for the edges
        layer.resolution = layer.resolution_fov + 4
        layer.D = layer.resolution * telescope.D / telescope.resolution
        layer.center = layer.resolution//2

        if telescope.src.tag == 'source':
            [x_z, y_z] = pol2cart(layer.altitude*np.tan(telescope.src.coordinates[0]/self.rad2arcsec) * layer.resolution / layer.D, np.deg2rad(telescope.src.coordinates[1]))
            center_x = int(y_z)+layer.resolution//2
            center_y = int(x_z)+layer.resolution//2
            layer.center_x = center_x
            layer.center_y = center_y
            layer.pupil_footprint = np.zeros([layer.resolution, layer.resolution], dtype=self.precision())
            layer.pupil_footprint[center_x-telescope.resolution//2:center_x+telescope.resolution // 2, center_y-telescope.resolution//2:center_y+telescope.resolution//2] = 1
        else:

            layer.pupil_footprint = []
            layer.extra_sx = []
            layer.extra_sy = []
            layer.center_x = []
            layer.center_y = []
            for i in range(telescope.src.n_source):
                [x_z, y_z] = pol2cart(layer.altitude*np.tan(telescope.src.coordinates[i][0]/self.rad2arcsec) * layer.resolution / layer.D, np.deg2rad(telescope.src.coordinates[i][1]))
                layer.extra_sx.append(int(x_z)-x_z)
                layer.extra_sy.append(int(y_z)-y_z)
                center_x = int(y_z)+layer.resolution//2
                center_y = int(x_z)+layer.resolution//2

                pupil_footprint = np.zeros([layer.resolution, layer.resolution], dtype=self.precision())
                pupil_footprint[center_x-telescope.resolution//2:center_x+telescope.resolution // 2, center_y-telescope.resolution//2:center_y+telescope.resolution//2] = 1
                layer.pupil_footprint.append(pupil_footprint)
                layer.center_x.append(center_x)
                layer.center_y.append(center_y)

        return layer

    def _footprint_slices(self, src):
        """Rows and columns of the altitude layer seen by `src` (the pupil footprint is a square block)."""
        layer = self.altitude_layer
        n = self.telescope.resolution
        if isinstance(layer.center_x, list):
            # one footprint per source of the asterism, indexed by the source's own position in it
            center_x, center_y = layer.center_x[src.ast_idx], layer.center_y[src.ast_idx]
        else:
            center_x, center_y = layer.center_x, layer.center_y
        return slice(center_x - n//2, center_x + n//2), slice(center_y - n//2, center_y + n//2)

    def get_OPD_altitude(self, src):
        self.set_pupil_footprint()
        # crop the part of the layer seen by the source: plain slicing, so the OPD stays on its backend (NumPy or CuPy)
        rows, cols = self._footprint_slices(src)
        OPD = self.OPD[rows, cols]
        if ~np.isinf(src.altitude):
            # cone effect: the interpolation runs on the CPU, the result is moved back to the backend of the DM OPD
            backend = _backend_of(OPD)
            OPD = _to_backend(OPD, np)
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
            OPD = np.asarray(np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out)).T)
            OPD = _to_backend(OPD, backend)

        return OPD

    def rotate_coordinates(self, x, y, angle):
        xOut = x*np.cos(angle)-y*np.sin(angle)
        yOut = y*np.cos(angle)+x*np.sin(angle)
        return xOut, yOut

    def anamorphosis_coordinates(self, x, y, angle, mRad, mNorm):

        mRad += 1
        mNorm += 1
        xOut = x * (mRad*np.cos(angle)**2 + mNorm * np.sin(angle)**2) + y * (mNorm*np.sin(2*angle)/2 - mRad*np.sin(2*angle)/2)
        yOut = y * (mRad*np.sin(angle)**2 + mNorm * np.cos(angle)**2) + x * (mNorm*np.sin(2*angle)/2 - mRad*np.sin(2*angle)/2)
        return xOut, yOut

    def modesComputation(self, i, j):
        x0 = i
        y0 = j
        cx = (1+self.misReg.radialScaling)*(self.resolution / self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))
        cy = (1+self.misReg.tangentialScaling)*(self.resolution / self.nActAlongDiameter)/np.sqrt(2*np.log(1./self.mechCoupling))

        # Radial direction of the anamorphosis
        theta = self.misReg.anamorphosisAngle*np.pi/180
        x = np.linspace(0, 1, self.resolution)*self.resolution
        X, Y = np.meshgrid(x, x)

        # Compute the 2D Gaussian coefficients
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
        # matplotlib needs NumPy arrays (e.g. when a GPU-resident dm.OPD is given)
        input_opd = _to_backend(input_opd, np)
        ax.imshow(input_opd, extent=[-self.D/2, self.D/2, -self.D/2, self.D/2])
        center = self.telescope.D/2
        [x_tel, y_tel] = pol2cart(self.D/2, np.linspace(0, 2*np.pi, 100, endpoint=True))
        cm = plt.get_cmap('gist_rainbow')
        col = []
        for i_source in range(len(list_src)):
            col.append(cm(1.*i_source/len(list_src)))
            [x_c, y_c] = pol2cart(self.telescope.initial_D/2, np.linspace(0, 2*np.pi, 100, endpoint=True))
            if self.altitude is None:
                h = list_src[i_source].altitude
            else:
                h = list_src[i_source].altitude-self.altitude
            if np.isinf(h):
                r = self.telescope.initial_D/2
            else:
                r = (h/list_src[i_source].altitude)*self.telescope.initial_D/2
            [x_cone, y_cone] = pol2cart(r, np.linspace(0, 2*np.pi, 100, endpoint=True))
            if self.altitude is None:
                [x_z, y_z] = [0, 0]
            else:
                [x_z, y_z] = pol2cart(self.altitude*np.tan((list_src[i_source].coordinates[0])/self.rad2arcsec), -np.deg2rad(list_src[i_source].coordinates[1]))
            center = 0
            [x_c, y_c] = pol2cart(self.D/2, np.linspace(0, 2*np.pi, 100, endpoint=True))
            nm = (list_src[i_source].type) + '@' + str(list_src[i_source].coordinates[0])+'"'
            ax.plot(x_cone+x_z+center, y_cone+y_z+center, '-', color=col[i_source], label=nm)
            ax.fill(x_cone+x_z+center, y_cone+y_z+center, y_z+center, alpha=0.1, color=col[i_source])
        ax.set_xlabel('[m]')
        ax.set_ylabel('[m]')
        ax.set_title('Altitude '+str(self.altitude)+' m')
        ax.plot(x_tel+center, y_tel+center, '--', color='k')
        ax.legend(loc='upper left')
        makeSquareAxes(plt.gca())
        return

    def _modes_times_coefs(self):
        """DM shape (flattened) = modes @ coefs.

        On the CPU the product uses self.modes directly. On the GPU a copy of the modes, at the working
        precision, is uploaded once and reused; it is uploaded again whenever self.modes is replaced by a new
        array (call reset_gpu_modes() after editing self.modes in place). The coefficients can be NumPy or
        CuPy arrays. The result stays on the GPU with GPU residency and is copied back to the CPU otherwise.
        """
        if not self.gpu_available:
            try:
                return np.matmul(self.modes, self._coefs)
            except (TypeError, ValueError):
                return self.modes @ self._coefs
        if self._gpu_modes_source is not self.modes:
            self._gpu_modes = None  # free the previous copy before uploading the new one
            self._gpu_modes = xp.asarray(self.modes, dtype=self.precision)
            self._gpu_modes_source = self.modes
        opd = self._gpu_modes @ xp.asarray(self._coefs, dtype=self.precision)
        return opd if self.gpu_resident else xp.asnumpy(opd)

    def reset_gpu_modes(self):
        """Drop the GPU copy of the modes: it is uploaded again at the next update of dm.coefs."""
        self._gpu_modes = None
        self._gpu_modes_source = None

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, val):
        if self.floating_precision == 32:
            # astype keeps CuPy arrays on the GPU (np.float32 would force an implicit, forbidden, conversion)
            self._coefs = val.astype(np.float32) if hasattr(val, 'astype') else np.float32(val)
        else:
            self._coefs = val
        if np.isscalar(val):
            if val == 0:
                self._coefs = np.zeros(self.nValidAct, dtype=self.precision())
                opd = self._modes_times_coefs()
                self.OPD = _backend_of(opd).asarray(opd, dtype=self.precision).reshape(
                    self.resolution, self.resolution)
            else:
                print('Error: wrong value for the coefficients')
        else:
            if len(val) == self.nValidAct:
                if np.ndim(val) == 1:
                    shape = [self.resolution, self.resolution]
                else:
                    shape = [self.resolution, self.resolution, val.shape[1]]
                opd = self._modes_times_coefs()
                self.OPD = _backend_of(opd).asarray(opd, dtype=self.precision).reshape(shape)
            else:
                print('Error: wrong value for the coefficients')
                sys.exit(0)
            self.current_coefs = self.coefs.copy()

    def set_coefs_value(self, actuator_index: list, actuator_coefs: list):
        """
        This functions allows to assign a value to a given actuator list instead of a full set of actuator commands.

        Parameters
        ----------
        actuator_index : list
            Index of the actuators to be updated.
        actuator_value : list
            Actuator Coefficient to be applied to the selected actuators
        """
        if np.isscalar(actuator_index) and np.isscalar(actuator_coefs):
            actuator_index = [actuator_index]
            actuator_coefs = [actuator_coefs]

        if len(actuator_index) != len(actuator_coefs):
            raise OopaoError('The list of actuator_index and actuator_coefs must have the same length')

        for i_act in range(len(actuator_index)):
            self.coefs[actuator_index[i_act]] = actuator_coefs[i_act]
        # update dm shape
        self.coefs = self.coefs
        return

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