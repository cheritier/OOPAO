# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

This module gathers all the functions to compute user-defined influence
functions for the different systems considered with OOPAO.

@author: cheritier
"""

import numpy as np

from OOPAO.tools.tools import OopaoError, centroid


class InfluenceFunctions:
    """
    Compute the influence functions of a user-defined system.

    The heavy computation is delegated to the system-specific function then generic operations (flips, sign) are applied.

    Attributes
    ----------
    influence_function_2D : np.ndarray
        Influence-function cube of shape [n_act, resolution, resolution],
        with flips and sign applied.
    coordinates : np.ndarray
        Actuator coordinates computed as the centroids of the (flipped)
        influence functions.
    sign : int
        Sign convention applied to the influence functions.
    """

    def __init__(self,
                 name_system: str,
                 diameter: float,
                 resolution: int,
                 loc: str = None,
                 mis_registration=None,
                 flip_lr: bool = False,
                 flip_ud: bool = False,
                 specific_parameters: dict = None,
                 sign: int = 1):
        """
        Parameters
        ----------
        name_system : str
            Name of the system. One of: 'ELT_M4', 'EKARUS_DM468', 'LBT_ASM',
            'GHOST_DM492', 'PAPYRUS_DM241', 'RAMA_DM97'.
        diameter : float
            Telescope diameter in [m].
        resolution : int
            Number of pixels across the pupil.
        loc : str, optional
            Path to the system input data.
        mis_registration : MisRegistration, optional
            Mis-registration to apply (shift, rotation, scaling, anamorphosis).
        flip_lr : bool, optional
            Flip the influence functions left/right. Default is False.
        flip_ud : bool, optional
            Flip the influence functions up/down. Default is False.
        specific_parameters : dict, optional
            System-specific parameters (see the corresponding compute function).
        sign : int, optional
            Sign convention applied to the influence functions. Default is 1.

        Raises
        ------
        OopaoError
            If `name_system` is not recognized.
        """
        compute_function = self._get_compute_function(name_system)
        print('Computing Influence Functions for ' + name_system + '...')

        IF_2D = compute_function(name_system=name_system,
                                 diameter=diameter,
                                 resolution=resolution,
                                 loc=loc,
                                 mis_registration=mis_registration,
                                 specific_parameters=specific_parameters)

        # ------------- general operations applied to the IFs -------------
        self.sign = sign

        # potential flips of the IFs
        if flip_lr:
            IF_2D = np.flip(IF_2D, axis=2)
        if flip_ud:
            IF_2D = np.flip(IF_2D, axis=1)

        self.influence_function_2D = IF_2D * self.sign
        self.coordinates = centroid(IF_2D)

    @staticmethod
    def _get_compute_function(name_system: str):
        """Return the compute function matching `name_system`.
        """
        from OOPAO.tools import user_defined_influence_functions as udif

        dispatch = {'ELT_M4':        udif.compute_ELT_M4_influence_functions,
                    'EKARUS_DM468':  udif.compute_EKARUS_DM468_influence_functions,
                    'LBT_ASM':       udif.compute_LBT_ASM_influence_functions,
                    'GHOST_DM492':   udif.compute_GHOST_DM492_influence_functions,
                    'PAPYRUS_DM241': udif.compute_PAPYRUS_DM241_influence_functions,
                    'RAMA_DM97':     udif.compute_RAMA_DM97_influence_functions}

        try:
            return dispatch[name_system]
        except KeyError:
            raise OopaoError(
                'The name of your system is not recognized or not implemented '
                'yet. Valid options are: ' + ', '.join(dispatch.keys()))