import numpy as np

from ..DeformableMirror import DeformableMirror
from ..MisRegistration import MisRegistration


def _param_get(param, *keys, default=None):
    """Return param[key] for the first key present in param (None values are skipped)."""
    if param is None:
        return default
    for key in keys:
        try:
            value = param[key]
        except (KeyError, TypeError, IndexError):
            continue
        if value is not None:
            return value
    return default


def _build_dm_from_param(tel, misreg, param, print_dm_properties, floating_precision):
    """Build a DM directly with the mis-registration `misreg` from a parameter file
    (used when no DM is available to start from)."""
    pitch = _param_get(param, 'pitch')
    coordinates = _param_get(param, 'dm_coordinates')
    n_subaperture = _param_get(param, 'nSubaperture', 'wfs_n_subaperture')
    mechanical_coupling = _param_get(param, 'mechanicalCoupling', 'dm_mechanical_coupling')

    if _param_get(param, 'isM4', default=False) is True:
        dm_tmp = DeformableMirror(telescope=tel,
                                  nSubap=n_subaperture,
                                  mechCoupling=mechanical_coupling,
                                  coordinates=coordinates,
                                  pitch=None,
                                  misReg=misreg,
                                  M4_param=param,
                                  print_dm_properties=print_dm_properties,
                                  floating_precision=floating_precision)
        label = 'M4'

    elif _param_get(param, 'isLBT', default=False) is True:
        if _param_get(param, 'new_IF', default=False):
            from lbt_tools import get_influence_functions_new as get_influence_functions
        else:
            from lbt_tools import get_influence_functions as get_influence_functions

        modes, coord, M2C, validAct = get_influence_functions(telescope=tel,
                                                              misReg=misreg,
                                                              filename_IF=param['filename_if'],
                                                              filename_mir_modes=param['filename_mir_modes'],
                                                              filename_coordinates=param['filename_coord'],
                                                              filename_M2C=param['filename_m2c'])
        param['isM4'] = False
        dm_tmp = DeformableMirror(telescope=tel,
                                  nSubap=n_subaperture,
                                  mechCoupling=mechanical_coupling,
                                  coordinates=coord,
                                  pitch=pitch,
                                  misReg=misreg,
                                  modes=np.reshape(modes, [tel.resolution**2, modes.shape[2]]),
                                  M4_param=param,
                                  print_dm_properties=print_dm_properties,
                                  floating_precision=floating_precision)
        label = 'user-defined DM'
    else:
        dm_tmp = DeformableMirror(telescope=tel,
                                  nSubap=n_subaperture,
                                  mechCoupling=mechanical_coupling,
                                  coordinates=coordinates,
                                  pitch=pitch,
                                  misReg=misreg,
                                  print_dm_properties=print_dm_properties,
                                  floating_precision=floating_precision)
        label = 'Synthetic DM'
    return dm_tmp, label


def applyMisRegistration(tel,
                         misRegistration_tmp,
                         param=None,
                         wfs=None,
                         extra_dm_mis_registration=None,
                         print_dm_properties=True,
                         floating_precision=64,
                         dm_input=None):
    """Return a DM with the mis-registration misRegistration_tmp (+ extra_dm_mis_registration) applied.

    - dm_input provided: dm_input.apply_mis_registration() is used. The mis-registration is absolute
      (applied to the initial geometry of dm_input, not cumulated with dm_input.misReg), and all the
      other properties are inherited from dm_input (floating_precision is therefore ignored).
    - dm_input not provided: the DM is built from `param`.
    - wfs provided (pyramid): the shifts are applied to the WFS, the rest to the DM.
    """
    # used when the wfs is provided to the function to apply a shift to both WFS and DM
    if extra_dm_mis_registration is None:
        extra_dm_mis_registration = MisRegistration()

    if wfs is None:
        misreg_dm = misRegistration_tmp + extra_dm_mis_registration
        if dm_input is not None:
            dm_tmp = dm_input.apply_mis_registration(misreg_dm)
            label = 'input DM'
        else:
            dm_tmp, label = _build_dm_from_param(tel, misreg_dm, param,
                                                 print_dm_properties, floating_precision)
        if print_dm_properties:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Mis-Registrations Applied on ' + label + '!')

    elif wfs.tag == 'pyramid':
        # mis-registration for the WFS
        misRegistration_wfs = MisRegistration()
        # sign change to match Pyramid pupil shift vs DM shift
        misRegistration_wfs.shiftX = -misRegistration_tmp.shiftX
        misRegistration_wfs.shiftY = misRegistration_tmp.shiftY
        wfs.apply_shift_wfs(misRegistration_wfs.shiftX, misRegistration_wfs.shiftY, units='m')  # [m], consistent with the dm shift

        # mis-registration for the DM (the shift is handled by the WFS)
        misRegistration_dm = MisRegistration()
        misRegistration_dm.rotationAngle = misRegistration_tmp.rotationAngle
        misRegistration_dm.anamorphosisAngle = misRegistration_tmp.anamorphosisAngle
        misRegistration_dm.tangentialScaling = misRegistration_tmp.tangentialScaling
        misRegistration_dm.radialScaling = misRegistration_tmp.radialScaling

        dm_tmp = applyMisRegistration(tel,
                                      misRegistration_dm,
                                      param,
                                      wfs=None,
                                      extra_dm_mis_registration=extra_dm_mis_registration,
                                      print_dm_properties=print_dm_properties,
                                      floating_precision=floating_precision,
                                      dm_input=dm_input)
        if print_dm_properties:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Mis-Registrations Applied on both DM and WFS!')
    else:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Wrong object passed as a wfs.. applying the mis-registrations to the DM only')
        dm_tmp = applyMisRegistration(tel,
                                      misRegistration_tmp,
                                      param,
                                      wfs=None,
                                      extra_dm_mis_registration=extra_dm_mis_registration,
                                      print_dm_properties=print_dm_properties,
                                      floating_precision=floating_precision,
                                      dm_input=dm_input)

    _ = dm_tmp.misReg.properties()
    return dm_tmp