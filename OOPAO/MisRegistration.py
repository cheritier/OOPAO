# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:01:10 2020

@author: cheritie
"""
import numpy as np


class MisRegistration:
    """Geometrical mis-registration between a DM and a WFS.

    Attributes
    ----------
    rotationAngle : float
        Rotation angle in [deg].
    shiftX, shiftY : float
        Shifts in [m].
    anamorphosisAngle : float
        Anamorphosis angle in [deg].
    radialScaling, tangentialScaling : float
        Scalings in fraction of the diameter (0.01 = 1 %). They include the
        contribution of the magnification.
    magnification : float
        Isotropic scaling. Setting it to a value v moves both radialScaling and
        tangentialScaling by (v - previous value), so the operation is
        idempotent and can be safely combined with the explicit scalings.
    """

    # parameters stored directly (magnification is a derived shortcut)
    _FIELDS = ('rotationAngle', 'shiftX', 'shiftY', 'anamorphosisAngle',
               'tangentialScaling', 'radialScaling')

    def __init__(self, param=None):
        self.tag = 'misRegistration'
        self.isInitialized = False
        self._magnification = 0
        for field in self._FIELDS:
            setattr(self, field, 0)

        if param is None:
            pass
        elif isinstance(param, dict):
            for field in self._FIELDS:
                setattr(self, field, param[field])
            # in a dictionary, the magnification is applied on top of the scalings
            self.magnification = param.get('magnification', 0)
        elif getattr(param, 'tag', None) == 'misRegistration':
            for field in self._FIELDS:
                setattr(self, field, getattr(param, field))
            # the scalings of param already include its magnification: copy the value only
            self._magnification = param.magnification
        else:
            raise TypeError('A MisRegistration can only be created from None, a dict or another MisRegistration.')

        self.properties()
        self.isInitialized = True

    # ----------------------------------------- Arithmetic -----------------------------------------

    def _combine(self, other, sign):
        if getattr(other, 'tag', None) != 'misRegistration':
            raise TypeError('A MisRegistration can only be combined with another MisRegistration.')
        out = MisRegistration()
        for field in self._FIELDS:
            setattr(out, field, getattr(self, field) + sign * getattr(other, field))
        out._magnification = self.magnification + sign * other.magnification
        return out

    def __add__(self, other):
        return self._combine(other, +1)

    def __sub__(self, other):
        return self._combine(other, -1)

    def as_array(self):
        """Values of rotationAngle, shiftX, shiftY, anamorphosisAngle, tangentialScaling, radialScaling, magnification."""
        return np.array([getattr(self, f) for f in self._FIELDS] + [self.magnification], dtype=float)

    def __eq__(self, other):
        if not isinstance(other, MisRegistration):
            return False
        return np.array_equal(self.as_array(), other.as_array())

    def __ne__(self, other):
        return not self.__eq__(other)

    # ----------------------------------------- Properties -----------------------------------------

    @property
    def magnification(self):
        return self._magnification

    @magnification.setter
    def magnification(self, val):
        delta = val - self._magnification
        self._magnification = val
        self.radialScaling += delta
        self.tangentialScaling += delta

    @property
    def misRegName(self):
        """Name of the mis-registration, used e.g. to name the folders of the sensitivity matrices."""
        fmt = '%.2f' if (self.radialScaling == 0 and self.tangentialScaling == 0) else '%.4f'
        return (('rot_%.2f_sX_%.2f_m_sY_%.2f_m_anam_%.2f_mR_' + fmt + '_mT_' + fmt)
                % (self.rotationAngle, self.shiftX, self.shiftY, self.anamorphosisAngle,
                   self.radialScaling + 1., self.tangentialScaling + 1.))

    # for backward compatibility
    def print_properties(self):
        print(self)

    # for backward compatibility
    def print_(self):
        print(self)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['rotation'] = f"{'Rotation [°]':<25s}|{self.rotationAngle:^9.3f}"
        self.prop['shift_x'] = f"{'Shift X [m]':<25s}|{self.shiftX:^9.3e}"
        self.prop['shift_y'] = f"{'Shift Y [m]':<25s}|{self.shiftY:^9.3e}"
        self.prop['anamophosis_angle'] = f"{'Anamorphosis angle [°]':<25s}|{self.anamorphosisAngle:^9.3f}"
        self.prop['tengential_scaling'] = f"{'Tangential scaling [%]':<25s}|{self.tangentialScaling*100:^9.3f}"
        self.prop['radial_scaling'] = f"{'Radial scaling [%]':<25s}|{self.radialScaling*100:^9.3f}"
        return self.prop

    def __repr__(self):
        self.properties()
        n_char = len(max(self.prop.values(), key=len))
        str_prop = ''.join(value + '\n' for value in self.prop.values())
        title = f'\n{"Misregistration":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        return title + str_prop + end_line