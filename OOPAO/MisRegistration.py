# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:01:10 2020

@author: cheritie
"""
import inspect


class MisRegistration:
    def __init__(self, param=None):

        self.tag = 'misRegistration'
        self.isInitialized = False
        if param is None:
            self.rotationAngle = 0                    # rotation angle in degrees
            self.shiftX = 0                           # shift X in m
            self.shiftY = 0                           # shift Y in m
            self.anamorphosisAngle = 0                # amamorphosis angle in degrees
            self.tangentialScaling = 0                # normal scaling in % of diameter
            self.radialScaling = 0                    # radial scaling in % of diameter
        else:
            if isinstance(param, dict):
                # print('MisRegistration object created from a ditionnary')
                # rotation angle in degrees
                self.rotationAngle = param['rotationAngle']
                # shift X in m
                self.shiftX = param['shiftX']
                # shift Y in m
                self.shiftY = param['shiftY']
                # amamorphosis angle in degrees
                self.anamorphosisAngle = param['anamorphosisAngle']
                # normal scaling in % of diameter
                self.tangentialScaling = param['tangentialScaling']
                # radial scaling in % of diameter
                self.radialScaling = param['radialScaling']
            else:
                if inspect.isclass(type(param)):
                    if param.tag == 'misRegistration':
                        # print('MisRegistration object created from an existing Misregistration object')

                        # rotation angle in degrees
                        self.rotationAngle = param.rotationAngle
                        self.shiftX = param.shiftX                           # shift X in m
                        self.shiftY = param.shiftY                           # shift Y in m
                        # amamorphosis angle in degrees
                        self.anamorphosisAngle = param.anamorphosisAngle
                        # normal scaling in % of diameter
                        self.tangentialScaling = param.tangentialScaling
                        self.radialScaling = param.radialScaling
                    else:
                        print(
                            'wrong type of object passed to a MisRegistration object')
                else:
                    print('wrong type of object passed to a MisRegistration object')

        if self.radialScaling == 0 and self.tangentialScaling == 0:
            self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                              'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                              'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                              'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                              'mR_' + str('%.2f' % (self.radialScaling+1.)) + '_'\
                              'mT_' + str('%.2f' % (self.tangentialScaling+1.))
        else:
            self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                'mR_' + str('%.4f' % (self.radialScaling+1.)) + '_'\
                'mT_' + str('%.4f' % (self.tangentialScaling+1.))
        _ = self.properties()
        self.isInitialized = True

#        mis-registrations can be added or sub-tracted using + and -
    def __add__(self, misRegObject):
        if misRegObject.tag == 'misRegistration':
            tmp = MisRegistration()
            tmp.rotationAngle = self.rotationAngle + misRegObject.rotationAngle
            tmp.shiftX = self.shiftX + misRegObject.shiftX
            tmp.shiftY = self.shiftY + misRegObject.shiftY
            tmp.anamorphosisAngle = self.anamorphosisAngle + misRegObject.anamorphosisAngle
            tmp.tangentialScaling = self.tangentialScaling + misRegObject.tangentialScaling
            tmp.radialScaling = self.radialScaling + misRegObject.radialScaling
        else:
            print(
                'Error you are trying to combine a MisRegistration Object with the wrong type of object')
        return tmp

    def __sub__(self, misRegObject):
        if misRegObject.tag == 'misRegistration':
            tmp = MisRegistration()

            tmp.rotationAngle = self.rotationAngle - misRegObject.rotationAngle
            tmp.shiftX = self.shiftX - misRegObject.shiftX
            tmp.shiftY = self.shiftY - misRegObject.shiftY
            tmp.anamorphosisAngle = self.anamorphosisAngle - misRegObject.anamorphosisAngle
            tmp.tangentialScaling = self.tangentialScaling - misRegObject.tangentialScaling
            tmp.radialScaling = self.radialScaling - misRegObject.radialScaling
        else:
            print(
                'Error you are trying to combine a MisRegistration Object with the wrong type of object')
        return tmp

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
# ----------------------------------------- Properties  -----------------------------------------

    @property
    def rotationAngle(self):
        return self._rotationAngle

    @rotationAngle.setter
    def rotationAngle(self, val):
        self._rotationAngle = val
        if self.isInitialized:
            if self.radialScaling == 0 and self.tangentialScaling == 0:
                self.misRegName = 'rot_' + str('%.2f' % val) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.2f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.2f' % (self.tangentialScaling+1.))
            else:
                self.misRegName = 'rot_' + str('%.2f' % val) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.4f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.4f' % (self.tangentialScaling+1.))

    @property
    def shiftX(self):
        return self._shiftX

    @shiftX.setter
    def shiftX(self, val):
        self._shiftX = val
        if self.isInitialized:
            if self.radialScaling == 0 and self.tangentialScaling == 0:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (val)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.2f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.2f' % (self.tangentialScaling+1.))
            else:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (val)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.4f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.4f' % (self.tangentialScaling+1.))

    @property
    def shiftY(self):
        return self._shiftY

    @shiftY.setter
    def shiftY(self, val):
        self._shiftY = val
        if self.isInitialized:
            if self.radialScaling == 0 and self.tangentialScaling == 0:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (val)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.2f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.2f' % (self.tangentialScaling+1.))
            else:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (val)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.4f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.4f' % (self.tangentialScaling+1.))

    @property
    def anamorphosisAngle(self):
        return self._anamorphosisAngle

    @anamorphosisAngle.setter
    def anamorphosisAngle(self, val):
        self._anamorphosisAngle = val
        if self.isInitialized:
            if self.radialScaling == 0 and self.tangentialScaling == 0:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % val) + '_'\
                    'mR_' + str('%.2f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.2f' % (self.tangentialScaling+1.))
            else:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % val) + '_'\
                    'mR_' + str('%.4f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.4f' % (self.tangentialScaling+1.))

    @property
    def radialScaling(self):
        return self._radialScaling

    @radialScaling.setter
    def radialScaling(self, val):
        self._radialScaling = val
        if self.isInitialized:
            if self.radialScaling == 0 and self.tangentialScaling == 0:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.2f' % (val+1.)) + '_'\
                    'mT_' + str('%.2f' % (self.tangentialScaling+1.))
            else:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_m_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_m_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mR_' + str('%.4f' % (val+1.)) + '_'\
                    'mT_' + str('%.4f' % (self.tangentialScaling+1.))

    @property
    def tangentialScaling(self):
        return self._tangentialScaling

    @tangentialScaling.setter
    def tangentialScaling(self, val):
        self._tangentialScaling = val
        if self.isInitialized:
            if self.radialScaling == 0 and self.tangentialScaling == 0:

                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'X_' + str('%.2f' % (self.shiftX)) + '_'\
                    'Y_' + str('%.2f' % (self.shiftY)) + '_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mN_' + str('%.2f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.2f' % (val+1.))
            else:
                self.misRegName = 'rot_' + str('%.2f' % self.rotationAngle) + '_'\
                    'sX_' + str('%.2f' % (self.shiftX)) + '_'\
                    'sY_' + str('%.2f' % (self.shiftY)) + '_'\
                    'anam_' + str('%.2f' % self.anamorphosisAngle) + '_'\
                    'mN_' + str('%.4f' % (self.radialScaling+1.)) + '_'\
                    'mT_' + str('%.4f' % (val+1.))

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
        str_prop = str()
        n_char = len(max(self.prop.values()))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{"Misregistration":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table
