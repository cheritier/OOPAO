# -*- coding: utf-8 -*-
"""
Created on February 2026

@author: RafaelSalgueiro58 & cheritie
"""

# imports
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_image
from OOPAO.tools.tools import OopaoError


class FieldTransformer:
    def __init__(self,
                 src,
                 shift_x=None,
                 shift_y=None,
                 rotation_angle=None,
                 anamorphosisAngle=None,
                 tangentialScaling=None,
                 radialScaling=None):
        """
        FieldTransformer object allows to perform a geometrical transformation to the Electromagnetic field (EMF),
        i.e., both phase and amplitude.
        It can be applied both for single sources or asterisms with several sources
        This object allows simulating Super Resolution, namely with SH wavefront sensors, by shifting each EMF before
        propagating towards the SH wfs

        Parameters
        ----------

        shift_x : list
        List containing the shits in the direction x for each guide star considered in units of pixels
        Sub-pixel shifts are possible (0.5, 0.25 or other)

        shift_y : list
        List containing the shits in the direction y for each guide star considered in units of pixels
        Sub-pixel shifts are possible (0.5, 0.25 or other)

        rotation_angle : list
        List containing the angular rotations for each guide star considered in units of degrees

        anamorphosisAngle: list
        List containing the anamorphosis angle for each guide star considered in units of degrees

        tangentialScaling: list
        List containing the tangential scaling for each guide star considered

        radialScaling: list
        List containing the radial scaling for each guide star considered

        ************************** EXEMPLE **************************

        shift = Shift(shift_x,shift_y,rotation_angle)

        ast**atm*tel*shift

        --> EM field is propagated from ast --> atm --> tel, and then the EM field from each guide star of the asterism
        is properly shifted and/rotated according to the given shift_x, shift_y and rotation_angle given as input.

        """
        if src.tag == 'source':
            src_list = [src]
        elif src.tag == 'asterism':
            src_list = src.src
        parameters = ['shift_x',
                      'shift_y',
                      'rotation_angle',
                      'anamorphosisAngle',
                      'tangentialScaling',
                      'radialScaling']
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.rotation_angle = rotation_angle
        self.anamorphosisAngle = anamorphosisAngle
        self.tangentialScaling = tangentialScaling
        self.radialScaling = radialScaling
        # initialize values
        for i_param in parameters:
            tmp_param = getattr(self, i_param)
            if tmp_param is None:
                setattr(self, i_param, [0]*len(src_list))
            if len(getattr(self, i_param)) != len(src_list):
                raise OopaoError('The transformation vector for each parameter length must be equal to the number of sources')
        self.tag = 'FieldTransformer'
        print(self)
        return

    def relay(self, src):
        self.src = src
        if src.tag == 'source':
            self.src_list = [src]
        elif src.tag == 'asterism':
            self.src_list = src.src
            # verify the given lists have all the same length (equal to the number of guide stars)
            lists = [self.shift_x,
                     self.shift_y,
                     self.rotation_angle,
                     self.anamorphosisAngle,
                     self.tangentialScaling,
                     self.radialScaling]

            lengths = {len(lst) for lst in lists}
            if lengths != {len(self.src_list)}:
                raise OopaoError(
                    'The transformation vector for each parameter length must be equal to the number of sources')

        for src in self.src_list:
            src.optical_path.append([self.tag, self])
            # shift the EM field amplitude (fluxMap)
            src.fluxMap = interpolate_image(image_in=src.fluxMap.copy(),
                                            pixel_size_in=1.0,
                                            pixel_size_out=1.0,
                                            resolution_out=len(src.fluxMap.copy()),
                                            rotation_angle=self.rotation_angle[src.ast_idx],
                                            shift_x=self.shift_x[src.ast_idx],
                                            shift_y=self.shift_y[src.ast_idx],
                                            anamorphosisAngle=self.anamorphosisAngle[src.ast_idx],
                                            tangentialScaling=self.tangentialScaling[src.ast_idx],
                                            radialScaling=self.radialScaling[src.ast_idx],
                                            shape_out=None,
                                            order=1)

            # shift the EM OPD and phase
            # by updating the OPD, the phase (and OPD_no_pupil) will be also updated accordingly automatically
            src.OPD = interpolate_image(image_in=src.OPD.copy(),
                                        pixel_size_in=1.0,
                                        pixel_size_out=1.0,
                                        resolution_out=len(src.OPD.copy()),
                                        rotation_angle=self.rotation_angle[src.ast_idx],
                                        shift_x=self.shift_x[src.ast_idx],
                                        shift_y=self.shift_y[src.ast_idx],
                                        anamorphosisAngle=self.anamorphosisAngle[src.ast_idx],
                                        tangentialScaling=self.tangentialScaling[src.ast_idx],
                                        radialScaling=self.radialScaling[src.ast_idx],
                                        shape_out=None,
                                        order=1)

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['shift_x'] = f"{'Shift_x [pix.]':<25s}|{self.shift_x}"
        self.prop['shift_y'] = f"{'Shift_y [pix.]':<25s}|{self.shift_y}"
        self.prop['rotation_angle'] = f"{'Rotation ang. [deg.]':<25s}|{self.rotation_angle}"
        self.prop['anamorphosisAngle'] = f"{'Anamorphosis ang. [deg.]':<25s}|{self.anamorphosisAngle}"
        self.prop['tangentialScaling'] = f"{'Tangential scaling':<25s}|{self.tangentialScaling}"
        self.prop['radialScaling'] = f"{'Radial scaling':<25s}|{self.radialScaling}"
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" EM field Transformer ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table
