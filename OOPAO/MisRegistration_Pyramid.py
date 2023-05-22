# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:05:05 2022

@author: cheritie
"""


import numpy as np
import inspect

class MisRegistration_Pyramid:
    def __init__(self,sx=None,sy=None):
        
        self.tag                  = 'misRegistration_Pyramid' 
        self.isInitialized = False  

        if sx is None:
            self.dX_1        = 0                    # rotation angle in degrees
            self.dX_2        = 0                           # shift X in m
            self.dX_3        = 0                           # shift Y in m
            self.dX_4        = 0                # amamorphosis angle in degrees
        else:
            self.dX_1        = sx[0]                    # rotation angle in degrees
            self.dX_2        = sx[1]                            # shift X in m
            self.dX_3        = sx[2]                            # shift Y in m
            self.dX_4        = sx[3]                 # amamorphosis angle in degrees
            
        if sy is None:
            self.dY_1        = 0                 # normal scaling in % of diameter
            self.dY_2        = 0                    # radial scaling in % of diameter
            self.dY_3        = 0                    # radial scaling in % of diameter
            self.dY_4        = 0                     # radial scaling in % of diameter
        else:
            self.dY_1        = sy[0]                 # normal scaling in % of diameter
            self.dY_2        = sy[1]                     # radial scaling in % of diameter
            self.dY_3        = sy[2]                    # radial scaling in % of diameter
            self.dY_4        = sy[3]                     # radial scaling in % of diameter
                    
        self.misRegName = 'dX_1'      + str('%.2f' %self.dX_1)            +'_'\
                          'dX_2'      + str('%.2f' %(self.dX_2))                 +'_'\
                          'dX_3'      + str('%.2f' %(self.dX_3))                 +'_'\
                          'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
                          'dY_1'      + str('%.2f' %self.dY_1)            +'_'\
                          'dY_2'      + str('%.2f' %(self.dY_2))                 +'_'\
                          'dY_3'      + str('%.2f' %(self.dY_3))                 +'_'\
                          'dY_4'      + str('%.2f' %self.dY_4)
                            
        self.isInitialized = True

#        mis-registrations can be added or sub-tracted using + and -       
    def __add__(self,misRegObject):
        if misRegObject.tag == 'misRegistration_Pyramid':
            tmp = MisRegistration_Pyramid()
            tmp.dX_1    = self.dX_1   + misRegObject.dX_1
            tmp.dX_2    = self.dX_2   + misRegObject.dX_2
            tmp.dX_3    = self.dX_3   + misRegObject.dX_3
            tmp.dX_4    = self.dX_4   + misRegObject.dX_4
            tmp.dY_1    = self.dY_1   + misRegObject.dY_1
            tmp.dY_2    = self.dY_2   + misRegObject.dY_2
            tmp.dY_3    = self.dY_3   + misRegObject.dY_3
            tmp.dY_4    = self.dY_4   + misRegObject.dY_4
        else:
            print('Error you are trying to combine a MisRegistration Object with the wrong type of object')
        return tmp
    
    def __sub__(self,misRegObject):
        if misRegObject.tag == 'misRegistration_Pyramid':
            tmp = MisRegistration_Pyramid()
            tmp.dX_1    = self.dX_1   - misRegObject.dX_1
            tmp.dX_2    = self.dX_2   - misRegObject.dX_2
            tmp.dX_3    = self.dX_3   - misRegObject.dX_3
            tmp.dX_4    = self.dX_4   - misRegObject.dX_4
            tmp.dY_1    = self.dY_1   - misRegObject.dY_1
            tmp.dY_2    = self.dY_2   - misRegObject.dY_2
            tmp.dY_3    = self.dY_3   - misRegObject.dY_3
            tmp.dY_4    = self.dY_4   - misRegObject.dY_4
        else:
            print('Error you are trying to combine a MisRegistration Object with the wrong type of object')
        return tmp
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
# ----------------------------------------- Properties  -----------------------------------------
    # @property
    # def dX_1(self):
    #     return self._dX_1
        
    # @dX_1.setter
    # def dX_1(self,val):
    #     self._dX1  = val
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %val)            +'_'\
    #               'dX_2'             + str('%.2f' %(self.dX_2))                 +'_'\
    #               'dX_3'             + str('%.2f' %(self.dX_3))                 +'_'\
    #               'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
    #               'dY_1'       + str('%.2f' %self.dY_1)            +'_'\
    #               'dY_2'             + str('%.2f' %(self.dY_2))                 +'_'\
    #               'dY_3'             + str('%.2f' %(self.dY_3))                 +'_'\
    #               'dY_4'      + str('%.2f' %self.dY_4)
         
    # @property
    # def dX_2(self):
    #     return self._dX2
        
    # @dX_2.setter
    # def dX_2(self,val):
    #     self._dX_2  = val
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %self.dX_1)            +'_'\
    #               'dX_2'             + str('%.2f' %(val))                 +'_'\
    #               'dX_3'             + str('%.2f' %(self.dX_3))                 +'_'\
    #               'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
    #               'dY_1'       + str('%.2f' %self.dY_1)            +'_'\
    #               'dY_2'             + str('%.2f' %(self.dY_2))                 +'_'\
    #               'dY_3'             + str('%.2f' %(self.dY_3))                 +'_'\
    #               'dY_4'      + str('%.2f' %self.dY_4)        

    
    # @property
    # def dX_3(self):
    #     return self._dX_3
    
    # @dX_3.setter
    # def dX_3(self,val):
    #     self._dX_3  = val  
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %self.dX_1)            +'_'\
    #               'dX_2'             + str('%.2f' %(self.dX_2))                 +'_'\
    #               'dX_3'             + str('%.2f' %(val))                 +'_'\
    #               'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
    #               'dY_1'       + str('%.2f' %self.dY_1)            +'_'\
    #               'dY_2'             + str('%.2f' %(self.dY_2))                 +'_'\
    #               'dY_3'             + str('%.2f' %(self.dY_3))                 +'_'\
    #               'dY_4'      + str('%.2f' %self.dY_4)
    # @property
    # def dX_4(self):
    #     return self._dX_4
    
    # @dX_4.setter
    # def dX_4(self,val):
    #     self._dX_4  = val                    
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %self.dX_1)            +'_'\
    #               'dX_2'             + str('%.2f' %(self.dX_2))                 +'_'\
    #               'dX_3'             + str('%.2f' %(self.dX_3))                 +'_'\
    #               'dX_4'      + str('%.2f' %val)        +'_'\
    #               'dY_1'       + str('%.2f' %self.dY_1)            +'_'\
    #               'dY_2'             + str('%.2f' %(self.dY_2))                 +'_'\
    #               'dY_3'             + str('%.2f' %(self.dY_3))                 +'_'\
    #               'dY_4'      + str('%.2f' %self.dY_4)
    # @property
    # def dY_1(self):
    #     return self._dY_1
        
    # @dY_1.setter
    # def dY_1(self,val):
    #     self._dY1  = val
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %self.dX_1)            +'_'\
    #               'dX_2'             + str('%.2f' %(self.dX_2))                 +'_'\
    #               'dX_3'             + str('%.2f' %(self.dX_3))                 +'_'\
    #               'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
    #               'dY_1'       + str('%.2f' %val)            +'_'\
    #               'dY_2'             + str('%.2f' %(self.dY_2))                 +'_'\
    #               'dY_3'             + str('%.2f' %(self.dY_3))                 +'_'\
    #               'dY_4'      + str('%.2f' %self.dY_4)        
    # @property
    # def dY_2(self):
    #     return self._dY2
        
    # @dY_2.setter
    # def dY_2(self,val):
    #     self._dY_2  = val
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %self.dX_1)            +'_'\
    #               'dX_2'             + str('%.2f' %(self.dX_2))                 +'_'\
    #               'dX_3'             + str('%.2f' %(self.dX_3))                 +'_'\
    #               'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
    #               'dY_1'       + str('%.2f' %self.dY_1)            +'_'\
    #               'dY_2'             + str('%.2f' %(val))                 +'_'\
    #               'dY_3'             + str('%.2f' %(self.dY_3))                 +'_'\
    #               'dY_4'      + str('%.2f' %self.dY_4)
    
    # @property
    # def dY_3(self):
    #     return self._dY_3
    
    # @dY_3.setter
    # def dY_3(self,val):
    #     self._dY_3  = val  
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %self.dX_1)            +'_'\
    #               'dX_2'             + str('%.2f' %(self.dX_2))                 +'_'\
    #               'dX_3'             + str('%.2f' %(self.dX_3))                 +'_'\
    #               'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
    #               'dY_1'       + str('%.2f' %self.dY_1)            +'_'\
    #               'dY_2'             + str('%.2f' %(self.dY_2))                 +'_'\
    #               'dY_3'             + str('%.2f' %(val))                 +'_'\
    #               'dY_4'      + str('%.2f' %self.dY_4)
    # @property
    # def dY_4(self):
    #     return self._dY_4
    
    # @dY_4.setter
    # def dY_4(self,val):
    #     self._dY_4  = val          
    #     if self.isInitialized:
    #         self.misRegName = 'dX_1'       + str('%.2f' %self.dX_1)            +'_'\
    #               'dX_2'             + str('%.2f' %(self.dX_2))                 +'_'\
    #               'dX_3'             + str('%.2f' %(self.dX_3))                 +'_'\
    #               'dX_4'      + str('%.2f' %self.dX_4)        +'_'\
    #               'dY_1'       + str('%.2f' %self.dY_1)            +'_'\
    #               'dY_2'             + str('%.2f' %(self.dY_2))                 +'_'\
    #               'dY_3'             + str('%.2f' %(self.dY_3))                 +'_'\
    #               'dY_4'      + str('%.2f' %val)                  
                  
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def print_(self):
            print('{: ^10s}'.format('dX_1 [pix]') + '\t' + '{: ^10s}'.format('dX_2 [pix]')+ '\t' + '{: ^10s}'.format('dX_3 [pix]')+ '\t' + '{: ^10s}'.format('dX_4 [pix]'))
            print("{: ^10s}".format(str(self.dX_1) )  + '\t' +'{: ^10s}'.format(str(self.dX_2))+'\t' + '{: ^10s}'.format(str(self.dX_3)) +'\t' +'{: ^10s}'.format(str(self.dX_4)))
            print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('{: ^10s}'.format('dY_1 [pix]') + '\t' + '{: ^10s}'.format('dY_2 [pix]')+ '\t' + '{: ^10s}'.format('dY_3 [pix]')+ '\t' + '{: ^10s}'.format('dY_4 [pix]'))
            print("{: ^10s}".format(str(self.dY_1) )  + '\t' +'{: ^10s}'.format(str(self.dY_2))+'\t' + '{: ^10s}'.format(str(self.dY_3)) +'\t' +'{: ^10s}'.format(str(self.dY_4)))
    def show(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        print(self.tag+':')
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                if not(a[0].startswith('_')):
                    if not np.shape(a[1]):
                        tmp=a[1]
                        try:
                            print('          '+str(a[0])+': '+str(tmp.tag)+' object') 
                        except:
                            print('          '+str(a[0])+': '+str(a[1])) 
                    else:
                        if np.ndim(a[1])>1:
                            print('          '+str(a[0])+': '+str(np.shape(a[1])))    
    def __repr__(self):
        self.print_()
        return ' '