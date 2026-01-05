# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:56:59 2020

@author: cheritie
"""
import os
import sys
import numpy as np

try:
    if os.get_terminal_size()[0] >= 79:
        print('\n')
        print('================================================================================')
        print('   ✸       *          °          *      *                                      ')
        print('        °   ✸         ▄██▄   ▄██▄  ▄███▄   ▄██▄ * ▄██▄    >           ▄▄▄▄     ')
        print('  ✸            °     ██* ██ ██  ██ ██  ██ ██  ██ ██  ██   ===>     ▄█▀▀  ▀▀█▄  ')
        print('   *   °    ✸        ██  ██ ██° ██ ██  ██ ██* ██ ██  ██   =>      █▀ ▄█▀▀█▄ ▀█ ')
        print('✸    *           °   ██  ██ ██  ██ ████▀  ██▄▄██ ██  ██   ====>  █▀ █▀ ▄▄ ▀█ ▀█')
        print('           ✸   °     ██* ██ ██  ██ ██     ██▀▀██ ██  ██   =====> █▄ █▄ ▀▀ ▄█ ▄█')
        print(' *    ✸     °        ██  ██ ██  ██ ██ *   ██  ██ ██* ██   =>      █▄ ▀█▄▄█▀ ▄█ ')
        print('    °        *    ✸   ▀██▀   ▀██▀  ██   ° ██  ██  ▀██▀    ==       ▀█▄▄  ▄▄█▀  ')
        print('         ✸       *        *         *                                 ▀▀▀▀     ')
        print('================================================================================')
        print('\n')
    else:
        print('\n')
        print('==================================')
        print('     °          *      *      ')
        print(' ▄██▄   ▄██▄  ▄███▄   ▄██▄ * ▄██▄ ')
        print('██* ██ ██  ██ ██  ██ ██  ██ ██  ██')
        print('██  ██ ██° ██ ██  ██ ██* ██ ██  ██')
        print('██  ██ ██  ██ ████▀  ██▄▄██ ██  ██')
        print('██* ██ ██  ██ ██     ██▀▀██ ██  ██')
        print('██  ██ ██  ██ ██ *   ██  ██ ██* ██')
        print(' ▀██▀   ▀██▀  ██   ° ██  ██  ▀██▀ ')
        print('      *         *             ')
        print('==================================')
        print('\n')
except:
    print('\n')
    print('==================================')
    print('     °          *      *      ')
    print(' ▄██▄   ▄██▄  ▄███▄   ▄██▄ * ▄██▄ ')
    print('██* ██ ██  ██ ██  ██ ██  ██ ██  ██')
    print('██  ██ ██° ██ ██  ██ ██* ██ ██  ██')
    print('██  ██ ██  ██ ████▀  ██▄▄██ ██  ██')
    print('██* ██ ██  ██ ██     ██▀▀██ ██  ██')
    print('██  ██ ██  ██ ██ *   ██  ██ ██* ██')
    print(' ▀██▀   ▀██▀  ██   ° ██  ██  ▀██▀ ')
    print('      *         *             ')
    print('==================================')
    print('\n')

OOPAO_path = [s for s in sys.path if "OOPAO" in s]
l = []
for i in OOPAO_path:
    l.append(len(i))
path = OOPAO_path[np.argmin(l)]
np.save(path+'/precision_oopao', 64)

show_message = True
message = '''Significant changes were done to the OOPAO repository, the Telescope class is no longer the "master" class and the Source is now carrying the EM-field info.'''
    
from OOPAO.tools.tools import warning

warning(message)


