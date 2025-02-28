# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:02 2020

@author: cheritie
"""

def initializeParameterFile():
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.06                                          # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                            # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [0.45, 0.1, 0.1, 0.25, 0.1]                   # Cn2 profile
    param['windSpeed'            ] = [6, 8, 7, 11, 16]                             # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0, 72, 144, 216, 288]                        # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0, 1000, 5000, 10000, 12000]                 # altitude of the different layers in [m]
                    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 1.52                                          # diameter in [m]
    param['nSubaperture'         ] = 20                                            # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 4                                             # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap'] # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']       # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/1500                                         # loop sampling time in [s]
    param['centralObstruction'   ] = 0.00                                          # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                             # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 0.2*0.33*0.5*0.8                              # reflectivity of the 798 segments
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 2                                             # magnitude of the guide star
    param['opticalBand'          ] = 'R'                                           # optical band of the guide star
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = 16                                            # number of actuators 
    param['mechanicalCoupling'   ] = 0.45
    param['isM4'                 ] = False                                         # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                          # tag for the deformable mirror class
    
    # mis-registrations                                                             
    param['shiftX'               ] = 0.006                                         # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = -0.003                                        # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0.69                                          # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                             # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = -0.049                                        # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = -0.041                                        # tangential scaling in percentage of diameter

    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = 5                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 4                                             # separation ratio between the PWFS pupils
    param['n_pix_edge'            ] = 2
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'slopesMaps'                                  # post-processing of the PWFS signals 
    
    return param
