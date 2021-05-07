# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:30:29 2021

@author: cheritie
"""

import numpy as np

def get_slaved_m4(dm):

    def get_segment_edge_actuators(n_petal):
        edges_index = np.zeros([n_petal,2,30])
        
        for i_spider in range(n_petal):
            edge_act_left = []
            edge_act_right = []
            count2 = 0+i_spider*892
            count = 3+i_spider*892
            for i in range(30):
                if i ==0:
                    added = 0
                    added2 = 0
                else:
                    added = 10+i
                    if i==1:
                        added2 = 9
                    else:
                        if i ==2:
                            added2 = 7+i-1
                        else:
                            if i==3:
                                added2 = 8+i-1
                            else:
                                added2 = 9+i
                count = count+added
                count2 = count2+added2
                edge_act_left.append(count)
                edge_act_right.append(count2)
            edges_index[i_spider,0,:] = np.asarray(edge_act_left)
            edges_index[i_spider,1,:] = np.asarray(edge_act_right)
            
       
        return edges_index.astype(int)
                      
    
    # get indexes for petals on the edges
    edge_index = get_segment_edge_actuators(n_petal= 6)
    
    
    index_petal_order = [0,1,2,3,4,5,0]
    index_side_order  = [1,0]
                         
    slaved_IF    = []

    for i in range(6):    
        index_1 = edge_index[index_petal_order[i],index_side_order[1]]    
        index_2 = edge_index[index_petal_order[i+1],index_side_order[0]]
        for j in range(len(index_1)):
            slaved = dm.modes[:,index_1[j]] + dm.modes[:,index_2[j]]
            slaved_IF.append(slaved)
    
    
    full_list    = np.reshape(edge_index, 6*2*30)
    act_rem_edge = []
    
    for i in range(5352):
        if (i not in full_list):
            act_rem_edge.append(i)
    
    normal_IF =  dm.modes[:,act_rem_edge]       
    
    slaved_IF = np.asarray(slaved_IF)
    
    print('Replacing the dm influence functions by slaved influence functions along the spiders...')
    
    dm.modes     = np.concatenate([normal_IF.T,slaved_IF]).T                    
    dm.nAct      = dm.modes.shape[1]
    dm.nValidAct = dm.modes.shape[1]
    
    return dm 