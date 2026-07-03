# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:03:22 2024

@author: cheritier
"""
import numpy as np

def compress_rama_data(frame,n_pix,n_pix_crop = None):
    sum_input = frame.sum()
    Q1 = frame[:n_pix,:n_pix]
    Q2 = frame[:n_pix,-n_pix:]
    Q3 = frame[-n_pix:,:n_pix]
    Q4 = frame[-n_pix:,-n_pix:]
    output_frame =  np.vstack((np.hstack((Q1,Q2)),np.hstack((Q3,Q4))))
    if n_pix_crop is not None:
        output_frame = output_frame[n_pix_crop:-n_pix_crop,n_pix_crop:-n_pix_crop]

    # if output_frame.sum()!=sum_input:
    #     print(output_frame.sum())
    #     print(sum_input)        
    #     raise OopaoError('You are missing valid pixel! select a larger n_pix_crop or n_pix value')
    return output_frame


def compute_rama_frame(signal,valid_signal):
    data = np.zeros(valid_signal.shape)
    data[valid_signal] = signal
    return data
