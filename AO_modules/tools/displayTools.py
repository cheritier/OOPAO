# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:33:20 2020

@author: cheritie
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def update_line(hl, new_dataX,new_dataY):
    
    
    hl.set_xdata(new_dataX)
    hl.set_ydata(new_dataY)
    
def displayMap(A,norma=False,axis=2,mask=0,returnOutput = False): 
    A = np.squeeze(A)
    
    if axis==0:
        A=np.moveaxis(A,0,-1)
    dimA = np.ndim(A)
        
    if dimA == 2:
        n1,n2 = A.shape
        if n1 == n2:
            plt.figure()
            plt.imshow(A)
            return -1

        else:
            if np.math.log(n1,np.sqrt(n1)) == 2.0:
                nImage = n2
                nPix1 = int(np.sqrt(n1))
                nPix2 = nPix1
                A = np.reshape(A,[nPix1,nPix2,nImage])
    else:
        if dimA==3:
            n1,n2,n3 = A.shape
            nImage = n3
            nPix1 = n1
            nPix2 = n2
        else:
            print('Error wrong size for the image cube')
            return -1
    
#    Create a meta Map
    
    nSide = int(np.ceil(np.sqrt(nImage)))
        
    S=np.zeros([nPix1*nSide-1,nPix2*nSide-1])
    
    count=0
    for i in range(nSide):
        for j in range(nSide):
            count+=1
            if count <= nImage:
                if np.ndim(mask)==2:
                    tmp = A[:,:,count-1]*mask
                else:
                    tmp = A[:,:,count-1] 
                if norma:
                    tmp = tmp/np.max(np.abs(tmp))
                S[ i*(nPix1-1) : nPix1 +i*(nPix1-1) , j*(nPix2-1) : nPix2 +j*(nPix2-1)] = tmp
    
    
    plt.figure()
    plt.imshow(S)
    if returnOutput:
        return S

def makeSquareAxes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

            
        
def displayPyramidSignals(wfs,signals,returnOutput=False, norma = False):
    
    A= np.zeros(wfs.validSignal.shape)
    print(A.shape)
    A[:]=np.Inf
    # one signal only
    if np.ndim(signals)==1:
        if wfs.validSignal.sum() == signals.shape:
            A[np.where(wfs.validSignal==1)]=signals
        plt.figure()
        plt.imshow(A)
        out =A
    else:
        B= np.zeros([wfs.validSignal.shape[0],wfs.validSignal.shape[1],signals.shape[1]])
        B[:]=np.Inf
        if wfs.validSignal.shape[0] == wfs.validSignal.shape[1]:
            B[wfs.validSignal,:]=signals
            out = displayMap(B,returnOutput=True)
        else:
            for i in range(signals.shape[1]):
                A[np.where(wfs.validSignal==1)]=signals[:,i]
                if norma:
                    A/= np.max(np.abs(signals[:,i]))
                B[:,:,i] = A
            out = displayMap(B,returnOutput=True)
    if returnOutput:
        return out
        


def interactive_plot(x,y,im_array, event_name ='button_press_event', n_fig = None):   
    # create figure and plot scatter
    if n_fig is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    else:
        fig = plt.figure(n_fig)        
        ax = plt.subplot(111)

    line, = ax.plot(x,y, ls="", marker="o", markersize = 10)
    
    # create the annotations box
    im = OffsetImage(im_array[0,:,:], zoom=2)
    xybox=(100., 100.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    
    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            ind, = line.contains(event)[1]["ind"]
            # get the figure size
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy =(x[ind], y[ind])
            # set the image corresponding to that point
            im.set_data(im_array[ind,:,:])

        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()
    
    # add callback for mouse moves
    fig.canvas.mpl_connect(event_name, hover)           
    plt.show()
        
def interactive_plot_text(x,y,text_array, event_name ='button_press_event'):   
    # create figure and plot scatter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    sc = plt.scatter(x,y)
    
    # create the annotations box
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
    
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                               " ".join([text_array[n] for n in ind["ind"]]))
        annot.set_text(text)
#        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)
    
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    # add callback for mouse moves
    fig.canvas.mpl_connect(event_name, hover)           
    plt.show()
    

    
    
    