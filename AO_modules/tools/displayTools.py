# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:33:20 2020

@author: cheritie
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from AO_modules.tools.tools import  emptyClass
import matplotlib.gridspec as gridspec
import matplotlib as mpl

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


def getColorOrder():
    color = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
    return color
            
        
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
        

def display_wfs_signals(wfs,signals,returnOutput=False, norma = False):
    
    if wfs.tag == 'pyramid':
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
    if wfs.tag == 'shackHartmann':
        A= np.zeros(wfs.valid_slopes_maps.shape)
        print(A.shape)
        A[:]=np.Inf
        # one signal only
        if np.ndim(signals)==1:
            if wfs.valid_slopes_maps.sum() == signals.shape:
                A[np.where(wfs.valid_slopes_maps==1)]=signals
            plt.figure()
            plt.imshow(A)
            out =A
        else:
            B= np.zeros([wfs.valid_slopes_maps.shape[0],wfs.valid_slopes_maps.shape[1],signals.shape[1]])
            B[:]=np.Inf
            if wfs.valid_slopes_maps.shape[0] == wfs.valid_slopes_maps.shape[1]:
                B[wfs.valid_slopes_maps,:]=signals
                out = displayMap(B,returnOutput=True)
            else:
                for i in range(signals.shape[1]):
                    A[np.where(wfs.valid_slopes_maps==1)]=signals[:,i]
                    if norma:
                        A/= np.max(np.abs(signals[:,i]))
                    B[:,:,i] = A
                out = displayMap(B,returnOutput=True)
        if returnOutput:
            return out

def interactive_plot(x,y,im_array, im_array_ref, event_name ='button_press_event', n_fig = None):   
    # create figure and plot scatter
    if n_fig is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    else:
        fig = plt.figure(n_fig)        
        ax = plt.subplot(111)

    line, = ax.plot(x,y, ls="",color = 'k', marker="o", markersize = 10)
    
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
            if event.button == 1:
                im.set_data(im_array[ind,:,:])
            if event.button == 3:
                im.set_data(im_array_ref[ind,:,:])
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
    

def compute_gif(cube, name, vect = None, vlim = None):
    from matplotlib import animation, rc
    rc('animation', html='html5')    
    data = cube.copy()
    
    plt.close('all')
    fig, ax = plt.subplots(figsize = [5,5])
    line = ax.imshow(np.fliplr(np.flip(data[0,:,:].T)))
    # fig.set_facecolor((0.94,0.85,0.05))
    # line.set_clim([-4,1])
    plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left = False,
            labelbottom=False,
            labelleft=False)
    plt.tight_layout()
    # SR =ax.text(50, 400,'SR: '+str(np.round(ao_res[20],1))+'%',color=(1,1,1),fontsize = 14,weight = 'bold')

    def init():
        tmp = np.copy(data[0,:,:])
        tmp[np.where(tmp==0)] = np.inf
        line.set_data(np.fliplr(np.flip(tmp.T)))
        # ax.set_title('Time '+str(0) + ' ms -- WFE '+str(wfe[0])+' nm')
        line.set_clim(vmin = np.min(data[0,:,:]), vmax = np.max(data[0,:,:]))
        return (line,)
        
        
        # animation function. This is called sequentially
    def animate(i):
        tmp = np.copy(data[i,:,:])
        tmp[np.where(tmp==0)] = np.inf
        line.set_data(np.fliplr(np.flip(tmp.T)))
        # SR.set_text('SR: '+str(np.round(ao_res[i],1))+'%')
        line.set_clim(vmin = np.min(data[i,:,:]), vmax = np.max(data[i,:,:]))
        return (line)
    # have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=data.shape[0], interval=100)
    
    folder = '//winhome/home/cheritie/My Pictures/gif_from_python/'
    anim.save(folder+name+'.gif', writer='imagemagick', fps=2)
    return

def cl_plot(list_fig,plt_obj= None, type_fig = None,fig_number = 20, list_ratio = None, list_title = None):
    
    n_im = len(list_fig)
    n_sp = int(np.ceil(np.sqrt(n_im)))
    n_add = 0
    if (n_sp**2-n_im)<2:
        n_add = 1
    n_sp_y = n_sp + n_add
    
    if plt_obj is None:
        if list_ratio is None:
            gs = gridspec.GridSpec(n_sp_y,n_sp, height_ratios=np.ones(n_sp_y), width_ratios=np.ones(n_sp), hspace=0.25, wspace=0.25)
        else:
            gs = gridspec.GridSpec(n_sp_y,n_sp, height_ratios=list_ratio[0], width_ratios=list_ratio[1], hspace=0.25, wspace=0.25)
        
        plt_obj = emptyClass()
        setattr(plt_obj,'gs',gs)
        plt_obj.keep_going = True
        f = plt.figure(fig_number)
        
        line_comm = ['Stop','Pause','Continue']
        col_comm = ['r','b','b']
        
        for i in range(2):
            
            setattr(plt_obj,'ax_0_'+str(i+1), plt.subplot(gs[n_sp_y-1,n_sp-2+i]))
            
            sp_tmp =getattr(plt_obj,'ax_0_'+str(i+1)) 
            
            annot = sp_tmp.annotate(line_comm[i],color ='k', fontsize=25, xy=(0.5,0.5), xytext=(0.5,0.5),bbox=dict(boxstyle="round", fc=col_comm[i]))
            setattr(plt_obj,'annot_'+str(i+1), annot)
        
            plt.axis('off')
        

        count = 0
        for i in range(n_sp):
            for j in range(n_sp):
                if count < n_im:
                    print(count)
                    setattr(plt_obj,'ax_'+str(count), plt.subplot(gs[i,j]))
                    sp_tmp =getattr(plt_obj,'ax_'+str(count))            

                    setattr(plt_obj,'type_fig_'+str(count),type_fig[count])
           # IMSHOW
                    if type_fig[count] == 'imshow':
                        data_tmp = list_fig[count]
                        if len(data_tmp)==3:
                            setattr(plt_obj,'im_'+str(count),sp_tmp.imshow(data_tmp[2],extent = [data_tmp[0][0],data_tmp[0][1],data_tmp[1][0],data_tmp[1][1]]))        
                        else:
                            setattr(plt_obj,'im_'+str(count),sp_tmp.imshow(data_tmp))        
                            

                        im_tmp =getattr(plt_obj,'im_'+str(count))
                        plt.colorbar(im_tmp)
           # PLOT     
                    if type_fig[count] == 'plot':
                        data_tmp = list_fig[count]
                        if len(data_tmp)==2:
                            line_tmp, = sp_tmp.plot(data_tmp[0],data_tmp[1],'-x')
                        else:
                            line_tmp, = sp_tmp.plot(data_tmp,'-o')                                
                        setattr(plt_obj,'im_'+str(count),line_tmp)        
           # SCATTER
                    if type_fig[count] == 'scatter':
                        data_tmp = list_fig[count]
                        
                        scatter_tmp = sp_tmp.scatter(data_tmp[0],data_tmp[1],c=data_tmp[2],marker = 'h', s =16)
                        setattr(plt_obj,'im_'+str(count),scatter_tmp)  
                        makeSquareAxes(plt.gca())
                        plt.colorbar(scatter_tmp)
                        plt.axis('off')
                    if list_title is not None:
                        plt.title(list_title[count])

                count+=1
                
        def hover(event):
            if event.inaxes == plt_obj.ax_0_1:
                cont, ind = f.contains(event)        
                if cont:
                    plt_obj.keep_going = False
                    plt_obj.annot_1.set_fontweight('bold')
                    plt_obj.annot_1.set_text('Stopped')
          
                        
            if event.inaxes == plt_obj.ax_0_2:
                cont, ind = f.contains(event)        
                if cont:
                    plt_obj.annot_2.set_text('Pause')
                    plt_obj.annot_2.set_backgroundcolor('g')
                    plt_obj.annot_2.set_fontweight('bold')
                    plt.waitforbuttonpress()
        f.canvas.mpl_connect('button_press_event', hover)   
        return plt_obj
    if plt_obj is not None:
        count = 0
        for i in range(n_sp):
            for j in range(n_sp):
                if count < n_im:
                    data = list_fig[count]
                    if getattr(plt_obj,'type_fig_'+str(count)) == 'imshow':
                        im_tmp =getattr(plt_obj,'im_'+str(count))
                        im_tmp.set_data(data)
                        im_tmp.set_clim(vmin=data.min(),vmax=data.max())
                    if getattr(plt_obj,'type_fig_'+str(count)) == 'plot':
                        if len(data)==2:
                            im_tmp =getattr(plt_obj,'im_'+str(count))
                            im_tmp.set_xdata(data[0])
                            im_tmp.set_ydata(data[1])
                            im_tmp.axes.set_ylim([np.min(data[1])-0.1*np.abs(np.min(data[1])),np.max(data[1])+0.1*np.abs(np.max(data[1]))])
                            im_tmp.axes.set_xlim([np.min(data[0])-0.1*np.abs(np.min(data[0])),np.max(data[0])+0.1*np.abs(np.max(data[0]))])

                        else:
                            im_tmp =getattr(plt_obj,'im_'+str(count))
                            im_tmp.set_ydata(data[0])
                            im_tmp.axes.set_ylim([np.min(data[0])-0.1*np.abs(np.min(data[0])),np.max(data[0])+0.1*np.abs(np.max(data[0]))])
                                                
                    if getattr(plt_obj,'type_fig_'+str(count)) == 'scatter':
                        n = mpl.colors.Normalize(vmin = min(data), vmax = max(data))
                        m = mpl.cm.ScalarMappable(norm=n, cmap=mpl.cm.afmhot)

                        im_tmp =getattr(plt_obj,'im_'+str(count))
                        im_tmp.set_facecolor(m.to_rgba(data))
                        im_tmp.set_clim(vmin=min(data), vmax=max(data))
                    plt.draw()
    
                count+=1
    
    