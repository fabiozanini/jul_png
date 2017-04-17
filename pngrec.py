# vim: fdm=indent
'''
author:     Fabio Zanini
date:       03/04/17
content:    Try to reconstruct PNGs iteratively.
'''
# Modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import dilation


# Script
if __name__ == '__main__':

    # Load matlab image
    fn = '091126_Si541_01_1_R3D_1_D3D_1.png'
    im = np.array(Image.open(fn))
    shape = im.shape[:2]
    dpi = 100

    x = np.arange(-5, 25 + 1)
    yr = np.linspace(0, 35, len(x))
    yg = np.linspace(0, 35, len(x))
    ys0 = np.concatenate([yr, yg])

    def prepare_png():
        fig, ax = plt.subplots(
                figsize=(shape[1] * 1.0 / dpi, shape[0] * 1.0 / dpi))
        ax.set_xlim(-5, 25)
        ax.set_ylim(0, 35)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_tick_params(which='both', direction='in')
        ax.xaxis.set_tick_params(which='both', direction='in')
        plt.tight_layout(rect=(0.099, 0.072, 0.925, 0.947))
        return {'fig': fig, 'ax': ax}

    def distance_pngs(ys, fig, ax):
        hr = None
        hg = None
        yr = ys[:len(x)]
        xr = [xi for xi, yri in zip(x, yr) if yri is not None]
        if len(xr):
            if len(xr) == 1:
                xr = [xr[0], xr[0] + 0.1]
                yr = [yr[0], yr[0]]
            hr = ax.plot(xr, yr[:len(xr)], color='#cc0033', lw=2)

        yg = ys[len(x):]
        xg = [xi for xi, ygi in zip(x, yg) if ygi is not None]
        if len(xg):
            if len(xg) == 1:
                xg = [xg[0], xg[0] + 0.1]
                yg = [yg[0], yg[0]]
            hg = ax.plot(xg, yg[:len(xg)], color='#009900', lw=2)

        # Save and reopen duplicate
        from io import BytesIO
        png_bytes = BytesIO()
        fig.savefig(png_bytes, dpi=dpi)
        im_dup = np.array(Image.open(png_bytes))
        if hr is not None:
            hr[0].remove()
        if hg is not None:
            hg[0].remove()

        # Compare them
        im_comp = (255 - im) ^ (255 - im_dup[:, :, :3])
        im_comp_bw = (im_comp // 3).sum(axis=2).astype('uint8')
        obj = im_comp_bw.sum()

        return obj

    def plot_distance_line(ds):
        fig, ax = plt.subplots()
        from matplotlib import cm
        colors = cm.jet(np.linspace(0, 1, len(ds)))
        for d, color in zip(ds, colors):
            x = d['ys']
            y = d['distances']
            ax.plot(x, y, lw=2, color=color)

        plt.ion()
        plt.show()


    # Minimize distance step by step
    from operator import itemgetter
    prep = prepare_png()
    n_points = 100
    ys = [None for y in ys0]
    ds = []
    for i in range(len(ys)):
        print('Minimizing i+1='+str(i+1)+' of '+str(len(ys)))
        dsi = []
        if i == 0:
            y_scans = np.linspace(0, 35, 100)
        else:
            y_scans = np.linspace(0, 35, 100)
        for iy, y_scan in enumerate(y_scans):
            ys[i] = y_scan
            d = {'distance': distance_pngs(ys, fig=prep['fig'], ax=prep['ax']),
                 'y': y_scan}
            dsi.append(d)
        dmin = min(dsi, key=itemgetter('distance'))
        if (i > 0) and (dmin['distance'] > ds[-1]['distance_min'] * 1.1):
            break
        ys[i] = dmin['y']
        dtot = {'distance_min': dmin['distance'],
                'y_min': dmin['y'],
                'distances': [d['distance'] for d in dsi],
                'ys': [d['y'] for d in dsi]}
        ds.append(dtot)

    plot_distance_line(ds)
    yr = ys[:len(x)]
    yg = ys[len(x):]
    print('x:', x)
    print('y red:', yr)
    print('y green:', yg)

    # Plot final figure
    fin = prepare_png()
    fin['ax'].plot(x, yr, lw=2, color='darkred')
    fin['ax'].plot(x, yg, lw=2, color='green')

    plt.ion()
    plt.show()

    sys.exit()

    fn_dup = '/tmp/test.png'
    im_dup = np.array(Image.open(fn_dup))
    im_comp = (255 - im) ^ (255 - im_dup[:, :, :3])
    im_comp_bw = (im_comp // 3).sum(axis=2).astype('uint8')

    # Plot comparison
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 12))
    axs2 = axs2.ravel()
    axs2[0].imshow(im)
    axs2[1].imshow(im_dup)
    axs2[2].imshow((255 - im_comp))
    axs2[3].imshow(im_comp_bw, cmap='Greys')

    plt.ion()
    plt.show()
