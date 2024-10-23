#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:06:24 2024



@author: yhc2080
"""

# Initialize
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator


#%% Class: Figure Base

class PBLplot:
    
    def __init__(self):
        pass
    
    def plot_hovmoller_pollutants(self, pollutant, xc, 
                              casename=None, title=r"Pollutant Hovmöller Diagram", 
                              figsize=[4,6], imgname=None, savefig=True, show=False, 
                              xbnd=[0, 256000], xLocator=3200,
                              tbnd=[0, 720], toffset_min=300, tresolution_min=2, 
                              shade_levels=np.arange(2, 20.1, 2), 
                              shade_cmap='YlOrBr', shade_extend='both', 
                              shade_cbar_label=r"Tracer [a.u.]"):
        """
        Plots Hovmöller diagram for pollutant concentrations.
    
        Parameters
        ----------
        pollutant : 2D-array in (time, x) dimension
            Pollutant concentration for shading. The shape should be (tbnd[1]-tbnd[0]+1, len(xc)).
        xc : 1D-array
            Coordinate of the x-axis grid points [m].
        casename : string, optional
            Casename. Ignore when None.
        title : string, optional
            Main title in upper left. The default is r"Pollutant Hovmöller Diagram".
        figsize : list, optional
            Figure size in inches. The default is [6,4].
        imgname : string, optional
            Filename for savefig if savefig is True (e.g. 'XXX.png'). The default is None.
        savefig : bool, optional
            Whether to save the figure or not. The default is True.
            But no figure is saved when imgname is None.
        show : bool, optional
            Whether to plt.show() or not. The default is False.
        xbnd : list with 2 elements, optional
            The range of x-axis in figure (spatial distance). The default is [0, 2000].
        xLocator : int, optional
            Ticks setting in ax.xaxis.set_major_locator(MultipleLocator(xLocator)). The default is 3200.        
        tbnd : list with 2 elements, optional
            The start and end outputstep number. Determine the range of y-axis 
            in figure (time). The default is [0, 720].
        toffset_min : int, optional
            Start time offset from 00:00LST in minute. The default is 300 for 05:00AM.
        tresolution_min : int, optional
            Time resolution in minute. The default is 2.
        shade_levels : list or 1D-array, optional
            The "levels" for cmap of shaded. The default is np.arange(2, 20.1, 2).
        shade_cmap : string, optional
            The colormap name in matplotib for cmap. The default is 'YlOrBr'.
        shade_extend : string, optional
            The "extend" for cmap and colorbar of shaded. The default is 'both'.
        shade_cbar_label : string, optional
            The "label" for colorbar of shaded. The default is r"Tracer [a.u.]".
        
        Returns
        -------
        None.
        """
    
        # Pre-processing 
        ## for time axis
        tstart = tbnd[0]
        tend = tbnd[1]
        tminute = toffset_min + tstart * tresolution_min
        startHHMM = f"{str(tminute // 60).zfill(2)}:{str(tminute % 60).zfill(2)}"
        time_range = pd.date_range(start=f"2024-01-01 {startHHMM}", periods=(tend - tstart + 1), freq=f"{tresolution_min}T")
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
        ax.set_xlim(xbnd)
        ax.xaxis.set_major_locator(MultipleLocator(xLocator))
        ax.set_ylabel('Local Time')
        ax.set_xlabel('x [m]')
        ax.set_ylim([time_range[0], time_range[-1]])
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.minorticks_on()
    
        ax.set_title(title, loc="left", fontsize=14, weight="bold")
        ax.set_title(casename, loc="right", fontsize=14, weight="bold")
    
        # Plot Shaded: Pollutant concentration
        cmap = plt.cm.get_cmap(shade_cmap, len(shade_levels) + 1)
        norm = BoundaryNorm(shade_levels, ncolors=len(shade_levels) + 1, clip=False, extend=shade_extend)
        PC0 = ax.pcolormesh(xc, time_range, pollutant, cmap=cmap, norm=norm)
    
        # Add Colorbar
        [x0, y0], [x1, y1] = ax.get_position().get_points()
        cax3 = fig.add_axes([x1 + 0.02, y0 + 0.002, 0.015, y1 - y0 - 0.004]) # x0, y0, width, height
        CB3 = plt.colorbar(PC0, cax=cax3, orientation='vertical', extend=shade_extend)
        CB3.set_label(shade_cbar_label, labelpad=0.2, fontsize=12)
    
        # Save or show the figure
        if savefig and imgname:
            plt.savefig(imgname, bbox_inches='tight')
        if show:
            plt.show()    
    
    def plot_vertical_theta_transport(self, PBLheight, wth, zc, 
                                      casename=None, title=r"Vertical $\theta$ transport", 
                                      figsize=[6,4], imgname=None, savefig=True, show=False, 
                                      zbnd=[0, 2000], 
                                      tbnd=[0, 720], toffset_min=300, tresolution_min=2, 
                                      line_color_list=None, line_style_list="-", 
                                      line_width_list=2.5, line_label_list=None, 
                                      shade_levels=np.arange(-0.1, 0.101, 0.01), 
                                      shade_cmap='coolwarm', shade_extend='both', 
                                      shade_cbar_label=r"$\overline{w'\theta'}$ [K]"):
        """
        

        Parameters
        ----------
        PBLheight : 1D-array or list of multiple 1D-array (t,)
            Boundary Layer Height [m] Time Series. Either single one or multiple.
            Each array should have length can fit in argument "tbnd" (defalut: [0, 720]) 
        wth : 2D-array in (z,t) dimension
            Eddy flux for shaded. The shape should identical to (size(zc), (tbnd[1]-tbnd[0]+1)).
        zc : 1D-array
            Coordinate of the height in each grid level.
        casename : string, optional
            Casename. Ignore when None.
        title : string, optional
            Main title in upper left. The default is r"Vertical $\theta$ transport".
        figsize : TYPE, optional
            Figure size in inches. The default is [6,4].
        imgname : TYPE, optional
            Filename for savefig if savefig is True (e.g. 'XXX.png'). The default is None.
        savefig : Bool, optional
            Whether to save figure or not. The default is True.
            But no figure is saved when imgname is None.
        show : Bool, optional
            Whether to plt.show() or not. The default is False.
        zbnd : list with 2 elements, optional
            The range of y-axis in figure (height). The default is [0, 2000].
        tbnd : list with 2 elements, optional
            The start and end outputstep number. Determine the range of x-axis 
            in figure (time). The default is [0, 720].
        toffset_min : int, optional
            Start time offset from 00:00LST in minute. The default is 300 for 05:00AM.
        tresolution_min : int, optional
            Time resolution in minute. The default is 2.
        line_color_list : str or list, optional
            The color for each PBL line. The default is None.
        line_style_list : str or list, optional
            The linestyle for each PBL line. The default is "-".
        line_width_list : str or list, optional
            The linewidth for each PBL line. The default is 2.0.
        line_label_list : str or list, optional
            Labels for each PBL line. The default is None and no legend would be drawn.
        shade_levels : list or 1D-array, optional
            The "levels" for cmap of shaded. The default is np.arange(-0.1, 0.101, 0.01).
        shade_cmap : string, optional
            The colormap name in matplotib for cmap. The default is 'coolwarm'.
        shade_extend : string, optional
            The "extend" for cmap and colorbar of shaded. The default is 'both'.
        shade_cbar_label : string, optional
            The "label" for colorbar of shaded. The default is r"$\overline{w'\theta'}$ [K]".

        Returns
        -------
        None.

        """
    
        # Pre-processing 
        ## for time axis
        tstart = tbnd[0]
        tend = tbnd[1]
        tminute = toffset_min + tstart * tresolution_min
        startHHMM = f"{str(tminute // 60).zfill(2)}:{str(tminute % 60).zfill(2)}"
        time_range = pd.date_range(start=f"2024-01-01 {startHHMM}", periods=(tend - tstart + 1), freq=f"{tresolution_min}T")
        ## for the line data
        line_data, line_color_list, line_style_list, line_width_list, line_label_list = self._linedata_preprocessing(PBLheight, 
                   line_color_list, line_style_list, line_width_list, line_label_list)
        
        ## for the shade data
        shade_data = wth
    
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
        ax.set_ylim(zbnd)
        ax.set_ylabel('z [m]')
        ax.set_xlabel('Local Time')
        ax.set_xlim([time_range[0], time_range[-1]])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.minorticks_on()
    
        ax.set_title(title, loc="left", fontsize=14, weight="bold")
        ax.set_title(casename, loc="right", fontsize=14, weight="bold")
    
        # Plot Lines: BLH
        for data, color, style, width, label in zip(line_data, line_color_list, 
                             line_style_list, line_width_list, line_label_list):
            ax.plot(time_range, data, color=color, ls=style, lw=width, label=label)
    
        # Plot Shaded: wth
        cmap = plt.cm.get_cmap(shade_cmap, len(shade_levels) + 1)
        norm = BoundaryNorm(shade_levels, ncolors=len(shade_levels) + 1, clip=False, extend=shade_extend)
        PC0 = ax.pcolormesh(time_range, zc, shade_data, cmap=cmap, norm=norm)
    
        # Add Colorbar
        [x0, y0], [x1, y1] = ax.get_position().get_points()
        cax3 = fig.add_axes([x1 + 0.02, y0 + 0.002, 0.015, y1 - y0 - 0.004]) # x0, y0, width, height
        CB3 = plt.colorbar(PC0, cax=cax3, orientation='vertical', extend=shade_extend)
        CB3.set_label(shade_cbar_label, labelpad=0.10, fontsize=12)
    
        # Add legend to plot
        if any(label != "" for label in line_label_list):
            ax.legend(loc=2, ncol=1, fontsize=12, borderpad=0.2, handleheight=0.9, handlelength=1.5, 
                      handletextpad=0.3, labelspacing=0.2, columnspacing=1.0, framealpha=0.90)
    
        # Save or show the figure
        if savefig and imgname:
            plt.savefig(imgname, bbox_inches='tight')
        if show:
            plt.show()
            
    def _linedata_preprocessing(self, line_data, line_color_list=None, line_style_list=None, line_width_list=None, line_label_list=None):
        # Step 1: 將 PBLheight 統一處理成 list of arrays
        if isinstance(line_data, list): 
            line_data = line_data
        else:  
            line_data = [line_data]
        
        # Step 2: 根據 line_data (PBLheight) 中 1D array 的數量，賦值 nlines
        nlines = len(line_data)
        
        # Step 3: 處理 line_color_list
        if line_color_list is None:
            cmap = plt.get_cmap('tab10')  # 使用 tab10 colormap
            line_color_list = [cmap(i % 10) for i in range(nlines)]  # 根據 nlines 設定顏色
        elif isinstance(line_color_list, str):  # 如果是字串，將其重複 nlines 次
            line_color_list = [line_color_list] * nlines
    
        # Step 4: 處理 line_style_list 和 line_width_list
        if line_style_list is None:  # 如果未給定，則設為全為 "-" 的 list
            line_style_list = ["-"] * nlines
        elif isinstance(line_style_list, str):  # 如果是字串，將其重複 nlines 次
            line_style_list = [line_style_list] * nlines
    
        if line_width_list is None:  # 如果未給定，則設為全為 2.5 的 list
            line_width_list = [2.5] * nlines
        elif isinstance(line_width_list, (int, float)):  # 如果是數值，將其重複 nlines 次
            line_width_list = [line_width_list] * nlines
    
        # Step 5: 處理 line_label_list
        if line_label_list is None:  # 如果未給定，則設為空字串 "" 的 list
            line_label_list = [""] * nlines
    
        # 回傳處理過的結果
        return line_data, line_color_list, line_style_list, line_width_list, line_label_list


    def generate_test_data(self, t0, t1, n_points=128):
        # 產生時間範圍 (t0 到 t1)，以 1 為間隔 (總共 t1-t0+1 個時間點)
        time_range = np.linspace(t0, t1, t1 - t0 + 1)
    
        # 產生空間範圍 (例如 128 個點)
        space_range = np.linspace(0, 2 * np.pi, n_points)
    
        # 使用 np.outer 來生成 2D array，基於 np.sin() 的時間和空間變化
        # 時間方向是 np.sin 的快速變化，空間方向是 np.sin 的較慢變化
        data = np.sin(np.outer(time_range, space_range))  # 時間和空間交互生成
    
        return data

