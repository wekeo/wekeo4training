import os
from matplotlib import pyplot as plt

import xarray as xr
from netCDF4 import Dataset
import numpy as np
import glob

from matplotlib import pyplot as plt
import matplotlib.colors
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.cm as cm

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)


def generate_masked_array(xarray, mask, threshold, operator, drop=True):
    """ 
    Applies a mask (e.g. a cloud mask) onto a given xarray.DataArray, based on a given threshold and operator.
    
    Parameters:
        xarray(xarray DataArray): a three-dimensional xarray.DataArray object
        mask(xarray DataArray): 1-dimensional xarray.DataArray, e.g. cloud fraction values
        threshold(float): any number specifying the threshold
        operator(str): operator how to mask the array, e.g. '<', '>' or '!='
        drop(boolean): default is True
        
    Returns:
        Masked xarray.DataArray with NaN values dropped, if kwarg drop equals True
    """
    if(operator=='<'):
        cloud_mask = xr.where(mask < threshold, 1, 0) #Generate cloud mask with value 1 for the pixels we want to keep
    elif(operator=='!='):
        cloud_mask = xr.where(mask != threshold, 1, 0)
    elif(operator=='>'):
        cloud_mask = xr.where(mask > threshold, 1, 0)
    else:
        cloud_mask = xr.where(mask == threshold, 1, 0)
            
    xarray_masked = xr.where(cloud_mask ==1, xarray, np.nan) #Apply mask onto the DataArray
    xarray_masked.attrs = xarray.attrs #Set DataArray attributes 
    if(drop):
        return xarray_masked[~np.isnan(xarray_masked)] #Return masked DataArray
    else:
        return xarray_masked



def slstr_frp_gridding(parameter_array, parameter, lat_min, lat_max, lon_min, lon_max, 
                       sampling_lat_FRP_grid, sampling_lon_FRP_grid, n_fire, lat_frp, lon_frp, **kwargs):
    """ 
    Produces gridded data of Sentinel-3 SLSTR NRT Fire Radiative Power Data
    
    Parameters:
        parameter_array(xarray.DataArray): xarray.DataArray with extracted data variable of fire occurences
        parameter(str): NRT S3 FRP channel - either `mwir`, `swir` or `swir_nosaa`
        lat_min, lat_max, lon_min, lon_max(float): Floats of geographical bounding box
        sampling_lat_FRP_grid, sampling_long_FRP_grid(float): Float of grid cell size
        n_fire(int): Number of fire occurences
        lat_frp(xarray.DataArray): Latitude values of occurred fire events
        lon_frp(xarray.DataArray): Longitude values of occurred fire events
        **kwargs: additional keyword arguments to be added. Required for parameter `swir_nosaa`, where the function
                  requires the xarray.DataArray with the SAA FLAG information.  

    Returns:
        the gridded xarray.Data Array and latitude and longitude grid information
    """ 
    n_lat = int( (np.float32(lat_max) - np.float32(lat_min)) / sampling_lat_FRP_grid ) + 1 # Number of rows per latitude sampling
    n_lon = int( (np.float32(lon_max) - np.float32(lon_min)) / sampling_lon_FRP_grid ) + 1 # Number of lines per longitude sampling

    
    slstr_frp_gridded = np.zeros( [n_lat, n_lon], dtype='float32' ) - 9999.

    lat_grid = np.zeros( [n_lat, n_lon], dtype='float32' ) - 9999.
    lon_grid = np.zeros( [n_lat, n_lon], dtype='float32' ) - 9999.
    
    if (n_fire >= 0):
    
    # Loop on i_lat: begins
        for i_lat in range(n_lat):
                    
        # Loop on i_lon: begins
            for i_lon in range(n_lon):
                        
                lat_grid[i_lat, i_lon] = lat_min + np.float32(i_lat) * sampling_lat_FRP_grid + sampling_lat_FRP_grid / 2.
                lon_grid[i_lat, i_lon] = lon_min + np.float32(i_lon) * sampling_lon_FRP_grid + sampling_lon_FRP_grid / 2.
                            
            # Gridded SLSTR FRP MWIR Night - All days
                if(parameter=='swir_nosaa'):
                    FLAG_FRP_SWIR_SAA_nc = kwargs.get('flag', None)
                    mask_grid = np.where( 
                        (lat_frp[:] >= lat_min + np.float32(i_lat) * sampling_lat_FRP_grid)  & 
                        (lat_frp[:] < lat_min + np.float32(i_lat+1) * sampling_lat_FRP_grid) & 
                        (lon_frp[:] >= lon_min + np.float32(i_lon) * sampling_lon_FRP_grid)  & 
                        (lon_frp[:] < lon_min + np.float32(i_lon+1) * sampling_lon_FRP_grid) &
                        (parameter_array[:] != -1.) & (FLAG_FRP_SWIR_SAA_nc[:] == 0), False, True)
                else:
                    mask_grid = np.where( 
                        (lat_frp[:] >= lat_min + np.float32(i_lat) * sampling_lat_FRP_grid)  & 
                        (lat_frp[:] < lat_min + np.float32(i_lat+1) * sampling_lat_FRP_grid) & 
                        (lon_frp[:] >= lon_min + np.float32(i_lon) * sampling_lon_FRP_grid)  & 
                        (lon_frp[:] < lon_min + np.float32(i_lon+1) * sampling_lon_FRP_grid) &
                        (parameter_array[:] != -1.),  False, True)
                            
                masked_slstr_frp_grid = np.ma.array(parameter_array[:], mask=mask_grid)
                            
                if len(masked_slstr_frp_grid.compressed()) != 0:
                    slstr_frp_gridded[i_lat, i_lon]  = np.sum(masked_slstr_frp_grid.compressed())
    return slstr_frp_gridded, lat_grid, lon_grid
    

def visualize_s3_frp(data, lat, lon, unit, longname, textstr_1, textstr_2, vmax, show=True):
    """ 
    Visualizes a numpy.Array (Sentinel-3 SLSTR NRT FRP data) with matplotlib's pcolormesh function and adds two
    text boxes to the plot.
    
    Parameters:
        data(numpy.MaskedArray): any numpy MaskedArray, e.g. loaded with the NetCDF library and the Dataset function
        lat(numpy.Array): array with longitude values
        lon(numpy.Array) : array with latitude values
        unit(str): unit of the resulting plot
        longname(str): Longname to be used as title
        textstr_1(str): String to fill box 1
        textstr_2(str): String to fill box 2
        vmax(float): Maximum value of color scale
    """
    fig=plt.figure(figsize=(20, 15))

    ax = plt.axes(projection=ccrs.PlateCarree())

    cmap = cm.autumn_r
    cmap.set_over('#8B0000')  # Dark red
    
    # Normalization
    norm = Normalize(vmin=0, vmax=vmax, clip=False)

    img = plt.pcolormesh(lon, lat, data, 
                        cmap=cmap, transform=ccrs.PlateCarree(),
                        norm=norm)

    ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.bottom_labels=False
    gl.right_labels=False
    gl.xformatter=LONGITUDE_FORMATTER
    gl.yformatter=LATITUDE_FORMATTER
    gl.xlabel_style={'size':14}
    gl.ylabel_style={'size':14}

    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', fraction=0.029, pad=0.025, extend='max')
    cbar.set_label(unit, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(longname, fontsize=20, pad=40.0) 

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box on the right side of the plot
    ax.text(1.1, 0.9, textstr_1, transform=ax.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(1.1, 0.85, textstr_2, transform=ax.transAxes, fontsize=16,
            verticalalignment='top', bbox=props)
    if show:
        plt.show()
    else:
        plt.close()



def visualize_s3_aod(aod_ocean, aod_land, latitude, longitude, title, unit, vmin, vmax, color_scale, projection, show):
    """ 
    Visualizes two xarray.DataArrays from the Sentinel-3 SLSTR NRT AOD dataset onto the same plot with 
    matplotlib's pcolormesh function.
    
    Parameters:
        aod_ocean(xarray.DataArray): xarray.DataArray with the Aerosol Optical Depth for ocean values
        aod_land(xarray.DataArray): xarray.DataArray with Aerosol Optical Depth for land values
        longitude(xarray.DataArray): xarray.DataArray holding the longitude values
        latitude(xarray.DataArray): xarray.DataArray holding the latitude values
        title(str): title of the resulting plot
        unit(str): unit of the resulting plot
        vmin(int): minimum number on visualisation legend
        vmax(int): maximum number on visualisation legend
        color_scale(str): string taken from matplotlib's color ramp reference
        projection(str): a projection provided by the cartopy library, e.g. ccrs.PlateCarree()
    """
    fig=plt.figure(figsize=(12, 12))

    ax=plt.axes(projection=projection)
    ax.coastlines(linewidth=1.5, linestyle='solid', color='k', zorder=10)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels=False
    gl.right_labels=False
    gl.xformatter=LONGITUDE_FORMATTER
    gl.yformatter=LATITUDE_FORMATTER
    gl.xlabel_style={'size':12}
    gl.ylabel_style={'size':12}


    img1 = plt.pcolormesh(longitude, latitude, aod_ocean, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=color_scale)
    img2 = plt.pcolormesh(longitude, latitude, aod_land, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=color_scale)
    ax.set_title(title, fontsize=20, pad=20.0)

    cbar = fig.colorbar(img1, ax=ax, orientation='vertical', fraction=0.04, pad=0.05)
    cbar.set_label(unit, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    if show:
        plt.show()
    else:
        plt.close()
