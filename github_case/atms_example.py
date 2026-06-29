

import glob
import h5py 
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import zoom

import pdb



def make_plot(atms_lon, atms_lat, atms_bt, title_str='', extent=None, 
              outfile='test.png', vmin=220, vmax=285, 
              flag='mesh', cmap='jet', bar_label=None):
                  
    """Generates a geographic plot of ATMS Swath or Super-Resolution prediction data."""
    # 1. Initialize the map figure with PlateCarree projection
    fig = plt.figure(figsize=(11, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 2. Add geographic baseline features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='gray', linestyle=':', zorder=2)

    # 3. Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5, zorder=3)
    gl.top_labels = False   # Hide top labels for a cleaner look
    gl.right_labels = False # Hide right labels

    # 4. Plot the ATMS data using pcolormesh
    if flag == 'mesh': 
        mesh = ax.pcolormesh(atms_lon, atms_lat, atms_bt, 
                            transform=ccrs.PlateCarree(),
                            cmap=cmap, 
                            shading='auto', 
                            vmin=vmin, 
                            vmax=vmax, 
                            zorder=1)
    
    elif flag == 'scatter': 
        # For scatter, coordinates must be flattened to 1D if they are 2D grids
        # 'c=' controls the color mapped values, and 's=' sets pixel point size
        mesh = ax.scatter(atms_lon.ravel(), atms_lat.ravel(), 
                          c=atms_bt.ravel(), 
                          transform=ccrs.PlateCarree(),
                          cmap=cmap, 
                          vmin=vmin, 
                          vmax=vmax, 
                          s=1,  # Set a small marker size so the satellite points don't blob together
                          zorder=1)
    

    # 5. Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03, shrink=0.7)
    #cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
    
    if bar_label is None: 
        cbar.set_label('ATMS Channel 16 Brightness Temp (K)', fontsize=11, fontweight='bold')
    else:
        cbar.set_label(bar_label)    

    # 6. Set the map's boundary extents
    if extent is None:
        extent = [
            np.min(atms_lon) - 1, np.max(atms_lon) + 1, 
            np.min(atms_lat) - 1, np.max(atms_lat) + 1
        ]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 7. Add Title
    plt.title(title_str, fontsize=13, fontweight='bold', pad=15)
    

    # 8. Save the figure to your disk and clean up memory
    plt.savefig(outfile, dpi=200)
    plt.close(fig) 
    print(f"  -> Successfully generated map: {outfile}") 


def main (): 
  
    ### get the ATMS file list 
    atms_geo_files = sorted(glob.glob('./data/GATMO_j01_d20260415*.h5'))
    atms_sdr_files = sorted(glob.glob('./data/SATMS_j01_d20260415*.h5'))
    
    ### Read Brightness Temperaute  
    sdrs = [h5py.File(filename, 'r') for filename in atms_sdr_files]
    BrightnessTemperature = np.concatenate([f['/All_Data/ATMS-SDR_All/BrightnessTemperature'] for f in sdrs])
    
    BrightnessTemperatureFactors=np.concatenate([f['/All_Data/ATMS-SDR_All/BrightnessTemperatureFactors'] for f in sdrs])
    BrightnessTemperature = BrightnessTemperature * BrightnessTemperatureFactors[0] + BrightnessTemperatureFactors[1]
    
    ### channel 16 only: 89GHz 
    bt = BrightnessTemperature[:, :, 15]
    
    
    ### read latitudes and longitude 
    geos = [h5py.File(filename, 'r') for filename in atms_geo_files]
    lat = np.concatenate([f['/All_Data/ATMS-SDR-GEO_All/Latitude'] for f in geos])
    lon = np.concatenate([f['/All_Data/ATMS-SDR-GEO_All/Longitude'] for f in geos])
    
    
    # Define a shared bounding box context
    margin = 0.0
    if (np.max(lon) - np.min(lon)) > 180:
        # Shift longitudes temporarily to [0, 360] to find the true localized min/max
        lon_shifted = np.mod(lon, 360)
        lon_min_shifted = np.min(lon_shifted) - margin
        lon_max_shifted = np.max(lon_shifted) + margin
    
        # Convert them back to the [-180, 180] space for Cartopy
        lon_min = (lon_min_shifted + 180) % 360 - 180
        lon_max = (lon_max_shifted + 180) % 360 - 180
    else:
        # Standard case (No date-line crossing)
        lon_min = np.min(lon) - margin
        lon_max = np.max(lon) + margin

    # 2. Latitude never crosses a wrapping boundary
    lat_min = np.min(lat) - margin
    lat_max = np.max(lat) + margin

    # 3. Create your localized extent
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    
    
    ## make plots for orginal resolotion 
    make_plot(lon, lat, bt, title_str='orig ', extent=extent, outfile='orig.png')
    
    ## make plots for high resolotion 
    # 3. Zoom coordinates linearly to match the 4x super-resolution prediction grid
    zoom_factor = 4
    pred_lat = zoom(lat, zoom_factor, order=1) 
    pred_lon = zoom(lon, zoom_factor, order=1)
    
    #make_plot(pred_lon, pred_lat, pred_bt, title_str='model predit', extent=extent, outfile='high.png')
    
    pdb.set_trace()
    


if __name__ == "__main__": 
    
    main()