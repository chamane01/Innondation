import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import MeasureControl, Draw
import streamlit as st
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from streamlit_folium import folium_static

# Function to load a TIFF file
def load_tiff(file_path):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read the first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds  # Source bounds
        return data, transform, crs, bounds
    except Exception as e:
        st.error(f"Error loading GeoTIFF file: {e}")
        return None, None, None, None

# Function to load and reproject a shapefile or GeoJSON
def load_and_reproject_shapefile(file_path, target_crs):
    try:
        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs(target_crs)  # Reproject to target CRS
        return gdf
    except Exception as e:
        st.error(f"Error loading Shapefile/GeoJSON file: {e}")
        return None

# Function to calculate volume using only MNS
def calculate_volume_without_mnt(mns, transform, polygon_gdf):
    # Create a mask from the polygon
    out_shape = mns.shape
    mask = rasterize([(polygon_gdf.geometry.iloc[0], 1)], out_shape=out_shape, transform=transform)

    # Calculate cell area
    pixel_width = transform[0]
    pixel_height = -transform[4]  # Negative because it's in the y-direction
    cell_area = pixel_width * pixel_height  # in m²

    # Extract MNS values within the polygon
    mns_in_polygon = mns[mask == 1]

    # Calculate volume
    volume = np.sum(mns_in_polygon) * cell_area  # in m³

    return volume

# Streamlit app
st.title("Volume Calculation using MNS")

# Buttons under the map
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Create Map"):
        st.info("Map creation feature under development.")
with col2:
    if st.button("Calculate Volumes"):
        st.session_state.show_volume_sidebar = True
with col3:
    if st.button("Detect Trees"):
        st.info("Tree detection feature under development.")

# Volume calculation parameters sidebar
if st.session_state.get("show_volume_sidebar", False):
    st.sidebar.title("Volume Calculation Parameters")

    # File uploaders
    mns_file = st.sidebar.file_uploader("Upload MNS file (TIFF)", type=["tif", "tiff"])
    polygon_file = st.sidebar.file_uploader("Upload polygon file (required)", type=["geojson", "shp"])

    if mns_file and polygon_file:
        mns, transform, mns_crs, mns_bounds = load_tiff(mns_file)
        polygon_gdf = load_and_reproject_shapefile(polygon_file, mns_crs)

        if mns is None or polygon_gdf is None:
            st.sidebar.error("Error loading files.")
        else:
            try:
                volume = calculate_volume_without_mnt(mns, transform, polygon_gdf)
                st.sidebar.write(f"Calculated volume within the polygon: {volume:.2f} m³")

                # Display the map
                center_lat = (mns_bounds[1] + mns_bounds[3]) / 2
                center_lon = (mns_bounds[0] + mns_bounds[2]) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                # Add MNS as an image overlay
                folium.raster_layers.ImageOverlay(
                    image=mns,
                    bounds=[[mns_bounds[1], mns_bounds[0]], [mns_bounds[3], mns_bounds[2]]],
                    opacity=0.7,
                    name="MNS"
                ).add_to(fmap)

                # Add the polygon
                folium.GeoJson(
                    polygon_gdf,
                    name="Polygon",
                    style_function=lambda x: {'fillOpacity': 0, 'color': 'red', 'weight': 2}
                ).add_to(fmap)

                # Add map controls
                fmap.add_child(MeasureControl(position='topleft'))
                fmap.add_child(Draw(position='topleft', export=True))
                fmap.add_child(folium.LayerControl(position='topright'))

                # Display the map
                folium_static(fmap, width=700, height=500)

            except Exception as e:
                st.sidebar.error(f"Error calculating volume: {e}")
