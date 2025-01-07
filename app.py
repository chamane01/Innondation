import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import MeasureControl, Draw
import streamlit as st
from rasterio.features import rasterize
from streamlit_folium import folium_static

# Function to load a TIFF file
def load_tiff(file_path, target_crs="EPSG:4326"):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read the first band
            src_crs = src.crs  # Source CRS
            transform = src.transform  # Affine transform
            bounds = src.bounds  # Source bounds

            # Reproject the data to target CRS
            target_transform, width, height = calculate_default_transform(src_crs, target_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': target_transform,
                'width': width,
                'height': height
            })
            data_reprojected = np.zeros((height, width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=data_reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

            # Reproject bounds to target CRS
            target_bounds = transform_bounds(src_crs, target_crs, *src.bounds)

        return data, transform, src_crs, data_reprojected, target_transform, target_crs, target_bounds
    except Exception as e:
        st.error(f"Error loading GeoTIFF file: {e}")
        return None, None, None, None, None, None, None

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
def calculate_volume_without_mnt(mns, mns_transform, mns_crs, polygon_gdf):
    # Create a mask from the polygon
    out_shape = mns.shape
    mask = np.zeros(out_shape, dtype=int)
    for geom in polygon_gdf.geometry:
        mask += rasterize([(geom, 1)], out_shape=out_shape, transform=mns_transform, fill=0, default_value=1)
    mask = mask.astype(bool)

    # Calculate cell area
    pixel_width = mns_transform[0]
    pixel_height = -mns_transform[4]  # Negative because it's in the y-direction
    cell_area = pixel_width * pixel_height  # in m²

    # Extract MNS values within the polygon
    mns_in_polygon = mns[mask]

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

    # Choice of calculation method
    method = st.sidebar.radio(
        "Choose calculation method:",
        ("Method 1: MNS - MNT", "Method 2: MNS only")
    )

    # File uploaders
    mns_file = st.sidebar.file_uploader("Upload MNS file (TIFF)", type=["tif", "tiff"])
    polygon_file = st.sidebar.file_uploader("Upload polygon file (required)", type=["geojson", "shp"])

    if method == "Method 1: MNS - MNT":
        mnt_file = st.sidebar.file_uploader("Upload MNT file (TIFF)", type=["tif", "tiff"])
    else:
        mnt_file = None

    if mns_file and polygon_file and (method == "Method 2: MNS only" or mnt_file):
        mns, mns_transform, mns_crs, mns_reprojected, mns_target_transform, mns_target_crs, mns_bounds = load_tiff(mns_file)
        polygons_gdf = load_and_reproject_shapefile(polygon_file, mns_crs)

        if mns is None or polygons_gdf is None:
            st.sidebar.error("Error loading files.")
        else:
            try:
                if method == "Method 1: MNS - MNT":
                    mnt, mnt_transform, mnt_crs, mnt_reprojected, mnt_target_transform, mnt_target_crs, mnt_bounds = load_tiff(mnt_file)
                    if mnt is None or mnt_bounds != mns_bounds:
                        st.sidebar.error("Files must have the same geographic bounds.")
                    else:
                        # Volume calculation using MNS - MNT
                        # (Implementation depends on specific requirements)
                        pass
                else:
                    volume = calculate_volume_without_mnt(mns, mns_transform, mns_crs, polygons_gdf)
                    st.sidebar.write(f"Calculated volume within the polygon: {volume:.2f} m³")

                # Display the map
                center_lat = (mns_bounds[1] + mns_bounds[3]) / 2
                center_lon = (mns_bounds[0] + mns_bounds[2]) / 2
                fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

                # Add MNS as an image overlay in EPSG:4326
                folium.raster_layers.ImageOverlay(
                    image=mns_reprojected,
                    bounds=[[mns_bounds[1], mns_bounds[0]], [mns_bounds[3], mns_bounds[2]]],
                    opacity=0.7,
                    name="MNS"
                ).add_to(fmap)

                # Add the polygon
                folium.GeoJson(
                    polygons_gdf.to_crs("EPSG:4326"),
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
