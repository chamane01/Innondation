import streamlit as st
import rasterio
import rasterio.warp
import folium
from folium import plugins
from folium.plugins import MeasureControl, Draw
from rasterio.plot import reshape_as_image
from PIL import Image
from streamlit_folium import folium_static

def reproject_tiff(input_tiff, target_crs):
    """Reproject a TIFF file to a target CRS."""
    with rasterio.open(input_tiff) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        reprojected_tiff = "reprojected.tiff"
        with rasterio.open(reprojected_tiff, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=rasterio.warp.Resampling.nearest
                )

    return reprojected_tiff

def add_image_overlay(map_object, tiff_path, bounds, name):
    """Add a TIFF image overlay to a Folium map."""
    with rasterio.open(tiff_path) as src:
        # Read the image and reshape it into a format compatible with Folium
        image = reshape_as_image(src.read())
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            name=name,
            opacity=0.6
        ).add_to(map_object)

# Streamlit app
def main():
    st.title("TIFF Viewer and Interactive Map")

    # Button to toggle sidebar visibility
    if st.button("Dessiner"):
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload a TIFF file", type=["tif", "tiff"])

        if uploaded_file is not None:
            tiff_path = uploaded_file.name
            with open(tiff_path, "wb") as f:
                f.write(uploaded_file.read())

            st.write("Reprojecting TIFF file...")

            # Reproject TIFF to target CRS (e.g., EPSG:4326)
            reprojected_tiff = reproject_tiff(tiff_path, "EPSG:4326")

            # Read bounds from reprojected TIFF file
            with rasterio.open(reprojected_tiff) as src:
                bounds = src.bounds

            # Create Folium map centered on the bounds
            center_lat = (bounds.top + bounds.bottom) / 2
            center_lon = (bounds.left + bounds.right) / 2
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Add reprojected TIFF as overlay
            add_image_overlay(fmap, reprojected_tiff, bounds, "TIFF Layer")

            # Add measure control
            fmap.add_child(MeasureControl(position='topleft'))

            # Add draw control
            draw = Draw(position='topleft', export=True,
                        draw_options={'polyline': {'shapeOptions': {'color': 'blue', 'weight': 4, 'opacity': 0.7}},
                                      'polygon': {'shapeOptions': {'color': 'green', 'weight': 4, 'opacity': 0.7}},
                                      'rectangle': {'shapeOptions': {'color': 'red', 'weight': 4, 'opacity': 0.7}},
                                      'circle': {'shapeOptions': {'color': 'purple', 'weight': 4, 'opacity': 0.7}}},
                        edit_options={'edit': True,}
            )
            fmap.add_child(draw)

            # Layer control
            folium.LayerControl().add_to(fmap)

            # Display map
            folium_static(fmap)

if __name__ == "__main__":
    main()
