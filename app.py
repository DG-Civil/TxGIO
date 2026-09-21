import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import shape
import requests
import xml.etree.ElementTree as ET
import os
import zipfile
import glob
import math
import io
import gc
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="TNRIS Data & Surface Suite", layout="wide")


# Initialize Session State
if 'map_center' not in st.session_state:
    st.session_state.map_center = [31.9686, -99.9018]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6
if 'map_key_version' not in st.session_state:
    st.session_state.map_key_version = 0
if 'basemap_choice' not in st.session_state:
    st.session_state.basemap_choice = "Satellite"
if 'last_drawing' not in st.session_state:
    st.session_state.last_drawing = None
if 'ready_zip_data' not in st.session_state:
    st.session_state.ready_zip_data = None
if 'ready_bat_data' not in st.session_state:
    st.session_state.ready_bat_data = None
if 'processed_batch_zip' not in st.session_state:
    st.session_state.processed_batch_zip = None
if 'filtered_tile_ids' not in st.session_state:
    st.session_state.filtered_tile_ids = None
if 'editor_suffix' not in st.session_state:
    st.session_state.editor_suffix = 0
if 'force_select' not in st.session_state:
    st.session_state.force_select = False

# Texas EPSG Coordinate Reference Systems
TEXAS_EPSG_DICT = {
    "NAD83(2011) / Texas North (ftUS) - EPSG:6582": "EPSG:6582",
    "NAD83(2011) / Texas North Central (ftUS) - EPSG:6584": "EPSG:6584",
    "NAD83(2011) / Texas Central (ftUS) - EPSG:6585": "EPSG:6585",
    "NAD83(2011) / Texas South Central (ftUS) - EPSG:6587": "EPSG:6587",
    "NAD83(2011) / Texas South (ftUS) - EPSG:6586": "EPSG:6586",
    "NAD83 / Texas North (ftUS) - EPSG:2275": "EPSG:2275",
    "NAD83 / Texas North Central (ftUS) - EPSG:2276": "EPSG:2276",
    "NAD83 / Texas Central (ftUS) - EPSG:2277": "EPSG:2277",
    "NAD83 / Texas South Central (ftUS) - EPSG:2278": "EPSG:2278",
    "NAD83 / Texas South (ftUS) - EPSG:2279": "EPSG:2279",
}

MAX_UPLOAD_MB = 150.0

import ctypes
import psutil


def enforce_cloud_memory_limit(limit_mb=900, hard_stop=True):
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

    parent = psutil.Process(os.getpid())
    processes = [parent] + parent.children(recursive=True)
    total_mb = sum(p.memory_info().rss for p in processes) / (1024 * 1024)

    if total_mb > limit_mb:
        if hard_stop:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.error(f"⚠️ **Memory Limit Exceeded ({total_mb:.1f} MB / {limit_mb} MB)**\n\nExecution stopped. Session purged.")
            if st.button("🔄 Hard Reset Server Memory", type="primary"):
                os._exit(0)
            st.stop()
        return False # Return False for soft stops inside loops
    return True # Safe

RAM_limit=1100

def get_zoom_from_bounds(minx, miny, maxx, maxy):
    max_diff = max(maxx - minx, maxy - miny)
    if max_diff == 0:
        return 15
    zoom = math.floor(math.log2(360 / max_diff))
    return max(0, min(18, zoom + 1))

@st.cache_resource
def load_shapefiles(shp_dir="shp"):
    shp_dict = {}
    if not os.path.exists(shp_dir):
        return shp_dict
        
    for file in glob.glob(os.path.join(shp_dir, "*.shp")):
        try:
            filename = os.path.basename(file)
            gdf = gpd.read_file(file).to_crs(epsg=4326)
            shp_dict[filename] = gdf
        except Exception as e:
            st.warning(f"Could not load {file}: {e}")
    return shp_dict

@st.cache_data(ttl=3600)
def fetch_tnris_collections():
    collections = []
    try:
        url = "https://tnris-data-warehouse.s3.us-east-1.amazonaws.com/?prefix=LCD/collection/&delimiter=/"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for elem in root.iter():
                if elem.tag.endswith('Prefix'):
                    val = elem.text
                    if val and val.startswith('LCD/collection/') and val != 'LCD/collection/':
                        collections.append(val.split('/')[-2])
    except Exception:
        pass
    return sorted(collections) if collections else ['stratmap-2024-50cm-hays-williamson-counties']

@st.cache_data(ttl=3600)
def fetch_tnris_items(collection_name):
    items = []
    try:
        prefix = f"LCD/collection/{collection_name}/items/"
        url = f"https://tnris-data-warehouse.s3.us-east-1.amazonaws.com/?prefix={prefix}&delimiter=/"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for elem in root.iter():
                if elem.tag.endswith('Prefix'):
                    val = elem.text
                    if val and val.startswith(prefix) and val != prefix:
                        items.append(val.split('/')[-2])
    except Exception:
        pass
    return sorted(items) if items else ['dem', 'hypso-2ft', 'las', 'dsm']

def geocode_address(address):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
        headers = {'User-Agent': 'TNRISDownloader/1.0'}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return None

@st.cache_data
def load_surface_adjustment_factors(csv_path="surface-adjustment-factors.csv"):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['County', 'Adjustment'])
            saf_dict = dict(zip(df['County'].astype(str).str.strip(), df['Adjustment'].astype(float)))
            return saf_dict
        except Exception as e:
            st.error(f"Error reading surface-adjustment-factors.csv: {e}")
    return {}

def convert_raster_to_landxml_str(data_arr, transform, nodata_val, surface_name="TNRIS_Surface", stride=2, z_scale=1.0, surface_factor=1.0):
    sub_arr = data_arr[::stride, ::stride]
    rows, cols = sub_arr.shape
    
    row_indices, col_indices = np.indices((rows, cols))
    flat_rows = row_indices.flatten() * stride
    flat_cols = col_indices.flatten() * stride
    
    xs, ys = rasterio.transform.xy(transform, flat_rows, flat_cols, offset='center')
    flat_z = sub_arr.flatten()
    
    valid_mask = (flat_z != nodata_val) & (~np.isnan(flat_z)) if nodata_val is not None else ~np.isnan(flat_z)
    
    grid_pt_ids = np.zeros(rows * cols, dtype=int)
    valid_indices = np.where(valid_mask)[0]
    
    grid_pt_ids[valid_indices] = np.arange(1, len(valid_indices) + 1)
    grid_pt_ids_2d = grid_pt_ids.reshape((rows, cols))
    
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            p1 = grid_pt_ids_2d[r, c]
            p2 = grid_pt_ids_2d[r+1, c]
            p3 = grid_pt_ids_2d[r+1, c+1]
            p4 = grid_pt_ids_2d[r, c+1]
            
            if p1 > 0 and p2 > 0 and p3 > 0:
                faces.append(f"          <F>{p1} {p2} {p3}</F>")
            if p1 > 0 and p3 > 0 and p4 > 0:
                faces.append(f"          <F>{p1} {p3} {p4}</F>")

    num_points = len(valid_indices)
    num_triangles = len(faces)
    now_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2" date="{now_str}" time="{time_str}">',
        '  <Units>',
        '    <Imperial linearUnit="USSurveyFoot" widthUnit="USSurveyFoot" heightUnit="USSurveyFoot" areaUnit="squareFoot" volumeUnit="cubicFeet" temperatureUnit="fahrenheit" pressureUnit="inHG"/>',
        '  </Units>',
        '  <Surfaces>',
        f'    <Surface name="{surface_name}">',
        f'      <Definition surfaceType="TIN" numPnts="{num_points}" numTris="{num_triangles}">',
        '        <Pnts>'
    ]
    
    for idx in valid_indices:
        pt_id = grid_pt_ids[idx]
        easting = xs[idx] * surface_factor
        northing = ys[idx] * surface_factor
        elev = flat_z[idx] * z_scale
        xml_lines.append(f'          <P id="{pt_id}">{northing:.4f} {easting:.4f} {elev:.4f}</P>')
        
    xml_lines.append('        </Pnts>')
    xml_lines.append('        <Faces>')
    xml_lines.extend(faces)
    xml_lines.append('        </Faces>')
    xml_lines.append('      </Definition>')
    xml_lines.append('    </Surface>')
    xml_lines.append('  </Surfaces>')
    xml_lines.append('</LandXML>')
    
    return "\n".join(xml_lines)

# ---------------------------------------------------------
# MAIN MULTI-TAB INTERFACE
# ---------------------------------------------------------

st.set_page_config(
    page_title="TxGIO Downloader",
    page_icon="🗺️",
    layout="wide",
)


st.markdown(
    "<h1 style='text-align: center;'>🗺️ TxGIO (TNRIS) GIS Data Downloader And Processor</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'><b><i>Developed by Dawit Ghebreyesus</i></b></p>", 
    unsafe_allow_html=True
)

st.markdown("---")  
tab1, tab2, tab3, tab4 = st.tabs([
    "1. TxGIO (TNRIS) Tile Downloader", 
    "2. Batch LandXML & Coordinate Converter", 
    "3. Batch Shapefile Reprojection",
    "4. 3D Terrain Viewer"
])

# =========================================================
# TAB 1: TILE DOWNLOADER
# =========================================================
with tab1:

    st.subheader("A. Area of Interest Selection")
    col_search1, col_search2 = st.columns([4, 1])
    with col_search1:
        address_query = st.text_input("🔍 Search Address or Location", placeholder="Enter address...", label_visibility="collapsed")
    with col_search2:
        if st.button("Go to Address", use_container_width=True):
            if address_query:
                coords = geocode_address(address_query)
                if coords:
                    st.session_state.map_center = coords
                    st.session_state.map_zoom = 14
                    st.session_state.map_key_version += 1
                    st.rerun()

    all_shapefiles = load_shapefiles("shp")
    if not all_shapefiles:
        st.error("⚠️ No shapefiles found in 'shp/' folder! Make sure the folder is included in your deployment repository.")
        selected_shp = None
        id_field = None
        collection_field = None
    else:
        col_sh1, col_sh2, col_sh3 = st.columns(3)
        with col_sh1:
            selected_shp = st.selectbox("Active Index Shapefile", list(all_shapefiles.keys()))
            active_gdf = all_shapefiles[selected_shp]
            available_fields = [c for c in active_gdf.columns if c != 'geometry']
        with col_sh2:
            default_id_index = next((i for i, f in enumerate(available_fields) if 'name' in f.lower() or 'id' in f.lower()), 0)
            id_field = st.selectbox("Index ID Field Name", available_fields, index=default_id_index)
        with col_sh3:
            default_col_index = next((i for i, f in enumerate(available_fields) if 'collection' in f.lower() or 'coll' in f.lower()), min(1, max(0, len(available_fields) - 1)))
            collection_field = st.selectbox("Shapefile Collection Field Name", available_fields, index=default_col_index)

    col1, col2 = st.columns([2, 1])
    intersecting_tiles = gpd.GeoDataFrame()
    tiles_to_download_ids = []

    with col1:
        st.subheader("Select Area of Interest")
        
        st.markdown("🗺️ Select Basemap") 
        basemap_choice = st.radio(
            "🗺️ Select Basemap", 
            ["Satellite", "OpenStreetMap"], 
            horizontal=True, 
            label_visibility="collapsed",
            key="basemap_choice"
        )
        
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles=None)
        
        if st.session_state.basemap_choice == "Satellite":
            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                attr="Google",
                name="Google Satellite",
                overlay=False,
                control=False
            ).add_to(m)
        else:
            folium.TileLayer(
                tiles="OpenStreetMap",
                name="OpenStreetMap",
                overlay=False,
                control=False
            ).add_to(m)
            
        if selected_shp:
            active_gdf = all_shapefiles[selected_shp]
            minx, miny, maxx, maxy = active_gdf.total_bounds
            folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]], color="blue", fill=False, weight=1).add_to(m)
            
        draw = folium.plugins.Draw(
            draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True},
            edit_options={'edit': False}
        )
        m.add_child(draw)
        
        if st.session_state.last_drawing and selected_shp:
            drawn_geom = shape(st.session_state.last_drawing["geometry"])
            active_gdf = all_shapefiles[selected_shp]
            intersecting_tiles = active_gdf[active_gdf.intersects(drawn_geom)]
            
            # Apply filter if rows were removed from the table
            if st.session_state.filtered_tile_ids is not None:
                intersecting_tiles = intersecting_tiles[intersecting_tiles[id_field].isin(st.session_state.filtered_tile_ids)]

            if not intersecting_tiles.empty:
                tiles_to_download_ids = intersecting_tiles[id_field].astype(str).tolist()
                tooltip_fields = [id_field]
                if collection_field and collection_field in intersecting_tiles.columns and collection_field != id_field:
                    tooltip_fields.append(collection_field)

                folium.GeoJson(
                    intersecting_tiles,
                    name="Selected Tiles",
                    style_function=lambda x: {'fillColor': 'green', 'color': 'green', 'weight': 2, 'fillOpacity': 0.5},
                    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields)
                ).add_to(m)

        map_data = st_folium(
            m, 
            width="100%", 
            height=450, 
            returned_objects=["last_active_drawing"], 
            key=f"tnris_map_{st.session_state.map_key_version}"
        )
        
        if map_data and map_data.get("last_active_drawing"):
            if map_data["last_active_drawing"] != st.session_state.last_drawing:
                st.session_state.last_drawing = map_data["last_active_drawing"]
                st.session_state.ready_zip_data = None
                st.session_state.ready_bat_data = None
                st.session_state.filtered_tile_ids = None
                st.session_state.force_select = False
                st.session_state.editor_suffix += 1
                
                try:
                    drawn_geom = shape(map_data["last_active_drawing"]["geometry"])
                    minx, miny, maxx, maxy = drawn_geom.bounds
                    
                    st.session_state.map_center = [(miny + maxy) / 2, (minx + maxx) / 2]
                    st.session_state.map_zoom = get_zoom_from_bounds(minx, miny, maxx, maxy)
                    st.session_state.map_key_version += 1
                except Exception:
                    pass 
                
                st.rerun()

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if not intersecting_tiles.empty and st.button("📍 Zoom to Selected", use_container_width=True):
                minx, miny, maxx, maxy = intersecting_tiles.total_bounds
                st.session_state.map_center = [(miny + maxy) / 2, (minx + maxx) / 2]
                st.session_state.map_zoom = get_zoom_from_bounds(minx, miny, maxx, maxy)
                st.session_state.map_key_version += 1
                st.rerun()
        with col_btn2:
            if st.session_state.last_drawing is not None and st.button("🗑️ Clear Selection", use_container_width=True):
                st.session_state.last_drawing = None
                st.session_state.ready_zip_data = None
                st.session_state.ready_bat_data = None
                st.session_state.filtered_tile_ids = None
                st.session_state.force_select = False
                st.session_state.map_key_version += 1
                st.session_state.editor_suffix += 1
                st.rerun()

    with col2:
        st.subheader("Intersecting Tiles")
        if not intersecting_tiles.empty:
            st.success(f"Matched {len(tiles_to_download_ids)} tiles.")
            display_cols = [id_field]
            if collection_field and collection_field in intersecting_tiles.columns and collection_field != id_field:
                display_cols.append(collection_field)
            
            # Dataframe must have a clean index for data_editor state tracking
            df_to_edit = intersecting_tiles[display_cols].copy().reset_index(drop=True)
            df_to_edit.insert(0, "Select", st.session_state.force_select)
            
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                if st.button("☑️ Select All", use_container_width=True):
                    st.session_state.force_select = True
                    st.session_state.editor_suffix += 1
                    st.rerun()
            with col_sel2:
                if st.button("☐ Unselect All", use_container_width=True):
                    st.session_state.force_select = False
                    st.session_state.editor_suffix += 1
                    st.rerun()
            
            edited_df = st.data_editor(
                df_to_edit, 
                hide_index=True, 
                use_container_width=True, 
                height=250,
                column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
                key=f"editor_{st.session_state.map_key_version}_{st.session_state.editor_suffix}"
            )
            
            col_rem1, col_rem2 = st.columns(2)
            with col_rem1:
                if st.button("🗑️ Remove Selected", use_container_width=True):
                    keep_ids = edited_df[~edited_df["Select"]][id_field].tolist()
                    st.session_state.filtered_tile_ids = keep_ids
                    st.session_state.force_select = False
                    st.session_state.editor_suffix += 1
                    st.rerun()
            with col_rem2:
                if st.button("🗑️ Remove Unselected", use_container_width=True):
                    keep_ids = edited_df[edited_df["Select"]][id_field].tolist()
                    st.session_state.filtered_tile_ids = keep_ids
                    st.session_state.force_select = False
                    st.session_state.editor_suffix += 1
                    st.rerun()
        else:
            st.info("Draw a bounding box on the map to select tiles.")
            
    st.markdown("---")    
    st.subheader("B. Data Source Selection (from Amazon AWS)")
    st.markdown("[The directory of the TxGIO AWS server](https://tnris-data-warehouse.s3.us-east-1.amazonaws.com/index.html#LCD/collection/)")

    col_ds1, col_ds2, col_ds3 = st.columns(3)
    
    with col_ds1:
        collections = fetch_tnris_collections()
        selected_collection = st.selectbox("TNRIS Collection (select the Matched Tiles Collection name)", collections)
    with col_ds2:
        items = fetch_tnris_items(selected_collection)
        selected_item = st.selectbox("Item Data Type ", items)
    with col_ds3:
        file_types = ['.tif', '.img', '.zip']
        selected_ft = st.selectbox("File Type (DEM .tif or .img and for contours .zip) ", file_types, index=(2 if 'hypso' in selected_item or 'contour' in selected_item else 0))

    st.markdown("---")
    
    download_method = st.radio(
        "Download Method", 
        ["Cloud Download (.zip)", "Local Script (.bat) - Recommended for large numbers of tiles"], 
        horizontal=True
    )

    if st.button("🚀 Start Download", type="primary", use_container_width=True):
        if not tiles_to_download_ids:
            st.error("Please draw a bounding box first.")
            st.stop()
            
        bucket_url = "https://tnris-data-warehouse.s3.us-east-1.amazonaws.com/"
        s3_prefix = f"LCD/collection/{selected_collection}/items/{selected_item}/"
        
        s3_files = []
        continuation_token = None
        
        with st.spinner("Querying TNRIS S3 Bucket..."):
            while True:
                params = {'prefix': s3_prefix, 'list-type': '2'}
                if continuation_token:
                    params['continuation-token'] = continuation_token
                try:
                    resp = requests.get(bucket_url, params=params, timeout=10)
                    resp.raise_for_status()
                    root = ET.fromstring(resp.content)
                    for elem in root.iter():
                        if elem.tag.endswith('Key') and elem.text.lower().endswith(selected_ft):
                            s3_files.append(elem.text)
                    is_truncated = False
                    for elem in root.iter():
                        if elem.tag.endswith('IsTruncated') and elem.text.lower() == 'true':
                            is_truncated = True
                        if elem.tag.endswith('NextContinuationToken'):
                            continuation_token = elem.text
                    if not is_truncated:
                        break
                except Exception as e:
                    st.error(f"S3 Query Failed: {e}")
                    st.stop()

        files_to_download = []
        for dem_id in tiles_to_download_ids:
            for s3_key in s3_files:
                filename = s3_key.split('/')[-1]
                name_no_ext = os.path.splitext(filename)[0]
                if name_no_ext.endswith(dem_id):
                    files_to_download.append((dem_id, bucket_url + s3_key, filename, name_no_ext))
                    break

        if not files_to_download:
            st.error("No matching files found on S3.")
            st.stop()

        if "Cloud Download (.zip)" in download_method:
            progress_bar = st.progress(0)
            status_text = st.empty()
            zip_buffer = io.BytesIO()
            
            # --- BEFORE TAB 1 DOWNLOAD LOOP ---
            enforce_cloud_memory_limit(limit_mb=RAM_limit, hard_stop=True)
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for i, (dem_id, download_url, filename, name_no_ext) in enumerate(files_to_download):
                    
                    # Soft RAM check inside the loop
                    if not enforce_cloud_memory_limit(limit_mb=RAM_limit-50, hard_stop=False):
                        st.warning(f"⚠️ Memory limit approaching. Stopping at {i} of {len(files_to_download)} files. The partial batch is ready for download below.")
                        break
    
                    status_text.text(f"Downloading ({i+1}/{len(files_to_download)}): {filename}")
                    try:
                        response = requests.get(download_url, stream=True)
                        response.raise_for_status()
                        content = response.content
                        zip_file.writestr(filename, content)
                    except Exception as e:
                        st.error(f"Failed on {filename}: {e}")
                    progress_bar.progress((i + 1) / len(files_to_download))
    
            status_text.text("✅ Download process completed.")
            zip_buffer.seek(0)
            st.session_state.ready_zip_data = zip_buffer.getvalue()
            st.session_state.ready_bat_data = None
            
            # --- AFTER TAB 1 DOWNLOAD LOOP ---
            enforce_cloud_memory_limit(limit_mb=RAM_limit, hard_stop=True)
            
        else:
            # Generate BAT script to bypass server memory limits
            bat_script = "@echo off\n"
            bat_script += f"echo Downloading {len(files_to_download)} TNRIS Tiles...\n"
            for dem_id, download_url, filename, name_no_ext in files_to_download:
                bat_script += f'curl -o "{filename}" "{download_url}"\n'
            bat_script += "echo Download complete!\npause\n"
            
            st.session_state.ready_bat_data = bat_script
            st.session_state.ready_zip_data = None
            st.success("✅ Batch script generated successfully!")

    if st.session_state.get('ready_zip_data') is not None:
        st.download_button(
            label="💾 Save Downloaded Tiles (.zip)",
            data=st.session_state.ready_zip_data,
            file_name=f"TNRIS_Downloads_{selected_item}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
        
    if st.session_state.get('ready_bat_data') is not None:
        st.download_button(
            label="💾 Save Local Download Script (.bat)",
            data=st.session_state.ready_bat_data,
            file_name=f"TNRIS_Downloads_{selected_item}.bat",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )

# =========================================================
# TAB 2: BATCH LANDXML & COORDINATE CONVERTER
# =========================================================
with tab2:
    st.subheader("Batch Tile Reprojection & LandXML Conversion")
    st.caption("Process tiles individually without merging to optimize memory usage and keep surfaces modular.")
            
    col_proc1, col_proc2 = st.columns(2)
    
    with col_proc1:
        st.markdown("##### A. Select Target Coordinate System")
        selected_crs_label = st.selectbox("Target Texas State Plane System", list(TEXAS_EPSG_DICT.keys()))
        target_epsg = TEXAS_EPSG_DICT[selected_crs_label]
        
        st.markdown("##### B. Elevation Unit Options and Surface Adjustment Factor ")
        unit_option = st.radio(
            "Elevation Conversion (Z-Axis)",
            ["Keep Original Units", "Convert Meters to US Survey Feet (* 3.2808333)", "Convert Meters to International Feet (* 3.2808399)"],
            index=1
        )
        
        z_scale = 1.0
        if "US Survey Feet" in unit_option:
            z_scale = 3.280833333333333
        elif "International Feet" in unit_option:
            z_scale = 3.280839895013123
        
        county_saf_dict = load_surface_adjustment_factors("surface-adjustment-factors.csv")
        
        if county_saf_dict:
            county_options = ["None (Grid Coordinates - 1.000000)"] + sorted(list(county_saf_dict.keys()))
            selected_county = st.selectbox(
                "Select Texas County (Surface Adjustment Factor)",
                options=county_options,
                index=0,
                help="Select a Texas county to scale grid coordinates to surface coordinates at the end of LandXML or GEOTIFF generation."
            )
            
            if selected_county != "None (Grid Coordinates - 1.000000)":
                surface_factor = county_saf_dict[selected_county]
                st.info(f"Applying **{selected_county} County** Surface Adjustment Factor: `{surface_factor:.6f}`")
            else:
                surface_factor = 1.0
        else:
            st.warning("⚠️ `surface-adjustment-factors.csv` not found in root folder. Defaulting factor to 1.0.")
            surface_factor = 1.0
    
    with col_proc2:
        st.markdown("##### C. Output Format Options")
        output_format = st.radio("Batch Export Format", ["LandXML Surface (.xml)", "Reprojected GeoTIFF (.tif)"])
        
        grid_stride = 1
        if output_format == "LandXML Surface (.xml)":
            grid_stride = st.slider("Grid Resolution Sampling Stride", min_value=1, max_value=10, value=2, 
                                    help="1 = Full resolution points. 2 = Sample every 2nd point (reduces file size by ~75%).")
    
    st.markdown("---")
    st.markdown("##### D. Upload .tif or .img files")
    
    st.markdown("""
    <style>
        [data-testid="stFileUploaderDropzone"] ~ * {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if 'batch_files_dict' not in st.session_state:
        st.session_state.batch_files_dict = {}
    if 'ignored_dem_files' not in st.session_state:
        st.session_state.ignored_dem_files = set()
    if 'dem_uploader_key' not in st.session_state:
        st.session_state.dem_uploader_key = 0

    uploaded_batch_files = st.file_uploader(
        "Upload DEM Files for Batch Conversion (.tif, .img)", 
        type=["tif", "img"], 
        accept_multiple_files=True,
        key=f"dem_file_uploader_{st.session_state.dem_uploader_key}"
    )

    if uploaded_batch_files:
        for f in uploaded_batch_files:
            if f.name not in st.session_state.batch_files_dict and f.name not in st.session_state.ignored_dem_files:
                st.session_state.batch_files_dict[f.name] = f

    total_size_bytes_tab2 = sum(f.size for f in st.session_state.batch_files_dict.values())
    total_size_mb_tab2 = total_size_bytes_tab2 / (1024 * 1024)
    is_over_limit_tab2 = total_size_mb_tab2 > MAX_UPLOAD_MB

    if st.session_state.batch_files_dict:
        col_hdr1, col_hdr2 = st.columns([4, 1])
        with col_hdr1:
            st.caption(f"📂 **Uploaded Files ({len(st.session_state.batch_files_dict)})** — Total Size: **{total_size_mb_tab2:.2f} MB / {MAX_UPLOAD_MB:.0f} MB**")
        with col_hdr2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.batch_files_dict.clear()
                st.session_state.ignored_dem_files.clear()
                st.session_state.dem_uploader_key += 1
                st.session_state.processed_batch_zip = None
                gc.collect()
                st.rerun()

        with st.container(height=180):
            files_to_delete = []
            for file_name, file_obj in list(st.session_state.batch_files_dict.items()):
                col_info, col_del = st.columns([5, 1])
                size_kb = file_obj.size / 1024
                size_str = f"{size_kb/1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
                
                with col_info:
                    st.markdown(f"• **`{file_name}`** &nbsp;·&nbsp; `{size_str}`")
                with col_del:
                    if st.button("❌", key=f"del_{file_name}", help=f"Remove {file_name}"):
                        files_to_delete.append(file_name)

            if files_to_delete:
                for fname in files_to_delete:
                    del st.session_state.batch_files_dict[fname]
                    st.session_state.ignored_dem_files.add(fname)
                st.rerun()

    if is_over_limit_tab2:
        st.error(f"⚠️ Total upload size ({total_size_mb_tab2:.1f} MB) exceeds the {MAX_UPLOAD_MB:.0f} MB limit! Please remove one or more files before processing.")

    uploaded_batch_files_list = list(st.session_state.batch_files_dict.values())
    process_disabled_tab2 = is_over_limit_tab2 or (len(uploaded_batch_files_list) == 0)
             
    if st.button("⚡ Process Batch Files", type="primary", use_container_width=True, disabled=process_disabled_tab2):
        st.session_state.processed_batch_zip = None
        gc.collect()
        
        raster_sources = []
        for up_f in uploaded_batch_files_list:
            raster_sources.append((up_f.name, io.BytesIO(up_f.read())))
            
        st.info(f"Starting batch conversion for {len(raster_sources)} tiles...")
        
        batch_zip_buffer = io.BytesIO()
        batch_prog = st.progress(0)
        batch_status = st.empty()
        
        # --- BEFORE TAB 2 PROCESSING LOOP ---
        enforce_cloud_memory_limit(limit_mb=RAM_limit, hard_stop=True)
        
        
        with zipfile.ZipFile(batch_zip_buffer, "w", zipfile.ZIP_DEFLATED) as b_zip:
            for idx, (filename, file_source) in enumerate(raster_sources):
                
                
                # Soft RAM check inside the loop
                if not enforce_cloud_memory_limit(limit_mb=RAM_limit-100, hard_stop=False):
                    st.warning(f"⚠️ Memory limit approaching. Processed {idx} out of {len(raster_sources)} files. The partial batch is ready for download.")
                    break
                
                batch_status.text(f"Processing ({idx+1}/{len(raster_sources)}): {filename}")
                base_name = os.path.splitext(filename)[0]
                
                try:
                    with rasterio.open(file_source) as src:
                        transform, width, height = calculate_default_transform(
                            src.crs, target_epsg, src.width, src.height, *src.bounds
                        )
                        
                        destination = np.zeros((height, width), dtype=src.dtypes[0])
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=destination,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_epsg,
                            resampling=Resampling.bilinear
                        )
                        
                        nodata_val = src.nodata
                        
                        if output_format == "LandXML Surface (.xml)":
                            xml_str = convert_raster_to_landxml_str(
                                data_arr=destination,
                                transform=transform,
                                nodata_val=nodata_val,
                                surface_name=base_name,
                                stride=grid_stride,
                                z_scale=z_scale,
                                surface_factor=surface_factor
                            )
                            out_filename = f"{base_name}_{target_epsg.replace(':', '_')}.xml"
                            b_zip.writestr(out_filename, xml_str)
                            del xml_str
                            
                        else:  
                            out_filename = f"{base_name}_{target_epsg.replace(':', '_')}.tif"
                            
                            if z_scale != 1.0:
                                valid_mask = (destination != nodata_val) if nodata_val is not None else ~np.isnan(destination)
                                destination[valid_mask] = destination[valid_mask] * z_scale
                            
                            from rasterio.transform import Affine
                            scaled_transform = Affine(
                                transform.a * surface_factor,
                                transform.b * surface_factor,
                                transform.c * surface_factor,
                                transform.d * surface_factor,
                                transform.e * surface_factor,
                                transform.f * surface_factor
                            )
                            
                            kwargs = src.meta.copy()
                            kwargs.update({
                                'driver': 'GTiff',
                                'crs': target_epsg,
                                'transform': scaled_transform,
                                'width': width,
                                'height': height,
                                'dtype': 'float32',
                                'nodata': nodata_val
                            })
                            
                            tif_mem = io.BytesIO()
                            with rasterio.open(tif_mem, 'w', **kwargs) as dst:
                                dst.write(destination.astype(np.float32), 1)
                                
                            tif_mem.seek(0)
                            b_zip.writestr(out_filename, tif_mem.getvalue())
                            tif_mem.close()
                            
                        del destination
                        gc.collect()

                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                    
                batch_prog.progress((idx + 1) / len(raster_sources))

        batch_status.text("✅ Batch processing complete!")
        batch_zip_buffer.seek(0)
        st.session_state.processed_batch_zip = batch_zip_buffer.getvalue()
        batch_zip_buffer.close()
        gc.collect()
        
        # --- AFTER TAB 2 PROCESSING LOOP ---
        enforce_cloud_memory_limit(limit_mb=RAM_limit, hard_stop=True)

    if st.session_state.processed_batch_zip is not None:
        st.download_button(
            label="💾 Save Batch Processed Files (.zip)",
            data=st.session_state.processed_batch_zip,
            file_name="TNRIS_Batch_Processed_Surfaces.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )

# =========================================================
# TAB 3: BATCH SHAPEFILE REPROJECTION
# =========================================================
with tab3:
    st.subheader("Batch Shapefile Reprojection & Coordinate Converter")
    st.caption("Process zipped shapefiles (e.g., contours) to reproject them and apply a TxDOT Surface Adjustment Factor.")
    
    col_shp1, col_shp2 = st.columns(2)
    
    with col_shp1:
        st.markdown("##### A. Select Target Coordinate System")
        selected_crs_label_shp = st.selectbox("Target Texas State Plane System (Shapefiles)", list(TEXAS_EPSG_DICT.keys()), key="shp_crs")
        target_epsg_shp = TEXAS_EPSG_DICT[selected_crs_label_shp]
        
    with col_shp2:
        st.markdown("##### B. Surface Adjustment Factor")
        county_saf_dict = load_surface_adjustment_factors("surface-adjustment-factors.csv")
        
        if county_saf_dict:
            county_options = ["None (Grid Coordinates - 1.000000)"] + sorted(list(county_saf_dict.keys()))
            selected_county_shp = st.selectbox(
                "Select Texas County (Surface Adjustment Factor)",
                options=county_options,
                index=0,
                key="shp_county",
                help="Select a Texas county to scale grid coordinates to surface coordinates."
            )
            
            if selected_county_shp != "None (Grid Coordinates - 1.000000)":
                surface_factor_shp = county_saf_dict[selected_county_shp]
                st.info(f"Applying **{selected_county_shp} County** Surface Adjustment Factor: `{surface_factor_shp:.6f}`")
            else:
                surface_factor_shp = 1.0
        else:
            st.warning("⚠️ `surface-adjustment-factors.csv` not found in root folder. Defaulting factor to 1.0.")
            surface_factor_shp = 1.0

    st.markdown("---")
    st.markdown("##### C. Upload .zip shapefiles")
    
    st.markdown("""
    <style>
        [data-testid="stFileUploaderDropzone"] ~ * {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if 'batch_shp_files_dict' not in st.session_state:
        st.session_state.batch_shp_files_dict = {}
    if 'ignored_shp_files' not in st.session_state:
        st.session_state.ignored_shp_files = set()
    if 'shp_uploader_key' not in st.session_state:
        st.session_state.shp_uploader_key = 0

    uploaded_shp_files = st.file_uploader(
        "Upload Zipped Shapefiles (.zip)", 
        type=["zip"], 
        accept_multiple_files=True,
        key=f"shp_file_uploader_{st.session_state.shp_uploader_key}"
    )

    if uploaded_shp_files:
        for f in uploaded_shp_files:
            if f.name not in st.session_state.batch_shp_files_dict and f.name not in st.session_state.ignored_shp_files:
                st.session_state.batch_shp_files_dict[f.name] = f

    total_size_bytes_tab3 = sum(f.size for f in st.session_state.batch_shp_files_dict.values())
    total_size_mb_tab3 = total_size_bytes_tab3 / (1024 * 1024)
    is_over_limit_tab3 = total_size_mb_tab3 > MAX_UPLOAD_MB

    if st.session_state.batch_shp_files_dict:
        col_hdr1, col_hdr2 = st.columns([4, 1])
        with col_hdr1:
            st.caption(f"📂 **Uploaded Files ({len(st.session_state.batch_shp_files_dict)})** — Total Size: **{total_size_mb_tab3:.2f} MB / {MAX_UPLOAD_MB:.0f} MB**")
        with col_hdr2:
            if st.button("🗑️ Clear All", key="clear_shp", use_container_width=True):
                st.session_state.batch_shp_files_dict.clear()
                st.session_state.ignored_shp_files.clear()
                st.session_state.shp_uploader_key += 1
                st.session_state.processed_batch_shp_zip = None
                gc.collect()
                st.rerun()

        with st.container(height=180):
            files_to_delete = []
            for file_name, file_obj in list(st.session_state.batch_shp_files_dict.items()):
                col_info, col_del = st.columns([5, 1])
                size_kb = file_obj.size / 1024
                size_str = f"{size_kb/1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
                
                with col_info:
                    st.markdown(f"• **`{file_name}`** &nbsp;·&nbsp; `{size_str}`")
                with col_del:
                    if st.button("❌", key=f"del_shp_{file_name}", help=f"Remove {file_name}"):
                        files_to_delete.append(file_name)

            if files_to_delete:
                for fname in files_to_delete:
                    del st.session_state.batch_shp_files_dict[fname]
                    st.session_state.ignored_shp_files.add(fname)
                st.rerun()

    if is_over_limit_tab3:
        st.error(f"⚠️ Total upload size ({total_size_mb_tab3:.1f} MB) exceeds the {MAX_UPLOAD_MB:.0f} MB limit! Please remove one or more files before processing.")

    uploaded_shp_list = list(st.session_state.batch_shp_files_dict.values())
    process_disabled_tab3 = is_over_limit_tab3 or (len(uploaded_shp_list) == 0)
    
    if st.button("⚡ Process Shapefiles", type="primary", use_container_width=True, disabled=process_disabled_tab3):
        st.session_state.processed_batch_shp_zip = None
        gc.collect()
        
        st.info(f"Starting batch conversion for {len(uploaded_shp_list)} shapefiles...")
        
        batch_shp_zip_buffer = io.BytesIO()
        batch_shp_prog = st.progress(0)
        batch_shp_status = st.empty()
        
        # --- BEFORE TAB 3 PROCESSING LOOP ---
        enforce_cloud_memory_limit(limit_mb=RAM_limit, hard_stop=True)
        
        
        with zipfile.ZipFile(batch_shp_zip_buffer, "w", zipfile.ZIP_DEFLATED) as b_zip:
            for idx, up_f in enumerate(uploaded_shp_list):
                
                # Soft RAM check inside the loop
                if not enforce_cloud_memory_limit(limit_mb=RAM_limit-50, hard_stop=False):
                    st.warning(f"⚠️ Memory limit approaching. Processed {idx} out of {len(uploaded_shp_list)} shapefiles. The partial batch is ready below.")
                    break
                
                filename = up_f.name
                batch_shp_status.text(f"Processing ({idx+1}/{len(uploaded_shp_list)}): {filename}")
                base_name = os.path.splitext(filename)[0]
                
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with zipfile.ZipFile(io.BytesIO(up_f.getvalue()), 'r') as z:
                            z.extractall(temp_dir)
                            
                        shp_files = glob.glob(os.path.join(temp_dir, "**", "*.shp"), recursive=True)
                        if not shp_files:
                            st.warning(f"No .shp file found in {filename}. Skipping.")
                            continue
                        
                        for shp_path in shp_files:
                            gdf = gpd.read_file(shp_path)
                            if gdf.crs is not None:
                                gdf = gdf.to_crs(target_epsg_shp)
                            else:
                                gdf.set_crs(target_epsg_shp, inplace=True)
                                
                            if surface_factor_shp != 1.0:
                                gdf['geometry'] = gdf.geometry.scale(
                                    xfact=surface_factor_shp, 
                                    yfact=surface_factor_shp, 
                                    zfact=1.0, 
                                    origin=(0, 0, 0)
                                )
                            
                            gdf.to_file(shp_path)
                            del gdf
                            
                        processed_zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(processed_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as p_zip:
                            for root, _, files in os.walk(temp_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, temp_dir)
                                    p_zip.write(file_path, arcname=arcname)
                                    
                        processed_zip_buffer.seek(0)
                        out_zip_filename = f"{base_name}_{target_epsg_shp.replace(':', '_')}.zip"
                        b_zip.writestr(out_zip_filename, processed_zip_buffer.read())
                        processed_zip_buffer.close()
                        
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                    
                batch_shp_prog.progress((idx + 1) / len(uploaded_shp_list))
                gc.collect()

        batch_shp_status.text("✅ Batch shapefile processing complete!")
        batch_shp_zip_buffer.seek(0)
        st.session_state.processed_batch_shp_zip = batch_shp_zip_buffer.getvalue()
        batch_shp_zip_buffer.close()
        gc.collect()

    if st.session_state.get('processed_batch_shp_zip') is not None:
        st.download_button(
            label="💾 Save Batch Processed Shapefiles (.zip)",
            data=st.session_state.processed_batch_shp_zip,
            file_name="TNRIS_Batch_Processed_Shapefiles.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
        
        
        
# =========================================================
# TAB 4: 3D TERRAIN VIEWER & ALIGNMENT PROFILER
# =========================================================
with tab4:
    st.subheader("Interactive 3D Terrain Viewer & Alignment Profiler")
    
    st.markdown("""
    **Navigation Controls:** 
    * 🔄 **Rotate:** Left-Click + Drag 
    * ✋ **Pan:** Right-Click + Drag (or `Shift` + Left-Click + Drag)
    * 🔍 **Zoom:** Scroll Wheel
    """)
    st.caption("Upload up to 4 files (DEMs or LandXMLs) and use checkboxes to toggle visibility. Models must share the same coordinate system.")
    
    col_v1, col_v2 = st.columns([3, 1])
    
    with col_v1:
        uploaded_3d_files = st.file_uploader("Upload up to 4 DEMs (.tif) or LandXMLs (.xml)", type=["tif", "xml"], accept_multiple_files=True, key="3d_uploader",help="Upload up to 4 raster DEMs (.tif) or LandXML meshes (.xml) sharing a common coordinate system for simultaneous 3D comparison.")
        
        if len(uploaded_3d_files) > 4:
            st.warning("Maximum of 4 files allowed. Only the first 4 will be available for viewing.")
            uploaded_3d_files = uploaded_3d_files[:4]
            
        selected_files = []
        if uploaded_3d_files:
            st.markdown("**Toggle Visibility:**")
            cols = st.columns(len(uploaded_3d_files))
            for i, f in enumerate(uploaded_3d_files):
                with cols[i]:
                    if st.checkbox(f.name, value=True, key=f"chk_{f.name}", help=f"Toggle 3D rendering visibility for {f.name}."):
                        selected_files.append(f)
        
    with col_v2:
        st.markdown("##### Viewer Settings")
        chart_height = st.slider("Viewer Height (px)", min_value=500, max_value=1200, value=900, step=50, help="Define the vertical display height (in pixels) of the interactive 3D Plotly rendering canvas." )
        z_exaggeration = st.slider("Vertical Exaggeration", min_value=1.0, max_value=20.0, value=10.0, step=0.5, help="Scale vertical elevation heights relative to horizontal dimensions (1.0 = true 1:1 scale).")
        downsample_factor = st.slider("GeoTIFF Downsample Factor", min_value=1, max_value=20, value=5, help="Skip grid pixels to reduce browser memory load and accelerate 3D rendering performance (1 = full resolution).")
        colorscale = st.selectbox("Color Theme", ["Earth", "Viridis", "Cividis", "Turbo", "Gray"], index=0, help="Select a color gradient palette theme for rendering surface elevations.")

    parsed_surfaces_for_profile = {}

    if selected_files:
        with st.spinner("Processing 3D Models..."):
            
            # --- BEFORE TAB 4 PLOT LOOP ---
            enforce_cloud_memory_limit(limit_mb=RAM_limit, hard_stop=True)
            
            
            parsed_data = []
            
            global_zmin, global_zmax = float('inf'), float('-inf')
            global_xmin, global_xmax = float('inf'), float('-inf')
            global_ymin, global_ymax = float('inf'), float('-inf')
            
            for file_obj in selected_files:
                
                # Soft RAM check inside the loop
                if not enforce_cloud_memory_limit(limit_mb=RAM_limit-50, hard_stop=False):
                    st.warning(f"⚠️ Rendering stopped early due to memory limits. Displaying the first {len(parsed_data)} models.")
                    break
                
                file_ext = os.path.splitext(file_obj.name)[1].lower()
                
                try:
                    if file_ext == ".tif":
                        with rasterio.open(file_obj) as src:
                            z_data = src.read(1)
                            nodata = src.nodata
                            
                            if nodata is not None:
                                z_data = np.where(z_data == nodata, np.nan, z_data)
                            
                            parsed_surfaces_for_profile[file_obj.name] = file_obj 
                            
                            z_render = z_data[::downsample_factor, ::downsample_factor]
                            left, bottom, right, top = src.bounds
                            x = np.linspace(left, right, z_render.shape[1])
                            y = np.linspace(top, bottom, z_render.shape[0])
                            
                            local_zmin, local_zmax = np.nanmin(z_data), np.nanmax(z_data)
                            global_zmin, global_zmax = min(global_zmin, local_zmin), max(global_zmax, local_zmax)
                            global_xmin, global_xmax = min(global_xmin, left), max(global_xmax, right)
                            global_ymin, global_ymax = min(global_ymin, bottom), max(global_ymax, top)
                            
                            parsed_data.append({
                                'type': 'surface',
                                'x': x, 'y': y, 'z': z_render,
                                'name': file_obj.name
                            })
                    
                    elif file_ext == ".xml":
                        file_obj.seek(0)
                        xml_str = file_obj.read().decode('utf-8')
                        root = ET.fromstring(xml_str)
                        
                        pts_dict, faces = {}, []
                        for elem in root.iter():
                            if elem.tag.endswith('P'):
                                pt_id = int(elem.attrib['id'])
                                coords = list(map(float, elem.text.split()))
                                if len(coords) >= 3:
                                    pts_dict[pt_id] = (coords[1], coords[0], coords[2])
                            elif elem.tag.endswith('F'):
                                face_indices = list(map(int, elem.text.split()))
                                if len(face_indices) == 3:
                                    faces.append(face_indices)
                        
                        if pts_dict and faces:
                            id_to_idx = {pid: i for i, pid in enumerate(pts_dict.keys())}
                            x = [pts_dict[pid][0] for pid in pts_dict.keys()]
                            y = [pts_dict[pid][1] for pid in pts_dict.keys()]
                            z = [pts_dict[pid][2] for pid in pts_dict.keys()]
                            i_idx = [id_to_idx[f[0]] for f in faces]
                            j_idx = [id_to_idx[f[1]] for f in faces]
                            k_idx = [id_to_idx[f[2]] for f in faces]
                            
                            local_zmin, local_zmax = min(z), max(z)
                            global_zmin, global_zmax = min(global_zmin, local_zmin), max(global_zmax, local_zmax)
                            global_xmin, global_xmax = min(global_xmin, min(x)), max(global_xmax, max(x))
                            global_ymin, global_ymax = min(global_ymin, min(y)), max(global_ymax, max(y))
                            
                            parsed_data.append({
                                'type': 'mesh',
                                'x': x, 'y': y, 'z': z,
                                'i': i_idx, 'j': j_idx, 'k': k_idx,
                                'name': file_obj.name
                            })
                            
                except Exception as e:
                    st.error(f"Error processing {file_obj.name}: {e}")
                    
            enforce_cloud_memory_limit(limit_mb=RAM_limit, hard_stop=True)    

            if parsed_data:
                fig = go.Figure()
                for idx, data in enumerate(parsed_data):
                    show_legend = (idx == 0)
                    if data['type'] == 'surface':
                        fig.add_trace(go.Surface(
                            x=data['x'], y=data['y'], z=data['z'],
                            colorscale=colorscale, cmin=global_zmin, cmax=global_zmax,
                            showscale=show_legend, colorbar=dict(title="Elevation") if show_legend else None,
                            name=data['name']
                        ))
                    elif data['type'] == 'mesh':
                        fig.add_trace(go.Mesh3d(
                            x=data['x'], y=data['y'], z=data['z'],
                            i=data['i'], j=data['j'], k=data['k'],
                            intensity=data['z'], colorscale=colorscale, cmin=global_zmin, cmax=global_zmax,
                            showscale=show_legend, colorbar=dict(title="Elevation") if show_legend else None,
                            name=data['name']
                        ))

                dx = global_xmax - global_xmin
                dy = global_ymax - global_ymin
                max_horizontal_dim = max(dx, dy) if max(dx, dy) > 0 else 1.0
                
                aspect_x = dx / max_horizontal_dim
                aspect_y = dy / max_horizontal_dim
                dz = global_zmax - global_zmin
                aspect_z = (dz / max_horizontal_dim) * z_exaggeration if dz > 0 else 0.1 * z_exaggeration

                fig.update_layout(
                    uirevision='locked_camera',
                    scene=dict(
                        aspectmode='manual',
                        aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
                        xaxis_title='Easting / X', yaxis_title='Northing / Y', zaxis_title='Elevation'
                    ),
                    height=chart_height,
                    margin=dict(l=0, r=0, b=0, t=0)
                )
                
                viewer_config = {
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan3d', 'orbitRotation', 'tableRotation', 'resetCameraDefault3d', 'resetCameraLastSave3d']
                }
                st.plotly_chart(fig, use_container_width=True, config=viewer_config)

    from pyproj import Transformer
    from affine import Affine
    #import matplotlib.cm as cm
    import branca.colormap as cmp
    import matplotlib.colors as mcolors
    import matplotlib as mpl
    
    
    # =========================================================
    # DYNAMIC ALIGNMENT PROFILE TOOL SECTION
    # =========================================================
    st.markdown("---")
    
    if "prof_map_key" not in st.session_state:
        st.session_state.prof_map_key = 0
    if "drawn_alignment" not in st.session_state:
        st.session_state.drawn_alignment = None
        
    with st.expander("📐 Dynamic Alignment Profile Tool (Interactive Map Cut)", expanded=False):
        st.markdown("📍 **How to use:** Use the **Draw Polyline** tool on the left side of the map to trace your alignment over the terrain. Double-click to finish the line.")
        
        if not parsed_surfaces_for_profile:
            st.warning("Please upload and process at least one GeoTIFF (.tif) DEM file above to generate the base map.")
        else:
            col_map1, col_map2 = st.columns([4, 1])
            
            with col_map1:
                init_lats = []
                init_lons = []
                for name, obj in parsed_surfaces_for_profile.items():
                    try:
                        obj.seek(0)
                        with rasterio.open(obj) as src:
                            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                            left, bottom, right, top = src.bounds
                            for x, y in [(left, bottom), (right, bottom), (left, top), (right, top)]:
                                lon, lat = transformer.transform(x, y)
                                init_lons.append(lon)
                                init_lats.append(lat)
                    except Exception:
                        pass

                if init_lats and init_lons:
                    min_lon, max_lon = min(init_lons), max(init_lons)
                    min_lat, max_lat = min(init_lats), max(init_lats)
                    prof_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
                    prof_zoom = get_zoom_from_bounds(min_lon, min_lat, max_lon, max_lat)
                else:
                    prof_center = st.session_state.map_center
                    prof_zoom = st.session_state.map_zoom

                m_profile = folium.Map(location=prof_center, zoom_start=prof_zoom, tiles="OpenStreetMap")
                map_bounds = []
                
                legend_css = """
                <style>
                svg text {
                    font-weight: 900 !important;
                    fill: #000000 !important;
                    text-shadow: 
                        2px 0px 0px #FFFFFF, 
                        -2px 0px 0px #FFFFFF, 
                        0px 2px 0px #FFFFFF, 
                        0px -2px 0px #FFFFFF, 
                        1px 1px 0px #FFFFFF, 
                        -1px -1px 0px #FFFFFF, 
                        1px -1px 0px #FFFFFF, 
                        -1px 1px 0px #FFFFFF !important;
                }
                </style>
                """
                m_profile.get_root().header.add_child(folium.Element(legend_css))
                
                #terrain_cmap = cm.get_cmap('terrain', 15)
                terrain_cmap = mpl.colormaps['terrain'].resampled(15)
                hex_colors = [mcolors.to_hex(terrain_cmap(i)) for i in np.linspace(0, 1, 15)]
                
                processed_dems = []
                global_min = float('inf')
                global_max = float('-inf')
                
                for name, obj in parsed_surfaces_for_profile.items():
                    obj.seek(0)
                    with rasterio.open(obj) as src:
                        dst_crs = 'EPSG:4326'
                        
                        transform, width, height = calculate_default_transform(
                            src.crs, dst_crs, src.width, src.height, *src.bounds
                        )
                        
                        max_dim = 500.0
                        ds_factor = int(max(1, max(width, height) / max_dim))
                        
                        dst_width = max(1, width // ds_factor)
                        dst_height = max(1, height // ds_factor)
                        scaled_transform = transform * Affine.scale(ds_factor)
                        
                        destination = np.zeros((dst_height, dst_width), dtype=np.float32)
                        
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=destination,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=scaled_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.average
                        )
                        
                        valid_mask = ~np.isnan(destination)
                        if src.nodata is not None:
                            valid_mask &= (destination != src.nodata)
                        
                        if valid_mask.any():
                            local_min = np.nanmin(destination[valid_mask])
                            local_max = np.nanmax(destination[valid_mask])
                            global_min = min(global_min, local_min)
                            global_max = max(global_max, local_max)
                            
                        from rasterio.transform import array_bounds
                        lon_min, lat_min, lon_max, lat_max = array_bounds(dst_height, dst_width, scaled_transform)
                        bounds = [[lat_min, lon_min], [lat_max, lon_max]]
                        map_bounds.extend(bounds)
                        
                        processed_dems.append({
                            'name': name,
                            'array': destination,
                            'mask': valid_mask,
                            'bounds': bounds
                        })

                if global_min < float('inf') and global_max > float('-inf'):
                    colormap = cmp.LinearColormap(
                        colors=hex_colors,
                        vmin=float(global_min),
                        vmax=float(global_max),
                        caption="Elevation (Project Units)"
                    )
                    colormap.add_to(m_profile)
                    
                    for dem in processed_dems:
                        dst_height, dst_width = dem['array'].shape
                        rgba_img = np.zeros((dst_height, dst_width, 4), dtype=np.uint8)
                        
                        if dem['mask'].any() and global_max > global_min:
                            norm = ((dem['array'] - global_min) / (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)
                            #colored = cm.terrain(norm / 255.0) * 255
                            colored = terrain_cmap(norm / 255.0) * 255
                            rgba_img = colored.astype(np.uint8)
                            
                        rgba_img[..., 3] = np.where(dem['mask'], 160, 0)
                        
                        folium.raster_layers.ImageOverlay(
                            image=rgba_img,
                            bounds=dem['bounds'],
                            opacity=0.9, 
                            name=dem['name']
                        ).add_to(m_profile)
                
                if map_bounds:
                    flat_lats = [pt[0] for pt in map_bounds]
                    flat_lons = [pt[1] for pt in map_bounds]
                    m_profile.fit_bounds([[min(flat_lats), min(flat_lons)], [max(flat_lats), max(flat_lons)]])

                draw = folium.plugins.Draw(
                    draw_options={
                        'polyline': {
                            'metric': False,  
                            'feet': True      
                        },
                        'polygon': False, 
                        'circle': False, 
                        'marker': False, 
                        'circlemarker': False, 
                        'rectangle': False
                    },
                    edit_options={'edit': True}
                )
                m_profile.add_child(draw)

                map_data = st_folium(
                    m_profile, 
                    key=f"prof_map_{st.session_state.prof_map_key}", 
                    width="100%", 
                    height=500,
                    returned_objects=["last_active_drawing"]
                )
                
                if map_data and map_data.get("last_active_drawing"):
                    geom = map_data["last_active_drawing"]["geometry"]
                    if geom["type"] == "LineString":
                        st.session_state.drawn_alignment = geom["coordinates"]                        
                        
            with col_map2:
                st.markdown("##### Map Tools")
                if st.button("🔍 Zoom to Files", use_container_width=True, help="Instantly re-center and zoom the map to the bounding box of your uploaded DEM files."):
                    st.session_state.prof_map_key += 1
                    st.rerun()
                if st.button("🗑️ Clear Alignment", use_container_width=True):
                    st.session_state.drawn_alignment = None
                    st.session_state.prof_map_key += 1
                    st.rerun()

        st.markdown("---")
        st.markdown("##### Process Alignment Coordinates")
        
        col_prof1, col_prof2 = st.columns([1, 2])
        with col_prof1:
            sample_step = st.number_input("Profile Sampling Interval (feet/meters)", min_value=0.5, max_value=200.0, value=5.0, step=1.0, help="Specify the distance spacing interval between cross-section elevation sample points extracted along the alignment line.")
        
        if st.button("Generate Elevation Profile Graph", type="primary"):
            if not st.session_state.drawn_alignment or len(st.session_state.drawn_alignment) < 2:
                st.error("Please draw a valid polyline on the map above first.")
            elif not parsed_surfaces_for_profile:
                st.warning("Please upload and process at least one GeoTIFF (.tif) DEM file above.")
            else:
                try:
                    profile_fig = go.Figure()
                    
                    for file_name, file_obj in parsed_surfaces_for_profile.items():
                        file_obj.seek(0)
                        with rasterio.open(file_obj) as src:
                            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                            native_vertices = [transformer.transform(lon, lat) for lon, lat in st.session_state.drawn_alignment]
                            
                            stations = [0.0]
                            sample_points = [native_vertices[0]]
                            current_station = 0.0
                            
                            for idx in range(len(native_vertices) - 1):
                                p1 = np.array(native_vertices[idx])
                                p2 = np.array(native_vertices[idx+1])
                                seg_len = np.linalg.norm(p2 - p1)
                                if seg_len == 0:
                                    continue
                                
                                num_steps = int(np.floor(seg_len / sample_step))
                                for step_i in range(1, num_steps + 1):
                                    t = (step_i * sample_step) / seg_len
                                    pt = p1 + t * (p2 - p1)
                                    current_station += sample_step
                                    stations.append(current_station)
                                    sample_points.append((pt[0], pt[1]))
                                    
                                remainder = seg_len - (num_steps * sample_step)
                                if remainder > 0.01:
                                    current_station += remainder
                                    stations.append(current_station)
                                    sample_points.append((p2[0], p2[1]))
                            
                            elevations = []
                            valid_stations = []
                            for stat, (x_coord, y_coord) in zip(stations, sample_points):
                                try:
                                    row, col = src.index(x_coord, y_coord)
                                    if 0 <= row < src.height and 0 <= col < src.width:
                                        window = rasterio.windows.Window(col, row, 1, 1)
                                        val = src.read(1, window=window)[0, 0]
                                        if val == src.nodata or np.isnan(val):
                                            elevations.append(None)
                                        else:
                                            elevations.append(float(val))
                                        valid_stations.append(stat)
                                    else:
                                        elevations.append(None)
                                        valid_stations.append(stat)
                                except Exception:
                                    elevations.append(None)
                                    valid_stations.append(stat)
                                    
                            profile_fig.add_trace(go.Scatter(
                                x=valid_stations, y=elevations,
                                mode='lines',
                                name=file_name,
                                line=dict(width=2)
                            ))
                            
                    profile_fig.update_layout(
                        title="Alignment Elevation Profile (Cross-Section)",
                        xaxis_title="Station (Length Units in Surface CRS)",
                        yaxis_title="Elevation",
                        hovermode="x unified",
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(profile_fig, use_container_width=True)
                    
                except Exception as profile_err:
                    st.error(f"Error generating profile: {profile_err}")