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
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="TNRIS Data & Surface Suite", layout="wide")

# Initialize Session State (Removed local output_dir)
if 'map_center' not in st.session_state:
    st.session_state.map_center = [31.9686, -99.9018]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 6
if 'last_drawing' not in st.session_state:
    st.session_state.last_drawing = None
if 'programmatic_zoom' not in st.session_state:
    st.session_state.programmatic_zoom = False
if 'ready_zip_data' not in st.session_state:
    st.session_state.ready_zip_data = None
if 'processed_batch_zip' not in st.session_state:
    st.session_state.processed_batch_zip = None

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
st.title("🗺️ TxGIO (TNRIS) GIS Data Downloader And Processor.")
tab1, tab2, tab3 = st.tabs([
    "1. TxGIO (TNRIS) Tile Downloader", 
    "2. Batch LandXML & Coordinate Converter", 
    "3. Batch Shapefile Reprojection"
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
                    st.session_state.programmatic_zoom = True 
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
        
        # 1. Initialize the map (defaults to OpenStreetMap)
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles="OpenStreetMap")
        
        # 2. Add Google Satellite Hybrid Layer
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite Hybrid",
            overlay=False,
            control=True
        ).add_to(m)
            
        # 3. Add the shapefile bounding box if selected
        if selected_shp:
            active_gdf = all_shapefiles[selected_shp]
            minx, miny, maxx, maxy = active_gdf.total_bounds
            folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]], color="blue", fill=False, weight=1).add_to(m)
            
        # 4. Add the drawing tool
        draw = folium.plugins.Draw(
            draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True},
            edit_options={'edit': False}
        )
        m.add_child(draw)
        
        # 5. Add previously drawn selection (if any)
        if st.session_state.last_drawing and selected_shp:
            drawn_geom = shape(st.session_state.last_drawing["geometry"])
            active_gdf = all_shapefiles[selected_shp]
            intersecting_tiles = active_gdf[active_gdf.intersects(drawn_geom)]
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

        # ---> ADD LAYER CONTROL HERE (LAST) <---
        folium.LayerControl(position="topright").add_to(m)

        # Render the map
        map_data = st_folium(m, width="100%", height=450, returned_objects=["last_active_drawing", "center", "zoom"])
        
        if map_data and map_data.get("center"):
            if st.session_state.get("programmatic_zoom"):
                st.session_state.programmatic_zoom = False
            else:
                st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
                st.session_state.map_zoom = map_data["zoom"]
            
        if map_data and map_data.get("last_active_drawing"):
            if map_data["last_active_drawing"] != st.session_state.last_drawing:
                st.session_state.last_drawing = map_data["last_active_drawing"]
                st.session_state.ready_zip_data = None
                st.rerun()

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if not intersecting_tiles.empty and st.button("📍 Zoom to Selected", use_container_width=True):
                minx, miny, maxx, maxy = intersecting_tiles.total_bounds
                st.session_state.map_center = [(miny + maxy) / 2, (minx + maxx) / 2]
                st.session_state.map_zoom = get_zoom_from_bounds(minx, miny, maxx, maxy)
                st.session_state.programmatic_zoom = True
                st.rerun()
        with col_btn2:
            if st.session_state.last_drawing is not None and st.button("🗑️ Clear Selection", use_container_width=True):
                st.session_state.last_drawing = None
                st.session_state.ready_zip_data = None
                st.rerun()

    with col2:
        st.subheader("Intersecting Tiles")
        if not intersecting_tiles.empty:
            st.success(f"Matched {len(tiles_to_download_ids)} tiles.")
            display_cols = [id_field]
            if collection_field and collection_field in intersecting_tiles.columns and collection_field != id_field:
                display_cols.append(collection_field)
            st.dataframe(intersecting_tiles[display_cols], hide_index=True, use_container_width=True, height=400)
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

        progress_bar = st.progress(0)
        status_text = st.empty()
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, (dem_id, download_url, filename, name_no_ext) in enumerate(files_to_download):
                status_text.text(f"Downloading ({i+1}/{len(files_to_download)}): {filename}")
                try:
                    response = requests.get(download_url, stream=True)
                    response.raise_for_status()
                    content = response.content
                    
                    # Write directly to memory zip
                    zip_file.writestr(filename, content)
                except Exception as e:
                    st.error(f"Failed on {filename}: {e}")
                progress_bar.progress((i + 1) / len(files_to_download))

        status_text.text("✅ All downloads complete!")
        zip_buffer.seek(0)
        st.session_state.ready_zip_data = zip_buffer.getvalue()

    if st.session_state.ready_zip_data is not None:
        st.download_button(
            label="💾 Save Downloaded Tiles (.zip)",
            data=st.session_state.ready_zip_data,
            file_name=f"TNRIS_Downloads_{selected_item}.zip",
            mime="application/zip",
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

    if st.session_state.batch_files_dict:
        col_hdr1, col_hdr2 = st.columns([4, 1])
        with col_hdr1:
            st.caption(f"📂 **Uploaded Files ({len(st.session_state.batch_files_dict)})**")
        with col_hdr2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.batch_files_dict.clear()
                st.session_state.ignored_dem_files.clear()
                st.session_state.dem_uploader_key += 1
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
                
                

    uploaded_batch_files_list = list(st.session_state.batch_files_dict.values())
             
    if st.button("⚡ Process Batch Files", type="primary", use_container_width=True):
        raster_sources = []
        
        if uploaded_batch_files_list:
            for up_f in uploaded_batch_files_list:
                raster_sources.append((up_f.name, io.BytesIO(up_f.read())))
        else:
            st.error("No raster files uploaded. Upload files first.")
            st.stop()
            
        st.info(f"Starting batch conversion for {len(raster_sources)} tiles...")
        
        batch_zip_buffer = io.BytesIO()
        batch_prog = st.progress(0)
        batch_status = st.empty()
        
        with zipfile.ZipFile(batch_zip_buffer, "w", zipfile.ZIP_DEFLATED) as b_zip:
            for idx, (filename, file_source) in enumerate(raster_sources):
                batch_status.text(f"Processing ({idx+1}/{len(raster_sources)}): {filename}")
                base_name = os.path.splitext(filename)[0]
                
                try:
                    with rasterio.open(file_source) as src:
                        transform, width, height = calculate_default_transform(
                            src.crs, target_epsg, src.width, src.height, *src.bounds
                        )
                        kwargs = src.meta.copy()
                        kwargs.update({
                            'crs': target_epsg,
                            'transform': transform,
                            'width': width,
                            'height': height
                        })
                        
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
                            
                            # Write directly to memory zip
                            b_zip.writestr(out_filename, xml_str)
                            
                        else:  # Reprojected GeoTIFF (.tif)
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
                            tif_bytes = tif_mem.getvalue()
                            
                            # Write directly to memory zip
                            b_zip.writestr(out_filename, tif_bytes)                            
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                    
                batch_prog.progress((idx + 1) / len(raster_sources))

        batch_status.text("✅ Batch processing complete!")
        batch_zip_buffer.seek(0)
        st.session_state.processed_batch_zip = batch_zip_buffer.getvalue()

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

    if st.session_state.batch_shp_files_dict:
        col_hdr1, col_hdr2 = st.columns([4, 1])
        with col_hdr1:
        # Fixed caption formatting
            st.caption(f"📂 **Uploaded Files ({len(st.session_state.batch_shp_files_dict)})**")
        with col_hdr2:
            if st.button("🗑️ Clear All", key="clear_shp", use_container_width=True):
                st.session_state.batch_shp_files_dict.clear()
                st.session_state.ignored_shp_files.clear()
                st.session_state.shp_uploader_key += 1
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


    uploaded_shp_list = list(st.session_state.batch_shp_files_dict.values())
    
    if st.button("⚡ Process Shapefiles", type="primary", use_container_width=True):
        if not uploaded_shp_list:
            st.error("Upload at least one zipped shapefile.")
            st.stop()
            
        st.info(f"Starting batch conversion for {len(uploaded_shp_list)} shapefiles...")
        
        batch_shp_zip_buffer = io.BytesIO()
        batch_shp_prog = st.progress(0)
        batch_shp_status = st.empty()
        
        with zipfile.ZipFile(batch_shp_zip_buffer, "w", zipfile.ZIP_DEFLATED) as b_zip:
            for idx, up_f in enumerate(uploaded_shp_list):
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
                        
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                    
                batch_shp_prog.progress((idx + 1) / len(uploaded_shp_list))

        batch_shp_status.text("✅ Batch shapefile processing complete!")
        batch_shp_zip_buffer.seek(0)
        st.session_state.processed_batch_shp_zip = batch_shp_zip_buffer.getvalue()

    if st.session_state.get('processed_batch_shp_zip') is not None:
        st.download_button(
            label="💾 Save Batch Processed Shapefiles (.zip)",
            data=st.session_state.processed_batch_shp_zip,
            file_name="TNRIS_Batch_Processed_Shapefiles.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )