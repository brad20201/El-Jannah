# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import googlemaps
import os
from datetime import datetime
import logging

# --- Configuration ---
CSV_FILENAME = 'el_jannah_locations.csv'
API_KEY_ENV_VAR = 'GOOGLE_MAPS_API_KEY' # Ensure this env var is set BEFORE running streamlit
SYDNEY_CENTER_LAT = -33.8688
SYDNEY_CENTER_LON = 151.2093
DEFAULT_ZOOM = 10

# --- Route highlight colors (Gold, Silver, Bronze) ---
ROUTE_COLORS = ['gold', 'silver', '#CD7F32'] # Hex for Bronze
ROUTE_WEIGHTS = [5, 4.5, 4] # Make faster routes slightly thicker

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Hardcoded Short Names ---
address_to_name_map = {
    "104 Princes Hwy, Albion Park Rail, NSW, 2527": "El Jannah Albion Park",
    "11B Chapel Rd, Bankstown, NSW, 2200": "El Jannah Bankstown",
    "351 Windsor Rd, Baulkham Hills, NSW, 2153": "El Jannah Baulkham Hills",
    "17 Patrick St, Blacktown, NSW, 2148": "El Jannah Blacktown",
    "11-13 Burwood Road, Burwood, NSW, 2134": "El Jannah Burwood",
    "321 Queen St, Campbelltown, NSW, 2560": "El Jannah Campbelltown",
    "160 Waldron Rd, Chester Hill, NSW, 2162": "El Jannah Chester Hill",
    "34 Willoughby Rd, Crows Nest, NSW, 2065": "El Jannah Crows Nest",
    "283 Homer St, Earlwood, NSW, 2206": "El Jannah Earlwood",
    "8 South St, Granville, NSW, 2142": "El Jannah Granville",
    "33 Village Cct, Gregory Hills, NSW, 2557": "El Jannah Gregory Hills",
    "100-102 Railway Parade, Kogarah, NSW, 2217": "El Jannah Kogarah",
    "938-944 Canterbury Road, Roselands, NSW, 2196": "El Jannah Lakemba (Roselands)",
    "279-287 Macquarie Street, Liverpool, NSW, 2170": "El Jannah Liverpool",
    "2 Ultimo Place, Marsden Park, NSW, 2765": "El Jannah Marsden Park",
    "156 King St, Newtown, NSW, 2042": "El Jannah Newtown",
    "4 Century Cct, Baulkham Hills, NSW, 2153": "El Jannah Norwest",
    "535 High St, Penrith, NSW, 2750": "El Jannah Penrith",
    "260 Jersey Rd, Plumpton, NSW, 2761": "El Jannah Plumpton",
    "1-5 Yato Rd, Prestons, NSW, 2170": "El Jannah Prestons",
    "701 Punchbowl Rd, Punchbowl, NSW, 2196": "El Jannah Punchbowl",
    "16 Smithfield Rd, Smithfield, NSW, 2164": "El Jannah Smithfield",
    "3 Potters Drive, Tahmoor, NSW, 2573": "El Jannah Tahmoor",
    "77 Crown St, Wollongong, NSW, 2500": "El Jannah Wollongong",
}

# --- Helper Functions ---

@st.cache_data
def load_data(filename):
    """Loads location data from CSV and adds short names."""
    try:
        df = pd.read_csv(filename)
        if 'Address' not in df.columns or 'Latitude' not in df.columns or 'Longitude' not in df.columns: return None
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        df['Name'] = df['Address'].map(address_to_name_map).fillna(df['Address'])
        logging.info(f"Loaded {len(df)} locations from {filename}.")
        return df
    except Exception as e: logging.error(f"Error loading CSV {filename}: {e}", exc_info=True); return None

def decode_polyline(polyline_str):
    """Decodes a Google Maps encoded polyline string."""
    index, lat, lng = 0, 0, 0; coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    while index < len(polyline_str):
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63; index += 1
                result |= (byte & 0x1f) << shift; shift += 5
                if not byte >= 0x20: break
            changes[unit] = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += changes['latitude']; lng += changes['longitude']
        coordinates.append((lat / 100000.0, lng / 100000.0))
    return coordinates

def find_address_matches(api_key, partial_address):
    """Uses Google Geocoding API to find full address matches."""
    if not api_key: st.error("Google Maps API Key not found."); return []
    if not partial_address or len(partial_address) < 5: st.warning("Enter min 5 chars."); return []
    try:
        gmaps = googlemaps.Client(key=api_key)
        geocode_result = gmaps.geocode(partial_address, components={"country": "AU"},
                                       bounds={'southwest': (-34.1, 150.5), 'northeast': (-33.5, 151.5)})
        if geocode_result: return [result['formatted_address'] for result in geocode_result][:5]
        else: st.warning(f"No matches found for '{partial_address}'."); return []
    except Exception as e: st.error(f"Address matching error: {e}"); return []

def get_drive_times(api_key, origin_address, destinations_df):
    """Gets drive times from origin to multiple destinations using Distance Matrix API."""
    if not api_key: st.error("API Key Error."); return None
    if not origin_address: st.warning("Enter starting address."); return None
    try:
        gmaps = googlemaps.Client(key=api_key)
        dest_coords = list(zip(destinations_df['Latitude'], destinations_df['Longitude']))
        dist_matrix = gmaps.distance_matrix(origins=[origin_address], destinations=dest_coords, mode="driving", units="metric",
                                           departure_time=datetime.now(), traffic_model="best_guess" )
        results = []
        if dist_matrix['status'] == 'OK' and len(dist_matrix['rows']) > 0:
            elements = dist_matrix['rows'][0]['elements']
            if len(elements) == len(dest_coords):
                for i, element in enumerate(elements):
                    if element['status'] == 'OK':
                        duration_value = element.get('duration_in_traffic', {}).get('value')
                        duration_text = element.get('duration_in_traffic', {}).get('text')
                        if duration_value is None:
                             duration_value = element.get('duration', {}).get('value')
                             duration_text = element.get('duration', {}).get('text', 'N/A') + " (no traffic data)"
                        distance_text = element.get('distance', {}).get('text', 'N/A')
                        results.append({ 'destination_index': i, 'drive_time_secs': duration_value,
                                         'drive_time_text': duration_text, 'distance_text': distance_text, 'status': 'OK' })
                    else: results.append({ 'destination_index': i, 'drive_time_secs': None, 'drive_time_text': f"Route Error ({element['status']})", 'distance_text': 'N/A', 'status': element['status'] })
            else: st.error("Mismatch results."); return None
        elif dist_matrix['status'] == 'REQUEST_DENIED': st.error(f"Gmaps Error: {dist_matrix['status']}. Enable Distance Matrix API?"); return None
        elif dist_matrix['status'] != 'OK': st.error(f"Gmaps Error: {dist_matrix['status']}."); return None
        else: st.error("Empty Gmaps response."); return None
        return results
    except Exception as e: st.error(f"Distance calc error: {e}"); return None


# --- REVERTED: Directions API Function only returns routes now ---
def get_top_routes(api_key, origin_address, top_destinations_info):
    """Gets route polylines for the top N destinations using Directions API."""
    if not api_key: st.error("API Key Error."); return []
    if not origin_address or not top_destinations_info: return []

    routes_data = []
    gmaps = googlemaps.Client(key=api_key)

    for i, dest_info in enumerate(top_destinations_info):
        dest_coords = (dest_info['Latitude'], dest_info['Longitude'])
        color = ROUTE_COLORS[i]
        weight = ROUTE_WEIGHTS[i]

        logging.info(f"Requesting directions for route #{i+1}: '{origin_address}' to {dest_coords}")
        try:
            directions_result = gmaps.directions(
                origin=origin_address, destination=dest_coords,
                mode="driving", departure_time=datetime.now() )

            if directions_result and 'overview_polyline' in directions_result[0]:
                encoded_polyline = directions_result[0]['overview_polyline']['points']
                decoded_path = decode_polyline(encoded_polyline)
                routes_data.append({ 'path': decoded_path, 'color': color, 'weight': weight })
                logging.info(f"Successfully got polyline for route #{i+1}")
            else:
                 logging.warning(f"Could not retrieve polyline for route #{i+1}. Result: {directions_result}")

        except googlemaps.exceptions.ApiError as e:
            st.warning(f"Directions API Error route #{i+1}: {e}. Skip polyline. Check Directions API is enabled.")
            logging.warning(f"Directions API Error route #{i+1}: {e}", exc_info=True)
            continue
        except Exception as e:
            st.warning(f"Unexpected error route #{i+1}: {e}. Skip polyline.")
            logging.warning(f"Unexpected error route #{i+1}: {e}", exc_info=True)
            continue

    return routes_data # Only return route path data


# --- REVERTED: Map Creation Function no longer needs toll_info_map ---
def create_map(locations_df, center_lat, center_lon, zoom,
               origin_coords=None, route_polylines=None):
    """Creates Folium map with markers, origin pin, and routes."""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="cartodbpositron")

    # Add Origin Marker
    if origin_coords:
        try:
            folium.Marker( location=[origin_coords[0], origin_coords[1]], popup="Your Starting Location",
                tooltip="Your Starting Location", icon=folium.Icon(color='blue', icon='home', prefix='fa')
            ).add_to(m)
        except Exception as e: logging.error(f"Failed to add origin marker at {origin_coords}: {e}")

    # Add Destination Markers
    for idx, row in locations_df.iterrows():
        popup_title = row.get('Name', row['Address'])
        popup_html = f"<b>{popup_title}</b>"
        drive_time_str = row.get('Drive Time')
        distance_str = row.get('Distance')

        if pd.notna(drive_time_str):
            popup_html += f"<br>Drive Time: {drive_time_str}"
            if pd.notna(distance_str): popup_html += f" ({distance_str})"
        # NO Toll info added here anymore

        tooltip_text = popup_title
        if pd.notna(drive_time_str): tooltip_text += f" - {drive_time_str}"

        folium.Marker( location=[row['Latitude'], row['Longitude']], popup=folium.Popup(popup_html, max_width=250),
            tooltip=tooltip_text, icon=folium.Icon(color='green', icon_color='black', icon='utensils', prefix='fa')
        ).add_to(m)

    # Add Route Polylines
    if route_polylines:
        for route in route_polylines:
            if route.get('path'):
                folium.PolyLine( locations=route['path'], color=route.get('color', 'blue'),
                    weight=route.get('weight', 4), opacity=0.7
                ).add_to(m)

    return m

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("🚗 El Jannah Drive Time Calculator (Sydney)")
st.markdown("Enter address, find matches, select, calculate drive times (with traffic). Top 3 routes shown.")

api_key = os.getenv(API_KEY_ENV_VAR)
if not api_key:
    st.sidebar.error("CRITICAL: GOOGLE_MAPS_API_KEY env var not detected.")
    st.sidebar.error("Set it permanently or in terminal BEFORE running.")
    st.sidebar.code("$env:GOOGLE_MAPS_API_KEY = 'YOUR_KEY_HERE'")
    st.stop()

locations_df_loaded = load_data(CSV_FILENAME)

if locations_df_loaded is not None:
    # Initialize Session State (Removed toll_info_map)
    default_state = {
        'start_address_input': "Sydney CBD, NSW", 'address_matches': [], 'selected_address': None,
        'drive_times_df': locations_df_loaded.copy(), 'route_polylines': [], 'origin_coords': None
    }
    for key, value in default_state.items():
        if key not in st.session_state: st.session_state[key] = value
    for col in ['Drive Time', 'Distance', 'Drive Time (secs)']:
         if col not in st.session_state.drive_times_df.columns: st.session_state.drive_times_df[col] = pd.NA

    # --- Sidebar ---
    st.sidebar.header("Your Location")
    st.session_state.start_address_input = st.sidebar.text_input( "Start Typing Address:",
        value=st.session_state.start_address_input, key="addr_input_widget" )
    find_matches_button = st.sidebar.button("Find Address Matches", key="find_button")

    if find_matches_button:
        st.session_state.selected_address = None; st.session_state.route_polylines = []
        st.session_state.origin_coords = None # Clear old data
        with st.spinner("Finding address matches..."):
            st.session_state.address_matches = find_address_matches(api_key, st.session_state.start_address_input)

    if st.session_state.address_matches:
        st.sidebar.markdown("---"); st.sidebar.markdown("**Select the correct address:**")
        st.session_state.selected_address = st.sidebar.radio( "Potential Matches:",
            options=st.session_state.address_matches, key="address_selector",
            index=None if st.session_state.selected_address not in st.session_state.address_matches else st.session_state.address_matches.index(st.session_state.selected_address) )
        st.sidebar.markdown("---")

    address_to_calculate = st.session_state.selected_address if st.session_state.selected_address else st.session_state.start_address_input
    calculate_button = st.sidebar.button( "Calculate Drive Times", key="calc_button", disabled=(not address_to_calculate) )

    # --- Calculation Logic ---
    if calculate_button:
        if address_to_calculate:
             # Clear previous results first
             st.session_state.route_polylines = []; st.session_state.origin_coords = None

             # Geocode origin address for the pin
             try:
                 with st.spinner(f"Locating '{address_to_calculate}'..."):
                     gmaps_client = googlemaps.Client(key=api_key)
                     geocode_origin_result = gmaps_client.geocode(address_to_calculate)
                     if geocode_origin_result:
                         loc = geocode_origin_result[0]['geometry']['location']
                         st.session_state.origin_coords = (loc['lat'], loc['lng'])
                         logging.info(f"Geocoded origin '{address_to_calculate}' to {st.session_state.origin_coords}")
                     else:
                         st.warning(f"Could not geocode starting address '{address_to_calculate}'. Origin pin won't be shown.")
             except Exception as e:
                  st.warning(f"Error geocoding starting address: {e}. Origin pin won't be shown.")
                  logging.error(f"Error geocoding origin '{address_to_calculate}': {e}", exc_info=True)

             # Calculate drive times
             with st.spinner(f"Calculating distances from '{address_to_calculate}'..."):
                 drive_time_results = get_drive_times(api_key, address_to_calculate, locations_df_loaded)

             if drive_time_results:
                 time_map, dist_map, secs_map = {}, {}, {}
                 valid_results_for_routes = []
                 for result in drive_time_results:
                     original_index = locations_df_loaded.index[result['destination_index']]
                     time_map[original_index] = result['drive_time_text']
                     dist_map[original_index] = result['distance_text']
                     secs_map[original_index] = result['drive_time_secs']
                     if result['drive_time_secs'] is not None:
                          valid_results_for_routes.append({'original_index': original_index, 'secs': result['drive_time_secs']})

                 st.session_state.drive_times_df['Drive Time'] = st.session_state.drive_times_df.index.map(time_map)
                 st.session_state.drive_times_df['Distance'] = st.session_state.drive_times_df.index.map(dist_map)
                 st.session_state.drive_times_df['Drive Time (secs)'] = pd.to_numeric(st.session_state.drive_times_df.index.map(secs_map), errors='coerce')

                 # Get Top 3 Routes (polylines only)
                 if valid_results_for_routes:
                     valid_results_for_routes.sort(key=lambda x: x['secs'])
                     top_3_indices = [res['original_index'] for res in valid_results_for_routes[:3]]
                     top_3_dest_info = [
                         {**locations_df_loaded.loc[idx, ['Latitude', 'Longitude']].to_dict(), 'original_index': idx}
                         for idx in top_3_indices ] # Pass original index if needed by route func

                     with st.spinner("Calculating fastest route paths..."):
                          # Call the reverted function, only getting polylines
                          st.session_state.route_polylines = get_top_routes(api_key, address_to_calculate, top_3_dest_info)

                 st.rerun()
             else: # Clear results if drive time calculation failed
                 st.session_state.drive_times_df['Drive Time'] = pd.NA; st.session_state.drive_times_df['Distance'] = pd.NA
                 st.session_state.drive_times_df['Drive Time (secs)'] = pd.NA
                 st.session_state.route_polylines = []; st.session_state.origin_coords = None
        else:
            st.sidebar.warning("Please enter an address or select a match first.")

    # --- Main Area Layout ---
    col1, col2 = st.columns([2, 1]) # Keep adjusted ratio [2, 1]

    with col1: # Map Column
        st.subheader("Map")
        folium_map = create_map(
            st.session_state.drive_times_df, SYDNEY_CENTER_LAT, SYDNEY_CENTER_LON, DEFAULT_ZOOM,
            origin_coords=st.session_state.get('origin_coords'),
            route_polylines=st.session_state.get('route_polylines') # Pass only routes now
            )
        map_output = st_folium(folium_map, width='100%', height=600, returned_objects=[])

        # --- Removed Legend ---

    with col2: # Table Column
        st.subheader("Locations & Drive Times")
        source_df = st.session_state.drive_times_df
        if 'Drive Time (secs)' in source_df.columns and source_df['Drive Time (secs)'].notna().any():
            sorted_df = source_df.sort_values(by='Drive Time (secs)', na_position='last')
        else: sorted_df = source_df
        # Use 'Name' column, hide index
        display_df_final = sorted_df[['Name', 'Drive Time', 'Distance']].copy()
        st.dataframe(display_df_final, height=600, use_container_width=True, hide_index=True)

    # --- Footer/Disclaimer ---
    st.sidebar.markdown("---")
    # Updated disclaimer to remove toll mention
    st.sidebar.caption("Tips: Enter address, find matches, select, then calculate. Ensure API key has Geocoding, Distance Matrix & Directions APIs enabled.")
    st.sidebar.caption(f"Data loaded from: {CSV_FILENAME}")

else:
    st.warning("Could not load El Jannah location data.")