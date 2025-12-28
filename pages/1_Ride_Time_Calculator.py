import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gpx_tool
import physics_engine

# --- HELPER FUNCTION ---
def get_wind_cardinal(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25)/22.5)
    return dirs[ix % 16]

st.set_page_config(page_title="Ride Time Calculator", page_icon="⏱️")

# --- SIDEBAR (Standard Inputs) ---
st.sidebar.title("🚴 Simulation Settings")

# 1. RIDER PROFILE
st.sidebar.header("1. Rider Profile")
rider_mass = st.sidebar.number_input("Rider Mass (kg)", value=75.0, step=0.5, min_value=0.0)
bike_mass = st.sidebar.number_input("Bike Mass (kg)", value=8.5, step=0.1, min_value=0.0)

# 2. AERODYNAMICS
st.sidebar.header("2. Aerodynamics")
cda_choice = st.sidebar.selectbox("Riding Position", 
    ["TT Bike (Aero)", "Road Bike (Drops)", "Road Bike (Hoods)", "Relaxed"], index=2)
cda_map = {"TT Bike (Aero)": 0.23, "Road Bike (Drops)": 0.29, "Road Bike (Hoods)": 0.32, "Relaxed": 0.38}
cda = cda_map[cda_choice]

# 3. ROLLING RESISTANCE
st.sidebar.header("3. Tyres & Road")
crr_choice = st.sidebar.selectbox("Tyre/Road Quality", 
    ["Race Tyres / Smooth Road", "Standard Tyres / Average Road", "Gravel / Rough Road"], index=1)
crr_map = {"Race Tyres / Smooth Road": 0.0035, "Standard Tyres / Average Road": 0.0050, "Gravel / Rough Road": 0.0080}
crr = crr_map[crr_choice]

# 4. WEATHER
st.sidebar.header("4. Weather")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 0.0, step=0.5)

# Wind Logic (Dropdown + Manual Override)
wind_cardinal = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
wind_choice = st.sidebar.selectbox("Wind Origin", list(wind_cardinal.keys()) + ["Manual Degrees"], index=0)

if wind_choice == "Manual Degrees":
    wind_deg = st.sidebar.number_input("Wind Direction (0-360°)", 0, 360, 0)
else:
    wind_deg = wind_cardinal[wind_choice]
    
st.sidebar.caption(f"Wind coming from: {wind_deg}° ({get_wind_cardinal(wind_deg)})")

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #666666; font-size: 12px;">
        Designed & Engineered by<br>
        <strong>Dyfan Davies</strong>
    </div>
    """,
    unsafe_allow_html=True
)

# --- MAIN PAGE ---
st.title("⏱️ Ride Time Calculator")
st.markdown("""
**Predict your finish time based on a fixed Power Output.**
\nThis tool runs the physics engine forward: *Input Watts $\\rightarrow$ Output Speed*.
""")

# --- FILE INPUT SECTION ---
col_upload, col_demo = st.columns([2, 1])

with col_upload: 
        uploaded_file = st.file_uploader(
            "Upload GPX Course", 
            type=['gpx'], 
            help="Ensure name of file ends with '.gpx'"
        )

with col_demo: 
    st.write("")
    st.write("")
    # Updated dropdown with distinct profiles
    demo_choice = st.selectbox(
        "📄 Load Demo Course", 
        [
            "None", 
            "Rolling / Punchy (Box Hill)", 
            "Hilly Time Trial (Tour de France)",
            "Flat Course (Vuelta a España)", 
            "Mountain Time Trial (Hourquette d'Ancizan)"
        ],
        index=0,
        help="Select a pre-loaded course to test different physics scenarios. If loading a custom file, ensure 'None' is selected."
    )

# Determine file path based on selection
file_path = None  # <--- Key Change: Variable name matches the rest of this file

if demo_choice != "None":
    # Map friendly names to actual filenames
    demo_files = {
        "Rolling / Punchy (Box Hill)": "Demo_Rolling_BoxHill.gpx",
        "Hilly Time Trial (Tour de France)": "Demo_Hilly_Tour.gpx",
        "Flat Course (Vuelta a España)": "Demo_Flat_Vuelta.gpx",
        "Mountain Time Trial (Hourquette d'Ancizan)": "Demo_Mountain_Hourquette_dAncizan.gpx"
    }
    
    selected_file = demo_files.get(demo_choice)
    
    if selected_file and os.path.exists(selected_file):
        file_path = selected_file
        st.info(f"✅ **{demo_choice} Loaded**")
    else:
        st.error(f"❌ Error: Demo file '{selected_file}' not found. Please ensure files are renamed.")

elif uploaded_file:
    # Save uploaded file to a temporary path
    with open("temp_time.gpx", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = "temp_time.gpx"

# --- CALCULATION ---
if file_path:
    try:
        gpx_result = gpx_tool.parse_gpx_file(file_path)
        
        # Plot Elevation
        st.subheader("Course Profile")
        plot_data = gpx_result['plot_data']
        if plot_data['distances_km']:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(plot_data['distances_km'], plot_data['smoothed_elevations_m'], color='black', linewidth=0.8)
            ax.fill_between(plot_data['distances_km'], plot_data['smoothed_elevations_m'], color='gray', alpha=0.2)
            ax.set_ylim(bottom=0)
            ax.axis('off')
            st.pyplot(fig)

            # --- [NEW CODE START] ---
            # Calculate and display metrics immediately after loading
            total_climb = gpx_tool.calculate_total_ascent(gpx_result['trackPoints'])
            
            c1, c2 = st.columns(2)
            c1.metric("Distance", f"{gpx_result['totalDistanceKm']:.2f} km")
            c2.metric("Elevation Gain", f"{total_climb:.0f} m")
            # --- [NEW CODE END] ---

        st.write("### Input")
        col_pwr, col_btn = st.columns([2, 1])
        
        with col_pwr:
            target_power = st.number_input("Target Average Power (Watts)", value=200, step=5, min_value=0)
        
        with col_btn:
            st.write("")
            st.write("")
            run_sim = st.button("🚀 Run Simulation")

        if run_sim:
            # --- INPUT VALIDATION ---
            # Check for zero values that break physics or logic
            if rider_mass <= 0 or bike_mass <= 0:
                st.error("❌ **Physics Error:** Mass cannot be zero.")
                st.warning("Please enter a valid weight for both the Rider and the Bike in the sidebar.")
                st.stop()
            
            if target_power <= 0:
                st.error("❌ **Input Error:** Power must be greater than 0 Watts.")
                st.stop()

            # Prepare Simulation Data
            course_data = {
                'gpx_track_points': gpx_result['trackPoints'],
                'total_course_distance_m': gpx_result['totalDistanceKm'] * 1000
            }
            # Create a constant power profile
            power_profile = [target_power] * len(gpx_result['trackPoints'])
            
            with st.spinner("Simulating ride physics..."):
                # Run Physics Engine
                time_seconds, _, _, _ = physics_engine.simulate_course(
                    rider_mass, bike_mass, cda, 0.96, # 0.96 efficiency default
                    gpx_result['trackPoints'][0]['ele'],
                    wind_speed, wind_deg, 20, # 20C temp default
                    True, 0.8, 0.95, # Cornering enabled
                    [[0, crr]], # Surface profile
                    gpx_result['trackPoints'],
                    gpx_result['totalDistanceKm'] * 1000,
                    power_profile,
                    None, # No segmentation map needed for basic sim
                    False, # No print output
                    rider_ftp=target_power # Assume FTP = Target for simple sim
                )

            # --- ERROR HANDLING FOR STALLS ---
            if time_seconds == float('inf'):
                st.error("🛑 **Simulation Failed: Stall Detected**")
                st.warning(f"""
                **{target_power} Watts is insufficient to complete this course.**
                
                The physics engine detected that on the steepest gradients, this power output was not enough to overcome gravity. 
                The rider's speed dropped to zero.
                
                **Suggestion:** Increase your power target or reduce the total system mass.
                """)
            else:
                # Success
                h, m, s = physics_engine.format_time_hms(time_seconds)
                st.success(f"### 🏁 Predicted Time: {h}h {m}m {s}s")
                
                # --- [UPDATED RESULTS SECTION] ---
                # Calculate metrics
                avg_speed = (gpx_result['totalDistanceKm'] * 1000 / time_seconds) * 3.6
                total_elevation = gpx_tool.calculate_total_ascent(gpx_result['trackPoints'])
                
                # Display in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Speed", f"{avg_speed:.1f} km/h")
                with col2:
                    st.metric("Elevation Gain", f"{int(total_elevation)} m")

    except Exception as e:
        st.error(f"Error: {e}")