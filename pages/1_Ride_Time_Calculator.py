#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APP MODULE: RIDE TIME CALCULATOR
--------------------------------
This is the "Forward" Simulator.
Input:  Constant Power (Watts)
Output: Speed & Time

Unlike the Optimizer (which solves for Power given a Time), this tool 
runs the physics engine in its standard direction to predict performance 
for a steady effort.
"""

import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk  

import gpx_tool
import physics_engine
import analytics

# --- PROJECT PATH SETUP ---
# We need to tell Python where to look for our custom modules (physics_engine, etc.)
# because they are located in the parent directory, not inside the 'pages' folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Inject Google Analytics (Optional tracking)
#analytics.inject_ga()

def inject_ga():
    # Replace G-XXXXXXXXXX with your actual Tracking ID
    GA_ID = "G-XXXXXXXXXX"
    
    # Note: Streamlit runs components in an iframe, so this isn't perfect, 
    # but it works without crashing the server.
    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{GA_ID}');
    </script>
    """
    st.components.v1.html(ga_code, height=0, width=0)

# _____________________________________________________________________________

# --- ### --- 1. HELPER FUNCTIONS --- ### ---

def get_wind_cardinal(deg):
    """
    Converts a wind bearing (0-360) into a text Label (N, NE, E...).
    Useful for UI feedback so the user knows '90 degrees' means 'East Wind'.
    """
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25)/22.5)
    return dirs[ix % 16]
# _____________________________________________________________________________

# --- ### --- 2. STREAMLIT UI CONFIGURATION --- ### ---

st.set_page_config(page_title="Ride Time Calculator", page_icon="🚴")

# --- SIDEBAR: GLOBAL PHYSICS SETTINGS ---
st.sidebar.title("🚴 Simulation Settings")

# A. RIDER BIO-METRICS
# These determine the weight penalty (Gravity) and frontal area scaling (Aerodynamics).
st.sidebar.header("1. Rider Profile")
rider_mass = st.sidebar.number_input("Rider Mass (kg)", value=75.0, step=0.5, min_value=0.0)
rider_height = st.sidebar.number_input("Rider Height (cm)", value=178, step=1, min_value=100)
bike_mass = st.sidebar.number_input("Bike Mass (kg)", value=8.5, step=0.1, min_value=0.0)

# B. AERODYNAMICS (CdA)
# We allow the user to pick a position textually, but we convert it to a number.
st.sidebar.header("2. Aerodynamics")

cda_options = {
    "TT Bike (Aero Bars)": 0.23,
    "Road Bike (Aero Bars)": 0.26,
    "Road Bike (Drops)": 0.29,
    "Road Bike (Hoods)": 0.32,
    "Road Bike (Tops)": 0.38,
    "Gravel Bike (Drops)": 0.33,
    "Gravel Bike (Hoods)": 0.36,
    "Gravel Bike (Tops)": 0.42,
    "Mountain Bike (Tops)": 0.45
}

cda_choice = st.sidebar.selectbox(
    "Riding Position",
    list(cda_options.keys()),
    index=2, 
    help="Select your dominant position for flat roads. The model automatically adjusts your aerodynamics for climbing (more upright) and descending (tucking)."
    )

# 1. Base CdA lookup (Standard 178cm rider)
base_cda = cda_options[cda_choice]

# 2. Dynamic Scaling: Taller riders push more wind.
cda = physics_engine.calculate_height_adjusted_cda(base_cda, rider_height)

# UI Feedback showing the math
if rider_height != 178:
    st.sidebar.caption(f"📏 Scaled for height: **{base_cda}** → **{cda:.3f}**")
else:
    st.sidebar.caption(f"Standard value: **{cda}**")

# C. ROLLING RESISTANCE (Crr)
# Crr = Tyre Rolling Resistance + Road Surface Roughness
st.sidebar.header("3. Rolling Resistance (Crr)")
tyre_options = {
    "Top-Tier Race (e.g. Latex/Tubeless)": 0.0030, "Mid-Range Performance (e.g. GP5000)": 0.0040,
    "Training / Endurance (e.g. Gatorskin)": 0.0060, "Gravel / Cyclocross": 0.0070, "MTB Knobby": 0.0100
}
surface_options = {
    "Smooth New Tarmac": 0.0010, "Average Road (e.g. Worn Tarmac)": 0.0025,
    "Rough Road / Chip Seal": 0.0050, "Light Gravel / Hardpack": 0.0070, "Heavy Gravel / Dirt": 0.0120
}
crr_mode = st.sidebar.radio("Input Mode", ["Auto-Calculate", "Manual Override"], horizontal=True)

if crr_mode == "Auto-Calculate":
    selected_tyre = st.sidebar.selectbox("Tyre Choice", list(tyre_options.keys()), index=1)
    selected_surface = st.sidebar.selectbox("Road Condition", list(surface_options.keys()), index=1)
    
    # Calculate Total Crr
    crr = tyre_options[selected_tyre] + surface_options[selected_surface]
    st.sidebar.info(f"Calc: {tyre_options[selected_tyre]:.4f} (Tyre) + {surface_options[selected_surface]:.4f} (Surface)")
else:
    # Manual Override for power users who know their exact coefficient
    crr = st.sidebar.number_input("Custom Crr", value=0.0050, step=0.0005, format="%.4f", min_value=0.0)

# D. DRIVETRAIN
st.sidebar.header("4. Mechanical Efficiency")
mech_dict = {
    "Optimised (Waxed / Ceramic)": 0.98,
    "Standard (Clean & Lubed)": 0.96,
    "Neglected (Dirty / Old)": 0.94
}
mech_key = st.sidebar.selectbox("Drivetrain Condition", list(mech_dict.keys()), index=1)
mechanical_efficiency = mech_dict[mech_key]

# E. ENVIRONMENT
st.sidebar.header("5. Weather")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 0.0, step=0.5)

# Wind Direction Logic: Dropdown OR Manual Degree Input
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

# _____________________________________________________________________________

# --- ### --- 3. MAIN PAGE LAYOUT --- ### ---
st.title("⏱️ Ride Time Calculator")
st.markdown("""
**Predict your finish time based on a fixed Power Output.**
\nThis tool runs the physics engine forward: *Input Watts $\\rightarrow$ Output Speed*.
""")

# Reminder to check sidebar
st.info("👈 **Configuration:** Please adjust the **Rider, Bike, and Environment settings** in the sidebar to match your specific scenario.")

# --- FILE INPUT SECTION ---
col_upload, col_demo = st.columns([2, 1])

with col_upload: 
        # Option A: Upload a custom GPX file
        uploaded_file = st.file_uploader(
            "Upload GPX Course", 
            type=['gpx'], 
            help="Ensure name of file ends with '.gpx'"
        )
        
        st.info(
            "⚠️ **File Type:** Upload a **.GPX** file only. HTML, FIT, or TCX files will not work."
        )

with col_demo: 
    st.write("")
    st.write("")
    # Option B: Use a pre-loaded demo file
    demo_choice = st.selectbox(
        "📄 Load Demo Course", 
        [
            "None", 
            "Rolling / Punchy (Box Hill)", 
            "Hilly Time Trial (Tour de France)",
            "Flat Course (Vuelta a España)", 
            "Mountain Summit Time Trial (Hourquette d'Ancizan)"
        ],
        index=0,
        help="Select a pre-loaded course to test different physics scenarios. If loading a custom file, ensure 'None' is selected."
    )

# Determine file path based on selection
file_path = None 

if demo_choice != "None":
    # Dictionary mapping friendly names to actual file paths on disk
    demo_files = {
        "Rolling / Punchy (Box Hill)": "Demo_Rolling_BoxHill.gpx",
        "Hilly Time Trial (Tour de France)": "Demo_Hilly_Tour.gpx",
        "Flat Course (Vuelta a España)": "Demo_Flat_Vuelta.gpx",
        "Mountain Summit Time Trial (Hourquette d'Ancizan)": "Demo_Mountain_Hourquette_dAncizan.gpx"
    }
    
    selected_file = demo_files.get(demo_choice)
    
    if selected_file and os.path.exists(selected_file):
        file_path = selected_file
        st.info(f"✅ **{demo_choice} Loaded**")
    else:
        st.error(f"❌ Error: Demo file '{selected_file}' not found. Please ensure files are renamed.")

elif uploaded_file:
    # Save the uploaded file to a temporary location so the parser can read it
    with open("temp_time.gpx", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path = "temp_time.gpx"

# _____________________________________________________________________________

# --- ### --- 4. SIMULATION EXECUTION --- ### ---
if file_path:
    try:
        # 1. PARSE GPX DATA
        gpx_result = gpx_tool.parse_gpx_file(file_path)
        
        # 2. PLOT COURSE MAP
        
        st.subheader("Course Map")

        # A. Prepare Data for PyDeck
        # We need a list of coordinates in the format [[lon, lat], [lon, lat], ...]
        # Note: PyDeck uses [Longitude, Latitude] order!
        path_data = [
            [p['lon'], p['lat']] for p in gpx_result['trackPoints']
        ]
        
        # B. Define the 'Layer' (The Red Line)
        layer = pdk.Layer(
            "PathLayer",
            data=[{"path": path_data}],  # Wrap in a list of dicts
            get_path="path",
            get_color=[255, 0, 0],       # Red Color [R, G, B]
            width_scale=20,              # Adjusts line thickness
            width_min_pixels=2,          # Minimum width on screen
            pickable=True,
            auto_highlight=True
        )

        # C. Define the 'View' (Camera Position)
        # Calculate the center of the map automatically
        avg_lat = sum(p['lat'] for p in gpx_result['trackPoints']) / len(gpx_result['trackPoints'])
        avg_lon = sum(p['lon'] for p in gpx_result['trackPoints']) / len(gpx_result['trackPoints'])

        view_state = pdk.ViewState(
            latitude=avg_lat,
            longitude=avg_lon,
            zoom=12,
            pitch=0,  # Set to 0 for "Bird's Eye" (Top-Down), or 45 for 3D
        )

        # D. Render the Map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Course Path"}
        ))
        
        # 3. COURSE PROFILE
        st.subheader("Course Profile")
        plot_data = gpx_result['plot_data']
        if plot_data['distances_km']:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(plot_data['distances_km'], plot_data['smoothed_elevations_m'], color='black', linewidth=0.8)
            ax.fill_between(plot_data['distances_km'], plot_data['smoothed_elevations_m'], color='gray', alpha=0.2)
            ax.set_ylim(bottom=0)
            ax.axis('off')
            st.pyplot(fig)

            # Calculate and display metrics immediately after loading
            total_climb = gpx_tool.calculate_total_ascent(gpx_result['trackPoints'])
            
            c1, c2 = st.columns(2)
            c1.metric("Distance", f"{gpx_result['totalDistanceKm']:.2f} km")
            c2.metric("Elevation Gain", f"{total_climb:.0f} m")
            

        # 4. USER INPUT: POWER TARGET
        st.write("### Input")
        col_pwr, col_btn = st.columns([2, 1])
        
        # Educational expander for beginners
        with st.expander("ℹ️ Not sure what Power to enter?", expanded=False):
            st.markdown("""
            **1. Estimate based on fitness (for a 75kg rider):**
            * **Beginner / Casual:** 100 W – 150 W
            * **Keen Amateur / Club Rider:** 180 W – 230 W
            * **Competitive Racer:** 250 W – 300 W+
            
            **2. Base it on a previous ride:**
            Have you completed this course (or a similar one) before?
            1. Switch to the **⚡ Power Plan Estimator** tool.
            2. Enter your previous finish time.
            3. The tool will calculate the average power you held during that ride. Use that number here as your baseline!
            """)
            
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

            # --- PREPARE SIMULATION DATA ---
            course_data = {
                'gpx_track_points': gpx_result['trackPoints'],
                'total_course_distance_m': gpx_result['totalDistanceKm'] * 1000
            }
            
            # Apply surface resistance to the track
            gpx_tool.enrich_track_with_surface_profile(course_data['gpx_track_points'], [[0, crr]])
            
            # Create a constant power profile
            power_profile = [target_power] * len(gpx_result['trackPoints'])
            
            with st.spinner("Simulation in progress. This may take a few minutes depending on course complexity."):
                # --- RUN PHYSICS ENGINE ---
                # We use the standard 'simulate_course' function but with a constant power input.
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

            # --- RESULT HANDLING ---
            if time_seconds == float('inf'):
                # The rider failed to complete the course (Physics Stall)
                st.error("🛑 **Simulation Failed: Stall Detected**")
                st.warning(f"""
                **{target_power} Watts is insufficient to complete this course.**
                
                The physics engine detected that on the steepest gradients, this power output was not enough to overcome gravity. 
                The rider's speed dropped to zero.
                
                **Suggestion:** Increase your power target or reduce the total system mass.
                """)
            else:
                # --- SUCCESS: DISPLAY METRICS ---
                h, m, s = physics_engine.format_time_hms(time_seconds)
                st.success(f"### 🏁 Predicted Time: {h}h {m}m {s}s")
                
                # Calculate derived metrics
                avg_speed = (gpx_result['totalDistanceKm'] * 1000 / time_seconds) * 3.6
                total_elevation = gpx_tool.calculate_total_ascent(gpx_result['trackPoints'])
                
                # Metrics Columns
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Speed", f"{avg_speed:.1f} km/h")
                with col2:
                    st.metric("Elevation Gain", f"{int(total_elevation)} m")
                
                # Realism Warning
                st.warning("""
                **⚠️ A Note on Real-World Pacing:**
                This calculation assumes a **perfectly steady mechanical effort**.
                
                **In the real world:**
                * No rider outputs perfectly constant watts.
                * Hills, corners, and fatigue cause power to fluctuate.
                * **Result:** To match this predicted time, your **Normalized Power (NP)** on the road will likely need to be **5-10% higher** than the Average Power shown here.
                """)
                
                # _____________________________________________________________________________
                
                # --- ### --- 5. NUTRITION CALCULATOR --- ### ---
                st.markdown("---")
                st.subheader("🍎 Nutrition Planner")
                
                # 1. Calculate Total Work (Watts * Seconds = Joules)
                total_work_joules = target_power * time_seconds
                
                # 2. Estimate Caloric & Carb needs
                # NOTE: Since this simple tool doesn't ask for FTP, we assume the ride is done at "Threshold" intensity (Power = FTP) to give a safe, high-end recommendation.
                metabolic_kcal, carbs_burned, intake_rec, total_intake = physics_engine.calculate_nutrition(
                    total_work_joules, time_seconds, target_power # Using Target as FTP for this simple tool
                )
                
                # 3. Display Nutrition Metrics
                n1, n2, n3 = st.columns(3)
                n1.metric("Total Energy Burned", f"{int(metabolic_kcal)} kcal")
                n2.metric("Est. Carbs Burned", f"{int(carbs_burned)} g")
                n3.metric("Required Carb Intake During Ride", f"{int(total_intake)} g Total")
                
                # 4. Educational Strategy Logic (Duration-based advice)
                duration_hours = time_seconds / 3600.0
                
                # CASE A: Short Rides (< 1h 15m)
                if duration_hours < 1.25:
                    st.info(
                        f"💧 **Strategy:** For this duration, aim for **{intake_rec}**.\n\n"
                        "**Why 0g?** Your body stores **300g–500g** of glycogen (fuel). "
                        f"This ride burns roughly **{int(carbs_burned)}g**, so you have plenty of reserves.\n\n"
                        "**The Deficit:** You are burning more than you eat, but for this short duration, "
                        "your liver and muscle stores can easily cover the gap without impacting performance."
                    )
                
                # CASE B: Medium Rides (1h 15m - 2h 30m)
                elif duration_hours < 2.5:
                    st.info(
                        f"⛽️ **Strategy:** For this duration, aim for **{intake_rec}**.\n\n"
                        "**Timing:** Start consuming simple carbs (gels/drink mix) **between 45-60 minutes**.\n"
                        "*Why?* Simple sugars absorb quickly (~15-20 mins). Taking them at the hour mark ensures the fuel hits your bloodstream *before* your stored energy drops around the 90-minute mark.\n\n"
                        "**Why is Intake < Burned?** You will notice your 'Required Intake' is lower than 'Carbs Burned'. This is intentional! "
                        "Your gut can only process ~60-90g/hr, but you might burn 150g+/hr. The difference is covered by the energy already stored in your muscles."
                    )
                    
                # CASE C: Long Rides (2h 30m - 4h)
                elif duration_hours < 4.0:
                    st.info(
                        f"⛽️ **Strategy:** For this duration, aim for **{intake_rec}**.\n\n"
                        "**Timing:** Start early, **within the first 20-30 minutes**.\n"
                        "*Why?* In long events, you cannot play catch-up. If you wait until you are hungry, your digestion will be too slow to reverse the depletion.\n\n"
                        "**Managing the Deficit:** Since you burn fuel 2-3x faster than you can digest it, your goal is to trickle fuel in constantly. "
                        "This spares your limited glycogen stores, extending the time before you hit 'empty'."
                    )
                
                # CASE D: Ultra Rides (> 4h)
                else:
                    st.info(
                        f"⛽️ **Strategy:** For this duration, aim for **{intake_rec}**.\n\n"
                        "**Why 90g+?** This is near the physiological limit of the human gut. "
                        "To absorb this much without sickness, you typically need **'Multiple Transportable Carbohydrates'** "
                        "(a mix of Glucose and Fructose) to use different absorption pathways in the stomach.\n\n"
                        "**The Ultra Reality:** The energy deficit here is massive. Your strategy is simply to jam as much fuel into the system "
                        "as your stomach can tolerate to keep the fire burning."
                    )
        
                    
                st.caption(
                    "⚠️ **Note on Accuracy:** These values are estimates based on standard metabolic efficiency (~24%). "
                    "**Crucially, this simple calculator assumes a high-intensity effort (Threshold).** "
                    "For precise nutrition planning that accounts for your specific fitness zones (e.g. burning more fat at lower intensities), "
                    "please use the **Race Strategy Optimiser**, which includes a full FTP-based physiological model."
                )
                
    except Exception as e:
        st.error(f"Error: {e}")
        
        
        
        
