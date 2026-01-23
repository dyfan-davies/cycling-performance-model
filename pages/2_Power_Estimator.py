#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APP MODULE: POWER TARGET ESTIMATOR
----------------------------------
This is the "Reverse" Simulator.
Input:  Target Time (e.g., "Finish in 4 hours")
Output: Required Power (Watts)

Logic:
It uses a Bisection Search algorithm (Binary Search) to iteratively run 
the physics engine, adjusting the power guess up or down until the 
simulated time matches the user's target time.
"""

import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk  

import gpx_tool
import physics_engine
from run_app import calculate_power_from_target_time
import analytics

# --- PROJECT PATH SETUP ---
# We need to tell Python where to look for our custom modules (physics_engine, etc.)
# because they are located in the parent directory, not inside the 'pages' folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Inject Google Analytics (Optional)
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
    Converts numeric degrees (0-360) to compass direction (N, NE, E...).
    Used for UI feedback.
    """
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25)/22.5)
    return dirs[ix % 16]

# _____________________________________________________________________________

# --- ### --- 2. STREAMLIT UI CONFIGURATION --- ### ---
st.set_page_config(page_title="Power Target Estimator", page_icon="🚴")

# --- SIDEBAR: ADVANCED PHYSICS SETTINGS ---
st.sidebar.title("🚴 Simulation Settings")

# A. RIDER PROFILE
st.sidebar.header("1. Rider Profile")
rider_mass = st.sidebar.number_input("Rider Mass (kg)", value=75.0, step=0.5, min_value=0.0)
rider_height = st.sidebar.number_input("Rider Height (cm)", value=178, step=1, min_value=100)
bike_mass = st.sidebar.number_input("Bike Mass (kg)", value=8.5, step=0.1, min_value=0.0)

# W' (Anaerobic Capacity) Input
# This is crucial for the Estimator. If the target time is very short (e.g. 1 min),
# the rider can output HUGE power because they are burning W'. 
# If the time is long (e.g. 4 hours), they are limited to Critical Power (CP).
w_prime_kj = st.sidebar.number_input(
    "Anaerobic Capacity (W' in kJ)",
    value=15.0,
    step=0.5,
    min_value=1.0,
    help=(
        "The finite battery of energy available above Critical Power (approx. FTP).\n\n"
        "Typical Values:\n"
        "• Untrained / Novice: 8–12 kJ\n"
        "• Trained Amateur: 12–18 kJ\n"
        "• Elite / Pro (All-Rounder): 18–25 kJ\n"
        "• World Class Sprinter/Puncheur: 25–35 kJ+"
    )
)
w_prime_j = w_prime_kj * 1000.0

# B. AERODYNAMICS
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

# Get the 'Base' CdA (for a standard 178cm rider)
base_cda = cda_options[cda_choice]

# Apply Height Adjustment
cda = physics_engine.calculate_height_adjusted_cda(base_cda, rider_height)

# Feedback to User (Optional but recommended)
if rider_height != 178:
    st.sidebar.caption(f"📏 Scaled for height: **{base_cda}** → **{cda:.3f}**")
else:
    st.sidebar.caption(f"Standard value: **{cda}**")

# C. ROLLING RESISTANCE
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
    
    crr = tyre_options[selected_tyre] + surface_options[selected_surface]
    st.sidebar.info(f"Calc: {tyre_options[selected_tyre]:.4f} (Tyre) + {surface_options[selected_surface]:.4f} (Surface)")
else:
    crr = st.sidebar.number_input("Custom Crr", value=0.0050, step=0.0005, format="%.4f", min_value=0.0)

# D. DRIVETRAIN EFFICIENCY
st.sidebar.header("4. Mechanical Efficiency")
mech_dict = {
    "Optimised (Waxed / Ceramic)": 0.98,
    "Standard (Clean & Lubed)": 0.96,
    "Neglected (Dirty / Old)": 0.94
}
mech_key = st.sidebar.selectbox("Drivetrain Condition", list(mech_dict.keys()), index=1)
mechanical_efficiency = mech_dict[mech_key]

# 5. ENVIRONMENT
st.sidebar.header("5. Weather")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 0.0, step=0.5)

wind_cardinal = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
wind_choice = st.sidebar.selectbox("Wind Origin", list(wind_cardinal.keys()) + ["Manual Degrees"], index=0)

if wind_choice == "Manual Degrees":
    wind_deg = st.sidebar.number_input("Degrees", 0, 360, 0)
else:
    wind_deg = wind_cardinal[wind_choice]

st.sidebar.caption(f"Wind from {wind_deg}° ({get_wind_cardinal(wind_deg)})")

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

st.title("⚡ Power Target Estimator")
st.markdown("""
**Calculate the power needed to achieve a specific Time Goal.**

This tool runs a physics simulation backwards to find the average watts required to finish in your target time.
""")

# --- SETTINGS REMINDER ---
st.info("👈 **Configuration:** Please adjust the **Rider, Bike, and Environment settings** in the sidebar to match your specific scenario.")

# Educational Expandable
with st.expander("ℹ️ How the Algorithm Works (Bisection Search)"):
    st.markdown("""
    When you provide an actual ride time, the calculator works backwards using an iterative **"Bisection Search"** method:
    
    1.  **Initial Guess:** It estimates a starting power output (e.g. 200W).
    2.  **Full Simulation:** It runs the physics engine for the entire course with that power.
    3.  **Compare:** It checks the simulated time against your target time.
    4.  **Refine:** If the simulation was *too slow*, it increases the power guess. If *too fast*, it decreases the power guess.
    5.  **Converge:** It repeats this process until the result matches your target time.
    """)

# --- FILE INPUT SECTION ---
col_upload, col_demo = st.columns([2, 1])

with col_upload: 
        # Option A: Upload Custom GPX
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
    # Option B: Use Demo File
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

# Determine file path
file_path_to_process = None

if demo_choice != "None":
    # Map friendly names to actual filenames
    demo_files = {
        "Rolling / Punchy (Box Hill)": "Demo_Rolling_BoxHill.gpx",
        "Hilly Time Trial (Tour de France)": "Demo_Hilly_Tour.gpx",
        "Flat Course (Vuelta a España)": "Demo_Flat_Vuelta.gpx",
        "Mountain Summit Time Trial (Hourquette d'Ancizan)": "Demo_Mountain_Hourquette_dAncizan.gpx"
    }
    
    selected_file = demo_files.get(demo_choice)
    
    if selected_file and os.path.exists(selected_file):
        file_path_to_process = selected_file
        st.info(f"✅ **{demo_choice} Loaded**")
    else:
        st.error(f"❌ Error: Demo file '{selected_file}' not found. Please ensure files are renamed correctly.")

elif uploaded_file:
    # Save uploaded file to a temporary path
    with open("temp_power.gpx", "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path_to_process = "temp_power.gpx"

# _____________________________________________________________________________

# --- ### --- 4. CALCULATION LOGIC --- ### ---
if file_path_to_process:
    try:
        # 1. PARSE GPX
        gpx_result = gpx_tool.parse_gpx_file(file_path_to_process)
        
        # --- DEMO LIMIT GATEKEEPER (COMPLEXITY SCORE) ---
        # Checks if file is too complex (too long or too steep) for the free web server.
        total_climb_m = gpx_tool.calculate_total_ascent(gpx_result['trackPoints'])
        total_dist_km = gpx_result['totalDistanceKm']
        
        # Heuristic: Score = Distance + (Elevation / 20)
        # Limit set to 90 to block complex files like Alpe d'Huez (Score ~103)
        # which take >9 mins to process.
        COMPLEXITY_LIMIT = 150.0
        
        complexity_score = total_dist_km + (total_climb_m / 20.0) + 10.0
        
        if complexity_score > COMPLEXITY_LIMIT:
            st.error(f"⛔ **Route Too Complex for Web Demo (Score: {complexity_score:.0f}/{int(COMPLEXITY_LIMIT)})**")
            
            st.warning(
                f"""
                To ensure the web app remains responsive, simulation load is limited by a **Complexity Score**.
                
                **Why was this file rejected?**
                * **Distance:** {total_dist_km:.1f} km
                * **Elevation:** {int(total_climb_m)} m
                
                Mountainous routes take significantly longer to process than flat ones due to slower climbing speeds.
                
                * **Flat Routes:** Accepted up to ~85km
                * **Mountain Routes:** Accepted up to ~30km
                """
            )
            
            st.info(
                "💡 **Tip:** Long-distance or high-mountain analysis is possible on dedicated offline hardware. "
                "Get in touch for a custom report, contact details are at the bottom of the ***About the Engineer*** section."
            )
            
            st.stop() 
        # -----------------------------------
        
        # 2. COURSE VISUALIZATION
        
        # Course Map
        
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
        
        # Course Profile
        st.subheader("Course Profile")
        plot_data = gpx_result.get('plot_data', {})
        dist_km = plot_data.get('distances_km', [])
        elev_m = plot_data.get('smoothed_elevations_m', [])
        
        if dist_km and elev_m:
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.plot(dist_km, elev_m, color='#1f77b4', linewidth=1)
            ax.fill_between(dist_km, elev_m, color='#1f77b4', alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.axis('off')
            st.pyplot(fig)
        
        # DISPLAY COURSE METRICS ON LOAD
        total_dist_km = gpx_result['totalDistanceKm']
        total_ascent_m = gpx_tool.calculate_total_ascent(gpx_result['trackPoints'])
        
        # Metrics Display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Distance", f"{total_dist_km:.2f} km")
        with col2:
            st.metric("Total Elevation", f"{int(total_ascent_m)} m")

        st.markdown("---")
        
        # 3. USER INPUT: TARGET TIME
        st.write("### Input")
        target_str = st.text_input("Target Finish Time (HH:MM:SS)", "00:30:00", help="e.g. 04:00:00 for a 4 hour century.")

        # 4. EXECUTION
        st.info("ℹ️ **Processing Note:** This tool must run the physics engine multiple times (iteratively) to find the exact match. This may take a few minutes for the calculation to complete.")
        
        if st.button("Calculate Required Watts"):
            
            # A. Parse Time String -> Seconds
            try:
                parts = list(map(int, target_str.split(':')))
                total_seconds = parts[0]*3600 + parts[1]*60 + parts[2]
            except:
                st.error("Invalid time format. Use HH:MM:SS")
                st.stop()

            # B. Validate Physics Inputs
            if rider_mass <= 0 or bike_mass <= 0:
                st.error("❌ **Physics Error:** Mass cannot be zero.")
                st.stop()

            if cda <= 0:
                st.error("❌ **Physics Error:** Aerodynamic Drag (CdA) cannot be zero.")
                st.stop()

            if crr <= 0 and crr_mode == "Manual Override":
                st.error("❌ **Physics Error:** Rolling Resistance (Crr) cannot be zero.")
                st.stop()
            
            # C. Pack Parameters
            course_data = {
                'gpx_track_points': gpx_result['trackPoints'],
                'total_course_distance_m': gpx_result['totalDistanceKm'] * 1000
            }
            
            rider_params = {
                'rider_mass': rider_mass, 
                'bike_mass': bike_mass, 
                'cda': cda, 
                'mechanical_efficiency': mechanical_efficiency,
                'w_prime_capacity_j': w_prime_j
            }
            
            sim_params = {
                'global_wind_speed_kmh': wind_speed, 'global_wind_direction_from_deg': wind_deg,
                'ambient_temperature_celsius': 20, 'surface_profile': [[0, crr]],
                'enable_cornering_model': True, 'friction_coefficient_mu': 0.8,
                'rider_skill_factor': 0.95, 'starting_elevation_m': gpx_result['trackPoints'][0]['ele']
            }

            gpx_tool.enrich_track_with_surface_profile(course_data['gpx_track_points'], sim_params['surface_profile'])
            
            with st.spinner("Simulation in progress. This may take a few minutes depending on course complexity."):
                # D. RUN THE BISECTION SEARCH
                # This function (imported from run_app) calls simulate_course repeatedly
                # until it finds the Watts that result in 'total_seconds'.
                required_watts = calculate_power_from_target_time(total_seconds, rider_params, sim_params, course_data)
                
                # E. VERIFICATION RUN
                # Run one final sim with the found watts to double-check the time and check for physics stalls
                verify_profile = [required_watts] * len(gpx_result['trackPoints'])
                
                verified_time, _, _, _ = physics_engine.simulate_course(
                    rider_mass, bike_mass, cda, mechanical_efficiency,
                    gpx_result['trackPoints'][0]['ele'],
                    wind_speed, wind_deg, 20,
                    True, 0.8, 0.95,
                    [[0, crr]],
                    gpx_result['trackPoints'],
                    gpx_result['totalDistanceKm'] * 1000,
                    verify_profile,
                    None, False, rider_ftp=required_watts,
                    rider_w_prime=w_prime_j
                )

            # --- RESULT HANDLING ---
            
            # Case 1: Physics Stall (e.g. 50W on a 20% gradient)
            if verified_time == float('inf'):
                st.error("⚠️ **Target Time Unreachable (Physics Constraint)**")
                st.warning(f"""
                To finish in **{target_str}**, you would need to ride at roughly **{required_watts:.1f} Watts**.
                However, this power output is **too low** to climb the steepest gradients on this course. 
                
                **The Physics:** At {required_watts:.1f}W, gravity is stronger than your legs on the climbs, causing you to stop.
                """)
            
            # Case 2: User asked for a time that is mathematically impossible (slower than walking speed)
            elif (total_seconds - verified_time) > (total_seconds * 0.1): # >10% Discrepancy
                h_v, m_v, s_v = physics_engine.format_time_hms(verified_time)
                st.warning(f"⚠️ **Target Time Unreachable (Too Slow)**")
                st.info(f"""
                You requested a time of **{target_str}**.
                However, the **minimum power** required to physically climb the hills on this course without stopping is **{required_watts:.1f} Watts**.
                
                Riding at this minimum power results in a time of **{h_v}h {m_v}m {s_v}s**, which is faster than your target.
                *You cannot ride slower than this without walking!*
                """)
                
            else:
                # Case 3 Success
                # Format time nicely using the physics_engine helper
                th, tm, ts = physics_engine.format_time_hms(total_seconds)
                
                # Logic: Only show hours if > 0
                if th > 0:
                    time_human = f"{th} hour{'s' if th!=1 else ''}, {tm} minute{'s' if tm!=1 else ''} and {ts} second{'s' if ts!=1 else ''}"
                else:
                    time_human = f"{tm} minute{'s' if tm!=1 else ''} and {ts} second{'s' if ts!=1 else ''}"
                    
                # Main Metric
                st.success(f"### Required Constant Power: {required_watts:.1f} Watts")
                st.info(f"To finish **{gpx_result['name']}** in **{time_human}**, you would need to cycle at a constant power of **{required_watts:.1f} Watts**.")
                
                # Explanation of Logic
                with st.expander("🧠 Why is this power value chosen?"):
                    st.markdown(f"""
                    This calculation uses the **Critical Power Model**.
                    
                    * **If the result is > FTP:** The ride is short enough that you can "burn" your anaerobic battery ($W'$) to sustain a higher intensity.
                    * **If the result is < FTP:** The ride is long, so you must ride below Threshold to preserve energy.
                    
                    **Your Stats:**
                    * Input Time: {time_human}
                    * Anaerobic Contribution: ~{w_prime_j / total_seconds:.1f} Watts (derived from your {w_prime_kj}kJ $W'$)
                    """)
                
                # W/kg Context
                total_mass = rider_params['rider_mass'] + rider_params['bike_mass']
                st.caption(f"This requires **{required_watts / total_mass:.2f} W/kg**.")
                
                st.warning("""
                **⚠️ Important Note on Pacing:**
                This calculation assumes you pedal at a **perfectly constant power**. 
                In reality, you would likely need a slightly higher Normalized Power (NP) to account for hills and fatigue.
                """)
                
                # _____________________________________________________________________________
                
                # --- ### --- 5. NUTRITION PLANNER --- ### ---
                st.markdown("---")
                st.subheader("🍎 Nutrition Planner")
                
                # 1. Calculate Total Work (Watts * Seconds = Joules)
                # We use the FINAL RESULT (required_watts) calculated above
                total_work_joules = required_watts * total_seconds
                
                # 2. Nutrion Math: Call the engine helper
                # NOTE: Since we don't know the rider's true FTP in this tool, we pass 'required_watts' 
                # as the FTP. This assumes a hard effort (Intensity Factor = 1.0), which provides 
                # a safe, high-carb recommendation.
                est_kcal, est_carbs, est_rec, est_intake = physics_engine.calculate_nutrition(
                    total_work_joules, total_seconds, required_watts
                )
                
                # 3. Display Metrics
                nc1, nc2, nc3 = st.columns(3)
                nc1.metric("Total Energy Burned", f"{int(est_kcal)} kcal")
                nc2.metric("Est. Carbs Burned", f"{int(est_carbs)} g")
                nc3.metric("Required Intake During Ride", f"{int(est_intake)} g")
                
                # 4. Educational Strategy Logic
                duration_hours = total_seconds / 3600.0
                
                # CASE 1: Short Rides (< 1h 15m)
                if duration_hours < 1.25:
                    st.info(
                        f"💧 **Strategy:** For this duration, aim for **{est_rec}**.\n\n"
                        "**Why 0g?** Your body stores **300g–500g** of glycogen (fuel). "
                        f"This ride burns roughly **{int(est_carbs)}g**, so you have plenty of reserves.\n\n"
                        "**The Deficit:** You are burning more than you eat, but for this short duration, "
                        "your liver and muscle stores can easily cover the gap without impacting performance."
                    )
                
                # CASE 2: Medium Rides (1h 15m - 2h 30m)
                elif duration_hours < 2.5:
                    st.info(
                        f"⛽️ **Strategy:** For this duration, aim for **{est_rec}**.\n\n"
                        "**Timing:** Start consuming simple carbs (gels/drink mix) **between 45-60 minutes**.\n"
                        "*Why?* Simple sugars absorb quickly (~15-20 mins). Taking them at the hour mark ensures the fuel hits your bloodstream *before* your stored energy drops around the 90-minute mark.\n\n"
                        "**Why is Intake < Burned?** You will notice your 'Required Intake' is lower than 'Carbs Burned'. This is intentional! "
                        "Your gut can only process ~60-90g/hr, but you might burn 150g+/hr. The difference is covered by the energy already stored in your muscles."
                    )
                    
                # CASE 3: Long Rides (2h 30m - 4h)
                elif duration_hours < 4.0:
                    st.info(
                        f"⛽️ **Strategy:** For this duration, aim for **{est_rec}**.\n\n"
                        "**Timing:** Start early, **within the first 20-30 minutes**.\n"
                        "*Why?* In long events, you cannot play catch-up. If you wait until you are hungry, your digestion will be too slow to reverse the depletion.\n\n"
                        "**Managing the Deficit:** Since you burn fuel 2-3x faster than you can digest it, your goal is to trickle fuel in constantly. "
                        "This spares your limited glycogen stores, extending the time before you hit 'empty'."
                    )
                
                # CASE 4: Ultra Rides (> 4h)
                else:
                    st.info(
                        f"⛽️ **Strategy:** For this duration, aim for **{est_rec}**.\n\n"
                        "**Why 90g+?** This is near the physiological limit of the human gut. "
                        "To absorb this much without sickness, you typically need **'Multiple Transportable Carbohydrates'** "
                        "(a mix of Glucose and Fructose) to use different absorption pathways in the stomach.\n\n"
                        "**The Ultra Reality:** The energy deficit here is massive. Your strategy is simply to jam as much fuel into the system "
                        "as your stomach can tolerate to keep the fire burning."
                    )

    except Exception as e:
        st.error(f"Error parsing GPX: {e}")
