import streamlit as st
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gpx_tool
import physics_engine
import optimizer
from run_app import run_selected_optimization

# --- HELPER FUNCTION: Wind Direction ---
def get_wind_cardinal(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((deg + 11.25)/22.5)
    return dirs[ix % 16]

# --- HELPER FUNCTION: Comparative Analysis (Power & Speed) ---
def plot_comparative_analysis(log_baseline: List[Dict], log_optimized: List[Dict], course_name: str):
    if not log_baseline or not log_optimized:
        st.warning("Warning: Insufficient data for comparative plot.")
        return None, None

    # Extract Data
    dist_base = [e['distance'] for e in log_baseline]
    power_base = [e['power'] for e in log_baseline]
    speed_base = [e['velocity'] for e in log_baseline]
    
    dist_opt = [e['distance'] for e in log_optimized]
    power_opt = [e['power'] for e in log_optimized]
    speed_opt = [e['velocity'] for e in log_optimized]
    
    elevations = [e['elevation'] for e in log_baseline]
    max_dist = max(dist_base)
    max_ele = max(elevations) if elevations else 100

    # --- FIGURE 1: POWER STRATEGY COMPARISON ---
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1_twin = ax1.twinx()
    y1_limit_min = min(elevations)-(min(elevations)*0.1)
    y1_limit_max = max_ele * 1.05
    ax1_twin.set_ylim(y1_limit_min,y1_limit_max)
    
    ax1_twin.fill_between(dist_base, elevations, 0, color='tab:green', alpha=0.15, label='Elevation')
    ax1_twin.set_ylabel('Elevation (m)', color='tab:green', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='tab:green')
    ax1_twin.grid(False) 

    ax1.plot(dist_base, power_base, label='Constant Power (Baseline)', color='tab:red', linewidth=2, alpha=0.7)
    ax1.plot(dist_opt, power_opt, label='Optimised Strategy', color='tab:blue', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('Power (Watts)', color='black', fontsize=12)
    ax1.set_xlim(0, max_dist)
    
    y_max_pwr = max(power_opt) * 1.5
    ax1.set_ylim(0, y_max_pwr)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', 
               bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", borderaxespad=0, ncol=3, frameon=False)

    plt.tight_layout()

    # --- FIGURE 2: SPEED COMPARISON ---
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    ax2.plot(dist_base, speed_base, label='Constant Power Speed', color='tab:red', linewidth=2, alpha=0.6)
    ax2.plot(dist_opt, speed_opt, label='Optimised Speed', color='tab:blue', linewidth=2, linestyle='--')

    ax2.set_xlabel('Distance (km)', fontsize=12)
    ax2.set_ylabel('Speed (km/h)', fontsize=12)
    ax2.set_xlim(0, max_dist)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, which='major', linestyle='--', alpha=0.5)

    ax2.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", borderaxespad=0, ncol=2, frameon=False)

    plt.tight_layout()

    return fig1, fig2

# --- HELPER FUNCTION: Pacing Guide Plotter ---
def plot_pacing_strategy_guide(course_name, gpx_track_points, optimized_power_profile, rider_ftp, macro_metrics):
    if not gpx_track_points or not optimized_power_profile:
        st.warning("Warning: Missing data for Pacing Guide.")
        return None

    distances_km = [p['cumulativeDistanceM'] / 1000.0 for p in gpx_track_points]
    elevations = [p['smoothedElevation'] if 'smoothedElevation' in p else p['ele'] for p in gpx_track_points]
    max_ele = max(elevations) if elevations else 100
    
    target_power_curve = []
    gpx_to_opt_segment_map = [] 
    
    current_seg_idx = 0
    for dist in distances_km:
        if current_seg_idx < len(macro_metrics) - 1:
            if dist >= macro_metrics[current_seg_idx]['segment_end_km']:
                current_seg_idx += 1
        gpx_to_opt_segment_map.append(current_seg_idx)
        watts = optimized_power_profile[min(current_seg_idx, len(optimized_power_profile)-1)]
        target_power_curve.append(watts)

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.set_title(f'Rider Pacing Strategy Guide: {course_name}', fontsize=16, fontweight='bold')

    color_elev = 'tab:gray'
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', color=color_elev, fontsize=12)
    y1_limit_min = min(elevations)-(min(elevations)*0.1)
    ax1.set_ylim(y1_limit_min, max_ele * 1.05) 
    
    ax1.fill_between(distances_km, elevations, 0, color=color_elev, alpha=0.2, label='Elevation')
    ax1.plot(distances_km, elevations, color=color_elev, linewidth=1.0, alpha=0.5)
    
    ax1.tick_params(axis='y', labelcolor=color_elev)
    ax1.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)
    ax1.set_xlim(distances_km[0], distances_km[-1])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Target Power (Watts)', fontsize=12, fontweight='bold')
    ax2.plot(distances_km, target_power_curve, color='black', linewidth=2.5, label='Target Power')

    def get_zone_color(watts):
        pct = watts / rider_ftp
        if pct < 0.75: return '#3498db'
        elif pct < 1.05: return '#f1c40f'
        else: return '#e74c3c'

    segment_boundaries = [0]
    for i in range(1, len(gpx_track_points)):
        if gpx_to_opt_segment_map[i] != gpx_to_opt_segment_map[i-1]:
            segment_boundaries.append(i)
    segment_boundaries.append(len(gpx_track_points)-1)

    for i in range(len(segment_boundaries)-1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i+1]
        
        seg_id = gpx_to_opt_segment_map[start_idx]
        seg_id = min(seg_id, len(optimized_power_profile)-1)
        watts = optimized_power_profile[seg_id]
        
        grad_str = ""
        if seg_id < len(macro_metrics):
             grad_val = macro_metrics[seg_id]['avg_gradient']
             grad_str = f"\n({grad_val:+.1f}%)"

        start_dist = distances_km[start_idx]
        end_dist = distances_km[end_idx]
        color = get_zone_color(watts)
        
        ax2.fill_between([start_dist, end_dist], 0, [watts, watts], color=color, alpha=0.4)
        
        if (end_dist - start_dist) > 0.3: 
            mid_dist = (start_dist + end_dist) / 2
            label_text = f"{int(watts)}W{grad_str}"
            ax2.text(mid_dist, watts + 10, label_text, ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

    max_power = max(optimized_power_profile) if optimized_power_profile else 300
    ax2.set_ylim(0, max_power * 1.5) 
    
    ftp_label = f"FTP ({int(rider_ftp)}W)"
    ax2.axhline(y=rider_ftp, color='gray', linestyle='--', linewidth=1, alpha=0.8, label=ftp_label)
    ax2.text(distances_km[0], rider_ftp + 5, ftp_label, color='gray', fontsize=9, ha='left', va='bottom')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    custom_lines = [Line2D([0], [0], color='#3498db', lw=4, alpha=0.6),
                    Line2D([0], [0], color='#f1c40f', lw=4, alpha=0.6),
                    Line2D([0], [0], color='#e74c3c', lw=4, alpha=0.6)]
    
    ax1.legend(lines_1 + lines_2 + custom_lines, 
               ['Elevation', 'Target Power', ftp_label, 'Easy (<75%)', 'Sustainable (75-105%)', 'Attack (>105%)'], 
               loc='upper left', bbox_to_anchor=(1.08, 1), borderaxespad=0.)

    plt.tight_layout()
    return fig

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Race Strategy Optimiser", page_icon="📈", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🚴 Simulation Settings")

st.sidebar.header("1. Rider Profile")
rider_mass = st.sidebar.number_input("Rider Mass (kg)", value=75.0, step=0.5, min_value=0.0)
bike_mass = st.sidebar.number_input("Bike Mass (kg)", value=8.5, step=0.1, min_value=0.0)
rider_ftp = st.sidebar.number_input("Rider FTP (Watts)", value=250, step=5, min_value=0)
w_prime_kj = st.sidebar.number_input(
    "Anaerobic Capacity (W' in kJ)",
    value=14.0,
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
st.sidebar.caption(
    "ℹ️ **Realism Constraint:** This value limits how long the rider can surge above Threshold. "
    "It prevents the model from choosing unrealistic strategies (e.g., sprinting for 10 minutes) "
    "by forcing the rider to recover if this battery is depleted."
)

st.sidebar.header("2. Aerodynamics (CdA)")
cda_presets = {
    "TT Bike (Aero Tuck)": 0.23, "Road Bike (Drops)": 0.29, 
    "Road Bike (Hoods)": 0.32, "Road Bike (Relaxed/Tops)": 0.38,
    "Gravel Bike (Hoods/Drops)": 0.35, "Mountain Bike (Flat Bars)": 0.45,
    "Enter Manually...": -1
}
cda_choice = st.sidebar.selectbox("Base Riding Position", list(cda_presets.keys()), index=2)
if cda_choice == "Enter Manually...":
    cda = st.sidebar.slider("Custom CdA", 0.0, 0.600, 0.320, step=0.005, format="%.3f")
else:
    cda = cda_presets[cda_choice]
    st.sidebar.number_input("Base CdA", value=cda, disabled=True, format="%.3f")

st.sidebar.header("3. Rolling Resistance (Crr)")
tyre_options = {
    "Top-Tier Race (Latex/Tubeless)": 0.0030, "Mid-Range Performance (GP5000)": 0.0040,
    "Training / Endurance (Gatorskin)": 0.0060, "Gravel / Cyclocross": 0.0070, "MTB Knobby": 0.0100
}
surface_options = {
    "Smooth New Asphalt": 0.0010, "Average Road (Worn Asphalt)": 0.0025,
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

st.sidebar.header("4. Mechanical Efficiency")
mech_dict = {"Optimised (Waxed / Ceramic)": 0.98, "Standard (Clean & Lubed)": 0.96, "Neglected (Dirty / Old)": 0.94}
mech_key = st.sidebar.selectbox("Drivetrain Condition", list(mech_dict.keys()), index=1)
mechanical_efficiency = mech_dict[mech_key]

st.sidebar.header("5. Weather")
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 0.0, step=0.5)
wind_cardinal = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
wind_choice = st.sidebar.selectbox("Wind Origin", list(wind_cardinal.keys()) + ["Manual Degrees"], index=0)
if wind_choice == "Manual Degrees":
    wind_deg = st.sidebar.number_input("Degrees", 0, 360, 0)
else:
    wind_deg = wind_cardinal[wind_choice]
st.sidebar.caption(f"Wind from {wind_deg}° ({get_wind_cardinal(wind_deg)})")

st.sidebar.markdown("---")
st.sidebar.markdown("""<div style="text-align: center; color: #666666; font-size: 12px;">Designed & Engineered by<br><strong>Dyfan Davies</strong></div>""", unsafe_allow_html=True)

# --- MAIN PAGE CONTENT ---
st.title("📈 Race Strategy Optimiser")
st.markdown("""
The model uses a Genetic Algorithm to attempt to find the optimal pacing strategy for the course.
""")

# --- CONFIG ---
st.subheader("1. Optimisation Settings")
with st.expander("⚙️ Configure Genetic Algorithm", expanded=True):
    col1, col2 = st.columns(2)
    with col1: 
        pop_size = st.number_input(
            "Population Size", 
            value=10, 
            min_value=2, 
            max_value=10, 
            disabled=False, 
            help="Number of candidate strategies to test per generation."
        )
    with col2: 
        num_gens = st.number_input(
            "Generations", 
            value=10, 
            min_value=1, 
            max_value=10, 
            disabled=False, 
            help="Number of evolutionary cycles to perform."
        )
    
    
    st.info("🔒 **Server Limits:** Parameters are capped at 10 to ensure server stability.")
    
    st.info("""
    **🫁 Physiological Constraints Active:**
    The Genetic Algorithm is bounded by the **Critical Power Model**. It will not generate a strategy that requires physically impossible power for a given segment duration.
    * **Sprints:** Allowed on short, steep segments (using $W'$).
    * **Endurance:** Enforced on long climbs (preserving $W'$).
    """)
    
    st.warning("⚠️ **Performance Warning:** For long courses (>50 km) or routes with frequent gradient changes, a setting of 10 can still lead to timeouts. If the app becomes unresponsive, reduce these values.")

    enable_cornering = st.checkbox("Enable Advanced Cornering Physics", value=False, help="Uncheck to prevent phantom braking on noisy GPX files.")
    st.caption("**Note:** Automatic segmentation is in beta. For optimal results using custom manual segmentation, please contact the engineer for a full analysis with manual segmentation.")
    
# --- FILE INPUT ---
st.subheader("2. Course Selection")
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

file_path = None  # <--- Ensured this matches the variable name used below

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
        st.success(f"✅ **{demo_choice} Loaded**")
    else:
        st.error(f"❌ Error: Demo file '{selected_file}' not found.")

elif uploaded_file:
    with open("temp_opt.gpx", "wb") as f: 
        f.write(uploaded_file.getbuffer())
    file_path = "temp_opt.gpx"

# --- RUN LOGIC ---
if file_path:
    try:
        gpx_result = gpx_tool.parse_gpx_file(file_path)
        
        # Profile Plot
        st.subheader("3. Course Profile")
        plot_data = gpx_result['plot_data']
        if plot_data['distances_km']:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(plot_data['distances_km'], plot_data['smoothed_elevations_m'], color='#d62728', linewidth=1)
            ax.fill_between(plot_data['distances_km'], plot_data['smoothed_elevations_m'], color='#d62728', alpha=0.1)
            #ax.set_ylim(bottom=0)
            ax.axis('off')
            st.pyplot(fig)
            
            total_climb = gpx_tool.calculate_total_ascent(gpx_result['trackPoints'])
            c1, c2 = st.columns(2)
            c1.metric("Distance", f"{gpx_result['totalDistanceKm']:.2f} km")
            c2.metric("Elevation Gain", f"{total_climb:.0f} m")

        st.markdown("---")
        if st.button("🧬 Start Evolutionary Optimisation", type="primary", use_container_width=True):
            if rider_mass <= 0 or bike_mass <= 0 or cda <= 0:
                st.error("❌ Physics Error: Inputs must be > 0.")
                st.stop()

            course_data = {'name': gpx_result['name'], 'gpx_track_points': gpx_result['trackPoints'], 'total_course_distance_m': gpx_result['totalDistanceKm'] * 1000}
            rider_params = {
                'rider_mass': rider_mass, 
                'bike_mass': bike_mass, 
                'cda': cda, 
                'mechanical_efficiency': mechanical_efficiency, 
                'rider_ftp_watts': rider_ftp, 
                'rider_target_power_watts': rider_ftp,
                'w_prime_capacity_j': w_prime_j
            }
            sim_params = {'global_wind_speed_kmh': wind_speed, 'global_wind_direction_from_deg': wind_deg, 'ambient_temperature_celsius': 20, 'surface_profile': [[0, crr]], 'enable_cornering_model': enable_cornering, 'friction_coefficient_mu': 0.8, 'rider_skill_factor': 0.95, 'starting_elevation_m': gpx_result['trackPoints'][0]['ele'], 'gpx_filename': file_path}
            
            optimizer.GA_POPULATION_SIZE = pop_size
            optimizer.GA_NUM_GENERATIONS = num_gens

            prog_bar = st.progress(0, text="Initialising Population...")
            # --- ADDED SPINNER HERE ---
            with st.spinner("Simulation in progress. This may take a few minutes depending on course complexity."):
                final_time, avg_pwr, full_profile, sim_log, strat_name, metrics, gradients, macro_profile = run_selected_optimization(
                    "genetic_algorithm", rider_params, sim_params, course_data, progress_bar=prog_bar
                )
            # --------------------------
            prog_bar.empty()

            if final_time != float('inf'):
                st.success("✅ **Optimisation Complete!**")
                
                # --- BASELINE COMPARISON ---
                baseline_watts = avg_pwr
                baseline_profile = [baseline_watts] * len(gpx_result['trackPoints'])
                base_time, base_avg, base_log, _ = physics_engine.simulate_course(
                    rider_mass, bike_mass, cda, mechanical_efficiency, gpx_result['trackPoints'][0]['ele'],
                    wind_speed, wind_deg, 20, enable_cornering, 0.8, 0.95, [[0, crr]],
                    gpx_result['trackPoints'], gpx_result['totalDistanceKm'] * 1000,
                    baseline_profile, None, False, rider_ftp=rider_ftp,
                    rider_w_prime=w_prime_j
                )
                
                # Formatter
                def fmt_time(s): return "{:d}h {:02d}m {:02d}s".format(*physics_engine.format_time_hms(s))
                
                # Data Prep
                time_opt_str = fmt_time(final_time)
                time_base_str = fmt_time(base_time)
                speed_opt = (gpx_result['totalDistanceKm'] * 1000 / final_time) * 3.6
                speed_base = (gpx_result['totalDistanceKm'] * 1000 / base_time) * 3.6
                time_diff = base_time - final_time
                diff_str = f"-{int(time_diff // 60)}m {int(time_diff % 60)}s" if time_diff > 0 else "+0s"
                speed_gain = f"+{speed_opt - speed_base:.1f} km/h"

                # --- 1. COMPARISON TABLE ---
                st.subheader("📊 Simulation Results Comparison")
                comparison_data = {
                    "Finish Time": [time_base_str, time_opt_str, diff_str],
                    "Avg Speed": [f"{speed_base:.1f} km/h", f"{speed_opt:.1f} km/h", speed_gain],
                    "Avg Power": [f"{baseline_watts:.0f} W", f"{avg_pwr:.0f} W", "0 W"]
                }
                st.table(pd.DataFrame(comparison_data, index=["Constant Power (Baseline)", "Optimised Strategy", "Difference"]))
                
                if time_diff > 0: st.info(f"⚡ **Optimisation Gain:** {diff_str} faster than riding at constant power.")

                # --- 2. COMPARATIVE CHARTS (Power & Speed) ---
                st.subheader("📈 Comparative Performance Analysis")
                st.caption("Comparison of where the strategy differs from a constant effort.")
                
                fig_pwr, fig_spd = plot_comparative_analysis(base_log, sim_log, gpx_result['name'])
                
                if fig_pwr: 
                    st.pyplot(fig_pwr)
                if fig_spd: 
                    st.pyplot(fig_spd)

                # --- 3. PACING STRATEGY GUIDE (CUSTOM FUNCTION) ---
                st.subheader("📋 Rider Pacing Strategy Guide")
                st.caption("Your race-day execution plan. This chart details the target power and gradient for each segment. Colours indicate intensity: **Blue** is Easy (Zones 1-2), **Yellow** is Sustainable (Zones 3-4), and **Red** is Attack (Zones 5-6).")
                
                fig_guide = plot_pacing_strategy_guide(gpx_result['name'], gpx_result['trackPoints'], macro_profile, rider_ftp, metrics)
                if fig_guide:
                    st.pyplot(fig_guide)

                # Table
                if metrics:
                    flat_metrics = []
                    for i, m in enumerate(metrics):
                        # FIX: Use actual power from the metric dictionary
                        actual_power = m['power'] 
                        
                        # Update Zone Label to reflect the actual effort (e.g. Zone 1 if blown up)
                        zone_label = optimizer.get_power_zone(actual_power, rider_ftp)
                        
                        flat_metrics.append({
                            "Segment": i+1, 
                            "Start (km)": f"{m['segment_start_km']:.2f}", 
                            "End (km)": f"{m['segment_end_km']:.2f}",
                            "Avg Grad (%)": f"{m['avg_gradient']:.1f}", 
                            "Avg Power (W)": f"{actual_power:.0f}", # <--- CHANGED: Shows reality
                            "Zone": zone_label
                        })
                    st.dataframe(flat_metrics, hide_index=True, use_container_width=True)

            else:
                total_mass = rider_mass + bike_mass
                st.error("🛑 **Optimisation Failed: Rider Stalled**")
                st.warning(f"""
                **The simulation could not complete the course.**
                
                This usually happens when the power output is too low to overcome gravity on steep gradients.
                
                **Diagnostics:**
                * **Input Power:** {rider_ftp} Watts
                * **Total Mass:** {total_mass} kg
                * **Power-to-Weight:** {rider_ftp/total_mass:.2f} W/kg
                
                **Suggestion:** Increase your FTP setting or check if the GPX file has extreme gradient spikes.
                """)

    except Exception as e:
        st.error(f"Error processing file: {e}")