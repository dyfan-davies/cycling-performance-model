#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:39:05 2025

@author: dyfanrhys
"""

### --- Project Module Imports --- ###
import math
from typing import Tuple, List, Dict, Optional 

### --- Constants --- ###

# --- Global Constants ---
GRAVITY = 9.81  # Acceleration due to gravity (m/s^2)
RHO_SEA_LEVEL = 1.225 # Air density at sea level (kg/m^3) at 15°C
T_SEA_LEVEL = 288.15 # Standard temperature at sea level (15°C in Kelvin)
TEMP_LAPSE_RATE = 0.0065 # Temperature lapse rate (K/m)
MOLAR_MASS_AIR = 0.0289644 # Molar mass of air (kg/mol)
UNIVERSAL_GAS_CONSTANT = 8.31447 # Universal gas constant (J/(mol·K))

# Numerical method values
TOLERANCE = 1e-9 
MAX_ITERATIONS = 1000 

# --- Dynamic Simulation Constants ---
DT = 1.0 
REPORT_INTERVAL_DISTANCE_KM = 1.0 
MIN_SPEED_FOR_FORCE_CALC_MPS = 0.5 
BRAKING_DECELERATION_MPS2 = -7.0 

# --- Corner Detection Physics Constants ---
CORNER_LOOKAHEAD_DISTANCE_M = 30.0 
BEARING_CHANGE_THRESHOLD_DEG = 45.0 
MIN_CORNER_RADIUS_M = 5.0 

# --- Dynamic CdA Constants ---
CDA_CLIMBING_THRESHOLD = 4.0  
CDA_DESCENDING_THRESHOLD = -2.0 
CDA_CLIMBING_FACTOR = 1.15    
CDA_DESCENDING_FACTOR = 0.85 

# --- Gearing Constants (Mechanical Realism) ---
MAX_EFFICIENT_CADENCE_SPEED_KMH = 65.0 
ABSOLUTE_MAX_SPEED_KMH = 75.0          

### --- Functions --- ###

def calculate_total_mass(rider_mass_kg: float, bike_mass_kg: float) -> float:
    return rider_mass_kg + bike_mass_kg

def convert_gradient_to_radians(gradient_percent: float) -> float:
    return math.atan(gradient_percent / 100)

def convert_km_to_metres(distance_km: float) -> float:
    return distance_km * 1000

def convert_kmh_to_mps(velocity_kmh: float) -> float:
    return velocity_kmh / 3.6

def convert_velocity_mps_to_kmh(velocity_mps: float) -> float:
    return velocity_mps * 3.6

def format_time_hms(total_seconds: float) -> Tuple[int, int, int]:
    if total_seconds == float('inf'):
        return float('inf'), float('inf'), float('inf')
    hours = int(total_seconds // 3600) 
    minutes = int((total_seconds % 3600) // 60) 
    seconds = int(total_seconds % 60) 
    return hours, minutes, seconds

def calculate_nutrition(total_energy_joules: float, duration_seconds: float, rider_ftp: float) -> Tuple[float, float, str, float]:
    """
    Calculates metabolic cost and fueling requirements based on total work and intensity.
    Returns: (kcal_burned, carbs_burned_grams, intake_recommendation_string, total_intake_needed_grams)
    """
    if duration_seconds <= 0:
        return 0.0, 0.0, "N/A", 0.0

    # 1. Total Metabolic Cost (1 kJ Mechanical Work ~= 1 kcal Metabolic Cost)
    # Human efficiency is ~20-25%. 1 kcal = 4.184 kJ. 
    # The math cancels out roughly 1:1. (e.g. 1000kJ output / 0.24 eff / 4.184 = 996 kcal)
    total_kcal = total_energy_joules / 1000.0
    
    # 2. Average Power & Intensity
    avg_power = total_energy_joules / duration_seconds
    intensity_factor = avg_power / rider_ftp if rider_ftp > 0 else 0.7

    # 3. Estimate Carb vs Fat Burn Ratio (Respiratory Exchange Ratio)
    # Zone 2 (0.6 IF) = ~40% Carbs | Threshold (1.0 IF) = 100% Carbs
    if intensity_factor < 0.6:
        carb_ratio = 0.4
    elif intensity_factor > 1.0:
        carb_ratio = 1.0
    else:
        # Linear interpolation
        carb_ratio = 0.4 + ((intensity_factor - 0.6) / 0.4) * 0.6
        
    total_carbs_burned_grams = (total_kcal * carb_ratio) / 4.0 # 4 kcal per gram of carb
    
    # 4. Intake Recommendation (Standard Guidelines)
    duration_hours = duration_seconds / 3600.0
    if duration_hours < 1.25:
        rec_str = "0g (Water/Mouth Rinse)"
        intake_g_per_hr = 0
    elif duration_hours < 2.5:
        rec_str = "30-60g per hour"
        intake_g_per_hr = 45
    elif duration_hours < 4.0:
        rec_str = "60-90g per hour"
        intake_g_per_hr = 75
    else:
        rec_str = "90g+ per hour"
        intake_g_per_hr = 90
        
    total_intake_needed = intake_g_per_hr * duration_hours
    
    return total_kcal, total_carbs_burned_grams, rec_str, total_intake_needed

def calculate_height_adjusted_cda(base_cda: float, rider_height_cm: float) -> float:
    """
    Scales Base CdA for rider height using geometric similarity.
    Standard reference height is 178cm.
    Scale factor approximates Frontal Area scaling (approx height^1.5).
    """
    STANDARD_HEIGHT_CM = 178.0
    if rider_height_cm <= 0: return base_cda
    
    scale_factor = (rider_height_cm / STANDARD_HEIGHT_CM) ** 1.5
    return base_cda * scale_factor

def calculate_average_speed_kmh(total_distance_km: float, total_time_seconds: float) -> float:
    if total_time_seconds <= 0: return 0.0
    return total_distance_km / (total_time_seconds / 3600)

def calculate_time_seconds(distance_m: float, velocity_mps: float) -> float:
    if velocity_mps <= 0: return float('inf')
    return distance_m / velocity_mps

def calculate_air_density(elevation_m: float, temperature_celsius: float) -> float:
    temp_kelvin = temperature_celsius + 273.15 
    rho_sea_level_actual_temp = RHO_SEA_LEVEL * (T_SEA_LEVEL / temp_kelvin)
    if elevation_m >= 0:
        rho = rho_sea_level_actual_temp * (1 - TEMP_LAPSE_RATE * elevation_m / T_SEA_LEVEL) ** \
              (GRAVITY * MOLAR_MASS_AIR / (UNIVERSAL_GAS_CONSTANT * TEMP_LAPSE_RATE))
    else:
        rho = rho_sea_level_actual_temp
    return rho

def calculate_cda(frontal_area_m2: float, cd_coefficient: float) -> float:
    return frontal_area_m2 * cd_coefficient

def calculate_dynamic_cda(base_cda: float, gradient_percent: float) -> float:
    if gradient_percent > CDA_CLIMBING_THRESHOLD:
        return base_cda * CDA_CLIMBING_FACTOR 
    elif gradient_percent < CDA_DESCENDING_THRESHOLD:
        return base_cda * CDA_DESCENDING_FACTOR 
    else:
        return base_cda 

# --- Force Calculation Functions ---

def calculate_gravity_force(mass_kg: float, gradient_radians: float) -> float:
    return mass_kg * GRAVITY * math.sin(gradient_radians)

def calculate_rolling_resistance_force(mass_kg: float, gradient_radians: float, crr: float) -> float:
    return mass_kg * GRAVITY * math.cos(gradient_radians) * crr

def calculate_drag_force(cda_m2: float, air_density_kg_m3: float, v_relative_to_air_signed: float) -> float:
    drag_magnitude = 0.5 * cda_m2 * air_density_kg_m3 * (v_relative_to_air_signed ** 2)
    if v_relative_to_air_signed > 0:
        return -drag_magnitude 
    elif v_relative_to_air_signed < 0:
        return drag_magnitude  
    else: 
        return 0.0

def calculate_physics_based_corner_speeds(gpx_track_points, friction_coefficient_mu, rider_skill_factor):
    corner_speed_limits = []
    num_points = len(gpx_track_points)
    if num_points < 2: return []

    for i in range(num_points - 1):
        p1 = gpx_track_points[i]
        p2_index = i + 1
        dist_so_far = 0.0
        
        while p2_index < num_points:
            dist_seg = gpx_track_points[p2_index]['cumulativeDistanceM'] - gpx_track_points[p2_index-1]['cumulativeDistanceM']
            dist_so_far += dist_seg
            if dist_so_far >= CORNER_LOOKAHEAD_DISTANCE_M:
                break
            if p2_index > i + 100: break 
            p2_index += 1
            
        if p2_index >= num_points: break 
            
        p2 = gpx_track_points[p2_index]
        bearing1 = p1['bearingDeg']
        bearing2 = p2['bearingDeg']
        delta_bearing = abs(bearing1 - bearing2)
        if delta_bearing > 180: delta_bearing = 360 - delta_bearing

        if delta_bearing >= BEARING_CHANGE_THRESHOLD_DEG:
            chord_distance_m = p2['cumulativeDistanceM'] - p1['cumulativeDistanceM']
            if chord_distance_m > 0 and delta_bearing > 0:
                try:
                    radius_m = (chord_distance_m / 2.0) / math.sin(math.radians(delta_bearing) / 2.0)
                except ValueError:
                    radius_m = MIN_CORNER_RADIUS_M
                radius_m = max(MIN_CORNER_RADIUS_M, radius_m) 
                max_speed_mps = math.sqrt(friction_coefficient_mu * GRAVITY * radius_m)
                
                if gpx_track_points[i]['segmentGradientPercent'] > 0.5:
                    max_speed_mps *= 3.0 
                
                max_speed_mps *= rider_skill_factor
                max_speed_kmh = convert_velocity_mps_to_kmh(max_speed_mps)
                corner_center_distance_m = (p1['cumulativeDistanceM'] + p2['cumulativeDistanceM']) / 2.0
                corner_speed_limits.append((corner_center_distance_m, max_speed_kmh, radius_m, delta_bearing))
                
    return corner_speed_limits

def get_course_properties_at_distance(distance_m, gpx_track_points, corner_speeds, total_course_distance_m):
    current_gradient_percent = 0.0
    current_bearing_deg = 0.0
    current_corner_speed_limit_kmh = None
    current_segment_index = 0 

    if not gpx_track_points: return 0.0, None, 0.0, 0 
    
    idx = 0
    # Simple search - can be optimized, but safe
    while idx < len(gpx_track_points) - 1 and gpx_track_points[idx+1]['cumulativeDistanceM'] <= distance_m:
        idx += 1
    current_segment_index = idx 
    current_gradient_percent = gpx_track_points[current_segment_index]['segmentGradientPercent']
    current_segment_bearing_deg = gpx_track_points[current_segment_index]['bearingDeg']
    
    applicable_speed_limit = float('inf')
    window_size = CORNER_LOOKAHEAD_DISTANCE_M 
    
    for corner_center_m, speed_kmh, _, _ in corner_speeds:
        if (corner_center_m - window_size) <= distance_m < (corner_center_m + window_size):
            applicable_speed_limit = min(applicable_speed_limit, speed_kmh)
    
    if applicable_speed_limit != float('inf'):
        current_corner_speed_limit_kmh = applicable_speed_limit

    return current_gradient_percent, current_corner_speed_limit_kmh, current_segment_bearing_deg, current_segment_index

def simulate_course(
    rider_mass: float, bike_mass: float, cda: float, mechanical_efficiency: float,
    starting_elevation_m: float,
    global_wind_speed_kmh: float, global_wind_direction_from_deg: float,
    ambient_temperature_celsius: float,
    enable_cornering_model: bool, friction_coefficient_mu: float, rider_skill_factor: float,
    surface_profile: List[Tuple[float, float]], 
    gpx_track_points: List[Dict], total_course_distance_m: float,
    power_profile_watts: List[float], 
    gpx_to_opt_segment_map: Optional[List[int]] = None, 
    report_progress: bool = True,
    rider_ftp: float = 200.0,
    rider_w_prime: float = 15000.0 
) -> Tuple[float, float, List[Dict], float]: 
    
    total_mass = calculate_total_mass(rider_mass, bike_mass) 
    GLOBAL_WIND_SPEED_MPS = convert_kmh_to_mps(global_wind_speed_kmh)

    # --- MODE DETECTION: MACRO OR MICRO? ---
    # If the power profile length matches the GPX points, we are in legacy (Micro) mode.
    # If it matches the number of segments in the map, we are in Macro mode.
    using_macro_strategy = False
    if gpx_to_opt_segment_map is not None:
        # Get the max index from the map to determine number of segments required
        required_segments = max(gpx_to_opt_segment_map) + 1
        if len(power_profile_watts) == required_segments:
            using_macro_strategy = True
    
    # If map is missing but we need one, create a 1-to-1 dummy map
    if gpx_to_opt_segment_map is None:
        gpx_to_opt_segment_map = list(range(len(gpx_track_points)))

    # --- PRE-CALCULATION PHASE ---
    corner_speeds_for_sim = [] 
    if enable_cornering_model and gpx_track_points:
        corner_speeds_for_sim = calculate_physics_based_corner_speeds(
            gpx_track_points, friction_coefficient_mu, rider_skill_factor
        )
        
    current_distance_m = 0.0
    current_time_seconds = 0.0
    current_velocity_mps = 0.1 
    cumulative_elevation_m = starting_elevation_m 
    total_energy_joules = 0.0 
    
    current_w_prime_balance = rider_w_prime 
    min_w_prime_balance = rider_w_prime

    if report_progress:
        next_report_distance_m = convert_km_to_metres(REPORT_INTERVAL_DISTANCE_KM)
    else:
        next_report_distance_m = float('inf')

    MAX_SIMULATION_DURATION_SECONDS = 3600 * 24 
    
    # --- CHECKPOINT INIT ---
    checkpoint_dist = 0.0
    checkpoint_time = 0.0
    checkpoint_w_prime = rider_w_prime
    checkpoint_ele = starting_elevation_m
    checkpoint_log_index = 0
    last_strategy_idx = -1
    
    simulation_log = [] 

    while current_distance_m < total_course_distance_m and current_time_seconds < MAX_SIMULATION_DURATION_SECONDS:
        
        is_braking = False
        is_coasting = False
        coasting_time_buffer_s = 2.0 

        # 1. RETRIEVE GRADIENT & SEGMENT INFO
        gradient_percent, corner_speed_limit_kmh, segment_bearing_deg, current_gpx_segment_index = \
            get_course_properties_at_distance(current_distance_m, gpx_track_points, corner_speeds_for_sim, total_course_distance_m)

        # --- STRATEGY LOOKUP ---
        # Determine which "Strategy Gene" applies to this location
        if using_macro_strategy:
            current_strategy_idx = gpx_to_opt_segment_map[current_gpx_segment_index]
        else:
            current_strategy_idx = current_gpx_segment_index

        # --- CHECKPOINT UPDATE ---
        # If we enter a new Strategy Segment, save state.
        if current_strategy_idx != last_strategy_idx:
            checkpoint_dist = current_distance_m
            checkpoint_time = current_time_seconds
            checkpoint_w_prime = current_w_prime_balance
            checkpoint_ele = cumulative_elevation_m
            checkpoint_log_index = len(simulation_log)
            last_strategy_idx = current_strategy_idx

        # 2. UPHILL & LOW SPEED OVERRIDE (Corner Safety)
        check_for_corners = True
        if gradient_percent > -1.0: check_for_corners = False 
        if current_velocity_mps < 11.1: check_for_corners = False 
        
        if check_for_corners:
            for corner_dist_m, corner_speed_kmh, _, _ in corner_speeds_for_sim:
                if corner_dist_m > current_distance_m: 
                    corner_speed_mps = convert_kmh_to_mps(corner_speed_kmh)
                    if current_velocity_mps > corner_speed_mps:
                        required_braking_dist = (corner_speed_mps**2 - current_velocity_mps**2) / (2 * BRAKING_DECELERATION_MPS2)
                        dist_to_corner = corner_dist_m - current_distance_m
                        if dist_to_corner <= required_braking_dist:
                            is_braking = True
                            break 
                        coasting_dist_threshold = required_braking_dist + (current_velocity_mps * coasting_time_buffer_s)
                        if dist_to_corner <= coasting_dist_threshold:
                            is_coasting = True

        # --- FORCE APPLICATION ---
        if is_braking:
            net_force = total_mass * BRAKING_DECELERATION_MPS2 
            simulated_power_watts = 0.0 
            effective_power_watts = 0.0 
            
        elif is_coasting:
            simulated_power_watts = 0.0 
            cumulative_elevation_m = gpx_track_points[current_gpx_segment_index]['ele']
            current_air_density = calculate_air_density(cumulative_elevation_m, ambient_temperature_celsius)
            
            current_cda = calculate_dynamic_cda(cda, gradient_percent) 
            relative_wind_angle_rad = math.radians(global_wind_direction_from_deg - segment_bearing_deg)
            headwind_component_mps = GLOBAL_WIND_SPEED_MPS * math.cos(relative_wind_angle_rad)
            crosswind_component_mps = GLOBAL_WIND_SPEED_MPS * math.sin(relative_wind_angle_rad)
            v_relative_to_air_signed = current_velocity_mps + headwind_component_mps
            
            if abs(v_relative_to_air_signed) > 0.1:
                apparent_yaw_rad = math.atan2(abs(crosswind_component_mps), abs(v_relative_to_air_signed))
            else:
                apparent_yaw_rad = 0.0
            
            yaw_factor = 1.0 + 0.5 * (math.sin(apparent_yaw_rad) ** 2)
            current_cda = current_cda * yaw_factor

            current_crr = gpx_track_points[current_gpx_segment_index].get('crr', 0.005)
            gradient_radians = convert_gradient_to_radians(gradient_percent)
            gravity_force = calculate_gravity_force(total_mass, gradient_radians)
            rolling_resistance_force = calculate_rolling_resistance_force(total_mass, gradient_radians, current_crr)
            drag_force = calculate_drag_force(current_cda, current_air_density, v_relative_to_air_signed)
            
            net_force = drag_force - (gravity_force + rolling_resistance_force)
            effective_power_watts = 0.0

        else:
            # --- POWER LOOKUP (MACRO AWARE) ---
            target_power_watts = power_profile_watts[min(current_strategy_idx, len(power_profile_watts) - 1)]
            simulated_power_watts = target_power_watts
                
            # --- GEARING LIMITS ---
            current_speed_kmh = convert_velocity_mps_to_kmh(current_velocity_mps)
            
            if current_speed_kmh > ABSOLUTE_MAX_SPEED_KMH:
                effective_power_watts = 0.0 
            elif current_speed_kmh > MAX_EFFICIENT_CADENCE_SPEED_KMH:
                eff_factor = 1.0 - ((current_speed_kmh - MAX_EFFICIENT_CADENCE_SPEED_KMH) / (ABSOLUTE_MAX_SPEED_KMH - MAX_EFFICIENT_CADENCE_SPEED_KMH))
                effective_power_watts = simulated_power_watts * mechanical_efficiency * eff_factor
            else:
                effective_power_watts = simulated_power_watts * mechanical_efficiency
            
            cumulative_elevation_m = gpx_track_points[current_gpx_segment_index]['ele']
            current_air_density = calculate_air_density(cumulative_elevation_m, ambient_temperature_celsius)
            
            current_cda = calculate_dynamic_cda(cda, gradient_percent) 
            relative_wind_angle_rad = math.radians(global_wind_direction_from_deg - segment_bearing_deg)
            headwind_component_mps = GLOBAL_WIND_SPEED_MPS * math.cos(relative_wind_angle_rad)
            crosswind_component_mps = GLOBAL_WIND_SPEED_MPS * math.sin(relative_wind_angle_rad)
            v_relative_to_air_signed = current_velocity_mps + headwind_component_mps
            
            if abs(v_relative_to_air_signed) > 0.1:
                apparent_yaw_rad = math.atan2(abs(crosswind_component_mps), abs(v_relative_to_air_signed))
            else:
                apparent_yaw_rad = 0.0
            
            yaw_factor = 1.0 + 0.5 * (math.sin(apparent_yaw_rad) ** 2)
            current_cda = current_cda * yaw_factor
            
            current_crr = gpx_track_points[current_gpx_segment_index].get('crr', 0.005)
            gradient_radians = convert_gradient_to_radians(gradient_percent)
            gravity_force = calculate_gravity_force(total_mass, gradient_radians)
            rolling_resistance_force = calculate_rolling_resistance_force(total_mass, gradient_radians, current_crr)
            drag_force = calculate_drag_force(current_cda, current_air_density, v_relative_to_air_signed)
            effective_speed_for_force = max(current_velocity_mps, MIN_SPEED_FOR_FORCE_CALC_MPS)
            propulsive_force = effective_power_watts / effective_speed_for_force
            net_force = propulsive_force + drag_force - (gravity_force + rolling_resistance_force)
        
        # --- PHYSIOLOGY UPDATE ---
        power_delta = simulated_power_watts - rider_ftp
        
        if power_delta > 0:
            current_w_prime_balance -= (power_delta * DT)
        else:
            d_cp = rider_ftp - simulated_power_watts
            tau_w_prime = 546.0 * math.exp(-0.01 * d_cp) + 316.0
            current_w_prime_balance = rider_w_prime - (rider_w_prime - current_w_prime_balance) * math.exp(-DT / tau_w_prime)
        
        if current_w_prime_balance > rider_w_prime: current_w_prime_balance = rider_w_prime
        if current_w_prime_balance < min_w_prime_balance: min_w_prime_balance = current_w_prime_balance

        # --- RESTART LOGIC (SMART CORRECTION) ---
        if current_w_prime_balance < 0:
            # We bonked! The optimizer tried to go too hard.
            # Instead of an arbitrary -10% penalty, let's find the PHYSICAL LIMIT.
            
            # 1. How much energy did we start this segment with?
            w_prime_available = checkpoint_w_prime 
            
            # 2. How long did we spend in this segment before dying?
            # We can approximate the duration of the segment based on distance and current speed,
            # but a safer heuristic for the Critical Power model is:
            # Max Power = CP + (Available W' / Time)
            # Since we don't know the exact Time yet (circular dependency), we can use a simpler 
            # "Damping" approach that is less punishing than 10%.
            
            # LOGIC: Reduce power to halfway between "Critical Power" and "Failed Power"
            # This gently walks the power down until it barely fits.
            failed_power = power_profile_watts[min(current_strategy_idx, len(power_profile_watts) - 1)]
            
            # If we are above FTP, nudge towards FTP.
            if failed_power > rider_ftp:
                # New Power = average of (Failed Attempt) and (Safe FTP)
                # This converges much faster and stays "high" rather than dropping like a stone.
                new_power = (failed_power + rider_ftp) / 2.0
                
                # If the difference is tiny, just clamp to FTP to stop infinite loops
                if abs(failed_power - new_power) < 1.0:
                    new_power = rider_ftp
                
                power_profile_watts[current_strategy_idx] = new_power
            else:
                # If we bonked BELOW FTP (shouldn't happen with this model, but safety first),
                # just reduce by 5%
                power_profile_watts[current_strategy_idx] *= 0.95

            # 3. Rewind: Reset variables to checkpoint
            current_distance_m = checkpoint_dist
            current_time_seconds = checkpoint_time
            current_w_prime_balance = checkpoint_w_prime
            cumulative_elevation_m = checkpoint_ele
            
            # 4. Clear History
            simulation_log = simulation_log[:checkpoint_log_index]
            
            # 5. Retry
            continue

        # --- MOTION UPDATE ---
        acceleration_t0 = net_force / total_mass
        distance_covered_in_step_m = current_velocity_mps * DT
        current_distance_m += distance_covered_in_step_m
        new_velocity_mps = current_velocity_mps + acceleration_t0 * DT
        
        if corner_speed_limit_kmh is not None and check_for_corners:
            corner_limit_mps = convert_kmh_to_mps(corner_speed_limit_kmh)
            if new_velocity_mps > corner_limit_mps: new_velocity_mps = corner_limit_mps
        
        if new_velocity_mps < 0: new_velocity_mps = 0.0
        current_velocity_mps = new_velocity_mps
        current_time_seconds += DT
        total_energy_joules += simulated_power_watts * DT

        # --- LOGGING ---
        simulation_log.append({
            'distance': current_distance_m / 1000,
            'time': current_time_seconds,
            'velocity': convert_velocity_mps_to_kmh(new_velocity_mps),
            'elevation': cumulative_elevation_m,
            'gradient': gradient_percent,
            'power': simulated_power_watts, 
            'macro_segment_index': current_strategy_idx if using_macro_strategy else -1,
            'w_prime': current_w_prime_balance 
        })
        
        if report_progress and current_distance_m >= next_report_distance_m:
            vel_kmh = convert_velocity_mps_to_kmh(new_velocity_mps)
            print(f"  At {current_distance_m/1000:.2f}km: {vel_kmh:.1f}km/h, {simulated_power_watts:.0f}W, W'={current_w_prime_balance/1000:.1f}kJ")
            next_report_distance_m += convert_km_to_metres(REPORT_INTERVAL_DISTANCE_KM)
        
        # Safety break if stuck
        if new_velocity_mps < 0.01 and effective_power_watts <= 0 and gradient_percent >= 0:
            break
    
    final_total_time_seconds = current_time_seconds
    final_actual_average_power_watts = 0.0
    if current_distance_m >= total_course_distance_m - 0.1 and final_total_time_seconds > 0:
        final_actual_average_power_watts = total_energy_joules / final_total_time_seconds
    else:
        final_total_time_seconds = float('inf')

    return final_total_time_seconds, final_actual_average_power_watts, simulation_log, min_w_prime_balance