#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:39:29 2025

@author: dyfanrhys
"""

### --- Project Module Imports --- ###
import physics_engine
import math
from xml.etree import ElementTree as ET
import re 
from scipy.signal import savgol_filter 
from typing import Tuple, List, Dict, Optional 

# --- GPX Smoothing and Data Cleaning Configuration ---

# --- CONFIGURATION ---
GPX_SMOOTHING_ORDER = 3 
SMOOTHING_DISTANCE_TARGET_M = 300.0 
MACRO_SMOOTHING_DISTANCE_TARGET_M = 750.0
GRADIENT_CLIP_THRESHOLD = 25.0 

# --- Dynamic Segmentation Configuration ---
SEGMENT_CLIMB_THRESHOLD = 2.0     
SEGMENT_DESCENT_THRESHOLD = -1.5  
MIN_SEGMENT_LENGTH_M = 250.0 

def calculate_bearing(lat1, lon1, lat2, lon2) -> float:
    """Calculates bearing between two lat/lon points in degrees (0-360)."""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lon = lon2_rad - lon1_rad
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - (math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    initial_bearing_rad = math.atan2(x, y)
    initial_bearing_deg = math.degrees(initial_bearing_rad)
    return (initial_bearing_deg + 360) % 360

def smooth_elevation_data(gpx_track_points: List[Dict], window_size: int, poly_order: int) -> List[Dict]:
    if window_size <= poly_order:
        print(f"Warning: Window size ({window_size}) too small for poly order ({poly_order}). Skipping smoothing.")
        return gpx_track_points
        
    elevation_values = [point['ele'] for point in gpx_track_points]
    smoothed_elevation = savgol_filter(elevation_values, window_size, poly_order)
    
    # PASS 1: Assign smoothed elevation to ALL points first
    for i, point in enumerate(gpx_track_points):
        point['smoothedElevation'] = smoothed_elevation[i]
        
    # PASS 2: Calculate gradients using the fully populated data
    for i, point in enumerate(gpx_track_points):
        if i < len(gpx_track_points) - 1:
            delta_elevation = gpx_track_points[i+1]['smoothedElevation'] - gpx_track_points[i]['smoothedElevation']
            delta_distance = gpx_track_points[i+1]['cumulativeDistanceM'] - gpx_track_points[i]['cumulativeDistanceM']
            
            if delta_distance > 0:
                gradient = (delta_elevation / delta_distance) * 100.0 
            else:
                gradient = 0.0
        else:
            gradient = gpx_track_points[i-1]['segmentGradientPercent']
        point['segmentGradientPercent'] = gradient
        
    return gpx_track_points

def parse_gpx_file(file_input, force_smoothing_window: Optional[int] = None, auto_downsample: bool = True) -> Dict:
    try:
        if isinstance(file_input, str):
            with open(file_input, 'r', encoding='utf-8') as f:
                xml_string = f.read()
        else:
            if hasattr(file_input, 'seek'):
                file_input.seek(0)
            content = file_input.read()
            xml_string = content.decode('utf-8') if isinstance(content, bytes) else content
    except Exception as e:
        raise ValueError(f"Could not read GPX input: {e}")

    xml_string = re.sub(r'\sxmlns="[^"]+"', '', xml_string, count=1)
    if not xml_string.strip():
        raise ValueError("GPX file content is empty.")
        
    # Parse XML
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML format: {e}")

    # --- FIX: STRIP NAMESPACES ROBUSTLY ---
    # This forces the parser to ignore the "http://..." prefix so .findall('trkpt') works
    # for both GPX 1.0 and 1.1 files.
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    # --------------------------------------
    track_points_raw = []
    trkpts = root.findall('.//trkpt')
    if not trkpts:
        raise ValueError("No track points found in GPX file.")
    
    for trkpt in trkpts:
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        ele_elem = trkpt.find('ele')
        ele = float(ele_elem.text) if ele_elem is not None else 0.0
        track_points_raw.append({'lat': lat, 'lon': lon, 'ele': ele})
        
    # Initialize flag
    was_downsampled = False
    original_count = len(track_points_raw)
    
    MAX_POINTS = 3500
    total_raw = len(track_points_raw)
    
    if auto_downsample and total_raw > MAX_POINTS:
        # Calculate step to keep roughly MAX_POINTS
        step = total_raw // MAX_POINTS
        # Ensure step is at least 1 to avoid division by zero errors
        step = max(1, step) 
        
        print(f"⚠️ Large file detected ({total_raw} pts). Downsampling by factor of {step}.")
        track_points_raw = track_points_raw[::step]
    elif total_raw > MAX_POINTS and not auto_downsample:
        print(f"ℹ️ High Precision Mode: Keeping all {total_raw} points (Performance may suffer).")
    # --------------------------------
        
    temp_cumulative_distance_m = 0.0
    prev_lat, prev_lon = None, None
    for point in track_points_raw:
        if prev_lat is not None:
            R = 6371e3
            phi1, phi2 = math.radians(prev_lat), math.radians(point['lat'])
            delta_lambda = math.radians(point['lon'] - prev_lon)
            a = math.sin((phi2 - phi1) / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            segment_distance_m = R * c
            temp_cumulative_distance_m += segment_distance_m
        point['cumulativeDistanceM'] = temp_cumulative_distance_m
        prev_lat, prev_lon = point['lat'], point['lon']
    
    total_distance_km = temp_cumulative_distance_m / 1000
    num_points = len(track_points_raw)
    
    smoothing_window_to_use = 0
    if force_smoothing_window and force_smoothing_window > GPX_SMOOTHING_ORDER:
        smoothing_window_to_use = force_smoothing_window
        print(f"  > Manual override active. Using forced smoothing window of {smoothing_window_to_use}.")
    else:
        dynamic_window = GPX_SMOOTHING_ORDER + 2 
        if total_distance_km > 0.1:
            points_per_km = num_points / total_distance_km
            target_points_in_window = int((points_per_km / 1000) * SMOOTHING_DISTANCE_TARGET_M)
            if target_points_in_window % 2 == 0:
                target_points_in_window += 1
            dynamic_window = max(GPX_SMOOTHING_ORDER + 2, min(target_points_in_window, num_points - 1))
            if dynamic_window % 2 == 0:
                dynamic_window = dynamic_window -1 if dynamic_window > GPX_SMOOTHING_ORDER + 2 else dynamic_window + 1
        smoothing_window_to_use = dynamic_window
        print(f"  > GPX data density: {points_per_km:.1f} points/km. Using dynamic smoothing window of {smoothing_window_to_use}.")

    track_points_smoothed_ele = smooth_elevation_data(track_points_raw, smoothing_window_to_use, GPX_SMOOTHING_ORDER)
    
    track_points_final = []
    cumulative_distance_m = 0.0
    prev_lat, prev_lon, prev_ele = None, None, None

    for i, point in enumerate(track_points_smoothed_ele):
        lat, lon, ele = point['lat'], point['lon'], point['ele']
        segment_distance_m = 0.0
        bearing_deg = 0.0
        segment_gradient_percent = 0.0
        
        if prev_lat is not None:
            R = 6371e3 
            phi1, phi2 = math.radians(prev_lat), math.radians(lat)
            delta_phi = math.radians(lat - prev_lat)
            delta_lambda = math.radians(lon - prev_lon)
            a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            segment_distance_m = R * c
            cumulative_distance_m += segment_distance_m
            
            elevation_change_m = ele - prev_ele
            if segment_distance_m > 1e-3: 
                segment_gradient_percent = (elevation_change_m / segment_distance_m) * 100
            
            if segment_gradient_percent > GRADIENT_CLIP_THRESHOLD:
                segment_gradient_percent = GRADIENT_CLIP_THRESHOLD
            elif segment_gradient_percent < -GRADIENT_CLIP_THRESHOLD:
                segment_gradient_percent = -GRADIENT_CLIP_THRESHOLD
            bearing_deg = calculate_bearing(prev_lat, prev_lon, lat, lon)

        track_points_final.append({
            'lat': lat, 'lon': lon, 'ele': ele,
            'smoothedElevation': point.get('smoothedElevation', ele), 
            'cumulativeDistanceM': cumulative_distance_m,
            'segmentDistanceM': segment_distance_m,
            'bearingDeg': bearing_deg,
            'segmentGradientPercent': segment_gradient_percent
        })
        prev_lat, prev_lon, prev_ele = lat, lon, ele

    if len(track_points_final) > 1:
        track_points_final[0]['bearingDeg'] = calculate_bearing(
            track_points_final[0]['lat'], track_points_final[0]['lon'],
            track_points_final[1]['lat'], track_points_final[1]['lon']
        )
        
    course_name_elem = root.find('.//name')
    course_name = course_name_elem.text if course_name_elem is not None else 'Unnamed Route'
    raw_elevations_for_plot = [p['ele'] for p in track_points_raw]
    
    plot_data = {
        'distances_km': [p['cumulativeDistanceM'] / 1000 for p in track_points_final],
        'raw_elevations_m': raw_elevations_for_plot,
        'smoothed_elevations_m': [p['ele'] for p in track_points_final],
        'smoothing_window': smoothing_window_to_use
    }

    return {
        'name': course_name,
        'trackPoints': track_points_final,
        'totalDistanceKm': cumulative_distance_m / 1000,
        'plot_data': plot_data,
        'was_downsampled': was_downsampled,
        'original_point_count': original_count,
        'final_point_count': len(track_points_final)
    }

def get_macro_segment_data(
    gpx_track_points: List[Dict], 
    total_course_distance_m: float,
    rider_params: Optional[Dict] = None, 
    manual_boundaries_km: Optional[List[float]] = None 
) -> Tuple[List[int], List[float], int]:
    
    if not gpx_track_points:
        return [], [], 0

    num_points = len(gpx_track_points)
    gpx_to_opt_segment_map = [0] * num_points
    macro_segments = []

    if manual_boundaries_km and len(manual_boundaries_km) > 0:
        print(f"  > Using MANUAL segmentation boundaries: {manual_boundaries_km}")
        boundaries_m = sorted([b * 1000.0 for b in manual_boundaries_km])
        current_boundary_idx = 0
        current_segment_start_idx = 0
        for i in range(num_points):
            dist = gpx_track_points[i]['cumulativeDistanceM']
            if current_boundary_idx < len(boundaries_m) and dist >= boundaries_m[current_boundary_idx]:
                macro_segments.append(gpx_track_points[current_segment_start_idx:i])
                current_segment_start_idx = i
                current_boundary_idx += 1
        macro_segments.append(gpx_track_points[current_segment_start_idx:])
    else:
        raw_gradients = [p.get('segmentGradientPercent', 0) for p in gpx_track_points]
        density = calculate_gpx_density(gpx_track_points, total_course_distance_m)
        window = calculate_smoothing_window_size(density, 500.0) 
        if window > 3 and len(raw_gradients) > window:
             smoothed_gradients = savgol_filter(raw_gradients, window, 1) 
        else:
             smoothed_gradients = raw_gradients

        point_states = []
        for g in smoothed_gradients:
            if g > SEGMENT_CLIMB_THRESHOLD:
                point_states.append(1) 
            elif g < SEGMENT_DESCENT_THRESHOLD:
                point_states.append(-1) 
            else:
                point_states.append(0) 
        
        temp_segments = []
        if not point_states: return [], [], 0

        current_state = point_states[0]
        current_points = []
        current_len_m = 0.0
        
        for i in range(num_points):
            p = gpx_track_points[i]
            if point_states[i] != current_state:
                temp_segments.append({
                    'state': current_state,
                    'points': current_points,
                    'length_m': current_len_m
                })
                current_state = point_states[i]
                current_points = []
                current_len_m = 0.0
            current_points.append(p)
            current_len_m += p['segmentDistanceM']
        temp_segments.append({'state': current_state, 'points': current_points, 'length_m': current_len_m})
        
        merged_segments = []
        if temp_segments:
            merged_segments.append(temp_segments[0])
            for i in range(1, len(temp_segments)):
                next_seg = temp_segments[i]
                prev_seg = merged_segments[-1]
                is_tiny = next_seg['length_m'] < MIN_SEGMENT_LENGTH_M
                if (prev_seg['state'] == next_seg['state']) or is_tiny:
                    prev_seg['points'].extend(next_seg['points'])
                    prev_seg['length_m'] += next_seg['length_m']
                else:
                    merged_segments.append(next_seg)
        macro_segments = [s['points'] for s in merged_segments]

    num_dynamic_segments = len(macro_segments)
    avg_macro_segment_gradients = []
    point_idx = 0
    for seg_idx, segment in enumerate(macro_segments):
        grad_sum = 0.0
        dist_sum = 0.0
        for i, p in enumerate(segment):
            if point_idx < num_points:
                gpx_to_opt_segment_map[point_idx] = seg_idx
                point_idx += 1
            d = p['segmentDistanceM']
            grad_sum += p['segmentGradientPercent'] * d
            dist_sum += d
        if dist_sum > 0:
            avg_macro_segment_gradients.append(grad_sum / dist_sum)
        else:
            avg_macro_segment_gradients.append(0.0)

    return gpx_to_opt_segment_map, avg_macro_segment_gradients, num_dynamic_segments

def calculate_macro_segment_metrics(
    simulation_log: List[Dict],
    num_optimization_segments: int,
    optimized_macro_power_profile: List[float],
    final_avg_macro_segment_gradients: List[float],
    total_course_distance_m: float
) -> List[Dict]:
    macro_segment_metrics = []
    grouped_log_entries = [[] for _ in range(num_optimization_segments)]
    for entry in simulation_log:
        macro_idx = entry.get('macro_segment_index')
        if macro_idx is not None and 0 <= macro_idx < num_optimization_segments:
            grouped_log_entries[macro_idx].append(entry)

    segment_start_dist_km = 0.0
    for i in range(num_optimization_segments):
        segment_entries = grouped_log_entries[i]
        # If the rider blew up, this will show the lower, real value.
        if segment_entries:
            # Calculate mean power from the simulation log for this segment
            total_pwr = sum(entry['power'] for entry in segment_entries)
            power = total_pwr / len(segment_entries)
        elif i < len(optimized_macro_power_profile):
            # Fallback if no simulation data exists for this segment
            power = optimized_macro_power_profile[i]
        else:
            power = 0.0
        if i < len(final_avg_macro_segment_gradients):
            avg_gradient = final_avg_macro_segment_gradients[i]
        else:
            avg_gradient = 0.0

        time_taken_s = 0.0
        avg_speed_kmh = 0.0
        actual_segment_distance_km = 0.0
        
        if segment_entries:
            first_entry_time = segment_entries[0]['time']
            last_entry_time = segment_entries[-1]['time']
            first_entry_distance_km = segment_entries[0]['distance']
            last_entry_distance_km = segment_entries[-1]['distance']
            
            time_taken_s = last_entry_time - first_entry_time
            actual_segment_distance_km = last_entry_distance_km - first_entry_distance_km

            if time_taken_s > 0:
                avg_speed_kmh = physics_engine.calculate_average_speed_kmh(actual_segment_distance_km, time_taken_s)

        segment_end_dist_km = segment_start_dist_km + actual_segment_distance_km

        macro_segment_metrics.append({
            'segment_start_km': segment_start_dist_km,
            'segment_end_km': segment_end_dist_km,
            'avg_gradient': avg_gradient,
            'power': power,
            'time_taken_s': time_taken_s,
            'avg_speed_kmh': avg_speed_kmh,
            'actual_distance_km': actual_segment_distance_km
        })
        segment_start_dist_km = segment_end_dist_km
    return macro_segment_metrics

def calculate_total_ascent(track_points: List[Dict]) -> float:
    total_ascent = 0.0
    for i in range(1, len(track_points)):
        elevation_change = track_points[i]['ele'] - track_points[i-1]['ele']
        if elevation_change > 0:
            total_ascent += elevation_change
    return total_ascent

def calculate_gpx_density(gpx_track_points: List[Dict], total_distance_m: float) -> float:
    if total_distance_m == 0: return 0.0
    num_points = len(gpx_track_points)
    total_distance_km = total_distance_m / 1000.0
    if total_distance_km == 0: return 0.0
    return num_points / total_distance_km

def calculate_smoothing_window_size(gpx_density: float, target_smoothing_distance_m: float) -> int:
    if gpx_density <= 0: return 1
    target_smoothing_distance_km = target_smoothing_distance_m / 1000.0
    approx_points = int(gpx_density * target_smoothing_distance_km)
    window_size = max(3, approx_points)
    if window_size % 2 == 0:
        window_size += 1
    return window_size

def enrich_track_with_surface_profile(gpx_track_points: List[Dict], surface_profile: List[Tuple[float, float]]):
    if not gpx_track_points: return
    if not surface_profile:
        for p in gpx_track_points: p['crr'] = 0.005
        return

    current_crr = surface_profile[0][1]
    profile_idx = 0
    next_change_dist_m = float('inf')
    
    if len(surface_profile) > 1:
        next_change_dist_m = surface_profile[1][0] * 1000.0

    for p in gpx_track_points:
        dist_m = p['cumulativeDistanceM']
        
        while dist_m >= next_change_dist_m:
            profile_idx += 1
            if profile_idx < len(surface_profile):
                current_crr = surface_profile[profile_idx][1]
                if profile_idx + 1 < len(surface_profile):
                    next_change_dist_m = surface_profile[profile_idx + 1][0] * 1000.0
                else:
                    next_change_dist_m = float('inf')
            else:
                next_change_dist_m = float('inf')
                
        p['crr'] = current_crr