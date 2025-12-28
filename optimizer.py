#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 18:39:45 2025

@author: dyfanrhys
"""

### --- Project Module Imports --- ###
import physics_engine
import gpx_tool
import random
import time
from typing import Tuple, List, Dict

### --- Constants --- ###

SEGMENT_UPHILL_THRESHOLD_PERCENT = 2.5 
SEGMENT_DOWNHILL_THRESHOLD_PERCENT = -1.0 
MIN_SEGMENT_DISTANCE_M = 2000.0

GA_POPULATION_SIZE = 10 
GA_NUM_GENERATIONS = 10
GA_MUTATION_RATE = 0.3      
GA_CROSSOVER_RATE = 0.8     
GA_ELITISM_COUNT = 2        
GA_MUTATION_FTP_FACTOR = 0.1 

GREEDY_QUICK_ATTEMPTS = 100 
GREEDY_MODERATE_ATTEMPTS = 50000
GREEDY_POWER_SWAP_FTP_FACTOR = 0.01      
GREEDY_NO_IMPROVEMENT_THRESHOLD = 500    

MAX_POWER_FACTOR = 1.2             
MIN_POWER_FACTOR_DOWNHILL = 0.1    
MIN_POWER_FACTOR_NON_DOWNHILL = 0.4 
MAX_POWER_FACTOR_DOWNHILL = 0.7    
DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER = -1.5 

UPHILL_GRADIENT_THRESHOLD_RULE_BASED = 1.5 
DOWNHILL_GRADIENT_THRESHOLD_RULE_BASED = -1.5 
FLAT_POWER_FACTOR_RULE_BASED = 0.95 

UPHILL_GRADIENT_THRESHOLD = 1.0 
DOWNHILL_GRADIENT_THRESHOLD = DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER 

POWER_FACTOR_UPHILL_AGG = 1.2   
POWER_FACTOR_FLAT_AGG = 1.0      
POWER_FACTOR_DOWNHILL_AGG = 0.1 

POWER_FACTOR_UPHILL_MOD = 1.05   
POWER_FACTOR_FLAT_MOD = 0.95      
POWER_FACTOR_DOWNHILL_MOD = 0.7 

RANDOM_UPHILL_POWER_FACTOR_MIN = 1.05 
RANDOM_UPHILL_POWER_FACTOR_MAX = 1.2 
RANDOM_FLAT_POWER_FACTOR_MIN = 0.85 
RANDOM_FLAT_POWER_FACTOR_MAX = 1.05 
RANDOM_DOWNHILL_POWER_FACTOR_MIN = 0.10 
RANDOM_DOWNHILL_POWER_FACTOR_MAX = 0.50 

ZONE_THRESHOLDS = {
    "Zone 7 (Neuromuscular)": 1.51,  
    "Zone 6 (Anaerobic)": 1.21,      
    "Zone 5 (VO2 Max)": 1.06,        
    "Zone 4 (Threshold)": 0.90,        
    "Zone 3 (Tempo)": 0.76,          
    "Zone 2 (Endurance)": 0.56,        
    "Zone 1 (Active Recovery)": 0.00,  
}

def get_power_zone(power_watts: float, rider_ftp_watts: float) -> str:
    if rider_ftp_watts <= 0:
        return "N/A (FTP not defined)"
    power_factor = power_watts / rider_ftp_watts
    sorted_zones = sorted(ZONE_THRESHOLDS.items(), key=lambda item: item[1], reverse=True)
    for zone_name, threshold in sorted_zones:
        if power_factor >= threshold:
            return zone_name
    return "Zone 1 (Active Recovery)" 

def normalize_power_profile_time_weighted(
    power_profile: List[float],
    target_avg_power: float,
    segment_times: List[float],
    max_power_overall: float,
    segment_gradients_avg: List[float],
    rider_ftp_watts: float
) -> List[float]:
    
    # 1. Safety Checks
    if len(segment_times) < len(power_profile):
        diff = len(power_profile) - len(segment_times)
        segment_times = list(segment_times) + [1.0] * diff
    if len(segment_gradients_avg) < len(power_profile):
        diff = len(power_profile) - len(segment_gradients_avg)
        segment_gradients_avg = list(segment_gradients_avg) + [0.0] * diff

    if not power_profile or sum(segment_times) < 1e-6:
        return power_profile 

    profile = list(power_profile)
    total_time = sum(segment_times)
    
    # --- NEW: DEFINE DYNAMIC PHYSIOLOGICAL LIMITS ---
    min_powers = []
    max_powers = []
    
    # Heuristic W' (Anaerobic Capacity) for the optimizer constraints.
    # We use a generous 25,000J here to allow aggressive strategies, 
    # but the physics engine (with the real W') will have the final say.
    SAFE_W_PRIME_J = 25000.0 
    
    for i, g in enumerate(segment_gradients_avg):
        duration_s = max(1.0, segment_times[i])
        
        # 1. Calculate Physiological Ceiling (Critical Power Curve)
        # Formula: Max Sustainable Power = FTP + (W' / Duration)
        # Example: 1 min climb -> FTP + (25000/60) = FTP + 416W (Sprint is OK)
        # Example: 20 min climb -> FTP + (25000/1200) = FTP + 20W (Must be close to Threshold)
        physio_limit_watts = rider_ftp_watts + (SAFE_W_PRIME_J / duration_s)
        
        # 2. Determine Max Power for this segment
        # It cannot exceed the overall max, AND it cannot exceed the physio limit for this duration
        segment_max = min(max_power_overall, physio_limit_watts)

        if g < DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER:
            # DOWNHILL
            min_powers.append(0.0) 
            max_powers.append(rider_ftp_watts * MAX_POWER_FACTOR_DOWNHILL)
        else:
            # UPHILL/FLAT
            min_powers.append(rider_ftp_watts * MIN_POWER_FACTOR_NON_DOWNHILL)
            max_powers.append(segment_max) # <--- Apply the new dynamic limit

    # 2. Iterative Normalization
    for _ in range(20):  
        # Clamp first
        profile = [max(min_p, min(max_p, p)) for p, min_p, max_p in zip(profile, min_powers, max_powers)]
        
        current_energy = sum(p * t for p, t in zip(profile, segment_times))
        target_energy = target_avg_power * total_time
        energy_error = target_energy - current_energy
        
        if abs(energy_error / total_time) < 0.5: 
            break
            
        if energy_error > 0:  
            # --- NEED TO ADD POWER: Prioritize CLIMBS ---
            adjustable_indices = [i for i, p in enumerate(profile) if p < max_powers[i]]
            if not adjustable_indices: break
            
            # Sort by gradient descending (Steepest First)
            adjustable_indices.sort(key=lambda i: segment_gradients_avg[i], reverse=True)
            
            remaining_energy_needed = energy_error
            for idx in adjustable_indices:
                if remaining_energy_needed <= 0.01: break
                
                # We are now limited by 'max_powers[idx]', which includes the duration cap
                room_to_grow_watts = max_powers[idx] - profile[idx]
                energy_capacity = room_to_grow_watts * segment_times[idx]
                
                energy_to_add = min(remaining_energy_needed, energy_capacity)
                power_boost = energy_to_add / segment_times[idx]
                
                profile[idx] += power_boost
                remaining_energy_needed -= energy_to_add

        elif energy_error < 0:  
            # --- NEED TO REMOVE POWER: Prioritize DESCENTS ---
            adjustable_indices = [i for i, p in enumerate(profile) if p > min_powers[i]]
            if not adjustable_indices: break
            
            # Sort by gradient ascending (Steepest Downhill First)
            adjustable_indices.sort(key=lambda i: segment_gradients_avg[i])
            
            energy_to_remove_total = abs(energy_error)
            for idx in adjustable_indices:
                if energy_to_remove_total <= 0.01: break
                
                room_to_cut_watts = profile[idx] - min_powers[idx]
                energy_capacity = room_to_cut_watts * segment_times[idx]
                
                energy_to_remove = min(energy_to_remove_total, energy_capacity)
                power_cut = energy_to_remove / segment_times[idx]
                
                profile[idx] -= power_cut
                energy_to_remove_total -= energy_to_remove

    # Final Clamp
    profile = [max(min_p, min(max_p, p)) for p, min_p, max_p in zip(profile, min_powers, max_powers)]
    return profile


def initialize_population(num_segments: int, rider_ftp_watts: float, population_size: int, macro_segment_gradients: List[float]) -> List[List[float]]:
    population = []
    population.append([rider_ftp_watts] * num_segments)
    aggressive_profile = []
    for avg_gradient in macro_segment_gradients:
        if avg_gradient > UPHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_UPHILL_AGG
        elif avg_gradient < DOWNHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_DOWNHILL_AGG
        else:
            power = rider_ftp_watts * FLAT_POWER_FACTOR_RULE_BASED 
        aggressive_profile.append(power)
    population.append(aggressive_profile)
    moderate_profile = []
    for avg_gradient in macro_segment_gradients:
        if avg_gradient > UPHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_UPHILL_MOD
        elif avg_gradient < DOWNHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_DOWNHILL_MOD
        else:
            power = rider_ftp_watts * FLAT_POWER_FACTOR_RULE_BASED 
        moderate_profile.append(power)
    population.append(moderate_profile)
    while len(population) < population_size:
        biased_random_profile = []
        for avg_gradient in macro_segment_gradients:
            if avg_gradient > UPHILL_GRADIENT_THRESHOLD_RULE_BASED:
                power_factor = random.uniform(RANDOM_UPHILL_POWER_FACTOR_MIN, RANDOM_UPHILL_POWER_FACTOR_MAX)
            elif avg_gradient < DOWNHILL_GRADIENT_THRESHOLD_RULE_BASED: 
                power_factor = random.uniform(RANDOM_DOWNHILL_POWER_FACTOR_MIN, RANDOM_DOWNHILL_POWER_FACTOR_MAX)
            else:
                power_factor = random.uniform(RANDOM_FLAT_POWER_FACTOR_MIN, RANDOM_FLAT_POWER_FACTOR_MAX)
            biased_random_profile.append(rider_ftp_watts * power_factor)
        population.append(biased_random_profile)
    return population

def mutate(profile: List[float], mutation_rate: float, perturbation_step: float, min_val: float, max_val: float) -> List[float]:
    mutated_profile = list(profile)
    for i in range(len(mutated_profile)):
        if random.random() < mutation_rate:
            change = random.uniform(-perturbation_step, perturbation_step)
            mutated_profile[i] = max(min_val, min(max_val, mutated_profile[i] + change))
    return mutated_profile

def crossover(parent1: List[float], parent2: List[float], crossover_rate: float) -> Tuple[List[float], List[float]]:
    # [FIX] Safety check: If there's only 1 segment, we can't split it. Return parents as-is.
    if len(parent1) < 2:
        return list(parent1), list(parent2)

    if random.random() < crossover_rate:
        # random.randint(a, b) includes both end points.
        # We need at least 1 segment on either side of the cut.
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    return list(parent1), list(parent2)

def create_full_power_profile_from_macro(optimized_macro_power_profile: List[float], gpx_track_points: List[Dict], gpx_to_opt_segment_map: List[int]) -> List[float]:
    full_power_profile = []
    for gpx_idx in range(len(gpx_track_points)):
        macro_segment_idx = gpx_to_opt_segment_map[gpx_idx]
        power = optimized_macro_power_profile[min(macro_segment_idx, len(optimized_macro_power_profile) - 1)]
        full_power_profile.append(power)
    return full_power_profile

def get_segment_times_from_log(simulation_log: List[Dict], num_segments: int) -> List[float]:
    segment_times = [0.0] * num_segments
    if not simulation_log:
        return segment_times
    grouped_log_entries = [[] for _ in range(num_segments)]
    for entry in simulation_log:
        macro_idx = entry.get('macro_segment_index')
        if macro_idx is not None and 0 <= macro_idx < num_segments:
            grouped_log_entries[macro_idx].append(entry)
    for i in range(num_segments):
        segment_entries = grouped_log_entries[i]
        if segment_entries:
            start_time = segment_entries[0]['time']
            end_time = segment_entries[-1]['time']
            segment_times[i] = end_time - start_time
    return segment_times

def optimize_pacing_ga(
    rider_params: Dict, sim_params: Dict, course_data: Dict, ga_params: Dict, num_segments: int, progress_bar=None
) -> Tuple[float, List[float], List[int], List[float], List[float], List[Dict]]: 
    gpx_track_points = course_data['gpx_track_points']
    total_course_distance_m = course_data['total_course_distance_m']
    rider_target_power_watts = rider_params['rider_target_power_watts']
    rider_ftp = rider_params['rider_ftp_watts']
    
    if not gpx_track_points:
        return float('inf'), [], [], [], [], []

    if 'gpx_to_opt_segment_map' in course_data and 'avg_macro_segment_gradients' in course_data:
        gpx_to_opt_segment_map = course_data['gpx_to_opt_segment_map']
        avg_macro_segment_gradients = course_data['avg_macro_segment_gradients']
    else:
        gpx_to_opt_segment_map, avg_macro_segment_gradients, _ = gpx_tool.get_macro_segment_data(
            gpx_track_points, total_course_distance_m, rider_params=rider_params
        )
        
    max_power_watts_overall = rider_ftp * MAX_POWER_FACTOR
    population = initialize_population(
        num_segments, rider_target_power_watts, ga_params['population_size'], 
        avg_macro_segment_gradients
    )
    
    best_overall_time = float('inf')
    best_overall_profile = []
    final_simulation_log = []

    sim_rider_params = rider_params.copy()
    sim_rider_params.pop('rider_target_power_watts', None)
    sim_rider_params.pop('rider_ftp_watts', None)
    
    # [CHANGE 3] Extract w_prime for simulation
    rider_w_prime_val = sim_rider_params.pop('w_prime_capacity_j', 14000.0) 
    
    mutation_step_watts = rider_target_power_watts * ga_params['mutation_ftp_factor']

    print("--- Starting Pacing Optimisation (Genetic Algorithm with %d generations) ---" % ga_params['num_generations'])
    print(f"  Population Size: {ga_params['population_size']}, Generations: {ga_params['num_generations']}, Mutation Rate (Prob): {ga_params['mutation_rate']*100:.1f}%")
    
    start_time_gen1 = time.time()
    for generation in range(ga_params['num_generations']):
        if progress_bar:
            progress = (generation) / ga_params['num_generations']
            progress_bar.progress(progress, text=f"Evolution in progress: Generation {generation + 1}/{ga_params['num_generations']}")
        
        fitness_scores = []
        for profile in population:
            current_profile = list(profile)
            segment_times = [1] * len(current_profile)
            
            for _ in range(2): 
                normalized_profile = normalize_power_profile_time_weighted(
                    current_profile, rider_target_power_watts, segment_times, max_power_watts_overall,
                    avg_macro_segment_gradients, rider_target_power_watts
                )
                
                
                # --- CHANGED: Passed Macro Profile directly to fix jitter ---
                sim_params_safe = sim_params.copy()
                keys_to_remove = ['gpx_filename', 'ga_population_size', 'ga_num_generations', 'ga_mutation_rate', 'ga_crossover_rate', 'ga_elitism_count', 'ga_mutation_ftp_factor']
                for key in keys_to_remove:
                    sim_params_safe.pop(key, None)
                
                time_taken, avg_power, sim_log, min_w_prime = physics_engine.simulate_course(
                    **sim_rider_params, **sim_params_safe,
                    gpx_track_points=gpx_track_points, total_course_distance_m=total_course_distance_m,
                    power_profile_watts=normalized_profile, # <--- KEY CHANGE: Use normalized_profile directly
                    gpx_to_opt_segment_map=gpx_to_opt_segment_map,
                    report_progress=False,
                    rider_ftp=rider_ftp,
                    rider_w_prime=rider_w_prime_val 
                )
                
                
                if time_taken == float('inf'): break
                segment_times = get_segment_times_from_log(sim_log, num_segments)
                current_profile = normalized_profile
                
            fitness_scores.append((time_taken, sim_log, current_profile))

            if time_taken < best_overall_time:
                best_overall_time = time_taken
                best_overall_profile = current_profile
                final_simulation_log = sim_log

        sorted_population_data = sorted(zip(fitness_scores, population), key=lambda pair: pair[0][0])
        sorted_population = [raw_profile for (fitness_tuple, raw_profile) in sorted_population_data]
        new_population = [list(p) for p in sorted_population[:ga_params['elitism_count']]]

        while len(new_population) < ga_params['population_size']:
            parent1, parent2 = random.sample(sorted_population[:len(sorted_population)//2], 2)
            child1, child2 = crossover(parent1, parent2, ga_params['crossover_rate'])
            child1 = mutate(child1, ga_params['mutation_rate'], mutation_step_watts, 0, max_power_watts_overall * 1.2)
            child2 = mutate(child2, ga_params['mutation_rate'], mutation_step_watts, 0, max_power_watts_overall * 1.2)
            new_population.append(child1)
            if len(new_population) < ga_params['population_size']:
                new_population.append(child2)
        
        population = new_population
        best_min, best_sec = physics_engine.format_time_hms(best_overall_time)[1:]
        print(f"  Generation {generation + 1}/{ga_params['num_generations']}: Current Best Time {best_min}m {best_sec}s")
        
        if generation == 0:
            time_gen1 = time.time() - start_time_gen1
            estimated_total_time = time_gen1 * ga_params['num_generations']
            est_min = int(estimated_total_time // 60)
            est_sec = int(estimated_total_time % 60)
            print(f"  Estimated total run time: {est_min}m {est_sec}s")

    print("--- Optimisation Complete ---")
    final_optimized_time_hms = physics_engine.format_time_hms(best_overall_time)
    print(f"Optimised Time: {final_optimized_time_hms[1]}m {final_optimized_time_hms[2]}s")

    if not best_overall_profile:
        print("Warning: Genetic Algorithm failed to find a valid optimised profile. Returning a constant power profile as fallback.")
        best_overall_profile = [rider_target_power_watts] * num_segments
    
    optimized_power_profile_full_gpx = create_full_power_profile_from_macro(best_overall_profile, gpx_track_points, gpx_to_opt_segment_map)

    if progress_bar:
        progress_bar.progress(1.0, text="Optimisation Complete!")

    return best_overall_time, optimized_power_profile_full_gpx, gpx_to_opt_segment_map, best_overall_profile, avg_macro_segment_gradients, final_simulation_log

def optimize_pacing_greedy_hill_climbing(
    rider_params: Dict, sim_params: Dict, course_data: Dict, greedy_params: Dict, num_segments: int, initial_profile_type: str
) -> Tuple[float, List[float], List[int], List[float], List[Dict]]: 
    gpx_track_points = course_data['gpx_track_points']
    total_course_distance_m = course_data['total_course_distance_m']
    rider_target_power_watts = rider_params['rider_ftp_watts']
    optimization_attempts = greedy_params['optimization_attempts']
    power_swap_ftp_factor = greedy_params['power_swap_ftp_factor']
    no_improvement_threshold = greedy_params['no_improvement_threshold']

    if not gpx_track_points or num_segments < 2:
        return float('inf'), [], [], [], []

    gpx_to_opt_segment_map, avg_macro_segment_gradients, _ = gpx_tool.get_macro_segment_data(
        gpx_track_points, total_course_distance_m, rider_params=rider_params
    )

    best_opt_power_profile = [rider_target_power_watts] * num_segments
    max_power_overall = rider_target_power_watts * MAX_POWER_FACTOR
    min_powers = [(rider_target_power_watts * MIN_POWER_FACTOR_DOWNHILL) if g < DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER else (rider_target_power_watts * MIN_POWER_FACTOR_NON_DOWNHILL) for g in avg_macro_segment_gradients]
    max_powers = [(rider_target_power_watts * MAX_POWER_FACTOR_DOWNHILL) if g < DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER else max_power_overall for g in avg_macro_segment_gradients]

    sim_rider_params = rider_params.copy()
    sim_rider_params.pop('rider_ftp_watts', None)
    
    # [CHANGE 3] Extract w_prime
    rider_w_prime_val = sim_rider_params.pop('w_prime_capacity_j', 14000.0)

    initial_full_power_profile = create_full_power_profile_from_macro(best_opt_power_profile, gpx_track_points, gpx_to_opt_segment_map)
    best_time, _, initial_sim_log, _ = physics_engine.simulate_course(
        **sim_rider_params, **sim_params,
        gpx_track_points=gpx_track_points, total_course_distance_m=total_course_distance_m,
        power_profile_watts=initial_full_power_profile,
        gpx_to_opt_segment_map=gpx_to_opt_segment_map,
        report_progress=False,
        rider_ftp=rider_target_power_watts,
        rider_w_prime=rider_w_prime_val # [CHANGE 3]
    )
    
    print(f"\n--- Starting Pacing Optimisation (Greedy Time-Weighted Swapping, {optimization_attempts} attempts) ---")
    final_simulation_log = initial_sim_log
    current_segment_times = get_segment_times_from_log(initial_sim_log, num_segments)
    no_improvement_count = 0
    power_change_from_target = rider_target_power_watts * power_swap_ftp_factor

    for i in range(optimization_attempts):
        try:
            donor_idx, receiver_idx = random.sample(range(num_segments), 2)
        except ValueError:
            break

        test_profile = list(best_opt_power_profile)
        if current_segment_times[donor_idx] <= 0 or current_segment_times[receiver_idx] <= 0:
            no_improvement_count += 1
            continue

        energy_change = power_change_from_target * current_segment_times[donor_idx]
        power_increase_for_receiver = energy_change / current_segment_times[receiver_idx]

        if (test_profile[donor_idx] - power_change_from_target >= min_powers[donor_idx] and
            test_profile[receiver_idx] + power_increase_for_receiver <= max_powers[receiver_idx]):
            
            test_profile[donor_idx] -= power_change_from_target
            test_profile[receiver_idx] += power_increase_for_receiver
            
            full_test_profile = create_full_power_profile_from_macro(test_profile, gpx_track_points, gpx_to_opt_segment_map)
            new_time, avg_power, new_sim_log, _ = physics_engine.simulate_course( 
                **sim_rider_params, **sim_params,
                gpx_track_points=gpx_track_points, total_course_distance_m=total_course_distance_m,
                power_profile_watts=full_test_profile,
                gpx_to_opt_segment_map=gpx_to_opt_segment_map,
                report_progress=False,
                rider_ftp=rider_target_power_watts,
                rider_w_prime=rider_w_prime_val # [CHANGE 3]
            )

            if new_time < best_time:
                best_time = new_time
                best_opt_power_profile = test_profile
                final_simulation_log = new_sim_log
                current_segment_times = get_segment_times_from_log(new_sim_log, num_segments)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
           no_improvement_count += 1

        if (i + 1) % 10 == 0:
            best_min, best_sec = physics_engine.format_time_hms(best_time)[1:]
            print(f"  Attempt {i + 1}/{optimization_attempts}: Current Best Time {best_min}m {best_sec}s")
        
        if no_improvement_count >= no_improvement_threshold:
            print(f"  No improvement in {no_improvement_threshold} attempts. Ending early.")
            break
            
    print("--- Optimisation Complete ---")
    final_optimized_time_hms = physics_engine.format_time_hms(best_time)
    print(f"Optimised Time: {final_optimized_time_hms[1]}m {final_optimized_time_hms[2]}s")
    optimized_power_profile_full_gpx = create_full_power_profile_from_macro(best_opt_power_profile, gpx_track_points, gpx_to_opt_segment_map)
    return best_time, optimized_power_profile_full_gpx, gpx_to_opt_segment_map, best_opt_power_profile, final_simulation_log