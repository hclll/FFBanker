import numpy as np
import pickle
from parser import Parser
from module import Instance, PlacementRow, FlipFlop, Gate, CellLibrary # Added Gate and CellLibrary
from cluster import perform_mean_shift_clustering
import math
from collections import defaultdict # Added defaultdict

def find_closest_row(y_coord: float, placement_rows: list[PlacementRow]) -> PlacementRow | None:
    """Finds the placement row closest to the given y-coordinate."""
    if not placement_rows:
        return None
    
    closest_row = min(placement_rows, key=lambda row: abs(row.start_y - y_coord))
    return closest_row

def calculate_placement(cluster_center_x: float, new_ff_width: int, row: PlacementRow) -> int | None:
    """Calculates a valid x-coordinate for the new flip-flop within the row."""
    
    # Snap the cluster center x to the nearest site start
    site_index = round((cluster_center_x - row.start_x) / row.site_width)
    potential_x = row.start_x + site_index * row.site_width
    
    # Ensure the potential x is within the row bounds
    potential_x = max(row.start_x, potential_x) 
    
    # Ensure the flip-flop fits within the row
    if potential_x + new_ff_width <= row.start_x + row.total_sites * row.site_width:
        return potential_x
    else:
        # Try placing at the end if it doesn't fit at the calculated spot
        # Correct calculation for fitting at the end:
        end_fit_site_index = row.total_sites - math.ceil(new_ff_width / row.site_width)
        end_fit_x = row.start_x + end_fit_site_index * row.site_width
        if end_fit_x >= row.start_x and end_fit_x + new_ff_width <= row.start_x + row.total_sites * row.site_width:
             return end_fit_x
        else:
            # Cannot fit even at the start/end
            print(f"Warning: Cannot fit FF with width {new_ff_width} in row starting at y={row.start_y} (Total width: {row.total_sites * row.site_width})")
            return None # Indicate placement failure


def cluster_and_merge_flip_flops(parser_obj: Parser):
    """
    Clusters 1-bit flip-flop instances and merges them into higher-bit flip-flops 
    aligned with placement rows.

    Args:
        parser_obj: A Parser object containing parsed design data.

    Returns:
        A tuple containing:
            - A new list of instances with merged flip-flops.
            - A dictionary mapping original 1-bit FF instance names to the 
              name of the merged FF instance they became part of.
    """
    old_ff_to_new_ff_map: dict[str, str] = {} # Initialize the mapping

    # 1. Filter 1-bit Flip-Flop Instances
    one_bit_ff_instances = []
    other_instances = []
    one_bit_ff_cell_types = {
        name: ff for name, ff in parser_obj.cell_library.flip_flops.items() if ff.bits == 1
    }
    
    if not one_bit_ff_cell_types:
        print("No 1-bit flip-flop types found in the library. Cannot merge.")
        return parser_obj.instances # Return original instances

    original_indices = {} # Keep track of original index for later removal
    for i, inst in enumerate(parser_obj.instances):
        if inst.cell_type in one_bit_ff_cell_types:
            one_bit_ff_instances.append(inst)
            original_indices[inst.name] = i
        else:
            other_instances.append(inst)

    if not one_bit_ff_instances:
        print("No 1-bit flip-flop instances found in the design.")
        return parser_obj.instances # Return original instances

    print(f"Found {len(one_bit_ff_instances)} 1-bit flip-flop instances to cluster.")

    # 2. Cluster Flip-Flops (using a temporary parser-like object for the function)
    class TempParser:
        def __init__(self, instances):
            self.instances = instances
            
    temp_parser = TempParser(one_bit_ff_instances)
    labels, cluster_centers, n_clusters, _ = perform_mean_shift_clustering(temp_parser)

    if labels is None or n_clusters <= 0:
        print("Clustering failed or found no clusters.")
        return parser_obj.instances

    # 3. Group Instances by Cluster
    clusters: dict[int, list[Instance]] = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(one_bit_ff_instances[i])

    print(f"Grouped instances into {n_clusters} clusters.")

    # 4. Merge Logic & 5. Update Instances
    new_instances = []
    merged_instance_names = set()
    instances_to_remove_indices = set()

    for cluster_label, cluster_instances in clusters.items():
        k = len(cluster_instances)
        if k <= 1: # Don't merge clusters of size 1
             continue 

        print(f"Processing cluster {cluster_label} with {k} instances.")

        # Check if a k-bit FF exists or find the largest smaller one
        exact_ff_type: FlipFlop | None = None
        exact_ff_name: str | None = None
        best_smaller_ff_type: FlipFlop | None = None
        best_smaller_ff_name: str | None = None
        max_smaller_bits = -1 # Initialize to track the largest bits < k

        for name, ff in parser_obj.cell_library.flip_flops.items():
            if ff.bits == k:
                exact_ff_type = ff
                exact_ff_name = name
                break # Found exact match, no need to look further
            elif ff.bits < k:
                if ff.bits > max_smaller_bits:
                    max_smaller_bits = ff.bits
                    best_smaller_ff_type = ff
                    best_smaller_ff_name = name

        target_ff_type: FlipFlop | None = None
        target_ff_name: str | None = None
        original_k = k # Store original k for potential adjustment message
        popped_instance_names = set() # Initialize empty set for popped instances
        # Create a copy of the cluster's instances to potentially modify
        current_cluster_instances = list(cluster_instances)

        if exact_ff_type and exact_ff_name:
            target_ff_type = exact_ff_type
            target_ff_name = exact_ff_name
            #print(f"  Found exact matching {k}-bit FF type: {target_ff_name}")
        elif best_smaller_ff_type and best_smaller_ff_name:
            target_ff_type = best_smaller_ff_type
            target_ff_name = best_smaller_ff_name
            target_bits = target_ff_type.bits
            #print(f"  No exact {original_k}-bit FF found. Using largest smaller FF: {target_ff_name} ({target_bits} bits).")

            # Adjust cluster: Remove furthest instances until len(current_cluster_instances) == target_bits
            if len(current_cluster_instances) > target_bits:
                num_to_remove = len(current_cluster_instances) - target_bits
                #print(f"  Adjusting cluster size from {original_k} to {target_bits}. Removing {num_to_remove} furthest instances.")

                cluster_center_x = cluster_centers[cluster_label][0]
                cluster_center_y = cluster_centers[cluster_label][1]

                # Calculate distances from cluster center
                distances = []
                for inst in current_cluster_instances:
                    dist = math.hypot(inst.x - cluster_center_x, inst.y - cluster_center_y)
                    distances.append((dist, inst))

                # Sort by distance descending (furthest first)
                distances.sort(key=lambda item: item[0], reverse=True)

                # Identify instances to pop (furthest) and keep (closest)
                popped_instances = distances[:num_to_remove]
                kept_instances_with_dist = distances[num_to_remove:]

                popped_instance_names = {inst.name for _, inst in popped_instances}
                # Update the list of instances for this cluster to only the kept ones
                current_cluster_instances = [inst for _, inst in kept_instances_with_dist]

                # Update k to reflect the new cluster size for placement logic
                k = len(current_cluster_instances)
                assert k == target_bits # Sanity check: k should now match the target FF bits

            else:
                 # This case might occur if k was already <= target_bits (e.g., if target_bits was the only smaller one)
                 #print(f"  Cluster size {original_k} is already <= target bits {target_bits}. No instances removed.")
                 k = len(current_cluster_instances) # Ensure k reflects the actual count

        # --- Placement Logic (using target_ff_type and potentially modified current_cluster_instances/k) ---
        if target_ff_type and target_ff_name:
            #print(f"  Attempting placement for {target_ff_type.bits}-bit FF {target_ff_name}")
            cluster_center_x = cluster_centers[cluster_label][0]
            cluster_center_y = cluster_centers[cluster_label][1]

            closest_row = find_closest_row(cluster_center_y, parser_obj.placement_rows)

            if closest_row:
                new_x = calculate_placement(cluster_center_x, target_ff_type.width, closest_row)

                if new_x is not None:
                    # Placement successful
                    new_y = closest_row.start_y
                    # Create new merged instance name using the target FF's bits, not the potentially reduced k
                    new_instance_name = f"merged_cluster_{cluster_label}_ff{target_ff_type.bits}"

                    new_merged_instance = Instance(new_instance_name, target_ff_name, new_x, new_y)
                    new_instances.append(new_merged_instance)
                    #print(f"  Created merged instance {new_instance_name} at ({new_x}, {new_y}) in row starting at y={closest_row.start_y}")

                    # Mark original instances that were *successfully merged* for removal
                    # and add them to the mapping.
                    # Use the potentially reduced current_cluster_instances list.
                    for inst in current_cluster_instances:
                        merged_instance_names.add(inst.name) # Keep track of names that formed the merge
                        if inst.name in original_indices:
                             instances_to_remove_indices.add(original_indices[inst.name])
                        # Add to the mapping
                        old_ff_to_new_ff_map[inst.name] = new_merged_instance.name

                    # Also mark instances that were *popped* due to size mismatch for removal,
                    # but only if the placement was successful.
                    for name in popped_instance_names:
                        if name in original_indices:
                             instances_to_remove_indices.add(original_indices[name])

                else:
                     # Placement failed
                     print(f"  Placement failed for cluster {cluster_label} (target FF: {target_ff_name}). Keeping original FFs from this cluster.")
                     # Do not mark any instances (merged or popped) from this cluster for removal if placement fails.
            else:
                # No suitable row found
                print(f"  Could not find suitable placement row for cluster {cluster_label}. Keeping original FFs.")
                # Do not mark any instances (merged or popped) from this cluster for removal if no row found.
        else:
            # No suitable FF found (neither exact nor smaller)
            print(f"  No suitable FF type found in library for cluster {cluster_label} (original size {original_k}). Keeping original FFs.")
            # Do not mark any instances from this cluster for removal if no suitable FF was found.

    # Create the final list of instances
    final_instances = []
    for i, inst in enumerate(parser_obj.instances):
        if i not in instances_to_remove_indices:
            final_instances.append(inst)
            
    final_instances.extend(new_instances) # Add the newly created merged instances

    print(f"Removed {len(instances_to_remove_indices)} original 1-bit FFs.")
    print(f"Added {len(new_instances)} merged FFs.")
    print(f"Final instance count: {len(final_instances)}")
    print(f"Found a total of {len(temp_parser.instances)} 1-bit ffs")
    print(f"Grouped them into {n_clusters} clusters.")
    print(f"Created mapping for {len(old_ff_to_new_ff_map)} original FFs to new merged FFs.")

    return final_instances, old_ff_to_new_ff_map

def create_site_instance_mappings(parser_obj: Parser) -> tuple[dict[tuple[int, int], list[str]], dict[str, list[tuple[int, int]]]]:
    """
    Creates mappings between placement sites and the instances occupying them.

    Args:
        parser_obj: A Parser object containing parsed design data including instances,
                    placement rows, and the cell library.

    Returns:
        A tuple containing two dictionaries:
        1. site_to_instances: Maps site coordinates (x, y) to a list of instance names occupying that site.
        2. instance_to_sites: Maps instance names to a list of site coordinates (x, y) occupied by that instance.
    """
    site_to_instances = defaultdict(list)
    instance_to_sites = defaultdict(list)
    cell_library = parser_obj.cell_library
    placement_rows_map = {row.start_y: row for row in parser_obj.placement_rows} # For quick row lookup by y

    for instance in parser_obj.instances:
        # Get instance width from cell library
        instance_width = 0
        if instance.cell_type in cell_library.flip_flops:
            instance_width = cell_library.flip_flops[instance.cell_type].width
        elif instance.cell_type in cell_library.gates:
            instance_width = cell_library.gates[instance.cell_type].width
        else:
            print(f"Warning: Cell type '{instance.cell_type}' for instance '{instance.name}' not found in library. Skipping mapping.")
            continue

        # Find the placement row the instance belongs to
        row = placement_rows_map.get(instance.y)
        if not row:
            # This might happen if instances are not perfectly aligned with row start_y
            # Or if the instance is outside defined rows (e.g., IO pads - though those aren't usually 'instances')
            # For now, we'll try finding the closest row, but ideally, instances should match a row.y
            closest_row = find_closest_row(instance.y, parser_obj.placement_rows)
            # Heuristic: If it's very close to the row's y, assume it belongs there.
            # Adjust tolerance as needed. A small tolerance accounts for potential float issues.
            if closest_row and abs(closest_row.start_y - instance.y) < 1e-6 : # Tolerance for float comparison
                 row = closest_row
            else:
                print(f"Warning: Instance '{instance.name}' at y={instance.y} does not align with any placement row start_y. Skipping mapping.")
                continue

        # Validate instance placement and width against row/site properties
        if instance.x < row.start_x or instance.x + instance_width > row.start_x + row.total_sites * row.site_width:
            print(f"Warning: Instance '{instance.name}' ({instance.x}, {instance.y}, width={instance_width}) falls outside row bounds ({row.start_x} to {row.start_x + row.total_sites * row.site_width}). Skipping mapping.")
            continue

        if row.site_width <= 0:
             print(f"Warning: Row at y={row.start_y} has non-positive site_width ({row.site_width}). Skipping instance '{instance.name}'.")
             continue

        # Check if instance starts at a site boundary
        if abs((instance.x - row.start_x) % row.site_width) > 1e-6: # Use tolerance
            print(f"Warning: Instance '{instance.name}' at x={instance.x} does not start on a site boundary in row y={row.start_y} (site width: {row.site_width}). Skipping mapping.")
            continue

        # Check if instance width is a multiple of site width
        num_sites_float = instance_width / row.site_width
        if abs(num_sites_float - round(num_sites_float)) > 1e-6: # Use tolerance
            print(f"Warning: Instance '{instance.name}' width ({instance_width}) is not a multiple of site width ({row.site_width}) in row y={row.start_y}. Skipping mapping.")
            continue
        num_sites = round(num_sites_float) # Use rounded value after check

        if num_sites <= 0:
             print(f"Warning: Instance '{instance.name}' calculates to occupy zero or negative sites ({num_sites}). Skipping mapping.")
             continue


        # Calculate the sites occupied by this instance
        start_site_index = round((instance.x - row.start_x) / row.site_width) # Use round after boundary check

        for i in range(num_sites):
            current_site_index = start_site_index + i
            # Ensure the site index is within the row's valid range
            if 0 <= current_site_index < row.total_sites:
                site_x = row.start_x + current_site_index * row.site_width
                site_y = row.start_y
                site_coords = (site_x, site_y)

                # Add to site -> instances mapping
                site_to_instances[site_coords].append(instance.name)

                # Add to instance -> sites mapping
                instance_to_sites[instance.name].append(site_coords)
            else:
                 print(f"Warning: Calculated site index {current_site_index} for instance '{instance.name}' is outside the valid range [0, {row.total_sites - 1}] for row y={row.start_y}. Site skipped.")


    return dict(site_to_instances), dict(instance_to_sites) # Convert back to regular dicts


def print_overlaps(site_to_instances: dict[tuple[int, int], list[str]], verbose=False):
    """
    Prints details of any placement sites occupied by more than one instance.

    Args:
        site_to_instances: A dictionary mapping site coordinates (x, y)
                           to a list of instance names occupying that site.
    """
    print("\n--- Checking for Overlapping Instances ---")
    overlap_found = False
    num_overlaps = 0
    for site, instances in site_to_instances.items():
        if len(instances) > 1:
            if verbose:
                print(f"  Overlap detected at site {site}: Instances {instances}")
            num_overlaps += 1
            overlap_found = True

    if not overlap_found:
        print("  No overlaps detected.")
    print(f"Found {num_overlaps} overlaps")
    return overlap_found


def find_adjacent_empty_site(instance_name: str, instance_width: int, current_site: tuple[int, int], row: PlacementRow, site_to_instances: dict[tuple[int, int], list[str]], max_search_dist: int = 100) -> tuple[int, int] | None:
    """
    Searches for the nearest sequence of empty sites adjacent (left or right)
    to the current_site that can accommodate the instance width.

    Args:
        instance_name: Name of the instance to move (used for logging).
        instance_width: Width of the instance in coordinate units.
        current_site: The starting site (x, y) of the overlap.
        row: The placement row the instance is in.
        site_to_instances: The current mapping of sites to instances.
        max_search_dist: Max number of sites to check left/right.

    Returns:
        The coordinates (x, y) of the new starting site if found, otherwise None.
    """
    if row.site_width <= 0: return None # Cannot calculate sites
    num_sites_needed = math.ceil(instance_width / row.site_width)
    if num_sites_needed <= 0: return None # Invalid width

    start_x, start_y = current_site

    # Search Right then Left
    for direction in [1, -1]: # 1 for right, -1 for left 
        for i in range(1, max_search_dist + 1): # TODO probably should switch to distance then direction
            potential_start_site_x = start_x + direction * i * row.site_width
            potential_site_coords = (potential_start_site_x, start_y)

            # Check if potential start site is within row bounds
            if not (row.start_x <= potential_start_site_x < row.start_x + row.total_sites * row.site_width):
                is_clear = False
                break # Went off the end of the row in this direction

            # Check if the required number of sites starting from here are empty and within bounds
            is_clear = True
            sites_to_occupy = []
            for j in range(num_sites_needed):
                check_site_x = potential_start_site_x + j * row.site_width
                check_site_coords = (check_site_x, start_y)

                # Check bounds for this specific site
                if not (row.start_x <= check_site_x < row.start_x + row.total_sites * row.site_width):
                    is_clear = False
                    break # Instance would go out of bounds

                # Check occupancy
                if check_site_coords in site_to_instances and site_to_instances[check_site_coords]:
                    # Allow overlap only if it's the instance we are trying to move (relevant if moving left)
                    assert len(site_to_instances[check_site_coords]) != 0
                    if not (len(site_to_instances[check_site_coords]) == 1 and site_to_instances[check_site_coords][0] == instance_name):
                        is_clear = False
                        break # Site is occupied by another instance

                sites_to_occupy.append(check_site_coords) # Keep track for potential update

            if is_clear:
                #print(f"    Found empty spot for {instance_name} at {potential_site_coords}")
                return potential_site_coords # Found a suitable empty spot
            

        # If searching right failed, the loop continues to search left (direction = -1)

    #print(f"    Could not find adjacent empty spot for {instance_name} near {current_site}")
    return None


def resolve_overlaps(parser_obj: Parser, max_iterations: int = 10) -> bool:
    """
    Attempts to resolve instance overlaps by nudging instances to adjacent empty sites.

    Args:
        parser_obj: The Parser object with instances to check and modify.
        max_iterations: Maximum attempts to resolve overlaps.

    Returns:
        True if all overlaps were resolved, False otherwise.
    """
    print("\n--- Resolving Overlapping Instances ---")
    cell_library = parser_obj.cell_library
    placement_rows_map = {row.start_y: row for row in parser_obj.placement_rows}
    instances_dict = {inst.name: inst for inst in parser_obj.instances} # For quick lookup

    for iteration in range(max_iterations):
        print(f"  Overlap Resolution Iteration {iteration + 1}/{max_iterations}")
        site_to_instances, instance_to_sites = create_site_instance_mappings(parser_obj)

        overlaps_found_this_iter = False
        sites_with_overlaps = set()
        for site, instances in site_to_instances.items():
            if len(instances) > 1:
                overlaps_found_this_iter = True
                sites_with_overlaps.add(site)
                #print(f"    Overlap detected at site {site}: Instances {instances}")

        if not overlaps_found_this_iter:
            print("  No overlaps detected in this iteration. Resolution complete.")
            return True # Success

        # Attempt to resolve overlaps found in this iteration
        moved_count = 0
        failed_moves = 0
        for site in sites_with_overlaps:
            if site not in site_to_instances or len(site_to_instances[site]) <= 1:
                 continue # Overlap might have been resolved by a previous move in this iteration

            overlapping_instances = site_to_instances[site]
            #print(f"    Attempting to resolve overlap at {site} for {overlapping_instances}")

            for instance_to_move_name in overlapping_instances:
            # Try moving the second instance in the list (simple strategy)
            #instance_to_move_name = overlapping_instances[1]
                instance_to_move = instances_dict.get(instance_to_move_name)
                if instance_to_move.cell_type in cell_library.flip_flops:
                    if not instance_to_move:
                        print(f"    Error: Instance '{instance_to_move_name}' not found in instances_dict. Skipping.")
                        continue

                    # Get instance width
                    instance_width = 0
                    if instance_to_move.cell_type in cell_library.flip_flops:
                        instance_width = cell_library.flip_flops[instance_to_move.cell_type].width
                    elif instance_to_move.cell_type in cell_library.gates:
                        assert False, "We should not move gates"
                    else:
                        print(f"    Warning: Cell type '{instance_to_move.cell_type}' for instance '{instance_to_move_name}' not found. Cannot determine width. Skipping move.")
                        continue

                    # Find the row
                    row = placement_rows_map.get(site[1]) # site[1] is the y-coordinate
                    if not row:
                        # This shouldn't happen if create_site_instance_mappings worked, but check anyway
                        print(f"    Warning: Could not find placement row for site {site}. Skipping move for {instance_to_move_name}.")
                        continue

                    # Find an adjacent empty site
                    new_site_coords = find_adjacent_empty_site(instance_to_move_name, instance_width, site, row, site_to_instances, max_search_dist=10**(iteration+1))

                    if new_site_coords:
                        new_x, new_y = new_site_coords
                        assert new_x != instance_to_move.x
                        #print(f"    Moving instance '{instance_to_move_name}' from ({instance_to_move.x}, {instance_to_move.y}) to ({new_x}, {new_y})")

                        # Update the instance object's coordinates directly
                        instance_to_move.x = new_x
                        instance_to_move.y = new_y # Y should remain the same (same row)

                        # Update site_to_instances incrementally for the next checks in *this* iteration
                        # 1. Remove instance from all its old sites
                        old_sites = instance_to_sites.get(instance_to_move_name, [])
                        #print("old_sites", old_sites)
                        for old_site in old_sites:
                            #print(site_to_instances[old_site])
                            if old_site in site_to_instances and instance_to_move_name in site_to_instances[old_site]:
                                site_to_instances[old_site].remove(instance_to_move_name)
                                # If list becomes empty, optionally remove the key: del site_to_instances[old_site]

                        # 2. Add instance to its new sites
                        num_sites_needed = math.ceil(instance_width / row.site_width)
                        for i in range(num_sites_needed):
                            current_site_x = new_x + i * row.site_width
                            current_site_coords = (current_site_x, new_y)
                            if current_site_coords not in site_to_instances:
                                site_to_instances[current_site_coords] = []
                            # Avoid adding duplicates if somehow already there
                            if instance_to_move_name not in site_to_instances[current_site_coords]:
                                site_to_instances[current_site_coords].append(instance_to_move_name)

                        # Update instance_to_sites for the moved instance (will be fully rebuilt next iteration)
                        instance_to_sites[instance_to_move_name] = [ (new_x + i * row.site_width, new_y) for i in range(num_sites_needed)]

                        moved_count += 1
                        break
                    else:
                        #print(f"    Failed to find a new location for '{instance_to_move_name}' near site {site}.")
                        failed_moves += 1
                        # If we fail to move, the overlap persists for the next iteration (or final failure)

        print(f"  Iteration {iteration + 1} summary: Moved {moved_count} instances, failed to move {failed_moves} instances involved in overlaps.")
        if moved_count == 0 and overlaps_found_this_iter:
            print(f"  Could not resolve remaining overlaps after iteration {iteration + 1}.")
            break # No progress made, exit loop


    # Final check after loop
    final_site_map, _ = create_site_instance_mappings(parser_obj)
    final_overlap_found = print_overlaps(final_site_map)

    if final_overlap_found:
        print("--- Overlap resolution failed: Overlaps still exist after maximum iterations. ---")
        return False
    else:
        print("--- Overlap resolution successful: No overlaps remain. ---")
        return True


def create_pin_mapping(original_instances: list[Instance], final_instances: list[Instance], old_ff_to_new_ff_map: dict[str, str], cell_library: CellLibrary) -> dict[str, str]:
    """
    Creates a mapping from old instance pin names to new instance pin names
    after flip-flop merging.

    Args:
        original_instances: The list of instances before merging.
        final_instances: The list of instances after merging.
        old_ff_to_new_ff_map: Dictionary mapping old 1-bit FF names to new merged FF names.
        cell_library: The CellLibrary object containing cell definitions.

    Returns:
        A dictionary mapping old full pin names (e.g., "old_ff/D") to
        new full pin names (e.g., "merged_ff/D0").
    """
    pin_mapping: dict[str, str] = {}
    final_instances_map = {inst.name: inst for inst in final_instances}
    original_instances_map = {inst.name: inst for inst in original_instances}

    # 1. Create reverse map: new_ff_name -> [old_ff_name1, old_ff_name2, ...]
    new_ff_to_old_ffs_map = defaultdict(list)
    for old_name, new_name in old_ff_to_new_ff_map.items():
        if old_name in original_instances_map: # Ensure the old FF actually exists
             new_ff_to_old_ffs_map[new_name].append(old_name)
        else:
             print(f"Warning: Old FF '{old_name}' from mapping not found in original instances. Skipping.")


    # 2. Process merged flip-flops
    for new_ff_name, old_ff_names in new_ff_to_old_ffs_map.items():
        if not old_ff_names: continue # Skip if list is empty

        if new_ff_name not in final_instances_map:
            print(f"Warning: New FF '{new_ff_name}' from mapping not found in final instances. Skipping mapping for its constituents.")
            continue

        new_instance = final_instances_map[new_ff_name]
        try:
            new_ff_library_cell = cell_library.flip_flops[new_instance.cell_type]
        except KeyError:
            print(f"Warning: Cell type '{new_instance.cell_type}' for merged FF '{new_ff_name}' not found in library. Skipping mapping.")
            continue

        # Get original instance objects and sort them by coordinate (Y then X)
        old_ff_instances = []
        for name in old_ff_names:
             if name in original_instances_map:
                 old_ff_instances.append(original_instances_map[name])
             # else: already warned above

        if not old_ff_instances: continue # Skip if no valid old instances found

        # Sort primarily by Y, secondarily by X to assign bit index consistently
        old_ff_instances.sort(key=lambda inst: (inst.original_y, inst.original_x))

        # Map pins for each original FF based on its sorted position (bit_index)
        for bit_index, old_instance in enumerate(old_ff_instances):
            try:
                old_ff_library_cell = cell_library.flip_flops[old_instance.cell_type]
            except KeyError:
                print(f"Warning: Cell type '{old_instance.cell_type}' for original FF '{old_instance.name}' not found in library. Skipping its pin mapping.")
                continue

            for old_pin_name in old_ff_library_cell.pins.keys():
                # Determine the corresponding new pin name convention
                # Common conventions: D->D0, Q->Q0, CLK->CLK, RST->RST etc.
                # This might need adjustment based on actual library conventions
                new_pin_name = ""
                is_shared_pin = old_pin_name.upper() in ["CLK", "RST", "SET", "CLR"] # Common shared pins

                if is_shared_pin:
                    new_pin_name = old_pin_name # Assume shared pins keep their name
                elif old_pin_name.upper() == 'D':
                    new_pin_name = f"D{bit_index}"
                elif old_pin_name.upper() == 'Q':
                     new_pin_name = f"Q{bit_index}"
                # Add more rules here if other pin naming conventions exist (e.g., TI, SI, SO for scan chains)
                else:
                    # Fallback for potentially indexed pins we don't explicitly handle
                    # Check if the old pin name itself suggests an index (less common for 1-bit)
                    # Or assume it might be indexed in the new FF
                    new_pin_name = f"{old_pin_name}{bit_index}" # Guessing convention
                    #print(f"Debug: Assuming indexed pin mapping for '{old_pin_name}' -> '{new_pin_name}'")


                # Check if the determined new pin name actually exists in the new FF type
                if new_pin_name in new_ff_library_cell.pins:
                    old_full_pin = f"{old_instance.name}/{old_pin_name}"
                    new_full_pin = f"{new_instance.name}/{new_pin_name}"
                    pin_mapping[old_full_pin] = new_full_pin
                    #print(f"  Mapped: {old_full_pin} -> {new_full_pin}")
                else:
                    # If the indexed version didn't exist, maybe it's a shared pin we missed?
                    # Try mapping to the original name if that exists in the new FF.
                    if old_pin_name in new_ff_library_cell.pins:
                         old_full_pin = f"{old_instance.name}/{old_pin_name}"
                         new_full_pin = f"{new_instance.name}/{old_pin_name}" # Map to non-indexed version
                         pin_mapping[old_full_pin] = new_full_pin
                         #print(f"  Mapped (as shared): {old_full_pin} -> {new_full_pin}")
                    else:
                         print(f"Warning: Could not map pin '{old_pin_name}' of old FF '{old_instance.name}'. "
                               f"Neither '{new_pin_name}' nor '{old_pin_name}' found in new FF '{new_instance.name}' ({new_instance.cell_type}).")


    print(f"Created pin mapping for {len(pin_mapping)} pins.")
    return pin_mapping


if __name__ == "__main__":
    # Example Usage:
    file_path = "bm/testcase1_0812.txt" # Or bm/sampleCase
    #file_path = "bm/sampleCase" # Or bm/sampleCase
    print(f"Parsing file: {file_path}")
    parser = Parser(file_path)
    try:
        parsed_data = parser.parse()
        print("Parsing complete.")
        
        if not parsed_data.placement_rows:
             print("Warning: No placement rows found in the input file.")

        print("\n--- Initial State Summary ---")
        # Create initial mappings before merging/resolving
        initial_site_map, initial_instance_map = create_site_instance_mappings(parsed_data)
        print(f"Initial instance count: {len(parsed_data.instances)}")
        print(f"Initial occupied sites: {len(initial_site_map)}")

        # --- Cluster and Merge Flip-Flops ---
        print("\n--- Clustering and Merging Flip-Flops ---")
        updated_instances, old_to_new_map = cluster_and_merge_flip_flops(parsed_data)
        print(create_pin_mapping(parsed_data.instances, updated_instances, old_to_new_map, parsed_data.cell_library))
        parsed_data.instances = updated_instances # Update instances in the parsed_data object

        print("\n--- Old FF to New FF Mapping (Sample) ---")
        map_sample_count = 0
        for old_ff, new_ff in old_to_new_map.items():
            if map_sample_count >= 5: break
            print(f"  Original: {old_ff} -> Merged: {new_ff}")
            map_sample_count += 1
        if not old_to_new_map:
            print("  No FFs were merged, mapping is empty.")
        # -----------------------------------------

        # --- Save intermediate state (optional) ---
        #with open("temp2.pkl", 'wb') as f:
        #    pickle.dump(parsed_data, f)
        #with open("temp1.pkl", 'rb') as f:
        #    parsed_data =  pickle.load(f)

        # --- Create Site/Instance Mappings AFTER merging ---
        print("\nCreating site-instance mappings post-merge...")
        site_map, instance_map = create_site_instance_mappings(parsed_data)
        print(f"Generated mapping for {len(instance_map)} instances across {len(site_map)} occupied sites.")

        # Optional: Print some mapping details for verification
        # print("\n--- Example Mappings Post-Merge ---")
        # print("Site to Instances (first 5 occupied sites):")
        # for i, (site, inst_list) in enumerate(site_map.items()):
        #     if i >= 5: break
        #     print(f"  Site {site}: {inst_list}")
        # print("\nInstance to Sites (first 5 mapped instances):")
        # for i, (inst, site_list) in enumerate(instance_map.items()):
        #      if i >= 5: break
        #      print(f"  Instance {inst}: {site_list}")

        # --- Check for Overlaps AFTER merging ---
        print("\n--- Overlap Check Post-Merge ---")
        initial_overlap_found = print_overlaps(site_map)
        # -----------------------------------------

        # --- Resolve Overlaps ---
        if initial_overlap_found:
             resolve_overlaps(parsed_data) # Modify parsed_data.instances in-place
        else:
             print("\nNo overlaps to resolve post-merge.")
        # ------------------------

        # --- Final Check and Summary ---
        print("\n--- Final State Summary ---")
        # Re-create mappings to reflect final state after potential resolution
        final_site_map, final_instance_map = create_site_instance_mappings(parsed_data)
        print(f"Final instance count: {len(parsed_data.instances)}")
        print(f"Final occupied sites: {len(final_site_map)}")
        print("\n--- Final Overlap Check ---")
        print_overlaps(final_site_map)
        # -----------------------------

        # Further analysis or output generation could go here
        # E.g., write the updated instance list to a file

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
