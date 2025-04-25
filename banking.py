import numpy as np
import pickle
from parser import Parser
from module import Instance, PlacementRow, FlipFlop, Gate, CellLibrary # Added Gate and CellLibrary
from cluster import perform_mean_shift_clustering
import math
from math import ceil # Import ceil
from collections import defaultdict
from preprocessing import debanking_all, Die
from module import Instance # Ensure Instance is imported
import time

def find_closest_row(y_coord, placement_rows):
    """Finds the placement row closest to the given y-coordinate."""
    if not placement_rows:
        return None
    
    # Find the row whose start_y is less than or equal to y_coord
    candidate_rows = [r for r in placement_rows if r.start_y <= y_coord]
    if not candidate_rows:
        # If no row starts at or below y_coord, find the one with the minimum start_y (closest from above)
        return min(placement_rows, key=lambda row: row.start_y) if placement_rows else None

    # Among candidates, find the one whose start_y is closest to y_coord
    closest_row = min(candidate_rows, key=lambda row: y_coord - row.start_y)
    return closest_row

def calculate_placement(cluster_center_x, new_ff_width, new_ff_height, start_row, all_placement_rows):
    """
    Calculates a valid x-coordinate for the new flip-flop, potentially spanning multiple rows.

    Args:
        cluster_center_x: The target x-coordinate from clustering.
        new_ff_width: The width of the flip-flop to place.
        new_ff_height: The height of the flip-flop to place.
        start_row: The starting placement row determined by find_closest_row.
        all_placement_rows: A list or dict of all placement rows for checking vertical neighbors.

    Returns:
        The calculated x-coordinate if placement is possible, otherwise None.
        The start_row might be adjusted if the initial one doesn't work but a nearby one does.
        Returns a tuple (x_coordinate, actual_start_row) or (None, None)
    """
    if not start_row or start_row.site_height <= 0 or start_row.site_width <= 0:
        print(f"Warning: Invalid start_row provided or zero site dimensions.")
        return None, None

    num_rows_needed = ceil(new_ff_height / start_row.site_height) # Use math.ceil

    # Verify consecutive rows exist and are compatible
    rows_to_occupy = [start_row]
    placement_rows_map = {row.start_y: row for row in all_placement_rows} # Quick lookup

    for i in range(1, num_rows_needed):
        next_row_y = start_row.start_y + i * start_row.site_height
        next_row = placement_rows_map.get(next_row_y)
        if not next_row:
            print(f"Warning: Cannot find consecutive row at y={next_row_y} needed for FF height {new_ff_height}. Required {num_rows_needed} rows.")
            return None, None # Cannot span vertically

        # Check compatibility (same start_x, width, site_width, total_sites) - adjust checks as needed
        if not (next_row.start_x == start_row.start_x and \
                next_row.site_width == start_row.site_width and \
                next_row.total_sites == start_row.total_sites):
             print(f"Warning: Consecutive row at y={next_row_y} is incompatible with start row at y={start_row.start_y}.")
             return None, None # Incompatible rows

        rows_to_occupy.append(next_row)

    # All necessary rows found and are compatible. Use properties from the start_row for placement calculation.
    row = start_row # Use the validated start_row for calculations below

    # Snap the cluster center x to the nearest site start
    site_index = round((cluster_center_x - row.start_x) / row.site_width)
    potential_x = row.start_x + site_index * row.site_width
    
    # Ensure the potential x is within the row bounds
    potential_x = max(row.start_x, potential_x)

    # Ensure the flip-flop fits horizontally within the row
    if potential_x + new_ff_width <= row.start_x + row.total_sites * row.site_width:
        return potential_x, row # Return the calculated x and the validated start row
    else:
        # Try placing at the very end if it doesn't fit at the calculated spot
        # Ensure there's enough horizontal space even at the end
        num_sites_needed_width = ceil(new_ff_width / row.site_width)
        if num_sites_needed_width > row.total_sites:
             print(f"Warning: FF width {new_ff_width} ({num_sites_needed_width} sites) exceeds total row width ({row.total_sites} sites) in row starting at y={row.start_y}.")
             return None, None # Cannot fit horizontally at all

        end_fit_site_index = row.total_sites - num_sites_needed_width
        end_fit_x = row.start_x + end_fit_site_index * row.site_width

        # Double-check the end fit calculation ensures it's within bounds
        if end_fit_x >= row.start_x and (end_fit_x + new_ff_width) <= (row.start_x + row.total_sites * row.site_width + 1e-9): # Add tolerance
             return end_fit_x, row # Return end fit x and the validated start row
        else:
            # Cannot fit even at the start/end horizontally
            print(f"Warning: Cannot fit FF with width {new_ff_width} horizontally in row starting at y={row.start_y} (Total width: {row.total_sites * row.site_width}). Tried x={potential_x} and end_fit_x={end_fit_x}.")
            return None, None # Indicate placement failure


def cluster_and_merge_flip_flops(parser_obj, index):
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
    old_ff_to_new_ff_map = {} # Initialize the mapping

    # 1. Filter 1-bit Flip-Flop Instances
    one_bit_ff_instances = []
    other_instances = []
    one_bit_ff_cell_types = {
        name: ff for name, ff in parser_obj.cell_library.flip_flops.items() if ff.bits == 1
    }
    
    if not one_bit_ff_cell_types:
        print("No 1-bit flip-flop types found in the library. Cannot merge.")
        return parser_obj.die.instances # Return original instances

    original_indices = {} # Keep track of original index for later removal
    for i, inst in enumerate(parser_obj.die.instances.values()):
        if inst.cell_type in one_bit_ff_cell_types:
            one_bit_ff_instances.append(inst)
            original_indices[inst.name] = i
        else:
            other_instances.append(inst)

    if not one_bit_ff_instances:
        print("No 1-bit flip-flop instances found in the design.")
        return parser_obj.die.instances.values() # Return original instances

    print(f"Found {len(one_bit_ff_instances)} 1-bit flip-flop instances to cluster.")

    # 2. Cluster Flip-Flops (using a temporary parser-like object for the function)
    class TempParser:
        def __init__(self, instances):
            self.instances = instances
            
    temp_parser = TempParser(one_bit_ff_instances)
    
    labels, cluster_centers, n_clusters = perform_mean_shift_clustering(temp_parser)

    if labels is None or n_clusters <= 0:
        print("Clustering failed or found no clusters.")
        return parser_obj.die.instances.values()

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

        #print(f"Processing cluster {cluster_label} with {k} instances.")

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
            elif ff.bits < k: #TODO should cluster all into two flip flops if k is twice max ff.bits
                if ff.bits > max_smaller_bits:
                    max_smaller_bits = ff.bits
                    best_smaller_ff_type = ff
                    best_smaller_ff_name = name

        target_ff_type: FlipFlop | None = None
        target_ff_name: str | None = None
        original_k = k # Store original k for potential adjustment message
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
                kept_instances_with_dist = distances[num_to_remove:]

                # Update the list of instances for this cluster to only the kept ones
                current_cluster_instances = [inst for _, inst in kept_instances_with_dist]

                # Update k to reflect the new cluster size for placement logic
                k = len(current_cluster_instances)
                assert k == target_bits # Sanity check: k should now match the target FF bits

            else:
                 # This case might occur if k was already <= target_bits (e.g., if target_bits was the only smaller one)
                 #print(f"  Cluster size {original_k} is already <= target bits {target_bits}. No instances removed.")
                 assert False, "should not happen"

        # --- Placement Logic (using target_ff_type and potentially modified current_cluster_instances/k) ---
        if target_ff_type and target_ff_name:
            #print(f"  Attempting placement for {target_ff_type.bits}-bit FF {target_ff_name}")
            cluster_center_x = cluster_centers[cluster_label][0]
            cluster_center_y = cluster_centers[cluster_label][1]

            # Find the potential starting row based on the cluster's y-center
            potential_start_row = find_closest_row(cluster_center_y, parser_obj.placement_rows)

            if potential_start_row:
                # Attempt placement, passing all rows for multi-row checks
                new_x, actual_start_row = calculate_placement(
                    cluster_center_x,
                    target_ff_type.width,
                    target_ff_type.height,
                    potential_start_row,
                    parser_obj.placement_rows # Pass all rows
                )

                if new_x is not None and actual_start_row:
                    # Placement successful
                    new_y = actual_start_row.start_y # Use the actual start row's y
                    # Create new merged instance name using the target FF's bits
                    new_instance_name = f"{index}_merged_cluster_{cluster_label}_ff{target_ff_type.bits}"

                    new_merged_instance = Instance(new_instance_name, target_ff_name, new_x, new_y)
                    new_instances.append(new_merged_instance)
                    #print(f"  Created merged instance {new_instance_name} at ({new_x}, {new_y}) starting in row y={actual_start_row.start_y}")

                    # Mark original instances that were *successfully merged* for removal
                    # and add them to the mapping.
                    # Use the potentially reduced current_cluster_instances list.
                    for inst in current_cluster_instances:
                        merged_instance_names.add(inst.name) # Keep track of names that formed the merge
                        if inst.name in original_indices:
                             instances_to_remove_indices.add(original_indices[inst.name])
                        # Add to the mapping
                        old_ff_to_new_ff_map[inst.name] = new_merged_instance.name

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
    for i, inst in enumerate(parser_obj.die.instances.values()):
        if i not in instances_to_remove_indices:
            final_instances.append(inst)
            
    final_instances.extend(new_instances) # Add the newly created merged instances

    #print(f"Removed {len(instances_to_remove_indices)} original 1-bit FFs.")
    #print(f"Added {len(new_instances)} merged FFs.")
    #print(f"Final instance count: {len(final_instances)}")
    #print(f"Found a total of {len(temp_parser.instances)} 1-bit ffs")
    #print(f"Grouped them into {n_clusters} clusters.")
    #print(f"Created mapping for {len(old_ff_to_new_ff_map)} original FFs to new merged FFs.")

    #print(f"Clustering took {time.time()-start} seconds")
    return {i.name:i for i in final_instances}, old_ff_to_new_ff_map

def create_site_instance_mappings(parser_obj):
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

    for instance in parser_obj.die.instances.values():
        # Get instance dimensions from cell library
        instance_width = 0
        instance_height = 0
        if instance.cell_type in cell_library.flip_flops:
            cell = cell_library.flip_flops[instance.cell_type]
            instance_width = cell.width
            instance_height = cell.height
        elif instance.cell_type in cell_library.gates:
            cell = cell_library.gates[instance.cell_type]
            instance_width = cell.width
            instance_height = cell.height
        else:
            print(f"Warning: Cell type '{instance.cell_type}' for instance '{instance.name}' not found in library. Skipping mapping.")
            continue

        if instance_width <= 0 or instance_height <= 0:
             print(f"Warning: Instance '{instance.name}' has non-positive dimensions ({instance_width}x{instance_height}). Skipping mapping.")
             continue

        # Find the starting placement row the instance belongs to (based on its bottom-left y)
        start_row = placement_rows_map.get(instance.y)
        if not start_row:
            # Try finding the closest row if not perfectly aligned
            potential_start_row = find_closest_row(instance.y, parser_obj.placement_rows)
            # Use tolerance for float comparison
            if potential_start_row and abs(potential_start_row.start_y - instance.y) < 1e-6 :
                 start_row = potential_start_row
            else:
                print(f"Warning: Instance '{instance.name}' at y={instance.y} does not align with any placement row start_y. Skipping mapping.")
                continue

        # Basic validation using the start_row
        if start_row.site_width <= 0 or start_row.site_height <= 0:
             print(f"Warning: Start row at y={start_row.start_y} has non-positive site dimensions ({start_row.site_width}x{start_row.site_height}). Skipping instance '{instance.name}'.")
             continue

        # Check if instance starts at a site boundary horizontally
        if abs((instance.x - start_row.start_x) % start_row.site_width) > 1e-6: # Use tolerance
            print(f"Warning: Instance '{instance.name}' at x={instance.x} does not start on a site boundary in row y={start_row.start_y} (site width: {start_row.site_width}). Skipping mapping.")
            continue

        # Calculate number of sites needed horizontally and vertically
        num_sites_horiz = ceil(instance_width / start_row.site_width)
        num_rows_needed = ceil(instance_height / start_row.site_height)

        if num_sites_horiz <= 0 or num_rows_needed <= 0:
             print(f"Warning: Instance '{instance.name}' calculates to occupy zero or negative sites/rows ({num_sites_horiz}x{num_rows_needed}). Skipping mapping.")
             continue

        # Validate horizontal placement within the start row
        if instance.x < start_row.start_x or instance.x + instance_width > start_row.start_x + start_row.total_sites * start_row.site_width + 1e-9: # Tolerance
            print(f"Warning: Instance '{instance.name}' ({instance.x}, {instance.y}, width={instance_width}) falls outside horizontal bounds of start row y={start_row.start_y}. Skipping mapping.")
            continue

        # Calculate the starting site index horizontally
        start_site_index_x = round((instance.x - start_row.start_x) / start_row.site_width)

        # Iterate through all rows and sites occupied by the instance
        instance_occupied_sites = [] # Store sites for this instance
        valid_placement = True
        for row_offset in range(num_rows_needed):
            current_row_y = start_row.start_y + row_offset * start_row.site_height
            current_row = placement_rows_map.get(current_row_y)

            if not current_row:
                print(f"Warning: Instance '{instance.name}' requires row at y={current_row_y}, but it wasn't found. Skipping mapping.")
                valid_placement = False
                break # Cannot map this instance

            # Verify row compatibility (important if instance spans rows)
            if not (current_row.start_x == start_row.start_x and \
                    current_row.site_width == start_row.site_width and \
                    current_row.total_sites == start_row.total_sites):
                 print(f"Warning: Instance '{instance.name}' spans incompatible rows (y={start_row.start_y} and y={current_row_y}). Skipping mapping.")
                 valid_placement = False
                 break

            for site_offset_x in range(num_sites_horiz):
                current_site_index = start_site_index_x + site_offset_x
                # Ensure the site index is within the row's valid range
                if 0 <= current_site_index < current_row.total_sites:
                    site_x = current_row.start_x + current_site_index * current_row.site_width
                    site_y = current_row.start_y # Y coordinate of the current row
                    site_coords = (site_x, site_y)
                    instance_occupied_sites.append(site_coords)
                else:
                     print(f"Warning: Calculated site index {current_site_index} for instance '{instance.name}' is outside the valid range [0, {current_row.total_sites - 1}] for row y={current_row.start_y}. Site skipped, mapping incomplete.")
                     # Decide if partial mapping is okay or if the whole instance should be skipped
                     valid_placement = False # Mark as invalid if any site is out of bounds
                     break # Stop processing sites for this row

            if not valid_placement:
                break # Stop processing rows for this instance

        # If all sites were valid and within bounds, add mappings
        if valid_placement:
            instance_to_sites[instance.name] = instance_occupied_sites
            for site_coords in instance_occupied_sites:
                site_to_instances[site_coords].append(instance.name)


    return dict(site_to_instances), dict(instance_to_sites) # Convert back to regular dicts


def banking_each_clock_net(parser_obj: Parser):
    """
    Separates flip-flop instances by clock net and performs banking on each group.

    Args:
        parser_obj: The Parser object containing the design data.

    Returns:
        A tuple containing:
            - A dictionary of final instances (name -> Instance object).
            - A dictionary mapping original 1-bit FF instance names to the
              name of the merged FF instance they became part of.
    """
    print("\n--- Banking Flip-Flops Separately for Each Clock Net ---")

    # 1. Identify FF instances and their clock nets
    ffs_by_clock_net = defaultdict(list)
    ff_instances = {}
    other_instances = {}
    ff_cell_types = set(parser_obj.cell_library.flip_flops.keys())

    for inst_name, inst in parser_obj.die.instances.items():
        if inst.cell_type in ff_cell_types:
            ff_instances[inst_name] = inst
        else:
            other_instances[inst_name] = inst

    print(f"Identified {len(ff_instances)} flip-flop instances and {len(other_instances)} other instances.")

    # Find clock net for each FF
    found_clocks_for = set()
    for net_name, net in parser_obj.netlist.nets.items():
        for pin in net.pins:
            # Assuming pin format is "instance_name/pin_name"
            if 'CLK' in pin:
                instance_name = pin[0]
                if instance_name in ff_instances and instance_name not in found_clocks_for:
                    ffs_by_clock_net[net_name].append(instance_name)
                    found_clocks_for.add(instance_name)
                    # Optimization: Stop checking pins for this net once a CLK is found,
                    # or continue if a net might drive multiple different FFs' clocks.
                    # The grouping logic handles multiple FFs on the same net correctly anyway.

    ffs_without_clock = len(ff_instances) - len(found_clocks_for)
    if ffs_without_clock > 0:
        print(f"Warning: Could not find clock net connection for {ffs_without_clock} flip-flop instances. They will be grouped separately.")

    print(f"Grouped FFs into {len(ffs_by_clock_net)} clock net groups (key=None for unconnected).")

    # 3. Process Each Group
    final_instances = other_instances # Start with non-FF instances
    final_old_to_new_map = {}
    index = 0

    for clock_net, ff_group in ffs_by_clock_net.items():
        group_name = f"Clock Net '{clock_net}'" if clock_net else "FFs with Unconnected Clock"
        print(f"\nProcessing {group_name} with {len(ff_group)} FFs...")

        if len(ff_group) <= 1:
            print("  Group size <= 1, skipping banking. Keeping original FF(s).")
            for inst in ff_group:
                final_instances[inst.name] = inst # Add the single/unbankable FF to the final list
            continue
        else:
            ff_group = {name:parser_obj.die.instances[name] for name in ff_group}

        # Create a temporary parser object containing only the FFs for this group
        # Need to copy necessary attributes from the original parser object
        temp_parser_for_group = Parser(parser_obj.file_path) # Use file_path for init, though parse() isn't called

        # Copy essential attributes needed by cluster_and_merge_flip_flops and its helpers
        temp_parser_for_group.cell_library = parser_obj.cell_library
        temp_parser_for_group.placement_rows = parser_obj.placement_rows
        # Copy parameters if they are used downstream (check cluster_and_merge and placement logic)
        temp_parser_for_group.alpha = parser_obj.alpha
        temp_parser_for_group.beta = parser_obj.beta
        temp_parser_for_group.gamma = parser_obj.gamma
        temp_parser_for_group.lambda_ = parser_obj.lambda_
        # Copy other potentially relevant info if needed
        # temp_parser_for_group.timing_info = parser_obj.timing_info
        # temp_parser_for_group.power_info = parser_obj.power_info
        # temp_parser_for_group.bin_constraints = parser_obj.bin_constraints

        # CRITICAL: Use a deep copy of the die object to isolate instance list modification
        #temp_parser_for_group.die = copy.deepcopy(parser_obj.die)
        # Set the instances for this temporary parser to *only* the FFs in the current group
        temp_parser_for_group.die = Die(0, 0, 10, 10)
        temp_parser_for_group.die.instances = ff_group

        print(f"  Calling cluster_and_merge_flip_flops for {len(temp_parser_for_group.die.instances)} FFs in this group.")
        # cluster_and_merge_flip_flops returns a dict of instances (merged + unmerged from the input group)
        # and the mapping for the merged ones within that group.
        merged_instances_for_net, map_for_net = cluster_and_merge_flip_flops(temp_parser_for_group, index)

        # Add results to the final collections
        print(f"  Merging results: {len(merged_instances_for_net)} instances returned from banking, {len(map_for_net)} mappings created for this group.")
        final_instances.update(merged_instances_for_net) # Update the main instance dict

        old_len = len(final_old_to_new_map)
        final_old_to_new_map.update(map_for_net) # Update the main mapping dict
        assert len(final_old_to_new_map) >= old_len
        index += 1

    print(f"\nBanking per clock net complete.")
    print(f"Final instance count: {len(final_instances)}")
    print(f"Total mappings created across all groups: {len(final_old_to_new_map)}")

    return final_instances, final_old_to_new_map


def print_overlaps(site_to_instances, verbose=False):
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


def find_adjacent_empty_site(instance_name, instance_width, instance_height, current_site, start_row, all_placement_rows, site_to_instances, max_search_dist = 100):
    """
    Searches for the nearest sequence of empty sites adjacent (left or right)
    to the current_site that can accommodate the instance dimensions (width and height).

    Args:
        instance_name: Name of the instance to move.
        instance_width: Width of the instance in coordinate units.
        instance_height: Height of the instance in coordinate units.
        current_site: The starting site (x, y) of the overlap (typically the bottom-left site).
        start_row: The placement row corresponding to current_site's y-coordinate.
        all_placement_rows: List/dict of all placement rows.
        site_to_instances: The current mapping of sites to instances.
        max_search_dist: Max number of horizontal sites to check left/right.

    Returns:
        The coordinates (x, y) of the new bottom-left starting site if found, otherwise None.
    """
    if not start_row or start_row.site_width <= 0 or start_row.site_height <= 0:
        print(f"    Error (find_adjacent): Invalid start_row provided for {instance_name}.")
        return None
    placement_rows_map = {row.start_y: row for row in all_placement_rows}

    num_sites_needed_horiz = ceil(instance_width / start_row.site_width)
    num_rows_needed = ceil(instance_height / start_row.site_height)

    if num_sites_needed_horiz <= 0 or num_rows_needed <= 0:
        print(f"    Error (find_adjacent): Invalid dimensions for {instance_name} ({num_sites_needed_horiz}x{num_rows_needed} sites).")
        return None

    start_x, start_y = current_site # Bottom-left site of the overlap

    # Search Right then Left from the original overlap site's x-coordinate
    for direction in [1, -1]: # 1 for right, -1 for left
        for i in range(0, max_search_dist + 1): # Check distance 0 first (original column if possible), then 1 site right/left, etc.
            potential_start_site_x = start_x + direction * i * start_row.site_width
            potential_start_site_coords = (potential_start_site_x, start_y) # Potential bottom-left

            # Check if the potential bottom-left site is horizontally within the start row bounds
            if not (start_row.start_x <= potential_start_site_x < start_row.start_x + start_row.total_sites * start_row.site_width):
                break# This starting X is invalid in the base row, try next distance

            # Check if the instance fits horizontally from this potential start X
            if potential_start_site_x + instance_width > start_row.start_x + start_row.total_sites * start_row.site_width + 1e-9: # Tolerance
                break # Doesn't fit horizontally, try next distance

            # Now, check if all sites (horizontally and vertically) are clear
            all_sites_clear = True
            for row_offset in range(num_rows_needed):
                check_row_y = start_y + row_offset * start_row.site_height
                check_row = placement_rows_map.get(check_row_y)

                if not check_row:
                    all_sites_clear = False # Required row doesn't exist
                    break
                # Optional: Add compatibility check if needed, though create_site_instance_mappings should prevent incompatible multi-row instances
                # if not (check_row.start_x == start_row.start_x and ...): all_sites_clear = False; break

                for site_offset_x in range(num_sites_needed_horiz):
                    check_site_x = potential_start_site_x + site_offset_x * check_row.site_width
                    check_site_coords = (check_site_x, check_row_y)

                    # Check horizontal bounds for this specific site in its row
                    if not (check_row.start_x <= check_site_x < check_row.start_x + check_row.total_sites * check_row.site_width):
                        all_sites_clear = False
                        break # Site out of horizontal bounds

                    # Check occupancy
                    occupying_instances = site_to_instances.get(check_site_coords, [])
                    if occupying_instances:
                        # Allow if the *only* occupant is the instance we are trying to move
                        if not (len(occupying_instances) == 1 and occupying_instances[0] == instance_name):
                            all_sites_clear = False
                            break # Site is occupied by another instance or multiple instances

                if not all_sites_clear:
                    break # Stop checking rows if a conflict was found

            if all_sites_clear:
                #print(f"    Found empty spot for {instance_name} (size {num_sites_needed_horiz}x{num_rows_needed}) at {potential_start_site_coords}")
                return potential_start_site_coords # Found a suitable empty spot
            

        # If searching right failed, the loop continues to search left (direction = -1)

    #print(f"    Could not find adjacent empty spot for {instance_name} near {current_site}")
    return None


def resolve_overlaps(parser_obj, max_iterations= 10):
    """
    Attempts to resolve instance overlaps by nudging instances to adjacent empty sites.

    Args:
        parser_obj: The Parser object with instances to check and modify.
        max_iterations: Maximum attempts to resolve overlaps.

    Returns:
        True if all overlaps were resolved, False otherwise.
    """
    print("\n--- Resolving Overlapping Instances ---")
    assert "C102260" in parser_obj.die.instances
    cell_library = parser_obj.cell_library
    placement_rows_map = {row.start_y: row for row in parser_obj.placement_rows}

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
                instance_to_move = parser_obj.die.instances.get(instance_to_move_name)
                if instance_to_move.cell_type in cell_library.flip_flops:
                    if not instance_to_move:
                        print(f"    Error: Instance '{instance_to_move_name}' not found in instances_dict. Skipping.")
                        continue

                    # Get instance dimensions
                    instance_width = 0
                    instance_height = 0
                    if instance_to_move.cell_type in cell_library.flip_flops:
                        cell = cell_library.flip_flops[instance_to_move.cell_type]
                        instance_width = cell.width
                        instance_height = cell.height
                    elif instance_to_move.cell_type in cell_library.gates:
                        # Assuming gates are not moved or handled differently
                        assert False, "Overlap resolution currently only handles moving Flip-Flops"
                        continue # Skip moving gates for now
                    else:
                        print(f"    Warning: Cell type '{instance_to_move.cell_type}' for instance '{instance_to_move_name}' not found. Cannot determine dimensions. Skipping move.")
                        continue

                    if instance_width <= 0 or instance_height <= 0:
                         print(f"    Warning: Instance '{instance_to_move_name}' has invalid dimensions ({instance_width}x{instance_height}). Skipping move.")
                         continue

                    # Find the starting row based on the instance's actual y-coordinate
                    start_row = placement_rows_map.get(instance_to_move.y)
                    if not start_row:
                        # This could happen if the instance's y doesn't match a row start, try finding closest
                        potential_start_row = find_closest_row(instance_to_move.y, parser_obj.placement_rows)
                        if potential_start_row and abs(potential_start_row.start_y - instance_to_move.y) < 1e-6:
                            start_row = potential_start_row
                        else:
                            print(f"    Warning: Could not determine valid start row for instance '{instance_to_move_name}' at y={instance_to_move.y}. Skipping move.")
                            continue

                    # Find an adjacent empty site considering height
                    new_site_coords = find_adjacent_empty_site(
                        instance_to_move_name,
                        instance_width,
                        instance_height,
                        site, # Pass the specific overlap site as the reference point for search
                        start_row, # Pass the instance's actual start row
                        parser_obj.placement_rows, # Pass all rows
                        site_to_instances,
                        max_search_dist=10**(iteration+1)
                    )

                    if new_site_coords:
                        new_x, new_y = new_site_coords # This is the new bottom-left corner
                        #print(f"    Moving instance '{instance_to_move_name}' from ({instance_to_move.x}, {instance_to_move.y}) to ({new_x}, {new_y})")

                        # --- Update site_to_instances incrementally ---
                        # 1. Remove instance from all its OLD sites
                        # Use the instance_to_sites map generated at the *start* of the iteration
                        old_sites_for_instance = instance_to_sites.get(instance_to_move_name, [])
                        if not old_sites_for_instance:
                             print(f"    Warning: No old sites found in instance_to_sites map for '{instance_to_move_name}'. Cannot reliably remove from site_to_instances.")
                             # As a fallback, try calculating old sites based on original position? Might be complex/risky.
                        else:
                            for old_site_coords in old_sites_for_instance:
                                if old_site_coords in site_to_instances and instance_to_move_name in site_to_instances[old_site_coords]:
                                    site_to_instances[old_site_coords].remove(instance_to_move_name)
                                    # Optional: Clean up empty lists if desired
                                    # if not site_to_instances[old_site_coords]:
                                    #     del site_to_instances[old_site_coords]

                        # Update the instance object's coordinates *after* removing from old sites
                        instance_to_move.x = new_x
                        instance_to_move.y = new_y

                        # 2. Add instance to its NEW sites
                        num_sites_horiz = ceil(instance_width / start_row.site_width) # Use start_row props
                        num_rows_needed = ceil(instance_height / start_row.site_height)
                        new_sites_list = [] # Keep track for updating instance_to_sites later

                        for row_offset in range(num_rows_needed):
                            current_row_y = new_y + row_offset * start_row.site_height
                            current_row = placement_rows_map.get(current_row_y)
                            if not current_row: # Should not happen if find_adjacent_empty_site worked
                                print(f"    Error: Row {current_row_y} needed for moved instance {instance_to_move_name} not found during update.")
                                continue # Skip adding to this row

                            for site_offset_x in range(num_sites_horiz):
                                current_site_x = new_x + site_offset_x * current_row.site_width
                                current_site_coords = (current_site_x, current_row_y)

                                # Check bounds just in case
                                if not (current_row.start_x <= current_site_x < current_row.start_x + current_row.total_sites * current_row.site_width):
                                     print(f"    Warning: New site {current_site_coords} for {instance_to_move_name} is out of bounds for row {current_row_y}. Skipping add.")
                                     continue

                                if current_site_coords not in site_to_instances:
                                    site_to_instances[current_site_coords] = []
                                # Avoid adding duplicates if somehow already there (shouldn't happen after removal)
                                if instance_to_move_name not in site_to_instances[current_site_coords]:
                                    site_to_instances[current_site_coords].append(instance_to_move_name)
                                new_sites_list.append(current_site_coords)

                        # Update instance_to_sites map for the moved instance (for the *next* iteration's reference)
                        instance_to_sites[instance_to_move_name] = new_sites_list

                        moved_count += 1
                        break
                    else:
                        print(f"      Failed to find a new location for '{instance_to_move_name}' near site {site}.")
                        failed_moves += 1
                        # If we fail to move, the overlap persists for the next iteration (or final failure)

        print(f"  Iteration {iteration + 1} summary: Moved {moved_count} instances, failed to move {failed_moves} instances involved in overlaps.")
        # Re-create site_to_instances and instance_to_sites after attempting moves in this iteration
        # This ensures the next iteration starts with an up-to-date view of occupied sites.
        # Although the incremental update logic above *should* keep it consistent,
        # re-creating provides a clean state for the next iteration's overlap detection.
        # site_to_instances, instance_to_sites = create_site_instance_mappings(parser_obj) # This is already done at the start of the loop
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


def create_pin_mapping(original_instances, final_instances, old_ff_to_new_ff_map, cell_library):
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

    # 1. Create reverse map: new_ff_name -> [old_ff_name1, old_ff_name2, ...]
    new_ff_to_old_ffs_map = defaultdict(list)
    for old_name, new_name in old_ff_to_new_ff_map.items():
        if old_name in original_instances: # Ensure the old FF actually exists
             new_ff_to_old_ffs_map[new_name].append(old_name)
        else:
            #print(f"Warning: Old FF '{old_name}' from mapping not found in original instances. Skipping.")
            pass


    # 2. Process merged flip-flops
    for new_ff_name, old_ff_names in new_ff_to_old_ffs_map.items():
        #print(f"{new_ff_name} is the banked results of these flip flops: {old_ff_names}")
        if not old_ff_names: continue # Skip if list is empty

        if new_ff_name not in final_instances:
            #print(f"Warning: New FF '{new_ff_name}' from mapping not found in final instances. Skipping mapping for its constituents.")
            continue

        new_instance = final_instances[new_ff_name]
        try:
            new_ff_library_cell = cell_library.flip_flops[new_instance.cell_type]
        except KeyError:
            #print(f"Warning: Cell type '{new_instance.cell_type}' for merged FF '{new_ff_name}' not found in library. Skipping mapping.")
            continue

        # Get original instance objects and sort them by coordinate (Y then X)
        old_ff_instances = []
        for name in old_ff_names:
             if name in original_instances:
                 old_ff_instances.append(original_instances[name])
             # else: already warned above

        if not old_ff_instances: continue # Skip if no valid old instances found

        # Sort primarily by Y, secondarily by X to assign bit index consistently
        old_ff_instances.sort(key=lambda inst: (inst.original_y, inst.original_x))

        new_instance.original_name = []
        # Map pins for each original FF based on its sorted position (bit_index)
        for bit_index, old_instance in enumerate(old_ff_instances):
            try:
                old_ff_library_cell = cell_library.flip_flops[old_instance.cell_type]
            except KeyError:
                print(f"Warning: Cell type '{old_instance.cell_type}' for original FF '{old_instance.name}' not found in library. Skipping its pin mapping.")
                continue

            new_instance.original_name.append(old_instance.original_name)
            for old_pin_name in old_ff_library_cell.pins.keys():
                assert isinstance(old_instance.original_name, str)
                # Determine the corresponding new pin name convention
                # Common conventions: D->D0, Q->Q0, CLK->CLK, RST->RST etc.
                # This might need adjustment based on actual library conventions
                new_pin_name = ""
                
                if old_pin_name.upper() == 'D':
                    new_pin_name = f"D{bit_index}"
                elif old_pin_name.upper() == 'Q':
                     new_pin_name = f"Q{bit_index}"
                # Add more rules here if other pin naming conventions exist (e.g., TI, SI, SO for scan chains)
                else:
                    assert old_pin_name == "CLK"


                # Check if the determined new pin name actually exists in the new FF type
                if new_pin_name:
                    assert new_pin_name in new_ff_library_cell.pins, f"{new_pin_name}, {new_ff_library_cell.pins}"
                    if old_instance.pin_mapping:
                        new_instance.pin_mapping[new_pin_name] = (old_instance.pin_mapping[old_pin_name])
                    else:
                        assert old_instance.name == old_instance.original_name
                        new_instance.pin_mapping[new_pin_name] = (old_instance.original_name, old_pin_name)


if __name__ == "__main__":
    # Example Usage:
    file_path = "bm/testcase1_0812.txt" # Or bm/sampleCase
    #file_path = "bm/sampleCase" # Or bm/sampleCase
    print(f"Parsing file: {file_path}")
    parser = Parser(file_path)
    try:
        parsed_data = parser.parse()
        debanking_all(parsed_data.die, parsed_data.cell_library, parsed_data.netlist)
        print("Parsing complete.")
        
        if not parsed_data.placement_rows:
             print("Warning: No placement rows found in the input file.")

        print("\n--- Initial State Summary ---")
        # Create initial mappings before merging/resolving
        initial_site_map, initial_instance_map = create_site_instance_mappings(parsed_data)
        print(f"Initial instance count: {len(parsed_data.die.instances)}")
        print(f"Initial occupied sites: {len(initial_site_map)}")

        # --- Cluster and Merge Flip-Flops (Per Clock Net) ---
        # Call the new function that handles banking per clock net
        #updated_instances, old_to_new_map = banking_each_clock_net(parsed_data)

        # TODO: Verify create_pin_mapping logic with per-clock-net results.

        #with open("temp1.pkl", 'wb') as f:
        #    pickle.dump([updated_instances, old_to_new_map], f)
        # --- Create Pin Mapping using original state and final state ---
        #with open("temp1.pkl", 'rb') as f:
        #    updated_instances, old_to_new_map = pickle.load(f)

        #print("\n".join(map(str, old_to_new_map.items())))
        #create_pin_mapping(parsed_data.die.instances, updated_instances, old_to_new_map, parsed_data.cell_library)

        # Update the main parser object's instances with the final result
        #parsed_data.die.instances = updated_instances

        #with open("temp1.pkl", 'wb') as f:
        #    pickle.dump(parsed_data, f)
        with open("temp1.pkl", 'rb') as f:
            parsed_data = pickle.load(f)


        # --- Old FF to New FF Mapping (Sample) --- # (This section seems fine)
        #print("\n--- Old FF to New FF Mapping (Sample) ---")
        #map_sample_count = 0
        #for old_ff, new_ff in old_to_new_map.items():
        #    if map_sample_count >= 5: break
        #    print(f"  Original: {old_ff} -> Merged: {new_ff}")
        #    map_sample_count += 1
        #if not old_to_new_map:
        #    print("  No FFs were merged, mapping is empty.")
        # -----------------------------------------

        # --- Create Site/Instance Mappings AFTER merging ---
        print("\nCreating site-instance mappings post-merge...")
        site_map, instance_map = create_site_instance_mappings(parsed_data)
        #print("\n".join(map(str, [(k, v, len(v)) for k, v in instance_map.items()])))
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
        print(f"Final instance count: {len(parsed_data.die.instances)}")
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
