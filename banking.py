import numpy as np
from parser import Parser
from module import Instance, PlacementRow, FlipFlop
from cluster import perform_mean_shift_clustering
import math

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
        # Try placing at the beginning if it doesn't fit at the calculated spot
        if row.start_x + new_ff_width <= row.start_x + row.total_sites * row.site_width:
             return row.start_x
        else:
            # Cannot fit even at the start
            print(f"Warning: Cannot fit FF with width {new_ff_width} in row starting at {row.start_x} with total width {row.total_sites * row.site_width}")
            return None # Indicate placement failure


def cluster_and_merge_flip_flops(parser_obj: Parser):
    """
    Clusters 1-bit flip-flop instances and merges them into higher-bit flip-flops 
    aligned with placement rows.

    Args:
        parser_obj: A Parser object containing parsed design data.

    Returns:
        A new list of instances with merged flip-flops, or the original list if no merging occurs.
    """
    
    # 1. Filter 1-bit Flip-Flop Instances
    one_bit_ff_instances = []
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

        # Check if a k-bit FF exists
        target_ff_type: FlipFlop | None = None
        target_ff_name: str | None = None
        for name, ff in parser_obj.cell_library.flip_flops.items():
            if ff.bits == k:
                target_ff_type = ff
                target_ff_name = name
                break
        
        if target_ff_type and target_ff_name:
            print(f"  Found matching {k}-bit FF type: {target_ff_name}")
            
            # Calculate placement
            cluster_center_x = cluster_centers[cluster_label][0]
            cluster_center_y = cluster_centers[cluster_label][1]
            
            closest_row = find_closest_row(cluster_center_y, parser_obj.placement_rows)
            
            if closest_row:
                new_x = calculate_placement(cluster_center_x, target_ff_type.width, closest_row)
                
                if new_x is not None:
                    new_y = closest_row.start_y
                    # Create new merged instance name (e.g., merged_cluster_0_ffk)
                    new_instance_name = f"merged_cluster_{cluster_label}_ff{k}" 
                    
                    new_merged_instance = Instance(new_instance_name, target_ff_name, new_x, new_y)
                    new_instances.append(new_merged_instance)
                    #print(f"  Created merged instance {new_instance_name} at ({new_x}, {new_y}) in row starting at y={closest_row.start_y}")

                    # Mark original instances for removal
                    for inst in cluster_instances:
                        merged_instance_names.add(inst.name)
                        if inst.name in original_indices:
                             instances_to_remove_indices.add(original_indices[inst.name])
                else:
                     print(f"  Placement failed for cluster {cluster_label}. Keeping original FFs.")
            else:
                print(f"  Could not find suitable placement row for cluster {cluster_label}. Keeping original FFs.")
        else:
            print(f"  No {k}-bit FF type found in library for cluster {cluster_label}. Keeping original FFs.")

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

    return final_instances




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
             
        print(f"Initial number of instances: {len(parsed_data.instances)}")

        # --- Run the clustering and merging ---
        updated_instances = cluster_and_merge_flip_flops(parsed_data)
        # ------------------------------------

        # You can now work with 'updated_instances'
        # For example, update the parser object if needed (optional)
        # parsed_data.instances = updated_instances 
        
        print("\n--- Merging Process Summary ---")
        print(f"Original instance count: {len(parser.instances)}")
        print(f"Instance count after merging: {len(updated_instances)}")
        
        # Further analysis or output generation could go here
        # E.g., write the updated instance list to a file

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
