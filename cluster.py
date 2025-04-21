import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from parser import Parser  # Assuming parser.py is in the same directory or accessible
from module import Instance # Import Instance if needed for type hinting or direct use

def perform_mean_shift_clustering(parser_obj: Parser):
    """
    Performs Mean Shift clustering on the instances provided by a Parser object.

    Args:
        parser_obj: An instance of the Parser class that has already parsed the data.

    Returns:
        A tuple containing:
        - labels: An array of cluster labels assigned to each instance.
        - cluster_centers: An array of coordinates for the identified cluster centers.
        - n_clusters: The estimated number of clusters.
    """
    if not parser_obj.instances:
        print("No instances found in the parser object.")
        return None, None, 0

    # Check if instance is a FF, and extract coordinates from FF instances
    coordinates = np.array([[inst.x, inst.y] for inst in parser_obj.instances if "FF" in  str(inst.cell_type)])
            
    # Calculate number of flip-flops from coordinates collected. This is to see if it correctly finds them.
    Coordcount = 0;
    for coords in coordinates:
        Coordcount = Coordcount + 1;

    # Extract coonates (x, y) from instances
    #coordinates = np.array([[inst.x, inst.y] for inst in parser_obj.instances])

    # Estimate bandwidth
    # quantile: Proportion of samples to use for bandwidth estimation (0.3 means 30%)
    # n_samples: Number of samples to use. If specified, quantile is ignored.
    # Adjust quantile or n_samples based on dataset size and characteristics if needed.
    #bandwidth = estimate_bandwidth(coordinates, quantile=0.002, n_samples=1000, random_state=42, n_jobs=-1)
    bandwidth = 6000

    if bandwidth <= 0:
        print(f"Estimated bandwidth is {bandwidth}. Clustering cannot proceed.")
        # Handle cases where bandwidth estimation fails (e.g., too few points, points are identical)
        # Assign all points to one cluster or handle as an error case.
        labels = np.zeros(len(coordinates), dtype=int)
        cluster_centers = np.mean(coordinates, axis=0, keepdims=True) if len(coordinates) > 0 else np.array([])
        n_clusters = 1 if len(coordinates) > 0 else 0
        return labels, cluster_centers, n_clusters, Coordcount


    print(f"Estimated bandwidth: {bandwidth}")

    # Apply Mean Shift
    # bin_seeding=True can speed up the process but might affect results slightly
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
    ms.fit(coordinates)

    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters = len(np.unique(labels))

    print(f"Estimated number of clusters: {n_clusters}")

    return labels, cluster_centers, n_clusters, Coordcount

if __name__ == "__main__":
    # Example usage:
    file_path = "bm/testcase1_0812.txt"  # Make sure this path is correct
    parser = Parser(file_path)

    FFcount = 0

    try:
        parsed_data = parser.parse()
        print(f"Successfully parsed data from {file_path}")
        #print(f"Number of instances to cluster: {len(parsed_data.instances)}")

        FFcount = 0
        for inst in parsed_data.instances:
            if str(inst.cell_type)[0]=="F":
                FFcount = FFcount + 1
        print("Number of FFs from parser:", FFcount)

        labels, centers, num_clusters, coordcount = perform_mean_shift_clustering(parsed_data)

        if labels is not None:
            print(f"\nClustering complete.")
            print(f"Number of clusters found: {num_clusters}")
            
            # You can add more detailed output here, e.g., print labels or centers
            print("Cluster labels for each instance:", labels)
            print("Cluster centers:", centers)
            print("Number of coords:",coordcount)

            # Example: Assign cluster labels back to instances if needed
            # for i, inst in enumerate(parsed_data.instances):
            #    inst.cluster_label = labels[i] # Add a cluster_label attribute if desired

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
    except Exception as e:
        print(f"An error occurred during parsing or clustering: {e}")
