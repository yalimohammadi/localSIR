import pandas as pd
import networkx as nx
import seaborn as sns
import numpy as np
from utils.constants import *

def create_copenhagen_graph() -> nx.Graph:
    """
    Reads a Bluetooth data CSV, filters rows based on timestamp and RSSI thresholds,
    and returns a graph built from the valid edges.
    
    Parameters:
        file_path (str): Path to the CSV file containing the data.
        
    Returns:
        nx.Graph: A networkx graph created from the filtered edge list.
    """
    # Read CSV file with explicit data types for the columns
    df = pd.read_csv(file_copenhagen, sep=',', dtype={'user_a': int, 'user_b': int, 'rssi': int})
    
    # Constants
    DIST_THRESHOLD = 1.83         # Distance threshold in meters (approx. 6 ft)
    RSSI_THRESHOLD = -74.25       # Minimum RSSI threshold for valid entries
    SECONDS_PER_HOUR = 3600       # Number of seconds in an hour
    
    # Define time filtering parameters
    hr_offset = 24 * 4            # Base offset (e.g., 96 hours)
    # Calculate start and end of the time window (the window is shifted by 8 hours)
    start_time = SECONDS_PER_HOUR * (hr_offset + 8)
    end_time = SECONDS_PER_HOUR * (hr_offset + 12 + 8)
    
    # Filter rows based on the "# timestamp" column (timestamps are assumed to be in seconds)
    time_filtered = df[df["# timestamp"].isin(range(start_time, end_time))]
    
    # Filter rows based on RSSI values:
    # - Keep values greater than or equal to the threshold
    # - Remove invalid data (e.g., RSSI >= 0)
    valid_rssi = time_filtered[
        (time_filtered["rssi"] >= RSSI_THRESHOLD) & (time_filtered["rssi"] < 0)
    ]
    
    # Create edge list and rename columns for graph construction
    edges = (
        valid_rssi[["user_a", "user_b"]]
        .rename(columns={"user_a": "source", "user_b": "target"})
    )
    # Ensure valid target values (if required by your use case)
    edges = edges[edges["target"] > -1]
    
    # Create a graph from the edge list
    graph = nx.from_pandas_edgelist(edges)
    
    return graph

import pickle
import networkx as nx

def load_sf_graph(file_path, hours=6):
    """
    Load a bipartite mobility graph from a pickle file containing hourly matrices.
    
    Parameters:
        file_path (str): Path to the pickle file containing the data.
        hours (int): Number of initial matrices (hours) to sum for constructing the graph.
        
    Returns:
        networkx.Graph: A bipartite graph constructed from the summed data.
        
    The pickle file should contain a list of matrices (e.g., sparse matrices) for each hour.
    The function sums the first `hours` matrices, then builds a bipartite graph with:
      - Partition X: nodes 0 to num_rows-1
      - Partition Y: nodes num_rows to num_rows+num_cols-1
      
    An edge is added between node i in partition X and node (num_rows + j) in partition Y 
    for each nonzero entry in the summed matrix, with the corresponding weight.
    """
    # Load the data from the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Sum the first `hours` matrices
    data_six_hr = sum(data[:hours])
    
    # Create an empty graph
    G = nx.Graph()
    num_rows, num_cols = data_six_hr.shape
    
    # Add nodes for the two bipartite partitions
    G.add_nodes_from(range(num_rows), bipartite=0)  # Partition X
    G.add_nodes_from(range(num_rows, num_rows + num_cols), bipartite=1)  # Partition Y
    
    # Add edges for each nonzero entry in the summed matrix
    for i, j, weight in zip(*data_six_hr.nonzero(), data_six_hr.data):
        G.add_edge(i, num_rows + j, weight=weight)
    
    return G


def create_rgg(n):
    """
    Create a random geometric graph with n nodes.
    The side length is set to sqrt(n) to roughly keep node density constant,
    and a fixed radius of 1.5 is used for connection.
    """
    side_length = np.sqrt(n)
    x_coords = np.random.uniform(0, side_length, n)
    y_coords = np.random.uniform(0, side_length, n)
    pos = {i: (x_coords[i], y_coords[i]) for i in range(n)}
    G = nx.random_geometric_graph(n, 1.5, pos=pos)
    return G


def create_pa(n):
    """
    Create a preferential attachment (Barabási–Albert) graph with n nodes.
    Here, each new node attaches to m=3 existing nodes.
    """
    G = nx.barabasi_albert_graph(n, m=3)
    return G

def generate_graphs(n_values, graph_type='rgg'):
    """
    Generates graphs for each n in n_values using the specified graph type.
    Prints the average degree for each graph.
    
    Parameters:
        n_values (list): List of integers for the number of nodes.
        graph_type (str): Type of graph to generate ('rgg' or 'pa').
    
    Returns:
        dict: A dictionary where keys are node counts and values are the generated graphs.
    """
    G_dicts = {}
    
    for n in n_values:
        if graph_type == 'rgg':
            G = create_rgg(n)
        elif graph_type == 'pa':
            G = create_pa(n)
        else:
            raise ValueError("Invalid graph type. Choose 'rgg' or 'pa'.")
        
        G_dicts[n] = G
        # Average degree = (2 * number_of_edges) / number of nodes
        avg_degree = G.number_of_edges() * 2 / n
        print(f"For n = {n}: Average Degree = {avg_degree:.2f}")
        
    return G_dicts



