import numpy as np
import random
import networkx as nx
import EoN

def directed_percolate_network(G, tau, gamma, weights = True,weight_func = None):
    #indirectly tested in test_estimate_SIR_prob_size
    r'''
    performs directed percolation, assuming that transmission and recovery 
    are Markovian
    
    
    From figure 6.13 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.  
    
    This performs directed percolation corresponding to an SIR epidemic
    assuming that transmission is at rate tau and recovery at rate 
    gamma

    :See Also:

    ``nonMarkov_directed_percolate_network`` which allows for duration and 
        time to infect to come from other distributions.
    
    ``nonMarkov_directed_percolate_network`` which allows for more complex 
        rules
    
    :Arguments: 

    **G**    networkx Graph
        The network the disease will transmit through.
    **tau**   positive float 
        transmission rate
    **gamma**   positive float 
        recovery rate
    **weights**   boolean    (default True)
        if True, then includes information on time to recovery
        and delay to transmission.  If False, just the directed graph.

    :Returns: 
        :
    **H**   networkx DiGraph  (directed graph)
        a u->v edge exists in H if u would transmit to v if ever 
        infected.
        
        The edge has a time attribute (time_to_infect) which gives the 
        delay from infection of u until transmission occurs.
        
        Each node u has a time attribute (duration) which gives the 
        duration of its infectious period.
        
    :SAMPLE USE:


    ::

        import networkx as nx
        import EoN
        
        G = nx.fast_gnp_random_graph(1000,0.002)
        H = EoN.directed_percolate_network(G, 2, 1)

    '''
    
    #simply calls directed_percolate_network_with_timing, using markovian rules.
    
    def trans_time_fxn(u, v, tau):
        if tau>0:
            return random.expovariate(tau)
        else:
            return float('Inf')
    trans_time_args = (tau,)
    
    def rec_time_fxn(u, gamma):
        if gamma>0:
            return random.expovariate(gamma)
        else:
            return float('Inf')
    rec_time_args = (gamma,)
    if not(weight_func is None):
        trans_rate_fxn, rec_rate_fxn = EoN._get_rate_functions_(G, tau, gamma, 
                                                "weight",
                                                None)
        print("percolation with weights")
    
    
    return EoN.nonMarkov_directed_percolate_network_with_timing(G, trans_time_fxn, 
                                                            rec_time_fxn,  
                                                            trans_time_args,
                                                            rec_time_args, 
                                                            weights=weights)

def find_max_in_dict(keys_to_check,my_dict):
    # Find the keys with the largest value among the specified keys
    max_value = float('-inf')  # Initialize the maximum value to negative infinity
    max_keys = []  # List to store keys with the maximum value

    for key in keys_to_check:
        value = my_dict[key]
        if value > max_value:
            max_value = value
            max_keys = [key]
        elif value == max_value:
            max_keys.append(key)
    return max_keys



def choose_start_node(perc_G_rev, weighted_by_degree=False):
    """
    Select a starting node from perc_G_rev.
    
    If weighted_by_degree is True, nodes are chosen with probability proportional
    to (degree + 0.01). Otherwise, a node is chosen uniformly at random.
    
    Parameters:
        perc_G_rev (networkx.Graph): Graph from which to choose a node.
        weighted_by_degree (bool): Flag to determine weighted sampling.
    
    Returns:
        node: The selected starting node.
    """
    nodes = list(perc_G_rev.nodes())
    if weighted_by_degree:
        weights = np.array([perc_G_rev.degree(node) + 0.01 for node in nodes])
        probabilities = weights / weights.sum()
        return np.random.choice(nodes, p=probabilities)
    else:
        return np.random.choice(nodes)


def local_tracing_modified_fast(k, tau, gamma, infection_prob, perc_G, perc_G_rev, weighted_by_degree=False):
    """
    Perform local contact tracing on a graph using a modified fast SIR process.
    
    The function starts from a randomly selected node (optionally weighted by degree)
    and computes its k-neighborhood. It then assigns initial infections randomly based on
    infection_prob and computes infection and recovery times via multi-source Dijkstra's
    algorithm. If no node is initially infected and the neighborhood is larger than k,
    the farthest node(s) (as determined by find_max_in_dict) are used as infection sources.
    
    Parameters:
        k (int): Radius (number of hops) to define the neighborhood.
        tau (float): Transmission rate (not directly used here).
        gamma (float): Recovery rate (not directly used here).
        infection_prob (float): Probability that a node in the neighborhood is initially infected.
        perc_G (networkx.Graph): The main graph.
        perc_G_rev (networkx.Graph): A modified/reversed version of the graph for tracing.
        weighted_by_degree (bool): Whether to weight node selection by degree.
    
    Returns:
        tuple: (inf_time, rec_time, flag) or (inf_time, rec_time, flag, degree_weight) if weighted_by_degree is True.
            - inf_time: Calculated infection time for the starting node.
            - rec_time: Recovery time of the starting node.
            - flag: Boolean flag that is True if alternate infection sources were used.
            - degree_weight (optional): Relative degree weight of the chosen node.
    """
    # Choose the starting node
    v = choose_start_node(perc_G_rev, weighted_by_degree)
    
    # Compute the k-neighborhood of v (nodes within k hops)
    shortest_paths = nx.single_source_shortest_path_length(perc_G_rev, v, cutoff=k)
    back_k_nodes = list(shortest_paths.keys())[:k+1]
    back_k_graph = perc_G.subgraph(back_k_nodes)
    
    # Randomly infect nodes in the neighborhood based on infection_prob
    randomization = random.choices([True, False], weights=[infection_prob, 1 - infection_prob], k=len(back_k_nodes))
    infected_k = [node for node, is_infected in zip(back_k_nodes, randomization) if is_infected]
    
    # Set default times (infinity means "no infection/recovery")
    inf_time = np.inf
    rec_time = np.inf
    flag = False
    
    if infected_k:
        # Compute infection time using the initially infected nodes as sources
        distances = nx.multi_source_dijkstra_path_length(back_k_graph, weight="delay_to_infection", sources=infected_k)
        inf_time = distances.get(v, np.inf)
        rec_time = back_k_graph.nodes[v].get("duration", np.inf)
    elif not infected_k and len(back_k_nodes) > k:
        flag = True
        # Use farthest nodes (from find_max_in_dict) as the infection source
        farthest_nodes = find_max_in_dict(back_k_nodes, shortest_paths)
        distances = nx.multi_source_dijkstra_path_length(back_k_graph, weight="delay_to_infection", sources=farthest_nodes[-1:])
        inf_time = distances.get(v, np.inf)
        rec_time = back_k_graph.nodes[v].get("duration", np.inf)
    
    if weighted_by_degree:
        # Compute and return the relative degree weight for node v
        nodes = list(perc_G_rev.nodes())
        weights = np.array([perc_G_rev.degree(node) + 0.01 for node in nodes])
        degree_weight = (perc_G_rev.degree(v) + 0.01) / weights.sum()
        return inf_time, rec_time, flag, degree_weight
    
    return inf_time, rec_time, flag

    
def local_tracing(G, k, tau, gamma,infection_prob):
    # same as the previous one but it doesn't infect initally infected node if all are suscdeptible.
    perc_G=EoN.directed_percolate_network(G, tau, gamma, weights = True)
    perc_G_rev= perc_G.reverse()
    # draw uniform random node v from G to start contac tracing
    v = np.random.choice(G.nodes(), size=1)[0]

    # draw the k neighborhood of v
    
    back_k_nodes = list(nx.single_source_shortest_path_length(perc_G_rev, v, cutoff=k).keys())[:k+1]
    back_k_graph = perc_G.subgraph(back_k_nodes)
    # draw initially infected nodes
    randomization = random.choices([True, False], weights=[infection_prob, 1-infection_prob], k=len(back_k_nodes))
    infected_k = [item for item, is_true in zip(back_k_nodes, randomization) if is_true]
    # find recovery and infection times
    
    inf_time = np.inf # if susceptible then infection time is at infinity
    rec_time = np.inf
    if len(infected_k)>0:
        inf_time = nx.multi_source_dijkstra_path_length(back_k_graph, weight="delay_to_infection", sources=infected_k)[v]
        rec_time = back_k_graph.nodes[v]["duration"]
    # print(infected_k)
    return inf_time,rec_time


def run_local_tracing_experiment(G, tau, gamma, infection_prob=0.01, K_range=range(2, 20), iterations=1000, weight_func="weight",weighted_by_degree=False):
    """
    Run the local tracing algorithm over a range of neighborhood sizes (K values) for a given graph.

    For each K in K_range, the function performs `iterations` runs of local_tracing_modified_fast
    and collects the query answers. It also prints some statistics about flags and infinite infection times.

    Parameters:
        G (networkx.Graph): The input graph.
        tau (float): Parameter tau used in the simulation.
        gamma (float): Parameter gamma used in the simulation.
        infection_prob (float): The initial infection probability (default is 0.01).
        K_range (iterable): Range or list of K values (neighborhood radii) to test.
        iterations (int): Number of simulation runs per K.
        weight_func (str): Weight function to use in directed_percolate_network.
        
    Returns:
        dict: A dictionary where each key is a K value and each value is a list of tuples (inf_time, rec_time)
              from the local_tracing_modified_fast runs.
    """
    dict_queries = dict()
    n = G.number_of_nodes()
    
    # Percolate the network and create its reverse
    perc_G = directed_percolate_network(G, tau, gamma, weights=True, weight_func=weight_func)
    perc_G_rev = perc_G.reverse()
    
    # Loop over each K value in the specified range
    for K in K_range:
        print(f"Graph nodes: {n}, K: {K}")
        flags = []
        infis = []
        ans_queries = []
        
        for i in range(iterations):
            ans = local_tracing_modified_fast(K, tau, gamma, infection_prob, perc_G, perc_G_rev,weighted_by_degree=weighted_by_degree)
            # ans is expected to be a tuple where the first two entries are relevant query answers.
            ans_queries.append(ans[:2])
            
            # Count flags if alternate infection source is used
            if ans[2]:
                flags.append(ans[2])
            # Count if the recovery time is infinity
            if ans[1] is np.inf:
                infis.append(0)
        
        dict_queries[K] = ans_queries
        print(f"Flags count: {len(flags)}")
        print(f"Infinite rec_time count: {len(infis)}")
    
    return dict_queries

# Example usage:
# G_safe should be your input graph.
# tau, gamma should be defined appropriately.
# dict_queries = run_local_tracing_experiment(G_safe, tau, gamma)


def run_local_tracing_on_graph_dict(G_dicts, tau, gamma, infection_prob=0.1, K_range=range(2, 10), iterations=10_000):
    """
    Run local tracing experiments on a dictionary of graphs.
    
    For each graph in G_dicts (keyed by the number of nodes), this function runs the
    local tracing experiment over a range of K values and collects the results.
    
    Parameters:
        G_dicts (dict): Dictionary where keys are node counts and values are networkx graphs.
        tau (float): Transmission parameter.
        gamma (float): Recovery parameter.
        infection_prob (float): Infection probability.
        K_range (iterable): Range or list of K values to test.
        iterations (int): Number of simulation runs per K value.
        
    Returns:
        dict: Nested dictionary where the outer keys are node counts and the inner keys are K values,
              with values as lists of results from local_tracing_modified_fast.
    """
    dict_queries = {}
    for n, G in G_dicts.items():
        print(f"\nProcessing graph with n = {n}")
        dict_queries[n] = run_local_tracing_experiment(G, tau, gamma, infection_prob, K_range, iterations)
    return dict_queries

