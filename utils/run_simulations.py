import pandas as pd
import EoN  # Ensure you have Epidemics on Networks installed

def create_dataframe_from_sim(iteration, times_true, S_true, I_true, R_true):
    """
    Create a DataFrame for a single simulation iteration.

    Parameters:
        iteration (int): The simulation iteration index.
        times_true (list of arrays): List of time arrays for each iteration.
        S_true (list of arrays): List of susceptible arrays.
        I_true (list of arrays): List of infected arrays.
        R_true (list of arrays): List of recovered arrays.

    Returns:
        pd.DataFrame: A DataFrame with columns ["iteration", "t", "S", "I", "R"].
    """
    data = {
        "iteration": [iteration] * len(times_true[iteration]),
        "t": times_true[iteration],
        "S": S_true[iteration],
        "I": I_true[iteration],
        "R": R_true[iteration],
    }
    return pd.DataFrame(data)

def run_simulation_for_n(n, iterations, G, tau, gamma, infection_prob,weighted=False):
    """
    Run the SIR simulation for a graph with n nodes for a specified number of iterations.
    
    Parameters:
        n (int): Number of nodes.
        iterations (int): Number of simulation runs.
        G (networkx.Graph): Graph on which to run the simulation.
        tau (float): Transmission rate.
        gamma (float): Recovery rate.
        infection_prob (float): Initial infection probability (rho).

    Returns:
        pd.DataFrame: A concatenated DataFrame of simulation results with S, I, R normalized by n.
    """
    times_true = []
    S_true = []
    I_true = []
    R_true = []
    
    # Run the simulation multiple times
    for iteration in range(iterations):
        if weighted:
            t, s, i, r = EoN.fast_SIR(G, tau, gamma, rho=infection_prob,transmission_weight="weight")
        else:   
            t, s, i, r = EoN.fast_SIR(G, tau, gamma, rho=infection_prob)
        times_true.append(t)
        S_true.append(s)
        I_true.append(i)
        R_true.append(r)
        print(f"n = {n}, iteration = {iteration}, final R fraction = {max(r)/n:.3f}")
    
    # Concatenate all iterations into a single DataFrame
    df_true_sims = pd.DataFrame()
    for iteration in range(iterations):
        df_iter = create_dataframe_from_sim(iteration, times_true, S_true, I_true, R_true)
        df_true_sims = pd.concat([df_true_sims, df_iter], ignore_index=True)
    
    # Normalize the state values by n
    df_true_sims["S"] /= n
    df_true_sims["I"] /= n
    df_true_sims["R"] /= n
    
    return df_true_sims

