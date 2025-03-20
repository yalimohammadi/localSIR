
import numpy as np
import bisect
import pandas as pd

def estimator(list_queries):
    # Total number of queries
    n_q = len(list_queries)
    
    # Extract infection times from queries where the infection time is not infinity
    inf_times = [l[0] for l in list_queries if not (l[0] is np.inf)]
    # Extract recovery times corresponding to valid queries (where infection time is not infinity)
    rec_times = [l[1] for l in list_queries if not (l[0] is np.inf)]
    
    # Compute combined times: sum of infection and recovery times for each query, then sort the array
    r_inf_times = sorted(np.array(inf_times) + np.array(rec_times))
    
    # Count how many queries have an infection time of 0
    inf_at_zero = inf_times.count(0)
    flag = False
    if inf_at_zero > 0:
        # If there is at least one query with infection time 0, set flag to True
        flag = True
        # Adjust count by subtracting one (to account for the initial infection)
        inf_at_zero = inf_at_zero - 1
    # Remove duplicate infection times and sort them
    inf_times = sorted(list(set(inf_times)))
    
    # Combine the sorted recovery times (r_inf_times) with the unique infection times, and sort them
    times = sorted(r_inf_times.copy() + inf_times.copy())
    i_sorted = []  # List to store the estimated number of infected individuals at each time
    
    # For each time point, estimate the number of infections
    for t in times:
        # Find the index where t would be inserted in inf_times (gives count of infection events <= t)
        i = bisect.bisect_right(inf_times, t)
        # Find the number of recovered individuals (using combined infection+recovery times) up to time t
        num_rec = bisect.bisect_right(r_inf_times, t)
        # Adjust infection count: add the initial infections (inf_at_zero) and subtract recoveries
        i_sorted.append(inf_at_zero + i - num_rec)
    
    # Normalize the estimated infection counts by the total number of queries
    i_ret = np.array(i_sorted.copy()) / n_q
    r_ret = []  # List to store the estimated number of recovered individuals at each time
    for t in times:
        # For each time point, count the number of recovered individuals (normalized by total queries)
        num_rec = bisect.bisect_right(r_inf_times, t)
        r_ret.append(num_rec / n_q)
    
    # If there was an infection at time 0, prepend time 0 and corresponding zero counts
    if flag:
        times = [0] + times
        i_ret = np.insert(i_ret, 0, 0)
        r_ret = [0] + r_ret
        
    # Ensure outputs are numpy arrays
    i_ret = np.array(i_ret)
    r_ret = np.array(r_ret)
    
    # Return the time points and the estimated infected and recovered proportions
    return times, i_ret, r_ret

    


def HT_estimator(list_queries, n):
    """
    Horvitz-Thompson estimator for epidemic quantities when each query is given by:
      (infection_time, recovery_time, weight)
    The weight should be the inclusion probability for that query.
    The estimator returns (times, i_ret, r_ret) where i_ret and r_ret are the HT estimates
    for the infected and recovered proportions at each time.
    """
    # Filter out queries with an infection time of infinity and extract valid ones
    valid = [(inf, rec, w) for (inf, rec, _, w) in list_queries if inf is not np.inf]
    if not valid:
        # If there are no valid queries, return empty results
        return [], np.array([]), np.array([])
    
    # Compute total Horvitz-Thompson weight (denominator) by summing contributions (each contribution is 1, here multiplied by n)
    total_ht = sum(1.0 * n  # The commented division by w is omitted as written in the original code
                   for (_, _, w) in valid)
    
    # Build infection data: each tuple contains the infection time and its HT weight (1/w)
    inf_data = [(inf, 1.0 / w) for (inf, rec, w) in valid]
    # Sort the infection data by infection time
    inf_data.sort(key=lambda x: x[0])
    # Extract sorted infection times and corresponding weights
    inf_times = [t for (t, _) in inf_data]
    inf_weights = [wt for (_, wt) in inf_data]
    # Compute cumulative sum of infection weights
    cum_inf = np.cumsum(inf_weights)
    
    # Build recovery data: each tuple contains (infection time + recovery delay) and its HT weight (1/w)
    rec_data = [(inf + rec, 1.0 / w) for (inf, rec, w) in valid]
    # Sort the recovery data by the combined time (infection + recovery)
    rec_data.sort(key=lambda x: x[0])
    # Extract sorted recovery times and corresponding weights
    rec_times = [t for (t, _) in rec_data]
    rec_weights = [wt for (_, wt) in rec_data]
    # Compute cumulative sum of recovery weights
    cum_rec = np.cumsum(rec_weights)
    
    # For queries with infection time 0, collect their HT weights
    zero_weights = [wt for (t, wt) in inf_data if t == 0]
    flag = False
    if len(zero_weights) > 0:
        flag = True
        # Subtract one query's contribution (the first one) to adjust for the initial infection at time 0
        initial_adjustment = sum(zero_weights) - zero_weights[0]
    else:
        initial_adjustment = 0.0

    # Define evaluation time points as the union of unique infection times and recovery times
    unique_inftimes = sorted(set(inf_times))
    times = sorted(rec_times + unique_inftimes)
    
    i_sorted = []  # List to store cumulative HT estimate for infections at each time
    r_sorted = []  # List to store cumulative HT estimate for recoveries at each time
    for t in times:
        # For infection: find the index in inf_times where t would be inserted (number of events <= t)
        idx_inf = bisect.bisect_right(inf_times, t)
        # Get cumulative infection weight up to time t (or 0 if none)
        val_inf = cum_inf[idx_inf - 1] if idx_inf > 0 else 0.0
        
        # For recovery: similarly, find cumulative recovery weight up to time t
        idx_rec = bisect.bisect_right(rec_times, t)
        val_rec = cum_rec[idx_rec - 1] if idx_rec > 0 else 0.0
        
        # Estimated number of infected individuals at time t (adjusted for the initial infection)
        i_sorted.append(initial_adjustment + val_inf - val_rec)
        # Estimated number of recovered individuals at time t
        r_sorted.append(val_rec)
    
    # Normalize the cumulative sums by the total HT weight to get proportions
    i_ret = np.array(i_sorted) / total_ht
    r_ret = np.array(r_sorted) / total_ht
    
    # If an infection occurs at time 0, prepend time 0 with zero estimates
    if flag:
        times = [0] + times
        i_ret = np.insert(i_ret, 0, 0)
        r_ret = np.insert(r_ret, 0, 0)
    
    # Return the evaluation times and the estimated infection and recovery proportions
    return times, i_ret, r_ret

def time_division_float(sim_df):
    sim_df=sim_df[sim_df["t"]<6]
    num_points =100
    equally_spaced_points = np.linspace(0, 6, num_points)
    def map_time_to_nearest(time_col):
        return time_col.apply(lambda x: equally_spaced_points[np.abs(equally_spaced_points - x).argmin()])


    sim_df_new = sim_df.groupby('iteration').apply(lambda x: x.assign(time_mapped=map_time_to_nearest(x['t'])))
    sim_df_new = sim_df_new.drop_duplicates(["iteration","time_mapped"])

    # Create a new DataFrame with one row for each time value in equally_spaced_points

    # Create a new DataFrame with one row for each time value in equally_spaced_points for each iteration
    new_dfs = []
    for group_name, group_df in sim_df_new.groupby('iteration'):
        new_df = pd.DataFrame({'iteration': [group_name] * num_points, 'time_mapped': equally_spaced_points})
        new_dfs.append(new_df)

    # Concatenate the list of DataFrames into a single DataFrame
    new_df = pd.concat(new_dfs, ignore_index=True)


    # Merge the new DataFrame with the original DataFrame on 'iterations' and 'time_mapped' columns
    result_df = pd.merge(new_df, sim_df_new, on=['iteration', 'time_mapped'], how='left')

    result_df["S"]=result_df.groupby('iteration')['S'].fillna(method='ffill')
    result_df["I"]=result_df.groupby('iteration')['I'].fillna(method='ffill')
    result_df["R"]=result_df.groupby('iteration')['R'].fillna(method='ffill')
    result_df["S"]=result_df.groupby('iteration')['S'].fillna(value=1)
    result_df["I"]=result_df.groupby('iteration')['I'].fillna(value=0)
    result_df["R"]=result_df.groupby('iteration')['R'].fillna(value=0)
    return result_df

def time_division_float_iterations(sim_df):
    results = []
    for i in range(sim_df["iteration"].max()+1):
        sim_iter = sim_df[sim_df["iteration"]==i]
        result_iter=time_division_float(sim_iter)
        results.append(result_iter)
    return pd.concat(results)
def process_infection_prob(dict_queries, G, time_division_float, estimator_func=estimator):
    """
    Process simulation query data for a single infection probability value.

    For the given infection probability, this function loops over neighborhood sizes (k values),
    divides the query data into segments, applies the estimator to each segment, aggregates the results
    into a DataFrame, and then processes that DataFrame using the time_division_float function.
    
    Parameters:
        infection_prob (float): The infection probability to process.
        dict_queries (dict): Nested dictionary of query data, indexed by infection probability and k.
        G (networkx.Graph): The graph, whose node count is used for normalization.
        time_division_float (function): Function to process the final simulation DataFrame.
        estimator_func (function): The estimator function to apply (e.g., HT_estimator or estimator).
    
    Returns:
        dict: A dictionary with keys as k values (from 2 to 9) and values as the processed DataFrames.
    """
    inner_dict = dict()
    # Loop over different neighborhood sizes (k values)
    for k in range(2, 10):
        print(G.number_of_nodes(), k)
        sim_df = pd.DataFrame()
        # Process the queries in segments (each segment has 10 query results)
        num_segments = int(len(dict_queries[k]) / 10)
        for i in range(num_segments):
            n = G.number_of_nodes()
            # Apply the specified estimator function to a slice of 10 queries
            t_ret, i_ret, r_ret = estimator_func(dict_queries[k][i*10 : i*10+10])
            # Compute susceptible proportion as one minus the sum of infected and recovered
            s_ret = 1 - i_ret - r_ret
            data = {}
            data["iteration"] = [i] * len(t_ret)  # Record the current segment iteration
            data["t"] = t_ret  # Time points from the estimator
            data["S"] = s_ret  # Estimated susceptible proportions
            data["I"] = i_ret  # Estimated infected proportions
            data["R"] = r_ret  # Estimated recovered proportions
            df = pd.DataFrame(data)
            # Concatenate the segment DataFrame with the overall simulation DataFrame
            sim_df = pd.concat([sim_df, df])
        
        # Process the aggregated DataFrame by applying the time_division_float function
        inner_dict[k] = time_division_float(sim_df)
    return inner_dict


def compute_simd_df_divded(dict_queries, G, time_division_float, estimator_func=estimator):
    """
    Compute simulation DataFrames divided by time using the specified estimator.

    This function loops over a set of infection probabilities, and for each infection probability,
    it calls process_infection_prob to process the query data for various neighborhood sizes (k values).
    The final processed DataFrames are stored in a nested dictionary keyed first by infection probability
    and then by k value.
    
    Parameters:
        dict_queries (dict): Nested dictionary of query data, indexed by infection probability and k.
        G (networkx.Graph): The graph, whose node count is used for normalization.
        time_division_float (function): Function to process the final simulation DataFrame.
        estimator_func (function): The estimator function to apply (e.g., HT_estimator or estimator).
    
    Returns:
        dict: A nested dictionary (sim_df_divded) with processed simulation DataFrames.
    """
    sim_df_divded = dict()
    
    # Loop over different infection probability values
    for infection_prob in dict_queries.keys():
        sim_df_divded[infection_prob] = process_infection_prob( dict_queries[infection_prob], G, time_division_float, estimator_func)
    
    return sim_df_divded

# Example usage:
# sim_df_divded = compute_simd_df_divded(dict_queries, G, time_division_float, estimator_func=estimator)

