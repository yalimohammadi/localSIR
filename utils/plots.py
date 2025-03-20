import matplotlib.pyplot as plt
import seaborn as sns

def plot_degree_distribution(G, bins=None):
    """
    Plot the degree distribution of a graph.

    Parameters:
        G (networkx.Graph): The graph whose degree distribution to plot.
        bins (int or sequence, optional): Number of bins or bin edges for the histogram.
    """
    # Get the degree for each node
    degree_sequence = [d for n, d in G.degree()]
    
    # If bins are not provided, create bins spanning the degree range
    if bins is None:
        bins = range(min(degree_sequence), max(degree_sequence) + 2)
    
    # Create the histogram
    sns.histplot(degree_sequence, stat='probability', bins=bins)
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title('Degree Distribution')
    plt.show()

# Example usage:
# plot_degree_distribution(G)
