import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_2d_RGG(G: nx.Graph, pos: dict) -> None:
    """Plot a 2D RGGmetric graph.

    Args:
        G: A networkx Graph object.
        pos: A dictionary mapping node IDs to (x, y) coordinates.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Random RGGmetric graph")
    plt.subplot(1, 2, 2)
    # Get the degrees of all nodes
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 1) - 0.5, density=True, alpha=0.75)
    plt.title("Degree Distribution Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()




def plot_spring_layout(G: nx.Graph) -> None:
    """Plot a spring layout graph.

    Args:
        G: A networkx Graph object.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G) 
    nx.draw_networkx_nodes(G, pos, node_size=70)
    nx.draw_networkx_edges(G, pos, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    # Get the degrees of all nodes
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 1) - 0.5, density=True, alpha=0.75)
    plt.title("Degree Distribution Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()
    plt.show()
        
def plot_adjacenty(trueGraphNp, preGraphsNp=[], campThis= "viridis"):
    maxValue= 0

    for preGraph in preGraphsNp:
        maxTemp= np.max(preGraph)
        maxValue= maxTemp if maxTemp>maxValue else maxValue

    plt.figure(figsize=(6*(len(preGraphsNp)+1), 6))
    plt.subplot(1, len(preGraphsNp)+1, 1)
    plt.imshow(trueGraphNp, cmap= campThis, norm=Normalize(vmin=0, vmax= maxValue))
    colorbar =plt.colorbar(shrink=0.01)
    colorbar.ax.set_axis_off()
    plt.title('Real graph')

    for i in range(len(preGraphsNp)-1):
        plt.subplot(1, len(preGraphsNp)+1, i+2)
        plt.imshow(preGraphsNp[2], cmap= campThis, norm=Normalize(vmin=0, vmax= maxValue))
        plt.title(f'{i+1} agents')
        colorbar =plt.colorbar(shrink=0.01)
        colorbar.ax.set_axis_off()
    
    plt.subplot(1, len(preGraphsNp)+1, len(preGraphsNp)+1)
    plt.imshow(preGraphsNp[-1], cmap= campThis, norm=Normalize(vmin=0, vmax= maxValue))
    plt.title(f'{len(preGraphsNp)} agents')

    plt.colorbar(label='',  shrink=0.4)
    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the plot
    plt.show()
