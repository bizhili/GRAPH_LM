import networkx as nx
import numpy as np

def eigenvector_centrality_embedding(G):
    centrality= nx.eigenvector_centrality(G, 500)
    centralityNp=np.array(list(centrality.values()))[::-1]
    return centralityNp


def closeness_centrality_embedding(G):
    centrality= nx.closeness_centrality(G)
    centralityNp=np.array(list(centrality.values()))[::-1]
    return centralityNp


def degree_centrality_embedding(G):
    centrality= nx.degree_centrality(G)
    centralityNp=np.array(list(centrality.values()))[::-1]
    return centralityNp

def load_centrality_embedding(G):
    centrality= nx.load_centrality(G)
    centralityNp=np.array(list(centrality.values()))[::-1]
    return centralityNp

def betweenness_centrality_embedding(G):
    centrality= nx.betweenness_centrality(G)
    centralityNp=np.array(list(centrality.values()))[::-1]
    return centralityNp