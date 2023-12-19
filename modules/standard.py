import numpy as np
import networkx as nx
import random
import torch

def generate_random_standard_graph(n, m):
    array_2d = np.zeros((n, n))
    for i in range(m):
        array_2d[n-i-2, n-1-i:]= 1
    for i in range(n-1-m):
        random_int = random.sample(range(i+1, n), m)
        array_2d[i, random_int]= 1

    return nx.Graph(array_2d)

def compress_left_order(graphNumpy):
    arr= np.sum(graphNumpy, axis=0)
    sortedIndices = np.argsort(arr)[::1]
    include= set()
    exclude= set()
    left= set(sortedIndices)
    for i in sortedIndices:
        if i in left:
            exclude.add(i)
            left.remove(i)
            tempInclude= set()
            for j in left:
                if graphNumpy[i, j]==1:
                    include.add(j)
                    tempInclude.add(j)
            for j in tempInclude:
                left.remove(j)
        if not left:
            break
    return include, exclude


def standardize(graph, node= 0, traversal=None):
    if traversal==None:
        traversal = list(nx.dfs_preorder_nodes(graph, node))
    graphNp=nx.to_numpy_array(graph)
    lent= len(traversal)
    n= graphNp.shape[0]-1
    graphNew= np.zeros_like(graphNp)
    nameDic= {}
    for i in range(lent):
        orinInd= traversal[i]
        nameDic[orinInd]= i
        neighbors = set(graph.neighbors(orinInd))&set(traversal[0:i])
        values = [n-nameDic[key] for key in neighbors]
        graphNew[n-i, values]= 1
    return graphNew

def power_m_embedding(standardgraphNP, Precision: int=5, length= 20, m=0.5):
    expandLength= Precision*length
    n= standardgraphNP.shape[0]
    #expand the original matrix to expandLength*expandLength
    expandedM = np.zeros((expandLength, expandLength))
    expandedM[-n:, -n:] = standardgraphNP
    #create sample [0.03125, 0.0625, 0.125, 0.25, 0.5]
    sampleV= [m**(Precision-i) for i in range(Precision)]
    embeddings=[]
    for i in range(expandLength):
        item= expandedM[expandLength-i-1, :]
        reshapedItem= np.resize(item, (length, Precision))
        reshapedItem= reshapedItem*sampleV
        id= np.array([(expandLength-i)/expandLength])
        feature=reshapedItem.sum(1)
        embedding= np.concatenate([id, feature])
        embeddings.append(embedding)
    return np.stack(embeddings)