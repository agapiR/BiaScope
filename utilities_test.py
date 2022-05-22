import numpy as np
import math
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
import csv

from utilities import recommended_nodes, group_unfairness_score, k_hop_InFoRM_scores_normalized, unfairness_scores_normalized

############ Group unfairness tests

# load example graph
path = 'demos/data/example3.edgelist'
# G = nx.read_edgelist(path) #ignores isolated vertices?
# W = nx.to_numpy_array(G)
W = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,1,0]])
G = nx.from_numpy_matrix(W)
print(G)
Y = np.array([[0,0],[0,2],[0,1],[0,4],[1,3]])
node_features = np.loadtxt(open("demos/data/example3_node_features.csv", "rb"), delimiter=",", skiprows=1).astype(int) #.sort(key = lambda x : x[0])
k=1
for i in range(5):
    rho_i = recommended_nodes(Y,W,i,k)
    print('rho_'+str(i),rho_i)
rho_1 = recommended_nodes(Y,W,1,k)
unfairness_0 = group_unfairness_score(Y, W, 0, node_features, 1, 0, k)
print('unfairness_0',unfairness_0)

for i in range(5):
    unfairness_i = group_unfairness_score(Y, W, i, node_features, 1, 0, k)
    print('unfairness_'+str(i),unfairness_i)

# get attributes
attrs = []
with open('demos/data/example3_node_features.csv', newline='') as f:
  reader = csv.reader(f)
  for row in reader:
    attrs = row[1:]
    break

# print('attrs',attrs)

##################################################

############ InFoRM unfairness tests

inform_k1 = k_hop_InFoRM_scores_normalized(Y, G, 1)
inform_prev = unfairness_scores_normalized(Y, W, G)

print('inform_k1',inform_k1)
print('inform_prev',inform_prev)