import numpy as np
import pandas as pd
import math
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
import csv
import random

import time
# Unfairness scores

def get_dict_node_id2idx(G):        
    dict_node_id2idx = {}
    for i,v in enumerate(G.nodes()):
        dict_node_id2idx[v] = i
    return dict_node_id2idx

def get_dict_node_idx2id(G):        
    dict_node_idx2id = {}
    for i,v in enumerate(G.nodes()):
        dict_node_idx2id[i] = v
    return dict_node_idx2id

def unfairness_score(Y, W, node_idx):
    '''
        Calculates the unfairness score for node 'node_idx' where Y is the nxd embedding matrix and W
        is the weighted adjacency matrix. 
        
        The unfairness score is \sum_{j=1}^n |Y_i - Y_j|^2 W[i,j]
    '''
    unfairness = 0.0
    for j in range(len(Y)):
        if W[node_idx][j] == 0:
            continue 
        unfairness += (np.linalg.norm(Y[node_idx] - Y[j])**2)*W[node_idx][j]
    return unfairness 

def unfairness_scores(Y, W):
    return [unfairness_score(Y, W, i) for i in range(len(Y))]

def unfairness_scores_normalized(Y, W, G):
    degrees = [G.degree[node] for node in G.nodes()]
    degree_normalized_scores = [unfairness_score(Y, W, i)/degrees[i] if degrees[i] > 0 
            else 0 
            for i in range(len(Y))]
    return degree_normalized_scores/np.max(degree_normalized_scores)

# k hop InFoRM
def k_hop_InFoRM_score(Y, G, node_idx, nr_hops):
    '''
        Calculates the k-hop InFoRM unfairness score for node 'node_idx' where Y is the nxd embedding matrix and W
        is the weighted adjacency matrix. 
        
        The k-hop InFoRM unfairness score for node u is \sum_{v : v != u and v in N(u,k)} |Y_i - Y_j|^2 
        where N(u,k) are the nodes reachable from u in at most k steps
    '''
    # compute N(node_idx, nr_hops)
    # ids from the nodes are not sorted, so we need to locate the correct node with node_idx
    N_nr_hops = nx.ego_graph(G,list(G.nodes())[node_idx],nr_hops,False).nodes
    unfairness = 0.0
    for v in N_nr_hops:
        # check that embedding matrix is sorted according to node indices
        # shouldn't the norm be squared?
        # get index from node to access embedding matrix 
        v_idx = list(G.nodes()).index(v)
        unfairness += np.linalg.norm(Y[node_idx] - Y[v_idx])**2
    return unfairness 

def k_hop_InFoRM_scores(Y, G, nr_hops):
    return [k_hop_InFoRM_score(Y, G, i, nr_hops) for i in range(len(Y))]

def k_hop_InFoRM_scores_normalized(Y, G, nr_hops):
    degrees = [G.degree[node] for node in G.nodes()]
    degree_normalized_scores = [k_hop_InFoRM_score(Y, G, i, nr_hops)/degrees[i] if degrees[i] > 0 
            else 0 
            for i in range(len(Y))]
    return degree_normalized_scores/np.max(degree_normalized_scores)

# group unfairness score

def recommended_nodes(Y,W,node_idx,k):
    '''
        Computes the set of recommended nodes for node 'node_idx' where Y is the nxd embedding matrix and W
        is the weighted adjacency matrix. 
        
        The set of recommended nodes is given by the top-$k$ most proximal ones in the embedding, using dot product similarity
    '''
    # compute top_k(<Y[u],Y[v]>)
    n = len(W)
    similarities = [(v,np.dot(Y[node_idx],Y[v])) for v in range(n) if (v != node_idx) and (not W[node_idx,v])]
    similarities.sort(key = lambda x : x[1],reverse=True)
    top_k = similarities[0:k]

    rho_u = [v for (v,_) in top_k]

    return rho_u      

def group_unfairness_score(G, Y, W, node_idx, node_features, S, z, k):
    '''
        Calculates the group unfairness score for node 'node_idx' where Y is the nxd embedding matrix, W
        is the weighted adjacency matrix, S is a sensitive attribute and z an attribute value for S. 
        
        The group unfairness score is 1/|Z^S| - z-share(u) where:
            - Z^S is the set of all possible values of attribute S
            - z-share(u) = |rho_z(u)|/|rho(u)|
            - rho(u) is the set of recommended nodes
            - rho_z(u) = {v : v in rho(u) and attr(v,S)=z}
    '''
    rho_u_idx = recommended_nodes(Y,W,node_idx,k)

    # get the number of values of attribute S
    nr_Svalues = len(np.unique(node_features[:,S]))

    # convert list of idx to id
    # dict_node_id2idx = {}
    dict_node_id2idx = get_dict_node_id2idx(G)
    dict_node_idx2id = get_dict_node_idx2id(G)
    rho_u_id = []
    for rec_idx in rho_u_idx:
        rho_u_id.append(dict_node_idx2id[rec_idx])

    # added list(node_features[:,0]).index(v) to avoid out of index  
    # accesses for nodes that do not have an S feature value
    # node_features is not ordered
    
    
    rho_u_z = [v for v in rho_u_id if v in list(node_features[:,0]) and node_features[list(node_features[:,0]).index(dict_node_id2idx[v]),S] == z]  #  attr(v,S) == z

    if rho_u_id == []:
        # check if assigning 0 for this case makes sense
        z_share_u = 0
    else:
        z_share_u = len(rho_u_z)/len(rho_u_id)

    return 1/nr_Svalues - z_share_u

def group_unfairness_scores(G, Y, W, node_features, S, z, k):
    return [group_unfairness_score(G, Y, W, i, node_features, S, z, k) for i in range(len(Y))]

def load_network(G, path_node_features, path_fairness_scores, fairness_notion, params, 
                title="Local Graph Topology", show_scale = True):
    #def load_network(edgelist_file, node_features_file):
    #G = nx.read_edgelist(path)
    W = nx.to_numpy_array(G)
    #node_features = np.loadtxt(open(path_node_features, "rb"), delimiter=",", skiprows=1).astype(int)

    # pos = nx.get_node_attributes(G2,'pos')

    # read in node features 
    node_features = {}
    with open(path_node_features, "r") as featuresCSV:
        #print(featuresCSV.read())
        features_lines = [line.strip().split(",") for line in featuresCSV.readlines()]
        keys = features_lines[0]
        for i in range(1, len(features_lines)):
            single_node_features = {}
            for j in range(len(keys)):
                single_node_features[keys[j]] = features_lines[i][j]
            node_features[single_node_features["id"]] = single_node_features

    edge_x = []
    edge_y = []
    edge_lengths = []
    for edge in G.edges():
        x0, y0 = float(node_features[edge[0]]["pos_x"]), float(node_features[edge[0]]["pos_y"])
        x1, y1 = float(node_features[edge[1]]["pos_x"]), float(node_features[edge[1]]["pos_y"])
        length = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        edge_lengths.append(length)
    edge_length_threshold = np.percentile(np.array(edge_lengths), 90)
    for edge in G.edges():
        x0, y0 = float(node_features[edge[0]]["pos_x"]), float(node_features[edge[0]]["pos_y"])
        x1, y1 = float(node_features[edge[1]]["pos_x"]), float(node_features[edge[1]]["pos_y"])
        length = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if length >= edge_length_threshold:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = float(node_features[node]["pos_x"]), float(node_features[node]["pos_y"])
        node_x.append(x)
        node_y.append(y)

    # standardize coloscale
    if fairness_notion == 'Individual (InFoRM)':
        [val_min, val_max] = [0, 1]
    elif fairness_notion == 'Group (Fairwalk)':
        [val_min, val_max] = [-1, 1]
    else:
        [val_min, val_max] = [0, 1]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=show_scale,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            reversescale=False,
            cmin=val_min,
            cmax=val_max,
            color=[],
            size=5,
            colorbar=dict(
                thickness=17,
                title='Unfairness scores <br> higher is more unfair',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    #node_trace.marker.color = node_adjacencies
    #node_trace.text = node_text
    # distinguish fairness notion and parameters
    if fairness_notion == 'Individual (InFoRM)':
        node_to_score = {}
        with open(path_fairness_scores, "r") as scores_file:
            lines = scores_file.readlines()
            header = lines[0].strip("\n").split(",")
            node_id_idx = header.index("id")
            nr_hops_idx = header.index("nr_hops")
            InFoRM_hops_idx = header.index("InFoRM_hops")
            for i in range(1, len(lines)):
                features = [feature.strip() for feature in lines[i].split(',')]
                if int(features[nr_hops_idx]) == params["nrHops"]:
                    try:
                        node_to_score[features[node_id_idx]] = float(features[InFoRM_hops_idx])
                    except:
                        #print(features)
                        node_to_score[features[node_id_idx]] = 0.0
        scores = [node_to_score[node] for node in G.nodes()]

    else:
        node_to_score = {}
        with open(path_fairness_scores, "r") as scores_file:
            lines = scores_file.readlines()
            header = lines[0].strip("\n").split(",")
            node_id_idx = header.index("node_id")
            attribute_idx = header.index("attribute")
            value_idx = header.index("value")
            k_idx = header.index("k")
            group_fairness_score_idx = header.index("group_fairness_score")
            for i in range(1, len(lines)):
                features = [feature.strip() for feature in lines[i].split(',')]
                if features[attribute_idx] == params["attribute"] and\
                    features[value_idx] == params["value"] and\
                    features[k_idx] == str(params["k"]):
                    try:
                        node_to_score[features[node_id_idx]] = float(features[group_fairness_score_idx])
                    except:
                        #print(features)
                        node_to_score[features[node_id_idx]] = 0.0

        scores = [node_to_score[node] for node in G.nodes()]

    node_text = [" node id: {} <br> unfairness score: {} "
                .format(n, round(scores[i],2)) for i,n in enumerate(G.nodes())]
    node_trace.marker.color = scores
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def get_statistical_summary(G):
    # computes the statistical summary of the graph G
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    density = nx.density(G)
    number_of_triangles = int(sum(nx.triangles(G).values()) / 3)
    avg_clust_coeff = nx.average_clustering(G)
    return n,m,density,number_of_triangles,avg_clust_coeff

def get_edgelist_file(networkName):
    name_to_edgelist = {"Facebook": "facebook_combined.edgelist",
                        "protein-protein": "ppi.edgelist",
                        "AutonomousSystems": "AS.edgelist",
                        "ca-HepTh": "ca-HepTh.edgelist",
                        "LastFM": "lastfm_asia_edges.edgelist",
                        "wikipedia": "wikipedia.edgelist"}
    if networkName in name_to_edgelist:
        return name_to_edgelist[networkName]
    else:
        return name_to_edgelist["Facebook"]