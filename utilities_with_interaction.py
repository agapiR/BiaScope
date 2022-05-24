from enum import unique
from genericpath import exists
from platform import node
from matplotlib import markers
import numpy as np
import math
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas
import re


import time


def draw_network(G, scores, focal, fairness_notion='Individual (InFoRM)',
                attributes = None, attribute_type = None, 
                title="Local Graph Topology",
                selection_local = None, selectedpoints = None):

    nodePos = nx.spring_layout(G, seed=42) # added seed argument for layout reproducibility

    edge_x = []
    edge_y = []
    edge_lengths = []
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
        length = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        edge_lengths.append(length)
    # for large and dense networks, show the longest 5% edges
    if nx.density(G)>0.8 and nx.number_of_nodes(G)>25:  # plot at most 25^2 * 0.8 = 500 edges
        edge_length_threshold = np.percentile(np.array(edge_lengths), 95)
    else:
        edge_length_threshold = 0 #np.min(np.array(edge_lengths))
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
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
        x, y = nodePos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        selectedpoints=selectedpoints,
        unselected={'marker': { 'opacity': 0.3 }},
        selected={'marker': { 'opacity': 1 }},
        marker=dict(
            size=11,
            line_width=2))

    # Focal Node pop-out
    # focal is an 1-hot vector
    node_trace.marker.line['color'] = ['#de2d26' if f==1 else '#696969' for f in focal]
    node_trace.marker.line['width'] = [4 if f==1 else 2 for f in focal]
    node_trace.marker.size = [18 if f==1 else 11 for f in focal]

    # color encoding
    if attributes:  # if attributes exist, color by attribute
        # choose colormap
        if fairness_notion == 'Individual (InFoRM)':
            # sequencial: https://colorbrewer2.org/#type=sequential&scheme=Reds&n=5
            if len(set(attributes))==2: # for 1-hop 
                colormap = ['#a50f15', '#fcae91',]
            else: # for 2-hop or more
                colormap = ['#a50f15', '#fcae91', '#fee5d9']
        elif fairness_notion == 'Group (Fairwalk)':
            # categorical: https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=7
            # supports only binary attributes
            colormap = ['#ffff99', '#386cb0'] #['#386cb0', '#beaed4', '#7fc97f','#fdc086','#ffff99']
            if len(set(attributes))==1: # for 1 attribute
                singlecolor = colormap[attributes[0]]
        if len(set(attributes))==1:
            node_trace.marker.color = singlecolor
        else:
            node_trace.marker.color = attributes
            node_trace.marker.colorscale = colormap
            node_trace.marker.showscale = False
            node_trace.marker.reversescale = False
    else:   # else color by score
        # standardize coloscale
        if fairness_notion == 'Individual (InFoRM)':
            [val_min, val_max] = [0, 1]
        elif fairness_notion == 'Group (Fairwalk)':
            [val_min, val_max] = [-1, 1]
        else:
            [val_min, val_max] = [0, 1]
            node_trace.marker.color = scores
            node_trace.marker.colorscale = 'Reds'
            node_trace.marker.showscale = False
            node_trace.marker.cmin = val_min
            node_trace.marker.cmax = val_max

    # node info text
    if attributes:
        node_text = [" node id: {} <br> unfairness score: {} <br> {}: {}"
                    .format(n, scores[i], attribute_type, attributes[i]) 
                    for i,n in enumerate(G.nodes())]
    else:
        node_text = [" node id: {} <br> unfairness score: {}"
                    .format(n, scores[i]) 
                    for i,n in enumerate(G.nodes())]
    node_trace.text = node_text

    # configure viz layout
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    dragmode='select',
                    #clickmode='select',
                    uirevision=True,
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.update_layout(legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ))

    # draw brush
    if selection_local and selection_local['range']:
        ranges = selection_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
        bound_width = 1
    else:
        selection_bounds = {'x0': 0, 'x1': 0,
                            'y0': 0, 'y1': 0}
        bound_width = 0

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': bound_width,'dash': 'dot','color': 'darkgrey'} },
                       **selection_bounds))

    return fig




def draw_embedding_2dprojection(G, projections, scores, focal, fairness_notion='Individual (InFoRM)',
                    attributes = None, attribute_type = None, 
                    type="TSNE",
                    selection_local = None, selectedpoints = None):

    # if type=="TSNE":
    #     tsne = TSNE(n_components=2, random_state=0)
    #     projections = tsne.fit_transform(embedding)
    # elif type=="UMAP":
    #     umap_2d = UMAP(n_components=2, init='random', random_state=0)
    #     projections = umap_2d.fit_transform(embedding)

    mark_trace = go.Scatter(
        x=list(projections[0]), y=list(projections[1]),
        mode='markers',
        hoverinfo='text',
        selectedpoints=selectedpoints,
        unselected={'marker': { 'opacity': 0.3 }},
        selected={'marker': { 'opacity': 1 }},
        marker=dict(
            size=11,
            line_width=2))

    # Focal Node pop-out
    # focal is an 1-hot vector
    mark_trace.marker.line['color'] = ['#de2d26' if f==1 else '#696969' for f in focal]
    mark_trace.marker.line['width'] = [4 if f==1 else 2 for f in focal]
    mark_trace.marker.size = [18 if f==1 else 11 for f in focal]

    # color encoding
    if attributes:  # if attributes exist, color by attribute
        # choose colormap
        if fairness_notion == 'Individual (InFoRM)':
            # sequencial: https://colorbrewer2.org/#type=sequential&scheme=Reds&n=5
            if len(set(attributes))==2: # for 1-hop 
                colormap = ['#a50f15', '#fcae91',]
            else: # for 2-hop or more
                colormap = ['#a50f15', '#fcae91', '#fee5d9']
        elif fairness_notion == 'Group (Fairwalk)':
            # categorical: https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=7
            # supports only binary attributes
            colormap = ['#ffff99', '#386cb0'] #['#386cb0', '#beaed4', '#7fc97f','#fdc086','#ffff99']
            if len(set(attributes))==1: # for 1 attribute
                singlecolor = colormap[attributes[0]]
        if len(set(attributes))==1:
            mark_trace.marker.color = singlecolor
        else:
            mark_trace.marker.color = attributes
            mark_trace.marker.colorscale = colormap
            mark_trace.marker.showscale = False
            mark_trace.marker.reversescale = False
    else:   # else color by score
        # standardize coloscale
        if fairness_notion == 'Individual (InFoRM)':
            [val_min, val_max] = [0, 1]
        elif fairness_notion == 'Group (Fairwalk)':
            [val_min, val_max] = [-1, 1]
        else:
            [val_min, val_max] = [0, 1]
            mark_trace.marker.color = scores
            mark_trace.marker.colorscale = 'Reds'
            mark_trace.marker.showscale = False
            mark_trace.marker.cmin = val_min
            mark_trace.marker.cmax = val_max

    # node info text
    mark_text = [" node id: {} <br> unfairness score: {} <br> {}: {}"
                .format(n, scores[i], attribute_type, attributes[i]) 
                for i,n in enumerate(G.nodes())]
    mark_trace.text = mark_text

    delta_x = np.abs(np.max(projections[0]) - np.min(projections[0]))
    delta_y = np.abs(np.max(projections[1]) - np.min(projections[1]))
    padding_x = delta_x*0.05
    padding_y = delta_y*0.05
    padding = max(padding_x, padding_y)
    fig = go.Figure(data=mark_trace,
                layout=go.Layout(
                    title="2D projection of node embeddings ({})".format(type),
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    dragmode='select',
                    #clickmode='select',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=True, zeroline=False, showticklabels=True, range=[np.min(projections[0])-padding, np.max(projections[0])+padding]),
                    yaxis=dict(showgrid=True, zeroline=False, showticklabels=True, range=[np.min(projections[1])-padding, np.max(projections[1])+padding])
                    )
                )

    if selection_local and selection_local['range']:
        ranges = selection_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
        bound_width = 1
    else:
        selection_bounds = {'x0': 0, 'x1': 0,
                            'y0': 0, 'y1': 0}
        bound_width = 0

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': bound_width,'dash': 'dot','color': 'darkgrey'} },
                       **selection_bounds))

    return fig


def draw_2d_scale(array2d_outer, array2d_inner, show_inner=False):
    
    # outer rectangle corners
    X_min = np.min(array2d_outer[0])
    X_max = np.max(array2d_outer[0])
    Y_min = np.min(array2d_outer[1])
    Y_max = np.max(array2d_outer[1])

    # inner rectangle corners
    x_min = np.min(array2d_inner[0])
    x_max = np.max(array2d_inner[0])
    y_min = np.min(array2d_inner[1])
    y_max = np.max(array2d_inner[1])

    delta_x = np.abs(x_min - x_max)
    delta_y = np.abs(y_min - y_max)
    padding_x = delta_x*0.05
    padding_y = delta_y*0.05
    padding = max(padding_x, padding_y)

    fig = go.Figure(go.Scatter(
        x=[x_min-padding, x_min-padding, x_max+padding, x_max+padding, x_min-padding], 
        y=[y_min-padding, y_max+padding, y_max+padding, y_min-padding, y_min-padding], 
        fill="toself",
        name = "displayed area",
        hoverinfo='none',
        fillcolor = 'rgba(255, 0, 0, 0.1)',
        marker=dict(
            size=0.3,
            line_width=0,
            color='rgba(255, 0, 0, 1)')
            )
        )

    # fig.update_xaxes(range=[X_min, X_max])
    # fig.update_yaxes(range=[Y_min, Y_max])

    # Create scatter trace of the outer array points
    fig.add_trace(go.Scatter(
        x=array2d_outer[0], y=array2d_outer[1],
        mode='markers',
        hoverinfo='none',
        name = "all points",
        marker=dict(
            size=1,
            line_width=0,
            color= '#000000' )
            )
        )

    # Create scatter trace of the selected array points, if given.
    if show_inner:
        fig.add_trace(go.Scatter(
            x=array2d_inner[0], y=array2d_inner[1],
            mode='markers',
            hoverinfo='none',
            name = "displayed points",
            marker=dict(
                size=2,
                line_width=0,
                color='rgba(255, 0, 0, 1)')
                )
            )

    
    fig.layout=go.Layout(
                title="",
                titlefont_size=16,
                paper_bgcolor='rgb(233, 233, 233,0.3)',
                #plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=True, zeroline=False, showticklabels=True, range=[X_min, X_max]),
                yaxis=dict(showgrid=True, zeroline=False, showticklabels=True, range=[Y_min, Y_max])
                )

    return fig


def get_scores(fairness_notion, params, path_fairness_scores):
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
    else:
        node_to_score = {}
        with open(path_fairness_scores, "r") as scores_file:
            lines = scores_file.readlines()
            header = lines[0].strip("\n").split(",")
            node_id_idx = header.index("id")
            attribute_idx = header.index("attribute")
            value_idx = header.index("value")
            k_idx = header.index("k")
            group_fairness_score_idx = header.index("group_fairness_score")
            for i in range(1, len(lines)):
                features = [feature.strip() for feature in lines[i].split(',')]
                if features[attribute_idx] == params["attribute"] and\
                    features[value_idx] == params["value"] and\
                    features[k_idx] == params["k"]:
                    try:
                        node_to_score[features[node_id_idx]] = float(features[group_fairness_score_idx])
                    except:
                        #print(features)
                        node_to_score[features[node_id_idx]] = 0.0

    return node_to_score

def get_node_features(path_node_features):
    '''read in node features''' 
    node_features = {}
    with open(path_node_features, "r") as featuresCSV:
        features_lines = [line.strip().split(",") for line in featuresCSV.readlines()]
        keys = features_lines[0]
        for i in range(1, len(features_lines)):
            single_node_features = {}
            for j in range(len(keys)):
                single_node_features[keys[j]] = features_lines[i][j]
            node_features[single_node_features["id"]] = single_node_features

    return node_features

def get_recommended_nodes(path_recommended_nodes):
    node_to_rec = {}
    with open(path_recommended_nodes, "r") as rec_file:
        lines = rec_file.readlines()
        header = lines[0].strip("\n").split(",")
        node_id_idx = header.index("id")
        k_idx = header.index("k")
        rec_list_idx = header.index("recommended_nodes")
        for i in range(1, len(lines)):
            features = [re.sub('\W+','', feature).strip() for feature in lines[i].split(',')]
            for rec_idx  in range(rec_list_idx-1, len(features)-1): # file has a redundant "," at the end of line
                try:
                    node_to_rec[features[node_id_idx]].append(features[rec_idx])
                except:
                    node_to_rec[features[node_id_idx]] = []
    return node_to_rec

def get_egoNet(G, node, k=1):
    '''Returns the k-hop ego net of a node, from node index.
    k: max distance of neighbors from node.'''
    tic = time.perf_counter()

    ego_net = nx.ego_graph(G, node, radius=k)

    toc = time.perf_counter()
    #print(f"Calculated the ego net in {toc - tic:0.4f} seconds")

    return ego_net,[idx for idx in ego_net.nodes()]

def get_induced_subgraph(G, node_list):
    '''Returns the induced subgraph, from a list of node indeces.'''
    tic = time.perf_counter()

    subgraph = nx.induced_subgraph(G, node_list)

    toc = time.perf_counter()
    #print(f"Calculated the induced subgraph in {toc - tic:0.4f} seconds")

    return subgraph,[idx for idx in subgraph.nodes()]

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
    node_to_score = get_scores(fairness_notion, params, path_fairness_scores)
    # if fairness_notion == 'Individual (InFoRM)':
    #     node_to_score = {}
    #     with open(path_fairness_scores, "r") as scores_file:
    #         lines = scores_file.readlines()
    #         header = lines[0].strip("\n").split(",")
    #         node_id_idx = header.index("id")
    #         nr_hops_idx = header.index("nr_hops")
    #         InFoRM_hops_idx = header.index("InFoRM_hops")
    #         for i in range(1, len(lines)):
    #             features = [feature.strip() for feature in lines[i].split(',')]
    #             if int(features[nr_hops_idx]) == params["nrHops"]:
    #                 try:
    #                     node_to_score[features[node_id_idx]] = float(features[InFoRM_hops_idx])
    #                 except:
    #                     print(features)
    #                     node_to_score[features[node_id_idx]] = 0.0
    #     scores = [node_to_score[node] for node in G.nodes()]

    # else:
    #     node_to_score = {}
    #     with open(path_fairness_scores, "r") as scores_file:
    #         lines = scores_file.readlines()
    #         header = lines[0].strip("\n").split(",")
    #         node_id_idx = header.index("node_id")
    #         attribute_idx = header.index("attribute")
    #         value_idx = header.index("value")
    #         k_idx = header.index("k")
    #         group_fairness_score_idx = header.index("group_fairness_score")
    #         for i in range(1, len(lines)):
    #             features = [feature.strip() for feature in lines[i].split(',')]
    #             if features[attribute_idx] == params["attribute"] and\
    #                 features[value_idx] == params["value"] and\
    #                 features[k_idx] == str(params["k"]):
    #                 try:
    #                     node_to_score[features[node_id_idx]] = float(features[group_fairness_score_idx])
    #                 except:
    #                     print(features)
    #                     node_to_score[features[node_id_idx]] = 0.0

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
                    #dragmode='select',
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
