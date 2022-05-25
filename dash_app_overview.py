# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys

from matplotlib.pyplot import scatter
from dash import Dash, html, dcc, Input, Output, State, callback, callback_context
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px
import csv
import json
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc



sys.path.insert(0, './../')
from utilities import (unfairness_scores,
                        unfairness_scores_normalized,
                        load_network,
                        get_statistical_summary,
                        get_edgelist_file)

sys.path.insert(0, "description-txt/")
import overview_description
import overview_conclusion

app = Dash(__name__)
app.title = "InfoViz-Final"

overview_layout = html.Div(
    children=[
        overview_description.description,
        html.P("Begin by selecting the network/graph (see 'Data') to analyze along with a fairness notion (See 'Fairness of Graph Embeddings'). "),
        html.Div([
                    html.Div([
                        html.H3('Network:'),
                    ],style={'align-items':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block'}),
                    html.Div([
                        #dcc.Dropdown(options=['Example 1','Example 2'], value='Example 1', id='networkDropdown',style={'width':'160px','align-items':'center'}),
                        dcc.Dropdown(options=['Facebook','protein-protein', 'LastFM', 'wikipedia'], value='Facebook', id='networkDropdown', clearable=False, style={'width':'150px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '0px'}),
                ],style=dict(display='flex')),
                html.Div( children = [
                    html.H2(children='Statistical summary of the network', style={'padding': '10px'}),
                    html.P("The table on the left provides key summary statistics for the network chosen and the bar chart on the right \
                        is a histogram of the number of edges connected to each node, also known as a degree distribution."),
                    html.Div(
                        children = [
                            html.Div([
                                html.Div(id='overviewContainerStats1',
                                children = [
                                        html.Table([
                                            html.Tr([html.Th('Variable'), html.Th('Value')]),
                                            html.Tr([html.Td('Number of Nodes'), html.Td(id='nodes')]),
                                            html.Tr([html.Td('Number of Edges'), html.Td(id='edges')]),
                                            html.Tr([html.Td(['Edge Density ',
                                                    html.Span(
                                                        html.Span("Number of edges / number of possible edges", className="tooltiptext"),
                                                        className="fa fa-question-circle tooltip")
                                                ]), html.Td(id='density')]),
                                            html.Tr([html.Td(['Number of Trianges ',
                                                    html.Span(
                                                        html.Span("Total number of triangles formed by edges", className="tooltiptext"),
                                                        className="fa fa-question-circle tooltip")
                                                ]), html.Td(id='triangles')]),
                                            html.Tr([html.Td(['Average Clustering Coef. ',
                                                    html.Span(
                                                        html.Span("A measure between 0 (less clustered) and 1 (more clustered) indicating how often neighbors of a node \
                                                            are connected.", className="tooltiptext"),
                                                        className="fa fa-question-circle tooltip")
                                                ]), html.Td(id='clustcoeff')])
                                        ], style={'width':'60%', 'border': '1px solid'})
                                ],style={'opacity':'1', 'paddingBottom': '70px','paddingLeft': '100px'}),
                            ])
                        ],
                        className="title six columns",
                        style={'width': '49%', 'display': 'inline-block'}
                    ),
                    html.Div(
                        children = [
                            html.Div(id='overviewContainerStats2',
                            children = [
                                dcc.Loading(
                                    id="loading-graph-overviewContainerStats2",
                                    type="circle",
                                    color="#800020",
                                    children=[]
                                )
                            ],
                            style={'paddingBottom': '0px'}),
                        ],
                        className="title six columns",
                        style={'width': '49%', 'display': 'inline-block'}
                    )
        ]),
        html.Div([
            html.Div([
                html.H3('Fairness notion:'),
            ],style={'align-items':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block', 'margin-left': '0px'}),
            html.Div([
                dcc.Dropdown(options=[], value='Individual (InFoRM)', id='fairnessNotion', clearable=False, style={'width':'160px','align-items':'center'}),
            ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '0px'})
        ],style={'display': 'block'}),
        html.Div([
            html.H3('Fairness parameters'),
            # div for fairness parameters - initialization
            html.Div(children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.H4('Number of hops:'),
                        ],style={'display': 'inline-block','align-items':'center','padding-right': '10px'}),
                        html.Span(
                            html.Span("Radius (in number of edges) of neighborhood", className="tooltiptext"),
                            className="fa fa-question-circle tooltip",
                            style={'display': 'inline-block'}),
                        html.Div([
                            dcc.Slider(id='nrHops', min=1, max=2, step=1, value=1)
                        ],style={'width':'150px','height':'11px','display': 'inline-block','padding-left': '5px'}),
                    ],id='indFairnessParams',style={'display': 'none'}),
                    html.Div([
                        html.Div([
                            html.H4('Node attribute:'),
                        ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                        html.Span(
                            html.Span("Sensitive attribute used to evaluate group fairness", className="tooltiptext"),
                            className="fa fa-question-circle tooltip"),
                        html.Div([
                            dcc.Dropdown(options=[], value='', id='sensitiveAttr', clearable=False, style={'width':'200px','align-items':'center'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','paddingLeft': '5px'}),
                        html.Div([
                            html.H4('Attribute value:'),
                        ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                        html.Span(
                        html.Span("Define group fairness based on recommendations to nodes with this attribute value", className="tooltiptext"),
                        className="fa fa-question-circle tooltip"),
                        html.Div([
                            dcc.RadioItems(options=[1,2], value=1, id='sensitiveAttrVal',inline=False,labelStyle={'display': 'block'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '21px'}),
                        html.Div([
                            html.H4('Value of k:'),
                        ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                        html.Span(
                            html.Span("Number of nodes recommended to each target node", className="tooltiptext"),
                            className="fa fa-question-circle tooltip"),
                        html.Div([
                            dcc.Slider(id='kVal', min=1, max=4, step=1, value=1)
                        ],style={'width':'150px','height':'11px','display': 'inline-block','vertical-align': 'middle','padding-top': '0px'})
                    ],id='groupFairnessParams',style={'display': 'none'})
                ])
            ],id='fairnessParams'),
        ],style={'align-items':'center','padding-right': '10px','padding-top': '10px','display': 'inline-block'}
                ),
        html.P("Select two different embedding algorithms to evaluate with the fairness notion. In both views, the network is \
            drawn with nodes colored by fairness score where lighter if fair and darker is unfair. In both plots, the nodes \
            are drawn in the same positions. To investigate why nodes are unfair for each plot, click 'Diagnose'"),
        html.Div([
                    html.Div([
                        html.H4('Embedding Algorithm:'),
                    ],style={'align-items':'center','paddingRight': '10px','paddingTop': '5px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Node2Vec','HOPE', 'HGCN', 'LaplacianEigenmap', 'SDNE', 'SVD'], value='Node2Vec', id='embDropdownLeft', clearable=False, style={'width':'160px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','paddingTop': '0px'}),
                    html.Div([
                        html.H4('Embedding Algorithm:'),
                    ],style={'align-items':'center', 'padding-left': '500px', 'padding-right': '10px','paddingTop': '5px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Node2Vec','HOPE', 'HGCN', 'LaplacianEigenmap', 'SDNE', 'SVD'], value='HOPE', id='embDropdownRight', clearable=False, style={'width':'160px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','paddingTop': '0px'})
                ],style=dict(display='flex')
                ),
        html.Div(
            children = [
                html.Div([
                    html.Div(id='overviewContainerLeft',
                    children = [
                        dcc.Loading(
                            id="loading-graph-overviewContainerLeft",
                            type="circle",
                            color="#800020",
                            children=[],
                        )
                    ],
                    style={'paddingTop': '50px','paddingBottom': '50px','paddingRight': '90px','paddingLeft': '90px'}),
                    html.Div([html.Center(html.A(href="/diagnostics", children="Diagnose", id= "redirectLeft", className="button"))],style={'paddingBottom': '50px'})
                ]),
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div(
            children = [
                html.Div(id='overviewContainerRight',
                children = [
                    dcc.Loading(
                        id="loading-graph-overviewContainerRight",
                        type="circle",
                        color="#800020",
                        children=[],
                    )
                ],
                style={'paddingTop': '50px','paddingBottom': '50px','paddingRight': '25px','paddingLeft': '25px'}),
                html.Div([html.Center(html.A(href="/diagnostics", children="Diagnose", id= "redirectRight", className="button"))])
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        dcc.Store(id='graph-data-overview', data=[], storage_type='memory'),
        overview_conclusion.conclusion_div
    ]
)
app.layout = overview_layout

# callbacks

# Store 'graph-data'
@callback(
    Output('graph-data-overview', 'data'),
    Input('networkDropdown', 'value')
)
def store_graph_data_overview(network_name):
    # Configure data sources
    edgelist_file = "edgelists/{}".format(get_edgelist_file(network_name))
    # load new network 
    G = nx.read_edgelist(edgelist_file)

    return nx.to_dict_of_dicts(G)

# callback for parameter passing configuration
@callback(
    Output('redirectRight', 'href'),
    Output('redirectLeft', 'href'),
    [Input('networkDropdown', 'value'),
    Input('embDropdownRight', 'value'),
    Input('embDropdownLeft', 'value'),]
)
def set_parameter_passing_ulr(networkDropdown, embDropdownRight, embDropdownLeft):
    href_right = "/diagnostics?net={}&emb={}".format(networkDropdown,embDropdownRight)
    href_left = "/diagnostics?net={}&emb={}".format(networkDropdown,embDropdownLeft)

    return href_right,href_left

# @callback(
#     Output('network-name', 'data'),
#     Output('embedding-name', 'data'),
#     State('networkDropdown', 'value'),
#     State('embDropdownRight', 'value'),
#     State('embDropdownLeft', 'value'),
#     Input('redirectRight', 'n_clicks'),
#     Input('redirectLeft', 'n_clicks'),
# )
# def set_parameter_passing(network_name, embedding_right, embedding_left, 
#                                 button_right, button_left):
    
#     # Identify callback source
#     ctx = callback_context
#     trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
#     print(trigger_id)
#     if trigger_id=='redirectRight':
#         embedding_name = embedding_right
#         #print("Clicked Right! Net : " + network_name + " Emb : " + embedding_name)
#     elif trigger_id=='redirectLeft':
#         embedding_name = embedding_left
#         #print("Clicked Left! Net : " + network_name + " Emb : " + embedding_name)

#     print("from")
#     print(network_name)
#     print(embedding_name)
#     return network_name, embedding_name



@callback(
    Output('fairnessNotion', 'options'),
    Output('fairnessNotion', 'value'),
    # individual fairness
    Output('nrHops', 'value'),
    Output('indFairnessParams', 'style'),
    # group fairness
    Output('sensitiveAttr', 'options'),
    Output('sensitiveAttr', 'value'),
    Output('sensitiveAttrVal', 'options'),
    Output('sensitiveAttrVal', 'value'),
    Output('kVal', 'max'),
    Output('kVal', 'value'),
    Output('groupFairnessParams', 'style'),
    [Input('networkDropdown', 'value'),
    Input('fairnessNotion', 'value')]
)
def display_fairness_parameters(networkDropdown, fairnessNotion):
    # print('display_fairness_parameters')
    # get path to selected network
    nr_hops_options_value = 1 
    ind_fairness_params_style = {'display': 'none'}
    fairness_notions_val = fairnessNotion
    fairness_notions = []
    if networkDropdown == "Facebook":
        fairness_notions = ['Individual (InFoRM)','Group (Fairwalk)']
    else:
        fairness_notions = ['Individual (InFoRM)']
    
    ctx = callback_context
    if ctx.triggered[0]['prop_id'] in ['networkDropdown.value','.'] :
        # restore state of parameters - default is Individual (InFoRM) with nr_hops = 1
        nr_hops_options_value = 1
        ind_fairness_params_style = {'display': 'inline-block','padding-top': '3px'}
        fairness_notions_val = 'Individual (InFoRM)'
        group_fairness_params_style = {'display': 'none'}
        # group fairness
        sensitive_attr_options = []
        sensitive_attr_value = ''
        sensitive_attr_val_options = []
        sensitive_attr_val_value = ''
        k_val_max = 1
        k_val_value = 1
        group_fairness_params_style = {'display': 'none'} 
    else:
        if fairnessNotion == 'Group (Fairwalk)':
            # get sensitive attributes
            config_file = "embeddings/{}/group_fairness_config.json".format(networkDropdown)
            with open(config_file, "r") as configFile:
                group_fairness_config = json.load(configFile)

            nr_hops_options_value = 1 
            ind_fairness_params_style = {'display': 'none'}

            sensitive_attr_options = group_fairness_config["sensitive_attrs"]
            sensitive_attr_value = group_fairness_config["sensitive_attrs"][0]
            sensitive_attr_val_options = group_fairness_config["sensitive_attrs_vals"]
            sensitive_attr_val_value = group_fairness_config["sensitive_attrs_vals"][0]
            k_val_max = max([int(e) for e in group_fairness_config["k_s"]])
            k_val_value = int(group_fairness_config["k_s"][0])
            group_fairness_params_style = {'display': 'inline-block'}

        else:
            nr_hops_options_value = 1 
            ind_fairness_params_style = {'display': 'inline-block','vertical-align': 'middle','padding-top': '3px'}

            sensitive_attr_options = []
            sensitive_attr_value = ''
            sensitive_attr_val_options = []
            sensitive_attr_val_value = ''
            k_val_max = 1
            k_val_value = 1
            group_fairness_params_style = {'display': 'none'}

    return fairness_notions, fairness_notions_val, nr_hops_options_value, ind_fairness_params_style, sensitive_attr_options, sensitive_attr_value, sensitive_attr_val_options, sensitive_attr_val_value, k_val_max, k_val_value, group_fairness_params_style

@callback(
    Output('loading-graph-overviewContainerLeft', 'children'),
    [Input('graph-data-overview', 'data'),
    Input('networkDropdown', 'value'),
    Input('embDropdownLeft', 'value'),
    Input('fairnessNotion', 'value'),
    Input('sensitiveAttr', 'value'),
    Input('sensitiveAttrVal', 'value'),
    Input('kVal', 'value'),
    Input('nrHops', 'value')]
)
def update_network1_fairness(graph_data, networkDropdown, embDropdownLeft, fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):
    edgelist_file = "edgelists/{}".format(get_edgelist_file(networkDropdown))
    node_features_file = "embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv".format(networkDropdown, 
                                                                                    embDropdownLeft, 
                                                                                    networkDropdown, 
                                                                                    embDropdownLeft)
    
    path_fairness_score = ""
    if fairnessNotion == 'Individual (InFoRM)':
        params = {"nrHops": nrHops}
        # currently the same as node features file
        path_fairness_score = "embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv".format(networkDropdown, 
                                                                                    embDropdownLeft, 
                                                                                    networkDropdown, 
                                                                                    embDropdownLeft)
    else: #Group (Fairwalk)
        params = {"attribute": sensitiveAttr, "value": sensitiveAttrVal, "k": kVal}
        path_fairness_score = "embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv".format(networkDropdown, 
                                                                                            embDropdownLeft, 
                                                                                            networkDropdown, 
                                                                                            embDropdownLeft)
    G = nx.from_dict_of_dicts(graph_data)
    fig = load_network(G, node_features_file, path_fairness_score, fairnessNotion, params, title=" ", show_scale=False)

    graph = dcc.Graph(
                    id='overview-graph-left',
                    figure=fig,
                    style={"width": "110%"}
                )
    return graph

@callback(
    Output('loading-graph-overviewContainerRight', 'children'),
    [Input('graph-data-overview', 'data'),
    Input('networkDropdown', 'value'),
    Input('embDropdownRight', 'value'),
    Input('fairnessNotion', 'value'),
    Input('sensitiveAttr', 'value'),
    Input('sensitiveAttrVal', 'value'),
    Input('kVal', 'value'),
    Input('nrHops', 'value')]
)
def update_network2_fairness(graph_data, networkDropdown, embDropdownRight, fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):
    edgelist_file = "edgelists/{}".format(get_edgelist_file(networkDropdown))
    node_features_file = "embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv".format(networkDropdown, 
                                                                                            embDropdownRight, 
                                                                                            networkDropdown, 
                                                                                            embDropdownRight)
    
    path_fairness_score = ""
    if fairnessNotion == 'Individual (InFoRM)':
        params = {"nrHops": nrHops}
        # currently the same as node features file
        path_fairness_score = "embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv".format(networkDropdown, 
                                                                                            embDropdownRight, 
                                                                                            networkDropdown, 
                                                                                            embDropdownRight)
    else: #Group (Fairwalk)
        params = {"attribute": sensitiveAttr, "value": sensitiveAttrVal, "k": kVal}
        path_fairness_score = "embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv".format(networkDropdown, 
                                                                                            embDropdownRight, 
                                                                                            networkDropdown, 
                                                                                            embDropdownRight)
    
    G = nx.from_dict_of_dicts(graph_data)
    fig = load_network(G, node_features_file, path_fairness_score, fairnessNotion, params, title=" ", show_scale=True)

    graph = dcc.Graph(
                    id='overview-graph-right',
                    figure=fig,
                )
    return graph

@callback(
    Output('nodes', 'children'),
    Output('edges', 'children'),
    Output('density', 'children'),
    Output('triangles', 'children'),
    Output('clustcoeff', 'children'),
    Output('loading-graph-overviewContainerStats2', 'children'),
    Input('graph-data-overview', 'data')
)
def update_statistical_summary(graph_data):
    #path = "edgelists/{}".format(get_edgelist_file(networkDropdown))


    # get graph
    #G = nx.read_edgelist(path)
    G = nx.from_dict_of_dicts(graph_data)
    # get properties
    n,m,density,number_of_triangles,avg_clust_coeff = get_statistical_summary(G)
    #deg_hist = nx.degree_histogram(G)
    #max_deg = len(deg_hist)
    #deg_range = [i for i in range(0,max_deg)]
    #df = pd.DataFrame(dict(degree=deg_range, nr_nodes=deg_hist))
    # create bar chart 
    #fig = px.bar(df, x=df.degree, y=df.nr_nodes, labels={'degree' : 'Degree', 'nr_nodes':'Number of Nodes'}, title='Degree distribution of the Network')
    #fig.update_xaxes(type='category')
    degrees = [G.degree(node) for node in G.nodes()]
    df = pd.DataFrame(dict(nodes=list(G.nodes()), degrees=degrees))
    fig = px.histogram(df, x="degrees",
                       title='Degree Distribution',
                       labels={'degrees':'Degree'}, # can specify one label per df column
                       opacity=0.8
                       )
    har_chart = dcc.Graph(figure=fig)
    # insert in Div
    return f'{n:,}',\
            f'{m:,}',\
            "{:.3f}".format(density),\
            f'{number_of_triangles:,}',\
            "{:.3f}".format(avg_clust_coeff),\
            har_chart
    

if __name__ == '__main__':
    app.run_server(debug=True)