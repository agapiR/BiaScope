# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
import os
import dash
from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import flask
import numpy as np
import networkx as nx
import pandas as pd
import json

import time
from urllib import parse

sys.path.insert(0, './../')
from utilities_with_interaction import (get_egoNet,
                                        get_induced_subgraph,
                                        draw_network,
                                        draw_embedding_2dprojection,
                                        draw_2d_scale, 
                                        get_recommended_nodes,
                                        get_scores,
                                        get_node_features)

## Define metadata 
graph_metadata = {"Facebook": {"edgelist": "edgelists/facebook_combined.edgelist"},
                 "LastFM": {"edgelist": "edgelists/lastfm_asia_edges.edgelist"},
                 "wikipedia": {"edgelist": "edgelists/wikipedia.edgelist"},
                  "protein-protein": {"edgelist": "edgelists/ppi.edgelist"},
                  "ca-HepTh": {"edgelist": "edgelists/ca-HepTh.edgelist"},
                  "AutonomousSystems": {"edgelist": "edgelists/AS.edgelist"},
                 }




## Pre-Load Data
# global G, node_features, scores

# networkDefault = "Facebook"
# embDefault = "Node2Vec"

# graph_dir = graph_metadata[networkDefault]["edgelist"]
# preprocessed_data_dir = 'embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv'.format(networkDefault,
#                                                                                                         embDefault,
#                                                                                                         networkDefault,
#                                                                                                         embDefault)
# preprocessed_group_fairness_dir = 'embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv'.format(networkDefault,
#                                                                                                             embDefault,
#                                                                                                             networkDefault,
#                                                                                                             embDefault)

# G = nx.read_edgelist(graph_dir)
# node_features = get_node_features(preprocessed_data_dir)
# params = {"nrHops": 1}
# scores_raw = get_scores('Individual (InFoRM)', params, preprocessed_data_dir)
# scores = np.array(scores_raw).round(decimals=2)


## Import descriptive text
sys.path.insert(0, "description-txt/")
import diagnostics_description, footer

## View Layout
app = Dash(__name__)
app.title = "InfoViz-Final"

diagnostics_layout = html.Div(
    children=[
        diagnostics_description.description,
        html.Div([
                    html.Div([
                        html.H3('Embedding Algorithm:'),
                    ],style={'alignItems':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Node2Vec','HGCN','HOPE','LaplacianEigenmap','SDNE','SVD'], 
                        value='', id='embeddingDropdown', clearable=False, style={'width':'160px','alignItems':'center'}),
                    ],style={'display': 'inline-block','verticalAlign': 'middle','paddingRight': 10})
                ],style=dict(display='flex')),
        html.Div([
                    html.Div([
                        html.H3('Network:'),
                    ],style={'alignItems':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Facebook','protein-protein', 'LastFM', 'wikipedia'], value='', id='networkDropdown_d', clearable=False, style={'width':'160px','alignItems':'center'}),
                    ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '0px'}),
                    html.Div([
                        html.H3('Fairness notion:'),
                        ],style={'alignItems':'center','paddingRight': '10px','paddingTop': '7px','display': 'inline-block', 'marginLeft': '50px'}),
                    html.Div([
                        dcc.Dropdown(options=[], value='Individual (InFoRM)', id='fairnessNotion_d', clearable=False, style={'width':'200px','alignItems':'center'}),
                        ],style={'display': 'inline-block','verticalAlign': 'middle','paddingTop': '0px'}),
                    html.Div([
                        html.H4('Projection Method:'),
                    ],style={'alignItems':'center', 'paddingLeft': '30px', 'paddingRight': '10px','display': 'none'}),
                    html.Div([
                        dcc.Dropdown(options=['PCA'], value='PCA', id='projectionDropdown', clearable=False, style={'width':'160px','alignItems':'center'}),
                    ],style={'display': 'none','padding': 20, 'verticalAlign': 'middle'})
                ],style=dict(display='flex')),
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
                            dcc.Slider(id='nrHops_d', min=1, max=2, step=1, value=1)
                        ],style={'width':'150px','height':'11px','display': 'inline-block','padding-left': '5px'}),
                    ],id='indFairnessParams_d',style={'display': 'none'}),
                    html.Div([
                        html.Div([
                            html.H4('Node attribute:'),
                        ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                        html.Span(
                            html.Span("Sensitive attribute used to evaluate group fairness", className="tooltiptext"),
                            className="fa fa-question-circle tooltip"),
                        html.Div([
                            dcc.Dropdown(options=[], value='', id='sensitiveAttr_d', clearable=False, style={'width':'200px','align-items':'center'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','paddingLeft': '5px'}),
                        html.Div([
                            html.H4('Attribute value:'),
                        ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                        html.Span(
                        html.Span("Define group fairness based on recommendations to nodes with this attribute value", className="tooltiptext"),
                        className="fa fa-question-circle tooltip"),
                        html.Div([
                            dcc.RadioItems(options=[1,2], value=1, id='sensitiveAttrVal_d',inline=False,labelStyle={'display': 'block'}),
                        ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '21px'}),
                        html.Div([
                            html.H4('Value of k:'),
                        ],style={'align-items':'center', 'padding-left': '150px', 'padding-right': '10px','display': 'inline-block'}),
                        html.Span(
                            html.Span("Number of nodes recommended to each target node", className="tooltiptext"),
                            className="fa fa-question-circle tooltip"),
                        html.Div([
                            dcc.Slider(min=5, max=25, step=5, value=5, id='kVal_d')
                        ],style={'width':'150px','height':'11px','display': 'inline-block','vertical-align': 'middle','padding-top': '0px'})
                    ],id='groupFairnessParams_d',style={'display': 'none'})
                ])
                ],id='fairnessParams_d'),
            ],style={'alignItems':'center','padding': 20,'display': 'block'}),
        html.Div([
            # html.Div([
                    # html.Div([],style={'display': 'inline-block','paddingLeft': 0}), # For shifting the legend
            html.Div(id='legend-indfairness',className="legend",
                children=[
                    html.H2('Legend'),
                    html.H3(children=['The focal node is highlighted in ',html.Span('red',style={"backgroundColor":"#de2d26","color":"#ededed"}),' and increased in ', html.Span('size',style={'fontSize': 24}),'.']),
                    html.H3(children=['The 1-hop neighbors of the focal node are colored in ',html.Span('coral',style={"backgroundColor":"#fcae91"}) ,', while the 2-hop neighbors in ',html.Span('light salmon',style={"backgroundColor":"#fee5d9"}),'.'])
                ],
                style={'width': '60%','display': 'inline-block','paddingLeft':'0px','paddingBottom':'0px'}),
            # ],style={'width': '65%','display': 'block','padding': 0}),
            # html.Div([
                    # html.Div([],style={'display': 'inline-block','paddingLeft': 300}), # For shifting the legend
            html.Div(id='legend-groupfairness',className="legend",
                children=[
                    html.H2('Legend'),
                    html.H3(children=['The focal node is increased in ',html.Span('size',style={'fontSize': 24}),' and its contour is highlighted in ',html.Span('red',style={"backgroundColor":"#de2d26","color":"#ededed"}),'.']),
                    html.H3(children=['The 1-hop neighbors of the focal node, together with itself, are colored according to their gender value: ', html.Span('yellow',style={"backgroundColor":"#ffff99"}),' for gender 0 and ',html.Span('blue',style={"backgroundColor":"#386cb0","color":"#ededed"}),' for gender 1.'])
                ],
            style={'width': '60%','display': 'none','paddingLeft':'0px','paddingBottom':'0px'}),
            # ],style={'width': '65%','display': 'block','padding': 0}),
            html.Div(
                    dcc.Loading(
                    id="loading-scale",
                    type="circle",
                    color="#800020",
                    children= dcc.Graph(id='embeddingScale', config={'displayModeBar': False}),
                    ),
                    className='main view',
                    style={'width': '40%', 'display': 'inline-block', 'paddingTop':'0px'}       
                )            
        ],style={'display': 'flex','padding':'0px 25px'}),
        html.Div([
            html.Div(
                dbc.Container([
                    dbc.Label('Click to select the focal node:'),
                    dash_table.DataTable(columns=[
                                        {'name': 'Node IDs', 'id': 'Node IDs', 'type': 'text'},
                                        {'name': 'Scores', 'id': 'Scores', 'type': 'numeric'},
                                    ],
                                    data = [],
                                    id='nodeList', 
                                    style_table={'overflowY': 'auto'},
                                    row_selectable='single',
                                    sort_action="native",
                                    filter_action='native',
                                    page_size= 10,
                                    style_data_conditional=[
                                        {
                                            "if": {"state": "selected"},
                                            "backgroundColor": "inherit !important",
                                            "border": "inherit !important"
                                        }
                                    ],
                                        # style_cell={
                                        #     'height': 'auto',
                                        #     # all three widths are needed
                                        #     'minWidth': '30px', 'width': '30px', 'maxWidth': '30px',
                                        #     'whiteSpace': 'normal'
                                        # }
                                    )
                ]),
                className='main view',
                style={'alignItems':'center', 'width': '20%', 'paddingRight': '20px', 'paddingBottom': '20px', 'paddingTop': '55px', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Loading(
                id="loading-graph",
                type="circle",
                color="#800020",
                children= dcc.Graph(id='overviewDiagnosticsGraph', config={'displayModeBar': False}),
                ),
                className='main view',
                style={'width': '40%', 'display': 'inline-block', 'paddingTop':'25px'}
            ),
            html.Div(
                dcc.Loading(
                id="loading-emb",
                type="circle",
                color="#800020",
                children= dcc.Graph(id='overviewDiagnosticsEmb', config={'displayModeBar': False}),
                ),
                className='main view',
                style={'width': '40%', 'display': 'inline-block', 'paddingTop':'25px'}       
            )
        ],style={'display': 'flex','padding':'0px 25px'}
        ),
        html.Div(
            children = html.Div([html.Center(html.A(href="/", children="Back", className="button"))]),
            className='main view',
            style={'width': '10%', 'display': 'inline-block', 'paddingTop': '25px',
            'paddingBottom': '50px','paddingRight': '0px','paddingLeft': '0px'}
        ),
        footer.conclusion_div,
        # Store Components
        # dcc.Store inside the user's current browser session
        #dcc.Store(id='network-name', data=[], storage_type='memory'), # 'local' or 'session'
        #dcc.Store(id='embedding-name', data=[], storage_type='memory'),
        #dcc.Store(id='focal-node-id', data=[], storage_type='memory'),
        dcc.Location(id='location'),
        dcc.Store(id='node-score-list', data=[], storage_type='memory'),
        dcc.Store(id='node-features', data=[], storage_type='memory'),
        dcc.Store(id='graph-data', data=[], storage_type='memory')
    ]
)
app.layout = diagnostics_layout



## Callbacks

# Set default values for network and embedding
@callback(
    Output('networkDropdown_d', 'value'),
    Output('embeddingDropdown', 'value'),
    Input('location', 'href'),
)
def set_defaults(url):
    parsed_url = parse.urlparse(url)
    if parse.parse_qs(parsed_url.query):
        network_name = parse.parse_qs(parsed_url.query)['net'][0]
        embedding_name = parse.parse_qs(parsed_url.query)['emb'][0]
    else:
        network_name = 'Facebook'
        embedding_name = 'Node2Vec'
    
    return network_name, embedding_name


# Store 'graph-data'
@callback(
    Output('graph-data', 'data'),
    Input('networkDropdown_d', 'value')
)
def store_graph_data(network_name):
    # Configure data sources
    graph_dir = graph_metadata[network_name]["edgelist"]
    # load new network 
    G = nx.read_edgelist(graph_dir)

    return nx.to_dict_of_dicts(G)

# Store 'node-features'
@callback(
    Output('node-features', 'data'),
    [Input('networkDropdown_d', 'value'),
     Input('embeddingDropdown', 'value')]
)
def store_node_features(network_name, embedding_name):
    # Configure data sources
    preprocessed_data_dir = 'embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores_with_gender.csv'.format(network_name,
                                                                                                        embedding_name,
                                                                                                        network_name,
                                                                                                        embedding_name)
    # Configure data sources
    recommended_nodes_dir = 'embeddings/{}/{}/{}_{}_64_embedding_recommended_nodes.csv'.format(network_name,
                                                                                            embedding_name,
                                                                                            network_name,
                                                                                            embedding_name)

    # read preprocessed data
    node_features = get_node_features(preprocessed_data_dir)

    if os.path.exists(recommended_nodes_dir):
        recommended_nodes = get_recommended_nodes(recommended_nodes_dir)
        for node in node_features.keys():
            try: 
                node_features[node]["recommended"] = [node]+recommended_nodes[node]
            except:
                node_features[node]["recommended"] = [node]

    return node_features

# Store 'node-score-list'
@callback(
    Output('node-score-list', 'data'),
    [# basic selection config
     Input('networkDropdown_d', 'value'),
     Input('embeddingDropdown', 'value'),
     # fairness config
     Input('fairnessNotion_d', 'value'),
     Input('sensitiveAttr_d', 'value'),
     Input('sensitiveAttrVal_d', 'value'),
     Input('kVal_d', 'value'),
     Input('nrHops_d', 'value')
    ]
)
def store_node_score_list(networkDropdown, embeddingDropdown,
                fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):
    # Configure data sources
    preprocessed_data_dir = 'embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores_with_gender.csv'.format(  networkDropdown,
                                                                                                embeddingDropdown,
                                                                                                networkDropdown,
                                                                                                embeddingDropdown)
    preprocessed_group_fairness_dir = 'embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv'.format(  
                                                                                                networkDropdown,
                                                                                                embeddingDropdown,
                                                                                                networkDropdown,
                                                                                                embeddingDropdown)

    if fairnessNotion=='Group (Fairwalk)':
        params = {"attribute": sensitiveAttr, "value": sensitiveAttrVal, "k": str(kVal)}
        node_to_score = get_scores(fairnessNotion, params, preprocessed_group_fairness_dir)
    else: #'Individual (InFoRM)'
        params = {"nrHops": nrHops}
        node_to_score = get_scores(fairnessNotion, params, preprocessed_data_dir)
    
    ## update node-score list
    node_ids = list(node_to_score.keys())
    scores = [round(s,2) for s in node_to_score.values()]
    # get indices of scores sorted in descending order
    idx_sort_by_score = np.argsort(-1*np.array(scores))
    # sort nodes and scores:
    node_ids_sorted = [node_ids[idx] for idx in idx_sort_by_score]
    scores_sorted = [scores[idx] for idx in idx_sort_by_score]
    # save list data in df
    node_list_dict = {'Node IDs': node_ids_sorted, 'Scores': scores_sorted}
    node_list = pd.DataFrame(data=node_list_dict)

    return node_list.to_dict('records')

# Parameter selection depending on selected fairness notion
@callback(
    Output('fairnessNotion_d', 'options'),
    Output('fairnessNotion_d', 'value'),
    # individual fairness
    Output('nrHops_d', 'value'),
    Output('indFairnessParams_d', 'style'),
    # group fairness
    Output('sensitiveAttr_d', 'options'),
    Output('sensitiveAttr_d', 'value'),
    Output('sensitiveAttrVal_d', 'options'),
    Output('sensitiveAttrVal_d', 'value'),
    Output('kVal_d', 'value'),
    Output('groupFairnessParams_d', 'style'),
    [Input('networkDropdown_d', 'value'),
    Input('fairnessNotion_d', 'value')]
)
def display_fairness_parameters(networkDropdown, fairnessNotion):
    # get path to selected network
    nr_hops_options_value = 1 
    ind_fairness_params_style = {'display': 'none'}
    fairness_notions_val = fairnessNotion
    fairness_notions = []
    if networkDropdown == "Facebook":
        fairness_notions = ['Individual (InFoRM)','Group (Fairwalk)']
    else:
        fairness_notions = ['Individual (InFoRM)']
    
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] in ['networkDropdown_d.value','.'] :
        # restore state of parameters - default is Individual (InFoRM) with nr_hops = 1
        nr_hops_options_value = 1
        ind_fairness_params_style = {'display': 'inline-block','verticalAlign': 'middle','paddingTop': '3px'}
        fairness_notions_val = 'Individual (InFoRM)'
        group_fairness_params_style = {'display': 'none'}
        # group fairness
        sensitive_attr_options = []
        sensitive_attr_value = ''
        sensitive_attr_val_options = []
        sensitive_attr_val_value = ''
        k_val_value = 5
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
            k_val_value = int(group_fairness_config["k_s"][0])
            group_fairness_params_style = {'display': 'inline-block'}

        else: 
            nr_hops_options_value = 1 
            ind_fairness_params_style = {'display': 'inline-block','verticalAlign': 'middle','paddingTop': '3px'}

            sensitive_attr_options = []
            sensitive_attr_value = ''
            sensitive_attr_val_options = []
            sensitive_attr_val_value = ''
            k_val_value = 5
            group_fairness_params_style = {'display': 'none'}

    return fairness_notions, fairness_notions_val, nr_hops_options_value, ind_fairness_params_style, sensitive_attr_options, sensitive_attr_value, sensitive_attr_val_options, sensitive_attr_val_value, k_val_value, group_fairness_params_style


# Configure Legends
@callback(
    Output('legend-indfairness', 'style'),
    Output('legend-groupfairness', 'style'),
    # fairness config
    Input('fairnessNotion_d', 'value'),
)
def updateLegend(fairnessNotion):

    legend_ind_fairness = {}
    legend_group_fairness = {}

    if fairnessNotion=='Group (Fairwalk)':
        legend_ind_fairness = {'display': 'none'}
        legend_group_fairness = {'display': 'inline-block','paddingLeft':'35px','paddingRight':'35px','paddingBottom':'10px','paddingTop':'55px'}
    else: #'Individual (InFoRM)'
        legend_group_fairness = {'display': 'none'}
        legend_ind_fairness = {'display': 'inline-block','paddingLeft':'35px','paddingRight':'35px','paddingBottom':'0px','paddingTop':'160px'}

    return legend_ind_fairness, legend_group_fairness

# Update Interactive Table
@callback(
    Output('nodeList', 'data'),
    Output('nodeList', 'selected_rows'),
    State('nodeList', 'selected_rows'),
    Input('node-score-list', 'data')
)
def updateNodeList(selectedRow, node_score_list):
    # # if: a row is already selected, then do not change the selection
    # if selectedRow:
    #     row = selectedRow
    # # else: select the first row of the table.
    # else:
    #     first_table_tuple = node_score_list[0]
    #     node_id_to_be_selected = first_table_tuple['Node IDs']
    #     node_idx_to_be_selected = int(0) #(I don't now how to find this)???
    #     row = [node_idx_to_be_selected]

    # Selct the first entry of the table.
    # Selects the most unfair node, assuming sorted table.
    row = [0]

    return node_score_list, row

# @callback(
#     Output('nodeList', 'data'),
#     [Input('embeddingDropdown', 'value'),
#      Input('networkDropdown_d', 'value'),
#      # fairness config
#      Input('fairnessNotion_d', 'value'),
#      Input('sensitiveAttr_d', 'value'),
#      Input('sensitiveAttrVal_d', 'value'),
#      Input('kVal_d', 'value'),
#      Input('nrHops_d', 'value')
#     ]
# )
# def updateNodeList(embeddingDropdown, networkDropdown, 
#                 fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):

#     global G, scores

#     # Configure data sources
#     graph_dir = graph_metadata[networkDropdown]["edgelist"]
#     preprocessed_data_dir = 'embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv'.format(  networkDropdown,
#                                                                                                 embeddingDropdown,
#                                                                                                 networkDropdown,
#                                                                                                 embeddingDropdown)
#     preprocessed_group_fairness_dir = 'embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv'.format(  
#                                                                                                 networkDropdown,
#                                                                                                 embeddingDropdown,
#                                                                                                 networkDropdown,
#                                                                                                 embeddingDropdown)

#     if fairnessNotion=='Group (Fairwalk)':
#         params = {"attribute": sensitiveAttr, "value": sensitiveAttrVal, "k": kVal}
#         scores_raw = get_scores(fairnessNotion, params, preprocessed_group_fairness_dir)
#     else: #'Individual (InFoRM)'
#         params = {"nrHops": nrHops}
#         scores_raw = get_scores(fairnessNotion, params, preprocessed_data_dir)
    
#     # update node-score list
#     scores = np.array(scores_raw).round(decimals=2)
#     node_ids = np.array(list(G.nodes()))
#     node_list_dict = {'Node IDs': node_ids, 'Scores': scores}
#     node_list = pd.DataFrame(data=node_list_dict)

#     return node_list.to_dict('records')


# Handle Interaction
@callback(
    Output('overviewDiagnosticsGraph', 'figure'),
    Output('overviewDiagnosticsEmb', 'figure'),
    Output('embeddingScale', 'figure'),
    [Input('graph-data', 'data'),
     Input('node-features', 'data'),
     Input('node-score-list', 'data'),
     # I dont know what we need from bellow
     Input('embeddingDropdown', 'value'),
     Input('networkDropdown_d', 'value'),
     Input('projectionDropdown', 'value'),
     # brushing & linking and filtering selections
     Input('nodeList', 'selected_rows'),
     Input('overviewDiagnosticsGraph', 'selectedData'),
     Input('overviewDiagnosticsEmb', 'selectedData'),
     # fairness config
     Input('fairnessNotion_d', 'value'),
     Input('sensitiveAttr_d', 'value'),
     Input('sensitiveAttrVal_d', 'value'),
     Input('kVal_d', 'value'),
     Input('nrHops_d', 'value')
    ]
)
def updateView(graph_data, node_features, node_score_list,
                embeddingDropdown, networkDropdown, projectionDropdown,
                selectedRow, selectionGraph, selectionEmb,
                fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):

    G = nx.from_dict_of_dicts(graph_data)
    node_score_df = pd.DataFrame(data=node_score_list)
    node_ids = np.array(node_score_df['Node IDs'].to_list())
    scores = np.array(node_score_df['Scores'].to_list())

    if fairnessNotion=='Group (Fairwalk)':
        attribute_type = sensitiveAttr
        hops = 1

    else: #'Individual (InFoRM)'
        attribute_type = "graph distance"
        hops = int(nrHops)

    # Identify callback source
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #print("updateView triggered by:", trigger_id)


    # UPDATE LOGIC:
    # 1) when we change network or embedding algo or focal node or fairness config brush selection should be cleared
    # 2) when we change projection method the brush should be released from the Emb only.
    
    if trigger_id in ['networkDropdown_d', 'embeddingDropdown', 
                    'fairnessNotion_d', 'sensitiveAttr_d', 'sensitiveAttrVal_d', 'kVal_d', 'nrHops_d']:
        # erase brushes
        selectionGraph = None
        selectionEmb = None

    elif trigger_id == 'projectionDropdown':
        # erase embedding brush
        selectionEmb = None
        # get projection method
        projectionAlgo = projectionDropdown

    elif trigger_id == 'nodeList':
        # erase brushes
        selectionGraph = None
        selectionEmb = None
        # get new focal node
        if selectedRow:
            focalNodeIdx = selectedRow[0]
    else:
        # no need to reload scores
        pass
       

    # handle selection triggers
    selection = False
    if trigger_id == 'overviewDiagnosticsEmb':
        # erase graph brush
        selectionGraph = None
        # get new selection
        if selectionEmb and selectionEmb['points']:
            selectedpoints = np.array([p['pointIndex'] for p in selectionEmb['points']])
            selection = True

    elif selectionGraph or trigger_id == 'overviewDiagnosticsGraph':
        # erase embedding brush
        selectionEmb = None
        # get new selection
        if selectionGraph and selectionGraph['points']:
            selectedpoints = np.array([p['pointIndex'] for p in selectionGraph['points']])
            selection = True

    else:
        selectionGraph = None
        selectionEmb = None

    n = nx.number_of_nodes(G)
    if not selection:
        selectedpoints = np.array(range(n))

    # get focal node
    if selectedRow:
        focalNodeIdx = selectedRow[0]
        focal_node = node_ids[focalNodeIdx]
    else:
        focal_node = node_ids[np.argmax(scores)]

    # get local topology and attributes
    if fairnessNotion=='Group (Fairwalk)':
        recommended = node_features[str(focal_node)]["recommended"]
        recommended_topk = recommended[0:int(kVal)+1]
        local_network, local_ids = get_induced_subgraph(G, recommended_topk)
        attributes = [int(node_features[idx][attribute_type]) for idx in local_ids]
        topology_title = "Top-{} proximal nodes in the embedding".format(kVal)
        #attributes = [0 if idx==focal_node else 1 for idx in local_ids]
    else: #'Individual (InFoRM)'
        local_network, local_ids = get_egoNet(G, str(focal_node), k=hops)
        distance_dict = nx.shortest_path_length(local_network, source=focal_node)
        attributes = [0 if n==focal_node else distance_dict[n] for n in local_network]
        topology_title = "{}-Hop Ego Network".format(hops)
    # focal node(s) list
    focal = [1 if idx==focal_node else 0 for idx in local_ids]
    # get scores of local nodes

    local_scores = []
    for idx in local_ids:
        slice = node_score_df.loc[node_score_df['Node IDs'] == idx]
        local_scores.append(slice['Scores'].to_list()[0])

    local_projections_x = np.array([float(node_features[idx]['proj_x']) for idx in local_ids])
    local_projections_y = np.array([float(node_features[idx]['proj_y']) for idx in local_ids])
    local_projections = np.vstack((local_projections_x,local_projections_y))

    tic = time.perf_counter()

    figGraph = draw_network(local_network, local_scores, focal, fairness_notion=fairnessNotion, 
                        attributes=attributes, attribute_type=attribute_type,
                        title = topology_title,
                        selection_local = selectionGraph, selectedpoints = selectedpoints)

    toc = time.perf_counter()

    #print(f"Network drawn in {toc - tic:0.4f} seconds")

    figEmb = draw_embedding_2dprojection(local_network, local_projections, local_scores, focal, fairness_notion=fairnessNotion, 
                                attributes=attributes, attribute_type=attribute_type,
                                type=projectionDropdown,
                                selection_local = selectionEmb, selectedpoints = selectedpoints)
    

    projections_x = np.array([float(node_features[idx]['proj_x']) for idx in node_features.keys()])
    projections_y = np.array([float(node_features[idx]['proj_y']) for idx in node_features.keys()])
    projections = np.vstack((projections_x,projections_y))
    figScale = draw_2d_scale(projections, local_projections, show_inner=True)

    return figGraph, figEmb, figScale


# @callback(
#     Output('embeddingScale', 'figure'),
#     [Input('graph-data', 'data'),
#      Input('node-features', 'data'),
#      Input('node-score-list', 'data'),
#      # I dont know what we need from bellow
#      Input('embeddingDropdown', 'value'),
#      Input('networkDropdown_d', 'value'),
#      Input('projectionDropdown', 'value'),
#      # Focal selection
#      Input('nodeList', 'selected_rows'),
#      # fairness config
#      Input('fairnessNotion_d', 'value'),
#      Input('sensitiveAttr_d', 'value'),
#      Input('sensitiveAttrVal_d', 'value'),
#      Input('kVal_d', 'value'),
#      Input('nrHops_d', 'value')
#     ]
# )
# def updateScaleLegend(graph_data, node_features, node_score_list,
#                     embeddingDropdown, networkDropdown, projectionDropdown,
#                     selectedRow,
#                     fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):
        
#     local_projections_x = np.array([float(node_features[idx]['proj_x']) for idx in local_ids])
#     local_projections_y = np.array([float(node_features[idx]['proj_y']) for idx in local_ids])
#     local_projections = np.vstack((local_projections_x,local_projections_y))

#     projections_x = np.array([float(node_features[idx]['proj_x']) for idx in node_features.keys()])
#     projections_y = np.array([float(node_features[idx]['proj_y']) for idx in node_features.keys()])
#     projections = np.vstack((projections_x,projections_y))

#     figScale = draw_2d_scale(projections, local_projections, show_inner=True)

#     return figScale




# @callback(
#     Output('overviewDiagnosticsGraph', 'figure'),
#     Output('overviewDiagnosticsEmb', 'figure'),
#     [Input('embeddingDropdown', 'value'),
#      Input('networkDropdown', 'value'),
#      Input('projectionDropdown', 'value'),
#      # brushing & linking and filtering selections
#      Input('nodeList', 'selected_rows'),
#      Input('overviewDiagnosticsGraph', 'selectedData'),
#      Input('overviewDiagnosticsEmb', 'selectedData'),
#      # fairness config
#      Input('fairnessNotion_d', 'value'),
#      Input('sensitiveAttr_d', 'value'),
#      Input('sensitiveAttrVal_d', 'value'),
#      Input('kVal_d', 'value'),
#      Input('nrHops_d', 'value')
#     ]
# )
# def updateView(embeddingDropdown, networkDropdown, projectionDropdown,
#                 selectedRow, selectionGraph, selectionEmb,
#                 fairnessNotion, sensitiveAttr, sensitiveAttrVal, kVal, nrHops):

#     global G, node_features, scores


#     if fairnessNotion=='Group (Fairwalk)':
#         # print("Debug: attr = {}  attr_val = {}  #recommendation = {}\n".format(sensitiveAttr, 
#         #                                                                         sensitiveAttrVal,
#         #                                                                         kVal))
#         attribute_type = sensitiveAttr
#         hops = 1
#     else: #'Individual (InFoRM)'
#         #print("Debug: #hops = {}\n".format(nrHops))
#         attribute_type = "distance"
#         hops = int(nrHops)

#     # Configure data sources
#     graph_dir = graph_metadata[networkDropdown]["edgelist"]
#     preprocessed_data_dir = 'embeddings/{}/{}/{}_{}_64_embedding_node_features_InFoRM_scores.csv'.format(  networkDropdown,
#                                                                                                 embeddingDropdown,
#                                                                                                 networkDropdown,
#                                                                                                 embeddingDropdown)
#     preprocessed_group_fairness_dir = 'embeddings/{}/{}/{}_{}_64_embedding_group_fairness_scores.csv'.format(  
#                                                                                                 networkDropdown,
#                                                                                                 embeddingDropdown,
#                                                                                                 networkDropdown,
#                                                                                                 embeddingDropdown)
    
#     # Identify callback source
#     ctx = dash.callback_context
#     trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
#     print("Trigger: " + trigger_id)

#     # UPDATE LOGIC:
#     # 1) when we change network or embedding algo everything should be reset: 
#     # clear brush and list selection 
#     # 2) when we change projection method the brush should be released from the Emb only.
    
#     # load new network 
#     G = nx.read_edgelist(graph_dir)
#     if trigger_id == 'networkDropdown':
#         # erase brushes
#         selectionGraph = None
#         selectionEmb = None
#         # read preprocessed data
#         node_features = get_node_features(preprocessed_data_dir)
#         # must reload scores
#         load_scores = True

#     elif trigger_id == 'embeddingDropdown':
#         # erase brushes
#         selectionGraph = None
#         selectionEmb = None
#         # read preprocessed data
#         node_features = get_node_features(preprocessed_data_dir)
#         # must reload scores
#         load_scores = True

#     elif trigger_id == 'projectionDropdown':
#         # erase embedding brush
#         selectionEmb = None
#         # no need to reload scores
#         load_scores = False
#         # get projection method
#         projectionAlgo = projectionDropdown

#     elif trigger_id == 'nodeList':
#         # erase brushes
#         selectionGraph = None
#         selectionEmb = None
#         # no need to reload scores
#         load_scores = False
#         # get new focal node
#         if selectedRow:
#             focalNodeIdx = selectedRow[0]

#     elif trigger_id in ['fairnessNotion_d', 'sensitiveAttr_d', 'sensitiveAttrVal_d', 'kVal_d', 'nrHops_d']:
#         # must reload scores for any change in the fairness config 
#         load_scores = True
    
#     else:
#         # no need to reload scores
#         load_scores = False
       

#     # handle selection triggers
#     selection = False
#     if trigger_id == 'overviewDiagnosticsEmb':
#         # erase graph brush
#         selectionGraph = None
#         # get new selection
#         if selectionEmb and selectionEmb['points']:
#             selectedpoints = np.array([p['pointIndex'] for p in selectionEmb['points']])
#             selection = True

#     elif selectionGraph or trigger_id == 'overviewDiagnosticsGraph':
#         # erase embedding brush
#         selectionEmb = None
#         # get new selection
#         if selectionGraph and selectionGraph['points']:
#             selectedpoints = np.array([p['pointIndex'] for p in selectionGraph['points']])
#             selection = True

#     else:
#         selectionGraph = None
#         selectionEmb = None

#     node_ids = np.array(list(G.nodes()))
#     n = nx.number_of_nodes(G)
#     if not selection:
#         selectedpoints = np.array(range(n))

#     # get focal node
#     if selectedRow:
#         focalNodeIdx = selectedRow[0]
#         focal_node = node_ids[focalNodeIdx]
#         #print("\nSELECTED ROW = {}\n".format(selectedRow[0]))
#         #print("FOCAL NODE = {}\n".format(focal_node))
#     else:
#         focal_node = node_ids[np.argmax(scores)]

#     # get local topology and attributes
#     local_network, local_ids = get_egoNet(G, str(focal_node), k=hops)
#     if fairnessNotion=='Group (Fairwalk)':
#         attributes = [int(node_features[idx][attribute_type]) for idx in local_ids]
#         #attributes = [0 if idx==focal_node else 1 for idx in local_ids]
#     else: #'Individual (InFoRM)'
#         distance_dict = nx.shortest_path_length(local_network, source=focal_node)
#         attributes = [0 if n==focal_node else distance_dict[n] for n in local_network]
#     # focal node(s) list
#     focal = [1 if idx==focal_node else 0 for idx in local_ids]

#     local_scores = scores[[int(idx) for idx in local_ids]]
#     local_projections_x = np.array([float(node_features[idx]['proj_x']) for idx in local_ids])
#     local_projections_y = np.array([float(node_features[idx]['proj_y']) for idx in local_ids])
#     local_projections = np.vstack((local_projections_x,local_projections_y))

#     tic = time.perf_counter()

#     figGraph = draw_network(local_network, local_scores, focal, fairness_notion=fairnessNotion, 
#                         attributes=attributes, attribute_type=attribute_type,
#                         title = "{}-Hop Ego Network".format(hops),
#                         selection_local = selectionGraph, selectedpoints = selectedpoints)

#     toc = time.perf_counter()

#     #print(f"Network drawn in {toc - tic:0.4f} seconds")

#     figEmb = draw_embedding_2dprojection(local_network, local_projections, local_scores, focal, fairness_notion=fairnessNotion, 
#                                 attributes=attributes, attribute_type=attribute_type,
#                                 type=projectionDropdown,
#                                 selection_local = selectionEmb, selectedpoints = selectedpoints)
    


#     return figGraph, figEmb




if __name__ == '__main__':
    app.run_server(debug=True)