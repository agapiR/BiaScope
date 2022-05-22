# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
from dash import Dash, html, dcc, Input, Output, callback
import numpy as np
import networkx as nx

sys.path.insert(0, './../')
from utilities import (unfairness_scores,
                        unfairness_scores_normalized,
                        load_network,
                        load_embedding_2dprojection,
                        load_embedding_2dprojection_go)

# we could use a global variable to avoid recomputing the position of the nodes
app = Dash(__name__)
app.title = "InfoViz-Final"

diagnostics_layout = html.Div(
    children=[
        html.H1(children='Diagnostics view'),
        html.Div([
                    html.Div([
                        html.H3('Select an Embedding Algorithm:'),
                    ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Embedding 1','Embedding 2'], value='Embedding 1', id='embeddingDropdown',style={'width':'200px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                ],style=dict(display='flex')),
        html.Div([
                    html.Div([
                        html.H4('Select a Network:'),
                    ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Example 1','Example 2'], value='Example 1', id='networkDropdown',style={'width':'160px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'}),
                    html.Div([
                        html.H4('Select a Projection Method:'),
                    ],style={'align-items':'center', 'padding-left': '30px', 'padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['UMAP','TSNE'], value='TSNE', id='projectionDropdown',style={'width':'160px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                ],style=dict(display='flex')),
        html.Div(
            children = [
                html.Div([
                    html.Div(id='overviewDiagnosticsNet',style={}),
                ])
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div(
            children = [
                html.Div(id='overviewDiagnosticsProj',style={'z-index': '2'}),
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div([html.Button(children=[
                    html.A(href="/", children="Back")])])
    ]
)
app.layout = diagnostics_layout

@callback(
    Output('overviewDiagnosticsNet', 'children'),
    [Input('networkDropdown', 'value'),
     Input('embeddingDropdown', 'value')]
)
def update_output1(networkDropdown,embeddingDropdown):
    if embeddingDropdown == "Embedding 1":
        embeddingAlgo = "embedding1"
    elif embeddingDropdown == "Embedding 2":
        embeddingAlgo = "embedding2"
    else:
        embeddingAlgo = "embedding1"
    if networkDropdown == "Example 1":
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/{}/example1_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    elif networkDropdown == "Example 2":
        graph_dir = 'data/example2.edgelist'
        embedding_dir = 'data2/example2/{}/example2_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    else:
        # empty selection
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/embedding1/example1_embedding1_10_embedding.npy'
    fig = load_network(graph_dir, embedding_dir)
    graph = dcc.Graph(
                    id='diagnostics-graph',
                    figure=fig
                )
    return graph

@callback(
    Output('overviewDiagnosticsProj', 'children'),
    [Input('networkDropdown', 'value'),
     Input('projectionDropdown', 'value'),
     Input('embeddingDropdown', 'value')]
)
def update_output2(networkDropdown, projectionDropdown, embeddingDropdown):
    if embeddingDropdown == "Embedding 1":
        embeddingAlgo = "embedding1"
    elif embeddingDropdown == "Embedding 2":
        embeddingAlgo = "embedding2"
    else:
        embeddingAlgo = "embedding1"
    if networkDropdown == "Example 1":
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/{}/example1_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    elif networkDropdown == "Example 2":
        graph_dir = 'data/example2.edgelist'
        embedding_dir = 'data2/example2/{}/example2_{}_10_embedding.npy'.format(embeddingAlgo,embeddingAlgo)
    else:
        # empty selection
        graph_dir = 'data/example1.edgelist'
        embedding_dir = 'data2/example1/embedding1/example1_embedding1_10_embedding.npy'
    fig = load_embedding_2dprojection_go(embedding_dir, graph_dir, type=projectionDropdown)
    graph = dcc.Graph(
                    id='diagnostics-embedding',
                    figure=fig
                )
    return graph
    

if __name__ == '__main__':
    app.run_server(debug=True)