# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px


# we could use a global variable to avoid recomputing the position of the nodes

def load_network(path):

    G = nx.read_edgelist(path)
    # pos = nx.get_node_attributes(G2,'pos')

    # nodePos = nx.circular_layout(G)
    # nodePos = nx.kamada_kawai_layout(G)
    nodePos = nx.spring_layout(G)
    # nodePos = nx.spectral_layout(G)

    # print(nodePos['0'])

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = nodePos[edge[0]]
        x1, y1 = nodePos[edge[1]]
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
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

app = Dash(__name__)
app.title = "Multipage Report"

app.layout = html.Div(
    children=[
        html.H1(children='Hello Dash'),
        html.Div([
                    html.Div([
                        html.H3('Select a Network:'),
                    ],style={'align-items':'center','padding-right': '10px','display': 'inline-block'}),
                    html.Div([
                        dcc.Dropdown(options=['Example 1','Example 2'], value='Example 1', id='demo-dropdown1',style={'width':'160px','align-items':'center'}),
                    ],style={'display': 'inline-block','vertical-align': 'middle','padding-top': '12px'})
                ],style=dict(display='flex')),
        html.Div(
            children = [
                html.Div([
                    html.Div(id='dd-output-container1',style={}),
                ])
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div(
            children = [
                html.Div(id='dd-output-container2',style={}),
            ],
            className="title six columns",
            style={'width': '49%', 'display': 'inline-block'}
        ),
        html.Div( children = [
            html.H2(children='Statistical summary of the network'),
            html.Div(
                children = [
                    html.Div([
                        html.Div(id='dd-output-container3',
                        children = [
                                html.Table([
                                    html.Tr([html.Th('Variable'), html.Th('Value')]),
                                    html.Tr([html.Td('n'), html.Td(id='nodes')]),
                                    html.Tr([html.Td('m'), html.Td(id='edges')]),
                                    html.Tr([html.Td('density'), html.Td(id='density')]),
                                    html.Tr([html.Td('nr. of triangles'), html.Td(id='triangles')]),
                                    html.Tr([html.Td('avg. clustering coeff.'), html.Td(id='clustcoeff')])
                                ], style={'width':'80%', 'border': '1px solid'})
                        ],style={'opacity':'1', 'padding-bottom': '180px'}),
                    ])
                ],
                className="title six columns",
                style={'width': '49%', 'display': 'inline-block'}
            ),
            html.Div(
                children = [
                    html.Div(id='dd-output-container4',style={'padding-bottom': '85px'}),
                ],
                className="title six columns",
                style={'width': '49%', 'display': 'inline-block'}
            )
        ])
    ]
)

# aux functions
def get_statistical_summary(G):
    # computes the statistical summary of the graph G
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    density = nx.density(G)
    number_of_triangles = int(sum(nx.triangles(G).values()) / 3)
    avg_clust_coeff = nx.average_clustering(G)
    return n,m,density,number_of_triangles,avg_clust_coeff

# callbacks

@app.callback(
    Output('dd-output-container1', 'children'),
    Input('demo-dropdown1', 'value')
)
def update_network1(value):
    if value == "Example 1":
        path = 'data/example1.edgelist'
    elif value == "Example 2":
        path = 'data/example2.edgelist'
    else:
        # empty selection
        path = 'data/example1.edgelist'
    fig = load_network(path)
    graph = dcc.Graph(
                    id='example-graph-1',
                    figure=fig
                )
    return graph

@app.callback(
    Output('dd-output-container2', 'children'),
    Input('demo-dropdown1', 'value')
)
def update_network2(value):
    if value == "Example 1":
        path = 'data/example1.edgelist'
    elif value == "Example 2":
        path = 'data/example2.edgelist'
    else:
        # empty selection
        path = 'data/example1.edgelist'
    fig = load_network(path)
    graph = dcc.Graph(
                    id='example-graph-1',
                    figure=fig
                )
    return graph

@app.callback(
    Output('nodes', 'children'),
    Output('edges', 'children'),
    Output('density', 'children'),
    Output('triangles', 'children'),
    Output('clustcoeff', 'children'),
    Output('dd-output-container4', 'children'),
    Input('demo-dropdown1', 'value')
)
def update_statistical_summary(value):
    if value == "Example 1":
        path = 'data/example1.edgelist'
    elif value == "Example 2":
        path = 'data/example2.edgelist'
    else:
        # empty selection
        path = 'data/example1.edgelist'
    # get graph
    G = nx.read_edgelist(path)
    # get properties
    n,m,density,number_of_triangles,avg_clust_coeff = get_statistical_summary(G)
    deg_hist = nx.degree_histogram(G)
    max_deg = len(deg_hist)
    deg_range = [i for i in range(0,max_deg)]
    df = pd.DataFrame(dict(degree=deg_range, nr_nodes=deg_hist))
    # create bar chart 
    fig = px.bar(df, x=df.degree, y=df.nr_nodes, labels={'degree' : 'Degree', 'nr_nodes':'Nr. of nodes'}, title='Degree distribution of the Network')
    fig.update_xaxes(type='category')
    bar_chart = dcc.Graph(figure=fig)
    # insert in Divs
    return n,m,density,number_of_triangles,avg_clust_coeff, bar_chart
    

if __name__ == '__main__':
    app.run_server(debug=True)