from dash import Dash, dcc, html, Input, Output, callback

import dash_app_overview
import dash_app_diagnostics_with_interaction

external_stylesheets = [
    {
        "href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
        "rel": "stylesheet"
    }
]

app = Dash(__name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
    #dcc.Store(id='network-name', data="Facebook", storage_type='memory'), # 'memory' or 'session'
    #dcc.Store(id='embedding-name', data="Node2Vec", storage_type='memory')
    #dcc.Store(id='focal-node-id', data=[], storage_type='memory')
])

@callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return dash_app_overview.overview_layout
    elif pathname == '/diagnostics':
        return dash_app_diagnostics_with_interaction.diagnostics_layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)