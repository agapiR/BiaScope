from dash import Dash, html, dcc, Input, Output, State, callback

conclusion_div = html.Div([
	html.P(["Our visualisations were developed using the ",html.A(href="https://plotly.com/dash/", children="Dash")," Python framework, written on top of Plotly.js and React.js."]),
	])