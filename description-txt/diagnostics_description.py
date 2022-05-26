from dash import Dash, html, dcc, Input, Output, State, callback

description = html.Div([
		html.H1("BiaScope: Visual Unfairness Diagnosis for Graph Embeddings"),
		#
		#
		#
		html.Ul([
			html.Li([html.A("Description", href="#description")]),
			html.Li([html.A("Usage", href="#usage")]),
			html.Li([html.A("Diagnose View", href="#visualization")])
			]),
	#
	#
	#
	html.H1("Description", id="description"),
	html.P("The purpose of this view is to facilitate the unfairness diagnostics task. \
	The view consists of tree components: the “node-score” interactive table, the local network visualization along with the corresponding local node embeddings visualization.\
	The user can specify the fairness notion configuration, which in turn determines the “node-score” table entries. Consecutively, the user can select the focal node from the table \
	based on the unfairness score or the node id. Given the focal node selection, which is indicated by color and increased size, the local network topology \
	and local node embeddings are produced. The default view uses as focal node the most unfair node according to the selected fairness notion configuration. " ),\
	#
	#
	#
	html.H1("Usage", id="usage"),
	#
	html.H3("Configuration and Filtering"),
	html.P("The “node-score” interactive table supports filtering as well as sorting in ascending or descending order with respect to either the node id \
		or the score value. Sort the entries by clicking on the arrows displayed in the header. Search by typing on the filter cells. \
		To select the focal node, use the radio buttons on the left of each row." ),

	#
	html.H3("Linked Views"),
	html.P("Diagnose View supports brushing and linking between the local network topology and the local node embeddings. \
		 Brush on either of the views and the selected nodes will be highlighted in the other view. \
		 Double click to clear the brush selection." ),\
	#
	#
	#
	html.H1("Diagnose View", id="visualization"),
	])