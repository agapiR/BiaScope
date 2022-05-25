from dash import Dash, html, dcc, Input, Output, State, callback

description = html.Div([
		html.H1("Visual Examination of Fairness for Graph Embedding Algorithms"),
		html.P("CS 7250 Information Visualization"),
		html.P("Agapi Rissaki, Bruno Scarone, David Liu"),
		#
		#
		#
		html.Ul([
			html.Li([html.A("Diagnostics: Description", href="#description")]),
			html.Li([html.A("Diagnostics: Usage", href="#usage")]),
			html.Li([html.A("Diagnostics: Visualization", href="#visualization")])
			]),
	#
	#
	#
	html.H2("Description", id="description"),
	html.P("The purpose of this view is to facilitate the “unfairness” diagnostics task, as described in our Task Analysis. \
	The view consists of tree components: the “node-score” interactive table, the local network visualization along with the corresponding local node embeddings visualization.\
	The three components allow the user to “drill-down” in order to discover the source of observed unfairness for a given network and embedding algorithm. The user can \
	specify the fairness notion configuration, which in turn determines the “node-score” table entries. Consecutively, the user can select the focal node from the table \
	based on the unfairness score or the node id. Given the focal node selection, which is indicated by color and increased size (utilizing the pop-out effect), the local network topology \
	and local node embeddings are produced. The default view uses as focal node the most unfair node according to the selected fairness notion configuration. \
	The local network topology is the Ego Network of the selected node with radius 1 (1-hop) or 2 (2-hop), which consists of the selected node, its neighbors within distance upper bounded by the \
	radius and all the edges between them. The local node embeddings consist of the embeddings of the selected node and its neighbors, projected in the 2-dimensional space using the well-known PCA." ),\
	#
	#
	#
	html.H2("Usage", id="usage"),
	#
	html.H3("Configuration and Filtering"),
	html.P("The fairness notion configuration along with the network and embedding selections are exactly the same as in the overview (simple dropdown lists) \
		to avoid confusion and allow for more flexibility." ),
	html.P("The “node-score” interactive table allows the user to sort the entries in ascending as well as descending order with respect to either the node id \
		or the score value. The user can sort the entries by clicking on the arrows displayed in the header. The custom sorting allows the user to lookup and browse \
		using either the node id or the score value. The user can select the focal node using the radio buttons on the left of each row." ),
	#
	html.H3("Color Encoding"),
	html.P("For both the local network topology and the local node embeddings we use color encoding. For the InFoRM score the mark color encodes the distance from the focal node. The distance attribute is ordinal. \
	For the Fairwalk score the mark color encodes the sensitive attribute value. For the networks we use this attribute, which is categorical. We also use the red color in the mark contour to indicate the focal node, utilizing the effective pop-out effect for quick lookup. \
	More information about the color encoding is provided in the legend." ),
	#
	html.H3("Brushing and Linking"),
	html.P("Our visualization supports brushing and linking between the local network topology and the local node embeddings. This functionality allows the user to identify and explore shared patterns \
	between the two views which can reveal the source of the unfairness for a selected node. The user can brush on either of the views and the selected nodes will be highlighted in the other view. \
	The user can double click to clear the brush selection." ),\
	#
	#
	#
	html.H2("Final Visualization: Diagnose View", id="visualization"),
	html.P(["Our visualisations were developed using the ",html.A(href="https://plotly.com/dash/", children="Dash")," Pyhton framework, written on top of Plotly.js and React.js."])
	])