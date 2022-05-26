from dash import Dash, html, dcc, Input, Output, State, callback

description = html.Div([
		html.H1("BiaScope: Visual Unfairness Diagnosis for Graph Embeddings"),
		#
		#
		html.Ul([
			html.Li([html.A("Demo Video", href="#demo")]),
			html.Li([html.A("Description", href="#description")]),
			html.Li([html.A("Overview", href="#visualization")])
			]),
	#
	#
	html.H1("Demo Video", id="demo"),
	html.Video(controls=True, width=500, children=[
		html.Source(src="/demo.mp4", type="video/mp4")]),
	#
	html.H1("Description", id="description"),
	html.H3("Graph Embeddings"),
	html.P("Graph embedding algorithms convert each node in a graph/network into a d-dimensional vector. \
		While there are dozens of graph embedding algorithms they nearly all share the same core idea: nodes that are \
		similar in the structural graph should be embedded similarly in the d-dimensional vector space. For example, \
		nodes that are in the same community should also be clustered together in vector space."),
	html.P(["Once nodes are represented as vectors, the embeddings can be passed to downstream machine learning \
		models. Common applications include classifying the functions of proteins, inferring the topics of documents, \
		and recommending connections between users in a social network. For a more in-depth overview of graph embeddings \
		and their applications please see William Hamilton's book titled ",
		html.A("Graph Representation Learning.", href="https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf")]),
	html.H3("Fairness of Graph Embeddings"),
	html.P("Recent work has evaluated the fairness of graph embeddings and builds upon broader fairness definitions \
		established in the algorithmic fairness community. Fairness definitions typically fall into two classes: \
		individual and group. Individual fairness ensures that algorithms treat two similar individuals similarly. \
		In contrast, group fairness ensures that two subpopulations, in aggregate, are treated similarly. Our \
		project incorporates embedding fairness definitions from both classes."),
	html.P([
			html.A("InFoRM:", href="https://dl.acm.org/doi/abs/10.1145/3394486.3403080"),
			" This is a definition of individual fairness for graph embeddings. InFoRM assigns a non-negative score \
			to each node. The score indicates how differently the node is embedded from other nodes in the neighborhood, \
			where the neighborhood is defined by the number of hops from the target node."
		]),
	html.P([
			html.A("Fairwalk:", href="https://www.ijcai.org/proceedings/2019/456"),
			" This is a definition of group fairness for graph embeddings. To start, a set of link recommendations \
			are made for each node. Recommendations are the top k closest embeddings. We then measure the proportion \
			of links belonging to a specific population (e.g. number of edges recommending men and women). A high score \
			suggests that one subpopulation is recommended disproportionately more."
		]),
	#
	#
	html.H1(children='Overview', id="visualization"),
	])