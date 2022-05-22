from dash import Dash, html, dcc, Input, Output, State, callback

conclusion_div = html.Div([
	html.H2("Data Analysis", id="data-analysis"),
	html.P("Using our visualization tool, we find that the fairness of graph embeddings does indeed depend on the graph embedding algorithm. For \
		instance when comparing the fairness scores for the LastFM graph, nearly all of the HGCN embeddings are unfair whereas the LaplacianEigenmap \
		embeddings are fair. In fact, we observe that the LaplacianEigenmap embeddings were fair for all networks in our dataset. "),
	html.P("A second observation is that the unfairness clustered in communities. This pattern is most clearly seen when evaluting the InFoRM fairness of \
		HGCN embeddings for the Facebook graph. We can see that three of the communities are all embedded unfairly whereas the three other \
		communities are lighter."),
	html.P("Finally, the brushing and linking functionality was very effective for mapping embeddings back to nodes in the graph. This functionality \
		is useful even outside of the fairness setting because graph embeddings are often hard to connect back to graph topology. The side by side \
		mapping provides a sanity trust for how the embeddings were generated. Further, with the additional sensitive attributes in the diagnose view, \
		users are also able to debug the fairness score to understand why it is high or low."),

	html.H2("Conclusion", id="conclusion"),
	html.P("Overall, our graph embedding fairness visualization tool contributes towards an active area of research. Our tool reveals that not all \
		graph embedding algorithms are equally fair. Further, our tool provides greater transparency into how current graph embedding algorithms and \
		fairness definitions operate, allowing the user to debug each. Looking ahead, one of the challenges is overcoming the scale of the data. \
		This involves not only improving scalability and responsiveness but also designing the visualization to avoid occlusion and redundancy. Future \
		work can also develop more mature fairness definitions as the current ones (InFoRM and Fairwalk) predominantly evalute proxity between embeddings.")
	])