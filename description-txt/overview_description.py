from dash import Dash, html, dcc, Input, Output, State, callback

description = html.Div([
		html.H1("Visual Examination of Fairness for Graph Embedding Algorithms"),
		html.P("CS 7250 Information Visualization"),
		html.P("Agapi Rissaki, Bruno Scarone, David Liu"),
		#
		#
		html.Ul([
			html.Li([html.A("Motivation", href="#motivation")]),
			html.Li([html.A("Background", href="#background")]),
			html.Li([html.A("Data", href="#data")]),
			html.Li([html.A("Task Analysis", href="#tasks")]),
			html.Li([html.A("Design Process", href="#design")]),
			html.Li([html.A("Visualization", href="#visualization")]),
			html.Li([html.A("Data Analysis", href="#data-analysis")]),
			html.Li([html.A("Conclusion", href="#conclusion")])
			]),
	#
	#
	html.H2("Motivation", id="motivation"),
	html.P("With the growing use of machine learning models to automate decisions, a rising concern \
		is to assure these are not biased (systematically unfair) against specific population groups. The algorithmic\
		fairness community has recently begun investigating the fairness of graph embeddings. Graph embeddings, synonymous\
		with node embeddings, transform every node to a d-dimensional vector. The vector representation\
		can then be fed into downstream machine learning applications such as drug discovery and social media \
		recommendations."),
	html.P("However, there are many difficulties with evaluating the fairness of graph embedding algorithms. First,\
		existing notions of embedding fairness assign fairness scores to nodes, but given the size of the graph,\
		it is difficult to know how the unfairness is distributed throughout the graph and why certain nodes are \
		embedded unfairly. Second, as there are many graph embedding algorithms, it is important to compare the \
		relative fairness of algorithms. Practitioners may compare the fairness of algorithms before applying \
		embeddings."),\
	#
	#
	html.H2("Background", id="background"),
	html.H3("Graph Embeddings"),
	html.P("Graph embedding algorithms convert each node in a graph/network into a d-dimensional vector. The process \
		is similar to that of word embeddings, which represent individual words as low-dimensional vectors (embeddings). \
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
	html.H2("Data", id="data"),
	html.P(children=["Our data consist of the graph embeddings for four real-world graphs, which are considered common benchmarks. These graphs span multiple domains \
		including social media (Facebook, LastFM) and biology (protein-protein). Each \
		network is embedded with six common graph embedding algorithms. These algorithms span the major classes \
		of embedding algorithms: matrix factorization (SVD, HOPE, Laplacian Eigenmap), random-walk (Node2Vec), \
		and deep learning (SDNE, HGCN). Further details about the data and the data itself are available at ",
		html.A("this repository", href="https://dliu18.github.io/embedding_repo/"),"."
		]),
	#
	#
	html.H2("Task Analysis", id="tasks"),
	# summary of interview and task table
	html.P("To better understand current challenges facing practitioners working with existing graph visualization and algorithmic fairness tools, we interviewed Professor Tina Eliassi-Rad and Dr. Brennan Klein. Professor Eliassi-Rad is a Professor of Computer Science and a core member of Northeastern’s Network Science Institute; she has led research in both of the domains of interest and is familiar with many of the real-world applications. Dr. Brennan Klein is a postdoctoral researcher at Northeastern’s Network Science Institute, who is also an expert in the network visualization domain."),
	html.P("Based on the interviews with our two domain experts, we decided to prioritize the following tasks, which are supported by our visualization: "),
	html.Ul([
			html.Li("Statistical Summary (T1): provide an overview of the graph and its statistical properties."),
			html.Li("Embedding comparison with respect to fairness (T2): compare how two sets of embeddings differ in \
				the fairness of individual node embeddings."),
			html.Li("Unfairness diagnostics (T3): for a given embedding, find which nodes contribute to unfairness and \
				why.")
		]),
	# html.P("These tasks are sequential and correspond to our workflow of an overview followed by a drill down investigation of unfairness. The design choices are justified in the next section"),
	html.P("To perform the tasks selection, we focused on the ones that best facilitate the goal of our project: to allow for a complete unfairness diagnosis workflow for graph embeddings. The sequential workflow consists of an overview step followed by a drill-down or diagnose step, which are described next:"),
	html.Ul([
			html.Li("Overview: Comparison between two embedding algorithms, augmented by relevant graph statistics. The user will be able to determine whether the embedding of interest displays signs of unfairness."),
			html.Li("Drill-down/Diagnosis: Identification of the unfairness source(s) for a given embedding. The user will be able to discover the determining factors that lead to the observed unfairness.")
	]),
	html.P("The design choices are justified in the next section."),
	#
	#
	html.H2("Design Process", id="design"),
	# sketches and design choices to justify final visualization
	#into? html.P("")
	html.P("We satisfy the statistical summary task with the “Statistical summary of the network” section. The summary provides key graph metrics to help the user understand the type of graph being studied. For the Degree Distribution the bar chart encoding was selected, since it makes use of the most effective magnitude channel to encode ordered attributes, according to the effectiveness ranking for visual channels presented in Munzer's VAD."),
	
	html.P("Next, we achieve the comparison task by providing side-by-side comparisons. The same network is shown in both sides but the node colors depend on the fairness of the respective embedding algorithm. The color channel is chosen to encode the score attribute, as other channels could negatively affect the representation of the network. A sequential colorscale is used, since the attribute being encoded is quantitative. We associated darker colors with a higher unfairness score based on the feedback collected from the Usability Testing."),

	html.P("In order to improve the network visibility, we filter out visually non-salient edges. Specifically, we only display edges in the top 10 percent of length. Filtering allows us to save on browser memory, as the number of edges far exceeds the number of nodes. At the same time, we found that the bottom 90 percent of shortest edges were often not visible anyways, so filtering offers large performance increases without loss in visual information."),

	# Both panels showing the network support T2. On top of them, the corresponding embedding algorithms can be selected from a dropdown list. On the right of them, the score function used can be selected. After executing the algorithms and computing the scores, these values are used to color the nodes of the network. The color scale used is shown below the selected score function. 

	html.P("Finally, we address the “unfairness” diagnostics task with the “diagnostics” view that provides the functionality to list the most unfair nodes along with their local topology. These local visualizations explain the fairness score and help to identify causes of unfairness. A red color pop-out effect is used to distinguish the selected focal node, together with an increase in its size. This allows for a more efficient search of it, which is an essential component to understand the score computation. Brushing and linking is supported between the network and the embedding space to analyze how different nodes contribute to the fairness score."),
	
	html.H2(children='Final Visualization: Overview', id="visualization"),
	html.P(["Our visualisations were developed using the ",html.A(href="https://plotly.com/dash/", children="Dash")," Python framework, written on top of Plotly.js and React.js."]),
	])
	# “Final Visualization” (i.e., final visualization, design justifications, packages utilized for coding, and UI walk-through), 
	# “Data Analysis” (i.e., summary of interesting results), and 
	# “Conclusion” (i.e., short summary of work completed and areas for improvement/future-work).  
	# Additional non-required sections may be added to the website to thoroughly explain/frame the final project.