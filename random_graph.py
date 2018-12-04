import networkx as nx

real_graph = None # todo: load in graph edges once we have it
din = list(real_graph.in_degree().values())
dout = list(real_graph.out_degree().values())

gnp_graph = nx.directed_configuration_model(din, dout)
# gnp_graph = nx.DiGraph(gnp_graph) # Remove multi-edges todo: I commented this out since we want multi edges?
gnp_graph = gnp_graph.remove_edges_from(gnp_graph.selfloop_edges()) # Remove parallel edges
