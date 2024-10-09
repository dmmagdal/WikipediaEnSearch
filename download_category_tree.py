# download_category_tree.py
# Source: https://github.com/rx-112358/Wiki-Category-Graph
# A quick script from rx-112358 that allows you to download the 
# wikipedia category tree.
# Python 3.9
# Windows/MacOS/Linux


import argparse
from collections import deque
import json
import os
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import msgpack
import networkx as nx
from tqdm import tqdm
import wikipediaapi


# Initialize Wikipedia API
WIKI = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')


def load_data_from_msgpack(path: str) -> Dict:
	'''
	Load a data file (to dictionary) from msgpack file given the path.
	@param: path (str), the path of the data file that is to be loaded.
	@return: Returns a python dictionary containing the structured data
		from the loaded data file.
	'''
	with open(path, 'rb') as f:
		byte_data = f.read()

	return msgpack.unpackb(byte_data)


def write_data_to_msgpack(path: str, data: Dict) -> None:
	'''
	Write data (dictionary) to a msgpack file given the path.
	@param: path (str), the path of the data file that is to be 
		written/created.
	@param: data (Dict), the structured data (dictionary) that is to be
		written to the file.
	@return: returns nothing.
	'''
	with open(path, 'wb+') as f:
		packed = msgpack.packb(data)
		f.write(packed)


# Function to recursively get all categories and subcategories
def get_all_subcategories(category: str, max_depth: int = 1, current_depth: int = 0, visited: Set[str] = None) -> Dict[str, List[str]]:
	'''
	Recursively retrieve all categories and subcategories in the 
		wikipedia category tree.
	@param: category (str), the current category.
	@param: max_depth (int), the max depth of the category tree that was 
		generated. Default is 1.
	@param: current_depth (int), the current depth in the category 
		tree. Default is 0.
	@param: visited (Set[str]), The set of all nodes (categories) 
		visited in the wikipedia category tree. Default is None.
	@return: returns all subcategories for the current category in the 
		wikipedia category graph.
	'''
	if visited is None:
		visited = set()

	if current_depth > max_depth:
		return {}
	
	# Get category page
	cat_page = WIKI.page(f"Category:{category}")
	
	# Return if this category was already processed
	if cat_page.title in visited:
		return {}
	
	visited.add(cat_page.title)
	
	subcategories = {}
	for member in cat_page.categorymembers.values():
		if member.ns == 14:  # Namespace 14 refers to categories
			subcategories[member.title] = get_all_subcategories(
				member.title.replace("Category:", ""), max_depth, current_depth + 1, visited
			)

	return subcategories


def get_all_subcategories_bfs(category: str, max_depth: int = 1, depth: int = 0) -> Dict[str, List[str]]:
	'''
	Use BFS to iteratively retrieve all categories and subcategories in
		the wikipedia category tree.
	@param: category (str), the current category.
	@param: max_depth (int), the max depth of the category tree that was 
		generated. Default is 1.
	@return: returns all subcategories for the current category in the 
		wikipedia category graph.
	'''
	visited = set()
	subcategories = dict()

	queue = deque([(category, depth)])

	while len(queue) != 0:
		current_category, current_depth = queue.popleft()

		if current_depth > max_depth:
			continue
	
		# Get category page
		cat_page = WIKI.page(f"Category:{current_category}")
	
		# Return if this category was already processed
		if cat_page.title in visited:
			continue
	
		visited.add(cat_page.title)
		
		subcategories[current_category] = list()
		for member in cat_page.categorymembers.values():
			if member.ns == 14:  # Namespace 14 refers to categories
				subcategory_name = member.title.replace("Category:", "")
				subcategories[current_category].append(subcategory_name)
				queue.append((subcategory_name, current_depth + 1))

	return subcategories


# Build a graph with all categories and subcategories
def build_full_graph(max_depth: int = 1, use_bfs: bool = False) -> nx.DiGraph:
	'''
	Build the wikipedia category graph.
	@param: max_depth (int), the maximum depth of category tree that
		will be explored. Default is 1.
	@return: returns the wikipedia category graph.
	'''
	G = nx.DiGraph()  # Directed graph
	all_categories = "Main topic classifications"  # Root category to start from

	# Recursively fetch all categories
	if use_bfs:
		subcategories = get_all_subcategories_bfs(
			all_categories, max_depth
		)
		
		for cat, subcats in subcategories.items():
			for subcat in subcats:
				G.add_edge(cat, subcat)
	else:
		subcategories = get_all_subcategories(
			all_categories, max_depth
		)

		# Helper function to add nodes/edges to the graph
		def add_to_graph(cat: str, subcats: Dict[str, List[str]]):
			for subcat, nested_subcats in subcats.items():
				G.add_edge(cat, subcat)
				add_to_graph(subcat, nested_subcats)

		add_to_graph(all_categories, subcategories)

	assert G.number_of_nodes() != 0, "Graph should have at least one node"
	assert G.number_of_edges() != 0, "Graph must have edges"

	return G


# Save the graph to a file
def save_graph(G: nx.DiGraph, file_name: str, format: str = "graphml") -> None:
	'''
	Save the graph to a file.
	@param: G (nx.DiGraph), the graph of the category tree that is to 
		be saved.
	@param: file_name (str), the filename to save the graph as.
	@param: format (str), how the graph should be saved. Default is
		"graphml".
	@return: returns nothing.
	'''
	assert file_name.endswith(format),\
		f"Expected filename {file_name} to end with appropriate extension. Recieved {format}"

	valid_extensions = ["graphml", "gml", "edgelist", "json", "msgpack"]

	if format == "graphml":
		nx.write_graphml(G, file_name)  # Save as GraphML
	elif format == "gml":
		nx.write_gml(G, file_name)      # Save as GML
	elif format == "edgelist":
		nx.write_edgelist(G, file_name) # Save as Edge List
	elif format == "json":
		with open(file_name, "w+") as f:
			json.dump(nx.to_dict_of_lists(G), f, indent=4)
			# json.dump(nx.node_link_data(G), f, indent=4)
	elif format == "msgpack":
		write_data_to_msgpack(file_name, nx.to_dict_of_lists(G))
		# write_data_to_msgpack(file_name, nx.node_link_data(G))
	else:
		raise ValueError(f"Unsupported format: choose {', '.join(valid_extensions)}.")


# Load the graph from a file
def load_graph(file_name: str, format: str = "graphml") -> nx.DiGraph:
	'''
	Load the graph from a file.
	@param: file_name (str), the filename to load the graph from.
	@param: format (str), how the graph should was saved. Default is
		"graphml".
	@return: returns  the graph of the category tree that is was to be
		loaded.
	'''
	assert file_name.endswith(format),\
		f"Expected filename {file_name} to end with appropriate extension. Recieved {format}"

	valid_extensions = ["graphml", "gml", "edgelist", "json", "msgpack"]

	if format == "graphml":
		graph = nx.read_graphml(file_name)
	elif format == "gml":
		graph = nx.read_gml(file_name)
	elif format == "edgelist":
		graph = nx.read_edgelist(file_name)
	elif format == "json":
		with open(file_name, "r") as f:
			graph = nx.from_dict_of_lists(json.load(f))
			# graph = nx.node_link_graph(json.load(f), directed=True)
	elif format == "msgpack":
		graph = nx.from_dict_of_lists(load_data_from_msgpack(file_name))
		# graph = nx.node_link_graph(load_data_from_msgpack(file_name))
	else:
		raise ValueError(f"Unsupported format: choose {', '.join(valid_extensions)}.")
	
	return graph.to_directed()


# Visualize the graph using Matplotlib
def draw_graph(G: nx.DiGraph, depth: int = 1) -> None:
	'''
	Draw the graph on matplotlib and save the figure to a png file.
	@param: G (nx.DiGraph), the graph of the category tree that is to 
		be illustrated and saved.
	@param: depth (int), the max depth of the category tree that was 
		generated. Default is 1.
	@return: returns nothing.
	'''
	plt.figure(figsize=(12, 8))
	pos = nx.spring_layout(G, k=0.8)
	nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
	# plt.show()
	plt.savefig(f"Wikipedia-category-graph_depth{depth}.png")


def convert_graph_file(depth: str, extension1: str, extension2: str) -> None:
	'''
	Load and save a graph from one file format to another.
	@param: depth (str), the max depth of the category tree that was 
		generated.
	@param: extension1 (str), the file format extension that is used 
		for the loaded file.
	@param: extension2 (str), the file format extension that is used 
		for the saved file.
	@return: returns nothing.
	'''
	valid_extensions = ["graphml", "gml", "edgelist", "json", "msgpack"]
	assert extension1 in valid_extensions,\
		f"invalid save extension used: {extension1}"
	assert extension2 in valid_extensions,\
		f"invalid save extension used: {extension2}"
	
	load_path = os.path.join(
		"./wiki_category_graphs",
		f"wiki_categories_depth{depth}.{extension1}"
	)
	save_path = os.path.join(
		"./wiki_category_graphs",
		f"wiki_categories_depth{depth}.{extension2}"
	)

	assert os.path.exists(load_path),\
		f"Could not find target file {load_path}"

	graph = load_graph(load_path, format=extension1)
	save_graph(graph, save_path, format=extension2)


def graphs_equal(graph1: nx.Graph, graph2: nx.Graph, use_isomorphic: bool = False, use_vf2pp: bool = False) -> bool:
	'''
	Determine if a graph is equal or isomorphic.
	@param: graph1 (nx.Graph), One of the graphs to be compared.
	@param: graph2 (nx.Graph), The other graph to be compared.
	@param: use_isomorphism (bool), Whether to use networkx's 
		is_isomorphic() to determine if the graphs are isomorphic.
		Default is False.
	@param: use_vf2pp (bool), Whether to use networkx's 
		vf2pp_is_isomorphic() to determine if the graphs are 
		isomorphic.Default is False.
	@return: returns whether the graphs are equal or isomorphic.
	'''
	if use_isomorphic:
		return nx.is_isomorphic(graph1, graph2)
	elif use_vf2pp:
		return nx.vf2pp_is_isomorphic(graph1, graph2)
	else:
		# NOTE:
		# Assumes the sorting of nodes (and edges) will result in the
		# entries being aligned. If the entries are not aligned, this
		# implies there is a difference which is evident of graphs that
		# do not match.

		# NOTE:
		# Checks edge sets and vertex sets between graphs.

		# Isolate the nodes.
		g1_nodes = sorted(list(graph1.nodes))
		g2_nodes = sorted(list(graph2.nodes))
		
		# Verify the number of nodes is the same.
		if len(g1_nodes) != len(g2_nodes):
			return False
		
		# Iterate through each node in the graph.
		for node_idx in tqdm(range(len(g1_nodes))):
			# Isolate the nodes.
			g1_node = g1_nodes[node_idx]
			g2_node = g2_nodes[node_idx]
			
			# Verify the number of edges from the node is the same.
			if g1_node != g2_node:
				return False
			
			# Isolate edges from the node.
			g1_node_edges = sorted(list(graph1.edges(g1_node)))
			g2_node_edges = sorted(list(graph1.edges(g2_node)))

			# Verify the number of edges from the node is the same.
			if len(g1_node_edges) != len(g2_node_edges):
				return False
			
			# Iterate through each edge in the node edges.
			for edge_idx in range(len(g1_node_edges)):
				# Isolate the edges.
				g1_node_edge = g1_node_edges[edge_idx]
				g2_node_edge = g2_node_edges[edge_idx]

				# Verify the number of edges from the node is the same.
				if g1_node_edge != g2_node_edge:
					return False

	# Return true since all checks pass.
	return True


def main():
	# Initialize argument parser and arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--depth",
		type=int, 
		default=2, 
		help="Max level of depth to go in the wikipedia category tree. Default is 2."
	)
	parser.add_argument(
		"--draw_graph",
		action="store_true",
		help="Whether to draw and save the graph to png. Default is false/not specified."
	)
	parser.add_argument(
		"--use_bfs",
		action="store_true",
		help="Whether to use BFS to the recursive DFS algorithms to build the graph. Default is false/not specified."
	)

	# Parse arguments.
	args = parser.parse_args()

	# Control the depth of subcategory exploration.
	depth = args.depth
	
	# Build the graph for all categories
	G = build_full_graph(depth, args.use_bfs)

	# Save the graph to a file
	save_path = os.path.join(
		"./wiki_category_graphs",
		f"wiki_categories_depth{depth}.graphml"
	)
	save_graph(G, save_path, format="graphml")  # Change format to 'gml' or 'edgelist' as needed

	if args.draw_graph:
		# Draw the graph
		draw_graph(G, depth)

	# Convert graph between formats.
	convert_graph_file(depth, "graphml", "json")
	convert_graph_file(depth, "json", "msgpack")
	graphml_graph = load_graph(
		f"./wiki_category_graphs/wiki_categories_depth{depth}.graphml", 
		"graphml"
	)
	msgpack_graph = load_graph(
		f"./wiki_category_graphs/wiki_categories_depth{depth}.msgpack", 
		"msgpack"
	)
	json_graph = load_graph(
		f"./wiki_category_graphs/wiki_categories_depth{depth}.json", 
		"json"
	)

	print(f"graphml matches json: {graphs_equal(graphml_graph, json_graph)}")								# True
	print(f"graphml matches msgpack: {graphs_equal(graphml_graph, msgpack_graph)}")							# True
	print(f"json matches msgpack: {graphs_equal(json_graph, msgpack_graph)}")								# True
	# print(f"graphml matches json: {graphs_equal(graphml_graph, json_graph, use_isomorphic=True)}")			# False
	# print(f"graphml matches msgpack: {graphs_equal(graphml_graph, msgpack_graph, use_isomorphic=True)}")	# False
	# print(f"json matches msgpack: {graphs_equal(json_graph, msgpack_graph, use_isomorphic=True)}") 			# Takes a REALLY long time for some reason compared to the above comparison
	# print(f"graphml matches json: {graphs_equal(graphml_graph, json_graph, use_vf2pp=True)}")				# False
	# print(f"graphml matches msgpack: {graphs_equal(graphml_graph, msgpack_graph, use_vf2pp=True)}")			# False
	# print(f"json matches msgpack: {graphs_equal(json_graph, msgpack_graph, use_vf2pp=True)}") 				# Takes a REALLY long time (still) for some reason compared to the above comparison

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()