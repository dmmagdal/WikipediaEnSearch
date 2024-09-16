# download_category_tree.py
# Source: https://github.com/rx-112358/Wiki-Category-Graph
# A quick script from rx-112358 that allows you to download the 
# wikipedia category tree.
# Python 3.9
# Windows/MacOS/Linux


import argparse
import json
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import networkx as nx
import wikipediaapi


# Initialize Wikipedia API
WIKI = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')


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


# Build a graph with all categories and subcategories
def build_full_graph(max_depth: int = 1) -> nx.DiGraph:
	'''
	Build the wikipedia category graph.
	@param: max_depth (int), the maximum depth of category tree that
		will be explored. Default is 1.
	@return: returns the wikipedia category graph.
	'''
	G = nx.DiGraph()  # Directed graph
	all_categories = "Main topic classifications"  # Root category to start from

	# Recursively fetch all categories
	subcategories = get_all_subcategories(all_categories, max_depth)

	# Helper function to add nodes/edges to the graph
	def add_to_graph(cat, subcats):
		for subcat, nested_subcats in subcats.items():
			G.add_edge(cat, subcat)
			add_to_graph(subcat, nested_subcats)

	add_to_graph(all_categories, subcategories)
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
	if format == "graphml":
		nx.write_graphml(G, file_name)  # Save as GraphML
	elif format == "gml":
		nx.write_gml(G, file_name)      # Save as GML
	elif format == "edgelist":
		nx.write_edgelist(G, file_name) # Save as Edge List
	elif format == "json":
		with open(file_name, "w+") as f:
			json.dump(nx.to_dict_of_dicts(G), f, indent=4)
	# elif format == "msgpack":
	# 	pass
	else:
		raise ValueError("Unsupported format: choose 'graphml', 'gml', or 'edgelist'.")


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


def main():
	# Initialize argument parser and arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--depth",
		type=int, 
		default=2, 
		help="Max level of depth to go in the wikipedia category tree. Default is 2."
	)

	# Parse arguments.
	args = parser.parse_args()

	# Control the depth of subcategory exploration
	depth = args.depth
	
	# Build the graph for all categories
	G = build_full_graph(depth)

	# Save the graph to a file
	save_graph(G, f"wiki_categories_depth{depth}.graphml", format="graphml")  # Change format to 'gml' or 'edgelist' as needed

	# Draw the graph
	draw_graph(G, depth)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()