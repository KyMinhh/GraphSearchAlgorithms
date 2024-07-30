import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from collections import deque
import os
import pandas as pd

class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Use directed graph for clear path directions

    def add_edge(self, u, v, weight=1):
        self.graph.add_edge(u, v, weight=weight)

    def remove_edge(self, u, v):
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)

    def add_node(self, u):
        self.graph.add_node(u)

    def remove_node(self, u):
        if self.graph.has_node(u):
            self.graph.remove_node(u)

    def dfs(self, start, goal):
        stack = [start]
        visited = set()
        parent = {start: None}

        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)

            if vertex == goal:
                path = []
                while vertex is not None:
                    path.append(vertex)
                    vertex = parent[vertex]
                return path[::-1]

            for neighbor in self.graph.neighbors(vertex):
                if neighbor not in visited:
                    stack.append(neighbor)
                    parent[neighbor] = vertex
        return []

    def bfs(self, start, goal):
        queue = deque([start])
        visited = {start}
        parent = {start: None}

        while queue:
            vertex = queue.popleft()
            if vertex == goal:
                path = []
                while vertex is not None:
                    path.append(vertex)
                    vertex = parent[vertex]
                return path[::-1]

            for neighbor in self.graph.neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    parent[neighbor] = vertex
        return []

def draw_initial_graph(graph):
    pos = nx.spring_layout(graph)  # Use spring layout for better visualization
    plt.figure(figsize=(10, 8))

    # Draw initial graph
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=14,
            font_weight='bold', arrows=True)

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black', font_size=12)

    plt.title('Initial Graph', fontsize=18)
    plt.savefig('initial_graph.png', format='png')  # Save initial graph image
    plt.close()

def draw_updated_graph(graph, path_dfs, path_bfs, start, goal):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))

    # Draw graph with paths
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=14,
            font_weight='bold', arrows=True)

    # Draw paths with different styles
    if path_dfs:
        edges_dfs = list(zip(path_dfs, path_dfs[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges_dfs, edge_color='red', width=2, style='dashed', label='DFS Path')

    if path_bfs:
        edges_bfs = list(zip(path_bfs, path_bfs[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges_bfs, edge_color='green', width=2, style='dotted',
                               label='BFS Path')

    # Highlight START and GOAL nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=[start, goal], node_color='orange', node_size=2500, label='START/GOAL')

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black', font_size=12)

    plt.title('Graph with Search Paths and Highlighted Nodes', fontsize=18)

    # Add legend
    plt.legend(prop={'size': 12})

    plt.savefig('graph.png', format='png')  # Save updated graph image
    plt.close()

def draw_graph_with_tables():
    # Draw and save the initial graph
    pos = nx.spring_layout(g.graph)
    plt.figure(figsize=(10,8))
    nx.draw(g.graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=14,
            font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(g.graph, 'weight')
    nx.draw_networkx_edge_labels(g.graph, pos, edge_labels=edge_labels, font_color='black', font_size=12)
    plt.title('Initial Graph', fontsize=18)
    plt.savefig('initial_graph.png', format='png')
    plt.close()

    # Create adjacency matrix, adjacency list, and edge list
    adj_matrix = nx.adjacency_matrix(g.graph).todense()
    adj_df = pd.DataFrame(adj_matrix, columns=range(g.graph.number_of_nodes()), index=range(g.graph.number_of_nodes()))

    adj_list = {node: list(g.graph.neighbors(node)) for node in g.graph.nodes()}
    adj_list_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in adj_list.items()]))

    edge_list = list(g.graph.edges)
    edge_list_df = pd.DataFrame(edge_list, columns=["From", "To"])

    # Create tables
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].axis('tight')
    ax[0].axis('off')
    ax[0].table(cellText=adj_df.values, colLabels=adj_df.columns, rowLabels=adj_df.index, cellLoc='center', loc='center')
    ax[0].set_title("Adjacency Matrix", fontsize=14)

    ax[1].axis('tight')
    ax[1].axis('off')
    ax[1].table(cellText=adj_list_df.values, colLabels=adj_list_df.columns, cellLoc='center', loc='center')
    ax[1].set_title("Adjacency List", fontsize=14)

    ax[2].axis('tight')
    ax[2].axis('off')
    ax[2].table(cellText=edge_list_df.values, colLabels=edge_list_df.columns, cellLoc='center', loc='center')
    ax[2].set_title("Edge List", fontsize=14)

    plt.savefig('graph_tables.png', format='png')
    plt.close()

    # Open the images
    graph_img = Image.open('initial_graph.png')
    tables_img = Image.open('graph_tables.png')

    # Concatenate images side by side
    combined_img_width = graph_img.width + tables_img.width + 20  # Added space
    combined_img = Image.new('RGB', (combined_img_width, max(graph_img.height, tables_img.height)), 'white')
    combined_img.paste(graph_img, (0, 0))
    combined_img.paste(tables_img, (graph_img.width + 10, 0))  # Added space between images

    # Save the final combined image
    combined_img.save('final_graph_with_tables.png')

    # Update the canvas with the combined image
    update_canvas_with_image(combined_img)

def update_plot(algorithm):
    start_node = start_entry.get()
    goal_node = goal_entry.get()

    if start_node not in g.graph.nodes or goal_node not in g.graph.nodes:
        messagebox.showerror("Error", "One or both nodes are not in the graph.")
        return

    path_dfs, path_bfs = [], []
    cost_dfs, cost_bfs = 0, 0

    if algorithm == 'DFS':
        path_dfs = g.dfs(start_node, goal_node)
        cost_dfs = len(path_dfs) - 1
    elif algorithm == 'BFS':
        path_bfs = g.bfs(start_node, goal_node)
        cost_bfs = len(path_bfs) - 1

    # Draw and save updated graph
    draw_updated_graph(g.graph, path_dfs, path_bfs, start_node, goal_node)

    # Check if the image file exists
    if not os.path.exists('graph.png'):
        messagebox.showerror("Error", "Image file not found. Ensure the graph is correctly drawn.")
        return

    # Update canvas with the new image
    try:
        updated_image = Image.open('graph.png')
        update_canvas_with_image(updated_image)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def update_canvas_with_image(image):
    global canvas
    if canvas is not None:
        canvas.get_tk_widget().destroy()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image)
    ax.axis('off')
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def add_node():
    node = node_entry.get()
    if node:
        g.add_node(node)
        draw_initial_graph(g.graph)
        update_canvas_with_image(Image.open('initial_graph.png'))

def remove_node():
    node = node_entry.get()
    if node:
        g.remove_node(node)
        draw_initial_graph(g.graph)
        update_canvas_with_image(Image.open('initial_graph.png'))

def add_edge():
    u = edge_start_entry.get()
    v = edge_end_entry.get()
    weight = edge_weight_entry.get()

    if u and v and weight:
        g.add_edge(u, v, weight=int(weight))
        draw_initial_graph(g.graph)
        update_canvas_with_image(Image.open('initial_graph.png'))

def remove_edge():
    u = edge_start_entry.get()
    v = edge_end_entry.get()
    if u and v:
        g.remove_edge(u, v)
        draw_initial_graph(g.graph)
        update_canvas_with_image(Image.open('initial_graph.png'))

# Create the graph and add edges
g = Graph()
g.add_edge('A', 'B', weight=1)
g.add_edge('A', 'C', weight=2)
g.add_edge('B', 'D', weight=1)
g.add_edge('C', 'D', weight=1)
g.add_edge('C', 'E', weight=3)
g.add_edge('D', 'E', weight=1)
g.add_edge('D', 'F', weight=2)
g.add_edge('E', 'F', weight=1)

# Create the main application window
root = tk.Tk()
root.title("Graph Search Algorithms")

# Create GUI components
frame = tk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a frame for graph display
graph_frame = tk.Frame(frame)
graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a frame for tables
table_frame = tk.Frame(frame)
table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

tk.Label(frame, text="Start Node:").pack(side=tk.TOP, anchor=tk.W)
start_entry = tk.Entry(frame)
start_entry.pack(side=tk.TOP, fill=tk.X)

tk.Label(frame, text="Goal Node:").pack(side=tk.TOP, anchor=tk.W)
goal_entry = tk.Entry(frame)
goal_entry.pack(side=tk.TOP, fill=tk.X)

tk.Label(frame, text="Node:").pack(side=tk.TOP, anchor=tk.W)
node_entry = tk.Entry(frame)
node_entry.pack(side=tk.TOP, fill=tk.X)

tk.Label(frame, text="Edge Start:").pack(side=tk.TOP, anchor=tk.W)
edge_start_entry = tk.Entry(frame)
edge_start_entry.pack(side=tk.TOP, fill=tk.X)

tk.Label(frame, text="Edge End:").pack(side=tk.TOP, anchor=tk.W)
edge_end_entry = tk.Entry(frame)
edge_end_entry.pack(side=tk.TOP, fill=tk.X)

tk.Label(frame, text="Edge Weight:").pack(side=tk.TOP, anchor=tk.W)
edge_weight_entry = tk.Entry(frame)
edge_weight_entry.pack(side=tk.TOP, fill=tk.X)

tk.Button(frame, text="Add Node", command=add_node).pack(side=tk.TOP, fill=tk.X)
tk.Button(frame, text="Remove Node", command=remove_node).pack(side=tk.TOP, fill=tk.X)
tk.Button(frame, text="Add Edge", command=add_edge).pack(side=tk.TOP, fill=tk.X)
tk.Button(frame, text="Remove Edge", command=remove_edge).pack(side=tk.TOP, fill=tk.X)

# Create menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# Create Algorithms menu
algorithms_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Algorithms", menu=algorithms_menu)

# Add algorithm options to menu
algorithms_menu.add_command(label="DFS", command=lambda: update_plot('DFS'))
algorithms_menu.add_command(label="BFS", command=lambda: update_plot('BFS'))

# Add option to display tables
menu_bar.add_command(label="Show Graph with Tables", command=draw_graph_with_tables)

# Draw and display the initial graph
draw_initial_graph(g.graph)

# Initialize the canvas
fig, ax = plt.subplots(figsize=(10, 8))
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Display the initial graph when the application starts
update_canvas_with_image(Image.open('initial_graph.png'))

root.mainloop()