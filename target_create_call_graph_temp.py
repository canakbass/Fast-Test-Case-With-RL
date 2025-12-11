"""Graph visualization for code structure and flow."""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any
import os
from code_analyzer import CodeAnalyzer


class GraphVisualizer:
    """Visualizes code structure and control flow using NetworkX and Graphviz."""
    
    def __init__(self, target_file: str):
        """
        Initialize graph visualizer.
        
        Args:
            target_file: Path to the Python file to visualize
        """
        self.target_file = target_file
        self.analyzer = CodeAnalyzer(target_file)
        self.analysis = self.analyzer.get_full_analysis()
        self.call_graph = self.analysis['call_graph']
    
    def create_call_graph(self) -> nx.DiGraph:
        """Create and return the function call graph."""
        return self.call_graph
    
    def visualize_call_graph(self, output_file: str = "call_graph.png"):
        """
        Visualize the function call graph.
        
        Args:
            output_file: Path to save the visualization
        """
        plt.figure(figsize=(14, 10))
        
        # Get node colors based on type
        node_colors = []
        for node in self.call_graph.nodes():
            node_type = self.call_graph.nodes[node].get('type', 'function')
            if node_type == 'class':
                node_colors.append('lightblue')
            elif node_type == 'method':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightcoral')
        
        # Layout
        pos = nx.spring_layout(self.call_graph, k=1, iterations=50)
        
        # Draw
        nx.draw(
            self.call_graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=2000,
            font_size=9,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            arrowstyle='->',
        )
        
        plt.title("Function Call Graph", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_complexity_graph(self) -> nx.Graph:
        """Create graph showing function complexity."""
        graph = nx.Graph()
        
        complexity_data = self.analysis['cyclomatic_complexity']
        
        for item in complexity_data:
            graph.add_node(
                item['name'],
                complexity=item['complexity'],
                rank=item['rank'],
            )
        
        return graph
    
    def visualize_complexity_graph(self, output_file: str = "complexity_graph.png"):
        """
        Visualize function complexity as a graph.
        
        Args:
            output_file: Path to save the visualization
        """
        complexity_graph = self.create_complexity_graph()
        
        if len(complexity_graph.nodes()) == 0:
            print("No complexity data to visualize")
            return None
        
        plt.figure(figsize=(14, 8))
        
        # Get complexity values for sizing
        complexities = [
            complexity_graph.nodes[node].get('complexity', 1)
            for node in complexity_graph.nodes()
        ]
        
        # Normalize sizes
        max_complexity = max(complexities) if complexities else 1
        node_sizes = [
            (c / max_complexity) * 3000 + 500
            for c in complexities
        ]
        
        # Get colors based on rank
        rank_colors = {
            'A': 'green',
            'B': 'lightgreen',
            'C': 'yellow',
            'D': 'orange',
            'E': 'red',
            'F': 'darkred',
        }
        
        node_colors = [
            rank_colors.get(complexity_graph.nodes[node].get('rank', 'A'), 'gray')
            for node in complexity_graph.nodes()
        ]
        
        # Layout - circular for better visibility
        pos = nx.circular_layout(complexity_graph)
        
        # Draw
        nx.draw(
            complexity_graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=9,
            font_weight='bold',
            edge_color='lightgray',
        )
        
        # Add complexity values as labels
        labels = {
            node: f"{node}\n(CC: {complexity_graph.nodes[node].get('complexity', 0)})"
            for node in complexity_graph.nodes()
        }
        nx.draw_networkx_labels(complexity_graph, pos, labels, font_size=7)
        
        plt.title("Cyclomatic Complexity by Function", fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Rank A (Low)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                      markersize=10, label='Rank C (Medium)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Rank E (High)'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_coverage_graph(self, covered_functions: set, covered_branches: set) -> nx.Graph:
        """
        Create graph showing coverage status.
        
        Args:
            covered_functions: Set of covered function names
            covered_branches: Set of covered branch IDs
        
        Returns:
            NetworkX graph with coverage information
        """
        graph = nx.Graph()
        
        functions = self.analysis['functions']
        for func in functions:
            is_covered = func['name'] in covered_functions
            graph.add_node(
                func['name'],
                covered=is_covered,
                type='function',
            )
        
        return graph
    
    def visualize_coverage_graph(
        self,
        covered_functions: set,
        covered_branches: set,
        output_file: str = "coverage_graph.png"
    ):
        """
        Visualize coverage status.
        
        Args:
            covered_functions: Set of covered function names
            covered_branches: Set of covered branch IDs
            output_file: Path to save the visualization
        """
        coverage_graph = self.create_coverage_graph(covered_functions, covered_branches)
        
        plt.figure(figsize=(12, 8))
        
        # Colors: green for covered, red for uncovered
        node_colors = [
            'lightgreen' if coverage_graph.nodes[node].get('covered', False) else 'lightcoral'
            for node in coverage_graph.nodes()
        ]
        
        pos = nx.spring_layout(coverage_graph, k=1.5)
        
        nx.draw(
            coverage_graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=2500,
            font_size=10,
            font_weight='bold',
        )
        
        covered_count = sum(1 for n in coverage_graph.nodes() if coverage_graph.nodes[n].get('covered', False))
        total_count = len(coverage_graph.nodes())
        coverage_pct = (covered_count / total_count * 100) if total_count > 0 else 0
        
        plt.title(f"Coverage Status: {covered_count}/{total_count} functions ({coverage_pct:.1f}%)", 
                 fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file


if __name__ == "__main__":
    # Test the visualizer
    visualizer = GraphVisualizer("target_code.py")
    
    print("Generating call graph...")
    call_graph_file = visualizer.visualize_call_graph()
    print(f"Call graph saved to: {call_graph_file}")
    
    print("\nGenerating complexity graph...")
    complexity_file = visualizer.visualize_complexity_graph()
    print(f"Complexity graph saved to: {complexity_file}")
    
    print("\nGenerating coverage graph...")
    # Example: some functions covered
    covered = {'factorial', 'is_prime', 'fibonacci'}
    coverage_file = visualizer.visualize_coverage_graph(covered, set())
    print(f"Coverage graph saved to: {coverage_file}")
