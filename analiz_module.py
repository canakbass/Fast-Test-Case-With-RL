import ast
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
import radon.raw as radon_raw
import networkx as nx
from graphviz import Digraph
import io

def get_metrics(code):
    """
    Calculates code metrics using radon.
    Returns a dictionary of metrics.
    """
    metrics = {}

    # Raw metrics
    raw = radon_raw.analyze(code)
    metrics['LOC'] = raw.loc
    metrics['LLOC'] = raw.lloc
    metrics['SLOC'] = raw.sloc
    metrics['Comments'] = raw.comments
    metrics['Multi-line comments'] = raw.multi
    metrics['Blank lines'] = raw.blank

    # Cyclomatic Complexity
    cc_blocks = radon_cc.cc_visit(code)
    if cc_blocks:
        metrics['Average Complexity'] = radon_cc.cc_rank(sum(c.complexity for c in cc_blocks) / len(cc_blocks))
        metrics['Total Complexity'] = sum(c.complexity for c in cc_blocks)
        metrics['Max Complexity'] = max(c.complexity for c in cc_blocks)
    else:
        metrics['Average Complexity'] = 'A'
        metrics['Total Complexity'] = 0
        metrics['Max Complexity'] = 0

    # Maintainability Index
    mi = radon_metrics.mi_visit(code, multi=True)
    metrics['Maintainability Index'] = mi
    metrics['MI Rank'] = radon_metrics.mi_rank(mi)

    # Halstead Metrics
    try:
        hal = radon_metrics.h_visit(code)
        metrics['Halstead Volume'] = hal.total.volume
        metrics['Halstead Difficulty'] = hal.total.difficulty
        metrics['Halstead Effort'] = hal.total.effort
        metrics['Halstead Time'] = hal.total.time
        metrics['Halstead Bugs'] = hal.total.bugs
    except Exception as e:
        metrics['Halstead Error'] = str(e)

    return metrics

def generate_call_graph(code):
    """
    Generates a function call graph from the code.
    Returns a graphviz Digraph object.
    """
    tree = ast.parse(code)
    graph = nx.DiGraph()

    # Find all function definitions
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    for func in functions:
        graph.add_node(func)

    # Find calls within functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            caller = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        callee = child.func.id
                        if callee in functions:
                            graph.add_edge(caller, callee)
                    elif isinstance(child.func, ast.Attribute):
                         # Handle method calls if possible, but keeping it simple for now
                         pass

    # Convert to Graphviz
    dot = Digraph(comment='Call Graph')
    for node in graph.nodes:
        dot.node(node, node)
    for edge in graph.edges:
        dot.edge(edge[0], edge[1])

    return dot
