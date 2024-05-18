from sympy.core.basic import Basic
from sympy import simplify, srepr, flatten, Add, Mul, Eq
from latex2sympy2 import latex2sympy, latex2latex

from difflib import SequenceMatcher

import torch
from sklearn.metrics.pairwise import cosine_similarity
from zss import Node, simple_distance
import math

COMMUTATIVE_FUNCTIONS = [Mul, Add]

# Data structure for zss algorithm

class TreeNode:
    def __init__(self, label, position=None):
        if isinstance(label, Basic):
            self.label = label.__class__.__name__
        else:
            self.label = label
        self.children = []
        self.position = position  # Track position for highlighting

    def add_child(self, child):
        self.children.append(child)

    def to_zss_node(self):
        zss_node = Node(self.label)
        for child in self.children:
            zss_node.addkid(child.to_zss_node())
        return zss_node

    def __repr__(self):
        return f'TreeNode({self.label})'


def build_tree(expr, position=None):
    if isinstance(expr, Basic):
        if expr.is_Atom:
            node = TreeNode(str(expr), position)
        else:
            node = TreeNode(expr.func, position)
            if expr.func in COMMUTATIVE_FUNCTIONS:
                args = flatten(expr.args)
                args = sorted(args, key=lambda x: str(x))
            else:
                args = expr.args
            for i, arg in enumerate(args):
                child_node = build_tree(arg, position=i)
                node.add_child(child_node)
    else:
        node = TreeNode(str(expr), position)

    return node


def find_differences(tree1, tree2):
    differences = []

    def recurse(t1, t2, pos1=(), pos2=()):
        if t1.label != t2.label:
            differences.append((pos1, pos2, t1.label, t2.label))
        for c1, c2 in zip(t1.children, t2.children):
            recurse(c1, c2, pos1 + (t1.position,), pos2 + (t2.position,))
        for c1 in t1.children[len(t2.children):]:
            differences.append((pos1 + (c1.position,), None, c1.label, None))
        for c2 in t2.children[len(t1.children):]:
            differences.append((None, pos2 + (c2.position,), None, c2.label))

    recurse(tree1, tree2)
    return differences

def compare_latex_expressions_with_differences(latex_expr1, latex_expr2):
    expr1, expr2 = simplify_latex_expression(latex_expr1), simplify_latex_expression(latex_expr2)
    tree1 = build_tree(expr1)
    tree2 = build_tree(expr2)
    differences = find_differences(tree1, tree2)
    return differences


# Simplify functions

def simplify_latex_expression(latex_expr):
    return latex2sympy(latex2latex(latex_expr))

def simplify_sympy_expression(sympy_expr):
    return simplify(sympy_expr.doit().doit())


# Compare functions

def compare_latex_expressions(latex_expr1, latex_expr2):

    # Convert and simplify expressions
    expr1, expr2 = simplify_latex_expression(latex_expr1), simplify_latex_expression(latex_expr2)
    # Check if the expressions are equal
    equations_are_equal = Eq(expr1, expr2) == True

    return equations_are_equal

def compare_sympy_expressions(sympy_expr1, sympy_expr2):

    # Simplify expressions
    expr1, expr2 = simplify_sympy_expression(sympy_expr1), simplify_sympy_expression(sympy_expr2)
    # Check if the expressions are equal
    equations_are_equal = Eq(expr1, expr2) == True

    return equations_are_equal


# For sequence similarity

def simpy_to_tree(sympy_expr):
    return srepr(simplify_sympy_expression(sympy_expr))

def latex_to_tree(latex_expr):
    return srepr(simplify_latex_expression(latex_expr))

def get_tree_sequence_similarity(tree1, tree2):
    matcher = SequenceMatcher(None, tree1, tree2)
    return matcher.ratio()



# For bert similarity

def get_bert_embeddings(expr, model, tokenizer):
    tokens = tokenizer(expr, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def get_text_similarity(embeddings1, embeddings2):
    return cosine_similarity([embeddings1], [embeddings2])[0][0]

def load_expr(expr):
    symbolic = simplify_latex_expression(expr)
    tree = build_tree(symbolic)
    
    return tree

def get_score(str1, str2):
    tree1 = load_expr(str1)
    tree2 = load_expr(str2)
    return simple_distance(tree1, tree2)

def get_numerical_score(str1, str2):
    tree1 = load_expr(str1)
    tree2 = load_expr(str2)

    distance = simple_distance(tree1, tree2)

    # score = math.exp(distance/)