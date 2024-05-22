from sympy.core.basic import Basic
from sympy import simplify, srepr, flatten, Add, Mul, Eq
from latex2sympy2 import latex2sympy, latex2latex

from difflib import SequenceMatcher

import torch
from sklearn.metrics.pairwise import cosine_similarity
from zss import Node, simple_distance

COMMUTATIVE_FUNCTIONS = [Mul, Add]


# Data structure for zss algorithm

class TreeNode:

    def __init__(self, label):
        if isinstance(label, Basic):
            self.label = label._class.name_
        else:
            self.label = label
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def build_tree(expr, parent=None):

    if isinstance(expr, Basic):
        if expr.is_Atom:
            node = TreeNode(str(expr))
        else:
            node = TreeNode(expr.func)
            if expr.func in COMMUTATIVE_FUNCTIONS:
                args = flatten(expr.args)
                args = sorted(args, key=lambda x: str(x))
            for arg in expr.args:
                child_node = build_tree(arg, node)
                node.add_child(child_node)
    else:
        node = TreeNode(str(expr))

    return node



# For our database

def parse_database_tree(expression):

    children = []
    
    if 'children' not in expression:
        return Node(expression['val'])
    for i in range(len(expression['children'])):
        children.append(parse_database_tree(expression['children'][i]))
    root = Node(expression['val'], children=children)
    
    return root



# Simplify functions
def simplify_latex_expression(latex_expr):
    try:
        result = latex2sympy(latex2latex(latex_expr))
    except:
        try:
            result = simplify(latex2sympy(latex_expr).doit())
        except:
            try:
                result = simplify(latex2sympy(latex_expr))
            except:
                result = latex2sympy(latex_expr)
    return result
     

def simplify_sympy_expression(sympy_expr):
    try:
        result = simplify(sympy_expr.doit().doit())
    except:
        try:
           result = simplify(sympy_expr.doit())
        except:
            try:
                result = simplify(sympy_expr)
            except:
                result = sympy_expr
    return result


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

# def give_feedback(answer, expected):
#     tree1 = load_expr(answer)
#     tree2 = load_expr(expected)
#     frontier1 = [tree1]
#     frontier2 = [tree2]
#     feedback = []
#     #Breath-first traversal
#     while len(frontier1) > 0 and len(frontier2) > 0:
#         print("Here!")
#         tree1 = frontier1.pop(0)
#         tree2 = frontier2.pop(0)

#         if len(tree1.children) == 0:
#             feedback.append("You forgot terms!")

#         if len(tree2.children) == 0:
#             feedback.append("You have extra terms!")

#         for child in tree1.children:
#             frontier1.append(child)
        
#         for child in tree2.children:
#             frontier2.append(child)

#         if tree1.label == Mul and type(tree2.label) is not str and tree2.label.is_symbol:
#             if tree2.children[0].label == '-1':
#                 feedback.append("You got one sign wrong!")

#         if tree2.label == Mul and tree1.label.is_symbol:
#             if tree1.children[0].label == '-1':
#                 feedback.append("You got one sign wrong!")
            

#     return feedback


def detect_number_terms_error(answer, expected, verbose=False):

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    tree1 = load_expr(answer)
    tree2 = load_expr(expected)

    #Traverse first tree

    frontier = [tree1]
    n_terms_1 = 0
    current = None
    while(len(frontier) > 0):
        current = frontier.pop(0)

        if type(current.label) is str and not is_number(current.label):
            print(current.label)
            n_terms_1 += 1
        frontier.extend(current.children)

    print("=================")

    frontier = [tree2]
    n_terms_2 = 0
    current = None

    while(len(frontier) > 0):
        current = frontier.pop(0)

        if type(current.label) is str and not is_number(current.label):
            print(current.label)
            n_terms_2 += 1
        frontier.extend(current.children)

    if verbose:
        print(f"------Terms expected: {n_terms_1}")
        print(f"------Terms received: {n_terms_2}")

    return n_terms_1 == n_terms_2

def signal_test(answer, expected):
    def has_sign_error(ans, exp):
        # Placeholder logic for sign error checking
        return False

    if has_sign_error(answer, expected):
        return "There's a sign error in your answer!"
    return ""

def number_terms_test(answer, expected):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def count_terms(tree):
        frontier = [tree]
        n_terms = 0
        while frontier:
            current = frontier.pop(0)
            if isinstance(current.label, str) and not is_number(current.label):
                n_terms += 1
            frontier.extend(current.children)
        return n_terms

    tree1 = load_expr(expected)
    tree2 = load_expr(answer)

    n_terms_1 = count_terms(tree1)
    n_terms_2 = count_terms(tree2)

    if n_terms_1 > n_terms_2:
        return "You forgot terms in your answer!"
    elif n_terms_1 < n_terms_2:
        return "You added wrong terms in your answer!"
    return ""

def not_simplified_test(answer, expected):
    tree1 = load_expr(expected)
    tree2 = load_expr(answer)

    def count_nodes(tree):
        counter = 0
        frontier = [tree]
        while frontier:
            current = frontier.pop(0)
            counter += len(current.children)
            frontier.extend(current.children)
        return counter

    counter1 = count_nodes(tree1)
    counter2 = count_nodes(tree2)

    if (counter2 - counter1) / counter1 > 0.5:
        return "You forgot to simplify!"
    return ""

# Example usage (make sure load_expr and build_tree are defined):
# answer = ...
# expected = ...
# feedback = give_feedback(answer, expected)
# print(feedback)import sympy as sp

import sympy as sp

def identify_sign_error(expr1, expr2):
    # Convert both expressions to expanded form
    expanded1 = sp.expand(expr1)
    expanded2 = sp.expand(expr2)
    
    # Break down both expressions into their respective terms
    terms1 = expanded1.as_ordered_terms()
    terms2 = expanded2.as_ordered_terms()

    # Create a dictionary of terms to their coefficients for each expression
    terms_dict1 = {term.as_coeff_Mul()[1]: term.as_coeff_Mul()[0] for term in terms1}
    terms_dict2 = {term.as_coeff_Mul()[1]: term.as_coeff_Mul()[0] for term in terms2}

    # Check for sign errors by comparing coefficients of the same terms
    for term in terms_dict1:
        if term in terms_dict2 and terms_dict1[term] + terms_dict2[term] == 0:
            # If terms are present in both expressions and their coefficients sum to zero
            return True

    return False

def give_feedback(answer, expected):
    feedback_list = []

    # Parse the input strings as SymPy expressions
    expr1 = sp.sympify(answer)
    expr2 = sp.sympify(expected)

    # Check for sign errors
    if identify_sign_error(expr1, expr2):
        feedback_list.append("There's a sign error in your answer!")

    return feedback_list

def extract_symbols(tree):
    symbols = set()
    frontier = [tree]
    while frontier:
        current = frontier.pop(0)
        if current is None:
            continue
        # Check if the current node's label is a symbol (not an operator or number)
        if isinstance(current.label, str) and not current.label.isnumeric():
            symbols.add(current.label)
        # Add the children of the current node to the frontier
        frontier.extend(current.children)
    return symbols

def give_feedback_symbol_analysis(answer, expected):
    feedback_list = []

    tree1 = load_expr(answer)
    tree2 = load_expr(expected)

    # Extract symbols from answer and expected expressions
    symbols_answer = set(extract_symbols(tree1))
    symbols_expected = set(extract_symbols(tree2))

    # Check if symbols in answer match symbols in expected
    if symbols_answer != symbols_expected:
        missing_symbols = symbols_expected - symbols_answer
        extra_symbols = symbols_answer - symbols_expected
        feedback = "Wrong symbols in your answer!"
        if missing_symbols:
            feedback += f" Missing symbols: {missing_symbols}"
        if extra_symbols:
            feedback += f" Extra symbols: {extra_symbols}"
        feedback_list.append(feedback)

    return feedback_list


if __name__ == '__main__':
    # Example use cases
    print(give_feedback('x+y+z', 'x+y-z'))
    print(give_feedback('x+y', 'x-z'))

    print(give_feedback('x+y+z', 'x+y+z'))
    print(give_feedback('x+y', 'x-y'))


    print("======================")
    print(give_feedback_symbol_analysis('x+y+z', 'x+y'))
