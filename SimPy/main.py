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

from abc import ABC, abstractmethod

class Test(ABC):

    def __init__(self):
        self.feedback = []
        pass

    @abstractmethod
    def test():
        pass

    @abstractmethod
    def feedback(expected, answer):
        pass

class SignalError(Test):

    def __init__():
        super()
        pass

    def test():
        raise NotImplementedError('Not implemented!')
    
    def feedback(self, expected, answer):
        
        error_sign = self.test(expected, answer)

        if error_sign:
            self.feedback.append("There's a sign error in your answer!")
        
        return "".joins(self.feedback)

class NumberTermsError():

    def __init__():
        super()
        pass

    def test(answer, expected, verbose=False):
        
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

        #Traverse second tree
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

        return n_terms_1 - n_terms_2
        
    def feedback(self, expected, answer):
        
        n_terms_error = self.test(expected, answer)

        if n_terms_error > 0:
            self.feedback.append("You forgot terms in your answer!")
        elif n_terms_error < 0:
            self.feedback.append("You added wrong terms in your answer!")
        
        return "".joins(self.feedback)
    


def give_feedback(answer, expected):
    # Assuming you have a function called load_expr that correctly loads expressions into tree structures
    #If return True then there's an error
    tests = [SignalError, NumberTermsError]
    
    feedback = []
    for test in tests:
        test_feedback = test.feedback(answer, expected)
        if test_feedback != "":
            feedback.append(test_feedback)

    return feedback
