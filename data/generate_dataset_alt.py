import sympy as sp
import random
from typing import Dict, List, Union, Tuple

# Define a list of allowed symbols
symbols = [sp.Symbol(chr(i)) for i in range(ord('a'), ord('z') + 1) if chr(i) != "i"]

# Define a list of allowed operations
operations = [sp.Add, sp.Mul, sp.Pow]
functions = [sp.sin, sp.cos, sp.tan, sp.exp, sp.log]

# Define a function to generate a random expression
def generate_expression(depth: int, max_depth: int) -> sp.Expr:
    if depth == max_depth:
        if random.random() < 0.5:
            return random.choice(symbols)
        else:
            return sp.Number(random.uniform(-10, 10)).n(chop=True)

    operation = random.choice(operations+functions)
    if operation in functions:
        expr = operation(generate_expression(depth + 1, max_depth))
    elif operation in operations:
        num_args = 2
        args = [generate_expression(depth + 1, max_depth) for _ in range(num_args)]
        expr = operation(*args)

    return expr

# Define a function to generate equivalent expressions
def generate_equivalent_expression(expr: sp.Expr) -> sp.Expr:
    if isinstance(expr, sp.Symbol):
        return expr

    if isinstance(expr, sp.Number):
        return expr

    op = expr.func
    if op in [sp.Add, sp.Mul]:
        args = list(expr.args)
        random.shuffle(args)
        return op(*args)

    if op == sp.Pow:
        base, exp = expr.args
        if random.random() < 0.5:
            new_base = generate_equivalent_expression(base)
            new_exp = exp
        else:
            new_base = base
            new_exp = generate_equivalent_expression(exp)
        return sp.Pow(new_base, new_exp)

    if op in [sp.sin, sp.cos, sp.tan, sp.exp, sp.log]:
        arg = expr.args[0]
        new_arg = generate_equivalent_expression(arg)
        return op(new_arg)

    raise ValueError(f"Unsupported operation: {op}")

# Define a function to generate non-equivalent expressions
def generate_non_equivalent_expression(expr: sp.Expr) -> sp.Expr:
    if isinstance(expr, sp.Symbol):
        return random.choice([sym for sym in symbols if sym != expr])

    if isinstance(expr, sp.Number):
        return sp.Number(random.uniform(-10, 10)).n(chop=True)
    print(expr)
    op = expr.func
    if op in [sp.Add, sp.Mul]:
        args = list(expr.args)
        random.shuffle(args)
        new_args = [generate_non_equivalent_expression(arg) for arg in args]
        return op(*new_args)

    if op == sp.Pow:
        base, exp = expr.args
        new_base = generate_non_equivalent_expression(base)
        new_exp = generate_non_equivalent_expression(exp)
        return sp.Pow(new_base, new_exp)

    if op in [sp.sin, sp.cos, sp.tan, sp.exp, sp.log]:
        arg = expr.args[0]
        new_arg = generate_non_equivalent_expression(arg)
        return op(new_arg)

    raise ValueError(f"Unsupported operation: {op}")

# Define a function to convert a SymPy expression to the desired format
def expr_to_dict(expr: sp.Expr) -> Dict[str, Union[str, int, float, List]]:
    if isinstance(expr, sp.Symbol):
        return {"id": str(expr), "val": str(expr),"type":"VARIABLE", "children": []}

    if isinstance(expr, sp.Number):
        return {"id": str(expr), "val": str(expr),"type":"NUMERICAL", "children": []}
    op = expr.func
    if not op:
        op = expr.atoms()
    children = [expr_to_dict(arg) for arg in expr.args]
    return {"id": str(expr), "val": str(op), "children": children}

# Define a function to generate a dataset
def generate_dataset(num_samples: int, max_depth: int) -> List[Dict[str, Union[str, int, float, List]]]:
    dataset = []
    for _ in range(num_samples):
        expr = generate_expression(0, max_depth)
        equiv_expr = generate_equivalent_expression(expr)
        non_equiv_expr = generate_non_equivalent_expression(expr)

        expr_dict = expr_to_dict(expr)
        equiv_dict = expr_to_dict(equiv_expr)
        non_equiv_dict = expr_to_dict(non_equiv_expr)

        dataset.append({"id": str(expr), "val": expr_dict, "score": 0})
        dataset.append({"id": str(equiv_expr), "val": equiv_dict, "score": 0})
        dataset.append({"id": str(non_equiv_expr), "val": non_equiv_dict, "score": random.randint(1, 10)})

    return dataset

# Example usage
dataset = generate_dataset(10, 3)
for sample in dataset:
    print(sample)