import random
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import decimal
from sympy import simplify, expand,  symbols, sin, cos, tan, exp, log, Add, Mul, Pow
import json
import sympy as sp
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class Node:

    def __init__(self, node_type, value=None, children=None, subtype=None):

        self.type = node_type

        self.subtype = subtype

        self.value = value

        self.children = children if children is not None else []



def drange(x, y, jump):

    """Generate a range of decimals."""

    while x < y:

        yield float(x)

        x += decimal.Decimal(jump)



OPERATIONS = ["ADD", "MUL", "FUNC", "POW"]

FUNCTIONS = ["SIN", "COS", "TAN", "EXP", "LOG"]

VARIABLE_ALPHABET = [chr(x) for x in range(ord('a'), ord('z')+1) if chr(x) not in ["e", "i"]]

CLASSIC_CONSTANTS = ["PI", "I", "g","e", "zoo"]

literal_atomics = list(drange(-10, 10, "0.5"))



def create_literal():

    """Create a literal atomic node."""

    return Node("LITERAL", value=random.choice(literal_atomics))



def create_variable():

    """Create a variable atomic node."""

    return Node("VARIABLE", value=random.choice(VARIABLE_ALPHABET))



def create_function_node(argument: Node):

    """Create a function node with a given argument."""

    return Node("FUNC", subtype=random.choice(FUNCTIONS), children=[argument])



def create_operation_node(left: Node, right:Node):

    """Create an operation node with two children."""

    operation = random.choice(["ADD", "MUL"])

    return Node("OPERATION", subtype=operation, children=[left, right])



def create_pow_node(base:Node, exponent: Node):

    """Create a power node."""

    return Node("POW", children=[base, exponent])



def generate_expression(depth: int) -> Node:

    """

    Generate an expression tree with a given depth.

    At each level, randomly decide between creating a function node, operation node, or atomic node.

    """

    if depth == 1:

        return random.choice([create_literal, create_variable])()

    else:

        node_type = random.choice(["FUNC", "OPERATION", "POW"])

        if node_type == "FUNC":

            return create_function_node(generate_expression(depth-1))

        elif node_type == "OPERATION":

            return create_operation_node(generate_expression(depth-1), generate_expression(depth-1))

        elif node_type == "POW":

            return create_pow_node(generate_expression(depth-1), create_literal())





def node_to_sympy(node):

    """

    Converts a Node structure into a Sympy expression.

    """
    try:


        if node.type == "LITERAL":
                        
            return node.value
        if node.type == "CONSTANT_LITERAL":
            const_lit_map = {
                "PI": sp.pi,
                "E": sp.E,
                "I": sp.I,
                "zoo": sp.zoo
            }
            return const_lit_map[node.value]

        elif node.type == "VARIABLE":

            return symbols(node.value)

        elif node.type == "FUNC":

            func_map = {

                "SIN": sin,

                "COS": cos,

                "TAN": tan,

                "EXP": exp,

                "LOG": log

            }

            return func_map[node.subtype](node_to_sympy(node.children[0]))

        elif node.type == "OPERATION":

            if node.subtype == "ADD":

                return Add(node_to_sympy(node.children[0]), node_to_sympy(node.children[1]))

            elif node.subtype == "MUL":

                return Mul(node_to_sympy(node.children[0]), node_to_sympy(node.children[1]))

        elif node.type == "POW":

            return Pow(node_to_sympy(node.children[0]), node_to_sympy(node.children[1]))

        else:

            raise ValueError("Unknown node type")
    except:
        print("[node_to_sympy]-> Failed at node: \n", serialize_node_to_json(node))



def sympy_to_node(expr:sp.Expr) -> Node:

    """

    Converts a Sympy expression into a Node structure.

    """
    try: 
        if expr.is_Symbol:

            return Node("VARIABLE", value=str(expr))

        
        elif expr.is_Number and expr.is_imaginary:
            
            return Node("LITERAL", value=complex(expr))
        
        elif expr.is_Number:

            return Node("LITERAL", value=float(expr))

        elif expr.is_Function:

            func_name = type(expr).__name__.upper()

            return Node("FUNC", subtype=func_name, children=[sympy_to_node(arg) for arg in expr.args])

        elif expr.is_Add or expr.is_Mul:

            op_type = "ADD" if expr.is_Add else "MUL"

            children = [sympy_to_node(arg) for arg in expr.args]

            return Node("OPERATION", subtype=op_type, children=children)

        elif expr.is_Pow:

            return Node("POW", children=[sympy_to_node(arg) for arg in expr.args])
        
        elif expr == sp.pi:
            return Node("CONSTANT_LITERAL", value="PI")
        elif expr == sp.E:
            return Node("CONSTANT_LITERAL", value="E")
        elif expr == sp.I:
            return Node("CONSTANT_LITERAL", value="I")
        elif expr == sp.zoo:
            return Node("CONSTANT_LITERAL", value="zoo")

        else:

            raise ValueError("Unsupported Sympy expression type")
    
    except:
        print("[sympy_to_node]-> Failed at expression: \n", expr, "\n")



def simplify_expression(node: Node) -> Node :
    """
    Simplifies a mathematical expression represented as a Node,
    and returns the simplified expression as a Node.
    """
    #try:
    sympy_expr = node_to_sympy(node)
    simplified_sympy_expr = simplify(sympy_expr)
    return sympy_to_node(simplified_sympy_expr)
    # except:
    #     print("Failed at node: \n", serialize_node_to_json(node))

def expand_expression(node:Node) -> Node:
    """
    Expands a mathematical expression represented as a Node,
    and returns the expanded expression as a Node.
    """
    sympy_expr = node_to_sympy(node)
    expanded_sympy_expr = expand(sympy_expr)
    return sympy_to_node(expanded_sympy_expr)

def node_to_dict(node: Node):
    # Convert a Node object to a dictionary for JSON serialization
    
    node_dict = {
        "type": node.type,
        "value": node.value,
        "subtype": node.subtype,
        "children": [node_to_dict(child) for child in node.children] if node.children else []
    }
    return node_dict

def dict_to_node(expr_dict: dict) -> Node:
    """Convert a dictionary object into a Node"""
    node = Node(expr_dict["type"],expr_dict["value"],[dict_to_node(child) for child in expr_dict["children"]] if expr_dict["children"] else [] ,expr_dict["subtype"])
    return node

def serialize_node_to_json(node):
    """
    Serializes a Node structure into a JSON string.
    """

    return json.dumps(node_to_dict(node), indent=4)


def flip_literal_sign(node):
    """
    Flips the sign of all literals in the expression.
    """
    if node.type == "LITERAL":
        node.value = -node.value
    else:
        for child in node.children:
            flip_literal_sign(child)
    return node

def change_constant_value(node):
    """
    Changes the value of a specific constant in the expression.
    """
    if node.type == "LITERAL":
        node.value = random.choice(literal_atomics)
    else:
        for child in node.children:
            change_constant_value(child)
    return node

def change_variable(node):
    """
    Changes a variable (e.g., 'x' to 'y') in the expression.
    """
    if node.type == "VARIABLE":
        node.value = random.choice(VARIABLE_ALPHABET)
    else:
        for child in node.children:
            change_variable(child)
    return node

def swap_cos_sin(node):
    """
    Changes cosine to sine or sine to cosine in the expression.
    """
    if node.type == "FUNC" and node.subtype in ["COS", "SIN"]:
        node.subtype = "SIN" if node.subtype == "COS" else "COS"
    else:
        for child in node.children:
            swap_cos_sin(child)
    return node

equivalent_transformations = [simplify_expression, expand_expression]
inequivalent_transformations = [flip_literal_sign, change_constant_value, change_variable, swap_cos_sin]
NUM_INEQUIVALENT_TRANSFORMATIONS = 3

def generate_expression_pairs(dataset_size):
    """
    Generates pairs of expressions by first applying equivalent transformations one by one,
    and then applying two out of the non-equivalent transformations three times to generate
    three dissimilar expression pairs.
    """
    
    def dataset_line(expr_true, expr_false, expr_anchor):
        return {"expr_true": expr_true, "expr_false": expr_false, "expr_anchor": expr_anchor}
    
    pairs = []
    for _ in range(dataset_size):
        expression_depth = random.choice([random.choice(list(range(2,4))),random.choice(list(range(2,5))), random.choice(list(range(2,6)))])
        base_expression = generate_expression(expression_depth)
        # print("================================================================================", "depth = ", expression_depth, "=======")
        # print(node_to_dict(base_expression))
        # print("=====================================================================================================")
        # Apply equivalent transformations
        simplified = simplify_expression(base_expression) # Can brick and take waaay too long

            
        expanded = expand_expression(base_expression)
        # Serialize and add to pairs
        # Apply non-equivalent transformations
        for _ in range(NUM_INEQUIVALENT_TRANSFORMATIONS):
            t1 = random.choice(inequivalent_transformations)
            t2 = random.choice(inequivalent_transformations)
            transformed_expression = t2(t1(base_expression))
            pairs.append(dataset_line(node_to_dict(simplified), node_to_dict(transformed_expression), node_to_dict(base_expression)))  # Non-equivalent
            pairs.append(dataset_line(node_to_dict(expanded), node_to_dict(transformed_expression), node_to_dict(base_expression)))  # Non-equivalent
            
        # Clear cut
        pairs.append(dataset_line(node_to_dict(base_expression), node_to_dict(generate_expression(expression_depth)), node_to_dict(simplified)) ) # Non-equivalent
        pairs.append(dataset_line(node_to_dict(simplified), node_to_dict(generate_expression(expression_depth)), node_to_dict(expanded)) ) # Non-equivalent

    return pairs


def generate_partial_dataset(size):
    """Wrapper function to generate a portion of the dataset."""
    return generate_expression_pairs(size)

# def parallel_dataset_generation(total_size, chunk_size=10, timeout=10, max_retries_per_chunk=2):
#     """Generates the dataset in parallel, handling timeouts and errors."""
#     dataset = []
#     futures = []
#     retry_limit = max_retries_per_chunk  # Maximum number of retries per task
#     retries = {}
#     with ProcessPoolExecutor() as executor:
#         # Submit tasks to generate chunks of the dataset in parallel
#         for _ in range(0, total_size, chunk_size):
#             futures.append(executor.submit(generate_partial_dataset, chunk_size))

#         # Monitor futures and collect results
#         for future in tqdm(as_completed(futures), total=len(futures)):
#             try:
#                 result = future.result(timeout=timeout)
#                 dataset.extend(result)
#             except TimeoutError:
#                 if retries.get(future, 0) < retry_limit:
#                     logging.info("Task exceeded the timeout limit, rescheduling...")
#                     new_future = executor.submit(generate_partial_dataset, chunk_size)
#                     futures.append(new_future)
#                     retries[new_future] = retries.get(future, 0) + 1
#                 else:
#                     logging.warning("Task reached maximum retry limit.")
#             except Exception as e:
#                 logging.error(f"Task failed with exception: {e}.")

#     return dataset

import zmq


def worker(task_id, chunk_size, address):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(address)
    
    try:
        # Simulate dataset generation (replace with actual function)
        result = generate_partial_dataset(chunk_size)
        socket.send_pyobj((task_id, result, None))  # Send result back to the main process
    except Exception as e:
        socket.send_pyobj((task_id, None, e))  # Send exception info back to the main process


import multiprocessing
import time
import zmq
import logging
from tqdm import tqdm


def parallel_dataset_generation(total_size, chunk_size=10, timeout=60*5, max_retries_per_chunk=2):
    dataset = []
    task_id = 0
    address = "tcp://127.0.0.1:5555"
    retries = {}
    
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(address)
    
    processes = []
    total_chunks = (total_size + chunk_size - 1) // chunk_size  # Ceiling division
    with tqdm(total=total_chunks) as pbar:
        for _ in range(total_chunks):
            retries[task_id] = 0
            process = multiprocessing.Process(target=worker, args=(task_id, chunk_size, address))
            process.start()
            processes.append((task_id, process, time.time()))
            task_id += 1
        
        while processes:
            new_processes = []
            for task_id, process, start_time in processes:
                if process.is_alive():
                    if time.time() - start_time > timeout:
                        process.terminate()
                        logging.info(f"Task {task_id} exceeded the timeout limit, rescheduling...")
                        if retries[task_id] < max_retries_per_chunk:
                            retries[task_id] += 1
                            new_process = multiprocessing.Process(target=worker, args=(task_id, chunk_size, address))
                            new_process.start()
                            new_processes.append((task_id, new_process, time.time()))
                        else:
                            logging.warning(f"Task {task_id} reached maximum retry limit.")
                            pbar.update(1)
                    else:
                        new_processes.append((task_id, process, start_time))
                else:
                    pbar.update(1)
            
            while socket.poll(2000):  # Check for messages with a 2-second timeout
                try:
                    task_id, result, error = socket.recv_pyobj(zmq.NOBLOCK)
                    if error:
                        logging.error(f"Task {task_id} failed with exception: {error}")
                    else:
                        dataset.extend(result)
                except zmq.Again:
                    break

            processes = new_processes
            time.sleep(2)

    socket.close()
    context.term()
    return dataset


def sequential_dataset_generation(total_size, chunk_size):
    dataset = []
    i = 0
    with tqdm(total=total_size) as pbar:
        while i < total_size:
            try:
                dataset.extend(generate_partial_dataset(chunk_size))
                i+= chunk_size
                pbar.update(chunk_size)
                pbar.set_description("Processing...")
            except Exception as e:
                pbar.set_description("Reprocessing chunk #" + str(i//chunk_size)+ " ...")
                continue
    return dataset
            
if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser(description='Generate dataset')
    # parser.add_argument('--size', type=int, default=16000, help='Desired dataset size')
    # parser.add_argument('--chunk_size', type=int, default=50, help='Chunk size')
    # parser.add_argument('--output', type=str, required=True, help='Output file path')
    # args = parser.parse_args()

    # generated_dataset = parallel_dataset_generation(args.size, chunk_size=args.chunk_size)
    # with open(args.output, "w") as f:
    #     json.dump(generated_dataset, f, indent=4)
    with open("math_datagen_triplet_40k.json", "r") as f:
        data = json.load(f)
        mathexpr = data[10]["expr_true"]
        print(node_to_sympy(dict_to_node(mathexpr)))


# import queue
# import time
# import signal
# from concurrent.futures import ThreadPoolExecutor

# def generate_partial_dataset(size):
#     """Wrapper function to generate a portion of the dataset."""
#     return generate_expression_pairs(size)

# def parallel_dataset_generation(total_size, chunk_size=10, timeout=30, max_workers=4, max_retries=3):
#     """Generates the dataset in parallel, handling timeouts and errors."""
#     dataset = []
#     retry_queue = queue.Queue()
#     completed_tasks = 0
#     pending_tasks = 0
#     retries = {}
#     should_terminate = False

#     pbar = tqdm(total=total_size, desc="Generating dataset", unit="items")

#     def signal_handler(sig, frame):
#         nonlocal should_terminate
#         should_terminate = True
#         print("Received interrupt signal. Terminating workers...")

#     signal.signal(signal.SIGINT, signal_handler)

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit initial tasks
#         for _ in range(0, total_size, chunk_size):
#             future = executor.submit(generate_partial_dataset, chunk_size)
#             pending_tasks += 1
#             retries[future] = 0

#         while completed_tasks < total_size and not should_terminate:
#             try:
#                 future = retry_queue.get(timeout=timeout)
#             except queue.Empty:
#                 # No completed tasks, check for timeouts
#                 for future, retries_count in retries.items():
#                     if future.done():
#                         completed_tasks += chunk_size
#                         pbar.update(chunk_size)
#                         try:
#                             result = future.result()
#                             dataset.extend(result)
#                         except Exception as e:
#                             logging.error(f"Task failed with exception: {e}")
#                     elif retries_count < max_retries:
#                         # Task timed out, retry
#                         new_future = executor.submit(generate_partial_dataset, chunk_size)
#                         pending_tasks += 1
#                         retries[new_future] = retries_count + 1
#                         del retries[future]
#             else:
#                 # Task completed
#                 pending_tasks -= 1
#                 try:
#                     result = future.result()
#                     dataset.extend(result)
#                     completed_tasks += chunk_size
#                     pbar.update(chunk_size)
#                 except Exception as e:
#                     logging.error(f"Task failed with exception: {e}")
#                 del retries[future]

#     pbar.close()
#     if should_terminate:
#         print("Dataset generation terminated.")
#     return dataset

# if __name__ == "__main__":
#     desired_dataset_size = 10000
#     generated_dataset = parallel_dataset_generation(desired_dataset_size, chunk_size=200)
#     with open("math_datagen_triplet.json", "w") as f:
#         json.dump(generated_dataset, f, indent=4)