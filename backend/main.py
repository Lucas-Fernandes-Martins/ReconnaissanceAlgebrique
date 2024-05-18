from urllib.request import Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from engine import get_score, compare_latex_expressions_with_differences, simplify_latex_expression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to match your React app's URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Define your Python function
def process_strings(string1: str, string2: str):
    # Perform some operation on the input strings
    result = f"Processed: {string1.upper()} and {string2.lower()}"
    return result

def parse_input(input_string):
    return input_string.replace("backl", "\\")
 

# Define a route to handle GET requests with two string arguments
@app.get('/api/get_score/')
def get_data(string1: str, string2: str):
    string1 = parse_input(string1)
    string2 = parse_input(string2)
    # Call the function with the provided input strings
    data = get_score(string1, string2)
    # Create JSON response
    response = {'result': [string1, string2, data]}
    # Return JSON response
    return response


@app.post("/get_score")
async def receive_data(data: dict):
    print(data)
    string1 = data['first_equation']
    string2 = data['second_equation']
    # Call the function with the provided input strings
    score = get_score(string1, string2)
    # Create JSON response
    response = {'result': [score]}
    # Return JSON response
    return response

@app.post("/get_differences")
async def get_differences(data: dict):
    string1 = data['first_equation']
    string2 = data['second_equation']
    differences = compare_latex_expressions_with_differences(string1, string2)
    simplified_expr1 = str(simplify_latex_expression(string1))
    simplified_expr2 = str(simplify_latex_expression(string2))
    response = {
        'differences': differences,
        'simplified_expr1': simplified_expr1,
        'simplified_expr2': simplified_expr2,
    }
    return response

