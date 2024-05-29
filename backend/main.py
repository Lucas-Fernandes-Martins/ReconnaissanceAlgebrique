from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

from engine import get_score, give_feedback, give_feedback_symbol_analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to match your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    id: int
    question: str
    answer: str

questions_db = []

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/questions", response_model=List[Question])
async def get_questions():
    logging.info("Fetching all questions")
    return questions_db

@app.post("/questions", response_model=Question)
async def create_question(question: Question):
    questions_db.append(question)
    logging.info(f"Question created: {question}")
    return question

@app.delete("/questions/{question_id}", response_model=Question)
async def delete_question(question_id: int):
    question = next((q for q in questions_db if q.id == question_id), None)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    questions_db.remove(question)
    logging.info(f"Question deleted: {question}")
    return question

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

@app.post("/get_feedback")
async def receive_data(data: dict):
    print(data)
    string1 = data['first_equation']
    string2 = data['second_equation']
    # Call the function with the provided input strings
    feedback_signs = give_feedback(string1, string2)
    feedback_symbol = give_feedback_symbol_analysis(string1, string2)
    feedback = "Mistakes: \n" + "".join(feedback_signs) + "\n" + "".join(feedback_symbol)
    # Create JSON response
    response = {'result': feedback}
    # Return JSON response
    return response