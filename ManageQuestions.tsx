import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';

interface Question {
  id: number;
  question: string;
  answer: string;
}

interface ManageQuestionsProps {
  onQuestionsUpdated: () => void;
}

function ManageQuestions({ onQuestionsUpdated }: ManageQuestionsProps) {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [newQuestion, setNewQuestion] = useState<string>("");
  const [newAnswer, setNewAnswer] = useState<string>("");
  const navigate = useNavigate();

  useEffect(() => {
    fetchQuestions();
  }, []);

  const fetchQuestions = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/questions');
      const data = await response.json();
      console.log("Fetched questions:", data);  // Debug statement
      setQuestions(data);
    } catch (error) {
      console.log('Error fetching questions:', error);
    }
  };

  const createQuestion = async () => {
    const question: Question = {
      id: questions.length ? questions[questions.length - 1].id + 1 : 1,
      question: newQuestion,
      answer: newAnswer,
    };
    try {
      const response = await fetch('http://127.0.0.1:8000/questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(question),
      });
      const data = await response.json();
      console.log("Created question:", data);  // Debug statement
      setQuestions([...questions, data]);
      setNewQuestion("");
      setNewAnswer("");
      onQuestionsUpdated();
      navigate('/');
    } catch (error) {
      console.log('Error creating question:', error);
    }
  };

  const deleteQuestion = async (id: number) => {
    try {
      await fetch(`http://127.0.0.1:8000/questions/${id}`, {
        method: 'DELETE',
      });
      setQuestions(questions.filter(q => q.id !== id));
      onQuestionsUpdated();
    } catch (error) {
      console.log('Error deleting question:', error);
    }
  };

  return (
    <div className="App">
      <div className="form">
        <h3>Manage Questions</h3>
        <ul>
          {questions.map(question => (
            <li key={question.id}>
              {question.question} - {question.answer}
              <button onClick={() => deleteQuestion(question.id)}>Delete</button>
            </li>
          ))}
        </ul>
        <input
          value={newQuestion}
          onChange={(e) => setNewQuestion(e.target.value)}
          placeholder="New question"
        />
        <input
          value={newAnswer}
          onChange={(e) => setNewAnswer(e.target.value)}
          placeholder="Expected answer"
        />
        <button onClick={createQuestion}>Add Question</button>
      </div>
    </div>
  );
}

export default ManageQuestions;
