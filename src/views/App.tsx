import React, { useState, useEffect } from 'react';
import './App.css';
import Header from '../components/Header';
import { BrowserRouter as Router, Route, Routes, Link, useNavigate } from 'react-router-dom';
import ManageQuestions from './ManageQuestions';

interface Question {
  id: number;
  question: string;
  answer: string;
}

function App() {
  const [firstEquation, setFirstEquation] = useState<string>("");
  const [secondEquation, setSecondEquation] = useState<string>("");
  const [result, setResult] = useState<string>("");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [selectedQuestionId, setSelectedQuestionId] = useState<number | null>(null);
  const [studentAnswer, setStudentAnswer] = useState<string>("");

  useEffect(() => {
    fetchQuestions();
  }, []);

  const fetchQuestions = async () => {
    try {
      const response = await fetch('https://deployed-backend-ra.onrender.com/questions');
      const data = await response.json();
      console.log("Fetched questions:", data);  // Debug statement
      setQuestions(data);
    } catch (error) {
      console.log('Error fetching questions:', error);
    }
  };

  const checkSimilarity = async () => {
    if (selectedQuestionId === null) {
      setResult("Please select a question.");
      return;
    }

    const selectedQuestion = questions.find(q => q.id === selectedQuestionId);
    if (!selectedQuestion) {
      setResult("Selected question not found.");
      return;
    }

    const data = {
      'first_equation': selectedQuestion.answer,
      'second_equation': studentAnswer
    };

    try {
      const response = await fetch('https://deployed-backend-ra.onrender.com/get_score', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const data = await response.json();
        setResult("Distance: " + data['result'][0]);
      } else {
        console.log('Error: Unable to process request');
      }
    } catch (error) {
      console.log('Error: Unable to process request');
    }
  };

  const SubmitFunction = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    checkSimilarity();
  };

  const handleQuestionsUpdated = () => {
    fetchQuestions();
  };

  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/manage-questions" element={<ManageQuestions onQuestionsUpdated={handleQuestionsUpdated} />} />
        <Route path="/" element={
          <div className="App">
            <div className="form">
              <form onSubmit={SubmitFunction}>
                <label>
                  Select Question:
                  <select
                    value={selectedQuestionId ?? ""}
                    onChange={(e) => setSelectedQuestionId(Number(e.target.value))}
                  >
                    <option value="" disabled>Select a question</option>
                    {questions.map(question => (
                      <option key={question.id} value={question.id}>
                        {question.question}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedQuestionId !== null && (
                  <>
                    <label>
                      Your Answer:
                      <input
                        name="student-answer"
                        className="input-field"
                        onChange={(e) => setStudentAnswer(e.target.value)}
                      />
                    </label>
                    <button type="submit" className="submit-button">Submit Answer</button>
                  </>
                )}
              </form>

              <div className="result">
                <p>{result}</p>
              </div>

              <div className="navigation">
                <Link to="/manage-questions">Manage Questions</Link>
              </div>
            </div>
          </div>
        } />
      </Routes>
    </Router>
  );
}

export default App;
