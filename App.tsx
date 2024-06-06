import React, { useState, useEffect } from 'react';
import './App.css';
import Header from '../components/Header';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import ManageQuestions from './ManageQuestions';

interface Question {
  id: number;
  question: string;
  answer: string;
}

function App() {
  const [result, setResult] = useState(100);
  const [feedback, setFeedback] = useState("");

  const [questions, setQuestions] = useState<Question[]>([]);
  const [selectedQuestionId, setSelectedQuestionId] = useState<number | null>(null);
  const [studentAnswer, setStudentAnswer] = useState<string>("");

  const [submitted, setSubmitted] = useState<boolean>(false);

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

  const checkSimilarity = async () => {
    if (selectedQuestionId === null) {
      setResult(0);
      setFeedback("Please select a question.");
      return;
    }

    const selectedQuestion = questions.find(q => q.id === selectedQuestionId);
    if (!selectedQuestion) {
      setResult(0);
      setFeedback("Selected question not found.");
      return;
    }

    const data = {
      'first_equation': selectedQuestion.answer,
      'second_equation': studentAnswer
    };

    try {
      const response = await fetch('http://127.0.0.1:8000/get_score', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data['result'][0]);
      } else {
        console.log('Error: Unable to process request');
      }
    } catch (error) {
      console.log('Error: Unable to process request');
    }
  };

  const getFeedback = async () => {
    if (selectedQuestionId === null) {
      setFeedback("Please select a question.");
      return;
    }

    const selectedQuestion = questions.find(q => q.id === selectedQuestionId);
    if (!selectedQuestion) {
      setFeedback("Selected question not found.");
      return;
    }

    const data = {
      'first_equation': selectedQuestion.answer,
      'second_equation': studentAnswer
    };

    try {
      const response = await fetch('http://127.0.0.1:8000/get_feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const data = await response.json();
        setFeedback(data['result']);
      } else {
        console.log('Error: Unable to process request');
      }
    } catch (error) {
      console.log('Error: Unable to process request');
    }
  };

  const SubmitFunction = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitted(true);
    checkSimilarity();
    getFeedback();
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
                    className="select-box"
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
                <p>{feedback}</p>
                <div className='Layout'>
                  <div className='LatexBlock'>
                    {submitted && (
                      <BlockMath>{questions.find(q => q.id === selectedQuestionId)?.answer}</BlockMath>
                    )}
                  </div>
                  <div className='CircularProgressBarLayout'>
                    <CircularProgressbar value={result} text={`${result}%`} />
                  </div>
                </div>
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