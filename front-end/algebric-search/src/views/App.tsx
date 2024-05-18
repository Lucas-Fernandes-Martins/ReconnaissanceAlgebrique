import React, { useState } from 'react';
import './App.css';
import Header from '../components/Header';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

function App() {
  const [firstEquation, setFirstEquation] = useState("");
  const [secondEquation, setSecondEquation] = useState("");
  const [result, setResult] = useState("");
  const [simplifiedExpr1, setSimplifiedExpr1] = useState("");
  const [simplifiedExpr2, setSimplifiedExpr2] = useState("");
  const [highlightedExpr1, setHighlightedExpr1] = useState("");
  const [highlightedExpr2, setHighlightedExpr2] = useState("");

  const data = {'first_equation': firstEquation, 'second_equation': secondEquation};

  const highlightDifferences = (expr1, expr2, differences) => {
    let highlightedExpr1 = expr1;
    let highlightedExpr2 = expr2;

    differences.forEach(([pos1, pos2, label1, label2]) => {
      if (pos1 !== null) {
        highlightedExpr1 = highlightedExpr1.replace(label1, `<span class="highlight">${label1}</span>`);
      }
      if (pos2 !== null) {
        highlightedExpr2 = highlightedExpr2.replace(label2, `<span class="highlight">${label2}</span>`);
      }
    });

    setHighlightedExpr1(highlightedExpr1);
    setHighlightedExpr2(highlightedExpr2);
  };

  const checkSimilarity = async () => {
    try {
      const response = await fetch('https://deployed-backend-ra.onrender.com/get_differences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (response.ok) {
        const data = await response.json();
        const differences = data['differences'];
        const simplifiedExpr1 = data['simplified_expr1'];
        const simplifiedExpr2 = data['simplified_expr2'];
        setSimplifiedExpr1(simplifiedExpr1);
        setSimplifiedExpr2(simplifiedExpr2);
        highlightDifferences(simplifiedExpr1, simplifiedExpr2, differences);
      } else {
        console.log('Error: Unable to process request');
      }
    } catch (error) {
      console.log('Error: Unable to process request');
    }
  };

  const SubmitFunction = (event: { preventDefault: () => void; }) => {
    event.preventDefault();
    checkSimilarity();
  };

  return (
    <div>
      <Header />
      <div className="App">
        <div className="form">
          <form onSubmit={SubmitFunction}>
            <input name="first-equation" className="input-field" onChange={(e) => setFirstEquation(e.target.value)} />
            <input name="second-equation" className="input-field" onChange={(e) => setSecondEquation(e.target.value)} />
            <button type="submit" className="submit-button">Similarity</button>
          </form>

          <div className="result">
            <p>{result}</p>
          </div>
          
          <div className="expressions">
            <h3>Original Expressions</h3>
            <BlockMath>{firstEquation}</BlockMath>
            <BlockMath>{secondEquation}</BlockMath>

            <h3>Simplified Expressions</h3>
            <BlockMath>{simplifiedExpr1}</BlockMath>
            <BlockMath>{simplifiedExpr2}</BlockMath>
          </div>

          <div className="highlighted-expressions">
            <h3>Highlighted Differences</h3>
            <div dangerouslySetInnerHTML={{ __html: highlightedExpr1 }}></div>
            <div dangerouslySetInnerHTML={{ __html: highlightedExpr2 }}></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
