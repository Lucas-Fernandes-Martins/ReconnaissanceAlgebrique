import React, { useState } from 'react';
import './App.css';
import Header from '../components/Header';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

function App() {
  const [firstEquation, setFirstEquation] = useState<string>("");
  const [secondEquation, setSecondEquation] = useState<string>("");
  const [simplifiedExpr1, setSimplifiedExpr1] = useState<string>("");
  const [simplifiedExpr2, setSimplifiedExpr2] = useState<string>("");
  const [highlightedExpr1, setHighlightedExpr1] = useState<string>("");
  const [highlightedExpr2, setHighlightedExpr2] = useState<string>("");

  const data = { 'first_equation': firstEquation, 'second_equation': secondEquation };

  const highlightDifferences = (expr1: string, expr2: string, differences: any[]) => {
    let highlightedExpr1 = expr1;
    let highlightedExpr2 = expr2;

    differences.forEach(([pos1, pos2, label1, label2]) => {
      if (pos1 !== null) {
        const regex = new RegExp(`\\b${label1}\\b`, 'g');
        highlightedExpr1 = highlightedExpr1.replace(regex, `\\textcolor{red}{${label1}}`);
      }
      if (pos2 !== null) {
        const regex = new RegExp(`\\b${label2}\\b`, 'g');
        highlightedExpr2 = highlightedExpr2.replace(regex, `\\textcolor{red}{${label2}}`);
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

  const SubmitFunction = (event: React.FormEvent<HTMLFormElement>) => {
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
            <BlockMath math={highlightedExpr1} />
            <BlockMath math={highlightedExpr2} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
