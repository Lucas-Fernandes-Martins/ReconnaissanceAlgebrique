import React, { useState } from 'react';
import './App.css';
import Header from '../components/Header';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';


function App() {

  const [firstEquation, setFirstEquation] = useState("");
  const [secondEquation, setSecondEquation] = useState("");
  const [result, setResult] = useState(100);

  const data = { 'first_equation': firstEquation, 'second_equation': secondEquation }
  console.log(data)
  const checkSimilarity = async () => {
    try {
      // Make GET request to API endpoint
      const response = await fetch('http://localhost:8000/get_score', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
      })
      // Check if request was successful
      if (response.ok) {
        // Parse response JSON
        const data = await response.json();
        console.log(data);
        console.log(data['result'][0]);
        setResult(data['result'][0])
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
            <input name="first-equation" className="input-field" onChange={(e) => setFirstEquation(e.target.value)}></input>
            <input name="second-equation" className="input-field" onChange={(e) => setSecondEquation(e.target.value)}></input>
            <button type="submit" className="submit-button">Similarity</button>
          </form>
        </div>
        <div className='Layout'>
          <div className='LatexBlock'>
            <BlockMath>{firstEquation}</BlockMath>
            <BlockMath>{secondEquation}</BlockMath>
          </div>
          <div className='CircularProgressBarLayout'>
            <CircularProgressbar value={result} text={`${result}%`} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
