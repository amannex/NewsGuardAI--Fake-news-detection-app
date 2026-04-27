document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const newsInput = document.getElementById('news-input');
    const loader = document.getElementById('loader');
    const btnText = document.getElementById('btn-text');
    
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const predLabel = document.getElementById('pred-label');
    const predConfidence = document.getElementById('pred-confidence');
    const highlightedText = document.getElementById('highlighted-text');
    const textualExplanation = document.getElementById('textual-explanation');
    const errorMessage = document.getElementById('error-message');

    // Make sure FastAPI is running locally or hosted on a server
    const API_URL = 'http://127.0.0.1:8000/predict';

    analyzeBtn.addEventListener('click', async () => {
        const text = newsInput.value.trim();
        
        // Reset state
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        
        if (!text) {
            showError("Please enter some text to analyze.");
            return;
        }

        // Show loading state
        setLoading(true);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Failed to analyze text. Ensure backend is running.");
            }

            displayResult(text, data);

        } catch (error) {
            showError("Error: Could not connect to backend server. Make sure the FastAPI server is running on localhost:8000.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            loader.classList.remove('hidden');
            btnText.textContent = "Analyzing...";
            analyzeBtn.disabled = true;
        } else {
            loader.classList.add('hidden');
            btnText.textContent = "Analyze";
            analyzeBtn.disabled = false;
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorContainer.classList.remove('hidden');
    }

    function displayResult(originalText, data) {
        // Set Header info
        predLabel.textContent = data.prediction;
        predLabel.style.color = data.prediction === 'REAL' ? 'var(--real-color)' : 'var(--fake-color)';
        predConfidence.textContent = data.confidence;

        // Render Explanation
        renderHighlights(originalText, data.explanation);
        renderTextualSummary(data.prediction, data.explanation);
        
        resultContainer.classList.remove('hidden');
    }

    function renderHighlights(originalText, explanation) {
        // Convert explanation array to a dictionary mapping word (lowercase) to its data
        const wordData = {};
        let maxWeight = 0;
        explanation.forEach(item => {
            wordData[item.word.toLowerCase()] = item;
            if (Math.abs(item.weight) > maxWeight) {
                maxWeight = Math.abs(item.weight);
            }
        });

        // Split original text into words maintaining punctuation using Regex
        const tokens = originalText.split(/(\b\w+\b)/);
        
        highlightedText.innerHTML = '';

        tokens.forEach(token => {
            const lowerToken = token.toLowerCase();
            const span = document.createElement('span');
            span.textContent = token;

            if (wordData[lowerToken]) {
                const item = wordData[lowerToken];
                span.classList.add('LIME-word');
                
                const normalizedWeight = Math.min(Math.abs(item.weight) / (maxWeight || 1), 1);
                const opacity = Math.max(0.3, normalizedWeight);
                
                if (item.indicator === 'REAL') {
                    span.classList.add('highlight-real');
                    span.style.backgroundColor = `rgba(34, 197, 94, ${opacity})`; 
                } else {
                    span.classList.add('highlight-fake');
                    span.style.backgroundColor = `rgba(239, 68, 68, ${opacity})`;
                }
                
                span.setAttribute('title', `Weight: ${item.weight.toFixed(4)}`);
            } else if (/\w+/.test(token)) {
                // It's a word but not highlighted by LIME
                span.classList.add('highlight-neutral');
            }

            highlightedText.appendChild(span);
        });
    }

    function renderTextualSummary(prediction, explanation) {
        const fakeWords = explanation.filter(e => e.indicator === 'FAKE').sort((a,b) => Math.abs(b.weight) - Math.abs(a.weight)).map(e => e.word);
        const realWords = explanation.filter(e => e.indicator === 'REAL').sort((a,b) => Math.abs(b.weight) - Math.abs(a.weight)).map(e => e.word);
        
        let summary = `<strong>Summary:</strong> The model predicted this article is <span style="color: ${prediction === 'REAL' ? 'var(--real-color)' : 'var(--fake-color)'}; font-weight: bold;">${prediction}</span>. `;
        
        if (prediction === 'FAKE' && fakeWords.length > 0) {
            summary += `The strongest indicators pushing towards this prediction were the words: <em>"${fakeWords.slice(0, 5).join('", "')}"</em>. `;
            if (realWords.length > 0) {
                summary += `Conversely, words like <em>"${realWords.slice(0, 3).join('", "')}"</em> pushed the model towards REAL, but were outweighed.`;
            }
        } else if (prediction === 'REAL' && realWords.length > 0) {
            summary += `The strongest indicators pushing towards this prediction were the words: <em>"${realWords.slice(0, 5).join('", "')}"</em>. `;
            if (fakeWords.length > 0) {
                summary += `Conversely, words like <em>"${fakeWords.slice(0, 3).join('", "')}"</em> pushed the model towards FAKE, but were outweighed.`;
            }
        } else {
            summary += `The model used the overall context to make this decision without strong individual word indicators.`;
        }
        
        textualExplanation.innerHTML = summary;
    }
});
