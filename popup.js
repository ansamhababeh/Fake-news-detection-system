window.onload = function () {
  chrome.storage.local.get(["selectedText"], function(result) {
    if (result.selectedText) {
      fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ article: result.selectedText })
      })
      .then(response => response.json())
      .then(data => {
        document.getElementById("loader").style.display = "none";
        document.getElementById("result").style.display = "block";

        document.getElementById("modalText").innerHTML = `
          <strong>Prediction:</strong> ${data.label}<br>
          <strong>Score:</strong> ${data.prediction.toFixed(2)}
        `;

        let explanationHTML = '<h3>Explanation (Top Influential Words):</h3>';
        explanationHTML += '<table><tr><th>Word</th><th>Impact</th><th>Why it matters</th></tr>';
        data.insights.forEach(item => {
          const className = item.impact >= 0 ? "positive" : "negative";
          explanationHTML += `<tr><td>${item.word}</td><td class="${className}">${item.impact.toFixed(3)}</td><td>${item.reason}</td></tr>`;
        });
        explanationHTML += '</table>';
        explanationHTML += `<p class="summary">${data.human_explanation}</p>`;
        document.getElementById("modalExplanation").innerHTML = explanationHTML;

        setTimeout(() => {
          const canvas = document.getElementById('modalChart');
          canvas.style.height = "300px"; // enforce height
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            console.error("Canvas context not available.");
            return;
          }

          const labels = data.insights.map(i => i.word);
          const values = data.insights.map(i => i.impact);
          const backgroundColors = values.map(val =>
            val >= 0 ? 'rgba(0, 128, 0, 0.6)' : 'rgba(255, 0, 0, 0.6)'
          );

          new Chart(ctx, {
            type: 'bar',
            data: {
              labels: labels,
              datasets: [{
                label: 'Impact per Word',
                data: values,
                backgroundColor: backgroundColors
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                tooltip: { enabled: true }
              },
              scales: {
                y: { title: { display: true, text: 'Impact' } },
                x: { title: { display: true, text: 'Word' } }
              }
            }
          });
        }, 500); // ensure canvas is loaded
      });
    }
  });
};