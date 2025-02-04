// File: ml_comparison_web/static/js/comparison.js

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Performance Metrics Chart
    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    const metricsChart = new Chart(metricsCtx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            datasets: [
                {
                    label: 'Model 1',
                    data: modelData[0],
                    backgroundColor: 'rgba(75, 192, 192, 0.6)'
                },
                {
                    label: 'Model 2',
                    data: modelData[1],
                    backgroundColor: 'rgba(54, 162, 235, 0.6)'
                },
                {
                    label: 'Model 3',
                    data: modelData[2],
                    backgroundColor: 'rgba(153, 102, 255, 0.6)'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison'
                }
            }
        }
    });

    // Initialize ROC Curve Chart
    const rocCtx = document.getElementById('rocChart').getContext('2d');
    const rocChart = new Chart(rocCtx, {
        type: 'line',
        data: {
            labels: rocData.fpr,
            datasets: [
                {
                    label: 'Model 1 ROC',
                    data: rocData.model1,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false
                },
                {
                    label: 'Model 2 ROC',
                    data: rocData.model2,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    fill: false
                },
                {
                    label: 'Model 3 ROC',
                    data: rocData.model3,
                    borderColor: 'rgba(153, 102, 255, 1)',
                    fill: false
                },
                {
                    label: 'Random Classifier',
                    data: rocData.random,
                    borderColor: 'rgba(128, 128, 128, 0.5)',
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'ROC Curves Comparison'
                }
            }
        }
    });

    // Function to update confusion matrices
    function updateConfusionMatrices(confusionData) {
        const models = ['Model 1', 'Model 2', 'Model 3'];
        models.forEach((model, index) => {
            const container = document.getElementById(`confusion-${model.replace(' ', '')}`);
            const matrix = confusionData[index];
            
            // Create confusion matrix visualization
            const table = document.createElement('table');
            table.className = 'confusion-matrix-table';
            
            // Add matrix values
            for (let i = 0; i < 2; i++) {
                const row = table.insertRow();
                for (let j = 0; j < 2; j++) {
                    const cell = row.insertCell();
                    cell.textContent = matrix[i][j];
                    cell.style.backgroundColor = `rgba(54, 162, 235, ${matrix[i][j] / Math.max(...matrix.flat())})`;
                }
            }
            
            container.innerHTML = `<h6>${model} Confusion Matrix</h6>`;
            container.appendChild(table);
        });
    }

    // Event listeners for tab changes
    document.querySelectorAll('a[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function (e) {
            if (e.target.id === 'confusion-tab') {
                // Update confusion matrices when tab is shown
                updateConfusionMatrices(confusionMatrixData);
            }
        });
    });
});