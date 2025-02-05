// กำหนดชุดสีที่จะใช้ในกราฟทั้งหมด ทำให้การแสดงผลมีความสอดคล้องกัน
const chartColors = {
    metrics: {
        accuracy: {
            fill: 'rgba(54, 162, 235, 0.7)',
            border: 'rgba(54, 162, 235, 1)'
        },
        precision: {
            fill: 'rgba(75, 192, 192, 0.7)',
            border: 'rgba(75, 192, 192, 1)'
        },
        recall: {
            fill: 'rgba(153, 102, 255, 0.7)',
            border: 'rgba(153, 102, 255, 1)'
        },
        f1_score: {
            fill: 'rgba(255, 159, 64, 0.7)',
            border: 'rgba(255, 159, 64, 1)'
        }
    },
    models: [
        'rgba(75, 192, 192, 1)',  // สีเขียวมิ้นท์
        'rgba(54, 162, 235, 1)',  // สีฟ้า
        'rgba(153, 102, 255, 1)'  // สีม่วง
    ]
};

// ฟังก์ชันสำหรับสร้างกราฟแท่งเปรียบเทียบเมทริกซ์
function createMetricsChart(data) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    
    // สร้างชุดข้อมูลสำหรับแต่ละเมทริกซ์ โดยใช้สีที่กำหนดไว้
    const datasets = Object.entries(data.metrics).map(([metric, values]) => ({
        label: metric.split('_')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' '),
        data: values,
        backgroundColor: chartColors.metrics[metric].fill,
        borderColor: chartColors.metrics[metric].border,
        borderWidth: 2,
        borderRadius: 5,
        categoryPercentage: 0.8,
        barPercentage: 0.9
    }));

    // สร้างกราฟแท่งด้วย Chart.js พร้อมการตั้งค่าที่เหมาะสม
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.models,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Metrics Comparison',
                    font: { size: 16, weight: 'bold' },
                    padding: 20
                },
                legend: {
                    position: 'top',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${(context.raw * 100).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: value => (value * 100).toFixed(0) + '%',
                        font: { size: 12 }
                    },
                    grid: {
                        color: 'rgba(200, 200, 200, 0.3)'
                    },
                    title: {
                        display: true,
                        text: 'Performance Score',
                        font: { size: 14, weight: 'bold' }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 12 },
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// ฟังก์ชันสร้าง Confusion Matrix ด้วยการแสดงผลแบบตาราง
function createConfusionMatrix(matrix, modelName, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    // สร้างตารางและกำหนดคลาส Bootstrap
    const table = document.createElement('table');
    table.className = 'confusion-matrix-table table table-bordered';
    
    // สร้างส่วนหัวของตาราง
    const header = table.createTHead();
    const headerRow = header.insertRow();
    ['', 'Predicted Negative', 'Predicted Positive'].forEach(label => {
        const th = document.createElement('th');
        th.textContent = label;
        headerRow.appendChild(th);
    });
    
    // สร้างเนื้อหาของตาราง พร้อม heatmap
    const body = table.createTBody();
    [
        ['Actual Negative', matrix[0][0], matrix[0][1]],
        ['Actual Positive', matrix[1][0], matrix[1][1]]
    ].forEach(row => {
        const tr = body.insertRow();
        row.forEach((cell, index) => {
            const td = tr.insertCell();
            td.textContent = cell;
            if (index > 0) {
                const intensity = cell / Math.max(...matrix.flat());
                td.style.backgroundColor = `rgba(54, 162, 235, ${intensity})`;
            }
        });
    });
    
    // เพิ่มหัวข้อและตารางลงในคอนเทนเนอร์
    const title = document.createElement('h6');
    title.textContent = modelName;
    container.appendChild(title);
    container.appendChild(table);
}

// ฟังก์ชันสร้างกราฟ ROC Curves เพื่อเปรียบเทียบประสิทธิภาพของโมเดล
function createROCCurves(data) {
    const ctx = document.getElementById('rocChart').getContext('2d');
    
    // สร้างชุดข้อมูลสำหรับแต่ละโมเดล
    const datasets = data.models.map((model, index) => ({
        label: model,
        data: data.roc_curves.fpr[index].map((fpr, i) => ({
            x: fpr,
            y: data.roc_curves.tpr[index][i]
        })),
        borderColor: chartColors.models[index],
        fill: false
    }));
    
    // เพิ่มเส้น Random Classifier เป็นเส้นอ้างอิง
    datasets.push({
        label: 'Random Classifier',
        data: [{x: 0, y: 0}, {x: 1, y: 1}],
        borderColor: 'rgba(128, 128, 128, 0.5)',
        borderDash: [5, 5],
        fill: false
    });

    // สร้างกราฟเส้น ROC
    new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate',
                        font: { size: 14, weight: 'bold' }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate',
                        font: { size: 14, weight: 'bold' }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'ROC Curves Comparison',
                    font: { size: 16, weight: 'bold' },
                    padding: 20
                }
            }
        }
    });
}

// เริ่มต้นการทำงานเมื่อโหลดหน้าเว็บ
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // ดึงข้อมูลจาก API
        const response = await fetch('/api/model-metrics');
        if (!response.ok) {
            throw new Error('Failed to fetch model metrics');
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        // สร้างการแสดงผลทั้งหมด
        createMetricsChart(data);
        data.models.forEach((model, index) => {
            createConfusionMatrix(
                data.confusion_matrices[index],
                model,
                `confusion-${model.replace(/\s+/g, '')}`
            );
        });
        createROCCurves(data);
        
    } catch (error) {
        console.error('Error creating visualizations:', error);
        // แสดงข้อความแจ้งเตือนเมื่อเกิดข้อผิดพลาด
        const containers = ['metricsChart', 'confusion-matrices', 'rocChart']
            .forEach(id => {
                const container = document.getElementById(id);
                if (container) {
                    container.innerHTML = `
                        <div class="alert alert-danger" role="alert">
                            <h4 class="alert-heading">Error Loading Data</h4>
                            <p>${error.message}</p>
                            <hr>
                            <p class="mb-0">Please try refreshing the page or contact support.</p>
                        </div>
                    `;
                }
            });
    }
});