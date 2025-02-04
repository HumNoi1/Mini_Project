"""# เปิด tunnel ของ ngrok"""

from pyngrok import ngrok

# ตรวจสอบว่ามี tunnel ที่เปิดอยู่หรือไม่
existing_tunnels = ngrok.get_tunnels()
if not existing_tunnels:
    public_url = ngrok.connect(5000)  # เปิด tunnel ใหม่ถ้าไม่มี
    print(f"Public URL: {public_url}")
else:
    print("Existing Tunnel:", existing_tunnels[0].public_url)  # ใช้ tunnel ที่มีอยู่

"""# วิเคราะห์ผลความรู้สึก และ Report"""

from flask import Flask, request, render_template_string, send_file
from flask_ngrok import run_with_ngrok
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pythainlp.tokenize import word_tokenize
import pickle
import re
import matplotlib.pyplot as plt
from io import BytesIO
import os
import signal
import base64

# ฟังก์ชันสำหรับประมวลผลข้อความ
def preprocess_text(text):
    if text is None:
        return ""
    text = str(text)
    words = word_tokenize(text)
    text = ' '.join(words)
    text = re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', text)
    return text

# ฟังก์ชันสำหรับแปลงข้อความเป็นฟีเจอร์
def extract_features(corpus, w2v_model):
    return np.array([np.mean([w2v_model.wv[word] for word in text.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0) for text in corpus])

# ฟังก์ชันสำหรับบันทึกข้อมูลลงไฟล์
def save_to_file(input_text, prediction):
    with open("analysis_results.txt", "a", encoding="utf-8") as file:
        file.write(f"ข้อความ: {input_text}\nผลการวิเคราะห์: {prediction}\n\n")

# ฟังก์ชันสำหรับนับประเภทของผลการวิเคราะห์
def count_sentiment_types(file_path):
    sentiment_counts = {'swear': 0, 'positive': 0, 'negative': 0}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        result_line = lines[i + 1]
        if 'swear' in result_line:
            sentiment_counts['swear'] += 1
        elif 'positive' in result_line:
            sentiment_counts['positive'] += 1
        elif 'negative' in result_line:
            sentiment_counts['negative'] += 1

    return sentiment_counts

# ฟังก์ชันสำหรับแสดง pie chart
def plot_pie_chart(sentiment_counts):
    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    total = sum(sizes)
    percentages = [f"{(size/total)*100:.2f}%" for size in sizes]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=[f"{label} ({size}, {percent})" for label, size, percent in zip(labels, sizes, percentages)], autopct='%1.1f%%', startangle=90)
    ax.set_title('ผลการวิเคราะห์ (Sentiment Analysis Results)')

    # Save pie chart to a BytesIO object and return the image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# ฟังก์ชันที่หยุดเซิร์ฟเวอร์ Flask
def shutdown_server():
    os.kill(os.getpid(), signal.SIGINT)

# โหลดข้อมูลและโมเดล
df = pd.read_excel('Thai_Sentiment.xlsx')
df['Text'] = df['Text'].apply(preprocess_text)

w2v_model = Word2Vec(sentences=[text.split() for text in df['Text']], vector_size=100, window=5, min_count=1, workers=4)
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# สร้าง Flask App
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
    return render_template_string('''
        <h1>Thai Sentiment Analysis</h1>
        <form method="POST" action="/predict">
            <label>กรอกข้อความภาษาไทย:</label><br>
            <input type="text" name="text" style="width: 300px;"><br><br>
            <input type="submit" value="วิเคราะห์">
        </form>
        <br>
        <a href="/result">ดูผลการวิเคราะห์ทั้งหมด</a><br><br>
        <form method="POST" action="/shutdown">
            <input type="submit" value="ปิดเซิร์ฟเวอร์">
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    processed_text = preprocess_text(user_text)
    features = extract_features([processed_text], w2v_model)
    prediction = rf_model.predict(features)
    prediction_text = prediction[0]

    # บันทึกข้อความและผลการวิเคราะห์ลงไฟล์
    save_to_file(user_text, prediction_text)

    return f"<h3>ผลการวิเคราะห์: {prediction_text}</h3><br><a href='/'>กลับหน้าหลัก</a>"

@app.route('/result')
def result():
    # นับผลการวิเคราะห์จากไฟล์
    sentiment_counts = count_sentiment_types('analysis_results.txt')

    # สร้าง pie chart
    img = plot_pie_chart(sentiment_counts)

    # แปลงภาพ pie chart เป็น base64 สำหรับการแสดงใน HTML
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template_string('''
        <h1>ผลการวิเคราะห์ทั้งหมด</h1>
        <img src="data:image/png;base64,{{img_data}}" alt="Pie Chart">
        <br><br>
        <a href="/">กลับหน้าหลัก</a>
    ''', img_data=img_base64)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run()