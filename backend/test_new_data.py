"""# Test by New data"""

import pickle
import numpy as np
import pandas as pd
import re
from pythainlp.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

# ฟังก์ชันสำหรับการประมวลผลข้อความ
def preprocess_text(text):
    return re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', ' '.join(word_tokenize(str(text or ""))))

# ฟังก์ชันสำหรับแปลงข้อความเป็นฟีเจอร์ด้วย Word2Vec
def extract_features_word2vec(corpus):
    model = Word2Vec(sentences=[text.split() for text in corpus], vector_size=100, window=5, min_count=1, workers=4)
    features = np.array([np.mean([model.wv[w] for w in text.split() if w in model.wv] or [np.zeros(100)], axis=0) for text in corpus])
    return features, model

# ฟังก์ชันสำหรับฝึกและบันทึกโมเดล
def train_and_save_model(features, labels, model_path='random_forest_model.pkl'):
    model = RandomForestClassifier().fit(features, labels)
    with open(model_path, 'wb') as f: pickle.dump(model, f)
    return model

# ฟังก์ชันสำหรับการทำนายผล
def predict_new_data(new_data, w2v_model, model_path='random_forest_model.pkl'):
    features = np.array([np.mean([w2v_model.wv[w] for w in preprocess_text(text).split() if w in w2v_model.wv] or [np.zeros(100)], axis=0) for text in new_data])
    with open(model_path, 'rb') as f: model = pickle.load(f)
    return model.predict(features)

# ฟังก์ชันหลัก
def main(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df['Text'] = df['Text'].fillna('').apply(preprocess_text)
    features, w2v_model = extract_features_word2vec(df['Text'])
    train_and_save_model(features, df['Class'])

    while True:
        text = input("กรุณากรอกข้อความ (หรือพิมพ์ 'exit' เพื่อออก): ")
        if text.lower() == 'exit': break
        print(f"ผลลัพธ์การทำนาย: {predict_new_data([text], w2v_model)[0]}")

# เรียกใช้งานฟังก์ชันหลัก
main('Thai_Sentiment.xlsx')