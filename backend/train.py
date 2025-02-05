# train_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import multiprocessing
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """แปลงข้อมูลให้พร้อมใช้งาน"""
    # ตรวจสอบและเติมค่าว่างในคอลัมน์ข้อความ
    df['processed_tweet'] = df['processed_tweet'].fillna('')
    
    # แปลง label เป็นตัวเลข
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['hashtage'])
    
    return df['processed_tweet'], y, label_encoder

def train_and_save_models(X, y, label_encoder):
    """เทรนโมเดลและบันทึกผล"""
    # สร้างโฟลเดอร์สำหรับเก็บโมเดล
    os.makedirs('saved_models', exist_ok=True)
    
    # แบ่งข้อมูลสำหรับเทรนและทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # เตรียม vectorizers
    bow_vectorizer = CountVectorizer(max_features=5000)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    
    # สร้าง features
    print("กำลังสร้าง features...")
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # สร้าง Word2Vec features
    print("กำลังสร้าง Word2Vec features...")
    sentences = [text.split() for text in X_train]
    w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    X_train_w2v = np.array([
        np.mean([w2v_model.wv[word] for word in text.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)
        for text in X_train
    ])
    
    X_test_w2v = np.array([
        np.mean([w2v_model.wv[word] for word in text.split() if word in w2v_model.wv] or [np.zeros(100)], axis=0)
        for text in X_test
    ])
    
    # เตรียมโมเดล
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    # เก็บผลลัพธ์
    results = []
    
    # ทดสอบทุกคู่ของ features และ classifiers
    for clf_name, clf in classifiers.items():
        # BoW
        print(f"\nกำลังเทรน {clf_name} ด้วย BoW...")
        clf_bow = clf.fit(X_train_bow, y_train)
        acc_bow = accuracy_score(y_test, clf_bow.predict(X_test_bow))
        results.append(('BoW', clf_name, acc_bow, clf_bow, bow_vectorizer))
        
        # TF-IDF
        print(f"กำลังเทรน {clf_name} ด้วย TF-IDF...")
        clf_tfidf = clf.fit(X_train_tfidf, y_train)
        acc_tfidf = accuracy_score(y_test, clf_tfidf.predict(X_test_tfidf))
        results.append(('TF-IDF', clf_name, acc_tfidf, clf_tfidf, tfidf_vectorizer))
        
        # Word2Vec
        print(f"กำลังเทรน {clf_name} ด้วย Word2Vec...")
        clf_w2v = clf.fit(X_train_w2v, y_train)
        acc_w2v = accuracy_score(y_test, clf_w2v.predict(X_test_w2v))
        results.append(('Word2Vec', clf_name, acc_w2v, clf_w2v, w2v_model))
    
    # เรียงลำดับตาม accuracy
    results.sort(key=lambda x: x[2], reverse=True)
    
    # บันทึก 3 โมเดลที่ดีที่สุด
    print("\nกำลังบันทึกโมเดลที่ดีที่สุด 3 อันดับ...")
    for rank, (feat_type, clf_name, acc, clf, vec) in enumerate(results[:3], 1):
        print(f"\nอันดับ {rank}:")
        print(f"ประเภท: {feat_type}")
        print(f"โมเดล: {clf_name}")
        print(f"Accuracy: {acc:.4f}")
        
        # สร้างชื่อไฟล์
        base_name = f"saved_models1/rank{rank}_{feat_type}_{clf_name}"
        
        # บันทึกโมเดล
        joblib.dump(clf, f"{base_name}_classifier.joblib")
        
        # บันทึก vectorizer
        if feat_type in ['BoW', 'TF-IDF']:
            joblib.dump(vec, f"{base_name}_vectorizer.joblib")
        else:  # Word2Vec
            joblib.dump(vec, f"{base_name}_word2vec.joblib")

def main():
    """ฟังก์ชันหลักสำหรับการเทรนและบันทึกโมเดล"""
    print("กำลังโหลดข้อมูล...")
    df = pd.read_excel('data/new_data1.xlsx')
    
    print("กำลังเตรียมข้อมูล...")
    X, y, label_encoder = preprocess_data(df)
    
    print("กำลังเทรนและบันทึกโมเดล...")
    train_and_save_models(X, y, label_encoder)
    
    print("\nเสร็จสิ้นการเทรนและบันทึกโมเดล!")

if __name__ == "__main__":
    main()