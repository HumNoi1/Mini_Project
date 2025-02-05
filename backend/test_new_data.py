import joblib
import numpy as np
import pandas as pd
import re
from pythainlp.tokenize import word_tokenize
from gensim.models import Word2Vec

class TextClassificationPredictor:
    """
    A class for making predictions using the three best trained models.
    This class handles text preprocessing and prediction using different feature extraction methods.
    """
    
    def __init__(self, models_dir='saved_models'):
        """
        Initialize the predictor by loading the saved models and vectorizers.
        
        Parameters:
            models_dir (str): Directory containing the saved models and vectorizers
        """
        self.models_dir = models_dir
        self.models = []
        
        # โหลดโมเดลที่ดีที่สุด 3 อันดับ
        for i in range(1, 4):  # สำหรับ rank 1-3
            try:
                # อ่านข้อมูลโมเดลจากไฟล์
                model_info = self._load_model_files(i)
                if model_info:
                    self.models.append(model_info)
                    print(f"โหลดโมเดลอันดับที่ {i} สำเร็จ: {model_info['type']}")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดลอันดับที่ {i}: {str(e)}")
    
    def _load_model_files(self, rank):
        """
        Load model files for a specific rank.
        Returns a dictionary containing the model type, classifier, and vectorizer.
        """
        # ค้นหาไฟล์โมเดลตาม rank
        for model_type in ['BoW', 'TF-IDF', 'Word2Vec']:
            try:
                base_path = f"{self.models_dir}/rank{rank}_{model_type}"
                
                # โหลด classifier
                classifier = joblib.load(f"{base_path}_classifier.joblib")
                
                # โหลด vectorizer ตามประเภทของโมเดล
                if model_type in ['BoW', 'TF-IDF']:
                    vectorizer = joblib.load(f"{base_path}_vectorizer.joblib")
                elif model_type == 'Word2Vec':
                    vectorizer = joblib.load(f"{base_path}_word2vec.joblib")
                
                return {
                    'type': model_type,
                    'classifier': classifier,
                    'vectorizer': vectorizer,
                    'rank': rank
                }
            except:
                continue
        return None

    def preprocess_text(self, text):
        """
        Preprocess the input text by removing special characters and tokenizing.
        """
        # ทำความสะอาดข้อความและตัดคำ
        return re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', ' '.join(word_tokenize(str(text or ""))))

    def extract_features(self, text, model_info):
        """
        Extract features from text using the appropriate vectorizer.
        """
        processed_text = self.preprocess_text(text)
        
        if model_info['type'] in ['BoW', 'TF-IDF']:
            # สำหรับ BoW และ TF-IDF ใช้ vectorizer โดยตรง
            return model_info['vectorizer'].transform([processed_text])
        
        elif model_info['type'] == 'Word2Vec':
            # สำหรับ Word2Vec สร้าง document vector
            words = processed_text.split()
            word_vectors = [
                model_info['vectorizer'].wv[word]
                for word in words
                if word in model_info['vectorizer'].wv
            ]
            if not word_vectors:
                return np.zeros((1, model_info['vectorizer'].vector_size))
            return np.mean(word_vectors, axis=0).reshape(1, -1)

    def predict(self, text):
        """
        Make predictions using all loaded models and return results.
        """
        results = []
        
        for model_info in self.models:
            try:
                # แปลงข้อความเป็น features
                features = self.extract_features(text, model_info)
                
                # ทำนายผล
                prediction = model_info['classifier'].predict(features)[0]
                probability = np.max(model_info['classifier'].predict_proba(features)[0])
                
                results.append({
                    'rank': model_info['rank'],
                    'type': model_info['type'],
                    'prediction': prediction,
                    'confidence': probability
                })
                
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการทำนายด้วยโมเดล {model_info['type']}: {str(e)}")
        
        return results

def main():
    """
    Main function to run the prediction system.
    """
    # สร้าง predictor object
    predictor = TextClassificationPredictor()
    
    # ตรวจสอบว่าโหลดโมเดลสำเร็จหรือไม่
    if not predictor.models:
        print("ไม่พบโมเดลที่บันทึกไว้ กรุณาเทรนโมเดลก่อนใช้งาน")
        return
    
    print("\nระบบพร้อมทำนายข้อความใหม่")
    print("สามารถพิมพ์ 'exit' เพื่อออกจากโปรแกรม")
    
    while True:
        # รับข้อความจากผู้ใช้
        text = input("\nกรุณากรอกข้อความที่ต้องการทำนาย: ")
        
        if text.lower() == 'exit':
            break
        
        # ทำนายผลด้วยทุกโมเดล
        results = predictor.predict(text)
        
        # แสดงผลการทำนาย
        print("\nผลการทำนายจากโมเดลทั้งหมด:")
        for result in results:
            print(f"\nโมเดลอันดับที่ {result['rank']} ({result['type']}):")
            print(f"ผลการทำนาย: {result['prediction']}")
            print(f"ความมั่นใจ: {result['confidence']*100:.2f}%")
        
        # แสดงผลการทำนายแบบ voting
        predictions = [r['prediction'] for r in results]
        if predictions:
            majority_vote = max(set(predictions), key=predictions.count)
            print(f"\nผลการทำนายแบบ Majority Vote: {majority_vote}")

if __name__ == "__main__":
    main()