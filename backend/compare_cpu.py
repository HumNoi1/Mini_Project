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
import warnings
warnings.filterwarnings('ignore')

class TextClassificationComparison:
    def __init__(self, random_state=42, min_samples_per_class=2):
        self.random_state = random_state
        self.min_samples_per_class = min_samples_per_class
        self.label_encoder = LabelEncoder()
        self.unique_classes = None
        
        # ดึงจำนวน CPU cores ที่มีทั้งหมด
        self.n_jobs = multiprocessing.cpu_count()
        print(f"\nกำลังใช้งาน CPU จำนวน {self.n_jobs} cores")
        
        # ปรับแต่งโมเดลให้ใช้งานหลาย CPU cores ในโมเดลที่รองรับ
        self.classifiers = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs  # RandomForest รองรับ parallel processing
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=random_state
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, 
                random_state=random_state
            )
        }
        
        # สร้าง vectorizers (ไม่มี n_jobs parameter)
        self.bow_vectorizer = CountVectorizer(max_features=5000)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.results = {}

    def load_and_preprocess_data(self, excel_path):
        """โหลดและเตรียมข้อมูลจากไฟล์ Excel"""
        print("\nกำลังโหลดและประมวลผลข้อมูล...")
        df = pd.read_excel(excel_path)
        
        class_counts = df['hashtage'].value_counts()
        print("\nการกระจายของข้อมูลในแต่ละคลาส:")
        print(class_counts.head())
        print(f"\nจำนวนคลาสทั้งหมด: {len(class_counts)}")
        
        valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
        df_filtered = df[df['hashtage'].isin(valid_classes)].copy()
        
        print(f"\nจำนวนคลาสที่เหลือหลังการกรอง (>= {self.min_samples_per_class} ตัวอย่าง): {len(valid_classes)}")
        print(f"จำนวนตัวอย่างที่เหลือ: {len(df_filtered)} จากทั้งหมด {len(df)}")
        
        self.unique_classes = sorted(valid_classes)
        y = self.label_encoder.fit_transform(df_filtered['hashtage'])
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df_filtered['processed_tweet'], 
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y
            )
        except ValueError as e:
            print("\nไม่สามารถใช้ stratified sampling ได้ จะใช้การสุ่มแบบปกติแทน")
            X_train, X_test, y_train, y_test = train_test_split(
                df_filtered['processed_tweet'],
                y,
                test_size=0.2,
                random_state=self.random_state
            )
        
        return X_train, X_test, y_train, y_test

    def create_word2vec_features(self, texts, vector_size=100):
        """สร้าง features ด้วย Word2Vec แบบ parallel"""
        print("กำลังสร้าง Word2Vec features แบบ parallel...")
        
        # แยกข้อความเป็นประโยคสำหรับ Word2Vec
        sentences = [text.split() for text in texts]
        
        # สร้างโมเดล Word2Vec ด้วย parallel processing
        w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=self.n_jobs  # ใช้ parallel processing
        )
        
        # ฟังก์ชันสำหรับประมวลผลแต่ละข้อความ
        def process_text(text):
            words = text.split()
            word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
            return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(vector_size)
        
        # ใช้ parallel processing ในการสร้าง document vectors
        doc_vectors = Parallel(n_jobs=self.n_jobs)(
            delayed(process_text)(text) for text in texts
        )
        
        return np.array(doc_vectors)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """เทรนและประเมินผลโมเดลแบบ parallel"""
        print("\nเริ่มการเทรนและประเมินผลโมเดล...")
        
        # สร้าง features ด้วยวิธีต่างๆ
        feature_sets = {}
        
        print("กำลังสร้าง Bag of Words features...")
        X_train_bow = self.bow_vectorizer.fit_transform(X_train)
        X_test_bow = self.bow_vectorizer.transform(X_test)
        feature_sets['BoW'] = (X_train_bow, X_test_bow)
        
        print("กำลังสร้าง TF-IDF features...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        feature_sets['TF-IDF'] = (X_train_tfidf, X_test_tfidf)
        
        X_train_w2v = self.create_word2vec_features(X_train)
        X_test_w2v = self.create_word2vec_features(X_test)
        feature_sets['Word2Vec'] = (X_train_w2v, X_test_w2v)

        # ฟังก์ชันสำหรับเทรนและประเมินผลแต่ละโมเดล
        def train_and_evaluate_model(feature_name, clf_name, clf, X_train_feat, X_test_feat):
            print(f"- เทรนโมเดล {clf_name} ด้วย {feature_name}...")
            clf.fit(X_train_feat, y_train)
            y_pred = clf.predict(X_test_feat)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, 
                y_pred,
                target_names=self.unique_classes,
                output_dict=True
            )
            return {
                'feature_name': feature_name,
                'clf_name': clf_name,
                'accuracy': accuracy,
                'report': report
            }

        # ประมวลผลแบบ parallel
        results = []
        for feature_name, (X_train_feat, X_test_feat) in feature_sets.items():
            print(f"\nกำลังทดสอบ {feature_name} แบบ parallel...")
            
            # ใช้ parallel processing เฉพาะการเทรนและทดสอบโมเดล
            feature_results = Parallel(n_jobs=self.n_jobs)(
                delayed(train_and_evaluate_model)(
                    feature_name, clf_name, clf, X_train_feat, X_test_feat
                )
                for clf_name, clf in self.classifiers.items()
            )
            results.extend(feature_results)

        # จัดเก็บผลลัพธ์
        for result in results:
            if result['feature_name'] not in self.results:
                self.results[result['feature_name']] = {}
            self.results[result['feature_name']][result['clf_name']] = {
                'accuracy': result['accuracy'],
                'report': result['report']
            }

    def plot_results(self):
        """สร้างกราฟแสดงผลการเปรียบเทียบ"""
        print("\nกำลังสร้างกราฟแสดงผลการเปรียบเทียบ...")
        accuracies = []
        for feature_name in self.results:
            for clf_name in self.results[feature_name]:
                accuracies.append({
                    'Feature': feature_name,
                    'Classifier': clf_name,
                    'Accuracy': self.results[feature_name][clf_name]['accuracy']
                })
        
        df_results = pd.DataFrame(accuracies)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_results, x='Feature', y='Accuracy', hue='Classifier')
        plt.title('การเปรียบเทียบประสิทธิภาพของโมเดล')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def print_detailed_results(self):
        """แสดงผลการวิเคราะห์แบบละเอียด"""
        print("\nผลการวิเคราะห์แบบละเอียด:")
        for feature_name in self.results:
            print(f"\n=== ผลลัพธ์สำหรับ {feature_name} ===")
            for clf_name in self.results[feature_name]:
                print(f"\n--- {clf_name} ---")
                print(f"Accuracy: {self.results[feature_name][clf_name]['accuracy']:.4f}")
                report = self.results[feature_name][clf_name]['report']
                print("\nรายงานการจำแนกประเภท:")
                report_df = pd.DataFrame(report).transpose().round(3)
                print(report_df)

def main():
    # สร้างอ็อบเจ็กต์สำหรับการเปรียบเทียบ
    comparison = TextClassificationComparison(min_samples_per_class=5)
    
    # โหลดและเตรียมข้อมูล
    X_train, X_test, y_train, y_test = comparison.load_and_preprocess_data('new_data1.xlsx')
    
    # เทรนและประเมินผลโมเดล
    comparison.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # แสดงผลด้วยกราฟ
    comparison.plot_results()
    
    # แสดงผลลัพธ์แบบละเอียด
    comparison.print_detailed_results()

if __name__ == "__main__":
    main()