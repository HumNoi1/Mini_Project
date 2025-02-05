import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import os
from pathlib import Path

def get_project_root():
    """ค้นหาโฟลเดอร์หลักของโปรเจค"""
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    return current_dir.parent.parent

def filter_rare_classes(df, min_samples=5):
    """
    กรองข้อมูลโดยเก็บเฉพาะคลาสที่มีจำนวนตัวอย่างมากกว่าค่าที่กำหนด
    
    Parameters:
        df: DataFrame ที่ต้องการกรอง
        min_samples: จำนวนตัวอย่างขั้นต่ำที่ยอมรับได้สำหรับแต่ละคลาส
    """
    # นับจำนวนตัวอย่างในแต่ละคลาส
    class_counts = df['hashtage'].value_counts()
    print("\nการกระจายของข้อมูลในแต่ละคลาส:")
    print(class_counts.head())
    
    # หาคลาสที่มีจำนวนตัวอย่างเพียงพอ
    valid_classes = class_counts[class_counts >= min_samples].index
    print(f"\nจำนวนคลาสทั้งหมด: {len(class_counts)}")
    print(f"จำนวนคลาสที่มีตัวอย่างมากกว่า {min_samples} ตัวอย่าง: {len(valid_classes)}")
    
    # กรองข้อมูล
    filtered_df = df[df['hashtage'].isin(valid_classes)].copy()
    print(f"จำนวนข้อมูลหลังกรอง: {len(filtered_df)} จากทั้งหมด {len(df)}")
    
    return filtered_df

def evaluate_model_with_real_data(model, vectorizer, data_file, save_dir):
    """
    ประเมินผลโมเดลโดยใช้ข้อมูลจริง พร้อมจัดการการเข้ารหัสคลาสอย่างถูกต้อง
    
    การทำงานหลัก:
    1. โหลดและกรองข้อมูล
    2. แปลง features ให้ตรงกับที่โมเดลต้องการ
    3. จัดการการเข้ารหัสคลาสให้ตรงกับที่ใช้ตอนเทรนโมเดล
    4. คำนวณและบันทึกผลการประเมิน
    """
    print(f"\nกำลังโหลดข้อมูลจากไฟล์: {data_file}")
    
    try:
        # โหลดและกรองข้อมูล
        df = pd.read_excel(data_file)
        print(f"จำนวนข้อมูลทั้งหมด: {len(df)} แถว")
        print(f"คอลัมน์ที่มี: {', '.join(df.columns)}")
        
        # กรองข้อมูลที่มีจำนวนตัวอย่างน้อยเกินไปออก
        df = filter_rare_classes(df, min_samples=5)
        
        # แบ่งข้อมูล
        X = df['processed_tweet']
        y = df['hashtage']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print("\nจำนวนข้อมูล:")
        print(f"ชุดฝึก (Training): {len(X_train)} ตัวอย่าง")
        print(f"ชุดทดสอบ (Test): {len(X_test)} ตัวอย่าง")
        
        # เริ่มจับเวลาการทำนาย
        start_time = time.time()
        
        # แสดงข้อมูล vectorizer
        print("\nข้อมูล Vectorizer:")
        print(f"ประเภท: {type(vectorizer).__name__}")
        if hasattr(vectorizer, 'vocabulary_'):
            print(f"จำนวนคำในคลังเดิม: {len(vectorizer.vocabulary_)}")
            print(f"ตัวอย่างคำในคลัง: {list(vectorizer.vocabulary_.keys())[:5]}")
        
        print("\nกำลังแปลงข้อความเป็น features...")
        
        # ใช้ vectorizer เดิมโดยตรง
        X_test_features = vectorizer.transform(X_test)
        print(f"รูปร่างของ features ก่อนปรับ: {X_test_features.shape}")
        
        # ตรวจสอบจำนวน features ที่โมเดลคาดหวัง
        n_features_model = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
        print(f"จำนวน features ที่โมเดลคาดหวัง: {n_features_model}")
        
        # ถ้าจำนวน features ไม่ตรงกัน ใช้วิธีเลือก features ที่สำคัญที่สุด
        if X_test_features.shape[1] != n_features_model:
            print("\nกำลังปรับจำนวน features ให้ตรงกับโมเดล...")
            
            # คำนวณความสำคัญของ features
            if isinstance(vectorizer, TfidfVectorizer):
                feature_importance = np.array(X_test_features.mean(axis=0)).flatten()
            else:  # CountVectorizer
                feature_importance = np.array(X_test_features.sum(axis=0)).flatten()
            
            # เลือก features ที่สำคัญที่สุด
            top_indices = np.argsort(feature_importance)[-n_features_model:]
            X_test_features = X_test_features[:, top_indices]
            
            print(f"รูปร่างของ features หลังปรับ: {X_test_features.shape}")
        
        # ทำนายผล
        print("\nกำลังทำนายผล...")
        y_pred = model.predict(X_test_features)
        y_pred_proba = model.predict_proba(X_test_features)
        
        print("\nตัวอย่างผลการทำนาย:")
        print("- ค่าจริง:", y_test.iloc[0])
        print("- ค่าทำนาย:", y_pred[0])
        print("- ประเภทข้อมูล (ค่าจริง):", type(y_test.iloc[0]))
        print("- ประเภทข้อมูล (ค่าทำนาย):", type(y_pred[0]))
        
        # ถ้าผลการทำนายเป็นตัวเลข แปลงกลับเป็นชื่อคลาส
        if np.issubdtype(y_pred.dtype, np.number):
            print("\nกำลังแปลงรหัสคลาสกลับเป็นชื่อ...")
            # โหลด label encoder จากโมเดล
            if hasattr(model, 'classes_'):
                class_mapping = {i: label for i, label in enumerate(model.classes_)}
                y_pred = np.array([class_mapping[pred] for pred in y_pred])
                print("แปลงค่าทำนายเรียบร้อย:", y_pred[0])
        
        prediction_time = time.time() - start_time
        
        # สร้าง label encoder สำหรับการประเมินผล
        unique_labels = sorted(set(y_test.unique()))
        print(f"\nคลาสที่พบในข้อมูลทดสอบ: {len(unique_labels)} คลาส")
        print(f"ตัวอย่างคลาส: {', '.join(list(unique_labels)[:5])}")
        
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        y_test_encoded = np.array([label_to_id[label] for label in y_test])
        y_pred_encoded = np.array([label_to_id[label] for label in y_pred])
        
        # คำนวณ metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test_encoded, y_pred_encoded, average='weighted'),
            'recall': recall_score(y_test_encoded, y_pred_encoded, average='weighted'),
            'f1_score': f1_score(y_test_encoded, y_pred_encoded, average='weighted'),
            'training_time': prediction_time,
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred_encoded),
            'class_labels': unique_labels,
            'class_distribution': dict(df['hashtage'].value_counts()),
            'n_features': X_test_features.shape[1],
            'vocabulary_size': len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else None
        }
        
        # คำนวณ ROC curve สำหรับคลาสแรก
        fpr, tpr, _ = roc_curve(
            (y_test_encoded == 0).astype(int),
            y_pred_proba[:, 0]
        )
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        
        print("\nผลการประเมิน:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"เวลาที่ใช้: {metrics['training_time']:.2f} วินาที")
        
        print("\nข้อมูล Features:")
        print(f"- จำนวน features ที่ใช้: {metrics['n_features']}")
        print(f"- ขนาดคลังคำศัพท์: {metrics['vocabulary_size']}")
        
        return metrics
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประเมินผล: {str(e)}")
        print("รายละเอียดเพิ่มเติม:", e.__class__.__name__)
        print("\nข้อมูลการดีบัก:")
        print("- ประเภทของโมเดล:", type(model).__name__)
        if hasattr(model, 'classes_'):
            print("- คลาสที่โมเดลรู้จัก:", model.classes_)
        import traceback
        traceback.print_exc()
        raise e

def evaluate_all_models():
    """ประเมินผลโมเดลทั้งหมด"""
    try:
        # ตั้งค่าพาธ
        project_root = get_project_root()
        models_dir = project_root / "saved_models"
        data_file = project_root / "new_data1.xlsx"
        
        print(f"กำลังค้นหาไฟล์โมเดลใน: {models_dir}")
        print(f"กำลังใช้ไฟล์ข้อมูล: {data_file}")
        
        os.makedirs(models_dir, exist_ok=True)
        
        if not data_file.exists():
            print(f"ไม่พบไฟล์ข้อมูล: {data_file}")
            return
            
        # รายชื่อโมเดล
        model_configs = [
            {
                'name': 'rank1_TF-IDF_RandomForest',
                'classifier_file': 'rank1_TF-IDF_RandomForest_classifier.joblib',
                'vectorizer_file': 'rank1_TF-IDF_RandomForest_vectorizer.joblib'
            },
            {
                'name': 'rank2_BoW_RandomForest',
                'classifier_file': 'rank2_BoW_RandomForest_classifier.joblib',
                'vectorizer_file': 'rank2_BoW_RandomForest_vectorizer.joblib'
            },
            {
                'name': 'rank3_BoW_GradientBoosting',
                'classifier_file': 'rank3_BoW_GradientBoosting_classifier.joblib',
                'vectorizer_file': 'rank3_BoW_GradientBoosting_vectorizer.joblib'
            }
        ]
        
        print("\nเริ่มการประเมินผลโมเดล...")
        
        for config in model_configs:
            try:
                print(f"\nกำลังประเมินโมเดล: {config['name']}")
                
                model_path = models_dir / config['classifier_file']
                vectorizer_path = models_dir / config['vectorizer_file']
                
                print(f"ตรวจสอบไฟล์:")
                print(f"- Model: {model_path}")
                print(f"- Vectorizer: {vectorizer_path}")
                
                if not model_path.exists() or not vectorizer_path.exists():
                    print("ไม่พบไฟล์โมเดลหรือ vectorizer")
                    continue
                
                print("กำลังโหลดโมเดลและ vectorizer...")
                model = joblib.load(model_path)
                vectorizer = joblib.load(vectorizer_path)
                
                # ประเมินผลโมเดล
                metrics = evaluate_model_with_real_data(model, vectorizer, data_file, models_dir)
                
                if metrics:
                    metrics_file = models_dir / f"{config['name']}_metrics.joblib"
                    joblib.dump(metrics, metrics_file)
                    print(f"บันทึก metrics ไปที่: {metrics_file}")
                
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการประเมินโมเดล {config['name']}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทำงาน: {str(e)}")

if __name__ == "__main__":
    evaluate_all_models()