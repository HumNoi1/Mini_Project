# text_classification.py

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

class TextClassificationComparison:
    """
    A comprehensive class for text classification model comparison and evaluation.
    This class handles data preprocessing, model training, evaluation, and model saving.
    """
    
    def __init__(self, random_state=42, min_samples_per_class=2):
        """
        Initialize the text classification comparison system.
        
        Parameters:
            random_state (int): Seed for reproducibility
            min_samples_per_class (int): Minimum number of samples required per class
        """
        self.random_state = random_state
        self.min_samples_per_class = min_samples_per_class
        self.label_encoder = LabelEncoder()
        self.unique_classes = None
        
        # Get number of CPU cores for parallel processing
        self.n_jobs = multiprocessing.cpu_count()
        print(f"\nUsing {self.n_jobs} CPU cores for processing")
        
        # Initialize classifiers with parallel processing where supported
        self.classifiers = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs
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
        
        # Initialize vectorizers for text feature extraction
        self.bow_vectorizer = CountVectorizer(max_features=5000)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        
        # Store results and trained models
        self.results = {}
        self.trained_models = {}
        
    def load_and_preprocess_data(self, excel_path):
        """
        Load and preprocess the data from Excel file.
        
        Parameters:
            excel_path (str): Path to the Excel file containing the data
            
        Returns:
            tuple: Training and testing data splits (X_train, X_test, y_train, y_test)
        """
        print("\nLoading and preprocessing data...")
        df = pd.read_excel(excel_path)
        
        # Analyze class distribution
        class_counts = df['hashtage'].value_counts()
        print("\nClass distribution:")
        print(class_counts.head())
        print(f"\nTotal number of classes: {len(class_counts)}")
        
        # Filter classes with insufficient samples
        valid_classes = class_counts[class_counts >= self.min_samples_per_class].index
        df_filtered = df[df['hashtage'].isin(valid_classes)].copy()
        
        print(f"\nClasses remaining after filtering (>= {self.min_samples_per_class} samples): {len(valid_classes)}")
        print(f"Samples remaining: {len(df_filtered)} out of {len(df)}")
        
        # Encode labels and split data
        self.unique_classes = sorted(valid_classes)
        y = self.label_encoder.fit_transform(df_filtered['hashtage'])
        
        try:
            # Try stratified split first
            return train_test_split(
                df_filtered['processed_tweet'],
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y
            )
        except ValueError:
            print("\nFalling back to regular split due to stratification error")
            return train_test_split(
                df_filtered['processed_tweet'],
                y,
                test_size=0.2,
                random_state=self.random_state
            )
            
    def create_word2vec_features(self, texts, vector_size=100):
        """
        Create document vectors using Word2Vec with parallel processing.
        
        Parameters:
            texts (list): List of text documents
            vector_size (int): Size of the word vectors
            
        Returns:
            numpy.ndarray: Document vectors
        """
        print("Creating Word2Vec features...")
        
        # Prepare sentences for Word2Vec
        sentences = [text.split() for text in texts]
        
        # Train Word2Vec model using parallel processing
        w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=self.n_jobs
        )
        
        # Create document vectors in parallel
        def process_text(text):
            words = text.split()
            word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
            return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(vector_size)
        
        doc_vectors = Parallel(n_jobs=self.n_jobs)(
            delayed(process_text)(text) for text in texts
        )
        
        return np.array(doc_vectors)
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models with different feature extraction methods.
        Also saves the top performing models.
        
        Parameters:
            X_train, X_test: Training and testing text data
            y_train, y_test: Training and testing labels
        """
        print("\nStarting model training and evaluation...")
        
        # Store features and vectorizers for each method
        feature_sets = {}
        self.trained_models = {}
        
        # Create Bag of Words features
        print("Creating Bag of Words features...")
        X_train_bow = self.bow_vectorizer.fit_transform(X_train)
        X_test_bow = self.bow_vectorizer.transform(X_test)
        feature_sets['BoW'] = (X_train_bow, X_test_bow)
        self.trained_models['BoW'] = {'vectorizer': self.bow_vectorizer}
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        feature_sets['TF-IDF'] = (X_train_tfidf, X_test_tfidf)
        self.trained_models['TF-IDF'] = {'vectorizer': self.tfidf_vectorizer}
        
        # Create Word2Vec features
        print("Creating Word2Vec features...")
        X_train_w2v = self.create_word2vec_features(X_train)
        X_test_w2v = self.create_word2vec_features(X_test)
        feature_sets['Word2Vec'] = (X_train_w2v, X_test_w2v)
        self.trained_models['Word2Vec'] = {'vectorizer': None}  # Will be created during saving
        
        # Train and evaluate models
        for feature_name, (X_train_feat, X_test_feat) in feature_sets.items():
            print(f"\nEvaluating {feature_name} features...")
            self.results[feature_name] = {}
            self.trained_models[feature_name]['classifiers'] = {}
            
            for clf_name, clf in self.classifiers.items():
                print(f"- Training {clf_name}...")
                
                # Train the model
                clf.fit(X_train_feat, y_train)
                self.trained_models[feature_name]['classifiers'][clf_name] = clf
                
                # Make predictions and evaluate
                y_pred = clf.predict(X_test_feat)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=self.unique_classes,
                    output_dict=True
                )
                
                self.results[feature_name][clf_name] = {
                    'accuracy': accuracy,
                    'report': report
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
        
        # Save the top performing models
        self.save_top_models(X_train)

    def save_top_models(self, X_train):
        """
        Save the top 3 performing models and their preprocessors.
        
        Parameters:
            X_train: Original training text data for Word2Vec recreation
        """
        print("\nRanking and saving top models...")
        
        # Create directory for saved models
        os.makedirs('saved_models', exist_ok=True)
        
        # Collect all models and their accuracies
        all_models = []
        for feature_name in self.results:
            for clf_name in self.results[feature_name]:
                all_models.append({
                    'feature_name': feature_name,
                    'clf_name': clf_name,
                    'accuracy': self.results[feature_name][clf_name]['accuracy'],
                    'classifier': self.trained_models[feature_name]['classifiers'][clf_name],
                    'vectorizer': self.trained_models[feature_name]['vectorizer']
                })
        
        # Sort models by accuracy and get top 3
        sorted_models = sorted(all_models, key=lambda x: x['accuracy'], reverse=True)
        top_3_models = sorted_models[:3]
        
        # Save top models
        print("\nTop 3 performing models:")
        for rank, model in enumerate(top_3_models, 1):
            print(f"\nRank {rank}:")
            print(f"Method: {model['feature_name']}")
            print(f"Model: {model['clf_name']}")
            print(f"Accuracy: {model['accuracy']:.4f}")
            
            # Create base filename
            base_filename = f"rank{rank}_{model['feature_name']}_{model['clf_name']}"
            
            # Save classifier
            clf_path = f"saved_models/{base_filename}_classifier.joblib"
            joblib.dump(model['classifier'], clf_path)
            print(f"Saved classifier to: {clf_path}")
            
            # Save vectorizer if applicable
            if model['vectorizer'] is not None:
                vec_path = f"saved_models/{base_filename}_vectorizer.joblib"
                joblib.dump(model['vectorizer'], vec_path)
                print(f"Saved vectorizer to: {vec_path}")
            
            # For Word2Vec, create and save a new model
            if model['feature_name'] == 'Word2Vec':
                sentences = [text.split() for text in X_train]
                w2v_model = Word2Vec(
                    sentences=sentences,
                    vector_size=100,
                    window=5,
                    min_count=1,
                    workers=self.n_jobs
                )
                w2v_path = f"saved_models/{base_filename}_word2vec.joblib"
                joblib.dump(w2v_model, w2v_path)
                print(f"Saved Word2Vec model to: {w2v_path}")

    def plot_results(self):
        """
        Create a bar plot comparing model performances.
        """
        print("\nCreating performance comparison plot...")
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
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def print_detailed_results(self):
        """
        Print detailed classification reports for all models.
        """
        print("\nDetailed Classification Reports:")
        for feature_name in self.results:
            print(f"\n=== Results for {feature_name} ===")
            for clf_name in self.results[feature_name]:
                print(f"\n--- {clf_name} ---")
                print(f"Accuracy: {self.results[feature_name][clf_name]['accuracy']:.4f}")
                report = self.results[feature_name][clf_name]['report']
                print("\nClassification Report:")
                report_df = pd.DataFrame(report).transpose().round(3)
                print(report_df)

def main():
    """
    Main function to run the text classification comparison.
    """
    # Create the comparison object
    print("Initializing Text Classification Comparison...")
    comparison = TextClassificationComparison(min_samples_per_class=5)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = comparison.load_and_preprocess_data('data/new_data1.xlsx')
    
    # Train and evaluate models
    comparison.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Show results
    comparison.plot_results()
    comparison.print_detailed_results()

if __name__ == "__main__":
    main()