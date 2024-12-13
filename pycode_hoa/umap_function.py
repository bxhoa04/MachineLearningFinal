import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import umap.umap_ as umap
import warnings
from data_preprocessor import DataPreprocessor

warnings.filterwarnings('ignore')

class UMAP_Processor:
    def __init__(self, data_path, target_column, n_components=2, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state

        # Sử dụng DataPreprocessor để xử lý dữ liệu
        preprocessor = DataPreprocessor(self.data_path, self.target_column)
        self.X, self.y, self.df = preprocessor.preprocess()

    def apply_umap_before_split(self, classifiers, show_plots=True):
        # Áp dụng UMAP trước khi chia dữ liệu thành tập train và test.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        imputer = SimpleImputer(strategy='mean')
        X_scaled = imputer.fit_transform(X_scaled)

        umap_model = umap.UMAP(n_components=self.n_components, n_jobs=-1)
        X_umap = umap_model.fit_transform(X_scaled)
        X_train_umap, X_test_umap, y_train, y_test = train_test_split(
            X_umap, self.y, test_size=self.test_size, random_state=self.random_state
        )
        if show_plots:
            plt.figure(figsize=(10, 8))
            plt.scatter(X_umap[:, 0], X_umap[:, 1], c=self.y, cmap='Spectral', s=5)
            plt.title('Áp dụng UMAP trước khi chia dữ liệu')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.colorbar(label=self.target_column)
            plt.show()
        results = self.evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)
        return results

    def apply_umap_after_split(self, classifiers, show_plots=True):
        # Áp dụng UMAP sau khi chia dữ liệu thành tập train và test.
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        imputer = SimpleImputer(strategy='mean')
        X_train_scaled = imputer.fit_transform(X_train_scaled)
        X_test_scaled = imputer.transform(X_test_scaled)

        umap_model = umap.UMAP(n_components=self.n_components, n_jobs=-1)
        X_train_umap = umap_model.fit_transform(X_train_scaled)
        X_test_umap = umap_model.transform(X_test_scaled)

        if show_plots:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap='Spectral', s=5)
            plt.title('Áp dụng UMAP sau khi chia dữ liệu (Train)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.colorbar(label=self.target_column)
            plt.subplot(1, 2, 2)
            plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=y_test, cmap='Spectral', s=5)
            plt.title('Áp dụng UMAP sau khi chia dữ liệu (Test)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.colorbar(label=self.target_column)
            plt.tight_layout()
            plt.show()

        results = self.evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)
        return results

    def evaluate_classifiers(self, X_train, X_test, y_train, y_test, classifiers):
        results = {}
        for clf in classifiers:
            clf_name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            classification_reports = classification_report(y_test, y_pred)
            matrix = confusion_matrix(y_test, y_pred)

            results[clf_name] = {
                "accuracy": accuracy,
                "classification_report": classification_reports,
                "matrix": matrix
            }
        return results

    def find_optimal_n_components_elbow(self, max_components=14):
        # Tìm số lượng thành phần tối ưu cho UMAP bằng phương pháp Elbow.

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        imputer = SimpleImputer(strategy='mean')
        X_scaled = imputer.fit_transform(X_scaled)
        inertia = []
        components_range = range(1, max_components + 1)
        for n in components_range:
            umap_model = umap.UMAP(n_components=n, n_jobs=-1)
            X_umap = umap_model.fit_transform(X_scaled)
            inertia.append(X_umap.var(axis=0).sum())

        results_df = pd.DataFrame({
            'n_component': components_range,
            'inertia': inertia
        })

        plt.figure(figsize=(10, 6))
        plt.plot(results_df['n_component'], results_df['inertia'], marker='o')
        plt.title('Elbow Method for Optimal n_components')
        plt.xlabel('Number of Components')
        plt.ylabel('Inertia (Variance)')
        plt.xticks(results_df['n_component'])
        plt.grid()
        plt.show()

        optimal_n = self._find_elbow_point(results_df['inertia'])
        print(f'Optimal number of dimensions: {optimal_n}')
        return results_df

    def _find_elbow_point(self, inertia):
        # Tìm điểm gãy trong đồ thị
        return np.argmax(np.diff(inertia)) + 1

def print_results(results):
    for clf_name, metrics in results.items():
        print(f"Accuracy {clf_name:<25}{metrics['accuracy']:<10.4f}")
        print(f"\nClassification Report for {clf_name}:")
        print(metrics['classification_report'])
        print(f"\nConfusion Matrix for {clf_name}:")
        print(metrics['matrix'])