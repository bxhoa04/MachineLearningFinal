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

        preprocessor = DataPreprocessor(self.data_path, self.target_column)
        self.X, self.y, self.df = preprocessor.preprocess()

    def _scale_and_impute(self, X):
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        return imputer.fit_transform(scaler.fit_transform(X))

    def _evaluate_classifiers(self, X_train, X_test, y_train, y_test, classifiers):
        results = {}
        for clf in classifiers:
            clf_name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[clf_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
                "matrix": confusion_matrix(y_test, y_pred)
            }
        return results

    def results_classifier_before_umap(self, classifiers):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                            random_state=self.random_state)
        X_train_scaled, X_test_scaled = self._scale_and_impute(X_train), self._scale_and_impute(X_test)
        return self._evaluate_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, classifiers)

    def apply_umap(self, before_split=True, classifiers=None):
        if before_split:
            X_scaled = self._scale_and_impute(self.X)
            X_umap = umap.UMAP(n_components=self.n_components, n_jobs=-1).fit_transform(X_scaled)
            plt.scatter(X_umap[:, 0], X_umap[:, 1], c=self.y, cmap='Spectral', s=5)
            plt.title('UMAP before splitting data')
            plt.colorbar(label=self.target_column)
            plt.show()
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                random_state=self.random_state)
            X_train_scaled, X_test_scaled = self._scale_and_impute(X_train), self._scale_and_impute(X_test)
            umap_model = umap.UMAP(n_components=self.n_components, n_jobs=-1)
            X_train_umap, X_test_umap = umap_model.fit_transform(X_train_scaled), umap_model.transform(X_test_scaled)

            for X_umap, title in zip([X_train_umap, X_test_umap], ['Train', 'Test']):
                plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_train if title == 'Train' else y_test, cmap='Spectral', s=5)
                plt.title(f'UMAP after splitting data ({title})')
                plt.colorbar(label=self.target_column)
                plt.show()

            return self._evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)

    def find_optimal_n_components_elbow(self, max_components=14):
        X_scaled = self._scale_and_impute(self.X)
        inertia = [umap.UMAP(n_components=n, n_jobs=-1, random_state=42).fit_transform(X_scaled).var(axis=0).sum() for n
                   in range(1, max_components + 1)]

        results_df = pd.DataFrame({'n_component': range(1, max_components + 1), 'inertia': inertia})

        return results_df

    def _find_elbow_point(self, inertia):
        if len(inertia) < 2:
            raise ValueError("Inertia list must have at least 2 values to find elbow point.")
        n_points = len(inertia)
        start_point = np.array([0, inertia.iloc[0]])
        end_point = np.array([n_points - 1, inertia.iloc[-1]])
        line_vector = end_point - start_point
        distances = [
            np.abs(np.cross(line_vector, np.array([i, inertia.iloc[i]]) - start_point)) / np.linalg.norm(line_vector)
            for i in range(n_points)]
        return np.argmax(distances) + 1

    def apply_optimal_n_components(self, classifiers):
        results_df = self.find_optimal_n_components_elbow(max_components=14)
        optimal_n = self._find_elbow_point(results_df['inertia'])
        X_scaled = self._scale_and_impute(self.X)
        umap_model = umap.UMAP(n_components=optimal_n, n_jobs=-1, random_state=42)
        X_umap = umap_model.fit_transform(X_scaled)
        X_train_umap, X_test_umap, y_train, y_test = train_test_split(X_umap, self.y, test_size=self.test_size,
                                                                      random_state=42)
        return self._evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)


def print_results(results):
    for clf_name, metrics in results.items():
        print(f"Accuracy {clf_name:<25}{metrics['accuracy']:<10.4f}")
        print(f"\nClassification Report for {clf_name}:")
        print(metrics['classification_report'])
        print(f"\nConfusion Matrix for {clf_name}:")
        print(metrics['matrix'])