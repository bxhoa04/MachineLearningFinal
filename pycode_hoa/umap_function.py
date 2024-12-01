import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
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

        # Lấy dữ liệu sau khi tiền xử lý
        self.X, self.y, self.df = preprocessor.preprocess()

    def apply_umap_before_split(self, classifiers):
        """
        Áp dụng UMAP trước khi chia dữ liệu thành tập train và test.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        imputer = SimpleImputer(strategy='mean')
        X_scaled = imputer.fit_transform(X_scaled)

        umap_model = umap.UMAP(n_components=self.n_components, n_jobs=-1)
        X_umap = umap_model.fit_transform(X_scaled)

        X_train_umap, X_test_umap, y_train, y_test = train_test_split(
            X_umap, self.y, test_size=self.test_size, random_state=self.random_state
        )

        self.visualize_umap(X_umap)

        results = self.evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)

        return results

    def apply_umap_after_split(self, classifiers):
        """
        Áp dụng UMAP sau khi chia dữ liệu thành tập train và test.
        """
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

        results = self.evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)

        return results

    def visualize_umap(self, X_umap):
        plt.figure(figsize=(10, 8))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=self.y, cmap='Spectral', s=5)
        plt.title('UMAP Projection of the Data')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.colorbar(label=self.target_column)
        plt.show()

    def evaluate_classifiers(self, X_train, X_test, y_train, y_test, classifiers):
        results = {}
        for clf in classifiers:
            clf_name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Tính các chỉ số hiệu suất
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']

            # Lưu kết quả
            results[clf_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            }

        return results

def print_results(results):
    print(f"{'Classifier':<25}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
    print("=" * 60)
    for clf_name, metrics in results.items():
        print(f"{clf_name:<25}{metrics['accuracy']:<10.4f}{metrics['precision']:<10.4f}{metrics['recall']:<10.4f}{metrics['f1_score']:<10.4f}")