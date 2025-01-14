import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessor import DataPreprocessor

class PCA_Processor:
    def __init__(self, data_path, target_column, n_components=2, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state

        # Sử dụng DataPreprocessor để xử lý dữ liệu
        preprocessor = DataPreprocessor(self.data_path, self.target_column)
        self.X, self.y, self.df = preprocessor.preprocess()

    def apply_pca_before_split(self, classifiers):
            """
            Áp dụng PCA trước khi chia dữ liệu thành tập train và test.
            """
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)
            imputer = SimpleImputer(strategy='mean')
            X_scaled = imputer.fit_transform(X_scaled)
            pca = PCA(n_components=self.n_components)
            X_pca = pca.fit_transform(X_scaled)
            X_train_pca, X_test_pca, y_train, y_test = train_test_split(
                X_pca, self.y, test_size=self.test_size, random_state=self.random_state
            )
            results = self.evaluate_classifiers(X_train_pca, X_test_pca, y_train, y_test, classifiers)
            return results

    def apply_pca_after_split(self, classifiers):
            """
            Áp dụng PCA sau khi chia dữ liệu thành tập train và test.
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
            pca = PCA(n_components=self.n_components)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            results = self.evaluate_classifiers(X_train_pca, X_test_pca, y_train, y_test, classifiers)
            return results

    def evaluate_classifiers(self, X_train, X_test, y_train, y_test, classifiers):
            results = {}

            for clf in classifiers:
                clf_name = clf.__class__.__name__
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                # Tính toán độ chính xác
                accuracy = accuracy_score(y_test, y_pred)
                classification_reports = classification_report(y_test, y_pred)
                matrix = confusion_matrix(y_test, y_pred)
                # Lưu kết quả
                results[clf_name] = {
                    "accuracy": accuracy,
                    "classification_report": classification_reports,
                    "matrix": matrix
                }
            return results

    def select_pca_components(self, variance_threshold=0.95):
            """
            Chọn số lượng thành phần PCA để đạt tỷ lệ phương sai giải thích tối thiểu.
            Parameters:
            - variance_threshold (float): Tỷ lệ phương sai cần đạt (ví dụ: 0.95 hoặc 0.99).
            """
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)
            pca = PCA()
            pca.fit(X_scaled)
            cumulative_variance = pca.explained_variance_ratio_.cumsum()
            n_components = (cumulative_variance >= variance_threshold).argmax() + 1

            return n_components

    def visualize_pca_2d_cases(self):
            scaler = StandardScaler()
            # Trường hợp 1: PCA trước khi chia dữ liệu
            X_scaled = scaler.fit_transform(self.X)
            pca = PCA(n_components=self.n_components)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(10, 6))
            plt.scatter(X_pca[self.y == 0, 0], X_pca[self.y == 0, 1], alpha=0.7, label="Class 0", color='blue')
            plt.scatter(X_pca[self.y == 1, 0], X_pca[self.y == 1, 1], alpha=0.7, label="Class 1", color='orange')
            plt.title("PCA Trước Khi Chia Dữ Liệu (PC1 vs PC2)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Trường hợp 2: PCA sau khi chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # figsize để điều chỉnh kích thước tổng thể
            axes[0].scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], alpha=0.7, label="Class 0",
                            color='blue')
            axes[0].scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], alpha=0.7, label="Class 1",
                            color='orange')
            axes[0].set_title("PCA Sau Khi Chia Dữ Liệu (Train) (PC1 vs PC2)")
            axes[0].set_xlabel("Principal Component 1")
            axes[0].set_ylabel("Principal Component 2")
            axes[0].legend()
            axes[0].grid(True)

            # Trực quan hóa tập test
            axes[1].scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], alpha=0.7, label="Class 0",
                            color='blue')
            axes[1].scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], alpha=0.7, label="Class 1",
                            color='orange')
            axes[1].set_title("PCA Sau Khi Chia Dữ Liệu (Test) (PC1 vs PC2)")
            axes[1].set_xlabel("Principal Component 1")
            axes[1].set_ylabel("Principal Component 2")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

def print_results(results):
        for clf_name, metrics in results.items():
            print(f"Accuracy {clf_name:<15}{metrics['accuracy']:<10.4f}")
            print(f"\nClassification Report for {clf_name}:")
            print(metrics['classification_report'])
            print(f"\nConfusion Matrix for {clf_name}:")
            print(metrics['matrix'])
