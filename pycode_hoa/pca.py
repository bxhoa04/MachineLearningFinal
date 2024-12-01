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

        # Lấy dữ liệu sau khi tiền xử lý
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

            # Tính các chỉ số hiệu suất
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']
            matrix = confusion_matrix(y_test, y_pred)

            # Lưu kết quả
            results[clf_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "matrix": matrix
            }

        return results

def print_results(results):
    print(f"{'Classifier':<25}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
    print("=" * 55)
    for clf_name, metrics in results.items():
        print(f"{clf_name:<25}{metrics['accuracy']:<10.4f}{metrics['precision']:<10.4f}{metrics['recall']:<10.4f}{metrics['f1_score']:<10.4f}")

        print(f"\nConfusion Matrix for {clf_name}:")
        print(metrics['matrix'])


def visualize_comparison(results_before, results_after):
    """
    Trực quan hóa so sánh kết quả giữa PCA trước khi chia và PCA sau khi chia.
    """
    classifiers = list(results_before.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        before_values = [results_before[clf][metric] for clf in classifiers]
        after_values = [results_after[clf][metric] for clf in classifiers]

        x = range(len(classifiers))

        plt.bar(x, before_values, width=0.4, label='PCA Before Split', align='center')
        plt.bar([i + 0.4 for i in x], after_values, width=0.4, label='PCA After Split', align='center')

        plt.xticks([i + 0.2 for i in x], classifiers, rotation=45, ha='right')
        plt.ylabel(metric.capitalize())
        plt.title(f'Comparison of {metric.capitalize()} between PCA Before and After Split')
        plt.legend()
        plt.tight_layout()
        plt.show()

def visualize_pca_scatter(processor):
    """
    Trực quan hóa PCA với cả hai trường hợp:
    1. PCA trước khi chia dữ liệu.
    2. PCA sau khi chia dữ liệu.

    Parameters:
    - processor: Đối tượng PCA_Processor chứa dữ liệu và thông tin cài đặt.
    """
    # Trường hợp 1: PCA trước khi chia dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(processor.X)

    pca = PCA(n_components=processor.n_components)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[processor.y == 0, 0], X_pca[processor.y == 0, 1], alpha=0.7, label="Class 0", color='blue')
    plt.scatter(X_pca[processor.y == 1, 0], X_pca[processor.y == 1, 1], alpha=0.7, label="Class 1", color='orange')
    plt.title("Áp dụng PCA trước khi chia dữ liệu ")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Trường hợp 2: PCA sau khi chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        processor.X,
        processor.y,
        test_size=processor.test_size,
        random_state=processor.random_state
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], alpha=0.7, label="Class 0", color='blue')
    plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], alpha=0.7, label="Class 1", color='orange')
    plt.title("Áp dụng PCA sau khi chia dữ liệu (Train)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], alpha=0.7, label="Class 0", color='blue')
    plt.scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], alpha=0.7, label="Class 1", color='orange')
    plt.title("Áp dụng PCA sau khi chia dữ liệu (Test)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()