import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import umap
import warnings
warnings.filterwarnings('ignore')

class UMAP_Processor:
    def __init__(self, data_path, target_column, n_components=2, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state

        # Load dữ liệu
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop(columns=['education'])

        # Xử lý giá trị thiếu
        self.preprocess_data()

        # Tách các đặc trưng (X) và biến mục tiêu (y)
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

    def preprocess_data(self):
        # Định nghĩa các cột nhị phân
        bin_cols = ["male", "currentSmoker", "prevalentStroke", "prevalentHyp", "diabetes"]
        # Điền giá trị thiếu cho các đặc trưng nhị phân bằng giá trị thường gặp nhất (mode)
        for col in bin_cols:
            mode_val = self.df[col].mode()[0]
            self.df[col] = self.df[col].fillna(mode_val)
        # Điền giá trị thiếu cho các đặc trưng số
        numeric_cols = ["cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]
        for col in numeric_cols:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val)

        # Tăng cường lớp thiểu số
        df_majority = self.df[self.df[self.target_column] == 0]
        df_minority = self.df[self.df[self.target_column] == 1]
        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=len(df_majority),
                                         random_state=42)
        self.df = pd.concat([df_majority, df_minority_upsampled])

    def results_classider_before_umap(self, classifiers):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        imputer = SimpleImputer(strategy='mean')
        X_train_scaled = imputer.fit_transform(X_train_scaled)
        X_test_scaled = imputer.transform(X_test_scaled)

        results = self.evaluate_classifiers(X_train_scaled, X_test_scaled, y_train, y_test, classifiers)

        return results

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
            X_umap, self.y, test_size=self.test_size, random_state=42
        )

        plt.figure(figsize=(10, 8))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=self.y, cmap='Spectral', s=5)
        plt.title('Áp dụng UMAP trước khi chia dữ liệu')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.colorbar(label=self.target_column)
        plt.show()

        results = self.evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)

        return results

    def apply_umap_after_split(self, classifiers):
        """
        Áp dụng UMAP sau khi chia dữ liệu thành tập train và test.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
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

        plt.tight_layout()  # Tự động căn chỉnh các biểu đồ
        plt.show()

        results = self.evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)

        return results

    def find_optimal_n_components_elbow(self, max_components=14):
        """
        Tìm số lượng thành phần tối ưu cho UMAP bằng phương pháp Elbow.
        Trả về DataFrame với số lượng thành phần và độ chính xác tương ứng.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        imputer = SimpleImputer(strategy='mean')
        X_scaled = imputer.fit_transform(X_scaled)

        inertia = []
        components_range = range(1, max_components + 1)

        for n in components_range:
            umap_model = umap.UMAP(n_components=n, n_jobs=-1, random_state=42)
            X_umap = umap_model.fit_transform(X_scaled)

            # Tính toán độ phân tán (inertia) của các điểm trong không gian UMAP
            inertia.append(X_umap.var(axis=0).sum())

        # Kiểm tra nếu inertia rỗng
        if len(inertia) == 0:
            raise ValueError("Không tính được inertia. Vui lòng kiểm tra lại dữ liệu đầu vào.")
        # Tạo DataFrame với n_component và inertia
        results_df = pd.DataFrame({
            'n_component': components_range,
            'inertia': inertia
        })

        # Vẽ biểu đồ để xác định số lượng thành phần tối ưu
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['n_component'], results_df['inertia'], marker='o')
        plt.title('Elbow Method for Optimal n_components')
        plt.xlabel('Number of Components')
        plt.ylabel('Inertia (Variance)')
        plt.xticks(results_df['n_component'])
        plt.grid()

        # Tìm số chiều tối ưu (n_components) bằng cách xác định điểm gãy (elbow)
        optimal_n = self._find_elbow_point(results_df['inertia'])

        # Vẽ đường nét đứt tại điểm tối ưu
        plt.plot([optimal_n, optimal_n], [results_df['inertia'].min(), results_df['inertia'].max()], 'k--', linestyle='--', color='red')

        plt.show()

        print(f'Số chiều tối ưu là: {optimal_n}')

        return results_df

    def _find_elbow_point(self, inertia):
        """
        Tìm điểm gãy (elbow point) trong danh sách độ phân tán (inertia).
        """
        # Kiểm tra độ dài của danh sách inertia
        if len(inertia) < 2:
            raise ValueError("Danh sách inertia phải có ít nhất 2 giá trị để tìm điểm gãy.")

        # Lấy số lượng điểm
        n_points = len(inertia)

        # Tạo vector từ điểm đầu đến điểm cuối
        start_point = np.array([0, inertia.iloc[0]])  # Truy cập phần tử đầu tiên
        end_point = np.array([n_points - 1, inertia.iloc[-1]])  # Truy cập phần tử cuối cùng
        line_vector = end_point - start_point

        # Tính khoảng cách từ mỗi điểm đến đoạn thẳng
        distances = []
        for i in range(n_points):
            point = np.array([i, inertia.iloc[i]])  # Truy cập giá trị tại chỉ số i
            distance = np.abs(np.cross(line_vector, point - start_point)) / np.linalg.norm(line_vector)
            distances.append(distance)

        # Tìm vị trí có khoảng cách lớn nhất (điểm gãy)
        elbow_point = np.argmax(distances) + 1  # +1 vì index bắt đầu từ 0
        return elbow_point

    def apply_optimal_n_components(self, classifiers):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        imputer = SimpleImputer(strategy='mean')
        X_scaled = imputer.fit_transform(X_scaled)

        # Tìm số chiều tối ưu
        results_df = self.find_optimal_n_components_elbow(max_components=14)
        optimal_n = self._find_elbow_point(results_df['inertia'])

        umap_model = umap.UMAP(n_components=optimal_n, n_jobs=-1, random_state=42)
        X_umap = umap_model.fit_transform(X_scaled)

        X_train_umap, X_test_umap, y_train, y_test = train_test_split(
            X_umap, self.y, test_size=self.test_size, random_state=42
        )

        results = self.evaluate_classifiers(X_train_umap, X_test_umap, y_train, y_test, classifiers)

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

def print_results(results):
    for clf_name, metrics in results.items():
        print(f"Accuracy {clf_name:<25}{metrics['accuracy']:<10.4f}")

        print(f"\nClassification Report for {clf_name}:")
        print(metrics['classification_report'])

        print(f"\nConfusion Matrix for {clf_name}:")
        print(metrics['matrix'])