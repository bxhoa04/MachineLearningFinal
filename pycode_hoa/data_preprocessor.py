# data_preprocessor.py
import pandas as pd
from sklearn.utils import resample


class DataPreprocessor:
    def __init__(self, data_path, target_column):
        """
        Khởi tạo preprocessor với đường dẫn file và cột mục tiêu
        Parameters:
        - data_path: Đường dẫn tới file CSV
        - target_column: Tên cột mục tiêu
        """
        self.data_path = data_path
        self.target_column = target_column
        # Load dữ liệu
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop(columns=['education'])

    def handle_missing_values(self):
        """
        Xử lý giá trị thiếu cho dữ liệu
        """
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

        return self.df

    def balance_classes(self):
        """
        Tăng cường lớp thiểu số bằng phương pháp resampling
        """
        df_majority = self.df[self.df[self.target_column] == 0]
        df_minority = self.df[self.df[self.target_column] == 1]

        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )

        self.df = pd.concat([df_majority, df_minority_upsampled])

        return self.df

    def preprocess(self):
        """
        Thực hiện toàn bộ quá trình tiền xử lý
        """
        # Xử lý giá trị thiếu
        self.handle_missing_values()

        # Cân bằng lớp
        self.balance_classes()

        # Tách đặc trưng và nhãn
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        return X, y, self.df

