import pandas as pd
from sklearn.utils import resample

class DataPreprocessor:
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop(columns=['education'])

    def handle_missing_values(self):
        # Xử lý giá trị thiếu cho dữ liệu
        bin_cols = ["male", "currentSmoker", "prevalentStroke", "prevalentHyp", "diabetes"]
        for col in bin_cols:
            mode_val = self.df[col].mode()[0]
            self.df[col] = self.df[col].fillna(mode_val)
        numeric_cols = ["cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]
        for col in numeric_cols:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val)

    def balance_classes(self):
        df_majority = self.df[self.df[self.target_column] == 0]
        df_minority = self.df[self.df[self.target_column] == 1]
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        self.df = pd.concat([df_majority, df_minority_upsampled])

    def preprocess(self):
        # Thực hiện toàn bộ quá trình tiền xử lý
        self.handle_missing_values()  # Đảm bảo xử lý giá trị thiếu
        self.balance_classes()
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y