import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from data_preprocessor import DataPreprocessor
from sklearn.metrics import classification_report, accuracy_score

class KMeansClusterer:
    def __init__(self, data_path, target_column, random_state=42):
        preprocessor = DataPreprocessor(data_path, target_column)
        self.X, self.y, self.df = preprocessor.preprocess()
        self.random_state = random_state

    def preprocess_data(self):
        scaler = StandardScaler()
        return scaler.fit_transform(self.X)

    def perform_clustering(self, n_clusters=2, save_plot=False):
        # Chia dữ liệu train và valid
        X_scaled = self.preprocess_data()
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, self.y, test_size=0.2, random_state=self.random_state)

        # Thực hiện K-means trên tập train và validation
        kmeans_train = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans_val = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)

        train_labels = kmeans_train.fit_predict(X_train)
        val_labels = kmeans_val.fit_predict(X_val)

        # Tính toán điểm Silhouette
        train_silhouette = silhouette_score(X_train, train_labels)
        val_silhouette = silhouette_score(X_val, val_labels)

        # Tính toán độ chính xác
        train_accuracy = accuracy_score(y_train, train_labels)
        val_accuracy = accuracy_score(y_val, val_labels)

        # Trực quan hóa kết quả
        self._visualize_clustering_results(X_train, X_val, train_labels, val_labels,
                                           kmeans_train.cluster_centers_, kmeans_val.cluster_centers_,
                                           train_silhouette, val_silhouette,
                                           save_plot)

        # Báo cáo phân loại
        self._generate_cluster_report(train_labels, val_labels, y_train, y_val,
                                      train_silhouette, val_silhouette,
                                      train_accuracy, val_accuracy)

        return train_labels, val_labels, kmeans_train, kmeans_val

    def _visualize_clustering_results(self, X_train, X_val, train_labels, val_labels,
                                      train_centers, val_centers,
                                      train_silhouette, val_silhouette,
                                      save_plot=False):
        # Giảm chiều dữ liệu
        pca_train = PCA(n_components=2)
        pca_val = PCA(n_components=2)
        X_train_pca = pca_train.fit_transform(X_train)
        X_val_pca = pca_val.fit_transform(X_val)
        train_centers_pca = pca_train.transform(train_centers)
        val_centers_pca = pca_val.transform(val_centers)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Trực quan hóa tập train
        scatter_train = ax1.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                                    c=train_labels, cmap='viridis', alpha=0.7)
        ax1.scatter(train_centers_pca[:, 0], train_centers_pca[:, 1],
                    c='red', marker='x', s=200, linewidths=3)
        ax1.set_title(f'Phân cụm Tập Train\nSilhouette: {train_silhouette:.4f}')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        plt.colorbar(scatter_train, ax=ax1)

        # Trực quan hóa tập validation
        scatter_val = ax2.scatter(X_val_pca[:, 0], X_val_pca[:, 1],
                                  c=val_labels, cmap='viridis', alpha=0.7)
        ax2.scatter(val_centers_pca[:, 0], val_centers_pca[:, 1],
                    c='red', marker='x', s=200, linewidths=3)
        ax2.set_title(f'Phân cụm Tập Validation\nSilhouette: {val_silhouette:.4f}')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        plt.colorbar(scatter_val, ax=ax2)

        plt.tight_layout()

        if save_plot:
            plt.savefig('clustering_results.png')
        else:
            plt.show()

    def _generate_cluster_report(self, train_labels, val_labels, y_train, y_val,
                                 train_silhouette, val_silhouette,
                                 train_accuracy, val_accuracy):

        print("\n--- BÁO CÁO PHÂN LOẠI PHÂN CỤM ---")
        print(f"Silhouette Tập Train: {train_silhouette:.4f}")
        print(f"Silhouette Tập Validation: {val_silhouette:.4f}")
        print(f"Độ chính xác Tập Train: {train_accuracy:.4f}")
        print(f"Độ chính xác Tập Validation: {val_accuracy:.4f}")

        print("\nPhân phối nhãn cụm:")
        print("Tập Train:")
        print(np.unique(train_labels, return_counts=True))
        print("Tập Validation:")
        print(np.unique(val_labels, return_counts=True))

        print("\nClassification Report: Validation")
        print(classification_report(y_val, val_labels))