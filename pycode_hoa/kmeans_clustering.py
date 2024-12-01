import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
from data_preprocessor import DataPreprocessor  # Import trực tiếp


class KMeansClusterer:
    def __init__(self, data_path, target_column,
                 n_clusters_range=(2, 10),
                 random_state=42):
        """
        Khởi tạo K-means Clustering
        """
        # Sử dụng DataPreprocessor để tiền xử lý
        preprocessor = DataPreprocessor(data_path, target_column)
        self.X, self.y, self.df = preprocessor.preprocess()

        # Các tham số
        self.n_clusters_range = n_clusters_range
        self.random_state = random_state

    def preprocess_data(self):
        """
        Chuẩn hóa dữ liệu sau khi tiền xử lý
        """
        scaler = StandardScaler()
        return scaler.fit_transform(self.X)

    def find_optimal_clusters(self, method='silhouette'):
        """
        Tìm số lượng cụm tối ưu

        Parameters:
        - method: Phương pháp chọn ('silhouette' hoặc 'elbow')
        """
        # Tiền xử lý dữ liệu
        X_scaled = self.preprocess_data()

        # Tính toán
        inertias = []
        silhouette_scores = []

        for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(X_scaled)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        # Chọn số lượng cụm tối ưu
        if method == 'silhouette':
            optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + self.n_clusters_range[0]
        else:
            knee_locator = KneeLocator(
                range(self.n_clusters_range[0], self.n_clusters_range[1] + 1),
                inertias,
                curve='convex',
                direction='decreasing'
            )
            optimal_clusters = knee_locator.elbow

        # Trực quan hóa
        self._visualize_cluster_selection(inertias, silhouette_scores, optimal_clusters)

        return optimal_clusters

    def _visualize_cluster_selection(self, inertias, silhouette_scores, optimal_clusters):
        """
        Trực quan hóa quá trình chọn số lượng cụm
        """
        plt.figure(figsize=(12, 5))

        # Subplot Elbow
        plt.subplot(1, 2, 1)
        plt.plot(range(self.n_clusters_range[0], self.n_clusters_range[1] + 1),
                 inertias, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Số lượng cụm')
        plt.ylabel('Inertia')
        plt.axvline(x=optimal_clusters, color='r', linestyle='--')

        # Subplot Silhouette
        plt.subplot(1, 2, 2)
        plt.plot(range(self.n_clusters_range[0], self.n_clusters_range[1] + 1),
                 silhouette_scores, marker='o')
        plt.title('Silhouette Score')
        plt.xlabel('Số lượng cụm')
        plt.ylabel('Điểm Silhouette')
        plt.axvline(x=optimal_clusters, color='r', linestyle='--')

        plt.tight_layout()
        plt.show()

    def perform_clustering(self, n_clusters):
        """
        Thực hiện phân cụm

        Parameters:
        - n_clusters: Số lượng cụm
        """
        # Tiền xử lý dữ liệu
        X_scaled = self.preprocess_data()

        # Thực hiện K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Trực quan hóa
        self._visualize_clusters(X_scaled, cluster_labels, n_clusters)

        return cluster_labels, kmeans

    def _visualize_clusters(self, X_scaled, cluster_labels, n_clusters):
        """
        Trực quan hóa các cụm
        """
        # Giảm chiều dữ liệu
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=cluster_labels,
            cmap='viridis',
            alpha=0.7
        )
        plt.title(f'Kết quả phân cụm (K = {n_clusters})')
        plt.xlabel('Thành phần chính 1')
        plt.ylabel('Thành phần chính 2')
        plt.colorbar(scatter)
        plt.show()

    def analyze_clusters(self, cluster_labels):
        """
        Phân tích đặc điểm của các cụm
        """
        # Thêm nhãn cụm vào DataFrame
        df_clustered = self.df.copy()
        df_clustered['Cluster'] = cluster_labels

        # Thống kê đặc điểm của từng cụm
        cluster_summary= df_clustered.groupby('Cluster').mean()

        print("Tóm tắt các cụm:")
        print(cluster_summary)

        return df_clustered, cluster_summary