import os
import wfdb
import pandas as pd
import numpy as np
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, adjusted_rand_score


class ECGClassifier:
    def __init__(self, base_dir, csv_path):
        self.base_dir = base_dir
        self.csv_path = csv_path
        self.disease_classes = {
            "Myocardial Infarction (MI)": ["IMI", "ASMI"],
            "ST/T Change (STTC)": ["NST_", "NDT", "DIG", "ISC_"],
            "Conduction Disturbance (CD)": ["LAFB"],
            "Hypertrophy (HYP)": ["LVH"],
            "Normal (NORM)": ["NORM"]
        }
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_and_prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df["scp_codes"] = df["scp_codes"].apply(eval)
        df["label"] = None

        for disease, codes in self.disease_classes.items():
            mask = df["scp_codes"].apply(lambda x: any(code in x for code in codes))
            df.loc[mask, "label"] = disease

        df_filtered = df[df["label"].notnull()].reset_index(drop=True)
        df_filtered["label_encoded"] = self.le.fit_transform(df_filtered["label"])

        features_list = []
        labels = []

        for _, row in df_filtered.iterrows():
            relative_path = row["filename_lr"]
            subfolder = relative_path.split('/')[1]
            filename = relative_path.split('/')[-1]
            record_path = os.path.join(self.base_dir, subfolder, filename)

            features = self.extract_features(record_path)
            if features:
                features_list.append(features)
                labels.append(row["label_encoded"])

        self.X = np.array(features_list)
        self.y = np.array(labels)

    def extract_features(self, record_path):
        try:
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 1][:1000]  # Use channel 1
            fs = 100
            _, _, Zxx = stft(signal, fs=fs, nperseg=128, noverlap=64)
            magnitude = np.abs(Zxx)

            features = [
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude),
                np.min(magnitude),
                np.percentile(magnitude, 25),
                np.percentile(magnitude, 75),
            ]
            return features
        except:
            return None

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Standardize features for SVM, KNN, and optionally KMeans
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_random_forest(self):
        print("\n=== Random Forest ===")
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_rf.fit(self.X_train, self.y_train)
        y_pred = clf_rf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_))

    def train_svm(self):
        print("\n=== Support Vector Machine ===")
        clf_svm = SVC(kernel='rbf', probability=True)
        clf_svm.fit(self.X_train, self.y_train)
        y_pred = clf_svm.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_))

    def train_knn(self):
        print("\n=== K-Nearest Neighbors ===")
        clf_knn = KNeighborsClassifier(n_neighbors=5)
        clf_knn.fit(self.X_train, self.y_train)
        y_pred = clf_knn.predict(self.X_test)
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_))

    def run_kmeans_clustering(self):
        print("\n=== K-Means Clustering ===")
        # Optional: use scaled data for KMeans
        X_scaled = self.scaler.transform(self.X)
        num_classes = len(np.unique(self.y))
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        ari = adjusted_rand_score(self.y, cluster_labels)
        print(f"Adjusted Rand Index (K-Means vs true labels): {ari:.3f}")

    def run_all(self):
        self.load_and_prepare_data()
        self.split_data()
        self.train_random_forest()
        self.train_svm()
        self.train_knn()
        self.run_kmeans_clustering()


if __name__ == "__main__":
    classifier = ECGClassifier(base_dir="ptb-xl/records100", csv_path="ptb-xl/ptbxl_database.csv")
    classifier.run_all()
