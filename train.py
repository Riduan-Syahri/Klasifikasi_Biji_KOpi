import os
import cv2
import zipfile
import shutil
import joblib
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report

# Ekstraksi Fitur Gambar
def extract_features_from_image(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, target_size)

    # Fitur HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean, h_std = cv2.meanStdDev(hsv[:, :, 0])
    s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
    v_mean, v_std = cv2.meanStdDev(hsv[:, :, 2])

    # Fitur Histogram RGB
    hist_b = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [8], [0, 256]).flatten()
    hist_r = cv2.calcHist([img], [2], None, [8], [0, 256]).flatten()

    # Fitur Tekstur Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_mean, g_std = cv2.meanStdDev(gray)

    features = np.hstack([
        [h_mean[0][0], h_std[0][0], s_mean[0][0], s_std[0][0], v_mean[0][0], v_std[0][0]],
        hist_b, hist_g, hist_r,
        [g_mean[0][0], g_std[0][0]]
    ])
    return features

# Load Dataset dari Zip
def load_dataset_from_zip(zip_filename="Datashetbijikopi.zip", extract_dir="./dataset_extracted"):
    if not os.path.exists(extract_dir) and os.path.exists(zip_filename):
        print(f"Mengekstrak {zip_filename}...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✅ Ekstraksi selesai.")

    image_paths, labels = [], []
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

    for root, _, filenames in os.walk(extract_dir):
        for filename in filenames:
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(root, filename)
                parent_folder = os.path.basename(root)
                if parent_folder.lower() in ['train', 'test', 'val']:
                    continue
                image_paths.append(full_path)
                labels.append(parent_folder)

    return pd.DataFrame({'image_path': image_paths, 'label': labels})

def main():
    print("--- 1. SCANNING DATASET ---")
    df_raw = load_dataset_from_zip()
    
    if len(df_raw) == 0:
        print("❌ Dataset tidak ditemukan. Pastikan Datashetbijikopi.zip diunggah.")
        return

    print(f"Total gambar ditemukan: {len(df_raw)}")
    print(df_raw['label'].value_counts())

    print("\n--- 2. EKSTRAKSI FITUR ---")
    X_list, y_list = [], []
    for _, row in df_raw.iterrows():
        feat = extract_features_from_image(row['image_path'])
        if feat is not None:
            X_list.append(feat)
            y_list.append(row['label'])

    X = np.array(X_list)
    y = np.array(y_list)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- 3. OPTIMASI PARAMETER SVM VIA GA ---")
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_c", random.uniform, -2, 3)
    toolbox.register("attr_gamma", random.uniform, -4, 1)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_c, toolbox.attr_gamma), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_svm(individual):
        c_val = 10 ** individual[0]
        gamma_val = 10 ** individual[1]
        clf = SVC(C=c_val, gamma=gamma_val, kernel='rbf', class_weight='balanced', random_state=42)
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        return (f1_score(y_test, preds, average='macro'),)

    toolbox.register("evaluate", evaluate_svm)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=20)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, verbose=True)

    best_ind = tools.selBest(pop, 1)[0]
    best_C = 10 ** best_ind[0]
    best_gamma = 10 ** best_ind[1]

    final_model = SVC(C=best_C, gamma=best_gamma, kernel='rbf', class_weight='balanced', random_state=42)
    final_model.fit(X_train_scaled, y_train)

    print("\n--- EVALUASI FINAL ---")
    y_pred = final_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs("./saved_models", exist_ok=True)
    joblib.dump(final_model, "./saved_models/svm_ga_model.pkl")
    joblib.dump(scaler, "./saved_models/scaler.pkl")
    joblib.dump(le, "./saved_models/label_encoder.pkl")
    print("✅ Model, Scaler, dan LabelEncoder berhasil disimpan di folder 'saved_models/'.")

if __name__ == "__main__":
    main()
