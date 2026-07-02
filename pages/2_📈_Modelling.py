import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier

st.title("📈 Modelling")

# Buat 2 tab
tab1, tab2 = st.tabs([
    "📊 Tanpa SMOTE",
    "⚖️ Dengan SMOTE"
])

if "X_train_scaled" not in st.session_state:
    st.warning("⚠️ Silakan lakukan proses analisis terlebih dahulu")
    st.stop()

X_train_scaled = st.session_state.X_train_scaled
X_test_scaled = st.session_state.X_test_scaled
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test
scaler = st.session_state.scaler
# ===============================
# TAB 1
# ===============================
with tab1:
    st.header("Modelling tanpa SMOTE")
    st.subheader("Weighted K-Nearest Neighbors (WKNN)")

    def hitung_euclidean(titik_a, titik_b):
        return np.sqrt(np.sum((titik_a - titik_b) **2))
    # fungsi prediksi wknn 
    def wknn_predict_all(X_train, y_train, X_test, k):
        predictions = []
        for idx_test in range(len(X_test)):
            x_test_row = (
                X_test[idx_test]
                # cek tipe data format numpy array atau DataFrame
                if isinstance(X_test, np.ndarray)
                else X_test.iloc[idx_test]
            )

            distances = []

            for i in range(len(X_train)):
                x_train_row = (
                    X_train[i]
                    if isinstance(X_train, np.ndarray)
                    else X_train.iloc[i]
                )
                jarak = hitung_euclidean(
                    np.array(x_train_row),
                    np.array(x_test_row)
                )
                distances.append((jarak, y_train.iloc[i]))

            # Urutkan berdasarkan jarak terkecil
            distances.sort(key=lambda x: x[0])

            # Ambil k tetangga terdekat
            neighbors = distances[:k]

            # Weighted Voting
            voting = {}
            for jarak, label in neighbors:
                bobot = 1 / (jarak + 0.0001)  # menghindari pembagian 0
                voting[label] = voting.get(label, 0) + bobot

            # Kelas dengan bobot terbesar
            hasil_prediksi = max(voting, key=voting.get)
            predictions.append(hasil_prediksi)

        return np.array(predictions)

    # ==========================================
    # GRID SEARCH DENGAN CROSS VALIDATION (5-FOLD)
    # ==========================================
    st.subheader("Grid Search WKNN")
    
    param_k = [3, 5, 7, 9]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    iteration_logs = []
    summary_results = []
    # ubah X_train_scaled menjadi format yang konsisten (numpy array untuk X) agar bisa digunakan di fungsi wknn_predict_all
    X_train_arr = np.array(X_train_scaled)
    # ubah y_train menjadi format yang konsisten (Series untuk y) agar bisa digunakan di fungsi wknn_predict_all
    y_train_ser = pd.Series(y_train).reset_index(drop=True)

    # Lakukan pencarian parameter
    for k_val in param_k:
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_arr)):
            X_tr, X_val = X_train_arr[train_idx], X_train_arr[val_idx]
            y_tr, y_val = y_train_ser.iloc[train_idx], y_train_ser.iloc[val_idx]
            
            y_val_pred = wknn_predict_all(X_tr, y_tr, X_val, k_val)
            score = accuracy_score(y_val, y_val_pred)
            fold_scores.append(score)
            
            # Simpan log per fold
            iteration_logs.append({
                "Iterasi Parameter": f"Pengujian k={k_val}",
                "Fold / Split ke-": f"Fold {fold_idx + 1}",
                "Akurasi Validasi": score
            })
            
        # HITUNG RATA-RATA K-VAL INI
        mean_score = np.mean(fold_scores)
        
        # MASUKKAN BARIS RATA-RATA KE DALAM TABEL LOG AGAR SINKRON DILIHAT
        iteration_logs.append({
            "Iterasi Parameter": f"Pengujian k={k_val}",
            "Fold / Split ke-": "✨ RATA-RATA (MEAN)",
            "Akurasi Validasi": mean_score
        })
        
        summary_results.append({
            'param_n_neighbors': k_val,
            'mean_test_score': mean_score
        })

    # Tampilkan log iterasi ke Streamlit (Sekarang ada baris RATA-RATA di tiap k)
    df_logs = pd.DataFrame(iteration_logs)
    st.dataframe(df_logs, use_container_width=True)
    st.write("---")

    # Mengurutkan hasil rangkuman untuk mencari k terbaik
    results_sorted = pd.DataFrame(summary_results).sort_values(by='mean_test_score', ascending=False)

    # Ambil 2 k terbaik
    best_k1 = int(results_sorted.iloc[0]['param_n_neighbors'])
    best_k2 = int(results_sorted.iloc[1]['param_n_neighbors'])

    # Tampilan score di sini akan sama persis dengan baris "✨ RATA-RATA (MEAN)" di dalam tabel
    st.write(f"🏆 k terbaik 1: {best_k1} (Score Rata-rata CV: {results_sorted.iloc[0]['mean_test_score']:.4f})")
    st.write(f"🥈 k terbaik 2: {best_k2} (Score Rata-rata CV: {results_sorted.iloc[1]['mean_test_score']:.4f})")
    st.write("---")

    # ==========================================
    # FUNCTION RUN MODEL (Evaluasi Data Test Akhir)
    # ==========================================
    def run_wknn(k):
        # --- Evaluasi Data Training ---
        y_pred_train = wknn_predict_all(X_train_scaled, y_train, X_train_scaled, k)
        # acc_train = accuracy_score(y_train, y_pred_train)
        # report_train = classification_report(y_train, y_pred_train)
        # cm_train = confusion_matrix(y_train, y_pred_train)
        # --- Evaluasi Data Testing ---
        y_pred_test = wknn_predict_all(X_train_scaled, y_train, X_test_scaled, k)
        acc_test = accuracy_score(y_test, y_pred_test)
        report_test = classification_report(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test)
        return (acc_test, report_test, cm_test)
        # return (acc_train, report_train, cm_train), (acc_test, report_test, cm_test)

    # Jalankan model untuk 2 k terbaik
    # train_res1, test_res1 = run_wknn(best_k1)
    test_res1 = run_wknn(best_k1)
    test_res2 = run_wknn(best_k2)

    # Unpack hasil k terbaik 1
    # acc_train1, report_train1, cm_train1 = train_res1
    acc_test1, report_test1, cm_test1 = test_res1

    # Unpack hasil k terbaik 2
    # acc_train2, report_train2, cm_train2 = train_res2
    acc_test2, report_test2, cm_test2 = test_res2

    # =========================
    # CONFUSION MATRIX FUNCTION
    # =========================
    def plot_confusion_matrix(cm, title):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        fig.colorbar(im)
        st.pyplot(fig)
        plt.close(fig)

    # =========================================
    # HASIL k TERBAIK 1
    # =========================================
    st.subheader(f"🏆 Hasil Akhir Model (k = {best_k1})")
    
    # Membuat 2 kolom berdampingan untuk Train dan Test
    # col_train1, col_test1 = st.columns(2)
    
    # with col_train1:
    #     st.markdown("### 📝 **Data Training**")
    #     st.write(f"Akurasi Pengujian Data Train: **{acc_train1:.4f}**")
    #     st.text("Classification Report (Train):")
    #     st.text(report_train1)
    #     plot_confusion_matrix(cm_train1, f"CM Train WKNN k={best_k1}")
        
    # with col_test1:
    st.markdown("### 🧪 **Data Testing**")
    st.write(f"Akurasi Pengujian Data Test: **{acc_test1:.4f}**")
    st.text("Classification Report (Test):")
    st.text(report_test1)
    plot_confusion_matrix(cm_test1, f"CM Test WKNN k={best_k1}")

    st.write("---")

    # =========================================
    # HASIL k TERBAIK 2
    # =========================================
    st.subheader(f"🥈 Hasil Akhir Model (k = {best_k2})")
    
    # Membuat 2 kolom berdampingan untuk Train dan Test
    # col_test2 = st.columns(1)
    
    # with col_train2:
    #     st.markdown("### 📝 **Data Training**")
    #     st.write(f"Akurasi Pengujian Data Train: **{acc_train2:.4f}**")
    #     st.text("Classification Report (Train):")
    #     st.text(report_train2)
    #     plot_confusion_matrix(cm_train2, f"CM Train WKNN k={best_k2}")

    st.markdown("### 🧪 **Data Testing**")
    st.write(f"Akurasi Pengujian Data Test: **{acc_test2:.4f}**")
    st.text("Classification Report (Test):")
    st.text(report_test2)
    plot_confusion_matrix(cm_test2, f"CM Test WKNN k={best_k2}")
# ====================================================================================
    st.subheader("Logistic Regression")
    logreg = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs'
        # algoritma optimasi yang digunakan untuk menemukan parameter terbaik
        # class_weight='balanced'
    )

    # Training
    logreg.fit(X_train_scaled, y_train)

    # Prediksi untuk Train dan Test
    y_pred_logreg_train = logreg.predict(X_train_scaled)
    y_pred_logreg_test = logreg.predict(X_test_scaled)

    # Evaluasi Skor
    acc_logreg_train = accuracy_score(y_train, y_pred_logreg_train)
    acc_logreg_test = accuracy_score(y_test, y_pred_logreg_test)
    
    report_logreg_train = classification_report(y_train, y_pred_logreg_train)
    report_logreg_test = classification_report(y_test, y_pred_logreg_test)
    
    cm_logreg_train = confusion_matrix(y_train, y_pred_logreg_train)
    cm_logreg_test = confusion_matrix(y_test, y_pred_logreg_test)

    # Fungsi Plot Confusion Matrix dengan Reset Canvas
    def plot_confusion_matrix(cm, title):
        plt.clf() # Membersihkan canvas agar tidak tumpang tindih
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        fig.colorbar(im)
        st.pyplot(fig)
        plt.close(fig)

    # Tampilan Kolom Berdampingan untuk Logistic Regression
    # col_lr_train, col_lr_test = st.columns(2)
    
    # with col_lr_train:
    #     st.markdown("### 📝 **Data Training**")
    #     st.write(f"Accuracy Logistic Regression (Train): **{acc_logreg_train:.4f}**")
    #     st.text("Classification Report (Train):")
    #     st.text(report_logreg_train)
    #     plot_confusion_matrix(cm_logreg_train, "CM Train Logistic Regression")

    st.markdown("### 🧪 **Data Testing**")
    st.write(f"Accuracy Logistic Regression (Test): **{acc_logreg_test:.4f}**")
    st.text("Classification Report (Test):")
    st.text(report_logreg_test)
    plot_confusion_matrix(cm_logreg_test, "CM Test Logistic Regression")

    st.write("---")

# ======================================================================
    st.subheader("Ensemble Soft Voting Classifier")
    
    # WKNN = KNN dengan bobot 'distance'
    wknn1 = KNeighborsClassifier(n_neighbors=best_k1, weights='distance')
    wknn2 = KNeighborsClassifier(n_neighbors=best_k2, weights='distance')

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, multi_class='multinomial')

    voting_clf = VotingClassifier(
        estimators=[
            ('wknn1', wknn1),
            ('wknn2', wknn2),
            ('lr', lr)
        ],
        voting='soft'   # gunakan probabilitas
    )
    
    # Training
    voting_clf.fit(X_train_scaled, y_train)
    
    # Prediksi untuk Train dan Test
    y_pred_vote_train = voting_clf.predict(X_train_scaled)
    y_pred_vote_test = voting_clf.predict(X_test_scaled)

    # Evaluasi Skor
    acc_vote_train = accuracy_score(y_train, y_pred_vote_train)
    acc_vote_test = accuracy_score(y_test, y_pred_vote_test)
    
    report_vote_train = classification_report(y_train, y_pred_vote_train)
    report_vote_test = classification_report(y_test, y_pred_vote_test)
    
    cm_vote_train = confusion_matrix(y_train, y_pred_vote_train)
    cm_vote_test = confusion_matrix(y_test, y_pred_vote_test)

    # Tampilan Kolom Berdampingan untuk Voting Classifier
    # col_vote_train, col_vote_test = st.columns(2)
    
    # with col_vote_train:
    #     st.markdown("### 📝 **Data Training**")
    #     st.write(f"Akurasi Voting Classifier (Train): **{acc_vote_train:.4f}**")
    #     st.text("Classification Report (Train):")
    #     st.text(report_vote_train)
    #     plot_confusion_matrix(cm_vote_train, "CM Train Voting Classifier")

    st.markdown("### 🧪 **Data Testing**")
    st.write(f"Akurasi Voting Classifier (Test): **{acc_vote_test:.4f}**")
    st.text("Classification Report (Test):")
    st.text(report_vote_test)
    plot_confusion_matrix(cm_vote_test, "CM Test Voting Classifier")
    st.write("---")
    
    # TABEL EVALUASI AKHIR (KOMPARASI TRAIN VS TEST)
    st.subheader("📊 Evaluasi Model Tanpa SMOTE (Komparasi)")
    
    hasil_all = pd.DataFrame({
        'Model': [
            f'WKNN k={best_k1}',
            f'WKNN k={best_k2}',
            'Logistic Regression',
            'Soft Voting Ensemble'
        ],
        # 'Akurasi Training': [
        #     acc_train1,
        #     acc_train2,
        #     acc_logreg_train,
        #     acc_vote_train
        # ],
        'Akurasi Testing': [
            acc_test1,
            acc_test2,
            acc_logreg_test,
            acc_vote_test
        ]
    })

    # Menampilkan tabel komparasi agar mudah dibaca
    st.dataframe(hasil_all, use_container_width=True)
# ===============================
# TAB 2
# ===============================
with tab2:
    st.header("Modelling dengan Penerapan SMOTE")

    # ===============================
    # SMOTE
    # ===============================
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    st.write("Sebelum SMOTE:", y_train.value_counts())
    st.write("Sesudah SMOTE:", y_train_smote.value_counts())

    # =========================
    # GRID SEARCH WKNN (SMOTE)
    # =========================
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['distance']
    }

    knn = KNeighborsClassifier()

    grid_search_smote = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid_search_smote.fit(X_train_smote, y_train_smote)

    results_smote = pd.DataFrame(grid_search_smote.cv_results_)
    results_sorted_smote = results_smote.sort_values(by='mean_test_score', ascending=False)

    best_k1_s = int(results_sorted_smote.iloc[0]['param_n_neighbors'])
    best_k2_s = int(results_sorted_smote.iloc[1]['param_n_neighbors'])

    st.write(f"🏆 k terbaik 1 (SMOTE): {best_k1_s}")
    st.write(f"🥈 k terbaik 2 (SMOTE): {best_k2_s}")

    # =========================
    # FUNCTION WKNN
    # =========================
    def run_wknn_smote(k):
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model.fit(X_train_smote, y_train_smote)

        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return acc, report, cm

    acc1_s, report1_s, cm1_s = run_wknn_smote(best_k1_s)
    acc2_s, report2_s, cm2_s = run_wknn_smote(best_k2_s)

    # =========================
    # CONFUSION MATRIX
    # =========================
    def plot_confusion_matrix(cm, title):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")

        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        fig.colorbar(im)
        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # HASIL WKNN
    # =========================
    st.subheader(f"Hasil WKNN SMOTE (k = {best_k1_s})")
    st.write("Akurasi:", acc1_s)
    st.text(report1_s)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        plot_confusion_matrix(cm1_s, f"Confusion Matrix k={best_k1_s} (SMOTE)")

    st.write("===================================")

    st.subheader(f"Hasil WKNN SMOTE (k = {best_k2_s})")
    st.write("Akurasi:", acc2_s)
    st.text(report2_s)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        plot_confusion_matrix(cm2_s, f"Confusion Matrix k={best_k2_s} (SMOTE)")

    # =========================
    # GRID SEARCH TABLE
    # =========================
    st.subheader("Hasil Grid Search (SMOTE)")
    st.dataframe(results_sorted_smote[['param_n_neighbors', 'mean_test_score']])

    # =========================
    # LOGISTIC REGRESSION (SMOTE)
    # =========================
    st.subheader("Logistic Regression (SMOTE)")

    logreg_smote = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs'
    )

    logreg_smote.fit(X_train_smote, y_train_smote)
    y_pred_logreg_s = logreg_smote.predict(X_test_scaled)

    acc_logreg_s = accuracy_score(y_test, y_pred_logreg_s)

    st.write("Akurasi:", acc_logreg_s)
    st.text(classification_report(y_test, y_pred_logreg_s))

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred_logreg_s),
                              "Confusion Matrix Logistic Regression (SMOTE)")

    # =========================
    # VOTING CLASSIFIER (SMOTE)
    # =========================
    st.subheader("Ensemble Soft Voting (SMOTE)")

    wknn1 = KNeighborsClassifier(n_neighbors=best_k1_s, weights='distance')
    wknn2 = KNeighborsClassifier(n_neighbors=best_k2_s, weights='distance')

    voting_smote = VotingClassifier(
        estimators=[
            ('wknn1', wknn1),
            ('wknn2', wknn2),
            ('lr', logreg_smote)
        ],
        voting='soft'
    )

    voting_smote.fit(X_train_smote, y_train_smote)
    y_pred_vote_s = voting_smote.predict(X_test_scaled)

    acc_vote_s = accuracy_score(y_test, y_pred_vote_s)

    st.write("Akurasi Voting:", acc_vote_s)
    st.text(classification_report(y_test, y_pred_vote_s))

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred_vote_s),
                              "Confusion Matrix Voting (SMOTE)")

    # =========================
    # RINGKASAN MODEL
    # =========================
    st.subheader("Evaluasi Model (SMOTE)")

    hasil_all = pd.DataFrame({
        'Model': [
            f'WKNN k={best_k1_s}',
            f'WKNN k={best_k2_s}',
            'Logistic Regression',
            'Soft Voting'
        ],
        'Akurasi': [
            acc1_s,
            acc2_s,
            acc_logreg_s,
            acc_vote_s
        ]
    })

    st.dataframe(hasil_all)