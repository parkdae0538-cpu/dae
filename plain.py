import os
import random
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    GlobalAveragePooling2D, Reshape, Dense, multiply,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

import optuna

# =========================================
# 0. 시드 고정
# =========================================
seed = 35
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# =========================================
# 1. NSE 함수
# =========================================
def nse(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)
    return 1 - (np.sum((predictions - targets) ** 2) /
                np.sum((targets - np.mean(targets)) ** 2))

# =========================================
# 2. SE 블록
# =========================================
def squeeze_excite_block(input_tensor, ratio=16):
    filters = int(input_tensor.shape[-1])
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    reduced = max(filters // ratio, 1)  # 0 방지
    se = Dense(reduced, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return multiply([input_tensor, se])

# =========================================
# 3. 데이터 로드
# =========================================
X_raw = np.load(r'C:\Users\USER\Downloads\in\HOT5_clean12345.npy')   # (N, 3, 3, 11) 가정
y_raw = np.load(r'C:\Users\USER\Downloads\out\HOT5_clean12345.npy')  # (N,) 또는 (N, 1)

# y shape 정리
if y_raw.ndim == 1:
    y_raw = y_raw.reshape(-1, 1)
elif y_raw.ndim == 2 and y_raw.shape[1] != 1:
    raise ValueError(f"y shape가 이상함: {y_raw.shape}")

print("X_raw shape:", X_raw.shape)
print("y_raw shape:", y_raw.shape)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw,
    test_size=0.25,      # ★ 7:3
    random_state=seed
)

input_shape = X_raw.shape[1:]  # (3,3,C)

# =========================================
# 4. 모델 빌더 (하이퍼파라미터 입력)
# =========================================
def build_model(input_shape, trial):
    # ====== 탐색 범위를 "조금" 좁힌 버전 ======
    conv1_filters = trial.suggest_int("conv1_filters", 16, 144, step=16)
    conv2_filters = trial.suggest_int("conv2_filters", conv1_filters, 256, step=32)
    conv3_filters = trial.suggest_int("conv3_filters", conv2_filters, 320, step=32)
    conv4_filters = trial.suggest_int("conv4_filters", conv2_filters, 320, step=32)
    conv5_filters = trial.suggest_int("conv5_filters", 256, 448, step=32)

    se_ratio = trial.suggest_categorical("se_ratio", [4, 8, 16])

    dense1_units = trial.suggest_int("dense1_units", 256, 1024, step=128)
    dense2_units = trial.suggest_categorical("dense2_units", [16, 32, 64])

    dropout_c1 = trial.suggest_float("dropout_c1", 0.0, 0.3)
    dropout_c2 = trial.suggest_float("dropout_c2", 0.0, 0.3)
    dropout_c3 = trial.suggest_float("dropout_c3", 0.0, 0.3)
    dropout_c4 = trial.suggest_float("dropout_c4", 0.0, 0.3)
    dropout_d1 = trial.suggest_float("dropout_d1", 0.2, 0.5)

    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-5, log=True)

    lr = 5e-5

    # -----------------------------
    # SE-CNN 구조
    # -----------------------------
    inputs = Input(shape=input_shape)
    x = inputs

    # Conv1
    x = Conv2D(conv1_filters, (3, 3), padding='same', strides=(1, 1),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c1)(x)

    # Conv2
    x = Conv2D(conv2_filters, (3, 3), padding='same', strides=(1, 1),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c2)(x)

    # Conv3
    x = Conv2D(conv3_filters, (2, 2), padding='same', strides=(1, 1),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c3)(x)

    # Conv4
    x = Conv2D(conv4_filters, (2, 2), padding='same', strides=(1, 1),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c4)(x)

    # 마지막 1×1 Conv
    x = Conv2D(conv5_filters, (1, 1))(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)

    # GAP + Dense
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense1_units, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_d1)(x)

    x = Dense(dense2_units, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=lr, clipnorm=1.0)
    )
    return model

# =========================================
# 5. Optuna objective 함수 (K-Fold CV)
# =========================================
def objective(trial):
    # ---- K-Fold 설정 (Train 부분만 사용) ----
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    nse_scores = []

    # ==== 각 Fold마다 모델 새로 만들고 학습 ====
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_raw)):
        X_tr_raw = X_train_raw[tr_idx]
        X_val_raw = X_train_raw[val_idx]
        y_tr_raw = y_train_raw[tr_idx]
        y_val_raw = y_train_raw[val_idx]

        # ---- 스케일링 (Fold의 Train으로만 fit) ----
        x_tr_flat = X_tr_raw.reshape(-1, X_tr_raw.shape[-1])
        x_val_flat = X_val_raw.reshape(-1, X_val_raw.shape[-1])

        x_scaler = MinMaxScaler()
        x_scaler.fit(x_tr_flat)
        X_tr_scaled = x_scaler.transform(x_tr_flat).reshape(X_tr_raw.shape)
        X_val_scaled = x_scaler.transform(x_val_flat).reshape(X_val_raw.shape)

        y_scaler = MinMaxScaler()
        y_scaler.fit(y_tr_raw)
        y_tr_scaled = y_scaler.transform(y_tr_raw).ravel()
        y_val_scaled = y_scaler.transform(y_val_raw).ravel()

        xxTr = tf.convert_to_tensor(X_tr_scaled, dtype=tf.float32)
        yyTr = tf.convert_to_tensor(y_tr_scaled, dtype=tf.float32)
        xxVal = tf.convert_to_tensor(X_val_scaled, dtype=tf.float32)
        yyVal = tf.convert_to_tensor(y_val_scaled, dtype=tf.float32)

        # ---- 모델 생성 ----
        model = build_model(input_shape, trial)

        # ---- 배치 사이즈 튜닝 ----
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 24, 32])

        es = EarlyStopping(
            monitor='val_loss',
            patience=500,         # ★ 조금 줄임
            restore_best_weights=True,
            verbose=0
        )
        rlrop = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=30,
            verbose=0,
            min_delta=1e-7,
            mode='min',
            min_lr=1e-7
        )

        model.fit(
            xxTr, yyTr,
            batch_size=batch_size,
            epochs=700,
            verbose=0,
            validation_data=(xxVal, yyVal),
            callbacks=[es, rlrop]
        )

        # ---- 검증 성능 (역정규화 후 NSE) ----
        y_val_pred_scaled = model.predict(xxVal, verbose=0).reshape(-1, 1)
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled).ravel()
        y_val_real = y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()

        nse_val = nse(y_val_real, y_val_pred)
        nse_scores.append(nse_val)

    # K-Fold 평균 NSE
    mean_nse = float(np.mean(nse_scores))
    return mean_nse  # maximize

# =========================================
# 6. Optuna 실행 + 최종 학습/평가
# =========================================
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="senet_hot5_kfold_lr5e5",
        direction="maximize"
    )
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    print("\n===== Optuna 결과 =====")
    print("Best Val NSE (CV 평균):", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = study.best_params

    # ---- Train 전체를 다시 스케일링 (Test는 건드리지 않음) ----
    x_train_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    x_test_flat = X_test_raw.reshape(-1, X_test_raw.shape[-1])

    x_scaler = MinMaxScaler()
    x_scaler.fit(x_train_flat)
    X_train_scaled = x_scaler.transform(x_train_flat).reshape(X_train_raw.shape)
    X_test_scaled = x_scaler.transform(x_test_flat).reshape(X_test_raw.shape)

    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train_raw)
    y_train_scaled = y_scaler.transform(y_train_raw).ravel()
    y_test_scaled = y_scaler.transform(y_test_raw).ravel()

    xxTrain = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
    yyTrain = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)
    xxTest = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
    yyTest = tf.convert_to_tensor(y_test_scaled, dtype=tf.float32)

    # ---- best_params를 쓰기 위한 dummy trial ----
    class DummyTrial:
        def __init__(self, params):
            self.params = params
        def suggest_int(self, name, low, high, step=1):
            return self.params[name]
        def suggest_categorical(self, name, choices):
            return self.params[name]
        def suggest_float(self, name, low, high, log=False):
            return self.params[name]

    dummy_trial = DummyTrial(best_params)
    final_model = build_model(input_shape, dummy_trial)

    es_final = EarlyStopping(
        monitor='loss',
        patience=500,
        restore_best_weights=True,
        verbose=1
    )
    rlrop_final = ReduceLROnPlateau(
        monitor='loss',
        factor=0.4,
        patience=30,
        verbose=1,
        min_delta=1e-7,
        mode='min',
        min_lr=1e-7
    )

    final_model.fit(
        xxTrain, yyTrain,
        batch_size=best_params["batch_size"],
        epochs=700,
        verbose=1,
        callbacks=[es_final, rlrop_final]
    )

    # ---- Train / Test 역정규화 후 성능 ----
    y_train_pred_scaled = final_model.predict(xxTrain, verbose=0).reshape(-1, 1)
    y_test_pred_scaled = final_model.predict(xxTest, verbose=0).reshape(-1, 1)

    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).ravel()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled).ravel()
    y_train_real = y_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
    y_test_real = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    print("\n===== 최종 모델 성능 (Train / Test, 7:3) =====")
    print(f"Train R² : {r2_score(y_train_real, y_train_pred):.4f}")
    print(f"Test  R² : {r2_score(y_test_real, y_test_pred):.4f}")

    print(f"Train NSE: {nse(y_train_real, y_train_pred):.4f}")
    print(f"Test  NSE: {nse(y_test_real, y_test_pred):.4f}")

    print(f"Train RMSE: {sqrt(mean_squared_error(y_train_real, y_train_pred)):.4f}")
    print(f"Test  RMSE: {sqrt(mean_squared_error(y_test_real, y_test_pred)):.4f}")

    # ---- 산점도 (Observed vs Predicted) ----
    max_val = max(
        y_train_real.max(), y_train_pred.max(),
        y_test_real.max(), y_test_pred.max()
    )
    ticks = np.linspace(0, max_val, 7)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot([0, max_val], [0, max_val], 'r-', linewidth=1.0, alpha=0.3)
    plt.scatter(y_train_real, y_train_pred)
    plt.title('Train (70%)')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.subplot(1, 2, 2)
    plt.plot([0, max_val], [0, max_val], 'r-', linewidth=1.0, alpha=0.3)
    plt.scatter(y_test_real, y_test_pred)
    plt.title('Test (30%)')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.tight_layout()
    plt.show()

    final_model.save(r'D:\model\senet_hot5_kfold_lr5e5_best.h5')
