import argparse
import os
import random
from math import sqrt

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Reshape,
    multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


SEED = 35


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def nse(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.array(targets)
    predictions = np.array(predictions)
    return 1 - (
        np.sum((predictions - targets) ** 2)
        / np.sum((targets - np.mean(targets)) ** 2)
    )


def squeeze_excite_block(input_tensor, ratio: int = 16):
    filters = int(input_tensor.shape[-1])
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    reduced = max(filters // ratio, 1)
    se = Dense(reduced, activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)
    return multiply([input_tensor, se])


def build_best_model(input_shape, l2_reg: float = 1e-5):
    conv1_filters = 128
    conv2_filters = 224
    conv3_filters = 256
    conv4_filters = 288
    conv5_filters = 384

    se_ratio = 16

    dense1_units = 768
    dense2_units = 64

    dropout_c1 = 0.1207623438708754
    dropout_c2 = 0.20885774341423669
    dropout_c3 = 0.17842710924442584
    dropout_c4 = 0.007375245138482419
    dropout_d1 = 0.23794352148265036

    lr = 5e-5

    inputs = Input(shape=input_shape)
    x = inputs

    x = Conv2D(conv1_filters, (3, 3), padding="same", strides=(1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c1)(x)

    x = Conv2D(conv2_filters, (3, 3), padding="same", strides=(1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c2)(x)

    x = Conv2D(conv3_filters, (2, 2), padding="same", strides=(1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c3)(x)

    x = Conv2D(conv4_filters, (2, 2), padding="same", strides=(1, 1), activation="relu")(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = Dropout(dropout_c4)(x)

    x = Conv2D(conv5_filters, (1, 1))(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x, ratio=se_ratio)

    x = GlobalAveragePooling2D()(x)
    x = Dense(dense1_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_d1)(x)

    x = Dense(dense2_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr, clipnorm=1.0))
    return model


def scale_features(X_train, X_test, y_train, y_test):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    x_scaler.fit(X_train_flat)
    X_train_scaled = x_scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = x_scaler.transform(X_test_flat).reshape(X_test.shape)

    y_scaler.fit(y_train)
    y_train_scaled = y_scaler.transform(y_train).ravel()
    y_test_scaled = y_scaler.transform(y_test).ravel()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler


def train_and_evaluate(X_raw, y_raw, model_output: str, scaler_prefix: str):
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=SEED
    )

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler = scale_features(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw
    )

    input_shape = X_raw.shape[1:]
    model = build_best_model(input_shape=input_shape)

    batch_size = 8

    es = EarlyStopping(
        monitor="loss",
        patience=200,
        restore_best_weights=True,
        verbose=1,
    )
    rlrop = ReduceLROnPlateau(
        monitor="loss",
        factor=0.4,
        patience=30,
        verbose=1,
        min_delta=1e-7,
        mode="min",
        min_lr=1e-7,
    )

    model.fit(
        tf.convert_to_tensor(X_train_scaled, dtype=tf.float32),
        tf.convert_to_tensor(y_train_scaled, dtype=tf.float32),
        batch_size=batch_size,
        epochs=700,
        verbose=1,
        callbacks=[es, rlrop],
    )

    y_train_pred_scaled = model.predict(X_train_scaled, verbose=0).reshape(-1, 1)
    y_test_pred_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1, 1)

    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).ravel()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled).ravel()

    y_train_real = y_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
    y_test_real = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    metrics = {
        "train_r2": r2_score(y_train_real, y_train_pred),
        "test_r2": r2_score(y_test_real, y_test_pred),
        "train_nse": nse(y_train_real, y_train_pred),
        "test_nse": nse(y_test_real, y_test_pred),
        "train_rmse": sqrt(mean_squared_error(y_train_real, y_train_pred)),
        "test_rmse": sqrt(mean_squared_error(y_test_real, y_test_pred)),
    }

    os.makedirs(os.path.dirname(model_output) or ".", exist_ok=True)
    model.save(model_output)

    if scaler_prefix:
        os.makedirs(os.path.dirname(scaler_prefix) or ".", exist_ok=True)
        joblib.dump(x_scaler, f"{scaler_prefix}_x_scaler.joblib")
        joblib.dump(y_scaler, f"{scaler_prefix}_y_scaler.joblib")

    return metrics, y_train_real, y_train_pred, y_test_real, y_test_pred


def plot_results(y_train_real, y_train_pred, y_test_real, y_test_pred):
    max_val = max(
        y_train_real.max(), y_train_pred.max(), y_test_real.max(), y_test_pred.max()
    )
    ticks = np.linspace(0, max_val, 7)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot([0, max_val], [0, max_val], "r-", linewidth=1.0, alpha=0.3)
    plt.scatter(y_train_real, y_train_pred)
    plt.title("Train (75%)")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.subplot(1, 2, 2)
    plt.plot([0, max_val], [0, max_val], "r-", linewidth=1.0, alpha=0.3)
    plt.scatter(y_test_real, y_test_pred)
    plt.title("Test (25%)")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train best-performing SE-CNN model.")
    parser.add_argument("--x-path", required=True, help="Path to the input features .npy file.")
    parser.add_argument("--y-path", required=True, help="Path to the target values .npy file.")
    parser.add_argument(
        "--model-output",
        default="models/senet_hot5_best.h5",
        help="Where to save the trained Keras model.",
    )
    parser.add_argument(
        "--scaler-prefix",
        default="models/senet_hot5",
        help="Prefix for saving MinMaxScalers (set empty to skip saving scalers).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib plots (useful for headless environments).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds()

    X_raw = np.load(args.x_path)
    y_raw = np.load(args.y_path)

    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)
    elif y_raw.ndim == 2 and y_raw.shape[1] != 1:
        raise ValueError(f"y shape가 이상함: {y_raw.shape}")

    metrics, y_train_real, y_train_pred, y_test_real, y_test_pred = train_and_evaluate(
        X_raw, y_raw, model_output=args.model_output, scaler_prefix=args.scaler_prefix
    )

    print("\n===== 최종 모델 성능 (Train / Test, 75:25) =====")
    print(f"Train R² : {metrics['train_r2']:.4f}")
    print(f"Test  R² : {metrics['test_r2']:.4f}")
    print(f"Train NSE: {metrics['train_nse']:.4f}")
    print(f"Test  NSE: {metrics['test_nse']:.4f}")
    print(f"Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test  RMSE: {metrics['test_rmse']:.4f}")

    if not args.no_plot:
        plot_results(y_train_real, y_train_pred, y_test_real, y_test_pred)


if __name__ == "__main__":
    main()
