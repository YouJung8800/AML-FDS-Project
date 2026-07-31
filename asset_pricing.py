import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import subprocess
import datetime

def calculate_oos_r2(y_true, y_pred):
    """
    논문(Gu, Kelly, and Xiu 2020)의 Out-of-Sample R^2 계산 공식.
    평균을 빼지 않고 단순 제곱합을 사용합니다.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    return 1 - (numerator / denominator)

def main():
    print("=== [1/4] 데이터 셋 준비 중 ===")
    # 논문 환경(주식 특성 94개, 거시변수 8개 등)과 유사하게 가상의 패널 데이터 생성
    np.random.seed(42)
    n_samples = 15000
    n_features = 100 

    X = np.random.randn(n_samples, n_features)
    # 비선형 상호작용(Interaction)이 포함된 가상의 수익률 데이터
    y = 0.05 * X[:, 0] + 0.1 * (X[:, 1] * X[:, 2]) - 0.02 * (X[:, 3] ** 2) + np.random.randn(n_samples) * 0.1

    # 시간순 분할 (Train, Validation, Test)
    train_idx, val_idx, test_idx = 9000, 12000, 15000
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:test_idx], y[val_idx:test_idx]

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}

    print("=== [2/4] Random Forest 모델 학습 중 ===")
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results['Random Forest'] = calculate_oos_r2(y_test, y_pred_rf)

    print("=== [3/4] Neural Network (NN3) 모델 학습 중 ===")
    # 논문의 NN3 아키텍처 (32, 16, 8 뉴런 + ReLU)
    nn3 = Sequential([
        Dense(32, activation='relu', input_dim=n_features),
        BatchNormalization(),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dense(1) # 수익률 예측 출력층
    ])

    nn3.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    
    # 조기 종료 (Early Stopping) 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    nn3.fit(X_train_scaled, y_train, 
            validation_data=(X_val_scaled, y_val),
            epochs=100, batch_size=256, callbacks=[early_stopping], verbose=0)

    y_pred_nn3 = nn3.predict(X_test_scaled, verbose=0).flatten()
    results['NN3'] = calculate_oos_r2(y_test, y_pred_nn3)

    print("\n=== Out-of-Sample R^2 예측 결과 ===")
    for model, r2 in results.items():
        print(f"{model}: {r2 * 100:.4f}%")

    print("\n=== [4/4] 깃허브 자동 연동 진행 ===")
    try:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Auto update: ML Asset Pricing code & results ({current_time})"
        
        subprocess.run(["git", "add", "asset_pricing.py"], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ 깃허브에 성공적으로 코드가 업로드되었습니다!")
    except subprocess.CalledProcessError:
        print("⚠️ 깃허브 업로드 실패: git 저장소 초기화(git init)가 되어있는지, 원격 저장소가 연결되어 있는지 확인해주세요.")

if __name__ == "__main__":
    main()
