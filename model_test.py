import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor

# 1. 가상 데이터 생성
np.random.seed(42)
n_samples = 200
Z = np.random.randn(n_samples, 1)
true_theta = 2.0
R = true_theta * Z.ravel() + np.random.normal(0, 1, n_samples)

# 이상치 주입
outlier_indices = [10, 45, 100, 150]
R[outlier_indices] += np.random.choice([-15, 15], size=len(outlier_indices))

# 2. 모델 학습
ols_model = LinearRegression()
ols_model.fit(Z, R)
R_pred_ols = ols_model.predict(Z)

huber_model = HuberRegressor(epsilon=1.35)
huber_model.fit(Z, R)
R_pred_huber = huber_model.predict(Z)

# 3. 결과 텍스트 출력
print("--- [모델 추정 파라미터(Theta) 비교] ---")
print(f"실제 설정값 (True Theta): {true_theta}")
print(f"OLS 추정치: {ols_model.coef_[0]:.4f} (이상치에 의해 왜곡됨)")
print(f"Huber 추정치: {huber_model.coef_[0]:.4f} (이상치에 강건하게 방어함)")

# 4. 시각화 및 파일 저장
plt.figure(figsize=(10, 6))
plt.scatter(Z, R, color='gray', alpha=0.7, label='Data (with Outliers)')
plt.plot(Z, true_theta * Z, color='black', linestyle='--', label='True Relationship')
plt.plot(Z, R_pred_ols, color='red', label='OLS Fit (Simple Linear)')
plt.plot(Z, R_pred_huber, color='blue', label='Huber Fit (Robust)')
plt.xlabel('Predictor (Z)')
plt.ylabel('Return (R+1)')
plt.title('Comparison of OLS and Huber Regression under Outliers')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('result.png')
print("\n[알림] 그래프가 현재 폴더에 'result.png' 파일로도 저장되었습니다.")
plt.show()

