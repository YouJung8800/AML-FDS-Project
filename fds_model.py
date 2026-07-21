import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ==========================================
# 1. 이전에 만든 교차 어텐션 모델 클래스 (그대로 가져옴)
# ==========================================
class AML_CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(AML_CrossAttention, self).__init__()
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(self, sender_seq, receiver_seq):
        Q = self.W_q(sender_seq)       
        K = self.W_k(receiver_seq)     
        V = self.W_v(receiver_seq)     
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, V)
        return context_vector, attention_weights

# ==========================================
# 2. 가상의 CSV 파일 생성 (테스트용 - 실제 파일이 있으면 이 부분은 삭제)
# ==========================================
if not os.path.exists('financial_data.csv'):
    print("가상의 금융 CSV 데이터를 생성합니다...")
    # 임의의 거래 데이터 100건 생성
    dummy_data = {
        'account_id': ['User_A']*50 + ['User_B']*50,
        'timestamp': pd.date_range(start='2026-07-01', periods=100, freq='H'),
        'amount': np.random.randint(1000, 5000000, 100),
        'ip_address': np.random.choice(['192.168.0.1', '10.0.0.2', '172.16.0.3'], 100),
        'merchant_category': np.random.choice(['Food', 'Crypto', 'Transfer', 'Retail'], 100)
    }
    pd.DataFrame(dummy_data).to_csv('financial_data.csv', index=False)

# ==========================================
# 3. CSV 데이터 로드 및 전처리 (Pandas의 영역)
# ==========================================
print("CSV 데이터를 불러오고 전처리를 시작합니다...")
df = pd.read_csv('financial_data.csv')

# (1) 시간순 정렬 (시계열 데이터에서 가장 중요)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['account_id', 'timestamp'])

# (2) 연속형 변수 스케일링 (금액 정규화)
# 금액(amount)이 너무 크면 모델의 기울기가 폭발하므로 0~1 사이로 압축합니다.
df['amount_scaled'] = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())

# (3) 범주형 변수 인코딩 (문자를 숫자로 변환)
# IP주소와 가맹점 카테고리를 신경망이 인식할 수 있는 정수형 코드로 변환합니다.
df['ip_encoded'] = df['ip_address'].astype('category').cat.codes
df['merchant_encoded'] = df['merchant_category'].astype('category').cat.codes

# 사용할 최종 피처 3개 (실제로는 64개까지 확장 가능)
feature_cols = ['amount_scaled', 'ip_encoded', 'merchant_encoded']
FEATURE_DIM = len(feature_cols) 
SEQ_LENGTH = 30 # 추출할 최대 거래 횟수

# ==========================================
# 4. 시퀀스 데이터 추출 및 Tensor 변환
# ==========================================
def get_user_sequence(dataframe, user_id, seq_len, features):
    """특정 유저의 최근 거래내역을 추출해 텐서로 변환하고 0으로 패딩(Padding)합니다."""
    user_data = dataframe[dataframe['account_id'] == user_id][features].values
    
    # 최근 seq_len(30)개의 거래만 가져옴
    if len(user_data) > seq_len:
        user_data = user_data[-seq_len:]
    
    # 거래 횟수가 30번보다 적으면, 모자란 만큼 0으로 채움 (Zero Padding)
    if len(user_data) < seq_len:
        padding = np.zeros((seq_len - len(user_data), len(features)))
        user_data = np.vstack([padding, user_data])
        
    return torch.tensor(user_data, dtype=torch.float32).unsqueeze(0) # [1, 30, feature_dim] 형태로 반환

# User_A(송금인)와 User_B(수취인)의 거래 내역을 Tensor로 변환
sender_tensor = get_user_sequence(df, 'User_A', SEQ_LENGTH, feature_cols)
receiver_tensor = get_user_sequence(df, 'User_B', SEQ_LENGTH, feature_cols)

print(f"변환 완료! 송금인 Tensor 형태: {sender_tensor.shape}")

# ==========================================
# 5. 모델 추론
# ==========================================
# 3차원으로 피처가 구성되었으므로 모델의 차원(feature_dim)을 3으로 맞춥니다.
model = AML_CrossAttention(feature_dim=FEATURE_DIM)

output_vector, attention_weights = model(sender_tensor, receiver_tensor)

print(f"\n--- 최종 모델 연산 결과 ---")
print(f"결과 벡터 형태: {output_vector.shape}")
print("실제 금융 데이터가 텐서로 변환되어 모델을 성공적으로 통과했습니다.")

