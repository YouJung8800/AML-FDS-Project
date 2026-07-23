import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
plt.rc("font", family="AppleGothic")
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

ax1 = fig.add_subplot(gs[0], projection='3d')
np.random.seed(42)
x_norm, y_norm, z_norm = np.random.normal(0, 1, 200), np.random.normal(0, 1, 200), np.random.normal(0, 1, 200)
x_anom, y_anom, z_anom = np.random.normal(3, 0.5, 40), np.random.normal(3, 0.5, 40), np.random.normal(-2, 0.5, 40)
ax1.scatter(x_norm, y_norm, z_norm, c='#3498db', alpha=0.2, s=15)
ax1.scatter(x_anom, y_anom, z_anom, c='#e74c3c', alpha=0.9, s=40, edgecolors='k', label='AML Hub')
ax1.set_title('Unconstrained 3D Latent Space', weight='bold', fontsize=11)

ax2 = fig.add_subplot(gs[1])
methods = ['Rule-based', 'CroCo 3D']
conversion = [4.2, 13.5]
ax2.bar(methods, conversion, color=['#95a5a6', '#2ecc71'], width=0.4)
ax2.set_title('FeeBiz Conversion Rate (%)', weight='bold', fontsize=11)
ax2.set_ylim(0, 16)

ax3 = fig.add_subplot(gs[2])
ax3.plot([0, 0.2, 1], [0, 0.8, 1], linestyle='--', color='#7f8c8d', label='Legacy GNN')
ax3.plot([0, 0.05, 1], [0, 0.85, 0.98], linewidth=2.5, color='#9b59b6', label='PanSt3R (25ms)')
ax3.set_title('Real-time AML Detection', weight='bold', fontsize=11)
ax3.legend(loc='lower right')

plt.tight_layout()
plt.savefig('ultimate_aml_fds_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ 대시보드 이미지 생성 완료!")
