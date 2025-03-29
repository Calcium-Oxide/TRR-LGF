import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 示例数据
data = [
    (0, 0.8989),
    (5, 0.911),
    (10, 0.8953),
    (15, 0.8977),
    (20, 0.9104),
    (25, 0.8946),
    (30, 0.8752),
    (50, 0.8650),
]

# 将数据分离为两个列表
mu = np.array([point[0] for point in data])
acc = np.array([point[1] for point in data])

# 使用样条插值生成光滑曲线
mu_smooth = np.linspace(mu.min(), mu.max(), 300)
acc_smooth = make_interp_spline(mu, acc)(mu_smooth)

# 绘制光滑曲线图
plt.figure(figsize=(10, 10))
plt.plot(mu_smooth, acc_smooth)
plt.title('The Regularization Coefficient $\mu$ 的影响', fontsize=25, fontname='Times New Roman')
plt.xlabel('$\mu$', fontsize=20, fontname='Times New Roman')
plt.ylabel('ACC', fontsize=20, fontname='Times New Roman')
plt.grid()
plt.xticks(fontname='Times New Roman', fontsize=18)
plt.yticks(fontname='Times New Roman', fontsize=18)
plt.ylim(0, 1)  # 设置纵坐标范围
plt.show()