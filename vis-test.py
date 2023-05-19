import numpy as np
import matplotlib.pyplot as plt

# 创建初始的4x6 ndarray
ndarray = np.random.rand(4, 6)

# 创建图形和子图
fig, ax = plt.subplots()

# 显示初始的ndarray
im = ax.imshow(ndarray, cmap='viridis')

# 添加颜色条
cbar = fig.colorbar(im, ax=ax)

# 设置图形的标题


# 更新函数
def update_figure():
    # 更新ndarray数据
    ndarray = np.random.rand(4, 6)

    # 更新图形的数据
    im.set_array(ndarray)

    # 更新颜色条的范围
    im.set_clim(vmin=ndarray.min(), vmax=ndarray.max())

    # 绘制图形
    plt.draw()

    # 设置更新频率（以毫秒为单位）
    plt.pause(0.1)

# 不断更新图形
while True:
    update_figure()
