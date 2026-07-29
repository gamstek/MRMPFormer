# utils/plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from scipy.ndimage import gaussian_filter
import bisect

def smooth_xic(inten, rt, sigma):
    if sigma == 0:
        smooth_intensity = inten
        smooth_rt = rt
    else:
        smooth_intensity = gaussian_filter(inten, sigma=sigma)
        smooth_rt = rt
    return smooth_rt, smooth_intensity

def calc_coordinate(info, intensity, rt, k, windows_size=2):
    t_rt = info[k][2]
    lrt = t_rt - windows_size / 2 if t_rt - windows_size / 2 > 0 else 0
    rrt = t_rt + windows_size / 2 if t_rt + windows_size / 2 < rt[-1] else rt[-1]
    lindex = bisect.bisect_left(rt, lrt)
    rindex = bisect.bisect_right(rt, rrt)
    if rindex - lindex >= 0:
        calc_intensity = intensity[lindex:rindex]
        calc_rt = rt[lindex:rindex]
    return calc_intensity, calc_rt
def plot_xic(rt, intensity, name, folder_path):
    # 创建图像对象
    fig = Figure(figsize=(4, 3), dpi=100)  # 4英寸 × 3英寸，DPI=100 -> 400×300 像素
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(rt, intensity, color='blue', linewidth=1.5)
    # 移除坐标轴、边框和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 调整布局以填满画布
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存为 JPEG 图像
    file_path = os.path.join(folder_path, f"{name}.jpeg")
    canvas.print_jpeg(file_path)

    # 打印日志信息（符合规范）
    print(f"[INFO] Generated ROI image: {file_path}")
    print(f"       Retention Time Range: {rt.min():.2f} - {rt.max():.2f} min")
    print(f"       Intensity Range: {intensity.min():.2e} - {intensity.max():.2e}")

    # 新增：保存输入 CNN 的图像（RGB 格式，归一化到 [0, 1]）
    # 将图像转换为 NumPy 数组
    canvas.draw()
    buf = canvas.tostring_rgb()  # 获取 RGB 数据
    image_array = np.frombuffer(buf, dtype=np.uint8).reshape((300, 400, 3))  # (H, W, C)

    # 归一化到 [0, 1] 并保存为 .npy 文件
    normalized_image = image_array.astype(np.float32) / 255.0
    cnn_input_path = os.path.join(folder_path, f"{name}_cnn_input.npy")
    np.save(cnn_input_path, normalized_image)

    # 打印 CNN 输入图像信息
    print(f"[INFO] Saved CNN input image: {cnn_input_path}")
    print(f"       Shape: {normalized_image.shape}")  # 应为 (300, 400, 3)
    print(f"       Value Range: [{normalized_image.min():.4f}, {normalized_image.max():.4f}]")

    plt.close(fig)