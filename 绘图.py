# 我将使用matplotlib绘制一个清晰的模型结构框架图
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置画布大小和背景
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# 绘制模块的矩形框
module_colors = ['#a8dadc', '#457b9d', '#f4a261']

# Spatio-Temporal Sampling Module
ax.add_patch(patches.FancyBboxPatch((0.02, 0.5), 0.25, 0.4, boxstyle="round,pad=0.02", color=module_colors[0], alpha=0.7))
ax.text(0.145, 0.85, 'Spatio-Temporal\nSampling Module', fontsize=12, ha='center', va='center', weight='bold')

# Multi-View Trajectory Encoder
ax.add_patch(patches.FancyBboxPatch((0.37, 0.3), 0.28, 0.6, boxstyle="round,pad=0.02", color=module_colors[1], alpha=0.7))
ax.text(0.51, 0.88, 'Multi-View Trajectory Encoder', fontsize=12, ha='center', va='center', weight='bold', color='white')

# GPS view
ax.add_patch(patches.FancyBboxPatch((0.39, 0.72), 0.24, 0.15, boxstyle="round,pad=0.01", color='white', alpha=0.9))
ax.text(0.51, 0.795, 'GPS (TDM)\n[Mamba-2]', fontsize=10, ha='center', va='center')

# Route view
ax.add_patch(patches.FancyBboxPatch((0.39, 0.53), 0.24, 0.15, boxstyle="round,pad=0.01", color='white', alpha=0.9))
ax.text(0.51, 0.605, 'Route\n[Transformer]', fontsize=10, ha='center', va='center')

# POI view
ax.add_patch(patches.FancyBboxPatch((0.39, 0.34), 0.24, 0.15, boxstyle="round,pad=0.01", color='white', alpha=0.9))
ax.text(0.51, 0.415, 'POI\n[Transformer]', fontsize=10, ha='center', va='center')

# Spatio-Temporal Meta-Learning Process
ax.add_patch(patches.FancyBboxPatch((0.73, 0.3), 0.25, 0.6, boxstyle="round,pad=0.02", color=module_colors[2], alpha=0.7))
ax.text(0.855, 0.88, 'Spatio-Temporal\nMeta-Learning Process', fontsize=12, ha='center', va='center', weight='bold')

# 内循环和外循环文本
ax.text(0.855, 0.68, 'Inner Loop:\nParameter Update', fontsize=10, ha='center', va='center')
ax.text(0.855, 0.48, 'Outer Loop:\nGlobal Optimization', fontsize=10, ha='center', va='center')
ax.text(0.855, 0.35, 'Contrastive Loss', fontsize=10, ha='center', va='center')

# 绘制箭头
ax.annotate('', xy=(0.37, 0.7), xytext=(0.27, 0.7), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.73, 0.6), xytext=(0.65, 0.6), arrowprops=dict(arrowstyle='->', lw=2))

# 数据输入和输出文本
ax.text(0.02, 0.45, 'Trajectory Dataset', fontsize=10, ha='left', va='center', rotation=90)
ax.annotate('', xy=(0.145, 0.5), xytext=(0.145, 0.47), arrowprops=dict(arrowstyle='->', lw=1.5))

ax.text(0.98, 0.6, 'Optimized\nTrajectory Representation', fontsize=10, ha='right', va='center', rotation=90)
ax.annotate('', xy=(0.855, 0.9), xytext=(0.855, 0.93), arrowprops=dict(arrowstyle='->', lw=1.5))

# 标题
plt.title('STMetaT Framework: Spatio-Temporal Meta-Learning for Trajectory Representation', fontsize=14, weight='bold')

plt.tight_layout()
plt.show()
