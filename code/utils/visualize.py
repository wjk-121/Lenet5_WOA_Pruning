# 图表绘制：中文字体支持、保存函数、常用图。
import os
from typing import List, Dict
import matplotlib
import matplotlib.pyplot as plt

def _setup_zh_font():
    # 优先尝试 Windows 下常见中文字体；失败则回退默认
    for f in ["SimHei", "Microsoft YaHei", "MSYH", "Noto Sans CJK SC"]:
        try:
            matplotlib.rcParams["font.sans-serif"] = [f]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue

def save_figure(fig, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_accuracy_compare(methods: List[str], accs: List[float], out_path: str, title: str = "准确率对比"):
    _setup_zh_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, accs, color=["#4e79a7"]*len(methods))
    ax.set_ylabel("准确率")
    ax.set_title(title)
    for i, v in enumerate(accs):
        ax.text(i, v, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=9)
    save_figure(fig, out_path)

def plot_params_compare(methods: List[str], params: List[int], out_path: str, title: str = "参数量对比"):
    _setup_zh_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, params, color=["#59a14f"]*len(methods))
    ax.set_ylabel("参数量")
    ax.set_title(title)
    for i, v in enumerate(params):
        ax.text(i, v, f"{v/1e3:.1f}K", ha="center", va="bottom", fontsize=9)
    save_figure(fig, out_path)

def plot_speed_compare(methods: List[str], ips: List[float], out_path: str, title: str = "推理速度对比"):
    _setup_zh_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, ips, color=["#e15759"]*len(methods))
    ax.set_ylabel("图片/秒")
    ax.set_title(title)
    for i, v in enumerate(ips):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    save_figure(fig, out_path)

def plot_woa_convergence(history: Dict[str, list], out_path: str, title: str = "WOA 收敛曲线"):
    _setup_zh_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["best_fitness"], label="最佳适应值", color="#4e79a7")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("适应值")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, out_path)
