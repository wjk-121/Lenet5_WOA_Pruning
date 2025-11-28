"""
日志到文件 + 控制台；控制台彩色分级（INFO=白、WARNING=黄、ERROR=红）。
文件日志不带颜色，便于查阅与比对。
"""
import logging
import os
import sys
from datetime import datetime

# ANSI 颜色码
RESET = "\033[0m"
COLORS = {
    logging.DEBUG:   "\033[36m",  # 青色
    logging.INFO:    "\033[97m",  # 白色
    logging.WARNING: "\033[33m",  # 黄色
    logging.ERROR:   "\033[31m",  # 红色
    logging.CRITICAL:"\033[91m",  # 亮红
}

def _init_console_color() -> bool:
    """
    尝试启用控制台颜色：
    1) 优先用 colorama（若已安装）
    2) Windows 下尝试开启 ANSI 虚拟终端序列
    3) 失败则返回 False（降级为无色）
    """
    # 1) colorama 优先（Windows/CMD 最稳）
    try:
        import colorama
        colorama.just_fix_windows_console()
        return True
    except Exception:
        pass

    # 2) Windows 尝试开启 ANSI
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                if kernel32.SetConsoleMode(handle, new_mode):
                    return True
        except Exception:
            pass

    # 3) 类 Unix 终端通常支持 ANSI，默认返回 True
    if os.name != "nt":
        return True

    print("警告：控制台颜色未启用，日志输出可能不带颜色。")
    return False

class ColorFormatter(logging.Formatter):
    """给控制台输出加颜色；文件输出仍用普通 Formatter。"""
    def __init__(self, fmt: str, datefmt: str = None, enable_color: bool = True):
        super().__init__(fmt, datefmt)
        self.enable_color = enable_color and _init_console_color()

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self.enable_color:
            color = COLORS.get(record.levelno)
            if color:
                msg = f"{color}{msg}{RESET}"
        return msg

def setup_logger(name: str, log_dir: str, filename_prefix: str = "run"):
    """
    统一日志：文件 + 控制台。
    - 控制台输出到 stdout，并按级别着色（INFO 白、WARNING 黄、ERROR 红）
    - 文件输出不着色（UTF-8）
    - 捕获 warnings 到相同输出通道
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{filename_prefix}_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    # 文件处理器（不带颜色）
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_fmt)

    # 控制台处理器（带颜色）-> stdout
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    color_fmt = ColorFormatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        enable_color=True
    )
    ch.setFormatter(color_fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # 捕获 warnings 到同样的处理器（避免走 stderr 变红）
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.setLevel(logging.WARNING)
    warn_logger.handlers.clear()
    warn_logger.propagate = False
    warn_logger.addHandler(fh)
    warn_logger.addHandler(ch)

    logger.info(f"日志文件: {log_path}")
    return logger, log_path