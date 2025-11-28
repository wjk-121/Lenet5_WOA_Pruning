"""
鲸鱼优化算法（WOA），用于搜索 [conv1_ratio, conv2_ratio]。
增加进度输出：每轮打印当前 best_fitness/best_ratio 与 ETA。
"""
from typing import Callable, Dict, List, Tuple, Optional
import random
import math
import time

class WOA:
    def __init__(self, pop_size: int, dim: int,
                 lb: List[float], ub: List[float],
                 iters: int,
                 fitness_fn: Callable[[List[float]], Tuple[float, dict]],
                 seed: int = 123,
                 logger=None,
                 progress: Optional[Callable[[int,int,float,List[float],float,float], None]] = None):
        """
        progress 回调签名: (iter_idx, total_iters, best_fit, best_pos, elapsed_sec, eta_sec)
        如果未提供 progress，则默认使用 logger.info 或 print。
        """
        self.pop_size = pop_size
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.iters = iters
        self.fitness_fn = fitness_fn
        self.logger = logger
        self.progress_cb = progress
        random.seed(seed)

        self.pop = [[random.uniform(lb[d], ub[d]) for d in range(dim)] for _ in range(pop_size)]
        self.fit = [None] * pop_size

        self.best_pos = None
        self.best_fit = -1e18
        self.history = {"best_fitness": []}

    def _clip(self, x: List[float]) -> List[float]:
        return [max(self.lb[d], min(self.ub[d], x[d])) for d in range(self.dim)]

    def _emit_progress(self, it_idx: int, elapsed: float, eta: float):
        # 自带进度条样式（ASCII），兼容日志文件
        bar_len = 30
        frac = it_idx / max(self.iters, 1)
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (f"[WOA] {it_idx:02d}/{self.iters} |{bar}| "
               f"best_fit={self.best_fit:.4f} best_ratio=[{self.best_pos[0]:.3f}, {self.best_pos[1]:.3f}] "
               f"eta={eta:.1f}s")
        if self.progress_cb:
            self.progress_cb(it_idx, self.iters, self.best_fit, self.best_pos, elapsed, eta)
        elif self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def optimize(self) -> Dict:
        # 初始评价
        start_all = time.time()
        for i in range(self.pop_size):
            f, _ = self.fitness_fn(self.pop[i])
            self.fit[i] = f
            if f > self.best_fit:
                self.best_fit, self.best_pos = f, self.pop[i][:]

        if self.logger:
            self.logger.info(f"WOA 初始化完成: best_fitness={self.best_fit:.4f}, "
                             f"best_ratio=[{self.best_pos[0]:.3f}, {self.best_pos[1]:.3f}]")

        # 迭代优化（记录每轮耗时用于 ETA）
        iter_times = []
        for t in range(self.iters):
            t0 = time.time()
            a = 2 - 2 * (t / max(self.iters - 1, 1))  # 从 2 线性到 0
            for i in range(self.pop_size):
                p = random.random()
                if p < 0.5:
                    A = 2 * a * random.random() - a
                    C = 2 * random.random()
                    if abs(A) < 1:  # 向最优解靠拢
                        X_star = self.best_pos
                        D = [abs(C * X_star[d] - self.pop[i][d]) for d in range(self.dim)]
                        X_new = [X_star[d] - A * D[d] for d in range(self.dim)]
                    else:  # 全局搜索
                        rand_idx = random.randrange(self.pop_size)
                        X_rand = self.pop[rand_idx]
                        D = [abs(C * X_rand[d] - self.pop[i][d]) for d in range(self.dim)]
                        X_new = [X_rand[d] - A * D[d] for d in range(self.dim)]
                else:
                    b = 1
                    l = random.uniform(-1, 1)
                    X_star = self.best_pos
                    D_prime = [abs(X_star[d] - self.pop[i][d]) for d in range(self.dim)]
                    X_new = [D_prime[d] * math.exp(b * l) * math.cos(2 * math.pi * l) + X_star[d]
                             for d in range(self.dim)]

                X_new = self._clip(X_new)
                f_new, _ = self.fitness_fn(X_new)

                # 贪婪选择
                if f_new > self.fit[i]:
                    self.pop[i] = X_new
                    self.fit[i] = f_new

                if f_new > self.best_fit:
                    self.best_fit, self.best_pos = f_new, X_new

            self.history["best_fitness"].append(self.best_fit)

            # 进度输出（本轮时间与 ETA）
            t1 = time.time()
            iter_times.append(t1 - t0)
            avg = sum(iter_times) / len(iter_times)
            remain = max(self.iters - (t + 1), 0)
            eta = avg * remain
            self._emit_progress(t + 1, elapsed=(t1 - start_all), eta=eta)

        return {
            "best_position": self.best_pos,
            "best_fitness": self.best_fit,
            "history": self.history
        }