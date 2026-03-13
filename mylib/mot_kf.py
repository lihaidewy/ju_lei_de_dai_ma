import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.optimize import linear_sum_assignment

INF = 1e9


def cv_F(dt: float) -> np.ndarray:
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]], dtype=float)


def H_pos() -> np.ndarray:
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)


def Q_white_acc(dt: float, sigma_a: float) -> np.ndarray:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    q = sigma_a * sigma_a
    return q * np.array([[dt4 / 4, 0,       dt3 / 2, 0],
                         [0,       dt4 / 4, 0,       dt3 / 2],
                         [dt3 / 2, 0,       dt2,     0],
                         [0,       dt3 / 2, 0,       dt2]], dtype=float)


def radial_speed_pred(x_state: np.ndarray) -> float:
    """预测径向速度：v_r = los · vel"""
    px, py, vx, vy = x_state
    pos = np.array([px, py], dtype=float)
    vel = np.array([vx, vy], dtype=float)
    n = np.linalg.norm(pos) + 1e-6
    los = pos / n
    return float(los @ vel)


@dataclass
class Measurement:
    frame: int
    z: np.ndarray          # (2,) rear-center
    v_median: float = np.nan
    width: float = np.nan
    n_points: int = 0
    snr_mean: float = np.nan


@dataclass
class Track:
    tid: int
    x: np.ndarray          # (4,) [px,py,vx,vy]
    P: np.ndarray          # (4,4)
    hits: int = 1
    age: int = 1
    missed: int = 0
    confirmed: bool = False

    recent_hits: list = field(default_factory=list)   # 最近 N 帧命中(1/0)
    trace: list = field(default_factory=list)         # [(px,py), ...]

    # ===== 观测属性缓存 =====
    last_v_median: float = np.nan
    last_width: float = np.nan
    last_n_points: int = 0
    last_snr_mean: float = np.nan

    # ===== 语义与置信度 =====
    tail_visible: bool = True   # True: 面向雷达侧更像尾部；False: 更像车头
    confidence: float = 0.0     # 0~1


class MOTKF:
    """
    KF(CV) + gating + Hungarian + M/N track management
    cost = Mahalanobis(d2) + w_v*|Δv_r| + w_w*|Δwidth|
    """

    def __init__(self,
                 dt=0.1,
                 sigma_a=3.0,
                 sigma_z=1.0,
                 gate_chi2=9.21,     # 2D pos, 99%
                 w_v=1.0,
                 w_w=0.5,            # ✅ 新增：width 一致性权重
                 M=3, N=5,           # M/N confirm
                 max_missed=10,
                 min_birth_points=2):
        self.dt = dt
        self.F = cv_F(dt)
        self.H = H_pos()
        self.Q = Q_white_acc(dt, sigma_a)
        self.R = (sigma_z ** 2) * np.eye(2)

        self.gate = gate_chi2
        self.w_v = w_v
        self.w_w = w_w

        self.M, self.N = M, N
        self.max_missed = max_missed
        self.min_birth_points = min_birth_points

        self.next_id = 1
        self.tracks: List[Track] = []

    # ----------------- helper: semantics & confidence -----------------

    def _tail_visible_from_motion(self, x_state: np.ndarray) -> bool:
        """
        True：目标沿 LOS 远离雷达（h·los>0）=> 面向雷达侧更像尾部
        False：目标沿 LOS 靠近雷达（h·los<0）=> 面向雷达侧更像车头
        """
        px, py, vx, vy = x_state
        pos = np.array([px, py], dtype=float)
        vel = np.array([vx, vy], dtype=float)

        pnorm = np.linalg.norm(pos) + 1e-6
        vnorm = np.linalg.norm(vel) + 1e-6

        los = pos / pnorm
        h = vel / vnorm

        return bool((h @ los) > 0.0)

    def _track_confidence(self, tr: Track) -> float:
        """0~1 置信度（规则法）：命中率 + confirmed + missed + 协方差 + 观测质量"""
        hit_ratio = tr.hits / max(tr.age, 1)

        c = 0.15 + 0.55 * hit_ratio
        if tr.confirmed:
            c += 0.2
        c -= 0.05 * min(tr.missed, 5)

        # 位置协方差越小越可靠
        pos_var = float(tr.P[0, 0] + tr.P[1, 1])
        c *= 1.0 / (1.0 + 0.05 * pos_var)

        # 观测质量轻微加分
        if tr.last_n_points >= 3:
            c += 0.05
        if np.isfinite(tr.last_snr_mean) and tr.last_snr_mean > 5:
            c += 0.05

        return float(np.clip(c, 0.0, 1.0))

    def _write_meas_attrs(self, tr: Track, m: Measurement):
        tr.last_v_median = m.v_median
        tr.last_width = m.width
        tr.last_n_points = m.n_points
        tr.last_snr_mean = m.snr_mean

    # ----------------- KF predict / update -----------------

    def _predict(self, tr: Track):
        tr.x = self.F @ tr.x
        tr.P = self.F @ tr.P @ self.F.T + self.Q

        tr.age += 1
        tr.missed += 1

        tr.recent_hits.append(0)
        if len(tr.recent_hits) > self.N:
            tr.recent_hits.pop(0)

        tr.trace.append(tr.x[:2].copy())

        # 预测后也可更新语义/置信度（保守做法：等 update 后再更可靠）
        tr.tail_visible = self._tail_visible_from_motion(tr.x)
        tr.confidence = self._track_confidence(tr)

    def _update(self, tr: Track, m: Measurement):
        z = m.z.reshape(2, 1)
        x = tr.x.reshape(4, 1)
        P = tr.P

        y = z - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)

        x_new = x + K @ y
        P_new = (np.eye(4) - K @ self.H) @ P

        tr.x = x_new.reshape(4,)
        tr.P = P_new

        tr.hits += 1
        tr.missed = 0

        # 命中记录
        if len(tr.recent_hits) == 0:
            tr.recent_hits.append(1)
        else:
            tr.recent_hits[-1] = 1

        # M/N confirm
        if (not tr.confirmed) and (sum(tr.recent_hits) >= self.M):
            tr.confirmed = True

        # 更新 trace 的最后一个点为更新后的状态
        if len(tr.trace) == 0:
            tr.trace.append(tr.x[:2].copy())
        else:
            tr.trace[-1] = tr.x[:2].copy()

        # 写入观测属性 + 更新语义/置信度
        self._write_meas_attrs(tr, m)
        tr.tail_visible = self._tail_visible_from_motion(tr.x)
        tr.confidence = self._track_confidence(tr)

    # ----------------- association cost -----------------

    def _cost(self, tr: Track, m: Measurement) -> float:
        z = m.z.reshape(2, 1)
        x = tr.x.reshape(4, 1)

        y = z - self.H @ x
        S = self.H @ tr.P @ self.H.T + self.R

        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return INF

        d2 = float((y.T @ Sinv @ y).squeeze())
        if d2 > self.gate:
            return INF

        cost = d2

        # 速度一致性项（径向速度）
        if np.isfinite(m.v_median):
            vpred = radial_speed_pred(tr.x)
            cost += self.w_v * abs(m.v_median - vpred)

        # width 一致性项（用 track 上一次 width）
        if np.isfinite(m.width) and np.isfinite(tr.last_width):
            cost += self.w_w * abs(m.width - tr.last_width)

        return cost

    # ----------------- step -----------------

    def step(self, measurements: "List[Measurement]") -> "List[Track]":
        # 1) predict all
        for tr in self.tracks:
            self._predict(tr)

        T = len(self.tracks)
        M = len(measurements)

        matches = []
        unmatched_tr = list(range(T))
        unmatched_meas = list(range(M))

        # 2) associate
        if T > 0 and M > 0:
            C = np.full((T, M), INF, dtype=float)
            for i, tr in enumerate(self.tracks):
                for j, m in enumerate(measurements):
                    C[i, j] = self._cost(tr, m)

            ri, ci = linear_sum_assignment(C)

            unmatched_tr = set(range(T))
            unmatched_meas = set(range(M))

            for r, c in zip(ri, ci):
                if C[r, c] >= INF / 2:
                    continue
                matches.append((r, c))
                unmatched_tr.discard(r)
                unmatched_meas.discard(c)

            unmatched_tr = sorted(list(unmatched_tr))
            unmatched_meas = sorted(list(unmatched_meas))

        # 3) update matched
        for ti, mj in matches:
            self._update(self.tracks[ti], measurements[mj])

        # 4) delete dead tracks
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]

        # 5) birth new tracks from unmatched measurements
        for j in unmatched_meas:
            m = measurements[j]
            if m.n_points < self.min_birth_points:
                continue

            px, py = m.z
            x0 = np.array([px, py, 0.0, 0.0], dtype=float)
            P0 = np.diag([4.0, 4.0, 25.0, 25.0])

            tr = Track(tid=self.next_id, x=x0, P=P0)

            tr.trace.append(tr.x[:2].copy())
            tr.recent_hits.append(1)

            # 写入初始观测属性
            self._write_meas_attrs(tr, m)
            tr.tail_visible = self._tail_visible_from_motion(tr.x)
            tr.confidence = self._track_confidence(tr)

            self.next_id += 1
            self.tracks.append(tr)

        return self.tracks

    # ----------------- vehicle output helper -----------------

    def get_confirmed_vehicles(self, conf_thr: float = 0.5):
        """
        返回 confirmed 且 confidence>=conf_thr 的车辆列表（dict），便于外部直接用。
        """
        vehicles = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            if tr.confidence < conf_thr:
                continue

            px, py, vx, vy = tr.x
            yaw = float(np.arctan2(vy, vx))
            vehicles.append({
                "id": tr.tid,
                "x": float(px), "y": float(py),
                "vx": float(vx), "vy": float(vy),
                "yaw": yaw,
                "speed": float(np.hypot(vx, vy)),
                "tail_visible": bool(tr.tail_visible),
                "confidence": float(tr.confidence),
                "width": float(tr.last_width) if np.isfinite(tr.last_width) else None,
                "snr_mean": float(tr.last_snr_mean) if np.isfinite(tr.last_snr_mean) else None,
                "n_points": int(tr.last_n_points),
                "age": int(tr.age),
                "missed": int(tr.missed),
            })
        return vehicles
