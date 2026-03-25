from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment


BIG_COST = 1e6


class BaseKalmanTrack(object):
    def __init__(
        self,
        track_id,
        center,
        dt=1.0,
        r_pos=1.50,
        init_pos_var=4.0,
        init_vel_var=9.0,
        init_acc_var=16.0,
        use_adaptive_r=False,
        adaptive_r_gain=0.25,
        min_r_scale=0.75,
        max_r_scale=4.0,
        enable_output_ema=False,
        output_ema_alpha=0.65,
    ):
        self.track_id = int(track_id)
        self.dt = float(dt)

        self.age = 1
        self.hit_count = 1
        self.hit_streak = 1
        self.miss_count = 0

        self.is_confirmed = False
        self.is_tentative = True

        self.raw_center = np.asarray(center, dtype=float).reshape(2)
        self.filtered_center = self.raw_center.copy()
        self.output_center = self.raw_center.copy()

        self.base_r_pos = float(r_pos)
        self.use_adaptive_r = bool(use_adaptive_r)
        self.adaptive_r_gain = float(adaptive_r_gain)
        self.min_r_scale = float(min_r_scale)
        self.max_r_scale = float(max_r_scale)

        self.enable_output_ema = bool(enable_output_ema)
        self.output_ema_alpha = float(output_ema_alpha)

        self._init_pos_var = float(init_pos_var)
        self._init_vel_var = float(init_vel_var)
        self._init_acc_var = float(init_acc_var)

        self.state = None
        self.P = None
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self._build_model(center)

    def _build_model(self, center):
        raise NotImplementedError

    @property
    def position(self):
        return self.state[:2].copy()

    @property
    def velocity(self):
        return self.state[2:4].copy()

    def confirm(self):
        self.is_confirmed = True
        self.is_tentative = False

    def on_update_success(self):
        self.hit_count += 1
        self.hit_streak += 1
        self.miss_count = 0
        self.age += 1

    def on_missed(self):
        self.miss_count += 1
        self.hit_streak = 0
        self.age += 1

    def predict(self):
        self.state = self.F.dot(self.state)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        self.filtered_center = self.state[:2].copy()
        self.output_center = self._apply_output_smoother(self.filtered_center)
        return self.output_center.copy()

    def _apply_output_smoother(self, center):
        center = np.asarray(center, dtype=float).reshape(2)
        if not self.enable_output_ema:
            self.output_center = center.copy()
            return self.output_center.copy()

        alpha = self.output_ema_alpha
        self.output_center = alpha * center + (1.0 - alpha) * self.output_center
        return self.output_center.copy()

    def _measurement_prediction(self):
        return self.H.dot(self.state)

    def innovation_cov(self):
        return self.H.dot(self.P).dot(self.H.T) + self.R

    def mahalanobis_distance(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(2)
        residual = z - self._measurement_prediction()
        s = self.innovation_cov()
        try:
            d2 = residual.T.dot(np.linalg.solve(s, residual))
        except np.linalg.LinAlgError:
            d2 = residual.T.dot(np.linalg.pinv(s)).dot(residual)
        return float(np.sqrt(max(d2, 0.0)))

    def euclidean_distance(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(2)
        return float(np.linalg.norm(z - self.position))

    def _maybe_adapt_R(self, residual):
        if not self.use_adaptive_r:
            return
        residual_norm = float(np.linalg.norm(residual))
        scale = 1.0 + self.adaptive_r_gain * residual_norm
        scale = min(max(scale, self.min_r_scale), self.max_r_scale)
        self.R = np.eye(2, dtype=float) * (self.base_r_pos * scale)

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(2)

        y = z - self.H.dot(self.state)
        self._maybe_adapt_R(y)

        s = self.H.dot(self.P).dot(self.H.T) + self.R
        pht = self.P.dot(self.H.T)
        try:
            k = np.linalg.solve(s, pht.T).T
        except np.linalg.LinAlgError:
            k = pht.dot(np.linalg.pinv(s))

        self.state = self.state + k.dot(y)

        i = np.eye(self.P.shape[0], dtype=float)
        kh = k.dot(self.H)
        self.P = (i - kh).dot(self.P).dot((i - kh).T) + k.dot(self.R).dot(k.T)

        self.filtered_center = self.state[:2].copy()
        self.output_center = self._apply_output_smoother(self.filtered_center)

        self.raw_center = z.copy()
        self.on_update_success()
        return self.output_center.copy()

    def mark_missed(self):
        self.on_missed()


class KalmanTrackCV(BaseKalmanTrack):
    def __init__(self, track_id, center, dt=1.0, q_pos=0.30, q_vel=0.50, **kwargs):
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        super(KalmanTrackCV, self).__init__(track_id=track_id, center=center, dt=dt, **kwargs)

    def _build_model(self, center):
        x, y = np.asarray(center, dtype=float).reshape(2)
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.diag([
            self._init_pos_var,
            self._init_pos_var,
            self._init_vel_var,
            self._init_vel_var,
        ]).astype(float)

        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        self.Q = np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel]).astype(float)
        self.R = np.eye(2, dtype=float) * float(self.base_r_pos)


class KalmanTrackCA(BaseKalmanTrack):
    def __init__(self, track_id, center, dt=1.0, q_pos=0.30, q_vel=0.50, q_acc=0.20, **kwargs):
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.q_acc = float(q_acc)
        super(KalmanTrackCA, self).__init__(track_id=track_id, center=center, dt=dt, **kwargs)

    def _build_model(self, center):
        x, y = np.asarray(center, dtype=float).reshape(2)
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([
            self._init_pos_var,
            self._init_pos_var,
            self._init_vel_var,
            self._init_vel_var,
            self._init_acc_var,
            self._init_acc_var,
        ]).astype(float)

        dt = self.dt
        dt2 = 0.5 * dt * dt

        self.F = np.array([
            [1.0, 0.0, dt, 0.0, dt2, 0.0],
            [0.0, 1.0, 0.0, dt, 0.0, dt2],
            [0.0, 0.0, 1.0, 0.0, dt, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=float)

        self.Q = np.diag([
            self.q_pos,
            self.q_pos,
            self.q_vel,
            self.q_vel,
            self.q_acc,
            self.q_acc,
        ]).astype(float)

        self.R = np.eye(2, dtype=float) * float(self.base_r_pos)


class OnlineTrackerManager(object):
    def __init__(
        self,
        method="cv",
        assoc_metric="euclidean",
        assoc_dist_thr=8.0,
        assoc_mahal_thr=3.5,
        max_misses=5,
        min_hits_to_confirm=2,
        max_tentative_misses=1,
        dt=1.0,
        q_pos=0.30,
        q_vel=0.50,
        q_acc=0.20,
        r_pos=1.50,
        init_pos_var=4.0,
        init_vel_var=9.0,
        init_acc_var=16.0,
        use_adaptive_r=False,
        adaptive_r_gain=0.25,
        min_r_scale=0.75,
        max_r_scale=4.0,
        enable_output_ema=False,
        output_ema_alpha=0.65,
    ):
        self.method = str(method).lower()
        self.assoc_metric = str(assoc_metric).lower()
        self.assoc_dist_thr = float(assoc_dist_thr)
        self.assoc_mahal_thr = float(assoc_mahal_thr)
        self.max_misses = int(max_misses)
        self.min_hits_to_confirm = int(min_hits_to_confirm)
        self.max_tentative_misses = int(max_tentative_misses)

        self.dt = float(dt)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.q_acc = float(q_acc)
        self.r_pos = float(r_pos)
        self.init_pos_var = float(init_pos_var)
        self.init_vel_var = float(init_vel_var)
        self.init_acc_var = float(init_acc_var)

        self.use_adaptive_r = bool(use_adaptive_r)
        self.adaptive_r_gain = float(adaptive_r_gain)
        self.min_r_scale = float(min_r_scale)
        self.max_r_scale = float(max_r_scale)

        self.enable_output_ema = bool(enable_output_ema)
        self.output_ema_alpha = float(output_ema_alpha)

        self.next_track_id = 1
        self.tracks = {}  # type: Dict[int, BaseKalmanTrack]

    def _make_track(self, track_id, center):
        common_kwargs = dict(
            dt=self.dt,
            r_pos=self.r_pos,
            init_pos_var=self.init_pos_var,
            init_vel_var=self.init_vel_var,
            init_acc_var=self.init_acc_var,
            use_adaptive_r=self.use_adaptive_r or (self.method == "cv_robust"),
            adaptive_r_gain=self.adaptive_r_gain,
            min_r_scale=self.min_r_scale,
            max_r_scale=self.max_r_scale,
            enable_output_ema=self.enable_output_ema,
            output_ema_alpha=self.output_ema_alpha,
        )

        if self.method == "ca":
            return KalmanTrackCA(
                track_id=track_id,
                center=center,
                q_pos=self.q_pos,
                q_vel=self.q_vel,
                q_acc=self.q_acc,
                **common_kwargs
            )

        return KalmanTrackCV(
            track_id=track_id,
            center=center,
            q_pos=self.q_pos,
            q_vel=self.q_vel,
            **common_kwargs
        )

    def _new_track(self, center):
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = self._make_track(tid, center)
        return tid

    def _distance(self, track, center):
        if self.assoc_metric == "mahalanobis":
            return track.mahalanobis_distance(center)
        return track.euclidean_distance(center)

    def _gate_threshold(self):
        if self.assoc_metric == "mahalanobis":
            return self.assoc_mahal_thr
        return self.assoc_dist_thr

    def _maybe_confirm_track(self, track):
        if (not track.is_confirmed) and track.hit_streak >= self.min_hits_to_confirm:
            track.confirm()

    def _should_output_track(self, track):
        return bool(track.is_confirmed)

    def _should_delete_track(self, track):
        if track.is_tentative:
            return track.miss_count > self.max_tentative_misses
        return track.miss_count > self.max_misses

    def _build_cost_matrix(self, cluster_centers):
        track_ids = list(self.tracks.keys())
        cluster_ids = list(cluster_centers.keys())
        cost = np.zeros((len(track_ids), len(cluster_ids)), dtype=float)

        for tid in track_ids:
            self.tracks[tid].predict()

        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            for j, cid in enumerate(cluster_ids):
                center = cluster_centers[cid]
                cost[i, j] = self._distance(track, center)

        return cost, track_ids, cluster_ids

    def get_active_track_states(self):
        out = {}
        for tid, track in self.tracks.items():
            out[int(tid)] = {
                "track_id": int(tid),
                "age": int(getattr(track, "age", 0)),
                "hit_count": int(getattr(track, "hit_count", 0)),
                "hit_streak": int(getattr(track, "hit_streak", 0)),
                "miss_count": int(getattr(track, "miss_count", 0)),
                "is_confirmed": bool(getattr(track, "is_confirmed", False)),
                "is_tentative": bool(getattr(track, "is_tentative", True)),
                "raw_x": float(track.raw_center[0]),
                "raw_y": float(track.raw_center[1]),
                "filtered_x": float(track.filtered_center[0]),
                "filtered_y": float(track.filtered_center[1]),
                "output_x": float(track.output_center[0]),
                "output_y": float(track.output_center[1]),
            }
        return out

    def step(self, cluster_centers):
        cluster_centers = {
            int(cid): np.asarray(center, dtype=float).reshape(2)
            for cid, center in cluster_centers.items()
        }

        filtered_centers = {}
        track_assignments = {}
        raw_centers = {}

        if len(cluster_centers) == 0:
            to_delete = []
            for tid, track in self.tracks.items():
                track.predict()
                track.mark_missed()
                if self._should_delete_track(track):
                    to_delete.append(tid)

            for tid in to_delete:
                del self.tracks[tid]

            return filtered_centers, track_assignments, raw_centers

        if len(self.tracks) == 0:
            for _, center in cluster_centers.items():
                self._new_track(center)
            return filtered_centers, track_assignments, raw_centers

        cost, track_ids, cluster_ids = self._build_cost_matrix(cluster_centers)
        gate_thr = self._gate_threshold()

        gated_cost = cost.copy()
        gated_cost[gated_cost > gate_thr] = BIG_COST
        row_ind, col_ind = linear_sum_assignment(gated_cost)

        matched_tracks = set()
        matched_clusters = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > gate_thr:
                continue

            tid = track_ids[r]
            cid = cluster_ids[c]
            track = self.tracks[tid]

            filtered = track.update(cluster_centers[cid])
            self._maybe_confirm_track(track)

            if self._should_output_track(track):
                filtered_centers[cid] = filtered.copy()
                track_assignments[cid] = tid
                raw_centers[cid] = cluster_centers[cid].copy()

            matched_tracks.add(tid)
            matched_clusters.add(cid)

        for cid in cluster_ids:
            if cid in matched_clusters:
                continue
            center = cluster_centers[cid]
            self._new_track(center)

        to_delete = []
        for tid, track in self.tracks.items():
            if tid in matched_tracks:
                continue
            track.mark_missed()
            if self._should_delete_track(track):
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return filtered_centers, track_assignments, raw_centers
