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
        track_vel_ema_alpha=0.6,

        # ------------------------------------------------------------
        # Step 2: quality-aware R
        # 当前先只使用：
        #   1) num_points   (主导项)
        #   2) vr_std       (辅助项)
        # ------------------------------------------------------------
        use_quality_aware_r=False,
        quality_r_min_scale=0.75,
        quality_r_max_scale=4.0,

        quality_singleton_penalty=2.20,
        quality_two_points_penalty=1.50,
        quality_three_points_penalty=1.15,
        quality_many_points_reward=0.90,
        quality_many_points_thr=4,

        quality_ref_vr_std=0.80,
        quality_high_vr_std_penalty=1.20,
        quality_low_vr_std_thr=0.20,
        quality_low_vr_std_reward=0.95,
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

        self.radial_speed = np.nan

        self.base_r_pos = float(r_pos)
        self.use_adaptive_r = bool(use_adaptive_r)
        self.adaptive_r_gain = float(adaptive_r_gain)
        self.min_r_scale = float(min_r_scale)
        self.max_r_scale = float(max_r_scale)

        self.enable_output_ema = bool(enable_output_ema)
        self.output_ema_alpha = float(output_ema_alpha)
        self.track_vel_ema_alpha = float(track_vel_ema_alpha)

        self._init_pos_var = float(init_pos_var)
        self._init_vel_var = float(init_vel_var)
        self._init_acc_var = float(init_acc_var)

        # ------------------------------------------------------------
        # quality-aware R config
        # ------------------------------------------------------------
        self.use_quality_aware_r = bool(use_quality_aware_r)
        self.quality_r_min_scale = float(quality_r_min_scale)
        self.quality_r_max_scale = float(quality_r_max_scale)

        self.quality_singleton_penalty = float(quality_singleton_penalty)
        self.quality_two_points_penalty = float(quality_two_points_penalty)
        self.quality_three_points_penalty = float(quality_three_points_penalty)
        self.quality_many_points_reward = float(quality_many_points_reward)
        self.quality_many_points_thr = int(quality_many_points_thr)

        self.quality_ref_vr_std = float(quality_ref_vr_std)
        self.quality_high_vr_std_penalty = float(quality_high_vr_std_penalty)
        self.quality_low_vr_std_thr = float(quality_low_vr_std_thr)
        self.quality_low_vr_std_reward = float(quality_low_vr_std_reward)

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
        """
        原始 residual-aware adaptive R 逻辑保留。
        注意：如果 use_adaptive_r=False，则这里不改 R。
        """
        if not self.use_adaptive_r:
            return

        residual_norm = float(np.linalg.norm(residual))
        scale = 1.0 + self.adaptive_r_gain * residual_norm
        scale = min(max(scale, self.min_r_scale), self.max_r_scale)
        self.R = np.eye(2, dtype=float) * (self.base_r_pos * scale)

    def _compute_quality_r_scale(self, cluster_meta):
        """
        当前只用 num_points + vr_std。
        其中 num_points 是主导项，vr_std 只是辅助修正。
        """
        if (not self.use_quality_aware_r) or (cluster_meta is None):
            return 1.0

        meta = dict(cluster_meta)
        scale = 1.0

        # ------------------------------------------------------------
        # 1) num_points：主导项
        # ------------------------------------------------------------
        num_points = meta.get("num_points", np.nan)
        try:
            num_points = int(num_points)
        except Exception:
            num_points = -1

        if num_points == 1:
            scale *= self.quality_singleton_penalty
        elif num_points == 2:
            scale *= self.quality_two_points_penalty
        elif num_points == 3:
            scale *= self.quality_three_points_penalty
        elif num_points >= self.quality_many_points_thr:
            scale *= self.quality_many_points_reward

        # ------------------------------------------------------------
        # 2) vr_std：辅助项
        # ------------------------------------------------------------
        vr_std = meta.get("vr_std", np.nan)
        try:
            vr_std = float(vr_std)
        except Exception:
            vr_std = np.nan

        if np.isfinite(vr_std):
            if vr_std > self.quality_ref_vr_std:
                scale *= self.quality_high_vr_std_penalty
            elif vr_std < self.quality_low_vr_std_thr:
                scale *= self.quality_low_vr_std_reward

        scale = min(max(scale, self.quality_r_min_scale), self.quality_r_max_scale)
        return float(scale)

    def _apply_quality_R(self, cluster_meta):
        """
        先根据 cluster 质量设置本次量测噪声。
        """
        scale = self._compute_quality_r_scale(cluster_meta)
        self.R = np.eye(2, dtype=float) * (self.base_r_pos * scale)
        return scale

    def update_cluster_velocity(self, cluster_vr):
        if cluster_vr is None:
            return
        try:
            v = float(cluster_vr)
        except Exception:
            return
        if not np.isfinite(v):
            return

        if not np.isfinite(self.radial_speed):
            self.radial_speed = v
        else:
            a = self.track_vel_ema_alpha
            self.radial_speed = a * v + (1.0 - a) * self.radial_speed

    def update(self, measurement, cluster_meta=None):
        z = np.asarray(measurement, dtype=float).reshape(2)

        # ------------------------------------------------------------
        # 先按 cluster 质量设置 R
        # ------------------------------------------------------------
        self._apply_quality_R(cluster_meta)

        y = z - self.H.dot(self.state)

        # 原始 adaptive R 逻辑保留；如果你 config 里关掉，它就不会覆盖上面的质量 R
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

        if cluster_meta is not None:
            self.update_cluster_velocity(cluster_meta.get("vr_median", np.nan))

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
        use_vel_assoc=True,
        assoc_vel_thr=2.0,
        assoc_w_pos=1.0,
        assoc_w_vel=0.8,
        track_vel_ema_alpha=0.6,
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

        # ------------------------------------------------------------
        # Step 2: quality-aware R
        # ------------------------------------------------------------
        use_quality_aware_r=False,
        quality_r_min_scale=0.75,
        quality_r_max_scale=4.0,

        quality_singleton_penalty=2.20,
        quality_two_points_penalty=1.50,
        quality_three_points_penalty=1.15,
        quality_many_points_reward=0.90,
        quality_many_points_thr=4,

        quality_ref_vr_std=0.80,
        quality_high_vr_std_penalty=1.20,
        quality_low_vr_std_thr=0.20,
        quality_low_vr_std_reward=0.95,
    ):
        self.method = str(method).lower()
        self.assoc_metric = str(assoc_metric).lower()
        self.assoc_dist_thr = float(assoc_dist_thr)
        self.assoc_mahal_thr = float(assoc_mahal_thr)

        self.use_vel_assoc = bool(use_vel_assoc)
        self.assoc_vel_thr = float(assoc_vel_thr)
        self.assoc_w_pos = float(assoc_w_pos)
        self.assoc_w_vel = float(assoc_w_vel)
        self.track_vel_ema_alpha = float(track_vel_ema_alpha)

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

        # quality-aware R
        self.use_quality_aware_r = bool(use_quality_aware_r)
        self.quality_r_min_scale = float(quality_r_min_scale)
        self.quality_r_max_scale = float(quality_r_max_scale)

        self.quality_singleton_penalty = float(quality_singleton_penalty)
        self.quality_two_points_penalty = float(quality_two_points_penalty)
        self.quality_three_points_penalty = float(quality_three_points_penalty)
        self.quality_many_points_reward = float(quality_many_points_reward)
        self.quality_many_points_thr = int(quality_many_points_thr)

        self.quality_ref_vr_std = float(quality_ref_vr_std)
        self.quality_high_vr_std_penalty = float(quality_high_vr_std_penalty)
        self.quality_low_vr_std_thr = float(quality_low_vr_std_thr)
        self.quality_low_vr_std_reward = float(quality_low_vr_std_reward)

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
            track_vel_ema_alpha=self.track_vel_ema_alpha,

            use_quality_aware_r=self.use_quality_aware_r,
            quality_r_min_scale=self.quality_r_min_scale,
            quality_r_max_scale=self.quality_r_max_scale,

            quality_singleton_penalty=self.quality_singleton_penalty,
            quality_two_points_penalty=self.quality_two_points_penalty,
            quality_three_points_penalty=self.quality_three_points_penalty,
            quality_many_points_reward=self.quality_many_points_reward,
            quality_many_points_thr=self.quality_many_points_thr,

            quality_ref_vr_std=self.quality_ref_vr_std,
            quality_high_vr_std_penalty=self.quality_high_vr_std_penalty,
            quality_low_vr_std_thr=self.quality_low_vr_std_thr,
            quality_low_vr_std_reward=self.quality_low_vr_std_reward,
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

    def _velocity_distance(self, track, meta):
        if (not self.use_vel_assoc) or meta is None:
            return 0.0

        cluster_vr = meta.get("vr_median", np.nan)
        try:
            cluster_vr = float(cluster_vr)
        except Exception:
            return 0.0

        if (not np.isfinite(cluster_vr)) or (not np.isfinite(track.radial_speed)):
            return 0.0

        return float(abs(cluster_vr - track.radial_speed))

    def _gate_threshold(self):
        if self.assoc_metric == "mahalanobis":
            return self.assoc_mahal_thr
        return self.assoc_dist_thr

    def _association_cost(self, track, center, meta):
        c_pos = self._distance(track, center)
        c_vel = self._velocity_distance(track, meta)
        return self.assoc_w_pos * c_pos + self.assoc_w_vel * c_vel

    def _maybe_confirm_track(self, track):
        if (not track.is_confirmed) and track.hit_streak >= self.min_hits_to_confirm:
            track.confirm()

    def _should_output_track(self, track):
        return bool(track.is_confirmed)

    def _should_delete_track(self, track):
        if track.is_tentative:
            return track.miss_count > self.max_tentative_misses
        return track.miss_count > self.max_misses

    def _build_cost_matrix(self, cluster_centers, cluster_meta=None):
        if cluster_meta is None:
            cluster_meta = {}

        track_ids = list(self.tracks.keys())
        cluster_ids = list(cluster_centers.keys())
        cost = np.zeros((len(track_ids), len(cluster_ids)), dtype=float)

        for tid in track_ids:
            self.tracks[tid].predict()

        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            for j, cid in enumerate(cluster_ids):
                center = cluster_centers[cid]
                meta = cluster_meta.get(cid, {})

                vel_dist = self._velocity_distance(track, meta)
                meta_vr = meta.get("vr_median", np.nan)
                try:
                    meta_vr = float(meta_vr)
                except Exception:
                    meta_vr = np.nan

                if self.use_vel_assoc and np.isfinite(track.radial_speed):
                    if np.isfinite(meta_vr) and vel_dist > self.assoc_vel_thr:
                        cost[i, j] = BIG_COST
                        continue

                cost[i, j] = self._association_cost(track, center, meta)

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
                "radial_speed": float(track.radial_speed) if np.isfinite(track.radial_speed) else np.nan,
            }
        return out

    def step(self, cluster_centers, cluster_meta=None):
        cluster_centers = {
            int(cid): np.asarray(center, dtype=float).reshape(2)
            for cid, center in cluster_centers.items()
        }
        if cluster_meta is None:
            cluster_meta = {}

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
            for cid, center in cluster_centers.items():
                tid = self._new_track(center)
                meta = cluster_meta.get(cid, {})
                self.tracks[tid].update_cluster_velocity(meta.get("vr_median", np.nan))
            return filtered_centers, track_assignments, raw_centers

        cost, track_ids, cluster_ids = self._build_cost_matrix(
            cluster_centers=cluster_centers,
            cluster_meta=cluster_meta,
        )
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

            filtered = track.update(
                cluster_centers[cid],
                cluster_meta=cluster_meta.get(cid, {}),
            )
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
            tid = self._new_track(center)
            meta = cluster_meta.get(cid, {})
            self.tracks[tid].update_cluster_velocity(meta.get("vr_median", np.nan))

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
