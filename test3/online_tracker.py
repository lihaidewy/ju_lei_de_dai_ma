import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanTrack:
    def __init__(self, track_id, center, dt=1.0, q_pos=0.30, q_vel=0.50, r_pos=1.50):
        self.track_id = int(track_id)
        self.dt = float(dt)

        x, y = center

        # state: [x, y, vx, vy]
        self.state = np.array([x, y, 0.0, 0.0], dtype=float)

        # covariance
        self.P = np.eye(4, dtype=float)

        # motion model
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # measurement model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # process noise
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)

        # measurement noise
        self.R = np.eye(2, dtype=float) * float(r_pos)

        self.age = 1
        self.hit_count = 1
        self.miss_count = 0

        self.raw_center = np.array(center, dtype=float)
        self.filtered_center = np.array(center, dtype=float)

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def update(self, measurement):
        z = np.array(measurement, dtype=float)

        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4, dtype=float) - K @ self.H) @ self.P

        self.filtered_center = self.state[:2].copy()
        self.raw_center = z.copy()

        self.hit_count += 1
        self.miss_count = 0
        self.age += 1

        return self.filtered_center.copy()

    def mark_missed(self):
        self.miss_count += 1
        self.age += 1


class OnlineTrackerManager:
    def __init__(
        self,
        assoc_dist_thr=8.0,
        max_misses=5,
        dt=1.0,
        q_pos=0.30,
        q_vel=0.50,
        r_pos=1.50,
    ):
        self.assoc_dist_thr = float(assoc_dist_thr)
        self.max_misses = int(max_misses)
        self.dt = float(dt)

        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_pos = float(r_pos)

        self.next_track_id = 1
        self.tracks = {}

    def _build_cost_matrix(self, cluster_centers):
        track_ids = list(self.tracks.keys())
        cluster_ids = list(cluster_centers.keys())

        cost = np.zeros((len(track_ids), len(cluster_ids)), dtype=float)
        predictions = {}

        for tid in track_ids:
            predictions[tid] = self.tracks[tid].predict()

        for i, tid in enumerate(track_ids):
            pred = predictions[tid]
            for j, cid in enumerate(cluster_ids):
                c = cluster_centers[cid]
                d = np.linalg.norm(pred - c)
                cost[i, j] = d

        return cost, track_ids, cluster_ids

    def _new_track(self, center):
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = KalmanTrack(
            track_id=tid,
            center=center,
            dt=self.dt,
            q_pos=self.q_pos,
            q_vel=self.q_vel,
            r_pos=self.r_pos,
        )
        return tid

    def step(self, cluster_centers):
        cluster_centers = {
            int(cid): np.asarray(c, dtype=float)
            for cid, c in cluster_centers.items()
        }

        filtered_centers = {}
        track_assignments = {}
        raw_centers = {}

        # 没有任何 cluster，所有现有 track 都记一次 missed
        if len(cluster_centers) == 0:
            to_delete = []
            for tid, track in self.tracks.items():
                track.mark_missed()
                if track.miss_count > self.max_misses:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return filtered_centers, track_assignments, raw_centers

        # 当前没有 track：直接初始化
        if len(self.tracks) == 0:
            for cid, center in cluster_centers.items():
                tid = self._new_track(center)
                filtered_centers[cid] = center.copy()
                track_assignments[cid] = tid
                raw_centers[cid] = center.copy()
            return filtered_centers, track_assignments, raw_centers

        cost, track_ids, cluster_ids = self._build_cost_matrix(cluster_centers)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_clusters = set()

        # 1) 匹配成功的 track 做 update
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.assoc_dist_thr:
                continue

            tid = track_ids[r]
            cid = cluster_ids[c]

            track = self.tracks[tid]
            filtered = track.update(cluster_centers[cid])

            filtered_centers[cid] = filtered.copy()
            track_assignments[cid] = tid
            raw_centers[cid] = cluster_centers[cid].copy()

            matched_tracks.add(tid)
            matched_clusters.add(cid)

        # 2) 未匹配 cluster：新建轨迹
        for cid in cluster_ids:
            if cid not in matched_clusters:
                center = cluster_centers[cid]
                tid = self._new_track(center)

                filtered_centers[cid] = center.copy()
                track_assignments[cid] = tid
                raw_centers[cid] = center.copy()

                matched_tracks.add(tid)  # 新建轨迹本帧视作已处理

        # 3) 未匹配旧轨迹：miss 计数并按需删除
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.mark_missed()
                if track.miss_count > self.max_misses:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return filtered_centers, track_assignments, raw_centers
