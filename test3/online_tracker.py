import numpy as np


class Track:
    def __init__(self, track_id, center, filter_type="ema", ema_alpha=0.8):
        self.track_id = int(track_id)
        self.filter_type = str(filter_type).lower()
        self.ema_alpha = float(ema_alpha)

        center = np.asarray(center, dtype=float)
        self.raw_center = center.copy()
        self.filtered_center = center.copy()

        self.age = 1
        self.hit_count = 1
        self.miss_count = 0

    def predict(self):
        # 第一版：EMA没有显式运动模型，直接用上一次 filtered_center 作为预测
        return self.filtered_center.copy()

    def update(self, center):
        center = np.asarray(center, dtype=float)
        self.raw_center = center.copy()
        self.age += 1
        self.hit_count += 1
        self.miss_count = 0

        if self.filter_type == "ema":
            self.filtered_center = (
                self.ema_alpha * center +
                (1.0 - self.ema_alpha) * self.filtered_center
            )
        else:
            # 先占位；后面如果你要上 Kalman，再替换这里
            self.filtered_center = center.copy()

        return self.filtered_center.copy()

    def mark_missed(self):
        self.age += 1
        self.miss_count += 1


class OnlineTrackerManager:
    def __init__(
        self,
        assoc_dist_thr=6.0,
        max_misses=3,
        min_hits=1,
        filter_type="ema",
        ema_alpha=0.8,
    ):
        self.assoc_dist_thr = float(assoc_dist_thr)
        self.max_misses = int(max_misses)
        self.min_hits = int(min_hits)
        self.filter_type = str(filter_type).lower()
        self.ema_alpha = float(ema_alpha)

        self.next_track_id = 1
        self.tracks = {}

    def _match_greedy(self, cluster_centers):
        cluster_ids = sorted(cluster_centers.keys())
        track_ids = sorted(self.tracks.keys())

        matches = {}
        used_tracks = set()

        for cid in cluster_ids:
            c = np.asarray(cluster_centers[cid], dtype=float)

            best_tid = None
            best_d = float("inf")

            for tid in track_ids:
                if tid in used_tracks:
                    continue

                pred = self.tracks[tid].predict()
                d = float(np.linalg.norm(c - pred))

                if d < best_d:
                    best_d = d
                    best_tid = tid

            if best_tid is not None and best_d <= self.assoc_dist_thr:
                matches[cid] = best_tid
                used_tracks.add(best_tid)

        unmatched_clusters = [cid for cid in cluster_ids if cid not in matches]
        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
        return matches, unmatched_clusters, unmatched_tracks

    def step(self, cluster_centers):
        cluster_centers = {
            int(cid): np.asarray(center, dtype=float)
            for cid, center in cluster_centers.items()
        }

        matches, unmatched_clusters, unmatched_tracks = self._match_greedy(cluster_centers)

        filtered_centers = {}
        track_assignments = {}
        raw_centers = {}

        # 1. 已匹配轨迹：更新
        for cid, tid in matches.items():
            raw_centers[cid] = cluster_centers[cid].copy()
            filtered = self.tracks[tid].update(cluster_centers[cid])
            filtered_centers[cid] = filtered
            track_assignments[cid] = tid

        # 2. 未匹配 cluster：新建轨迹
        for cid in unmatched_clusters:
            tid = self.next_track_id
            self.next_track_id += 1

            self.tracks[tid] = Track(
                track_id=tid,
                center=cluster_centers[cid],
                filter_type=self.filter_type,
                ema_alpha=self.ema_alpha,
            )

            raw_centers[cid] = cluster_centers[cid].copy()
            filtered_centers[cid] = self.tracks[tid].filtered_center.copy()
            track_assignments[cid] = tid

        # 3. 未匹配旧轨迹：miss 计数
        to_delete = []
        for tid in unmatched_tracks:
            self.tracks[tid].mark_missed()
            if self.tracks[tid].miss_count > self.max_misses:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return filtered_centers, track_assignments, raw_centers
