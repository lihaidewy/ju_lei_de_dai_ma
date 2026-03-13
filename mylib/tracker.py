import numpy as np
from mylib.extract_cluster_measurements import extract_cluster_measurements

class Track:
    def __init__(self, tid, x, y, frame_id):
        self.tid = tid
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

        self.first_frame = frame_id
        self.last_frame = frame_id

        self.hits = 1          # 被成功关联的次数
        self.misses = 0        # 连续丢失次数
        self.age = 1           # 存在的帧数（含miss帧）
        self.history = [(frame_id, x, y)]

    def predict(self, dt=1.0):
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        return self.x, self.y

    def update(self, meas_x, meas_y, frame_id, alpha=0.6, dt=1.0):
        # 用观测更新速度（非常简化）
        new_vx = (meas_x - self.x) / dt
        new_vy = (meas_y - self.y) / dt
        self.vx = alpha * new_vx + (1 - alpha) * self.vx
        self.vy = alpha * new_vy + (1 - alpha) * self.vy

        self.x = meas_x
        self.y = meas_y

        self.last_frame = frame_id
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.history.append((frame_id, self.x, self.y))

    def mark_missed(self):
        self.misses += 1
        self.age += 1

def track_across_frames(frame_data, frame_ids, get_labels_fn,
                        gate_dist=3.0, max_misses=2, min_hits=3, dt=1.0):
    """
    gate_dist: 关联门限(米)——预测点到观测质心距离 < gate_dist 才允许匹配
    max_misses: 连续丢失多少帧后删 track
    min_hits: 至少命中多少次才认为“真实目标”(用于最后过滤短命多径)
    get_labels_fn(fid) -> labels  (你用 cluster_frame_dbscan 得到)
    """
    tracks = []
    finished = []
    next_tid = 1

    for fid in frame_ids:
        labels = get_labels_fn(fid)
        meas = extract_cluster_measurements(frame_data, fid, labels)

        # 1) 预测
        for tr in tracks:
            tr.predict(dt=dt)

        # 2) 计算代价矩阵（距离）
        if len(tracks) > 0 and len(meas) > 0:
            cost = np.zeros((len(tracks), len(meas)), dtype=float)
            for i, tr in enumerate(tracks):
                for j, mj in enumerate(meas):
                    dx = tr.x - mj["x"]
                    dy = tr.y - mj["y"]
                    cost[i, j] = np.hypot(dx, dy)

            # 3) 贪心匹配（足够用；想更优可换匈牙利）
            used_tr = set()
            used_meas = set()
            pairs = []
            # 按距离从小到大取
            flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
            flat.sort(key=lambda t: t[0])

            for d, i, j in flat:
                if d > gate_dist:
                    break
                if i in used_tr or j in used_meas:
                    continue
                used_tr.add(i); used_meas.add(j)
                pairs.append((i, j, d))

            # 4) 更新匹配到的 track
            for i, j, d in pairs:
                tr = tracks[i]
                mj = meas[j]
                tr.update(mj["x"], mj["y"], fid, dt=dt)

            # 5) 未匹配的 track：miss
            for i, tr in enumerate(tracks):
                if i not in used_tr:
                    tr.mark_missed()

            # 6) 未匹配的观测：新建 track
            for j, mj in enumerate(meas):
                if j not in used_meas:
                    tr = Track(next_tid, mj["x"], mj["y"], fid)
                    tracks.append(tr)
                    next_tid += 1
        else:
            # 没有 track 或没有观测：全部 miss 或全部新建
            if len(meas) == 0:
                for tr in tracks:
                    tr.mark_missed()
            else:
                for mj in meas:
                    tr = Track(next_tid, mj["x"], mj["y"], fid)
                    tracks.append(tr)
                    next_tid += 1

        # 7) 删除丢失太久的 track
        alive = []
        for tr in tracks:
            if tr.misses > max_misses:
                finished.append(tr)
            else:
                alive.append(tr)
        tracks = alive

    # 把还活着的也收尾
    finished.extend(tracks)

    # 最终过滤：hits 太少的认为“短命多径/假目标”
    real_tracks = [tr for tr in finished if tr.hits >= min_hits]
    ghost_tracks = [tr for tr in finished if tr.hits < min_hits]
    return real_tracks, ghost_tracks
