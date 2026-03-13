import numpy as np
from scipy.spatial import cKDTree

def mydbscan_ellipse_vel(points, vel, eps_x, eps_y, eps_v, min_pts):
    """
    改进DBSCAN：椭球距离(按eps缩放) + 速度维度
    points: (N,2) -> [x,y]
    vel:    (N,)  -> v
    eps_x, eps_y, eps_v: 阈值
    min_pts: 最小邻域点数（含自身）
    return:
      labels: (N,)  噪声=-1, 未分配=0(函数结束时不会剩0), 簇从1开始
      core_samples_mask: (N,) bool  核心点掩码
    """
    points = np.asarray(points, dtype=float)
    vel = np.asarray(vel, dtype=float).reshape(-1)
    n = points.shape[0]

    # 1) 归一化/缩放到单位空间
    data_scaled = np.column_stack([
        points[:, 0] / eps_x,
        points[:, 1] / eps_y,
        vel / eps_v
    ])

    # 2) KDTree 范围搜索：半径=1.0
    tree = cKDTree(data_scaled)
    neighbors = tree.query_ball_point(data_scaled, r=1.0)  # list of lists

    # 3) 标准 DBSCAN 逻辑
    labels = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    # 核心点：邻居数 >= min_pts
    core_samples_mask = np.array([len(nb) >= min_pts for nb in neighbors], dtype=bool)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        N = neighbors[i]
        if len(N) < min_pts:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id

        # 扩展簇（用队列更高效；并用 set 去重避免无限增长）
        queue = list(N)
        in_queue = set(queue)

        k = 0
        while k < len(queue):
            pt = queue[k]

            if not visited[pt]:
                visited[pt] = True
                N_pt = neighbors[pt]
                if len(N_pt) >= min_pts:
                    # 合并邻居：只加入未出现过的
                    for q in N_pt:
                        if q not in in_queue:
                            in_queue.add(q)
                            queue.append(q)

            if labels[pt] == 0 or labels[pt] == -1:
                labels[pt] = cluster_id

            k += 1

    return labels, core_samples_mask
