# Radar Cluster Center Evaluation Pipeline

> 这是一份基于当前已提供代码整理的详细 README。
> 本 README 聚焦于 **雷达点云聚类、cluster center 估计、在线时序跟踪、与 GT 的匹配评估、统计汇总、结果导出、可视化渲染** 这一整条处理链。

---

## 1. 项目概览

这个项目实现的是一条 **面向逐帧雷达点云的目标聚类与中心点评估流水线**。
从代码结构上看，核心目标不是“训练模型”，而是对已经加载好的雷达点数据进行工程化处理，并生成可用于分析和实验对比的结果。

整体能力包括：

- 读取雷达数据与地面真值（GT）
- 选取 radar 和 GT 都存在的帧
- 对每一帧执行聚类
- 为每个 cluster 估计中心点
- 支持不同中心估计策略
- 支持按 Doppler 速度做中心点 refinement
- 支持按距离分段的中心偏置修正
- 支持在线多目标跟踪与中心点平滑
- 以 cluster center 为基础与 GT 做匹配评估
- 统计 TP / FP / FN、precision / recall / F1、中心误差与偏差
- 导出点级结果表和 TP 匹配样本
- 可视化原始点、聚类结果、匹配关系和跟踪 ID

从工程用途看，这套代码尤其适合：

1. 比较不同 cluster center 估计策略
2. 验证速度过滤和距离偏置修正是否有效
3. 比较启用 / 不启用在线时序滤波时的中心误差变化
4. 输出可复盘的点级表与可视化结果，用于实验分析和调参

---

## 2. 当前已看到的代码文件

本 README 基于以下已提供文件整理：

- `centers.py`
- `data_pipeline.py`
- `online_tracker.py`
- `config.py`
- `exporters.py`
- `stats_utils.py`
- `viz_utils.py`

代码中还引用了若干 **未随本次代码一起提供** 的依赖模块，因此本 README 会在说明中明确区分：

### 已提供

- 中心计算与偏置修正
- 主数据流水线
- 在线跟踪器
- 配置项
- 统计汇总
- 结果导出
- 可视化渲染

### 未提供但被引用

- `load_data2.load_data`
- `plot_gt_main.load_gt_reference`
- `mylib.cluster_frame_dbscan`
- `plot_raw_and_clusters_multi_prior_v2.plot_raw_and_clusters`
- `eval_clusters2_multi_prior_v2.GT_DIM`

因此，**本 README 可以准确解释项目的核心处理逻辑和模块关系，但无法完整确认外部依赖模块的内部实现细节**。

---

## 3. 项目在做什么

### 3.1 输入

项目假设存在两类输入数据：

1. **雷达点级数据**

   - 至少包含 `X`, `Y`, `V`
   - 其中：
     - `X` 通常表示横向位置
     - `Y` 通常表示前向距离
     - `V` 表示速度或 Doppler 速度
   - 在某些模式下还可能包含 `SNR`
2. **GT 参考数据**

   - 至少包含：
     - `Frame`
     - `ID`
     - `X`
     - `Y`
     - `model`

### 3.2 每帧的核心处理步骤

对每一帧，代码执行的逻辑大致如下：

1. 从原始雷达帧中取出点坐标和速度
2. 调用 DBSCAN 风格聚类器得到点所属的 cluster label
3. 对每个有效 cluster：
   - 取出 cluster 内所有点
   - 计算 cluster center
   - 应用偏置修正
4. 如果启用了在线 tracker：
   - 将当前 cluster centers 与已有 tracks 关联
   - 用 Kalman Filter 做时序平滑
   - 生成 `track_id`
5. 以 cluster centers 与当前帧 GT 做简单匹配评估
6. 生成点级结果表：
   - 为每个点补充 `Label`
   - 对所属 cluster 写入 `gid`
   - 对所属 cluster 写入 `track_id`
   - 写入原始中心与滤波后中心
7. 将单帧结果存入 cache，供可视化和汇总统计使用

### 3.3 输出

主要输出有三类：

#### A. 单帧结构化结果

由 `process_one_frame(...)` 返回，包含：

- `point_table`
- `metrics`
- `metrics_raw`
- `cache_item`
- GT 相关映射结构

#### B. 多帧统计结果

由 `stats_utils.py` 汇总打印，包括：

- TP / FP / FN
- Precision / Recall / F1
- 中心误差统计
- 中心偏差统计
- 分类别统计
- 分距离段 Y 偏差统计

#### C. 导出文件

由 `exporters.py` 导出：

- 点级标注表 CSV / Excel
- TP 匹配样本 CSV / Excel

---

## 4. 项目整体架构

从职责划分来看，当前代码可以概括为以下结构：

```text
config.py
    ↓
data_pipeline.py
    ├── centers.py
    ├── online_tracker.py
    ├── stats_utils.py
    ├── exporters.py
    └── viz_utils.py
```

也可以从“数据流”的角度理解为：

```text
Radar + GT
   ↓
Frame selection
   ↓
Per-frame clustering
   ↓
Cluster center estimation
   ↓
Optional bias correction
   ↓
Optional online tracking / smoothing
   ↓
GT matching and evaluation
   ↓
Point-level table / cache
   ↓
Global statistics / export / visualization
```

---

## 5. 各文件详细说明

---

## 5.1 `config.py`

### 作用

集中定义整个项目的实验参数和默认配置，是整个项目的“中央控制面板”。

### 主要配置项

#### 1）数据路径

- `RADAR_PATH`
- `GT_PATH`

用于指定雷达数据和 GT 数据的默认位置。

#### 2）聚类参数

- `EPS_X`
- `EPS_Y`
- `EPS_V`
- `MIN_PTS`

这些参数会传入 `cluster_frame_dbscan(...)`，决定聚类的空间 / 速度邻域阈值以及最小点数。

#### 3）评估阈值

- `DIST_THR`
- `IOU_THR`

当前可见代码中，中心评估主要依赖 `DIST_THR`；`IOU_THR` 在已提供代码里尚未真正用到。

#### 4）帧控制

- `MAX_FRAMES_TO_VIEW`
- `FRAMES_TO_SHOW`

用于控制要处理哪些帧。

#### 5）固定 box 先验

- `FIXED_BOX_PRIORS`
- `FIXED_BOX_YAW`
- `FIXED_BOX_SCORE_LAMBDA`
- `FIXED_BOX_INSIDE_MARGIN`
- `FIXED_BOX_ALPHA_OUT`
- `FIXED_BOX_BETA_IN`

这组参数主要服务于可视化或 box 相关打分逻辑。
当前上传代码中，它们被 `viz_utils.py` 传入底层绘图函数。

#### 6）cluster center 模式

- `CLUSTER_CENTER_MODE`

支持的模式由 `centers.py` 决定，目前包括：

- `mean`
- `median`
- `snr_mean`

#### 7）速度过滤参数

- `USE_VELOCITY_FILTER`
- `VELOCITY_FILTER_THR`
- `VELOCITY_FILTER_MIN_POINTS`

用于控制 Doppler velocity filtering 是否启用，以及过滤阈值和最少保留点数。

#### 8）中心偏置修正参数

- `CENTER_BIAS_X`
- `USE_RANGE_BIAS_Y`
- `BIAS_SPLIT_Y`
- `BIAS_Y_NEAR`
- `BIAS_Y_FAR`

用于控制 cluster center 的后处理偏置修正。

#### 9）导出路径

- `EXPORT_CSV_PATH`
- `EXPORT_XLSX_PATH`
- `TP_MATCH_CSV_PATH`
- `TP_MATCH_XLSX_PATH`

#### 10）在线 tracker 参数

- `USE_ONLINE_TRACKER`
- `TRACK_ASSOC_DIST_THR`
- `TRACK_MAX_MISSES`
- `KF_DT`
- `KF_Q_POS`
- `KF_Q_VEL`
- `KF_R_POS`

#### 11）调试开关

- `ENABLE_TEMPORAL_DEBUG`

---

## 5.2 `centers.py`

### 作用

负责 **cluster center 的估计与后处理修正**。

### 已实现的中心估计策略

#### `compute_center_mean`

对 cluster 内所有点按列求平均，得到几何均值中心。

适合点分布较稳定、离群点不强的情况。

---

#### `compute_center_median`

对每个维度取中位数，得到中值中心。

相比均值，抗异常点能力更强。
如果点云中存在少量偏离很大的离群点，这个模式通常更稳。

---

#### `compute_center_snr_mean`

使用 `SNR` 作为加权信号，对点坐标做加权平均。

实现中采用了：

1. 对 SNR 做下界截断
2. 开平方
3. 归一化
4. 作为点权重参与加权平均

这意味着：

- 高 SNR 点更重要
- 但不会直接按 SNR 线性放大，避免个别极高 SNR 点完全主导中心

这个模式要求 `frame_item` 中存在 `SNR` 字段，否则会抛出错误。

---

### Doppler 速度过滤

#### `compute_center_velocity_filtered_mean`

这是对 `mean` 模式的一种 refinement：

1. 取 cluster 内速度中位数作为参考速度
2. 只保留与参考速度足够接近的点
3. 对保留点做均值
4. 如果保留点数不足，回退到普通均值

这种做法的目的，是在同一 cluster 中进一步剔除速度不一致的杂点，
让中心更靠近“运动一致”的主成分。

它只在下面两个条件同时满足时启用：

- `cfg.USE_VELOCITY_FILTER == True`
- `cfg.CLUSTER_CENTER_MODE == "mean"`

换句话说：

- `median` 模式不会触发速度过滤
- `snr_mean` 模式也不会触发速度过滤

---

### 中心估计策略分发

#### `get_center_function(mode)`

根据字符串选择中心函数。
是上层 pipeline 注入中心策略时的关键入口。

---

### 偏置修正

#### `apply_no_bias`

不做任何修正，直接返回中心。

#### `apply_two_segment_bias`

对中心点做二段式偏置修正：

- X 方向统一叠加 `CENTER_BIAS_X`
- Y 方向如果启用，则按 `BIAS_SPLIT_Y` 分成近距与远距：
  - 近距加 `BIAS_Y_NEAR`
  - 远距加 `BIAS_Y_FAR`

这种设计适用于：

- 传感器在不同距离上存在系统性偏差
- 希望用简单规则做经验补偿

---

### 这个模块的定位

`centers.py` 不是在做聚类，而是在 **“聚类结果已经给定”的前提下，估计每个 cluster 的代表中心**。
它是整条流水线里直接影响评估误差的关键模块之一。

---

## 5.3 `data_pipeline.py`

### 作用

这是整个项目的主流程模块。
它负责把数据加载、帧筛选、聚类、中心构建、时序滤波、评估、点表构建串成一条完整 pipeline。

---

### 5.3.1 数据加载

#### `load_all_data(cfg)`

从配置中读取路径，调用：

- `load_gt_reference(...)`
- `load_data(...)`

得到：

- `gt_df`
- `radar_data`

这一步依赖外部模块，但接口很明确：

- `gt_df` 是 GT 的 DataFrame
- `radar_data` 是以 `frame_id` 为 key 的帧数据容器

---

### 5.3.2 帧筛选

#### `get_frame_ids(radar_data, gt_df, cfg, args)`

这个函数的目的，是找出 **雷达和 GT 同时存在的帧**。

处理逻辑：

1. 取 radar 全部 frame id
2. 取 GT 中全部 frame id
3. 做交集
4. 如果 `FRAMES_TO_SHOW is None`：
   - 返回交集中前 `args.max_frames` 帧
5. 如果 `FRAMES_TO_SHOW` 被显式指定：
   - 只保留既在该列表中，又在交集中存在的帧

它还包含两个安全检查：

- 若交集为空，直接报错
- 若用户指定帧但一个都不在交集中，也报错

---

### 5.3.3 GT 结构构造

#### `build_gt_list_for_frame(gt_df, fid)`

把当前帧的 GT 转成列表结构：

```python
[
    {"id": ..., "x": ..., "y": ..., "model": ...},
    ...
]
```

#### `build_gt_maps(gt_list)`

进一步把 GT 列表展开成：

- `gt_map`
- `gt_model_map`
- `gt_pos_map`

这样后续不同阶段可以各取所需：

- 查完整信息
- 查类别
- 查位置
- 做距离计算

---

### 5.3.4 聚类

#### `cluster_one_frame(radar_data, fid, cfg)`

调用 `cluster_frame_dbscan(...)` 完成单帧聚类。

从调用参数看，这个聚类器至少考虑：

- `X`
- `Y`
- `V`

对应参数：

- `EPS_X`
- `EPS_Y`
- `EPS_V`
- `MIN_PTS`

当前上传代码未包含 `cluster_frame_dbscan` 的内部实现，因此这里只能确认它的输入接口和外部职责。

---

### 5.3.5 有效 cluster 过滤

#### `iter_valid_cluster_ids(labels)`

只把 `labels` 中 `cid >= 1` 的聚类标签当作有效 cluster。

这意味着：

- `0` 或负数标签不会被视为有效 cluster
- 当前项目内部有效 cluster 的约定是“正整数 ID”

---

### 5.3.6 构建 cluster 中心

#### `build_cluster_centers(labels, pts, frame_item, center_fn, bias_fn, cfg)`

这是单帧处理中非常核心的一步。

工作过程：

1. 遍历当前帧所有有效 cluster id
2. 为每个 cluster 取出对应点集
3. 调用 `compute_center_with_optional_velocity_filter(...)` 求中心
4. 对中心应用 `bias_fn(...)`
5. 保存到 `{cid: center}` 字典

注意这里的设计非常重要：

- **中心计算函数 `center_fn` 由外部注入**
- **偏置函数 `bias_fn` 也由外部注入**

因此，pipeline 本身并不强绑定某一种中心估计策略，扩展性很好。

---

### 5.3.7 时序滤波接口

#### A. `temporal_filter_cluster_centers_with_matches(...)`

这是一个“基于当前匹配结果”的版本。
逻辑是把当前帧 match 中的 `gid` 当成 track id，然后去更新 tracker。

从整体结构看，这更像早期验证版或辅助版。

---

#### B. `temporal_filter_cluster_centers_online(...)`

这是更标准的在线版本。

特点：

- 不依赖 GT
- 直接调用 `tracker.step(cluster_centers)`
- 返回：
  - `filtered_centers`
  - `track_assignments`
  - `raw_centers`

这才是和 `online_tracker.py` 正式对接的主入口。

---

### 5.3.8 点级结果表构造

#### `build_point_level_table_from_centers(...)`

作用是把 cluster 级结果映射回点级表。

它会：

1. 复制原始 `frame_item`
2. 增加 `Label`
3. 根据匹配结果增加 `gid`
4. 根据 tracking 结果增加 `track_id`
5. 增加：
   - `Raw_Center_X`
   - `Raw_Center_Y`
   - `Center_X`
   - `Center_Y`

这意味着最终导出的表，既保留原始点属性，又附带了每个点所属 cluster 的分析结果。

这种设计非常适合：

- 离线分析
- 可视化验证
- 误差追踪
- 与 Excel / CSV 联动排查问题

---

### 5.3.9 评估逻辑

#### `_build_eval_summary(matches, cids, gts)`

根据匹配结果生成单帧评估指标。

内容包括：

##### 检测类指标

- `TP`
- `FP`
- `FN`
- `precision`
- `recall`
- `f1`

##### 中心误差统计

- `mean_center_error`
- `median_center_error`
- `p90_center_error`
- `p95_center_error`
- `acc_0p3m`
- `acc_0p5m`

##### 偏差统计

- `mean_dx_error`
- `mean_dy_error`
- `median_dx_error`
- `median_dy_error`
- `std_dx_error`
- `std_dy_error`

##### 详细样本

- `center_errors`
- `dx_errors`
- `dy_errors`
- `matches`
- `unmatched_clusters`
- `unmatched_gts`

##### 分 model 统计

- `model_counts`

---

### 需要注意的一点

当前可见实现中，`model_counts` 的 `FP` 并没有真正按预测端做完整统计。逻辑上更偏向：

- GT 被匹配到了，就算对应 model 的 TP
- GT 没被匹配到，就算对应 model 的 FN

因此，README 中建议把这一项理解为 **“偏 GT 侧的 per-model TP/FN 统计”**，而不是严格完整的 per-model confusion matrix。

---

### 5.3.10 GT 匹配评估

#### `evaluate_with_given_centers(cluster_centers, gt_list, dist_thr=6.0)`

这个函数对“外部给定的 cluster centers”做简单评估。

匹配策略是：

- 逐个 cluster center 找最近 GT
- 同一 GT 只能被匹配一次
- 距离小于阈值才算匹配成功

这是一个 **greedy nearest matching**，优点是简单，适合快速验证。
但它不是全局最优匹配，因此在密集目标场景下，可能不是最理想的评估器。

---

### 5.3.11 主入口

#### `process_one_frame(fid, radar_data, gt_df, fit_mode, cfg, center_fn, bias_fn, tracker=None)`

这是整个项目按帧处理的总入口。

步骤如下：

1. 读取当前帧原始数据
2. 提取 `pts` 和 `v`
3. 调聚类器得到 `labels`
4. 读取当前帧 GT
5. 构造 GT map
6. 计算 `cluster_centers_raw`
7. 若提供 tracker，则做在线时序滤波
8. 计算 `metrics_raw`
9. 计算 `metrics`
10. 构建 `point_table`
11. 构建 `cache_item`
12. 返回本帧所有关键结果

---

### 返回值说明

#### `point_table`

点级结果表，适合导出与离线分析。

#### `metrics`

基于最终 center（可能经过滤波）得到的评估结果。

#### `metrics_raw`

基于原始 center（未经过 tracker）得到的评估结果。

#### `cache_item`

供后续可视化、交互式浏览和调试使用的缓存结构，包含：

- 点坐标
- 速度
- labels
- GT 信息
- raw / filtered metrics
- raw / filtered centers
- track assignments

---

## 5.4 `online_tracker.py`

### 作用

实现一个 **在线多目标跟踪器**，用于：

- 将相邻帧 cluster center 关联起来
- 为 cluster 赋予稳定的 `track_id`
- 通过 Kalman Filter 平滑中心点

---

### 5.4.1 `KalmanTrack`

表示一条轨迹。

#### 状态定义

使用 4 维状态：

```text
[x, y, vx, vy]
```

含义：

- `x, y`：位置
- `vx, vy`：速度

#### 初始化

新轨迹建立时：

- 初始位置来自当前 cluster center
- 初始速度设为 0

#### 状态转移模型

使用标准的 constant velocity 模型。
意味着假设目标在相邻帧之间做近似匀速运动。

#### 观测模型

只观测位置 `[x, y]`，不直接观测速率。

#### 噪声参数

- `Q`：过程噪声
- `R`：测量噪声

#### 轨迹辅助状态

每条轨迹还维护：

- `age`
- `hit_count`
- `miss_count`
- `raw_center`
- `filtered_center`

---

### 5.4.2 `predict()`

对轨迹做一步时间预测，输出预测位置。

---

### 5.4.3 `update(measurement)`

用当前观测位置更新 Kalman 状态。

实现细节上，更新阶段使用了数值更稳健的写法：

- `np.linalg.solve(...)`
- Joseph form 协方差更新

这说明实现者对数值稳定性有一定考虑。

---

### 5.4.4 `mark_missed()`

当某条轨迹在当前帧没有匹配到 cluster 时，增加 miss 计数。

---

### 5.4.5 `OnlineTrackerManager`

这是多轨迹管理器。

它负责：

1. 保存所有活跃轨迹
2. 预测轨迹下一位置
3. 计算轨迹与 cluster 的代价矩阵
4. 做数据关联
5. 为新 cluster 建立新轨迹
6. 删除长期丢失的轨迹

---

### 初始化参数

- `assoc_dist_thr`
- `max_misses`
- `dt`
- `q_pos`
- `q_vel`
- `r_pos`

这些参数在 `config.py` 中有对应项。

---

### 5.4.6 `_build_cost_matrix(cluster_centers)`

流程：

1. 对每条已有轨迹先 `predict()`
2. 计算预测位置到每个 cluster center 的欧式距离
3. 形成代价矩阵

这是后续 Hungarian 匹配的基础。

---

### 5.4.7 `_new_track(center)`

为新出现的 cluster 分配新的 `track_id`，并创建 `KalmanTrack`。

---

### 5.4.8 `step(cluster_centers)`

这是 tracker 的总入口。

输入：

```python
{cid: np.array([x, y])}
```

输出：

- `filtered_centers`
- `track_assignments`
- `raw_centers`

具体逻辑分三种情况：

#### 情况 1：当前帧没有 cluster

- 对所有轨迹做预测
- 标记 miss
- 删除 miss 过多的轨迹
- 返回空结果

#### 情况 2：当前没有历史轨迹

- 为每个 cluster 新建轨迹
- `filtered_center == raw_center`
- 直接返回

#### 情况 3：既有轨迹，也有当前 cluster

1. 建代价矩阵
2. 对超阈值距离做门控
3. 用匈牙利算法做全局最优匹配
4. 对成功匹配的轨迹做 Kalman 更新
5. 未匹配 cluster 新建轨迹
6. 未匹配旧轨迹记 miss，超过阈值删除

---

### 跟踪器特点总结

#### 优点

- 在线工作，不依赖 GT
- 支持 cluster 到 track 的稳定关联
- 用 Kalman 平滑抑制帧间抖动
- 用 Hungarian 避免简单贪心带来的局部错误
- 结构清晰，适合工程使用

#### 局限

- 关联代价只基于二维位置距离
- 没使用速度、SNR、类别等更丰富信息
- 初始速度固定为 0
- 未包含复杂遮挡 / 交叉场景的高级逻辑

---

## 5.5 `stats_utils.py`

### 作用

负责跨帧累积统计，输出全局实验结果。

---

### 5.5.1 `init_stats()`

初始化全局统计容器，主要包括：

- 全局 TP / FP / FN
- 所有 TP 的中心误差列表
- 所有 TP 的 dx / dy 偏差列表
- 分 model 的累计统计

---

### 5.5.2 `update_stats(stats, metrics)`

把单帧 `metrics` 累加进全局 `stats`。

---

### 5.5.3 `update_range_bias_stats(range_bias_stats, range_bins, matches, gt_list)`

按 GT 的 `y` 距离把 TP 样本的 `dy` 归入不同 range bin。

这个设计很适合分析：

- 近距离是否偏大 / 偏小
- 中距离是否存在稳定 bias
- 远距离误差是否更大

也正好呼应了 `centers.py` 中的 “range-dependent Y bias correction” 设计。

---

### 5.5.4 `print_global_summary(frame_ids, stats, range_bins, range_bias_stats)`

负责输出全局实验报告。

主要内容包括：

#### 1）全局检测统计

- 总帧数
- 总 TP / FP / FN
- Overall P / R / F1

#### 2）TP 中心误差统计

- mean
- median
- p90
- p95
- `<=0.3m`
- `<=0.5m`

#### 3）TP 中心偏差统计

- `mean_dx`
- `mean_dy`
- `median_dx`
- `median_dy`
- `std_dx`
- `std_dy`

并明确了偏差的定义是：

```text
GT - Pred
```

#### 4）Per-model 统计

按 `model=0/1/2` 输出 TP / FP / FN、P / R / F1。

#### 5）Range-wise Y bias

按距离段输出：

- 样本数
- mean_dy
- median_dy
- std_dy

---

## 5.6 `exporters.py`

### 作用

负责把分析结果导出到磁盘。

---

### 5.6.1 `export_point_table(point_tables, csv_path, excel_path)`

把多帧点级表拼接后导出为：

- CSV
- Excel

导出的表是整个项目里最重要的中间 / 最终结果之一，因为它能把点级原始数据和 cluster / GT / track / center 信息放在一起。

---

### 5.6.2 `export_tp_matches(rows, csv_path)`

导出 TP 匹配样本到 CSV。

通常可以用于：

- bias 分析
- 误差分布统计
- 手动抽样复核

---

### 5.6.3 `export_tp_matches_excel(rows, path)`

与上面类似，但导出为 Excel。

---

### 这个模块的特点

- 逻辑简单
- 与主流程解耦
- 方便实验脚本统一调用

一个需要注意的小点是：当前实现里没有显式创建目录，因此使用前需要保证目标目录已存在。

---

## 5.7 `viz_utils.py`

### 作用

在底层绘图函数基础上，为单帧结果加标题、坐标设置和匹配标注。

它不是底层散点绘图实现本身，而是 **面向分析展示的可视化增强层**。

---

### 5.7.1 标题构造

#### `_build_title(...)`

标题中包含：

- frame id
- 当前索引 / 总帧数
- fit mode
- TP / FP / FN
- precision / recall / F1
- mean center error

这样每张图的关键信息都集中在标题里。

---

### 5.7.2 坐标轴设定

#### `_configure_axes(axes)`

统一设置：

- `x ∈ [-30, 30]`
- `y ∈ [0, 250]`
- 网格
- 坐标轴标题

这样不同帧的图能保持统一尺度，方便对比。

---

### 5.7.3 cluster / track 归一化

#### `_normalize_cluster_centers(...)`

把 cluster center 字典规范成普通 Python 数值。

#### `_normalize_track_assignments(...)`

把 `track_id` 映射规范成普通 Python int。

这对后续标注和兼容性有帮助。

---

### 5.7.4 GT 位置获取

#### `_get_gt_positions(item)`

优先使用 `gt_pos_map`，否则从 `gt_list` 中重建。

---

### 5.7.5 匹配关系标注

#### `_annotate_matches(...)`

对匹配成功的 cluster 画出类似：

```text
C{cid}/T{tid}→GT{gid}
d=...
```

如果存在 IoU，也会一起显示。

#### `_annotate_unmatched_clusters(...)`

对未匹配 cluster 标注：

```text
C{cid}/T{tid}→FP
```

#### `_annotate_unmatched_gts(...)`

对未匹配 GT 标注：

```text
GT{gid}(FN)
```

---

### 5.7.6 主入口 `render_frame(...)`

处理流程：

1. 根据索引确定当前 frame
2. 从 cache 中取出当前帧结果
3. 构建标题
4. 调用底层 `plot_raw_and_clusters(...)`
5. 统一坐标轴
6. 取出 cluster centers、track_assignments、GT positions
7. 叠加匹配、FP、FN 标注
8. 刷新画布

这说明 `data_pipeline.py` 返回的 `cache_item` 设计得非常贴合可视化层需求。

---

## 6. 模块之间的协作关系

下面给出一个更完整的模块协作过程。

### 6.1 启动阶段

1. 构造 `Config`
2. 根据 `CLUSTER_CENTER_MODE` 选择 `center_fn`
3. 根据实验需求选择 `bias_fn`
4. 如果启用 tracker，则创建 `OnlineTrackerManager`

---

### 6.2 数据准备阶段

1. `load_all_data(cfg)`
2. `get_frame_ids(...)`

得到后续要处理的帧序列。

---

### 6.3 单帧处理阶段

对每个 `fid`：

1. `cluster_one_frame(...)`
2. `build_gt_list_for_frame(...)`
3. `build_gt_maps(...)`
4. `build_cluster_centers(...)`
5. 可选：`temporal_filter_cluster_centers_online(...)`
6. `evaluate_with_given_centers(...)` for raw
7. `evaluate_with_given_centers(...)` for filtered
8. `build_point_level_table_from_centers(...)`
9. 结果写入 cache / point_tables / stats

---

### 6.4 多帧汇总阶段

1. `update_stats(...)`
2. `update_range_bias_stats(...)`
3. `print_global_summary(...)`

---

### 6.5 输出阶段

1. `export_point_table(...)`
2. `export_tp_matches(...)`
3. `export_tp_matches_excel(...)`

---

### 6.6 可视化阶段

通过 `render_frame(...)` 浏览或展示单帧结果。

---

## 7. 典型处理流程示意

下面给出一个 **基于当前接口推测出的主脚本伪代码**。
注意：这不是仓库中现成上传的主程序，只是根据现有函数接口整理出的推荐调用方式。

```python
from config import Config
from centers import get_center_function, apply_no_bias, apply_two_segment_bias
from data_pipeline import load_all_data, get_frame_ids, process_one_frame
from online_tracker import OnlineTrackerManager
from stats_utils import init_stats, update_stats, update_range_bias_stats, print_global_summary
from exporters import export_point_table, export_tp_matches, export_tp_matches_excel

cfg = Config()

center_fn = get_center_function(cfg.CLUSTER_CENTER_MODE)
bias_fn = apply_two_segment_bias if cfg.USE_RANGE_BIAS_Y or cfg.CENTER_BIAS_X != 0 else apply_no_bias

tracker = None
if cfg.USE_ONLINE_TRACKER:
    tracker = OnlineTrackerManager(
        assoc_dist_thr=cfg.TRACK_ASSOC_DIST_THR,
        max_misses=cfg.TRACK_MAX_MISSES,
        dt=cfg.KF_DT,
        q_pos=cfg.KF_Q_POS,
        q_vel=cfg.KF_Q_VEL,
        r_pos=cfg.KF_R_POS,
    )

radar_data, gt_df = load_all_data(cfg)
frame_ids = get_frame_ids(radar_data, gt_df, cfg, args)

stats = init_stats()
cache = {}
point_tables = []
tp_rows = []

range_bins = [(0, 50), (50, 100), (100, 150), (150, 1e9)]
range_bias_stats = {rb: [] for rb in range_bins}

for fid in frame_ids:
    out = process_one_frame(
        fid=fid,
        radar_data=radar_data,
        gt_df=gt_df,
        fit_mode="some_mode",
        cfg=cfg,
        center_fn=center_fn,
        bias_fn=bias_fn,
        tracker=tracker,
    )

    point_tables.append(out["point_table"])
    cache[fid] = out["cache_item"]

    metrics = out["metrics"]
    update_stats(stats, metrics)
    update_range_bias_stats(range_bias_stats, range_bins, metrics["matches"], out["gt_list"])

    tp_rows.extend(metrics["matches"])

print_global_summary(frame_ids, stats, range_bins, range_bias_stats)

export_point_table(point_tables, cfg.EXPORT_CSV_PATH, cfg.EXPORT_XLSX_PATH)
export_tp_matches(tp_rows, cfg.TP_MATCH_CSV_PATH)
export_tp_matches_excel(tp_rows, cfg.TP_MATCH_XLSX_PATH)
```

---

## 8. 输入 / 输出接口约定

---

## 8.1 radar_data 的预期结构

从代码使用方式来看，`radar_data[fid]` 应该至少满足：

- 能按列访问
- 支持 `frame_item["X"]`
- 支持 `frame_item["Y"]`
- 支持 `frame_item["V"]`
- 在某些模式下支持 `frame_item["SNR"]`
- 支持 `.copy()` 和 `.reset_index(drop=True)` 的调用方式（说明很可能是 DataFrame）

因此，最合理的推断是：

- `radar_data` 是一个以 `frame_id -> pandas.DataFrame` 的映射结构

---

## 8.2 GT DataFrame 的预期列

从 `build_gt_list_for_frame(...)` 可见，GT DataFrame 需要至少包含：

- `Frame`
- `ID`
- `X`
- `Y`
- `model`

---

## 8.3 point_table 的结果列

当前代码明确会追加：

- `Label`
- `gid`
- `track_id`
- `Raw_Center_X`
- `Raw_Center_Y`
- `Center_X`
- `Center_Y`

这些列会追加在原始 frame_item 的右侧。

---

## 8.4 metrics 的关键字段

单帧 metrics 通常包含：

- `TP`
- `FP`
- `FN`
- `precision`
- `recall`
- `f1`
- `mean_center_error`
- `median_center_error`
- `p90_center_error`
- `p95_center_error`
- `acc_0p3m`
- `acc_0p5m`
- `mean_dx_error`
- `mean_dy_error`
- `median_dx_error`
- `median_dy_error`
- `std_dx_error`
- `std_dy_error`
- `center_errors`
- `dx_errors`
- `dy_errors`
- `matches`
- `unmatched_clusters`
- `unmatched_gts`
- `model_counts`

---

## 9. 当前项目最重要的设计思想

---

## 9.1 “聚类”和“中心估计”分离

聚类器负责给出 cluster membership；
中心怎么计算，完全交给 `centers.py`。

这样做的好处是：

- 容易比较不同 center 策略
- 不需要改动聚类器就能做大量实验
- 对工程验证很友好

---

## 9.2 “原始中心”和“滤波中心”并存

`process_one_frame(...)` 同时保留：

- `cluster_centers_raw`
- `cluster_centers_filtered`
- `metrics_raw`
- `metrics`

这让你可以直接比较：

- 不做 tracking 时的误差
- 做 tracking 后的误差

是非常实用的实验设计。

---

## 9.3 点级结果表是最终分析核心

虽然 cluster center 是 cluster 级结果，但代码最终会把结果映射回点级表。这意味着：

- 每个点都能找到它的 cluster
- 每个点都能关联 cluster 的中心、GT、track

这对排错与数据复盘非常有价值。

---

## 9.4 统计与可视化都围绕 cache 和 metrics 统一展开

这说明整个项目的数据组织相对统一：

- 单帧以 `metrics` 和 `cache_item` 为核心
- 多帧统计以 `stats` 为核心
- 导出以 `point_table` 和 `matches` 为核心

---

## 10. 当前代码的优点

### 10.1 结构清晰

职责拆分明确：

- 中心估计
- 主流程
- 跟踪
- 统计
- 导出
- 可视化

### 10.2 适合实验

特别适合做 A/B 测试，例如比较：

- `mean` vs `median`
- 是否启用速度过滤
- 是否启用 Y 偏置修正
- 是否启用在线跟踪

### 10.3 结果可追溯

点表、cache、matches、统计输出都比较完整。

### 10.4 工程可扩展

中心函数、偏置函数、tracker 都是“外部注入”的，不是写死在一个大函数里。

---

## 11. 当前代码的局限与注意事项

### 11.1 没有上传主入口脚本

目前看到了完整的模块，但没有看到最顶层运行脚本。
因此 README 中的运行方式是基于函数接口推导，而不是仓库中现成的 `main.py`。

### 11.2 外部依赖未提供

以下关键模块当前缺失：

- 数据加载器
- GT 加载器
- DBSCAN 聚类实现
- 底层绘图函数
- GT_DIM 定义

所以项目还不能仅靠当前上传文件单独运行。

### 11.3 `fit_mode` 的语义未完全确定

`process_one_frame(...)` 和 `viz_utils.py` 会接收 `fit_mode`，但当前上传代码里没有它的定义来源和可选值全集。

### 11.4 per-model FP 统计可能不完整

当前 `model_counts` 更接近 GT 侧 TP/FN 统计，而不是严格完整的 per-model confusion 统计。

### 11.5 评估匹配器比较简单

当前 `evaluate_with_given_centers(...)` 是 greedy nearest matching，不是 Hungarian，也没有 IoU 联合匹配。

### 11.6 tracker 关联代价较朴素

当前 tracker 仅使用位置距离做关联，复杂场景中可能出现 ID switch。

### 11.7 导出函数默认目录已存在

`exporters.py` 里没有自动创建目录。

### 11.8 配置路径偏向 Windows 风格

`config.py` 中默认路径写法是 Windows 风格字符串，跨平台时建议统一为 `Path`。

---

## 12. 如何扩展这个项目

---

## 12.1 扩展新的中心估计方法

可以在 `centers.py` 中新增函数，例如：

- `trimmed_mean`
- `robust_mean`
- `range_weighted_mean`

然后在 `get_center_function(...)` 中注册即可。

---

## 12.2 扩展更复杂的偏置修正

当前只支持：

- 无偏置
- 二段式 Y 偏置 + X 常量偏置

可以进一步扩展为：

- 多段距离 bias
- 多项式 bias
- 基于 SNR / range / angle 的联合 bias

---

## 12.3 替换跟踪器

`data_pipeline.py` 对 tracker 的依赖接口很简单，只要有：

```python
step(cluster_centers) -> filtered_centers, track_assignments, raw_centers
```

就可以替换成：

- EMA tracker
- 更复杂的 Kalman / IMM
- JPDA / MHT 风格管理器
- 加速度模型的状态滤波器

---

## 12.4 替换评估器

当前评估器主要看中心距离。如果以后想更精细，可以扩展：

- Hungarian on distance
- Distance + IoU 联合匹配
- 类别约束匹配
- 时序一致性评估

---

## 12.5 增加误差分析工具

在现有统计基础上可以增加：

- 不同速度段误差
- 不同类别误差
- 不同 cluster 点数的误差
- 跟踪稳定性指标
- ID switch 计数

---

## 13. 建议的仓库目录组织方式

如果后续你要把这个项目整理成更标准的仓库，建议可以采用：

```text
project/
├─ data/
│  ├─ radar_serious.csv
│  └─ reference_serious.csv
├─ mylib/
│  └─ cluster_frame_dbscan.py
├─ centers.py
├─ data_pipeline.py
├─ online_tracker.py
├─ config.py
├─ exporters.py
├─ stats_utils.py
├─ viz_utils.py
├─ main.py
├─ requirements.txt
└─ README.md
```

如果继续扩展，也可以进一步拆成：

```text
project/
├─ src/
│  ├─ pipeline/
│  ├─ tracking/
│  ├─ evaluation/
│  ├─ visualization/
│  └─ io/
├─ configs/
├─ scripts/
├─ data/
└─ README.md
```

---

## 14. 建议补充的内容

若想让项目 README 更完整，建议后续再补充：

1. `requirements.txt`
2. 主入口脚本 `main.py`
3. 示例输入数据格式
4. 示例导出结果表截图
5. 示例可视化截图
6. `fit_mode` 的全部可选值说明
7. `GT_DIM` 与 model 编号的对应关系
8. 聚类器 `cluster_frame_dbscan(...)` 的具体实现说明

---

## 15. 快速上手建议

如果你准备继续维护这个项目，建议优先做以下四件事：

### 第一步

补一个明确的主入口脚本 `main.py`

### 第二步

把当前 README 中的“推测式调用流程”替换成可直接运行的命令示例

### 第三步

补充依赖说明：

- Python 版本
- 第三方库
- 外部模块来源

### 第四步

加一份小样本数据或 mock 数据，便于验证流程

---

## 16. 简短总结

这个项目本质上是一套 **雷达点云 cluster center 实验与评估框架**，核心特点是：

- 先聚类，再估计中心
- 中心估计策略可插拔
- 支持速度过滤和偏置修正
- 支持在线 Kalman 跟踪平滑
- 支持点级结果导出
- 支持全局统计和单帧可视化
- 非常适合实验验证与工程调参

如果把它用一句话概括：

> **这是一个围绕“cluster center 是否估得更准”而设计的工程化分析 pipeline。**

---

## 17. 许可与说明

本 README 根据当前提供代码整理，优先追求：

- 结构清晰
- 逻辑完整
- 术语准确
- 对缺失部分保持诚实

对于未上传的外部模块，本 README 仅根据接口进行说明，不对其内部实现做超出代码证据范围的推断。
