%% 读取CSV文件并提取数据
clear; clc; close all;

%% 1. 读取CSV文件
TruePoint = 'reference.csv'; % 请替换为您的实际文件名
Measurement = 'S1m.csv';
vehicleInfo = readtable(TruePoint);
pointcloud = readtable(Measurement);
radar_H = 7.0; %雷达安装高度
%% 2. 提取数据（方法1：按列名提取）
V_frame = vehicleInfo.Frame;           % 帧号
M_frame = pointcloud.Frame;
vehicleID = vehicleInfo.ID;
%获取唯一帧号
uniqueFrames = unique(V_frame);

V_info = cell(length(uniqueFrames),1);
M_info = cell(length(uniqueFrames),1);

for i=1:length(uniqueFrames)
    currentFrame = uniqueFrames(i);  % 当前处理的帧号
    idx = (V_frame == currentFrame);    % 找出属于当前帧的行
    % 将当前帧的所有数据存储到第i个元胞中
    V_info{i} = vehicleInfo(idx, 2:6);  % 每个元胞存储一个表格
    idx = (M_frame == currentFrame);
    M_info{i} = pointcloud(idx, 2:5);
end
%% 3. 查看点云图
figure('Position', [100, 100, 200, 800]);
hold on;
grid on;
% 设置坐标轴范围


xlabel('y(m)');
ylabel('x(m)');

title('帧动画 - 车辆轨迹');
colors = lines(max(vehicleID)); % 为不同车辆准备颜色

% 动画循环
for i = 1:max(uniqueFrames)
    clf; % 清除当前图形
    hold on;
    grid on;

    xlabel('y(m)');
    ylabel('x(m)');
    xlim([-10, 10]);
    ylim([0, 200]);
    title(sprintf('帧 %d/%d', i, length(uniqueFrames)));
    % 获取当前帧数据
    currentData = V_info{i};

    if not(isempty(currentData))
        for j = 1:height(currentData)
            vehicle = currentData.ID(j);
            dist = currentData.Range(j);
            ang = currentData.Angle(j);
            x = -dist*sind(ang);% 矩形中心x坐标
            y = sqrt((dist*cosd(ang))^2 - radar_H^2);% 矩形中心y坐标
            % 绘制车辆位置（小圆点）
            plot(x, y, 'o', 'MarkerSize', 6, ...
             'MarkerFaceColor', colors(vehicle, :), ...
             'MarkerEdgeColor', 'k', 'LineWidth', 2);
            % 在指定位置(x,y)绘制宽度为w、高度为h的空心矩形
            switch mod(vehicle, 3)
                case 0
                    l = 5.06;
                    w = 2.22;
                case 1
                    l = 4.32;
                    w = 2.19;
                case 2
                    l = 3.55;
                    w = 2.58;
            end

            % 绘制空心矩形（中心在(x,y)）
            rectangle('Position', [x-w/2, y-l/2, w, l], ...
            'EdgeColor', 'k', ...      % 黑色边框
            'LineWidth', 1.5, ...       % 边框粗细
            'LineStyle', '-');          % 实线（可选：'--'虚线, ':'点线）
        end
    end
    currentMesure = M_info{i};
    if not(isempty(currentMesure))
        for j = 1:height(currentMesure)
            dist = currentMesure.Range(j);
            ang = currentMesure.Angle(j);
            x = -dist*sind(ang);
            y = sqrt((dist*cosd(ang))^2 - radar_H^2);
            % 绘制车辆位置（小圆点）
            plot(x, y, '^', 'MarkerSize', 6, ...
             'MarkerFaceColor', 'none', ...
             'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        end
    end
     % 添加图例
    drawnow; % 立即更新图形
    pause(0.1); % 暂停0.1秒，控制动画速度
end