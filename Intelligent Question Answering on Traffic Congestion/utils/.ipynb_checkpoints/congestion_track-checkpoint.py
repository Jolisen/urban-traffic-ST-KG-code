import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint, GeometryCollection
from shapely import ops
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_fid_time_summary(node_df):
    """
    使用 NumPy 优化 groupby 聚合过程
    """
    # 提前去重，获取每个 honeycomb_name 对应的固定值
    honeycomb_distances = node_df['honeycomb_name'].drop_duplicates()

    # 分组并用 NumPy 操作实现聚合
    grouped = node_df.groupby('honeycomb_name')
    start_time = grouped['datetime'].min()
    end_time = grouped['datetime'].max()
    min_name = grouped['trajectory_name'].min()
    max_name = grouped['trajectory_name'].max()
    count = max_name - min_name + 1


    # 创建 DataFrame
    fid_time_summary = pd.DataFrame({
        'honeycomb_name': start_time.index,
        'start_time': start_time.values,
        'end_time': end_time.values,
        'count': count.values,
        'min_name': min_name.values,
        'max_name': max_name.values
    })

    # 过滤 count > 1
    fid_time_summary = fid_time_summary[fid_time_summary['count'] > 1]

    # 合并 max_distance 和 min_distance
    fid_time_summary = fid_time_summary.merge(honeycomb_distances, on='honeycomb_name', how='left')

    # 计算持续时间
    fid_time_summary['duration'] = (fid_time_summary['end_time'] - fid_time_summary['start_time']).dt.total_seconds()

    # 按 start_time 排序
    fid_time_summary = fid_time_summary.sort_values(by='min_name', ignore_index=True)

    return fid_time_summary

def get_road_gdf(honeycomb_name, roadcrs, honeycomb_cache):
    """
    根据 honeycomb_name 从 honeycomb_cache 中查询与之关联的道路信息，
    并返回包含道路几何信息的 GeoDataFrame。
    
    参数：
    - honeycomb_name: 要查询的格网名称。
    - roadcrs: 坐标参考系，用于创建 GeoDataFrame。
    
    返回：
    - road_gdf: 包含道路名称和几何信息的 GeoDataFrame，带有指定的坐标参考系。
                 如果未找到数据，则返回 None。
    """
    # 获取指定 honeycomb_name 的数据
    road_data = honeycomb_cache.get(honeycomb_name)
    if road_data is None:
        return None
    
    # 将数据转换为 DataFrame
    road_df = pd.DataFrame(road_data)
    
    # 将 'geometry' 列中的 WKT 字符串转换为 Shapely 几何对象
    road_df['geometry'] = road_df['geometry'].apply(wkt.loads)
    
    # 转换为 GeoDataFrame，并设置坐标参考系为 roadcrs
    road_gdf = gpd.GeoDataFrame(road_df, geometry='geometry', crs=roadcrs)
    
    return road_gdf

def handle_multilinestring(merged_linestring, first_point, last_point, roadcrs, buffer_distance=1):
    """
    处理 MultiLineString，根据起点和终点选择或合并相应的 LineString。

    参数:
    - merged_linestring: 输入的 MultiLineString。
    - first_point: 起点的 Shapely Point。
    - last_point: 终点的 Shapely Point。
    - buffer_distance: 点缓冲区的距离，用于相交判断。
    - roadcrs: 坐标参考系，用于构建 GeoDataFrame。

    返回:
    - shortest_merged_linestring: 根据起点和终点处理后的单一 LineString。
    """
    # 将 MultiLineString 转为 GeoDataFrame
    line_gdf = gpd.GeoDataFrame(geometry=[line for line in merged_linestring.geoms], crs=roadcrs)

    start_idx, end_idx = None, None  # 初始化索引

    # 遍历 line_gdf 中的每个 LineString，检查与起点和终点的关系
    for idx, row in line_gdf.iterrows():
        line = row.geometry  # 获取当前 LineString

        # 检查与 start_point 和 end_point 的相交关系
        if line.intersects(first_point.buffer(buffer_distance)):
            start_idx = idx  # 匹配到起点的索引
        if line.intersects(last_point.buffer(buffer_distance)):
            end_idx = idx  # 匹配到终点的索引

        # 如果两点都匹配到同一个 `LineString`，直接使用
        if start_idx is not None and start_idx == end_idx:
            return line

    # 如果起点和终点匹配到不同的 `LineString`
    if start_idx is not None and end_idx is not None and start_idx != end_idx:
        # 获取包含起点和终点的 `LineString`
        start_line = line_gdf.loc[start_idx].geometry
        end_line = line_gdf.loc[end_idx].geometry

        # 合并起点和终点的 `LineString` 坐标
        coords = []
        for line in [start_line, end_line]:
            coords.extend(line.coords)

        # 创建新的单一 `LineString`
        return LineString(coords)

    # 如果没有匹配结果，返回 None
    return None

def get_honeycomb_adjacency(graph, prev_honeycomb_name, honeycomb_name):
    """
    检查在 honeycomb 属性节点中，prev_honeycomb_name 与 honeycomb_name 是否存在 adjacency 关系。
    
    :param graph: Neo4j 图数据库连接对象
    :param prev_honeycomb_name: 上一个 honeycomb 节点的 name
    :param honeycomb_name: 当前 honeycomb 节点的 name
    :return: True 如果存在 adjacency 关系，否则 False
    """
    # 构建 Cypher 查询
    cypher = f"""
    MATCH (prev:honeycomb {{name: {prev_honeycomb_name}}})-[:adjacency]-(current:honeycomb {{name: {honeycomb_name}}})
    RETURN count(*) > 0 AS is_adjacent
    """
    # 执行查询并返回结果
    result = graph.run(cypher).data()
    
    # 检查结果
    if result:
        return result[0]['is_adjacent']
    return False

def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(v1, v2)  # 计算点积
    norm_v1 = np.linalg.norm(v1)  # 计算向量 v1 的模长
    norm_v2 = np.linalg.norm(v2)  # 计算向量 v2 的模长
    return dot_product / (norm_v1 * norm_v2)

def trkg_matching_begin(paths, road_gdf, between_rows, tolerance=1e-3):
    """
    根据投影点计算最短路径，并通过首尾点的 intersects 匹配确定最后一个点对应的最邻近道路。
    优化：在投影点计算之前，使用 bounding box 进行相交判断。
    
    参数:
    - paths: 包含所有候选路径的信息（如路径点和几何形状）。
    - road_gdf: 包含道路几何信息的 GeoDataFrame。
    - between_rows: 包含点集的 GeoDataFrame，用于计算投影点。
    
    返回:
    - shortest_projection_points: 最短路径对应的投影点列表。
    - shortest_path: 距离最短的路径。
    - last_point_road: 最后一个点对应的道路。
    - projected_lines: shortest_merged_linestring 上的投影子线段。
    """
    shortest_projection_points = []  # 用于存储最短路径对应的投影点
    shortest_distance_sum = float('inf')  # 初始化为正无穷大
    shortest_path = None  # 用于存储距离最短的 path
    last_point_road = None  # 最后一个点对应的道路
    shortest_merged_linestring = None  # 最短路径对应的线段
    projected_lines = None
    prev_cosine_sim = -1  # 用于跟踪前一个路径的余弦相似度
    # 计算 between_rows 的首尾点向量
    start_point_raw = between_rows['geometry'].iloc[0]
    end_point_raw = between_rows['geometry'].iloc[-1]
    vector_raw = np.array([end_point_raw.x - start_point_raw.x,
                           end_point_raw.y - start_point_raw.y])

    # 遍历所有子列表
    for entry in paths:
        path = entry['path']
        merged_linestring = entry['geometry']  # 直接使用存储的 merged_linestring

        # 投影点计算：将 between_rows 中的点投影到 merged_linestring 上
        projected_points = []
        distances = []
        for point in between_rows['geometry']:
            # 计算点到 merged_linestring 的最近点
            projected_point = merged_linestring.interpolate(merged_linestring.project(point))
            projected_points.append(projected_point)
            
            # 计算点到最近点的距离
            distance = point.distance(projected_point)
            distances.append(distance)
        
        # 计算 distances 的总长度
        total_distance = sum(distances)

    # 计算 projected_points 的首尾点向量并计算余弦相似度
        # 获取首尾点
        start_point_projected = projected_points[0]
        end_point_projected = projected_points[-1]
        # 计算首尾点在 merged_linestring 上的投影长度
        start_dist = merged_linestring.project(start_point_projected)
        end_dist = merged_linestring.project(end_point_projected)
        if start_dist > end_dist:
            start_point_projected , end_point_projected = end_point_projected , start_point_projected
        vector_projected = np.array([end_point_projected.x - start_point_projected.x,
                                        end_point_projected.y - start_point_projected.y])
        # 计算余弦相似度
        cosine_sim = cosine_similarity(vector_raw, vector_projected)
        # print(f"Cosine Similarity between vectors: {cosine_sim}")

        # 如果当前路径的总距离小于之前的最短距离，则更新
        if total_distance < shortest_distance_sum:
            shortest_distance_sum = total_distance
            shortest_projection_points = projected_points
            shortest_path = path  # 更新为当前最短路径
            shortest_merged_linestring = merged_linestring  # 保存最短路径的线段
            prev_cosine_sim = cosine_sim  # 更新余弦相似度
        # ② **如果当前路径的总距离和之前的最短距离几乎相等（接近 tolerance），则比较余弦相似度**
        elif abs(total_distance - shortest_distance_sum) < tolerance:
            if cosine_sim > prev_cosine_sim:  # 如果当前余弦相似度更优，则更新
                shortest_distance_sum = total_distance
                shortest_projection_points = projected_points
                shortest_path = path  # 更新为当前最短路径
                shortest_merged_linestring = merged_linestring  # 保存最短路径的线段
                prev_cosine_sim = cosine_sim  # 更新余弦相似度
            

    # 确定 shortest_projection_points 的首尾点对应的道路
    if shortest_projection_points and shortest_path is not None:
        # 获取首尾点
        first_point = shortest_projection_points[0]
        last_point = shortest_projection_points[-1]

        # 计算首尾点在 merged_linestring 上的投影长度
        start_dist = shortest_merged_linestring.project(first_point)
        end_dist = shortest_merged_linestring.project(last_point)
        
        # 如果 shortest_path 只有一个元素
        if len(shortest_path) == 1:
            last_point_road = shortest_path[0]
            
        else:
            if start_dist > end_dist:
                start_dist, end_dist = end_dist, start_dist
                last_point_road = shortest_path[0]  # 顺序相反，取第一个点对应的道路
            else:
                last_point_road = shortest_path[-1]  # 顺序一致，取最后一个点对应的道路

        # 处理 MultiLineString 的逻辑
        if isinstance(shortest_merged_linestring, MultiLineString):
            shortest_merged_linestring = handle_multilinestring(shortest_merged_linestring, first_point, last_point, road_gdf.crs)

        # 提取 substring（部分线段）
        projected_lines = ops.substring(shortest_merged_linestring, start_dist, end_dist)


    return shortest_path, last_point_road, projected_lines

def get_roads_selectrange(graph, prev_honeycomb_name, honeycomb_name):
    """
    获取从 prev_honeycomb_name 到 honeycomb_name 的所有最短路径中，
    中间节点关联的 road 的集合。
    
    :param graph: Neo4j 图数据库连接对象
    :param prev_honeycomb_name: 起始 honeycomb 节点的 name
    :param honeycomb_name: 目标 honeycomb 节点的 name
    :return: 与路径中间节点关联的 road 名称集合（Set）。
    """
    # 构建 Cypher 查询
    cypher = """
    MATCH p = allShortestPaths((prev:honeycomb {name: $prev_honeycomb_name})-[:adjacency*]-(current:honeycomb {name: $honeycomb_name}))
    UNWIND nodes(p)[1..-1] AS honeycomb_node  // 排除首尾节点
    MATCH (honeycomb_node)-[:within]-(road:road)  // 查询与 honeycomb 节点关联的 road
    RETURN COLLECT(DISTINCT road.name) AS road_set;
    """
    
    # 执行查询
    result = graph.run(cypher, prev_honeycomb_name=prev_honeycomb_name, honeycomb_name=honeycomb_name).data()
    
    # 检查结果并返回 road 集合
    if result and result[0]['road_set'] is not None:
        return set(result[0]['road_set'])
    return set()

def get_road_shortest_step_MINdistance(graph, last_point_road, path, roads_selectrange_set):
    """
    计算从 last_point_road 到 path 中目标节点的所有最短路径的距离总和（不包含首尾节点），
    使用 roads_selectrange_set 作为搜索范围，并返回最短的距离总和。
    如果 road_set 中有不在 roads_selectrange_set 中的元素，则返回 None。

    :param graph: Neo4j 图数据库连接对象
    :param last_point_road: 起始 road 节点的 name
    :param path: 目标节点的 name 列表
    :param roads_selectrange_set: 限制计算的 road 名称集合
    :return: 最短的距离总和（浮点数），如果存在不匹配的元素，返回 None。
    """
    # 获取 path 的首尾元素
    path_edges = [path[0], path[-1]]
    # 构建 Cypher 查询
    cypher = """
    MATCH p = allshortestPaths((r1:road {name: $last_point_road})-[:touch*]-(r2:road))
    WHERE r2.name IN $path_edges
    UNWIND nodes(p)[1..-1] AS node  // 跳过首尾节点
    WITH r2.name AS target, SUM(node.distance) AS distance_sum, COLLECT(DISTINCT node.name) AS road_set
    ORDER BY distance_sum ASC  // 按照路径总距离升序排序
    LIMIT 1  // 只返回最小的一行
    RETURN target, distance_sum AS min_distance_sum, road_set;
    """
    
    # 执行查询
    result = graph.run(cypher, last_point_road=last_point_road, path_edges=path_edges).data()
    
    # 检查查询结果
    if result and result[0]['road_set']:
        # print(result)
        road_set = set(result[0]['road_set'])  # 将 road_set 转换为集合
        # 检查不匹配部分
        unmatched_roads = road_set - roads_selectrange_set  # 不在 roads_selectrange_set 中的部分
        if unmatched_roads:
            # print(f"Unmatched roads: {unmatched_roads}")
            return None  # 存在不匹配的元素
        else:
            return result[0]['min_distance_sum']  # 返回最短的距离总和
    return None  # 查询无结果或其他情况

def get_road_touch(g, last_point_road, path):
    """
    检查 name=last_point_road 的道路与 name in path 的道路之间是否存在 touch 关系。

    参数:
    - g: py2neo 的 Graph 对象，用于连接 Neo4j 数据库。
    - last_point_road: 需要检查的最后一个道路名称。
    - path: 一个包含多个道路名称的列表。

    返回:
    - 如果存在 touch 关系，返回 True；否则返回 False。
    """
    # 构建 Cypher 查询
    query = """
    MATCH (r1:road {name: $last_point_road})-[:touch]-(r2:road)
    WHERE r2.name IN $path
    RETURN COUNT(r2) > 0 AS has_touch
    """

    # 执行查询
    result = g.run(query, last_point_road=last_point_road, path=path).data()
    # 返回布尔值
    return result[0]['has_touch'] if result else False


def trkg_matching_forward(g, paths, road_gdf, between_rows, last_point_road, is_adjacent, prev_honeycomb_name, honeycomb_name, tolerance=1e-3):
    """
    根据路径和 `last_point_road` 的 touch 关系筛选最短路径，并通过首尾点的 intersects 匹配确定最后一个点对应的最邻近道路。

    参数:
    - g: py2neo 的 Graph 对象，用于连接 Neo4j 数据库。
    - paths: 包含多个路径（子列表）的列表。
    - road_gdf: 包含道路信息的 GeoDataFrame。
    - between_rows: 包含投影点的 DataFrame。
    - last_point_road: 最后一个点对应的道路名称。

    返回:
    - shortest_projection_points: 投影点的列表（最短路径）。
    - shortest_path: 对应的路径。
    - last_point_road: 最后一个点对应的道路。
    """
    shortest_projection_points = []  # 用于存储最短路径对应的投影点
    shortest_distance_sum = float('inf')  # 初始化为正无穷大
    shortest_path = None  # 用于存储距离最短的 path
    shortest_merged_linestring = None  # 最短路径对应的线段
    road_shortest_dist = float('inf')  # 初始化为正无穷大
    projected_lines = None
    prev_cosine_sim = -1  # 用于跟踪前一个路径的余弦相似度
    # 计算 between_rows 的首尾点向量
    start_point_raw = between_rows['geometry'].iloc[0]
    end_point_raw = between_rows['geometry'].iloc[-1]
    vector_raw = np.array([end_point_raw.x - start_point_raw.x,
                           end_point_raw.y - start_point_raw.y])

    if is_adjacent:
        # 遍历所有子列表
        for entry in paths:
            path = entry['path']
            merged_linestring = entry['geometry']  # 直接使用存储的 merged_linestring
    
            # 检查是否与 last_point_road 存在 touch 关系
            if not get_road_touch(g, last_point_road, path):
                continue  # 如果没有 touch 关系，跳过当前路径
    
            # 投影点计算：将 between_rows 中的点投影到 merged_linestring 上
            projected_points = []
            distances = []
            for point in between_rows['geometry']:
                # 计算点到 merged_linestring 的最近点
                projected_point = merged_linestring.interpolate(merged_linestring.project(point))
                projected_points.append(projected_point)
                
                # 计算点到最近点的距离
                distance = point.distance(projected_point)
                distances.append(distance)
            
            # 计算 distances 的总长度
            total_distance = sum(distances)

        # 计算 projected_points 的首尾点向量并计算余弦相似度
            # 获取首尾点
            start_point_projected = projected_points[0]
            end_point_projected = projected_points[-1]
            # 计算首尾点在 merged_linestring 上的投影长度
            start_dist = merged_linestring.project(start_point_projected)
            end_dist = merged_linestring.project(end_point_projected)
            if start_dist > end_dist:
                start_point_projected , end_point_projected = end_point_projected , start_point_projected
            vector_projected = np.array([end_point_projected.x - start_point_projected.x,
                                            end_point_projected.y - start_point_projected.y])
            # 计算余弦相似度
            cosine_sim = cosine_similarity(vector_raw, vector_projected)
            # print(f"Cosine Similarity between vectors: {cosine_sim}")
            
            # 如果当前路径的总距离小于之前的最短距离，则更新
            if total_distance < shortest_distance_sum:
                shortest_distance_sum = total_distance
                shortest_projection_points = projected_points
                shortest_path = path  # 更新为当前最短路径
                shortest_merged_linestring = merged_linestring  # 保存最短路径的线段
                prev_cosine_sim = cosine_sim  # 更新余弦相似度
            # ② **如果当前路径的总距离和之前的最短距离几乎相等（接近 tolerance），则比较余弦相似度**
            elif abs(total_distance - shortest_distance_sum) < tolerance:
                if cosine_sim > prev_cosine_sim:  # 如果当前余弦相似度更优，则更新
                    shortest_distance_sum = total_distance
                    shortest_projection_points = projected_points
                    shortest_path = path  # 更新为当前最短路径
                    shortest_merged_linestring = merged_linestring  # 保存最短路径的线段
                    prev_cosine_sim = cosine_sim  # 更新余弦相似度

    if not is_adjacent:
        roads_selectrange_set = get_roads_selectrange(g, prev_honeycomb_name, honeycomb_name)
        # print(f'roads_selectrange_set:{roads_selectrange_set}')
        # 遍历所有子列表
        for entry in paths:
            path = entry['path']
            merged_linestring = entry['geometry']  # 直接使用存储的 merged_linestring
            
            road_shortest_step_mindist = get_road_shortest_step_MINdistance(g, last_point_road, path, roads_selectrange_set)
            
            # print(f'road_shortest_step_mindist:{road_shortest_step_mindist}')
            
            if road_shortest_step_mindist is None or road_shortest_step_mindist > road_shortest_dist:
                continue
                
            # 投影点计算：将 between_rows 中的点投影到 merged_linestring 上
            projected_points = []
            distances = []
            for point in between_rows['geometry']:
                # 计算点到 merged_linestring 的最近点
                projected_point = merged_linestring.interpolate(merged_linestring.project(point))
                projected_points.append(projected_point)
                
                # 计算点到最近点的距离
                distance = point.distance(projected_point)
                distances.append(distance)
            
            # 计算 distances 的总长度
            total_distance = sum(distances)

        # 计算 projected_points 的首尾点向量并计算余弦相似度
            # 获取首尾点
            start_point_projected = projected_points[0]
            end_point_projected = projected_points[-1]
            # 计算首尾点在 merged_linestring 上的投影长度
            start_dist = merged_linestring.project(start_point_projected)
            end_dist = merged_linestring.project(end_point_projected)
            if start_dist > end_dist:
                start_point_projected , end_point_projected = end_point_projected , start_point_projected
            vector_projected = np.array([end_point_projected.x - start_point_projected.x,
                                            end_point_projected.y - start_point_projected.y])
            # 计算余弦相似度
            cosine_sim = cosine_similarity(vector_raw, vector_projected)
            # print(f"Cosine Similarity between vectors: {cosine_sim}")
            
            # 如果当前路径的总距离小于之前的最短距离，则更新
            if total_distance < shortest_distance_sum:
                shortest_distance_sum = total_distance
                shortest_projection_points = projected_points
                shortest_path = path  # 更新为当前最短路径
                shortest_merged_linestring = merged_linestring  # 保存最短路径的线段
                prev_cosine_sim = cosine_sim  # 更新余弦相似度
            # ② **如果当前路径的总距离和之前的最短距离几乎相等（接近 tolerance），则比较余弦相似度**
            elif abs(total_distance - shortest_distance_sum) < tolerance:
                if cosine_sim > prev_cosine_sim:  # 如果当前余弦相似度更优，则更新
                    shortest_distance_sum = total_distance
                    shortest_projection_points = projected_points
                    shortest_path = path  # 更新为当前最短路径
                    shortest_merged_linestring = merged_linestring  # 保存最短路径的线段
                    prev_cosine_sim = cosine_sim  # 更新余弦相似度
                
        

    # 确定 shortest_projection_points 的首尾点对应的道路
    if shortest_projection_points and shortest_path is not None:
        # 获取首尾点
        first_point = shortest_projection_points[0]
        last_point = shortest_projection_points[-1]

        # 计算首尾点在 merged_linestring 上的投影距离
        start_dist = shortest_merged_linestring.project(first_point)
        end_dist = shortest_merged_linestring.project(last_point)

        # 如果 shortest_path 只有一个元素
        if len(shortest_path) == 1:
            last_point_road = shortest_path[0]
            
        else:
            if start_dist > end_dist:
                start_dist, end_dist = end_dist, start_dist
                last_point_road = shortest_path[0]  # 顺序相反，取第一个点对应的道路
            else:
                last_point_road = shortest_path[-1]  # 顺序一致，取最后一个点对应的道路

        # 处理 MultiLineString 的逻辑
        if isinstance(shortest_merged_linestring, MultiLineString):
            shortest_merged_linestring = handle_multilinestring(shortest_merged_linestring, first_point, last_point, road_gdf.crs)

        # 提取 substring（部分线段）
        projected_lines = ops.substring(shortest_merged_linestring, start_dist, end_dist)

    return shortest_path, last_point_road, projected_lines

def classify_congestion_road(v_acc):
    # 初始化状态
    state = 'none'
    
    # 根据 v_acc 设置对应的拥堵等级
    if v_acc > 8.33:
        state = 'smooth'
    elif 5.56 <= v_acc <= 8.33:
        state = 'light_congestion'
    elif 2.78 <= v_acc < 5.56:
        state = 'congestion'
    elif v_acc < 2.78:
        state = 'severe_congestion'
    
    # 返回状态
    return state