import folium
from folium.plugins import MarkerCluster
from coord_convert.transform import wgs2gcj  # Convert WGS-84 to GCJ-02
from PIL import Image, ImageDraw, ImageFont

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

import geopandas as gpd
import pandas as pd
import os
from shapely import wkt
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPoint, GeometryCollection

# WGS-84 → GCJ-02 坐标转换函数
def convert_to_gcj02(geometry):
    if geometry.geom_type == "Point":
        lng, lat = wgs2gcj(geometry.x, geometry.y)
        return Point(lng, lat)  # 生成新的 Point 对象
    elif geometry.geom_type == "Polygon":
        coords = [wgs2gcj(x, y) for x, y in geometry.exterior.coords]
        return Polygon(coords)  # 生成新的 Polygon 对象
    elif geometry.geom_type == "LineString":  # Handling Polyline
        coords = [wgs2gcj(x, y) for x, y in geometry.coords]
        return LineString(coords)  # Generate a new LineString object
    return geometry  # 其他类型保持不变

# 获取地图中心点（取 honeycomb_id 的平均坐标）
def get_map_center(gdf):
    if not gdf.empty:
        return [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
    return [31.2304, 121.4737]  # 默认上海坐标

# 截图为 PNG
def save_map_as_image(html_path, img_path):
    """使用 Selenium 将 HTML 地图截图为 PNG"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 无头模式
    options.add_argument("--window-size=1200x800")
    driver = webdriver.Chrome(service=Service("D:\myJupyter\paper2\chromedriver-win64\chromedriver.exe"), options=options)  
    file_url = f"{os.path.abspath(html_path)}"
    driver.get(file_url)
    time.sleep(2)  # 等待加载
    driver.save_screenshot(img_path)
    driver.quit()

def crop_and_add_title(input_img_path, output_img_path, title_text, left=150, top=220, right=750, bottom=750, title_height=40):
    """裁剪 PNG 图像并添加标题"""
    # 加载图片
    img = Image.open(input_img_path)
    # 定义裁剪区域（left, top, right, bottom）
    crop_box = (left, top, right, bottom)
    # 裁剪图像
    cropped_img = img.crop(crop_box)
    # 获取裁剪后图像的大小
    width, height = cropped_img.size
    # 创建一个新的图像，比裁剪后的图像高，增加的空间用来放标题
    new_img = Image.new('RGB', (width, height + title_height), (255, 255, 255))  # 白色背景
    # 将裁剪后的图像粘贴到新的图像下方
    new_img.paste(cropped_img, (0, title_height))
    # 创建一个可以在图像上绘制的对象
    draw = ImageDraw.Draw(new_img)
    # 设置字体和标题位置（你可以根据需要调整字体大小和颜色）
    try:
        font = ImageFont.truetype("arialbd.ttf", 12)  # 使用 Arial 字体，大小 30
    except IOError:
        font = ImageFont.load_default()
    # 使用 textbbox 计算文本大小
    bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # 设置标题文本颜色和位置（居中）
    position = ((width - text_width) / 2, 5)  # 使文本居中，距离顶部一定距离
    # 在新的图像顶部绘制文本
    draw.text(position, title_text, font=font, fill="black")
    # 保存修改后的图像
    new_img.save(output_img_path)
    
def plot_traffic_congestion(honeycomb_ids_ls, road_ids_ls, poi_ids_ls, honeycomb_list, roadcrs, g, plot_type): 
    if honeycomb_ids_ls is not None:
        honeycomb_ids_str = ', '.join(f"{honeycomb_id}" for honeycomb_id in honeycomb_ids_ls)
        query = f"""
                MATCH (h:honeycomb)
                WHERE h.name IN [{honeycomb_ids_str}]
                RETURN h.name AS honeycomb_id, h.geometry AS honeycomb_geometry
                """
        honeycomb_ids_result = g.run(query).data()
        honeycomb_ids_resultdf = pd.DataFrame(honeycomb_ids_result)
        honeycomb_ids_resultdf['honeycomb_geometry'] = honeycomb_ids_resultdf['honeycomb_geometry'].apply(wkt.loads)
        honeycomb_ids_result_gdf = gpd.GeoDataFrame(honeycomb_ids_resultdf, geometry='honeycomb_geometry', crs=roadcrs)
        honeycomb_listdf = pd.DataFrame(honeycomb_list)
    if road_ids_ls is not None:
        road_ids_str = ', '.join(f"{road_id}" for road_id in road_ids_ls)
        query = f"""
                MATCH (r:road)
                WHERE r.name IN [{road_ids_str}]
                RETURN r.name AS road_id, r.roadname AS road_name, r.geometry AS road_geometry
                """
        road_ids_result = g.run(query).data()
        road_ids_resultdf = pd.DataFrame(road_ids_result)
        road_ids_resultdf['road_geometry'] = road_ids_resultdf['road_geometry'].apply(wkt.loads)
        road_ids_result_gdf = gpd.GeoDataFrame(road_ids_resultdf, geometry='road_geometry', crs=roadcrs)       
    if poi_ids_ls is not None:
        poi_ids_str = ', '.join(f"{poi_id}" for poi_id in poi_ids_ls)
        query = f"""
                MATCH (p:POI)
                WHERE p.name IN [{poi_ids_str}]
                RETURN p.name AS poi_id, p.poi_name AS poi_name, p.geometry AS poi_geometry
                """
        poi_ids_result = g.run(query).data()
        poi_ids_resultdf = pd.DataFrame(poi_ids_result)
        poi_ids_resultdf['poi_geometry'] = poi_ids_resultdf['poi_geometry'].apply(wkt.loads)
        poi_ids_result_gdf = gpd.GeoDataFrame(poi_ids_resultdf, geometry='poi_geometry', crs=roadcrs)   
        
    # **转换数据坐标系到 EPSG:4326（Web Mercator）以适配底图**
    honeycomb_ids_result_gdf = honeycomb_ids_result_gdf.to_crs(epsg=4326)
    road_ids_result_gdf = road_ids_result_gdf.to_crs(epsg=4326)
    poi_ids_result_gdf = poi_ids_result_gdf.to_crs(epsg=4326)
    
    # Convert all geometries to GCJ-02
    honeycomb_ids_result_gdf["honeycomb_geometry"] = honeycomb_ids_result_gdf["honeycomb_geometry"].apply(convert_to_gcj02)
    road_ids_result_gdf["road_geometry"] = road_ids_result_gdf["road_geometry"].apply(convert_to_gcj02)
    poi_ids_result_gdf["poi_geometry"] = poi_ids_result_gdf["poi_geometry"].apply(convert_to_gcj02)
        
    # 颜色映射字典（Nature 期刊配色 + 透明度 40%）
    color_map = {
        "smooth": "#38A700",
        "light_congestion": "#FFFF00",
        "congestion": "#FFAA06",
        "severe_congestion": "#FE5401"
    }
    
    # 按 batch 分组
    grouped = honeycomb_listdf.groupby('batch')
    
    if plot_type == 'grid congestion':
        # 遍历每个 batch 进行绘图
        for batch_name, group in grouped:
            # print(f"Processing batch\n: {batch_name}")
            # print(f"group\n: {group}")
            # 使用 inner join 只保留符合 expanded_road 的数据
            merged_grid = group.merge(honeycomb_ids_result_gdf, on='honeycomb_id', how='right')
            # print(f"merged_grid\n: {merged_grid}")
            # 获取中心点
            map_center = get_map_center(poi_ids_result_gdf)
        
            # 创建 Folium 地图
            m = folium.Map(location=map_center, zoom_start=16)
            folium.TileLayer(
                tiles="https://webrd02.is.autonavi.com/appmaptile?lang=en&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
                attr="高德-纯英文对照"
            ).add_to(m)
        
            # 绘制 honeycomb（Polygon）
            for _, row in merged_grid.iterrows():
                if row["honeycomb_geometry"].geom_type == "Polygon":
                    level = row["grid_congestion_level"]
                    if level not in color_map:
                        continue  # ❌ 不在 color_map 中 → 跳过绘制
                    fill_color = color_map[level]
                    folium.Polygon(
                        locations=[(y, x) for x, y in row["honeycomb_geometry"].exterior.coords],
                        color="rgba(69, 117, 149, 0.4)",
                        weight=0.5,
                        fill_color=fill_color,
                        fill_opacity=0.4,
                        zIndexOffset=0,
                        popup=f"Honeycomb ID: {row['honeycomb_id']}\nCongestion: {row['grid_congestion_level']}"
                    ).add_to(m)
        
            # 绘制道路 (road_geometry)
            if not road_ids_result_gdf.empty:
                for _, row in road_ids_result_gdf.iterrows():
                    road_coords = list(row["road_geometry"].coords)
                    folium.PolyLine(
                        locations=[(y, x) for x, y in road_coords],
                        color="rgba(0, 0, 0, 0.4)",
                        weight=2,
                        zIndexOffset=1,
                        tooltip=row["road_name"]
                    ).add_to(m)
        
            # 绘制 POI (poi_geometry)
            if not poi_ids_result_gdf.empty:
                poi_cluster = MarkerCluster().add_to(m)
                for _, row in poi_ids_result_gdf.iterrows():
                    folium.Marker(
                        location=(row["poi_geometry"].y, row["poi_geometry"].x),
                        icon=folium.Icon(color="blue", icon="info-sign"),
                        zIndexOffset=2,
                        tooltip=row["poi_name"]
                    ).add_to(poi_cluster)
                    
            # **缩放至所有绘制对象**
            m.fit_bounds(m.get_bounds())  # 🚀 自动缩放到所有对象
        
            # 设置地图标题
            title_html = f"""
            <h3 align="center" style="font-size:16px"><b>Batch: {batch_name}</b></h3>
            """
            m.get_root().html.add_child(folium.Element(title_html))
            # 打印 Folium 地图对象
            print(f"Map for batch: {batch_name}")
            display(m)  # 适用于 Jupyter Notebook
                        
            # **保存 HTML & 转换为 png**
            output_folder="F:\\paper2\\result\\maps\\grid_congestion"
            os.makedirs(output_folder, exist_ok=True)
            html_path = os.path.join(output_folder, f"{batch_name.replace(':', '_')}.html")
            img_path = os.path.join(output_folder, f"{batch_name.replace(':', '_')}.png")
            m.save(html_path)
            save_map_as_image(html_path, img_path)

            # **裁剪保存的图像**
            cropped_img_with_title_path = os.path.join(output_folder, f"{batch_name.replace(':', '_')}_cropped_with_title.png")
            crop_and_add_title(img_path, cropped_img_with_title_path, batch_name)
    
    if plot_type == 'road congestion':
        # 遍历每个 batch 进行绘图
        for batch_name, group in grouped:
            # print(f"Processing batch\n: {batch_name}")
            # print(f"group\n: {group}")
            # 提取 group 中所有 honeycomb_id
            group_honeycomb_ids = group["honeycomb_id"].unique()
            # 过滤 honeycomb_ids_result_gdf，只保留当前 group 中出现的 honeycomb
            filtered_honeycomb = honeycomb_ids_result_gdf[
                honeycomb_ids_result_gdf["honeycomb_id"].isin(group_honeycomb_ids)
            ]
            # 展开 shortest_path，使每行只对应一个 road_id
            expanded_road = group[['batch', 'shortest_path', 'road_congestion_level']].explode('shortest_path')
            expanded_road = expanded_road.rename(columns={'shortest_path': 'road_id'})
            roads_in_grid = gpd.sjoin(
                road_ids_result_gdf,
                filtered_honeycomb[["honeycomb_id", "honeycomb_geometry"]],
                how="inner",
                predicate="intersects"  # 也可以用 "within"，取决于你定义
            )
            # 使用 inner join 只保留符合 expanded_road 的数据
            merged_road = expanded_road.merge(roads_in_grid, on='road_id', how='inner')
            # 获取未在 merged_road 中的道路（即非拥堵道路）
            non_congested_roads = roads_in_grid[
                ~roads_in_grid["road_id"].isin(expanded_road["road_id"])
            ]
            
        # 绘图
            # 获取中心点
            map_center = get_map_center(poi_ids_result_gdf)
                
            # 创建 Folium 地图
            m = folium.Map(location=map_center, zoom_start=16)
            folium.TileLayer(
                tiles="https://webrd02.is.autonavi.com/appmaptile?lang=en&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
                attr="高德-纯英文对照"
            ).add_to(m)   
                
            # 绘制道路 (road_geometry)
            if not non_congested_roads.empty:
                for _, row in non_congested_roads.iterrows():
                    road_coords = list(row["road_geometry"].coords)
                    folium.PolyLine(
                        locations=[(y, x) for x, y in road_coords],
                        color="rgba(0, 0, 0, 1.0)",
                        weight=2,
                        zIndexOffset=0,
                        tooltip=row["road_name"]
                    ).add_to(m)
                
            # 绘制拥堵道路 (road_geometry)
            if not merged_road.empty:
                for _, row in merged_road.iterrows():
                    road_level = row["road_congestion_level"]
                    # ❌ 不在 color_map 中就跳过
                    if road_level not in color_map:
                        continue
                    road_coords = list(row["road_geometry"].coords)
                    road_color = color_map[road_level]
                    folium.PolyLine(
                        locations=[(y, x) for x, y in road_coords],
                        color=road_color,
                        weight=2,
                        # opacity=0.8,  # 设置透明度
                        zIndexOffset=1,
                        tooltip=row["road_name"],
                        popup=f"Road ID: {row['road_id']}\nCongestion: {row['road_congestion_level']}"
                    ).add_to(m)
                
            # 绘制 honeycomb（Polygon）
            if not filtered_honeycomb.empty:
                for _, row in filtered_honeycomb.iterrows():
                    folium.Polygon(
                        locations=[(y, x) for x, y in row["honeycomb_geometry"].exterior.coords],  # 转换坐标格式
                        color="rgba(69, 117, 149, 0.8)",  # 边框颜色（深蓝色）
                        weight=1,  # 线条粗细
                        fill=False,  # ❌ 取消填充，只显示边框
                        zIndexOffset=2,
                        popup=f"Honeycomb ID: {row['honeycomb_id']}"
                    ).add_to(m)  # 添加到地图
                
            # 绘制 POI (poi_geometry)
            if not poi_ids_result_gdf.empty:
                poi_cluster = MarkerCluster().add_to(m)
                for _, row in poi_ids_result_gdf.iterrows():
                    folium.Marker(
                        location=(row["poi_geometry"].y, row["poi_geometry"].x),
                        icon=folium.Icon(color="blue", icon="info-sign"),
                        zIndexOffset=3,
                        tooltip=row["poi_name"]
                    ).add_to(poi_cluster)
                        
            # **缩放至所有绘制对象**
            m.fit_bounds(m.get_bounds())  # 🚀 自动缩放到所有对象
        
            # 设置地图标题
            title_html = f"""
            <h3 align="center" style="font-size:16px"><b>Batch: {batch_name}</b></h3>
            """
            m.get_root().html.add_child(folium.Element(title_html))
            
            # 打印 Folium 地图对象
            print(f"Map for batch: {batch_name}")
            display(m)  # 适用于 Jupyter Notebook
                        
            # **保存 HTML & 转换为 png**
            output_folder="F:\\paper2\\result\\maps\\road_congestion"
            os.makedirs(output_folder, exist_ok=True)
            html_path = os.path.join(output_folder, f"{batch_name.replace(':', '_')}.html")
            img_path = os.path.join(output_folder, f"{batch_name.replace(':', '_')}.png")
            m.save(html_path)
            save_map_as_image(html_path, img_path)

            # **裁剪保存的图像**
            cropped_img_with_title_path = os.path.join(output_folder, f"{batch_name.replace(':', '_')}_cropped_with_title.png")
            crop_and_add_title(img_path, cropped_img_with_title_path, batch_name)
