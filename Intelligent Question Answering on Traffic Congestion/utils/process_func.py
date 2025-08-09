import json
import ast
import pandas as pd
import re

def is_datetime_with_date(time_str):
    """判断时间字符串是否包含年月日"""
    return bool(re.match(r"\d{4}-\d{2}-\d{2}", time_str))

# 解析 query_results
def parse_query_results(query_results):
    """
    尝试解析 query_results:
    - 如果是 JSON 格式字符串，使用 `json.loads()`
    - 如果是 Python `str(list)`，使用 `ast.literal_eval()`
    
    Args:
        query_results (str | list | dict): 需要解析的数据
    
    Returns:
        list | dict: 解析成功返回原始 JSON 数据（list 或 dict），失败返回 {"error": "..."}
    """
    # 如果 `query_results` 已经是 `list` 或 `dict`，直接返回
    if isinstance(query_results, (list, dict)):
        return query_results

    # 如果 `query_results` 是字符串，尝试解析
    if isinstance(query_results, str):
        try:
            # 先尝试解析 JSON 格式
            parsed_data = json.loads(query_results)
            print(f"\n✅ [DEBUG] JSON 格式解析成功")
            return parsed_data
        except json.JSONDecodeError:
            try:
                # 如果 JSON 解析失败，尝试 `ast.literal_eval()` 解析 Python `str(list)`
                parsed_data = ast.literal_eval(query_results)
                print(f"\n✅ [DEBUG] Python 列表解析成功")
                return parsed_data
            except (SyntaxError, ValueError):
                print(f"\n❌ [ERROR] query_results 不是有效的 JSON 或 Python 列表: {query_results}\n")
                return {"error": "query_results 不是有效的 JSON 格式"}

    # 如果 `query_results` 不是字符串、列表或字典，则返回错误
    return {"error": "query_results 不是有效的格式"}

def calculate_df_summary(df_sorted):
    """
    使用 NumPy 优化 groupby 聚合过程
    """
    # 分组并用 NumPy 操作实现聚合
    grouped = df_sorted.groupby(['batch', 'vehicle', 'grid_congestion_level'])
    min_name = grouped['index_helper'].min()
    max_name = grouped['index_helper'].max()
    count = max_name - min_name + 1

    # 创建 DataFrame
    df_summary = pd.DataFrame({
        'vehicle': min_name.index.get_level_values('vehicle'),
        'batch': min_name.index.get_level_values('batch'),
        'grid_congestion_level': min_name.index.get_level_values('grid_congestion_level'),
        'min_name': min_name.values,
        'max_name': max_name.values,
        'count': count.values
    })
    # 过滤 count > 1
    df_summary = df_summary[df_summary['count'] > 1]
    # 按 start_time 排序
    df_summary = df_summary.sort_values(by='min_name', ignore_index=True)
    return df_summary
    













    