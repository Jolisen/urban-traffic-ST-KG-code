import pandas as pd
import os

def save_honeycomb_list_to_file(honeycomb_list, output_path, file_type="xlsx"):
    """
    将 honeycomb_list 保存为表格文件（CSV 或 Excel）

    Args:
        honeycomb_list (list[dict]): 格式如 [{'honeycomb_id': ..., 'batch': ..., 'grid_congestion_level': ...}, ...]
        output_path (str): 文件保存的完整路径（不带后缀）
        file_type (str): 文件类型 "csv" 或 "xlsx"
    """
    try:
        df = pd.DataFrame(honeycomb_list)
        print(f"df:\n{df}")

        # 自动创建目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if file_type == "csv":
            file_name = output_path + ".csv"
            df.to_csv(file_name, index=False, encoding='utf-8-sig')
        elif file_type == "xlsx":
            file_name = output_path + ".xlsx"
            df.to_excel(file_name, index=False)
        else:
            raise ValueError("仅支持文件类型 'csv' 或 'xlsx'")

        print(f"✅ 文件已保存: {file_name}")

    except Exception as e:
        print(f"❌ 保存失败: {e}")