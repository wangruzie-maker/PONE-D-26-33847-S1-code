# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import re
import pandas as pd


def extract_fields(lines, patterns):
    """从多行文本中提取指定字段"""
    extracted_data = []
    current_entry = {}
    in_abstract = False

    for line in lines:
        line = line.strip()

        # 检查是否遇到了新的记录开始标志（如PT）
        if line.startswith('PT'):
            if current_entry:  # 如果当前条目不为空，则添加到结果列表
                extracted_data.append(current_entry)
                current_entry = {}
                in_abstract = False

        matched_field = None

        # 尝试匹配所有模式，找到第一个匹配的字段
        for field, pattern in patterns.items():
            match = re.match(pattern, line)
            if match:
                matched_field = field
                break

        if matched_field == 'AB':  # 对于摘要，累积多行内容直到遇到下一个字段标识符
            in_abstract = True
            if 'AB' not in current_entry:
                current_entry['AB'] = match.group(1).strip()
            else:
                current_entry['AB'] += ' ' + match.group(1).strip()
        elif in_abstract and not any(line.startswith(key) for key in patterns.keys()):
            # 如果在摘要模式中且当前行不是其他字段的开头，则继续累积摘要
            current_entry['AB'] += ' ' + line.strip()
        elif matched_field:
            # 如果匹配到了其他字段，则停止累积摘要，并更新current_entry
            in_abstract = False
            current_entry[matched_field] = match.group(1).strip()

    # 添加最后一个条目（如果有的话）
    if current_entry:
        extracted_data.append(current_entry)

    return extracted_data


def clean_wos_data(input_path, output_excel_path):
    """清洗WOS数据并导出包含摘要和发表时间的记录到Excel"""
    # 定义要提取的字段及其正则表达式模式
    field_patterns = {
        'PY': r'^PY\s+(\d{4})',  # 发表年份
        'AB': r'^AB\s+(.*)',  # 摘要
        'PD': r'^PD\s+(.*)',  # 其他字段标识符，用于确定摘要结束
        'PT': r'^PT\s+(.*)',  # 记录开始标识符
        # 可以在这里添加更多的字段标识符以确保准确性
    }

    # 读取原始文件
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 提取摘要和发表日期
    extracted_data = extract_fields(lines, field_patterns)

    # 将提取的数据转换为DataFrame
    df = pd.DataFrame(extracted_data)

    # 重命名列以更清晰地表示内容
    df.rename(columns={'PY': '发表年份', 'AB': '摘要'}, inplace=True)

    # 保存到Excel文件
    df.to_excel(output_excel_path, index=False, engine='openpyxl')

    print(f"Cleaned data has been saved to {output_excel_path}")


# 定义路径
input_file_path = r'C:\Users\15610\Desktop\城市形象传播wos2.txt'
output_excel_path = r'C:\Users\15610\Desktop\外文.xlsx'

# 执行清洗和导出
clean_wos_data(input_file_path, output_excel_path)