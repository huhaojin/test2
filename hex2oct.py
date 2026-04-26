import pandas as pd
import numpy as np
import re


def hex_to_excel(input_file, output_file, rows, cols):
    print("1. 正在读取文件...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # 如果是纯二进制文件并非文本，尝试以二进制读取（备选方案）
        print("警告：文本读取失败，尝试作为纯文本解析...")
        return

    # 2. 数据清洗
    # 去除所有空格、换行符、制表符，只保留十六进制字符
    print("2. 正在清洗数据...")
    # 使用正则表达式替换掉非十六进制的字符
    clean_hex = re.sub(r'[^0-9a-fA-F]', '', content)

    # 检查数据长度是否足够
    # 每个数字由2个字节组成（例如 45 92），在清理后的字符串中占 4 个字符
    expected_chars = rows * cols * 4
    if len(clean_hex) != expected_chars:
        print(f"注意：数据长度不完全匹配。")
        print(f"预期字符数: {expected_chars}, 实际字符数: {len(clean_hex)}")
        # 这里你可以选择截断或者报错，目前代码选择截断或处理现有所有数据

    print("3. 正在转换进制 (Hex -> Decimal)...")
    decimal_data = []

    # 每 4 个字符截取一次 (例如 '4592') 并转换为十进制
    # range(start, stop, step)
    for i in range(0, len(clean_hex), 4):
        chunk = clean_hex[i:i + 4]
        if len(chunk) < 4:
            break  # 忽略最后不足一位的数据
        # int(str, 16) 将16进制字符串转为10进制
        val = int(chunk, 16)
        decimal_data.append(val)

    print(f"转换完成，共得到 {len(decimal_data)} 个数字。")

    # 4. 重塑矩阵 (Reshape)
    # 确保数据量能够被 reshape，如果数据多了或少了，numpy会报错
    # 这里我们只取前 rows * cols 个数据以防万一
    total_needed = rows * cols
    if len(decimal_data) >= total_needed:
        data_matrix = np.array(decimal_data[:total_needed]).reshape(rows, cols)

        print(f"4. 正在保存到 Excel ({output_file})... 这可能需要几秒钟")
        # 转换为 DataFrame
        df = pd.DataFrame(data_matrix)

        # 保存为 Excel，不包含行索引和列头
        df.to_excel(output_file, index=False, header=False, engine='openpyxl')
        print("成功！文件已保存。")
    else:
        print(f"错误：数据量不足以填充 {rows}x{cols} 的矩阵。")
        print(f"现有数据: {len(decimal_data)}, 需要数据: {total_needed}")


# --- 配置区 ---
INPUT_FILENAME = 'data.hex'  # 你的源文件名 (如果是txt也改成对应的名字)
OUTPUT_FILENAME = 'output.xlsx'  # 输出的Excel文件名
ROWS = 656  # 行数
COLS = 514  # 列数

# 执行函数
if __name__ == '__main__':
    hex_to_excel(INPUT_FILENAME, OUTPUT_FILENAME, ROWS, COLS)