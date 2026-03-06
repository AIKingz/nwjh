import hashlib
import json
import os
import random
import re
import string

import cv2

# def get_sub_rectangles(rect, x, indices, direction='horizontal'):
#     """
#     获取切分后的某些编号的小矩形的四个顶点坐标（编号从0开始）
#     坐标值会被四舍五入为整数。
#
#     参数:
#         rect: 原始矩形的四个顶点坐标，格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
#               顺序为 左上 -> 右上 -> 右下 -> 左下
#         x: 切分数量
#         indices: 要提取的小矩形编号列表（从0开始）
#         direction: 'horizontal' 或 'vertical'
#
#     返回:
#         list of lists: 每个元素是一个小矩形的四个顶点坐标（整数）
#     """
#     # 提取矩形的边界
#     left_top = rect[0]
#     right_top = rect[1]
#     right_bottom = rect[2]
#     left_bottom = rect[3]
#
#     width = right_top[0] - left_top[0]
#     height = right_bottom[1] - right_top[1]
#
#     result = []
#
#     for idx in indices:
#         if not (0 <= idx < x):
#             raise ValueError(f"Index {idx} out of range [0, {x - 1}]")
#
#         if direction == 'horizontal':
#             segment_width = width / x
#             start_x = left_top[0] + idx * segment_width
#             end_x = start_x + segment_width
#
#             top_left = [round(start_x), round(left_top[1])]
#             top_right = [round(end_x), round(right_top[1])]
#             bottom_right = [round(end_x), round(right_bottom[1])]
#             bottom_left = [round(start_x), round(left_bottom[1])]
#
#         elif direction == 'vertical':
#             segment_height = height / x
#             start_y = left_top[1] + idx * segment_height
#             end_y = start_y + segment_height
#
#             top_left = [round(left_top[0]), round(end_y)]
#             top_right = [round(right_top[0]), round(end_y)]
#             bottom_right = [round(right_bottom[0]), round(start_y)]
#             bottom_left = [round(left_bottom[0]), round(start_y)]
#
#         else:
#             raise ValueError("Direction must be 'horizontal' or 'vertical'")
#
#         result.append([top_left, top_right, bottom_right, bottom_left])
#
#     return result


def get_sub_rectangles(rect, x, indices, direction='horizontal'):
    """
    获取切分后的某些编号的小矩形的四个顶点坐标（编号从0开始）
    坐标值会被四舍五入为整数。

    参数:
        rect: 原始矩形的四个顶点坐标，格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
              顺序为 左上 -> 右上 -> 右下 -> 左下
        x: 切分数量
        indices: 要提取的小矩形编号列表（从0开始）
        direction: 'horizontal' 或 'vertical'

    返回:
        list of lists: 每个元素是一个小矩形的四个顶点坐标（整数）
    """

    # x = x+1

    # print('输入参数为', 'rect:', rect, 'x:', x, 'indices:', indices, 'direction:', direction)
    # 提取矩形的边界
    left_top = rect[0]
    right_top = rect[1]
    right_bottom = rect[2]
    left_bottom = rect[3]

    width = right_top[0] - left_top[0]
    height = right_bottom[1] - right_top[1]

    result = []

    # 如果长度为1，直接返回原始矩形的整数化坐标
    if len(indices) == 1:
        return [[
            [round(left_top[0]), round(left_top[1])],
            [round(right_top[0]), round(right_top[1])],
            [round(right_bottom[0]), round(right_bottom[1])],
            [round(left_bottom[0]), round(left_bottom[1])]
        ]]

    # print('indices:', len(indices), indices)
    for i, idx in enumerate(indices):
        # print('i:', i, 'idx:', idx, 'indices:', indices, 'x:', x, 'direction:', direction)
        if not (0 <= idx < x):
            # continue
            raise ValueError(f"Index {idx} out of range [0, {x - 1}]")

        # 特殊处理第一个和最后一个 index
        if i == 0:
            # adjusted_idx = max(idx - 1, 0)
            adjusted_idx = 0
        elif i == len(indices) - 1:
            # adjusted_idx = min(idx + 1, x - 1)
            adjusted_idx = x-1
        else:
            adjusted_idx = idx

        if direction == 'horizontal':
            segment_width = width / x
            start_x = left_top[0] + adjusted_idx * segment_width
            end_x = start_x + segment_width

            top_left = [round(start_x), round(left_top[1])]
            top_right = [round(end_x), round(right_top[1])]
            bottom_right = [round(end_x), round(right_bottom[1])]
            bottom_left = [round(start_x), round(left_bottom[1])]

        elif direction == 'vertical':
            segment_height = height / x
            start_y = left_top[1] + adjusted_idx * segment_height
            end_y = start_y + segment_height

            top_left = [round(left_top[0]), round(end_y)]
            top_right = [round(right_top[0]), round(end_y)]
            bottom_right = [round(right_bottom[0]), round(start_y)]
            bottom_left = [round(left_bottom[0]), round(start_y)]

        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")

        result.append([top_left, top_right, bottom_right, bottom_left])

    return result

# def paddle2rappid(paddle_result):
#     """
#     将PaddleOCR的结果转换为RapidOCR的格式
#     """

#     # print('开始转换为rappid格式')
#     # print('paddle_result:', len(paddle_result))

#     rappid_result = []
#     print('开始转换为rappid格式')

#     # 如果不是list，直接返回
#     if type(paddle_result) != list:
#         return rappid_result



#     for i in range(len(paddle_result)):
#         # print('第', i, '个paddle_result:', paddle_result[i])

#         paddle = paddle_result[i]

#         box = paddle[0]
#         info = paddle[1]

#         text, conf, single_info = info
#         spilt_num = int(single_info[0]) + 1
# #TODO 订单合同
# # Traceback (most recent call last):
# #   File "/root/autodl-tmp/fujianwuzi/ocr.py", line 108, in get_ocr_image
# #     ocr_result = get_ocr(image_path, is_gpu=is_gpu, json_save_path=json_save_path)
# #   File "/root/autodl-tmp/fujianwuzi/ocr.py", line 39, in get_ocr
# #     ocr_result = paddle2rappid(ocr_result)
# #   File "/root/autodl-tmp/fujianwuzi/ocr_utils.py", line 178, in paddle2rappid
# #     single_text = single_info[1][0]
#         single_text = single_info[1][0]
#         points = single_info[2][0]

#         # print('text:', text,'split_num:', spilt_num,
#         #       'points:', points)
#         # print('处理第', i, '个paddle_result:')
#         single_rec = get_sub_rectangles(box, spilt_num, points, 'horizontal')

#         rappid = [
#             box,
#             text,
#             conf,
#             single_rec,
#             single_text
#         ]

#         rappid_result.append(rappid)

#     print('转换为rappid格式完成')

#     return rappid_result


def paddle2rappid(paddle_result):
    """
    将PaddleOCR的结果转换为RapidOCR的格式
    """
    rappid_result = []
    print('开始转换为rappid格式')

    if not isinstance(paddle_result, list):
        return rappid_result

    for paddle in paddle_result:
        try:
            box = paddle[0]
            info = paddle[1]
            text, conf, single_info = info
            spilt_num = int(single_info[0]) + 1
            
            # --- Safely access nested data here ---
            # Check if single_info has at least 2 elements and the second is a non-empty list
            if len(single_info) > 1 and len(single_info[1]) > 0:
                single_text = single_info[1][0]
            else:
                # Handle cases where single_info[1] is missing or empty
                single_text = ""
                print(f"Warning: Missing single_text info for result: {text}")

            # Check if single_info has at least 3 elements and the third is a non-empty list
            if len(single_info) > 2 and len(single_info[2]) > 0:
                points = single_info[2][0]
            else:
                # Handle cases where points are missing
                points = []
                print(f"Warning: Missing points info for result: {text}")

            single_rec = get_sub_rectangles(box, spilt_num, points, 'horizontal')

            rappid = [
                box,
                text,
                conf,
                single_rec,
                single_text
            ]
            rappid_result.append(rappid)
        except Exception as e:
            # This catch-all block is a last resort to prevent crashes
            # and helps pinpoint problematic data.
            print(f"Error processing a PaddleOCR result: {e}")
            print(f"Problematic data structure: {paddle}")
            continue # Skips to the next item

    print('转换为rappid格式完成')
    return rappid_result

def split_ocr_results(ocr_results, threshold):
    """
    对OCR识别结果进行切分，当相邻数字字符的距离大于阈值时，将文字识别框切分为两个。
    :param ocr_results: OCR识别结果，格式为文字识别框构成的List。
    :param threshold: 距离阈值，用于判断是否切分。
    :return: 切分后的OCR识别结果。
    """
    def is_digit_pair(char1, char2):
        """判断两个字符是否都是数字字符。"""
        return char1.isdigit() and char2.isdigit()

    def calculate_distance(coord1, coord2):
        """计算两个坐标之间的欧氏距离。"""
        x1, y1 = coord1
        x2, y2 = coord2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def create_new_box(box_coords, char_coords, text, confidence, chars, split_index):
        """
        创建一个新的文字识别框。
        :param box_coords: 原始文字识别框的坐标。
        :param char_coords: 单字坐标列表。
        :param text: 原始文本。
        :param confidence: 置信度。
        :param chars: 单字字符列表。
        :param split_index: 切分位置的索引。
        :return: 新的文字识别框。
        """
        new_box_coords = [
            [box_coords[0][0], box_coords[0][1]],
            [char_coords[split_index][1][0], box_coords[0][1]],
            [char_coords[split_index][2][0], box_coords[2][1]],
            [box_coords[3][0], box_coords[3][1]]
        ]
        return [
            new_box_coords,
            text[:split_index + 1],
            confidence,
            char_coords[:split_index + 1],
            chars[:split_index + 1]
        ]

    def update_existing_box(box_coords, char_coords, text, confidence, chars, split_index):
        """
        更新原始文字识别框。
        :param box_coords: 原始文字识别框的坐标。
        :param char_coords: 单字坐标列表。
        :param text: 原始文本。
        :param confidence: 置信度。
        :param chars: 单字字符列表。
        :param split_index: 切分位置的索引。
        :return: 更新后的文字识别框。
        """
        return [
            [
                [char_coords[split_index + 1][0][0], box_coords[0][1]],
                [box_coords[1][0], box_coords[1][1]],
                [box_coords[2][0], box_coords[2][1]],
                [char_coords[split_index + 1][3][0], box_coords[3][1]]
            ],
            text[split_index + 1:],
            confidence,
            char_coords[split_index + 1:],
            chars[split_index + 1:]
        ]

    # 遍历每个文字识别框
    i = 0
    while i < len(ocr_results):
        ocr_box = ocr_results[i]
        print(f"正在处理文字识别框：{ocr_box}")
        box_coords, text, confidence, char_coords_list, chars = ocr_box
        # print(f"正在处理文字识别框：{text}")

        # 遍历每个单字
        for j in range(len(chars) - 1):
            if is_digit_pair(chars[j], chars[j + 1]):
                distance = calculate_distance(char_coords_list[j][1], char_coords_list[j + 1][0])
                # print(f"字符 '{chars[j]}' 和 '{chars[j + 1]}' 之间的距离为：{distance:.2f}")

                if distance > threshold:
                    # print(f"距离大于阈值 {threshold}，进行切分。")
                    # 创建新的文字识别框
                    new_box = create_new_box(box_coords, char_coords_list, text, confidence, chars, j)
                    # 更新原始文字识别框
                    ocr_results[i] = update_existing_box(box_coords, char_coords_list, text, confidence, chars, j)
                    # 插入新的文字识别框
                    ocr_results.insert(i, new_box)
                    # print(f"切分后的文字识别框：{new_box[1]} 和 {ocr_results[i+1][1]}")
                    break  # 切分后，重新处理新的文字识别框
        i += 1

    return ocr_results

# 根据输入


# 在ocr结果中搜索字符串
def search_in_ocr_results(ocr_results, search_text):
    """
    在OCR识别结果中搜索指定字符串。
    :param ocr_results: OCR识别结果。
    :param search_text: 搜索字符串。
    :return: 搜索结果。
    """
    results = []
    for index, ocr_result in enumerate(ocr_results):
        for char_index, char in enumerate(ocr_result[4]):
            if char == search_text:
                results.append((index, char_index))
    return results

# 判断ocr结果是否存在
def is_ocr_result_exist(image_path, save_path="json_data"):
    """
    判断OCR识别结果是否存在。
    :param image_path: 图片路径。
    :return: 是否存在。
    """
    # 获取图片名
    image_name = os.path.basename(image_path)
    # 构造JSON文件路径
    json_path = os.path.join(save_path, f"{image_name}.json")
    return os.path.exists(json_path)

# 将ocr结果根据图片路径（获取图片名）暂存到json文件中
def save_ocr_result(ocr_result, image_path, save_path="json_data"):
    """
    将OCR识别结果保存到JSON文件中。
    :param ocr_result: OCR识别结果。
    :param image_path: 图片路径。
    :return: JSON文件路径。
    """
    # 创建保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 获取图片名
    image_name = os.path.basename(image_path)
    # 创建JSON文件路径
    json_path = os.path.join(save_path, f"{image_name}.json")
    # 保存OCR识别结果到JSON文件
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(ocr_result, file, ensure_ascii=False, indent=4)
    return json_path

# 从json文件中读取ocr结果
def load_ocr_result(image_path, save_path="json_data"):
    """
    从JSON文件中加载OCR识别结果。
    :param image_path: JSON文件路径。
    :return: OCR识别结果。
    """
    # 构造JSON文件路径
    image_name = os.path.basename(image_path)
    json_path = os.path.join(save_path, f"{image_name}.json")
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as file:
        ocr_result = json.load(file)
        return ocr_result

# 从删除ocr结果
def remove_ocr_result(image_path, save_path="json_data"):
    """
    删除OCR识别结果。
    :param image_path: 图片路径。
    :return: 是否删除成功。
    """
    # 获取图片名
    image_name = os.path.basename(image_path)
    # 构造JSON文件路径
    json_path = os.path.join(save_path, f"{image_name}.json")
    # 删除JSON文件
    if os.path.exists(json_path):
        os.remove(json_path)
        return True
    return False

# 绘制并保存ocr结果
def draw_ocr_results(image_path, boxes_list, save_path="ocr_mark_data"):
    """
    绘制OCR识别结果并保存图片。
    :param image_path: 图片路径。
    :param boxes: 文字识别框列表。
    :param save_path: 保存路径。
    :return: 保存的图片路径。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 读取图片
    img = cv2.imread(image_path)

    # 绘制方框
    for index, boxes in enumerate(boxes_list):
        for box in boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), get_color(index), 2)


            # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    # 保存图片
    save_image_path = os.path.join(save_path, os.path.basename(image_path))
    cv2.imwrite(save_image_path, img)
    return save_image_path


def generate_unique_filename_safe(original_filepath, suffix_length=6, directory="."):
    name, ext = os.path.splitext(os.path.basename(original_filepath))
    dir_path = os.path.dirname(original_filepath) or directory

    max_attempts = 1000
    for _ in range(max_attempts):
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=suffix_length))
        new_filename = f"{name}_{suffix}{ext}"
        new_filepath = os.path.join(dir_path, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath

    raise FileExistsError(f"无法在 {dir_path} 中生成唯一文件名（尝试次数过多）")

# 根据索引，获取一个颜色
def get_color(index):
    """
    根据索引获取一个颜色。
    :param index: 索引。
    :return: 颜色。
    """
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 128),
        (128, 128, 128),
        (255, 128, 0),
        (255, 0, 128),
        (128, 255, 0),
        (0, 255, 128),
        (128, 0, 255),
        (0, 128, 255),
        (255, 128, 128),
        (128, 255, 128),
        (128, 128, 255),
        (255, 255, 128),
        (255, 128, 255),
        (128, 255, 255),
        (255, 255, 255)
    ]
    return colors[index % len(colors)]

def re_search(a, b):
    """
    使用正则表达式在字符串列表b中搜索字符串a。
    :param a:
    :param b:
    :return:
    """

    if len(a) <= 1:
        return []

    results = []
    a = replace_punctuation_with_space(a)  # 删除标点符号

    # 拷贝一份
    a_tmp = a

    a_str = re.escape(a).replace("\\ ", "\\s*")  # 将目标字符串中的空格替换为 \s*
    a_str = re.compile(a_str, re.IGNORECASE)  # 忽略大小写
    # print(a_str.pattern)
    # a_str = remove_punctuation(a_str)  # 删除标点符号
    for index, string in enumerate(b):
        string = replace_punctuation_with_space(string)
        match = a_str.search(string)
        if match:
            results.append((index, match.start(), match.end()))
        else:
            string = replace_punctuation_with_space(string)

            # 如果长度小于等于1，直接跳过
            if len(string) <= 1:
                continue

            b_str = re.escape(string).replace("\\ ", "\\s*")
            b_str = re.compile(b_str, re.IGNORECASE)



            match = b_str.search(a)

            if match:
                start_index = match.start()
                end_index = match.end()

                # 将a_tmp中的匹配部分替换为空格
                a_tmp = replace_matched_text(a_tmp, b_str, "")

                # print('匹配到的字符串:', string, '匹配到的字符串在a中的位置:', start_index, end_index)

                results.append((index, 0, len(string)))

    # 如果a_tmp中还有字符，说明a_tmp中的字符没有在b中匹配到
    if len(a_tmp) >= 2:
        # 重新搜索一次a_tmp
        a_tmp = re.escape(a_tmp).replace("\\ ", "\\s*")
        a_tmp = re.compile(a_tmp, re.IGNORECASE)
        # print('a_tmp:', a_tmp, 'a:', a)
        for index, string in enumerate(b):
            string = replace_punctuation_with_space(string)
            match = a_tmp.search(string)
            if match:
                # print('匹配到a_tmp:', a_tmp, 'string:', string)
                results.append((index, match.start(), match.end()))
    else:
        print('过短a_tmp:', a_tmp)

    return results


# 删除字符串中的符号
def remove_punctuation(text):
    """
    删除字符串中的标点符号。
    :param text: 字符串。
    :return: 删除标点符号后的字符串。
    """
    return re.sub(r'[^\w\s]', '', text)

# 将字符串中的标点符号替换为空格
def replace_punctuation_with_space(text):
    """
    将字符串中的标点符号替换为空格。
    :param text: 字符串。
    :return: 替换后的字符串。
    """
    return re.sub(r'[^\w\s]', ' ', text)


def generate_unique_filename(file_path, algorithm='md5'):
    """
    根据文件内容生成唯一的文件名。

    :param file_path: 输入文件的路径（如图片文件）。
    :param algorithm: 哈希算法，默认为 'md5'，可选 'sha256' 或其他支持的算法。
    :return: 唯一的文件名（包含原文件扩展名）。
    """
    # 支持的哈希算法
    supported_algorithms = {'md5': hashlib.md5, 'sha256': hashlib.sha256}

    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported hash algorithm. Choose from {list(supported_algorithms.keys())}.")

    # 初始化哈希对象
    hash_func = supported_algorithms[algorithm]()

    # 读取文件内容并更新哈希值
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):  # 按块读取文件，避免内存占用过高
            hash_func.update(chunk)

    # 获取文件哈希值
    file_hash = hash_func.hexdigest()

    # 获取文件扩展名
    _, file_extension = os.path.splitext(file_path)

    # 生成唯一文件名
    unique_filename = f"{file_hash}{file_extension}"
    return unique_filename

def generate_custom_filename(prefix="upload", extension="txt", length=6):
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{suffix}.{extension}"

def generate_unique_filename_from_content(file_content, original_file_path, algorithm='md5'):
    """
    根据文件内容生成唯一的文件名。

    :param file_content: 已读取的文件内容（bytes 类型）。
    :param original_file_path: 原始文件路径，用于获取扩展名。
    :param algorithm: 哈希算法，默认为 'md5'，可选 'sha256' 或其他支持的算法。
    :return: 唯一的文件名（包含原文件扩展名）。
    """
    # 支持的哈希算法
    supported_algorithms = {'md5': hashlib.md5, 'sha256': hashlib.sha256}

    if algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported hash algorithm. Choose from {list(supported_algorithms.keys())}.")

    # 初始化哈希对象
    hash_func = supported_algorithms[algorithm]()

    # 更新哈希值
    hash_func.update(file_content)

    # 获取文件哈希值
    file_hash = hash_func.hexdigest()

    # 获取文件扩展名
    _, file_extension = os.path.splitext(original_file_path)

    # 生成唯一文件名
    unique_filename = f"{file_hash}{file_extension}"

    return unique_filename
def replace_matched_text(text, pattern, placeholder=" "):
    """
    将被正则表达式匹配到的文本替换为指定占位符（默认为空格）。

    参数:
        text (str): 原始字符串。
        pattern (str 或 re.Pattern): 正则表达式模式。
        placeholder (str): 用于替换匹配文本的占位符，默认为空格。

    返回:
        str: 替换后的字符串。
    """
    # 如果传入的是字符串形式的正则表达式，则编译它
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    # 使用 re.sub 进行替换
    return pattern.sub(placeholder, text)


def ocr_result_to_text(ocr_results, y_threshold=10):
    """
    将 paddleocr 的识别结果排序后拼接为一个字符串。

    参数:
        ocr_results: list
            paddleocr 的识别结果，如：
            [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], '文本'), ...]
        y_threshold: int
            判断是否属于同一行的 y 坐标差值阈值

    返回:
        str: 拼接后的文本结果
    """
    # 提取每个文本的最小 x 和 y 作为排序依据（左上角）
    lines = []

    print('开始转换为文本格式')

    if not ocr_results:
        return ""

    # print("ocr_results:", ocr_results)


    for ocr_r in ocr_results:

        box = ocr_r[0]  # 取出第一个元素
        text = ocr_r[1][0]
        min_x = min(p[0] for p in box)
        min_y = min(p[1] for p in box)
        lines.append((min_x, min_y, text))

    # 先按 y 排序（上到下），再按 x 排序（左到右）
    lines.sort(key=lambda x: (round(x[1] / y_threshold) * y_threshold, x[0]))

    # print("lines:", lines)

    # 合并文本
    result_lines = []
    current_y = None
    line_texts = []

    for x, y, text in lines:
        if current_y is None or abs(y - current_y) > y_threshold:
            if line_texts:
                result_lines.append(" ".join(line_texts))
                line_texts = []
            current_y = y
        line_texts.append(text)

    # print("line_texts:", line_texts)

    if line_texts:
        result_lines.append("".join(line_texts))

    print('转换为文本格式完成')

    return "".join(result_lines)



if __name__ == "__main__":
    s = "2024：11.12"

    print(replace_punctuation_with_space(s))

    print(replace_punctuation_with_space("2024.11.12"))

    print(replace_punctuation_with_space("1，375，421.63"))

