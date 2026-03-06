import mimetypes
import os
import uuid
from urllib.parse import urlparse

import requests
from fastapi import HTTPException

# 下载文件
def download_file(url, save_dir):
    """
    下载指定 URL 文件并保存到本地目录。

    如果传入的 URL 不是以 http:// 或 https:// 开头，则认为是本地路径，直接返回原路径。

    参数:
        url (str): 文件的 URL 地址或本地路径。
        save_dir (str): 本地保存图片的目录路径。

    返回:
        str: 下载后保存的本地文件路径；若 URL 是本地路径，则直接返回原路径。

    异常:
        HTTPException: 当网络请求失败时抛出 400 错误。
    """
    if not (url.startswith('http://') or url.startswith('https://')):
        return url

    try:
        response = requests.get(url)
        response.raise_for_status()

        # 提取文件后缀
        parsed_url = urlparse(url)
        file_suffix = parsed_url.path.split('.')[-1] if '.' in parsed_url.path else ''

        # 生成 UUID 作为文件名
        random_uuid = uuid.uuid4()

        # 定义保存路径
        # save_path = f"{save_dir}/{random_uuid}.{file_suffix}"
        save_path = os.path.join(save_dir, f"{random_uuid}.{file_suffix}")

        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"文件已成功下载到 {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=e)

# 上传文件
def upload_file(file_path, upload_url):
    """
    上传指定路径的文件到给定的URL，并返回服务器响应文本。

    参数:
    file_path (str): 图片在本地的路径。
    upload_url (str): 文件上传的目标URL地址。

    返回:
    str | None: 成功时返回服务器响应文本，失败返回 None。
    """
    # 根据文件扩展名猜测MIME类型
    mime_type, _ = mimetypes.guess_type(file_path)
    # if not mime_type or not mime_type.startswith('image/'):
    #     print(f"错误：文件 {file_path} 不是有效的图片文件。")
    #     return None

    try:
        with open(file_path, 'rb') as f:
            files = [('file', (file_path, f, mime_type))]
            response = requests.post(upload_url, files=files, proxies={})
            response.raise_for_status()  # 如果响应状态码不是200，将抛出异常
            return response.text
    except FileNotFoundError as e:
        print(f"错误：文件未找到 - {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"网络请求出错: {e}")
        return None

# 删除文件
def remove_file(file_path):
    """
    删除指定路径的文件。

    参数:
    file_path (str): 要删除的文件路径。
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # print(f"文件 {file_path} 已成功删除。")
        # else:
            # print(f"错误：文件 {file_path} 不存在。")
            # return None
    except Exception as e:
        print(f"删除文件时发生错误: {e}")

if __name__ == '__main__':
    print(upload_file(
        "/root/autodl-tmp/nwjh/LLMServe/data/广东电网有限责任公司信息中心2024年客户服务平台（网级95598语音平台V1.0）建设开发实施合同.pdf",
        "https://ioc.zhi-zai.com/process-factory/admin-api/infra/file/upload"))