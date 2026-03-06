import base64
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Request

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
import config
from pydantic import BaseModel
import fitz # PyMuPDF库，用于处理PDF
from PIL import Image
import re
import os
from openai import OpenAI # OpenAI客户端
from fastapi.middleware.cors import CORSMiddleware # CORS跨域支持
from fastapi.staticfiles import StaticFiles # 静态文件服务

from classify import predict_list # 文档分类预测
from ocr import get_ocr_list, get_ocr_image, get_ocr, get_ocr_image_list
from torchvision import transforms, models
import requests
from urllib.parse import urlparse
import uuid # 生成唯一ID
from typing import Any
# from sign_seal_detect import sign_seal_detect
import functools
#TODO 修改 ocr ,返回标注框
app = FastAPI()

# 配置CORS中间件（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"], # 允许所有HTTP方法
    allow_headers=["*"] # 允许所有头
)

# 挂载静态文件目录，设置静态文件服务目录/data对应本地data/文件夹
app.mount("/data", StaticFiles(directory="data"), name="data")

# 确保上传目录存在
current_directory = os.getcwd()
upload_dir = os.path.join(current_directory, "data/")
images_dir = upload_dir + "img/"
pdfs_dir = upload_dir + "content/"
os.makedirs(images_dir, exist_ok=True)
os.makedirs(pdfs_dir, exist_ok=True)

def ocr_decorator(func):
    """
   装饰器，在抽取函数执行后添加OCR。
   被装饰的函数返回值应该为：[
        {
                "page": 页码,
                "pageUrl": 对应页码的文件路径,
                "result": 对应页码的抽取结果（llm返回的 json 字符串）
        },
        ....
    ]
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs): 
        # 执行原始函数
        results = func(*args, **kwargs)
        
        # 钩子：在函数执行后执行的代码
        for item in results:
            page_url = item.get("pageUrl")
            img_json = item.get("result")
            if page_url and img_json:
                # 调用OCR函数，获取新的URL
                new_url = get_ocr_image(image_path=page_url, img_json=img_json)
                
                # 用新的URL替换原始的pageUrl
                item["pageUrl"] = new_url
        
        return results
    
    return wrapper



def extract_json(final_response: str):
    # 优先匹配 ```json ... ```
    code_block_match = re.search(r"```json\s*(.*?)\s*```", final_response, re.DOTALL | re.IGNORECASE)
    json_str = None

    if code_block_match:
        json_str = code_block_match.group(1).strip()
    # 如果失败或为空，降级用 re.search
    if not json_str:
        fallback_match = re.search(r'{(.*)}', final_response, re.DOTALL)
        if fallback_match:
            json_str = fallback_match.group().strip()
    return json_str
  
def run_conv(text_prompt, base64_image_list):
    """
    调用视觉语言模型:
    将图片和提示词发送给视觉语言模型
    使用阿里云的Qwen-VL模型
    从响应中提取JSON格式的结果
    """
    print("begin qwen3-vl-32b-instruct ask")

    contents = []
    for base64_image in base64_image_list:
        contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    contents.append({
        "type": "text",
        "text": text_prompt
    })
    # 构造请求消息
    messages = [
        {
            "role": "user",
            "content": contents
        }
    ]
    # 创建OpenAI客户端（使用阿里云DashScope服务）
    client = OpenAI(
        api_key="sk-40720d0026aa4fb9863330531995daed",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    response = client.chat.completions.create(
        model="qwen3-vl-32b-instruct",
        messages=messages
    )

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": text_prompt
    #             },
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image_list}"}
    #             }
    #         ]
    #     }
    # ]
    # client = OpenAI(api_key="none",
    #                 base_url="http://localhost:8000/v1")
    # response = client.chat.completions.create(
    #     model = "InternVL3-8B",
    #     messages=messages
    # )

    # 处理响应，排除json以外的内容
    final_response = response.choices[0].message.content
    # match = re.search(r'{(.*)}', final_response, re.DOTALL)
    res =extract_json(final_response)
    print("llm final result: ", res)
    return res

def encode_image(image_path):
    """将图片文件转换为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
#20260202修改
import os
import uuid
import requests
from fastapi import HTTPException
from urllib.parse import urlparse, unquote

def download_image(url: str, save_dir: str) -> str:
    """
    下载远程图片到本地，如果是本地路径则直接返回。
    
    参数:
        url (str): 图片地址（支持 http/https 或本地路径）
        save_dir (str): 远程图片下载后保存的目录
    
    返回:
        str: 本地图片路径
    """
    # 如果不是远程 URL，视为本地路径直接返回
    if not (url.startswith("http://") or url.startswith("https://")): 
        if os.path.exists(url):
            return url
        else:
            raise HTTPException(status_code=400, detail="Local file does not exist")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # 提取文件后缀（从 Content-Disposition 或 URL 中获取）
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition and "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"\' ')
        else:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or "." not in filename:
                filename = f"{uuid.uuid4()}.jpg"
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"图片已成功下载到 {save_path}")
        return save_path

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

@ocr_decorator
def pic_extract(fileUrl: str, category: str) -> list[Any]:
    """
    处理图像或PDF文档（兼容两者）
    使用原始提示词，保持原有字段提取逻辑和返回格式
    """
    data = []

    # 获取提示词（完全沿用原有逻辑）
    text_prompt = getattr(config, category)
    
    # 判断文件类型
    file_extension = os.path.splitext(fileUrl)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.psd', '.svg']
    pdf_extension = ['.pdf']

    if file_extension in image_extensions:
        # 是图像文件，按原有逻辑处理
        base64_image = [encode_image(fileUrl)]
        output = run_conv(text_prompt, base64_image)
        new_url = get_ocr_image(image_path=fileUrl, img_json=output)
        res = {
            "page": 0,
            "pageUrl": new_url,
            "result": output
        }
        data.append(res)

    elif file_extension in pdf_extension:
        # 是PDF文件，转换每一页为图像并处理
        pdf_document = fitz.open(fileUrl)
        total_pages = len(pdf_document)

        for page_num in range(total_pages):
            image_filename = pdf_2_images(pdf_document, images_dir, page_num)
            base64_image = encode_image(image_filename)
            output = run_conv(text_prompt, [base64_image])
            new_url = get_ocr_image(image_path=image_filename, img_json=output)

            res = {
                "page": page_num + 1,
                "pageUrl": new_url,
                "result": output
            }
            data.append(res)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return data

@ocr_decorator
def pic_extract_v2(fileUrl: str, category: str,needUrl=True) -> list[Any]:
    """
    处理图像或PDF文档（兼容两者）
    参数：needUrl 指示 pageUrl 是否带上文件地址
    返回：[
        {
                "page": 页码,
                "pageUrl": 对应页码的文件路径,
                "result": 对应页码的抽取结果
        },
        ....
    ]
    """
    data = []

    # 获取提示词（完全沿用原有逻辑）
    text_prompt = getattr(config, category)
    # 判断文件类型
    file_extension = os.path.splitext(fileUrl)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.psd', '.svg']
    pdf_extension = ['.pdf']

    if file_extension in image_extensions:
        # 是图像文件，按原有逻辑处理
        base64_image = [encode_image(fileUrl)]
        output = run_conv(text_prompt, base64_image)
        res = {
            "page": 0,
            "pageUrl": fileUrl if needUrl else "",
            "result": output
        }
        data.append(res)
    elif file_extension in pdf_extension:
        # 是PDF文件，转换每一页为图像并处理
        pdf_document = fitz.open(fileUrl)
        total_pages = len(pdf_document)
        for page_num in range(total_pages):
            image_filename = pdf_2_images(pdf_document, images_dir, page_num)
            base64_image = encode_image(image_filename)
            output = run_conv(text_prompt, [base64_image])
            res = {
                "page": page_num + 1,
                "pageUrl": image_filename  if needUrl else "",
                "result": output
            }
            data.append(res)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return data

# 20260203修改：发票抽取支持单页或两页 PDF，两页时合并为一份结果
INVOICE_2PAGE_PROMPT_SUFFIX = """
注意：本次提供的是同一张发票的第1页与第2页共2张图片，请将两页内容合并为一份完整的发票信息（一个JSON），字段取两页中的有效值，若同一字段在两页都有则优先采用第1页。"""


@ocr_decorator
def pic_extract_invoice_2page(fileUrl: str, category: str) -> list[Any]:
    """
    发票抽取：支持单页或两页 PDF。
    - 单页：与 pic_extract 一致，返回一页一个 result。
    - 两页：将两页图片一并传给模型，合并为一份发票结果（一个 result）。
    - 超过两页的 PDF：仅取前两页做合并抽取。
    """
    data = []
    text_prompt = getattr(config, category)
    file_extension = os.path.splitext(fileUrl)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.psd', '.svg']
    pdf_extension = ['.pdf']

    if file_extension in image_extensions:
        base64_image = [encode_image(fileUrl)]
        output = run_conv(text_prompt, base64_image)
        data.append({"page": 0, "pageUrl": fileUrl, "result": output})
        return data

    if file_extension not in pdf_extension:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    pdf_document = fitz.open(fileUrl)
    total_pages = len(pdf_document)

    if total_pages == 1:
        image_filename = pdf_2_images(pdf_document, images_dir, 0)
        base64_image = encode_image(image_filename)
        output = run_conv(text_prompt, [base64_image])
        data.append({"page": 1, "pageUrl": image_filename, "result": output})
        return data

    # 两页或更多：仅用前两页，合并为一份发票结果
    base64_images = []
    first_page_image = None
    for page_num in range(min(2, total_pages)):
        image_filename = pdf_2_images(pdf_document, images_dir, page_num)
        if first_page_image is None:
            first_page_image = image_filename
        base64_images.append(encode_image(image_filename))
    merged_prompt = text_prompt + INVOICE_2PAGE_PROMPT_SUFFIX
    output = run_conv(merged_prompt, base64_images)
    data.append({"page": 1, "pageUrl": first_page_image, "result": output})
    return data

# def inject_sign_seal(output:str,fileUrl):
#     #向 llm 输出的 json 字符串中插入签章检测结果
#     try:
#         # 检查输出是否为 JSON 字符串
#         if isinstance(output, str):
#             try:
#                 j = json.loads(output)
#             except json.JSONDecodeError as e:
#                 raise ValueError(f"解析 JSON 失败: {e}")
#         else:
#             raise TypeError("run_conv 返回值不是字符串，无法解析 JSON。")
        
#         # 获取签章检测结果，单个图片所以取第一个结果
#         sign_seal = sign_seal_detect(images=[fileUrl],debug=False)
#         if not isinstance(sign_seal[0], dict) or 'counts' not in sign_seal[0]:
#             raise ValueError("sign_seal_detect 返回格式不正确或缺少 'counts' 键。")
        
#         # 遍历并更新字典
#         counts = sign_seal[0].get('counts', 0)
#         if isinstance(counts, dict):
#             # 如果 count 本身是字典
#             for key, val in counts.items():
#                 j[key] = val
#         else:
#             raise TypeError(f"未预期的 count 类型: {type(counts)}")
#         # 转换回 JSON 字符串
#         return json.dumps(j, ensure_ascii=False)
        
#     except Exception as e:
#         print(f"发生错误: {e}")

# def pic_extract_sign_seal(fileUrl: str, category: str) -> list[Any]:
#     """
#     处理图像或PDF文档（兼容两者）
#     使用原始提示词，保持原有字段提取逻辑和返回格式
#     签章检测（提示词不需要签章）
#     """
#     results = pic_extract_v2( fileUrl,category)
#     for item in results:
#         page_url = item.get("pageUrl")
#         output = item.get("result")
#         if page_url and output:
#             item["result"] = inject_sign_seal(output,page_url) #插入签章检测结果
#     return results
    
  

import json
import re
from typing import Any

def pic_extract_with_processing(fileUrl: str, category: str) -> dict:
    """
    调用pic_extract并直接完成后处理，返回处理后的结果
    
    Args:
        fileUrl: 文件路径
        category: 类别
    
    Returns:
        处理后的结果，包含凭证不含税金额和凭证增值税数组
    """
    try:
        # 调用原始pic_extract函数
        raw_data = pic_extract_v2(fileUrl, category)
        
        # 检查返回值是否为None或空
        if raw_data is None:
            return []
        
        # 检查返回值是否为空列表或其他类型
        if not isinstance(raw_data, list):
            return []
        
        if len(raw_data) == 0:
            return []
        
        # 处理每页数据
        processed_data = []
        for page_data in raw_data:
            # 严格检查page_data
            if page_data is None:
                continue
            
            if not isinstance(page_data, dict):
                continue
                
            if "result" not in page_data:
                continue
            
            if page_data["result"] is None:
                continue
            try:
                # 解析result中的JSON字符串
                result_json = json.loads(page_data["result"])
                
                # -------------------------------------------------------
                # 1. 提取并处理凭证合计金额 (为后续计算做准备)
                # -------------------------------------------------------
                total_amount_str = result_json.get("凭证合计金额", "")
                total_amount = 0.0
                
                if total_amount_str:
                    # 清理金额字符串，去掉逗号等，转换为数字
                    cleaned_total = re.sub(r'[^\d.-]', '', str(total_amount_str))
                    try:
                        total_amount = float(cleaned_total)
                    except ValueError:
                        total_amount = 0.0
                
                # 更新处理结果字典
                processed_result = {
                    "合同编号": result_json.get("合同编号", ""),
                    "公司名称": result_json.get("公司名称", ""),
                    "凭证合计金额": total_amount_str # 保持原始字符串，或者存数字 total_amount 也可以
                }
                
                # -------------------------------------------------------
                # 2. 提取增值税金额 (仅保留提取增值税的逻辑)
                # -------------------------------------------------------
                mingxi = result_json.get("明细", [])
                if mingxi is None:
                    mingxi = []
                
                vat_amounts = []          # 增值税金额列表
                
                for item in mingxi:
                    # 检查item是否为None
                    if item is None:
                        continue
                        
                    subject_name = item.get("科目名称", "")
                    amount_str = item.get("借方金额", "")
                    
                    # 跳过包含"(辅)"的项目
                    if "(辅)" in subject_name:
                        continue
                        
                    # 清理金额字符串
                    if amount_str:
                        cleaned_amount = re.sub(r'[^\d.-]', '', str(amount_str))
                        try:
                            amount = float(cleaned_amount)
                            
                            # 判断是否为增值税相关科目
                            vat_keywords = ["增值税", "进项税", "销项税", "应交税费", "进项税额", "销项税额"]
                            if any(keyword in subject_name for keyword in vat_keywords):
                                vat_amounts.append(amount)
                                
                        except ValueError:
                            continue
                
                # -------------------------------------------------------
                # 3. 确定最终的增值税值
                # -------------------------------------------------------
                final_vat_amount = 0.0
                if vat_amounts:
                    # 如果列表里有值，取最小值 (根据原逻辑保持不变)
                    final_vat_amount = min(vat_amounts, key=float)
                
                processed_result["凭证增值税金额"] = final_vat_amount

                # -------------------------------------------------------
                # 4. 计算不含税金额 (逻辑修改：直接相减)
                # -------------------------------------------------------
                # 不含税金额 = 凭证合计金额 - 增值税金额
                # 使用 round 防止浮点数运算出现 100.00000001 这种情况
                final_no_tax_amount = round(total_amount - final_vat_amount, 2)
                
                processed_result["凭证不含税金额"] = final_no_tax_amount
                
                # 构造处理后的页面数据
                processed_page = {
                    "page": page_data["page"],
                    "pageUrl": page_data["pageUrl"], 
                    "result": json.dumps(processed_result, ensure_ascii=False, indent=4)
                }
                
                processed_data.append(processed_page)
                
            except json.JSONDecodeError as e:
                print(f"!!! JSON解析失败: {e}") 
                continue
            except Exception as e:
                # 处理其他异常，跳过该页
                print(f"!!! 页面处理异常: {e}") 
                import traceback
                traceback.print_exc() 
                continue
        
        return processed_data
        
    except Exception as e:
        # 整体异常处理
        print(f"!!! 整体流程异常: {e}") 
        import traceback
        traceback.print_exc() 
        return []


# def pdf_extract(fileUrl: str, category: str) -> list[Any]:
#     """处理PDF文档"""
#     data = []
#     # 验证文件类型
#     file_extension = os.path.splitext(fileUrl)[1].lower()
#     pdf_extension = ['.pdf']
#     if file_extension not in pdf_extension:
#         raise HTTPException(status_code=400, detail="Unsupported file type")

#     # 处理合同首页信息（前6页）
#     new_url, output = contract_homepage_info(fileUrl)

#     page = [re.search(r'page_(\d+)\.png$', path).group(1) for path in new_url]
#     # for path in new_url:
#     #     match = re.search(r'page_(\d+)\.png$', path)
#     #     page.append(match.group(1))

#     data.append({
#         "page": page,
#         "pageUrl": new_url,
#         "result": output
#     })

#     # 合同价款、是否按合同约定票据比例支付
#     output, page, picpath = contract_payment(fileUrl)
#     data.append({
#         "page": page,
#         "pageUrl": picpath,
#         "result": output
#     })

#     # 签字、盖章

#     return data

import cv2
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os

# 20260202公司名称等抽取结果的通用纠错规则：(错误片段, 正确片段)，按顺序替换
EXTRACT_CORRECTIONS = [
    ("勘研研究院", "勘察研究院"),
    ("勘测研究院", "勘察研究院"),
    # 后续可在此追加，例如：
    # ("其它误识别", "正确名称"),
]


def apply_extract_corrections(text: str, rules: list = None) -> str:
    """
    对抽取得到的文本做通用纠错。传入规则列表 [(错误, 正确), ...]，默认使用 EXTRACT_CORRECTIONS。
    """
    if not text or not isinstance(text, str):
        return text
    for wrong, right in (rules or EXTRACT_CORRECTIONS):
        if wrong in text:
            text = text.replace(wrong, right)
    return text


@ocr_decorator
def pic_extract_merge(fileUrl: str, category: str) -> dict:
    """
    处理转账凭证（transfer_voucher）：
    1. 范围：仅处理PDF的最后两页（如果是图片则处理单张）。
    2. 逻辑：
       - 凭证合计金额：使用最后一次抽取到的非空值（覆盖）。
       - 合同编号/公司名称：使用第一次抽取到的非空值（不覆盖）。
    """
    data = []
    
    # 获取提示词
    text_prompt = getattr(config, category)
    
    # 判断文件类型
    file_extension = os.path.splitext(fileUrl)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.psd', '.svg']
    pdf_extension = ['.pdf']

    # 1. 执行抽取 (只抽取最后5页)
    if file_extension in image_extensions:
        # 图片按单页处理
        base64_image = [encode_image(fileUrl)]
        output = run_conv(text_prompt, base64_image)
        res = {
            "page": 1, # 图片默认为第1页
            "pageUrl": fileUrl,
            "result": output
        }
        data.append(res)

    elif file_extension in pdf_extension:
        # PDF 文件：只处理最后 5页
        pdf_document = fitz.open(fileUrl)
        total_pages = len(pdf_document)
        
        # 计算起始页码：如果总页数大于5，从倒数第5页开始；否则从第0页开始
        start_page = max(0, total_pages - 5)

        for page_num in range(start_page, total_pages):
            image_filename = pdf_2_images(pdf_document, images_dir, page_num)
            base64_image = encode_image(image_filename)
            output = run_conv(text_prompt, [base64_image])
            
            res = {
                "page": page_num + 1,
                "pageUrl": image_filename,
                "result": output
            }
            data.append(res)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # 2. 合并结果
    merged_result = {
        "合同编号": {"value": ""},
        "公司名称": {"value": ""},
        "凭证合计金额": {"value": ""},
    }

    last_page = None 
    last_pageUrl = None 

    for item in data:
        # 更新最后页码信息
        if item.get("page") is not None:
            last_page = item.get("page")
        if item.get("pageUrl") is not None:
            last_pageUrl = item.get("pageUrl")

        try:
            page_result = json.loads(item.get("result", "{}"))
        except json.JSONDecodeError:
            page_result = {}

        # --- 核心修改逻辑开始 ---
        
        # 字段1：凭证合计金额 (逻辑：总是覆盖，取最后出现的值)
        amount_val = page_result.get("凭证合计金额")
        if amount_val:
            merged_result["凭证合计金额"]["value"] = amount_val

        # 字段2：合同编号 & 公司名称 (逻辑：如果有值且当前为空，才赋值 -> 即保留第一个找到的值)
        for key in ["合同编号", "公司名称"]:
            val = page_result.get(key)
            # 只有当提取到了值(val)，且当前结果库里还没值(result为空)时，才写入
            # 这样后面的页面就不会覆盖前面页面提取到的值
            if val and not merged_result[key]["value"]:
                merged_result[key]["value"] = val
        
        # --- 核心修改逻辑结束 ---

    # 设置默认值为“未找到”
    for key in merged_result:
        if not merged_result[key]["value"]:
            merged_result[key]["value"] = "未找到"

    # 20260202公司名称通用纠错
    merged_result["公司名称"]["value"] = apply_extract_corrections(merged_result["公司名称"]["value"])
    
    combined_dict = {
        "合同编号": merged_result["合同编号"]["value"],
        "公司名称": merged_result["公司名称"]["value"],
        "凭证合计金额": merged_result["凭证合计金额"]["value"]
    }
    
    combined_result_str = json.dumps(combined_dict, ensure_ascii=False)

    final_result = []
    res = {
        "page": last_page,
        "pageUrl": last_pageUrl,
        "result": combined_result_str
    }
    final_result.append(res)
    
    return final_result

# #warehouse_receipt
# def warehouse_receipt_form(fileUrl: str, category: str) -> list:
#     """
#     处理带二维码的入库单：
#     - 如果是 PDF，则每一页转为图像后识别二维码
#     - 如果是图片，直接识别二维码
#     - 使用视觉语言模型提取所有字段
#     - 如果二维码中存在内容（默认就是入库单号），则覆盖模型结果中的“入库单号”字段
#     返回包含二维码信息的结果列表
#     """
#     data = []

#     def detect_qr(image_path, page_num=0):
#         """识别指定图像中的二维码"""
#         image = cv2.imread(image_path)
#         decoded_objects = pyzbar.decode(image)

#         if not decoded_objects:
#             return None

#         for obj in decoded_objects:
#             payload = obj.data.decode("utf-8")
#             try:
#                 qr_data = json.loads(payload)
#                 # 如果有 "入库单号" 字段，优先使用
#                 if "入库单号" in qr_data:
#                     receipt_code = qr_data["入库单号"]
#                 else:
#                     # 否则尝试从 payload 中提取 receiptCode=
#                     match = re.search(r'receiptCode=(.+)', payload)
#                     receipt_code = match.group(1) if match else payload
#             except json.JSONDecodeError:
#                 # 如果不是 JSON 格式，直接尝试提取 receiptCode=
#                 match = re.search(r'receiptCode=(.+)', payload)
#                 receipt_code = match.group(1) if match else payload

#             return {
#                 "page": page_num + 1,
#                 "pageUrl": image_path,
#                 "result": {
#                     "二维码内容": {"入库单号": receipt_code}
#                 }
#             }

#     def merge_results(model_result, qr_result):
#         """合并视觉模型结果和二维码结果"""
#         if qr_result and "二维码内容" in qr_result["result"]:
#             model_result["入库单号"] = qr_result["result"]["二维码内容"]["入库单号"]
#         return model_result

#     file_extension = os.path.splitext(fileUrl)[1].lower()
#     image_extensions = ['.jpg', '.jpeg', '.png']
#     pdf_extension = ['.pdf']

#     if file_extension in image_extensions:
#         # 图像文件：识别二维码 + 模型提取字段
#         qr_result = detect_qr(fileUrl, page_num=0)
#         base64_image = encode_image(fileUrl)
#         model_output = run_conv(config.warehouse_receipt_form, [base64_image])
#         model_result = json.loads(model_output)

#         # 合并结果
#         final_result = merge_results(model_result, qr_result)

#         data.append({
#             "page": 0,
#             "pageUrl": fileUrl,
#             "result": json.dumps(final_result, ensure_ascii=False)
#         })

#     elif file_extension in pdf_extension:
#         # PDF 文件：逐页处理
#         pdf_document = fitz.open(fileUrl)
#         total_pages = len(pdf_document)

#         for page_num in range(total_pages):
#             image_filename = pdf_2_images(pdf_document, images_dir, page_num)

#             # 识别二维码
#             qr_result = detect_qr(image_filename, page_num)
#             base64_image = encode_image(image_filename)

#             # 使用模型提取字段
#             model_output = run_conv(config.warehouse_receipt_form, [base64_image])
#             model_result = json.loads(model_output)

#             # 合并结果
#             final_result = merge_results(model_result, qr_result)

#             data.append({
#                 "page": page_num + 1,
#                 "pageUrl": image_filename,
#                 "result": json.dumps(final_result, ensure_ascii=False)
#             })
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported file type")

#     return data

def get_keys_by_image_url(merged_result):
    """
    根据图片URL获取对应的键值对。
    
    Args:
        merged_result (dict): 包含合同信息和页面URL的字典。
        
    Returns:
        dict: 键是图片URL，值是包含该图片上所有键值对的列表。
    """
    results_by_url = {}
    
    # 遍历 merged_result 字典中的所有键值对
    for key, data in merged_result.items():
        # 检查 pageUrls 列表是否为空
        if data["pageUrls"]:
            # 遍历每个键值对中的 pageUrls
            for url in data["pageUrls"]:
                # 如果 URL 还没有在 results_by_url 字典中，则创建一个空列表
                if url not in results_by_url:
                    results_by_url[url] = []
                
                # 将当前键和值添加到对应的 URL 列表中
                results_by_url[url].append({
                    "key": key,
                    "value": data["value"]
                })
                
    return results_by_url

import json
def contract_Order_contracts_v2(fileUrl: str, category: str) -> dict:
    """
    处理物资订单合同，从整个PDF的所有页面中提取：
    - 合同编号 (合同编号)
    - 合同卖方 (合同卖方)
    - 合同买方 (合同买方)
    - 合同账户名 (合同账户名)
    - 合同结算账号 (合同结算账号)
    - 开户银行 (开户银行)
    - 支付比例 (支付比例)
    - 合同总金额 (合同总金额)
    - 买方印章内容 (买方印章内容)
    - 卖方印章内容 (卖方印章内容)
    - 买方签名章内容 (买方签名章内容)
    - 卖方签名章内容 (卖方签名章内容)
    - 签名 (签名)
    - 盖章 (盖章)
    返回合并后的 merged_result
    """
    # 调用YOLO接口获取所有页面结果 (此函数未提供，假设其返回结果为包含页面信息的列表)
    # results = pic_extract_sign_seal(fileUrl, category)

    results = pic_extract_v2(fileUrl,category)
    # 初始化合并结果
    merged_result = {
        "合同编号": {"value": "", "pages": [], "pageUrls": []},
        "合同卖方": {"value": "", "pages": [], "pageUrls": []},
        "合同买方": {"value": "", "pages": [], "pageUrls": []}, # 添加合同买方
        "合同结算账号": {"value": "", "pages": [], "pageUrls": []},
        "合同账户名": {"value": "", "pages": [], "pageUrls": []},
        "开户银行": {"value": "", "pages": [], "pageUrls": []},
        "支付比例": {"value": "", "pages": [], "pageUrls": []},
        "合同总金额": {"value": "", "pages": [], "pageUrls": []},
        "买方印章内容": {"value": 0, "pages": [], "pageUrls": []},
        "卖方印章内容": {"value": 0, "pages": [], "pageUrls": []},
        "买方签名章内容": {"value": 0, "pages": [], "pageUrls": []},
        "卖方签名章内容": {"value": 0, "pages": [], "pageUrls": []},
        "签名": {"value": 0, "pages": [], "pageUrls": []},
        "盖章": {"value": 0, "pages": [], "pageUrls": []},
    }

    for item in results:
        page = item.get("page")
        pageUrl = item.get("pageUrl")
        
        # try:
        #     # 解析 result（JSON字符串转为字典）
        #     page_result = json.loads(item.get("result", "{}"))
        # except json.JSONDecodeError:
        #     page_result = {}
        try:
            # 获取 result，如果为 None 或空字符串，则默认为 "{}"
            raw_result = item.get("result")
            if not raw_result:  # 这里涵盖了 None 和 "" 的情况
                raw_result = "{}"  
            page_result = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            # 同时捕获解析错误和类型错误
            page_result = {}


        for key in merged_result.keys():
            if key in ["签名", "盖章"]:
                # 提取签名和盖章数量
                sign_count = page_result.get("签名", 0)
                seal_count = page_result.get("盖章", 0)
                
                # 只有当当前页同时有签名和盖章时，更新记录 （记录最后一页）
                if isinstance(sign_count, int) and isinstance(seal_count, int) and sign_count > 0 and seal_count > 0:
                    merged_result["签名"]["value"] = sign_count
                    merged_result["签名"]["pages"] = [page]
                    merged_result["签名"]["pageUrls"] = [pageUrl]

                    merged_result["盖章"]["value"] = seal_count
                    merged_result["盖章"]["pages"] = [page]
                    merged_result["盖章"]["pageUrls"] = [pageUrl]

            else:
                # 更新记录 （记录最后一次出现的非空值）
                value = page_result.get(key)
                if value:
                    merged_result[key]["value"] = value
                    merged_result[key]["pages"].append(page)
                    merged_result[key]["pageUrls"].append(pageUrl)
                   
                    

    # 设置默认值为“未找到”
    for key in merged_result:
        if not merged_result[key]["value"]:
            merged_result[key]["value"] = "未找到"

    # 构造最终输出，使其与 pic_extract 一致
    final_result = []
    for key, value_info in merged_result.items():
        # 如果没有抽取到有效结果，`pages` 和 `pageUrls`可能为空
        page_list = value_info["pages"] if value_info["pages"] else []
        url_list = value_info["pageUrls"] if value_info["pageUrls"] else []
        
        res = {
            "page": page_list,
            "pageUrl": url_list,
            "result": f'{{"{key}": "{value_info["value"]}"}}'
        }
        final_result.append(res)

    return final_result



# def contract_Order_contracts_v2(fileUrl: str, category: str) -> dict:
#     """
#     处理物资订单合同，从整个PDF的所有页面中提取：
#     - 合同编号 (合同编号)
#     - 合同卖方 (合同卖方)
#     - 合同账户名 (合同账户名)
#     - 合同结算账号 (合同结算账号)
#     - 开户银行 (开户银行)
#     - 支付比例 (支付比例)
#     - 合同总金额 (合同总金额)
#     - 签名 (签名)
#     - 盖章 (盖章)
#     返回合并后的 merged_result
#     """
#     # 调用接口获取所有页面结果
#     results = pic_extract_sign_seal(fileUrl, category)

#     # 初始化合并结果
#     merged_result = {
#         "合同编号": {"value": "", "pages": [], "pageUrls": []},
#         "合同卖方": {"value": "", "pages": [], "pageUrls": []},
#         "合同结算账号": {"value": "", "pages": [], "pageUrls": []},
#         "合同账户名": {"value": "", "pages": [], "pageUrls": []},
#         "开户银行": {"value": "", "pages": [], "pageUrls": []},
#         "支付比例": {"value": "", "pages": [], "pageUrls": []},
#         "合同总金额": {"value": "", "pages": [], "pageUrls": []},
#         "签名": {"value": 0, "pages": [], "pageUrls": []},
#         "盖章": {"value": 0, "pages": [], "pageUrls": []},
#     }

#     for item in results:
#         page = item.get("page")
#         pageUrl = item.get("pageUrl")
#         try:
#             # 解析 result（JSON字符串转为字典）
#             page_result = json.loads(item.get("result", "{}"))
#         except json.JSONDecodeError:
#             page_result = {}

#         for key in merged_result.keys():
#             if key in ["签名", "盖章"]:
#                  # 当前页签名/盖章数量
#                  # 提取签名和盖章数量
#                 sign_count = page_result.get("签名", 0)
#                 seal_count = page_result.get("盖章", 0)
                
#                 #  签名盖章合计，但是签名容易错误检测
#                 # if isinstance(count, int) and count > 0:
#                 #     merged_result[key]["value"] += count
#                 #     merged_result[key]["pages"].append(page)
#                 #     merged_result[key]["pageUrls"].append(pageUrl)

#                 # 只有当当前页同时有签名和盖章时，更新记录 （记录最后一页）
#                 if isinstance(sign_count, int) and isinstance(seal_count, int) and sign_count > 0 and seal_count > 0:
#                     merged_result["签名"]["value"] = sign_count
#                     merged_result["签名"]["pages"] = [page]
#                     merged_result["签名"]["pageUrls"] = [pageUrl]

#                     merged_result["盖章"]["value"] = seal_count
#                     merged_result["盖章"]["pages"] = [page]
#                     merged_result["盖章"]["pageUrls"] = [pageUrl]

#             else:
#                 #更新记录 （记录最后一次出现的非空值）
#                 value = page_result.get(key)
#                 if value:
#                     merged_result[key]["value"] = value
#                     merged_result[key]["pages"].append(page)
#                     merged_result[key]["pageUrls"].append(pageUrl)

#     # 设置默认值为“未找到”
#     for key in merged_result:
#         if not merged_result[key]["value"]:
#             merged_result[key]["value"] = "未找到"
 
#     #return merged_result
    
#     # 构造最终输出，使其与 pic_extract 一致
#     final_result = []
#     for key, value_info in merged_result.items():
#         # 如果没有抽取到有效结果，`pages` 和 `pageUrls`可能为空
#         page_list = value_info["pages"] if value_info["pages"] else []
#         url_list = value_info["pageUrls"] if value_info["pageUrls"] else []
        
#         res = {
#             "page": page_list,
#             "pageUrl": url_list,
#             "result": f'{{"{key}": "{value_info["value"]}"}}'
#         }
#         final_result.append(res)

#     return final_result

    
    
def contract_Order_contracts(fileUrl: str, category: str) -> list:
    """
    处理物资订单合同，从整个PDF的所有页面中提取：
    - 合同编号 (contract_no)
    - 合同卖方 (contract_seller)
    - 合同账户名 (contract_account_no)
    - 合同结算账号 (contract_settlement_account)
    - 开户银行 
    - 支付比例 
    - 合同总金额 
    - 签名 
    - 盖章
    返回包含 page, pageUrl 和 result 的列表
    """


    # 初始化字段存储结构（使用中文字段名）
    # merged_result 结构不变，用于累积跨页面的结果
    merged_result = {
        "合同编号": {"value": "", "pages": [], "pageUrls": []},
        "合同卖方": {"value": "", "pages": [], "pageUrls": []},
        "合同结算账号": {"value": "", "pages": [], "pageUrls": []},
        "合同账户名": {"value": "", "pages": [], "pageUrls": []},
        "开户银行": {"value": "", "pages": [], "pageUrls": []},
        "支付比例": {"value": "", "pages": [], "pageUrls": []},
        "合同总金额": {"value": "", "pages": [], "pageUrls": []},
        "签名": {"value": "", "pages": [], "pageUrls": []},
        "盖章": {"value": "", "pages": [], "pageUrls": []},
    }

    # 加载 PDF 文档
    pdf_document = fitz.open(fileUrl)
    total_pages = len(pdf_document)

    for page_num in range(total_pages):  # 遍历所有页面
        # 确保 images_dir 已定义
        images_dir = "/tmp"  # 或者你实际的图片目录
        image_filename = pdf_2_images(pdf_document, images_dir, page_num)
        base64_image = encode_image(image_filename)

        try:
            output = run_conv(config.contract_order, [base64_image])
            # 清洗输出：提取 ```json``` 中的内容
            clean_output = output.strip()
            if clean_output.startswith('```json'):
                clean_output = clean_output[7:]  # 去掉 ```json
            if clean_output.endswith('```'):
                clean_output = clean_output[:-3]  # 去掉 ```
            clean_output = clean_output.strip()
            json_output = json.loads(clean_output)
        except (json.JSONDecodeError, Exception):  # 捕获所有解析异常
            continue

        current_page = page_num + 1  # 页面编号从 1 开始
        
        # 获取当前页面的抽取结果
        current_page_results = {k: v for k, v in json_output.items() if k in merged_result}
        
        # 对当前页面的抽取结果进行 OCR
        # get_ocr_image 函数需要支持处理json格式的img_json
        ocr_result_url = get_ocr_image(image_path=image_filename, img_json=json.dumps({"value": current_page_results}))
        
        # 提取字段并记录来源页码和 OCR 后的图片路径
        def update_field(field_key):
            if field_key in json_output and json_output[field_key] is not None and str(json_output[field_key]).strip() != "":
                value = json_output[field_key]
                if not merged_result[field_key]["value"]:
                    merged_result[field_key]["value"] = value
                # 更新 pages 和 pageUrls
                merged_result[field_key]["pages"].append(current_page)
                merged_result[field_key]["pageUrls"].append(ocr_result_url)

        # 遍历所有字段更新
        for key in merged_result:
            update_field(key)

    # 去重页码和图片路径
    # 由于现在每页都会有一个唯一的 OCR URL，去重逻辑简化
    for key in merged_result:
        seen_pages = set()
        unique_page_info = [] # 存储 (page, url) 元组
        for p, u in zip(merged_result[key]["pages"], merged_result[key]["pageUrls"]):
            if p not in seen_pages:
                seen_pages.add(p)
                unique_page_info.append((p, u))
        
        # 解压回 pages 和 pageUrls 列表
        merged_result[key]["pages"] = [info[0] for info in unique_page_info]
        merged_result[key]["pageUrls"] = [info[1] for info in unique_page_info]
    
    # 设置默认值为“未找到”
    for key in merged_result:
        if not merged_result[key]["value"]:
            merged_result[key]["value"] = "未找到"

    # 构造最终输出，使其与 pic_extract 一致
    final_result = []
    for key, value_info in merged_result.items():
        # 如果没有抽取到有效结果，`pages` 和 `pageUrls`可能为空
        page_list = value_info["pages"] if value_info["pages"] else []
        url_list = value_info["pageUrls"] if value_info["pageUrls"] else []
        
        res = {
            "page": page_list,
            "pageUrl": url_list,
            "result": f'{{"{key}": "{value_info["value"]}"}}'
        }
        final_result.append(res)

    return final_result
# # 物资框架合同
# def contract_Goods_contracts(fileUrl: str, category: str) -> list:
#     """
#     处理物资框架合同，从整个PDF的所有页面中提取：
#     - 合同编号 (contract_no)
#     - 合同支付期限 (stipulated_payment_term)
#     返回包含 page, pageUrl 和 result 的列表
#     """

#     # 初始化结果字典
#     result = {
#         "合同编号": {"value": "", "page": "", "pageUrl": ""},
#         "合同支付期限": {"value": "", "page": "", "pageUrl": ""}
#     }

#     pdf_document = fitz.open(fileUrl)
#     total_pages = len(pdf_document)  # 获取总页数

#     for page_num in range(total_pages):  # 遍历所有页面
#         image_filename = pdf_2_images(pdf_document, images_dir, page_num)
#         base64_image = encode_image(image_filename)

#         try:
#             output = run_conv(config.contract_goods, [base64_image])
#             json_output = json.loads(output)
#         except (json.JSONDecodeError, KeyError):
#             continue  # 忽略解析失败或格式错误的响应

#         current_page = str(page_num + 1)

#         # 提取合同编号：只保留第一次找到的
#         if "合同编号" in json_output and json_output["合同编号"] and not result["合同编号"]["value"]:
#             result["合同编号"]["value"] = json_output["合同编号"]
#             result["合同编号"]["page"] = current_page
#             result["合同编号"]["pageUrl"] = image_filename

#         # 提取合同支付期限：每次都更新，最终保留最后一次
#         if "合同支付期限" in json_output and json_output["合同支付期限"]:
#             result["合同支付期限"]["value"] = json_output["合同支付期限"]
#             result["合同支付期限"]["page"] = current_page
#             result["合同支付期限"]["pageUrl"] = image_filename

#     # 设置默认值
#     if not result["合同编号"]["value"]:
#         result["合同编号"]["value"] = "未找到"
#     if not result["合同支付期限"]["value"]:
#         result["合同支付期限"]["value"] = "未找到"

#     # 构造最终输出，使其与 pic_extract 一致
#     final_result = []
#     for key, value_info in result.items():
#         res = {
#             "page": [value_info["page"]],
#             "pageUrl": [value_info["pageUrl"]],
#             "result": f'{{"{key}": "{value_info["value"]}"}}'
#         }
#         final_result.append(res)

#     return final_result


# 前端类型到内部标识的映射
type_mapping = {
    "发票": "invoice",
    "订单合同":"contract_order",
    "商城到货单":"store_order",
    "质保金":"deposit",
    "银付凭证":"certificate",
    "转账凭证":"transfer_voucher",
    "支付申请单或XX证明":"payment_receipt_o_XXcertificate",
    # 2026.3.3新增场景
    "火车票订单":"train_ticket_order",
    "酒店订单":"hotel_order",
    "出差申请单":"business_travel_application",
    "机票订单":"flight_ticket_order"
}
# 类型到处理函数的映射
type_to_function = {
    #"invoice": pic_extract,
    "invoice": pic_extract_invoice_2page,  # 发票：支持单页/两页 PDF，两页合并为一份结果
    "contract_order": contract_Order_contracts_v2,
    "store_order": pic_extract,
    "deposit":pic_extract,
    # "deposit":pic_extract_sign_seal,
    "certificate":pic_extract_with_processing,
    # "certificate":pic_extract,
    "transfer_voucher":pic_extract_merge,
    "payment_receipt_o_XXcertificate":pic_extract,
    # 2026.3.3新增场景
    "train_ticket_order": pic_extract,
    "hotel_order": pic_extract,
    "business_travel_application": pic_extract,
    "flight_ticket_order": pic_extract
}

# 使用Pydantic定义API请求的数据模型
from typing import Optional
from pydantic import BaseModel

class ExtractBody(BaseModel):
    classify: str
    fileUrl: str
    base64: Optional[str] = None

class DetectBody(BaseModel):
    fileUrl: str
    base64: str

# 信息抽取端点
@app.post("/nfdw/api/v1/model/extract")
async def extract(body: ExtractBody):
    print("抽取接口")
    classify = body.classify
    url = body.fileUrl
    base64_data = body.base64

    # 如果提供了 base64 数据，则优先处理
    if base64_data:
        try:
            # 解码 base64 数据
            image_data = base64.b64decode(base64_data)
            # 生成临时文件路径
            temp_file_path = os.path.join(images_dir, f"temp_{uuid.uuid4()}.png")
            with open(temp_file_path, "wb") as f:
                f.write(image_data)
            fileUrl = temp_file_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 decode failed: {str(e)}")
    else:
        # 如果没有提供 base64，则从 fileUrl 下载文件
        try:
            fileUrl = download_image(url, images_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

    # 获取文档类型映射
    type = type_mapping.get(classify)
    if not type:
        raise HTTPException(status_code=400, detail="Unsupported type")

    handler = type_to_function.get(type)
    if not handler:
        return JSONResponse(status_code=400, content={"code": 400, "message": f"Unsupported type: {body.type}", "data": None})

    # 处理文档
    try:
        result = handler(fileUrl, type)
        # for item in result:
        #     item['pageUrl'] = [url]
        return JSONResponse(content={"code": 200, "message": "Success", "data": result})
    except Exception as e:
        return JSONResponse(status_code=400, content={"code": 400, "message": str(e), "data": ""})


# 签名检测端点
@app.post("/nfdw/api/v1/model/detect")
async def detect(body: DetectBody):
    print("检测接口")
    url = body.fileUrl
    base64 = body.base64
    data = []
    # 下载文件或使用提供的base64
    fileUrl = download_image(url, images_dir)

    if fileUrl is not None:
        base64 = encode_image(fileUrl)
    # 检测签名和盖章
    try:
        result = run_conv(config.sign_seal_prompt, [base64])

        count_json = json.loads(result)
        # sign_count = count_json.get("签名")
        # seal_count = count_json.get("盖章")

        data.append({
            "page": 0,
            "pageUrl": fileUrl,
            "sign": count_json.get("签名"),
            "seal": count_json.get("盖章")
        })

        return JSONResponse(content={"code": 200, "message": "Success", "data": data})

    except Exception as e:
        return JSONResponse(status_code=400, content={"code": 400, "message": str(e), "data": ""})


# # 修改后的签名检测端点,使用yolo
# @app.post("/nfdw/api/v1/model/detect")
# async def detect(body: DetectBody):
#     print("检测接口")
#     url = body.fileUrl
#     data = []

#     try:
#         # 下载文件或使用提供的base64
#         fileUrl = download_image(url, images_dir)

#         # 直接调用 YOLO 模型的签章检测函数
#         detection_results = sign_seal_detect(images=[fileUrl], debug=False)

#         # 检查检测结果是否有效
#         if not detection_results or 'counts' not in detection_results[0]:
#             raise ValueError("Sign/seal detection failed or returned an invalid format.")
        
#         counts = detection_results[0]['counts']
#         sign_count = counts.get("签名", 0)
#         seal_count = counts.get("盖章", 0)

#         data.append({
#             "page": 0,
#             "pageUrl": fileUrl,
#             "sign": sign_count,
#             "seal": seal_count
#         })

#         return JSONResponse(content={"code": 200, "message": "Success", "data": data})

#     except Exception as e:
#         return JSONResponse(status_code=400, content={"code": 400, "message": str(e), "data": ""})

# 文档分类端点
@app.post('/nfdw/model/classify')
async def get_images(request: Request):
    try:
        body = await request.json()
        img_list = body.get("list")
        bussiness_type = body.get("businessType")

        # --- 修改开始：更稳健的下载逻辑 ---
        valid_img_list = []  # 存储下载成功的图片
        failed_results = []  # 存储下载失败的结果

        for img_info in img_list:
            original_url = img_info.get('fileUrl', '')
            try:
                # 尝试下载所需要的图片
                img_info['fileUrl'] = download_image(img_info['fileUrl'], images_dir)
                valid_img_list.append(img_info['fileUrl'])
            except Exception as download_err:
                # 如果下载失败，记录错误，不要抛出异常
                print(f"图片下载失败: {original_url}, 错误: {download_err}")
                failed_results.append({
                    'classify': '下载失败',
                    'sorce': 0.0,
                    'angle': 0,
                    'fileUrl': original_url,
                    'fileId': img_info.get('fileId'),
                    'fileName': img_info.get('fileName'),
                    'message': f"下载失败: {str(download_err)}"
                })
        # 2026.3.3 增加新场景
        if bussiness_type == "工程报销支付申请_电网管理平台(深圳局)—物资":
            class_names =["其它", "发票", "商城到货单", "订单合同", "转账凭证"]      # 训练权重路径
            # weights_path = "Models/checkpoints/classify_6.pth"
        elif bussiness_type == "非工程物资报销申请单":
            class_names = ["其它", "发票", "订单合同", "质保金", "银付凭证"]
            # weights_path = "Models/checkpoints/model_epoch_10.pth"
        elif bussiness_type == "工程报销支付申请_电网管理平台(深圳局)":
            class_names =['其它', '发票', '支付申请单或XX证明', '订单合同', '转账凭证']               # 训练权重路径
            # weights_path = "Models/checkpoints/classify_6.pth"
        elif bussiness_type == "工程报销支付申请(扫描中心)_发票池":
             class_names =['其它', '发票', '商城到货单', '支付申请单或XX证明', '订单合同', '转账凭证']               # 训练权重路径
            # weights_path = "Models/checkpoints/classify_6.pth"
        elif bussiness_type == "工程报销支付审批单(扫描中心)_电网管理平台":
            class_names =['其它', '发票', '质保金']               # 训练权重路径
            # weights_path = "Models/checkpoints/classify_6.pth"
        elif bussiness_type == "工程报销支付审批单(扫描中心)":
            class_names =['其它', '发票', '质保金']               # 训练权重路径
            # weights_path = "Models/checkpoints/classify_6.pth"
        elif bussiness_type == "差旅费季度报销_扫描中心":
            class_names =['其它','出差申请单', '发票','机票订单','火车票订单', '酒店订单','银付凭证']               # 训练权重路径
            # weights_path = "Models/checkpoints/classify_6.pth"
        elif bussiness_type == "工程差旅费报销审批流程(扫描中心) _发票池_电网管理平台":
            class_names =['其它','出差申请单', '发票','机票订单','火车票订单', '酒店订单','银付凭证']               # 训练权重路径
            # weights_path = "Models/checkpoints/classify_6.pth"

        # 对图像进行变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        result = predict_list(img_list, transform, class_names,bussiness_type)
        # for i in range(min(len(result['data']), len(temp))):
        #     result['data'][i]['fileUrl'] = temp[i]    
        json_result = jsonable_encoder(result)
        return JSONResponse(content=json_result)
    except Exception as e:
        json_result = jsonable_encoder({
            'code': 500,
            'message': f"An error occurred: {str(e)}",
            'data': None
        })
        return JSONResponse(content=json_result)



def pdf_2_images(pdf_document, output_folder, page_num):
    """PDF处理辅助函数，返回指定页码的单张图片保存路径"""
    # 获取页面
    page = pdf_document.load_page(page_num)

    # 将页面转换成图像, 渲染为像素图
    pix = page.get_pixmap()

    # 使用PIL保存图像
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # match = re.search(r'\\([^\\]+)\.pdf$', pdf_document.name)
    # contract_name = match.group(1)
    contract_name = os.path.basename(pdf_document.name)
    image_filename = "{}{}page_{}.png".format(output_folder, contract_name, page_num + 1)
    img.save(image_filename, "PNG")
    print("合同分页保存位置：", image_filename)
    return image_filename

def contract_homepage_info(pdf_path):
    """合同首页(前6页)处理"""
    new_url = []
    pdf_document = fitz.open(pdf_path)

    base64_image_list = []
    for page_num in range(6):
        image_filename = pdf_2_images(pdf_document, images_dir, page_num)
        ## TODO new_url  框图片
        base64_image_list.append(encode_image(image_filename))

    output = run_conv(config.contract, base64_image_list)

    return new_url, output

# def contract_payment(pdf_path):
#     """合同价款处理"""
#     page = []
#     picpath = []

#     # json_data = """
#     #     {
#     #         "合同价款":"",
#     #         "是否按合同约定票据比例支付":""
#     #     }
#     # """
#     # result = json.loads(json_data)
#     result = {"合同价款": "", "是否按合同约定票据比例支付": ""}
#     pdf_document = fitz.open(pdf_path)
#     is_find_payment = False

#     for page_num in range(len(pdf_document)):
#         image_filename = pdf_2_images(pdf_document, images_dir, page_num)
#         # 根据状态选择提示词
#         if is_find_payment:
#             prompt = config.contract_method
#         else:
#             prompt = config.contract_amount

#         data = run_conv(prompt, [encode_image(image_filename)])

#         if json.loads(data)["boolean"] == 1:
#             result["合同价款"] = json.loads(data)["amount"]
#             page.append(re.search(r'(\d+)\.png', image_filename).group(1))
#             picpath.append(image_filename)
#             is_find_payment = True
#         if json.loads(data)["boolean"] == 2:
#             page.append(re.search(r'(\d+)\.png', image_filename).group(1))
#             picpath.append(image_filename)
#             result["是否按合同约定票据比例支付"] = json.loads(data)["result"]
#             break

#     if result["合同价款"] == "":
#         result["合同价款"] = "未找到"
#     if result["是否按合同约定票据比例支付"] == "":
#         result["是否按合同约定票据比例支付"] = "否"

#     return result, page, picpath


if __name__ == "__main__":
    # ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # ssl_context.load_cert_chain(certfile="/home/oem/llm/certificate.crt",keyfile="/home/oem/llm/private.key")
#    config = Config(app=app, host="0.0.0.0", port=8000, ssl_context=ssl_context)
#    uvicorn.run(app, host="0.0.0.0", port=8000,ssl=ssl_context)

    uvicorn.run(app, host="0.0.0.0", port=8001)

