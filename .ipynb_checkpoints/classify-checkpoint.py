import json
import requests
import random
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.encoders import jsonable_encoder
from torchvision import transforms, models
from PIL import Image
from transformers import BertTokenizer, BertModel
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

from nwai_detector import NwaiDetector


class MedicalImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MedicalImageClassifier, self).__init__()
        # 使用更轻量级的ResNet模型
        from torchvision.models import ResNet18_Weights
        # 使用新的weights API
        #weights = ResNet18_Weights.IMAGENET1K_V1 
        # 直接从本地加载 ResNet18 的权重
        self.resnet = models.resnet18(pretrained=False)  # 不加载预训练权重
        self.resnet.load_state_dict(torch.load('Models/resnet/resnet18-5c106cde.pth', map_location=torch.device('cpu'),weights_only=False))
        self.resnet.fc = nn.Identity()
        
        # 使用更小的BERT模型
        # self.bert = BertModel.from_pretrained('Models/bert')
        self.bert = BertModel.from_pretrained('Models/saft_bert', use_safetensors=True)
        
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, encoded_text):
        # 获取图像和文本的特征
        image_features = self.resnet(image)
        text_features = self.bert(**encoded_text)[0]  # 使用序列输出
        
        # 应用注意力机制
        text_features, _ = self.attention(text_features, text_features, text_features)
        text_features = torch.mean(text_features, dim=1)
        # 融合图像与文本的特征
        combined_features = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined_features)




def predict_single2(image_path, weights_path, transform, class_names):
    """
    使用预训练权重预测单张图片的类别。

    参数:
    - image_path: 要预测的图片路径。
    - weights_path: 预训练模型权重的路径。
    - transform: 图片预处理的transform。
    - class_names: 类别名称的列表。
    """

    # 加载预训练模型权重
    model = MedicalImageClassifier(num_classes=len(class_names))

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型

    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # 图片预处理
    image = Image.open(image_path).convert('RGB')

    # paddleOCR提取文本信息
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    ocrreader = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='ch')
    ocr_result = ocrreader.ocr(img_cv)
    text = ""
    keywords = class_names
    for result in ocr_result:
        if result:
            for item in result:
                if item[1]:
                    text_temp, _ = item[1]  # 提取文本，忽略置信度
                    text += text_temp + " "
                else:
                    text += "无识别文本"
                    # item[1] 是一个元组，其中第一个元素是识别的文本
                if any(keyword in text for keyword in keywords):
                    print(f"找到关键字，停止提取。找到的关键字1：'{text}'")
                    break
            if any(keyword in text for keyword in keywords):
                print(f"找到关键字，停止提取。找到的关键字2：'{text}'")
                break
        else:
            text += "无识别文本"
    text = ' '.join([str(item) for item in ocr_result]) if ocr_result else "无识别文本"
    max_length = 512
    tokenizer = BertTokenizer.from_pretrained('Models/bert')
    encoded_text = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length = max_length,
                padding = 'max_length',
                truncation = True,
                return_attention_mask=True,
                return_tensors='pt'
            )


    image = transform(image).unsqueeze(0)  # 增加batch维度

    # 将图片移动到设备（CPU或GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    encoded_text = encoded_text.to(device)
    model = model.to(device)

    # 预测
    with torch.no_grad():
        # outputs = model(image, {'input_ids': torch.zeros((1, 1), dtype=torch.long).to(device), 
        #                        'attention_mask': torch.zeros((1, 1), dtype=torch.long).to(device)})
        outputs = model(image, encoded_text)

    # 获取预测结果
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

    return predicted_class

def get_image_url(image_name,image_path):
    url = "https://ioc.zhi-zai.com/process-factory/admin-api/infra/file/upload"

    payload = {}
    # 以二进制模式读取图片
    with open(image_path, 'rb') as file:
        file_data = file.read()
    # 构造请求参数（根据接口要求调整字段名）
    mime_type = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'pdf': 'application/pdf'
    }

    files = {
        'file': (image_name, file_data, mime_type[image_name.split('.')[-1]])  # 关键：直接传递二进制数据
    }
    headers = {
    'Content-Type': 'application/json'
    }
    # 发送POST请求（根据接口需要添加headers/params/data）
    response = requests.post(url, files=files)

    print(response.text)
    result = response.json()  # 解析JSON响应
    # 检查code是否为0（成功）
    if result.get('code') == 0:
        return result['data']  # 返回图片URL
    else:
        raise Exception(f"上传失败: {result.get('msg', '未知错误')}")

# def detect_rotation_angle(image: np.ndarray,img_name='',img_path=''):
#     """
#     使用边缘检测和霍夫变换检测文档的旋转角度。
# 
#     参数:
#         image: 输入图像，类型为 numpy 数组
#     返回值:
#         int: 检测到的旋转角度（以度为单位，转换为整数）。如果发生错误，返回 0。
#     """
#     minio_url = ''
#     try:
#         # 如果图像是三通道（彩色图像），转换为灰度图像
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image
# 
#         # 使用 Canny 边缘检测算法提取图像边缘
#         edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# 
#         # 使用霍夫变换检测直线
#         lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
# 
#         angles = []
#         if lines is not None:
#             # 遍历每条检测到的直线
#             for rho, theta in lines[:, 0]:
#                 # 将弧度转换为角度
#                 angle = np.degrees(theta)
#                 # 调整角度范围为 [-90, 90]
#                 if angle > 90:
#                     angle = angle - 180
# 
#                 # 过滤掉不在 [-30, 30] 范围内的角度，只选在范围内的角度
#                 if -30 <= angle <= 30:
#                     angles.append(angle)
# 
#         # 如果没有检测到符合条件的直线，返回 0
#         if not angles:
#             return 0
#         # 如果存在角度
#         if img_name and img_path:
#             minio_url = get_image_url(img_name,img_path)
#         else:
#             minio_url = ''
# 
#         # 返回中位数角度，并将其转换为整数
#         return int(np.median(angles)),minio_url
# 
#     except Exception as e:
#         # 捕获任何异常并返回 0
#         print(f"检测角度发生错误: {e}")  # 打印错误信息以便调试
#         return 0,minio_url



def predict_folder(images_dir, weights_path, transform, class_names):
    """
    使用预训练权重预测文件夹下所有图片的类别。

    参数:
    - images_dir: 包含要预测图片的文件夹路径。
    - weights_path: 预训练模型权重的路径。
    - transform: 图片预处理的transform。
    - class_names: 类别名称的列表。
    """
    # 加载预训练模型权重
    model = MedicalImageClassifier(num_classes=len(class_names))

    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    # 确保文件夹路径存在
    if not os.path.exists(images_dir) or not os.path.isdir(images_dir):
        raise ValueError(f"提供的路径不是一个有效的目录：{images_dir}")

    # 初始化OCR
    ocrreader = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='ch', det_model_dir='Models/ch_PP-OCRv4_det_infer', rec_model_dir='Models/ch_PP-OCRv4_rec_infer', cls_model_dir='Models/ch_ppocr_mobile_v2.0_cls_infer')
    tokenizer = BertTokenizer.from_pretrained('Models/bert')

    keywords = class_names
    text = ""
    images_info = []


    # 遍历文件夹中的每个图片文件
    for img_filename in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_filename)
        
        # 检查文件是否是图片
        if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 图片预处理
            image = Image.open(img_path).convert('RGB')
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            # 使用OCR提取文本信息
            ocr_result = ocrreader.ocr(img_cv)
            for result in ocr_result:
                if result:
                    for item in result:
                        if item[1]:
                            text_temp, _ = item[1]  # 提取文本，忽略置信度
                            text += text_temp + " "
                        else:
                            text += "无识别文本"
                else:
                    text += "无识别文本"
            
            text = text[:50]
            max_length = 512
            # 将文本映射为数值张量
            encoded_text = tokenizer.encode_plus(
                text,
                add_special_tokens=True, # 添加特殊标记（如[CLS]开头、[SEP]结尾）
                max_length=max_length, # 序列最大长度
                padding='max_length', # 短文本补零至max_length。
                truncation=True, # 长文本截断至max_length。
                return_attention_mask=True, # 生成注意力掩码
                return_tensors='pt' # 返回PyTorch张量格式
            )
            
            image = transform(image).unsqueeze(0)  # 增加batch维度

            # 将图片和文本编码移动到设备（CPU或GPU）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image = image.to(device)
            encoded_text = {key: val.to(device) for key, val in encoded_text.items()}
            model = model.to(device)

            # 预测
            with torch.no_grad():
                outputs = model(image, encoded_text)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

            # 打印或保存预测结果
            image_info = {
                '图像地址': img_path,
                'class': predicted_class
            }
            images_info.append(image_info)
            print(f"图片：{img_filename} 预测类别：{predicted_class}")

    return images_info


# 将图片和文本编码移动到设备（CPU或GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['其他', '发票', '叠票', '阶段工作支付申请表', '阶段工作确认表', '验收证书']
# 加载预训练模型权重
model = MedicalImageClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load("Models/checkpoints/classify_6.pth"))
model.eval()
model = model.to(device)

# 初始化OCR
# ocrreader = PaddleOCR(use_angle_cls=True, lang='ch',
#                       det_model_dir='Models/ch_PP-OCRv4_det_infer',
#                       rec_model_dir='Models/ch_PP-OCRv4_rec_infer',
#                       use_gpu=True)

# # 初始化OCR
ocrreader = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='ch',
                      det_model_dir='Models/ch_PP-OCRv4_det_server_infer',
                      rec_model_dir='Models/ch_PP-OCRv4_rec_server_infer',
                      cls_model_dir='Models/ch_ppocr_mobile_v2.0_cls_infer')
tokenizer = BertTokenizer.from_pretrained('Models/bert')

def init_model(weights_path,class_names):
    """
    初始化模型和OCR
    """
    # 加载预训练模型权重
    model = MedicalImageClassifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(weights_path))
    return model

# 2026.2.2修改
def predict_list(images_list, transform,class_names=[],bussiness_type='信息化项目'):
    """
    预测列表
    """
    try:

        # # 将图片和文本编码移动到设备（CPU或GPU）
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # # 加载预训练模型权重
        # model = MedicalImageClassifier(num_classes=len(class_names))
        # model.load_state_dict(torch.load(weights_path))
        # model.eval()
        # model = model.to(device)
        #
        # # 初始化OCR
        # ocrreader = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='ch',
        #                       det_model_dir='Models/ch_PP-OCRv4_det_infer',
        #                       rec_model_dir='Models/ch_PP-OCRv4_rec_infer',
        #                       cls_model_dir='Models/ch_ppocr_mobile_v2.0_cls_infer')
        # tokenizer = BertTokenizer.from_pretrained('Models/bert')

        text = ""
        image_info_list = []
        
        # # 非法文件列表
        # unlegal_files = []
        # 初始化NwaiDetector
        detector = NwaiDetector(gpu_id=0)
        

        # 遍历文件夹中的每个图片文件
        for img_info in images_list:
            img_path = img_info['fileUrl']
            minio_url = img_path
            img_id = img_info['fileId']
            img_Name = img_info['fileName']

            # 判断文件是否存在，或者是否是图片
            # 文件不存在或不可访问时，归为「其它」，不报错
            if not (os.path.isfile(img_path)):
                # unlegal_files.append({
                #     "fileId": img_id,
                #     "fileName": img_Name,
                #     "fileUrl": img_path
                image_info_list.append({
                    'classify': '其它',
                    'sorce': 0.0,
                    'fileUrl': minio_url,
                    'fileId': img_id,
                    'fileName': img_Name
                })
                continue

            # 获取文件扩展名
            file_ext = os.path.splitext(img_path)[1].lower()

            # 处理PDF文件
            if file_ext == '.pdf':
                # minio_url = get_image_url(img_Name,img_path)
                try:
                    # 将PDF转换为图片
                    pdf_images = convert_pdf_to_images(img_path)
                    if not pdf_images:
                        raise ValueError(f"无法转换PDF文件: {img_path}")

                    # 对每个PDF页面图片进行处理
                    # for page_num, image_path in enumerate(pdf_images):
                    # 图片预处理
                    image = Image.open(pdf_images[0]).convert('RGB')
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

		    # angle,_ = detect_rotation_angle(img_cv)
                    # angle_result = detect_rotation_angle(img_cv)
                    # if isistance(angle_result, tuple):
                    #     angle, _ = angle_result
                    # else:
                    #     angle = 0

                    # 使用OCR提取文本信息
                    ocr_result = ocrreader.ocr(img_cv)
                    text = ""
                    for result in ocr_result:
                        if result:
                            for item in result:
                                if item[1]:
                                    text_temp, _ = item[1]  # 提取文本，忽略置信度
                                    text += text_temp + " "
                                else:
                                    text += "无识别文本"
                        else:
                            text += "无识别文本"

                    text = text[:50]
                    max_length = 512
                    encoded_text = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )

                    image = transform(image).unsqueeze(0)  # 增加batch维度
                    # image = image.to(device)
                    encoded_text = {key: val.to(device) for key, val in encoded_text.items()}

                    # model = init_model(weights_path, class_names)
                    # model.eval()
                    # model = model.to(device)
                    
                    
                    # # 预测
                    # with torch.no_grad():
                    #     outputs = model(image, encoded_text)

                    # # 获取预测结果
                    # if outputs.dim() == 1:
                    #     outputs = outputs.unsqueeze(0)

                    # outputs = F.softmax(outputs, dim=1)
                    # confidence, predicted = torch.max(outputs, 1)
                    ai_param = {
                        "ai_param":{
                            "bussiness_type": bussiness_type,
                            "encoded_text": encoded_text,
                        }
                    }
                    class_index,confidence = detector.detect(image,ai_param)[0]

                    predicted_class = class_names[class_index]

                    # 为PDF页面创建信息
                    image_info = {
                        'classify': predicted_class,
                        'sorce': confidence,
                        # 'angle': angle,
                        'fileUrl': minio_url, # 上传pdf的url
                        'fileId': img_id,
                        'fileName': img_Name,
                        # 'originalPdf': img_path  # 保留原始PDF路径
                    }
                    image_info_list.append(image_info)

                except Exception as e:
                    print(f"处理PDF文件 {img_path} 时出错: {str(e)}")
                    image_info = {
                        'classify': '其它',
                        'sorce': 0.0,
                        # 'angle': 0.0,
                        'fileUrl': img_path,
                        'fileId': img_id,
                        'fileName': img_Name
                    }
                    image_info_list.append(image_info)
            
            # 处理图片文件
            elif file_ext in ('.png', '.jpg', '.jpeg'):
                try:
                    # 图片预处理
                    image = Image.open(img_path).convert('RGB')
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

                    # angle,minio_url = detect_rotation_angle(img_cv,img_Name,img_path)

                    # 使用OCR提取文本信息
                    ocr_result = ocrreader.ocr(img_cv)
                    text = ""
                    for result in ocr_result:
                        if result:
                            for item in result:
                                if item[1]:
                                    text_temp, _ = item[1]  # 提取文本，忽略置信度
                                    text += text_temp + " "
                                else:
                                    text += "无识别文本"
                        else:
                            text += "无识别文本"

                    text = text[:50]
                    max_length = 512
                    encoded_text = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )

                    image = transform(image).unsqueeze(0)  # 增加batch维度
                    image = image.to(device)
                    encoded_text = {key: val.to(device) for key, val in encoded_text.items()}

                    # model = init_model(weights_path, class_names)
                    # model.eval()
                    # model = model.to(device)
                    # # 预测
                    # with torch.no_grad():
                    #     outputs = model(image, encoded_text)

                    # # 获取预测结果
                    # if outputs.dim() == 1:
                    #     outputs = outputs.unsqueeze(0)

                    # outputs = F.softmax(outputs, dim=1)
                    # confidence, predicted = torch.max(outputs, 1)
                    ai_param = {
                        "ai_param":{
                            "bussiness_type": bussiness_type,
                            "encoded_text": encoded_text,
                        }
                    }
                    class_index,confidence = detector.detect(image,ai_param)[0]

                    predicted_class = class_names[class_index]

                    # 打印或保存预测结果
                    image_info = {
                        'classify': predicted_class,
                        'sorce': confidence,
                        # 'angle': angle,
                        'fileUrl': minio_url,# 远程url
                        'fileId': img_id,
                        'fileName': img_Name
                    }
                    image_info_list.append(image_info)
                except Exception as e:
                    traceback.print_exc()
                    print(f"处理图片文件 {img_path} 时出错: {str(e)}")
                    image_info = {
                        'classify': '分类出错',
                        'sorce': 0.0,
                        # 'angle': 0.0,
                        'fileUrl': minio_url,
                        'fileId': img_id,
                        'fileName': img_Name
                    }
                    image_info_list.append(image_info)
            else:
                # unlegal_files.append({
                #     "fileId": img_id,
                #     "fileName": img_Name,
                #     "fileUrl": img_path
                # 格式不支持（非 pdf/png/jpg/jpeg）时，归为「其它」，不报错
                image_info_list.append({
                    'classify': '其它',
                    'sorce': 0.0,
                    'fileUrl': minio_url,
                    'fileId': img_id,
                    'fileName': img_Name
                })

        # if len(unlegal_files) > 0:
        #     return {
        #         "code": 500,
        #         "message": "以下文件格式不支持或不存在: " + str(unlegal_files),
        #         "data": image_info_list  # 仍然返回已处理成功的文件结果
        #     }

        return {
            'code': 200,
            'message': "分类完毕",
            'data': image_info_list
        }

    except Exception as e:
        print(f"发生错误: {e}")
        traceback.print_exc()
        return {
            'code': 500,
            'message': str(e),
            'data': None
        }

def convert_pdf_to_images(pdf_path):
    """将PDF文件转换为图片列表"""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import os
        
        # 创建临时目录存放转换后的图片
        temp_dir = os.path.join(os.path.dirname(pdf_path), "temp_pdf_images")
        os.makedirs(temp_dir, exist_ok=True)
        
        pdf_document = fitz.open(pdf_path)
        image_paths = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            
            # 生成图片文件名
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            image_filename = os.path.join(temp_dir, f"{pdf_name}_page_{page_num+1}.png")
            
            # 使用PIL保存图像
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(image_filename, "PNG")
            
            image_paths.append(image_filename)
            print(f"PDF页面 {page_num+1} 已转换为图片: {image_filename}")
        
        return image_paths
    
    except Exception as e:
        print(f"PDF转换失败: {str(e)}")
        return []



if __name__ == '__main__':
    weights_path = "Models/checkpoints/npu_model1.pth"
    imag_path = 'test_data/信息化项目混合测试'

    # 对图像进行变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = ['其他', '发票', '叠票', '阶段工作支付申请表', '阶段工作确认表', '验收证书']

    # 获取所有图片文件路径
    image_files = [os.path.join(imag_path, f) for f in os.listdir(imag_path) if
                   os.path.isfile(os.path.join(imag_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 构建测试数据
    img_info_list = []
    for img_path in image_files:
        img_info = {
            'fileUrl': img_path,
            'fileId': 1,
            'fileName': 'test'
        }
        img_info_list.append(img_info)

    result_list= (predict_list(img_info_list, weights_path, transform, class_names))
    print(result_list)


    # 保存到文件
    with open('result_list.json', 'w', encoding='utf-8') as f:
        json.dump(result_list, f, indent=4)
