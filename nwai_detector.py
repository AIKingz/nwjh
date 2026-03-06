import os
import numpy as np
import sys
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import requests
import torch.nn.functional as F

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

class NwaiDetector:
    """ 
    __init__       模型初始化函数,注意不能自行增加删除函数入参
    gpu_id         设备id
    gpu_rate       设备使用率,默认为None
    注意           请在try except内编写推理代码便于获取错误
    """
    def __init__(self,gpu_id,gpu_rate=None):
        try:
            self.gpu_id = gpu_id

            model_path_scene1 = "./Models/checkpoints/4090_model1.pth"
            self.class_names_scene1 = ["其它", "发票", "商城到货单", "订单合同", "转账凭证"]
            self.model_scene1 = MedicalImageClassifier(num_classes=len(self.class_names_scene1))
            self.model_scene1.load_state_dict(torch.load(model_path_scene1))

            model_path_scene2 = "./Models/checkpoints/4090_model2.pth"
            self.class_names_scene2 = ["其它", "发票", "订单合同", "质保金", "银付凭证"]
            self.model_scene2 = MedicalImageClassifier(num_classes=len(self.class_names_scene2))
            self.model_scene2.load_state_dict(torch.load(model_path_scene2))

            model_path_scene3 = "./Models/checkpoints/4090_model3.pth"
            self.class_names_scene3 = ['其它', '发票', '支付申请单或XX证明', '订单合同', '转账凭证']
            self.model_scene3 = MedicalImageClassifier(num_classes=len(self.class_names_scene3))
            self.model_scene3.load_state_dict(torch.load(model_path_scene3))

            model_path_scene4 = "./Models/checkpoints/4090_model4.pth"
            self.class_names_scene4 = ['其它', '发票', '商城到货单', '支付申请单或XX证明', '订单合同', '转账凭证']
            self.model_scene4 = MedicalImageClassifier(num_classes=len(self.class_names_scene4))
            self.model_scene4.load_state_dict(torch.load(model_path_scene4))

            model_path_scene56 = "./Models/checkpoints/4090_model56_epoch_8.pth"
            self.class_names_scene56 = ['其它', '发票', '质保金']
            self.model_scene56 = MedicalImageClassifier(num_classes=len(self.class_names_scene56))
            self.model_scene56.load_state_dict(torch.load(model_path_scene56))
            
            # 2026.3.3 增加新场景
            model_path_scene6 = "./Models/checkpoints/4090_model6.pth"
            self.class_names_scene6 = ['其它','出差申请单', '发票','机票订单','火车票订单', '酒店订单','银付凭证']
            self.model_scene6 = MedicalImageClassifier(num_classes=len(self.class_names_scene6))
            self.model_scene6.load_state_dict(torch.load(model_path_scene6))
            
            model_path_scene7 = "./Models/checkpoints/4090_model7.pth"
            self.class_names_scene7 = ['其它','出差申请单', '发票','机票订单','火车票订单', '酒店订单','银付凭证']
            self.model_scene7 = MedicalImageClassifier(num_classes=len(self.class_names_scene7))
            self.model_scene7.load_state_dict(torch.load(model_path_scene7))

            self.current_model = self.model_scene1  # 默认使用第一个模型
            self.current_class_names = self.class_names_scene1  # 默认类别名称
            self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        except Exception as err:
            raise Exception("[Error] NwaiDetector模型初始化函数异常.错误信息:{}".format(ExceptionMessage(err)))
    
    
    def detect(self,image_nparray,ai_param=None):
        """
        当前默认处理单张图片，没有加多图的循环判断，因为多图同时需要多encoded_text
        定义ai_param为字典，目前格式：
        {
            "bussiness_type": "信息化项目",
            "encoded_text": {},
        }
        """
        try:
            image_nparray = image_nparray.to(self.device)

            if isinstance(ai_param, dict):
                param=ai_param["ai_param"]
                bussiness_type = param["bussiness_type"]
                encoded_text = param["encoded_text"]
            else:
                bussiness_type = "工程物资"
                encoded_text = {}
            detect_result = []
            # 切换对应场景模型
            if bussiness_type == "工程报销支付申请_电网管理平台(深圳局)—物资":
                self.current_model = self.model_scene1
                self.current_class_names = self.class_names_scene1
            elif bussiness_type == "非工程物资报销申请单":
                self.current_model = self.model_scene2
                self.current_class_names = self.class_names_scene2
            elif bussiness_type == "工程报销支付申请_电网管理平台(深圳局)":
                self.current_model = self.model_scene3
                self.current_class_names = self.class_names_scene3
            elif bussiness_type == "工程报销支付申请(扫描中心)_发票池":
                self.current_model = self.model_scene4
                self.current_class_names = self.class_names_scene4
            elif bussiness_type == "工程报销支付审批单(扫描中心)_电网管理平台":
                self.current_model = self.model_scene56
                self.current_class_names = self.class_names_scene56
            elif bussiness_type == "工程报销支付审批单(扫描中心)":
                self.current_model = self.model_scene56
                self.current_class_names = self.class_names_scene56
                
            # 2026.3.3 增加新场景
            elif bussiness_type == "差旅费季度报销_扫描中心":
                self.current_model = self.model_scene6
                self.current_class_names = self.class_names_scene6
            elif bussiness_type == "工程差旅费报销审批流程(扫描中心) _发票池_电网管理平台":
                self.current_model = self.model_scene7
                self.current_class_names = self.class_names_scene7

            self.current_model.eval()
            self.current_model.to(self.device)
            # 预测
            with torch.no_grad():
                outputs = self.current_model(image_nparray, encoded_text)
            # 获取预测结果
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)

            outputs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(outputs, 1)

            # predicted_class = self.current_class_names[int(predicted.item())]
            detect_result.append([int(predicted.item()),confidence.item()])
            return detect_result
        except Exception as err:
            print("[Error] NwaiDetector.detect函数推理异常.错误信息:{}".format(ExceptionMessage(err)))
            return err

### 获取异常文件+行号+信息
def ExceptionMessage(err):
    err_message=(
        str(err.__traceback__.tb_frame.f_globals["__file__"])
        +":"
        +str(err.__traceback__.tb_lineno)
        +"行:"
        +str(err)
    )
    return err_message

def get_image_url(image_name,image_path):
    # url = "https://ioc.zhi-zai.com/process-factory/admin-api/infra/file/upload"
    url = "http://10.123.216.38:28489/process-factory/admin-api/infra/file/upload"

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

def detect_rotation_angle(image: np.ndarray,img_name='',img_path=''):
    """
    使用边缘检测和霍夫变换检测文档的旋转角度。

    参数:
        image: 输入图像，类型为 numpy 数组
    返回值:
        int: 检测到的旋转角度（以度为单位，转换为整数）。如果发生错误，返回 0。
    """
    minio_url = ''
    try:
        # 如果图像是三通道（彩色图像），转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 使用 Canny 边缘检测算法提取图像边缘
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 使用霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        angles = []
        if lines is not None:
            # 遍历每条检测到的直线
            for rho, theta in lines[:, 0]:
                # 将弧度转换为角度
                angle = np.degrees(theta)
                # 调整角度范围为 [-90, 90]
                if angle > 90:
                    angle = angle - 180

                # 过滤掉不在 [-30, 30] 范围内的角度，只选在范围内的角度
                if -30 <= angle <= 30:
                    angles.append(angle)

        # 如果没有检测到符合条件的直线，返回 0
        if not angles:
            return 0
        # 如果存在角度
        if img_name and img_path:
            minio_url = get_image_url(img_name,img_path)
        else:
            minio_url = ''

        # 返回中位数角度，并将其转换为整数
        return int(np.median(angles)),minio_url

    except Exception as e:
        # 捕获任何异常并返回 0
        print(f"检测角度发生错误: {e}")  # 打印错误信息以便调试
        return 0,minio_url

if __name__ == '__main__':
    from PIL import Image
    detector = NwaiDetector(gpu_id=0)
    image_path = "./testimage.jpg"
    image_nparray = cv2.imread(image_path)
    image_nparray = cv2.cvtColor(image_nparray, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_nparray)
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    image_tensor = transform(pil_image).unsqueeze(0)  # 增加batch维度
    
    
    detect_result = detector.detect(image_tensor)
    print('detect_result:', detect_result)
    ###模拟传入AI参数示例
    ai_param={
        "bussiness_type": "信息化项目",
        "encoded_text": {},
    }
    detect_result = detector.detect(image_tensor,ai_param)
    print('detect_result:', detect_result)

    ai_param={
        "bussiness_type": "非工程物资",
        "encoded_text": {},
    }
    detect_result = detector.detect(image_tensor,ai_param)
    print('detect_result:', detect_result)

