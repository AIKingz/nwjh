from collections import Counter
from ultralytics import YOLO
import os
import cv2
import numpy as np
from typing import List, Dict, Any

# ------------------------------
# 配置部分
# ------------------------------
MODEL_PATH = "./Models/yolo/best.pt"
model = YOLO(MODEL_PATH)

BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
DEBUG_RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "predicted_results")
os.makedirs(DEBUG_RESULTS_DIR, exist_ok=True)

# 英文->中文映射
CLASS_NAME_MAP = {
    "seal": "盖章",
    "sign": "签名"
}

def translate_class_name(name: str) -> str:
    """将模型类名翻译为中文"""
    return CLASS_NAME_MAP.get(name, name)  # 未映射的类名保持原样

def apply_nms_by_class(results, iou_threshold: float = 0.5):
    """
    对每个类别独立应用非极大值抑制。
    
    Args:
        results: 模型推理结果，包含一个或多个图像的Results对象。
        iou_threshold (float): 非极大值抑制的IoU阈值。
    
    Returns:
        List[Dict]: 包含每个图像经过NMS处理后的检测框列表。
    """
    final_results = []
    for res in results:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            final_results.append([])
            continue

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            detections.append({
                "box": [x1, y1, x2, y2],
                "conf": conf,
                "cls_id": cls_id
            })

        final_detections = []
        # 按类别分组检测框
        detections_by_class = {}
        for d in detections:
            cls_id = d["cls_id"]
            if cls_id not in detections_by_class:
                detections_by_class[cls_id] = []
            detections_by_class[cls_id].append(d)
        
        # 对每个类别应用NMS
        for cls_id in detections_by_class:
            class_detections = detections_by_class[cls_id]
            # 按置信度排序
            class_detections.sort(key=lambda x: x["conf"], reverse=True)
            
            while class_detections:
                best_detection = class_detections.pop(0)
                final_detections.append(best_detection)
                
                # 移除与最佳检测框重叠的检测框
                class_detections = [
                    d for d in class_detections
                    if calculate_iou(best_detection["box"], d["box"]) < iou_threshold
                ]
        
        final_results.append(final_detections)

    return final_results

def calculate_iou(box1, box2):
    """计算两个边界框的IoU。"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def sign_seal_detect(images: List[str], conf: float = 0.3, iou: float = 0.5, debug: bool = False):
    """
    检测图像中的签章数量。

    Args:
        images (List[str]): 图像文件路径列表
        conf (float, optional): 置信度阈值. 默认 0.3
        iou (float, optional): 非极大值抑制的IoU阈值. 默认 0.5
        debug (bool, optional): 是否保存检测可视化结果. 默认 False

    Returns:
        List[Dict]: 返回包含每张图片检测结果的字典
        [
            {
                "image": 图像路径,
                "counts": {类名: 数量, ...},
                "saved_path": 保存路径或None
            },
            ...
        ]
       
    """
    if not images:
        raise ValueError("图像列表不能为空。")

    try:
        results = model(images, conf=conf, device="0", iou=iou)
    except Exception as e:
        raise RuntimeError(f"模型推理失败: {e}")

    # 将YOLOv8内置的NMS替换为自定义的按类别NMS
    # results_processed = apply_nms_by_class(results, iou_threshold=iou)
    
    class_names = model.names
    details = []

    for i, (img_path, res) in enumerate(zip(images, results), start=1):
        if not os.path.exists(img_path):
            details.append({
                "image": img_path,
                "counts": {},
                "saved_path": None,
                "error": "图像路径不存在"
            })
            continue

        # 将 results.boxes 转换为一个列表，方便处理
        detections = []
        if res.boxes is not None:
            for box in res.boxes:
                detections.append({
                    "box": box.xyxy[0].cpu().numpy(),
                    "conf": box.conf[0].cpu().numpy(),
                    "cls_id": int(box.cls[0].cpu().numpy())
                })
        
        # 按类别应用NMS
        final_detections = []
        classes = {d['cls_id'] for d in detections}
        for cls_id in classes:
            class_detections = [d for d in detections if d['cls_id'] == cls_id]
            class_detections.sort(key=lambda x: x['conf'], reverse=True)
            
            while len(class_detections) > 0:
                best_detection = class_detections.pop(0)
                final_detections.append(best_detection)
                
                class_detections = [
                    d for d in class_detections
                    if calculate_iou(best_detection['box'], d['box']) < iou
                ]
        
        cls_ids = [d['cls_id'] for d in final_detections]
        counts = Counter(cls_ids)
         # 将类别名翻译为中文
        class_counts = {}
        for k, v in counts.items():
            en_name = class_names.get(k, f"未知类别({k})")
            zh_name = translate_class_name(en_name)
            class_counts[zh_name] = v

        saved_path = None
        if debug:
            try:
                # 重新绘制图像，只使用经过 NMS 后的检测框
                img = cv2.imread(img_path)
                for det in final_detections:
                    box = det['box'].astype(int)
                    cls_id = det['cls_id']
                    conf = det['conf']
                    label = f"{translate_class_name(class_names[cls_id])}: {conf:.2f}"
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                saved_path = os.path.join(DEBUG_RESULTS_DIR, os.path.basename(img_path))
                cv2.imwrite(saved_path, img)
            except Exception as e:
                saved_path = f"保存失败: {e}"

        details.append({
            "image": img_path,
            "counts": class_counts,
            "saved_path": saved_path
        })

    return details
    


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    test_images = [
    "test_data/data/质保金/5 003.jpg", 
    ]
    result = sign_seal_detect(test_images, conf=0.3, iou=0.5, debug=True)
    print(result)
# ***************************
# *****************************上面是原始代码


# from collections import Counter
# from ultralytics import YOLO
# import os
# import cv2
# import numpy as np
# from typing import List, Dict, Any
# from paddleocr import PaddleOCR

# # ------------------------------
# # 配置部分
# # ------------------------------
# MODEL_PATH = "./Models/yolo/best.pt"
# model = YOLO(MODEL_PATH)

# BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "tmp")
# DEBUG_RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "predicted_results")
# os.makedirs(DEBUG_RESULTS_DIR, exist_ok=True)

# # 英文->中文映射
# CLASS_NAME_MAP = {
#     "seal": "盖章",
#     "sign": "签名"
# }

# # 初始化 PaddleOCR 模型，只在程序启动时加载一次
# # 使用英文识别模型，对于印章文字识别可能需要进一步调优
# try:
#     # 修复：移除 to_args，直接实例化 PaddleOCR
#     ocr_model = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False)
# except Exception as e:
#     ocr_model = None
#     print(f"警告：PaddleOCR 初始化失败。将跳过印章内容识别。错误：{e}")

# def translate_class_name(name: str) -> str:
#     """将模型类名翻译为中文"""
#     return CLASS_NAME_MAP.get(name, name)

# def apply_nms_by_class(results, iou_threshold: float = 0.5):
#     """
#     对每个类别独立应用非极大值抑制。
    
#     Args:
#         results: 模型推理结果，包含一个或多个图像的Results对象。
#         iou_threshold (float): 非极大值抑制的IoU阈值。
    
#     Returns:
#         List[List[Dict]]: 包含每个图像经过NMS处理后的检测框列表。
#     """
#     final_results = []
#     for res in results:
#         boxes = res.boxes
#         if boxes is None or len(boxes) == 0:
#             final_results.append([])
#             continue

#         detections = []
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             conf = box.conf[0].cpu().numpy()
#             cls_id = int(box.cls[0].cpu().numpy())
#             detections.append({
#                 "box": [x1, y1, x2, y2],
#                 "conf": conf,
#                 "cls_id": cls_id
#             })

#         final_detections = []
#         # 按类别分组检测框
#         detections_by_class = {}
#         for d in detections:
#             cls_id = d["cls_id"]
#             if cls_id not in detections_by_class:
#                 detections_by_class[cls_id] = []
#             detections_by_class[cls_id].append(d)
        
#         # 对每个类别应用NMS
#         for cls_id in detections_by_class:
#             class_detections = detections_by_class[cls_id]
#             # 按置信度排序
#             class_detections.sort(key=lambda x: x["conf"], reverse=True)
            
#             while class_detections:
#                 best_detection = class_detections.pop(0)
#                 final_detections.append(best_detection)
                
#                 # 移除与最佳检测框重叠的检测框
#                 class_detections = [
#                     d for d in class_detections
#                     if calculate_iou(best_detection["box"], d["box"]) < iou_threshold
#                 ]
        
#         final_results.append(final_detections)

#     return final_results

# def calculate_iou(box1: List[float], box2: List[float]) -> float:
#     """计算两个边界框的IoU。"""
#     x1_min, y1_min, x1_max, y1_max = box1
#     x2_min, y2_min, x2_max, y2_max = box2
    
#     xi1 = max(x1_min, x2_min)
#     yi1 = max(y1_min, y2_min)
#     xi2 = min(x1_max, x2_max)
#     yi2 = min(y1_max, y2_max)
    
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
#     box1_area = (x1_max - x1_min) * (y1_max - y1_min)
#     box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
#     union_area = box1_area + box2_area - inter_area
    
#     return inter_area / union_area if union_area > 0 else 0

# def sign_seal_detect(images: List[str], conf: float = 0.3, iou: float = 0.5, debug: bool = False) -> List[Dict[str, Any]]:
#     """
#     检测图像中的签章数量及盖章内容。
#     ...
#     """
#     if not images:
#         raise ValueError("图像列表不能为空。")

#     try:
#         results = model(images, conf=conf, iou=iou)
#     except Exception as e:
#         raise RuntimeError(f"模型推理失败: {e}")
    
#     final_detections_list = apply_nms_by_class(results, iou_threshold=iou)

#     class_names = model.names
#     all_image_results = []

#     for img_path, final_detections in zip(images, final_detections_list):
#         image_result = {
#             "image": img_path,
#             "counts": {},
#             "details": [],
#             "saved_path": None
#         }

#         if not os.path.exists(img_path):
#             image_result["error"] = "图像路径不存在"
#             all_image_results.append(image_result)
#             continue
        
#         cls_ids = [d['cls_id'] for d in final_detections]
#         counts = Counter(cls_ids)
#         class_counts = {translate_class_name(class_names.get(k, f"未知类别({k})")): v for k, v in counts.items()}
#         image_result["counts"] = class_counts
        
#         img = cv2.imread(img_path)
        
#         for det in final_detections:
#             en_name = class_names.get(det['cls_id'])
#             zh_name = translate_class_name(en_name)
#             box_coords = np.array(det['box']).astype(int)
            
#             det_info = {
#                 "class": zh_name,
#                 "confidence": float(det['conf']),
#                 "box": box_coords.tolist()
#             }
            
#             # 如果是盖章，则进行内容识别
#             if zh_name == "盖章" and ocr_model:
#                 try:
#                     x1, y1, x2, y2 = box_coords
#                     cropped_seal = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
                    
#                     b, g, r = cv2.split(cropped_seal)
#                     seal_content_img = r
                    
#                     # PaddleOCR识别，不需要边界框信息，因此 det=False
#                     ocr_results = ocr_model.ocr(seal_content_img, det=False, rec=True)
#                     content_list = []

#                     if ocr_results and ocr_results[0] is not None:
#                         for line in ocr_results[0]:
#                             content_list.append(line[0])
                    
#                     # 按照您的要求，将识别到的所有文本内容作为列表赋给 "盖章内容" 键
#                     det_info["盖章内容"] = content_list
                    
#                 except Exception as e:
#                     det_info["盖章内容"] = []
#                     det_info["ocr_error"] = f"OCR识别失败: {e}"
            
#             image_result["details"].append(det_info)

#     return all_image_results

# if __name__ == "__main__":
#     # test_images = [
#     #     "test_data/非工程物资/质保金/3 003.jpg",
#     # ]
#     # # 你的 YOLO 模型需要能够检测到“seal”类别
#     # result = sign_seal_detect(test_images, conf=0.3, iou=0.5, debug=True)
#     # print(result)
#     test_images = [
#         "test_data/data/质保金/5 003.jpg",
#     ]
#     result = sign_seal_detect(test_images, conf=0.3, iou=0.5, debug=True)
    
#     print(result)
