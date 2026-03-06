import json
import time
from minio import Minio
from minio.error import S3Error
from rapidocr_onnxruntime import RapidOCR
from Models.paddleocr.paddleocr import PaddleOCR
from ocr_utils import split_ocr_results, save_ocr_result, is_ocr_result_exist, load_ocr_result, re_search, \
    draw_ocr_results, remove_ocr_result, paddle2rappid

import os
from logger import logger
current_directory = os.getcwd()
upload_dir = os.path.join(current_directory, "data/")


ocr_engine_cpu = RapidOCR(det_model_path='Models/ch_PP-OCRv4_det_infer.onnx',
                          rec_model_path='Models/ch_PP-OCRv4_rec_infer.onnx',
                          box_thresh=0.3, return_word_box=True)

ocr_engine_gpu = PaddleOCR(use_mindir=True, lang='ch',
                          det_model_path='Models/paddleocr/inference/det/m_model.mindir',
                          rec_model_path='Models/paddleocr/inference/rec/m_model.mindir',
                          use_angle_cls=False, return_word_box=True)

def get_ocr(image_path, num_threshold = 15, is_gpu=True, json_save_path = upload_dir + "json_data"):

    if is_gpu:
        # ocr_engine = RapidOCR_paddle()
        ocr_engine = ocr_engine_gpu
        ocr_result = ocr_engine.ocr(image_path)[0]
        ## 
        ocr_result = paddle2rappid(ocr_result) 
    else:
        ocr_engine = ocr_engine_cpu
        ocr_result, _ = ocr_engine(image_path, box_thresh=0.3, return_word_box=True)
        for res in ocr_result:
            # 删除最后一个元素
            res.pop()

    if ocr_result == None:
        ocr_result = []

    return ocr_result


def get_ocr_list(image_path_list,num_threshold = 15, is_gpu=True, json_save_path = upload_dir + "json_data"):
    """
    获取多个图片的ocr结果
    :param image_path_list: 图片地址列表
    :param num_threshold: 相邻数字字符之间的分割阈值
    :param is_gpu: 是否使用gpu
    :param json_save_path: json保存地址
    :return: [ocr_result1,ocr_result2,...]
    ocr_result1: [box1,box2,...]
    """
    # 如果是单个图片地址
    if type(image_path_list) == str:
        return [get_ocr(image_path_list,num_threshold)]

    output = []
    for image_path in image_path_list:
        output.append(get_ocr(image_path,num_threshold,is_gpu,json_save_path=json_save_path))
    return output


def get_ocr_boxes(image_path, img_json, json_save_path = upload_dir + "json_data",  is_gpu = True):
    """
    获取ocr标记框
    :param image_path: 图片地址
    :param img_json: 图片json
    :return: 图片标记框
    """
     # 需要标注的方框坐标
    logger.debug(f"img_json:{img_json}")
    boxes = []

    try:
        ocr_result = get_ocr(image_path, is_gpu=is_gpu, json_save_path=json_save_path)
        ocr_result_only_str = [item[1] for item in ocr_result]
        img_json = json.loads(img_json)
        logger.debug(f"ocr_result_only_str:{ocr_result_only_str}")
        
        # 遍历 json
        for key, value in img_json.items():
            if not value:  # 为空直接跳过
                continue

            # 确保统一成 list 处理
            values = value if isinstance(value, list) else [value]

            for v in values:
                if not v:
                    continue
              
                # 创建一个列表来存放所有需要搜索的字符串
                search_targets = []
                
                # 判断 v 的类型
                if isinstance(v, dict):
                    # 如果 v 是一个字典，遍历它的所有值
                    for item_value in v.values():
                        if item_value:  # 确保字典中的值不为空
                            search_targets.append(str(item_value))
                else:
                    # 如果 v 不是字典（字符串、数字等），直接添加
                    search_targets.append(str(v))

                # 遍历提取出的所有目标字符串进行搜索
                for target_str in search_targets:
                    if not target_str or target_str == "":
                        continue

                    search_result = re_search(target_str, ocr_result_only_str)
                    if search_result:
                        box = []
                        for item in search_result:
                            txt_index, start_index, end_index = item
                            # 提取左上角和右下角坐标
                            t_box = ocr_result[txt_index][0]
                            x_min = int(min(point[0] for point in t_box))
                            y_min = int(min(point[1] for point in t_box))
                            x_max = int(max(point[0] for point in t_box))
                            y_max = int(max(point[1] for point in t_box))
                            box.append([x_min, y_min, x_max, y_max])
                        boxes.append(box)

        logger.debug(f"boxes:{len(boxes)}")
        return boxes
    except Exception as e:
        print(f"获取 ocr 图片失败，err:{e}")
        # 在调试时打印更详细的错误堆栈信息
        import traceback
        traceback.print_exc()
        return []  # 返回空表示发生错误
    


def get_ocr_image(image_path, img_json, json_save_path = upload_dir + "json_data", img_save_path = upload_dir + "ocr_mark_data", is_gpu = True):
    """
    获取ocr标记图片地址
    :param image_path: 图片地址
    :param img_json: 图片json
    :return: 标记图片地址
    """
    
    boxes = get_ocr_boxes(image_path,img_json,json_save_path,is_gpu=is_gpu)
    if boxes is None or len(boxes) == 0:
        return ''
    ocr_results_path = draw_ocr_results(image_path, boxes, save_path=img_save_path)
    return ocr_results_path

# #与上面的区别：可上传到minIO并可下载图片
# def get_ocr_image(image_path, img_json, json_save_path = upload_dir + "json_data", img_save_path = upload_dir + "ocr_mark_data", is_gpu = True):
#     """
#     获取ocr标记图片地址
#     :param image_path: 图片地址
#     :param img_json: 图片json
#     :return: 标记图片地址
#     """
#     boxes = get_ocr_boxes(image_path,img_json,json_save_path,is_gpu=is_gpu)
#     if boxes is None or len(boxes) == 0:
#         return ''
#     ocr_results_path = draw_ocr_results(image_path, boxes, save_path=img_save_path)
   
#     client = Minio("62.234.210.249:9002",
#         access_key="root",
#         secret_key="szaudit1022",
#         secure=False
#     )
#     # The destination bucket and filename on the MinIO server
#     bucket_name = "sz-audit-mark"
#     # Make the bucket if it doesn't exist.
#     found = client.bucket_exists(bucket_name)
#     # if not found:
#     #     client.make_bucket(bucket_name)
#     #     print("Created bucket", bucket_name)
#     # else:
#     #     print("Bucket", bucket_name, "already exists")

#     # Upload the file, renaming it in the process
#     client.fput_object(
#         bucket_name, os.path.basename(ocr_results_path), ocr_results_path,
#     )
#     # print(
#     #     ocr_results_path, "successfully uploaded as object",
#     #     os.path.basename(ocr_results_path), "to bucket", bucket_name,
#     # )
#     url = client.presigned_get_object(
#     bucket_name,
#     os.path.basename(ocr_results_path),
#     response_headers={
#         "response-content-disposition": "attachment"  # 强制浏览器下载
#     })
#     return url
   
# 返回ocr标记图片地址
def get_ocr_image_list(image_path_list, img_json, json_save_path = upload_dir + "json_data", img_save_path = upload_dir + "ocr_mark_data"):
    """
    获取ocr标记图片地址
    :param image_path_list: 图片地址列表
    :param img_json: 图片json
    :param json_save_path: json保存地址
    :param img_save_path: 图片保存地址
    :return: 标记图片地址[ocr_img1,ocr_img2,...]
    """

    # 读取图片ocr结果
    ocr_result_list = []

    for i in range(len(image_path_list)):
        img_json = img_json
        image_path = image_path_list[i]

        ocr_img = get_ocr_image(image_path, img_json, json_save_path, img_save_path)

        ocr_result_list.append(ocr_img)

    return ocr_result_list











if __name__ == '__main__':
    # print(get_ocr("images/1/ivc-591248001.jpg",is_gpu=True))
    # with open("ocr_res.json", "w", encoding="utf-8") as file:
    #     json.dump(get_ocr_list("images/1/ivc-591248001.jpg",is_gpu=True), file, ensure_ascii=False, indent=4)
    # json_text = get_ocr("/data/nfdw/code/LLMServe/data/img/1/ivc-591248001.jpg")
    s = """
        {
            "发票号码": "24372000000241341837",
            "购买方名称": "广东电网有限责任公司信息中心",
            "购买方纳税号": "91440104693553264E",
            "销售方名称": "烟台海颐软件股份有限公司",
            "销售方纳税号": "913706007508888383",
            "项目单价": "1297567.575472",
            "金额": "1297567.58",
            "税额": "77854.05",
            "价税合计小写": "1375421.63",
            "备注": "广东电网有限责任公司2024年数字服务工单项目（营销系统工单）开发实施框架合同之（电能表参数配置等）工单委托函合同编号：0375002024030102YG00046；收款人：王玉群；复核人：鲁昱廷；"
        }
    """

    get_ocr_image(image_path= "data/img/1/ivc-591248001.jpg", img_json =s)



