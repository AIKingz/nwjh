import json
import time

from rapidocr_onnxruntime import RapidOCR
from Models.paddleocr.paddleocr import PaddleOCR
# from rapidocr_paddle import RapidOCR as RapidOCR_paddle
from ocr_utils import split_ocr_results, save_ocr_result, is_ocr_result_exist, load_ocr_result, re_search, \
    draw_ocr_results, remove_ocr_result, paddle2rappid
# from wired_table_rec import WiredTableRecognition
# from wired_table_rec.table_line_rec import TableLineRecognition
# from wired_table_rec.utils import LoadImage
import os

current_directory = os.getcwd()
upload_dir = os.path.join(current_directory, "data/")


ocr_engine_cpu = RapidOCR(det_model_path='Models/ch_PP-OCRv4_det_infer.onnx',
                          rec_model_path='Models/ch_PP-OCRv4_rec_infer.onnx',
                          box_thresh=0.3, return_word_box=True)

ocr_engine_gpu = PaddleOCR(use_mindir=True, lang='ch',
                          det_model_path='Models/paddleocr/inference/det/m_model.mindir',
                          rec_model_path='Models/paddleocr/inference/rec/m_model.mindir',
                          use_angle_cls=False, return_word_box=True)

# ocr_engine_cpu = ocr_engine_gpu

# ocr_engine_gpu = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
# ocr_engine_gpu = ocr_engine_cpu

def get_ocr(image_path, num_threshold = 15, is_gpu=True, json_save_path = upload_dir + "json_data"):

    if is_gpu:
        # ocr_engine = RapidOCR_paddle()
        ocr_engine = ocr_engine_gpu
        ocr_result = ocr_engine.ocr(image_path)[0]
        ocr_result = paddle2rappid(ocr_result)
    else:
        ocr_engine = ocr_engine_cpu
        ocr_result, _ = ocr_engine(image_path, box_thresh=0.3, return_word_box=True)
        for res in ocr_result:
            # 删除最后一个元素
            res.pop()

    if ocr_result == None:
        ocr_result = []

    # 将ocr识别结果保存为json
    # save_ocr_result(ocr_result, image_path, save_path=json_save_path)

    return ocr_result

    # # 文字识别结果切分
    # ocr_result = split_ocr_results(ocr_result, num_threshold)
    #
    # # 识别表格
    # wired_engine = WiredTableRecognition()
    #
    # html, elasp, table_result, logic_points, ocr_res, cell_box_det_map, not_match_orc_boxes = wired_engine(image_path, ocr_result=ocr_result)
    #
    # if cell_box_det_map == None:
    #     return [[item[1]]  for item in ocr_result]
    #
    # fin_list = list(cell_box_det_map.values()) + list(not_match_orc_boxes.values())
    # # 对每个子列表，获取第二个元素
    # output = [item[1] for item in fin_list]
    # print("ocr result:", output)
    # return output

    # print('output:', output)

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


def get_ocr_image(image_path, img_json, json_save_path = upload_dir + "json_data", img_save_path = upload_dir + "ocr_mark_data", is_gpu = True):
    """
    获取ocr标记图片地址
    :param image_path: 图片地址
    :param img_json: 图片json
    :return: 标记图片地址
    """

    # 需要标注的方框坐标

    boxes = []

    try:
        # 读取图片ocr结果
        # if is_ocr_result_exist(image_path, save_path=json_save_path):
        #     ocr_result = load_ocr_result(image_path, save_path=json_save_path)
        # else:
        #     ocr_result = get_ocr(image_path, is_gpu=is_gpu, json_save_path=json_save_path)
        #     ocr_result = load_ocr_result(image_path, save_path=json_save_path)

        ocr_result = get_ocr(image_path, is_gpu=is_gpu, json_save_path=json_save_path)
        ocr_result_only_str = [item[1] for item in ocr_result]
        img_json = json.loads(img_json)

        # 遍历json
        for key, value in img_json.items():
            if value == "":
                continue

            search_result = re_search(value, ocr_result_only_str)

            if search_result:
                box = []
                for item in search_result:
                    txt_index, start_index, end_index = item
                    box.append(ocr_result[txt_index][3][start_index][0] + ocr_result[txt_index][3][end_index-1][2])
                boxes.append(box)

        # 绘制保存ocr结果
        ocr_results_path = draw_ocr_results(image_path, boxes, save_path=img_save_path)

        return ocr_results_path

    except Exception as e:

        return ''  # 返回空字符串表示发生错误

        # 在这里可以根据需要记录错误日志等操作
        # raise
    # finally:
    #     # 删掉ocr的json文件
    #     remove_ocr_result(image_path, save_path=json_save_path)

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

    # print(get_ocr("/data/nfdw/code/LLMServe/data/img/1/ivc-591248001.jpg"))

    # out = get_ocr("images/1/ivc-591248004.jpg")
    #
    # # 保存到文件
    # with open("ocr_res.json", "w", encoding="utf-8") as file:
    #     json.dump(out, file, ensure_ascii=False, indent=4)

