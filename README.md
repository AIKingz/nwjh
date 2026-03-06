# 智能稽核系统

基于 AI 的财务与合同类文档智能处理系统，支持文档分类、关键信息抽取、签章检测及 OCR 标注，面向商旅通/工程物资等业务场景。

---

## 功能特性

- **文档分类**：按业务类型（工程物资 / 非工程物资 / 商旅通报销等）对发票、订单合同、转账凭证、质保金、银付凭证、商城到货单、支付申请单、火车票订单、酒店订单、出差申请单、机票订单等文档进行自动分类。
- **信息抽取**：结合视觉大模型（Qwen / InternVL 等）与定制提示词，从图片/PDF 中抽取结构化字段（合同编号、金额、买卖方、账户信息等）。
- **签章检测**：基于 YOLO 的签名、盖章检测（当前代码中签章检测为注释状态，可按需启用）。
- **OCR 与标注**：支持 PaddleOCR（含 MindSpore 推理）与 RapidOCR（ONNX），可对抽取结果在图像上绘制标注框并保存。
- **多格式支持**：支持单图、多图及 PDF（按页转图后逐页处理）。

---

## 技术栈

| 模块         | 技术 |
|--------------|------|
| Web 服务     | FastAPI、Uvicorn |
| 文档分类     | PyTorch、ResNet18、BERT、PaddleOCR |
| 信息抽取     | 视觉大模型（Qwen / InternVL3-8B 等）、自定义 Prompt |
| 签章检测     | Ultralytics YOLO（可选） |
| OCR          | PaddleOCR、RapidOCR（ONNX）、MindSpore 推理 |
| 文档处理     | PyMuPDF (fitz)、Pillow |

---

## 项目目录结构（LLMServe）

```
LLMServe/
├── ai_detector.py          # 分类检测接口
├── classify.py             # 分类调用
├── confidence.json         # 输出配置文件
├── config.py               # 提示词配置文件
├── config.yaml             # 配置文件
├── data/                   # 存放数据
├── file_utils.py           # 从 minio 地址中上传下载文件
├── Models                  # 存放预训练模型参数相关文件
├── ocr.py                  # ocr 主要的调用方法
├── ocr_utils.py            # ocr 工具
├── prompt.py               # 存放提示词
├── sign_seal_detect.py     # 签章检测
├── server.py               # 服务启动
├── test_data/              # 测试数据
├── requirements.txt/       # 依赖清单文件
├── tmp/                    # 临时文件
└── predicted_results/      # yolo 识别结果可视化，需传入 debug=True
```

模型与权重需放置于 `Models/` 下（如 `Models/bert/`、`Models/resnet/`、`Models/paddleocr/`、`Models/yolo/` 等），本目录中仅包含代码与配置。

---

## 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速 OCR 与分类）
- 可访问的视觉大模型服务（如 Qwen2.5-VL、InternVL3-8B）用于信息抽取

---

## 安装与运行

### 1. 依赖

本项目提供 `requirements.txt`。

部分模型需从 PaddleOCR、HuggingFace 等获取并放入 `Models/` 对应目录。

### 2. 配置

- **config.py**：已内置各文档类型的抽取提示词，新增场景时需在此增加对应提示词及在 `server.py` 的 `type_mapping` / `type_to_function` 中登记。
- **config.yaml**：可配置 `file_upload_url`、大模型 API（如 `OpenAI.base_url`、`OpenAI.model`，当前示例为 InternVL3-8B）。
- **server.py** 内大模型请求地址、app_key、签名等需按实际部署修改（如 `run_conv` 中的 URL、鉴权参数）。


## API 说明

### 信息抽取

- **POST** `/nfdw/api/v1/model/extract`
- **Body**：`{ "classify": "发票"|"订单合同"|"质保金"|"银付凭证"|"转账凭证"|"商城到货单"|"支付申请单或XX证明"|"火车票订单"|"酒店订单"|"出差申请单"|"机票订单", "fileUrl": "图片/PDF 地址", "base64": "可选" }`
- **说明**：按 `classify` 选择文档类型进行关键信息抽取；订单合同走合同专用逻辑；转账凭证等有后处理（如不含税/税额）。
- **返回**：`{ "code": 200, "message": "Success", "data": [ { "page", "pageUrl", "result" }, ... ] }`

### 签章检测

- **POST** `/nfdw/api/v1/model/detect`
- **Body**：`{ "fileUrl": "图片地址", "base64": "可选" }`
- **返回**：`{ "code": 200, "data": [ { "page", "pageUrl", "sign", "seal" } ] }`

### 文档分类

- **POST** `/nfdw/model/classify`
- **Body**：`{ "list": [ { "fileUrl", "fileId", "fileName" }, ... ], "businessType": "工程物资"|"非工程物资"|"工程报销支付申请_电网管理平台(深圳局)" 等 }`
- **返回**：`{ "code": 200, "data": [ { "classify", "sorce", "fileUrl", "fileId", "fileName" }, ... ] }`


---

## 支持的文档类型与分类映射

| 前端展示类型       | 内部类型                       | 处理方式说明                     |
|--------------------|--------------------------------|----------------------------------|
| 发票               | invoice                       | 单页/两页 PDF 合并抽取           |
| 订单合同           | contract_order                | 合同字段 + 签章（可选）          |
| 商城到货单         | store_order                   | 通用抽取                         |
| 质保金             | deposit                       | 通用抽取                         |
| 银付凭证           | certificate                   | 抽取 + 后处理（不含税/税额等）   |
| 转账凭证           | transfer_voucher              | 抽取 + 合并与后处理              |
| 支付申请单或XX证明 | payment_receipt_o_XXcertificate | 通用抽取                         |
| 火车票订单         | train_ticket_order            | 通用抽取          |
| 酒店订单           | hotel_order                   | 通用抽取          |
| 出差申请单         | business_travel_application  | 通用抽取          |
| 机票订单           | flight_ticket_order           | 通用抽取          |

工程物资 / 非工程物资等在分类接口中对应不同 `businessType` 与类别集合。

---


## 注意事项

1. **模型与权重**：需自行准备并放置到 `Models/` 下相应路径（如 BERT、PaddleOCR、YOLO、分类 checkpoint），本目录可能仅包含配置或占位路径。
2. **大模型服务**：信息抽取依赖外部视觉大模型接口，请确保 `config.yaml` 及 `server.py` 中 URL、鉴权（app_key、签名等）配置正确。
3. **数据目录**：运行时使用并创建 `data/img/`、`data/content/` 等，请保证写权限。
4. **敏感信息**：上传 URL、API Key、签名密钥等请勿提交到公开仓库，建议使用环境变量或独立配置。

---

## License

请按项目实际许可协议使用；若为公司内部项目，请遵循公司规范。
