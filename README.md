# DeepSeek-OCR-2 Webui

基于 DeepSeek-OCR-2 的文档识别与转 Markdown 演示。

目前支持的功能有：
- 文档转 Markdown（带版面/ grounding）
- 自由 OCR

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 使用需求

1. 显存建议大于 8G；显卡最好支持 BF16（不支持将使用 FP32，显存占用增大）。
2. 需安装 Flash Attention 2（`_attn_implementation='flash_attention_2'`），否则请自行修改 `glut.py` 中的加载方式。

## 环境与模型

- 项目使用 `.glut` 目录作为 Python 环境（复制自 Tongbi 项目环境）。
- 模型需事先下载到 `models/deepseek-ai/DeepSeek-OCR-2` 目录（可从 [Hugging Face 镜像站 hf-mirror.com](https://hf-mirror.com) 或官方获取）。
- 使用 `01运行程序.bat` 启动时，已设置 `HF_HOME=%CD%\models` 等环境变量。

## 安装依赖（可选，若使用 .glut 可跳过）

```bash
conda create -n DeepSeek-OCR-2 python=3.12
conda activate DeepSeek-OCR-2
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
# 若使用 Flash Attention 2：
# pip install flash-attn --no-build-isolation
```

## 开始运行

使用一键包时，双击 `01运行程序.bat` 即可。

或命令行：

```bash
python glut.py
```

默认会在 `http://127.0.0.1:7891` 启动服务。如需修改 IP 和端口：

```bash
python glut.py --server_name 0.0.0.0 --server_port 7891
```

## 参考项目

- [Tongbi](https://github.com/gluttony-10/Tongbi)
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
