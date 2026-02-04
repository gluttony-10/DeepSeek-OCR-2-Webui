# -*- coding: utf-8 -*-
import socket
import sys
import os
import io
import re
import tempfile
import contextlib
import psutil
import argparse
import zipfile
from datetime import datetime
import torch
import gradio as gr
from PIL import Image
import numpy as np


@contextlib.contextmanager
def _quiet_stdout():
    """临时屏蔽 stdout，仅保留进度由我们自行 print"""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()

print(" 启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892")
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
if torch.cuda.is_available():
    device = "cuda"
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，尝试FP32\033[0m')
        dtype = torch.float32
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"

# 启用 CUDA 加速优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 矩阵乘法
    torch.backends.cudnn.allow_tf32 = True  # 允许 TF32 加速

# 解决冲突端口（感谢licyk酱提供的代码~）
def find_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            print(f"端口 {port} 已被占用，正在寻找可用端口...")
            return find_port(port=port + 1)
        else:
            return port


from transformers import AutoModel, AutoTokenizer

# 模型路径：本地 models/deepseek-ai/DeepSeek-OCR-2（需事先下载到该目录）
model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "deepseek-ai", "DeepSeek-OCR-2")
tokenizer = None
model = None


def load_model():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    print("正在加载 DeepSeek-OCR-2 模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 指定 dtype 并直接加载到 GPU，满足 Flash Attention 2 要求
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        device_map="cuda" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model = model.eval()
    if device == "cuda":
        model = model.to(dtype)
    print("模型加载完成。")


os.makedirs("outputs", exist_ok=True)


def _file_path(f):
    """从 Gradio 文件对象或路径得到本地路径"""
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("name") or f.get("path")
    return getattr(f, "name", None) or getattr(f, "path", None)


def _fix_md_image_paths(content, prefix):
    """将 md 中的图片相对路径加上前缀，使相对于 outputs/<时间戳>/ 正确"""
    if not content or not prefix:
        return content
    # ![](path) 或 ![alt](path)，path 非 http 且非 / 开头时加前缀
    def repl(m):
        alt, path = m.group(1), m.group(2).strip()
        if path.startswith("http") or path.startswith("/") or path.startswith("file:"):
            return m.group(0)
        return f"![{alt}]({prefix}{path})"
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl, content)


def _zip_folder(folder_path):
    """将文件夹打包成 zip 文件，返回 zip 文件的绝对路径"""
    zip_path = folder_path.rstrip(os.sep) + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zf.write(file_path, arcname)
    return os.path.abspath(zip_path)


def _infer_one_image(model, tokenizer, prompt, image_path, output_path):
    """对单张图片执行 model.infer，返回识别文本（控制台不打印模型内部信息）"""
    with _quiet_stdout():
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=True,
        )
    # 模型默认写出 result.mmd，统一改为 result.md
    mmd_path = os.path.join(output_path, "result.mmd")
    if os.path.exists(mmd_path):
        os.rename(mmd_path, os.path.join(output_path, "result.md"))
    text = (res if isinstance(res, str) else str(res)) if res else ""
    if "<|endoftext|>" in text:
        text = text.replace("<|endoftext|>", "")
    return text


def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """将 PDF 转为 PIL 图像列表（参考 DeepSeek-OCR-2 run_dpsk_ocr2_pdf.py）"""
    if fitz is None:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        images.append(img)
    pdf_document.close()
    return images


MAX_TABS = 20  # 分标签 MD 显示的最大文件数


def run_ocr_unified(files, prompt_choice):
    """统一识别：图片与 PDF 一起上传，按顺序逐文件识别；输出 (md路径, 提示信息, 各标签md内容列表)"""
    _empty_visible = [gr.update(visible=False)] * MAX_TABS
    _empty_content = [""] * MAX_TABS
    if not files:
        yield [], "请先上传至少一个文件（图片或 PDF）。", *_empty_visible, *_empty_content
        return
    if not isinstance(files, list):
        files = [files]
    if fitz is None and any(_file_path(f) and (_file_path(f) or "").lower().endswith(".pdf") for f in files):
        yield [], "未安装 PyMuPDF，无法识别 PDF，请执行: pip install pymupdf", *_empty_visible, *_empty_content
        return
    progress_lines = []
    page_marker = "\n<--- Page Split --->\n"
    file_contents = []  # 每个文件对应合并后的 md 内容，用于分标签显示
    file_output_paths = []  # 每个文件对应合并 md 的路径，用于文件输出（多文件列表）

    def _msg(s):
        print(s)
        progress_lines.append(s)
        return "\n".join(progress_lines)

    def _tabs_content():
        return list(file_contents) + [""] * (MAX_TABS - len(file_contents))

    def _tabs_visible():
        n = len(file_contents)
        return [gr.update(visible=(i < n)) for i in range(MAX_TABS)]

    def _file_output_list():
        return list(file_output_paths)  # 多文件列表供 gr.File 显示

    yield [], _msg("正在加载模型..."), *_tabs_visible(), *_tabs_content()
    with _quiet_stdout():
        load_model()
    yield [], _msg("模型已加载，开始识别。"), *_tabs_visible(), *_tabs_content()
    prompts = {
        "文档转 Markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
        "自由 OCR": "<image>\nFree OCR. ",
    }
    prompt = prompts.get(prompt_choice, prompts["文档转 Markdown"])
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    base_out = os.path.join("outputs", time_str)
    os.makedirs(base_out, exist_ok=True)
    total_files = len(files)
    for idx, item in enumerate(files):
        path = _file_path(item)
        if not path or not os.path.isfile(path):
            file_contents.append(f"[第 {idx+1} 个] 无效文件，已跳过。")
            yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] 无效文件，已跳过。"), *_tabs_visible(), *_tabs_content()
            continue
        basename = os.path.splitext(os.path.basename(path))[0]
        ext = (os.path.splitext(path)[1] or "").lower()
        if ext == ".pdf":
            if fitz is None:
                file_contents.append(f"[第 {idx+1} 个] {basename}.pdf 需要 PyMuPDF，已跳过。")
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] {basename}.pdf 需要 PyMuPDF，已跳过。"), *_tabs_visible(), *_tabs_content()
                continue
            out_dir = os.path.join(base_out, f"file_{idx+1}_{basename}")
            os.makedirs(out_dir, exist_ok=True)
            try:
                images = pdf_to_images_high_quality(path)
            except Exception as e:
                file_contents.append(f"[第 {idx+1} 个] {basename}.pdf 转图片失败：{e}")
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] {basename}.pdf 转图片失败。"), *_tabs_visible(), *_tabs_content()
                continue
            if not images:
                file_contents.append(f"[第 {idx+1} 个] {basename}.pdf 无有效页面。")
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] {basename}.pdf 无有效页面。"), *_tabs_visible(), *_tabs_content()
                continue
            temp_paths = []
            rel_dir = f"file_{idx+1}_{basename}"
            total_pages = len(images)
            for i, pil_image in enumerate(images):
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] {basename}.pdf 第 {i+1}/{total_pages} 页识别中..."), *_tabs_visible(), *_tabs_content()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    pil_image.save(f.name)
                    image_path = f.name
                temp_paths.append(image_path)
                output_path = os.path.join(out_dir, f"page_{i+1}")
                try:
                    _infer_one_image(model, tokenizer, prompt, image_path, output_path)
                except Exception as e:
                    for p in temp_paths:
                        try:
                            os.unlink(p)
                        except Exception:
                            pass
                    file_contents.append(f"[第 {idx+1} 个] {basename}.pdf 第 {i+1} 页识别出错：{e}")
                    yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] 第 {i+1} 页出错：{e}"), *_tabs_visible(), *_tabs_content()
                    break
            else:
                for p in temp_paths:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
                # 合并每一页的 result.md，并调整图片链接（相对合并 md 所在文件夹）
                raw_pages = []
                for j in range(total_pages):
                    p = os.path.join(out_dir, f"page_{j+1}", "result.md")
                    if os.path.exists(p):
                        with open(p, encoding="utf-8") as f:
                            raw_pages.append(f.read())
                    else:
                        raw_pages.append("")
                # 合并 md 放在对应文件夹：图片路径加 page_i/ 前缀
                merged_folder = page_marker.join(
                    _fix_md_image_paths(raw_pages[j], f"page_{j+1}/") for j in range(total_pages)
                )
                single_md_path = os.path.join(out_dir, f"{basename}.md")
                with open(single_md_path, "w", encoding="utf-8") as f:
                    f.write(f"## {basename}.pdf ({len(images)} 页)\n\n{merged_folder}")
                merged_display = f"## {basename}.pdf ({len(images)} 页)\n\n{merged_folder}"
                file_contents.append(merged_display)
                # 将该文件的输出目录打包成 zip
                zip_path = _zip_folder(out_dir)
                file_output_paths.append(zip_path)
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] {basename}.pdf 完成，共 {total_pages} 页。"), *_tabs_visible(), *_tabs_content()
        else:
            yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] {os.path.basename(path)} 识别中..."), *_tabs_visible(), *_tabs_content()
            try:
                pil_image = Image.open(path).convert("RGB")
            except Exception as e:
                file_contents.append(f"[第 {idx+1} 个] 无法打开图片：{e}")
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] 无法打开图片。"), *_tabs_visible(), *_tabs_content()
                continue
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                pil_image.save(f.name)
                image_path = f.name
            rel_dir = f"file_{idx+1}_{basename}"
            output_path = os.path.join(base_out, rel_dir)
            try:
                _infer_one_image(model, tokenizer, prompt, image_path, output_path)
                # 单图：合并即该页 result.md，放在对应文件夹；图片与 md 同目录，无需改链接
                result_md = os.path.join(output_path, "result.md")
                if os.path.exists(result_md):
                    with open(result_md, encoding="utf-8") as f:
                        content = f.read()
                else:
                    content = ""
                single_md_path = os.path.join(output_path, f"{basename}.md")
                single_display = f"## {os.path.basename(path)}\n\n{content}"
                with open(single_md_path, "w", encoding="utf-8") as f:
                    f.write(single_display)
                file_contents.append(single_display)
                # 将该文件的输出目录打包成 zip
                zip_path = _zip_folder(output_path)
                file_output_paths.append(zip_path)
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] {os.path.basename(path)} 完成。"), *_tabs_visible(), *_tabs_content()
            except Exception as e:
                try:
                    os.unlink(image_path)
                except Exception:
                    pass
                file_contents.append(f"[第 {idx+1} 个] {os.path.basename(path)} 识别出错：{e}")
                yield _file_output_list(), _msg(f"  [{idx+1}/{total_files}] 识别出错：{e}"), *_tabs_visible(), *_tabs_content()
            else:
                try:
                    os.unlink(image_path)
                except Exception:
                    pass
    done_msg = f"识别完成，共 {len(files)} 个文件。"
    yield _file_output_list(), _msg(done_msg), *_tabs_visible(), *_tabs_content()


with gr.Blocks() as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">DeepSeek-OCR-2-Webui</h2>
            </div>
            <div style="text-align: center;">
                十字鱼
                <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                |DeepSeek-OCR-2-Webui
                <a href="https://github.com/gluttony-10/DeepSeek-OCR-2-Webui">🌐github</a> 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            """)
    prompt_dropdown = gr.Dropdown(
        label="识别模式",
        choices=["文档转 Markdown", "自由 OCR"],
        value="文档转 Markdown",
    )
    with gr.Row():
        with gr.Column():
            files_in = gr.File(
                label="批量上传（图片 + PDF 可混合多选）",
                file_count="multiple",
                file_types=["image", ".pdf"],
            )
            run_btn = gr.Button("🖼️ 开始识别", variant="primary")
            file_out = gr.File(label="文件输出", file_count="multiple", interactive=False)
        with gr.Column():
            info_out = gr.Textbox(
                label="提示信息",
                lines=8,
                max_lines=12,
                interactive=False,
                placeholder="识别过程中将显示进度…",
            )
            tab_accs = []
            tab_mds = []
            for i in range(MAX_TABS):
                acc = gr.Accordion(f"文件{i + 1}", open=(i == 0), visible=False)
                tab_accs.append(acc)
                with acc:
                    tab_mds.append(gr.Markdown())
    run_btn.click(
        fn=run_ocr_unified,
        inputs=[files_in, prompt_dropdown],
        outputs=[file_out, info_out] + tab_accs + tab_mds,
    )


if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name,
        server_port=find_port(args.server_port),
        share=args.share,
        mcp_server=args.mcp_server,
        inbrowser=True,
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans")]),
    )
