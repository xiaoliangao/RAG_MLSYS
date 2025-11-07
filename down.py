# download_models.py
import os
import sys

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:
    print("错误：未找到 'modelscope' 库。")
    print("请先运行: pip install modelscope")
    sys.exit(1)
except TypeError:
    # 捕获旧版 modelscope 可能的导入问题
    pass

# (兼容旧版 modelscope)
if 'snapshot_download' not in locals():
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("错误：无法导入 'snapshot_download'。")
        print("请尝试更新 modelscope: pip install --upgrade modelscope")
        sys.exit(1)


def download_all_models():
    """
    下载项目所需的所有核心模型
    """
    print("="*50)
    print(" 🚀 开始下载项目所需的核心模型 🚀")
    print("="*50)

    # --- 1. 下载 BGE Embedding Model ---
    bge_model_id = 'BAAI/bge-large-zh-v1.5'
    bge_save_path = './models/bge-large-zh-v1.5'
    
    if not os.path.exists(os.path.join(bge_save_path, "pytorch_model.bin")):
        print(f"\n--- 1. 正在下载 [Embedding模型] ---")
        print(f"    ID: {bge_model_id}")
        print(f"    目标: {bge_save_path}")
        try:
            snapshot_download(
                bge_model_id,
                local_dir=bge_save_path,
                revision='master'
            )
            print(f"    ✓ BGE 模型下载完成！\n")
        except Exception as e:
            print(f"    ✗ BGE 模型下载失败: {e}\n")
    else:
        print(f"\n--- 1. [Embedding模型] 已存在于: {bge_save_path} ---\n")

    # --- 2. 下载 Qwen2.5-7B-Instruct LLM ---
    # (已根据您的最新选择更新)
    llm_model_id = 'qwen/Qwen2.5-7B-Instruct'
    llm_save_path = './models/Qwen2.5-7B-Instruct'
    
    if not os.path.exists(os.path.join(llm_save_path, "config.json")):
        print(f"--- 2. 正在下载 [LLM / 大语言模型] ---")
        print(f"    ID: {llm_model_id}")
        print(f"    目标: {llm_save_path}")
        print("    （模型约15GB，请耐心等待...）")
        try:
            snapshot_download(
                llm_model_id,
                local_dir=llm_save_path,
                revision='master'
            )
            print(f"    ✓ Qwen2.5 LLM 下载完成！\n")
        except Exception as e:
            print(f"    ✗ Qwen2.5 LLM 下载失败: {e}\n")
    else:
        print(f"--- 2. [LLM / 大语言模型] 已存在于: {llm_save_path} ---\n")

    print("="*50)
    print("✅ 所有核心模型均已准备就绪！")
    print("="*50)

if __name__ == "__main__":
    # 确保 'models' 目录存在
    os.makedirs("./models", exist_ok=True)
    download_all_models()