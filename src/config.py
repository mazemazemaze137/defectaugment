# src/config.py
import yaml
import os

def load_config(config_path="config.yaml"):
    # 获取 config.yaml 的绝对路径（相对于当前文件）
    config_abs_path = os.path.join(os.path.dirname(__file__), "..", config_path)
    with open(config_abs_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)