"""通用工具函数"""

import os


def get_env_int(name: str, default: int) -> int:
    """从环境变量读取整数，解析失败时返回默认值并打印警告。"""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: invalid integer for {name}={value!r}, using {default}")
        return default


def get_env_float(name: str, default: float) -> float:
    """从环境变量读取浮点数，解析失败时返回默认值并打印警告。"""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: invalid float for {name}={value!r}, using {default}")
        return default
