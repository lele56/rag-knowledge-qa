"""统一 token 计数工具，各模块共享。

tiktoken 精确计数优先，回退到中英文启发式估算。
"""
import re
from utils.logger import logger


def count_tokens(text: str) -> int:
    """使用 tiktoken 精确计数，回退到启发式估算。

    中文 token 约 1.5~2 字符/token，英文约 4 字符/token。
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))
    except Exception as e:
        logger.debug(f"tiktoken 不可用: {e}，使用启发式估算")
        chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
        other = len(text) - chinese
        return int(chinese / 1.5 + other / 4)