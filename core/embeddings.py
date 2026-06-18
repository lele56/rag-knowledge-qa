try:
    # LangChain >= 0.2 推荐用 langchain_huggingface（避免弃用警告）
    from langchain_huggingface import HuggingFaceEmbeddings
    _USING_NEW_PACKAGE = True
except ImportError:
    # 兼容旧版本 langchain_community
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _USING_NEW_PACKAGE = False
from langchain_core.embeddings import Embeddings
from config.settings import settings
from utils.logger import logger
from utils.device import get_device

_embeddings = None


def get_embeddings() -> Embeddings:
    """获取 embedding 模型（自动选择 langchain_huggingface 或 langchain_community 的实现）。"""
    global _embeddings
    if _embeddings is None:
        device = get_device()
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 64,
            },
        )
        source = "langchain_huggingface" if _USING_NEW_PACKAGE else "langchain_community (legacy)"
        logger.info(f"embeddings model 加载完成 [{source}] ({settings.EMBEDDING_MODEL_PATH}, device={device})")
    return _embeddings