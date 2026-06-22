# utils/device.py
from config.settings import settings
from utils.logger import logger
def get_device() -> str:
    """根据配置 + 实际硬件情况，返回应该使用的 device（"cuda" 或 "cpu"）。"""
    device = "cpu"
    try:
        use_cuda_flag = getattr(settings, "USE_CUDA", False)
        if not use_cuda_flag:
            return device
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("device: 使用 GPU (CUDA)")
            else:
                logger.warning("device: settings.USE_CUDA=True 但 CUDA 不可用，回退到 CPU")
        except Exception as e:
            logger.warning(f"device: 检测 CUDA 失败，用 CPU: {e}")
    except Exception:
        pass
    return device