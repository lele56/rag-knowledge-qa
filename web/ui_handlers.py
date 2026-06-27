# web/ui_handlers.py
"""UI 事件处理 — 统一导出。

拆分说明:
    - web.ui_chat   — Tab 1: 智能问答 + CSS + 辅助函数
    - web.ui_upload — Tab 2: 文件上传
    - web.ui_admin  — Tab 3: 文档管理 + Tab 4: 系统状态/调试
"""

from web.ui_chat import (
    CUSTOM_CSS,
    qa,
    doc_svc,
    _file_to_path,
    _render_uploaded_list,
    chat_respond,
    chat_retry,
    chat_undo,
    refresh_history,
)
from web.ui_upload import (
    upload_files,
    refresh_upload_status,
)
from web.ui_admin import (
    get_doc_stats,
    list_all_docs,
    clear_memory_action,
    refresh_status,
    debug_retrieve,
)

__all__ = [
    "CUSTOM_CSS",
    "qa",
    "doc_svc",
    "_file_to_path",
    "_render_uploaded_list",
    "chat_respond",
    "chat_retry",
    "chat_undo",
    "upload_files",
    "refresh_upload_status",
    "get_doc_stats",
    "list_all_docs",
    "clear_memory_action",
    "refresh_status",
    "debug_retrieve",
    "refresh_history",
]