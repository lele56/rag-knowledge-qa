# -*- coding: utf-8 -*-
# web/gradio_app.py
"""知识库问答助手 — Gradio Web 界面入口

职责:
  - 构建 UI 布局（上传文档、智能问答、文档管理、调试）
  - 绑定事件处理函数（委托给 ui_handlers.py）
  - 启动 Gradio 服务（端口/主题/CSS 配置）

架构:
  UI 布局 → 事件绑定 → ui_handlers → qa_service → rag_agent
"""

import gradio as gr
from config.settings import settings
from utils.logger import logger
from web.ui_handlers import (
    CUSTOM_CSS,
    _render_uploaded_list,
    chat_respond,
    chat_retry,
    chat_undo,
    upload_files,
    refresh_upload_status,
    get_doc_stats,
    list_all_docs,
    clear_memory_action,
    refresh_status,
    debug_retrieve,
    refresh_history,
)


# ================================================================
# UI 常量
# ================================================================

APP_TITLE = """
# 📚 知识库问答助手
LangChain + Qdrant + Neo4j  |  混合检索  |  HyDE/多查询  |  图增强  |  智能分块
"""

CHAT_EXAMPLES = [
    "这些文档中提到了哪些核心方法？",
    "请对比一下文档A和文档B的异同",
    "总结一下文档的主要内容",
]

CHAT_TIPS = """
- 上传文档后即可开始提问
- 支持跨文档对比问答
- 回答会标注信息来源
"""


def _build_upload_tab():
    """构建上传文档 Tab。
    
    Returns:
        tuple: (uploaded_state, upload_status, uploaded_list, file_input)
    """
    uploaded_state = gr.State(value=[])
    file_input = gr.File(
        file_count="multiple",
        file_types=[".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"],
        label="选择文件（可多选）",
        type="filepath",
    )
    with gr.Row():
        upload_btn = gr.Button("🚀 添加到知识库", variant="primary", size="lg")
        clear_upload_btn = gr.Button("🗑 清空选择", variant="secondary", size="lg")
    with gr.Row():
        upload_status = gr.Markdown(value="_等待上传..._")
        refresh_upload_btn = gr.Button("🔄 刷新写入状态", variant="secondary", size="sm")
    uploaded_list = gr.Markdown(value="*暂无上传记录*")

    upload_btn.click(
        fn=upload_files,
        inputs=[file_input, uploaded_state],
        outputs=[upload_status, uploaded_list, file_input, uploaded_state],
    )
    clear_upload_btn.click(
        fn=lambda: (None, "_等待上传..._", _render_uploaded_list([]), []),
        outputs=[file_input, upload_status, uploaded_list, uploaded_state],
    )
    refresh_upload_btn.click(
        fn=refresh_upload_status,
        inputs=[uploaded_state],
        outputs=[uploaded_list, uploaded_state],
    )
    
    return uploaded_state, upload_status, uploaded_list, file_input


def _build_chat_tab():
    """构建智能问答 Tab。
    
    Returns:
        tuple: (chatbot, msg, clear_mem_msg)
    """
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="输入您的问题...",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("🗑 清空对话", variant="secondary", size="sm")
                retry_btn = gr.Button("🔄 重新生成", variant="secondary", size="sm")
                undo_btn = gr.Button("↩️ 撤销", variant="secondary", size="sm")
            gr.Examples(
                examples=CHAT_EXAMPLES,
                inputs=msg,
            )
        with gr.Column(scale=1):
            gr.Markdown("### 💡 使用提示")
            gr.Markdown(CHAT_TIPS)
            clear_mem_btn = gr.Button("🧹 清空对话记忆", variant="secondary", size="sm")
            clear_mem_msg = gr.Markdown("")
            clear_mem_btn.click(fn=clear_memory_action, outputs=clear_mem_msg)

    submit_btn.click(fn=chat_respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(fn=chat_respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    retry_btn.click(fn=chat_retry, inputs=[msg, chatbot], outputs=[msg, chatbot])
    undo_btn.click(fn=chat_undo, inputs=[chatbot], outputs=[msg, chatbot])
    clear_btn.click(fn=lambda: ([], []), outputs=[msg, chatbot])
    
    return chatbot, msg, clear_mem_msg


def _build_docs_tab():
    """构建文档管理 Tab。
    
    Returns:
        tuple: (doc_stats, all_docs_list)
    """
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📊 知识库概览")
            doc_stats = gr.Markdown(value=get_doc_stats())
            gr.Button("🔄 刷新统计", variant="secondary", size="sm").click(
                fn=get_doc_stats, outputs=doc_stats
            )
        with gr.Column(scale=2):
            gr.Markdown("### 📋 所有文档")
            all_docs_btn = gr.Button("🔄 刷新文档列表", variant="secondary", size="sm")
            all_docs_list = gr.Markdown(value=list_all_docs())
            all_docs_btn.click(fn=list_all_docs, outputs=all_docs_list)
    
    return doc_stats, all_docs_list


def _build_debug_tab():
    """构建状态 & 调试 Tab。
    
    Returns:
        tuple: (debug_output, chat_history)
    """
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 检索调试")
            debug_input = gr.Textbox(label="关键词/问题", placeholder="输入关键词测试检索...")
            debug_btn = gr.Button("🔍 测试检索", variant="primary")
            debug_output = gr.Markdown(value="_等待测试..._")
            debug_btn.click(fn=debug_retrieve, inputs=debug_input, outputs=debug_output)
        with gr.Column(scale=1):
            gr.Markdown("### 📜 对话历史")
            refresh_hist_btn = gr.Button("🔄 刷新", variant="secondary", size="sm")
            chat_history = gr.Markdown(value="_点击刷新查看对话记录_")
            refresh_hist_btn.click(fn=refresh_history, outputs=chat_history)
    
    return debug_output, chat_history


def run():
    """构建并启动 Gradio 应用。"""
    with gr.Blocks(title="知识库问答助手") as demo:
        # ===== 顶部栏 =====
        gr.Markdown(APP_TITLE)

        # ===== 状态栏 =====
        with gr.Row():
            system_status = gr.Markdown(value="🔄 正在连接...")
            gr.Button("🔄 刷新状态", variant="secondary", size="sm").click(
                fn=refresh_status, outputs=system_status
            )

        # ===== Tab 构建 =====
        uploaded_state, upload_status, uploaded_list, file_input = _build_upload_tab()
        chatbot, msg, clear_mem_msg = _build_chat_tab()
        doc_stats, all_docs_list = _build_docs_tab()
        debug_output, chat_history = _build_debug_tab()

        # ===== 底部 =====
        gr.HTML('<div class="footer">知识库问答助手 v2.0  |  Powered by LangChain + Qdrant + Neo4j</div>')

        # ===== 页面加载事件 =====
        demo.load(fn=get_doc_stats, outputs=doc_stats)
        demo.load(fn=list_all_docs, outputs=all_docs_list)

    # ===== 启动服务 =====
    try:
        demo.launch(
            server_name=settings.server.host,
            server_port=settings.server.port,
            share=False,
            theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate"),
            css=CUSTOM_CSS,
        )
    except OSError as e:
        if "Address already in use" in str(e) or "地址已在使用" in str(e):
            logger.error(f"端口 {settings.server.port} 已被占用，请关闭其他实例或修改 SERVER_PORT")
        else:
            logger.error(f"Gradio 启动失败：{e}")
        raise


if __name__ == "__main__":
    run()