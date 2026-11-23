# Shopkeeper Brain

A RAG assistant for product manuals: parse PDFs, index them in a vector
database, and answer questions with citations and images.

## 功能

- **文档导入**：PDF → MinerU 解析 → 结构化切分 → 商品名识别 → BGE-M3 向量化 → 写入 Milvus
- **智能问答**：商品名对齐 → 混合检索（dense + sparse）+ HyDE + 联网搜索 → RRF 融合 → BGE 重排 → LLM 生成
- **流式输出**：SSE 逐字返回，也支持一次性返回
- **会话与文件**：MongoDB 存会话历史，MinIO 存原始文件

## 技术栈

FastAPI · LangGraph · Milvus · BGE-M3 / bge-reranker-large · MongoDB · MinIO · DashScope (Qwen)

## 目录结构

    knowledge/
    ├── api/          # FastAPI 入口（query_router / import_router）
    ├── core/         # 依赖注入与路径
    ├── processor/    # LangGraph 流程（import_process / query_process）
    ├── services/     # 业务服务
    ├── utils/        # Milvus/Mongo/SSE/模型客户端
    ├── prompts/      # 提示词模板
    ├── schema/       # 请求与响应模型
    └── front/        # 前端页面

## 快速开始

### 环境要求

- Python 3.12
- 可访问的 Milvus、MongoDB、MinIO
- 本地 BGE-M3 与 bge-reranker-large 模型

### 安装

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r knowledge/requirements.txt

### 配置

在 `knowledge/` 下新建 `.env`（参考 `.env.example`）：

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | DashScope API Key |
| `OPENAI_API_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_DEFAULT_MODEL` | 例如 `qwen-flash` |
| `BGE_M3_PATH` / `BGE_RERANKER_LARGE` | 本地模型路径 |
| `MILVUS_URL` | 例如 `http://<host>:19530` |
| `MONGO_URL` | 例如 `mongodb://<host>:27017` |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 对象存储 |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | 图数据库（可选） |

### 启动

在项目根目录执行：

    python -m uvicorn knowledge.api.query_router:app  --port 8001   # 问答服务
    python -m uvicorn knowledge.api.import_router:app --port 8000   # 导入服务

- 问答页面：http://localhost:8001/chat/chat.html
- 上传页面：http://localhost:8000/front/import.html
- 接口文档：http://localhost:8001/docs 、http://localhost:8000/docs

## 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/query` | 提问（`query` / `session_id` / `is_stream`） |
| GET | `/stream/{task_id}` | SSE 流式答案 |
| GET | `/history/{session_id}` | 查询会话历史 |
| DELETE | `/history/{session_id}` | 清空会话历史 |
| POST | `/upload` | 上传 PDF/MD 并触发导入 |
| GET | `/status/{task_id}` | 导入进度 |

## 说明

- `.env` 含密钥，已被 `.gitignore` 排除，请勿提交
- GPU/CPU 由 `BGE_DEVICE`、`BGE_RERANKER_DEVICE` 控制
