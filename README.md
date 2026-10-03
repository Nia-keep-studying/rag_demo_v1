# RAG Demo V1

一个用于学习的 Python RAG 项目：读取 Markdown 示例制度，按章节切片，使用本地 Embedding 模型检索，再调用 DeepSeek 根据资料回答并标注来源。

当前包含 3 份示例制度、12 个片段。这些文档用于练习，不代表真实企业政策。

## 当前流程

```text
Markdown 文档 → 按二级标题切片 → 章节与正文向量化
用户问题 → 问题向量 → 相似度排序 → Top 3 片段
Top 3 片段 → 带来源的参考资料 → DeepSeek 回答
```

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `rag_loader.py` | 读取文档并保留来源、章节、片段编号 |
| `rag_search.py` | 关键词检索练习与上下文 JSON 构造 |
| `vector_search_demo.py` | 人工二维向量的余弦相似度、Top K 和阈值练习 |
| `embedding_demo.py` | 加载真实模型、生成片段向量、按相似度检索 |
| `rag_answer.py` | 向量检索与 DeepSeek 回答的完整流程；保留旧关键词函数作学习对照 |
| `documents/` | 三份示例制度文档 |

## 环境与安装

已在 Windows、Python 3.12 环境中运行。代码使用 Python 3.12 的 f-string 语法，请使用 Python 3.12 或更高版本；其他版本与平台尚未完整验证。依赖版本来自当前可运行环境，尚未进行全新环境安装验收。

在 PowerShell 中执行：

```powershell
git clone https://github.com/Nia-keep-studying/rag_demo_v1.git
cd rag_demo_v1
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

如果没有 `py` 启动器，用本机 Python 3.12 的可执行文件路径替换 `py -3.12`。命令直接指定虚拟环境中的 Python，因此无需先执行激活脚本。

## 配置与运行

完整问答需要 DeepSeek API Key。首次配置时复制模板，并在本机编辑 `.env`：

```powershell
Copy-Item .env.example .env
```

将 `.env` 中的占位符替换为自己的 Key。已有 `.env` 时不要覆盖。真实 Key 不要写进源码或提交到仓库。

分别运行：

```powershell
# 只检查文档加载和切片，不调用外部模型 API
.\.venv312\Scripts\python.exe rag_loader.py

# 本地向量检索，输出候选片段与分数
.\.venv312\Scripts\python.exe embedding_demo.py

# 完整 RAG 问答，会调用 DeepSeek API
.\.venv312\Scripts\python.exe rag_answer.py
```

本地 Embedding 使用 `cnmoro/multilingual-e5-small-distilled-16m`，首次运行需要从 Hugging Face 下载模型，缓存在项目 `.model_cache/` 下。模型以归一化方式生成 64 维向量，使用点积计算相似度。文档与问题由本地模型编码；完整问答会将问题和召回的资料发送给 DeepSeek，并产生 API 费用。

这个环境曾遇到 FastEmbed / ONNX Runtime 原生 DLL 加载失败，目前使用 Model2Vec，不需要安装 FastEmbed。

## 已观察到的结果

- 文档加载得到 3 份文档、12 个片段。
- “发货后还能修改收货地址吗？”：地址修改片段进入前三名，回答引用对应制度。
- “可以开电子发票吗？”：虽然检索返回候选，模型在本次人工验收中回答资料不足、无法确认。
- “货物寄出去以后还能调整收件地点吗？”：曾观察到地址修改排第 7，Top 3 漏召回。这是保留的失败案例。

以上为当前小样例的观察，不代表对所有提问都能稳定回答或拒答。相似度分数不是正确率，具体排名也不是业务事实。

## 当前边界与下一步

- `embedding_demo.py` 的真实检索目前只有 Top K，还没有 `min_score`；人工向量练习中的阈值不会自动作用于真实检索。
- 无依据拒答目前依赖提示词和生成模型，不能只凭一次成功验收保证稳定性。
- 每次运行会重新生成文档向量；尚未使用持久化向量数据库。
- Markdown 切片只适配当前二级标题结构；尚未处理复杂表格、PDF、跨章节指代和版本冲突。
- `rag_answer.py` 当前是交互式入口脚本，导入时也会执行输入及运行流程；请直接运行该脚本。
- 尚未加入生产环境的权限控制、重排序、自动评估和完整异常处理。

下一步添加可配置的最低相似度过滤，比较正确召回、错误候选和无依据问题，再用固定测试问题评估改进。

`.env`、虚拟环境、模型缓存、数据库和 Python 缓存已通过 `.gitignore` 排除。
