# RAG Demo V1

一个 Python RAG 学习项目：读取 Markdown 制度、按标题切片，通过关键词与本地向量两路检索合并候选，按开关执行本地重排序，再调用 DeepSeek 根据资料回答并标注来源。

当前是命令行演示，包含 3 份示例制度、12 个片段；样例不代表真实企业政策。本项目尚未做成服务，也没有接入订单 Agent。

## 当前数据流

```text
Markdown → 按二级标题切片 → 保留来源与章节 → 片段向量
用户问题 → 表达映射 → 关键词候选
用户问题 → 问题向量 → 阈值过滤与倒序排序 → 最多 5 个向量候选
两路候选 → 合并并按 (source, chunk_id) 去重
候选为空 → 程序直接返回无法确认，不调用回答 API
候选非空 → 可选重排序 → 最终片段 → messages → DeepSeek 回答
```

当前 `use_rerank=True`，重排序后最多保留 3 个片段；关闭时使用全部合并候选，数量可能超过向量一路的 Top K。

## 环境与安装

已在 Windows、Python 3.12 中运行。源码包含 Python 3.12 的 f-string 语法，请使用 Python 3.12 或以上版本。依赖已固定到学习环境版本；尚未验证全新环境安装、其他 Python 版本或操作系统。

在 PowerShell 执行：

```powershell
git clone https://github.com/Nia-keep-studying/rag_demo_v1.git
cd rag_demo_v1
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip check
```

如果没有 `py` 启动器，将 `py -3.12` 替换为本机 Python 3.12 可执行文件路径。下面的命令直接指定虚拟环境，无需先激活。学习环境使用 CPU 版 PyTorch；需要 CPU 版安装源时，可先运行 `.\.venv312\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`，再安装依赖清单。

## 配置模型与 Key

完整问答需要在项目目录配置 `.env`：

```powershell
# 仅首次配置时复制，已有 .env 不要覆盖
Copy-Item .env.example .env
```

在本机将模板占位符换成自己的 DeepSeek Key。文档加载、检索评估和重排序演示不需要回答 API Key。问答入口在开始时创建客户端，因此即使候选为空，也仍需提供 Key。

| 用途 | 模型 | 执行位置 |
| --- | --- | --- |
| 片段与问题向量 | `cnmoro/multilingual-e5-small-distilled-16m` | 本地，生成归一化 64 维向量 |
| 重排序 | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 本地 CPU，对问题与候选共同评分 |
| 最终回答 | `deepseek-chat` | DeepSeek API |

首次使用本地模型需要下载权重，缓存位于 `.model_cache/`。本地编码和重排序不产生按次回答 API 费用；有候选时，问题和最终资料会发送到 DeepSeek。缓存文件、真实 Key 和虚拟环境不提交。

Windows 曾遇到 PyTorch DLL 初始化失败；学习者更新 Microsoft VC++ 运行库后导入成功，不据此认定所有同类报错的根因相同。见 [Microsoft 官方运行库说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)。曾失败的 FastEmbed / ONNX Runtime 方案已经改为 Model2Vec，当前不需要安装 FastEmbed。

## 运行与最小演示

下面命令均在项目根目录运行：

```powershell
# 文档读取与切片，应得到 3 份文档和 12 个片段
.\.venv312\Scripts\python.exe rag_loader.py

# 查看全部片段的向量排名，需要输入问题
.\.venv312\Scripts\python.exe embedding_demo.py

# 三题混合检索评估，不调用回答 API，覆盖当前结果 JSON
.\.venv312\Scripts\python.exe eval_retrieval.py

# 对上述 JSON 第二题运行重排序演示，不调用回答 API
.\.venv312\Scripts\python.exe rerank_demo.py

# 完整问答，需要输入问题和配置 Key
.\.venv312\Scripts\python.exe rag_answer.py
```

完整问答可依次输入：

1. `订单已经发货，还能修改收货地址吗？`：查看“最终交给模型的片段”，确认有地址修改；回答应区分订单系统修改与联系物流处理。
2. `包裹已经寄出，还能换个地方接受吗？`：保留这个真实使用的表达，查看正确章节是否保留。不要悄悄替换为“接收”后声称原问题已通过。
3. `可以开电子发票吗？`：样例无相关规则，即使有候选，回答也应明确无法确认。

学习者已报告地址问题能依据资料回答、电子发票问题正确拒答；收尾检查只运行本地模型与消息组装，没有再次调用回答 API。这组案例不代表对所有问题都能稳定处理。

## 可调整的参数

问答配置集中在 `rag_answer.py` 顶部，实际调用使用这些变量：

```python
use_rerank = True
vector_top_k = 5
vector_min_score = 0.4
rerank_top_k = 3
```

| 配置 | 向量 Top K | 向量阈值 | 重排序 | 最终候选 |
| --- | --- | --- | --- | --- |
| 10 月 5 日历史方案 | 3 | 0.526 | 关闭 | 全部合并候选 |
| 10 月 6 日当前实验 | 5 | 0.4 | 开启 | 最多 3 个 |
| 当前实验关闭重排序 | 5 | 0.4 | 关闭 | 全部合并候选 |

最后一行与历史方案不同。一次同时修改多个参数后，不能将效果变化单独归因于重排序。当前选择开启是保留学习实验，尚未证明整体优于关闭方案。

各入口的参数相互独立：

- `eval_retrieval.py` 显式用 5 和 0.4，只评价重排序前的合并候选。
- `embedding_demo.py` 的检索函数默认仍是 3 和 0.6；直接运行时则用全部片段、阈值 -1 查看完整排名。
- `rerank_chunks()` 默认保留 2；问答显式传 3，独立重排序演示显式传 5。
- `rerank_demo.py` 的问题来自 JSON 第二题；不会自动接收问答入口的输入。

## 评估结果如何理解

`eval_retrieval.py` 对有依据题检查预期文件与章节是否进入合并候选；对无依据题采用“候选应为空”的严格检索标准。它不运行重排序，也不评价最终答案。`evaluation_results.json` 保存本轮结果，每次运行覆盖。

当前参数下的离线复核为 **2/3**：两道地址题命中；电子发票题返回无关片段，检索检查未通过。学习者实测回答仍正确拒答，两者分别记录。旧配置的 3/3 属于历史结果，当前第二题措辞也改变过，不能直接当作严格的同组参数对照。

详细结果见 [实验记录](docs/EXPERIMENTS.md)、[离线候选与排序快照](docs/experiment_2026-10-06.json) 和 [早期记录](docs/HISTORY.md)。

## 文件与返回结构

| 文件 | 职责 |
| --- | --- |
| `rag_loader.py` | 文档加载与切片，返回带 source、section、chunk_id、content 的片段 |
| `rag_search.py` | 关键词映射、检索、合并去重与上下文 JSON |
| `embedding_demo.py` | 模型加载、片段向量和向量检索，结果为 `{chunk, score}` 列表 |
| `rerank_demo.py` | 模型加载、重排序、空输入处理与独立演示，结果为 `{chunk, score}` 列表 |
| `rag_answer.py` | 交互式问答、配置开关、最终候选打印和消息组装 |
| `eval_retrieval.py` | 三题重排序前的检索检查、汇总与结果保存 |
| `vector_search_demo.py` | 人工二维向量的基础练习 |

关键词结果直接是片段列表；向量与重排序结果外层含片段和分数。`merge_results()` 与重排序接入处分别取出 `chunk`，让 `build_messages()` 接收纯片段列表。分数越高不等于制度事实越可靠；重排序原始分数不能复用向量阈值。

## 边界与后续

- 重排序只能筛选已进入候选的片段，不能修复所有召回失败；当前口语地址问题仍可能被排第三。
- 降低阈值、扩大候选增加了无关资料。拒答目前依赖提示词与回答模型，不能保证所有问题都正确拒答。
- 每次启动重新编码片段；未实现持久化向量库、版本冲突、权限过滤、复杂 PDF 或表格处理。
- `rag_answer.py` 和 `eval_retrieval.py` 仍是执行式入口；不要将它们作为可安全导入的模块。`rerank_demo.py` 导入不读取演示 JSON 或加载权重，但依赖库仍会被导入。
- 三个自动开发用例及少量离线候选检查不代表整体质量；更广评估、干净环境安装和服务部署待补。

基础 RAG 链路及可选重排序接入已完成。下一阶段先回顾订单 Agent，再设计返回资料与来源的 `search_policy(query)` 工具。综合作品按实际能力逐步建设。
