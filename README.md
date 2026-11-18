# Reasoning-Guided Retrieval for KGQA

本项目用于在 WebQSP 和 ComplexWebQuestions（CWQ）上运行推理引导的知识图谱问答。本地学生先生成检索规划，程序将规划逐步对齐到 KG 的实体和关系，分别检索证据，再用 SGS（S-expression Graph Serialization）编码子图，由同一个学生生成答案。DeepSeek-R1 用于离线生成教师规划。

本仓库在早期 Retrieval-and-Reasoning-on-KGs 工程的提示词、LoRA 训练、权重合并和评分代码基础上，按论文方法实现当前流程。仓库随附格式样例和测试。完整基准评测需要相应的数据、模型权重和检索配置。

## 1. 在本地检查安装

在项目根目录执行以下命令。六项自动化测试和小图示例只依赖 Python 3.10 或更高版本的标准库。

```bash
python3 -m unittest discover -s tests -v
python3 run_qa.py demo --output workspace/demo_predictions.jsonl
```

示例用固定规划和一张小图检查三条路径，即首次对齐成功、重写后成功、两次对齐均失败后回退到一跳。输出中的 `fallback` 表示是否触发过重写，`direct_one_hop` 表示是否最终使用了一跳检索。正式评测使用下文的 `infer` 命令。

## 2. 准备实验环境

建议在 Linux GPU 服务器上单独创建 Python 3.11 环境。依赖见 [requirements-ml.txt](requirements-ml.txt)。默认训练使用两张 GPU、BF16 和 DeepSpeed ZeRO-1，详见 [default_config.yaml](default_config.yaml)。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# 先为服务器的 CUDA 环境安装兼容的 PyTorch 2.5.1
python -m pip install -r requirements-ml.txt
python -m pip install deepspeed
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.is_bf16_supported())"
```

## 3. 接入数据和权重

复制 [paths.json](paths.json) 为 `paths.local.json`，在副本中填写自己的路径。相对路径以该配置文件所在目录为起点。Git 已忽略本机路径配置和 `workspace/`。

```bash
cp paths.json paths.local.json
```

| 配置项 | 所需文件 |
| --- | --- |
| `backbone` | 本地 Hugging Face 格式的学生基座，包含模型配置、tokenizer 和权重。 |
| `bge` | 冻结的 bge-base-en-v1.5 检索器。使用调优版时记录具体 checkpoint。 |
| `teacher` | 在线下生成规划时加载的本地教师模型。已有教师候选文件时可直接导入。 |
| `data_dir` | 基准划分目录，如 `webqsp/train.jsonl`、`webqsp/test.jsonl`。也支持旧 RoG 格式的 parquet、`q_entity`、`answer` 和已归档的 `webcwq` 数据目录。 |
| `graph` | 每题自带图时设为 `null`。也可使用全局三元组 JSONL、小型 graph.json 或磁盘 SQLite 索引。输入图的边覆盖范围决定四跳邻域能取到哪些事实。 |
| `schema` | 实体别名、关系描述与类型，格式见 [kg_schema.json](examples/kg_schema.json)。设为 `null` 时使用局部图名称，可用的类型检查相应减少。 |
| `output_dir` | 保存规划、阶段样本、适配器、合并权重和逐题预测的目录。不同实验可设置不同输出根目录。 |

每题需有唯一 `id`、`question`、已链接的主题实体 `topics` 和答案列表 `answers`。按题提供三元组时，还需 `graph`。蒸馏及第三阶段构造另需训练参考三元组 `gold` 和约束短语 `required_phrases`。训练参考证据应提前从相应版本 KG 上的 SPARQL 标注获得。可选的 `answer_ids` 用于核验终点。完整输入示例见 [train.jsonl](examples/train.jsonl)。

如果训练证据另存，每条记录使用 `dataset`、`split=train`、`id` 与原题对应；在下文 **distill 和 prepare 两条命令**中均加上 `--reference-evidence /path/to/train_reference.jsonl`。测试题只需评测输入和标准答案。

若使用全局 KG，可将每行形如 `["head","relation","tail"]` 的三元组建成磁盘索引，再把 `graph` 指向该索引。

```bash
python -m rgr.graph_index /path/to/triples.jsonl /path/to/freebase.sqlite
```

## 4. 运行一个完整实验

以下以 WebQSP 为例，CWQ 使用独立的训练规划、适配器和输出目录。

**① 获取教师规划。** 有离线候选时使用第一条命令，文件格式见 [teacher_plans.jsonl](examples/teacher_plans.jsonl)。也可以使用第二条命令在本地生成。完整 DeepSeek 检查点需要支持其模型架构和精度的环境。

```bash
python run_qa.py distill --paths paths.local.json --dataset webqsp --teacher-plans /path/to/webqsp_train_plans.jsonl
# 或在本地生成
python run_qa.py distill --paths paths.local.json --dataset webqsp --teacher-device auto
```

每题最多处理 3 个候选，最多保留 1 条规划。输出 `plans.jsonl` 逐题记录尝试次数、拒绝原因和保留结果。过滤检查图上可遍历的关系、变量依赖、可用类型及指定约束短语。涉及比较、聚合或多分支汇合的问题，仍需结合原训练标注检查。

**② 构造三阶段数据并训练。** Stage 1 学习结构任务，Stage 2 学习度数偏置随机游走生成的 SGS，Stage 3 使用问答规划及带干扰证据的输入。每个阶段结束后把 LoRA 更新合并到当前骨干，下个阶段以合并后的模型为起点。

```bash
python run_qa.py prepare --dataset webqsp --paths paths.local.json
bash tune_rm/first.sh  webqsp --paths paths.local.json
bash tune_rm/second.sh webqsp --paths paths.local.json
bash tune_rm/third.sh  webqsp --paths paths.local.json
```

| 阶段 | 初始学习率 | 轮数 | 每卡批量 | 双卡有效批量 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2 × 10⁻⁶ | 3 | 2 | 128 |
| 2 | 2 × 10⁻⁶ | 3 | 2 | 128 |
| 3 | 3 × 10⁻⁶ | 5 | 1 | 64 |

各阶段累积 32 个微批次，分别初始化 AdamW 和线性调度器，warm-up 为 0。LoRA 配置为 rank 16、alpha 16、dropout 0.05，覆盖七类投影模块。当前实现的采样种子 42、游走长度 4、关系相似度阈值 0.45 和训练长度上限 4096 等默认值可在 [config.json](config.json) 中查看或调整。若适配器已保存、权重合并中断，可在对应启动脚本后添加 `--merge-only`。

**③ 推理与评测。** 默认加载当前基准目录中的 `merged3`。若已有合并好的学生，直接在下列命令后添加 `--checkpoint /path/to/merged_student`，即可从推理开始。

```bash
python run_qa.py infer --dataset webqsp --split test --paths paths.local.json --device cuda --retriever-device cuda
```

推理对半径 4 的局部图按每条结构化查询检索最多 50 条候选，去重并保留与主题实体连通的分量。SGS 的深度上限为 8，每个遍历关系最多保留 20 个邻居。控制台汇总问题数、回退次数、Hits 和 F1；逐题证据和回答保存在 `output_dir/webqsp/test_predictions.jsonl`。

默认 Hits 使用任一标准答案命中的规则，默认 F1 使用归一化答案集合。旧工程的 F1 可通过 `--f1 legacy` 选择，两种评分口径可能产生不同结果。复现实验时记录实际采用的评分方式。比较两个方法时，提供同一基准、同一划分、问题 ID 一一对应的逐题预测。

```bash
python run_qa.py bootstrap --left /path/to/ours.jsonl --right /path/to/baseline.jsonl
```

该命令按题配对重采样 10,000 次，给出 Hits 差值及其 95% 区间，单位为百分点。

## 5. 用 GitHub 管理实验版本

在已配置私有仓库访问权限的服务器上，用 `git clone` 获取源码。把模型和基准数据放在服务器本地，逐台填写 `paths.local.json`。运行前保存 `git rev-parse HEAD` 输出的提交号、实际参数、数据版本和 checkpoint 标识，以便把后续结果对应到运行过的版本。

日常改动可以在独立分支上练习。例如调整训练说明后，先检查差异和测试，再推送分支，在 GitHub 打开 pull request 查看修改。

```bash
git switch -c docs/experiment-notes
# 编辑 README.md 后
git diff -- README.md
python3 -m unittest discover -s tests -v
git add README.md
git commit -m "Document the experiment setup"
git push -u origin docs/experiment-notes
```

本仓库的自动化测试覆盖命令行数据流、教师筛选、回放构造、局部图与 SGS、推理回退和评分。正式基准分数、多卡训练耗时和显存使用量应从配套权重和数据上的实际运行取得。
