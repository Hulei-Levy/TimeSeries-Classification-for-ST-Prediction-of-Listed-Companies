# 首次 ST 分类基线

所有代码均为本项目独立实现，不导入旧工程。原始 `datasets/` 保持不变。

## 任务与样本

- 预测日为 2006—2024 年每年 6 月 30 日；目标为未来 12 个月 `(cutoff, cutoff+1year]` 是否首次 ST/*ST。
- 排除预测日前已发生过 ST 的公司窗口。负样本要求完整随访；正样本要求至事件日的完整随访。
- ST 标签为台账支持的研究候选标签，不是交易所认证的“历史上绝对首次 ST”，也不等同于经济意义上的破产或违约。
- 输入为此前五个完整财年的 20 个季度。2024-06-30 对应 2019Q1—2023Q4。上市不足五年保留有效历史；上市前位置只作 padding。整窗无上市后记录不纳入。
- 本次 10,723 窗口，100 阳性，1,536 家公司。母体为上传的 1,545 公司清单，不能把本样本阳性率称为整个 A 股市场的自然发生率。
- 固定五组公司，各组 20 个阳性。第 k 折以组 k 测试、组 `(k+1)%5` 验证，另外三组训练。公司每折内绝不跨集合，五折各测试一次；比例约 60/20/20，精确数量见 `cache/data_audit.json`。
- 此协议评价未见公司的泛化，不是严格的按未来年份留出的时间外推。样本之间的公司内窗口重叠仅存在于同一集合。

## 时间口径及缺失

20 个节点按报告所属季度排列，是在预测日回顾历史报告，不是重现每一个历史季度的实时信息集。源报告期不得晚于对应节点，也不带入五年窗口外的记录。

可用性筛选采用 `max(报告期末+6个月, 已知源记录发布日期) <= 预测日`。六个月是未知披露日期时的统一探索性代理规则，不是实际发布日期。已知晚发布的值会屏蔽；原始数据部分为当前版本快照，完整披露日期及历史修订版本仍未核验，因此 **point_in_time_verified=false**。公司隔离与此代理规则都不能证明全流程具备严格 PIT 性质。

- 财务真实缺失不作前向填充。
- 年度治理在所属 Q4 节点进入，正常保持至下一年 Q3；新年度缺失字段不继续沿用更旧年度。
- 半年度 MD&A 在 Q2 进入并保持至 Q3；年报在 Q4 进入并保持至下一 Q1。新一期报告缺失则重置为缺失。
- 记录年龄为预测日距源报告期末的季度数，模型输入换算为年；不是距实际披露日的间隔。例：2024-06-30 使用 2023Q4 记录时，报告年龄为 2 个季度，即半年。
- 连续变量用本折训练公司的已观测有效窗口值计算中位数、均值和标准差。缺失以中位数占位后标准化，附观测标记。全空训练列用固定 0 占位、缩放 1。
- 类别变量按训练集词表 one-hot，另外保留 missing 和 unseen 类别；不把 Y/N 或性质类别当成连续量求中位数。
- 上市前位置不参与上述统计。训练统计按窗口观测计权；公司代码、预测年份和 ST 日期不输入模型。
- 保留全部 30 个财务和原有 31 个治理字段。某些治理字段历史覆盖很少，保留掩码不代表恢复了其真实历史；后续可做仅掩码及覆盖充分子样本的敏感性检验。

## 文本

年报和半年报 MD&A 全文，使用 `google-bert/bert-base-chinese`，固定版本 `8f23c25b06e129b6c986331a13d8d025a92cf0ea`。通过公开镜像取得相同版本文件，来源和 SHA256 位于 `cache/bert-base-chinese/source.json`。

BERT 冻结且 eval。正文分为不重叠的 510-token 块，加 CLS/SEP 后不超过 512；取最后隐藏层，排除 CLS/SEP/PAD，对全部正文 token 平均。这等价于按有效 token 数对块向量加权，生成每份报告 768 维向量。各块独立编码，不声称有跨块注意力。

缓存仅保存冻结向量，不含 ST 标签、公司训练状态或已训练投影层。各模型、各折单独初始化并训练 `Linear(768,64)`，随后与数值、类别、掩码及报告年龄拼接。缺失报告映射后仍强制置零。旧五项 MD&A 计数指标完全不使用。

## 模型与训练

14 个模型：DLinear、TiRex、XLSTMMixer、LightTS、Pyraformer、TimeFilter、TimesNet、Reformer、Informer、Autoformer、ETSformer、FEDformer、TimeMoE、SEMPO。

这些是 **从头训练、固定小容量的架构分类适配**，不是官方代码的逐位复现，也不代表原论文预训练模型的零样本能力。各自保留的算子、参数与适配边界位于两个模型文件的 `MODEL_NOTES` 和来源说明中。主要改动包括分类读出替代预测输出、无需的 forecasting decoder 省略，以及禁用会消除绝对水平的逐窗口输入 RevIN；常规隐藏层 LayerNorm 保留。

LightTS 沿用用户旧代码实际采用的采样 MLP，正确来源为 Zhang et al. (2022), *Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures*, https://arxiv.org/abs/2207.01186 。它不是 Campos et al. 的 SIGMOD 2023 集成蒸馏同名方法。

固定配置在 `config.json`：隐藏维 64、深度 2、dropout 0.1；AdamW 学习率 0.001、weight decay 0.0001；有效批量 64、最多 100 epoch，验证 AP 连续 15 epoch 未改善则停止。TimeFilter 以 microbatch 16 累积到同样有效批量 64，以控制显存。架构专有参数在各类构造器中固定，不通过测试集选择。

所有样本进入训练，无负样本欠采样。BCEWithLogitsLoss 的正类权重为本折训练集负/正数量。需要路由辅助损失的模型使用固定系数 0.01。梯度裁剪为 1。各折随机种子固定为 42，不额外重复种子。相同种子不能保证不同模型具有完全相同初始化或 GPU 运算逐位一致。

按验证集 **Average Precision (AP)** 选择 checkpoint；测试阈值固定 0.5。AP 不冒称梯形积分 PR-AUC。报告 Accuracy、Precision、Recall、F1、TNR、MCC、ROC-AUC、AP、Brier、10-bin ECE、前 10% 捕获率。类加权训练的输出不自动视为已校准概率。

汇总包含五折均值、样本标准差和每个样本恰好一次的 OOF 预测指标。折间标准差不是置信区间，也不估计随机初始化带来的波动。所有模型使用同一数据和分组。

## 文件与运行

基线使用四个源文件：`data.py`（构建和审计）、`models_classic.py`（九种骨干）、`models_recent.py`（五种骨干）、`run.py`（BERT、训练、评估与双 GPU 调度）。另有本说明和配置文件；自有模型新增的两个源文件见下节。缓存、日志、结果分别集中在 `cache/`、`logs/`、`results/`。

宿主机 Python 已有 pandas/pyarrow，可重建样本：

```bash
python3 -B data.py build
```

训练使用现有 Docker 镜像 `huawei-q2-lhx:20260923`，Python 3.11、PyTorch 2.6.0+cu124、Transformers 4.48.3、scikit-learn 1.6.1；不修改宿主全局 Python 环境。将整个项目挂载到容器原绝对路径并以项目所有者 UID/GID 运行。首次使用前须已取得上述固定 BERT 权重。

```bash
python -u run.py pipeline
# 独立工程检查（合成文本向量；绝不计入实验结果）
python -u run.py smoke
# 汇总已完成的五折结果
python -u run.py summary
```

`pipeline` 先在两张 GPU 上分片编码全部文档，检查模型，再各用一张 GPU 执行独立训练任务。失败会留下明确日志并停止对应阶段；完整结果只在所有步骤成功后生成。冻结向量可断点复用；已完成训练只有数据/代码/配置身份一致时才会跳过。

运行状态见 `logs/` 与 `cache/bert_progress_*.json`；每折完整指标、预测、预处理参数和最佳 checkpoint 在 `results/<model>/fold_<k>/`。最终汇总为 `results/summary.csv` 和 `results/summary.json`。正式结果生成前不得将工程 smoke 检查当成性能。


## PerSeM 历史对照与多频率消融

新增 `models_persem.py` 和 `run_persem.py`。前者独立重写旧 `ThisWork.py` 的 PerSeM 主干和三个新模块；后者只复用本次新工程 `run.py` 的数据、预处理、划分、BERT 缓存与评分工具，不导入旧工程。基线源文件和 `config.json` 保持原版本，不中断或改变正在进行的 70 次训练。

八个配置为 PerSeM、A、B、C、AB、AC、BC、ABC，各五折，共 40 次从头训练。A/B/C 是下面的三个独立开关。全部配置保留相同的完整五年主分支及分类头；同一 seed 下公共主分支和文本投影的初始参数一致。新模块提供附加表示，不能把模块收益自动等同于排除了参数量增加后的机制收益；结果中同时报告可训练参数量。

- **主干**：季度标记和位置嵌入、FFT 周期折叠、多核二维卷积、时序注意力及内外残差，保留 20 个时间位置，通过 flatten→64→单 logit 分类。逐公司选周期，修正旧实现跨 batch 平均选周期的问题。固定小容量、适配新输入，不能将结果称为原论文旧数据实验的逐位复现。
- **A，多尺度自身历史参照**：三模态各自按原生记录构造五个独立年度向量。最近一年固定为 query，与之前 1／2／4 个日历年中有效历史的均值比较；输入当前、参照、有符号差值、绝对差值、覆盖率和报告年龄差，通过门控选择尺度，再融合各模态的比较表示。
- **B，自身历史检索比较**：每家公司、每个模态独立从前四年中检索与最近一年余弦相似度最高的两个有效年度片段，再学习权重做差异比较。仅有一年历史就使用一年，无 query 或无历史时该模态分支为零。检索相似度中的年度内容向量不包含绝对年份或报告年龄；年龄差在比较阶段作为辅助。没有跨公司记忆库，也不按未来 ST 标签筛所谓健康历史。
- **C，多频率融合**：财务保留 20 个季度，治理只取每年 Q4 共 5 个年度节点，MD&A 只取 Q2／Q4 共 10 个半年节点；分别编码。以财务原生季度表示加时间位置为 query，通过带报告期距离偏置的交叉注意力读取各模态，并在每个季度门控融合，再保留 20 节点读出。C 分支不读取季度重复持有的治理／文本记录；原主分支仍保留统一季度输入。缺失财务季度可由时间 query 读取其他有效模态。

历史年度向量在任何全五年卷积、注意力、FFT 混合之前，先从原生输入拆分年度并独立编码，避免最新一年混入历史参照。缺失标记用于区分真实观察与训练中位数占位；不能把填充值当成有效历史。全窗口分类及 C 的注意力为预测截点下的回顾性编码，可在窗口内双向交互；不声称重建了每个历史季度的实时可用信息。时间特征是报告期年龄，仍受上文披露日期代理及历史修订版本未核验的限制。

训练参数、样本、五折公司划分、单层 BERT 映射、类别权重、验证 AP 早停及固定 0.5 阈值与基线一致。没有额外调参或额外随机种子重复。先做 CPU 工程验证，正式训练由完成门禁控制：基线 supervisor 在两 GPU worker 均退出后写出完整 70 次汇总，且结果与当前数据/源码指纹一致，才允许启动自有模型。

```bash
# 只做少量样本前反向工程验证，不生成正式性能指标
python -u -B run_persem.py smoke --device cpu
# 后台等待基线全部完成，然后双 GPU 执行 40 次训练
python -u -B run_persem.py pipeline
# 汇总已完成的自有模型结果，不覆盖基线 summary
python -u -B run_persem.py summary
```

状态在 `cache/persem_pipeline_status.json`；两 GPU 日志在 `logs/persem_gpu0.log` 和 `logs/persem_gpu1.log`。每折产物位于 `results/PerSeM*/fold_<k>/`。自有模型汇总保存为 `results/persem_summary.json` 和 `results/persem_summary.csv`；合并对照表为 `results/comparison.csv`，包含 14 个基线及最多 8 个完成五折的自有模型配置。它们不覆盖原基线 `results/summary.*`。

修改本说明前的六个原文件已备份并逐字节验证于 `cache/backups/before_persem_20261022.zip`。正式启动前另保存包括新模型与调度器的源码快照。若正式训练后代码、配置、数据或文本缓存变动，指纹检查会拒绝混用旧结果。


## 2026-10-23 SEMPO 执行后端修复

原 13 个基线的 65 次训练已完成且 OOF 核验通过。SEMPO 在完整 batch64 下将 batch、特征和四个视图展开为 69,376 个注意力实例，默认高效 CUDA SDPA 内核报 invalid configuration；之前 batch2 工程检查未覆盖这个形状。采用标准 `torch.nn.attention.SDPBackend.MATH` 后，完整 batch 的前向、反向及 AdamW 更新通过；峰值分配显存约 7.21 GiB，小 batch eval 与默认后端最大误差约 1.91e-6。

修复只在 SEMPO 实际训练子进程内切换注意力执行后端，保留 batch64、模型结构、优化器、样本、seed 和早停设置。基线的源文件、配置及指纹未改变。后端选择记于 `cache/sempo_recovery.json` 和新完成的 SEMPO 折指标。已完成的 65 份指标和预测在恢复前后逐文件 SHA256 核对，不重新训练或覆盖。

现有 `run_persem.py` 增加恢复入口，不新增脚本文件：

```bash
python -u -B run_persem.py recover
```

它先用两 GPU 补跑 SEMPO 五折，生成完整 70 次基线汇总，再进入原定 40 次 PerSeM 消融队列。恢复前代码、汇总、失败日志和状态备份在 `cache/backups/before_sempo_recovery_20261023.zip`。PerSeM 尚无正式结果；这次只修改调度入口，已逐 AST 核对 PerSeM 训练、输入、评分、流水线函数及模型文件均未改变，源码身份与快照更新为 2026-10-23 版本。


## 2026-10-23 训练集欠采样试验（sampling_v1）

本轮仅预定 PerSeM、PerSeM_AB、PerSeM_ABC，分别使用阳性∶阴性 1∶3 和 1∶5，各五折，共 30 次新增训练。原有 110 次全训练集结果保留，作为同模型的对照。两个比例均为预先指定的实验条件，分别报告验证与测试结果，不根据测试结果选择比例、权重或模型，也不自动启动全部基线及八组消融。

公司分组与标签不重新构建。先取原五折的训练公司池，再保留全部阳性并抽取阴性；验证、测试窗口完全不抽样。阴性仍为截点后一年未首次 ST 且随访完整的合法窗口；不排除未来会 ST 的公司的早期阴性。每家公司最多贡献一个阴性窗口，阳性窗口另行全部保留。因严格按阳性预测年份配比，某训练折没有阳性的年份不会贡献独立分类训练窗口，但仍可存在于入选样本内部的历史，以及完整验证和测试集。

规模使用样本末个原生治理节点的 AF_CFEATUREPROFILE-CompanySize，其历史补充口径含期初资产代理，不是市值；仅使用当前样本中已通过原可得性屏蔽的值。匹配严格同年，在对数规模距离上做软匹配，使用局部 seed=42 随机数处理同成本选择；规模缺失时不采用规模距离。现有数据没有可靠历史行业字段，本轮不启用行业匹配。先全局分配每个阳性对应的五个不同公司阴性，再取其中距离最近的三个形成 1∶3 子集；两比例清单固定，三个模型共享。若无法满足年份、公司不重复和数量约束，直接失败，不跨年、不放回、不静默改变比例。

预处理继续使用各折抽样前完整训练池拟合的插补、标准化及类别词表，并逐项比对旧结果，维度与模型容量保持一致。未入选窗口参与这些无监督预处理统计，但不进入分类损失；验证与测试不参与拟合。BCE 正类权重按实际入选阴性/阳性数量重算，分别为 3 和 5，不沿用约 106。保持原 batch64、最多100 epoch、patience15、学习率、种子42、验证AP选checkpoint与测试阈值0.5，不新增对比损失。每折分别为60+180=240或60+300=360训练窗口，即每epoch约4/6次更新；与原全训练池并非相同更新步数预算，必须连同实际epoch、步数、样本暴露次数一起解释结果。

历史参照只依赖每个入选预测样本自己的五年输入：最近一年为query，前四年是候选。原A/B/C模型不改，不读取其他独立分类样本的入选状态或未来标签。数值原生节点至少有一个真实观测字段才有效；文本需有可用报告；年度至少一个有效原生节点。部分缺失沿用占位值、掩码、覆盖率及报告年龄。query整年无效或历史不足一条时该模态比较分支为零；B只有一个候选时只使用该候选，不要求更长历史，也不假设历史健康。

仅新增 sampling.py 负责固定匹配清单；run_persem.py 复用原训练函数增加可选采样分支与独立调度，其他模型、数据构建、基础训练和config源文件保持不变。修改前已备份，原结果文件身份记录在 cache/sampling_v1/legacy_identity.json。清单、协议、工程验证、源码快照和状态统一放在 cache/sampling_v1；新结果只写 results/sampling_v1/ratio_3 或 ratio_5，不能覆盖原模型目录或原 comparison.csv。新汇总是 results/sampling_v1/summary.json、comparison.csv 和 validation_summary.csv；日志为 logs/sampling_gpu0.log、sampling_gpu1.log。每折记录训练前后/验证/测试公司数、窗口数、阳性率、年份匹配、抽样清单哈希、源码指纹和实际更新次数；保存独立的验证及测试预测文件。

```bash
python -u -B run_persem.py sampling-prepare
# 完成清单、共享输入及完整batch工程检查，并绑定validation.json至当前源码指纹后：
python -u -B run_persem.py sampling-pipeline
python -u -B run_persem.py sampling-summary
```


## 2026-10-23 欠采样基线扩展（sampled_baselines_v1）

用户确认继续补跑同一14个基线，在既有1∶3和1∶5条件下各五折，共140次新增训练，不重新选择比例。直接加载 cache/sampling_v1/fold_0..4.json 已固定清单，与已完成的 PerSeM／AB／ABC 两比例30次试验共享训练样本、原完整训练池预处理、公司分组及完整验证／测试分布。BERT、原模型结构、seed42、config.json、正类权重按实际3／5重算、验证AP早停选checkpoint、0.5分类阈值均保持。TimeFilter保留microbatch16累积有效batch64及eval16，所有基线保留原架构所需辅助损失及系数；SEMPO在实际训练进程全程使用SDPBackend.MATH，不改batch64。

仅新增 run_sampled_baselines.py 复用本项目公共数据、模型和评分接口；现有源代码、模型文件、配置与两份既有结果对照表不修改。修改说明前的9个源/配置文件已备份于 cache/backups/before_sampled_baselines_20261023T071518Z.zip；旧908件结果的身份在 cache/sampled_baselines_v1/legacy_identity.json。新协议、工程验证、源码快照与状态集中在 cache/sampled_baselines_v1；正式结果写 results/sampled_baselines_v1/ratio_3 或 ratio_5 下，日志为 logs/sampled_baselines_gpu0.log 和 sampled_baselines_gpu1.log。

新 summary.json 记录28个基线条件的五折测试及验证统计，comparison.csv 合并这28个条件与已完成自有模型6条件共34行，引用时保留来源和源码身份；validation_summary.csv 单独报告验证指标。不会根据测试表现选择比例或调整训练。此前已完成110次全样本训练（70基线＋40自有消融），以及30次采样自有模型训练；本轮另新增140次采样基线训练，分别计数。

```bash
python -u -B run_sampled_baselines.py prepare
# 完成全批量工程检查并保存同指纹validation.json后
python -u -B run_sampled_baselines.py pipeline
python -u -B run_sampled_baselines.py summary
```

当前主控计划为Docker容器 st-sampled-baselines-20261023，使用两张GPU；20分钟心跳跟踪本轮140次新基线，不把先前已完成任务计入新完成数。全部140次和34行合并表核验通过后暂停心跳。
