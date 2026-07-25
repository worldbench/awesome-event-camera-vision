# 人工变更记录

本文件位于主论文项目之外，独立于 Git，用于记录主项目的所有实质修改。Agent 不执行
Git 添加、提交或推送；用户在 Windows 中自行完成版本控制。

## 记录规则

- 只追加，不覆盖旧记录。
- 主项目内容修改必须同时记录修改前和修改后。
- 大批量修改应列出范围、数量、代表性条目和可恢复来源。
- 旧记录若有错误，应追加更正记录。
- 每项记录应包含原因、证据、验证结果和待用户处理事项。

## 模板

```text
### YYYY-MM-DD-NN：简短标题

- 执行者：
- 范围：
- 修改前：
- 修改后：
- 原因/证据：
- 验证：
- 待用户处理：
```

## 变更记录

### 2026-07-24-01：确立 Overleaf/LaTeX 主项目并整理外围工作区

- 执行者：Codex
- 范围：外围目录、Agent 规则和私有管理文档；不修改主论文正文、BibTeX、宏或模板。
- 修改前：
  - 根目录混放主 LaTeX、awesome-list、旧 `筛查/`、`new_check/`、旧草稿、分类稿、脚本、缓存和压缩快照。
  - `AGENTS.md` 仍把 awesome-list 和 `筛查/main.bib` 作为主要工作材料。
  - 没有独立于 Git 的统一人工变更记录。
  - 主 Bib 有 572 个唯一条目、388 个唯一被引 key、0 个缺失引用、107 个 `others` 作者占位。
  - `main.tex` SHA-256：`87385960d87cd63685662b43e610e502f538087847f7aea7847a9c925eca252a`。
  - `main.bib` SHA-256：`71e66d15d01b9612ee8ad2f56574892bc5ffe3fa89e7f5cad9cf24e1acda3ca9`。
- 修改后：
  - `TPAMI2026_Survey_Event_Camera_Vision/` 是唯一主项目，内部未增加管理或审计文档。
  - 外层新增 `PROJECT_STATUS.md`、`CHANGELOG_MANUAL.md` 和 `docs/`。
  - 仍有效的研究材料整理到 `reference_materials/`。
  - 旧 README 工程、筛查、草稿、脚本、缓存和快照移入 `bin/`。
  - `AGENTS.md` 改为以当前 Tex/Bib 为事实来源，并禁止 Agent 执行 Git 操作。
- 原因/证据：
  - 用户决定未来以 Overleaf 中的 Tex/Bib 为导向，并自行在 Windows 处理 Git。
  - 静态依赖检查确认 `main.tex` 不依赖待归档的外围文件。
  - 旧筛查记录与当前 `main.bib` 的人工确认错误存在冲突。
- 验证：
  - 目录移动后重新检查 `main.tex` 的 `input`/`bibliography` 依赖。
  - 重新统计 Bib 条目、被引 key、缺失引用和占位作者。
  - 对比整理前后 `main.tex`、`main.bib`、`sections/*.tex` SHA-256。
- 待用户处理：
  - 在 Windows 检查新目录布局。
  - 建立 GitHub/Overleaf 连接后自行执行 Git 添加与提交。

### 2026-07-24-02：重建部分纯 arXiv 文献删减初筛

- 执行者：Codex
- 范围：外层 `docs/reconstruction_screening/` 与 `tools/reconstruction_screening/`；未修改主项目 Tex/Bib，未编译，未执行 Git。
- 修改前：
  - reconstruction catalog 共 117 项，其中 16 项标为 arXiv。
  - 没有按“正式发表状态、引用量、公开年龄、代码和数据”统一交叉核验的当前报告。
  - 自动标题匹配曾把 `EVDI++` 错连到 `EvDiff`；`Self-EHDRI` 未在当前 Bib 中找到；`ESHDR` 的 Bib arXiv ID 与官方页面冲突。
- 修改后：
  - 建立 117 项清单及 16 项 arXiv 候选的显式身份映射。
  - 批量缓存 Semantic Scholar 和 OpenAlex 数据，引用量保守取两库较大值。
  - 对 16 项逐一核验正式发表和官方代码/数据，生成按建议层级排序的 `docs/reconstruction_screening/REPORT.md` 及 CSV/JSON。
  - 确认最强删减候选为 `EventDiff`；三个 2025 年末零引新稿进入观察期；六项实际已正式发表，不应按纯 arXiv 删除。
- 原因/证据：
  - 用户要求优先减少纯 arXiv、低引用、无代码/数据工作，同时考虑论文年龄并增加 LLM/VLM/基础模型相关内容。
  - 身份与发表状态优先采用出版社/正式 proceedings、arXiv 与官方项目仓库；引用数由 Semantic Scholar/OpenAlex 交叉。
  - arXiv 批量 API 持续返回 429，故不把 API 缺失当负面证据，改用官方页面定点复核。
- 验证：
  - Semantic Scholar 32 个候选身份记录无错误；OpenAlex 32 个查询无网络错误。
  - 报告最终仅纳入 catalog 明写为 arXiv 的 16 项，避免重复计算。
  - 重新核对主 `main.tex`、`main.bib` SHA-256，必须与基线一致。
- 待用户处理：
  - 人工决定是否先删除 `EventDiff`，以及是否继续讨论 `ESHDR`、`E2VIDiff`、`EBAD-Gaussian`。
  - 后续另开一批修改 catalog/Bib 的过时 venue 与 `ESHDR` arXiv ID；本批只报告，不应用修改。

### 2026-07-24-03：六篇重建论文由 arXiv 条目更新为正式出版 BibTeX

- 执行者：Codex
- 范围：主项目 `main.bib` 中六个既有 citation key；外层新增专项核验说明并刷新筛查报告。未修改 Tex/catalog/appendix，未编译，未执行 Git。
- 修改前：
  - `chen2024lase`：`@article`，`journal={arXiv preprint arXiv:2407.05547}`，无正式页码和 DOI。
  - `zhang2025evdiplus`：`@article`，`journal={arXiv preprint arXiv:2509.08260}`，`year={2025}`。
  - `zhang2023crosszoom`：arXiv `2309.16949`，`year={2023}`；题名误写为 `Simultaneously Motion Deblurring`。
  - `li2024hdr`：arXiv `2404.08640`，`year={2024}`；该 arXiv ID 错误，作者顺序也与正式版不一致。
  - `wang2024evggs`：`@article`，`journal={arXiv preprint arXiv:2405.14959}`。
  - `zhang2024eliteevgs`：`@article`，`journal={arXiv preprint arXiv:2409.13392}`，`year={2024}`。
  - `main.bib` SHA-256：`71e66d15d01b9612ee8ad2f56574892bc5ffe3fa89e7f5cad9cf24e1acda3ca9`。
- 修改后：
  - `chen2024lase`：`@inproceedings`，NeurIPS 37 (2024), 70406--70430，DOI `10.52202/079017-2250`。
  - `zhang2025evdiplus`：TPAMI Early Access (2026), 1--18，DOI `10.1109/TPAMI.2026.3697759`；key 保持不变。
  - `zhang2023crosszoom`：TPAMI 46(12), 8209--8227 (2024)，DOI `10.1109/TPAMI.2024.3402972`；正式题名改为 `Simultaneous`。
  - `li2024hdr`：Pattern Recognition 180, 114265 (2026)，DOI `10.1016/j.patcog.2026.114265`；正式作者顺序改为 Zeng、Li、Fan、Zhao、Deng、Yu；key 保持不变。
  - `wang2024evggs`：`@inproceedings`，ICML 2024 / PMLR 235, 50561--50579。
  - `zhang2024eliteevgs`：`@inproceedings`，ICRA 2025, 13972--13978，DOI `10.1109/ICRA55743.2025.11127518`。
  - `main.bib` SHA-256：`af261498234e79cfb3e6623de0bd5c460a947c03b0e54e1a482d589795f57ef4`。
- 原因/证据：
  - 六篇的正式题名和作者均与原 arXiv 身份对应；没有根据相似标题推断。
  - NeurIPS、PMLR、Elsevier 正式页面与 Crossref DOI 注册元数据交叉一致。
  - 完整修改前后条目和逐篇链接见 `docs/reference_audit/reconstruction_six_formal_publication_bib_update.md`。
- 验证：
  - `main.bib` 仍为 572 个条目且 572 个 key 唯一。
  - Tex 共引用 388 个唯一 key，缺失引用为 0。
  - 六个 citation key 均未重命名。
  - `main.tex` SHA-256 仍为 `87385960d87cd63685662b43e610e502f538087847f7aea7847a9c925eca252a`。
- 待用户处理：
  - catalog 和 appendix 中仍存在相应旧 arXiv venue；若要修改，应作为下一批 Tex 变更单独记录。
  - 在 Windows/Overleaf 检查参考文献显示效果，并自行执行 Git 添加与提交。

### 2026-07-24-04：重建删减排序加入综述覆盖价值

- 执行者：Codex
- 范围：外层重建筛查报告、人工证据和排序脚本；未修改主项目 Tex/Bib。
- 修改前：排序主要依据正式发表、引用量、公开年龄、代码和数据，容易低估稀缺任务及大模型相关新工作的综述价值。
- 修改后：
  - 新增“综述覆盖价值”，奖励稀缺任务、基础模型/语言相关性和独特技术演进位置。
  - 最终排序分改为“基础风险分 − 综述覆盖价值”，原始引用和开源事实保持不变。
  - 优先删减复核调整为 `EBAD-Gaussian`、`EventDiff`。
  - `ESHDR`、`E2VIDiff` 上调为建议保留；`EvDiff`、`IE2Video` 列为新论文优先保留观察；`DESSERT` 为一般观察。
- 原因/证据：用户明确要求文献较少的分类和近年的大模型相关工作，即使低引或仅 arXiv，也应比老旧、重复度高的工作获得更高保留优先级。
- 验证：报告仍包含 16 个当前 catalog arXiv 候选；Semantic Scholar/OpenAlex 原始引用数未改；主 `main.tex`、`main.bib` 哈希未变。
- 待用户处理：人工确认排序原则后，再决定实际删除项；本批没有删除任何论文。

### 2026-07-24-05：EBAD-Gaussian 更新为 ICASSP 2026 正式版并撤回纯 arXiv 结论

- 执行者：Codex
- 范围：主项目 `main.bib` 中 `deng2025ebadgaussian`；外层 Agent 规则、专项核验说明、项目状态、重建筛查证据/报告及其生成源。未修改 Tex/catalog/appendix，未编译，未执行 Git。
- 修改前：
  - `deng2025ebadgaussian` 是 `@article`，题名为 `EBAD-Gaussian: Event-Driven Bundle Adjusted Deblur Gaussian Splatting`，作者为 arXiv:2504.10012 的九位作者，`journal` 为 arXiv preprint，年份 2025。
  - `docs/reconstruction_screening/REPORT.md`、`PROJECT_STATUS.md` 和人工证据将其视为纯 arXiv，并列入最高优先级删减复核。
  - 报告生成脚本固定写着“六项并非纯 arXiv”，且没有把 catalog 旧名显式映射到更名后的正式论文。
  - `AGENTS.md` 要求追加人工变更记录，但没有明确单条 Bib 修改也必须同任务记录、Bib 核验需保留完整前后条目、推翻旧结论时需同步更新生成源。
  - `main.bib` SHA-256：`af261498234e79cfb3e6623de0bd5c460a947c03b0e54e1a482d589795f57ef4`。
- 修改后：
  - citation key 保持 `deng2025ebadgaussian`；条目改为 `@inproceedings`，正式题名为 `EBAD-GS: Deblurring Gaussian Splatting with Event-Driven Bundle Adjustment`，采用 IEEE 正式版六位作者。
  - 增加 ICASSP 2026 正式 booktitle、页码 10902--10906、DOI `10.1109/ICASSP55912.2026.11464704`、DOI URL 和年份 2026。
  - 新增 `docs/reference_audit/ebad_gs_icassp2026_bib_update.md`，保留完整修改前/后条目、作者差异、判断边界和权威证据。
  - EBAD 从纯 arXiv/优先删减复核移除；当前优先删减复核只剩 `EventDiff`，正式发表组由六项改为七项。
  - 在 inventory 生成源中加入 `EBAD-Gaussian` catalog 旧名、arXiv ID 与正式 `EBAD-GS` Bib 条目的显式身份映射，并重新生成 CSV、JSON 和报告。
  - `AGENTS.md` 明确：每次修改均须在同任务留下前后记录；Bib 修改还须有专项核验说明；新证据推翻当前报告时要同步更新状态、报告和生成源，不得等待用户重复提醒。
  - `main.bib` SHA-256：`489865f042a12c067b054ce9160c5e8f93f577fc0a11ba6240058ca4c03761de`。
- 原因/证据：
  - IEEE ICASSP 2026 官方记录和 DOI 元数据确认论文正式发表，题名、六位作者、会议、年份和页码一致。
  - 正式版六位作者均包含于原预印本九位作者中，且专名、任务和 event-driven bundle adjustment 技术核心高度一致；因此按正式会议版替换，但不使用预印本作者表覆盖会议版。
  - 完整证据链接见 `docs/reference_audit/ebad_gs_icassp2026_bib_update.md`。
- 验证：
  - `main.bib` 仍有 572 个条目和 572 个唯一 key，无重复 key。
  - Tex 共引用 388 个唯一 key，缺失引用为 0；citation key 未改名。
  - 重建 inventory 仍有 117 行、16 个 catalog 明写为 arXiv 的候选，16 个均保有 arXiv 身份；重新生成的报告把 EBAD 放入 7 项正式发表组。
  - `main.tex` SHA-256 仍为 `87385960d87cd63685662b43e610e502f538087847f7aea7847a9c925eca252a`。
- 待用户处理：
  - catalog/appendix 仍显示旧名称 `EBAD-Gaussian` 和 `arXiv'25`；如需改为 `EBAD-GS` / `ICASSP'26`，应作为单独 Tex 修改记录。
  - 在 Windows/Overleaf 检查参考文献显示效果，并自行执行 Git 添加与提交。

### 2026-07-24-06：重建删减报告由纯 arXiv 扩展为全部 117 项

- 执行者：Codex
- 范围：外层 `docs/reconstruction_screening/`、`tools/reconstruction_screening/`、`PROJECT_STATUS.md` 和本修改历史；未修改主项目 Tex/Bib，未编译，未执行 Git。
- 修改前：
  - 主 `REPORT.md` 只覆盖 catalog 明写为 arXiv 的 16 项，无法发现“已经正式发表但老旧、低引、同质化或 venue 可见度较低”的论文。
  - `EvDiff` 仅因新近、大模型相关而列为优先保留观察，尚未记录 ECCV 2026 接收信息和用户要求的作者团队/机构因素。
  - 当前优先删减复核只剩纯 arXiv `EventDiff`，不能回答传统 CNN、NeRF、3DGS 文献是否过多。
  - 全量引用缓存、全量机器可读结果和“低引但具有独特价值”的保护清单均不存在。
- 修改后：
  - 新 `REPORT.md` 覆盖 reconstruction catalog 全部 117 项；原 16 项报告移为 `ARXIV_ONLY_APPENDIX.md`。
  - `EvDiff` 改为明确保留：Ming-Hsuan Yang 官方论文列表已列其为 ECCV 2026 conference paper；正式 proceedings 尚未上线，因此本批不改主 Bib。
  - 当前已发表高优先级删减复核为 `Revisit-EBVFI`、`Ev3DGS`；中优先级为 `SaENeRF`、`BeSplat`，纯 arXiv `EventDiff` 排在已发表冗余候选之后。
  - 新增低引保护清单：`STLR`、`EvINR`、`Event-ID`、`Sim2Real-EVFI`、`CMTA`；它们分别因范式、稀缺任务、数据或正式 venue 价值受保护。
  - 新增较低可见度 venue 保护/复核表：venue 只作附加降权，不单独决定删除；`E2GS`、`Ev-GS`、`E-3DGS` 因引用、早期代表性或独特硬件/数据/代码保留。
  - 新增 `fetch_full_evidence.py`、`generate_full_report.py`、`full_review_evidence.json` 及全量 CSV/JSON/缓存；README 记录可复现运行顺序。
  - 发现 catalog 有 10 项未绑定当前主 Bib；身份未完成前不把引用缺失当作零引用。
  - 新发现 `SaENeRF` 已正式发表于 IJCNN 2025，DOI `10.1109/IJCNN64981.2025.11227637`；当前 Bib 不仅仍为 arXiv，作者表也完全错误。已新增 `docs/reference_audit/saenerf_pending_bib_correction.md`，保留当前错误条目与建议完整替换条目；本轮仅记录，不越过筛查范围修改 Bib。
- 原因/证据：
  - 用户要求已发表但老旧、低引、技术普通以及较低知名度 venue 的论文也进入报告，并希望压缩大模型之前的普通增量工作、保护后大模型工作。
  - 引用元数据采用 Semantic Scholar 的 DOI/arXiv 稳定身份和精确题名查询；技术与代码/数据价值由正式 proceedings、arXiv、官方作者页和 GitHub 定点核验。
  - `EvDiff` 的 ECCV 2026 状态来自 Ming-Hsuan Yang 官方论文列表；`Revisit-EBVFI` 的正式 IROS 记录和 DOI、`Ev3DGS` 的 APSIPA DOI、`SaENeRF` 的 IJCNN DOI、`BeSplat` 的 CVF workshop 页面均已逐项核对。
  - OpenAlex 本日匿名额度已耗尽并返回 429；报告明确把未返回写成缺失，不把它们误判为零引用。
- 验证：
  - 全量 inventory 仍为 117 行，其中 107 项绑定当前 Bib、10 项未绑定。
  - `full_scope_results.csv/json` 均生成 117 条记录；Semantic Scholar 缓存含 107 个请求记录，限流错误原样保留。
  - 报告生成脚本通过只读语法编译检查；主 `main.tex` 和 `main.bib` 均未修改。
  - `main.tex` SHA-256 仍为 `87385960d87cd63685662b43e610e502f538087847f7aea7847a9c925eca252a`；`main.bib` SHA-256 仍为 `489865f042a12c067b054ce9160c5e8f93f577fc0a11ba6240058ca4c03761de`。
- 待用户处理：
  - 人工决定是否先从 `Revisit-EBVFI`、`Ev3DGS` 开始删除；本批没有删除任何论文。
  - 下一批可先补齐 10 项未绑定论文身份，并专项更正 `SaENeRF` 正式 BibTeX，再继续扩展删除候选。
  - 若决定更改 catalog/正文，需作为 Tex 修改单独记录并由用户在 Overleaf 编译。

### 2026-07-24-07：恢复52项限流引文并生成117项完整对照表

- 执行者：Codex
- 范围：外层 `docs/reconstruction_screening/`、`tools/reconstruction_screening/`、`PROJECT_STATUS.md` 和本修改历史；未修改主项目 Tex/Bib，未编译，未执行 Git。
- 修改前：
  - 已绑定主 Bib 的107项中，52项因 Semantic Scholar 标题查询触发51次 HTTP 429 和1次 HTTP 504而缺少引文值。
  - 报告只把缺失标为“—”，没有可供逐项比较的117项总表；旧人工摘要仍写 `CMTA=0`、`BeSplat=7`。
  - 原脚本逐篇查询标题，失败重跑容易再次消耗限额；生成器也没有把身份恢复证据呈现到报告。
- 修改后：
  - 新增 `recover_failed_citations.py`：仅处理失败缓存，通过 Crossref/DBLP 解析稳定身份，要求标题相似度至少0.93且年份差不超过1年，再以 Semantic Scholar batch 查询；成功项不重复请求。
  - 52项旧失败全部恢复；107个已绑定条目均有 Semantic Scholar 引文值，当前接口失败缺失为0。
  - `REPORT.md` 新增按12个重建子类展开的117项全表、子类中位数、2024及以前低引数量和粗略“引用/年”；10个未绑定条目继续显示“—”，不视为零引。
  - 统一稳定身份快照后，`CMTA` 更正为21引，`BeSplat` 更正为0引；相应人工判断文字已同步。
  - 身份解析候选、标题相似度、年份差及最终稳定标识保存在 `cache/failed_identity_resolution.json`；机器可读总表同步更新。
- 原因/证据：
  - Semantic Scholar 单篇标题接口受匿名速率限制，但 batch 稳定标识接口可用；Crossref/DBLP用于论文身份解析，不用于伪造引文数。
  - 对 DOI 无法被 Semantic Scholar 索引的10项，使用经 arXiv、DBLP或CVF标题核验的 arXiv ID 作为备用入口。
  - 引文数字统一来自2026-07-24 Semantic Scholar 快照，避免把不同数据库口径混在同一列。
- 验证：
  - `full_scope_results.json` 共117行：107行有引文值、0个已绑定条目缺失、10行未绑定。
  - `REPORT.md` 含117个 catalog 数据行；旧“52项仍受限流影响”和旧 `CMTA=0` 表述已不存在。
  - 恢复脚本报告 `targeted_failed_records=10, recovered=10, still_failed=0`；结合此前分段缓存，`_recovered` 标记共52项。
  - 主项目 Tex/Bib 未修改。
- 待用户处理：
  - 根据完整总表先决定需要做“技术冗余/venue/开源”人工深查的候选；本批没有删除或改写任何参考文献。
  - 10项未绑定主 Bib 的身份问题是独立任务，不影响本批52项限流恢复结论。

### 2026-07-24-08：修正EMVS的Semantic Scholar重复空记录

- 执行者：Codex
- 范围：外层完整筛查报告、报告生成器、机器结果、项目状态和本修改历史；未修改主项目 Tex/Bib。
- 修改前：`EMVS` 通过 DOI `10.1007/s11263-017-1050-6` 命中 Semantic Scholar 的同题名空壳记录，显示0引，因而被全量表错误标为“较老且低引”。
- 修改后：
  - 保留 Semantic Scholar 原始0引值用于审计；
  - 决策用引用改为 Crossref 同一 DOI 的 `is-referenced-by-count=243`；
  - 全量表增加“来源”列，EMVS 明示为“Crossref（S2重复空记录修正）”，其余条目仍为 Semantic Scholar；
  - 机器结果新增 `raw_s2_citations` 和 `citation_correction_reason`，避免覆盖原始证据。
- 原因/证据：同一 DOI、题名和期刊记录在 Crossref 返回243引；第三方 DOI 索引也显示243，证明 Semantic Scholar 的0引来自记录拆分，而不是论文真实低引。
- 验证：复核所有“2024及以前且不超过5引”的条目；PPLNs、ESHDR、Ev3DGS未发现同类重复空记录，EMVS是唯一需要修正者。
- 待用户处理：无；该修正只影响筛查排序，不修改论文参考文献。

### 2026-07-24-09：删除Revisit-EBVFI与Ev3DGS

- 执行者：Codex
- 范围：主项目重建 catalog、3DGS 正文概述和 `main.bib`；外层筛查数据、报告、脚本、项目状态及本修改历史。未编译，未执行 Git。
- 修改前：
  - reconstruction catalog 共117项：第29项为 `Revisit-EBVFI / IROS'23`，第103项为 `Ev3DGS / APSIPA'24`。
  - 3DGS 正文写为：`E2GS and Ev3DGS enhanced 3DGS with event data for improved quality.`
  - `main.bib` 有572条、Tex引用388个唯一 key，包含以下完整条目：

```bibtex
@inproceedings{chen2023revisiting,
    title   = {Revisiting Event-based Video Frame Interpolation},
    author  = {Jiaben Chen and Yichen Zhu and Dongze Lian and Jiaqi Yang and Yifu Wang and Renrui Zhang and Xinhang Liu and Shenhan Qian and Laurent Kneip and Shenghua Gao},
    booktitle = {IROS},
    year    = {2023}
}

@inproceedings{huang2024ev3dgs,
  title={{Ev3DGS}: Event Enhanced {3D} Gaussian Splatting from Blurry Images},
  author={Huang, Junwu and Wan, Zhexiong and Lu, Zhicheng and Zhu, Juanjuan and He, Mingyi and Dai, Yuchao},
  booktitle={2024 Asia Pacific Signal and Information Processing Association Annual Summit and Conference},
  pages={1--6},
  doi={10.1109/APSIPAASC63619.2025.10848695},
  year={2024}
}
```

  - `main.bib` SHA-256：`489865f042a12c067b054ce9160c5e8f93f577fc0a11ba6240058ca4c03761de`。
- 修改后：
  - 两项从 catalog 删除；其余重建条目连续重编号为1--115。
  - 3DGS 正文只保留 `E2GS enhanced 3DGS with event data for improved quality.`，并移除 `\cite{huang2024ev3dgs}`。
  - 上述两个完整 BibTeX 条目均从 `main.bib` 删除；Bib剩570条，Tex引用387个唯一 key。
  - reconstruction inventory/report 改为115项：106项有引文身份、9项未绑定；删除项不再出现在当前候选表。
  - `main.bib` SHA-256：`ab108b050f18ea8aee0c2320d6e36dae6f118533fb62d7eaba8ac59f7aa430df`。
- 原因/证据：
  - `Revisit-EBVFI` 公开约三年后引用较低，位于拥挤的传统CNN/光流VFI路线，综述位置可由 TimeLens、TimeReplayer、CBMNet、REFID 等更具代表性的工作覆盖，且未发现官方代码。
  - `Ev3DGS` 引用较低、未发现官方代码，核心 blur/event rendering loss 与 E2GS、EaDeblur-GS、EvaGaussians、DiET-GS 等高度重合，在拥挤的event-3DGS路线中替代性较强。
  - 用户明确决定先删除这两篇。
- 验证：
  - reconstruction catalog 恰有115行，编号从1到115连续且无重复。
  - `main.bib` 有570个条目和570个唯一 key；Tex有387个唯一引用 key，缺失引用为0。
  - 主项目内已无 `Revisit-EBVFI`、`Ev3DGS`、`chen2023revisiting` 或 `huang2024ev3dgs`。
  - 完整报告有115个编号数据行；已绑定106项均有引文值，9项未绑定不视为零引。
  - `main.tex` SHA-256 仍为 `87385960d87cd63685662b43e610e502f538087847f7aea7847a9c925eca252a`。
- 待用户处理：
  - 在 Overleaf 自行编译并检查 catalog 跨页与参考文献显示效果。
  - 在 Windows 环境自行执行 Git 添加与提交。

### 2026-07-25-10：将EBAD-Gaussian同步为正式EBAD-GS

- 执行者：Codex
- 范围：主项目 reconstruction catalog 和3DGS正文；外层筛查数据、报告、生成源、专项审计、项目状态及本修改历史。`main.bib` 未改，未编译，未执行 Git。
- 修改前：
  - catalog 第111项为 `EBAD-Gaussian / arXiv'25`。
  - 3DGS正文使用旧名称 `EBAD-Gaussian`。
  - 完整报告显示 `2026 / arXiv'25`；纯 arXiv 附录仍把它计入16项候选。
  - `main.bib` 实际已经只有一个正式条目：citation key 为 `deng2025ebadgaussian`，题名为 `EBAD-GS: Deblurring Gaussian Splatting with Event-Driven Bundle Adjustment`，六位正式版作者，ICASSP 2026，页码10902--10906，DOI `10.1109/ICASSP55912.2026.11464704`。不存在第二份独立 arXiv BibTeX。
- 修改后：
  - catalog 第111项改为 `EBAD-GS / ICASSP'26`，citation key 保持不变。
  - 3DGS正文方法名同步为 `EBAD-GS`。
  - 完整报告第111项显示 `2026 / ICASSP'26`，仍使用正式论文的 Semantic Scholar 身份。
  - 纯 arXiv 附录由16项减为15项，EBAD-GS 已从该附录和机器筛选结果移除。
  - 删除 inventory 中仅为旧 catalog 别名服务的 `EBAD-Gaussian` override；当前由显式 `\cite{deng2025ebadgaussian}` 直接绑定正式 Bib。
- 原因/证据：
  - IEEE/DOI正式记录已在2026-07-24专项核验，完整前后 BibTeX 和证据见 `docs/reference_audit/ebad_gs_icassp2026_bib_update.md`。
  - 本次问题是 catalog/正文/附录未同步，不是主 Bib 仍保留 arXiv 版本。
- 验证：
  - inventory 仍为115项；第111项方法名 `EBAD-GS`、venue `ICASSP'26`、Bib 类型 `inproceedings`、DOI与正式记录一致。
  - catalog 明写为 arXiv 的条目由16项变为15项；纯 arXiv 筛查结果恰有15项且不含 EBAD-GS。
  - `main.bib` 仍为570个条目和570个唯一 key；Tex有387个唯一引用 key，缺失引用为0。
  - `main.bib` SHA-256 仍为 `ab108b050f18ea8aee0c2320d6e36dae6f118533fb62d7eaba8ac59f7aa430df`。
- 待用户处理：
  - 在 Overleaf 自行编译，检查方法名和 venue 的最终显示。
  - citation key 为内部稳定标识，不需要随论文正式名称改名。

### 2026-07-25-11：建立论文源文件与人工变更记录的公共仓库同步规则

- 执行者：Codex
- 范围：根目录 `AGENTS.md`、本修改历史，以及独立公共仓库工作目录 `../awesome-event-camera-vision-public/`；未修改主项目 Tex/Bib，未编译，未执行 Git。
- 修改前：
  - `AGENTS.md` 只规定主论文事实来源、人工变更记录和 Git/Overleaf 操作边界，没有要求把可公开论文材料同步到独立公共仓库。
  - 新建公共仓库工作目录已连接 `https://github.com/worldbench/awesome-event-camera-vision.git`，但尚未包含当前 `TPAMI2026_Survey_Event_Camera_Vision/` 和根目录 `CHANGELOG_MANUAL.md`。
  - 公共仓库已有 `.gitignore`、`LICENSE`、`README.md`、`README_old.md` 和 `docs/figures/.keep` 的本地未提交修改。
- 修改后：
  - `AGENTS.md` 新增“公共仓库同步”规则：每次修改主论文 Overleaf/LaTeX 文件或人工变更记录后，必须在同一任务内将完整论文目录和 `CHANGELOG_MANUAL.md` 单向复制到 `../awesome-event-camera-vision-public/`。
  - 同步白名单严格限定为 `TPAMI2026_Survey_Event_Camera_Vision/` 与 `CHANGELOG_MANUAL.md`；明确禁止自动复制 `PROJECT_STATUS.md`、`AGENTS.md`、`docs/`、`reference_materials/`、`bin/`、`tools/` 和 Git 元数据。
  - 公共仓库原有 README、LICENSE、`.gitignore`、`docs/` 及其既有本地修改均保持不变。
  - 当前论文目录和包含本条记录的 `CHANGELOG_MANUAL.md` 已复制到公共仓库工作目录。
- 原因/证据：
  - 用户要求在不影响私人 Git 的前提下，把当前 Overleaf 项目及专用人工变更记录整理并复制到已创建的公共仓库目录，并把后续自动复制要求写入 Agent 规则。
  - 两个工作目录各自拥有独立 `.git`，复制仅传递白名单文件内容，不传递私人仓库历史或其他管理、审计和历史材料。
- 验证：
  - 公共仓库远程 `origin` 指向 `https://github.com/worldbench/awesome-event-camera-vision.git`，当前分支为 `main`。
  - 复制后比较源与目标的相对文件清单和 SHA-256；论文目录全部文件及 `CHANGELOG_MANUAL.md` 内容一致。
  - 公共仓库既有文件未被本次同步改写；Git 状态仅在原有修改之外新增白名单目标。
- 待用户处理：
  - 在 Windows 环境检查公共仓库差异，确认 `CHANGELOG_MANUAL.md` 适合公开，并自行执行 Git 添加、提交和推送。
  - 公共仓库当前已有若干与本次同步无关的本地修改，提交前需由用户决定是否与论文同步内容分开提交。

### 2026-07-25-12：Reconstruction 文献第一轮删减

- 执行者：Codex
- 范围：主项目 `sections/3_method.tex`、`sections/catalog_tables.tex`、`main.bib`；外层 reconstruction 筛查报告、数据、生成源、专项审计、项目状态和本修改历史；未修改 `main.tex`，未执行 Git。
- 修改前：
  - reconstruction catalog 共115项，Bib 共570条，Tex 共引用387个唯一 citation key。
  - 用户候选清单包含多个状态过时或具有强团队/独特资源价值的条目，不能整批机械删除。
  - `E3NeRF`、`SaENeRF`、`BeSplat` 同时存在于 catalog、正文和 Bib：
    - `qi2024e3nerf`：arXiv 2024 的 blurry-image/event NeRF；
    - `zhang2025saenerf`：旧 Bib 错写作者表并仍标 arXiv，正式版实际为 IJCNN 2025；
    - `matta2025besplat`：WACV 2025 Workshop 的单模糊图像+事件 3DGS。
  - catalog 把 `Self-EHDRI` 标为 `arXiv'24`、把 `EvGGS` 标为 `arXiv'24`；`ESHDR` 无显式 citation，且 Bib 的 journal 字段错误写成 `arXiv:2412.19067`。
- 修改后：
  - 本次第一轮新增删除 `E3NeRF`、`SaENeRF`、`BeSplat` 三篇：从 catalog 删除并连续重编号，从正文对应概述删除，从 `main.bib` 删除完整条目。
  - reconstruction catalog 从115项减为112项；累计已删除的 reconstruction 条目为 `Revisit-EBVFI`、`Ev3DGS`、`E3NeRF`、`SaENeRF`、`BeSplat`。
  - Bib 从570条减为567条；Tex 当前引用386个唯一 citation key，缺失引用为0。
  - `Self-EHDRI` 改为 `Self-EHDRI~\cite{li2024hdr} / PR'26`；`EvGGS` 改为 `ICML'24`；`ESHDR` 增加 `\cite{guo2024event}`。
  - `guo2024event` 的 `journal` 从 `arXiv preprint arXiv:2412.19067` 更正为 `arXiv preprint arXiv:2412.14705`；题名、作者和年份不变。
  - 全量报告、纯 arXiv 附录、inventory、机器结果、筛查规则和项目状态同步为112项；旧 SaENeRF 待修报告追加状态更正，不删除历史诊断。
  - 完整删除前 Bib 条目、逐篇强团队/独特性判断、保留依据和 ESHDR 前后条目见 `docs/reference_audit/reconstruction_pruning_round1_2026-07-25.md`。
- 原因/证据：
  - `E3NeRF` 截至核验日仍未找到正式主会/期刊版本，且位于 E2NeRF、EvDeblurNeRF、Deblur-e-NeRF、EBAD-NeRF 已覆盖的拥挤路线。
  - `SaENeRF` 虽正式发表于 IJCNN 2025且有代码，但贡献主要是 event-NeRF 极性归一化和伪影正则，属于可由现有代表工作覆盖的增量改良。
  - `BeSplat` 是 WACV Workshop 论文，将单模糊图像+事件设置迁移到 3DGS，与现有 blurry-event 3DGS 簇高度重合。
  - 强团队检查后保留：NUS Gim Hee Lee 团队的 `Deblur-e-NeRF`、MPI-INF Christian Theobalt/Vladislav Golyanik 团队的 `DynEventNeRF`、USTC Zhiwei Xiong 团队的 `EventBoosted-3DGS`，以及 Tianfan Xue/Jinwei Gu 等团队的 `Sim2Real-EVFI`/`ESHDR`。
  - 资源/任务检查后保留：`PAEv3d` 的101对象数据集、`Event-ID` 的 intrinsic decomposition、`E-3DGS` 的 exposure-event 硬件/数据、ICML 2024 正式论文 `EvGGS`。
  - `Self-EHDRI` 的 Pattern Recognition 正式记录 DOI 为 `10.1016/j.patcog.2026.114265`；`EvGGS` 的 PMLR/ICML 正式页为 `https://proceedings.mlr.press/v235/wang24w.html`；ESHDR 的 arXiv 官方身份为 `2412.14705`；EBAD-GS 已是 ICASSP 2026 正式版，本轮保留且不存在第二个旧 arXiv Bib 条目。
- 验证：
  - inventory 重建结果为112行，编号1--112连续；其中103项绑定 Bib 并有引文缓存，9项未绑定，catalog 明写 arXiv 的条目为12项。
  - `main.bib` 有567个条目和567个唯一 key；Tex 有386个唯一引用 key，缺失引用为0。
  - 主项目内 `E3NeRF`、`SaENeRF`、`BeSplat` 及三个删除 key 的命中均为0。
  - `main.tex` SHA-256 仍为 `87385960d87cd63685662b43e610e502f538087847f7aea7847a9c925eca252a`；`main.bib` SHA-256 为 `ff858d6e358e33ccf20157d7f11de46cf3dd15697275a43959f83f27ba6244ce`。
  - 当前环境没有 `latexmk`、`pdflatex` 或 `bibtex`，因此未做完整 LaTeX 编译；静态引用、JSON、报告再生成和文件一致性检查均通过。
- 待用户处理：
  - 在 Overleaf 编译 `main.tex`，重点检查 reconstruction catalog 删行后的跨页、行底色和参考文献显示。
  - 在 Windows 环境自行检查 Git 差异并提交；建议把本轮主稿、Bib、审计和变更日志作为同一批提交。

### 2026-07-25-13：第一轮补充删减EventDiff并保留EvDiff

- 执行者：Codex
- 范围：主项目 `sections/3_method.tex`、`sections/catalog_tables.tex`、`main.bib`；外层 reconstruction 报告、数据、生成源、专项审计、项目状态和本修改历史；未修改 `main.tex`，未执行 Git。
- 修改前：
  - `EventDiff` 位于 reconstruction catalog 第35项，正文 Frame Interpolation 段以 `\cite{chen2025eventdiff}` 概述，并在 Bib 中保留以下完整条目：

```bibtex
@article{chen2025eventdiff,
  title={{EventDiff}: A Unified and Efficient Diffusion Model Framework for Event-based Video Frame Interpolation},
  author={Zheng, Hanle and Han, Xujie and Peng, Zegang and Zhang, Shangbin and Du, Guangxun and Zou, Zhuo and Wang, Xilin and Wu, Jibin and Guo, Hao and Deng, Lei},
  journal={arXiv preprint arXiv:2505.08235},
  year={2025}
}
```

  - catalog 共112项，Bib 共567条，Tex 共引用386个唯一 citation key。
  - `EventDiff` 和 `EvDiff` 同为2025年扩散工作；前者仍是纯 arXiv 的 VFI 方法，后者已有 Ming-Hsuan Yang 官方论文列表的 ECCV 2026 接收记录。
- 修改后：
  - 从 catalog、正文和 `main.bib` 删除 `EventDiff` 及 citation key `chen2025eventdiff`；其后 catalog 条目连续重编号。
  - 保留 `EvDiff` 及 `li2025evdiff`，不修改其当前 Bib。
  - catalog 从112项减为111项；Bib 从567条减为566条；Tex 唯一 citation key 从386减为385。
  - 全量报告、纯 arXiv 附录、inventory、机器结果、筛查规则和项目状态同步为111项；`EventDiff` 从“中优先级复核”改为“第一轮已删除”。
- 原因/证据：
  - 用户明确决定二选一时删除 `EventDiff`。
  - `EventDiff` 截至核验日仍为 arXiv:2505.08235，未找到代码或数据，且 diffusion VFI 已有 REVDM、EGVD、EPA 等代表工作。
  - `EvDiff` 作者包括 Ming-Hsuan Yang、Luc Van Gool、Danda Pani Paudel；Ming-Hsuan Yang 官方论文列表已将其列为 ECCV 2026，且一阶段扩散和 surrogate training 在生成先验演进中更具代表性。
  - 完整比较和删除前 Bib 条目已追加到 `docs/reference_audit/reconstruction_pruning_round1_2026-07-25.md`。
- 验证：
  - inventory 为111行，编号1--111连续；102项绑定 Bib，9项未绑定，catalog 明写 arXiv 的条目为11项。
  - `main.bib` 有566个条目和566个唯一 key；Tex 有385个唯一引用 key，缺失引用为0。
  - 主项目内 `EventDiff` 和 `chen2025eventdiff` 命中为0；`EvDiff` 和 `li2025evdiff` 仍存在。
  - `main.tex` SHA-256 仍为 `87385960d87cd63685662b43e610e502f538087847f7aea7847a9c925eca252a`；`main.bib` SHA-256 为 `2da1ff9845c8bd1834fc63476fd8cfa74af89419ffda060e80c0019c3bb3a735`。
  - 当前环境没有 LaTeX 工具链，未做完整编译；静态引用、JSON、报告再生成和连续编号检查通过。
- 待用户处理：
  - 在 Overleaf 编译并检查 catalog 重编号后的行底色、分页和参考文献显示。
  - 在 Windows 环境自行执行 Git 添加、提交和推送。

### 2026-07-25-14：第一轮补充删除E-3DGS

- 执行者：Codex
- 范围：主项目 reconstruction 正文、catalog、`main.bib`；外层报告、机器数据、生成源、专项审计、项目状态和本修改历史；未执行 Git。
- 修改前：
  - `E-3DGS: 3D Gaussian Splatting with Exposure and Motion Events` 位于 catalog 第102项，正文 Event-Enhanced 3DGS 段引用 `yin2025e3dgs`。
  - Bib 条目为 Applied Optics 64(14):3897，DOI `10.1364/AO.557565`，7位作者，2025年。
  - 旧筛查结论因 exposure events、硬件、EME-3D 数据和标注代码而保护该论文。
  - catalog 111项，Bib 566条，Tex 385个唯一 citation key。
- 修改后：
  - 从正文、catalog 和 `main.bib` 删除该 E-3DGS 及 `yin2025e3dgs`；不影响另一篇 3DV 2025 的 `E-3DGS-LargeScale`。
  - 后续 catalog 连续重编号；catalog 110项，Bib 565条，Tex 384个唯一 citation key。
  - 当前报告把 E-3DGS 从“较低可见度venue但保护”撤回为“第一轮已删除”；所有仍作为当前依据的状态、报告和生成源同步更新。
- 原因/证据：
  - 用户实际核查后确认论文标注的 GitHub 长期未更新，属于假开源，并明确要求删除。
  - 在拥挤的 event-3DGS 路线中，低引用与不可用复现资产共同削弱其保留价值。
  - 完整删除前 Bib 和证据链接见 `docs/reference_audit/reconstruction_pruning_round1_2026-07-25.md`。
- 验证：
  - inventory 为110行，编号1--110连续；101项绑定 Bib，9项未绑定。
  - `main.bib` 有565个条目和565个唯一 key；Tex 有384个唯一引用 key，缺失引用为0。
  - `yin2025e3dgs` 和目标完整题名在主项目中命中为0；`E-3DGS-LargeScale`、`zahid2025e3dgslarge` 仍保留。
  - `main.bib` SHA-256 为 `9e9e285544ebe26bc33ad704046e57b3c1130b827ec4a0f134f69c7a6b58539d`。
  - 当前环境没有 LaTeX 工具链，未做完整编译；静态引用、JSON、报告再生成和编号检查通过。
- 待用户处理：
  - 在 Overleaf 编译并检查 catalog 行底色、分页和参考文献。
  - 在 Windows 环境自行执行 Git 添加、提交和推送。

### 2026-07-25-15：第一轮第二批删除九篇重建论文

- 执行者：Codex
- 范围：主项目 `sections/3_method.tex`、`sections/catalog_tables.tex`、`main.bib`；外层当前报告、机器清单、报告生成源、第一轮专项审计、项目状态和本修改历史；未修改 `main.tex`，未执行 Git。
- 修改前：
  - reconstruction catalog 为110项；Bib 为565条；Tex 为384个唯一 citation key。
  - `STLR`、`PPLNs`、`E2VIDiff`、`DESSERT`、`EvINR`、`CrossZoom`、`ESHDR`、`Event-ID`、`DEGS` 均在 catalog 和 Bib 中。
  - 正文还分别概述 `E2VIDiff`、`EvINR`、`DEGS`，方法演进图含 `E2VIDiff` 节点；当前报告仍把 ESHDR、E2VIDiff、Event-ID 等列为保护或建议保留对象。
- 修改后：
  - 删除上述九项的 catalog 行及完整 Bib 条目；删除正文中的三处对应概述和方法演进图中的 `E2VIDiff` 节点/连线。
  - catalog 后续条目连续重编号，从110项减为101项；Bib 从565条减为556条；Tex 唯一 citation key 从384减为378。
  - 重新生成 inventory、全量报告、纯 arXiv 附录及机器结果；撤回旧的 ESHDR、E2VIDiff、Event-ID 保护结论。
  - 删除前身份、理由和证据集中记录在 `docs/reference_audit/reconstruction_pruning_round1_2026-07-25.md`，历史结论保留并追加撤回说明。
- 修改原因与证据：
  - 用户明确要求删除全部九项；其中 ESHDR 的官方仓库长期只预告资产，被用户认定为假开源；Event-ID 的 GitHub 关注度为0；DEGS 未开源。
  - 其余项目依据用户给出的低引用、纯 arXiv 状态、技术位置可替代性进行本轮压缩。
  - `CrossZoom` 虽已有 TPAMI 正式版本、`STLR`/`EvINR` 虽为 ECCV、`PPLNs` 虽为 NeurIPS、`Event-ID` 虽为 ACM MM、`DEGS` 虽为 TVCG，仍按用户明确决定删除；未把 venue 状态误记为 arXiv。
- 验证：
  - reconstruction inventory 为101行，编号1--101连续；92项绑定 Bib，9项未绑定；catalog 明写 arXiv 的条目为7项。
  - `main.bib` 有556个条目和556个唯一 key；Tex 有378个唯一引用 key，缺失引用为0。
  - 九个删除 key 和九个方法名在主项目 `.tex`/`.bib` 中命中均为0。
  - `main.bib` SHA-256 为 `e90682555710238861e98424b709a260e53bbe9098ee9e2c2a56ec0f36797d3d`。
  - 当前环境没有 `latexmk` 或 `pdflatex`，因此未做完整 LaTeX 编译；静态引用、连续编号、报告再生成和机器清单检查均通过。
- 待用户处理：
  - 在 Overleaf 编译 `main.tex`，重点检查 catalog 删除九行后的行底色、分页及方法演进图布局。
  - 在 Windows 环境自行检查 Git 差异并提交、推送。

### 2026-07-25-16：修复REPORT按旧编号误分重建子类

- 执行者：Codex
- 范围：外层 `tools/reconstruction_screening/build_inventory.py`、`generate_full_report.py`、重建 inventory/全量结果、`docs/reconstruction_screening/REPORT.md`、`PROJECT_STATUS.md` 和本修改历史；未修改主项目论文文件，未执行 Git。
- 修改前：
  - 报告生成器用固定 catalog 编号区间推断类别，例如74--92强制归入 NeRF、93--110强制归入3DGS。
  - 删除论文并连续重编号后，当前3DGS实际范围已变为85--101，导致 EvGGS 至 IncEventGS 共8项被误列进 NeRF。
  - `REPORT.md` 因而错误显示 NeRF 19项、3DGS 9项；NeurIPS 2024 `Event-3DGS` 虽未删除，却出现在 NeRF 表中。
- 修改后：
  - inventory 直接解析 catalog 每行的 `\circnum{}`，保存 `catalog_category_no` 和 `catalog_category`。
  - 全量报告生成器读取该显式类别，不再依据会随删减变化的编号区间。
  - 重新生成后 NeRF 为18项、3DGS为17项；`Event-3DGS`、Event3DGS、EvGGS 等全部回到3DGS表。
- 修改原因与证据：
  - `sections/catalog_tables.tex` 明确用 `\circnum{11}` 标识 NeRF、`\circnum{12}` 标识3DGS，这是比连续序号稳定的当前事实来源。
  - 用户主要依据 `REPORT.md` 决定删减，因此错误分组会直接影响人工判断，必须修复生成源而非只手改报告。
- 验证：
  - 重建 inventory 共101项，12个子类合计101项，无 `unknown` 类别。
  - 3DGS 表完整列出编号85--101的17项；NeurIPS 2024 `Event-3DGS` 为第91项、当前引用30，位于3DGS表。
  - NeRF 表为编号67--84的18项；报告重建成功，未访问网络或刷新引用缓存。
- 待用户处理：
  - 后续以修复后的 `docs/reconstruction_screening/REPORT.md` 继续进行人工删减。

### 2026-07-25-17：为全量REPORT增加代码、stars和数据集审计

- 执行者：Codex
- 范围：外层 `tools/reconstruction_screening/audit_open_source.py`、`generate_full_report.py`、GitHub 审计缓存、机器可读结果、`docs/reconstruction_screening/REPORT.md`、`PROJECT_STATUS.md` 和本修改历史；未修改主项目论文文件，未执行 Git。
- 修改前：
  - `REPORT.md` 只有 venue、引用量、引用/年和引文初筛提示，无法直接判断论文是否真正开放代码/数据。
  - 旧人工材料只覆盖少数论文，stars 没有统一快照；“有 GitHub 链接”与“仓库实际含代码”未区分。
- 修改后：
  - 新增可重复运行的 `audit_open_source.py`：用少量宽泛 GitHub 搜索加未匹配项精确搜索发现候选，所有查询、仓库元数据、文件树和 README 结果均缓存，默认重跑不联网，显式刷新才重新请求。
  - 对仓库递归文件树检测实质代码文件；README 写有 coming soon/TODO 且缺少代码时标为“占位/待发布”；空仓库标为“仓库无代码”；历史链接404标为“仓库失效”。
  - 记录 GitHub stars 快照，并区分数据下载入口、仓库内数据/脚本、承诺未发布和本轮未检出。
  - `REPORT.md` 每个子类总表新增“代码/仓库”“GitHub ★”“数据集”三列，并在报告末尾加入保守解释和状态图例。
  - 为避免同名污染，完整题名/arXiv 身份或既有人工证据才视为高置信；仅仓库名相同的结果显示“同名候选”，不直接认定为论文官方开源。
- 修改原因与证据：
  - 用户明确把真实开源、开放数据集、假开源和 GitHub stars 作为下一轮删减的重要依据。
  - GitHub API 提供仓库 stars、默认分支和递归文件树；公开 README 用于判断代码/数据是否仍是 coming soon/TODO。
  - 初轮宽松匹配曾把多个 NeRF 方法指向同一个 `enerf` 仓库，并把 EvDiff 指向 EVDI；最终版提高阈值并重新生成，撤回这些误匹配。
- 验证：
  - 当前 catalog 101项均有一行开源审计结果；检出37个 GitHub 仓库，其中30个为人工证据或完整题名/身份高置信匹配，7个为明确标注的同名候选。
  - 检测到32项仓库含至少3个实质代码文件，3项为占位或无代码风险，15项 README 含明确数据下载入口。
  - CoRL 2024 `Event3DGS` 未检出高置信仓库；NeurIPS 2024 `Event-3DGS` 独立匹配 `lanpokn/Event-3DGS`，未再混用同一身份。
  - 报告仍保留“本轮未检出不等于绝对未开源”的限制；stars 是2026-07-25 GitHub API 快照，会随时间变化。
- 待用户处理：
  - 下一轮删减时优先复核“占位/待发布”“仓库无代码”“仓库失效”和低-star项；点击报告中的仓库链接确认作者身份及最新状态。
