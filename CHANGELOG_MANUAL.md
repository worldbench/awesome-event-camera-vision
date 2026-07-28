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

### 2026-07-25-18：开源仓库漏检二次检查

- 执行者：Codex
- 范围：外层 `tools/reconstruction_screening/audit_open_source.py`、`generate_full_report.py`、GitHub/来源页面缓存、机器可读开源结果、`docs/reconstruction_screening/REPORT.md`、`PROJECT_STATUS.md` 和本修改历史；未修改主项目论文文件，未执行 Git。
- 修改前：
  - 第一轮依靠宽泛 GitHub 搜索、方法题名精确搜索和已有人工证据，共检出37个仓库。
  - 其余64项标为“本轮未检出”，但尚未系统从论文身份页面反向提取链接，也未统一使用 arXiv ID或作者身份做第二查询。
- 修改后：
  - 新增 `--second-pass` 增量模式：只处理仍未匹配项，优先从 arXiv、Bib URL、DOI/官方页面提取 GitHub 仓库；仍无结果时执行一次 arXiv ID或“方法名+第一作者”GitHub身份查询。
  - Semantic Scholar 缓存中的 arXiv ID作为补充身份，不需要重新调用论文数据库；页面、搜索结果和仓库核验均逐条缓存。
  - `REPORT.md` 结论区新增二次检查覆盖量和新增仓库数，便于判断“未检出”的核验深度。
- 修改原因与证据：
  - 用户要求降低 GitHub 漏检率，同时控制提示词消耗；本轮完全由本地脚本和缓存完成，没有把逐篇网页内容送入模型。
  - 直接页面链接和 arXiv ID比单纯方法名搜索更能发现仓库名与论文名不一致的情况；作者身份查询用于补充无 arXiv ID的正式论文。
- 验证：
  - 反向检查54个 arXiv/官方/DOI 页面，其中53个成功读取、1个返回HTTP 403。
  - 对仍未匹配项执行64个缓存化身份查询；新增通过严格身份阈值的仓库为0个。
  - 最终仍为37个检出仓库：30个高置信、7个明确标注的同名候选；32项含实质代码，3项为占位或无代码风险，15项有明确数据下载入口。
  - 二次检查没有为了提高覆盖数字而接受低阈值结果；“本轮未检出”仍不等同于绝对不存在 GitHub。
- 待用户处理：
  - 下一轮删减可把二次检查后的“本轮未检出”作为较强的负面资源信号，但不应单独作为删除理由。

### 2026-07-26-01：第二轮删除EVDI++、CMTA和EventBoosted-3DGS

- 执行者：Codex
- 范围：主项目 `sections/3_method.tex`、`sections/catalog_tables.tex`、`sections/appendix.tex`、`main.bib`；外层当前报告、开源审计结果、报告生成源、专项删减审计、项目状态和本修改历史；未修改 `main.tex`，未执行 Git。
- 修改前：
  - reconstruction catalog 为101项；Bib 为556条；Tex 为378个唯一 citation key。
  - `EVDI++` 位于 catalog 第19项，正文以 `zhang2025evdiplus` 描述其迭代改进；当前正式身份为 TPAMI 2026、DOI `10.1109/TPAMI.2026.3697759`，报告快照为0引且二次开源审计未检出代码/数据。
  - `CMTA` 位于 catalog 第35项，Bib key 为 `kim2024cmta`；ECCV 2024、21引，官方 `intelpro/CMTA` 仓库17 stars且有数据入口，但文件树未检测到实质代码，README 状态为占位/待发布。
  - `EventBoosted-3DGS` 位于 catalog 第94项，正文和附录均引用 `xu2025eventboosted3dgs`；ICCV 2025、2引，二次审计未检出高置信代码/数据仓库。
- 修改后：
  - 从 catalog 和 `main.bib` 删除上述三项；删除 EVDI++、EventBoosted-3DGS 的正文概述及 EventBoosted 的附录比较行。
  - catalog 后续条目连续重编号，从101项减为98项；Bib 从556条减为553条；Tex 唯一 citation key 从378减为376。
  - 重新生成 inventory、全量报告、纯 arXiv 附录和开源审计结果；3DGS 从17项减为16项。
  - 当前人工证据撤回对 CMTA、EventBoosted-3DGS 的保护结论；完整删除身份和理由追加到 `docs/reference_audit/reconstruction_pruning_round1_2026-07-25.md`。
- 修改原因与证据：
  - 用户明确决定删除三项。EVDI++ 因当前引用为0、未检出代码且不承担大模型演进位置而降权删除。
  - CMTA 因官方仓库长期仍无实质代码，被用户按假开源风险删除；已开放数据不抵消核心代码未发布问题。
  - EventBoosted-3DGS 虽为 ICCV 2025，但当前仅2引、未检出高置信代码/数据，用户判断其在现有动态3DGS路线中重要性不足。
- 验证：
  - reconstruction inventory 为98行，编号1--98连续；89项绑定 Bib，9项未绑定。
  - `main.bib` 有553个条目和553个唯一 key；Tex 有376个唯一 citation key，缺失引用为0。
  - 三个删除方法名和三个 citation key 在主项目 `.tex`/`.bib` 中命中均为0。
  - `main.bib` SHA-256 为 `b93a70494c4d8f51b5de15070ef136242f0ec7cdd2db8d6ccf7dfcb8d29de979`。
  - 当前环境未执行完整 LaTeX 编译；静态引用、catalog 连续编号、报告重建及机器清单检查通过。
- 待用户处理：
  - 在 Overleaf 编译 `main.tex`，重点检查 catalog 重编号、正文段落和附录删行后的排版。
  - 在 Windows 环境自行检查 Git 差异并提交、推送。

### 2026-07-27-01：在REPORT记录下一轮建议删减20篇

- 执行者：Codex
- 范围：外层 `tools/reconstruction_screening/generate_full_report.py`、`docs/reconstruction_screening/REPORT.md`、`PROJECT_STATUS.md` 和本修改历史；未修改主项目论文文件，未执行 Git。
- 修改前：
  - `REPORT.md` 有当前98篇的引用、开源和数据审计，但没有集中记录“若再删20篇”的建议名单。
  - 用户需要结合论文总主旨而非现有临时分类，逐篇查看建议及其取舍依据。
- 修改后：
  - 新增“下一轮建议删减20篇（尚未执行）”章节，明确围绕 “Event Camera Vision in the Era of Large Models” 压缩传统任务增量。
  - 名单分为优先删除12篇和为达到20篇再压缩8篇；每篇动态展示当前venue、引用及仓库状态，并给出叙事层面的删除理由。
  - 新增明确保护名单，覆盖大模型/新范式、历史任务锚点及NeRF到3DGS的表示转折。
  - 建议写入报告生成源，后续重建不会丢失；Elite-EvGS 在建议表中优先显示已核实的正式 Bib venue，而不是过时catalog中的arXiv标签；同名候选仓库不再显示为已确认官方开源。
- 修改原因与证据：
  - 用户要求把建议写入 `REPORT.md` 便于继续人工决策。
  - 选择综合论文中心主旨、范式代表性、正式发表层级、引用、真实代码/数据及同类重叠度；不按现有子类机械分配删除名额。
- 验证：
  - 报告建议章节包含12+8共20篇，无重复方法；所有方法仍存在于当前98项inventory。
  - 章节明确标注“尚未执行”；主项目 `.tex`、`.bib` 均未变化。
- 待用户处理：
  - 在 `REPORT.md` 审阅20篇建议并决定实际删除名单；未获明确决定前不执行删减。

### 2026-07-27-02：停止常规哈希计算与记录

- 执行者：Codex
- 范围：根目录 `AGENTS.md`、`PROJECT_STATUS.md`、`tools/reconstruction_screening/audit_open_source.py` 和本修改历史；未修改主项目论文文件，未执行 Git。
- 修改前：
  - 公共仓库同步规则要求比较文件清单及内容校验值，容易被执行为每次逐文件计算并记录 SHA-256。
  - `PROJECT_STATUS.md` 长期保存 `main.tex` 和 `main.bib` 的基线 SHA-256。
  - 开源审计脚本每次运行都会计算并输出缓存 JSON 自身的 SHA-256。
- 修改后：
  - 公共同步默认只比较相对文件清单和实际内容，同一任务仅验收一次；仅在无法直接比较、怀疑传输损坏或用户明确要求时使用哈希。
  - 明确禁止把常规哈希写入变更日志、项目状态或报告，也不再为未修改文件维护基线哈希。
  - 删除项目状态中的两项基线 SHA-256，以及审计脚本无实际消费方的 `cache_sha256` 输出和 `hashlib` 依赖。
- 修改原因与证据：
  - 用户更新全局哈希规则并明确要求“能不用就不用”；当前项目的重复 SHA-256 记录没有帮助论文引用核验、编译验证或人工回溯。
  - 本地文件与公共副本可直接逐字节比较，无需先转换为摘要；论文正确性应由引用解析、条目检查和编译等针对性验证判定。
- 验证：
  - 审计脚本通过 Python 语法检查，且代码中不再存在 `hashlib` 或 `cache_sha256`。
  - 当前规则中仅保留哈希的例外使用条件；历史日志中的旧摘要未改写。
  - 公共仓库同步采用相对文件清单和直接内容比较，不生成或记录哈希。
- 待用户处理：
  - 无；后续任务默认不再计算或记录哈希。

### 2026-07-27-03：重构下一轮20篇删减建议的论文与GitHub信息展示

- 执行者：Codex
- 范围：外层 `tools/reconstruction_screening/generate_full_report.py`、`docs/reconstruction_screening/REPORT.md` 和本修改历史；未修改主项目论文文件，未执行 Git。
- 修改前：
  - 两张建议表只显示方法简称、“venue；引用；仓库状态”的压缩文本和删除理由，缺少论文全名、仓库直达链接、stars与数据开放状态。
  - “二次审计未检出仓库”重复出现在每行，无法快速看出20篇候选的整体开源结论，也不易区分未检出和同名候选。
- 修改后：
  - 表格逐篇显示可点击方法名、完整论文题名、发表信息、引用、GitHub审计结论、stars、数据情况及删除理由。
  - 使用✅、⚠️、❌区分已确认官方资产、候选/占位风险和二次检索未发现官方仓库；有仓库时状态文字直接链接GitHub。
  - 在两表之前集中说明：20篇中19篇未发现官方仓库；SuperFast仅有一个26-star、含代码和数据链接但作者身份未确认的同名候选；当前没有一篇被确认为官方仓库且含实质代码。
- 修改原因与证据：
  - 用户需要快速了解每篇论文名字和真实GitHub情况，原有“当前证据快照”列信息过度压缩。
  - 展示内容直接来自当前 inventory、Bib元数据及二次开源审计结果，没有把“未检出”扩大解释为绝对未开源。
- 验证：
  - 报告生成脚本成功重建 `REPORT.md`；两组仍为12篇和8篇且没有重复。
  - 20行均显示完整题名、GitHub状态、stars和数据状态；唯一仓库链接为明确标注“未确认官方”的SuperFast同名候选。
  - 生成脚本通过Python语法检查；未计算或记录文件哈希。
- 待用户处理：
  - 根据新版表格复核删除建议；尚未执行这20篇的实际删减。

### 2026-07-27-04：逐篇重查20篇删减建议并撤回错误结论

- 执行者：Codex
- 范围：外层 `tools/reconstruction_screening/audit_open_source.py`、`generate_full_report.py`、`docs/reconstruction_screening/full_review_evidence.json`、开源审计缓存与机器可读结果、`docs/reconstruction_screening/REPORT.md` 和本修改历史；未修改主项目论文文件，未执行 Git。
- 修改前：
  - 原建议把12篇列为优先删除、8篇列为凑足20篇的压缩项，技术理由主要依据方法简称、引用和旧开源审计，没有逐篇重新阅读论文。
  - 旧版错误声称20篇均未确认官方实质代码；SuperFast被标为未确认官方的同名候选。
  - TimeTracker因“不属于扩散、基础模型或其他关键范式转折”被优先删除，未说明其连续点轨迹范式和非线性运动数据；Elite-EvGS、EvLight++与预训练先验/基础模型的直接关系也被遗漏。
- 修改后：
  - 逐篇核对论文摘要/正文、官方 proceedings、作者项目页和GitHub，将原20篇重新判为：11篇撤回删除、7篇降为压缩备选、仅DA-Deblur和EaDeblur-GS两篇仍建议删除。
  - 找回并确认10个官方仓库：E-SAI、EvDeraining、SAN、DA-Deblur、EvDNeRF、LSE-NeRF、SuperFast、eSL-Net++、NeurImg-HDR和EvLight++；报告显示仓库链接、stars、代码完整度和数据状态。
  - SuperFast通过论文作者Siqi Li与仓库README作者表确认是官方仓库；EvLight官方仓库明确写明EvLight++视频代码、模型和数据已经发布。
  - 将EvDNeRF、SuperFast和eSL-Net++标为“有限代码”：前者缺论文声明的数据生成组件，后两者有实质推理代码、模型和数据但未提供训练流程；它们不是空仓库式假开源，也不再冒充完整开源。
  - 明确撤回TimeTracker旧理由：它不使用扩散/基础模型这一点属实，但连续点轨迹VFI属于方法范式变化；Elite-EvGS蒸馏现成E2V先验，EvLight++使用基础模型生成分割/深度伪标签，均应保留在大模型时代叙事中。
- 修改原因与证据：
  - 用户要求通过逐篇搜索替代其人工阅读，并检查技术删除理由是否真实。
  - 主要权威证据来自CVF/ECVA/AAAI论文页、arXiv论文作者声明、作者项目页和论文作者的GitHub仓库；仓库文件树及README用于区分完整代码、有限代码和占位仓库。
  - 代表性纠错证据包括 `dvs-whu/E-SAI`、`booker-max/Unsupervised-Deraining-with-Event-Camera`、`XiangZ-0/GEM`、`anish-bhattacharya/EvDNeRF`、`ubc-vision/LSENeRF`、`lisiqi19971013/SuperFast`、`EthanLiang99/EvLight` 等官方仓库。
- 验证：
  - 开源审计从37个旧匹配更新为45个仓库；当前全catalog中38项完整代码、3项有限代码、2项占位/无代码风险、23项有明确数据入口。
  - 复核表包含20篇且无重复：11个绿色撤回、7个黄色备选、2个红色仍删；10篇显示已确认官方仓库，另外10篇仍未检出官方仓库。
  - 两个脚本通过Python语法检查，人工证据JSON格式有效，报告由生成脚本成功重建；未计算或记录文件哈希。
- 待用户处理：
  - 不要直接执行旧20篇名单；如仍需再删20篇，应在当前98篇全表中另选至少18篇替代候选。

### 2026-07-27-05：从40篇候选复核形成新的20篇删减建议

- 执行者：Codex
- 范围：外层 `docs/reconstruction_screening/` 的40篇复核证据、开源审计结果与 `REPORT.md`，以及 `tools/reconstruction_screening/generate_full_report.py`；未修改主项目论文 `.tex`/`.bib`，未执行实际删文或Git操作。
- 修改前：
  - 报告只保留上一版20篇错误建议的纠错结果：11篇撤回、7篇备选、2篇仍建议删除，无法满足“仍需再删20篇”的人工决策需求。
  - 旧开源表仍漏掉TimeLens-XL、EGDeblurring、STIR、Robust-e-NeRF和BeNeRF的官方GitHub，容易再次用错误的“未检出代码”作为删除依据。
  - Sim2Real-EVFI与Ev-GS仍显示旧的保护标签，没有体现强制继续压缩时的相对取舍。
- 修改后：
  - 新建 `pruning_review_40.json`，从当前98篇中列出40篇候选；每篇记录真实技术贡献、同簇替代关系、权威论文链接与最终结论。
  - 40篇无重复并严格分为20篇“建议删除”和20篇“复核后保留”。建议删除为：SPADE-E2VID、EvDeraining、REFID、SuperFast、EVFI-DS、DA-Deblur、E-CIR、eSL-Net++、HDRev-Net、EvLowLight、Sim2Real-EVFI、Ev-NeRF、DE-NeRF、EBAD-NeRF、AE-NeRF、Event3DGS、EaDeblur-GS、Ev-GS、SweepEvGS、EBAD-GS。
  - 报告明确把EvDeraining、E-CIR、HDRev-Net、EvLowLight、Sim2Real-EVFI、Ev-NeRF、AE-NeRF和Event3DGS标为8篇边界项：它们不是差论文，只是在必须继续压缩时相对可替代。
  - 复核后保留的20篇为：E2VID+、EVSNN、HyperE2VID、CBMNet、TimeLens-XL、SAN、EGDeblurring、ClearSight、UniINR、STIR、Self-EHDRI、ERetinex、Robust-e-NeRF、EvDNeRF、BeNeRF、EvHDR-NeRF、E2GS、EF-3DGS、EventSplat、Elite-EvGS。
  - 找回并审计5个官方GitHub：TimeLens-XL、EGDeblurring、STIR、Robust-e-NeRF、BeNeRF；另将论文页面直接声明的SPADE-E2VID和REFID仓库提升为官方证据。全量结果现为50个仓库、43项实质代码、2项占位/无代码风险、27项明确数据入口。
  - 新表逐篇显示完整题名、真实技术位置、venue/引用、GitHub链接、stars、代码完整度、数据状态和比较后的取舍理由；同名候选不再显示成“官方代码”。
- 修改原因与证据：
  - 用户要求重新选择40篇可能删减项，逐篇搜索后再选20篇，以节省其自行阅读论文的时间。
  - 取舍围绕论文总题目“Event Camera Vision in the Era of Large Models”，优先保护扩散/预训练、SNN/INR/超网络、独特数据与传感器物理，以及NeRF到3DGS和位姿联合优化等范式节点；引用、venue和开源只作为交叉证据。
  - 技术证据来自CVF、ECVA、AAAI、PMLR、OpenReview、出版社DOI/arXiv、作者项目页及论文直接链接的GitHub；未用题名相似或二手概述替代技术核验。
- 验证：
  - 40篇证据JSON格式有效，方法数40、唯一方法数40，结论计数为20+20，全部存在于当前inventory。
  - 两个审计/报告脚本通过语法解析；生成器成功重建 `REPORT.md`，当前inventory仍为98项、9项未绑定主Bib。
  - 新表中的20篇建议仅为人工决策，不曾修改主项目 `.tex`、`main.bib` 或catalog；未计算或记录哈希。
- 待用户处理：
  - 优先审阅报告标出的8篇边界项；确认最终名单后，再执行主稿、Bib、catalog与附录的一致删减。

### 2026-07-27-06：执行第三轮19篇删减并保留Event3DGS

- 执行者：Codex
- 范围：主项目 `sections/3_method.tex`、`sections/5_appl.tex`、`sections/catalog_tables.tex`、`sections/appendix.tex` 和 `main.bib`；外层 `docs/reconstruction_screening/` 当前报告、40篇复核结论、人工证据与机器可读清单，以及 `tools/reconstruction_screening/generate_full_report.py` 和本修改历史；未修改 `main.tex`，未执行 Git。
- 修改前：
  - reconstruction catalog 为98项，`main.bib` 为553个唯一条目，主项目 TeX 使用376个唯一 citation key。
  - 40篇候选复核原拟删除20篇，其中包含 `Event3DGS`；报告仍把该项列为建议删除。
  - 待删论文仍分布于方法正文、应用正文、catalog、附录对照表和 Bib，不能只删单一位置。
- 修改后：
  - 按用户确认删除19篇：SPADE-E2VID、EvDeraining、REFID、SuperFast、EVFI-DS、DA-Deblur、E-CIR、eSL-Net++、HDRev-Net、EvLowLight、Sim2Real-EVFI、Ev-NeRF、DE-NeRF、EBAD-NeRF、AE-NeRF、EaDeblur-GS、Ev-GS、SweepEvGS、EBAD-GS。
  - `Event3DGS: Event-Based 3D Gaussian Splatting for High-Speed Robot Egomotion` 撤回删除并保留；其正文叙述、catalog行和 `xiong2024event3dgs` Bib条目均保持存在。
  - 删除上述19篇在方法正文、应用正文、catalog和附录中的全部引用或表格行，并从 `main.bib` 删除对应完整条目；reconstruction inventory 从98项减为79项，Bib从553条减为534条，TeX唯一 citation key从376减为357。
  - `REPORT.md` 更新为“第三轮已删除19篇、Event3DGS用户决定保留、其余20篇复核后保留”，并让生成脚本能够显示已经不在当前catalog中的历史删减项。
- 修改原因与证据：
  - 用户明确要求执行已复核的删减名单，随后明确撤回对Event3DGS的删除。
  - Event3DGS为CoRL正式论文，报告快照为33引，面向高速机器人自运动重建；当前未检出官方GitHub不足以单独构成删除理由。
  - 其余19篇的技术位置、正式发表身份、引用、开源/数据状态和同簇替代关系已在 `pruning_review_40.json` 逐篇记录，本轮不重新扩大或改写用户确认的删减范围。
- 验证：
  - 重建后的 inventory 为79行；开源状态表同为79行，检出41个仓库、36项实质代码、2项占位或无代码风险、20项数据入口。
  - `main.bib` 有534个条目且key全部唯一；主项目共有357个唯一 citation key，Bib缺失引用为0。
  - 19个删除 citation key在主项目 `.tex`/`.bib` 中均为0命中；`xiong2024event3dgs` 仍在正文、catalog和Bib中命中。
  - 40篇复核JSON格式有效，报告结论计数为19篇已删除、1篇用户决定保留、20篇复核后保留；报告生成脚本语法检查通过。
  - 本机未安装 `latexmk`、`pdflatex`、`xelatex`、`lualatex` 或 `tectonic`，因此未执行本地LaTeX编译；未计算或记录文件哈希。
- 待用户处理：
  - 在Overleaf以 `main.tex` 为入口编译，重点检查catalog删行后的表格分页、附录对照表和参考文献输出。
  - 在Windows环境检查私人工作区和公共发布仓库的差异后，自行执行Git提交与推送。

### 2026-07-28-01：建立Reconstruction分类重构的分阶段执行方案

- 执行者：Codex
- 范围：外层私有文档 `docs/reconstruction_taxonomy/PLAN.md`、`docs/README.md`、
  `PROJECT_STATUS.md` 和本修改历史；未修改主项目 `.tex`、`.bib`、模板或附件，
  未开始逐篇重新分类，未执行Git。
- 修改前：
  - reconstruction重构只有讨论中的分类草案，没有可跨会话续做的阶段计划、逐篇台账字段、
    Foundation-Prior严格判据、人工复核入口或LaTeX落地顺序。
  - `PROJECT_STATUS.md`仍保留删减前的565条Bib、384个引用及98项catalog等过时基线，
    近期优先级仍以继续删减为主。
  - 讨论曾使用`Event-Domain`和`Task-Specific`等容易产生歧义的名称，也没有固定
    “外部基础模型必须实际影响核心重建结果”的准入条件。
- 修改后：
  - 新建专用方案文档，暂定一级分类为`Dedicated Reconstruction Paradigms`
    和`Foundation-Prior Reconstruction`；明确前者仍需在正式落稿前优化名称。
  - 2D方案先区分Event-only与Event-guided RGB，后者以VFI、Deblur、SR、
    Low-Light、HDR和Unified等具体任务标注；3D方案区分Event-only与
    Event-assisted multimodal，并另记Geometry/NeRF/3DGS表示及位姿、动态和泛化属性。
  - 规定Foundation-Prior必须核实基础模型名称、预训练范围、权重使用方式和核心作用；
    从头训练的Diffusion、普通任务预训练、README宣传词及仅用于评价/附带标注的模型
    不能自动进入。
  - 固定六阶段流程：冻结当前范围、脚本候选扫描、逐篇权威核验、人工复核、
    先处理Bib身份、再按依赖顺序修改LaTeX并验证；规划结构化台账和争议项review queue，
    以便长任务或compact后继续。
  - 项目状态更新为当前534条唯一Bib、357个唯一citation key和79项reconstruction catalog。
- 修改原因与证据：
  - 用户要求先规划再执行，预计任务较长且可能经历compact；因此必须让分类判据、证据和
    用户决定落到文件中，而不是依赖聊天上下文。
  - 用户确认保留`Foundation-Prior Reconstruction`名称，并允许
    `Dedicated Reconstruction Paradigms`暂用但继续寻找更准确名称。
  - 当前主项目及最新筛查inventory静态统计分别支持534条Bib、357个唯一引用和79项catalog。
- 验证：
  - 方案文档包含目标、分类草案、Foundation-Prior准入/排除条件、台账字段、六阶段顺序、
    四个人工验收门槛、LaTeX修改依赖和中断续做规则。
  - `docs/README.md`已加入方案入口；`PROJECT_STATUS.md`的当前基线和近期优先级已同步。
  - 本轮没有修改主论文文件，不需要新增引用或Bib检查；未计算或记录哈希。
- 待用户处理：
  - 审阅并批准方案文档第7节的四项框架决定；未批准前不开始Phase 0或逐篇分类。
  - 后续正式落稿前，为`Dedicated Reconstruction Paradigms`确定最终名称。

### 2026-07-28-02：执行Reconstruction分类重构并同步正文、表格与Bib

- 执行者：Codex
- 范围：
  - 主项目 `sections/catalog_tables.tex`、`sections/3_method.tex`、`sections/1_intro.tex`、
    `sections/0_abstract.tex`、`sections/6_conclusion.tex`、`sections/appendix.tex` 和
    `main.bib`；
  - 外层 `docs/reconstruction_taxonomy/`、`docs/reference_audit/`、`docs/README.md`、
    `PROJECT_STATUS.md`、`tools/reconstruction_taxonomy/` 及本修改历史；
  - 未修改模板、样式、图片、`bin/` 历史材料或Git历史。
- 修改前：
  - reconstruction catalog为79项单表，一级分类混合E2V训练方式、VFI/去模糊任务、
    低光/HDR/SR任务以及NeRF/3DGS表示，不能清楚回答输入条件或大规模预训练知识是否
    进入核心重建。
  - 正文继续沿用E2V生成方式、VFI/deblur、enhancement和3D表示的旧组织，容易把
    task-trained Diffusion、NeRF或3DGS本身误写成“大模型时代”的证据。
  - 正文中若干方法未进入大表；旧表还包含无法核实的`EventMM`、3个event-to-event
    超分方法，以及把`EIF-BiOFNet`误作独立论文的重复行。
  - `main.bib`中Robust-e-NeRF和E-3DGS各有重复身份；E2EGS、EGVD和DeblurSplat仍为
    预印本元数据；Fourier low-light与Semantic-E2VID只有非正式标签或占位元数据。
- 修改后：
  - reconstruction采用两条一级路线：暂名`Dedicated Reconstruction Paradigms`和
    `Foundation-Prior Reconstruction`。Foundation-Prior只接纳外部大规模预训练模型
    通过适配、生成先验、几何初始化、蒸馏或核心监督直接影响重建的方法。
  - Dedicated按可观测输入分为2D Event-Only（13篇）、2D Event-Guided RGB（30篇）、
    3D Event-Only（22篇）及3D Event-Assisted Multimodal（12篇）；VFI、Deblur、
    SR、Low-Light、HDR、Unified及Geometry/NeRF/3DGS保留为直观任务或表示属性。
  - Foundation-Prior为2D 9篇、3D 2篇；表中明确列出SD/video diffusion、SAM、DINO、
    CogVideoX-I2V、LTX-Video、DUSt3R等具体先验及其在重建中的作用。视觉重建大表共
    88篇，77篇Dedicated、11篇Foundation-Prior。
  - 重写reconstruction正文的演进叙事，并同步abstract、intro概览与贡献、
    conclusion和附录；明确区分task-trained Diffusion、普通E2V teacher、辅助FM与
    真正Foundation-Prior。Understanding表中的Semantic-E2VID基础模型标记由错误的
    `SD`更正为`SAM`。
  - 从视觉重建表排除`BMCNet`、`NeurSR`、`UPNSNN`三个event-to-event stream
    super-resolution方法；排除无法核实为ECCV重建论文的`EventMM`；移除
    `EIF-BiOFNet`重复行。以上5项不从审计历史删除，仍在review queue和完整映射中说明。
  - Bib保留`low2023robustenerf`和`zahid2025e3dgslarge`作为各自唯一身份；将E2EGS、
    EGVD、DeblurSplat更新至CVPR 2026、ICCVW 2025和TMM 2026正式版本；补全并去重
    Fourier low-light与Semantic-E2VID。逐条完整前后Bib和权威链接见
    `docs/reference_audit/reconstruction_taxonomy_bib_update_2026-07-28.md`。
  - 新增确定性审计生成器以及JSON/CSV台账、完整映射和人工复核入口。
    `REVIEW_QUEUE.md`先列11个低置信或边界项，再列其余高置信建议；脚本只提取和生成，
    Foundation归类由论文/proceedings人工核验覆盖决定。
- 修改原因与证据：
  - 用户批准按新分类方案落地，并要求尽量不依赖人工逐项搜索，同时把原始方法理解用于
    表格和正文重写，而不是只移动旧LaTeX段落。
  - 方法身份与先验作用优先依据CVF、NeurIPS proceedings、出版社DOI、AAAI、arXiv论文
    和官方项目材料；不以标题关键词、README宣传语或单纯网络架构决定分类。
  - Foundation边界中特别保留`EvDiff`与`EPA`供人工复核；`HDRev-Diff`、
    `Elite-EvGS`、`EventSplat`和`EvLight++`的普通任务先验或辅助FM作用已明确写出。
- 验证：
  - `main.bib`共532条且citation key全部唯一；全项目使用367个唯一citation key，
    未解析引用为0。
  - 全项目LaTeX label无重复、交叉引用无缺失；`\begin`/`\end`数量一致，花括号数量
    一致；旧reconstruction小节label、旧四分类措辞和已删除重复Bib key均未残留。
  - 审计台账含88个表内方法、88个唯一citation key，分组计数为13+30+22+12+9+2；
    另保留4个排除项和1个重复行诊断。11个Foundation-Prior方法均有模型名称、作用和
    权威证据链接。
  - 本机未安装`latexmk`、`pdflatex`或`tectonic`，因此未执行本地排版编译；未计算或
    记录文件哈希。
- 待用户处理：
  - 先查看 `docs/reconstruction_taxonomy/REVIEW_QUEUE.md` 前半部分，重点决定
    `EvDiff`、`EPA`及范围边界项；后半部分高置信方法仅需抽查。
  - 在Overleaf以`main.tex`编译，检查三张reconstruction表的分页、字号、交叉引用和
    参考文献输出。
  - `Dedicated Reconstruction Paradigms`仍是暂定名称，可在不改变方法映射的情况下
    继续优化。
  - 在Windows环境自行检查差异并执行私人仓库和公共发布仓库的Git提交、推送。

### 2026-07-28-03：放宽Foundation-Prior范围并补入EvLight++与HDRev-Diff

- 执行者：Codex
- 范围：主项目 `sections/catalog_tables.tex`、`sections/3_method.tex`、
  `sections/0_abstract.tex`、`sections/1_intro.tex`；外层
  `docs/reconstruction_taxonomy/`、`docs/README.md`、`PROJECT_STATUS.md`、
  `tools/reconstruction_taxonomy/build_taxonomy_audit.py`及本修改历史；未修改Bib或Git。
- 修改前：
  - Foundation-Prior要求大规模预训练模型直接影响核心重建，共11篇；仅用于数据集
    伪标签或结构化下游评价的`EvLight++`仍放在Dedicated。
  - `HDRev-Diff`被记录为只使用任务预训练HDRev编码器，未识别其固定参数的预训练
    latent diffusion model，因此也被放在Dedicated。
  - 当前计数为Dedicated 77篇、Foundation-Prior 11篇。
- 修改后：
  - 判据改为：外部大规模预训练模型只要对论文的方法贡献产生实质作用，即可通过推理、
    训练监督、生成先验、几何初始化、数据集知识增强或结构化能力验证进入
    Foundation-Prior；不再建立`Core/Auxiliary`子类，具体作用只在`Role`列说明。
  - `EvLight++`移入Foundation-Prior 2D：SAM与单目深度基础模型用于语义/深度伪标签，
    并检验增强视频恢复的场景结构和下游可用性。
  - `HDRev-Diff`移入Foundation-Prior 2D：ICCV论文明确采用固定参数的预训练latent
    diffusion model，并通过预训练事件图像编码器和控制路径注入HDR条件。此项明确撤回
    记录`2026-07-28-02`中把它视为普通task-trained diffusion的旧结论。
  - Foundation-Prior现为13篇（2D 11、3D 2），Dedicated为75篇
    （2D Event-Only 13、2D Event-Guided RGB 28、3D Event-Only 22、
    3D Event-Assisted Multimodal 12）；总数仍为88篇。
  - `REVIEW_QUEUE.md`仍把9个低置信、任务先验和范围/身份边界项放在前部；
    `EvLight++`和`HDRev-Diff`作为已核实高置信项放在后部。
- 修改原因与证据：
  - 用户指出论文标题强调“大模型时代”，基础模型带来的数据标注和下游结构验证同样是
    实质受益，不应因未进入核心reconstructor而排除。
  - `EvLight++`证据来自TPAMI DOI `10.1109/TPAMI.2025.3617801`及arXiv
    `2408.16254`；`HDRev-Diff`证据来自ICCV 2025正式论文，其方法图和正文明确写出
    fixed pretrained latent diffusion parameters。
  - 同时复核TRG-Diffusion、EGDeblurring、NEC-Diff、SEE、ClearSight、Fourier prior、
    Self-EHDRI、Elite-EvGS与EventSplat，未发现需要按该标准继续扩入的方法。逐项依据见
    `docs/reconstruction_taxonomy/FOUNDATION_SCOPE_RECHECK.md`。
- 验证：
  - 生成台账仍有88个唯一表内方法；分组计数为13+28+22+12+11+2。
  - 13个Foundation-Prior方法均记录具体模型、作用及证据链接。
  - Bib共532个唯一key，全文367个唯一citation key，缺失引用为0；LaTeX环境、花括号、
    label及交叉引用静态检查均通过。
  - 本机无LaTeX引擎，未执行本地编译；未计算或记录哈希。
- 待用户处理：
  - 在Overleaf以`main.tex`检查更新后的Foundation-Prior表分页和正文排版。
  - 优先查看`REVIEW_QUEUE.md`前部9项；其余高置信项只需抽查。
