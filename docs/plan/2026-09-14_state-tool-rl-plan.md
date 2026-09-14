StateTool-RL项目
流程
数据筛选不是只在SFT、RL 前做一轮。完整路径是冷启动 SFT 打底，推理导向 RL 强化，拒绝采样用 RL 产出的高质量数据再做一轮 SFT，最后对齐导向 RL 收尾。

评测Base模型
主模型：qwen3-4b
构造数据用强模型：

数据构造：Reject sampling 筛选成功轨迹
1. 离线难度过滤：准备候选任务，rollout采样，取任务通过率在20%到80%窗口的题
2. step-RFT，规则Verifier对正确答案的推理过程验证
可验证的过程奖励:
- 调用了与当前诊断相关的工具
- 正确读取并利用 tool observation
- 状态是否向目标状态推进
- 失败后是否进入可恢复状态
- 是否避免重复调用已证明无效的 action
3. LLM as Judge从多维度yes/no打分筛选。把答案对但思维链、过程冗长像在凑的样本过滤
4. 对长思维链改写 （可选）
改写后的轨迹，都必须重新过：schema verifier→ state replay→ terminal verifier
5. 抽检
6. 数据格式构造成 SFT 样本

RFT rollout 采样规则（按任务自适应，不固定每题同一条数）：
- 仅从 train split 构造 RFT 数据；dev/test 不参与候选题筛选、rollout 或数据回灌。
- 每个任务先 rollout 16 条。
- 每条 rollout 的成功判定必须同时通过 Verifier 的两部分：
  - 终态任务成功验证（terminal verifier）：环境终态满足任务 assertion / official success；
  - 过程规则验证（process verifier）：tool schema 与参数合法、正确利用 observation、状态有可验证推进、未重复已证伪 action，且不存在编造结果、越权或不可恢复的违规行为。
- 目标是每个任务收集 4 条“终态 + 过程”双重验证通过、并按 action 序列/关键状态去重后的成功轨迹；达到配额即停止该题采样。
- 未达到配额时，每次追加 8 条 rollout；单任务最多 rollout 32 条。
- 保留成功与失败的所有rollout轨迹，最后做统计量总结

数据筛选规则
- 对高难度的任务（通过率不到某个数值的轨迹），放入recovery候选池
- 对简单的任务（前 16 条中双重验证通过 >=13 条），最多保留 2 条去重后的代表性轨迹，避免简单模板主导训练集。
- 根据任务难度阶梯选择rollout轨迹进入训练集


数据构造： Switch-policy recovery trajectory
训练模型从错误中恢复的能力
做法：prefix state s_t → 弱策略 / 较差采样走一步错误 action a_bad → 得到错误 observation → 强策略从错误上下文接管 → 产出 recovery continuation
SFT 数据格式：正常 prefix ->错误动作 -> 工具返回 / 用户反馈 -> 恢复动作序列

注意事项：
prefix state s_t 必须能被精确恢复。需要保证：同一个任务实例、同一个用户 persona / simulator seed、同一个工具数据库状态、同一个历史工具调用结果、同一个系统 prompt 与 tool schema
每条数据记录并且能恢复加载State snapshot / replay layer
- 每个 prefix 记录 task_id、seed、初始状态、action history、observation history；
- 支持从第 t 步 replay / reset；
- 记录生成模型、checkpoint、temperature、top-p、tool schema hash；
- 对 chosen/rejected replay 一次，验证二者确实从同一 state 出发。


数据构造：step level & segment-level DPO
使用上步骤的Switch-policy recovery策略
step level DPO数据格式：
共享上下文 c：用户问题 + 已有对话 + 当前 tool state
chosen y⁺：强策略在当前状态的正确诊断 / 正确 tool call / 正确回复
rejected y⁻：弱策略在完全相同状态下的错误诊断 / 错误参数 / 无效回复
trajectory -level DPO数据格式：
共享前缀：正常执行历史 → 弱 policy 在某一步犯错 → tool/env 返回错误结果或用户反馈“没解决”
chosen（正样本）：强 policy 基于该失败上下文做 diagnosis + recovery action
rejected（负样本）：弱 policy 在同一失败上下文下继续采取错误/无效动作

样本筛选条件：
1. 完全相同的决策上下文：同一任务、同一状态、同一工具返回历史；
2. chosen 有 verifier 支撑：格式正确、动作类型合法、参数有效，且能提高 partial score / 最终成功；
3. rejected 是明确可归因的坏动作，而非仅“最终碰巧失败”；
4. 拒绝采样不要全是格式错，要覆盖：
  - 错误动作类型；
  - 错误参数；
  - 忽略 tool observation；
  - 重复无效调用；
  - 过早结束；
  - 错误后的无效恢复。
5. 防止 preference leakage：不能因为 chosen 更长、更礼貌或多写解释，就被判为正样本；最好保持 action schema 和输出长度接近。

训练：RFT
拒绝采样微调RFT：从模型自己采样、只保留答对的轨迹做 SFT。
希望在在线 RL 之前，先用一个轻量的 RFT（Rejection-sampling Fine-Tuning，拒绝采样微调）对模型做一次冷启动：让它先学会稳定地产出格式正确、答案正确的轨迹，再交给 RL 去探索工具的上限。
数据：使用前面构造的RFT数据
实验：
数据配比实验


训练：DPO


训练：GRPO/DAPO
奖励：R = R_terminal + α × R_verified_progress - λ × R_violation
奖励设计：
1. terminal reward为主 
任务最终完成 / 环境终态满足 assertion → 1 
否则 → 0
2. R_verified_progress 
- 正确读取了当前 tool observation
- 做出了满足 precondition 的 action
- 从失败状态进入可恢复状态
- 正确识别之前的 action 无效，并停止重复调用
3. R_violation：强负奖励或直接归零
- 非法 tool call
- 参数不合法
- 越权操作
- 编造 tool result / 编造已完成
4. LLM judge过程奖励（可选）
 从多维打分，每个维度尽量独立，打分前首先给出rational说清依据
5. 长度奖励（可选）
软惩罚：在答案正确的前提下，对超过预设上限的输出按长度渐进式扣分。更进一步是分段式的超长惩罚，配合 token 级的策略梯度损失，让长序列里每一个 token 都参与优化，而不是被短样本稀释，试下来比固定阈值平滑很多。

训练过程中检查：
每隔 3 到 5 个 checkpoint 抽检：
- 高reward样本是否在reward hack-> 检查verifier 质量
- 低reward badcase -> 收集做数据增广
实时监控：
- 组内reward是否均为0或1，组内优势归零，奖励坍塌 -> DAPO动态采样
- 检查policy entropy，是否发生熵坍塌 
-> 解决：DAPO 下界保持紧、上界放宽，给探索型 token 更大的上升空间
把 response 级别的 entropy 单独记录、按 checkpoint 画曲线、配告警阈值，连续掉几个 checkpoint 就及时干预。大规模训练会先经历初步发现阶段，再进入精细化打磨阶段，entropy 在发现阶段掉说明探索不足，在打磨阶段掉可能是正常收敛，按训练阶段设不同的预期区间，比统一阈值有效得多。
- 检查policy entropy，是否发生模板坍缩-> 解决：模型输出看起来多样，其实跟输入无关，全是流畅但空洞的套话，entropy 数值还是健康的。跨输入互信息监控，拿同一批 prompt 看输出之间能不能互相区分，区分度掉下去就说明模型开始背模板了

训练trick:
课程式数据，训练中模型能力在变，同一个题的难度是动态的，我们用一个小型代理模型动态评估每个 prompt 的训练价值，价值低的直接跳过，再按难度配比组织 batch，简单中难大致三比五比二。
奖励解耦。结果奖励锚定正确性，过程奖励只在答案正确的样本里排推理质量，两套信号互不干扰
rollout group size动态调节。GRPO 里这个值不是越大越好，thinking 很长的时候，group size 超过 8 边际收益就很低。我们做过一组对比，同样的梯度预算，group size 16 跑 250 步和 group size 4 跑 1000 步，后者在高难度数学评测集上明显更稳、涨得更多。group size 要结合退化组比例和任务难度动态调，难的题大一点，简单的可以小。


数据增广 数据飞轮
从grpo/dapo训练过程中rollout筛选高质量成功轨迹、收集badcase

RFT
筛选成功与高价值恢复轨迹，回灌至 SFT/DPO


评测
业务指标
tau2测试集

模型指标
推理速度、效率
推理长度比较：如分段式超长软惩罚替代硬截断，推理长度缩短 20%
训推一致性
