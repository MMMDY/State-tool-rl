### Agent 后训练（SFT + RL）
**组内 benchmark 成功率 32.6% → 46.7%**

基于[内容遮挡]落地 SFT → 离线 step-RFT → 在线 GiGPO 后训练全链路，组内 256 题端到端 benchmark 成功率由 32.6% 提升至 46.7%。

- 多源 SFT 轨迹数据 pipeline + 配比实验：[内容遮挡，可辨识片段包含后处理、采样优化相关表述，指向效果提升]
- **switch-policy 纠错轨迹**：构造弱策略先执行错误步、强策略接管纠正的切换轨迹，补足失败-恢复覆盖；错误 step 屏蔽 loss，避免模仿被纠正的动作。
- **离线 step-level GRPO（step-RFT）**：冻结轨迹上下文对单步动作做组内 GRPO，规则 verifier 三段式打分（格式 / 动作类型 / 参数）；配合纠错轨迹占比高的 SFT ckpt 增益显著，+2pp。
- **可验证稠密奖励设计**：progress 主项 × side-effect / false-complete / post-success-abort / overdue 多重惩罚，并对 AnswerSheet 错答做去伪校正——堵住此前加法式 reward 被「假完成刷分」训歪的漏洞。
- **GiGPO 两层优势 + DAPO clip-higher**：episode 级按任务组、step 级按 action-prefix 状态组归一，实现比轨迹级 GRPO 更细粒度的信用分配。