# Recovery 数据训练设计：两个关键问题的处理方式

## 背景

- **推理时**：检测到错误 → VLM 输出 reflection + correction → 在 recovery 阶段把 correction 与当前预测的 subtask 一起喂给 Action Expert，直到 recovery 结束。
- **训练时**：随机 batch 采样，无法保证同一 batch 内同时出现「某 episode 的 error_attempt 帧」和「该 episode 的 recovery 帧」，也无法保证同时出现「error_attempt 的起始帧」和「结束帧」。

---

## 问题 A：Recovery 帧随机进 batch 时，不知道要加哪条 correction

### 现象

- 希望：recovery 段的文本 = `[Correction] <对应错误的 correction> [Subtask] <当前 phase 的 subtask>`。
- 随机 batch 里可能只有 episode 1 的某几帧 recovery，没有 episode 1 的 error_attempt，模型无法从当前 batch 推断「该加哪条 correction」。

### 做法：**在数据集里按 episode 固定写入 correction（或 error_type）**

1. **构建 dataset 时（预处理或 dataloader 内）**  
   - 对每个 episode，从 metadata 读出 `error_types`（如 `["premature_close"]`），用固定模板生成该 episode 的 `correction` 和（可选）`reflection` 文本，**存到该 episode 的 instructions 或单独 JSON 里**。  
   - 例如在 `instructions.json` 或 `phase_info` 旁增加：
     ```json
     "recovery_context": {
       "error_type": "premature_close",
       "correction": "[Correction] Open the gripper, continue moving down to the correct height, and then close it."
     }
     ```
     （或只存 `error_type`，dataloader 里用 `REFLECTION_TEMPLATES[error_type]["correction"]` 现算。）

2. **每个样本带 episode_id（或 segment_id）**  
   - 每个 (observation, action) 样本在 dataset 里都有 `episode_id`（以及可选的 `phase_label`: normal_pre / recovery / normal_post）。

3. **Dataloader 逻辑**  
   - 若当前样本属于 **recovery 段**：  
     - 用 `episode_id` 查该 episode 的 `recovery_context.correction`（或由 `error_type` 查表得到 correction）；  
     - 文本 = `correction + " " + current_subtask`（或你约定的拼接方式）。  
   - 若属于 normal 段：  
     - 文本 = 当前 phase 的 subtask，不加 correction。

这样 **随机 batch 里只要拿到的是 recovery 帧，就通过 episode_id 查表得到 correction**，不需要同一 batch 里同时出现 error_attempt 与 recovery。

---

## 问题 B：Reflection VLM 需要 error_attempt 的「起止帧」作为输入，但 batch 是随机单帧

### 现象

- 希望：Reflection 头输入 = (error_attempt **开始帧** 图像, error_attempt **结束帧** 图像)，输出 = reflection + correction。
- 若和 Action Expert 共用「随机单帧」dataloader，几乎不可能在同一个 batch 里凑齐「某 episode 的 start 与 end 帧」。

### 做法：**Reflection 头用单独的数据集，按「(start, end) 帧对」采样**

1. **单独建 ReflectionDataset**  
   - 每个样本 = 一个 episode 的一条「错误实例」：  
     - 输入：`(image_start, image_end, task_instruction)`  
       - `image_start` = 该 episode 的 `error_attempt_start_frame` 的图像  
       - `image_end` = 该 episode 的 `error_attempt_end_frame` 的图像  
     - 标签：`(reflection_text, correction_text)`  
       - 可由 metadata 的 `error_types` + 模板生成（与上面 `REFLECTION_TEMPLATES` 一致），或后续用人工/弱监督细化。

2. **训练方式**  
   - Reflection 头 / 双图 VLM：**只**用 ReflectionDataset 训练；batch 为「随机的 (start, end) 帧对」，每个 batch 内多对来自不同 episode。  
   - Action Expert：仍用原来的「随机单帧（或短序列）」dataloader，其中 recovery 帧的文本通过问题 A 的方式挂上 correction。

3. **不要求**  
   - 不需要 Action Expert 的 batch 里出现 error_attempt 起止帧；  
   - 不需要 Reflection 的 batch 里出现 recovery 帧。  
   - 两个头用两套数据流即可。

---

## 小结

| 问题 | 处理方式 |
|------|----------|
| **A：recovery 帧不知道加哪条 correction** | 数据集里按 episode 存好 `error_type` / `correction`；dataloader 对 recovery 段样本用 episode_id 查表拼到文本里。 |
| **B：Reflection 需要起止帧对** | Reflection 专用 Dataset，每个样本 = (image_start, image_end) + 标签；单独训 Reflection 头，不与单帧 Action batch 混合。 |

这样既保留「随机 batch」训练 Action Expert，又保证 recovery 帧始终带正确的 correction，且 Reflection 头有稳定的 (start, end) 输入。

---

## 附录：Reflection / Correction 模板

与推理时 VLM 输出一致，构建 recovery_context 或 Reflection 标签时用 metadata 的 `error_types[0]` 查表：

- **grasp_position_offset**: reflection = "[Reflection] Grasp missed due to position offset. The gripper is empty." ; correction = "[Correction] Open the gripper, adjust the XY position to align with the object, and retry grasping."
- **grasp_orientation_mismatch**: reflection = "[Reflection] Grasp failed due to incorrect orientation. The gripper collided with the object." ; correction = "[Correction] Open the gripper, adjust the wrist rotation to match the object's angle, and retry."
- **premature_close**: reflection = "[Reflection] The gripper closed prematurely in the air before reaching the target." ; correction = "[Correction] Open the gripper, continue moving down to the correct height, and then close it."
- **grasp_slip**: reflection = "[Reflection] The object slipped and dropped from the gripper during transport." ; correction = "[Correction] Track the object's new location, return to it, and grasp it again securely."

Recovery 段输入给 Action Expert 的文本 = correction + 当前 phase 的 subtask；recovery 结束后不再加 correction。
