import os
import json
import glob
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
# 模型路径 (Qwen3-VL 也可以当作纯文本模型用，或者你可以换成 Qwen2.5-7B-Instruct)
MODEL_PATH = "/mnt/data/linmin/Models/Qwen3-VL-4B-Thinking"

# 数据路径 (绝对路径)
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-aloha-agilex_randomized_500-200"

# vLLM 设置 (纯文本不需要 limit_mm_per_prompt)
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 4096
# ===========================================

def build_text_prompt(high_level_instr, template_subtasks):
    """
    构建纯文本 Prompt
    让模型根据 High-level 指令的细节（颜色、材质）去改写 Low-level 子任务
    """
    phases_str = "\n".join([f"{i+1}. {task}" for i, task in enumerate(template_subtasks)])
    
    prompt = f"""<|im_start|>system
You are a robotic instruction expert. You specialize in aligning low-level subtasks with high-level descriptions.
<|im_end|>
<|im_start|>user
I have a robot task.
The **High-Level Instruction** is: "{high_level_instr}"

The **Standard Subtasks** are:
{phases_str}

**Task:**
Rewrite the Standard Subtasks so they match the details (objects, adjectives, verbs) in the High-Level Instruction.
- Keep the **same number of steps**.
- Use the specific nouns/adjectives from the instruction (e.g., if it says "silver hammer", use "silver hammer" in the subtasks).
- Make the language natural and diverse.

**Output strictly in JSON format:**
{{
  "new_subtasks": [
    "step_1_rewritten",
    "step_2_rewritten",
    ...
  ]
}}
<|im_end|>
<|im_start|>assistant
<|think|>
"""
    return prompt

def main():
    # 1. 初始化 vLLM (纯文本模式，不需要 image 参数)
    print(f"Initializing vLLM with model: {MODEL_PATH}")
    try:
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            # 注意：移除了 limit_mm_per_prompt，因为我们不传图片
        )
    except Exception as e:
        print(f"Failed to initialize vLLM: {e}")
        return
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        stop=["<|im_end|>"]
    )

    # 2. 扫描 Episodes
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    print(f"Found {len(episode_dirs)} episodes.")
    
    prompts = []
    metadata = []

    # 3. 准备 Prompt 数据
    print("Preparing text prompts...")
    for ep_dir in episode_dirs:
        json_path = os.path.join(ep_dir, "instructions.json")
        
        if not os.path.exists(json_path):
            continue
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except:
            continue
            
        # 获取 High-level 指令列表
        instructions = data.get("instructions", [])
        if not instructions:
            continue
            
        # 获取模版 (取第一个现有的 subtasks)
        if "subtasks" not in data or not data["subtasks"]:
            continue
        template = data["subtasks"][0]
        
        # === 关键：为每一条指令单独生成一个 Prompt ===
        # 这样确保 instruction[i] 和 subtasks[i] 完美对应
        episode_prompts = []
        for instr in instructions:
            prompt_text = build_text_prompt(instr, template)
            episode_prompts.append(prompt_text)
            
            # 记录元数据，以便后续回填
            prompts.append(prompt_text)
            
        metadata.append({
            "json_path": json_path,
            "original_data": data,
            "count": len(instructions), # 这一集有多少条指令
            "template_len": len(template)
        })

    if not prompts:
        print("No valid data found.")
        return

    # 4. 批量推理
    print(f"Running inference on {len(prompts)} items...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # 5. 解析结果并回填
    print("Saving results...")
    
    # 指针，用于追踪当前的 outputs 属于哪个 episode
    output_idx = 0
    success_count = 0

    for meta in tqdm(metadata):
        count = meta["count"]
        # 取出这一集对应的所有 outputs
        episode_outputs = outputs[output_idx : output_idx + count]
        output_idx += count
        
        new_subtasks_list = []
        
        # 遍历每一条生成结果
        for output in episode_outputs:
            text = output.outputs[0].text
            content = text.split("</think>")[-1] if "</think>" in text else text
            
            try:
                # JSON 解析
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end != -1:
                    result = json.loads(content[start:end])
                    new_sub = result.get("new_subtasks", [])
                    
                    # 验证长度
                    if len(new_sub) == meta["template_len"]:
                        new_subtasks_list.append(new_sub)
                    else:
                        # 如果生成长度不对，回退到模版，防止报错
                        # print("Length mismatch, using template")
                        if "subtasks" in meta["original_data"] and meta["original_data"]["subtasks"]:
                             new_subtasks_list.append(meta["original_data"]["subtasks"][0])
                else:
                    # 解析失败，回退
                    if "subtasks" in meta["original_data"] and meta["original_data"]["subtasks"]:
                         new_subtasks_list.append(meta["original_data"]["subtasks"][0])
            except:
                if "subtasks" in meta["original_data"] and meta["original_data"]["subtasks"]:
                     new_subtasks_list.append(meta["original_data"]["subtasks"][0])

        # 如果生成的列表不为空，更新 JSON
        if new_subtasks_list:
            data = meta["original_data"]
            data["subtasks"] = new_subtasks_list
            
            with open(meta["json_path"], 'w') as f:
                json.dump(data, f, indent=2)
            success_count += 1

    print(f"Done! Updated {success_count}/{len(metadata)} episodes.")

if __name__ == "__main__":
    main()