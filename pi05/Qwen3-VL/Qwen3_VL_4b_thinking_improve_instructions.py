import json
import re
from pathlib import Path
from vllm import LLM, SamplingParams

# --- 配置路径 ---
model_path = "/mnt/data/linmin/Models/Qwen3-VL-4B-Thinking"
instr_path = Path(
    "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/"
    "beat_block_hammer-aloha-agilex_randomized_500-200/episode_0/instructions.json"
)
out_path = instr_path.with_name("instructions_augmented_vllm.json")

def clean_text(text):
    """提取模型输出并清洗掉序号、思考过程和多余标点"""
    # 移除 "Rewritten Sentence:" 等引导词
    if "Rewritten" in text:
        text = text.split("Rewritten")[-1]
    # 移除类似 "1. ", "- ", "Answer: " 的前缀
    text = re.sub(r'^[^\w\s]+', '', text) # 移除开头特殊字符
    text = re.sub(r'^\d+\.\s*', '', text)  # 移除数字编号
    return text.strip().strip('"')

def process_instructions_vllm():
    # 1. 加载数据
    if not instr_path.exists():
        print(f"错误: 找不到文件 {instr_path}")
        return
        
    with instr_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    subtasks = data.get("subtasks", [])
    # 展平所有指令，以便一次性并行处理
    flat_instructions = []
    for pair in subtasks:
        if isinstance(pair, list) and len(pair) == 2:
            flat_instructions.extend(pair)

    if not flat_instructions:
        print("未发现有效指令。")
        return

    # 2. 初始化 vLLM (使用 6 号显卡)
    # tensor_parallel_size=1 表示单卡，如果显存足够不需要切分
    llm = LLM(
        model=model_path, 
        trust_remote_code=True,
        gpu_memory_utilization=0.8, # 占用 80% 显存，留一点余量
    )

    # 设置生成参数：强制简洁，不采样以获得最稳定结果
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=64, 
        stop=["\n", "Original", "Thought:"]
    )

    # 构造 Prompts
    prompts = [
        f"Rewrite this robotic task instruction into a natural variant. Output ONLY the result.\nOriginal: {text}\nRewritten:" 
        for text in flat_instructions
    ]

    print(f"正在并行处理 {len(prompts)} 条指令...")
    
    # 执行批量推理
    outputs = llm.generate(prompts, sampling_params)
    
    # 提取结果
    rewritten_list = [clean_text(output.outputs[0].text) for output in outputs]

    # 3. 重新组装回 [s1, s2] 结构
    new_subtasks = []
    for i in range(0, len(rewritten_list), 2):
        new_subtasks.append([rewritten_list[i], rewritten_list[i+1]])

    # 4. 保存结果
    data["subtasks_augmented"] = new_subtasks
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"--- 处理完成 ---")
    print(f"原始指令数: {len(subtasks)}")
    print(f"处理后存入: {out_path}")

if __name__ == "__main__":
    # 注意：运行此脚本前确保没有其他进程占用显卡 6
    # 命令行执行: CUDA_VISIBLE_DEVICES=6 python 此脚本.py
    process_instructions_vllm()