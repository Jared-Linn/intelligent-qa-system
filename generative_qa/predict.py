"""
千问 LoRA 推理脚本

加载微调后的 LoRA 权重进行问答。

用法:
    python generative_qa/predict.py --question "什么是机器学习"
    python generative_qa/predict.py --interactive
"""

import sys
import argparse
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


def parse_args():
    parser = argparse.ArgumentParser(description="千问 LoRA 推理")
    parser.add_argument("--question", "-q", type=str, help="单次查询的问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/qwen_lora/",
                        help="LoRA 权重路径")
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="基座模型名称")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def load_model(base_model_name: str, lora_path: str):
    """加载基座模型 + LoRA 权重"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from peft import PeftModel
    except ImportError:
        print("请安装依赖: pip install torch transformers peft")
        sys.exit(1)

    print(f"[加载] 基座模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA 权重（如果存在）
    lora_path = Path(lora_path)
    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        print(f"[加载] LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))
    else:
        print(f"[警告] 未找到 LoRA 权重: {lora_path}")
        print("[提示] 使用基座模型直接推理（未微调）")

    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_length: int = 512,
                    temperature: float = 0.7) -> str:
    """生成答案"""
    import torch

    messages = [
        {"role": "system", "content": "你是一个专业的中文问答助手，请准确简洁地回答问题。"},
        {"role": "user", "content": question},
    ]

    # Qwen 的 chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_length).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.base_model, args.model_path)

    if args.question:
        answer = generate_answer(model, tokenizer, args.question,
                                 args.max_length, args.temperature)
        print(f"\n[问题] {args.question}")
        print(f"[答案] {answer}")

    if args.interactive or not args.question:
        print("\n" + "=" * 50)
        print("  千问问答 — 交互模式")
        print("  输入 exit 退出")
        print("=" * 50)
        while True:
            try:
                q = input("\n[问题]: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            answer = generate_answer(model, tokenizer, q,
                                     args.max_length, args.temperature)
            print(f"[答案] {answer}")


if __name__ == "__main__":
    main()
