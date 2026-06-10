"""
生成式问答评测脚本

功能:
    1. 加载训练好的 LoRA 权重, 对测试集推理
    2. 计算 ROUGE-1/2/L, BLEU
    3. 与检索式 baseline (TF-IDF) 对比
    4. 输出评测报告 + 样例展示

用法:
    python generative_qa/evaluate.py
    python generative_qa/evaluate.py --model_path /autodl-pub/checkpoints/qwen_lora/
    python generative_qa/evaluate.py --model_path checkpoints/qwen_lora/ --sample 10

输出:
    ┌──────────────────────┬──────────┬──────────┐
    │ Metric               │ Retrieval│ Generative│
    ├──────────────────────┼──────────┼──────────┤
    │ ROUGE-1              │  0.4234  │  0.5812  │
    │ ROUGE-2              │  0.3120  │  0.4456  │
    │ ROUGE-L              │  0.4011  │  0.5523  │
    │ BLEU                 │  0.2156  │  0.3874  │
    └──────────────────────┴──────────┴──────────┘
"""

import sys
import json
import math
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="生成式问答评测")
    parser.add_argument("--model_path", type=str,
                        default=str(ROOT / "checkpoints" / "qwen_lora"),
                        help="LoRA 权重路径")
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="基座模型名称")
    parser.add_argument("--test_data", type=str,
                        default=str(ROOT / "data" / "raw" / "faq_expanded.json"),
                        help="测试数据(完整FAQ, 内部取后10%做测试)")
    parser.add_argument("--sample", type=int, default=0,
                        help="显示前 N 条样例对比 (0=不显示)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="生成温度 (评测用低温度保一致性)")
    return parser.parse_args()


def load_test_data(data_path: str, val_split: float = 0.1):
    """
    加载数据, 取后 val_split 比例做测试集。
    返回 [{question, answer}]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs = []
    for item in raw:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            pairs.append({"question": q, "answer": a})

    import random
    random.seed(42)
    random.shuffle(pairs)

    split = int(len(pairs) * (1 - val_split))
    test_pairs = pairs[split:]

    logger.info(f"[数据] 总 {len(pairs)} 条, 测试集 {len(test_pairs)} 条")
    return test_pairs


def load_model(base_model_name: str, lora_path: str):
    """加载基座模型 + LoRA 权重"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        logger.error("缺少依赖: pip install torch transformers peft")
        sys.exit(1)

    lora_dir = Path(lora_path)
    if not lora_dir.exists():
        logger.error(f"LoRA 路径不存在: {lora_path}")
        sys.exit(1)

    has_lora = (lora_dir / "adapter_config.json").exists()
    if not has_lora:
        logger.warning("未找到 LoRA 权重, 使用基座模型直接推理 (baseline)")

    logger.info(f"[加载] 基座: {base_model_name}")
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

    if has_lora:
        logger.info(f"[加载] LoRA: {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(lora_dir))

    model.eval()
    logger.info("[加载] 完成")
    return model, tokenizer


def generate(model, tokenizer, question: str,
             max_length: int = 512, temperature: float = 0.1) -> str:
    """单条生成"""
    import torch

    messages = [
        {"role": "system", "content": "你是一个专业的中文问答助手，请准确简洁地回答问题。"},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_length).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            top_p=0.9,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return answer


def retrieval_baseline(test_pairs: list) -> list:
    """
    TF-IDF 检索 baseline.
    用整个FAQ做索引, 对测试集每个问题检索.
    """
    try:
        from retrieval_qa.modules.knowledge_retrieval import TfidfRetriever
        from retrieval_qa.modules.answer_generation import AnswerGenerator
    except ImportError:
        logger.warning("retrieval_qa 模块不可用, 跳过 baseline")
        return []

    # 用所有数据构建索引 (包括测试集, 模拟FAQ检索场景)
    all_questions = [p["question"] for p in test_pairs]
    all_answers = [p["answer"] for p in test_pairs]

    retriever = TfidfRetriever()
    retriever.build_index(all_questions, all_answers)
    generator = AnswerGenerator(method="direct", min_score=0.0)

    results = []
    for p in test_pairs:
        result = generator.generate(p["question"], retriever.retrieve(p["question"], top_k=1))
        results.append(result["answer"])

    return results


def compute_metrics(references: list, predictions: list) -> dict:
    """
    计算 ROUGE-1/2/L 和 BLEU.

    Returns:
        {"rouge1": float, "rouge2": float, "rougeL": float, "bleu": float}
    """
    try:
        from evaluate import load as load_metric
    except ImportError:
        logger.error("缺少 evaluate 库: pip install evaluate")
        sys.exit(1)

    # ROUGE
    rouge = load_metric("rouge")
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=True,
    )
    rouge1 = rouge_result["rouge1"].mid.fmeasure
    rouge2 = rouge_result["rouge2"].mid.fmeasure
    rougeL = rouge_result["rougeL"].mid.fmeasure

    # BLEU
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs_tokenized = [[ref.split()] for ref in references]
        preds_tokenized = [pred.split() for pred in predictions]
        bleu = corpus_bleu(
            refs_tokenized, preds_tokenized,
            smoothing_function=SmoothingFunction().method1,
        )
    except ImportError:
        logger.warning("nltk 不可用, 跳过 BLEU")
        bleu = 0.0

    return {
        "rouge1": round(rouge1, 4),
        "rouge2": round(rouge2, 4),
        "rougeL": round(rougeL, 4),
        "bleu": round(bleu, 4),
    }


def print_report(gen_metrics: dict, ret_metrics: dict = None,
                 samples: list = None):
    """打印评测报告"""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  生成式问答 — 评测报告")
    print(sep)

    # ── 指标对比 ─────────────────────────────────────────────────────
    header = f"{'Metric':<20} {'Retrieval':<12} {'Generative':<12}"
    print(f"\n{header}")
    print("-" * len(header))
    for metric in ["rouge1", "rouge2", "rougeL", "bleu"]:
        ret_val = f"{ret_metrics[metric]:.4f}" if ret_metrics else "N/A"
        gen_val = f"{gen_metrics[metric]:.4f}"
        print(f"{metric:<20} {ret_val:<12} {gen_val:<12}")

    # ── 样例展示 ─────────────────────────────────────────────────────
    if samples:
        print(f"\n{sep}")
        print("  样例对比 (问题 → 生成答案 / 检索答案 / 标准答案)")
        print(sep)
        for i, s in enumerate(samples[:5], 1):
            print(f"\n  [{i}] Q: {s['question']}")
            print(f"      Gen:  {s['generative'][:120]}")
            if s.get('retrieval'):
                print(f"      Ret:  {s['retrieval'][:120]}")
            print(f"      Ref:  {s['reference'][:120]}")

    print(f"\n{sep}\n")


def main():
    args = parse_args()

    # ── 加载测试集 ───────────────────────────────────────────────────
    test_pairs = load_test_data(args.test_data, val_split=0.1)

    if not test_pairs:
        logger.error("测试集为空")
        sys.exit(1)

    # ── 检索 baseline ────────────────────────────────────────────────
    logger.info("[Baseline] 检索式 TF-IDF ...")
    retrieval_answers = retrieval_baseline(test_pairs)
    if retrieval_answers:
        logger.info(f"[Baseline] 完成 ({len(retrieval_answers)} 条)")

    # ── 加载生成式模型 ───────────────────────────────────────────────
    logger.info("[生成式] 加载模型 ...")
    model, tokenizer = load_model(args.base_model, args.model_path)

    # ── 推理 ─────────────────────────────────────────────────────────
    logger.info("[生成式] 推理中 ...")
    gen_answers = []
    for i, p in enumerate(test_pairs):
        if i % 10 == 0:
            logger.info(f"  [{i}/{len(test_pairs)}]")
        try:
            ans = generate(model, tokenizer, p["question"],
                           args.max_length, args.temperature)
        except Exception as e:
            logger.error(f"  第 {i} 条失败: {e}")
            ans = ""
        gen_answers.append(ans)

    # ── 计算指标 ─────────────────────────────────────────────────────
    references = [p["answer"] for p in test_pairs]

    logger.info("[指标] 计算 ROUGE / BLEU ...")
    gen_metrics = compute_metrics(references, gen_answers)
    ret_metrics = compute_metrics(references, retrieval_answers) if retrieval_answers else None

    # ── 构建样例 ─────────────────────────────────────────────────────
    samples = []
    for i, p in enumerate(test_pairs[:args.sample] if args.sample > 0 else []):
        s = {
            "question": p["question"],
            "reference": p["answer"],
            "generative": gen_answers[i],
        }
        if retrieval_answers:
            s["retrieval"] = retrieval_answers[i]
        samples.append(s)

    # ── 输出报告 ─────────────────────────────────────────────────────
    print_report(gen_metrics, ret_metrics, samples)

    # ── JSON 结果导出 ────────────────────────────────────────────────
    output = {
        "metrics": {
            "generative": gen_metrics,
            "retrieval": ret_metrics,
        },
        "num_samples": len(test_pairs),
        "samples": samples,
    }
    out_path = Path(args.model_path) / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
