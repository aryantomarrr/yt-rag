import json, re, sys
import numpy as np
import pandas as pd
from rag_core import RAGPipeline, load_youtube, get_embeddings, BASELINE, ADVANCED

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _judge(llm, prompt):
    out = llm.chat("You are a strict evaluator. Reply with JSON only.", prompt,
                   temperature=0.0, max_tokens=250)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def judge_all(llm, question, answer, chunks, ground_truth):
    """One LLM call -> faithfulness, context_precision, context_recall."""
    # No retrieved context = the mode failed this question, so it scores 0 (not NaN)
    if not chunks:
        return 0.0, 0.0, 0.0
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    r = _judge(llm, f"""Question: {question}
Ground truth: {ground_truth}

Context passages:
{ctx}

Answer to evaluate:
{answer}

Reply with JSON only, exactly this shape:
{{"claims_total": int, "claims_supported": int, "relevant": [true or false per passage, in order], "truth_total": int, "truth_covered": int}}
- claims_*: split the Answer into factual claims; count how many are supported by the context.
- relevant: is each passage useful for answering the question.
- truth_*: split the Ground truth into statements; count how many are found in the context.""")
    if not r:
        return np.nan, np.nan, np.nan
    try:
        total = max(int(r["claims_total"]), 1)
        faith = 1.0 if "don't know" in answer.lower() else r["claims_supported"] / total
        rel = r["relevant"]
        prec = float(np.mean([bool(x) for x in rel])) if len(rel) == len(chunks) else np.nan
        rec = r["truth_covered"] / max(int(r["truth_total"]), 1)
        return faith, prec, rec
    except Exception:
        return np.nan, np.nan, np.nan


def answer_relevancy(question, answer):
    emb = get_embeddings()          # local model, no API cost
    return float(np.dot(emb.embed_query(question), emb.embed_query(answer)))


def run_eval(pipe, items, cfgs):
    rows = []
    for name, cfg in cfgs.items():
        for it in items:
            res = pipe.answer(it["question"], cfg)
            # judge on what the model actually saw (compressed text for Advanced)
            chunks = [s["text"] for s in res["sources"]]
            f, p, r = judge_all(pipe.llm, it["question"], res["answer"], chunks, it["ground_truth"])
            rows.append({"mode": name, "question": it["question"], "answer": res["answer"],
                         "n_sources": len(chunks),
                         "faithfulness": f,
                         "answer_relevancy": answer_relevancy(it["question"], res["answer"]),
                         "context_precision": p, "context_recall": r,
                         "latency_s": round(res["latency"], 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":      # python evaluate.py <youtube_url> eval_set.json
    pipe = RAGPipeline(load_youtube(sys.argv[1]))
    items = json.load(open(sys.argv[2], encoding="utf-8"))
    df = run_eval(pipe, items, {"baseline": BASELINE, "advanced": ADVANCED})
    df.to_csv("eval_results.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby("mode")[METRICS + ["latency_s"]].mean().round(3)
    summary.to_csv("eval_summary.csv", encoding="utf-8-sig")
    print("\n=== AVERAGE SCORES ===")
    print(summary.to_string())
    print("\nSaved: eval_results.csv, eval_summary.csv")