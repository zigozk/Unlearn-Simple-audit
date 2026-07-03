import json

def inspect_eval_json(path, n=3):
    d = json.load(open(path, "r"))
    triples = d["generated_text"]  # list of [prompt, gen, gt]
    def norm(x): return "" if x is None else str(x)

    empty_gt = [i for i,(p,g,gt) in enumerate(triples) if norm(gt).strip() == ""]
    empty_gen = [i for i,(p,g,gt) in enumerate(triples) if norm(g).strip() == ""]
    leak = [i for i,(p,g,gt) in enumerate(triples) if norm(gt).strip() and norm(gt).strip() in norm(p)]

    print(f"\n== {path} ==")
    print("N =", len(triples))
    print("empty_gt =", len(empty_gt), f"({len(empty_gt)/len(triples):.1%})",
          "| empty_gen =", len(empty_gen), f"({len(empty_gen)/len(triples):.1%})",
          "| prompt_contains_gt =", len(leak), f"({len(leak)/len(triples):.1%})")

    print("\nExamples:")
    for i in range(min(n, len(triples))):
        p,g,gt = triples[i]
        print("----", i)
        print("PROMPT:", norm(p)[-350:])
        print("GEN   :", norm(g)[:250])
        print("GT    :", norm(gt)[:250])

# 把路径换成你 Llama3 run 对应的 eval json
inspect_eval_json("/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/llama31_forget/simnpo_grad_diff_forget05_lr1e-5_beta0.1_lambda0.3_bs4_ga8_ep5/checkpoint-31/eval_log_forget.json")
inspect_eval_json("/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/llama31_forget/simnpo_grad_diff_forget05_lr1e-5_beta0.1_lambda0.3_bs4_ga8_ep5/checkpoint-31/eval_log_forget.json")
inspect_eval_json("/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/llama31_forget/simnpo_grad_diff_forget05_lr1e-5_beta0.1_lambda0.3_bs4_ga8_ep5/checkpoint-31/eval_real_author_wo_options.json")
inspect_eval_json("/home/zkzhang/unlearning/Unlearn-Simple/TOFU/unlearned/llama31_forget/simnpo_grad_diff_forget05_lr1e-5_beta0.1_lambda0.3_bs4_ga8_ep5/checkpoint-31/eval_real_world_wo_options.json")
