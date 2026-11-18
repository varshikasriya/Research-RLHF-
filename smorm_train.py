"""
SMORM with a weighted Eq. 2:

    loss = w1 * BT_loss(single head, Skywork pairs) + w2 * MSE_loss(multi head, HelpSteer2)

w1 = w2 = 1 reproduces Eq. 2 of the paper exactly.

Shared backbone (decoder-only LLM, last-token hidden state) + two linear heads:
  head_s : d -> 1   (Bradley-Terry score)
  head_m : d -> 5   (helpfulness, correctness, coherence, complexity, verbosity)

Example (real run, on a GPU machine):
  python smorm_train.py --model google/gemma-2b-it --w1 1 --w2 1 --out runs/w1_1_w2_1

Example (tiny local smoke test, CPU):
  python smorm_train.py --smoke --model Qwen/Qwen2.5-0.5B --out runs/smoke
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ATTRS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# RewardBench section -> subsets (standard grouping)
RB_SECTIONS = {
    "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard",
             "mt-bench-easy", "mt-bench-med"],
    "Chat Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
                  "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "Safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
               "xstest-should-respond", "donotanswer"],
    "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js",
                  "hep-python", "hep-rust"],
}


# --------------------------------------------------------------------------- model
class SMORM(nn.Module):
    def __init__(self, name, k=len(ATTRS), dtype=torch.float32):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name, torch_dtype=dtype)
        d = self.backbone.config.hidden_size
        self.head_s = nn.Linear(d, 1, bias=False)   # w_S in R^{d x 1}
        self.head_m = nn.Linear(d, k, bias=False)   # w_M in R^{d x k}
        for h in (self.head_s, self.head_m):
            nn.init.normal_(h.weight, std=0.02)

    def features(self, ids, mask):
        """f_theta(x (+) y): hidden state of the last non-pad token (right padding)."""
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(1) - 1
        return h[torch.arange(h.size(0), device=h.device), last].float()


# --------------------------------------------------------------------------- data
def to_text(tok, x):
    """x is either a raw string or a list of {'role','content'} messages."""
    if isinstance(x, str):
        return x
    return tok.apply_chat_template(x, tokenize=False)


def encode(tok, texts, max_len, device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len, add_special_tokens=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def make_loader(ds, bs, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                       collate_fn=lambda b: b, generator=g, drop_last=True)


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def score_pairs(model, tok, pairs, max_len, device, bs):
    """pairs: list of (chosen_text, rejected_text). Returns per-pair booleans for F and L."""
    model.eval()
    ok_f, ok_l = [], []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        texts = [c for c, _ in chunk] + [r for _, r in chunk]
        ids, mask = encode(tok, texts, max_len, device)
        f = model.features(ids, mask)
        s = model.head_s(f).squeeze(-1)
        m = model.head_m(f).mean(-1)
        n = len(chunk)
        ok_f += (s[:n] > s[n:]).tolist()
        ok_l += (m[:n] > m[n:]).tolist()
    return ok_f, ok_l


def eval_rewardbench(model, tok, max_len, device, bs, max_samples=None):
    ds = load_dataset("allenai/reward-bench", split="filtered")
    if max_samples:
        ds = ds.shuffle(seed=0).select(range(min(max_samples, len(ds))))
    pairs, subsets = [], []
    for ex in ds:
        p = [{"role": "user", "content": ex["prompt"]}]
        pairs.append((to_text(tok, p + [{"role": "assistant", "content": ex["chosen"]}]),
                      to_text(tok, p + [{"role": "assistant", "content": ex["rejected"]}])))
        subsets.append(ex["subset"])
    ok_f, ok_l = score_pairs(model, tok, pairs, max_len, device, bs)
    out = {}
    for tag, ok in (("F", ok_f), ("L", ok_l)):
        sec_acc = {}
        for sec, subs in RB_SECTIONS.items():
            idx = [j for j, s in enumerate(subsets) if s in subs]
            if idx:
                sec_acc[sec] = 100 * sum(ok[j] for j in idx) / len(idx)
        sec_acc["Avg"] = sum(sec_acc.values()) / max(len(sec_acc), 1)  # mean of sections
        out[tag] = sec_acc
    return out


@torch.no_grad()
def eval_helpsteer_val(model, tok, max_len, device, bs, max_samples=None):
    ds = load_dataset("nvidia/HelpSteer2", split="validation")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    model.eval()
    se = torch.zeros(len(ATTRS))
    n = 0
    for i in range(0, len(ds), bs):
        chunk = ds.select(range(i, min(i + bs, len(ds))))
        texts = [to_text(tok, [{"role": "user", "content": e["prompt"]},
                               {"role": "assistant", "content": e["response"]}]) for e in chunk]
        tgt = torch.tensor([[e[a] for a in ATTRS] for e in chunk], dtype=torch.float32)
        ids, mask = encode(tok, texts, max_len, device)
        pred = model.head_m(model.features(ids, mask)).cpu()
        se += ((pred - tgt) ** 2).sum(0)
        n += len(chunk)
    return {a: (se[j] / n).item() for j, a in enumerate(ATTRS)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2b-it")
    ap.add_argument("--w1", type=float, default=1.0, help="weight on BT loss")
    ap.add_argument("--w2", type=float, default=1.0, help="weight on regression loss")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--bs", type=int, default=16, help="pairs per step AND attribute samples per step (per micro-batch)")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--n_pref", type=int, default=None, help="limit # Skywork pairs")
    ap.add_argument("--n_attr", type=int, default=None, help="limit # HelpSteer2 samples")
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--rb_max", type=int, default=None, help="limit # RewardBench examples")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run")
    args = ap.parse_args()

    if args.smoke:
        args.n_pref, args.n_attr, args.bs, args.accum = 16, 16, 2, 1
        args.max_len, args.rb_max, args.eval_bs = 128, 40, 4

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"device={device}  w1={args.w1}  w2={args.w2}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"   # keep the end of the response (last token is the feature)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = SMORM(args.model, dtype=dtype).to(device)
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()

    # ---- data
    d_s = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2", split="train").shuffle(seed=args.seed)
    d_m = load_dataset("nvidia/HelpSteer2", split="train").shuffle(seed=args.seed)
    if args.n_pref:
        d_s = d_s.select(range(min(args.n_pref, len(d_s))))
    if args.n_attr:
        d_m = d_m.select(range(min(args.n_attr, len(d_m))))
    print(f"D_S (pairs) = {len(d_s)}   D_M (attribute samples) = {len(d_m)}")

    it_s = cycle(make_loader(d_s, args.bs, args.seed))
    it_m = cycle(make_loader(d_m, args.bs, args.seed + 1))

    steps_per_epoch = max(1, len(d_s) // (args.bs * args.accum))
    total_steps = steps_per_epoch * args.epochs

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)

    # ---- train
    log = []
    model.train()
    for step in range(1, total_steps + 1):
        opt.zero_grad(set_to_none=True)
        bt_acc = reg_acc = 0.0
        for _ in range(args.accum):
            bs_pref, bs_attr = next(it_s), next(it_m)
            B = len(bs_pref)
            texts = ([to_text(tok, e["chosen"]) for e in bs_pref] +
                     [to_text(tok, e["rejected"]) for e in bs_pref])
            ids_p, mask_p = encode(tok, texts, args.max_len, device)

            texts_m = [to_text(tok, [{"role": "user", "content": e["prompt"]},
                                     {"role": "assistant", "content": e["response"]}])
                       for e in bs_attr]
            ids_m, mask_m = encode(tok, texts_m, args.max_len, device)
            tgt = torch.tensor([[e[a] for a in ATTRS] for e in bs_attr],
                               dtype=torch.float32, device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                f_p = model.features(ids_p, mask_p)
                f_m = model.features(ids_m, mask_m)

            r = model.head_s(f_p).squeeze(-1)
            bt = -F.logsigmoid(r[:B] - r[B:]).mean()                      # Bradley-Terry
            reg = ((model.head_m(f_m) - tgt) ** 2).sum(-1).mean()         # ||w_M^T f - r||_2^2
            loss = args.w1 * bt + args.w2 * reg                           # weighted Eq. 2

            (loss / args.accum).backward()
            bt_acc += bt.item() / args.accum
            reg_acc += reg.item() / args.accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        log.append({"step": step, "bt": bt_acc, "reg": reg_acc})
        if step % 10 == 0 or step == 1 or step == total_steps:
            print(f"step {step}/{total_steps}  bt={bt_acc:.4f}  reg={reg_acc:.4f}")

    # ---- eval
    print("Evaluating on RewardBench ...")
    rb = eval_rewardbench(model, tok, args.max_len, device, args.eval_bs, args.rb_max)
    print("Evaluating on HelpSteer2 validation ...")
    hs = eval_helpsteer_val(model, tok, args.max_len, device, args.eval_bs,
                            args.rb_max if args.smoke else None)

    results = {"args": vars(args), "rewardbench": rb, "helpsteer2_val_mse": hs, "train_log": log}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"rewardbench": rb, "helpsteer2_val_mse": hs}, indent=2))

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()