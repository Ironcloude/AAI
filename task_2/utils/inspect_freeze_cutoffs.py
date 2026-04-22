"""
Inspect model structure and per-module parameter distributions.
Shows exactly what PARTIAL_FREEZE would unfreeze per architecture, so the
cutoff can be justified (or revised) from visual evidence rather than guesswork.

Run: uv run python task_2/inspect_freeze_cutoffs.py
"""
from torchvision.models import get_model, get_weight

ARCHS = ["maxvit_t", "efficientnet_v2_s", "swin_s"]

WEIGHT_STRING = {
    "maxvit_t":          "MaxVit_T_Weights.IMAGENET1K_V1",
    "efficientnet_v2_s": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
    "swin_s":            "Swin_S_Weights.IMAGENET1K_V1",
}

# How the current PARTIAL_FREEZE patch selects "last stage" per arch.
LAST_STAGE_SELECTOR = {
    "maxvit_t":          lambda m: [("blocks[-1]", m.blocks[-1])],
    "efficientnet_v2_s": lambda m: [("features[-2]", m.features[-2]),
                                    ("features[-1]", m.features[-1])],
    "swin_s":            lambda m: [("features[-1]", m.features[-1])],
}


def fmt(n: int) -> str:
    return f"{n:>12,}"


def inspect(arch: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {arch}")
    print(f"{'='*70}")
    weights = get_weight(WEIGHT_STRING[arch])
    model = get_model(arch, weights=weights)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}\n")

    # Top-level children with param counts and percentages
    print(f"  {'Top-level module':<22} {'params':>12}   {'% of total':>10}")
    print(f"  {'-'*22} {'-'*12}   {'-'*10}")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        pct = 100 * n / total
        print(f"  {name:<22} {fmt(n)}   {pct:>9.2f}%")

    # Zoom into the "last stage" being unfrozen by PARTIAL_FREEZE
    print(f"\n  --- PARTIAL_FREEZE unfreeze targets ---")
    unfrozen_total = 0
    for label, mod in LAST_STAGE_SELECTOR[arch](model):
        n = sum(p.numel() for p in mod.parameters())
        unfrozen_total += n
        pct = 100 * n / total
        print(f"  {label:<22} {fmt(n)}   {pct:>9.2f}%  ({type(mod).__name__})")

    # What's in those modules, one level deeper
    print(f"\n  --- Children of unfreeze targets ---")
    for label, mod in LAST_STAGE_SELECTOR[arch](model):
        for child_name, child in mod.named_children():
            n = sum(p.numel() for p in child.parameters())
            if n == 0:
                continue
            pct = 100 * n / total
            print(f"  {label}.{child_name:<18} {fmt(n)}   {pct:>9.2f}%")

    # Head will be replaced (new Linear) so it always trains in FREEZE/PARTIAL_FREEZE
    head_name = "classifier" if hasattr(model, "classifier") else "head"
    head = getattr(model, head_name)
    head_n = sum(p.numel() for p in head.parameters())
    print(f"\n  Head ({head_name}): {head_n:,} params "
          f"({100*head_n/total:.2f}%) — always trainable (replaced with binary Linear)")

    total_unfrozen = unfrozen_total  # (head gets replaced, new params vary with NUM_CLASSES)
    print(f"\n  >>> PARTIAL_FREEZE backbone unfreeze: "
          f"{total_unfrozen:,} / {total:,} = {100*total_unfrozen/total:.1f}%")


if __name__ == "__main__":
    for a in ARCHS:
        inspect(a)
