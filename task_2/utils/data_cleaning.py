import re
import shutil
from pathlib import Path
from collections import Counter

SRC = Path(__file__).resolve().parent.parent / "data" / "Fruit_And_Vegetable_Diseases_Dataset_no_identical"
DEST = Path(__file__).resolve().parent.parent / "data" / "Fruit_And_Vegetable_Diseases_Dataset_no_identical_no_aug"

AUG_KEYWORDS = [
    # Geometric
    "rotated",          # was: "rotated", "rotate", "rot" — "rot" false-matches "rotten"
    "flipped",          # was: "flipped", "flip" — "flip" is redundant
    "mirror",
    "hflip", "vflip",   # common abbreviations
    "sheared", "shear",
    "scaled", "resized",
    "perspective", "affine",

    # Photometric  
    "brightness",       # was: "brightness", "bright" — keep longest
    "contrast",
    "darkened",         # safer than bare "dark" (matches common product names)
    "jitter",
    "gamma", "saturation",

    # Noise / quality
    "blur", "gaussian",
    "noise",

    # Crop / zoom
    "zoom", "cropped",  # "cropped" less ambiguous than "crop" in agriculture context

    # Generic markers
    "augmented",        # was: "aug", "augmented" — "aug" is too short (matches "august", "gauge" reversed, etc.)
    "_aug_",            # anchored variant — safe specifically for augmentation markers
]
def is_augmented(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in AUG_KEYWORDS)

                                                                            
removed = Counter()
kept    = Counter()

for p in list(SRC.rglob("*.jpg")) + list(SRC.rglob("*.png")) + list(SRC.rglob("*.jpeg")):
    print(p)
    flag = "REMOVE" if is_augmented(p.name) else "keep  "
    print(flag, p.parent.name, "/", p.name)
for img_path in list(SRC.rglob("*.jpg")) + list(SRC.rglob("*.png")) + list(SRC.rglob("*.jpeg")):
    folder = img_path.parent.name
    if is_augmented(img_path.name):
        removed[folder] += 1
        continue
    dest_dir = DEST / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dest_dir / img_path.name)
    kept[folder] += 1

print(f"{'Folder':<30} {'Kept':>6} {'Removed':>8}")
print("-" * 46)
for folder in sorted(set(kept) | set(removed)):
    print(f"{folder:<30} {kept[folder]:>6} {removed[folder]:>8}")
print(f"\nTotal kept:    {sum(kept.values())}")
print(f"Total removed: {sum(removed.values())}")

