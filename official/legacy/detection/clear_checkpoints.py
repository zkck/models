import sys
from pathlib import Path

model_dir = Path(sys.argv[1])

protected = set()
with (model_dir / "checkpoint").open() as f:
    for line in f:
        protected.add(line.split(":")[-1].strip().strip('"'))

print(f"Protecting {protected}")

for checkpoint in model_dir.glob("*.ckpt-*.*"):
    if checkpoint.stem not in protected:
        print(f"Removing {checkpoint}")
        checkpoint.unlink()
