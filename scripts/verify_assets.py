#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "data/assets-manifest.json").read_text())
    failures = []
    for relative_path, metadata in manifest["files"].items():
        path = root / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
        elif path.stat().st_size != metadata["size"]:
            failures.append(f"wrong size: {relative_path}")
        elif sha256(path) != metadata["sha256"]:
            failures.append(f"wrong checksum: {relative_path}")
    if failures:
        raise SystemExit("Asset verification failed:\n" + "\n".join(failures))
    print(f"Verified {len(manifest['files'])} screen assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
