#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    load_dotenv(Path(__file__).parents[1] / ".env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--arial-font",
        type=Path,
        help="Legally obtained Arial 2.82 arial.ttf used by the paper implementation",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.model_dir.expanduser().resolve()
    manifest = json.loads((Path(__file__).parents[1] / "models/manifest.json").read_text())

    if args.arial_font:
        source = args.arial_font.expanduser().resolve()
        expected = manifest["local_files"]["arial.ttf"]
        actual = sha256(source) if source.is_file() else "missing"
        if actual != expected:
            raise SystemExit(
                f"Arial font checksum mismatch: expected {expected}, got {actual}"
            )
        root.mkdir(parents=True, exist_ok=True)
        destination = root / "arial.ttf"
        if source != destination:
            shutil.copy2(source, destination)

    if not args.verify_only:
        from huggingface_hub import snapshot_download

        for name, source in manifest["sources"].items():
            snapshot_download(
                repo_id=source["repo_id"],
                revision=source["revision"],
                local_dir=root / name,
                allow_patterns=(
                    ["groundingdino_swint_ogc.pth"]
                    if name == "GroundingDINO"
                    else ["icon_detect/*", "icon_caption_florence/*"]
                ),
            )

    expected_files = {**manifest["files"], **manifest["local_files"]}
    failures = []
    for relative_path, expected in expected_files.items():
        path = root / relative_path
        actual = sha256(path) if path.is_file() else "missing"
        if actual != expected:
            failures.append(f"{relative_path}: expected {expected}, got {actual}")
    if failures:
        hint = (
            "\nProvide a licensed Arial 2.82 file with --arial-font."
            if not (root / "arial.ttf").is_file()
            else ""
        )
        raise SystemExit("Model verification failed:\n" + "\n".join(failures) + hint)
    print(f"Verified {len(expected_files)} model and rendering artifacts in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
