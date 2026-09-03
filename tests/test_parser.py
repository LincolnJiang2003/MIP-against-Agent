import json
from pathlib import Path

import pytest
from PIL import Image

from mip_attack.parser import AgentParser, PAPER_FONT_SHA256


def test_parser_backends_are_lazy():
    parser = AgentParser(model_root="models")
    assert parser._extractor is None
    assert parser._omni_proposal is None


def test_paper_font_checksum_is_manifested():
    manifest_path = Path(__file__).parents[1] / "models/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["local_files"]["arial.ttf"] == PAPER_FONT_SHA256


def test_parser_reports_missing_font(tmp_path):
    parser = AgentParser(model_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="Arial 2.82"):
        parser._verified_font()


def test_unknown_parser_is_rejected_before_loading_assets(tmp_path):
    parser = AgentParser(model_root=tmp_path)
    with pytest.raises(ValueError, match="Unsupported parser origin"):
        parser.parse(Image.new("RGB", (8, 8)), None, "unknown")
