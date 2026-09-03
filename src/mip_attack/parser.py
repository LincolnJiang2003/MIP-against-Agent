import hashlib
from pathlib import Path

import torch

from mip_attack.parsing.utils.obs import parser_to_prompt


DEBUG_COLORS = {
    "image": "red",
    "text": "blue",
    "icon": "green",
    "text/html": "magenta",
}
PROMPT_COLORS = {"image": "red", "icon": "green"}
PAPER_FONT_SHA256 = "35c0f3559d8db569e36c31095b8a60d441643d95f59139de40e23fada819b833"
SUPPORTED_ORIGINS = {"a11y", "oss", "mixed-oss", "omni", "mixed-omni"}


def _rectangles(regions):
    return [
        [
            int(entity["shape"]["x"]),
            int(entity["shape"]["y"]),
            int(entity["shape"]["x"] + entity["shape"]["width"]),
            int(entity["shape"]["y"] + entity["shape"]["height"]),
        ]
        for entity in regions
    ]


class AgentParser:
    """Load only the parser backend requested by the experiment."""

    def __init__(self, model_root="models", oss_box_threshold=0.03):
        self.model_root = Path(model_root).expanduser().resolve()
        self.font_path = self.model_root / "arial.ttf"
        self.oss_box_threshold = oss_box_threshold
        self._extractor = None
        self._omni_proposal = None
        self._font_verified = False

    def _verified_font(self):
        if self._font_verified:
            return self.font_path
        if not self.font_path.is_file():
            raise FileNotFoundError(
                f"Missing Arial 2.82 font: {self.font_path}. See models/README.md."
            )
        digest = hashlib.sha256(self.font_path.read_bytes()).hexdigest()
        if digest != PAPER_FONT_SHA256:
            raise ValueError(
                f"Arial font checksum mismatch: expected {PAPER_FONT_SHA256}, got {digest}"
            )
        self._font_verified = True
        return self.font_path

    @property
    def extractor(self):
        if self._extractor is None:
            from mip_attack.parsing.parser import ScreenParser

            self._extractor = ScreenParser(
                {
                    "pipeline": ["groundingdino", "ocr"],
                    "groundingdino": {
                        "box_threshold": self.oss_box_threshold,
                        "weights_path": str(
                            self.model_root
                            / "GroundingDINO"
                            / "groundingdino_swint_ogc.pth"
                        ),
                    },
                    "ocr": {"class_name": "TesseractOCR"},
                }
            )
        return self._extractor

    @property
    def omni_proposal(self):
        if self._omni_proposal is None:
            from mip_attack.parsing.omniparser.omniparser import Omniparser

            omni_root = self.model_root / "OmniParser"
            self._omni_proposal = Omniparser(
                config={
                    "omni_base_path": str(omni_root),
                    "som_model_path": str(omni_root / "icon_detect" / "model.pt"),
                    "caption_model_path": str(
                        omni_root / "icon_caption_florence"
                    ),
                }
            )
        return self._omni_proposal

    @staticmethod
    def _mixed_regions(image, accessibility_tree, propose_visual_regions):
        from mip_attack.a11y import (
            filter_entities_with_mask,
            get_mask_from_entities,
            propose_ents,
        )
        from mip_attack.parsing.utils.som import filter_entities

        a11y_regions = filter_entities(propose_ents(accessibility_tree))
        mask = get_mask_from_entities(a11y_regions, image.width, image.height)
        visual_regions = propose_visual_regions(image)
        return a11y_regions + filter_entities_with_mask(visual_regions, mask, th=0.5)

    @torch.no_grad()
    def parse(self, image, accessibility_tree, som_origin):
        if som_origin not in SUPPORTED_ORIGINS:
            raise ValueError(f"Unsupported parser origin: {som_origin}")
        font_path = self._verified_font()

        if som_origin == "a11y":
            from mip_attack.a11y import propose_ents

            regions = propose_ents(accessibility_tree)
            debug_colors = prompt_colors = {"a11y": "red"}

        elif som_origin == "oss":
            regions = self.extractor.propose_ents(image)
            debug_colors, prompt_colors = DEBUG_COLORS, PROMPT_COLORS

        elif som_origin == "mixed-oss":
            regions = self._mixed_regions(
                image,
                accessibility_tree,
                self.extractor.propose_ents,
            )
            debug_colors = {"a11y": "red", **DEBUG_COLORS}
            prompt_colors = {"a11y": "red", **PROMPT_COLORS}

        elif som_origin == "omni":
            regions = self.omni_proposal.propose_ents(image, with_captions=True)
            debug_colors, prompt_colors = DEBUG_COLORS, PROMPT_COLORS

        elif som_origin == "mixed-omni":
            from mip_attack.a11y import (
                filter_entities_with_mask,
                filter_nonvis_and_oob_entities,
                get_mask_from_entities,
                propose_ents,
            )
            from mip_attack.parsing.utils.som import filter_entities

            a11y_regions = filter_nonvis_and_oob_entities(
                filter_entities(propose_ents(accessibility_tree)),
                image.width,
                image.height,
            )
            mask = get_mask_from_entities(a11y_regions, image.width, image.height)
            omni_regions = self.omni_proposal.propose_ents(
                image, with_captions=False
            )
            regions = a11y_regions + filter_entities_with_mask(
                omni_regions, mask, th=0.5
            )

            # Preserve the paper implementation's square-icon heuristic.
            for region in a11y_regions:
                width = region["shape"]["width"]
                height = region["shape"]["height"]
                region["type"] = "icon" if 0.9 <= width / height <= 1.1 else "text"
            debug_colors, prompt_colors = DEBUG_COLORS, PROMPT_COLORS

        image_debug, image_prompt, text = parser_to_prompt(
            image,
            regions,
            debug_colors,
            prompt_colors,
            font_path=font_path,
        )
        return regions, _rectangles(regions), image_debug, image_prompt, text, "N/A"
