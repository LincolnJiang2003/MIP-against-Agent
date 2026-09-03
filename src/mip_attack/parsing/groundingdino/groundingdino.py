from pathlib import Path

import torch

from mip_attack.parsing.groundingdino.icon_localization import det, load_dino_model


class GroundingDino:
    def __init__(
        self,
        prompts=("icon",),
        config_path=None,
        weights_path=None,
        device=None,
        box_threshold=0.03,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prompts = prompts
        self.box_threshold = box_threshold
        config_path = (
            Path(config_path)
            if config_path
            else Path(__file__).parent / "config/GroundingDINO_SwinT_OGC.py"
        )
        weights_path = (
            Path(weights_path)
            if weights_path
            else Path("models/GroundingDINO/groundingdino_swint_ogc.pth")
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"GroundingDINO weights not found: {weights_path}")
        self.model = load_dino_model(
            str(config_path), str(weights_path), device=self.device
        ).eval()

    def propose_ents(self, image):
        entities = []
        for prompt in self.prompts:
            for x1, y1, x2, y2 in det(
                image,
                prompt,
                self.model,
                box_threshold=self.box_threshold,
            ):
                entities.append(
                    {
                        "from": "groundingdino",
                        "type": prompt,
                        "shape": {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                        },
                    }
                )
        return entities
