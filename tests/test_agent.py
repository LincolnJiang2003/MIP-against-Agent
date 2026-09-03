from PIL import Image

from mip_attack.agent import place_png_patch


def test_png_patch_replacement(tmp_path):
    patch_path = tmp_path / "patch.png"
    Image.new("RGB", (1, 1), (10, 20, 250)).save(patch_path)
    carrier = Image.new("RGB", (1, 1), (250, 10, 20))

    result = place_png_patch(carrier, patch_path, [0, 1], [0, 1])

    assert result.getpixel((0, 0)) == (10, 20, 250)
