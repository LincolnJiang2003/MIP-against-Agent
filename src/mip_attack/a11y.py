"""Accessibility-tree helpers adapted from Windows Agent Arena."""

import xml.etree.ElementTree as ET

import numpy as np


STATE = "{uri:deskat:state.at-spi.gnome.org}"
COMPONENT = "{uri:deskat:component.at-spi.gnome.org}"


def _is_leaf(node):
    return len(list(node)) == 0


def _is_visible(node):
    return node.get(f"{STATE}visible", "false") == "true"


def _is_enabled(node):
    return node.get(f"{STATE}enabled", "false") == "true"


def _has_coords(node):
    return bool(node.get(f"{COMPONENT}screencoord", "")) and bool(
        node.get(f"{COMPONENT}size", "")
    )


def _heuristic(node):
    return _is_leaf(node) and _is_visible(node) and _is_enabled(node) and _has_coords(node)


def _center(position, size):
    return position[0] + size[0] // 2, position[1] + size[1] // 2


def _rect_overlaps(center, position, size):
    x, y = center
    return (
        position[0] <= x <= position[0] + size[0]
        and position[1] <= y <= position[1] + size[1]
    )


def _overlap_ratio(position_a, size_a, position_b, size_b):
    area_a = size_a[0] * size_a[1]
    area_b = size_b[0] * size_b[1]
    if area_a == 0 or area_b == 0:
        return 0
    x_overlap = max(
        0,
        min(position_a[0] + size_a[0], position_b[0] + size_b[0])
        - max(position_a[0], position_b[0]),
    )
    y_overlap = max(
        0,
        min(position_a[1] + size_a[1], position_b[1] + size_b[1])
        - max(position_a[1], position_b[1]),
    )
    return 2 * x_overlap * y_overlap / (area_a + area_b)


def _filter_redundant(entities, threshold=0.8):
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            first, second = entities[i]["shape"], entities[j]["shape"]
            if _overlap_ratio(
                (first["x"], first["y"]),
                (first["width"], first["height"]),
                (second["x"], second["y"]),
                (second["width"], second["height"]),
            ) > threshold:
                entities.pop(j)
                break
    return entities


def filter_nonvis_and_oob_entities(entities, width, height):
    filtered = []
    for entity in entities:
        shape = entity["shape"]
        x = min(width, max(0, shape["x"]))
        y = min(height, max(0, shape["y"]))
        x2 = min(width, max(0, x + shape["width"]))
        y2 = min(height, max(0, y + shape["height"]))
        w, h = x2 - x, y2 - y
        if x2 < x or y2 < y or w * h < 5:
            continue
        entity["shape"] = {"x": x, "y": y, "width": w, "height": h}
        filtered.append(entity)
    return filtered


def get_mask_from_entities(entities, width, height):
    mask = np.zeros((height, width))
    for entity in entities:
        shape = entity["shape"]
        x = max(0, min(width, shape["x"]))
        y = max(0, min(height, shape["y"]))
        w = max(0, min(width, shape["width"]))
        h = max(0, min(height, shape["height"]))
        mask[y : y + h, x : x + w] = 1
    return mask


def filter_entities_with_mask(entities, mask, th=0.3):
    filtered = []
    for entity in entities:
        shape = entity["shape"]
        x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
        area = w * h
        if area > 0 and mask[y : y + h, x : x + w].sum() / area < th:
            filtered.append(entity)
    return filtered


def detections_to_entities(detections):
    entities = []
    for detection_type, detection_group in detections.items():
        for detection in detection_group.values():
            x1, y1, x2, y2 = detection["rect"]
            text = detection.get(
                "label",
                detection.get(
                    "text",
                    str(detection.get("words", f"ocr_{detection.get('id', '')}")),
                ),
            )
            entities.append(
                {
                    "type": detection_type,
                    "shape": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    },
                    "text": text,
                }
            )
    return entities


def _parse_pair(value):
    return [int(part.strip("()")) for part in value.split(",")]


def propose_ents(accessibility_tree):
    if accessibility_tree is None:
        return []
    desktop = ET.fromstring(accessibility_tree)
    children = list(desktop)[::-1]
    desktop.clear()
    desktop.extend(children)
    entities_with_centers = []

    def visit(node):
        if not _is_visible(node):
            return
        if _heuristic(node):
            position = _parse_pair(node.get(f"{COMPONENT}screencoord"))
            size = _parse_pair(node.get(f"{COMPONENT}size"))
            entities_with_centers.append(
                (
                    _center(position, size),
                    {
                        "from": "a11y",
                        "type": "a11y",
                        "shape": {
                            "x": position[0],
                            "y": position[1],
                            "width": size[0],
                            "height": size[1],
                        },
                        "text": node.attrib.get("name", ""),
                    },
                )
            )
        for child in node:
            visit(child)

    for window in desktop:
        if not _is_visible(window):
            continue
        position = _parse_pair(window.get(f"{COMPONENT}screencoord"))
        size = _parse_pair(window.get(f"{COMPONENT}size"))
        entities_with_centers = [
            item
            for item in entities_with_centers
            if not _rect_overlaps(item[0], position, size)
        ]
        visit(window)
    return _filter_redundant([entity for _, entity in entities_with_centers])
