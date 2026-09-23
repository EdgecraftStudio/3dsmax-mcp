"""Object-ID views and silhouette comparison against a reference drawing."""
from __future__ import annotations

import os
import tempfile
from typing import Any
from uuid import uuid4

from ..helpers import silhouette as si
from ..server import client, mcp

_OUT_DIR = os.path.join(tempfile.gettempdir(), "3dsmax-mcp")


def _save_png(prefix: str, width: int, height: int, rgb: bytes | bytearray) -> str:
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, f"{prefix}_{uuid4().hex[:12]}.png")
    with open(path, "wb") as f:
        f.write(si.png_bytes(width, height, rgb))
    return path


def _id_view(names: list[str], view: str, resolution: int, padding: float, max_nodes: int) -> dict[str, Any]:
    raw = client.send_command(si.build_id_script(names, view, resolution, padding, max_nodes)).get("result", "")
    return si.parse_id_report(str(raw))


@mcp.tool()
def object_id_view(
    names: list[str] | None = None,
    view: str = "front",
    resolution: int = 256,
    padding: float = 0.05,
    limit: int = 50,
    max_nodes: int = 200,
) -> dict[str, Any]:
    """Draw which object is visible at each pixel of an orthographic view, one flat colour per object.

    view: front | back | left | right | top | bottom (Max's own axes). The view
    frames the nodes with `padding` (fraction of the larger extent). resolution
    is the long side in pixels (32..1024). Scope: names, else the selection,
    else all visible geometry.

    Returns a PNG (read `file`) plus, per object: colour, visible pixels, share of
    the silhouette, pixel bbox and visible width/height in scene units. An object
    with 0 visible pixels is fully hidden behind others in this view.

    Deterministic: one parallel ray per pixel against the evaluated meshes, so
    shading, antialiasing and viewport settings play no part, and the scene,
    selection and viewports are not touched. Cost grows with resolution^2.
    """
    names_l = si.check_names(names)
    view = si.check_view(view)
    si.check_int(resolution, "resolution", 32, 1024)
    si.check_float(padding, "padding", 0.0, 1.0)
    si.check_int(limit, "limit", 1, 500)
    si.check_int(max_nodes, "max_nodes", 1, 2000)
    rep = _id_view(names_l, view, resolution, padding, max_nodes)
    path = _save_png("object_id", rep["width"], rep["height"], si.id_image(rep["ids"], len(rep["nodes"])))
    return {
        "file": path, "mime_type": "image/png",
        "width": rep["width"], "height": rep["height"], "view": view,
        "pixel_size": rep["pixel_size"], "units": rep["units"],
        "objects": si.object_stats(rep, limit),
        "elapsed_ms": rep["elapsed_ms"],
        "notes": ["Background is white. Colours are stable for a given node order, not across calls with different scopes."],
    }


@mcp.tool()
def silhouette_compare(
    reference: str,
    names: list[str] | None = None,
    view: str = "front",
    fit: str = "height",
    crop: list[int] | None = None,
    threshold: int = 160,
    close_gaps: int = 1,
    largest_only: bool = True,
    resolution: int = 384,
) -> dict[str, Any]:
    """Compare the model's silhouette in an orthographic view with a reference drawing.

    reference: path to a PNG/JPG/BMP/TIFF on this machine, typically one elevation
    of a technical drawing. Use crop=[x,y,w,h] (source pixels) to pick one view off
    a sheet. Line drawings are filled: everything enclosed by the outer outline
    counts as inside. Images with transparency use alpha instead. threshold
    (0..255) decides what counts as ink; close_gaps dilates ink by that many
    pixels to seal small breaks; largest_only drops title blocks and text.

    fit decides how the two are scaled before overlap is measured:
      height  - same height, bottom-centre aligned (default; proportions show as width error)
      width   - same width, bottom-centre aligned
      stretch - both bboxes stretched onto each other (pure shape, ignores proportions)

    Returns IoU, the share of the union that only the model or only the reference
    covers, aspect ratios and their error, a 3x3 map of where they disagree, and a
    diff PNG (grey = agree, red = model only, blue = reference only). Also saves the
    model's object-ID view. Read only; nothing in the scene changes.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("reference must be a path to an image file")
    names_l = si.check_names(names)
    view = si.check_view(view)
    if fit not in {"height", "width", "stretch"}:
        raise ValueError("fit must be height, width or stretch")
    crop_l = si.check_crop(crop)
    si.check_int(threshold, "threshold", 1, 254)
    si.check_int(close_gaps, "close_gaps", 0, 5)
    si.check_int(resolution, "resolution", 64, 1024)
    if not isinstance(largest_only, bool):
        raise ValueError("largest_only must be true or false")

    img = si.parse_image_report(str(client.send_command(
        si.build_image_script(reference.strip(), crop_l, 1024)).get("result", "")))
    w, h = img["width"], img["height"]
    ink, source = si.ink_mask(img["lum"], img["alpha"], threshold)
    if source == "ink" and close_gaps:
        ink = si.dilate(ink, w, h, close_gaps)
    ref_mask = si.fill_enclosed(ink, w, h) if source == "ink" else ink
    components = 1
    if largest_only:
        ref_mask, components = si.keep_largest(ref_mask, w, h)
    ink_px = sum(ink) or 1
    filled_px = sum(ref_mask)
    warnings = []
    if source == "ink" and filled_px < ink_px * 1.5:
        warnings.append("The reference barely filled: its outline is probably open. Raise close_gaps, lower threshold, or crop tighter.")

    rep = _id_view(names_l, view, resolution, 0.02, 2000)
    model = si.fill_enclosed(bytearray(1 if v else 0 for v in rep["ids"]), rep["width"], rep["height"])
    cmp = si.compare_masks(model, rep["width"], rep["height"], ref_mask, w, h, fit)
    cw, ch, a, b = cmp.pop("canvas")
    diff_path = _save_png("silhouette_diff", cw, ch, si.diff_image(a, b))
    id_path = _save_png("object_id", rep["width"], rep["height"], si.id_image(rep["ids"], len(rep["nodes"])))
    mb = cmp.pop("model_bbox_px")
    rb = cmp.pop("reference_bbox_px")
    px = rep["pixel_size"]
    model_size = [round((mb[2] - mb[0]) * px, 4), round((mb[3] - mb[1]) * px, 4)]
    return {
        "view": view, "fit": fit, "units": rep["units"],
        **cmp,
        "model_size": model_size,
        "reference_bbox_px": list(rb),
        "reference_source": source,
        "reference_components": components,
        "reference_image": {"source_size": [img["source_width"], img["source_height"]], "analysed_size": [w, h], "crop": crop_l},
        "diff_file": diff_path,
        "object_id_file": id_path,
        "warnings": warnings,
        "notes": [
            "IoU 1.0 = identical silhouettes after fitting. aspect_error_pct > 0 means the model is wider for its height than the drawing.",
            "Only the outer silhouette is compared: holes fully enclosed by the outline are filled on both sides, interior lines are ignored.",
        ],
    }
