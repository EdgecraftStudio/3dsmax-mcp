"""Object-ID views and silhouette comparison against a reference drawing.

Everything that decides a result lives here and is plain Python, so it can be
tested without 3ds Max. Max does two cheap things:

* ``build_id_script``: casts one parallel ray per pixel of an orthographic view
  through ``RayMeshGridIntersect`` and returns which node is nearest at each
  pixel as a run-length report. No viewport, render or scene state is touched.
* ``build_image_script``: decodes a reference image (PNG, JPG, BMP, TIFF, GIF)
  with .NET, optionally crops it, scales it down and returns raw BGRA as base64.
  No file is written, so both work with ``safe_mode=true``.

Masks are flat ``bytearray`` rows (1 = inside), row-major, top row first.
"""
from __future__ import annotations

import base64
import colorsys
import math
import struct
import zlib
from collections import deque
from typing import Any

from .maxscript import safe_string

# Orthographic views as Max names them: (view direction, screen right, screen up).
VIEWS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "front": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "back": ((0, -1, 0), (-1, 0, 0), (0, 0, 1)),
    "left": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "right": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "top": ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
    "bottom": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
}

BACKGROUND = (255, 255, 255)


# ----------------------------------------------------------------- validation

def check_view(view: Any) -> str:
    if not isinstance(view, str) or view.lower() not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}")
    return view.lower()


def check_int(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}")
    return value


def check_float(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be a number from {low} to {high}")
    return float(value)


def check_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and v for v in value):
        raise ValueError("names must be a list of node names")
    return list(dict.fromkeys(value))


def check_crop(crop: Any) -> list[int] | None:
    if crop is None:
        return None
    if (not isinstance(crop, (list, tuple)) or len(crop) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in crop)
            or crop[0] < 0 or crop[1] < 0 or crop[2] < 1 or crop[3] < 1):
        raise ValueError("crop must be [x, y, width, height] in source-image pixels")
    return list(crop)


# ------------------------------------------------------------- MAXScript: ids

def _p3(v: tuple[int, int, int]) -> str:
    return "[" + ",".join(str(c) for c in v) + "]"


def build_id_script(names: list[str], view: str, resolution: int, padding: float, max_nodes: int) -> str:
    d, r, u = VIEWS[view]
    names_ms = "#(" + ", ".join(f'"{safe_string(n)}"' for n in names) + ")"
    return _ID_TEMPLATE % {
        "names": names_ms, "d": _p3(d), "r": _p3(r), "u": _p3(u),
        "res": resolution, "pad": format(padding, ".6g"), "max_nodes": max_nodes,
    }


_ID_TEMPLATE = r'''(
    fn siIsMesh n = isValidNode n and (isKindOf n GeometryClass) and not (isKindOf n TargetObject) and (canConvertTo n TriMeshGeometry)
    local rm = undefined
    local result = undefined
    try (
        local t0 = timeStamp()
        local namesIn = %(names)s
        local nodes = #()
        if namesIn.count > 0 then (
            for nm in namesIn do (
                local m = getNodeByName nm exact:true all:true
                if m.count != 1 do throw ("Node name must resolve uniquely: " + nm)
                if not (siIsMesh m[1]) do throw ("Not a mesh-convertible geometry node: " + nm)
                append nodes m[1]
            )
        ) else if selection.count > 0 then (
            nodes = for n in selection where siIsMesh n collect n
        ) else (
            nodes = for n in geometry where not n.isHiddenInVpt and siIsMesh n collect n
        )
        if nodes.count == 0 do throw "No mesh nodes to draw (pass names, select nodes, or show some geometry)"
        if nodes.count > %(max_nodes)s do throw ("Too many nodes (" + nodes.count as string + "); pass names or raise max_nodes")
        local dv = %(d)s, rv = %(r)s, uv = %(u)s
        local minR = 1e30, maxR = -1e30, minU = 1e30, maxU = -1e30, minD = 1e30, maxD = -1e30
        local rects = #()
        for n in nodes do (
            local a = n.min, b = n.max
            local nr0 = 1e30, nr1 = -1e30, nu0 = 1e30, nu1 = -1e30
            for c in #([a.x,a.y,a.z],[b.x,a.y,a.z],[a.x,b.y,a.z],[b.x,b.y,a.z],[a.x,a.y,b.z],[b.x,a.y,b.z],[a.x,b.y,b.z],[b.x,b.y,b.z]) do (
                local pr = dot c rv, pu = dot c uv, pd = dot c dv
                if pr < nr0 do nr0 = pr
                if pr > nr1 do nr1 = pr
                if pu < nu0 do nu0 = pu
                if pu > nu1 do nu1 = pu
                if pd < minD do minD = pd
                if pd > maxD do maxD = pd
            )
            append rects #(nr0, nr1, nu0, nu1)
            if nr0 < minR do minR = nr0
            if nr1 > maxR do maxR = nr1
            if nu0 < minU do minU = nu0
            if nu1 > maxU do maxU = nu1
        )
        local span = amax (maxR - minR) (maxU - minU)
        if span <= 0 do throw "Nodes have no extent in this view"
        local pad = %(pad)s * span
        minR -= pad; maxR += pad; minU -= pad; maxU += pad
        local px = (amax (maxR - minR) (maxU - minU)) / %(res)s
        local W = amax 1 ((ceil ((maxR - minR) / px)) as integer)
        local H = amax 1 ((ceil ((maxU - minU) / px)) as integer)
        local depth0 = minD - (maxD - minD) - px
        local total = W * H
        local ids = for i = 1 to total collect 0
        local dist = for i = 1 to total collect 1e30
        for k = 1 to nodes.count do (
            local rc = rects[k]
            local x0 = amax 0 ((floor ((rc[1] - minR) / px)) as integer)
            local x1 = amin (W - 1) ((ceil ((rc[2] - minR) / px)) as integer)
            local y0 = amax 0 ((floor ((maxU - rc[4]) / px)) as integer)
            local y1 = amin (H - 1) ((ceil ((maxU - rc[3]) / px)) as integer)
            rm = RayMeshGridIntersect()
            rm.Initialize 20
            rm.addNode nodes[k]
            rm.buildGrid()
            for y = y0 to y1 do (
                local cu = maxU - (y + 0.5) * px
                local rowBase = y * W + 1
                for x = x0 to x1 do (
                    local p = rv * (minR + (x + 0.5) * px) + uv * cu + dv * depth0
                    if (rm.intersectRay p dv true) > 0 do (
                        local t = rm.getHitDist (rm.getClosestHit())
                        local i = rowBase + x
                        if t < dist[i] do (dist[i] = t; ids[i] = k)
                    )
                )
            )
            rm.free()
            rm = undefined
        )
        local ss = stringStream ""
        format "V|%%|%%|%%|%%|%%|%%|%%\n" W H (formattedPrint px format:".9g") (formattedPrint minR format:".9g") (formattedPrint maxU format:".9g") (units.SystemType as string) (timeStamp() - t0) to:ss
        for k = 1 to nodes.count do (
            format "N|%%|%%|%%\n" k (formattedPrint ((getHandleByAnim nodes[k]) as integer64) format:"d") ((dotNetClass "System.Convert").ToBase64String ((dotNetClass "System.Text.Encoding").UTF8.GetBytes nodes[k].name)) to:ss
        )
        for y = 0 to H - 1 do (
            format "R|" to:ss
            local base = y * W
            local cur = ids[base + 1], run = 0
            for x = 1 to W do (
                local v = ids[base + x]
                if v == cur then run += 1 else (format "%%,%%;" cur run to:ss; cur = v; run = 1)
            )
            format "%%,%%\n" cur run to:ss
        )
        format "END\n" to:ss
        result = ss as string
    ) catch (
        result = "__ERROR__|" + (getCurrentException() as string)
    )
    if rm != undefined do try (rm.free()) catch ()
    result
)'''


def parse_id_report(raw: str) -> dict[str, Any]:
    if raw.startswith("__ERROR__|"):
        raise RuntimeError(raw.split("|", 1)[1])
    lines = [ln for ln in raw.replace("\r", "").split("\n") if ln]
    try:
        head = lines[0].split("|")
        if head[0] != "V":
            raise ValueError("missing header")
        width, height = int(head[1]), int(head[2])
        pixel, min_r, max_u = float(head[3]), float(head[4]), float(head[5])
        units, elapsed = head[6], int(head[7])
        nodes: dict[int, dict[str, Any]] = {}
        ids: list[int] = []
        rows = 0
        done = False
        for line in lines[1:]:
            if line.startswith("N|"):
                _, k, handle, name = line.split("|", 3)
                nodes[int(k)] = {"name": base64.b64decode(name, validate=True).decode("utf-8"), "handle": int(handle)}
            elif line.startswith("R|"):
                count = 0
                for run in line[2:].split(";"):
                    value, length = run.split(",")
                    n = int(length)
                    ids.extend([int(value)] * n)
                    count += n
                if count != width:
                    raise ValueError("row width mismatch")
                rows += 1
            elif line == "END":
                done = True
            else:
                raise ValueError("unknown record")
        if not done or rows != height:
            raise ValueError("report truncated")
    except (ValueError, IndexError, UnicodeError) as exc:
        raise RuntimeError(f"Invalid object-ID report: {exc}") from exc
    return {"width": width, "height": height, "pixel_size": pixel, "origin_right": min_r,
            "origin_up": max_u, "units": units, "elapsed_ms": elapsed, "nodes": nodes, "ids": ids}


# ----------------------------------------------------------- MAXScript: image

def build_image_script(path: str, crop: list[int] | None, max_side: int) -> str:
    crop_ms = "undefined" if crop is None else "#(" + ",".join(str(c) for c in crop) + ")"
    return _IMAGE_TEMPLATE % {"path": safe_string(path), "crop": crop_ms, "max_side": max_side}


_IMAGE_TEMPLATE = r'''(
    local src = undefined, dst = undefined
    local result = undefined
    try (
        local path = "%(path)s"
        if not (doesFileExist path) do throw ("Reference image not found: " + path)
        src = dotNetObject "System.Drawing.Bitmap" path
        local sw = src.Width, sh = src.Height
        local crop = %(crop)s
        local cx = 0, cy = 0, cw = sw, ch = sh
        if crop != undefined do (
            cx = crop[1]; cy = crop[2]; cw = crop[3]; ch = crop[4]
            if cx + cw > sw or cy + ch > sh do throw ("crop lies outside the " + sw as string + "x" + sh as string + " image")
        )
        local s = amin 1.0 ((%(max_side)s as float) / (amax cw ch))
        local tw = amax 1 ((cw * s + 0.5) as integer), th = amax 1 ((ch * s + 0.5) as integer)
        local fmt = (dotNetClass "System.Drawing.Imaging.PixelFormat").Format32bppArgb
        dst = dotNetObject "System.Drawing.Bitmap" tw th fmt
        local g = (dotNetClass "System.Drawing.Graphics").FromImage dst
        g.Clear ((dotNetClass "System.Drawing.Color").Transparent)
        g.InterpolationMode = (dotNetClass "System.Drawing.Drawing2D.InterpolationMode").HighQualityBilinear
        g.PixelOffsetMode = (dotNetClass "System.Drawing.Drawing2D.PixelOffsetMode").HighQuality
        local unitPx = (dotNetClass "System.Drawing.GraphicsUnit").Pixel
        g.DrawImage src (dotNetObject "System.Drawing.Rectangle" 0 0 tw th) (dotNetObject "System.Drawing.Rectangle" cx cy cw ch) unitPx
        g.Dispose()
        local bd = dst.LockBits (dotNetObject "System.Drawing.Rectangle" 0 0 tw th) ((dotNetClass "System.Drawing.Imaging.ImageLockMode").ReadOnly) fmt
        local n = bd.Stride * th
        local bytes = dotNetObject "System.Byte[]" n
        (dotNetClass "System.Runtime.InteropServices.Marshal").Copy bd.Scan0 bytes 0 n
        local stride = bd.Stride
        dst.UnlockBits bd
        result = "I|" + sw as string + "|" + sh as string + "|" + tw as string + "|" + th as string + "|" + stride as string + "|" + ((dotNetClass "System.Convert").ToBase64String bytes)
    ) catch (
        result = "__ERROR__|" + (getCurrentException() as string)
    )
    if dst != undefined do try (dst.Dispose()) catch ()
    if src != undefined do try (src.Dispose()) catch ()
    result
)'''


def parse_image_report(raw: str) -> dict[str, Any]:
    if raw.startswith("__ERROR__|"):
        raise RuntimeError(raw.split("|", 1)[1])
    try:
        tag, sw, sh, w, h, stride, b64 = raw.strip().split("|", 6)
        if tag != "I":
            raise ValueError("missing header")
        w, h, stride = int(w), int(h), int(stride)
        data = base64.b64decode(b64, validate=True)
        if len(data) != stride * h:
            raise ValueError("pixel data size mismatch")
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError(f"Invalid image report: {exc}") from exc
    lum = bytearray(w * h)
    alpha = bytearray(w * h)
    for y in range(h):
        row = y * stride
        for x in range(w):
            i = row + 4 * x
            b, g, r, a = data[i], data[i + 1], data[i + 2], data[i + 3]
            lum[y * w + x] = (299 * r + 587 * g + 114 * b) // 1000
            alpha[y * w + x] = a
    return {"source_width": int(sw), "source_height": int(sh), "width": w, "height": h, "lum": lum, "alpha": alpha}


# ------------------------------------------------------------------ masks

def ink_mask(lum: bytearray, alpha: bytearray, threshold: int) -> tuple[bytearray, str]:
    """1 where the drawing has ink. Uses alpha when the image has transparency."""
    transparent = sum(1 for a in alpha if a < 128)
    if transparent > len(alpha) // 100:
        return bytearray(1 if a >= 128 else 0 for a in alpha), "alpha"
    return bytearray(1 if (a >= 128 and v < threshold) else 0 for v, a in zip(lum, alpha)), "ink"


def dilate(mask: bytearray, w: int, h: int, radius: int) -> bytearray:
    out = bytearray(mask)
    for _ in range(radius):
        src = bytes(out)
        for y in range(h):
            row = y * w
            for x in range(w):
                if src[row + x]:
                    continue
                if ((x > 0 and src[row + x - 1]) or (x < w - 1 and src[row + x + 1])
                        or (y > 0 and src[row - w + x]) or (y < h - 1 and src[row + w + x])):
                    out[row + x] = 1
    return out


def fill_enclosed(ink: bytearray, w: int, h: int) -> bytearray:
    """Everything not reachable from the image border without crossing ink."""
    outside = bytearray(w * h)
    queue: deque[int] = deque()
    for x in range(w):
        for i in (x, (h - 1) * w + x):
            if not ink[i] and not outside[i]:
                outside[i] = 1
                queue.append(i)
    for y in range(h):
        for i in (y * w, y * w + w - 1):
            if not ink[i] and not outside[i]:
                outside[i] = 1
                queue.append(i)
    while queue:
        i = queue.popleft()
        x = i % w
        for j in ((i - 1) if x > 0 else -1, (i + 1) if x < w - 1 else -1, i - w, i + w):
            if 0 <= j < w * h and not outside[j] and not ink[j]:
                outside[j] = 1
                queue.append(j)
    return bytearray(0 if o else 1 for o in outside)


def bbox(mask: bytearray, w: int, h: int) -> tuple[int, int, int, int] | None:
    xs0, ys0, xs1, ys1 = w, h, -1, -1
    for y in range(h):
        row = mask[y * w:(y + 1) * w]
        if 1 not in row:
            continue
        ys0 = min(ys0, y)
        ys1 = y
        xs0 = min(xs0, row.index(1))
        xs1 = max(xs1, w - 1 - row[::-1].index(1))
    return None if xs1 < 0 else (xs0, ys0, xs1 + 1, ys1 + 1)


def keep_largest(mask: bytearray, w: int, h: int) -> tuple[bytearray, int]:
    """Largest 4-connected component; drops title blocks, text and stray marks."""
    label = [0] * (w * h)
    best, best_size, count, current = 0, 0, 0, 0
    for start in range(w * h):
        if not mask[start] or label[start]:
            continue
        current += 1
        count += 1
        size = 0
        queue = deque([start])
        label[start] = current
        while queue:
            i = queue.popleft()
            size += 1
            x = i % w
            for j in ((i - 1) if x > 0 else -1, (i + 1) if x < w - 1 else -1, i - w, i + w):
                if 0 <= j < w * h and mask[j] and not label[j]:
                    label[j] = current
                    queue.append(j)
        if size > best_size:
            best, best_size = current, size
    return bytearray(1 if v == best else 0 for v in label), count


def resample(mask: bytearray, w: int, h: int, box: tuple[int, int, int, int],
             out_w: int, out_h: int, off_x: int, off_y: int, canvas_w: int, canvas_h: int) -> bytearray:
    """Nearest-neighbour copy of mask[box] into an out_w x out_h rect at (off_x, off_y)."""
    out = bytearray(canvas_w * canvas_h)
    x0, y0, x1, y1 = box
    sx = (x1 - x0) / out_w
    sy = (y1 - y0) / out_h
    for oy in range(out_h):
        cy = off_y + oy
        if not 0 <= cy < canvas_h:
            continue
        src_y = min(h - 1, y0 + int((oy + 0.5) * sy))
        row = src_y * w
        base = cy * canvas_w
        for ox in range(out_w):
            cx = off_x + ox
            if 0 <= cx < canvas_w and mask[row + min(w - 1, x0 + int((ox + 0.5) * sx))]:
                out[base + cx] = 1
    return out


def compare_masks(model: bytearray, mw: int, mh: int, ref: bytearray, rw: int, rh: int,
                  fit: str, grid: int = 256) -> dict[str, Any]:
    mb, rb = bbox(model, mw, mh), bbox(ref, rw, rh)
    if mb is None:
        raise RuntimeError("The model silhouette is empty in this view")
    if rb is None:
        raise RuntimeError("No silhouette found in the reference image; check threshold or crop")
    m_w, m_h = mb[2] - mb[0], mb[3] - mb[1]
    r_w, r_h = rb[2] - rb[0], rb[3] - rb[1]
    # model bbox fills the grid on its long side; the reference is scaled by `fit`
    k = grid / max(m_w, m_h)
    gm_w, gm_h = max(1, round(m_w * k)), max(1, round(m_h * k))
    if fit == "stretch":
        gr_w, gr_h = gm_w, gm_h
    elif fit == "width":
        s = gm_w / r_w
        gr_w, gr_h = gm_w, max(1, round(r_h * s))
    else:  # height
        s = gm_h / r_h
        gr_w, gr_h = max(1, round(r_w * s)), gm_h
    cw, ch = max(gm_w, gr_w) + 4, max(gm_h, gr_h) + 4
    # bottom-centre alignment: furniture and fixtures stand on their base
    m_off = ((cw - gm_w) // 2, ch - 2 - gm_h)
    r_off = ((cw - gr_w) // 2, ch - 2 - gr_h)
    a = resample(model, mw, mh, mb, gm_w, gm_h, m_off[0], m_off[1], cw, ch)
    b = resample(ref, rw, rh, rb, gr_w, gr_h, r_off[0], r_off[1], cw, ch)
    both = sum(1 for p, q in zip(a, b) if p and q)
    only_m = sum(1 for p, q in zip(a, b) if p and not q)
    only_r = sum(1 for p, q in zip(a, b) if q and not p)
    union = both + only_m + only_r
    # where the disagreement sits, in a 3x3 grid over the canvas
    regions: dict[str, dict[str, float]] = {}
    names_y, names_x = ("top", "middle", "bottom"), ("left", "centre", "right")
    for gy in range(3):
        for gx in range(3):
            x0, x1 = cw * gx // 3, cw * (gx + 1) // 3
            y0, y1 = ch * gy // 3, ch * (gy + 1) // 3
            em = er = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    i = y * cw + x
                    if a[i] and not b[i]:
                        em += 1
                    elif b[i] and not a[i]:
                        er += 1
            if union and (em + er) / union >= 0.01:
                regions[f"{names_y[gy]}-{names_x[gx]}"] = {
                    "model_only": round(100 * em / union, 1), "reference_only": round(100 * er / union, 1)}
    return {
        "iou": round(both / union, 4) if union else 0.0,
        "model_only_pct": round(100 * only_m / union, 2) if union else 0.0,
        "reference_only_pct": round(100 * only_r / union, 2) if union else 0.0,
        "model_aspect": round(m_w / m_h, 4),
        "reference_aspect": round(r_w / r_h, 4),
        "aspect_error_pct": round(100 * ((m_w / m_h) / (r_w / r_h) - 1), 2),
        "regions": regions,
        "canvas": (cw, ch, a, b),
        "model_bbox_px": mb, "reference_bbox_px": rb,
    }


# ------------------------------------------------------------------- images

def palette(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        hue = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def png_bytes(width: int, height: int, rgb: bytes | bytearray) -> bytes:
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y * stride:(y + 1) * stride])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def id_image(ids: list[int], count: int) -> bytearray:
    colors = [BACKGROUND] + palette(count)
    out = bytearray()
    for v in ids:
        out.extend(colors[v])
    return out


def diff_image(a: bytearray, b: bytearray) -> bytearray:
    """Grey = agree, red = model only, blue = reference only, white = neither."""
    out = bytearray()
    for p, q in zip(a, b):
        out.extend((150, 150, 150) if p and q else (220, 40, 40) if p else (40, 90, 230) if q else (255, 255, 255))
    return out


def object_stats(report: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    w, h, px = report["width"], report["height"], report["pixel_size"]
    ids = report["ids"]
    acc: dict[int, list[int]] = {}
    for i, v in enumerate(ids):
        if not v:
            continue
        x, y = i % w, i // w
        s = acc.get(v)
        if s is None:
            acc[v] = [1, x, y, x, y]
        else:
            s[0] += 1
            if x < s[1]: s[1] = x
            if y < s[2]: s[2] = y
            if x > s[3]: s[3] = x
            if y > s[4]: s[4] = y
    covered = sum(s[0] for s in acc.values()) or 1
    colors = palette(len(report["nodes"]))
    rows = []
    for k, node in report["nodes"].items():
        s = acc.get(k)
        rows.append({
            "name": node["name"], "handle": node["handle"],
            "color": "#%02x%02x%02x" % colors[k - 1],
            "visible_pixels": s[0] if s else 0,
            "share_of_silhouette_pct": round(100 * s[0] / covered, 2) if s else 0.0,
            "bbox_px": [s[1], s[2], s[3] + 1, s[4] + 1] if s else None,
            "visible_size": [round((s[3] + 1 - s[1]) * px, 4), round((s[4] + 1 - s[2]) * px, 4)] if s else None,
        })
    rows.sort(key=lambda r: -r["visible_pixels"])
    return rows[:limit]
