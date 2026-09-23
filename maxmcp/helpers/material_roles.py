"""Normalized map-role reads: which file actually plays which role in a material.

`inspect_material_network` reports a role per wired slot and a flat file
manifest, but the two are not joined: a slot points at a wrapper node
(CoronaColorCorrect, CoronaNormal, a mix) and the bitmap hangs one or more
levels below it. Every caller then has to walk that graph itself.

This module does the walk once and answers the question directly: for each
role, the file behind it, the wrapper chain it passes through, and whether the
file name disagrees with the slot it is wired into.

Slot wiring is the ground truth. File names are only a hint, used to flag
disagreements, never to decide the role of a wired map.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..max_client import MaxClient
from ..tools.material_detection import _DEFAULT_CHANNEL_PATTERNS, _detect_texture_channel

# Native slot roles -> the short vocabulary used across this project.
SLOT_ROLE_TO_VOCAB: dict[str, str] = {
    "base_color": "dif",
    "basecolor": "dif",
    "diffuse": "dif",
    "color": "dif",
    "roughness": "rough",
    "glossiness": "gloss",
    "gloss": "gloss",
    "metalness": "met",
    "metallic": "met",
    "normal": "nrm",
    "bump": "bump",
    "displacement": "disp",
    "height": "disp",
    "ao": "ao",
    "occlusion": "ao",
    "opacity": "opac",
    "alpha": "opac",
    "ior": "ior",
    "refraction": "trans",
    "translucency": "trans",
    "transmission": "trans",
    "emission": "emis",
    "emissive": "emis",
    "self_illumination": "emis",
    "sss": "sss",
    "scattering": "sss",
    "reflection": "refl",
    "specular": "refl",
}

# Legacy and generic slots report role "map"; the slot's own name still says
# what it is. Longest key wins.
SLOT_NAME_TO_VOCAB: dict[str, str] = {
    "texmapdiffuse": "dif",
    "texmapreflectglossiness": "gloss",
    "texmapreflect": "refl",
    "texmaprefractglossiness": "gloss",
    "texmaprefract": "trans",
    "texmapbump": "bump",
    "texmapopacity": "opac",
    "texmapdisplacement": "disp",
    "texmapselfillum": "emis",
    "texmaptranslucency": "trans",
    "texmapfresnelior": "ior",
    "diffuse_map": "dif",
    "bump_map": "bump",
    "roughness_map": "rough",
    "normal_map": "nrm",
}

# Detected file-name channels -> the same vocabulary.
FILE_CHANNEL_TO_VOCAB: dict[str, str] = {
    "diffuse": "dif",
    "ao": "ao",
    "orm": "orm",
    "roughness": "rough",
    "glossiness": "gloss",
    "metallic": "met",
    "normal": "nrm",
    "bump": "bump",
    "displacement": "disp",
    "opacity": "opac",
    "emission": "emis",
    "translucency": "trans",
    "ior": "ior",
    "specular": "refl",
}

# Disagreements that are normal in production and must not be reported.
# AO wired into roughness is deliberately NOT here: it is a common substitute for
# a missing roughness map, but it is still worth surfacing so the artist decides.
_ACCEPTED_MISMATCH = {
    ("rough", "gloss"),   # Corona reads either, roughnessMode decides
    ("gloss", "rough"),
    ("nrm", "bump"),      # a normal map wired through a bump slot
    ("bump", "nrm"),
    ("dif", "ao"),        # AO multiplied into base color
    ("refl", "met"),
    ("disp", "bump"),
}

_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def _clean(value: Any) -> Any:
    """Strip control characters that some class names carry (a known upstream bug)."""
    return _CONTROL_CHARS.sub("", value) if isinstance(value, str) else value


def fetch_network(client: MaxClient, name: str, depth: int = 4, max_nodes: int = 120) -> dict[str, Any]:
    payload = {
        "name": name,
        "sub_material_index": 0,
        "depth": depth,
        "scope": "wired",
        "include_values": False,
        "verify_files": True,
        "max_nodes": max_nodes,
        "profile": "auto",
    }
    response = client.send_command(
        json.dumps(payload, separators=(",", ":")),
        cmd_type="native:inspect_material_network",
    )
    return json.loads(response.get("result", "{}"))


def _file_hint(path: str) -> tuple[str | None, str | None]:
    """Return (vocab_role, detected_channel) guessed from a file name.

    Single-letter aliases (``_s``, ``_g``, ``_m``) are too weak to contradict the
    wiring on their own: ``cloud_fabric_2_s`` is not evidence of a specular map.
    """
    detected = _detect_texture_channel(Path(path), _DEFAULT_CHANNEL_PATTERNS)
    if detected is None:
        return None, None
    channel, _, alias = detected
    if len(alias.strip("_")) < 3:
        return None, channel
    return FILE_CHANNEL_TO_VOCAB.get(channel), channel


def _descend_to_file(node_id: str, by_id: dict[str, dict], children: dict[str, list[dict]],
                     seen: set[str] | None = None) -> tuple[dict | None, list[str]]:
    """Walk down from a slot's node to the first node carrying a file."""
    seen = seen or set()
    if node_id in seen:
        return None, []
    seen.add(node_id)
    node = by_id.get(node_id)
    if node is None:
        return None, []
    chain = [_clean(node.get("class", "?"))]
    if node.get("files"):
        return node, chain
    for child in children.get(node_id, []):
        found, sub_chain = _descend_to_file(child["id"], by_id, children, seen)
        if found is not None:
            return found, chain + sub_chain
    return None, chain


def roles_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nodes = payload.get("nodes") or []
    by_id = {n["id"]: n for n in nodes if "id" in n}
    children: dict[str, list[dict]] = {}
    for node in nodes:
        parent = node.get("parentId")
        if parent:
            children.setdefault(parent, []).append(node)

    # inputType 2 is the native parameter slot and is authoritative; the UI-label
    # duplicates (inputType 0) only fill in roles the native pass did not name.
    best_slots: dict[str, dict] = {}
    for slot in payload.get("wiredSlots") or []:
        node_id = slot.get("nodeId")
        if not node_id:
            continue
        current = best_slots.get(node_id)
        if current is None or (slot.get("inputType") == 2 and current.get("inputType") != 2):
            best_slots[node_id] = slot

    entries: list[dict[str, Any]] = []
    for node_id, slot in best_slots.items():
        raw_role = str(slot.get("role") or "").lower()
        slot_name = str(_clean(slot.get("slot")) or "")
        role = SLOT_ROLE_TO_VOCAB.get(raw_role)
        if role is None:
            key = slot_name.lower().replace(" ", "").replace(".", "")
            role = next((v for k, v in sorted(SLOT_NAME_TO_VOCAB.items(),
                                              key=lambda kv: -len(kv[0])) if key.startswith(k)), None)
        if role is None:
            # "map" on a Multi/Sub-Object entry is a sub-material, not a texture role.
            role = "submaterial" if slot_name.lower().startswith("materiallist") else (raw_role or "unknown")
        file_node, chain = _descend_to_file(node_id, by_id, children)
        entry: dict[str, Any] = {
            "role": role,
            "slot": _clean(slot.get("slot")),
            "slot_role_raw": raw_role,
            "node_chain": chain,
            "file": None,
            "node_name": None,
            "file_hint": None,
            "mismatch": False,
        }
        if file_node is not None:
            first_file = (file_node.get("files") or [{}])[0]
            path = first_file.get("path")
            entry["file"] = path
            entry["node_name"] = _clean(file_node.get("name"))
            if path:
                hint, channel = _file_hint(path)
                entry["file_hint"] = hint or channel
                comparable = role not in {"submaterial", "map", "unknown", ""}
                if comparable and hint and hint != role and (role, hint) not in _ACCEPTED_MISMATCH:
                    entry["mismatch"] = True
        entries.append(entry)

    entries.sort(key=lambda e: (e["role"], e["slot"] or ""))
    root = payload.get("root") or {}
    return {
        "material": _clean(payload.get("query")),
        "class": _clean(root.get("class")),
        "renderer": root.get("rendererProfile"),
        "owner": _clean(payload.get("owner")),
        "texture_folder": (payload.get("hints") or {}).get("textureFolderGuess"),
        "roles": entries,
        "missing_files": [
            _clean(f.get("path"))
            for f in (payload.get("fileManifest") or [])
            if f.get("exists") is False
        ],
        "mismatches": [e for e in entries if e["mismatch"]],
        "issues": payload.get("issues") or [],
    }


def scene_material_names(client: MaxClient, limit: int = 200) -> list[str]:
    maxscript = f"""(
    local out = ""
    local shown = 0
    for m in sceneMaterials while shown < {int(limit)} do (
        out += m.name + "\\n"
        shown += 1
    )
    out
)"""
    response = client.send_command(maxscript)
    return [line.strip() for line in str(response.get("result", "")).splitlines() if line.strip()]
