"""Which file plays which role in a material, resolved from the wiring."""

from __future__ import annotations

from typing import Any

from ..server import mcp, client
from ..helpers import material_roles as impl


@mcp.tool()
def material_roles(names: list[str] | None = None, scan_scene: bool = False,
                   limit: int = 25, only_problems: bool = False) -> dict:
    """Resolve each material's maps to roles from the wiring, not from file names.

    Use when: you need to know what a material actually does — which file is the
    base colour, the roughness, the normal — before editing, converting or
    rebuilding it, or to audit a scene for maps wired into the wrong slot and for
    missing texture files.
    Not when: you want the full node graph with parameters — use
    inspect_material_network.
    Walks through wrapper nodes (colour correct, CoronaNormal, mixes) down to the
    bitmap, so the answer is one row per role with its file. File names are used
    only as a hint and a disagreement is reported, never used to decide the role.
    scan_scene reads the scene's own material list; only_problems returns just
    the materials with a mismatch or a missing file.
    """
    if scan_scene:
        wanted = impl.scene_material_names(client, limit=max(1, min(int(limit), 200)))
    else:
        wanted = [n for n in (names or []) if str(n).strip()]
    if not wanted:
        raise ValueError("Supply names=[...] or scan_scene=true.")
    wanted = wanted[:max(1, min(int(limit), 200))]

    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for name in wanted:
        try:
            payload = impl.fetch_network(client, str(name))
        except Exception as exc:  # a broken material must not kill the whole scan
            failed.append({"material": str(name), "error": str(exc)})
            continue
        if not payload.get("ok", True):
            failed.append({"material": str(name), "error": str(payload.get("error") or "inspect failed")})
            continue
        resolved = impl.roles_from_payload(payload)
        if only_problems and not resolved["mismatches"] and not resolved["missing_files"]:
            continue
        results.append(resolved)

    return {
        "materials": results,
        "checked": len(wanted),
        "returned": len(results),
        "with_mismatch": sum(1 for r in results if r["mismatches"]),
        "with_missing_files": sum(1 for r in results if r["missing_files"]),
        "failed": failed,
    }
