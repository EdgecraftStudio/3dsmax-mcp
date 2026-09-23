"""OpenCode configuration shared by the Python installer and Windows setup."""

import json
import os
import re
from pathlib import Path

SERVER = "3dsmax-mcp"


def opencode_config_path(directory: Path) -> Path:
    return next((directory / name for name in ("opencode.jsonc", "opencode.json", "config.json")
                 if (directory / name).exists()), directory / "opencode.json")


def resolve_opencode_config(home: Path) -> Path:
    if os.environ.get("OPENCODE_CONFIG"):
        return Path(os.environ["OPENCODE_CONFIG"])
    base = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "opencode"
    return opencode_config_path(Path(os.environ.get("OPENCODE_CONFIG_DIR", base)))


def update_opencode(text: str, entry: dict) -> bytes:
    """Edit only our MCP entry, preserving surrounding JSONC comments and layout."""
    string = r'"(?:[^"\\]|\\.)*"'
    masked = re.sub(string + r'|//[^\r\n]*|/\*[\s\S]*?\*/',
                    lambda m: m[0] if m[0].startswith('"') else
                    re.sub(r'[^\r\n]', ' ', m[0]), text)
    masked = re.sub(string + r'|,(\s*[}\]])',
                    lambda m: ' ' + m[1] if m[1] is not None else m[0], masked)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate OpenCode configuration key: {key}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    data = decoder.decode(masked)
    if not isinstance(data, dict) or not isinstance(data.get("mcp", {}), dict):
        raise ValueError("Invalid OpenCode MCP configuration")

    def skip_space(index):
        while masked[index].isspace():
            index += 1
        return index

    def edit(start, keys, value):
        index = skip_space(start + 1)
        while masked[index] != '}':
            key, end = decoder.raw_decode(masked, index)
            value_start = skip_space(skip_space(end) + 1)
            _, value_end = decoder.raw_decode(masked, value_start)
            if key == keys[0]:
                if len(keys) > 1:
                    return edit(value_start, keys[1:], value)
                return text[:value_start] + json.dumps(value, ensure_ascii=False) + text[value_end:]
            index = skip_space(value_end)
            if masked[index] == ',':
                index = skip_space(index + 1)
        for key in reversed(keys[1:]):
            value = {key: value}
        addition = json.dumps(keys[0]) + ': ' + json.dumps(value, ensure_ascii=False)
        comma = ',' if masked[skip_space(start + 1)] != '}' else ''
        return text[:start + 1] + '\n  ' + addition + comma + text[start + 1:]

    if data.get("mcp", {}).get(SERVER) == entry:
        return text.encode("utf-8")
    return edit(skip_space(0), ["mcp", SERVER], entry).encode("utf-8")
