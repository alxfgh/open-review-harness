"""Prompts added after the first manifests were frozen, pinned by their own hash.

They live outside `prompts/` on purpose: `prompts.prompts_sha()` is part of every frozen manifest,
and adding a file there would invalidate them. An amendment file (see `framing.py`) records this
folder's hash against a parent manifest's hash before any of these prompts is used.
"""

import hashlib
from importlib import resources


def _files() -> list:
    return sorted(
        (f for f in resources.files(__package__).iterdir() if f.name.endswith(".md")),
        key=lambda f: f.name,
    )


def load(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8").strip()


def prompts_sha() -> str:
    digest = hashlib.sha256()
    for file in _files():
        digest.update(file.name.encode())
        digest.update(file.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return digest.hexdigest()
