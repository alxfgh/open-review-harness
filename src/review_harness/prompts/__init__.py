"""Prompts live in files so a single hash can pin them to a frozen evaluation."""

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
    """Changes if any prompt changes, so results can be tied to the exact wording used."""
    digest = hashlib.sha256()
    for file in _files():
        digest.update(file.name.encode())
        digest.update(file.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return digest.hexdigest()
