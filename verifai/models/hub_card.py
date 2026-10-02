"""What a Hub repository's model card says about itself, in machine-readable form.

A model card is a `README.md` whose YAML header carries the fields the Hub
indexes — `license`, `datasets`, `tags`, `base_model`, `library_name`,
`pipeline_tag` — followed by prose. Only the header and the first `# ` heading
are read: they are facts the author declared in a fixed place, which a page can
quote with their source. The prose is not parsed; what it says is linked to,
never paraphrased.

Read when a scenario runs, at the revision it pins, and stored in
`report.meta["hub_card"]`, so the showcase quotes the card as it was at the
commit that was evaluated and never contacts the Hub itself.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# The header fields a page may quote. Anything else in the header is left out
# rather than passed through: the showcase should not render keys it has no
# wording for.
FIELDS = ("license", "datasets", "tags", "base_model", "library_name", "pipeline_tag")
LISTS = ("datasets", "tags", "base_model")


def parse_card(text: str) -> dict[str, Any]:
    """The header fields and the title of one `README.md`."""
    header: dict[str, Any] = {}
    body = text
    m = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, flags=re.S)
    if m:
        header = yaml.safe_load(m.group(1)) or {}
        body = text[m.end():]
    out: dict[str, Any] = {"present": True}
    for k in FIELDS:
        v = header.get(k)
        if v in (None, "", []):
            continue
        # The Hub accepts one string where it means a list; store the list.
        out[k] = [v] if k in LISTS and isinstance(v, str) else v
    title = re.search(r"^# +(.+?)\s*$", body, flags=re.M)
    if title:
        out["title"] = title.group(1).strip()
    return out


def read_hub_card(repo_id: str, revision: str | None) -> dict[str, Any]:
    """The card of `repo_id` at `revision`, or `{"present": False}` when it has none.

    Only a file the Hub says does not exist counts as absent. Any other failure
    is raised, as for the processor file: reporting "no card" because the lookup
    failed would publish a missing licence that is not missing.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        path = hf_hub_download(repo_id=repo_id, filename="README.md", revision=revision)
    except EntryNotFoundError:
        return {"present": False, "repo_id": repo_id, "revision": revision}
    card = parse_card(Path(path).read_text(encoding="utf-8"))
    return {**card, "repo_id": repo_id, "revision": revision}
