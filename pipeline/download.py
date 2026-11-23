"""Download pinned dataset files and verify each against its published sha256.

    python -m pipeline.download                          # FineWeb-Edu shard 0
    python -m pipeline.download oasst2-train dolly-15k   # any names from SOURCES

Pinned to one dataset commit, so the corpus can be rebuilt byte for byte later.
Interrupted downloads resume with an HTTP Range request.
"""

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_CHUNK_BYTES = 8 * 1024 * 1024
_USER_AGENT = "tinygpt-data-pipeline/0.1"


@dataclass(frozen=True)
class PinnedFile:
    repository: str
    revision: str
    path: str
    size_bytes: int
    sha256: str
    licence: str
    local_name: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{self.path}"
        )


_OASST2 = ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75", "Apache-2.0")
_SQUAD = ("rajpurkar/squad", "7b6d24c440a36b6815f21b70d25016731768db1f", "CC BY-SA 4.0")

# Revisions and hashes were read from the Hugging Face API on 2026-11-23.
SOURCES: dict[str, PinnedFile] = {
    "fineweb-edu-0": PinnedFile(
        repository="HuggingFaceFW/fineweb-edu",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        path="sample/10BT/000_00000.parquet",
        size_bytes=2_152_819_114,
        sha256="b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
        licence="ODC-By 1.0",
        local_name="fineweb-edu-000_00000.parquet",
    ),
    "oasst2-train": PinnedFile(
        *_OASST2[:2],
        path="data/train-00000-of-00001-88ba0162028a73fc.parquet",
        size_bytes=63_494_018,
        sha256="9638227b41b0a4fb69afa5d5c4b52e3fb6abfd56c9870d81bae32c9981e769b1",
        licence=_OASST2[2],
        local_name="oasst2-train.parquet",
    ),
    "oasst2-validation": PinnedFile(
        *_OASST2[:2],
        path="data/validation-00000-of-00001-1deeef95c3248fe0.parquet",
        size_bytes=3_180_111,
        sha256="fd46c63babc0756909a957ee40975bcf5610d58c5d1ab5ff2c57c0f57be1f273",
        licence=_OASST2[2],
        local_name="oasst2-validation.parquet",
    ),
    "dolly-15k": PinnedFile(
        repository="databricks/databricks-dolly-15k",
        revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        path="databricks-dolly-15k.jsonl",
        size_bytes=13_085_339,
        sha256="2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
        licence="CC BY-SA 3.0",
        local_name="dolly-15k.jsonl",
    ),
    "squad-train": PinnedFile(
        *_SQUAD[:2],
        path="plain_text/train-00000-of-00001.parquet",
        size_bytes=14_458_314,
        sha256="ea7f52bac024f6b1bdc7aaa2a4ee302cba8c2fdc8d4a235cf18a9a5196b6175b",
        licence=_SQUAD[2],
        local_name="squad-train.parquet",
    ),
    "squad-validation": PinnedFile(
        *_SQUAD[:2],
        path="plain_text/validation-00000-of-00001.parquet",
        size_bytes=1_819_889,
        sha256="8c6646d36bd5a95061e076788cf3161d11f6f3e7d625dac7a83bbed0a49f69f7",
        licence=_SQUAD[2],
        local_name="squad-validation.parquet",
    ),
}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_remaining_bytes(pinned: PinnedFile, partial_path: Path) -> None:
    already_downloaded = partial_path.stat().st_size if partial_path.exists() else 0
    if already_downloaded >= pinned.size_bytes:
        return

    request = urllib.request.Request(
        pinned.url,
        headers={"User-Agent": _USER_AGENT, "Range": f"bytes={already_downloaded}-"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        # 206 means the server honoured the Range header; 200 means it sent the
        # whole file, so appending would corrupt it.
        if already_downloaded and response.status != 206:
            raise RuntimeError(f"Server ignored resume request (HTTP {response.status})")
        with partial_path.open("ab") as handle:
            while chunk := response.read(_CHUNK_BYTES):
                handle.write(chunk)


def download_file(pinned: PinnedFile, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / pinned.local_name
    if final_path.exists():
        print(f"Already downloaded: {final_path}")
        return final_path

    partial_path = final_path.with_suffix(final_path.suffix + ".partial")
    _fetch_remaining_bytes(pinned, partial_path)

    actual_size = partial_path.stat().st_size
    if actual_size != pinned.size_bytes:
        raise RuntimeError(f"Size mismatch: expected {pinned.size_bytes}, got {actual_size}")
    actual_sha256 = sha256_of_file(partial_path)
    if actual_sha256 != pinned.sha256:
        partial_path.unlink()
        raise RuntimeError(f"sha256 mismatch: expected {pinned.sha256}, got {actual_sha256}")

    partial_path.replace(final_path)
    print(f"Downloaded and verified {final_path} ({pinned.licence})")
    return final_path


def main(argv: list[str]) -> None:
    names = argv or ["fineweb-edu-0"]
    unknown = [name for name in names if name not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown sources {unknown}; choose from {sorted(SOURCES)}")
    for name in names:
        download_file(SOURCES[name], Path("data/raw"))


if __name__ == "__main__":
    main(sys.argv[1:])
