from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class ByteTokenizer:
    """Deterministic byte-level fallback used only for local tests/debug smoke runs."""

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self) -> None:
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.vocab_size = 260

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [b + 4 for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        data = bytes([i - 4 for i in ids if 4 <= int(i) < 260])
        return data.decode("utf-8", errors="replace")

    def save_meta(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"type": "byte_fallback", "vocab_size": self.vocab_size}, f, indent=2)


class TokenizerWrapper:
    def __init__(self, tokenizer_path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The tokenizers package is required to load a trained tokenizer. "
                "Install dependencies manually with: python -m pip install -r requirements.txt"
            ) from exc

        self.path = Path(tokenizer_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.path}")
        self.tokenizer = Tokenizer.from_file(str(self.path))
        if self.tokenizer.decoder is None:
            # ByteLevel BPE stores spaces/newlines as visible byte-level symbols
            # such as Ġ/Ċ unless a matching decoder is attached.
            try:
                from tokenizers.decoders import ByteLevel as ByteLevelDecoder

                self.tokenizer.decoder = ByteLevelDecoder()
            except ImportError:
                pass
        self.pad_id = self._token_id("<pad>")
        self.bos_id = self._token_id("<bos>")
        self.eos_id = self._token_id("<eos>")
        self.unk_id = self._token_id("<unk>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required special token: {token}")
        return int(token_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(map(int, ids)), skip_special_tokens=True)


def load_tokenizer(path: str | Path | None, debug_byte_fallback: bool = False):
    if path is None or str(path).lower() in {"byte", "debug", "none"}:
        if debug_byte_fallback:
            return ByteTokenizer()
        raise ValueError("A tokenizer path is required unless debug_byte_fallback=True")
    return TokenizerWrapper(path)


def write_tokenizer_meta(tokenizer_path: str | Path, meta_path: str | Path, extra: dict | None = None) -> None:
    tok = TokenizerWrapper(tokenizer_path)
    payload = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tok.vocab_size,
        "pad_id": tok.pad_id,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "unk_id": tok.unk_id,
    }
    payload.update(extra or {})
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
