"""PDF resume text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pdfplumber
from PyPDF2 import PdfReader


@dataclass
class ResumeParseResult:
    text: str
    error: str | None = None


def _extract_with_pdfplumber(file_obj: BinaryIO) -> str:
    file_obj.seek(0)
    parts: list[str] = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def _extract_with_pypdf2(file_obj: BinaryIO) -> str:
    file_obj.seek(0)
    reader = PdfReader(file_obj)
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def extract_text_from_pdf(file_obj: BinaryIO | str | Path) -> ResumeParseResult:
    """Extract readable text using pdfplumber first and PyPDF2 as fallback."""
    try:
        if isinstance(file_obj, (str, Path)):
            with open(file_obj, "rb") as handle:
                return extract_text_from_pdf(handle)

        text = _extract_with_pdfplumber(file_obj)
        if not text:
            text = _extract_with_pypdf2(file_obj)

        if not text:
            return ResumeParseResult(
                text="",
                error="No readable text was found. The PDF may be scanned or image-only.",
            )

        return ResumeParseResult(text=text)
    except Exception as exc:
        try:
            text = _extract_with_pypdf2(file_obj)  # type: ignore[arg-type]
            if text:
                return ResumeParseResult(text=text)
        except Exception:
            pass
        return ResumeParseResult(text="", error=f"Unable to extract resume text: {exc}")
