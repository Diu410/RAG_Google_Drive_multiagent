"""Formula-preserving document extraction and conservative text preparation."""
from __future__ import annotations

import logging
import os
import re
import shutil
import unicodedata
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from extraction_limits import (bounded_extraction, check_extraction_budget, ocr_timeout,
    MAX_DOCX_XML_BYTES, MAX_PDF_PAGES, MAX_OCR_PIXELS, ExtractionLimitError, IncompleteExtractionError)

logger = logging.getLogger(__name__)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}
MATH_SYMBOLS = "∑∫√±×÷≤≥≠≈∞∂∇∈∉⊂⊃⊆⊇∪∩→←↔°µΩ"
MATH_RE = re.compile(rf"[{re.escape(MATH_SYMBOLS)}]|[₀-₉₊₋₌₍₎]|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]|\\(?:frac|sqrt|sum|int)\b")
MIN_PDF_TEXT_LAYER_CHARS = 80


def clean_formula_safe_text(text: str) -> str:
    """Normalize encoding and whitespace without compatibility-folding math glyphs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = re.sub(r"([^\W\d_])-\n([^\W\d_])", r"\1\2", text, flags=re.UNICODE)
    text = re.sub(r"(?im)^\s*page \d+\s*$", "", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _local(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _omml_plain(node: ET.Element) -> str:
    tag = _local(node)
    children = list(node)
    if tag == "t":
        return node.text or ""
    if tag == "f":
        num = next((_omml_plain(c) for c in children if _local(c) == "num"), "")
        den = next((_omml_plain(c) for c in children if _local(c) == "den"), "")
        return f"({num})/({den})"
    if tag in {"sSup", "sSub", "sSubSup"}:
        base = next((_omml_plain(c) for c in children if _local(c) == "e"), "")
        sub = next((_omml_plain(c) for c in children if _local(c) == "sub"), "")
        sup = next((_omml_plain(c) for c in children if _local(c) == "sup"), "")
        return base + (f"_{sub}" if sub else "") + (f"^{sup}" if sup else "")
    if tag == "rad":
        body = next((_omml_plain(c) for c in children if _local(c) == "e"), "")
        return f"√({body})"
    return "".join(_omml_plain(child) for child in children)


def _omml_latex(node: ET.Element) -> str:
    tag = _local(node)
    children = list(node)
    if tag == "t":
        return node.text or ""
    if tag == "f":
        num = next((_omml_latex(c) for c in children if _local(c) == "num"), "")
        den = next((_omml_latex(c) for c in children if _local(c) == "den"), "")
        return rf"\frac{{{num}}}{{{den}}}"
    if tag in {"sSup", "sSub", "sSubSup"}:
        base = next((_omml_latex(c) for c in children if _local(c) == "e"), "")
        sub = next((_omml_latex(c) for c in children if _local(c) == "sub"), "")
        sup = next((_omml_latex(c) for c in children if _local(c) == "sup"), "")
        return base + (f"_{{{sub}}}" if sub else "") + (f"^{{{sup}}}" if sup else "")
    if tag == "rad":
        body = next((_omml_latex(c) for c in children if _local(c) == "e"), "")
        degree = next((_omml_latex(c) for c in children if _local(c) == "deg"), "")
        return rf"\sqrt[{degree}]{{{body}}}" if degree else rf"\sqrt{{{body}}}"
    if tag == "nary":
        char = next((n.attrib.get(f"{{{M_NS}}}val", "") for n in node.iter() if _local(n) == "chr"), "∑")
        operator = {"∑": r"\sum", "∫": r"\int", "∏": r"\prod"}.get(char, char)
        sub = next((_omml_latex(c) for c in children if _local(c) == "sub"), "")
        sup = next((_omml_latex(c) for c in children if _local(c) == "sup"), "")
        body = next((_omml_latex(c) for c in children if _local(c) == "e"), "")
        return operator + (f"_{{{sub}}}" if sub else "") + (f"^{{{sup}}}" if sup else "") + " " + body
    return "".join(_omml_latex(child) for child in children)


@bounded_extraction
def load_docx_with_formulas(file_path: str) -> list[Document]:
    """Read Word paragraphs in order and preserve OMML as raw, plain and LaTeX."""
    documents: list[Document] = []
    with zipfile.ZipFile(file_path) as archive:
        if (archive.getinfo('word/document.xml').file_size > MAX_DOCX_XML_BYTES
                or sum(item.file_size for item in archive.infolist()) > MAX_DOCX_XML_BYTES * 2):
            raise ExtractionLimitError('DOCX unpacked content exceeds size limit')
    characters = 0
    try:
        with zipfile.ZipFile(file_path) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
        body = root.find(".//w:body", NS)
        paragraphs = list(body.iter(f"{{{W_NS}}}p")) if body is not None else []
        for position, paragraph in enumerate(paragraphs):
            check_extraction_budget(characters)
            parents = {child: parent for parent in paragraph.iter() for child in parent}
            parts, latex_parts, raw_formulas = [], [], []
            for child in paragraph.iter():
                check_extraction_budget()
                tag = _local(child)
                if tag in {"oMath", "oMathPara"}:
                    if any(_local(parent) in {"oMath", "oMathPara"} for parent in _ancestors(paragraph, child, parents)):
                        continue
                    raw = ET.tostring(child, encoding="unicode")
                    latex = clean_formula_safe_text(_omml_latex(child))
                    plain = clean_formula_safe_text(_omml_plain(child))
                    if latex or plain:
                        parts.append(rf"\({latex}\)" if latex else plain)
                        latex_parts.append(latex)
                        raw_formulas.append(raw)
                    # Descendant math nodes are handled by this top-level node.
                    continue
                if tag == "t" and not any(_local(parent) in {"oMath", "oMathPara"} for parent in _ancestors(paragraph, child, parents)):
                    parts.append(child.text or "")
                elif tag in {"tab"}:
                    parts.append("\t")
                elif tag in {"br", "cr"}:
                    parts.append("\n")
            text = clean_formula_safe_text("".join(parts))
            characters += len(text)
            check_extraction_budget(characters)
            if not text:
                continue
            style = paragraph.find("./w:pPr/w:pStyle", NS)
            section = style.attrib.get(f"{{{W_NS}}}val", "") if style is not None else ""
            metadata = {"source": str(file_path), "source_position": position, "section": section,
                        "block_type": "formula" if raw_formulas else "text"}
            if raw_formulas:
                metadata.update({"formula_format": "omml+latex", "formula_latex": "\n".join(latex_parts),
                                 "formula_raw": "\n".join(raw_formulas)})
            documents.append(Document(page_content=text, metadata=metadata))
    except ExtractionLimitError:
        raise
    except Exception:
        logger.exception("DOCX structural formula extraction failed; falling back to text extraction")
        from langchain_community.document_loaders import Docx2txtLoader
        documents = Docx2txtLoader(file_path).load()
        for doc in documents:
            doc.metadata.update({"block_type": "text", "formula_status": "unavailable"})
    return documents


def _mathml_latex(node) -> str:
    tag = getattr(node, "name", "")
    children = [child for child in getattr(node, "children", []) if getattr(child, "name", None)]
    if tag in {"mi", "mn", "mo", "mtext"}:
        return node.get_text("", strip=True)
    values = [_mathml_latex(child) for child in children]
    if tag == "mfrac" and len(values) >= 2:
        return rf"\frac{{{values[0]}}}{{{values[1]}}}"
    if tag == "msup" and len(values) >= 2:
        return f"{values[0]}^{{{values[1]}}}"
    if tag == "msub" and len(values) >= 2:
        return f"{values[0]}_{{{values[1]}}}"
    if tag == "msubsup" and len(values) >= 3:
        return f"{values[0]}_{{{values[1]}}}^{{{values[2]}}}"
    if tag == "msqrt":
        return rf"\sqrt{{{''.join(values)}}}"
    return "".join(values) or getattr(node, "get_text", lambda *args, **kwargs: "")("", strip=True)


@bounded_extraction
def load_html_with_formulas(file_path: str) -> list[Document]:
    """Preserve MathML structure and existing LaTeX while keeping ordinary HTML text."""
    from bs4 import BeautifulSoup

    if Path(file_path).stat().st_size > MAX_DOCX_XML_BYTES:
        raise ExtractionLimitError('HTML exceeds extraction size limit')
    raw_html = Path(file_path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")
    raw_formulas, latex_formulas = [], []
    for math in soup.find_all("math"):
        check_extraction_budget()
        raw_formulas.append(str(math))
        latex = _mathml_latex(math)
        latex_formulas.append(latex)
        math.replace_with(rf" \({latex}\) ")
    text = clean_formula_safe_text(soup.get_text("\n"))
    if not text:
        return []
    metadata = {"source": str(file_path), "source_position": 0,
                "block_type": "formula" if raw_formulas else "text"}
    if raw_formulas:
        metadata.update({"formula_format": "mathml+latex", "formula_latex": "\n".join(latex_formulas),
                         "formula_raw": "\n".join(raw_formulas)})
    return [Document(page_content=text, metadata=metadata)]


def _ancestors(root: ET.Element, target: ET.Element, parents=None):
    if parents is None:
        parents = {child: parent for parent in root.iter() for child in parent}
    current = parents.get(target)
    while current is not None:
        yield current
        current = parents.get(current)


@bounded_extraction
def load_pdf_with_formulas(file_path: str) -> list[Document]:
    """Preserve ordered PyMuPDF spans and Unicode math from text-layer PDFs."""
    import fitz

    documents: list[Document] = []
    with fitz.open(file_path) as pdf:
        if len(pdf) > MAX_PDF_PAGES:
            raise ExtractionLimitError('PDF page count exceeds limit')
        for page_number, page in enumerate(pdf):
            check_extraction_budget(sum(len(doc.page_content) for doc in documents))
            page_docs = []
            raw = page.get_text("dict", sort=True)
            for block_index, block in enumerate(raw.get("blocks", [])):
                if "lines" not in block:
                    continue
                lines = []
                for line in block["lines"]:
                    spans = sorted(line.get("spans", []), key=lambda span: span.get("bbox", [0])[0])
                    lines.append("".join(span.get("text", "") for span in spans))
                text = clean_formula_safe_text("\n".join(lines))
                if not text:
                    continue
                formula = bool(MATH_RE.search(text))
                page_docs.append(Document(page_content=text, metadata={
                    "source": str(file_path), "page": page_number, "source_position": block_index,
                    "block_type": "formula" if formula else "text",
                    "extraction_method": "text_layer",
                    "formula_format": "unicode" if formula else "",
                    "bbox": ",".join(str(round(value, 2)) for value in block.get("bbox", [])),
                }))
            documents.extend(page_docs)
    return documents


def missing_pdf_pages(documents: list[Document], page_count: int) -> list[int]:
    """Return zero-based pages for which the text layer yielded no usable text."""
    covered = {
        int(document.metadata["page"])
        for document in documents
        if document.page_content.strip() and document.metadata.get("page") is not None
    }
    return [page for page in range(page_count) if page not in covered]


@bounded_extraction
def load_pdf_pages_ocr(file_path: str, page_numbers: list[int]) -> list[Document]:
    """OCR only pages missing from the PDF text layer."""
    if not page_numbers:
        return []
    import fitz
    import pytesseract
    from PIL import Image
    from pytesseract import Output

    configured = os.getenv("TESSERACT_CMD")
    candidates = [configured, shutil.which("tesseract")]
    if os.name == "nt":
        candidates.extend([
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ])
    executable = next((value for value in candidates if value and Path(value).is_file()), None)
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
    documents: list[Document] = []
    with fitz.open(file_path) as pdf:
        for page_number in page_numbers:
            check_extraction_budget(sum(len(doc.page_content) for doc in documents))
            # Rendering through PyMuPDF keeps OCR self-contained and avoids the
            # external Poppler dependency required by pdf2image.
            page = pdf[page_number]
            if (page.rect.width * 300 / 72) * (page.rect.height * 300 / 72) > MAX_OCR_PIXELS:
                raise ExtractionLimitError('PDF page raster exceeds pixel limit')
            pixmap = page.get_pixmap(dpi=300, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            try:
                data = pytesseract.image_to_data(image, lang="rus+eng", output_type=Output.DICT, timeout=ocr_timeout())
            finally:
                image.close()
            tokens, confidences = [], []
            for token, raw_confidence in zip(data.get("text", []), data.get("conf", [])):
                try:
                    confidence = float(raw_confidence)
                except (TypeError, ValueError):
                    continue
                if token.strip() and confidence >= 30:
                    tokens.append(token)
                    confidences.append(confidence)
            text = clean_formula_safe_text(" ".join(tokens))
            if not text:
                continue
            average = sum(confidences) / len(confidences) if confidences else 0
            documents.append(Document(page_content=text, metadata={
                "source": str(file_path), "page": page_number, "source_position": 0,
                "block_type": "text", "extraction_method": "ocr",
                "ocr_status": "OCR_UNCERTAIN" if average < 60 else "OCR_OK",
                "ocr_confidence": round(average, 2), "formula_status": "uncertain",
            }))
    return documents


@bounded_extraction
def load_pdf_complete(file_path: str) -> list[Document]:
    """Extract every PDF page, filling missing text-layer pages with OCR."""
    import fitz

    with fitz.open(file_path) as pdf:
        page_count = len(pdf)
        if page_count > MAX_PDF_PAGES:
            raise ExtractionLimitError('PDF page count exceeds limit')
        raster_pages = {page for page in range(page_count) if pdf[page].get_images(full=True)}
    try:
        documents = load_pdf_with_formulas(file_path)
    except ExtractionLimitError:
        raise
    except Exception:
        logger.exception("PDF text-layer extraction failed; attempting OCR for every page")
        documents = []
    missing = missing_pdf_pages(documents, page_count)
    text_lengths = {
        page: sum(len(doc.page_content.strip()) for doc in documents if doc.metadata.get("page") == page)
        for page in range(page_count)
    }
    ocr_pages = [
        page for page in sorted(raster_pages)
        if page in missing or text_lengths.get(page, 0) < MIN_PDF_TEXT_LAYER_CHARS
    ]
    if ocr_pages:
        logger.warning("PDF text layer missed %s/%s pages; applying OCR to pages %s",
                       len(ocr_pages), page_count, [page + 1 for page in ocr_pages])
        try:
            documents.extend(load_pdf_pages_ocr(file_path, ocr_pages))
        except ExtractionLimitError:
            raise
        except Exception as exc:
            raise IncompleteExtractionError(
                f"OCR failed for PDF pages {[page + 1 for page in ocr_pages]}; "
                "the document was not indexed partially"
            ) from exc
    documents.sort(key=lambda doc: (
        int(doc.metadata.get("page", 0)), int(doc.metadata.get("source_position", 0))
    ))
    ocr_covered = {
        int(doc.metadata["page"]) for doc in documents
        if doc.metadata.get("extraction_method") == "ocr" and doc.page_content.strip()
    }
    unresolved = set(ocr_pages) - ocr_covered
    if unresolved:
        raise IncompleteExtractionError(
            f"OCR returned no text for PDF pages {[page + 1 for page in sorted(unresolved)]}; "
            "the document was not indexed partially"
        )
    covered = page_count - len(missing_pdf_pages(documents, page_count))
    logger.info("PDF extraction coverage: %s/%s pages", covered, page_count)
    return documents


def split_structured_documents(documents: list[Document], chunk_size=1500, chunk_overlap=300) -> list[Document]:
    """Keep short formula blocks atomic; split long prose with formula-safe separators."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=min(chunk_overlap, max(0, chunk_size // 5)),
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks: list[Document] = []
    section_heading = ""
    heading_page = None
    content_since_heading = False
    for document in documents:
        text = document.page_content.strip()
        source = str(document.metadata.get("source", "")).casefold()
        is_pdf = source.endswith(".pdf") or document.metadata.get("extraction_method") in {"text_layer", "ocr"}
        is_heading = (
            is_pdf
            and document.metadata.get("extraction_method") == "text_layer"
            and 1 <= len(text) <= 120
            and text.count("\n") <= 1
            and not re.search(r"[.!?;:]\s*$", text)
        )
        if is_heading:
            page = document.metadata.get("page")
            if section_heading and page == heading_page and not content_since_heading:
                section_heading = f"{section_heading}\n{text}"
            else:
                section_heading = text
            heading_page = page
            content_since_heading = False
            continue

        prepared = deepcopy(document)
        if is_pdf and section_heading and text:
            prepared.page_content = f"Раздел: {section_heading}\n{text}"
            prepared.metadata["section"] = section_heading
            prepared.metadata["section_context"] = True
            content_since_heading = True
        if prepared.metadata.get("block_type") == "formula" and len(prepared.page_content) <= chunk_size:
            chunks.append(prepared)
        else:
            chunks.extend(splitter.split_documents([prepared]))
    return chunks
