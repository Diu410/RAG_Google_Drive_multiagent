"""Budgets shared by uploads, Drive and workspace extraction."""
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from threading import BoundedSemaphore
from functools import wraps

MAX_EXTRACTED_CHARACTERS = int(os.getenv('MAX_EXTRACTED_CHARACTERS', '2000000'))
MAX_DOCX_XML_BYTES = int(os.getenv('MAX_DOCX_XML_BYTES', str(16 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv('MAX_PDF_PAGES', '500'))
MAX_OCR_PIXELS = int(os.getenv('MAX_OCR_PIXELS', '20000000'))
EXTRACTION_SECONDS = float(os.getenv('EXTRACTION_SECONDS', '120'))
_slots = BoundedSemaphore(max(1, int(os.getenv('EXTRACTION_CONCURRENCY', '2'))))
_deadline = ContextVar('extraction_deadline', default=None)


class ExtractionLimitError(ValueError):
    pass


class IncompleteExtractionError(ValueError):
    pass


def check_extraction_budget(characters=0):
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise ExtractionLimitError('Document extraction time limit exceeded')
    if characters > MAX_EXTRACTED_CHARACTERS:
        raise ExtractionLimitError('Extracted text exceeds size limit; document was not indexed partially')


@contextmanager
def extraction_budget():
    if _deadline.get() is not None:
        check_extraction_budget()
        yield
        return
    if not _slots.acquire(timeout=5):
        raise ExtractionLimitError('Document extraction is busy; retry later')
    token = _deadline.set(time.monotonic() + EXTRACTION_SECONDS)
    try:
        yield
        check_extraction_budget()
    finally:
        _deadline.reset(token)
        _slots.release()


def bounded_extraction(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with extraction_budget():
            documents = function(*args, **kwargs)
            check_extraction_budget(sum(len(doc.page_content) for doc in documents))
            return documents
    return wrapped


def ocr_timeout():
    remaining = (_deadline.get() or (time.monotonic() + 30)) - time.monotonic()
    check_extraction_budget()
    return max(.01, min(30, remaining))
