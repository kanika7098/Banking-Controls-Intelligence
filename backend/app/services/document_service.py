

from pathlib import Path
import shutil

import pymupdf
import pytesseract
from PIL import Image


# Default Windows installation path for Tesseract OCR.
TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)
elif shutil.which("tesseract") is None:
    # Leave the default command in place; pytesseract will report
    # a clear error if Tesseract cannot be located.
    pass


def extract_pdf_text(file_path: str) -> dict:
    """
    Extract text from a PDF using normal text extraction first.
    Use OCR on pages that contain no selectable text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError("The PDF file was not found.")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    pages = []

    try:
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            if len(pdf) == 0:
                raise ValueError("The PDF contains no pages.")

            for page_number, page in enumerate(pdf, start=1):
                # First, try extracting selectable text.
                text = page.get_text("text").strip()

                # If the page has no selectable text, use OCR.
                if not text:
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(2, 2),
                        alpha=False,
                    )

                    image = Image.frombytes(
                        "RGB",
                        [pixmap.width, pixmap.height],
                        pixmap.samples,
                    )

                    try:
                        text = pytesseract.image_to_string(
                            image,
                            lang="eng",
                        ).strip()
                    except pytesseract.TesseractNotFoundError as exc:
                        raise RuntimeError(
                            "Tesseract OCR was not found. "
                            "Install Tesseract or configure its executable path."
                        ) from exc
                    except pytesseract.TesseractError as exc:
                        raise RuntimeError(
                            f"OCR failed on PDF page {page_number}."
                        ) from exc
                    finally:
                        image.close()

                if text:
                    pages.append(
                        {
                            "page_number": page_number,
                            "text": text,
                        }
                    )

            return {
                "filename": path.name,
                "total_pages": len(pdf),
                "pages_with_text": len(pages),
                "pages": pages,
            }

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(
            "The PDF could not be processed."
        ) from exc
