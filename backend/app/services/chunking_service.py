from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int


def chunk_document_pages(
    pages,
    document_id: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be greater than or equal to zero "
            "and smaller than chunk_size."
        )

    chunks = []
    chunk_index = 0
    step = chunk_size - chunk_overlap

    for page in pages:
        page_number = page["page_number"]
        text = page.get("text", "")

        if not text or not text.strip():
            continue

        text = text.strip()
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer a word boundary near the target end.
            # Keep chunk starts on a predictable configured step.
            if end < len(text):
                boundary = text.rfind(" ", start + 1, end)

                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        document_id=document_id,
                        filename=filename,
                        page_number=page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break

            start += step

    return chunks
