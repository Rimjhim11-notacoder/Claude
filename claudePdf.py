import argparse
import base64
import os
import sys
from pathlib import Path

import anthropic


MODEL = "claude-haiku-4-5-20251001"
MAX_PDF_BYTES = 23_000_000


def summarize_pdf(pdf_path: Path) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first.")

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF is too large for this script's direct upload. Try a smaller PDF.")

    pdf_base64 = base64.b64encode(pdf_bytes).decode("ascii")
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Summarize this PDF. Include its main purpose, key points, "
                            "important conclusions, and any action items. Be concise "
                            "and do not add information that is not in the document."
                        ),
                    },
                ],
            }
        ],
    )

    return "\n".join(
        block.text for block in response.content if block.type == "text"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PDF with Claude Haiku.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    args = parser.parse_args()

    try:
        print(summarize_pdf(args.pdf))
    except (OSError, ValueError, RuntimeError, anthropic.APIError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
