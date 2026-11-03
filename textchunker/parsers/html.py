"""HTML extraction with explicitly unknown raw-HTML source offsets."""

from __future__ import annotations

from .text import parse_extracted_text


def parse_html(html: str, source_uri: str = "memory://html"):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "Install structure extras to parse HTML: pip install -e '.[structure]'"
        ) from exc
    soup = BeautifulSoup(html, "html.parser")
    text = "\n".join(
        element.get_text(" ", strip=True)
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]
        )
    )
    return parse_extracted_text(text, source_uri, "html")
