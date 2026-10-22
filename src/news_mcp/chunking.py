"""Split articles into passages ("chunks") for search.

Structure first: the splitter tries paragraph breaks, then line breaks, then sentence ends,
then clause punctuation, and only then spaces. Sizes are counted in tokens of the embedding
model, so a chunk holds a similar amount of meaning in every language.
"""

from typing import Callable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from news_mcp.config import CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def make_splitter(count_tokens: Callable[[str], int]) -> RecursiveCharacterTextSplitter:
    """
    :param count_tokens: function returning the number of tokens of a text
    """
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        keep_separator="end",  # a sentence keeps its full stop
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,
    )


def split_article(article: dict, splitter: RecursiveCharacterTextSplitter) -> List[dict]:
    """
    Split one article body into chunks that can be traced back to the article.
    :param article: dict with at least "id", "title" and "body"
    :return: list of {"article_id", "chunk_index", "start_char", "end_char", "text"}
    """
    body = article["body"]
    chunks = []
    search_from = 0
    for index, text in enumerate(splitter.split_text(body)):
        start = body.find(text, search_from)  # chunks are exact pieces of the body, in order
        search_from = start + 1
        chunks.append({
            "article_id": article["id"],
            "chunk_index": index,
            "start_char": start,
            "end_char": start + len(text),
            "text": text,
        })
    return chunks


def text_to_embed(title: str, chunk_text: str) -> str:
    """The title is added as a header so a chunk can be understood on its own."""
    return f"{title}\n\n{chunk_text}"
