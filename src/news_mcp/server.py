"""The MCP server: six read-only tools over Streamable HTTP.

    uv run news-mcp-serve      # http://127.0.0.1:8000/mcp, header "Authorization: Bearer test-token"
"""

import json
import os
import re
from typing import Annotated, Any, Literal, Optional

import numpy as np
import uvicorn
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from news_mcp import config
from news_mcp.auth import StaticTokenVerifier, caller_has_scope
from news_mcp.embeddings import SentenceEmbeddings
from news_mcp.search import HybridSearch
from news_mcp.store import Store

# Sent to the client when it connects; most clients show it to the model.
INSTRUCTIONS = """\
Tools for exploring ~26,000 news articles (1998-2025) about nuclear energy from 12 outlets in
Poland, the Czech Republic, Slovakia and Hungary. Articles are in Polish, Czech, Slovak and
Hungarian (not translated). Every result has article_id, source, country, date and url: cite
them, and use get_article to read and verify the full text.

Which tool to use:
- A specific fact: search_articles with a focused query, filtered by country/date when the
  question implies them. English queries work, but adding key words in the article language
  improves keyword matching, e.g. "LNG world trade 2021 obchod zkapalněný plyn".
- Counts, rankings, trends, comparisons: aggregate_articles (group by country, source,
  language, year or month). Use nuclear_related_only=true for "articles about nuclear
  energy". Report article_count / sentiment_rated_count together with means.
- Summaries or comparisons of a topic across countries/outlets/years: search_by_group.
- The articles behind a number: list_articles with the same filters, or example_article_ids.
- Unsure about outlets, fields or scales: describe_dataset.

Scores (LLM annotations by Gemini, GPT-4o and DeepSeek; consensus = their mean):
- nuclear_sentiment: how the article portrays nuclear power, 1 very negative .. 3 neutral ..
  5 very positive; null when the article is not about nuclear power.
- headline_sentiment: 1 very pessimistic .. 5 very optimistic.
- headline_alarmism: 1 calm .. 5 strongly alarmist.
"""

DATA_NOTES = [
    "Scores are zero-shot LLM annotations, not human labels; the models sometimes disagree "
    "(aggregate_articles(sentiment_model=...) shows each model separately).",
    "Coverage per outlet and year is uneven: compare means and shares rather than raw counts.",
    "Some Slovak texts lost letters with diacritics at scraping time (e.g. 'Vekom' for 'Veľkom').",
    "Article bodies are cut at 32,767 characters in the source data.",
    "Passages carry start_char/end_char: their position in the article body from get_article.",
]

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

mcp = MCPServer(
    name="v4-nuclear-news",
    title="V4 Nuclear News Explorer",
    instructions=INSTRUCTIONS,
    token_verifier=StaticTokenVerifier(),
    auth=AuthSettings(
        issuer_url=f"http://{config.HOST}:{config.PORT}",  # static tokens: no separate login server
        resource_server_url=None,
        required_scopes=["news:search"],
    ),
)

# Loaded by load_index() at start-up.
store: Store = None
searcher: HybridSearch = None


def load_index(data_dir: str, model: SentenceEmbeddings = None) -> None:
    """Open the database and embeddings built by news-mcp-ingest."""
    global store, searcher
    if model is None:  # use the same model preset as the one used to build the index
        with open(os.path.join(data_dir, "manifest.json")) as f:
            preset = config.EMBEDDING_PRESETS[json.load(f)["embedding_preset"]]
        model = SentenceEmbeddings(os.path.join(data_dir, "embedding_model"), preset)
    store = Store(os.path.join(data_dir, "news.sqlite"))
    embeddings = np.load(os.path.join(data_dir, "embeddings.npy")).astype(np.float32)
    searcher = HybridSearch(store, embeddings, model)


def make_filters(countries=None, sources=None, languages=None, date_from=None, date_to=None,
                 nuclear_related_only=False, keywords=None) -> dict[str, Any]:
    """Check the filter arguments of a tool call and turn them into a filters dict."""
    filters = {}
    for name, values in (("country", countries), ("source", sources), ("language", languages)):
        if values:
            known = {v.lower(): v for v in store.distinct_values(name)}
            unknown = [v for v in values if v.lower() not in known]
            if unknown:
                raise ToolError(f"Unknown {name} {unknown}. Valid values: {list(known.values())}")
            filters[name] = [known[v.lower()] for v in values]
    for name, value, suffix in (("date_from", date_from, "-01-01"), ("date_to", date_to, "-12-31")):
        if value:
            if not re.fullmatch(r"\d{4}(-\d{2}){0,2}", value):
                raise ToolError(f"Invalid {name} {value!r}: use YYYY, YYYY-MM or YYYY-MM-DD.")
            filters[name] = value + suffix[len(value) - 4:]  # "2020" -> "2020-01-01" / "2020-12-31"
    if nuclear_related_only:
        filters["nuclear_related_only"] = True
    if keywords:
        filters["keywords"] = keywords
    return filters


# ---- parameters shared by several tools (their descriptions are shown to the model) ----

Countries = Annotated[Optional[list[str]], Field(
    description="Only these countries: Poland, Czech Republic, Slovakia, Hungary.")]
Sources = Annotated[Optional[list[str]], Field(description="Only these outlets (names: see describe_dataset).")]
Languages = Annotated[Optional[list[str]], Field(description="Only these languages: pl, cs, sk, hu.")]
DateFrom = Annotated[Optional[str], Field(description="Start date, inclusive: YYYY, YYYY-MM or YYYY-MM-DD.")]
DateTo = Annotated[Optional[str], Field(description="End date, inclusive: YYYY, YYYY-MM or YYYY-MM-DD.")]
NuclearOnly = Annotated[bool, Field(description="Skip articles the annotators judged not about nuclear power.")]
Keywords = Annotated[Optional[list[str]], Field(description=(
    "Only articles containing at least one of these words. Give translations, e.g. "
    '["nuclear weapon", "broń jądrowa", "jaderná zbraň", "jadrová zbraň", "atomfegyver"].'))]


@mcp.tool(annotations=READ_ONLY)
def describe_dataset() -> dict[str, Any]:
    """Outlets per country (article counts, years covered), date range, the meaning of the
    score fields and known data caveats. Call this first if unsure about names or scales."""
    return {**store.overview(), "notes": DATA_NOTES}


@mcp.tool(annotations=READ_ONLY)
def search_articles(
    query: Annotated[str, Field(description="What to find, in any language. Native-language key words help.")],
    countries: Countries = None,
    sources: Sources = None,
    languages: Languages = None,
    date_from: DateFrom = None,
    date_to: DateTo = None,
    nuclear_related_only: NuclearOnly = False,
    mode: Annotated[Literal["hybrid", "keyword", "semantic"],
                    Field(description="hybrid (default) = keyword + semantic search combined.")] = "hybrid",
    limit: Annotated[int, Field(ge=1, le=30)] = 10,
) -> dict[str, Any]:
    """Find the most relevant articles for a query, each with its best-matching passage as
    evidence. Best for specific facts and for finding coverage of a topic."""
    filters = make_filters(countries, sources, languages, date_from, date_to, nuclear_related_only)
    return {"query": query, "filters": filters, "results": searcher.search(query, filters, limit, mode)}


@mcp.tool(annotations=READ_ONLY)
def search_by_group(
    query: Annotated[str, Field(description="Topic to find evidence for, in any language.")],
    group_by: Literal["country", "source", "language", "year"] = "country",
    per_group: Annotated[int, Field(ge=1, le=10)] = 3,
    countries: Countries = None,
    sources: Sources = None,
    languages: Languages = None,
    date_from: DateFrom = None,
    date_to: DateTo = None,
    nuclear_related_only: NuclearOnly = False,
) -> dict[str, Any]:
    """Top passages for a topic separately for each country / outlet / language / year, so a
    summary or comparison is balanced and not dominated by the biggest group.
    Example: 'how did media in the four countries cover X in 2023?'."""
    filters = make_filters(countries, sources, languages, date_from, date_to, nuclear_related_only)
    values = store.distinct_values(group_by, filters)
    if len(values) > 30:
        raise ToolError(f"{len(values)} groups is too many (max 30); narrow the filters.")
    groups = []
    for value in values:
        results = searcher.search(query, {**filters, group_by: [value]}, limit=per_group)
        groups.append({group_by: value, "results": results})
    return {"query": query, "filters": filters, "group_by": group_by, "groups": groups}


@mcp.tool(annotations=READ_ONLY)
def aggregate_articles(
    group_by: Annotated[list[Literal["country", "source", "language", "year", "month"]], Field(
        max_length=3, description="e.g. ['year'] or ['country', 'year']; [] for overall totals.")],
    countries: Countries = None,
    sources: Sources = None,
    languages: Languages = None,
    date_from: DateFrom = None,
    date_to: DateTo = None,
    nuclear_related_only: NuclearOnly = False,
    keywords: Keywords = None,
    sentiment_model: Annotated[Literal["consensus", "gemini", "gpt", "deepseek"], Field(
        description="Annotator used for sentiment; consensus = mean of the three.")] = "consensus",
    order_by: Annotated[Literal["group", "count"], Field(
        description="'count' ranks groups by article_count, largest first.")] = "group",
    limit: Annotated[int, Field(ge=1, le=500, description="Maximum number of groups.")] = 100,
) -> dict[str, Any]:
    """Count articles and summarise sentiment per group: article_count, sentiment_rated_count,
    mean_nuclear_sentiment (1-5), share_positive (>= 3.5), share_negative (<= 2.5),
    mean_headline_sentiment, mean_headline_alarmism, plus example_article_ids as evidence.
    Use for rankings, trends over time and comparisons."""
    filters = make_filters(countries, sources, languages, date_from, date_to, nuclear_related_only, keywords)
    groups = store.aggregate(filters, group_by, sentiment_model, order_by, limit)
    return {"filters": filters, "group_by": group_by, "sentiment_model": sentiment_model, "groups": groups}


@mcp.tool(annotations=READ_ONLY)
def list_articles(
    countries: Countries = None,
    sources: Sources = None,
    languages: Languages = None,
    date_from: DateFrom = None,
    date_to: DateTo = None,
    nuclear_related_only: NuclearOnly = False,
    keywords: Keywords = None,
    sort_by: Literal["date", "nuclear_sentiment", "headline_alarmism"] = "date",
    descending: bool = True,
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """List articles matching filters (not ranked by relevance), with the total number of
    matches. Use it to look inside a group from aggregate_articles, e.g. the most negative
    Hungarian articles of 2022 (sort_by='nuclear_sentiment', descending=false)."""
    filters = make_filters(countries, sources, languages, date_from, date_to, nuclear_related_only, keywords)
    return {"filters": filters, **store.list_articles(filters, sort_by, descending, limit, offset)}


@mcp.tool(annotations=READ_ONLY)
def get_article(
    article_id: Annotated[str, Field(description="article_id from a search, aggregate or list result.")],
) -> dict[str, Any]:
    """One article in full: metadata, url, complete original text and every annotator score.
    Needs the news:read_full permission."""
    if not caller_has_scope("news:read_full"):
        raise ToolError("Your access policy does not allow full article text (scope news:read_full). "
                        "Use the passages returned by search_articles as evidence instead.")
    article = store.get_article(article_id)
    if article is None:
        raise ToolError(f"No article with id {article_id!r}.")
    return article


@mcp.resource("news://dataset", mime_type="application/json", description="Dataset overview.")
def dataset_resource() -> str:
    return json.dumps(describe_dataset(), indent=2, ensure_ascii=False)


def main():
    load_index(config.DATA_DIR)
    print(f"MCP server: http://{config.HOST}:{config.PORT}/mcp")
    uvicorn.run(mcp.streamable_http_app(host=config.HOST), host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
