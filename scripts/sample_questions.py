"""Exercise the running server with the tool calls an MCP client would make for the
sample questions in the brief, and print compact results for manual inspection.

    uv run news-mcp-serve &          # in another terminal
    uv run python scripts/sample_questions.py [--url http://127.0.0.1:8000/mcp]
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

SCENARIOS = [
    (
        "According to Czech media, how much LNG was traded on world markets in 2021?",
        "search_articles",
        {"query": "how much LNG was traded worldwide in 2021 světový obchod LNG milionů tun",
         "countries": ["Czech Republic"], "date_from": "2021", "limit": 5},
    ),
    (
        "How did sentiment towards nuclear energy change in Poland between 2010 and 2020?",
        "aggregate_articles",
        {"group_by": ["year"], "countries": ["Poland"], "date_from": "2010", "date_to": "2020",
         "nuclear_related_only": True},
    ),
    (
        "How would you compare sentiment towards nuclear energy in Slovakia and Hungary in 2022?",
        "aggregate_articles",
        {"group_by": ["country"], "countries": ["Slovakia", "Hungary"], "date_from": "2022", "date_to": "2022",
         "nuclear_related_only": True},
    ),
    (
        "Which Polish publications had the highest number of articles about nuclear energy in 2020?",
        "aggregate_articles",
        {"group_by": ["source"], "countries": ["Poland"], "date_from": "2020", "date_to": "2020",
         "nuclear_related_only": True, "order_by": "count"},
    ),
    (
        "Summarise how media across the four countries covered topics related to nuclear weapons in 2023.",
        "search_by_group",
        {"query": "nuclear weapons broń jądrowa jaderné zbraně jadrové zbrane atomfegyver",
         "group_by": "country", "date_from": "2023", "date_to": "2023", "per_group": 3},
    ),
]  # fmt: skip


def compact(result: dict) -> str:
    def shorten(value):
        if isinstance(value, dict):
            return {k: shorten(v) for k, v in value.items() if k not in ("url", "language", "passage_index")}
        if isinstance(value, list):
            return [shorten(v) for v in value]
        if isinstance(value, str) and len(value) > 1000:
            return value[:1000] + " …"
        return value

    return json.dumps(shorten(result), ensure_ascii=False, indent=1)


async def main(url: str, token: str) -> None:
    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=60)
    async with http, Client(streamable_http_client(url, http_client=http)) as client:
        for question, tool, arguments in SCENARIOS:
            print("=" * 100, f"\nQ: {question}\n-> {tool}({json.dumps(arguments, ensure_ascii=False)})")
            result = await client.call_tool(tool, arguments)
            print(result.content[0].text if result.is_error else compact(result.structured_content))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--token", default="test-token")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.token))
