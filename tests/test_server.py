"""End-to-end tests: real HTTP requests to the running server, with the official MCP client."""

import asyncio

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}}
ACCEPT = {"Accept": "application/json, text/event-stream"}


def call_tool(url, token, tool, arguments=None):
    """Connect as an MCP client with this token and call one tool (tool=None: list the tools)."""
    async def run():
        http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"})
        async with http, Client(streamable_http_client(url, http_client=http)) as client:
            if tool is None:
                return await client.list_tools()
            return await client.call_tool(tool, arguments or {})
    return asyncio.run(run())


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-token", "Basic dGVzdDp0ZXN0", "test-token"])
def test_missing_or_invalid_token_is_rejected(server_url, authorization):
    headers = {**ACCEPT, "Authorization": authorization} if authorization else ACCEPT
    response = httpx.post(server_url, json=INITIALIZE, headers=headers)
    assert response.status_code == 401


def test_test_token_is_accepted(server_url):
    response = httpx.post(server_url, json=INITIALIZE, headers={**ACCEPT, "Authorization": "Bearer test-token"})
    assert response.status_code == 200


def test_tools_are_listed_with_descriptions(server_url):
    tools = call_tool(server_url, "test-token", None).tools
    assert {t.name for t in tools} == {"describe_dataset", "search_articles", "search_by_group",
                                       "aggregate_articles", "list_articles", "get_article"}
    assert all(t.description and t.annotations.read_only_hint for t in tools)


def test_search_then_read_full_article(server_url):
    found = call_tool(server_url, "test-token", "search_articles", {"query": "LNG 2021", "countries": ["czech republic"]})
    top = found.structured_content["results"][0]
    assert top["article_id"] == "1" and top["url"].startswith("https://")
    article = call_tool(server_url, "test-token", "get_article", {"article_id": "1"})
    assert "372 milionů tun LNG" in article.structured_content["body"]


def test_search_only_token_cannot_read_full_text(server_url):
    assert not call_tool(server_url, "search-only-token", "search_articles", {"query": "Paks"}).is_error
    article = call_tool(server_url, "search-only-token", "get_article", {"article_id": "9"})
    assert article.is_error and "news:read_full" in article.content[0].text


def test_invalid_arguments_give_readable_errors(server_url):
    result = call_tool(server_url, "test-token", "aggregate_articles", {"group_by": ["year"], "countries": ["Austria"]})
    assert result.is_error and "Valid values" in result.content[0].text
    result = call_tool(server_url, "test-token", "list_articles", {"date_from": "last year"})
    assert result.is_error and "YYYY" in result.content[0].text


def test_search_by_group_gives_every_group_its_own_results(server_url):
    result = call_tool(server_url, "test-token", "search_by_group",
                       {"query": "atomfegyver jadrové zbrane", "date_from": "2023", "per_group": 2})
    groups = {g["country"]: g["results"] for g in result.structured_content["groups"]}
    assert set(groups) == {"Hungary", "Slovakia"}
    assert groups["Hungary"][0]["article_id"] == "6"


def test_aggregate_echoes_filters(server_url):
    result = call_tool(server_url, "test-token", "aggregate_articles",
                       {"group_by": ["source"], "countries": ["Poland"], "date_from": "2020", "date_to": "2020",
                        "nuclear_related_only": True, "order_by": "count"})
    assert result.structured_content["groups"][0]["source"] == "Onet_pl"
    assert result.structured_content["filters"] == {"country": ["Poland"], "date_from": "2020-01-01",
                                                   "date_to": "2020-12-31", "nuclear_related_only": True}
