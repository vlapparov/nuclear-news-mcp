from news_mcp.search import reciprocal_rank_fusion
from news_mcp.store import keyword_query


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion({"keyword": [1, 2, 3], "semantic": [3, 4]})
    assert list(fused)[0] == 3  # found by both retrievers
    assert set(fused) == {1, 2, 3, 4}


def test_keyword_query_is_safe_fts_syntax():
    assert keyword_query('LNG "trade" 2021 a') == '"LNG" OR "trade"* OR "2021"*'


def test_needle_is_found_with_its_passage(index):
    results = index.searcher.search("LNG obchodovalo 2021", {"country": ["Czech Republic"]}, limit=3)
    assert results[0]["article_id"] == "1"
    assert "372 milionů tun" in results[0]["passage"]
    assert results[0]["matched_by"] == ["keyword", "semantic"]


def test_search_respects_filters(index):
    results = index.searcher.search("elektrowni jądrowej", {"country": ["Poland"], "date_from": "2020-01-01"})
    assert results and all(r["country"] == "Poland" for r in results)


def test_semantic_mode(index):
    results = index.searcher.search("Mochovce blok", {}, limit=3, mode="semantic")
    assert results[0]["article_id"] == "8"
    assert results[0]["matched_by"] == ["semantic"]
