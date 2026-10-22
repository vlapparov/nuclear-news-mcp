import pytest


def test_count_by_source(index):
    groups = index.store.aggregate({"country": ["Poland"]}, ["source"], order_by="count")
    assert [(g["source"], g["article_count"]) for g in groups] == [("Onet_pl", 2), ("Fakt_pl", 1)]


def test_nuclear_related_filter(index):
    groups = index.store.aggregate({"country": ["Poland"], "nuclear_related_only": True}, ["source"])
    assert [(g["source"], g["article_count"]) for g in groups] == [("Onet_pl", 2)]


def test_sentiment_statistics_and_examples(index):
    group = index.store.aggregate({"country": ["Poland"]}, ["year"])[0]
    assert group["article_count"] == 3
    assert group["sentiment_rated_count"] == 2  # the football article is not about nuclear power
    assert group["mean_nuclear_sentiment"] == pytest.approx(((4 + 4 + 5) / 3 + 2) / 2, abs=1e-3)
    assert (group["share_positive"], group["share_negative"]) == (0.5, 0.5)
    assert set(group["example_article_ids"]) <= {"3", "4", "5"}


def test_single_annotator_option(index):
    group = index.store.aggregate({"source": ["Fakt_pl"]}, [], sentiment_model="gpt")[0]
    assert group["mean_nuclear_sentiment"] == 3  # gpt rated the football article 3


def test_keyword_filter(index):
    groups = index.store.aggregate({"keywords": ["atomfegyver", "zbrane"]}, ["country"])
    assert {g["country"]: g["article_count"] for g in groups} == {"Hungary": 1, "Slovakia": 1}


def test_list_articles_sorted_by_sentiment(index):
    result = index.store.list_articles({"nuclear_related_only": True}, "nuclear_sentiment", descending=False)
    assert result["total_matching"] == 8
    assert result["articles"][0]["article_id"] == "6"  # the most negative one


def test_get_article(index):
    article = index.store.get_article("1")
    assert article["article_id"] == "1" and "372 milionů tun LNG" in article["body"]
    assert index.store.get_article("unknown") is None
