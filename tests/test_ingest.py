import pytest
from conftest import CORPUS, make_row

from news_mcp.ingest import clean_article, repair_text


def test_dates_and_language_are_normalised():
    article = clean_article(CORPUS[1])
    assert article["date"] == "2021-06-02"
    assert (article["year"], article["month"]) == (2021, "2021-06")
    assert article["language"] == "cs"  # the dataset uses "cz"


@pytest.mark.parametrize(
    "ratings,related,sentiment",
    [
        ((5, 4, 5), 1, 4.667),  # all three models rate it
        ((0, 4, 5), 1, 4.5),  # a 0 ("not about nuclear power") is left out of the mean
        ((0, 0, 3), 0, None),  # two models say it is unrelated
    ],
)
def test_consensus_sentiment(ratings, related, sentiment):
    article = clean_article(make_row("x", "Poland", "pl", "Onet_pl", "01-01-2020", "t", "b", rating=ratings))
    assert article["nuclear_related"] == related
    assert article["nuclear_sentiment"] == sentiment


def test_alarmism_is_flipped_so_higher_means_more_alarmist():
    calm = clean_article(make_row("a", "Poland", "pl", "Onet_pl", "01-01-2020", "t", "b", alarm=(5, 5, 5)))
    alarmist = clean_article(make_row("b", "Poland", "pl", "Onet_pl", "01-01-2020", "t", "b", alarm=(1, 1, 1)))
    assert (calm["headline_alarmism"], alarmist["headline_alarmism"]) == (1, 5)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("v prevádzke.Projekt za", "v prevádzke. Projekt za"),  # glued sentences are separated
        ("vybudovať.Šéf firmy", "vybudovať. Šéf firmy"),  # also before an accented capital
        ("do roku 2018.Projekt za", "do roku 2018.Projekt za"),  # a digit before the dot: left alone
        ("web aktualne.cz uvádí", "web aktualne.cz uvádí"),  # domain names are not touched
    ],
)
def test_repair_text(text, expected):
    assert repair_text(text) == expected
