"""SQLite database: articles, chunks and a keyword index. Every number the server reports
comes from a plain SQL query over this file."""

import re
import sqlite3
from typing import List, Tuple

SCHEMA = """
CREATE TABLE articles (
    id TEXT PRIMARY KEY,
    country TEXT, language TEXT, source TEXT,
    date TEXT,                -- yyyy-mm-dd
    year INTEGER, month TEXT, -- month = yyyy-mm
    url TEXT, title TEXT, body TEXT,
    nuclear_related INTEGER,  -- 1 if at least 2 of the 3 annotator models say it is about nuclear power
    nuclear_sentiment REAL,   -- 1 very negative .. 5 very positive (mean of the models); NULL if unrelated
    headline_sentiment REAL,  -- 1 very pessimistic .. 5 very optimistic
    headline_alarmism REAL,   -- 1 calm .. 5 strongly alarmist
    gemini_article_rating INTEGER, gemini_headline_sentiment INTEGER, gemini_headline_alarmistic INTEGER,
    deepseek_article_rating INTEGER, deepseek_headline_sentiment INTEGER, deepseek_headline_alarmistic INTEGER,
    gpt_article_rating INTEGER, gpt_headline_sentiment INTEGER, gpt_headline_alarmistic INTEGER
);
CREATE INDEX articles_by_country ON articles(country, date);
CREATE INDEX articles_by_source ON articles(source, date);

CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,  -- also the row number in the embeddings matrix
    article_id TEXT, chunk_index INTEGER, start_char INTEGER, end_char INTEGER, text TEXT
);
CREATE INDEX chunks_by_article ON chunks(article_id);

-- Keyword index over each chunk and its article title (rowid = chunk_id).
CREATE VIRTUAL TABLE chunks_fts USING fts5(title, text, content='', tokenize='unicode61 remove_diacritics 2');
"""

ARTICLE_SUMMARY = (
    "id AS article_id, title, source, country, language, date, url, "
    "nuclear_sentiment, headline_sentiment, headline_alarmism"
)

# Article sentiment column per annotator. "consensus" is the mean of the three models;
# a 0 rating means "not about nuclear power", so it becomes NULL (ignored by AVG).
SENTIMENT_COLUMNS = {
    "consensus": "nuclear_sentiment",
    "gemini": "NULLIF(gemini_article_rating, 0)",
    "gpt": "NULLIF(gpt_article_rating, 0)",
    "deepseek": "NULLIF(deepseek_article_rating, 0)",
}


def keyword_query(text: str) -> str:
    """
    Turn free text into a safe FTS5 query: words OR-ed together, longer words as prefixes
    (helps with Polish, Czech, Slovak and Hungarian word endings).
    Example: 'LNG trade 2021' -> '"LNG" OR "trade"* OR "2021"*'
    """
    words = [w for w in re.findall(r"\w+", text) if len(w) > 1]
    return " OR ".join(f'"{w}"*' if len(w) > 3 else f'"{w}"' for w in words)


def where_clause(filters: dict) -> Tuple[str, list]:
    """
    Build the SQL condition for a filters dict (see server.make_filters).
    :return: (sql, params) to use as `SELECT ... FROM articles WHERE <sql>`
    """
    conditions, params = ["1 = 1"], []
    for column in ("country", "source", "language", "year"):
        if filters.get(column):
            conditions.append(f"{column} IN ({', '.join('?' * len(filters[column]))})")
            params += filters[column]
    if filters.get("date_from"):
        conditions.append("date >= ?")
        params.append(filters["date_from"])
    if filters.get("date_to"):
        conditions.append("date <= ?")
        params.append(filters["date_to"])
    if filters.get("nuclear_related_only"):
        conditions.append("nuclear_related = 1")
    if filters.get("keywords"):
        conditions.append(
            "id IN (SELECT article_id FROM chunks WHERE chunk_id IN "
            "(SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?))"
        )
        params.append(keyword_query(" ".join(filters["keywords"])))
    return " AND ".join(conditions), params


class Store:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def query(self, sql: str, params: list = ()) -> List[dict]:
        return [dict(row) for row in self.conn.execute(sql, params)]

    def distinct_values(self, column: str, filters: dict = None) -> list:
        where, params = where_clause(filters or {})
        rows = self.query(f"SELECT DISTINCT {column} AS v FROM articles WHERE {where} ORDER BY v", params)
        return [row["v"] for row in rows]

    # ---- search helpers ----

    def keyword_search(self, text: str, filters: dict, limit: int) -> List[int]:
        """Best-matching chunk ids by BM25 (title words count twice)."""
        where, params = where_clause(filters)
        rows = self.query(
            "SELECT chunks.chunk_id FROM chunks_fts "
            "JOIN chunks ON chunks.chunk_id = chunks_fts.rowid "
            "JOIN articles ON articles.id = chunks.article_id "
            f"WHERE chunks_fts MATCH ? AND {where} "
            "ORDER BY bm25(chunks_fts, 2.0, 1.0) LIMIT ?",
            [keyword_query(text)] + params + [limit],
        )
        return [row["chunk_id"] for row in rows]

    def article_ids(self, filters: dict) -> List[str]:
        where, params = where_clause(filters)
        return [row["id"] for row in self.query(f"SELECT id FROM articles WHERE {where}", params)]

    def chunk_article_ids(self) -> List[str]:
        """Article id of every chunk, in chunk_id order (= embeddings matrix order)."""
        return [row["article_id"] for row in self.query("SELECT article_id FROM chunks ORDER BY chunk_id")]

    def passage(self, chunk_id: int) -> dict:
        """A chunk with its article's metadata: one search result."""
        return self.query(
            f"SELECT {ARTICLE_SUMMARY}, chunk_index, start_char, end_char, chunks.text AS passage "
            "FROM chunks JOIN articles ON articles.id = chunks.article_id WHERE chunk_id = ?",
            [chunk_id],
        )[0]

    # ---- analytics ----

    def aggregate(self, filters: dict, group_by: List[str], sentiment_model: str = "consensus",
                  order_by: str = "group", limit: int = 100) -> List[dict]:
        """
        Count articles and summarise sentiment per group.
        :param group_by: column names, e.g. ["country", "year"]; [] for one overall row
        :param sentiment_model: "consensus", "gemini", "gpt" or "deepseek"
        :param order_by: "group" (by group values) or "count" (largest groups first)
        """
        s = SENTIMENT_COLUMNS[sentiment_model]
        where, params = where_clause(filters)
        group_columns = ", ".join(group_by)
        rows = self.query(
            f"""
            SELECT {group_columns + ',' if group_by else ''}
                   COUNT(*) AS article_count,
                   COUNT({s}) AS sentiment_rated_count,
                   ROUND(AVG({s}), 3) AS mean_nuclear_sentiment,
                   ROUND(AVG({s} >= 3.5), 3) AS share_positive,
                   ROUND(AVG({s} <= 2.5), 3) AS share_negative,
                   ROUND(AVG(headline_sentiment), 3) AS mean_headline_sentiment,
                   ROUND(AVG(headline_alarmism), 3) AS mean_headline_alarmism
            FROM articles WHERE {where}
            {'GROUP BY ' + group_columns if group_by else ''}
            ORDER BY {group_columns if group_by and order_by == 'group' else 'article_count DESC'}
            LIMIT ?
            """,
            params + [limit],
        )
        # Evidence: for every group, the 3 articles with the clearest (least neutral) sentiment.
        for row in rows:
            group_condition = "".join(f" AND {column} = ?" for column in group_by)
            examples = self.query(
                f"SELECT id FROM articles WHERE {where}{group_condition} "
                f"ORDER BY ABS(COALESCE({s}, 3) - 3) DESC, date DESC LIMIT 3",
                params + [row[column] for column in group_by],
            )
            row["example_article_ids"] = [example["id"] for example in examples]
        return rows

    def list_articles(self, filters: dict, sort_by: str = "date", descending: bool = True,
                      limit: int = 20, offset: int = 0) -> dict:
        """Articles matching the filters, sorted by date or by a score column."""
        where, params = where_clause(filters)
        total = self.query(f"SELECT COUNT(*) AS n FROM articles WHERE {where}", params)[0]["n"]
        rows = self.query(
            f"SELECT {ARTICLE_SUMMARY} FROM articles WHERE {where} "
            f"ORDER BY {sort_by} IS NULL, {sort_by} {'DESC' if descending else 'ASC'}, id LIMIT ? OFFSET ?",
            params + [limit, offset],
        )
        return {"total_matching": total, "offset": offset, "articles": rows}

    def get_article(self, article_id: str) -> dict:
        """Full article with all scores, or None if the id is unknown."""
        rows = self.query("SELECT * FROM articles WHERE id = ?", [article_id])
        if not rows:
            return None
        article = rows[0]
        article["article_id"] = article.pop("id")
        return article

    def overview(self) -> dict:
        totals = self.query(
            "SELECT COUNT(*) AS articles, SUM(nuclear_related) AS nuclear_related_articles, "
            "MIN(date) AS first_date, MAX(date) AS last_date FROM articles"
        )[0]
        sources = self.query(
            "SELECT country, language, source, COUNT(*) AS articles, MIN(year) AS first_year, "
            "MAX(year) AS last_year FROM articles GROUP BY country, language, source ORDER BY country, source"
        )
        return {**totals, "sources": sources}
