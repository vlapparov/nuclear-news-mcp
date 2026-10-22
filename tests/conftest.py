"""Shared test fixtures: a tiny hand-written corpus and a fake embedding model, so the tests
run in about a second without downloading anything."""

import hashlib
import re
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from news_mcp import config, server
from news_mcp.ingest import build_index


class FakeEmbeddings:
    """Bag-of-words hashing: texts that share words get similar vectors. Tokens = words."""

    def encode(self, texts):
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    def encode_queries(self, queries):
        return self.encode(queries)

    def encode_passages(self, passages):
        return self.encode(passages)

    def count_tokens(self, text):
        return len(text.split())


def make_row(id, country, language, source, date, title, body, rating=(3, 3, 3), alarm=(3, 3, 3)):
    """One article in the same format as the Hugging Face dataset."""
    row = {"id": id, "country": country, "language": language, "source": source, "date": date,
           "year": date[-4:], "url": f"https://example.org/{id}", "title": title, "article body": body}
    for i, model in enumerate(["gemini", "deepseek", "gpt"]):
        row[f"{model}_article_rating"] = rating[i]
        row[f"{model}_headline_sentiment"] = 3
        row[f"{model}_headline_alarmistic"] = alarm[i]
    return row


CORPUS = [
    make_row("1", "Czech Republic", "cz", "iROZHLAS", "15-03-2022", "Svět obchodoval s LNG",
             "Úvodní odstavec o energetice. Podle zprávy se v roce 2021 na světových trzích obchodovalo "
             "372 milionů tun LNG. Dodávky rostly hlavně do Asie."),
    make_row("2", "Czech Republic", "cz", "novinky.cz", "02-06-2021", "Dukovany získají nový blok",
             "Vláda schválila tendr na nový jaderný blok v Dukovanech.", rating=(5, 4, 5), alarm=(5, 5, 5)),
    make_row("3", "Poland", "pl", "Onet_pl", "10-01-2020", "Polska buduje elektrownię jądrową",
             "Rząd ogłosił plan budowy pierwszej elektrowni jądrowej na Pomorzu.", rating=(4, 4, 5)),
    make_row("4", "Poland", "pl", "Onet_pl", "11-05-2020", "Atom w Polsce budzi emocje",
             "Mieszkańcy obawiają się skutków budowy elektrowni jądrowej.", rating=(2, 2, 2), alarm=(2, 2, 1)),
    make_row("5", "Poland", "pl", "Fakt_pl", "12-07-2020", "Mecz piłkarski w Gdańsku",
             "Relacja z meczu piłkarskiego. Nie ma tu nic o atomie.", rating=(0, 0, 3)),
    make_row("6", "Hungary", "hu", "index.hu", "05-04-2023", "Atomfegyver Oroszország",
             "Oroszország atomfegyver telepítését jelentette be Fehéroroszországban.", rating=(2, 1, 2)),
    make_row("7", "Slovakia", "sk", "sme.sk", "06-04-2023", "Jadrové zbrane v Bielorusku",
             "Rusko chce rozmiestniť jadrové zbrane v Bielorusku, varujú analytici.", rating=(2, 2, 2)),
    make_row("8", "Slovakia", "sk", "pravda.sk", "20-09-2022", "Mochovce spustili tretí blok",
             "Tretí blok jadrovej elektrárne Mochovce dosiahol plný výkon.", rating=(5, 5, 4)),
    make_row("9", "Hungary", "hu", "origo.hun", "21-10-2022", "Paks II engedélyt kapott",
             "A Paks II atomerőmű építési engedélyt kapott az országos hivataltól.", rating=(4, 4, 4)),
]


@pytest.fixture(scope="session")
def index(tmp_path_factory):
    """Build the test index once and load it into the server module."""
    data_dir = str(tmp_path_factory.mktemp("data"))
    build_index(CORPUS, data_dir, FakeEmbeddings())
    server.load_index(data_dir, model=FakeEmbeddings())
    return server


@pytest.fixture(scope="session")
def server_url(index):
    """Run the real HTTP server in a background thread, with an extra search-only token."""
    config.TOKENS["search-only-token"] = {"client_id": "partner", "scopes": ["news:search"]}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = server.mcp.streamable_http_app(host="127.0.0.1")
    http_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=http_server.run, daemon=True).start()
    while not http_server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    http_server.should_exit = True
    del config.TOKENS["search-only-token"]
