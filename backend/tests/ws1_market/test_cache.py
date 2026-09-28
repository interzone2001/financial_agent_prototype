from datetime import timedelta

from app.cache import SqliteCache

TTL = timedelta(seconds=60)


def test_roundtrip_returns_body_and_fetch_time(tmp_path, clock):
    cache = SqliteCache(tmp_path / "c.sqlite", clock=clock)
    cache.set("GLOBAL_QUOTE:AAPL", {"a": 1}, clock())
    assert cache.get("GLOBAL_QUOTE:AAPL", TTL) == ({"a": 1}, clock())


def test_miss_returns_none(tmp_path, clock):
    assert SqliteCache(tmp_path / "c.sqlite", clock=clock).get("nope", TTL) is None


def test_entry_expires_at_ttl(tmp_path, clock):
    cache = SqliteCache(tmp_path / "c.sqlite", clock=clock)
    cache.set("k", {"a": 1}, clock())
    clock.advance(seconds=59)
    assert cache.get("k", TTL) is not None
    clock.advance(seconds=1)
    assert cache.get("k", TTL) is None


def test_set_overwrites_and_persists_across_instances(tmp_path, clock):
    path = tmp_path / "nested" / "c.sqlite"  # parent dir must be created
    SqliteCache(path, clock=clock).set("k", {"v": 1}, clock())
    SqliteCache(path, clock=clock).set("k", {"v": 2}, clock())
    assert SqliteCache(path, clock=clock).get("k", TTL)[0] == {"v": 2}
