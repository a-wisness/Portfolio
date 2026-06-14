"""Tests for bot/modules/qa.py — KB CRUD and FTS5 search."""
from bot.modules.qa import add_entry, delete_entry, list_entries, search

# Use guild IDs unlikely to clash with test_history.py
G = 500


async def test_add_returns_entry():
    e = await add_entry(G, "Python basics", "Python is a programming language.")
    assert e.id is not None
    assert e.title == "Python basics"
    assert e.guild_id == G


async def test_search_finds_entry():
    await add_entry(G + 1, "FastAPI intro", "FastAPI is a modern web framework.")
    results = await search(G + 1, "FastAPI", limit=5)
    assert len(results) >= 1
    assert any(r["title"] == "FastAPI intro" for r in results)


async def test_search_empty_query_returns_nothing():
    results = await search(G, "   ", limit=5)
    assert results == []


async def test_search_no_match():
    await add_entry(G + 2, "Cats", "Cats are independent animals.")
    results = await search(G + 2, "supercalifragilistic", limit=5)
    assert results == []


async def test_search_guild_isolation():
    """A query must not return entries from a different guild."""
    await add_entry(G + 3, "Secret topic", "Confidential guild content.")
    # Search from a different guild — should find nothing.
    results = await search(G + 4, "Secret", limit=5)
    assert all(r["title"] != "Secret topic" for r in results)


async def test_delete_removes_entry():
    e = await add_entry(G + 5, "Temp entry", "Will be deleted.")
    deleted = await delete_entry(e.id, G + 5)
    assert deleted is True
    results = await search(G + 5, "Temp entry", limit=5)
    assert all(r["title"] != "Temp entry" for r in results)


async def test_delete_wrong_guild_fails():
    e = await add_entry(G + 6, "Cross-guild entry", "Belongs to guild G+6.")
    deleted = await delete_entry(e.id, G + 7)  # wrong guild
    assert deleted is False


async def test_delete_nonexistent_returns_false():
    deleted = await delete_entry(99999, G)
    assert deleted is False


async def test_list_entries_ordered_by_recency():
    await add_entry(G + 8, "First", "content a")
    await add_entry(G + 8, "Second", "content b")
    entries = await list_entries(G + 8)
    titles = [e.title for e in entries]
    assert titles.index("Second") < titles.index("First")


async def test_list_entries_empty_guild():
    entries = await list_entries(G + 9)
    assert entries == []
