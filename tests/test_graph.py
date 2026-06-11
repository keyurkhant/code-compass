from pathlib import Path

import pytest

from codecompass.graph.model import GraphDB, GraphEdge, GraphNode, make_node_id


@pytest.fixture
def db(tmp_path):
    return GraphDB(tmp_path / "test_graph.db")


def test_upsert_and_get_node(db):
    node = GraphNode(id=make_node_id("r", "a.py"), repo="r", path="a.py", node_type="file")
    db.upsert_node(node)
    result = db.get_node(node.id)
    assert result is not None
    assert result.path == "a.py"


def test_impact_of_change(db):
    a_id = make_node_id("r", "a.py")
    b_id = make_node_id("r", "b.py")
    c_id = make_node_id("r", "c.py")
    for nid, path in [(a_id, "a.py"), (b_id, "b.py"), (c_id, "c.py")]:
        db.upsert_node(GraphNode(id=nid, repo="r", path=path, node_type="file"))
    db.upsert_edge(GraphEdge(source_id=b_id, target_id=a_id, edge_type="imports"))
    db.upsert_edge(GraphEdge(source_id=c_id, target_id=b_id, edge_type="imports"))

    impacted = db.impact_of_change(a_id)
    assert b_id in impacted
    assert c_id in impacted


def test_fts_search(db):
    node = GraphNode(
        id=make_node_id("r", "auth.py", "authenticate"),
        repo="r",
        path="auth.py",
        symbol_name="authenticate",
        node_type="function",
        summary="Handles user authentication and session management",
    )
    db.upsert_node(node)
    results = db.fts_search("authentication")
    assert any(n.symbol_name == "authenticate" for n in results)


def test_incremental_skip(tmp_path):
    from codecompass.graph.extractor import build_graph

    tiny_repo = Path(__file__).parent / "fixtures" / "tiny_repo"
    db = GraphDB(tmp_path / "g.db")
    stats1 = build_graph(tiny_repo, "tiny", db, incremental=True)
    stats2 = build_graph(tiny_repo, "tiny", db, incremental=True)
    assert stats2["files_skipped"] == stats1["files_processed"] + stats1["files_skipped"]


def test_stats(db):
    db.upsert_node(GraphNode(id="x", repo="r", path="x.py", node_type="file"))
    s = db.stats()
    assert s["nodes"] >= 1


def test_delete_file(db):
    nid = make_node_id("r", "a.py")
    db.upsert_node(GraphNode(id=nid, repo="r", path="a.py", node_type="file"))
    db.delete_file("r", "a.py")
    assert db.get_node(nid) is None
