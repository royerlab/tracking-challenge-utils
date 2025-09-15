import tracksdata as td

from tracking_challenge_utils.metrics import _compute_score, _evaluate_matched_graph


def test_metric() -> None:
    # graph from google docs
    """"
    graph:

    0 - 1 - 2 - 4
          \\   \
            3   5

    gt_graph:

    0 - 1 - 2
          \
            3

    correspondence (node_id, gt_node_id):

    (1, 0), (2, 1), (5, 3)

    """
    graph = td.graph.InMemoryGraph()

    graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 0})
    graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 1})
    graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 2})
    graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 2})
    graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 3})
    graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 3})

    graph.add_edge(0, 1, {})
    edge_1 = graph.add_edge(1, 2, {})
    graph.add_edge(1, 3, {})
    graph.add_edge(2, 4, {})
    edge_4 = graph.add_edge(2, 5, {})

    graph.add_node_attr_key(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID, -1)
    graph.add_edge_attr_key(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK, False)

    graph.update_node_attrs(
        node_ids=[1, 2, 5],
        attrs={td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: [0, 1, 3]},
    )

    graph.update_edge_attrs(
        edge_ids=[edge_1, edge_4],
        attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: True},
    )

    gt_graph = td.graph.InMemoryGraph()

    gt_graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 1})
    gt_graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 2})
    gt_graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 3})
    gt_graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: 3})

    gt_graph.add_edge(0, 1, {})
    gt_graph.add_edge(1, 2, {})
    gt_graph.add_edge(1, 3, {})

    edge_attrs = _evaluate_matched_graph(graph, gt_graph)

    edge_overlap = (
        edge_attrs.filter(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK)
        .select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET)
        .to_numpy()
        .tolist()
    )

    # checking edges that overlap with ground truth (intersection)
    expected_overlap = [[1, 2], [2, 5]]
    assert len(edge_overlap) == len(expected_overlap)
    for expected_edge in expected_overlap:
        assert expected_edge in edge_overlap

    # checking edges where we can validate the prediction
    edge_pred_valid = (
        edge_attrs.filter("pred_valid")
        .select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET)
        .to_numpy()
        .tolist()
    )

    expected_pred_valid = [[1, 2], [1, 3], [2, 4], [2, 5]]

    assert len(edge_pred_valid) == len(expected_pred_valid)

    for expected_edge in expected_pred_valid:
        assert expected_edge in edge_pred_valid

    jaccard = _compute_score(edge_attrs, gt_graph.num_edges, metric="jaccard")

    assert jaccard == 2 / 5

    dice = _compute_score(edge_attrs, gt_graph.num_edges, metric="dice")

    assert dice == 4 / 7
