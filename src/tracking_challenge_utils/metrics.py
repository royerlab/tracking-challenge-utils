from typing import Literal

import polars as pl
import tracksdata as td


# function is split for easier testing
def _evaluate_matched_graph(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
) -> pl.DataFrame:
    edge_attrs = graph.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK])
    node_attrs = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])

    # I'm assuming valid ground-truth edges are always 100% correct if they have an edge.
    # Therefore, we don't have cases where the cell divided, but not in the ground truth.
    gt_node_ids = gt_graph.node_ids()
    gt_node_attrs = pl.DataFrame(
        {
            td.DEFAULT_ATTR_KEYS.NODE_ID: gt_node_ids,
            "out_degree": gt_graph.out_degree(gt_node_ids),
            "in_degree": gt_graph.in_degree(gt_node_ids),
        }
    ).with_columns(
        (pl.col("out_degree") > 0).alias("out_valid"),
        (pl.col("in_degree") > 0).alias("in_valid"),
    )

    # merging ground truth graph into the predicted graph
    node_attrs = node_attrs.join(
        gt_node_attrs,
        left_on=td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).with_columns(
        pl.col("out_valid").fill_null(False),
        pl.col("in_valid").fill_null(False),
    )

    # merge out valid into source and in valid into target
    edge_attrs = edge_attrs.join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, "out_valid"),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, "in_valid"),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    )

    edge_attrs = edge_attrs.with_columns(
        (pl.col("out_valid") | pl.col("in_valid")).alias("pred_valid"),
    )

    # sanity check that `pred_valid` is a superset of all matched edges
    assert edge_attrs.filter(td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK)["pred_valid"].all()

    return edge_attrs


def _compute_score(
    edge_attrs: pl.DataFrame,
    gt_num_edges: int,
    metric: Literal["jaccard", "dice"],
) -> float:
    intersection = edge_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK].sum()
    n_valid_pred_edges = edge_attrs["pred_valid"].sum()

    if metric == "jaccard":
        score = intersection / (gt_num_edges + n_valid_pred_edges - intersection)
    elif metric == "dice":
        score = 2 * intersection / (gt_num_edges + n_valid_pred_edges)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    return score


def evaluate(
    graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    metric: Literal["jaccard", "dice"] = "jaccard",
) -> float:
    """
    Evaluate the performance of a graph against a ground truth graph.
    Both graphs must have a `mask` attribute used for computing the matching score.

    Parameters
    ----------
    graph : tracksdata.graph.BaseGraph
        The predicted graph.
    gt_graph : tracksdata.graph.BaseGraph
        The ground truth graph.
    metric : Literal["jaccard", "dice"], optional
        The metric to use for evaluation, by default "jaccard".

    Returns
    -------
    float
        The score of the graph.
    """

    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in graph.node_attr_keys:
        raise ValueError("Graph already matched")

    for g_name, g in [("graph", graph), ("gt_graph", gt_graph)]:
        if td.DEFAULT_ATTR_KEYS.MASK not in g.node_attr_keys:
            raise ValueError(
                f"`mask` attribute not found in '{g_name}'. "
                "Call `td.nodes.MaskDiskAttrs(...).add_node_attrs(g)` to add it."
            )

    graph.match(gt_graph)

    edge_attrs = _evaluate_matched_graph(graph, gt_graph)

    return _compute_score(edge_attrs, gt_graph.num_edges, metric)
