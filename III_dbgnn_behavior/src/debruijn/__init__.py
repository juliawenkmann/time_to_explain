"""Utilities specific to De Bruijn / higher-order representations."""

from .translate import (
    ho_edge_df,
    ho_path_df,
    project_ho_edges_to_first_order_edges,
)

from .order2 import (
    Order2DeBruijn,
    build_order2_debruijn,
    pair_cluster,
    pair_group_id,
    pair_group_labels,
    sort_pairs_by_group,
    reorder_adjacency,
    block_matrix_from_edges,
    spectral_embedding_undirected,
)

__all__ = [
    "ho_edge_df",
    "ho_path_df",
    "project_ho_edges_to_first_order_edges",
    "Order2DeBruijn",
    "build_order2_debruijn",
    "pair_cluster",
    "pair_group_id",
    "pair_group_labels",
    "sort_pairs_by_group",
    "reorder_adjacency",
    "block_matrix_from_edges",
    "spectral_embedding_undirected",
]
