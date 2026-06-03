"""CellChat-like CCC analysis and plotting for Scanpy/AnnData."""

from .analysis import CCCResult, compute_communication, compute_pathway_communication, identify_overexpressed_genes, identify_overexpressed_interactions
from .cellchat_bridge import export_cellchat, export_cellchat_merged
from .database import CellChatDB, load_cellchatdb, load_lr_table, toy_lr_table
from .diff import DifferentialCCC, compare_communication, compare_samples, pairwise_pathway_embedding, rank_pathway_similarity, signaling_changes
from .expression import signaling_expression_frame, signaling_expression_values
from .liana_bridge import from_liana_results, run_liana
from .patterns import CommunicationPatterns, communication_pattern_matrix, compute_communication_patterns, compute_pathway_clusters, compute_pathway_embedding, compute_pathway_similarity, select_communication_pattern_number
from .report import save_cellchat_report
from .interactive import interactive_pathway_river, interactive_pattern_river, save_interactive_html

__all__ = [
    "CCCResult",
    "CellChatDB",
    "CommunicationPatterns",
    "DifferentialCCC",
    "communication_pattern_matrix",
    "compare_communication",
    "compare_samples",
    "compute_communication",
    "compute_communication_patterns",
    "compute_pathway_clusters",
    "compute_pathway_embedding",
    "compute_pathway_communication",
    "compute_pathway_similarity",
    "export_cellchat",
    "export_cellchat_merged",
    "from_liana_results",
    "identify_overexpressed_genes",
    "identify_overexpressed_interactions",
    "interactive_pathway_river",
    "interactive_pattern_river",
    "load_cellchatdb",
    "load_lr_table",
    "pairwise_pathway_embedding",
    "rank_pathway_similarity",
    "run_liana",
    "save_cellchat_report",
    "save_interactive_html",
    "select_communication_pattern_number",
    "signaling_expression_frame",
    "signaling_expression_values",
    "signaling_changes",
    "toy_lr_table",
]
