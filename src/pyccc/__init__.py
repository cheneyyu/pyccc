"""CellChat-like CCC analysis and plotting for Scanpy/AnnData."""

from .analysis import CCCResult, compute_communication, compute_pathway_communication, identify_overexpressed_genes, identify_overexpressed_interactions
from .cellchat_bridge import export_cellchat, export_cellchat_merged
from .database import CellChatDB, filter_lr_table, load_cellchatdb, load_lr_table, load_omnipath_interactions, toy_lr_table
from .density import build_predicted_lr_table, estimate_lr_density_prior
from .diff import DifferentialCCC, compare_communication, compare_samples, pairwise_pathway_embedding, rank_pathway_similarity, signaling_changes
from .embeddings import embed_proteins_esmc
from .expression import signaling_expression_frame, signaling_expression_values
from .liana_bridge import from_liana_results, run_liana
from .lr_prediction import generate_lr_candidates_dbfree, predict_lr_dbfree, score_lr_candidates, train_lr_link_predictor
from .lr_resources import load_training_lr_resources, training_lr_to_cellchatdb
from .patterns import CommunicationPatterns, communication_pattern_matrix, compute_communication_patterns, compute_pathway_clusters, compute_pathway_embedding, compute_pathway_similarity, select_communication_pattern_number
from .pair_features import LRPairFeatures, make_lr_pair_features
from .report import save_cellchat_report
from .interactive import interactive_pathway_river, interactive_pattern_river, save_interactive_html
from .roles import predict_protein_roles, train_protein_role_classifier
from .sequence import load_cds_translations, load_protein_fasta, match_expression_genes
from .spatial_validation import SpatialValidationReport, validate_spatial_lr_table
from .training_data import LRTrainingTable, build_lr_training_table

__all__ = [
    "CCCResult",
    "CellChatDB",
    "CommunicationPatterns",
    "DifferentialCCC",
    "LRPairFeatures",
    "LRTrainingTable",
    "SpatialValidationReport",
    "build_lr_training_table",
    "build_predicted_lr_table",
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
    "filter_lr_table",
    "from_liana_results",
    "embed_proteins_esmc",
    "estimate_lr_density_prior",
    "generate_lr_candidates_dbfree",
    "identify_overexpressed_genes",
    "identify_overexpressed_interactions",
    "interactive_pathway_river",
    "interactive_pattern_river",
    "load_cellchatdb",
    "load_cds_translations",
    "load_lr_table",
    "load_omnipath_interactions",
    "load_protein_fasta",
    "load_training_lr_resources",
    "make_lr_pair_features",
    "match_expression_genes",
    "pairwise_pathway_embedding",
    "predict_lr_dbfree",
    "predict_protein_roles",
    "rank_pathway_similarity",
    "run_liana",
    "save_cellchat_report",
    "save_interactive_html",
    "score_lr_candidates",
    "select_communication_pattern_number",
    "signaling_expression_frame",
    "signaling_expression_values",
    "signaling_changes",
    "toy_lr_table",
    "train_lr_link_predictor",
    "train_protein_role_classifier",
    "training_lr_to_cellchatdb",
    "validate_spatial_lr_table",
]
