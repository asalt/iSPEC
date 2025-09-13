from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches
import numpy as np
import logging

from ispec.logging import get_logger

logger = get_logger(__file__)

# Preload model
_default_model = SentenceTransformer("all-MiniLM-L6-v2")


def encode_column_names(column_names: List[str], model: Optional[SentenceTransformer] = None):
    model = model or _default_model
    return model.encode(column_names, convert_to_tensor=True)


def score_matches(
    source_columns: List[str],
    target_columns: List[str],
    model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """
    Compute similarity score matrix between source and target column names.
    Returns a NumPy array of shape (len(source_columns), len(target_columns)).
    """
    model = model or _default_model
    source_emb = encode_column_names(source_columns, model)
    target_emb = encode_column_names(target_columns, model)
    sim_matrix = util.cos_sim(source_emb, target_emb).cpu().numpy()
    return sim_matrix


def match_columns(
    source_columns: List[str],
    target_columns: List[str],
    model: Optional[SentenceTransformer] = None,
    threshold: float = 0.6,
    fallback: bool = True,
    verbose: bool = False,
) -> Dict[str, Optional[str]]:
    """
    Match source column names to target columns using semantic similarity.
    Falls back to difflib.get_close_matches if score below threshold.

    Parameters
    ----------
    source_columns : List[str]
        Columns we are trying to map.
    target_columns : List[str]
        Possible target columns to match against.
    model : Optional[SentenceTransformer]
        Embedding model to use; defaults to a preloaded miniLM model.
    threshold : float, optional
        Minimum similarity score to accept a match, by default 0.6.
    fallback : bool, optional
        When True, attempt a difflib-based match if semantic match is below
        threshold.
    verbose : bool, optional
        If True, emit debug logs detailing the matching process.
    """
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    model = model or _default_model
    sim_matrix = score_matches(source_columns, target_columns, model)

    matches = {}
    for i, src in enumerate(source_columns):
        logger.debug("looking for best match for %s", src)
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = sim_matrix[i][best_idx]
        if best_score >= threshold:
            best_match = target_columns[best_idx]
            logger.debug("best match : %s", best_match)
            matches[src] = best_match
        elif fallback:
            # Use difflib if transformer match is too weak
            logger.debug("couldn't find a top match")
            close = get_close_matches(src, target_columns, n=1, cutoff=0.0)
            logger.debug("close matches : %s", close)
            matches[src] = close[0] if close else None
        else:
            matches[src] = None
    return matches


def print_column_matches(match_dict: Dict[str, Optional[str]]):
    for src, tgt in match_dict.items():
        arrow = "→" if tgt else "↛"
        match_str = tgt if tgt else "(no good match)"
        print(f"{src:25s} {arrow} {match_str}")


if __name__ == "__main__":
    # from column_matching import match_columns, print_column_matches

    expected = ["PatientID", "Age", "Sex", "BloodPressure"]
    incoming = ["pat_id", "years_old", "gender", "bp"]

    matches = match_columns(expected, incoming)
    print_column_matches(matches)

    # View full similarity matrix if needed
    sim_matrix = score_matches(expected, incoming)
    print(sim_matrix)
