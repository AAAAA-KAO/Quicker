from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _to_search_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_to_search_text(item) for item in value.values()).strip()
    if isinstance(value, Iterable):
        return " ".join(_to_search_text(item) for item in value).strip()
    return str(value)


def _to_candidate_list(candidates) -> list:
    if candidates is None:
        return []
    if isinstance(candidates, str):
        return [candidates]
    if isinstance(candidates, Iterable):
        return list(candidates)
    return [candidates]


def similarity_match(query, candidates, threshold=0.9):
    """
    Check whether there are similar strings in the candidates whose similarity exceeds the threshold. If so, return the most similar string.

    Args:
        query: The string or string-like structure to compare.
        candidates: A candidate or a list of candidates to compare.
        threshold (float): The threshold of similarity. The default is 0.9.

    Returns:
        The most similar original candidate. If there is no similar candidate, return None.
    """

    # Initialize the most similar string and the highest similarity.
    query_text = _to_search_text(query)
    if not query_text:
        return None

    most_similar_candidate = None
    highest_similarity = 0
    vectorizer = TfidfVectorizer()

    # Compare the similarity between the query and each candidate.
    for candidate in _to_candidate_list(candidates):
        candidate_text = _to_search_text(candidate)
        if not candidate_text:
            continue
        tfidf_matrix = vectorizer.fit_transform([query_text, candidate_text])
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity = cos_sim[0][0]

        # Update the most similar string and the highest similarity if the similarity exceeds the threshold.
        if similarity > threshold and similarity > highest_similarity:
            most_similar_candidate = candidate
            highest_similarity = similarity

    return most_similar_candidate


if __name__ == "__main__":
    query = "Patients with dementia and agitation\/aggressive behavior"
    candidates = [
        "Patients with dementia and agitation/aggressive behavior",
        "Patients with dementia and agitation/aggressive behavior.",
        "Patients with dementia and agitation\aggressive behavior",
    ]
    result = similarity_match(query, candidates)
    print(result)  # Output: I want to buy a computer
