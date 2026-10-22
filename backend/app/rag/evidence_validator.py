
class EvidenceValidator:
    """
    Filters retrieved chunks using cosine distance.

    Lower distance means higher semantic similarity.
    This is a preliminary relevance filter, not proof
    that a chunk contains sufficient answer evidence.
    """

    def __init__(self, max_distance=0.65):
        if not 0 <= max_distance <= 2:
            raise ValueError(
                "max_distance must be between 0 and 2"
            )

        self.max_distance = max_distance

    def filter_relevant_chunks(self, results):
        """
        Return chunks whose cosine distance
        is within the configured threshold.
        """

        filtered = []

        for result in results:
            distance = result["distance"]

            if distance <= self.max_distance:
                filtered.append(result)

        return filtered

    def has_sufficient_evidence(self, results):
        """
        Preliminary check: at least one chunk passes
        the similarity threshold.

        This does not verify factual answerability.
        """

        filtered = self.filter_relevant_chunks(results)

        return len(filtered) > 0
