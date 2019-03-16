import re
import numpy as np
import spacy
import logging
from collections import Counter
from urllib.parse import unquote
from src.generators import DataGenerator


logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')

    def _samples_with_queries(self, data_generator):
        for sample in data_generator:
            # some requests may not have a query, which makes them invalid observations
            if "query" not in sample["request"]:
                continue
            yield sample

    def _is_attack(self, sample):
        return sample["class"]["type"] == "SqlInjection"

    def _tokenize(self, txt: str):
        return [self.nlp(w)[0] for w in re.split(r'\W+', txt) if w]

    def _find_black_tokens(self, data_generator, n_tokens: int):
        black_tokens = Counter()

        for sample in self._samples_with_queries(data_generator):
            if not self._is_attack(sample):
                continue

            attack = self._tokenize(unquote(sample["request"]["query"]))
            for token in attack:
                if token.is_alpha and token.has_vector and len(token.lower_) > 2:
                    black_tokens[token.lower_] += 1

        return [w for w, _ in black_tokens.most_common(n_tokens)]

    def _extract_features(self, sample, black_tokens):
        query = self._tokenize(unquote(sample["request"]["query"]))
        query_vector = np.zeros((300,))
        for token in query:
            if token.lower_ in black_tokens:
                query_vector += token.vector
        return query_vector / len(query)

    def extract(self, data_generator: DataGenerator):
        """
        Returns an iterator containing (features: np.ndarray, is_attack: bool) pairs
        """
        logger.debug("-----------------------")
        logger.debug("BLACK TOKENS:")
        logger.debug("-----------------------")
        black_tokens = self._find_black_tokens(data_generator, 300)
        logger.debug(str(black_tokens))
        features = []
        labels = []
        for i, sample in enumerate(self._samples_with_queries(data_generator)):
            logger.debug(f"extracting features ({i})")
            features.append(self._extract_features(sample, black_tokens))
            labels.append(self._is_attack(sample))
        return features, labels