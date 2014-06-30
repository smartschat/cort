from __future__ import division

__author__ = 'smartschat'


class MultigraphDecoder:
    def __init__(self, positive_features, negative_features, do_distance_reweighting=True):
        self.positive_features = positive_features
        self.negative_features = negative_features
        self.do_distance_reweighting = do_distance_reweighting

    def decode(self, corpus):
        for doc in corpus:
            for mention in doc.system_mentions:
                mention.attributes["set_id"] = None

            self.decode_for_one_document(doc.system_mentions)

    def decode_for_one_document(self, mentions):
        for i in range(0, len(mentions)):
            m = mentions[i]
            weights = []
            for j in range(i-1, -1, -1):
                n = mentions[j]

                negative_feature_active = False

                for negative_feature in self.negative_features:
                    if negative_feature(m, n):
                        negative_feature_active = True

                if negative_feature_active:
                    continue

                weight = 0

                for positive_feature in self.positive_features:
                    if positive_feature(m, n):
                        weight += 1

                if weight > 0:
                    if self.do_distance_reweighting:
                        weight /= m.attributes["sentence_id"] - n.attributes["sentence_id"] + 1
                    weights.append((weight, n))

            if len(weights) > 0 and sorted(weights)[-1][0] > 0:
                antecedent = sorted(weights)[-1][1]
                if antecedent.attributes["set_id"] is None:
                    antecedent.attributes["set_id"] = mentions.index(antecedent)
                m.attributes["set_id"] = antecedent.attributes["set_id"]