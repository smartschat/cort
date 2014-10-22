__author__ = 'smartschat'


def for_each_relation_with_distance(anaphor,
                                    antecedent,
                                    relations,
                                    relation_weights):
    weight = 0.0

    if len(relations["negative_relations"]) > 0:
        return float("-inf")

    if len(relations["positive_relations"]) == 0:
        return 0

    for relation in relations["positive_relations"]:
        weight += relation_weights[relation]

    weight /= (anaphor.attributes["sentence_id"] -
               antecedent.attributes["sentence_id"]
               + 1)

    return weight
