""" Contains features for coreference resolution.

Features takes as input either a mention or a pair of mentions, and always
output a list of ``(feature_name, value)`` tuples. ``feature_name`` is the name of the
feature (a string), and ``value`` is the value of the feature. ``value`` can be
either a string, a bool, or a scalar.
"""

from __future__ import division


import re


from cort.core import spans, external_data


__author__ = 'smartschat'


def fine_type(mention):
    """ Compute fine-grained type of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('fine_type', TYPE), where where TYPE is one of 'NAM', 'DEF',
        'NOTDEF', 'DEM', 'VRB', 'i', 'you', 'he', 'she', 'it', 'we' and 'they'.
    """
    return [("fine_type", mention.attributes["fine_type"])]


def gender(mention):
    """ Compute gender of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('gender', GENDER), where GENDER is one of 'MALE',
        'FEMALE', 'NEUTRAL', 'PLURAL' and 'UNKNOWN'.
    """
    return [("gender", mention.attributes["gender"])]


def number(mention):
    """ Compute number of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('number', NUMBER), where NUMBER is one of 'SINGULAR',
        'PLURAL' and 'UNKNOWN'.
    """
    return [("number", mention.attributes["number"])]


def sem_class(mention):
    """ Compute semantic class of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('sem_class', SEM_CLASS), where SEM_CLASS is one of
        'PERSON', 'OBJECT', 'NUMERIC' and 'UNKNOWN'.
    """
    return [("sem_class", mention.attributes["semantic_class"])]


def gr_func(mention):
    """ Compute grammatical function of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('gr_func', GR_FUNC), where GR_FUNC is one of 'SUBJECT',
        'OBJECT' and 'OTHER'.
    """
    return [("gr_func", mention.attributes["grammatical_function"])]


def governor(mention):
    """ Compute governor of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('governor', GOVERNOR), where GOVERNOR is the governor
        of the mention.
    """
    return [("governor", mention.attributes["governor"].lower())]


def deprel(mention):
    """ Compute dependency relation of a mention to its governor.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('deprel', DEPREL), where DEPREL is the dependency relation
        of the mention to its governor.
    """
    return [("deprel", mention.attributes["deprel"])]


def head(mention):
    """ Compute head of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('head', HEAD), where HEAD is the (lowercased) head of
        the mention.
    """
    return [("head", mention.attributes["head_as_lowercase_string"])]


def head_ner(mention):
    """ Compute named entity tag of a mention's head.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('ner', NER), where NER is the named entity tag of the
        mention's head word. If the mention is not a named entity, NER is
        set to 'NONE'.
    """
    return [("ner", mention.attributes["ner"][mention.attributes["head_index"]])]


def length(mention):
    """ Compute length of a mention in tokens.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('length' LENGTH), where LENGTH is the length of the
        mention in tokens. The length is stored as a string.
    """
    return [("length", str(len(mention.attributes["tokens"])))]


def first(mention):
    """ Compute the first token of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('first', TOKEN), where TOKEN is the (lowercased) first
        token of the mention.
    """
    return [("first", mention.attributes["tokens"][0].lower())]


def last(mention):
    """ Compute the last token of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('last', TOKEN), where TOKEN is the (lowercased) last token
        of the mention.
    """
    return [("last", mention.attributes["tokens"][-1].lower())]


def preceding_token(mention):
    """ Compute the token preceding a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('preceding', TOKEN), where TOKEN is the (lowercased) token
        preceding the mention. If no such token exists, set TOKEN to 'NONE'.
    """
    prec = mention.get_context(-1)

    if prec:
        return [("preceding", prec[0].lower())]
    else:
        return [("preceding", "NONE")]


def next_token(mention):
    """ Compute the token following a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('next', TOKEN), where TOKEN is the (lowercased) token
        following the mention. If no such token exists, set TOKEN to 'NONE'.
    """
    next_t = mention.get_context(1)

    if next_t:
        return [("next", next_t[0].lower())]
    else:
        return [("next", "NONE")]


def ancestry(mention):
    """ Compute the ancestry of a mention.

    We follow the definition of the ancestry by Durrett and Klein (2013). For
    more information, have a look at their paper:

    Greg Durrett and Dan Klein. Easy Victories and Uphill Battles in
    Coreference Resolution. In Proceedings of EMNLP 2013.
    http://anthology.aclweb.org/D/D13/D13-1203.pdf

    Args:
        mention (Mention): A mention.

    Returns:
        A list containing the tuple ('ancestry', ANCESTRY), where ANCESTRY is the ancestry of
        the mention.
    """
    return [("ancestry", mention.attributes["ancestry"])]


def exact_match(anaphor, antecedent):
    """ Compute whether the tokens of two mentions match exactly.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('exact_match', MATCH), where MATCH is True if the mentions
        match exactly (ignoring case), and False otherwise.
    """

    match = anaphor.attributes["tokens_as_lowercase_string"] == \
            antecedent.attributes["tokens_as_lowercase_string"]

    return [("exact_match", match)]


def head_match(anaphor, antecedent):
    """ Compute whether the heads of two mentions match.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('head_match', MATCH), where MATCH is True if the heads of
        the mentions match exactly (ignoring case), and False otherwise.
    """
    match = anaphor.attributes["head_as_lowercase_string"] == \
            antecedent.attributes["head_as_lowercase_string"]

    return [("head_match", match)]


def tokens_contained(anaphor, antecedent):
    """ Compute whether one mention is a substring of the other mention.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('tokens_contained', CONTAINED), where CONTAINED is True if
        the anaphor contains the antecedent, or vice versa (ignoring case).
    """
    ana_tokens = anaphor.attributes["tokens_as_lowercase_string"]
    ante_tokens = antecedent.attributes["tokens_as_lowercase_string"]

    contained = ana_tokens in ante_tokens or ante_tokens in ana_tokens

    return [("tokens_contained", contained)]


def head_contained(anaphor, antecedent):
    """ Compute whether one mention's head is a substring of the other
    mention's head.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('head_contained', CONTAINED), where CONTAINED is True if
        the anaphor's head contains the antecedent's head, or vice versa
        (ignoring case).
    """
    ana_head = anaphor.attributes["head_as_lowercase_string"]
    ante_head = antecedent.attributes["head_as_lowercase_string"]

    contained = ana_head in ante_head or ante_head in ana_head

    return [("head_contained", contained)]


def sentence_distance(anaphor, antecedent):
    """ Compute the sentence distance between two mentions (capped at 5).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('sentence_distance', DIST), where DIST is one of '0', '1',
        '2', '3', '4' and '>=5'.
    """
    return [("sentence_distance", __compute_sentence_distance(anaphor,
                                                            antecedent))]


def token_distance(anaphor, antecedent):
    """ Compute the token distance between two mentions (capped at 10).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('token_distance'=DIST), where DIST is one of '0', '1',
        '2', '3', '4' and '>=10'.
    """
    return [("token_distance", __compute_token_distance(anaphor, antecedent))]

def alias(anaphor, antecedent):
    """ Compute whether the mentions are aliases of each other.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('alias', ALIAS), where ALIAS is True if anaphor and
        antecedent are in an alias relation, False otherwise.
    """
    return [("alias", __are_alias(anaphor, antecedent))]


def same_speaker(anaphor, antecedent):
    """ Compute whether the speakers  of two mentions are the same.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('same_speaker', SAME), where SAME is True if anaphor and
        antecedent have the same speaker, False otherwise.
    """
    same = anaphor.attributes["speaker"] == antecedent.attributes["speaker"]

    return [("same_speaker", same)]


def embedding(anaphor, antecedent):
    """ Compute whether one mention embeds the other.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('embedding', EMB), where EMB is True if one of the mentions
        embeds the other, False otherwise.
    """
    emb = anaphor.span.embeds(antecedent.span) or \
          antecedent.span.embeds(anaphor.span)

    return [("embedding", emb)]


def modifier(anaphor, antecedent):
    """ Compute modifier agreement.

    In particular, compute whether the anaphor has modifiers that do not appear
    in the antecedent (ignoring demonstratives and prepositions).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('modifier', MOD), where MOD is True if the anaphor has
        modifiers that do not appear in the antecedent, False otherwise.
    """
    mod = not __get_modifier(anaphor).issubset(__get_modifier(antecedent))

    return [("modifier", mod)]


def relative_overlap(anaphor, antecedent):
    """ Compute relative overlap of the mentions (ignoring case).

    For example, "the new president" and "the president" have relative overlap
    2/3.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('relative_overlap', OVERLAP), where OVERLAP (a float) is
        the relative overlap of anaphor and antecedent.
    """
    ana_tokens = set([tok.lower() for tok in anaphor.attributes["tokens"]])
    ante_tokens = set([tok.lower() for tok
                             in antecedent.attributes["tokens"]])

    overlap = len(ana_tokens & ante_tokens)/max(len(ana_tokens),
                                                len(ante_tokens))

    return [("relative_overlap", overlap)]


def __compute_sentence_distance(anaphor, antecedent):
    dist = anaphor.attributes['sentence_id'] - antecedent.attributes[
        'sentence_id']

    if dist >= 5:
        return ">=5"
    else:
        return str(dist)


def __compute_token_distance(anaphor, antecedent):
    dist = anaphor.span.begin - antecedent.span.end
    if dist >= 10:
        return ">=10"
    else:
        return str(dist)


def __get_modifier(mention):
    head_span_in_mention = spans.Span(
        mention.attributes["head_span"].begin - mention.span.begin,
        mention.attributes["head_span"].end - mention.span.begin)

    modifiers = set()

    for index, (token, pos) in enumerate(
            zip(mention.attributes["tokens"], mention.attributes["pos"])):
        if (token.lower() not in ["the", "this", "that", "those", "these",
                                  "a", "an"]
            and pos not in ["POS", "IN"]
            and (index < head_span_in_mention.begin
                 or index > head_span_in_mention.end)):
            modifiers.add(token.lower())

    return modifiers


def __are_alias(anaphor, antecedent):
    if anaphor.attributes["type"] != "NAM" or antecedent.attributes["type"] \
            != "NAM":
        return False
    elif anaphor.attributes["head_as_lowercase_string"] == \
            antecedent.attributes["head_as_lowercase_string"]:
        return False
    else:
        anaphor_cleaned_tokens = anaphor.attributes["head"]
        antecedent_cleaned_tokens = antecedent.attributes["head"]

        category = __get_category_for_alias(
            anaphor.attributes["ner"][
                anaphor.attributes["head_index"]
            ], antecedent.attributes["ner"][
                antecedent.attributes["head_index"]]
        )

        if category == "PERSON":
            return __person_alias(anaphor_cleaned_tokens,
                                  antecedent_cleaned_tokens)
        elif category == "LOC":
            return __loc_alias(anaphor_cleaned_tokens,
                               antecedent_cleaned_tokens)
        elif category == "ORG":
            return __org_alias(anaphor_cleaned_tokens,
                               antecedent_cleaned_tokens)
        else:
            return False


def __get_category_for_alias(anaphor_ner, antecedent_ner):
    if anaphor_ner == "PERSON" and antecedent_ner == "PERSON":
        return "PERSON"
    elif re.match(r"LOC", anaphor_ner) and re.match(r"LOC", antecedent_ner):
        return "LOC"
    elif re.match(r"ORG", anaphor_ner) and re.match(r"(ORG)", antecedent_ner):
        return "ORG"


def __loc_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    return __starts_with(anaphor_cleaned_tokens, antecedent_cleaned_tokens) or \
        __is_abbreviation(anaphor_cleaned_tokens, antecedent_cleaned_tokens)


def __org_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    return __starts_with(anaphor_cleaned_tokens, antecedent_cleaned_tokens) or \
        __is_abbreviation(anaphor_cleaned_tokens, antecedent_cleaned_tokens)


def __person_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    if len(anaphor_cleaned_tokens) == 1 or len(antecedent_cleaned_tokens) == 1:
        return anaphor_cleaned_tokens[0] == antecedent_cleaned_tokens[0] \
            or anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1]
    elif (len(anaphor_cleaned_tokens) == 2 and anaphor_cleaned_tokens[
        0].lower() in ["mr", "ms", "mr.", "ms."]
        or len(antecedent_cleaned_tokens) == 2 and antecedent_cleaned_tokens[
            0].lower() in ["mr", "ms", "mr.", "ms."]):
        return anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1]
    elif anaphor_cleaned_tokens[0] == antecedent_cleaned_tokens[0] and \
            anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1]:
        return True
    elif len(anaphor_cleaned_tokens) > 1 and len(antecedent_cleaned_tokens) > 1:
        return anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1] and \
            anaphor_cleaned_tokens[-2] == antecedent_cleaned_tokens[-2]

    return False


def __starts_with(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    for ana_token, ante_token in zip(anaphor_cleaned_tokens,
                                     antecedent_cleaned_tokens):
        if ana_token != ante_token:
            return False

    return True


def __is_abbreviation(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    if " ".join(anaphor_cleaned_tokens).replace(".", "") == " ".join(
            antecedent_cleaned_tokens).replace(".", ""):
        return True
    else:
        if len(anaphor_cleaned_tokens) > len(antecedent_cleaned_tokens):
            return " ".join(antecedent_cleaned_tokens) in set(
                __get_acronyms(anaphor_cleaned_tokens))
        else:
            return " ".join(anaphor_cleaned_tokens) in set(
                __get_acronyms(antecedent_cleaned_tokens))


def __get_acronyms(cleaned_tokens):
    company_designator = r'assoc|bros|co|coop|corp|devel|inc|llc|ltd\.?'
    tokens_without_designator = [token for token in cleaned_tokens if
                                 not re.match(company_designator,
                                              token.lower())]

    return " ".join(tokens_without_designator), \
           "".join([token[0] for token in tokens_without_designator if
                    token[0].isupper()]), \
           ".".join([token[0] for token in tokens_without_designator if
                     token[0].isupper()]) + "."


def cluster_size_antecedent_cluster(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute size of antecedent cluster.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing the tuples ('cluster_size-', CL_SIZE), where CL_SIZE (a string) is
        the binned size (1, 2, 3, 4, 5, <=10, <=15, <=20, 20+) of the antecedent cluster. Also
        output feature combinations with the fine-grained type of the last mention in the antecedent
        cluster and the first mention in the anaphor cluster.
    """
    first_in_ana = cluster_anaphor[-1]
    last = cluster_antecedent[0]
    ft_a = first_in_ana.attributes["fine_type"]
    ft_b = last.attributes["fine_type"]

    size = _get_bin(len(cluster_antecedent))

    return [
        ("cluster_size-", size),
        ("cluster_size-ana-" + ft_a, size),
        ("cluster_size-ante-" + ft_b, size),
        ("cluster_size-ana-ante-" + ft_a + "-" + ft_b, size)
    ]


def cluster_size_both_clusters(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute sizes of anaphor and antecedent clusters.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing the tuples ('cluster_size-', CL_SIZE_A + "-" + CL_SIZE_B ),
        where CL_SIZE_A (a string) is the binned size (1, 2, 3, 4, 5, <=10, <=15, <=20, 20+)
        of the anaphor cluster, and CL_SIZE_B the analogously represented size of the antecedent
        cluster. Also output feature combinations with the fine-grained type of the last mention
        in the antecedent cluster and the first mention in the anaphor cluster.
    """
    first_in_ana = cluster_anaphor[-1]
    last = cluster_antecedent[0]
    ft_a = first_in_ana.attributes["fine_type"]
    ft_b = last.attributes["fine_type"]

    size = _get_bin(len(cluster_anaphor)) + "-" + _get_bin(len(cluster_antecedent))

    return [
        ("cluster_size-", size),
        ("cluster_size-ana-" + ft_a, size),
        ("cluster_size-ante-" + ft_b, size),
        ("cluster_size-ana-ante-" + ft_a + "-" + ft_b, size)
    ]


def _get_bin(number):
    if number <= 5:
        return str(number)
    elif number <= 10:
        return "<=10"
    elif number <= 15:
        return "<=15"
    elif number <= 20:
        return "<=20"
    else:
        return "20+"


def _create_cluster_feature(feature, cluster_anaphor, cluster_antecedent):
    first_in_ana = cluster_anaphor[-1]

    ana_fine_type = first_in_ana.attributes["fine_type"]

    last_fine_type = cluster_antecedent[0].attributes["fine_type"]

    size = len(cluster_antecedent)

    feature_count = 0

    feature_name = feature.__name__

    for mention in cluster_antecedent:
        evaluated_features = feature(first_in_ana, mention)
        if len(evaluated_features) > 1:
            raise ValueError("Error when creating cluster feature from "
                             + str(feature) + ": can only create cluster features "
                             + "from feature functions that output one feature, "
                             + str(feature) + " outputs " + str(len(evaluated_features))
                             + " features.")
        _, feature_holds = feature(first_in_ana, mention)[0]

        if feature_holds:
            feature_count += 1

    fraction = feature_count / size

    if fraction == 1.0:
        return [(feature_name + "-ALL", True),
                (feature_name + "-ana-" + ana_fine_type + "-ALL", True),
                (feature_name + "-ante-" + last_fine_type + "-ALL", True),
                (feature_name + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-ALL", True),
                ]
    elif fraction >= 0.5:
        return [(feature_name + "-MOST", True),
                (feature_name + "-ana-" + ana_fine_type + "-MOST", True),
                (feature_name + "-ante-" + last_fine_type + "-MOST", True),
                (feature_name + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-MOST", True),
                ]
    elif fraction > 0:
        return [(feature_name + "-SOME", True),
                (feature_name + "-ana-" + ana_fine_type + "-SOME", True),
                (feature_name + "-ante-" + last_fine_type + "-SOME", True),
                (feature_name + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-SOME", True),
                ]
    else:
        return [(feature_name + "-NONE", True),
                (feature_name + "-ana-" + ana_fine_type + "-NONE", True),
                (feature_name + "-ante-" + last_fine_type + "-NONE", True),
                (feature_name + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-NONE", True),
                ]


def cluster_exact_match(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level exact match feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster with which the anaphor has an exact match.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('exact_match-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if for all mentions there is an exact match, otherwise MOST if for more than
        one half of the mentions there is an exact match, otherwise SOME if for at least one
        of the mentions there is an exact match, NONE otherwise.
    """
    return _create_cluster_feature(exact_match, cluster_anaphor, cluster_antecedent)


def cluster_head_match(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level head match feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster with which the anaphor has a head match.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('head_match-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if for all mentions there is a head match, otherwise MOST if for more than
        one half of the mentions there is a head match, otherwise SOME if for at least one
        of the mentions there is an head match, NONE otherwise.
    """
    return _create_cluster_feature(head_match, cluster_anaphor, cluster_antecedent)


def cluster_alias(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level alias feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster with which the anaphor is in an alias relation.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('alias-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if for all mentions there is an alias relation, otherwise MOST if for more than
        one half of the mentions there is an alias relation, otherwise SOME if for at least one
        of the mentions there is an alias relation, NONE otherwise.
    """
    return _create_cluster_feature(alias, cluster_anaphor, cluster_antecedent)


def cluster_tokens_contained(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level tokens contained feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster which have token overlap with the anaphor.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('tokens_contained-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if for all mentions there is token overlap with the anaphor, otherwise MOST if for more than
        one half of the mentions there is token overlap, otherwise SOME if for at least one
        of the mentions there is token overlap, NONE otherwise.
    """
    return _create_cluster_feature(tokens_contained, cluster_anaphor, cluster_antecedent)


def cluster_head_contained(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level head contained feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster which have head token overlap with the anaphor.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('head_contained-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if for all mentions there is head token overlap with the anaphor, otherwise MOST if for more than
        one half of the mentions there is head token overlap, otherwise SOME if for at least one
        of the mentions there is head token overlap, NONE otherwise.
    """
    return _create_cluster_feature(head_contained, cluster_anaphor, cluster_antecedent)


def cluster_modifier(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level modifier agreement feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster which have modifier agreement with the anaphor.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('modifier-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if for all mentions there is modifier agreement with the anaphor, otherwise MOST if for more than
        one half of the mentions there is modifier agreement, otherwise SOME if for at least one
        of the mentions there is modifier agreement, NONE otherwise.
    """
    return _create_cluster_feature(modifier, cluster_anaphor, cluster_antecedent)


def cluster_embedding(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level embedding feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster which embed the anaphor.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('embedding-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if all mentions embed the anaphor, otherwise MOST if more than
        one half of the mentions embed the anaphor, otherwise SOME if at least one
        of the mentions embeds the anaphor, NONE otherwise.
    """
    return _create_cluster_feature(embedding, cluster_anaphor, cluster_antecedent)


def cluster_same_speaker(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level same speaker feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster which have the same speaker as the anaphor.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('same_speaker-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if all mentions have the same speaker as the anaphor, otherwise MOST if more than
        one half of the mentions have the same speaker as the anaphor, otherwise SOME if at least one
        mention has the same speaker as the anaphor, NONE otherwise.
    """
    return _create_cluster_feature(same_speaker, cluster_anaphor, cluster_antecedent)


def cluster_compatibility(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level compatibility feature.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the fraction of mentions in the
    antecedent cluster which are compatible with the anaphor w.r.t. gender, number, semantic class
    and named entity type.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('same_speaker-'+PRED, True) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        PRED is ALL if all mentions are compatible with the anaphor, otherwise MOST if more than
        one half of the mentions are compatible, otherwise SOME if at least one mention is compatible, NONE otherwise.
    """
    size = len(cluster_antecedent)

    first_in_ana = cluster_anaphor[-1]

    ana_fine_type = first_in_ana.attributes["fine_type"]

    last_fine_type = cluster_antecedent[0].attributes["fine_type"]

    types = ["gender", "number", "semantic_class", "ner"]

    output = []

    for t in types:
        feature_count = 0
        ana_attribute = first_in_ana.attributes[t]
        ana_head_ner = first_in_ana.attributes["ner"][first_in_ana.attributes["head_index"]]
        for mention in cluster_antecedent:
            ante_attribute = mention.attributes[t]
            if t != "ner":
                if (ana_attribute == "UNKNOWN" or ante_attribute == "UNKNOWN"
                    or ana_attribute == ante_attribute):
                    feature_count += 1
            else:
                ante_head_ner = mention.attributes["ner"][mention.attributes["head_index"]]
                if (ana_head_ner == "NONE" or ante_head_ner == "NONE"
                    or ana_head_ner == ante_head_ner):
                    feature_count += 1

        fraction = feature_count / size

        if fraction == 1.0:
            output.append((t + "-ALL", True))
            output.append((t + "-ana-" + ana_fine_type + "-ALL", True))
            output.append((t + "-ante-" + last_fine_type + "-ALL", True))
            output.append((t + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-ALL", True))
        elif fraction >= 0.5:
            output.append((t + "-MOST", True))
            output.append((t + "-ana-" + ana_fine_type + "-MOST", True))
            output.append((t + "-ante-" + last_fine_type + "-MOST", True))
            output.append((t + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-MOST", True))
        elif fraction > 0:
            output.append((t + "-SOME", True))
            output.append((t + "-ana-" + ana_fine_type + "-SOME", True))
            output.append((t + "-ante-" + last_fine_type + "-SOME", True))
            output.append((t + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-SOME", True))
        else:
            output.append((t + "-NONE", True))
            output.append((t + "-ana-" + ana_fine_type + "-NONE", True))
            output.append((t + "-ante-" + last_fine_type + "-NONE", True))
            output.append((t + "-ana-ante-" + ana_fine_type + "-" + last_fine_type + "-NONE", True))

        return output


def cluster_sentence_distance(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level sentence distance.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the sentence distance from the anaphor to
    the last mention in the antecedent cluster.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('sentence_distance', DIST) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        DIST is the capped sentence distance from the anaphor to the last mention in the antecedent cluster.
    """

    dist = __compute_sentence_distance(cluster_anaphor[-1], cluster_antecedent[0])

    ana = cluster_anaphor[-1]

    ana_fine_type = ana.attributes["fine_type"]

    last_fine_type = cluster_antecedent[0].attributes["fine_type"]

    return [("sentence_distance", dist),
            ("sentence_distance-ana-" + ana_fine_type, dist),
            ("sentence_distance-ante-" + last_fine_type, dist),
            ("sentence_distance-ana-ante-" + ana_fine_type + "-" + last_fine_type, dist)]


def cluster_token_distance(cluster_anaphor, cluster_antecedent, clustering):
    """ Compute cluster-level token distance.

    This feature should only be used when cluster_anaphor contains only one mention (i.e. the
    model is a mention-entity model). It computes the token distance from the anaphor to
    the last mention in the antecedent cluster.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.

    Returns:
        A list containing tuples of the form ('sentence_distance', DIST) which are concatenated
        with the fine-grained mention types of the anaphor and the last mention in the antecedent cluster.
        DIST is the capped token distance from the anaphor to the last mention in the antecedent cluster.
    """
    dist = __compute_token_distance(cluster_anaphor[-1], cluster_antecedent[0])

    ana = cluster_anaphor[-1]

    ana_fine_type = ana.attributes["fine_type"]

    last_fine_type = cluster_antecedent[0].attributes["fine_type"]

    return [("token_distance", dist),
            ("token_distance-ana-" + ana_fine_type, dist),
            ("token_distance-ante-" + last_fine_type, dist),
            ("token_distance-ana-ante-" + ana_fine_type + "-" + last_fine_type, dist)]


def dynamic_ante_has_ante(anaphor, antecedent, clustering):
    """ Compute whether the antecedent already has an antecedent in the current clustering.

    As this feature is a cluster-level dynamic feature which cannot be precomputed,
    also compute feature combinations with fine-grained types.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        A list containing the tuple ('exact_match', MATCH), where MATCH is True if the mentions
        match exactly (ignoring case), and False otherwise.

    Args:
        cluster_anaphor (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        cluster_antecedent (list(Mention)): A list of mentions, representing the cluster of
            the anaphor. The list must be sorted in descending order.
        clustering (Clustering): A clustering representing the current coreference clusterings for
            the document.

    Returns:
        A list containing tuples of the form ('ante_has_ante', ANTE_HAS_ANTE) which are concatenated
        with the fine-grained mention types of anaphor and antecedents.
        ANTE_HAS_ANTE is True if the antecedent already has an antecedent in the provided clustering,
        False otherwise.
    """
    ft_ana = anaphor.attributes["fine_type"]
    ft_ante = antecedent.attributes["fine_type"]

    if (antecedent in clustering.outgoing_links
        and not clustering.outgoing_links[antecedent].is_dummy()):

        return [
            ("ante_has_ante-", True),
            ("ante_has_ante-ana-" + ft_ana, True),
            ("ante_has_ante-ante-" + ft_ante, True),
            ("ante_has_ante-ana-ante-" + ft_ana + "-" + ft_ante, True),
        ]
    else:
        return [
            ("ante_has_ante-", False),
            ("ante_has_ante-ana-" + ft_ana, False),
            ("ante_has_ante-ante-" + ft_ante, False),
            ("ante_has_ante-ana-ante-" + ft_ana + "-" + ft_ante, False),
        ]