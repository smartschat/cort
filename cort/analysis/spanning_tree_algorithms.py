""" Algorithms for computing spanning trees of entity graphs. """


__author__ = 'smartschat'


def precision_system_output(entity, partitioned_entity):
    """ Compute a spanning tree from antecedent information.

    All edges in the spanning tree correspond to anaphor-antecedent pairs. In
    order to access this antecedent information, the attribute "antecedent" of
    the mentions in the entity must be set.

    Args:
        entity (EntityGraph): The EntityGraph for the entity for which the
            spanning tree should be computed.
        partitioned_entity (EntityGraph): A partition of the entity -- not
            used for this algorithm.

    Returns:
        list(Mention, Mention): A list of mention pairs, which constitute the
        edges of the spanning tree. For a pair (m, n), n appears later in
        the text than m.
    """
    edges = []
    for mention in entity.edges:
        # just look at system output
        if ("antecedent" in mention.attributes
                and mention.attributes["antecedent"] in entity.edges[mention]):
            edges.append((mention, mention.attributes["antecedent"]))

    return sorted(edges)


def recall_closest(entity, partitioned_entity):
    """ Compute a spanning tree by always taking the closest mention in the same
    entity.

    Args:
        entity (EntityGraph): The EntityGraph for the entity for which the
            spanning tree should be computed.
        partitioned_entity (EntityGraph): A partition of the entity -- not
            used for this algorithm.

    Returns:
        list(Mention, Mention): A list of mention pairs, which constitute the
        edges of the spanning tree. For a pair (m, n), n appears later in
        the text than m.
    """
    edges = []
    for mention in entity.edges:
        # always take closest (except for first mention in entity, which does
        # not have any antecedent)
        if entity.edges[mention]:
            if mention in partitioned_entity.edges:
                antecedent = sorted(partitioned_entity.edges[mention],
                                    reverse=True)[0]
            else:
                antecedent = sorted(entity.edges[mention], reverse=True)[0]
            edges.append((mention, antecedent))

    return sorted(edges)


def recall_accessibility(entity, partitioned_entity):
    """ Compute a spanning tree by choosing edges according to the accessibility
    of the antecedent.

    First, if a mention has an out-degree of at least one in the partitioned
    entity, take the edge with the closest mention distance as an edge for
    the spanning tree. Otherwise, proceed as follows.

    If a mention m is a proper name or a common noun, choose an antecedent as
    follows:

        - if a proper name antecedent exists, take the closest and output this
          pair as an edge
        - else if a common noun antecedent exists, take the closest and output
          this pair as an edge
        - else take the closest preceding mention and output this pair as an
          edge

    For all other mentions, take the closest preceding mention and output
    this pair as an edge.

    Args:
        entity (EntityGraph): The EntityGraph for the entity for which the
            spanning tree should be computed.
        partitioned_entity (EntityGraph): A partition of the entity -- not
            used for this algorithm.

    Returns:
        list(Mention, Mention): A list of mention pairs, which constitute the
        edges of the spanning tree. For a pair (m, n), n appears later in
        the text than m.
    """
    edges = []
    for mention in entity.edges:
        if entity.edges[mention]:
            # mention is not the first in subentity? take closest!
            if mention in partitioned_entity.edges:
                antecedent = sorted(partitioned_entity.edges[mention],
                                    reverse=True)[0]
            else:
                antecedent = __get_antecedent_by_type(mention,
                                                      entity.edges[mention])

            edges.append((mention, antecedent))

    return sorted(edges)


def __get_antecedent_by_type(mention, candidates):
    # make sure...
    candidates_reversed = sorted(candidates, reverse=True)
    # mention is (demonstrative) pronoun? take closest!
    if (mention.attributes["type"] == "PRO" or
            mention.attributes["type"] == "DEM"):
        return candidates_reversed[0]
    # otherwise chose by type, back off to closest
    elif __get_by_pos(candidates_reversed, "NAM"):
        return __get_by_pos(candidates_reversed, "NAM")
    elif __get_by_pos(candidates_reversed, "NOM"):
        return __get_by_pos(candidates_reversed, "NOM")
    else:
        return candidates_reversed[0]


def __get_by_pos(candidates, pos):
    for mention in candidates:
        if mention.attributes["type"] == pos:
            return mention
