""" Contains features for coreference resolution.

Features takes as input either a mention or a pair of mentions, and always
output a tuple ``(feature_name, value)``. ``feature_name`` is the name of the
feature (a string), and ``value`` is the value of the feature. ``value`` can be
either a string, a bool, or a scalar.
"""

from __future__ import division


import re


from cort.core import spans


__author__ = 'smartschat'


def fine_type(mention):
    """ Compute fine-grained type of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('fine_type', TYPE), where where TYPE is one of 'NAM', 'DEF',
        'INDEF', 'DEM', 'VRB', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        and 'NONE'.
    """
    if mention.attributes["type"] == "NOM":
        mention_fine_type = mention.attributes["fine_type"]
    elif mention.attributes["type"] == "PRO":
        mention_fine_type = mention.attributes["citation_form"]
    else:
        mention_fine_type = mention.attributes["type"]

    if not mention_fine_type:
        mention_fine_type = "NONE"

    return "fine_type", mention_fine_type


def gender(mention):
    """ Compute gender of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('gender', GENDER), where GENDER is one of 'MALE',
        'FEMALE', 'NEUTRAL', 'PLURAL' and 'UNKNOWN'.
    """
    return "gender", mention.attributes["gender"]


def number(mention):
    """ Compute number of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('number', NUMBER), where NUMBER is one of 'SINGULAR',
        'PLURAL' and 'UNKNOWN'.
    """
    return "number", mention.attributes["number"]


def sem_class(mention):
    """ Compute semantic class of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('sem_class', SEM_CLASS), where SEM_CLASS is one of
        'PERSON', 'OBJECT', 'NUMERIC' and 'UNKNOWN'.
    """
    return "sem_class", mention.attributes["semantic_class"]


def gr_func(mention):
    """ Compute grammatical function of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('gr_func', GR_FUNC), where GR_FUNC is one of 'SUBJECT',
        'OBJECT' and 'OTHER'.
    """
    return "gr_func", mention.attributes["grammatical_function"]


def governor(mention):
    """ Compute governor of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('governor', GOVERNOR), where GOVERNOR is the governor
        of the mention.
    """
    return "governor", mention.attributes["governor"].lower()


def deprel(mention):
    """ Compute dependency relation of a mention to its governor.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('deprel', DEPREL), where DEPREL is the dependency relation
        of the mention to its governor.
    """
    return "deprel", mention.attributes["deprel"]


def head(mention):
    """ Compute head of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('head', HEAD), where HEAD is the (lowercased) head of
        the mention.
    """
    return "head", mention.attributes["head_as_lowercase_string"]


def head_ner(mention):
    """ Compute named entity tag of a mention's head.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('ner', NER), where NER is the named entity tag of the
        mention's head word. If the mention is not a named entity, NER is
        set to 'NONE'.
    """
    return "ner", mention.attributes["ner"][mention.attributes["head_index"]]


def length(mention):
    """ Compute length of a mention in tokens.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('length' LENGTH), where LENGTH is the length of the
        mention in tokens. The length is stored as a string.
    """
    return "length", str(len(mention.attributes["tokens"]))


def first(mention):
    """ Compute the first token of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('first', TOKEN), where TOKEN is the (lowercased) first
        token of the mention.
    """
    return "first", mention.attributes["tokens"][0].lower()


def last(mention):
    """ Compute the last token of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('last', TOKEN), where TOKEN is the (lowercased) last token
        of the mention.
    """
    return "last", mention.attributes["tokens"][-1].lower()


def preceding_token(mention):
    """ Compute the token preceding a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('preceding', TOKEN), where TOKEN is the (lowercased) token
        preceding the mention. If no such token exists, set TOKEN to 'NONE'.
    """
    prec = mention.get_context(-1)

    if prec:
        return "preceding", prec[0].lower()
    else:
        return "preceding", "NONE"


def next_token(mention):
    """ Compute the token following a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        The tuple ('next', TOKEN), where TOKEN is the (lowercased) token
        following the mention. If no such token exists, set TOKEN to 'NONE'.
    """
    next_t = mention.get_context(1)

    if next_t:
        return "next", next_t[0].lower()
    else:
        return "next", "NONE"

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
        The tuple ('ancestry', ANCESTRY), where ANCESTRY is the ancestry of
        the mention.
    """
    return "ancestry", mention.attributes["ancestry"]

def genre(anaphor, antecedent):
        return "genre", anaphor.document.identifier[1:anaphor.document.identifier.index("/")]

def exact_match(anaphor, antecedent):
    """ Compute whether the tokens of two mentions match exactly.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('exact_match', MATCH), where MATCH is True if the mentions
        match exactly (ignoring case), and False otherwise.
    """

    match = anaphor.attributes["tokens_as_lowercase_string"] == \
            antecedent.attributes["tokens_as_lowercase_string"]

    return "exact_match", match


def head_match(anaphor, antecedent):
    """ Compute whether the heads of two mentions match.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('head_match', MATCH), where MATCH is True if the heads of
        the mentions match exactly (ignoring case), and False otherwise.
    """
    match = anaphor.attributes["head_as_lowercase_string"] == \
            antecedent.attributes["head_as_lowercase_string"]

    return "head_match", match


def tokens_contained(anaphor, antecedent):
    """ Compute whether one mention is a substring of the other mention.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('tokens_contained', CONTAINED), where CONTAINED is True if
        the anaphor contains the antecedent, or vice versa (ignoring case).
    """
    ana_tokens = anaphor.attributes["tokens_as_lowercase_string"]
    ante_tokens = antecedent.attributes["tokens_as_lowercase_string"]

    contained = ana_tokens in ante_tokens or ante_tokens in ana_tokens

    return "tokens_contained", contained


def head_contained(anaphor, antecedent):
    """ Compute whether one mention's head is a substring of the other
    mention's head.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('head_contained', CONTAINED), where CONTAINED is True if
        the anaphor's head contains the antecedent's head, or vice versa
        (ignoring case).
    """
    ana_head = anaphor.attributes["head_as_lowercase_string"]
    ante_head = antecedent.attributes["head_as_lowercase_string"]

    contained = ana_head in ante_head or ante_head in ana_head

    return "head_contained", contained


def sentence_distance(anaphor, antecedent):
    """ Compute the sentence distance between two mentions (capped at 5).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('sentence_distance', DIST), where DIST is one of '0', '1',
        '2', '3', '4' and '>=5'.
    """
    return "sentence_distance", __compute_sentence_distance(anaphor,
                                                            antecedent)


def token_distance(anaphor, antecedent):
    """ Compute the token distance between two mentions (capped at 10).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('token_distance'=DIST), where DIST is one of '0', '1',
        '2', '3', '4' and '>=10'.
    """
    return "token_distance", __compute_token_distance(anaphor, antecedent)

def alias(anaphor, antecedent):
    """ Compute whether the mentions are aliases of each other.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('alias', ALIAS), where ALIAS is True if anaphor and
        antecedent are in an alias relation, False otherwise.
    """
    return "alias", __are_alias(anaphor, antecedent)


def same_speaker(anaphor, antecedent):
    """ Compute whether the speakers  of two mentions are the same.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('same_speaker', SAME), where SAME is True if anaphor and
        antecedent have the same speaker, False otherwise.
    """
    same = anaphor.attributes["speaker"] == antecedent.attributes["speaker"]

    return "same_speaker", same


def embedding(anaphor, antecedent):
    """ Compute whether one mention embeds the other.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('embedding', EMB), where EMB is True if one of the mentions
        embeds the other, False otherwise.
    """
    emb = anaphor.span.embeds(antecedent.span) or \
          antecedent.span.embeds(anaphor.span)

    return "embedding", emb


def modifier(anaphor, antecedent):
    """ Compute modifier agreement.

    In particular, compute whether the anaphor has modifiers that do not appear
    in the antecedent (ignoring demonstratives and prepositions).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('modifier', MOD), where MOD is True if the anaphor has
        modifiers that do not appear in the antecedent, False otherwise.
    """
    mod = not __get_modifier(anaphor).issubset(__get_modifier(antecedent))

    return "modifier", mod


def relative_overlap(anaphor, antecedent):
    """ Compute relative overlap of the mentions (ignoring case).

    For example, "the new president" and "the president" have relative overlap
    2/3.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        The tuple ('relative_overlap', OVERLAP), where OVERLAP (a float) is
        the relative overlap of anaphor and antecedent.
    """
    ana_tokens = set([tok.lower() for tok in anaphor.attributes["tokens"]])
    ante_tokens = set([tok.lower() for tok
                             in antecedent.attributes["tokens"]])

    overlap = len(ana_tokens & ante_tokens)/max(len(ana_tokens),
                                                len(ante_tokens))

    return "relative_overlap", overlap


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

def has_exact_match(mention, all_mentions):
    for m in all_mentions:
        if (not m.is_dummy() and m != mention
            and mention.attributes["tokens_as_lowercase_string"] == m.attributes["tokens_as_lowercase_string"]):
            return "has_exact_match", True
    return "has_exact_match", False

def has_head_match(mention, all_mentions):
    for m in all_mentions:
        if (not m.is_dummy() and m != mention and
        mention.attributes["head_as_lowercase_string"] ==
            m.attributes["head_as_lowercase_string"]):
            return "has_head_match", True
    return "has_head_match", False

def pre_pre_token(mention):
    prec = mention.get_context(-2)

    if prec:
        return "prePreToken", prec[0].lower()
    else:
        return "PrePreToken", "NONE"
        
def preceding_token_pos(mention):
    prec = mention.get_pos_context(-1)

    if prec:
        return "prePOS", prec[0]
    else:
        return "prePOS","NONE"


def pre_pre_token_pos(mention):
    prec = mention.get_pos_context(-2)

    if prec:
        return "prePrePOS", prec[0]
    else:
        return "prePrePOS","NONE"
        
def next_next_token(mention):
    next_t = mention.get_context(2)

    if next_t:
        return "nextNextToken",next_t[1].lower()
    else:
        return "nextNextToken","NONE"
        
def next_token_pos(mention):
    next_t = mention.get_pos_context(1)

    if next_t:
        return "nextPOS",next_t[0].lower()
    else:
        return "nextPOS","NONE"

def next_next_token_pos(mention):
    next_t = mention.get_pos_context(2)

    if next_t:
        return "nextNextPOS",next_t[1].lower()
    else:
        return "nextNextPOS","NONE"
        
def singleton_score (mention):
    return "singletonScore",  int(mention.attributes["singletonScore"]/10)

