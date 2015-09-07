""" Contains features for coreference resolution."""


import re


from cort.core import spans


__author__ = 'smartschat'


def fine_type(mention):
    """ Compute fine-grained type of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'fine_type=TYPE', where TYPE is one of 'NAM', 'DEF',
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

    return "fine_type=" + mention_fine_type


def gender(mention):
    """ Compute gender of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'gender=GENDER', where GENDER is one of 'MALE',
        'FEMALE', 'NEUTRAL', 'PLURAL' and 'UNKNOWN'.
    """
    return "gender=" + mention.attributes["gender"]


def number(mention):
    """ Compute number of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'number=NUMBER', where NUMBER is one of 'SINGULAR',
        'PLURAL' and 'UNKNOWN'.
    """
    return "number=" + mention.attributes["number"]


def sem_class(mention):
    """ Compute semantic class of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'sem_class=SEM_CLASS', where SEM_CLASS is one of
        'PERSON', 'OBJECT', 'NUMERIC' and 'UNKNOWN'.
    """
    return "sem_class=" + mention.attributes["semantic_class"]


def gr_func(mention):
    """ Compute grammatical function of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'gr_func=GR_FUNC', where GR_FUNC is one of 'SUBJECT',
        'OBJECT' and 'OTHER'.
    """
    return "gr_func=" + mention.attributes["grammatical_function"]


def governor(mention):
    """ Compute governor of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'governor=GOVERNOR', where GOVERNOR is the governor
        of the mention.
    """
    return "governor=" + mention.attributes["governor"].lower()


def deprel(mention):
    return "deprel=" + mention.attributes["deprel"]


def head(mention):
    """ Compute head of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'head=HEAD', where HEAD is the (lowercased) head of
        the mention.
    """
    return "head=" + mention.attributes["head_as_lowercase_string"]


def head_ner(mention):
    """ Compute named entity tag of a mention's head.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'ner=NER', where NER is the named entity tag of the
        mention's head word. If the mention is not a named entity, NER is
        set to 'NONE'.
    """
    return "ner=" + mention.attributes["ner"][mention.attributes["head_index"]]


def length(mention):
    """ Compute length of a mention in tokens.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'length=LENGTH', where LENGTH is the length of the
        mention in tokens.
    """
    return "length=" + str(len(mention.attributes["tokens"]))


def first(mention):
    """ Compute the first token of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'first=TOKEN', where TOKEN is the first token of
        the mention.
    """
    return "first=" + mention.attributes["tokens"][0].lower()


def last(mention):
    """ Compute the last token of a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'last=TOKEN', where TOKEN is the last token of
        the mention..
    """
    return "last=" + mention.attributes["tokens"][-1].lower()


def preceding_token(mention):
    """ Compute the token preceding a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'preceding=TOKEN', where TOKEN is the token
        preceding the mention. If no such token exists, set TOKEN to 'NONE'.
    """
    prec = mention.get_context(-1)

    if prec:
        return "preceding=" + prec[0].lower()
    else:
        return "preceding=NONE"


def next_token(mention):
    """ Compute the token following a mention.

    Args:
        mention (Mention): A mention.

    Returns:
        str: The string 'next=TOKEN', where TOKEN is the token
        following the mention. If no such token exists, set TOKEN to 'NONE'.
    """
    next_t = mention.get_context(1)

    if next_t:
        return "next=" + next_t[0].lower()
    else:
        return "next=NONE"


def ancestry(mention):
    return "ancestry=" + mention.attributes["ancestry"]


def exact_match(anaphor, antecedent):
    """ Compute whether the tokens of two mentions match exactly.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'exact_match' if the tokens of anaphor and antecedent
        match exactly (ignoring case), None otherwise.
    """
    if (anaphor.attributes["tokens_as_lowercase_string"] ==
            antecedent.attributes["tokens_as_lowercase_string"]):
        return "exact_match"


def head_match(anaphor, antecedent):
    """ Compute whether the heads of two mentions match.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'head_match' if the heads of anaphor and antecedent
        match (ignoring case), None otherwise.
    """
    if (anaphor.attributes["head_as_lowercase_string"] ==
            antecedent.attributes["head_as_lowercase_string"]):
        return "head_match"


def tokens_contained(anaphor, antecedent):
    ana_tokens = anaphor.attributes["tokens_as_lowercase_string"]
    ante_tokens = antecedent.attributes["tokens_as_lowercase_string"]

    if ana_tokens in ante_tokens or ante_tokens in ana_tokens:
        return "tokens_contained"


def head_contained(anaphor, antecedent):
    ana_head = anaphor.attributes["head_as_lowercase_string"]
    ante_head = antecedent.attributes["head_as_lowercase_string"]

    if ana_head in ante_head or ante_head in ana_head:
        return "head_contained"


def sentence_distance(anaphor, antecedent):
    """ Compute the sentence distance between two mentions (capped at 5).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'sentence_distance=DIST', where DIST is one of '0', '1',
        '2', '3', '4' and '>=5'.
    """
    return "sentence_distance=" + __compute_sentence_distance(anaphor, antecedent)


def token_distance(anaphor, antecedent):
    """ Compute the token distance between two mentions (capped at 10).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'sentence_distance=DIST', where DIST is one of '0', '1',
        '2', '3', '4' and '>=5'.
    """
    return "token_distance=" + __compute_token_distance(anaphor, antecedent)

def alias(anaphor, antecedent):
    """ Compute whether the mentions are aliases of each other.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'alias' if anaphor and antecedent are in an alias
            relation. None otherwise.
    """
    if __are_alias(anaphor, antecedent):
        return "alias"


def same_speaker(anaphor, antecedent):
    """
    Compute whether the speakers  of two mentions are the same..

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'same speaker' if the mentions have the same speaker,
        None otherwise.
    """
    if anaphor.attributes["speaker"] == \
            antecedent.attributes["speaker"]:
        return "same_speaker"


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


def embedding(anaphor, antecedent):
    """ Compute whether one mention embeds the other.

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'embedding' if one of the mentions embeds the other,
        None otherwise.
    """
    if (anaphor.span.embeds(antecedent.span) or
            antecedent.span.embeds(anaphor.span)):
        return "embedding"


def modifier(anaphor, antecedent):
    """ Compute modifier agreement.

    In particular, compute whether the anaphor has modifiers that do not appear
    in the antecedent (ignoring demonstratives and prepositions).

    Args:
        anaphor (Mention): A mention.
        antecedent (Mention): Another mention, preceding the anaphor.

    Returns:
        str: 'modifier' if the anaphor has modifiers that do not
        appear in the antecedent. None otherwise.
    """
    if not __get_modifier(anaphor).issubset(__get_modifier(antecedent)):
        return "modifier"


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
