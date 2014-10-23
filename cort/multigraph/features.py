import re

from cort.core import external_data
from cort.core import spans
from cort.core import util


__author__ = 'smartschat'


def not_singleton(anaphor, antecedent):
    singleton_data = external_data.SingletonMentions.get_instance()
    anaphor = " ".join(anaphor.attributes["tokens"])
    antecedent = " ".join(antecedent.attributes["tokens"])

    if (anaphor in singleton_data.singletons and
            singleton_data.singletons[anaphor] >= 25):
        return True

    if (antecedent in singleton_data.singletons and
            singleton_data.singletons[antecedent] >= 25):
        return True


def pronoun_parallelism(anaphor, antecedent):
    return (anaphor.attributes["type"] == "PRO"
            and (anaphor.attributes["citation_form"]
                 in ["he", "she", "it", "they"])
            and (antecedent.attributes["type"] != "PRO"
                 or (antecedent.attributes["citation_form"]
                     in ["he", "she", "it", "they"]))
            and (antecedent.attributes["grammatical_function"] ==
                 anaphor.attributes["grammatical_function"])
            and (antecedent.attributes["grammatical_function"]
                 in ["SUBJECT", "OBJECT"]))


def antecedent_is_subject(anaphor, antecedent):
    return (anaphor.attributes["type"] == "PRO"
            and (anaphor.attributes["citation_form"]
                 in ["he", "she", "it", "they"])
            and (antecedent.attributes["type"] != "PRO"
                 or (antecedent.attributes["citation_form"]
                     in ["he", "she", "it", "they"]))
            and antecedent.attributes["grammatical_function"] == "SUBJECT")


def antecedent_is_object(anaphor, antecedent):
    return (anaphor.attributes["type"] == "PRO"
            and (anaphor.attributes["citation_form"]
                 in ["he", "she", "it", "they"])
            and (antecedent.attributes["type"] != "PRO"
                 or (antecedent.attributes["citation_form"]
                     in ["he", "she", "it", "they"]))
            and antecedent.attributes["grammatical_function"] == "OBJECT")


def anaphor_pronoun(anaphor, antecedent):
    return (anaphor.attributes["type"] == "PRO"
            and (anaphor.attributes["citation_form"]
                 in ["he", "she", "it", "they"])
            and (antecedent.attributes["type"] != "PRO"
                 or (antecedent.attributes["citation_form"]
                     in ["he", "she", "it", "they"])))


def lexical(anaphor, antecedent):
    lexical_data = external_data.LexicalData.get_instance()
    if ((anaphor.attributes["type"] == "NAM"
         and antecedent.attributes["type"] == "NAM")
        or (anaphor.attributes["type"] == "NOM"
            and anaphor.attributes["fine_type"] == "DEF"
            and antecedent.attributes["type"] in ["NAM", "NOM"])):
        return lexical_data.look_up(anaphor, antecedent)


def non_pronominal_string_match(anaphor, antecedent):
    if anaphor.attributes["type"] in ["PRO", "DEM", "VRB"]:
        return False
    elif antecedent.attributes["type"] in ["PRO", "DEM", "VRB"]:
        return False
    else:
        return (" ".join(util.clean_via_pos(anaphor.attributes["tokens"],
                                            anaphor.attributes["pos"])).lower()
                == " ".join(util.clean_via_pos(
                            antecedent.attributes["tokens"],
                            antecedent.attributes["pos"])).lower())


def head_match(anaphor, antecedent):
    if anaphor.attributes["type"] in ["PRO", "DEM", "VRB"]:
        return False
    elif antecedent.attributes["type"] in ["PRO", "DEM", "VRB"]:
        return False
    elif (anaphor.attributes["semantic_class"] == "NUMERIC" or
          antecedent.attributes["semantic_class"] == "NUMERIC"):
        return False
    else:
        return (anaphor.attributes["head"] != ["and"] and
                (" ".join(anaphor.attributes["head"]).lower()
                 == " ".join(antecedent.attributes["head"]).lower()))


def substring(anaphor, antecedent):
    if anaphor.attributes["type"] in ["PRO", "DEM", "VRB"]:
        return False
    elif antecedent.attributes["type"] != "NAM":
        return False
    elif (anaphor.attributes["semantic_class"] == "NUMERIC" or
          antecedent.attributes["semantic_class"] == "NUMERIC"):
        return False
    elif anaphor.attributes["head"] == ["and"]:
        return False
    else:
        cleaned = util.clean_via_pos(
            anaphor.attributes["tokens"],
            anaphor.attributes["pos"])

        return (" ".join(cleaned).lower()
                in " ".join(antecedent.attributes["tokens"]).lower())


def pronoun_same_canonical_form(anaphor, antecedent):
    return (anaphor.attributes["type"] == "PRO"
            and antecedent.attributes["type"] == "PRO"
            and (anaphor.attributes["citation_form"] ==
                 antecedent.attributes["citation_form"]))


def speaker(anaphor, antecedent):
    speaker_anaphor = anaphor.attributes["speaker"]
    speaker_antecedent = antecedent.attributes["speaker"]

    if speaker_anaphor == "-" and speaker_antecedent == "-":
        return False
    else:
        if (anaphor.attributes["type"] == "PRO"
                and antecedent.attributes["type"] == "PRO"):
            if (anaphor.attributes["citation_form"] == "i"
                    and antecedent.attributes["citation_form"] == "i"):
                return speaker_anaphor == speaker_antecedent
            elif ((anaphor.attributes["citation_form"] == "i"
                    and antecedent.attributes["citation_form"] == "you")
                  or (anaphor.attributes["citation_form"] == "you"
                      and antecedent.attributes["citation_form"] == "i")):
                return (nothing_between(anaphor, antecedent)
                        and speaker_anaphor != speaker_antecedent)
        elif (anaphor.attributes["type"] == "PRO"
              or antecedent.attributes["type"] == "PRO"):
            if (anaphor.attributes["type"] == "PRO"
                    and anaphor.attributes["citation_form"] == "i"):
                return (speaker_anaphor.replace("_", " ").lower() in
                        [" ".join(antecedent.attributes["tokens"]).lower(),
                         " ".join(antecedent.attributes["head"]).lower()])
            elif (antecedent.attributes["type"] == "PRO"
                    and antecedent.attributes["citation_form"] == "i"):
                return (speaker_antecedent.replace("_", " ").lower() in
                        [" ".join(anaphor.attributes["tokens"]).lower(),
                         " ".join(anaphor.attributes["head"]).lower()])


def nothing_between(anaphor, antecedent):
    if not anaphor.document:
        return True

    if anaphor.span < antecedent.span:
        start = anaphor.span.begin
        end = antecedent.span.end
    else:
        start = antecedent.span.begin
        end = anaphor.span.end

    speakers = anaphor.document.speakers[start:end+1]

    allowed_speakers = [speakers[0], speakers[-1]]
    for particular_speaker in speakers:
        if particular_speaker not in allowed_speakers:
            return False

    return True


def not_anaphoric(anaphor, antecedent):
    return not (anaphor.attributes["type"] in ["NAM", "PRO"]
                or (anaphor.attributes["type"] == "NOM"
                    and anaphor.attributes["fine_type"] == "DEF"))


def not_speaker(anaphor, antecedent):
    speaker_anaphor = anaphor.attributes["speaker"]
    speaker_antecedent = antecedent.attributes["speaker"]

    if speaker_anaphor == "-" or speaker_antecedent == "-":
        return False
    else:
        if (anaphor.attributes["type"] == "PRO"
                and antecedent.attributes["type"] == "PRO"):
            if ((anaphor.attributes["citation_form"] == "i"
                 and antecedent.attributes["citation_form"] == "i")
                or (anaphor.attributes["citation_form"] == "we"
                    and antecedent.attributes["citation_form"] == "we")
                or (anaphor.attributes["citation_form"] == "you"
                    and antecedent.attributes["citation_form"] == "you")):
                return speaker_anaphor != speaker_antecedent
            elif ((anaphor.attributes["citation_form"] == "i"
                   and antecedent.attributes["citation_form"] == "you")
                  or (anaphor.attributes["citation_form"] == "you"
                      and antecedent.attributes["citation_form"] == "i")):
                return speaker_anaphor == speaker_antecedent


def not_pronoun_distance(anaphor, antecedent):
    return (anaphor.attributes["type"] == "PRO"
            and anaphor.attributes["citation_form"] == "it"
            and (anaphor.attributes["sentence_id"]
                 - antecedent.attributes["sentence_id"] > 1))


def not_embedding(anaphor, antecedent):
    return (antecedent.span.embeds(anaphor.span)
            and (anaphor.attributes["fine_type"]
                 not in ["REFL", "POSS", "POSS_ADJ"]))


def not_compatible(anaphor, antecedent):
    if (" ".join(util.clean_via_pos(anaphor.attributes["tokens"],
                                    anaphor.attributes["pos"])).lower() ==
            " ".join(util.clean_via_pos(antecedent.attributes["tokens"],
                                        antecedent.attributes["pos"])).lower()):
        return False

    gender = (anaphor.attributes["gender"] == "UNKNOWN"
              or antecedent.attributes["gender"] == "UNKNOWN"
              or anaphor.attributes["gender"]
              == antecedent.attributes["gender"])

    number = (anaphor.attributes["number"] == "UNKNOWN"
              or antecedent.attributes["number"] == "UNKNOWN"
              or anaphor.attributes["number"]
              == antecedent.attributes["number"])

    semantic_class = (anaphor.attributes["semantic_class"] == "UNKNOWN"
                      or antecedent.attributes["semantic_class"] == "UNKNOWN"
                      or anaphor.attributes["semantic_class"]
                      == antecedent.attributes["semantic_class"])

    return not (gender and number and semantic_class)


def not_modifier(anaphor, antecedent):
    if (anaphor.attributes["type"] == "NAM"
            and antecedent.attributes["type"] == "NAM"):
        return False
    elif (anaphor.attributes["type"] in ["PRO", "DEM", "VRB"]
          or antecedent.attributes["type"] in ["PRO", "DEM", "VRB"]):
        return False
    else:
        return not get_modifier(anaphor).issubset(get_modifier(antecedent))


def get_modifier(mention):
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


def alias(anaphor, antecedent):
    if (anaphor.attributes["type"] != "NAM"
            or antecedent.attributes["type"] != "NAM"):
        return False
    elif (" ".join(anaphor.attributes["head"]).lower()
          == " ".join(antecedent.attributes["head"]).lower()):
        return False
    else:
        anaphor_cleaned_tokens = anaphor.attributes["head"]
        antecedent_cleaned_tokens = antecedent.attributes["head"]

        category = get_category_for_alias(
            anaphor.attributes["ner"][anaphor.attributes["head_index"]],
            antecedent.attributes["ner"][antecedent.attributes["head_index"]])

        if category == "PERSON":
            return person_alias(anaphor_cleaned_tokens,
                                antecedent_cleaned_tokens)
        elif category == "LOC":
            return loc_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens)
        elif category == "ORG":
            return org_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens)
        else:
            return False


def get_category_for_alias(anaphor_ner, antecedent_ner):
    if anaphor_ner == "PERSON" and antecedent_ner == "PERSON":
        return "PERSON"
    elif re.match(r"LOC", anaphor_ner) and re.match(r"LOC", antecedent_ner):
        return "LOC"
    elif re.match(r"ORG", anaphor_ner) and re.match(r"(ORG)", antecedent_ner):
        return "ORG"


def loc_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    return (starts_with(anaphor_cleaned_tokens, antecedent_cleaned_tokens)
            or is_abbreviation(anaphor_cleaned_tokens,
                               antecedent_cleaned_tokens))


def org_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    return (starts_with(anaphor_cleaned_tokens, antecedent_cleaned_tokens)
            or is_abbreviation(anaphor_cleaned_tokens,
                               antecedent_cleaned_tokens))


def person_alias(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    if len(anaphor_cleaned_tokens) == 1 or len(antecedent_cleaned_tokens) == 1:
        return (anaphor_cleaned_tokens[0] == antecedent_cleaned_tokens[0]
                or anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1])
    elif (len(anaphor_cleaned_tokens) == 2
            and anaphor_cleaned_tokens[0].lower() in ["mr", "ms", "mr.", "ms."]
          or len(antecedent_cleaned_tokens) == 2
            and antecedent_cleaned_tokens[0].lower() in ["mr", "ms", "mr.",
                                                         "ms."]):
        return anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1]
    elif (anaphor_cleaned_tokens[0] == antecedent_cleaned_tokens[0]
          and anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1]):
        return True
    elif len(anaphor_cleaned_tokens) > 1 and len(antecedent_cleaned_tokens) > 1:
        return (anaphor_cleaned_tokens[-1] == antecedent_cleaned_tokens[-1]
                and anaphor_cleaned_tokens[-2] == antecedent_cleaned_tokens[-2])

    return False


def starts_with(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    for ana_token, ante_token in zip(anaphor_cleaned_tokens,
                                     antecedent_cleaned_tokens):
        if ana_token != ante_token:
            return False

    return True


def is_abbreviation(anaphor_cleaned_tokens, antecedent_cleaned_tokens):
    if (" ".join(anaphor_cleaned_tokens).replace(".", "")
            == " ".join(antecedent_cleaned_tokens).replace(".", "")):
        return True
    else:
        if len(anaphor_cleaned_tokens) > len(antecedent_cleaned_tokens):
            return (" ".join(antecedent_cleaned_tokens)
                    in set(get_acronyms(anaphor_cleaned_tokens)))
        else:
            return (" ".join(anaphor_cleaned_tokens)
                    in set(get_acronyms(antecedent_cleaned_tokens)))


def get_acronyms(cleaned_tokens):
    company_designator = r'assoc|bros|co|coop|corp|devel|inc|llc|ltd\.?'
    tokens_without_designator = [token for token in cleaned_tokens
                                 if not re.match(company_designator,
                                                 token.lower())]

    return (" ".join(tokens_without_designator),
            "".join([token[0] for token in tokens_without_designator
                    if token[0].isupper()]),
            ".".join([token[0] for token in tokens_without_designator
                      if token[0].isupper()])+".")
