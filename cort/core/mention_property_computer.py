""" Compute attributes of mentions. """

import re

from nltk.corpus import wordnet as wn

from cort.core import external_data
from cort.core import spans


__author__ = 'smartschat'


class MentionPropertyComputer:
    def __init__(self, head_finder):
        self.head_finder = head_finder

    def compute_properties(self, mention, mention_id):
        pass

    def needs_parse_trees(self):
        return False

    def needs_dependency_trees(self):
        return False


class EnglishMentionPropertyComputer(MentionPropertyComputer):
    def compute_properties(self, mention, mention_id):
        attributes = mention.attributes

        attributes["parse_tree"] = self.get_relevant_subtree(
            mention.span, mention.document)

        attributes["is_apposition"] = \
            self._is_apposition(attributes)

        attributes["grammatical_function"] = \
            self._get_grammatical_function(attributes)

        (head, in_mention_span, head_index) = \
            self._compute_head_information(attributes)

        attributes["head"] = head
        attributes["head_span"] = spans.Span(
            mention.span.begin + in_mention_span.begin,
            mention.span.begin + in_mention_span.end
        )
        attributes["head_index"] = head_index

        attributes["head_as_lowercase_string"] = " ".join(attributes[
            "head"]).lower()

        attributes["type"] = self._get_type(attributes)

        if attributes["type"] == "PRO":
            attributes["citation_form"] = self._get_citation_form(attributes)

        attributes["fine_type"] = self._get_fine_type(
            attributes)

        attributes["number"] = self._compute_number(attributes)
        attributes["gender"] = self._compute_gender(attributes)

        attributes["semantic_class"] = self._compute_semantic_class(attributes)

        i, sentence_span = mention.document.get_sentence_id_and_span(mention.span)

        dep_tree = mention.document.dep[i]

        index = mention.span.begin + head_index - sentence_span.begin

        governor_id = dep_tree[index].head - 1

        if governor_id == -1:
            attributes["governor"] = "NONE"
        else:
            attributes["governor"] = dep_tree[governor_id].form.lower()

        attributes["ancestry"] = self._get_ancestry(dep_tree, index)

        attributes["deprel"] = dep_tree[index].deprel

        attributes["id"] = mention_id

    def _compute_number(self, attributes):
        """ Compute the number of a mention.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                values for "type", "head_index" and "pos".

        Returns:
            str: the number of the mention -- one of UNKNOWN, SINGULAR and PLURAL.
        """
        number = "UNKNOWN"
        head_index = attributes["head_index"]
        pos = attributes["pos"][head_index]

        if attributes["type"] == "PRO":
            if attributes["citation_form"] in ["i", "you", "he", "she", "it"]:
                number = "SINGULAR"
            else:
                number = "PLURAL"
        elif attributes["type"] == "DEM":
            if attributes["head"][0].lower() in ["this", "that"]:
                number = "SINGULAR"
            else:
                number = "PLURAL"
        elif attributes["type"] in ["NOM", "NAM"]:
            if pos == "NNS" or pos == "NNPS":
                number = "PLURAL"
            else:
                number = "SINGULAR"

        if pos == "CC":
            number = "PLURAL"

        return number

    def _compute_gender(self, attributes):
        """ Compute the gender of a mention.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                values for "type", "head", "head_index" and, if the mention is a
                pronoun, "citation_form".

        Returns:
            str: the number of the mention -- one of UNKNOWN, MALE, FEMALE,
                NEUTRAL and PLURAL.
        """
        gender = "NEUTRAL"
        head_index = attributes["head_index"]
        gender_data = external_data.GenderData.get_instance()

        if self._compute_number(attributes) == "PLURAL":
            gender = "PLURAL"
        elif attributes["type"] == "PRO":
            if attributes["citation_form"] == "he":
                gender = "MALE"
            elif attributes["citation_form"] == "she":
                gender = "FEMALE"
            elif attributes["citation_form"] == "it":
                gender = "NEUTRAL"
            elif attributes["citation_form"] in ["you", "we", "they"]:
                gender = "PLURAL"
        elif attributes["type"] == "NAM":
            if re.match(r"^mr(\.)?$", attributes["tokens"][0].lower()):
                gender = "MALE"
            elif re.match(r"^(miss|ms|mrs)(\.)?$",
                          attributes["tokens"][0].lower()):
                gender = "FEMALE"
            elif not re.match(r"(PERSON|NONE)", attributes["ner"][head_index]):
                gender = "NEUTRAL"
            elif gender_data.look_up(attributes):
                gender = gender_data.look_up(attributes)
        elif attributes["type"] == "NOM":
            if self.__wordnet_lookup_gender(
                    " ".join(attributes["head"])):
                gender = self.__wordnet_lookup_gender(
                    " ".join(attributes["head"]))
            elif gender_data.look_up(attributes):
                gender = gender_data.look_up(attributes)

        if (gender == "NEUTRAL" and
            self._compute_semantic_class(attributes) == "PERSON"):
            gender = "UNKNOWN"

        return gender

    def _compute_semantic_class(self, attributes):
        """ Compute the semantic class of a mention.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                values for "type", "head", "head_index" and, if the mention is a
                pronoun, "citation_form".

        Returns:
            str: the semantic class of the mention -- one of PERSON, OBJECT,
            NUMERIC and UNKNOWN.
        """
        semantic_class = "UNKNOWN"
        head_index = attributes["head_index"]

        if attributes["type"] == "PRO":
            if attributes["citation_form"] in ["i", "you", "he", "she", "we"]:
                semantic_class = "PERSON"
            elif attributes["citation_form"] == "they":
                semantic_class = "UNKNOWN"
            elif attributes["citation_form"] == "it":
                semantic_class = "OBJECT"
        elif attributes["type"] == "DEM":
            semantic_class = "OBJECT"
        elif attributes["ner"][head_index] != "NONE":
            ner_tag = attributes["ner"][head_index]
            if ner_tag == "PERSON":
                semantic_class = "PERSON"
            elif re.match("DATE|TIME|NUMBER|QUANTITY|MONEY|PERCENT", ner_tag):
                semantic_class = "NUMERIC"
            else:
                semantic_class = "OBJECT"
        # wordnet lookup
        elif (attributes["type"] == "NOM" and
                self.__wordnet_lookup_semantic_class(
                        " ".join(attributes["head"]))):
            semantic_class = self.__wordnet_lookup_semantic_class(
                " ".join(attributes["head"]))

        return semantic_class

    def __wordnet_lookup_semantic_class(self, head):
        synsets = wn.synsets(head)

        while synsets:
            lemma_name = synsets[0].lemma_names()[0]

            if lemma_name == "person":
                return "PERSON"
            elif lemma_name == "object":
                return "OBJECT"

            synsets = synsets[0].hypernyms()

    def __wordnet_lookup_gender(self, head):
        synsets = wn.synsets(head)

        while synsets:
            lemma_name = synsets[0].lemma_names()[0]

            if lemma_name == "man" or lemma_name == "male":
                return "MALE"
            elif lemma_name == "woman" or lemma_name == "female":
                return "FEMALE"
            elif lemma_name == "person":
                return
            elif lemma_name == "entity":
                return "NEUTRAL"

            synsets = synsets[0].hypernyms()

    def _is_apposition(self, attributes):
        """ Compute whether the mention is an apposition, as in "Secretary of
        State Madeleine Albright" or "Barack Obama, the US president".

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                a value for "parse_tree".

        Returns:
            bool: Whether the mention is an apposition.
        """
        tree = attributes["parse_tree"]

        if tree.label() == "NP" and len(tree) > 1:
            if len(tree) == 2:
                return (tree[0].label() == "NP" and
                        tree[1].label() == "NP" and
                        self.__head_pos_starts_with(tree[1], "NNP"))
            elif len(tree) == 3:
                return (tree[0].label() == "NP" and
                        tree[1].label() == "," and
                        tree[2].label() == "NP" and
                        self.__any_child_head_starts_with(tree, "NNP") and
                        "DT" in set([child.pos()[0][1] for child in tree]))
            elif len(tree) == 4:
                return (tree[0].label() == "NP" and
                        tree[1].label() == "," and
                        tree[2].label() == "NP" and
                        tree[3].label() == "," and
                        self.__any_child_head_starts_with(tree, "NNP") and
                        "DT" in set([child.pos()[0][1] for child in tree]))

    def __any_child_head_starts_with(self, tree, pos_tag):
        for child in tree:
            if self.__head_pos_starts_with(child, pos_tag):
                return True

        return False

    def __head_pos_starts_with(self, tree, pos_tag):
        return self.head_finder.get_head(tree).pos()[0][1].startswith(pos_tag)

    def _compute_head_information(self, attributes):
        """ Compute the head of the mention.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                values for "tokens", "parse_tree", "pos", "ner", "is_apposition"

        Returns:
            (list(str), Span, int): The head, the head span (in the document) and
            the starting index of the head (in the mention).
        """
        mention_subtree = attributes["parse_tree"]

        head_index = 0
        head = [attributes["tokens"][0]]

        if len(mention_subtree.leaves()) == len(attributes["tokens"]):
            head_tree = self.head_finder.get_head(mention_subtree)
            head_index = self.get_head_index(head_tree, mention_subtree.pos())
            head = [head_tree[0]]

        in_mention_span = spans.Span(head_index, head_index)

        if attributes["pos"][head_index].startswith("NNP"):
            in_mention_span, head = \
                self.head_finder.adjust_head_for_nam(
                    attributes["tokens"],
                    attributes["pos"],
                    attributes["ner"][head_index])

        # proper name mention: head index last word of head
        # (e.g. "Obama" in "Barack Obama")
        head_index = in_mention_span.end

        # special handling for appositions
        if attributes["is_apposition"]:
            # "Secretary of State Madeleine Albright"
            # => take "Madeleine Albright" as head
            if len(mention_subtree) == 2:
                head_tree = mention_subtree[1]
                head = head_tree.leaves()
                in_mention_span = spans.Span(
                    len(mention_subtree[0].leaves()),
                    len(attributes["tokens"]) - 1)
                head_index = in_mention_span.end
            else:
                start = 0
                for child in mention_subtree:
                    if self.__head_pos_starts_with(child, "NNP"):
                        end = min(
                            [start + len(child.leaves()),
                             len(attributes["tokens"])])
                        head_index = end - 1
                        in_mention_span, head = \
                            self.head_finder.adjust_head_for_nam(
                                attributes["tokens"][start:end],
                                attributes["pos"][start:end],
                                attributes["ner"][head_index])
                        break
                    start += len(child.leaves())

        return head, in_mention_span, head_index

    def get_relevant_subtree(self, span, document):
        """ Get the fragment of the parse tree and the input span.

        Args:
            span (Span): A span in a document.
            document (CoNLLDocument): A document.

        Returns:
            nltk.ParentedTree: The fragment of the parse tree at the span in the
            document.
        """
        in_sentence_ids = document.in_sentence_ids[span.begin:span.end+1]
        in_sentence_span = spans.Span(in_sentence_ids[0], in_sentence_ids[-1])

        sentence_id, sentence_span = document.get_sentence_id_and_span(span)

        sentence_tree = document.parse_trees[sentence_id]

        spanning_leaves = sentence_tree.treeposition_spanning_leaves(
            in_sentence_span.begin, in_sentence_span.end+1)
        mention_subtree = sentence_tree[spanning_leaves]

        if mention_subtree in sentence_tree.leaves():
            mention_subtree = sentence_tree[spanning_leaves[:-2]]

        return mention_subtree

    def _get_grammatical_function(self, attributes):
        """ Compute the grammatical function of a mention in its sentence.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                a value for "parse_tree".

        Returns:
            str: The grammatical function of the mention in its sentence, one of
            SUBJECT, OBJECT and OTHER.
        """
        tree = attributes["parse_tree"]
        parent = tree.parent()

        if parent is None:
            return "OTHER"
        else:
            parent_label = parent.label()

            if re.match(r"^(S|FRAG)", parent_label):
                return "SUBJECT"
            elif re.match(r"VP", parent_label):
                return "OBJECT"
            else:
                return "OTHER"

    def get_head_index(self, head_with_pos, all_leaves):
        head_index = -1

        for i in range(0, len(all_leaves)):
            if head_with_pos[0] == all_leaves[i][0]:
                head_index = i

        return head_index

    def _get_type(self, attributes):
        """ Compute mention type.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                values for "pos", "ner" and "head_index".

        Returns:
            str: The mention type, one of NAM (proper name), NOM (common noun),
            PRO (personal pronoun),
            DEM (demonstrative pronoun), VRB (verb) and MISC (remaining).
        """
        pos = attributes["pos"][attributes["head_index"]]
        head_ner = attributes["ner"][attributes["head_index"]]

        if pos.startswith("NNP"):
            return "NAM"
        elif head_ner != "NONE":
            return "NAM"
        elif pos.startswith("PRP"):
            return "PRO"
        elif pos.startswith("DT"):
            return "DEM"
        elif pos.startswith("VB"):
            return "VRB"
        elif pos.startswith("NN"):
            return "NOM"
        else:
            return "MISC"

    def _get_fine_type(self, attributes):
        """ Compute fine-grained mention type.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                values for "type", "tokens", "pos" and "citation_form".

        Returns:
            str: The fine-grained mention type, one of

                - DEF (definite noun phrase),
                - NONDEF (non-definite noun phrase),
                - i, you, we, he, she, it, they (the citation form of the
                  pronoun),
                - DEM (demonstrative pronoun),
                - NAM (proper name),
                - VRB (verb),
                - MISC (remaining
        """
        coarse_type = attributes["type"]
        start_token = attributes["tokens"][0]
        head_index = attributes["head_index"]

        if coarse_type == "NOM":
            if re.match("^(the|this|that|these|those|my|your|his|her|its|our|" +
                        "their)$", start_token):
                return "DEF"
            elif "POS" in attributes["pos"][:head_index]:
                return "DEF"
            else:
                return "NOTDEF"
        elif coarse_type == "PRO" and attributes["citation_form"]:
            return attributes["citation_form"]
        else:
            return coarse_type

    def _get_citation_form(self, attributes):
        """ Compute the citation form of a pronominal mention.

        Args:
            attributes (dict(str, object)): Attributes of the mention, must contain
                the key "tokens".

        Returns:
            str: The citation form of the pronoun, one of "i", "you", "he", "she",
            "it", "we", "they" and None.
        """
        pronoun = attributes["tokens"][0]

        pronoun = pronoun.lower()
        if re.match("^(he|him|himself|his)$",  pronoun):
            return "he"
        elif re.match("^(she|her|herself|hers|her)$",  pronoun):
            return "she"
        elif re.match("^(it|itself|its)$",  pronoun):
            return "it"
        elif re.match("^(they|them|themselves|theirs|their)$",  pronoun):
            return "they"
        elif re.match("^(i|me|myself|mine|my)$",  pronoun):
            return "i"
        elif re.match("^(you|yourself|yourselves|yours|your)$",  pronoun):
            return "you"
        elif re.match("^(we|us|ourselves|ours|our)$",  pronoun):
            return "we"

    def _get_ancestry(self, dep_tree, index, level=0):
        if level >= 2:
            return ""
        else:
            governor_id = dep_tree[index].head - 1

            direction = "L"

            if governor_id > index:
                direction = "R"

            if governor_id == -1:
                return  "-" + direction + "-NONE"
            else:
                return "-" + direction + "-" + dep_tree[governor_id].pos + \
                    self._get_ancestry(dep_tree, governor_id, level+1)

    def needs_parse_trees(self):
        return True

    def needs_dependency_trees(self):
        return True


class DummyMentionPropertyComputer(MentionPropertyComputer):
    def compute_properties(self, mention, mention_id):
        attributes = mention.attributes

        attributes["parse_tree"] = None

        attributes["is_apposition"] = False

        attributes["grammatical_function"] = "UNKNOWN"

        head_index = len(attributes["tokens"]) - 1
        head = [attributes["tokens"][-1]]

        attributes["head"] = head
        attributes["head_span"] = spans.Span(
            mention.span.begin + head_index,
            mention.span.begin + head_index
        )
        attributes["head_index"] = head_index

        attributes["type"] = "UNKNOWN"
        attributes["fine_type"] = "UNKNOWN"

        attributes["citation_form"] = "UNKNOWN"

        attributes["number"] = "UNKNOWN"
        attributes["gender"] = "UNKNOWN"

        attributes["semantic_class"] = "UNKNOWN"

        attributes["governor"] = "UNKNOWN"

        attributes["ancestry"] = "UNKNOWN"

        attributes["deprel"] = "UNKNOWN"

        attributes["id"] = mention_id


class NoDependenciesEnglishMentionPropertyComputer(EnglishMentionPropertyComputer):
    def compute_properties(self, mention, mention_id):
        attributes = mention.attributes

        attributes["parse_tree"] = self.get_relevant_subtree(
            mention.span, mention.document)

        attributes["is_apposition"] = \
            self._is_apposition(attributes)

        attributes["grammatical_function"] = \
            self._get_grammatical_function(attributes)

        (head, in_mention_span, head_index) = \
            self._compute_head_information(attributes)

        attributes["head"] = head
        attributes["head_span"] = spans.Span(
            mention.span.begin + in_mention_span.begin,
            mention.span.begin + in_mention_span.end
        )
        attributes["head_index"] = head_index

        attributes["head_as_lowercase_string"] = " ".join(attributes[
            "head"]).lower()

        attributes["type"] = self._get_type(attributes)

        if attributes["type"] == "PRO":
            attributes["citation_form"] = self._get_citation_form(attributes)

        attributes["fine_type"] = self._get_fine_type(
            attributes)

        attributes["number"] = self._compute_number(attributes)
        attributes["gender"] = self._compute_gender(attributes)

        attributes["semantic_class"] = self._compute_semantic_class(attributes)

        attributes["id"] = mention_id

    def needs_parse_trees(self):
        return True

    def needs_dependency_trees(self):
        return False