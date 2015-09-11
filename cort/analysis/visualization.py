from __future__ import print_function

import codecs
import shutil
import html
import os
import webbrowser
from random import randint
import collections

import cort
from cort.core import spans
from cort.analysis import data_structures


__author__ = "Thierry Goeckel, Sebastian Martschat"


class Visualizer:
    def __init__(self, structured_coreference_analysis, corpus_name,
                 highlight_error=None, for_raw_input=False):
        self.html_header = \
            "<!doctype html>\n" \
            "<html>\n" \
            "\t<head>\n" \
            "\t\t<title>cort visualization</title>\n" \
            "\t\t<link rel=\"stylesheet\" type=\"text/css\" " \
            "href=\"visualization/style.css\">\n" \
            "\t\t<script src=\"visualization/lib/jquery-2.1.1.min.js\">" \
            "</script>\n" \
            "\t\t<script src=\"visualization/lib/jquery.jsPlumb-1.6.4.js\"></script>\n" \
            "\t\t<script src=\"visualization/lib/cort.js\"></script>"

        self.chain_to_colour = {}
        self.colours = []
        self.navi = {}
        self.structured_coreference_analysis = structured_coreference_analysis
        self.highlight_error = highlight_error
        self.corpus_name = corpus_name
        self.for_raw_input = for_raw_input

    def run(self):
        documents_html = ""
        documents_navi = "\n\t\t<div id=\"documentsNavi\"><h3>Documents</h3>" \
                         "\n\t\t\t<ul>"
        errors_source = "\n\t\t<script>\n" \
                        "\t\t\terrors = [ "

        system_corpus = self.structured_coreference_analysis.corpora[
            self.corpus_name]

        for document in sorted(self.structured_coreference_analysis.reference.documents,
                               key=lambda doc:
                               doc.get_html_friendly_identifier()):
            doc_id = document.get_html_friendly_identifier()
            documents_navi += "\n\t\t\t\t<li>" + doc_id + "</li>"

            document_mentions = document.annotated_mentions

            if not self.for_raw_input:
                system_mentions = system_corpus.documents[
                                        system_corpus.documents.index(document)
                                    ].annotated_mentions
                document_mentions = sorted(document_mentions + system_mentions)

            if self.for_raw_input:
                text_source = self.__generate_html_for_raw(document,
                                                           document_mentions)
            else:
                text_source = self.__generate_html_for_errors(document,
                                                              document_mentions)

            if self.for_raw_input:
                documents_html = documents_html + "\n\t\t<div id=\"" + doc_id + \
                                "\" class=\"document\">" + \
                                "\n\t\t\t<div class=\"navcontainer\">" + \
                                self.navi["system"] + \
                                "\n\t\t\t\t\t</ul>\n\t\t\t\t</div>" \
                                "\n\t\t\t</div>" + \
                                text_source + "\n\t\t</div>"

            else:
                documents_html = documents_html + "\n\t\t<div id=\"" + doc_id + \
                                "\" class=\"document\">" \
                                "\n\t\t\t<div class=\"navcontainer\">" + \
                                self.__generate_errors_navi_by_mention_type(document) + \
                                self.navi["gold"] + \
                                "\n\t\t\t\t\t</ul>\n\t\t\t\t</div>" + \
                                self.navi["system"] + \
                                "\n\t\t\t\t\t</ul>\n\t\t\t\t</div>" \
                                "\n\t\t\t</div>" + \
                                text_source + \
                                "\n\t\t</div>"

            recall_errors = self.structured_coreference_analysis[
                self.corpus_name]["recall_errors"]["all"].filter(
                            lambda err: err[0].document == document)
            precision_errors = self.structured_coreference_analysis[
                self.corpus_name]["precision_errors"]["all"].filter(
                            lambda err: err[0].document == document)
            decisions = self.structured_coreference_analysis[
                self.corpus_name]["decisions"]["all"].filter(
                            lambda err: err[0].document == document)

            if isinstance(recall_errors, data_structures.StructuredCoreferenceAnalysis):
                for category in recall_errors.keys():
                    errors_source += self.__generate_errors_source(
                        recall_errors[category],
                        category,
                        document_mentions,
                        "Recall")
            else:
                errors_source += self.__generate_errors_source(
                    recall_errors,
                    "",
                    document_mentions,
                    "Recall")

            if isinstance(precision_errors, data_structures.StructuredCoreferenceAnalysis):
                for category in precision_errors.keys():
                    errors_source += self.__generate_errors_source(
                        precision_errors[category],
                        category,
                        document_mentions,
                        "Precision")
            else:
                errors_source += self.__generate_errors_source(
                    precision_errors,
                    "",
                    document_mentions,
                    "Precision")

            if isinstance(decisions, data_structures.StructuredCoreferenceAnalysis):
                for category in decisions.keys():
                    errors_source += self.__generate_errors_source(
                        decisions[category],
                        category,
                        document_mentions,
                        "Decision")
            else:
                errors_source += self.__generate_errors_source(
                    decisions,
                    "",
                    document_mentions,
                    "Decision")

        errors_source = errors_source[:-2] + " ];\n\t\t\tchain_to_colour = {"

        for chain in self.chain_to_colour.keys():
            errors_source = errors_source + chain + ": \"" + self.chain_to_colour[chain] + "\", "

        errors_source = errors_source[:-2] + "};\n\t\t</script>"

        html_source = self.html_header + \
            "\n\t</head>\n\t<body>" \
            "\n\t\t<div id=\"header\"><h1>cort visualization: <span id=\"document_name\">" \
            "</span></h1></div>" + documents_navi + \
            "\n\t\t\t</ul>\n\t\t</div>"

        html_source += documents_html + errors_source + "\n\t</body>\n</html>"

        if not os.path.exists("temp/output"):
            os.makedirs("temp/output")

        output = "temp/output/error_analysis.html"

        f = codecs.open(output, "w", "utf-8")

        abs_path = os.path.abspath(output)

        print("Writing " + abs_path)

        f.write(html_source)

        f.close()

        # copy js/css
        package_dir = cort.__path__[0]
        if not os.path.exists("temp/output/visualization"):
            shutil.copytree(package_dir + "/analysis/visualization",
                            "temp/output/visualization")

        if self.for_raw_input:
            shutil.copy(package_dir + "/analysis/visualization/lib/cort-for-raw.js",
                        "temp/output/visualization/lib/cort.js")

        webbrowser.open_new_tab("file://" + abs_path)

    def __generate_html_for_errors(self, document, mentions):
        document_html = "\n\t\t\t<ol class=\"text\">\n" \
                        "\t\t\t\t<li class=\"sentence\">"

        self.navi["gold"] = "\n\t\t\t\t<div class=\"goldNavi\">" \
                            "<h3>Reference Entities</h3>" \
                            "<span class=\"tease\">show all</span>" \
                            "\n\t\t\t\t\t<ul>"

        self.navi["system"] = "\n\t\t\t\t<div class=\"systemNavi\">" \
                              "<h3>System Entities</h3>" \
                              "<span class=\"tease\">show all</span>" \
                              "\n\t\t\t\t\t<ul>"

        chains = set()

        index = 0

        sentence_id, sentence_span = document.get_sentence_id_and_span(
            spans.Span(0, 0))

        annotated_mentions = set(document.annotated_mentions)

        for token in document.tokens:
            token = html.escape(token)

            mention_id = 0

            mention_text = ""

            processed_gold_mentions = set()

            for mention in mentions:
                if mention.span.begin > index:
                    break

                if mention.span.end < index:
                    mention_id += 1
                    continue

                mention_tokens = html.escape(" ".join(mention.attributes[
                    'tokens']), True)

                mention_head = html.escape(" ".join(mention.attributes[
                    'head']), True)

                mention_type = html.escape("".join(mention.attributes[
                    'type']), True)

                mention_span = str(mention.span)

                if mention in annotated_mentions and \
                   mention not in processed_gold_mentions:
                    system = "gold"

                    processed_gold_mentions.add(mention)
                else:
                    system = "system"

                chain_id = system + str(mention.attributes['annotated_set_id'])

                if chain_id not in chains:
                    self.navi[system] += "\n\t\t\t\t\t\t<li class=\"" + \
                                         chain_id +\
                                         "\">" + mention_tokens + "</li>"
                    chains.add(chain_id)

                if chain_id not in self.chain_to_colour.keys():
                    while True:
                        r = lambda: randint(170, 255)
                        colour = '#%02X%02X%02X' % (r(), r(), r())
                        if colour not in self.colours:
                            self.colours.append(colour)
                            break

                    self.chain_to_colour[chain_id] = colour

                span_id = document.get_html_friendly_identifier() + "_" + \
                          str(mention_id)

                temp_text = "<span " \
                            "id=\"" + span_id + "\" " \
                            "class=\"" + chain_id + " mention\" " \
                            "data-mentiontype=\"" + mention_type + "\" " \
                            "data-mentionhead=\"" + mention_head + "\" " \
                            "data-span=\"" + mention_span + "\">"

                if mention.span.begin == index and mention.span.end == index:
                    if mention_text.endswith("</span> "):
                        mention_text = temp_text + mention_text.strip() + \
                            "</span> "
                    elif mention_text == "":
                        mention_text = temp_text + token + "</span> "
                elif mention.span.begin == index:
                    if mention_text == "":
                        mention_text = temp_text + token + " "
                    else:
                        mention_text = temp_text + mention_text
                elif mention.span.end == index:
                    if mention_text == "":
                        mention_text = token + "</span> "
                    else:
                        mention_text = mention_text.strip() + "</span> "

                mention_id += 1

            if mention_text == "":
                mention_text = token + " "

            token_span = spans.Span(index, index)

            if document.get_sentence_id_and_span(token_span) is None or \
                    sentence_span != document.get_sentence_id_and_span(
                            token_span)[1]:
                mention_text = "</li>\n" \
                               "\t\t\t\t<li class=\"sentence\">" + mention_text

                sentence_id, sentence_span = document.get_sentence_id_and_span(token_span)

            document_html += mention_text

            index += 1

        document_html.strip()

        return document_html + "</li>\n\t\t\t</ol>"

    def __generate_html_for_raw(self, document, mentions):
        document_html = "\n\t\t\t<ol class=\"text\">\n" \
                        "\t\t\t\t<li class=\"sentence\">"

        self.navi["gold"] = "\n\t\t\t\t<div class=\"goldNavi\">" \
                            "<h3>Reference Entities</h3>" \
                            "<span class=\"tease\">show all</span>" \
                            "\n\t\t\t\t\t<ul>"

        self.navi["system"] = "\n\t\t\t\t<div class=\"systemNavi\">" \
                              "<h3>System Entities</h3>" \
                              "<span class=\"tease\">show all</span>" \
                              "\n\t\t\t\t\t<ul>"

        chains = set()

        index = 0

        sentence_id, sentence_span = document.get_sentence_id_and_span(
            spans.Span(0, 0))

        annotated_mentions = set(document.annotated_mentions)

        temp_navi = {
            "gold": {},
            "system": {}
        }

        chain_counter = {
            "gold": collections.Counter(),
            "system": collections.Counter()
        }

        for system in ["gold", "system"]:
            for mention in annotated_mentions:
                chain_counter[system].update([system + str(mention.attributes[
                    "annotated_set_id"])])

        for token in document.tokens:
            token = html.escape(token)

            mention_id = 0

            mention_text = ""

            processed_gold_mentions = set()

            for mention in mentions:
                if mention.span.begin > index:
                    break

                if mention.span.end < index:
                    mention_id += 1
                    continue

                mention_tokens = html.escape(" ".join(mention.attributes[
                    'tokens']), True)

                mention_head = html.escape(" ".join(mention.attributes[
                    'head']), True)

                mention_type = html.escape("".join(mention.attributes[
                    'type']), True)

                mention_span = str(mention.span)

                system = "system"

                chain_id = system + str(mention.attributes['annotated_set_id'])

                if chain_id not in chains:
                    temp_navi[system][chain_id] = "\n\t\t\t\t\t\t<li " \
                                                  "class=\"" + \
                                         chain_id +\
                                         "\">" + mention_tokens + "</li>"
                    chains.add(chain_id)

                if chain_id not in self.chain_to_colour.keys():
                    while True:
                        r = lambda: randint(170, 255)
                        colour = '#%02X%02X%02X' % (r(), r(), r())
                        if colour not in self.colours:
                            self.colours.append(colour)
                            break

                    self.chain_to_colour[chain_id] = colour

                span_id = document.get_html_friendly_identifier() + "_" + \
                          str(mention_id)

                style = ""

                if chain_counter[system][chain_id] > 1:
                    style = "style=\"background-color:" + self.chain_to_colour[
                        chain_id] + "\" "

                temp_text = "<span " \
                            "id=\"" + span_id + "\" " \
                            "class=\"" + chain_id + " mention\" "  + \
                            style + \
                            "data-mentiontype=\"" + mention_type + "\" " \
                            "data-mentionhead=\"" + mention_head + "\" " \
                            "data-span=\"" + mention_span + "\">"

                if mention.span.begin == index and mention.span.end == index:
                    if mention_text.endswith("</span> "):
                        mention_text = temp_text + mention_text.strip() + \
                            "</span> "
                    elif mention_text == "":
                        mention_text = temp_text + token + "</span> "
                elif mention.span.begin == index:
                    if mention_text == "":
                        mention_text = temp_text + token + " "
                    else:
                        mention_text = temp_text + mention_text
                elif mention.span.end == index:
                    if mention_text == "":
                        mention_text = token + "</span> "
                    else:
                        mention_text = mention_text.strip() + "</span> "

                mention_id += 1

            if mention_text == "":
                mention_text = token + " "

            token_span = spans.Span(index, index)

            if document.get_sentence_id_and_span(token_span) is None or \
                    sentence_span != document.get_sentence_id_and_span(
                            token_span)[1]:
                mention_text = "</li>\n" \
                               "\t\t\t\t<li class=\"sentence\">" + mention_text

                sentence_id, sentence_span = document.get_sentence_id_and_span(token_span)

            document_html += mention_text

            index += 1

        document_html.strip()

        for system in ["system"]:
            for key, val in chain_counter[system].items():
                if val > 1:
                    self.navi[system] += temp_navi[system][key]

        return document_html + "</li>\n\t\t\t</ol>"

    def __generate_errors_source(self, errors, category, sorted_mentions,
                                 error_type):
        errors_source = ""

        for error in errors:
            doc_id = error[0].document.get_html_friendly_identifier()
            antecedent_id = -1
            anaphor_id = -1
            mention_id = 0
            for mention in sorted_mentions:
                if mention == error[1] and antecedent_id == -1:
                    antecedent_id = mention_id
                if mention == error[0] and anaphor_id == -1:
                    anaphor_id = mention_id
                if antecedent_id > -1 and anaphor_id > -1:
                    break
                mention_id += 1
            error_source = \
                "{ anaphor: \"" + doc_id + "_" + str(anaphor_id) + "\", " \
                "antecedent: \"" + doc_id + "_" \
                + str(antecedent_id) + "\", " \
                "category: \"" + str(category) + "\", " \
                "type: \"" + error_type + "\""
            if error == self.highlight_error:
                error_source += ", highlight: \"true\" }, "
            else:
                error_source += " }, "
            errors_source += error_source

        return errors_source

    def __generate_errors_navi_by_mention_type(self, document):
        recall_errors = self.structured_coreference_analysis[
                            self.corpus_name]["recall_errors"]["all"].filter(
            lambda x: x[0].document == document)
        precision_errors = self.structured_coreference_analysis[
                            self.corpus_name]["precision_errors"]["all"].filter(
            lambda x: x[0].document == document)

        recall_error_count = len(recall_errors)

        precision_error_count = len(precision_errors)

        # Precision errors
        precision_errors_navi = "\n\t\t\t\t\t<div><h4>Precision (" + \
                            str(precision_error_count) +\
                            ")</h4><span class=\"tease\">show all</span>" \
                            "\n\t\t\t\t\t\t<ul class=\"precisionErrors\">"

        # Recall errors
        recall_errors_navi = "\n\t\t\t\t\t\t<div><h4>Recall (" + \
                            str(recall_error_count) + \
                            ")</h4><span class=\"tease\">show all</span>" \
                            "\n\t\t\t\t\t\t<ul class=\"recallErrors\">"

        recall_categories = set()
        precision_categories = set()

        if isinstance(recall_errors, data_structures.StructuredCoreferenceAnalysis):
            for cat in recall_errors.keys():
                if len(recall_errors[cat]) > 0:
                    recall_categories.add(cat)

        if isinstance(precision_errors, data_structures.StructuredCoreferenceAnalysis):
            for cat in precision_errors.keys():
                if len(precision_errors[cat]) > 0:
                    precision_categories.add(cat)

        recall_categories = sorted([str(cat) for cat in recall_categories])
        precision_categories = sorted([str(cat) for cat
                                       in precision_categories])

        for cat in recall_categories:
            recall_errors_navi += "\n\t\t\t\t\t\t\t<li>"

            error_count_cat = len(recall_errors[cat])

            recall_errors_navi += str(cat) + ": " + \
                str(error_count_cat)

            recall_errors_navi += "</li>"

        for cat in precision_categories:
            precision_errors_navi += "\n\t\t\t\t\t\t\t<li>"

            error_count_cat = len(precision_errors[cat])

            precision_errors_navi += str(cat) + ": " + \
                str(error_count_cat)

            precision_errors_navi += "</li>"

        precision_errors_navi += "\n\t\t\t\t\t\t</ul>" \
                            "\n\t\t\t\t\t</div>"
        recall_errors_navi += "\n\t\t\t\t\t\t</ul>" \
                            "\n\t\t\t\t\t</div>"
        return "\n\t\t\t\t<div class=\"errorsNavi\">" \
                "\n\t\t\t\t\t<h3>Errors (" + \
                str(precision_error_count + recall_error_count) + ")</h3>" \
                "\n\t\t\t\t\t<span class=\"tease\">show all</span>" + \
                precision_errors_navi + recall_errors_navi + \
                "\n\t\t\t\t</div>"
