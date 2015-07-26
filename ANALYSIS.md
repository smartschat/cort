# Error Analysis with cort

With __cort__, you can analyze recall and precision errors of your coreference 
resolution systems just with a few lines in python. In this readme, we show
the capabilities of __cort__ by going through an example.

## Contents

* [Reading in Data](#reading)
* [Extracting Errors](#extracting)
* [Filtering and Categorizing](#filtering)
* [Visualization](#visualization)
* [Plotting](#plotting)
* [Mention Attributes](#attributes)
* [Format for Antecedent Data](#antecedents)
* [Advanced Examples](#advanced)

## <a name="reading"></a> Reading in Data

So far, __cort__ only supports data in [the format from the CoNLL shared tasks 
on coreference resolution](http://conll.cemantix.org/2012/data.html). Let us 
assume that you have some data in this format: all documents with reference 
annotations are in the file `reference.conll`. You have output of two 
coreference resolution systems, the output of the first system is in the file
`pair-output.conll`, while the output of the other system is in the file
 `tree-output.conll`. If you want to compute precision errors, you also need a 
[file that contains antecedent decisions](#antecedents). Let us assume  
we have such a file for the first system, the file is called`pair.antecedents`.

First, let us load the data:

```python
from cort.core import corpora

reference = corpora.Corpus.from_file("reference", open("reference.conll"))
pair = corpora.Corpus.from_file("pair", open("pair-output.conll"))
tree = corpora.Corpus.from_file("tree", open("tree-output.conll"))

# optional -- not needed when you only want to compute recall errors
pair.read_antecedents(open('pair.antecedents'))
```

## <a name="extracting"></a> Extracting Errors

We now want to extract the errors. For this, we use an `ErrorExtractor`. 
In addition to the corpora, we need to provide the `ErrorExtractor` with the 
algorithms it should use to extract recall and precision errors. We stick to 
the algorithms described in the EMNLP'14 paper.

```python
from cort.analysis import error_extractors
from cort.analysis import spanning_tree_algorithms

extractor = error_extractors.ErrorExtractor(
	reference,
    spanning_tree_algorithms.recall_accessibility,
    spanning_tree_algorithms.precision_system_output
)

extractor.add_system(pair)
extractor.add_system(tree)

errors = extractor.get_errors()
```

That's it! `errors` now contains all errors of the two systems under 
consideration. The errors can be accessed like in a nested dict:
`errors["tree"]["recall_errors"]["all"]` contains all recall errors of the
second system, while `errors["pair"]["precision_errors"]["all"]` contains
all precision errors of the first system. If you supplied antecedent data,
`errors["pair"]["decisions"]["all"]` contains all antecedent decisions. 
`errors` is an instance of the class `StructuredCoreferenceAnalysis`.

## <a name="filtering"></a> Filtering and Categorizing

For further analysis, you will want to filter and categorize the errors you've 
extracted. That's why `StructuredCoreferenceAnalysis` provides the member 
functions `filter` and `categorize`. These take as input functions which 
filter or categorize errors.

Each error is internally represented as a tuple of two `Mention` objects, the 
anaphor and the antecedent. Given an error `e`, we can access these with `e[0]` 
and `e[1]` or with `anaphor, antecedent = e`.

Hence, we can obtain all errors where the anaphor is a pronoun and the
antecedent is a proper name as follows:

```python
pron_anaphor_errors = errors.filter(
	lambda error: error[0].attributes['type'] == "PRO" and 
	              error[1].attributes['type'] == "NAM"
)
```

Or we only do this filtering for recall errors of the second system:

```python
pron_anaphor_tree_recall_errors = errors["tree"]["recall_errors"].filter(
	lambda error: error[0].attributes['type'] == "PRO" and 
	              error[1].attributes['type'] == "NAM"
)
```

We can categorize each error by the mention types of the anaphor:

```python
errors_by_type = errors.categorize(
	lambda error: error[0].attributes['type']
)
```

The corresponding errors can now be accessed with 
`errors_by_type["pair"]["recall_errors"]["all"]["NOM"]`.

For more information on the attributes of the mentions which you can access, 
have a look at the documentation of `Mention`, or consult [the list included 
in this readme](#attributes).


## <a name="visualization"></a> Visualization

Errors of one system can be visualized by providing the name of the system:

```python
errors_by_type.visualize("pair")
```

This opens a visualization of the errors in a web browser. Below is a
screenshot of the visualization.

![Screenshot of the visualization](visualization.png)

The header displays the identifier of the document in focus. The left bar 
contains the navigation panel, which includes
* a list of all documents in the corpus,
* a summary of all errors for the document in focus, and
* lists of reference and system entities for the document in focus.

To the right of the navigation panel, the document in focus is shown. Mentions in
reference entities have a gold border, mention in system entities have a blue
border.

When the user picks a reference or system entity from the corresponding list, 
__cort__ displays all recall and precision errors for all mentions which are 
contained in the entity (as labeled red arrows between mentions). Furthermore,
it shows all antecedent antecedent decisions for the entity (as blue arrows).
The user can also click on mentions. Then all related errors and antecedent decisions
are displayed. Alternatively, the user can choose an error category from the error summary. 
In that case, all errors of that category are displayed.

We use color to distinguish between entities: mentions in different entities 
have different background colors. Additionally mentions in reference entities 
have a yellow border, while mentions in system entities have a blue border.

The visualization relies on [jQuery](https://jquery.org/) and 
[jsPlumb](http://www.jsplumb.org/). The libraries are contained in our toolkit.

## <a name="plotting"></a> Plotting

To assess differences in error distributions, __cort__ provides plotting
functionality.

```python
from cort.analysis import plotting

pair_errs = errors_by_type["pair"]["recall_errors"]["all"]
tree_errs = errors_by_type["tree"]["recall_errors"]["all"]

plotting.plot(
    [("pair", [(cat, len(errs)) for cat, errs in pair_errs.items()]),
     ("tree", [(cat, len(errs)) for cat, errs in tree_errs.items()])],
    "Recall Errors",
    "Type of anaphor",
    "Number of Errors")
```

This produces the following plot:

![An example plot](plot.png)


## <a name="attributes"></a> Mention Attributes

You can access an attribute of a mention `m` via 
`m.attributes['attribute_name']`.

Name | Type | Description
---- | ---- | -----------
tokens | list(str) | the tokens of the mention
head | list(str) | the head words of the mention
pos | list(str) | the part-of-speech tag of the mention
ner | list(str) | the named entity tags of the mention, as found in the data
type | str | the mention type, one of NAM (proper name), NOM (common noun), PRO (pronoun), DEM (demonstrative pronoun), VRB (verb)
fine_type | str | only set when the mention is a nominal or a pronoun, for nominals values DEF (definite noun phrase) or INDEF (bare plural or indefinite), for pronouns values PERS_NOM (personal pronoun, nominative case), PERS_ACC (personal pronoun, accusative), REFL (reflexive pronoun), POSS (possessive pronoun) or POSS_ADJ (possessive adjective, e.g. 'his')
citation_form | str | only set if the mention is a pronoun, then the canonical form of the pronoun, i.e. one of i, you, he, she, it, we, they
grammatical_function | str | either SUBJECT, OBJECT or OTHER
number | str | either SINGULAR, PLURAL or UNKNOWN
gender | str | either MALE, FEMALE, NEUTRAL, PLURAL or UNKNOWN
semantic_class | str | either PERSON, OBJECT or UNKNOWN
sentence_id | int | the sentence id of the mention's sentence (starting at 0)
parse_tree | nltk.ParentedTree | the parse tree of the mention
speaker | str | the speaker of the mention
antecedent | cort.core.mentions.Mention | the antecedent of the mention (intially None)
annotated_set_id | str | the set id of the mention as found in the data
set_id | str | the set id of the mention computed by a coreference resolution approach (initially None)
head_span | cort.core.spans.Span | the span of the head (in the document)
head_index | int | the mention-internal index of the start of the head
is_apposition | bool| whether the mention contains an apposition


## <a name="antecedents"></a>  Format for Antecedent Data

There is no standardized format for storing antecedent decisions on CoNLL 
coreference data. The toolkit expects the following format:

Files should have one antecedent decision per line. Entries in each line are
seperated by tabs. The format is

``doc_identifier	(anaphor_start, anaphor_end)	(ante_start, ante_end)``

where

* doc_identifier is the identifier in the first line of an CoNLL document
  after #begin document, such as (bc/cctv/00/cctv_0000); part 000
* `anaphor_start` is the position in the document where the anaphor begins 
  (counting from 0),
* `anaphor_end` is the position where the anaphor ends (inclusive),
* `ante_start`, `ante_end` analogously for the antecedent.

An example is

``(bc/cctv/00/cctv_0000); part 000   (10, 11) (1, 1)``

## <a name="advanced"></a> Advanced Examples

Here we show some advanced functionality.

### Categorizing by Status of Anaphor in the Reference Corpus

```python
all_gold = set()
for doc in reference:
    for mention in doc.annotated_mentions:
        all_gold.add(mention)


def is_anaphor_gold(mention):
    if mention in all_gold:
        return "is_gold"
    else:
        return "is_not_gold"

is_ana_gold = errors_by_type.categorize(lambda err: is_anaphor_gold(err[0]))
```

### Head Statistics for Common Noun Errors errors

```python
from collections import Counter

for system in ["pair", "tree"]:
    nom_rec_errs = errors_by_type[system]["recall_errors"]["all"]["NOM"]
    all_heads = [" ".join(err[0].attributes["head"]).lower() for err in nom_rec_errs]
    most_common = Counter(all_heads).most_common(10)
    print(system, most_common)
```

### Common Errors

```python
common = {
    "common": {
        "recall_errors": {},
        "precision_errors": {}
    }
}

common["common"]["recall_errors"]["all"] = errors["pair"]["recall_errors"][
    "all"].intersection(errors["tree"]["recall_errors"]["all"])

common["common"]["precision_errors"]["all"] = errors["pair"]["precision_errors"][
    "all"].intersection(errors["tree"]["precision_errors"]["all"])

from cort.analysis import data_structures
common = data_structures.StructuredCoreferenceAnalysis(
    common, errors.reference, errors.corpora
)
```

### Plot Decisions

```python
decs = errors_by_type["pair"]["decisions"]["all"]
prec_errs = errors_by_type["pair"]["precision_errors"]["all"]

plotting.plot(
    [("decisions", [(cat, len(errs)) for cat, errs in decs.items()]),
     ("errors", [(cat, len(errs)) for cat, errs in prec_errs.items()])],
    "Decisions and Errors",
    "Type of anaphor",
    "Number")
```