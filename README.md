# crtk

__crtk__ is a __c__oreference __r__esolution __t__ool__k__it. It implements the 
coreference resolution error analysis framework described in our 
[EMNLP'14 paper](#references), and also ships with a well-performing 
deterministic coreference resolution system. It needs Python 2.6+ and 
NLTK 2.0.4+. 

If you have any questions or comments, drop me an e-mail at 
[sebastian.martschat@gmail.com](mailto:sebastian.martschat@gmail.com).


## Error Analysis

With __crtk__, you can analyze recall and precision errors of your coreference 
resolution systems just with a few lines in python. Let us go through an 
example.

So far, __crtk__ only supports data in [the format from the CoNLL shared tasks 
on coreference resolution](http://conll.cemantix.org/2012/data.html). Let us 
assume that you have some data in this format: all documents with reference 
annotations are in the file `reference.data`, while the system output is in the 
file `output.data`. If you want to compute precision errors, you also need a 
file `antecedents.data` with antecedent decisions of the system.

First, let us load the data:

```python
from crtk.core import corpora

reference = corpora.Corpus.from_file("reference corpus", open("reference.data"))
output = corpora.Corpus.from_file("output corpus", open("output.data"))

# optional -- not needed when you only want to compute recall errors
output.read_antecedents(open('antecedents.data'))
```

We now want to extract the errors. For this, we use an `ErrorAnalysis`. 
In addition to the corpora, we need to provide the `ErrorAnalysis` with the 
algorithms it should use to extract recall and precision errors. We stick to 
the algorithms described in the EMNLP'14 paper.

```python
from crtk.analysis import error_representations
from crtk.analysis import spanning_tree_algorithms

errors = error_representations.ErrorAnalysis(
	reference,
    output,
    spanning_tree_algorithms.recall_accessibility,
    spanning_tree_algorithms.precision_system_output
)
```

That's it! You now can access all recall errors by `errors.recall_errors` and 
all precision errors by `errors.precision_errors`. These are both instances of 
`ErrorSet`.

For further analysis, you will want to filter and categorize the errors you've 
extracted. That's why `ErrorSet` provides the member functions `filter` and 
`categorize`. These take as input functions which filter or categorize errors.

Each error is internally represented as a tuple of two `Mention` objects, the 
anaphor and the antecedent. Given an error `e`, we can access these with `e[0]` 
and `e[1]` or with `anaphor, antecedent = e`.

Hence, we can obtain all recall errors where the anaphor is a pronoun as 
follows:

```python
pron_anaphor_recall_errors = errors.recall_errors.filter(
	lambda error: error[0].attributes['type'] == "PRO"
)
```

Or we can categorize each error by the mention types of anaphor and antecedent:

```python
# by_type is a dict, mapping string tuples to ErrorSets
by_type = errors.recall_errors.categorize(
	lambda error: (error[0].attributes['type'], error[1].attributes['type'])
)
```

For more information on the attributes of the mentions which you can access, 
have a look at the documentation of `Mention`, or consult [the list included 
in this readme](#attributes).



## Coreference Resolution

This toolkit also contains a well-performing deterministic coreference 
resolution system. You can run this system using the script `run-coref` as 
follows:

```shell
./run-coref -in reference.data -out out.data
```

With the optional argument `-ante`, antecedent decisions are also written to a 
file:

```shell
./run-coref -in reference.data -out out.data -ante antecedents_out.data
```


## <a name="attributes"></a> Mention Attributes

You can access an attribute of a mention `m` via 
`m.attributes['attribute_name']`.

Name | Type | Description
---- | ---- | -----------
tokens | list(str) | the tokens of the mention
head | list(str) | the head words of the mention
pos | list(str) | the part-of-speech tags of the mention
ner | list(str) | the named entity tags of the mention, as found in the data
type | str | the mention type, one of NAM (proper name), NOM (common noun), PRO (pronoun), DEM (demonstrative pronoun), VRB (verb)
fine_type | str | only set when the mention is a nominal or a pronoun, for nominals values DEF (definite noun phrase) or INDEF (bare plural ore indefinite), for pronouns values PERS_NOM (personal pronoun, nominative case), PERS_ACC (personal pronoun, accusative), REFL (reflexive pronoun), POSS (possessive pronoun) or POSS_ADJ (possessive adjective, e.g. 'his')
citation_form | str | only set if the mention is a pronoun, then the canonical form of the pronoun, i.e. one of i, you, he, she, it, we, they
grammatical_function | str | either SUBJECT, OBJECT or OTHER
number | str | either SINGULAR, PLURAL or UNKNOWN
gender | str | either MALE, FEMALE, NEUTRAL, PLURAL or UNKNOWN
semantic_class | str | either PERSON, OBJECT or UNKNOWN
sentence_id | int | the sentence id of the mention's sentence (starting at 0)
parse_tree | nltk.ParentedTree | the parse tree of the mention
speaker | str | the speaker of the mention
antecedent | crtk.core.mentions.Mention | the antecedent of the mention (intially None)
annotated_set_id | str | the set id of the mention as found in the data
set_id | str | the set id of the mention computed by a coreference resolution approach (initially None)
head_span | crtk.core.spans.Span | the span of the head (in the document)
head_index | int | the mention-internal index of the start of the head,
is_apposition | bool| whether the mention contains an apposition


## Format for Antecedent Data

There is no standardized format for storing antecedent decisions on CoNLL 
coreference data. The toolkit expects the following format:

Files should have one antecedent decision per line, in the format

	doc_id doc_part anaphor_start anaphor_end ante_start ante_end

where

* `doc_id` is the id as in the first column of the CoNLL original data,
* `doc_part` is the part number (without trailing 0s),
* `anaphor_start` is the position in the document where the anaphor begins 
  (counting from 0),
* `anaphor_end` is the position where the anaphor ends (inclusive),
* `ante_start`, `ante_end` analogously for the antecedent.


## <a name="references"></a> References

If you use this toolkit in your research, please cite the following publication:

**Sebastian Martschat and Michael Strube (2013).** Recall Error Analysis for 
Coreference Resolution. In *Proceedings of the 2014 Conference on Empirical 
Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 25-29 October 
2014, pages 2070-2081. http://aclweb.org/anthology/D/D14/D14-1221.pdf
