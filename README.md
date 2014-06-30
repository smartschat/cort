Multigraph
==========

This is an implementation of the coreference resolution system described in 
Martschat (2013) (see the reference below), extended with distance reweighting.

For questions or comments, drop me an email: sebastian.martschat@gmail.com.

### Usage ###

...
$ python run_coref.py input.conll output.conll
...

input.conll should be following the format of the CoNLL shared tasks on 
coreference resolution (see http://conll.cemantix.org/2012/data.html).

### Requirements ###

Python 2.6+
NLTK 2.0.4+

### Results ###

Here are the results on CoNLL'12 English development and test data, obtained 
with the [CoNLL reference scorer](http://code.google.com/p/reference-coreference-scorers/):

CoNLL'12 English dev:
---------------------
MUC Recall: 66.62%, Precision: 69.45%, F1: 68.01%
BCUB Recall: 53.06%, Precision: 62.14%, F1: 57.24%
CEAFE Recall: 56.90%, Precision: 52.26%, F1: 54.48%

CoNLL average F1: 59.91

CoNLL'12 English test:
----------------------
MUC Recall: 66.66%, Precision: 68.82%, F1: 67.72%
BCUB Recall: 51.29%, Precision: 60.03%, F1: 55.32%
CEAFE Recall: 54.61%, Precision: 50.14%, F1: 52.28%

CoNLL average F1: 58.44

### Results without Distance Reweighting ###

In the results above, the weights between two mentions are reweighted by 
sentence distance (via w' = w/(sentenceDistance+1). This reweighting was not
used in Martschat (2013). To turn this feature off, you have to invoke
MultigraphDecoder with do_distance_reweighting=False, i.e. change line 44 of 
run_coref.py to 

decoder = MultigraphDecoder(positive_features, negative_features, do_distance_reweighting=False)

The results then are as follows:

CoNLL'12 English dev:
---------------------
MUC Recall: 65.59%, Precision: 68.37%, F1: 66.95%
BCUB Recall: 51.91%, Precision: 59.38%, F1: 55.40%
CEAFE Recall: 51.56%, Precision: 51.93%, F1: 51.74%

CoNLL average F1: 58.03

CoNLL'12 English test:
----------------------
MUC Recall: 65.61%, Precision: 67.74%, F1: 66.66%
BCUB Recall: 50.23%, Precision: 57.24%, F1: 53.51%
CEAFE Recall: 49.37%, Precision: 49.21%, F1: 49.29%

CoNLL average F1: 56.48

They differ from Martschat (2013) since a different scorer is used, and 
preprocessing and features are slightly different.

### Reference ###

If you use this system in your research, please cite the following publication:

Sebastian Martschat (2013).
Multigraph Clustering for Unsupervised Coreference Resolution.
In Proceedings of the Student Research Workshop at the 51st Annual Meeting 
of the Association for Computational Linguistics, Sofia, Bulgaria, 5-7 August 
2013, pages 81-88.
http://aclweb.org/anthology-new/P/P13/P13-3012.pdf
