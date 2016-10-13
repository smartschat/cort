# LEA Coreference Scorer

Implementation of the **LEA** coreference evaluation metric integrated into the CoNLL official scorer v8.01.

## Description

LEA is a Link-Based Entity-Aware metric that is designed to overcome the shortcomings of the previous evaluation metrics.
For each entity, **LEA** considers how important the entity is and how well it is resolved, i.e. importance(entity) * resolution-score(entity).

The number of unique links in an entity with "n" mentions is link(entity)=n*(n-1)/2.
The resolution score of an entity is computed as the fraction of correctly resolved coreference links.

In the provided implementation, we consider the size of an entity as a measure of importance, i.e. importance(entity)=|entity|.
Therefore, the more prominent entities of the text get higher importance values.
However, according to the end-task or domain used, one can choose other importance measures based on other factors like the entity type or the included mention types.

## Usage

**LEA** is integrated into the official CoNLL scorer v8.01 available at http://conll.github.io/reference-coreference-scorers.  
The usage of the official CoNLL scorer (Pradhan et al., 2014) is as follows:


     perl scorer.pl <metric> <key> <response> [<document-id>]


     <metric>: the metric desired to score the results. one of the following values:

     muc: MUCScorer (Vilain et al, 1995)
     bcub: B-Cubed (Bagga and Baldwin, 1998)
     ceafm: CEAF (Luo et al., 2005) using mention-based similarity
     ceafe: CEAF (Luo et al., 2005) using entity-based similarity
     blanc: BLANC (Luo et al., 2014) BLANC metric for gold and predicted mentions
     lea: LEA (Moosavi and Strube, 2016)
     all: uses all the metrics to score

     <key>: file with expected coreference chains in CoNLL-2011/2012 format

     <response>: file with output of coreference system (CoNLL-2011/2012 format)
 
     <document-id>: optional. The name of the document to score. If name is not
                    given, all the documents in the dataset will be scored. If given
                    name is "none" then all the documents are scored but only total
                    results are shown.

##References

    Nafise Sadat Moosavi and Michael Strube. 2016. 
    Which Coreference Evaluation Metric Do You Trust? A Proposal for a Link-based Entity Aware Metric. 
    In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics.

    Sameer Pradhan, Xiaoqiang Luo, Marta Recasens, Eduard Hovy, Vincent Ng, and Michael Strube. 2014. 
    Scoring coreference partitions of predicted mentions: A reference implementation. 
    In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),
    Baltimore, Md., 22–27 June 2014, pages 30–35.

