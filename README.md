# cort

__cort__ is a <b>co</b>reference <b>r</b>esolution <b>t</b>oolkit. It consists
of two parts: the *coreference resolution* component implements a framework for 
coreference resolution based on latent variables, which allows you to rapidly 
devise approaches to coreference resolution, while the *error analysis* component 
provides extensive functionality for analyzing and visualizing errors made by 
coreference resolution systems.

If you have any questions or comments, drop me an e-mail at 
[sebastian.martschat@gmail.com](mailto:sebastian.martschat@gmail.com).

## Documentation

* <a href="COREFERENCE.md">coreference resolution with cort</a>
* <a href="ANALYSIS.md">error analysis with cort</a>
* <a href="MULTIGRAPH.md">running the multigraph system</a>

## Installation

__cort__ is available on PyPi. You can install it via

```
pip install cort
```
Dependencies (automatically installed by pip) are 
[nltk](http://www.nltk.org/), [numpy](http://www.numpy.org/), 
[matplotlib](http://matplotlib.org), 
[mmh3](https://pypi.python.org/pypi/mmh3),
[cython](http://cython.org/),
[future](https://pypi.python.org/pypi/future),
[jpype](https://pypi.python.org/pypi/jpype1) and
[beautifulsoup](https://pypi.python.org/pypi/beautifulsoup4). It ships with 
[PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies),
[stanford_corenlp_pywrapper](https://github.com/brendano/stanford_corenlp_pywrapper)
and [the reference implementation of the CoNLL scorer]
(https://github.com/conll/reference-coreference-scorers).

__cort__ is written for use on Linux with either Python 2.7+ or Python 3.3+.
However, I recommend running __cort__ with Python 3, since the Python 3 version is 
more efficient.

## References

Sebastian Martschat and Michael Strube (2015). **Latent Structures for 
Coreference Resolution**. To appear in *Transactions of the Association for 
Computational Linguistics*.
[PDF](http://smartschat.de/papers/tacl15.pdf)

Sebastian Martschat, Patrick Claus and Michael Strube (2015). **Plug Latent 
Structures and Play Coreference Resolution**. To appear in *Proceedings of 
the 53rd Annual Meeting of the Association for Computational Linguistics: 
System Demonstrations*, Beijing, China, 26-31 July 2015.
[PDF](https://smartschat.de/papers/acl15demo.pdf)

Sebastian Martschat, Thierry Göckel and Michael Strube (2015). **Analyzing and 
Visualizing Coreference Resolution Errors**. In *Proceedings of the 2015 
Conference of the North American Chapter of the Association for Computational 
Linguistics: Demonstrations*, Denver, Colorado, USA, 31 May-5 June 2015,
pages 6-10.  
[PDF](https://aclweb.org/anthology/N/N15/N15-3002.pdf)

Sebastian Martschat and Michael Strube (2014). **Recall Error Analysis for 
Coreference Resolution**. In *Proceedings of the 2014 Conference on Empirical 
Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 25-29 October 
2014, pages 2070-2081.  
[PDF](http://aclweb.org/anthology/D/D14/D14-1221.pdf)

Sebastian Martschat (2013). **Multigraph Clustering for Unsupervised 
Coreference Resolution**. In *Proceedings of the Student Research Workshop 
at the 51st Annual Meeting of the Association for Computational Linguistics*, 
Sofia, Bulgaria, 5-7 August 2013, pages 81-88.  
[PDF](http://aclweb.org/anthology/P/P13/P13-3012.pdf)

If you use the error analysis component in your research, please cite the
EMNLP'14 paper. If you use the coreference component in your research, please
cite the TACL paper. If you use the multigraph system, please cite the 
ACL'13-SRW paper.

## Changelog

__Tuesday, 21 July 2015__   
Updated to status of TACL paper.

__Wednesday, 3 June 2015__  
Improvements to visualization (mention highlighting and scrolling).

__Monday, 1 June 2015__  
Fixed a bug in mention highlighting for visualization.

__Sunday, 31 May 2015__  
Updated to status of NAACL'15 demo paper.

__Wednesday, 13 May 2015__  
Fixed another bug in the documentation regarding format of antecedent data.

__Tuesday, 3 February 2015__  
Fixed a bug in the documentation: part no. in antecedent file must be with trailing 0s.

__Thursday, 30 October 2014__  
Fixed data structure bug in documents.py. The results from the paper are not affected by this bug.

__Wednesday, 22 October 2014__  
Initial release.
