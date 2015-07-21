# Running cort's multigraph system

**cort** ships with a deterministic coreference resolution system based on
multigraph clustering. The input must follow  [the 
format from the CoNLL shared tasks on coreference resolution](http://conll.cemantix.org/2012/data.html).

To run the multigraph system, use

```shell
run-multigraph -in my_data.data -out out.data
```

With the optional argument `-ante`, antecedent decisions are also written to a 
file:

```shell
run-multigraph -in my_data.data -out out.data -ante antecedents_out.data
```