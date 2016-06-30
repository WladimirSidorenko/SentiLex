Dictionary-, Corpus-, and NWE-based Generation of Sentiment Lexicons
====================================================================

This project provides scripts and executable files for generating
sentiment lexicons using `GermaNet` (German equivalent of `WordNet`),
corpora, and neural word embeddings.

Examples
--------

To generate a sentiment lexicon using the `SentiWordNet` method of
`Esuli and Sebastiani, 2006`_, you should envoke the following
command:

```shell

./scripts/generate_lexicon.py esuli
--form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt
data/turney_littman_seedset.txt data/GermaNet_v9.0/

```

.. _`Esuli and Sebastiani, 2006`: http://ontotext.fbk.eu/Publications/sentiWN-TR.pdf
