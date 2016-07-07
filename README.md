# Dictionary-, Corpus-, and NWE-based Generation of Sentiment Lexicons

This project provides scripts and executable files for generating
sentiment lexicons using `GermaNet` (German equivalent of `WordNet`),
corpora, and neural word embeddings.

## Examples

### Hu and Liu (2004)

For generating a sentiment lexicon using the method of
[Hu and Liu (2004)](), you should use the following command:

```shell

./scripts/generate_lexicon.py hu \
--form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/turney_littman_gi_seedset.txt data/GermaNet_v9.0

```

### Esuli and Sebastiani (2004)

To generate a sentiment lexicon using the `SentiWordNet` method of
[Esuli and Sebastiani (2006)](http://ontotext.fbk.eu/Publications/sentiWN-TR.pdf),
you should envoke the following command:

```shell

./scripts/generate_lexicon.py esuli \
--form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/turney_littman_gi_seedset.txt data/GermaNet_v9.0

```

Evaluation
----------

You can evaluate the resulting sentiment lexicon by issuing the
following command:

```shell

./scripts/evaluate.py -l data/form2lemma.txt \
	data/es/esuli_sebastiani_tlg_seedset.txt \
	../PotTS/corpus/basedata/ ../PotTS/corpus/annotator-2/markables/

```
