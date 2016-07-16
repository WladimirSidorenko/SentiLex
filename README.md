# Dictionary-, Corpus-, and NWE-based Generation of Sentiment Lexicons

This project provides scripts and executable files for generating
sentiment lexicons using `GermaNet` (German equivalent of `WordNet`),
corpora, and neural word embeddings.

## Examples

### Blair-Goldensohn (2008)

For generating a sentiment lexicon using the method of
[Blair-Goldensohn et
al. (2008)](http://www.australianscience.com.au/research/google/34368.pdf),
use the following command:

```shell

./scripts/generate_lexicon.py blair-goldensohn \
 --ext-syn-rels --seed-pos=adj \
 --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
 data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0/

```

### Hassan (2010)

To generate a sentiment lexicon using the method of
[Hassan and Radev (2010)](https://www.aclweb.org/anthology/P/P10/P10-1041.pdf),
you should use the following command:

```shell

./scripts/generate_lexicon.py awdallah --ext-syn-rels \
--seed-pos=adj --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0/

```

### Esuli and Sebastiani (2006)

For generating a sentiment lexicon using the method of
[Esuli and Sebastiani (2006)](http://ontotext.fbk.eu/Publications/sentiWN-TR.pdf),
you should use the following command:

```shell

./scripts/generate_lexicon.py esuli \
--form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/turney_littman_gi_seedset.txt data/GermaNet_v9.0

```

### Hu and Liu (2004)

To generate a sentiment lexicon using the `SentiWordNet` method of
[Hu and Liu (2004)](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf),
you should envoke the following command:

```shell

./scripts/generate_lexicon.py hu \
--form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0

```

Evaluation
----------

You can evaluate the resulting sentiment lexicon on the
[PotTS](https://github.com/WladimirSidorenko/PotTS) dataset by using
the following command and providing a valid path to the downloaded
corpus files:

```shell

./scripts/evaluate.py -l data/form2lemma.txt \
	data/es/esuli_sebastiani_tlg_seedset.txt \
	${PATH_TO_PotTS}/corpus/basedata/ ${PATH_TO_PotTS}/corpus/annotator-2/markables/

```
