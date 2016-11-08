# Sentiment Lexicon Generation Suite

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This project provides executable files and scripts for generating
sentiment lexicons from `GermaNet` (a German equivalent of the English
`WordNet`), raw text corpora, and neural word embeddings.

## Building

For generating a sentiment lexcion from pre-trained neural word
embeddings, you first need to compile the C++ code by running the
following commands:

```shell
cd build/
cmake ../
make
```

Afterwards, an executable called `vec2dic` wil apper in
the subdirectory `build/bin`.  You can exectute this file by envoking:

```shell
./build/vec2dic [OPTIONS] --type=TYPE VECTOR_FILE SEED_FILE
```

where the `TYPE` argument (an integer from zero to three) will
determine the algorithm to use for inducing a sentiment lexicon,
`VECTORE_FILE` denotes a path to a text file with pre-trained
`word2vec` embeddings (note that the file should be in the raw text
format with space separated values), and `SEED_FILE`.  We currently
support the following types of algorithms:
- 0 -- nearest centroids (default);
- 1 -- KNN;
- 2 -- PCA;
- 3 -- linear projection.

## Examples

In addition to the C++ executables, we also provide several
reimplementations of popular alternative approaches which generate
sentiment lexcions from lexical taxonomies (e.g., `GermaNet`) or raw
unlabeled text corpora.  Please note that in order to use
dictionary-based methods, you need to download
[GermaNet](http://www.sfs.uni-tuebingen.de/GermaNet/), which is not
included here by default due to license restrictions, and place its
files in the directory `data/GermaNet_v9.0/`.  For corpus-based
algorithms, you need to provide a pre-lemmatized corpus in the format
similar to the one used in `data/snapshot_corpus_data/example.txt`.
Alternatively, for the method of Takamura et al. (2005), you need to
provide a list of coordinately conjoined pairs similar to the one
provided in `data/corpus/cc_light.txt`.

Below, you can find a short summary and command examples of the
provided systems.

### Hu and Liu (2004)

For generating a sentiment lexicon with the method of [Hu and Liu
(2004)](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf),
you should envoke the following command:

```shell

./scripts/generate_lexicon.py hu-liu \
 --ext-syn-rels --seed-pos=adj \
--form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0

```

### Blair-Goldensohn (2008)

If you want to generate a sentiment lexicon using the method of
[Blair-Goldensohn et
al. (2008)](http://www.australianscience.com.au/research/google/34368.pdf),
you should envoke the following command:

```shell

./scripts/generate_lexicon.py blair-goldensohn \
 --ext-syn-rels --seed-pos=adj \
 --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
 data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0/

```

### Kim-Hovy (2004)

For generating a sentiment lexicon with the method of [Kim and Hovy,
(2004)](https://www.cs.cmu.edu/~hovy/papers/04Coling-opinion-valences.pdf),
use the following command:

```shell

./scripts/generate_lexicon.py kim-hovy \
 --ext-syn-rels --seed-pos=adj \
 --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
 data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0/

```

### Takamura et al. (2005)

To generate a sentiment lexicon with the method of
[Takamura et al. (2005)](http://delivery.acm.org/10.1145/1220000/1219857/p133-takamura.pdf?ip=77.179.90.234&id=1219857&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=830128042&CFTOKEN=27668650&__acm__=1472910085_b90c7157c9757c8c7e7ccacc73a39bb5),
use the following command instead (note that the file
`data/corpus/cc.txt` is not included in this repository due to its big
size):

```shell

./scripts/generate_lexicon.py takamura \
    --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
    data/seeds/turney_littman_2003.txt data/GermaNet_v9.0/ data/corpus/cc.txt -1

```

### Esuli and Sebastiani (2006)

For generating a sentiment lexicon using the `SentiWordNet` method of
[Esuli and Sebastiani (2006)](http://ontotext.fbk.eu/Publications/sentiWN-TR.pdf),
you should use the following command:

```shell

./scripts/generate_lexicon.py esuli --ext-syn-rels \
--seed-pos=adj --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0

```

### Rao and Ravichandran (2009)

In order to generate a sentiment lexicon with the min-cut approach of
[Rao and Ravichandran (2009)](http://www.aclweb.org/anthology/E09-1077),
use the below command:

```shell

./scripts/generate_lexicon.py rao-min-cut --ext-syn-rels \
--seed-pos=adj --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0

```

If you want to test the label propagation algorithm described by these
authors, you should specify the following arguments:

```shell

./scripts/generate_lexicon.py rao-lbl-prop --ext-syn-rels \
--seed-pos=adj --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0

```

### Awdallah and Radev (2010)

To generate a sentiment lexicon using the method of
[Awdallah and Radev (2010)](https://www.aclweb.org/anthology/P/P10/P10-1041.pdf),
you should use the following command:

```shell

./scripts/generate_lexicon.py awdallah --ext-syn-rels \
--seed-pos=adj --form2lemma=data/GermaNet_v9.0/gn_form2lemma.txt \
data/seeds/hu_liu_seedset.txt data/GermaNet_v9.0/

```

### Velikovich et al. (2010)

For generating a sentiment lexicon using the algorithm of
[Velikovich et al. (2010)](http://www.aclweb.org/anthology/N10-1119),
you can use the following command:

```shell

./scripts/generate_lexicon.py velikovich \
data/seeds/hu_liu_seedset.txt -1 data/snapshot_corpus_data/example.txt

```

### Kiritchenko et al. (2014)

In order to generate a sentiment lexicon using the system of
[Kiritchenko et al. (2014)](https://www.jair.org/media/4272/live-4272-8102-jair.pdf),
you should use the following command:

```shell

./scripts/generate_lexicon.py kiritchenko \
data/seeds/hu_liu_seedset.txt -1 data/snapshot_corpus_data/example.txt

```

### Severyn and Moschitti (2014)

For generating a sentiment lexicon using the approach of
[Severyn and Moschitti (2014)](http://www.aclweb.org/anthology/N15-1159),
you should use the following command:

```shell

./scripts/generate_lexicon.py severyn \
data/seeds/hu_liu_seedset.txt -1 data/snapshot_corpus_data/example.txt

```

## Evaluation

You can evaluate the resulting sentiment lexicon on the
[PotTS](https://github.com/WladimirSidorenko/PotTS) dataset by using
the following command and providing a valid path to the downloaded
corpus data:

```shell

./scripts/evaluate.py -l data/form2lemma.txt \
	data/results/esuli-sebastiani/esuli-sebastiani.ext-syn-rels.turney-littman-seedset.txt \
	${PATH_TO_PotTS}/corpus/basedata/ ${PATH_TO_PotTS}/corpus/annotator-2/markables/

```
