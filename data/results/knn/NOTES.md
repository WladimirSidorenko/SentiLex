Description of Files Located in this Directory
==============================================

Files:
------

All files were obtained while being at revision
`782225a084403dbab4eb4ecf6adcfedb143a3429` of the SentiLex project.

1. `knn.word2vec.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=1 -L -M ../data/vectors/vectors.word2vec ../data/seeds/turney_littman_2003.txt > ../data/results/knn/knn.word2vec.L.M.turney-littman-seedset
```

1.1. `knn.word2vec.L.M.turney-littman-seedset.5120`

Command (from ~/Projects/SentiLex/build):
```
# determine the optimal cardinality of the `knn.word2vec.L.M.turney-littman-seedset` file

./scripts/eval_cardinality data/results/knn/knn.word2vec.L.M.turney-littman-seedset
./scripts/analyze_eval_log knn.eval.log

# Obtain the first 5,120 entries
head -5120 data/results/knn/knn.word2vec.L.M.turney-littman-seedset > \
data/results/knn/knn.word2vec.L.M.turney-littman-seedset.5120
```

2. `knn.w2v.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=1 -L -M ../data/vectors/vectors.word2vec ../data/seeds/turney_littman_2003.txt > ../data/results/knn/knn.w2v.L.M.turney-littman-seedset
```

3. `knn.ts.emoseedset.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=1 -L -M ../data/vectors/vectors.ts.emoseedset ../data/seeds/turney_littman_2003.txt > ../data/results/knn/knn.ts.emoseedset.L.M.turney-littman-seedset
```
