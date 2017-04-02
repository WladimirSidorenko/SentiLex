Description of Files Located in this Directory
==============================================

Files:
------

All files were obtained while being at revision
`782225a084403dbab4eb4ecf6adcfedb143a3429` of the SentiLex project.

1. `ncentroids.word2vec.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=0 -L -M ../data/vectors/vectors.word2vec ../data/seeds/turney_littman_2003.txt > ../data/results/ncentroids/ncentroids.word2vec.L.M.turney-littman-seedset
```

1.1. `ncentroids.word2vec.L.M.turney-littman-seedset.60`

Command (from ~/Projects/SentiLex/build):
```
# determine the optimal cardinality of the `ncentroids.word2vec.L.M.turney-littman-seedset` file

./scripts/eval_cardinality data/results/ncentroids/ncentroids.word2vec.L.M.turney-littman-seedset
./scripts/analyze_eval_log ncentroids.eval.log

# Obtain the first 60 entries
head -60 data/results/ncentroids/ncentroids.word2vec.L.M.turney-littman-seedset > \
data/results/ncentroids/ncentroids.word2vec.L.M.turney-littman-seedset.60
```

2. `ncentroids.w2v.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=0 -L -M ../data/vectors/vectors.word2vec ../data/seeds/turney_littman_2003.txt > ../data/results/ncentroids/ncentroids.w2v.L.M.turney-littman-seedset
```

3. `ncentroids.ts.emoseedset.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=0 -L -M ../data/vectors/vectors.ts.emoseedset ../data/seeds/turney_littman_2003.txt > ../data/results/ncentroids/ncentroids.ts.emoseedset.L.M.turney-littman-seedset
```
