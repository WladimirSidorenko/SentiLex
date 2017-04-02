Description of Files Located in this Directory
==============================================

Files:
------

All files were obtained while being at revision
`782225a084403dbab4eb4ecf6adcfedb143a3429` of the SentiLex project.

1. `linproj.word2vec.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=3 -L -M ../data/vectors/vectors.word2vec ../data/seeds/turney_littman_2003.txt > ../data/results/linproj/linproj.word2vec.L.M.turney-littman-seedset
```

1.1. `linproj.word2vec.L.M.turney-littman-seedset.40`

Command (from ~/Projects/SentiLex/build):
```
# determine the optimal cardinality of the `linproj.word2vec.L.M.turney-littman-seedset` file

./scripts/eval_cardinality data/results/linproj/linproj.word2vec.L.M.turney-littman-seedset
./scripts/analyze_eval_log linproj.eval.log

# Obtain the first 40 entries
head -40 data/results/linproj/linproj.word2vec.L.M.turney-littman-seedset > \
data/results/linproj/linproj.word2vec.L.M.turney-littman-seedset.40
```

2. `linproj.w2v.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=3 -L -M ../data/vectors/vectors.word2vec ../data/seeds/turney_littman_2003.txt > ../data/results/linproj/linproj.w2v.L.M.turney-littman-seedset
```

2.1. `linproj.w2v.L.M.turney-littman-seedset.5160`

Command (from ~/Projects/SentiLex/build):
```
# determine the optimal cardinality of the `linproj.w2v.L.M.turney-littman-seedset` file

./scripts/eval_cardinality linproj.w2v.L.M.turney-littman-seedset
./scripts/analyze_eval_log linproj.eval.log

# Obtain the first 40 entries
head -5160 data/results/linproj/linproj.w2v.L.M.turney-littman-seedset > \
data/results/linproj/linproj.w2v.L.M.turney-littman-seedset.40
```

3. `linproj.ts.emoseedset.L.M.turney-littman-seedset`

Command (from ~/Projects/SentiLex/build):
```
./bin/vec2dic --type=3 -L -M ../data/vectors/vectors.ts.emoseedset ../data/seeds/turney_littman_2003.txt > ../data/results/linproj/linproj.ts.emoseedset.L.M.turney-littman-seedset
```
