#!/usr/bin/awk -f

##################################################################
BEGIN {
    if (ARGC != 3) {
	printf("Incorrect number of arguments.") > "/dev/stderr"
	exit 1
    }

    # obtain words to fetch
    ifile = ARGV[1]
    while ((ret = (getline < ifile)) > 0) {
	if (!NF)
	    continue

        gsub(/[[:space:]]+$/, "")
        gsub(/^[[:space:]]+/, "")
        if ($0 in WORD2SEEK)
          continue

	WORD2SEEK[$0] = ""
        ++w_cnt
    }
    ARGV[1] = ""
}

##################################################################
NF && FNR > 1 && ($1 in WORD2SEEK) {
  print $0
  if (--w_cnt <= 0)
    exit
}
