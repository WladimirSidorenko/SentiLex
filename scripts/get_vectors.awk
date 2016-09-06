#!/usr/bin/awk -f

##################################################################
BEGIN {
    if (ARGC != 4) {
	printf("Incorrect number of arguments.") > "/sys/stderr"
	exit 1
    }

    ret = 0
    WCL = 0 + ARGV[1]
    # obtain words to fetch
    ifile = ARGV[2]
    while ((ret = (getline < ifile)) > 0) {
	if (!NF)
	    continue
	else if (NF != 2) {
	    printf("Incorrect line format: %s", $0) > "/sys/stderr"
	    exit 2
	}
        $1 = lower($1)
        if ($1 in WORD2SEEK)
          continue

	WORD2SEEK[$1] = ""
        ++w_cnt
    }
    ARGV[1] = ARGV[2] = ""
    # obtain random line numbers to fetch
    srand()
    wcl = WCL > 1? WCL - 1: 0
    for (i = 1; i < WCL; ++i) {
	randarr[i] = rand() * wcl + 1
    }
    asorti(randarr, tmparr)
    wcl = WCL / 500
    for (i = 1; i < wcl; ++i) {
	FNR2SEEK[tmparr[i]] = ""
    }
}

##################################################################
NF && FNR > 1 && ($1 in WORD2SEEK || FNR in FNR2SEEK)
