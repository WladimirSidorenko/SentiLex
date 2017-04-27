#!/usr/bin/awk -f

##################################################################
# Methods
function usage(ecode) {
    printf(\
	"Script for extracting relevant word vectors from embeddings file.\n\
\n\
USAGE:\n\
%s MIN_TERMS RELEVANT_TOKENS VECTOR_FILE\n\n\
OPTIONS:\n\
-h|--help    print this screen and exit\n", ARGV[0]) > "/dev/stderr"
    exit ecode
}

function read_relevant_tokens(words2seek, fname) {
    # read relevant tokens
    ret = w_cnt = 0
    while ((ret = (getline < fname)) > 0) {
	if (!NF)
	    continue
        $0 = tolower($0)
        if ($0 in words2seek)
	    continue

	words2seek[$0] = ""
        ++w_cnt
    }
    close(fname)
    if (ret != 0) {
	printf("Error while reading token file.\n") > "/dev/stderr"
	exit 2
    }
    return w_cnt
}

function read_vectors(line_stat, lines2print, words2seek, fname) {
    # read relevant tokens
    fnr = decl_seen = ret = knwn_cnt = unk_cnt = 0
    while ((ret = (getline < fname)) > 0) {
	++fnr
	# remember line numbers which can be used for sampling
	if (!NF) {
	    continue
	} else if (!decl_seen) {
	    decl_seen = 1
	    continue
	} else if (tolower($1) in words2seek) {
	    ++knwn_cnt
	    continue
	} else {
	    lines2print[++unk_cnt] = fnr
	}
    }
    close(fname)
    if (ret != 0) {
	printf("Error while reading vector file.\n") > "/dev/stderr"
	exit 2
    }
    line_stat[0] = knwn_cnt
    line_stat[1] = unk_cnt
}

function output_vectors(N, lines2print, words2seek, fname) {
    # read relevant tokens
    fnr = decl_seen = 0
    while ((ret = (getline < fname)) > 0) {
	++fnr
	# remember line numbers which can be used for sampling
	if (!NF) {
	    continue
	} else if (!decl_seen) {
	    decl_seen = 1
	    # output new declaration
	    printf("%d %s\n", N, $2)
	    decl_seen = 1
	    continue
	} else if (tolower($1) in words2seek || fnr in lines2print) {
	    printf("%s\n", $0)
	}
    }
    close(fname)
    if (ret != 0) {
	printf("Error while reading vector file.\n") > "/dev/stderr"
	exit 3
    }
}

##################################################################
# BEGIN
BEGIN {
    # parse options
    arg = ""
    argc = ARGC
    for (i = 1; i < ARGC; ++i) {
	arg = ARGV[i]
	if (arg == "-h" || arg == "--help") {
	    usage(0)
	} else if (arg == "--") {
	    ARGV[i] = ""
	    --argc
	    break
	} else if (arg ~ /^-/) {
	    printf("Unknown option: '%s'.  Type --help to see usage.\n",
		   arg) > "/dev/stderr"
	    exit 1
	} else {
	    break
	}
    }
    if (argc != 4) {
	printf("Incorrect number of arguments.  Type --help to see usage.\n") > "/dev/stderr"
	exit 1
    }

    min_N = ARGV[ARGC - 3]
    w_cnt = read_relevant_tokens(words2seek, ARGV[ARGC - 2])
    ARGV[ARGC - 3] = ARGV[ARGC - 2] = ""

    # obtain random line numbers to fetch
    vector_fname = ARGV[ARGC - 1]
    ARGV[ARGC - 1] = ""
    # remember line numbers for random sampling
    read_vectors(line_stat, unk_lines, words2seek, vector_fname)
    knwn_cnt = line_stat[0]
    unk_cnt = line_stat[1]
    # randomly shuffle line numbers
    unk_needed = min_N - knwn_cnt
    if (unk_needed < unk_cnt) {
	srand()
	for (i = 1; i <= unk_cnt; ++i) {
	    randarr[i] = rand()
	}
    }
    asorti(randarr, line_indices)
    for (i = 1; i <= unk_needed; ++i) {
	lines2print[unk_lines[line_indices[i]]] = ""
    }
    # re-iterate over the vector file and output embeddings of
    # relevant tokens and randomly chosen vectors
    output_vectors(unk_needed < 0? knwn_cnt: knwn_cnt + unk_needed, \
		   lines2print, words2seek, vector_fname)
}
