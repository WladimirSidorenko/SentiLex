/** @file vec2dic.cpp
 *
 *  @brief Generate sentiment lexicons from neural word embeddings.
 *
 *  This file provides main method for generating sentiment lexicons
 *  on the basis of previously computed neural word embeddings.
 */

//////////////
// Includes //
//////////////
#include "src/vec2dic/expansion.h"
#include "src/vec2dic/optparse.h"

#include <cctype>         // std::isspace()
#include <clocale>        // setlocale()
#include <cmath>          // sqrt(), fabs()
#include <cstdio>         // sscanf()
#include <cstdlib>        // std::exit(), std::strtoul()
#include <cstring>        // strcmp(), strlen()

#include <algorithm>
#include <armadillo>      // arma::mat
#include <fstream>        // std::ifstream
#include <functional>
#include <iostream>       // std::cerr, std::cout
#include <locale>
#include <stdexcept>      // std::domain_error()
#include <string>         // std::string
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::make_pair

/////////////
// Classes //
/////////////

/**
 * Type of algorithm to use for lexicon expansion.
 */
enum class ExpansionType: int {
  NC_CLUSTERING = 0,          // Nearest centroids algorithm
    KNN_CLUSTERING,           // K-nearest neighbors
    PCA_CLUSTERING,           // Proincipal component analysis
    PRJ_CLUSTERING,           // Projection-based clustering
    MAX_SENTINEL              // Unused type that serves as a sentinel
    };

// forward declaration of `usage()` method
static void usage(int a_ret = EXIT_SUCCESS);

/**
 * Custom option handler
 */
class Option: public optparse {
public:
  // Members
  /// input file containing seed polarity terms
  std::ifstream m_seedfile {};
  /// default number of nearest neighbors to consider by the KNN algorithm
  int knn = 5;
  /// maximum number of new terms to extract (-1 means all new terms)
  int n_terms = -1;
  /// learning rate for gradient methods
  double alpha = DFLT_ALPHA;
  /// minimum required improvement for gradient methods
  double delta = DFLT_DELTA;
  /// maximum number of gradient updates
  unsigned long max_iters = MAX_ITERS;
  /// elongation coefficient for vector lengths (useful for linear transformation)
  double coefficient = 1.;
  /// do not normalize length of the vectors
  bool no_length_normalize = false;
  /// do not center means of the vectors
  bool no_mean_normalize = false;
  /// algorithm to use for expansion
  ExpansionType etype = ExpansionType::NC_CLUSTERING;

  Option() {}

  BEGIN_OPTION_MAP_INLINE()
  ON_OPTION(SHORTOPT('L') || LONGOPT("no-length-normalizion"))
  no_length_normalize = true;

  ON_OPTION(SHORTOPT('M') || LONGOPT("no-mean-normalizion"))
  no_mean_normalize = true;

  ON_OPTION_WITH_ARG(SHORTOPT('a') || LONGOPT("alpha"))
  alpha = std::atof(arg);

  ON_OPTION_WITH_ARG(SHORTOPT('c') || LONGOPT("coefficient"))
  coefficient = std::atof(arg);

  ON_OPTION_WITH_ARG(SHORTOPT('d') || LONGOPT("delta"))
  delta = std::atof(arg);

  ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
  usage();

  ON_OPTION_WITH_ARG(SHORTOPT('i') || LONGOPT("max-iterations"))
  max_iters = std::strtoul(arg, nullptr, 10);

  ON_OPTION_WITH_ARG(SHORTOPT('k') || LONGOPT("k-nearest-neighbors"))
  knn = std::atoi(arg);
  if (knn < 1)
    throw optparse::invalid_value("k-nearest-neighbors should be >= 1");

  ON_OPTION_WITH_ARG(SHORTOPT('n') || LONGOPT("n-terms"))
  n_terms = std::atoi(arg);

  ON_OPTION_WITH_ARG(SHORTOPT('t') || LONGOPT("type"))
  int itype = std::atoi(arg);
  if (itype < 0 || itype >= static_cast<int>(ExpansionType::MAX_SENTINEL))
    throw invalid_value("Invalid type of expansion algorithm.");

  etype = static_cast<ExpansionType>(itype);

  END_OPTION_MAP()
};

/// Pair of c_string and polarity
typedef struct WP {
  /// polar word
  const char *m_word = nullptr;
  /// polarity class of the word
  Polarity m_polarity = Polarity::NEUTRAL;
  /// polarity score of the word
  dist_t m_score = 0.;

  /// copy constructor
  WP(const char *a_word, const Polarity a_polarity,
     dist_t a_score):
    m_word(a_word), m_polarity(a_polarity), m_score(fabs(a_score))
  {}

  /// copy constructor
  WP(const char *a_word, const ps_t *a_ps):
    m_word(a_word), m_polarity(a_ps->first), m_score(fabs(a_ps->second))
  {}
} wp_t;

/// Vector of pairs of c_string and their polarities
typedef std::vector<wp_t> wpv_t;

/////////////////////////////
// Variables and Constants //
/////////////////////////////

/// string representing positive polarity class
const char *positive  = "positive";
/// string representing negative polarity class
const char *negative = "negative";
/// string representing neutral polarity class
const char *neutral = "neutral";

/// Mapping from words to the index of their NWE vector
static w2v_t word2vecid;
/// Mapping from the index of NWE vectors to their respective words
static v2w_t vecid2word;
/// Mapping from word to its polarity
static w2ps_t word2polscore;
/// Matrix of neural word embeddings
static arma::mat NWE;
/// Output debug information
const bool debug = false;

/////////////
// Methods //
/////////////

/**
 * Print usage message and exit
 *
 * @param a_ret - exit code for the program
 *
 * @return \c void
 */
static void usage(int a_ret) {
  std::cerr << "Expand initial seed set of subjective terms by"
      " applying clustering" << std::endl;
  std::cerr << "to neural word embeddings." << std::endl << std::endl;
  std::cerr << "Usage:" << std::endl;
  std::cerr << "vec2dic [OPTIONS] VECTOR_FILE SEED_FILE"
            << std::endl << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "-L|--no-length-normalizion  do not normalize length"
      " of word vectors" << std::endl;
  std::cerr << "-M|--no-mean-normalizion  do not normalize means"
      " of word vectors" << std::endl;
  std::cerr << "-a|--alpha  learning rate for gradient methods"
      " (default " << DFLT_ALPHA << ")" << std::endl;
  std::cerr << "-c|--coefficient  elongate vectors by the"
      " coefficient (useful for linear projection, implies -L)" << std::endl;
  std::cerr << "-d|--delta  learning rate for gradient methods"
      " (default " << DFLT_DELTA << ")" << std::endl;
  std::cerr << "-h|--help  show this screen and exit" << std::endl;
  std::cerr << "-i|--max-iterations  maximum number of gradient"
      " updates (default " << MAX_ITERS << ")" << std::endl;
  std::cerr << "-k|--k-nearest-neighbors  set the number of neighbors"
      " for KNN algorithm" << std::endl;
  std::cerr << "-n|--n-terms  number of terms to extract (default:"
      " -1 (unlimited))" << std::endl;
  std::cerr << "-t|--type  type of expansion algorithm to use:" << std::endl;
  std::cerr << "           (0 - nearest centroids (default), "
      "1 - KNN, 2 - PCA dimension," << std::endl;
  std::cerr << "            3 - linear projection)" << std::endl << std::endl;
  std::cerr << "Exit status:" << std::endl;
  std::cerr << EXIT_SUCCESS << " on sucess, non-" << EXIT_SUCCESS
            << " otherwise" << std::endl;
  std::exit(a_ret);
}

/**
 * Output polar terms in sorted alphabetic order
 *
 * @param a_stream - output stream to use
 * @param a_vecid2polscore - mapping from vector id's to their respective polarities
 *
 * @return \c void
 */
static void output_terms(std::ostream &a_stream,
                         const v2ps_t *a_vecid2polscore) {
  // add new words to `word2polscore` map
  v2w_t::iterator v2w_it;

  for (auto &v2ps : *a_vecid2polscore) {
    // we assume that the word is always found
    v2w_it = vecid2word.find(v2ps.first);
    word2polscore.emplace(v2w_it->second, v2ps.second);
  }

  // populate word/polarity vector
  wpv_t wpv;
  wpv.reserve(word2polscore.size());
  for (auto &w2p : word2polscore) {
    wpv.push_back(WP {w2p.first.c_str(), &w2p.second});
  }
  // sort words
  std::sort(wpv.begin(), wpv.end(), [](const wp_t& wp1, const wp_t& wp2) \
            {return wp1.m_score < wp2.m_score;});

  // output sorted dict to the requested stream
  for (auto &wp : wpv) {
    switch (wp.m_polarity) {
      case Polarity::POSITIVE:
        a_stream << wp.m_word << '\t' << positive << '\t' << wp.m_score;
        break;
      case Polarity::NEGATIVE:
        a_stream << wp.m_word << '\t' << negative << '\t' << wp.m_score;
        break;
      case Polarity::NEUTRAL:
        continue;
    default:
      throw std::domain_error("Unknown polarity type");
    }
    a_stream << std::endl;
  }
}

/**
 * Auxiliary function for removing blanks from the left end of a string
 *
 * @param s - string to be trimmed
 *
 * @return reference to the original string with leading blanks truncated
 */
static inline std::string *ltrim(std::string *s) {
  s->erase(s->begin(),
           std::find_if(s->begin(), s->end(),
                        std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

/**
 * Auxiliary function for removing blanks from the right end of a string
 *
 * @param s - string to be trimmed
 *
 * @return reference to the original string with trailing blanks truncated
 */
static inline std::string *rtrim(std::string *s) {
  s->erase(std::find_if(s->rbegin(), s->rend(),
                        std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
           s->end());
  return s;
}

/**
 * Auxiliary function for removing blanks from both ends of a string
 *
 * @param s - string to be trimmed
 *
 * @return original string with leading and trailing blanks removed
 */
static inline std::string *normalize(std::string *s) {
  // strip leading and trailing whitespaces
  ltrim(rtrim(s));
  // downcase the string
  std::transform(s->begin(), s->end(), s->begin(), ::tolower);;
  return s;
}

/**
 * Perform length-normalization of word vectors
 *
 * @param a_nwe - Armadillo matrix of word vectors (each word vector
 *                is a column in this matrix)
 *
 * @return \c void
 */
static void _length_normalize(arma::mat *a_nwe) {
  dist_t ilength = 0., tmp_j;
  size_t j, n_rows = a_nwe->n_rows;
  for (size_t i = 0; i < a_nwe->n_cols; ++i) {
    ilength = 0.;
    // compute the unnormalized length of the vector
    for (j = 0; j < n_rows; ++j) {
      tmp_j = (*a_nwe)(j, i);
      ilength += tmp_j * tmp_j;
    }
    // normalize the length
    ilength = sqrt(ilength);
    // normalize the vector by dividing it by normalized length
    if (ilength)
      a_nwe->col(i) /= float(ilength);
  }
}

/**
 * Perform mean-normalization of word vectors
 *
 * @param a_nwe - Armadillo matrix of word vectors (each word vector
 *                is a column in this matrix)
 *
 * @return \c void
 */
static void _mean_normalize(arma::mat *a_nwe) {
  arma::vec vmean = arma::mean(*a_nwe, 1);
  arma::vec vstddev = arma::stddev(*a_nwe, 0, 1);

  for (vid_t i = 0; i < vmean.n_rows; ++i) {
    a_nwe->row(i) -= vmean[i];

    if (vstddev[i])
      a_nwe->row(i) /= vstddev[i];
  }
}

/**
 * Read NWE vectors for words.
 *
 * @param a_ret - exit code for the program
 * @param a_option - pointer to user's options
 *
 * @return \c 0 on success, non-\c 0 otherwise
 */
static int read_vectors(const char *a_fname, const Option *a_option) {
  float iwght;
  const char *cline;
  std::string iline;
  size_t space_pos, tab_pos;
  int nchars;
  vid_t mrows = 0, ncolumns = 0, icol = 0, irow = 0;
  const int coefficient = a_option->coefficient;
  const bool no_length_normalize = a_option->no_length_normalize;
  const bool no_mean_normalize = a_option->no_mean_normalize;
  std::cerr << "Reading word vectors ... ";

  std::ifstream is(a_fname);
  if (!is) {
    std::cerr << "Cannot open file " << a_fname << std::endl;
    goto error_exit;
  }
  // skip empty lines at the beginning of file
  while (std::getline(is, iline) && iline.empty()) {}
  // initialize matrix (columns represent words, rows are coordinates)
  if (sscanf(iline.c_str(), "%llu %llu", &ncolumns, &mrows) != 2) {
    std::cerr << "Incorrect declaration line format: '"
              << iline.c_str() << std::endl;
    goto error_exit;
  }

  // allocate space for map and matrix
  word2vecid.reserve(ncolumns); vecid2word.reserve(ncolumns);
  NWE.set_size(mrows, ncolumns);

  while (icol < ncolumns && std::getline(is, iline)) {
    tab_pos = iline.find_first_of('\t');
    space_pos = iline.find_first_of(' ');
    if (tab_pos < space_pos)
      space_pos = tab_pos;

    while (space_pos > 0 && std::isspace(iline[space_pos])) {--space_pos;}
    if (space_pos == 0 && std::isspace(iline[space_pos])) {
      std::cerr << "Incorrect line format (missing word): "
                << iline << std::endl;
      goto error_exit;
    }
    ++space_pos;
    word2vecid.emplace(std::move(iline.substr(0, space_pos)), std::move(icol));
    vecid2word.emplace(std::move(icol), std::move(iline.substr(0, space_pos)));

    cline = &(iline.c_str()[space_pos]);
    for (irow = 0; irow < mrows
           && sscanf(cline, " %f%n", &iwght, &nchars) == 1; ++irow) {
      NWE(irow, icol) = iwght;
      cline += nchars;
    }
    if (irow != mrows) {
      std::cerr << "Incorrect line format (declared vector size " << mrows
                << " differs from the actual size " << irow << "):\n"
		<< iline << std::endl;
      goto error_exit;
    }
    ++icol;
  }

  if (!is.eof() && is.fail()) {
    std::cerr << "Failed to read vector file " << a_fname << std::endl;
    goto error_exit;
  }
  if (icol != ncolumns) {
    std::cerr << "Incorrect file format: declared number of vectors "
              << ncolumns << " differs from the actual number "
              << icol << std::endl;
    goto error_exit;
  }
  is.close();
  // normalize lengths of word vectors
  if (coefficient != 1)
    NWE *= coefficient;

  // normalize lengths of word vectors
  if (!no_length_normalize)
    _length_normalize(&NWE);

  // normalize means of word vectors
  if (!no_mean_normalize)
    _mean_normalize(&NWE);

  std::cerr << "done (read " << mrows << " rows with "
            << ncolumns << " columns)" << std::endl;
  return 0;

 error_exit:
  is.close();            // basic guarantee
  word2vecid.clear();
  vecid2word.clear();
  NWE.reset();
  return 1;
}

/**
 * Read seed set of polarity terms
 *
 * @param a_ret - exit code for the program
 *
 * @return \c 0 on success, non-\c 0 otherwise
 */
static int read_seed_set(const char *a_fname) {
  Polarity ipol;
  std::string iline;
  size_t tab_pos, tab_pos_orig;

  std::cerr << "Reading seed set file ...";

  std::ifstream is(a_fname);
  if (!is) {
    std::cerr << "Cannot open file " << a_fname << std::endl;
    goto error_exit;
  }

  // read input file
  while (std::getline(is, iline)) {
    if (iline.empty() || iline.compare(0, 3, "###") == 0)
      continue;

    // remove leading and trailing whitespaces
    normalize(&iline);
    // find first tab character
    tab_pos = iline.find_first_of('\t');
    tab_pos_orig = tab_pos;
    // skip leading whitespaces
    while (iline[++tab_pos] && std::isspace(iline[tab_pos])) {}
    if (tab_pos == std::string::npos || !iline[tab_pos]) {
      std::cerr << "Incorrect line format (missing polarity): "
                << iline << std::endl;
      goto error_exit;
    }
    // determine polarity class
    if (iline.compare(tab_pos, strlen(positive), positive) == 0) {
      ipol = Polarity::POSITIVE;
    } else if (iline.compare(tab_pos, strlen(negative), negative) == 0) {
      ipol = Polarity::NEGATIVE;
    } else if (iline.compare(tab_pos, strlen(neutral), neutral) == 0) {
      ipol = Polarity::NEUTRAL;
    } else {
      std::cerr << "Unrecognized polarity class at line '"
                << iline.substr(tab_pos) << "'" << std::endl;
      goto error_exit;
    }

    while (tab_pos_orig > 0
           && std::isspace(iline[tab_pos_orig])) {--tab_pos_orig;}
    if (tab_pos_orig == 0 && std::isspace(iline[tab_pos_orig])) {
      std::cerr << "Incorrect line format (missing word): "
                << iline << std::endl;
      goto error_exit;
    }
    ++tab_pos_orig;
    word2polscore.emplace(
        std::move(iline.substr(0, tab_pos_orig)),
        std::make_pair(ipol, 0.));
  }

  if (!is.eof() && is.fail()) {
    std::cerr << "Failed to read seed set file " << a_fname << std::endl;
    goto error_exit;
  }
  is.close();
  std::cerr << "done (read " << word2polscore.size() << " entries)"
            << std::endl;
  return 0;

 error_exit:
  is.close();        // basic guarantee
  word2polscore.clear();
  return 1;
}

//////////
// Main //
//////////

/**
 * Main method for expanding sentiment lexicons
 *
 * @param argc - number of command line arguments
 * @param argv - array of command line arguments
 *
 * @return 0 on success, non-0 otherwise
 */
int main(int argc, char *argv[]) {
  int nargs = 0, ret = EXIT_SUCCESS;

  // set appropriate locale
  setlocale(LC_ALL, NULL);

  Option opt {};
  int argused = 1 + opt.parse(&argv[1], argc-1);  // Skip argv[0].

  if ((nargs = argc - argused) != 2) {
    std::cerr << "Incorrect number of arguments "
              << nargs << " (2 arguments expected).  "  \
      "Type --help to see usage." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // clean up options
  if (opt.coefficient != 1)
    opt.no_length_normalize = true;

  // read word vectors
  if ((ret = read_vectors(argv[argused++], &opt)))
    return ret;

  // read seed sets
  if ((ret = read_seed_set(argv[argused++])))
    return ret;

  // generate mapping from vector ids to the polarities of respective
  // words
  v2ps_t vecid2polscore;
  int seed_cnt = 0;
  w2v_t::const_iterator vecid, vecend = word2vecid.end();
  for (auto &w2p : word2polscore) {
    if (w2p.second.first != Polarity::NEUTRAL)
      ++seed_cnt;

    vecid = word2vecid.find(w2p.first);
    if (vecid == vecend)
      continue;

    vecid2polscore.emplace(vecid->second, w2p.second);
    if (debug) {
      std::cerr << "word: " << w2p.first
		<< ", vecid = " << vecid->second
		<< ", polarity = " << (int) w2p.second.first << std::endl;
    }
  }

  if (opt.n_terms > 0) {
    opt.n_terms -= seed_cnt;
    if (opt.n_terms < 1)
      goto print_steps;
  }

  // apply the requested expansion algorithm
  switch (opt.etype) {
  case ExpansionType::NC_CLUSTERING:
    expand_nearest_centroids(&vecid2polscore, &NWE, opt.n_terms);
    break;
  case ExpansionType::KNN_CLUSTERING:
    expand_knn(&vecid2polscore, &NWE, opt.n_terms, opt.knn);
    break;
  case ExpansionType::PCA_CLUSTERING:
    expand_pca(&vecid2polscore, &NWE, opt.n_terms);
    break;
  case ExpansionType::PRJ_CLUSTERING:
    expand_prjct(&vecid2polscore, &NWE, opt.n_terms,
                 opt.alpha, opt.delta, opt.max_iters);
    break;
  default:
    throw std::invalid_argument("Invalid type of seed set"
                                " expansion algorithm.");
  }
  // output new terms in sorted alphabetic order
 print_steps:
  output_terms(std::cout, &vecid2polscore);
  return ret;
}
