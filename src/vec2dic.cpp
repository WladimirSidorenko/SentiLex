//////////////
// Includes //
//////////////
#include "expansion.h"
#include "optparse.h"

#include <algorithm>
#include <armadillo>		// arma::mat
#include <cctype>		// std::isspace()
#include <clocale>		// setlocale()
#include <cstdio>		// sscanf()
#include <cstdlib>		// std::exit()
#include <fstream>		// std::ifstream
#include <functional>
#include <iostream>		// std::cerr, std::cout
#include <locale>
#include <string>		// std::string
#include <unordered_map>	// std::unordered_map
#include <utility>		// std::make_pair

/////////////
// Classes //
/////////////
enum class ExpansionType: int {
  KNN_CLUSTERING = 0,		// K-nearest neighbors
    NC_CLUSTERING,		// Nearest centroids algorithm
    PRJ_CLUSTERING,		// Projection-based clustering
    PRJ_LENGTH,			// Projection-length
    LIN_TRANSFORM,		// Linear transformation
    MAX_SENTINEL		// Unused type that serves as a sentinel
    };

// forward declaration of `usage()` method
static void usage(int a_ret = EXIT_SUCCESS);

class Option: public optparse {
public:
  // Members
  std::ifstream m_seedfile();
  int n_terms = -1;
  ExpansionType etype = ExpansionType::KNN_CLUSTERING;

  Option() {}

  BEGIN_OPTION_MAP_INLINE()
  ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
  usage();

  ON_OPTION_WITH_ARG(SHORTOPT('n') || LONGOPT("n-terms"))
  n_terms = std::atoi(arg);

  ON_OPTION_WITH_ARG(SHORTOPT('t') || LONGOPT("type"))
  int itype = std::atoi(arg);
  if (itype < 0 || itype >= static_cast<int>(ExpansionType::MAX_SENTINEL))
    throw invalid_value("Invalid type of expansion algorithm.");

  etype = static_cast<ExpansionType>(itype);

  END_OPTION_MAP()
};

/////////////////////////////
// Variables and Constants //
/////////////////////////////

// String epresenting polarity clases
const std::string positive {"positive"};
const std::string negative {"negative"};
const std::string neutral {"neutral"};

/// Mapping from Polarity enum to string
static const std::string pol2pol[] {positive, negative, neutral};
/// Mapping from words to the index of their NWE vector
static w2v_t word2vecid;
/// Mapping from the index of NWE vectors to their respective words
static v2w_t vecid2word;
/// Mapping from word to its polarity
static w2p_t word2pol;
/// Matrix of neural word embeddings
static arma::mat NWE;

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
  std::cerr << "Expand initial seed set of subjective terms by applying clustering" << std::endl;
  std::cerr << "to neural word embeddings." << std::endl << std::endl;
  std::cerr << "Usage:" << std::endl;
  std::cerr << "vec2dic [OPTIONS] VECTOR_FILE SEED_FILE" << std::endl << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "-h|--help  show this screen and exit" << std::endl;
  std::cerr << "-n|--n-terms  number of terms to extract (default: -1 (unlimited))" << std::endl;
  std::cerr << "-t|--type  type of expansion algorithm to use:" << std::endl;
  std::cerr << "           (0 - KNN, 1 - nearest centroids, 2 - projection," << std::endl;
  std::cerr << "            3 - projection length, 4 - linear transformation)" << std::endl << std::endl;
  std::cerr << "Exit status:" << std::endl;
  std::cerr << EXIT_SUCCESS << " on sucess" <<  "non-" << EXIT_SUCCESS << " otherwise" << std::endl;
  std::exit(a_ret);
}

/**
 * Auxiliary function for removing blanks from the left end of a string
 *
 * @param s - string to be trimmed
 *
 * @return reference to the original string with leading blanks truncated
 */
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

/**
 * Auxiliary function for removing blanks from the right end of a string
 *
 * @param s - string to be trimmed
 *
 * @return reference to the original string with trailing blanks truncated
 */
static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

/**
 * Auxiliary function for removing blanks from both ends of a string
 *
 * @param s - string to be trimmed
 *
 * @return original string with leading and trailing blanks removed
 */
static inline std::string &normalize(std::string &s) {
  // strip leading and trailing whitespaces
  ltrim(rtrim(s));
  // downcase the string
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);;
  return s;
}

/**
 * Read NWE vectors for word terms
 *
 * @param a_ret - exit code for the program
 *
 * @return \c 0 on success, non-\c 0 otherwise
 */
static int read_vectors(const char *a_fname) {
  float iwght;
  const char *cline;
  std::string iline;
  size_t space_pos;
  vid_t mrows = 0, ncolumns = 0, icol = 0, irow = 0;
  std::cerr << "Reading word vectors ... ";

  std::ifstream is(a_fname);
  if (! is) {
    std::cerr << "Cannot open file " << a_fname << std::endl;
    goto error_exit;
  }
  // skip empty lines at the beginning of file
  while (std::getline(is, iline) && iline.empty()) {}
  // initialize matrix (columns represent words, rows are coordinates)
  if (sscanf(iline.c_str(), "%u %u", &ncolumns, &mrows) != 2) {
    std::cerr << "Incorrect declaration line format: '" << iline.c_str() << std::endl;
    goto error_exit;
  }
  // allocate space for map and matrix
  word2vecid.reserve(ncolumns); vecid2word.reserve(ncolumns);
  NWE.set_size(mrows, ncolumns);

  while (icol < ncolumns && std::getline(is, iline)) {
    space_pos = iline.find_first_of(' ');
    while (space_pos > 0 && std::isspace(iline[space_pos])) {--space_pos;}
    if (space_pos == 0  && std::isspace(iline[space_pos])) {
      std::cerr << "Incorrect line format (missing word): " << iline << std::endl;
      goto error_exit;
    }
    ++space_pos;
    word2vecid.emplace(std::move(iline.substr(0, space_pos)), std::move(icol));
    vecid2word.emplace(std::move(icol), std::move(iline.substr(0, space_pos)));

    cline = &iline[space_pos];

    for (irow = 0; irow < mrows && sscanf(cline, " %f", &iwght) == 1; ++irow) {
      NWE(irow, icol) = iwght;
    }
    if (irow != mrows) {
      std::cerr << "cline = " << cline << std::endl;
      std::cerr << "Incorrect line format: declared vector size " << mrows << \
	" differs from the actual size " << irow << std::endl;
      goto error_exit;
    }
    ++icol;
  }

  if (!is.eof() && is.fail()) {
    std::cerr << "Failed to read vector file " << a_fname << std::endl;
    goto error_exit;
  }
  if (irow != mrows) {
    std::cerr << "Incorrect file format: declared number of rows " << mrows << \
      " differs from the actual number " << irow << std::endl;
    goto error_exit;
  }
  is.close();
  std::cerr << "done (read " << ncolumns << " columns with " << mrows << " rows)" << std::endl;
  return 0;

 error_exit:
  is.close();			// basic guarantee
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
  if (! is) {
    std::cerr << "Cannot open file " << a_fname << std::endl;
    goto error_exit;
  }

  // read input file
  while (std::getline(is, iline)) {
    if (iline.empty())
      continue;

    // remove leading and trailing whitespaces
    normalize(iline);
    // find first tab character
    tab_pos = iline.find_first_of('\t');
    tab_pos_orig = tab_pos;
    // skip leading whitespaces
    while (iline[++tab_pos] && std::isspace(iline[tab_pos])) {}
    if (tab_pos == std::string::npos || ! iline[tab_pos]) {
      std::cerr << "Incorrect line format (missing polarity): " << iline << std::endl;
      goto error_exit;
    }
    // determine polarity class
    if (iline.compare(tab_pos, positive.length(), positive) == 0)
      ipol = Polarity::POSITIVE;
    else if (iline.compare(tab_pos, negative.length(), negative) == 0)
      ipol = Polarity::NEGATIVE;
    else if (iline.compare(tab_pos, neutral.length(), neutral) == 0)
      ipol = Polarity::NEUTRAL;
    else {
      std::cerr << "Unrecognized polarity class " << iline.substr(tab_pos) << std::endl;
      goto error_exit;
    }

    while (tab_pos_orig > 0 && std::isspace(iline[tab_pos_orig])) {--tab_pos_orig;}
    if (tab_pos_orig == 0 && std::isspace(iline[tab_pos_orig])) {
      std::cerr << "Incorrect line format (missing word): " << iline << std::endl;
      goto error_exit;
    }
    ++tab_pos_orig;
    word2pol.emplace(std::move(iline.substr(0, tab_pos_orig)), std::move(ipol));
  }

  if (!is.eof() && is.fail()) {
    std::cerr << "Failed to read seed set file " << a_fname << std::endl;
    goto error_exit;
  }
  is.close();
  std::cerr << "done (read " << word2pol.size() << " entries)" << std::endl;
  return 0;

 error_exit:
  is.close();		// basic guarantee
  word2pol.clear();
  return 1;
}

//////////
// Main //
//////////
int main(int argc, char *argv[]) {
  int nargs;
  int ret = EXIT_SUCCESS;

  // set appropriate locale
  setlocale(LC_ALL, NULL);

  Option opt {};
  int argused = 1 + opt.parse(&argv[1], argc-1); // Skip argv[0].

  if ((nargs = argc - argused) != 2) {
    std::cerr << "Incorrect number of arguments " << nargs << " (2 arguments expected).  " \
      "Type --help to see usage." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // read word vectors
  if ((ret = read_vectors(argv[argused++])))
    return ret;

  // read seed sets
  if ((ret = read_seed_set(argv[argused++])))
    return ret;

  // generate mapping from vector id's to the known polarity of
  // respective words
  v2p_t vecid2pol;
  w2v_t::const_iterator vecid, vecend = word2vecid.end();
  for (auto& w2p: word2pol) {
    vecid = word2vecid.find(w2p.first);
    if (vecid != vecend)
      vecid2pol.emplace(vecid->second, w2p.second);
  }

  // apply the requested expansion algorithm
  switch (opt.etype) {
  case ExpansionType::KNN_CLUSTERING:
    expand_knn(&vecid2pol, &NWE, opt.n_terms);
    break;
  case ExpansionType::NC_CLUSTERING:
    expand_nearest_centroids(&vecid2pol, &NWE, opt.n_terms);
    break;
  case ExpansionType::PRJ_CLUSTERING:
    expand_projected(&vecid2pol, &NWE, opt.n_terms);
    break;
  case ExpansionType::PRJ_LENGTH:
    expand_projected_length(&vecid2pol, &NWE, opt.n_terms);
    break;
  case ExpansionType::LIN_TRANSFORM:
    expand_linear_transform(&vecid2pol, &NWE, opt.n_terms);
    break;
  default:
    throw std::invalid_argument("Invalid type of seed set expansion algorithm.");
  }

  // output new terms
  // for (auto& v2p: vecid2pol) {
  // }

  return ret;
}
