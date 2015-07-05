//////////////////////////////////////////////////////////////////
// Includes
#include "optparse.h"

#include <algorithm>
#include <armadillo>		// arma::mat
#include <cctype>		// std::isspace()
#include <cstdio>		// sprintf(), sscanf()
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
static void usage(int a_ret = EXIT_SUCCESS);

class option : public optparse {
public:
  // Members
  std::ifstream m_seedfile();

  option() {}

  BEGIN_OPTION_MAP_INLINE()
  ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
  usage();

  END_OPTION_MAP()
};

enum class Polarity: char {
  POSITIVE = 0,
    NEGATIVE,
    NEUTRAL
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
/// Mapping from words in seed set to their polarity
static std::unordered_map<std::string, Polarity> word2pol;
/// Mapping from words to the index of their NWE vector
static std::unordered_map<std::string, unsigned int> word2vecid;
/// Mapping from the index of NWE vectors to their respective words
static std::unordered_map<unsigned int, std::string> vecid2word;
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
  std::cerr << "DESCRIPTION:" << std::endl;
  std::cerr << "Expand initial seed set of subjective terms by applying clustering" << std::endl;
  std::cerr << "to neural word embeddings." << std::endl << std::endl;
  std::cerr << "USAGE:" << std::endl;
  std::cerr << "vec2dic [OPTIONS] SEED_FILE VECTOR_FILE" << std::endl << std::endl;
  std::cerr << "OPTIONS:" << std::endl;
  std::cerr << "-h|--help  show this screen and exit" << std::endl << std::endl;
  std::cerr << "EXIT STATUS" << std::endl;
  std::cerr << EXIT_SUCCESS << " on sucess" << std::endl;
  std::cerr << "non-" << EXIT_SUCCESS << " on failure" << std::endl;
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
    word2pol.insert(std::make_pair<std::string, Polarity>		\
		    (std::move(iline.substr(0, tab_pos_orig)), std::move(ipol)));
  }

  if (is.bad() || is.bad()) {
    std::cerr << "Failed to read file " << a_fname << std::endl;
    goto error_exit;
  }
  return 0;

 error_exit:
  word2pol.clear();		// basic guarantee
  return 1;
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
  size_t mrows, ncolumns, irow = 0, icolumn = 0;

  std::ifstream is(a_fname);
  if (! is) {
    std::cerr << "Cannot open file " << a_fname << std::endl;
    goto error_exit;
  }
  // skip empty lines at the beginning of file
  while (std::getline(is, iline) && iline.empty()) {}
  // initialize matrix
  sscanf(iline.c_str(), "%zu %zu", &mrows, &ncolumns);
  // allocate space for map and matrix
  word2vecid.reserve(mrows); vecid2word.reserve(mrows);
  NWE.set_size(mrows, ncolumns);

  while (irow < mrows && std::getline(is, iline)) {
    space_pos = iline.find_first_of(' ');
    while (space_pos > 0 && std::isspace(iline[space_pos])) {--space_pos;}
    if (space_pos == 0  && std::isspace(iline[space_pos])) {
      std::cerr << "Incorrect line format (missing word): " << iline << std::endl;
      goto error_exit;
    }
    ++space_pos;
    word2vecid.insert(std::make_pair<std::string, size_t> \
		      (std::move(iline.substr(0, space_pos)), std::move(irow)));
    vecid2word.insert(std::make_pair<size_t, std::string>	\
		      (std::move(irow), std::move(iline.substr(0, space_pos))));

    cline = &iline[space_pos];

    for (icolumn = 0; icolumn < ncolumns && sscanf(cline, " %f", &iwght) == 1; ++icolumn) {
      NWE(irow, icolumn) = iwght;
    }
    if (icolumn != ncolumns) {
      std::cerr << "cline = " << cline << std::endl;
      std::cerr << "Incorrect line format: declared vector size " << ncolumns << \
	" differs from the actual size " << icolumn << std::endl;
      goto error_exit;
    }
    ++irow;
  }

  if (is.bad() || is.bad()) {
    std::cerr << "Failed to read file " << a_fname << std::endl;
    goto error_exit;
  }
  if (irow != mrows) {
    std::cerr << "Incorrect file format: declared number of rows " << mrows << \
      " differs from the actual number " << irow << std::endl;
    goto error_exit;
  }

  return 0;

 error_exit:
  word2vecid.clear();		// basic guarantee
  vecid2word.clear();		// basic guarantee
  NWE.reset();
  return 1;
}

//////////
// Main //
//////////
int main(int argc, char *argv[]) {
  int nargs;
  int ret = EXIT_SUCCESS;

  option opt {};
  int argused = 1 + opt.parse(&argv[1], argc-1); // Skip argv[0].

  if ((nargs = argc - argused) != 2) {
    std::cerr << "Incorrect number of arguments " << nargs << " (2 arguments expected).  " \
      "Type --help to see usage." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // read seed sets
  if ((ret = read_seed_set(argv[argused++])))
    return ret;

  // read word vectors
  if ((ret = read_vectors(argv[argused++])))
    return ret;

  return ret;
}
