/** @file expansion.h
 *
 *  @brief Methods for generating sentiment lexicons from neural word embeddings.
 *
 *  This file declares several methods for generating sentiment
 *  lexicons on the basis of previously computed neural word
 *  embeddings.
 */

#ifndef VEC2DIC_EXPANSION_H_
# define VEC2DIC_EXPANSION_H_ 1

//////////////
// Includes //
//////////////
#include <armadillo>      // arma::mat
#include <forward_list>   // std::forward_list
#include <limits>         // std::numeric_limits
#include <string>         // std::string
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::pair

///////////
// Types //
///////////

/**
 * Polarity types.
 */
enum class Polarity:
char {
  POSITIVE = 0,   //< positive lexical polarity
    NEGATIVE,     //< negative lexical polarity
    NEUTRAL,      //< neutral lexical polarity
    MAX_SENTINEL  //< auxiliary polarity type (used for
                  //< determining maximum number of polarity
                  //< types)
    };

/** Integral type for distance measure */
using dist_t = double;
const dist_t MAX_DIST = std::numeric_limits<double>::max();

/** Integral type for vector id */
using vid_t = unsigned int;

/** Polarity-score pair */
using ps_t = std::pair<Polarity, dist_t>;

/** Map from word to its polarity */
using w2ps_t = std::unordered_map<std::string, ps_t>;

/** Map from string to the index of its vector */
using w2v_t = std::unordered_map<std::string, vid_t>;

/** Map from word index to its polarity and score */
using v2ps_t = std::unordered_map<vid_t, ps_t>;

/** Map from vector index to string */
using v2w_t = std::unordered_map<vid_t, std::string>;

/** Forward list of vector id's */
using vid_flist_t = std::forward_list<vid_t>;

/** Default learning rate for gradient methods */
extern const double DFLT_ALPHA;

/** Minimum required improvement for gradient methods */
extern const double DFLT_DELTA;

/** Maximum number of gradient updates */
extern const unsigned long MAX_ITERS;

/////////////
// Methods //
/////////////

/**
 * Apply nearest centroids clustering algorithm to expand seed sets of polar terms
 *
 * @param a_vecid2polscore - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of new terms to extract (these terms will have
 *              minimal distance to their respective centroids)
 * @param a_early_break - only apply one iteration (i.e. only assign words to the
 *                      centroids of known polarity term clusters)
 *
 * @return \c void (`a_vecid2polscore` is modified in place)
 */
void expand_nearest_centroids(v2ps_t *a_vecid2polscore,
                              const arma::mat *a_nwe, const int a_N,
                              const bool a_early_break = false);
/**
 * Apply K-nearest neighbors clustering algorithm to expand seed sets of polar terms
 *
 * @param a_vecid2polscore - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 * @param a_K - number of nearest neighbors to use
 *
 * @return \c void (`a_vecid2polscore` is modified in place)
 */
void expand_knn(v2ps_t *a_vecid2polscore, const arma::mat *a_nwe,
                const int a_N, const int a_K = 5);

/**
 * Apply principal component analysis to expand seed sets of polar terms
 *
 * This algorithm applies the PCA algorithm to obtain the subspace of
 * polar terms and then projects remaining terms on this subspace.
 *
 * @param a_vecid2polscore - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 *
 * @return \c void (`a_vecid2polscore` is modified in place)
 */
void expand_pca(v2ps_t *a_vecid2polscore,
                const arma::mat *a_nwe, const int a_N);

/**
 * Apply projection to expand seed sets of polar terms
 *
 * This algorithm projects all unknown terms on the vector subspace
 * defined by the known polar items and then extends seed sets of
 * known polar terms according to the lengths of projection vectors.
 *
 * @param a_vecid2polscore - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 * @param a_alpha - learning rate
 * @param a_delta - minimum required improvement
 * @param a_max_iters - maximum number of updates
 *
 * @return \c void (`a_vecid2polscore` is modified in place)
 */
void expand_prjct(v2ps_t *a_vecid2polscore,
                  const arma::mat *a_nwe, const int a_N,
                  const double a_alpha = DFLT_ALPHA,
                  const double a_delta = DFLT_DELTA,
                  const unsigned long a_max_iters = MAX_ITERS);

#endif    // VEC2DIC_EXPANSION_H_
