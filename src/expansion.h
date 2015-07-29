#ifndef VEC2DIC_EXPANSION_H_
# define VEC2DIC_EXPANSION_H_ 1

//////////////
// Includes //
//////////////
#include <armadillo>		// arma::mat
#include <string>		// std::string
#include <unordered_map>	// std::unordered_map

///////////
// Types //
///////////
enum class Polarity: char {
  POSITIVE = 0,
    NEGATIVE,
    NEUTRAL,
    MAX_SENTINEL
};

/** Integral type for distance measure */
using dist_t = double;

/** Integral type for vector id */
using vid_t = unsigned int;

/** Map from word to its polarity */
using w2p_t = std::unordered_map<std::string, Polarity>;

/** Map from string to the index of its vector */
using w2v_t = std::unordered_map<std::string, vid_t>;

/** Map from word index to its polarity */
using v2p_t = std::unordered_map<vid_t, Polarity>;

/** Map from vector index to string */
using v2w_t = std::unordered_map<vid_t, std::string>;

/////////////
// Methods //
/////////////

/**
 * Apply nearest centroids clustering algorithm to expand seed sets of polar terms
 *
 * @param a_vecid2pol - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of new terms to extract (these terms will have
 *              minimal distance to their respective centroids)
 * @param a_early_break - only apply one iteration (i.e. only assign words to the
 *                      centroids of known polarity term clusters)
 *
 * @return \c void (`a_vecid2pol` is modified in place)
 */
void expand_nearest_centroids(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N, \
			      const bool a_early_break = false);
/**
 * Apply K-nearest neighbors clustering algorithm to expand seed sets of polar terms
 *
 * @param a_vecid2pol - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 * @param a_K - number of nearest neighbors to use
 *
 * @return \c void (`a_vecid2pol` is modified in place)
 */
void expand_knn(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N, const int a_K = 5);

/**
 * Apply principal component analysis to expand seed sets of polar terms
 *
 * This algorithm applies the PCA algorithm to obtain the subspace of
 * polar terms and then projects remaining terms on this subspace.
 *
 * @param a_vecid2pol - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 *
 * @return \c void (`a_vecid2pol` is modified in place)
 */
void expand_pca(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N);

/**
 * Apply projection to expand seed sets of polar terms
 *
 * This algorithm projects all unknown terms on the vector subspace
 * defined by the known polar items and then extends seed sets of
 * known polar terms according to the lengths of projection vectors.
 *
 * @param a_vecid2pol - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 *
 * @return \c void (`a_vecid2pol` is modified in place)
 */
void expand_projected(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N);

/**
 * Derive projection matrix to expand seed sets of polar terms
 *
 * This algorithm tries to derive a projection matrix which makes all
 * known neutral terms the picture of the projection, and maps all
 * known positive and negative terms to vectos (1; 0) and (0; -1)
 * respectively.
 *
 * @param a_vecid2pol - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 *
 * @return \c void (`a_vecid2pol` is modified in place)
 */
void expand_linear_transform(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N);

#endif	// VEC2DIC_EXPANSION_H_
