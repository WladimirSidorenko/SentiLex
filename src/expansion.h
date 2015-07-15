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

/** Map from word to its polarity */
typedef std::unordered_map<std::string, Polarity> w2p_t;

/** Map from string to the index of its vector */
typedef std::unordered_map<std::string, unsigned int> w2v_t;

/** Map from word index to its polarity */
typedef std::unordered_map<unsigned int, Polarity> v2p_t;

/** Map from vector index to string */
typedef std::unordered_map<unsigned int, std::string> v2w_t;

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
 *
 * @return \c void (`a_vecid2pol` is modified in place)
 */
void expand_nearest_centroids(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N);

/**
 * Apply K-nearest neighbors clustering algorithm to expand seed sets of polar terms
 *
 * @param a_vecid2pol - dictionary mapping known vector id's to the
 *                      polarities of their respective words
 * @param a_nwe - matrix of neural word embeddings
 * @param a_N - number of polar terms to extract
 *
 * @return \c void (`a_vecid2pol` is modified in place)
 */
void expand_knn(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N);

/**
 * Apply projection to expand seed sets of polar terms
 *
 * This algorithm first projects all unknown terms on the vector
 * subspace defined by the known polar items and then applies standard
 * clustering algorithm on the projections.
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
void expand_projected_length(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N);

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
