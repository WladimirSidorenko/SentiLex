//////////////
// Includes //
//////////////
#include "expansion.h"

#include <algorithm>		// std::swap
#include <cassert>		// assert
#include <cstdlib>		// size_t
#include <cstdint>		// MAX_SIZE
#include <iostream>		// std::cerr
#include <unordered_set>	// std::unordered_set
#include <utility>		// std::make_pair

///////////
// Types //
///////////

/** Set of NWE id to polarity id */
typedef std::unordered_set<unsigned int> v_id_t;

/** Map from polarity id to a set of respective NWE id's */
typedef std::unordered_map<size_t, v_id_t> pi2v_t;

/** Map from NWE id to polarity id */
typedef std::unordered_map<unsigned int, size_t> v2pi_t;

/** Pair of vector id and its distance to one of the centroids */
typedef std::pair<size_t, double> vid_dist_t;

/** Vector of pairs of vector id and vector distance to one of the
    centroids */
typedef std::vector<vid_dist_t> vdist_t;

/////////////
// Methods //
/////////////
void expand_knn(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}

/**
 * Find cluster whose centroid is nearest to the given word vector
 *
 * @param a_centroids - matrix of pre-computed cluster centroids
 * @param a_vec - word vector whose nearest cluster should be found
 *
 * @return id of the cluster with the nearest centroid
 */
static size_t _nc_find_cluster(const arma::mat *a_centroids,	\
				   const double *a_vec) {
  double tmp_j;
  size_t ret = 0;
  const double *centroid;
  size_t j, idistance, mindistance = SIZE_MAX;
  for (size_t i = 0; i < a_centroids->n_cols; ++i) {
    idistance = 0;
    centroid = a_centroids->colptr(i);
    // compute Euclidean distance from vector to centroid
    for (j = 0; j < a_centroids->n_rows; ++j) {
      tmp_j = a_vec[j] - centroid[j];
      idistance += tmp_j * tmp_j;
    }
    // compare distance with
    if (idistance < mindistance) {
      mindistance = idistance;
      ret = i;
    }
  }
  return ret;
}

/**
 * Assign word vectors to the newly computed centroids
 *
 * @param a_polid2vecids - map from polarity id's to vector id's
 * @param a_vecid2polid - map from vector id's to polarity id's
 * @param a_centroids - newly computed centroids
 * @param a_nwe - matrix containing neural word embeddings
 *
 * @return \c true if clusters changed, \c false otherwise
 */
static bool _nc_assign(pi2v_t *a_polid2vecids, v2pi_t *a_vecid2polid, \
		       const arma::mat *a_centroids, const arma::mat *a_nwe) {
  size_t polid;
  bool ret = false;
  v_id_t::iterator v_id_pos;
  size_t N = a_nwe->n_cols;
  bool is_absent = false, differs = false;
  v2pi_t::iterator it, it_end = a_vecid2polid->end();
  // iterate over each word, find its new cluster, and reassign if
  // necessary
  for (size_t vecid = 0; vecid < N; ++vecid) {
    polid = _nc_find_cluster(a_centroids, a_nwe->colptr(vecid));
    // obtain previous polarity of this vector
    it = a_vecid2polid->find(vecid);
    // assign vecid to new cluster if necessary
    if ((is_absent = (it == it_end)) || (differs = (it->second != polid))) {
      // remove word vector from the previous polarity class
      if (differs) {
	v_id_pos = (*a_polid2vecids)[it->second].find(vecid);
	// obtain position of the vector id in its previous class set
	if (v_id_pos != (*a_polid2vecids)[it->second].end())
	  (*a_polid2vecids)[it->second].erase(v_id_pos);

	differs = false;
      }
      (*a_vecid2polid)[vecid] = polid;
      (*a_polid2vecids)[polid].insert(vecid);
      ret = true;
    }
  }
  return ret;
}

/**
 * Check if two matrices are equal or not
 *
 * @param a_mat1 - 1-st matrix to compare
 * @param a_mat2 - 2-nd matrix to compare
 *
 * @return \c true if both matrices are equal, \c false otherwise
 */
static bool _cmp_mat(const arma::mat *a_mat1, const arma::mat *a_mat2) {
  bool ret = ((a_mat1->n_rows == a_mat2->n_rows) && (a_mat1->n_cols == a_mat2->n_cols));
  if (!ret)
    return ret;

  size_t i, j;
  for (i = 0; i < a_mat1->n_cols; ++i) {
    for (j = 0; j < a_mat1->n_rows; ++j) {
      if (!(ret = ((*a_mat1)(j, i) == (*a_mat2)(j, i))))
	return ret;
    }
  }

  return ret;
}

/**
 * Compute centroids of previously populated clusters
 *
 * @param a_new_centroids - container for storing new centroid coordinates
 * @param a_old_centroids - container storing previously computed centroids
 * @param a_pol2vecids - previously populated clusters
 * @param a_nwe - matrix containing neural word embeddings
 *
 * @return \c true if centroids changed, \c false otherwise
 */
static bool _nc_compute_centroids(arma::mat *a_new_centroids, const arma::mat *a_old_centroids, \
				  const pi2v_t *a_pol2vecids, const arma::mat *a_nwe) {
  // zero-out new centroids
  a_new_centroids->zeros();
  // compute new centroids
  size_t c_id;
  for (auto& p2v: *a_pol2vecids) {
    c_id = p2v.first;
    std::cerr << "a) centroid #" << c_id << ": " << a_new_centroids->col(c_id) << std::endl;
    // sum-up coordinates of all the vectors pertaining to the given
    // polarity
    for (auto& vecid: p2v.second)
      a_new_centroids->col(c_id) += a_nwe->col(vecid);

    std::cerr << "b) centroid #" << c_id << ": " << a_new_centroids->row(c_id) << std::endl;
    std::cerr << "   p2v.second.size() =" << p2v.second.size() << std::endl;
    // take the mean of the new centroid
    a_new_centroids->col(c_id) /= float(p2v.second.size());
    std::cerr << "c) centroid #" << c_id << ": " << a_new_centroids->col(c_id) << std::endl;
  }
  return _cmp_mat(a_new_centroids, a_old_centroids);
}

/**
 * Perform a single iteration of the nearest centroids algorithm
 *
 * @param a_new_centroids - container for storing new centroid coordinates
 * @param a_old_centroids - container storing previously computed centroids
 * @param a_pol2vecids - previously populated clusters
 * @param a_nwe - matrix containing neural word embeddings of single terms
 *
 * @return \c true if clusters changes, \c false otherwise
 */
static inline bool _nc_run(arma::mat *a_new_centroids, arma::mat *a_old_centroids, \
			   pi2v_t *a_polid2vecids, v2pi_t *a_vecid2polid, \
			   const arma::mat *a_nwe) {
  bool ret = false;
  // calculate centroids
  if ((ret = _nc_compute_centroids(a_new_centroids, a_old_centroids, a_polid2vecids, a_nwe)))
    // assign new items to their new nearest centroids
    ret = _nc_assign(a_polid2vecids, a_vecid2polid, a_new_centroids, a_nwe);

  return ret;
}

/**
 * Expand polarity sets by adding terms that are closest to centroids
 *
 * @param a_vecid2pol - matrix of pre-computed cluster centroids
 * @param a_vec - word vector whose nearest cluster should be found
 *
 * @return id of the cluster with the nearest centroid
 */
static void _nc_expand(v2p_t *a_vecid2pol, const arma::mat *new_centroids, \
		       const arma::mat *a_nwe, const int a_N) {
  // vid_dist_t v_dist;
  // populate
  for (unsigned int i = 0; i < a_nwe->n_rows; ++i) {
    if (a_vecid2pol->find(i) == a_vecid2pol->end())
      continue;

  }
}

void expand_nearest_centroids(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {
  // determine the number of clusters
  size_t n_clusters = static_cast<size_t>(Polarity::MAX_SENTINEL);
  // create two matrices for storing centroids
  arma::mat centroids = arma::mat(a_nwe->n_rows, n_clusters);
  arma::mat new_centroids = arma::mat(a_nwe->n_rows, n_clusters);
  arma::mat *centr_p = &centroids, *new_centr_p = &new_centroids;
  // populate intial clusters
  size_t polid;
  pi2v_t polid2vecids;
  polid2vecids.reserve(n_clusters);
  v2pi_t vecid2polid;
  vecid2polid.reserve(a_nwe->n_cols);
  for (auto &v2p: *a_vecid2pol) {
    polid = static_cast<size_t>(v2p.second);
    polid2vecids[polid].insert(v2p.first);
    vecid2polid[v2p.first] = polid;
  }
  // run the algorithm until convergence
  while (_nc_run(new_centr_p, centr_p, &polid2vecids, &vecid2polid, a_nwe)) {
    // swap the centroids
    std::swap(new_centr_p, centr_p);
  }

  // add new terms to the polarity sets based on their distance to
  // centroids (centroids and new centroids should contain identical
  // values here)
  assert(_cmp_mat(new_centr_p, centr_p));
  _nc_expand(a_vecid2pol, new_centr_p, a_nwe, a_N);
}

void expand_projected(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}

void expand_projected_length(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}

void expand_linear_transform(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}
