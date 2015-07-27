//////////////
// Includes //
//////////////
#include "expansion.h"

#include <algorithm>		// std::swap(), std::sort()
#include <cassert>		// assert
#include <cfloat>		// DBL_MAX
#include <climits>		// UINT_MAX
#include <cmath>		// sqrt()
#include <cstdlib>		// size_t
#include <cstdint>		// MAX_SIZE
#include <iostream>		// std::cerr
#include <unordered_set>	// std::unordered_set
#include <queue>		// std::priority_queue

///////////
// types //
///////////

/** Integral type for polarity id */
using pol_t = size_t;

/** Set of NWE id's */
using vids_t = std::unordered_set<vid_t>;

/** Map from polarity id to a set of respective NWE id's */
using pi2v_t = std::unordered_map<pol_t, vids_t>;

/** Map from NWE id to polarity id */
using v2pi_t = std::unordered_map<vid_t, pol_t>;

/** 3-tuple of vector id, vector polarity, and its distance to the
    nearest centroid */
using vpd_t = struct VPD {
  dist_t m_distance {};
  pol_t m_polarity {};
  vid_t m_vecid {};

  VPD(void) {}

  VPD(dist_t a_distance, pol_t a_polarity, vid_t a_vecid):
    m_distance{a_distance}, m_polarity{a_polarity}, m_vecid{a_vecid}
  {}

  bool operator <(const VPD& rhs) const {
    return m_distance < rhs.m_distance;
  }
};

/** vector of 3-tuples of vector id, vector polarity, and distctance
    to the centroid */
using vpd_v_t = std::vector<vpd_t>;

/** priority queue of 3-tuples of vector id, vector polarity, and
    distctance to the centroid */
using vpd_pq_t = std::priority_queue<vpd_t>;

/////////////
// Methods //
/////////////

/**
 * Check if two matrices are equal
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
 * Compute unnormaized Euclidean distance between two vectors
 *
 * @param a_vec1 - 1-st vector
 * @param a_vec2 - 2-nd vector
 * @param a_N - number of elements in each vector
 *
 * @return unnormaized Euclidean distance between vectors
 */
static inline dist_t _unnorm_eucl_distance(const dist_t *a_vec1, const dist_t *a_vec2, size_t a_N) {
  dist_t tmp_i = 0., idistance = 0;

  for (size_t i = 0; i < a_N; ++i) {
    tmp_i = a_vec1[i] - a_vec2[i];
    idistance += tmp_i * tmp_i;
  }
  return idistance;
}

/**
 * Add newly extracted terms to the polarity lexicon
 *
 * @param a_vecid2pol - target dictionary mapping vector id's to polarities
 * @param a_vpds - source vector of NWE ids, their polarities, and distances
 * @param a_j - actual number of new terms
 * @param a_N - maximum number of terms to extract (-1 means unlimited)
 *
 * @return \c void
 */
static inline void _add_terms(v2p_t *a_vecid2pol, vpd_v_t *a_vpds, const int a_j, const int a_N) {
  // sort vpds according to their distances to the centroids
  std::sort(a_vpds->begin(), a_vpds->end());
  // add new terms to the dictionary
  vpd_t *ivpd;
  for (int i = 0; (a_N < 0 || i < a_N) && i < a_j; ++i) {
    ivpd = &(*a_vpds)[i];
    a_vecid2pol->emplace(ivpd->m_vecid, static_cast<Polarity>(ivpd->m_polarity));
  }
}

/**
 * Find cluster whose centroid is nearest to the given word vector
 *
 * @param a_centroids - matrix of pre-computed cluster centroids
 * @param a_vec - word vector whose nearest cluster should be found
 * @param a_dist - (optional) pointer to a variable in which actual
 *                  Euclidean distance to the vector should be stored
 * @return id of the cluster with the nearest centroid
 */
static pol_t _nc_find_cluster(const arma::mat *a_centroids,	\
			       const double *a_vec,		\
			       dist_t *a_dist = nullptr) {
  pol_t ret = 0;
  const double *centroid;
  dist_t idistance, mindistance = DBL_MAX;
  for (size_t i = 0; i < a_centroids->n_cols; ++i) {
    centroid = a_centroids->colptr(i);
    // compute Euclidean distance from vector to centroid
    idistance = _unnorm_eucl_distance(a_vec, centroid, a_centroids->n_rows);
    // compare distance with
    if (idistance < mindistance) {
      mindistance = idistance;
      ret = i;
    }
  }
  if (a_dist != nullptr)
    *a_dist = sqrt(idistance);

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
  bool ret = false;
  bool is_absent = false, differs = false;
  pol_t polid;
  vid_t N = a_nwe->n_cols;
  vids_t::iterator v_id_pos;
  v2pi_t::iterator it, it_end = a_vecid2polid->end();
  // iterate over each word, find its new cluster, and reassign if
  // necessary
  for (vid_t vecid = 0; vecid < N; ++vecid) {
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
  pol_t c_id;
  for (auto& p2v: *a_pol2vecids) {
    c_id = p2v.first;
    // sum-up coordinates of all the vectors pertaining to the given
    // polarity
    for (auto& vecid: p2v.second) {
      a_new_centroids->col(c_id) += a_nwe->col(vecid);
    }
    // take the mean of the new centroid
    if (p2v.second.size())
      a_new_centroids->col(c_id) /= float(p2v.second.size());
  }

  return !_cmp_mat(a_new_centroids, a_old_centroids);
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
    // assign new items to their new nearest centroids (can return
    // `ret =` here, but then remove assert from )
    _nc_assign(a_polid2vecids, a_vecid2polid, a_new_centroids, a_nwe);

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
static void _nc_expand(v2p_t *a_vecid2pol, const arma::mat *const a_centroids, \
		       const arma::mat *a_nwe, const int a_N) {
  // vector of word vector ids, their respective polarities (aka
  // nearest centroids), and distances to the nearest centroids
  vpd_v_t vpds;
  vpds.reserve(a_vecid2pol->size());

  dist_t idist;
  pol_t ipol;
  int j = 0;
  v2p_t::const_iterator v2p_end = a_vecid2pol->end();
  // populate
  for (unsigned int i = 0; i < a_nwe->n_cols; ++i) {
    if (a_vecid2pol->find(i) != v2p_end)
      continue;

    // obtain polarity class and minimum distance to the nearest
    // centroid
    ipol = _nc_find_cluster(a_centroids, a_nwe->colptr(i), &idist);

    // add new element to the vector
    vpds.push_back(VPD {idist, ipol, i});
    ++j;
  }
  _add_terms(a_vecid2pol, &vpds, j, a_N);
}

void expand_nearest_centroids(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N, \
			      const bool a_early_break) {
  // determine the number of clusters
  size_t n_clusters = static_cast<size_t>(Polarity::MAX_SENTINEL);
  // create two matrices for storing centroids
  arma::mat *centroids = new arma::mat(a_nwe->n_rows, n_clusters);
  arma::mat *new_centroids = new arma::mat(a_nwe->n_rows, n_clusters);

  // populate intial clusters
  pol_t polid;
  pi2v_t polid2vecids;
  polid2vecids.reserve(n_clusters);
  v2pi_t vecid2polid;
  vecid2polid.reserve(a_nwe->n_cols);
  for (auto &v2p: *a_vecid2pol) {
    polid = static_cast<size_t>(v2p.second);
    polid2vecids[polid].insert(v2p.first);
    vecid2polid[v2p.first] = polid;
  }
  int i = 0;
  // run the algorithm until convergence
  while (_nc_run(centroids, new_centroids, &polid2vecids, &vecid2polid, a_nwe)) {
    std::cerr << "Run #" << i++ << '\r';
    // early break
    if (a_early_break) {
      *centroids = *new_centroids;
      break;
    }
    // swap the centroids
    std::swap(centroids, new_centroids);
  }
  std::cerr << std::endl;

  // add new terms to the polarity sets based on their distance to the
  // centroids (centroids and new centroids should contain identical
  // values here)
  assert(_cmp_mat(centroids, new_centroids));
  _nc_expand(a_vecid2pol, new_centroids, a_nwe, a_N);

  delete new_centroids;
  delete centroids;
}

/**
 * Find K known neighbors nearest to the vector `a_vid`
 *
 * @param a_vid - id of the vector whose neighbors should be found
 * @param a_nwe - matrix of neural word embeddings
 * @param a_vecid2pol - map of vector id's with known polarities
 * @param a_knn - vector for storing K earest neighbors
 * @param a_K - number of nearest neighbors to use
 *
 * @return \c void
 */
static void _knn_find_nearest(vid_t a_vid, const arma::mat *a_nwe, const v2p_t * const a_vecid2pol, \
			      vpd_pq_t *a_knn, const int a_K) {
  bool filled = false;
  int added = 0;

  // reset KNN vector
  while (! a_knn->empty()) {
    a_knn->pop();
  }

  dist_t idistance, mindistance = DBL_MAX;
  const dist_t *ivec = a_nwe->colptr(a_vid);
  const size_t n_rows = a_nwe->n_rows;
  // iterate over each known vector and find K nearest ones
  for (auto& v2p: *a_vecid2pol) {
    // compute distance between
    idistance = _unnorm_eucl_distance(ivec, a_nwe->colptr(v2p.first), n_rows);

    if (idistance >= mindistance && filled)
      continue;

    // check if container is full and pop one element if necessary
    if (filled)
      a_knn->pop();
    else
      filled = (++added == a_K);

    a_knn->push(VPD {idistance, static_cast<pol_t>(v2p.second), v2p.first});
    mindistance = a_knn->top().m_distance;
  }
}

/**
 * Compute most probable polarity class for given item from its K neighbors
 *
 * @param a_vpd - element in which to store the result
 * @param a_vid - id of the vector in question
 * @param a_knn - priority queue of K nearest neighbors
 * @param a_workbench - workbench for constructing polarities
 *
 * @return \c void
 */
static void _knn_add(vpd_t *a_vpd, const vid_t a_vid, vpd_pq_t *a_knn, \
		     vpd_v_t *a_workbench) {

  // reset workbench
  for (auto& vpd: *a_workbench) {
    vpd.m_vecid = 0;		// will serve as neighbor counter
    vpd.m_distance = 0.;	// will store the sum of the distances
  }

  const vpd_t *vpd;
  // iterate over neighbors
  while (! a_knn->empty()) {
    vpd = &a_knn->top();
    ++(*a_workbench)[vpd->m_polarity].m_vecid;
    (*a_workbench)[vpd->m_polarity].m_distance += vpd->m_distance;
    a_knn->pop();
  }

  dist_t idistance, maxdistance = 0.;
  pol_t pol, maxpol = static_cast<pol_t>(Polarity::MAX_SENTINEL);
  for (pol_t ipol = 0; ipol < maxpol; ++ipol) {
    if ((*a_workbench)[ipol].m_distance == 0)
      continue;

    // square the number of neighbors for that class and divide by the distance
    idistance = (*a_workbench)[ipol].m_vecid;
    idistance *= idistance;
    idistance /= (*a_workbench)[ipol].m_distance;

    if (idistance > maxdistance) {
      maxdistance = idistance;
      pol = ipol;
    }
  }
  assert(pol != static_cast<pol_t>(Polarity::MAX_SENTINEL));
  *a_vpd = VPD {maxdistance, pol, a_vid};
}

void expand_knn(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N, const int a_K) {
  vpd_v_t vpds;
  vpds.reserve(a_nwe->n_cols);
  vpd_v_t _knn(a_K);
  vpd_pq_t knn(_knn.begin(), _knn.end());
  vpd_v_t workbench(static_cast<size_t>(Polarity::MAX_SENTINEL));

  vpd_t ivpd;
  size_t i = 0;
  v2p_t::const_iterator v2p_end = a_vecid2pol->end();

  // iterate over each word vector and find k-nearest neigbors for it
  for (vid_t vid = 0; vid < a_nwe->n_cols; ++vid) {
    a_vecid2pol->find(vid);

    // skip vector if its polarity is already known
    if (a_vecid2pol->find(vid) != v2p_end)
      continue;

    _knn_find_nearest(vid, a_nwe, a_vecid2pol, &knn, a_K);
    _knn_add(&vpds[i++], vid, &knn, &workbench);
  }
  _add_terms(a_vecid2pol, &vpds, i, a_N);
}

static vid_t _pca_find_best_pc(const v2p_t *a_vecid2pol, const arma::mat *a_prjctd) {
  return 0;
}

void expand_pca(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N, \
		const bool a_use_means, const bool a_get_best_pc) {
  // obtain matrix of vectors with known polarities
  vid_t i = 0;
  arma::mat pol_mtx (a_nwe->n_rows, a_vecid2pol->size());
  for (auto &v2p: *a_vecid2pol) {
    pol_mtx.col(i++) = a_nwe->col(v2p.first);
  }
  // obtain PCA coefficients and project the data
  arma::mat pca_coeff, prjctd;
  arma::princomp(pca_coeff, prjctd, pol_mtx.t());
  // look for the principal component with the maximum deviation for
  // polarity, if asked to
  vid_t trg_pc = 0;
  if (a_get_best_pc)
    trg_pc = _pca_find_best_pc(a_vecid2pol, &prjctd);

}

void expand_projected(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}

void expand_linear_transform(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}
