//////////////
// Includes //
//////////////
#include "expansion.h"

#include <algorithm>		// std::swap(), std::sort()
#include <cassert>		// assert
#include <cfloat>		// DBL_MAX
#include <cmath>		// sqrt()
#include <cstdlib>		// size_t
#include <cstdint>		// MAX_SIZE
#include <iostream>		// std::cerr
#include <unordered_set>	// std::unordered_set

///////////
// Types //
///////////

/** Integral type for polarity id */
typedef size_t pol_t;

/** Set of NWE id's */
typedef std::unordered_set<vid_t> vids_t;

/** Map from polarity id to a set of respective NWE id's */
typedef std::unordered_map<pol_t, vids_t> pi2v_t;

/** Map from NWE id to polarity id */
typedef std::unordered_map<vid_t, pol_t> v2pi_t;

/** 3-tuple of vector id, vector polarity, and its distance to the
    nearest centroid */
typedef struct VPD {
  dist_t m_distance {};
  pol_t m_polarity {};
  vid_t m_vecid {};

  VPD(void) {}

  VPD(dist_t a_distance, pol_t a_polarity, vid_t a_vecid):
    m_distance{a_distance}, m_polarity{a_polarity}, m_vecid{a_vecid}
  {}
} vpd_t;

/** vector of 3-tuples of vector id, vector polarity, and distctance
    to the centroid */
typedef std::vector<vpd_t> vpd_v_t;

/////////////
// Methods //
/////////////
void expand_knn(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}

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
  pol_t j, ret = 0;
  const double *centroid;
  dist_t tmp_j, idistance, mindistance = DBL_MAX;
  for (size_t i = 0; i < a_centroids->n_cols; ++i) {
    idistance = 0.;
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
      // std::cerr << "word vector = " << a_nwe->col(vecid) << std::endl;
      a_new_centroids->col(c_id) += a_nwe->col(vecid);
      // std::cerr << "centroid = " << a_new_centroids->col(c_id) << std::endl;
    }
    // std::cerr << "b) centroid #" << c_id << ": " << a_new_centroids->col(c_id) << std::endl;
    // std::cerr << "   p2v.second.size() =" << p2v.second.size() << std::endl;
    // take the mean of the new centroid
    a_new_centroids->col(c_id) /= float(p2v.second.size());
    // std::cerr << "c) centroid #" << c_id << ": " << a_new_centroids->col(c_id) << std::endl;
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
  // sort vpds according to their distances to the centroids
  std::sort(vpds.begin(), vpds.end(), [](const vpd_t& vpd1, const vpd_t& vpd2) \
	    {return vpd1.m_distance < vpd2.m_distance;});
  // add new terms to the dictionary
  vpd_t *ivpd;
  for (int i = 0; (a_N < 0 || i < a_N) && i < j; ++i) {
    ivpd = &vpds[i];
    a_vecid2pol->emplace(ivpd->m_vecid, static_cast<Polarity>(ivpd->m_polarity));
  }
}

void expand_nearest_centroids(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {
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

void expand_projected(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}

void expand_projected_length(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}

void expand_linear_transform(v2p_t *a_vecid2pol, const arma::mat *a_nwe, const int a_N) {

}
