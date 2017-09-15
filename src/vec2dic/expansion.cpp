//////////////
// Includes //
//////////////
#include "src/vec2dic/expansion.h"

#include <cassert>                      // assert

#include <algorithm>                    // std::swap(), std::sort()
#include <cstdlib>                      // size_t
#include <iostream>                     // std::cerr
#include <cmath>			// fabs(), sqrt()
#include <unordered_set>                // std::unordered_set
#include <queue>                        // std::priority_queue

///////////
// Types //
///////////

/** Integral type for polarity id */
using pol_t = size_t;

/** Set of NWE id's */
using vids_t = std::unordered_set<vid_t>;

/** Map from polarity id to a set of respective NWE id's */
using pi2v_t = std::unordered_map<size_t, vids_t>;

/** Map from NWE id to polarity id */
using v2pi_t = std::unordered_map<vid_t, size_t>;

/** 3-tuple of vector id, vector polarity, and its distance to the
    nearest centroid */
using vpd_t = struct VPD {
  vid_t m_vecid {};
  pol_t m_polarity {};
  dist_t m_distance {};

  VPD(void) {}

  VPD(vid_t a_vecid, pol_t a_polarity, dist_t a_distance):
    m_vecid{a_vecid}, m_polarity{a_polarity}, m_distance{a_distance}
  {}

  // be careful with this operator as it only makes sense for sorting
  // centroid vectors (since vectors that are closer to centroids will
  // be at the beginning of the list)
  bool operator <(const VPD& rhs) const {
    return m_distance < rhs.m_distance;
  }

  bool operator >(const VPD& rhs) const {
    return m_distance > rhs.m_distance;
  }

  friend std::ostream& operator<<(std::ostream &ostream, const struct VPD vpd) {
    ostream << "VPD { vecid: " << vpd.m_vecid
    << "; polarity: " << vpd.m_polarity
    << "; distance: " << vpd.m_distance << '}';
    return ostream;
  }
};

/** vector of 3-tuples of vector id, vector polarity, and distctance
    to the centroid */
using vpd_v_t = std::vector<vpd_t>;

/** set of 3-tuples of vector id, vector polarity, and distctance
    to the centroid */
using pd_t = std::pair<pol_t, dist_t>;
using vpd_m_t = std::unordered_map<vid_t, pd_t>;

/** priority queue of 3-tuples of vector id, vector polarity, and
    distctance to the centroid */
using vpd_pq_t = std::priority_queue<vpd_t>;

/** struct comprising means and variances of polarity vectors  */
using pol_stat_t = struct {
  // dimension with the biggest distance between subjective and
  // neutral terms
  vid_t m_subj_dim = 0;
  // dimension with the biggest distance between positive and negative
  // terms
  vid_t m_pol_dim = 0;

  size_t m_n_pos = 0;		// number of positive vectors
  dist_t m_pos_mean = 0.;	// mean of positive vectors

  size_t m_n_neg = 0;		// number of negative vectors
  dist_t m_neg_mean = 0.;	// mean of negative vectors

  size_t m_n_neut = 0;		// number of neutral vectors
  dist_t m_neut_mean = 0.;	// mean of neutral vectors

  size_t m_n_subj = 0;		// number of subjective vectors
  dist_t m_subj_mean = 0.;	// mean of subjective vectors

  void reset() {
    m_subj_dim = 0;
    m_pol_dim = 0;
    m_n_pos = 0;
    m_pos_mean = 0.;
    m_n_neg = 0;
    m_neg_mean = 0.;
    m_n_neut = 0;
    m_neut_mean = 0.;
    m_n_subj = 0;
    m_subj_mean = 0.;
  }
};

///////////////
// Constants //
///////////////
const double DFLT_ALPHA = 1e-5;
const double DFLT_DELTA = 1e-10;
const int MAX_ITERS = 1e6;

const vid_t POS_VID = static_cast<vid_t>(POSITIVE);
const vid_t NEG_VID = static_cast<vid_t>(NEGATIVE);
const vid_t NEUT_VID = static_cast<vid_t>(NEUTRAL);

const bool debug = false;

const size_t POS_IDX = 0, NEG_IDX = 1, NEUT_IDX = 2, SUBJ_IDX = 3;
const size_t POLID2IDX[] = {10, POS_IDX, NEG_IDX, SUBJ_IDX, NEUT_IDX};
const pol_t IDX2POLID[] = {1, 2, 4, 10, 10};

/////////////
// Methods //
/////////////

/**
 * Check whether two matrices are equal.
 *
 * @param a_mat1 - 1-st matrix to compare
 * @param a_mat2 - 2-nd matrix to compare
 *
 * @return \c true if both matrices are equal, \c false otherwise
 */
static bool _cmp_mat(const arma::mat *a_mat1, const arma::mat *a_mat2) {
  bool ret = ((a_mat1->n_rows == a_mat2->n_rows) \
              && (a_mat1->n_cols == a_mat2->n_cols));
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
static inline dist_t _unnorm_eucl_distance(const dist_t *a_vec1,
                                           const dist_t *a_vec2,
                                           size_t a_N) {
  dist_t tmp_i = 0., idistance = 0;

  for (size_t i = 0; i < a_N; ++i) {
    tmp_i = a_vec1[i] - a_vec2[i];
    idistance += tmp_i * tmp_i;
  }
  return idistance;
}

/**
 * Add newly extracted terms to the polarity lexicon.
 *
 * @param a_vecid2pol - target dictionary mapping vector id's to polarities
 * @param a_vpds - source vector of NWE ids, their polarities, and distances
 * @param a_j - actual number of new terms
 * @param a_N - maximum number of terms to extract (-1 means unlimited)
 *
 * @return \c void
 */
static inline void _add_terms(v2ps_t *a_vecid2pol,
                              vpd_v_t *a_vpds,
                              const int a_j, const int a_N) {
  // sort vpds according to their distances to the centroids
  std::sort(a_vpds->begin(), a_vpds->end());
  // add new terms to the dictionary
  vpd_t *ivpd;
  // `i' is the actual number of added terms and `j' is the total
  // counter
  for (int i = 0, j = 0; (a_N < 0 || i < a_N) && j < a_j; ++j) {
    ivpd = &(*a_vpds)[j];

    if (ivpd->m_polarity == NEUTRAL)
      continue;

    a_vecid2pol->emplace(ivpd->m_vecid,
			 std::make_pair(
					static_cast<Polarity>(ivpd->m_polarity),
					ivpd->m_distance));
    ++i;
  }
}

/**
 * Find cluster whose centroid is nearest to the given word vector.
 *
 * @param a_centroids - matrix of pre-computed cluster centroids
 * @param a_vec - word vector whose nearest cluster should be found
 * @param a_dist - (optional) pointer to a variable in which actual
 *                  Euclidean distance to the vector should be stored
 * @return id of the cluster with the nearest centroid
 */
static pol_t _nc_find_cluster(const arma::mat *a_centroids,
                              const double *a_vec,
                              dist_t *a_dist = nullptr) {
  pol_t ret = 0;
  const double *centroid;
  dist_t idistance = 1., mindistance = std::numeric_limits<double>::max();
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
static bool _nc_compute_centroids(arma::mat *a_new_centroids,
                                  const arma::mat *a_old_centroids,
                                  const pi2v_t *a_pol2vecids,
                                  const arma::mat *a_nwe) {
  // zero-out new centroids
  a_new_centroids->zeros();
  // compute new centroids
  pol_t c_id;
  for (auto& p2v : *a_pol2vecids) {
    c_id = p2v.first;
    // sum-up coordinates of all the vectors pertaining to the given
    // polarity
    for (auto& vecid : p2v.second) {
      a_new_centroids->col(c_id) += a_nwe->col(vecid);
    }
    // take the mean of the new centroid
    if (p2v.second.size())
      a_new_centroids->col(c_id) /= static_cast<float>(p2v.second.size());
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
static inline bool _nc_run(arma::mat *a_new_centroids,
                           arma::mat *a_old_centroids,
                           pi2v_t *a_polid2vecids,
                           v2pi_t *a_vecid2polid,
                           const arma::mat *a_nwe) {
  bool ret = false;
  // calculate centroids
  if ((ret = _nc_compute_centroids(a_new_centroids,
                                   a_old_centroids,
                                   a_polid2vecids, a_nwe)))
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
static void _nc_expand(v2ps_t *a_vecid2pol, const arma::mat *const a_centroids,
                       const arma::mat *a_nwe, const int a_N) {
  // vector of word vector ids, their respective polarities (aka
  // nearest centroids), and distances to the nearest centroids
  vpd_v_t vpds;
  vpds.reserve(a_vecid2pol->size());

  dist_t idist;
  size_t pol_idx;
  pol_t pol_i;
  int j = 0;
  v2ps_t::const_iterator v2p_end = a_vecid2pol->end();
  // populate
  for (vid_t i = 0; i < a_nwe->n_cols; ++i) {
    if (a_vecid2pol->find(i) != v2p_end)
      continue;

    // obtain polarity class and minimum distance to the nearest
    // centroid
    pol_idx = _nc_find_cluster(a_centroids, a_nwe->colptr(i), &idist);
    pol_i = IDX2POLID[pol_idx];

    // by default, all polarities are shifted by one
    if (pol_i == NEUTRAL)
      continue;

    // add new element to the vector
    vpds.push_back(VPD {i, pol_i, idist});
    ++j;
  }
  _add_terms(a_vecid2pol, &vpds, j, a_N);
}

void expand_nearest_centroids(v2ps_t *a_vecid2pol,
                              const arma::mat *a_nwe,
                              const int a_N,
                              const bool a_early_break) {
  // create two matrices for storing centroids
  arma::mat *centroids = new arma::mat(a_nwe->n_rows, N_POLARITIES);
  arma::mat *new_centroids = new arma::mat(a_nwe->n_rows, N_POLARITIES);

  // populate intial clusters
  pol_t polid;
  size_t pol_idx;
  pi2v_t pol_idx2vecids;
  pol_idx2vecids.reserve(N_POLARITIES);
  v2pi_t vecid2pol_idx;
  vecid2pol_idx.reserve(a_nwe->n_cols);

  for (auto &v2p : *a_vecid2pol) {
    polid = static_cast<pol_t>(v2p.second.first);
    pol_idx = POLID2IDX[polid];

    pol_idx2vecids[pol_idx].insert(v2p.first);
    vecid2pol_idx[v2p.first] = pol_idx;
  }
  int i = 0;
  // run the algorithm until convergence
  while (_nc_run(centroids, new_centroids,
                 &pol_idx2vecids, &vecid2pol_idx, a_nwe)) {
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
static void _knn_find_nearest(vid_t a_vid,
                              const arma::mat *a_nwe,
                              const v2ps_t * const a_vecid2pol,
                              vpd_pq_t *a_knn, const int a_K) {
  int added = 0;
  bool filled = false;

  // reset KNN vector
  while (!a_knn->empty()) {
    a_knn->pop();
  }

  const dist_t *ivec = a_nwe->colptr(a_vid);
  const size_t n_rows = a_nwe->n_rows;
  dist_t idistance, mindistance = std::numeric_limits<dist_t>::max();
  // iterate over each known vector and find K nearest ones
  for (auto& v2p : *a_vecid2pol) {
    // compute distance between
    idistance = _unnorm_eucl_distance(ivec, a_nwe->colptr(v2p.first), n_rows);

    if (idistance >= mindistance && filled)
      continue;

    // check if container is full and pop one element if necessary
    if (filled)
      a_knn->pop();
    else
      filled = (++added == a_K);

    a_knn->push(VPD {v2p.first, POLID2IDX[v2p.second.first], idistance});
    mindistance = a_knn->top().m_distance;
  }
}

/**
 * Compute most probable polarity class from K neighbors.
 *
 * @param a_vpd - element in which to store the result
 * @param a_vid - id of the vector in question
 * @param a_knn - priority queue of K nearest neighbors
 * @param a_workbench - workbench for constructing polarities
 *
 * @return \c void
 */
static void _knn_add(vpd_t *a_vpd, const vid_t a_vid,
                     vpd_pq_t *a_knn, vpd_v_t *a_workbench) {
  // reset the workbench
  for (auto& vpd : *a_workbench) {
    vpd.m_vecid = 0;        // will serve as neighbor counter
    vpd.m_distance = 0.;    // will store the sum of the distances
  }

  const vpd_t *vpd;
  // iterate over neighbors
  while (!a_knn->empty()) {
    vpd = &a_knn->top();
    ++(*a_workbench)[vpd->m_polarity].m_vecid;
    (*a_workbench)[vpd->m_polarity].m_distance += vpd->m_distance;
    a_knn->pop();
  }

  pol_t pol = 0;
  dist_t idistance, mindistance = std::numeric_limits<dist_t>::max();
  for (pol_t ipol = 0; ipol < N_POLARITIES; ++ipol) {
    if ((*a_workbench)[ipol].m_distance == 0)
      continue;

    // divide the total distance to the nearest neighbors of that
    // class by the square number
    idistance = (*a_workbench)[ipol].m_vecid;
    idistance *= idistance;
    idistance = (*a_workbench)[ipol].m_distance / idistance;

    if (idistance < mindistance) {
      mindistance = idistance;
      pol = IDX2POLID[ipol];
    }
  }
  *a_vpd = VPD {a_vid, pol, mindistance};
}

void expand_knn(v2ps_t *a_vecid2pol,
                const arma::mat *a_nwe,
                const int a_N, const int a_K) {
  vpd_v_t vpds;
  vpds.reserve(a_nwe->n_cols);
  vpd_v_t _knn(a_K);
  vpd_pq_t knn(_knn.begin(), _knn.end());
  vpd_v_t workbench(N_POLARITIES);

  vpd_t ivpd;
  size_t i = 0;
  v2ps_t::const_iterator v2p_end = a_vecid2pol->end();

  // iterate over each word vector and find k-nearest neigbors for it
  for (vid_t vid = 0; vid < a_nwe->n_cols; ++vid) {
    // skip vector if its polarity is already known
    if (a_vecid2pol->find(vid) != v2p_end)
      continue;

    _knn_find_nearest(vid, a_nwe, a_vecid2pol, &knn, a_K);
    _knn_add(&vpds[i++], vid, &knn, &workbench);
  }
  _add_terms(a_vecid2pol, &vpds, i, a_N);
}

/**
 *  Divide column vector by int unless int is zero.
 *
 *  @param mtx - matrix whose column should be normalized
 *  @param col_idx - index of the column to be normalized
 *  @param int - normalization factor
 *
 *  @return void
 *
 *  @note modifies `col` in place
 */
static void _divide(arma::mat *mtx, size_t col_idx, const int n) {
  if (n != 0)
    mtx->col(col_idx) /= n;
}

/**
 * Compute means of polarity vectors on the dimension with the biggest
 * deviation.
 *
 * @param a_vecid2polscore - mapping from vector id's to polarities
 *   and scores
 * @param a_prjctd - NWE matrix projected on the PCA space
 *
 * @return \c void
 */
static arma::mat _pca_compute_means(const v2ps_t *a_vecid2polscore,
				    const arma::mat *a_prjctd,
				    pol_stat_t *a_pol_stat) {
  a_pol_stat->reset();

  // means of polarity vectors (the `+ 1` column is reserved for
  // subjective terms)
  arma::mat pol_means(a_prjctd->n_cols, N_POLARITIES + 1);
  pol_means.zeros();

  // obtain unnormalized means of polarity vectors
  size_t pol_idx;
  for (auto &v2p : *a_vecid2polscore) {
    pol_idx = POLID2IDX[v2p.second.first];
    switch (pol_idx) {
    case POS_IDX:
      ++a_pol_stat->m_n_pos;
      break;
    case NEG_IDX:
      ++a_pol_stat->m_n_neg;
      break;
    default:
      ++a_pol_stat->m_n_neut;
      break;
    }
    pol_means.col(pol_idx) += a_prjctd->row(v2p.first).t();
  }
  a_pol_stat->m_n_subj = a_pol_stat->m_n_pos + a_pol_stat->m_n_neg;
  pol_means.col(SUBJ_IDX) = pol_means.col(POS_IDX) + pol_means.col(NEG_IDX);

  _divide(&pol_means, POS_IDX, a_pol_stat->m_n_pos);
  _divide(&pol_means, NEG_IDX, a_pol_stat->m_n_neg);
  _divide(&pol_means, NEUT_IDX, a_pol_stat->m_n_neut);
  _divide(&pol_means, SUBJ_IDX, a_pol_stat->m_n_subj);

  return pol_means;
}

 /**
 * Find axis with the biggest difference between vectors with opposite
 * polarities.
 *
 * @param mtx - embedding matrix
 * @param a_vecid2pol - mapping from vector id's to polarities and scores
 *
 * @return
 */
static vid_t _pca_find_axis(const arma::mat *a_mtx,
			    const v2ps_t *a_vecid2polscore,
			    const size_t a_idx1,
			    const size_t a_idx2) {
  arma::vec axis(a_mtx->n_cols, arma::fill::zeros);
  arma::vec vec1(a_mtx->n_cols);

  vid_t vec_id;
  size_t pol_idx;
  for (auto &v2p_1 : *a_vecid2polscore) {
    pol_idx = POLID2IDX[v2p_1.second.first];
    if (pol_idx != a_idx1)
      continue;

    vec_id = v2p_1.first;
    vec1 = a_mtx->row(vec_id).t();
    for (auto &v2p_2 : *a_vecid2polscore) {
      pol_idx = POLID2IDX[v2p_2.second.first];
      if (pol_idx != a_idx2)
	continue;

      vec_id = v2p_2.first;
      axis += arma::abs(vec1 - a_mtx->row(vec_id).t());
    }
  }

  size_t axis_i = 0;
  size_t n = axis.n_elem;
  dist_t max_delta = std::numeric_limits<dist_t>::min();
  for (size_t i = 0; i < n; ++i) {
    if (axis(i) > max_delta) {
      axis_i = i;
      max_delta = axis(i);
    }
  }
  return axis_i;
}

/**
 * Compute means of polarity vectors on the dimension with the biggest
 * deviation.
 *
 * @param a_vecid2pol - mapping from vector id's to polarities
 * @param a_prjctd - NWE matrix projected on the PCA space
 * @param a_pol_stat - struct comprising statiscs about polarity
 *                     vectors
 *
 * @return \c void
 *
 * @note modifies `pol_stat` in place
 */
static void _pca_find_means_axes(v2ps_t *a_vecid2polscore,
				 const arma::mat *a_prjctd,
				 pol_stat_t *a_pol_stat) {
  arma::mat means = _pca_compute_means(a_vecid2polscore,
				       a_prjctd, a_pol_stat);

  // look for the dimension with the biggest difference between
  // distinct polarity classes
  a_pol_stat->m_subj_dim = _pca_find_axis(a_prjctd, a_vecid2polscore,
					  SUBJ_IDX, NEUT_IDX);
  a_pol_stat->m_pol_dim = _pca_find_axis(a_prjctd, a_vecid2polscore,
					 POS_IDX, NEG_IDX);

  // set means of the polarity classes
  a_pol_stat->m_pos_mean = means(a_pol_stat->m_pol_dim, POS_IDX);
  a_pol_stat->m_neg_mean = means(a_pol_stat->m_pol_dim, NEG_IDX);
  a_pol_stat->m_neut_mean = means(a_pol_stat->m_subj_dim, NEUT_IDX);
  a_pol_stat->m_subj_mean = means(a_pol_stat->m_subj_dim, SUBJ_IDX);
}

/**
 * Expand polarity sets by adding terms that are farthermost to the
 * mean of neutral vectors
 *
 * @param a_vecid2pol - mapping from vector id's to polarities
 * @param a_pca_nwe - neural-word embedding matrix projected onto the
 *                    PCA space
 * @param a_pol_stat - struct comprising statiscs about polarity
 *                     vectors
 * @param a_N - maximum number of terms to extract (-1 means unlimited)
 *
 * @return \c void
 */
static void _pca_expand(v2ps_t *a_vecid2pol, const arma::mat *a_pca_nwe, \
			const pol_stat_t *a_pol_stat, const int a_N) {
  vpd_v_t vpds;
  vpds.reserve(a_pca_nwe->n_rows - a_vecid2pol->size());

  // find maximum values of subjective and polar scores
  vid_t subj_dim = a_pol_stat->m_subj_dim;
  vid_t pol_dim = a_pol_stat->m_pol_dim;

  arma::vec subj_scores = a_pca_nwe->col(subj_dim);
  arma::vec pol_scores = a_pca_nwe->col(pol_dim);

  dist_t origin_subj = (a_pol_stat->m_neut_mean - a_pol_stat->m_subj_mean) / 2.;
  dist_t max_subj = arma::abs(subj_scores - origin_subj).max();
  dist_t origin_pol = (a_pol_stat->m_pos_mean - a_pol_stat->m_neg_mean) / 2.;
  dist_t max_pol = arma::abs(pol_scores - origin_pol).max();

  int j = 0;
  pol_t pol_i;
  vid_t n = a_pca_nwe->n_rows;
  dist_t subj_score_i, pol_score_i, score_i;
  dist_t neut_delta, subj_delta, pos_delta, neg_delta;
  v2ps_t::const_iterator v2p_end = a_vecid2pol->end();
  // populate (since we are sorting the terms in the ascending order
  // of their distances, we use negative values here)
  for (vid_t i = 0; i < n; ++i) {
    if (a_vecid2pol->find(i) != v2p_end)
      continue;

    // determine subjectivity score
    neut_delta = fabs(subj_scores(i) - a_pol_stat->m_neut_mean);
    subj_delta = fabs(subj_scores(i) - a_pol_stat->m_subj_mean);

    if (neut_delta > subj_delta) {
      subj_score_i = 1 + fabs(subj_scores(i) - origin_subj) / max_subj;
    } else {
      subj_score_i = 1 - fabs(subj_scores(i) - origin_subj) / max_subj;
    }

    // determine polarity score
    pos_delta = fabs(pol_scores(i) - a_pol_stat->m_pos_mean);
    neg_delta = fabs(pol_scores(i) - a_pol_stat->m_neg_mean);

    if (pos_delta > neg_delta)
      pol_i = NEGATIVE;
    else
      pol_i = POSITIVE;

    pol_score_i = 1 + fabs(pol_scores(i) - origin_pol) / max_pol;

    score_i = 1000./(subj_score_i + pol_score_i);
    vpds.push_back(VPD {i, pol_i, score_i});
    ++j;
  }
  _add_terms(a_vecid2pol, &vpds, j, a_N);
}

void expand_pca(v2ps_t *a_vecid2polscore,
                const arma::mat *a_nwe, const int a_N) {
  // obtain PCA coordinates for the neural word embeddings data
  arma::mat pca_coeff, prjctd;
  // `a_nwe` elements are stored in column-major format, i.e., each
  // word corresponds to a column. `princomp()`, however, requires
  // that columns represent variables and rows are observations
  // (``Each row of X is an observation and each column is a
  // variable'').  Therefore, we transpose the embedding matrix.
  arma::princomp(pca_coeff, prjctd, a_nwe->t());

  // look for the principal component with the maximum distance
  // between the means of the vectors pertaining to different
  // polarities
  pol_stat_t pol_stat;
  _pca_find_means_axes(a_vecid2polscore, &prjctd, &pol_stat);
  // add new terms
  _pca_expand(a_vecid2polscore, &prjctd, &pol_stat, a_N);
}
