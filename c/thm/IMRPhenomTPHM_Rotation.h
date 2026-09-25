/*
 * LAL-free IMRPhenomTPHM mode-rotation interface.
 * Copyright (C) 2020 Hector Estelles
 * Port and subsequent modifications: Copyright (C) 2026 Neil Cornish
 * SPDX-License-Identifier: GPL-2.0-or-later
 * See the GNU GPL for copying terms and warranty disclaimer.
 */

#ifndef IMRPHENOMTPHM_ROTATION_H
#define IMRPHENOMTPHM_ROTATION_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMRPHENOMTPHM_MIN_ELL 2
#define IMRPHENOMTPHM_MAX_ELL 5
#define IMRPHENOMTPHM_MAX_M_COUNT (2*IMRPHENOMTPHM_MAX_ELL + 1)

/*
 * A rotation within one ell multiplet.  Rows are output/inertial m and
 * columns are input/co-precessing m'.  Only the leading (2*ell+1)^2 entries
 * are active; m and m' are stored at indices m+ell and m'+ell.
 */
typedef struct
{
    int ell;
    double complex element[IMRPHENOMTPHM_MAX_M_COUNT]
                          [IMRPHENOMTPHM_MAX_M_COUNT];
} IMRPhenomTPHMRotation;

typedef struct
{
    int ell;
    int abs_emm;
    /* Coefficients multiplying the +m and -m co-precessing modes. */
    double complex positive;
    double complex negative;
} IMRPhenomTPHMFoldedProjection;

/*
 * Build the active rotation
 *
 *   h^I_lm = sum_m' R^l_{m m'} h^cop_lm'
 *
 * using the IMRPhenomTPHM convention
 *
 *   R^l_{m m'} = exp(-i m alpha) d^l_{m' m}(beta)
 *                 exp(-i m' gamma).
 *
 * These are the physical Euler angles of Eqs. (4)--(7) of
 * arXiv:2105.05872.  The LAL helper reverses alpha and gamma at one call site;
 * this interface deliberately does not expose that implementation detail.
 */
int IMRPhenomTPHMBuildRotation(int ell,
                               double alpha,
                               double beta,
                               double gamma,
                               IMRPhenomTPHMRotation *rotation);

/* out = left * right, so right acts first. */
int IMRPhenomTPHMComposeRotations(const IMRPhenomTPHMRotation *left,
                                  const IMRPhenomTPHMRotation *right,
                                  IMRPhenomTPHMRotation *out);

int IMRPhenomTPHMAdjointRotation(const IMRPhenomTPHMRotation *rotation,
                                 IMRPhenomTPHMRotation *adjoint);

int IMRPhenomTPHMRotateMultipole(const IMRPhenomTPHMRotation *rotation,
                                 const double complex *input,
                                 double complex *output);

/* Spin-weight -2 spherical harmonic in the convention used by THM TDI. */
double complex IMRPhenomTPHMSpinWeightedYMinus2(int ell,
                                                int emm,
                                                double theta,
                                                double phi);

/*
 * Compose a rotation with the observer projection.  The returned coefficient
 * obeys
 *
 *   hplus - i hcross = sum_m' coefficient[m'+ell] h^cop_lm'.
 *
 * Polarization is applied as exp(2 i psi), matching PhenomTHM_TDI.c.
 */
int IMRPhenomTPHMBuildProjection(const IMRPhenomTPHMRotation *rotation,
                                 double theta,
                                 double phi,
                                 double psi,
                                 double complex *coefficient);

int IMRPhenomTPHMProjectMultipole(const double complex *modes,
                                  int ell,
                                  double theta,
                                  double phi,
                                  double psi,
                                  double complex *strain);

int IMRPhenomTPHMBuildFoldedProjection(
    const IMRPhenomTPHMRotation *rotation,
    int abs_emm,
    double theta,
    double phi,
    double psi,
    IMRPhenomTPHMFoldedProjection *projection);

/*
 * Evaluate one aligned-spin +/-m co-precessing pair using
 * h_lm=A exp(-i phase) and h_l,-m=(-1)^l conj(h_lm).
 */
double complex IMRPhenomTPHMEvaluateFoldedPair(
    const IMRPhenomTPHMFoldedProjection *projection,
    double amplitude,
    double phase);

/*
 * Fast projection of the populated co-precessing +/-m pairs through the
 * time-dependent coprecessing-to-J rotation.  j_projection is formed once
 * with IMRPhenomTPHMBuildProjection(j_to_l0, ...); it therefore already
 * combines the constant J-to-L0 rotation with the observer projection.
 *
 * The low-ell implementation evaluates the finite Wigner-d polynomials for
 * ell=2,...,5 directly from powers of sin(beta/2) and cos(beta/2).  No full
 * rotation matrix is constructed.  positive_mode[p] is h_{ell,+abs_m[p]};
 * the corresponding negative-m mode is supplied by
 * h_{ell,-m}=(-1)^ell conj(h_{ell,+m}).
 */
int IMRPhenomTPHMProjectFoldedPairsFast(
    int ell,
    double alpha,
    double beta,
    double gamma,
    const double complex *j_projection,
    int npairs,
    const int *abs_m,
    const double complex *positive_mode,
    double complex *strain);

#ifdef __cplusplus
}
#endif

#endif
