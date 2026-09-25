/*
 * LAL-free numerical IMRPhenomTPHM precession interface.
 * Copyright (C) 2020 Hector Estelles
 * Port and subsequent modifications: Copyright (C) 2026 Neil Cornish
 * SPDX-License-Identifier: GPL-2.0-or-later
 * See the GNU GPL for copying terms and warranty disclaimer.
 */

#ifndef IMRPHENOMTPHM_PRECESSION_H
#define IMRPHENOMTPHM_PRECESSION_H

#include "IMRPhenomTHM.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    /*
     * Dimensionless spin vectors at tau_ref in the L0 frame.  Their z
     * components must equal the aligned spins used to initialize carrier.
     */
    double chi1[3];
    double chi2[3];
    /* Dimensionless source time tau=t/M; tau=0 is the 22 amplitude peak. */
    double tau_ref;
    /* Set non-positive values to use the defaults below. */
    double absolute_tolerance;
    double relative_tolerance;
    double initial_step;
} IMRPhenomTPHMPrecessionConfig;

typedef struct
{
    double final_spin;
    double aligned_final_spin_at_peak;
    double ringdown_alpha_slope;
    /* Numerical-TPHM Euler-angle gauge at the reference epoch. */
    double alpha_reference;
    double gamma_reference;
    /* J-frame basis vectors expressed in the input L0 frame. */
    double j_frame_x[3];
    double j_frame_y[3];
    double j_frame_z[3];
} IMRPhenomTPHMPrecessionSummary;

/*
 * Evolve the orbit-averaged PN spin equations used by numerical TPHM.
 *
 * The carrier supplies v(t)=[omega_22(t)/2]^(1/3).  The spin vectors are
 * specified at tau_ref in the L0 frame, where Lhat(tau_ref)=(0,0,1).
 * Output angles rotate co-precessing modes into the J frame using
 *
 *   alpha = atan2(Lhat_y,Lhat_x),  beta = acos(Lhat_z),
 *   dgamma = -cos(beta) dalpha.
 *
 * tau must be strictly increasing.  The PN evolution is used through tau=0;
 * later samples use the QNM Euler-angle continuation.  The output arrays are
 * caller-owned and may be sampled sparsely or nonuniformly.
 */
int IMRPhenomTPHMEvolveEulerAngles(
    const IMRPhenomTHM *carrier,
    const IMRPhenomTPHMPrecessionConfig *config,
    int n,
    const double *tau,
    double *alpha,
    double *beta,
    double *gamma,
    IMRPhenomTPHMPrecessionSummary *summary);

/* Reproduce the sampled reference anchoring, endpoint continuation, and
 * gamma-integration conventions used by LAL's numerical TPHM implementation.
 * In particular, LAL writes the reference state into two neighboring samples;
 * this makes the result depend on the uniform waveform cadence.  This entry
 * point is intended for validation, not as the default continuous evolution. */
int IMRPhenomTPHMEvolveEulerAnglesLALCompatible(
    const IMRPhenomTHM *carrier,
    const IMRPhenomTPHMPrecessionConfig *config,
    int n,
    const double *tau,
    double *alpha,
    double *beta,
    double *gamma,
    IMRPhenomTPHMPrecessionSummary *summary);

#ifdef __cplusplus
}
#endif

#endif
