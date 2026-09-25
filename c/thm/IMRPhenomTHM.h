/*
 * LAL-free IMRPhenomTHM interface, based on LALSimulation IMRPhenomTHM.
 * Copyright (C) 2020 Hector Estelles
 * Port and subsequent modifications: Copyright (C) 2026 Neil Cornish
 * SPDX-License-Identifier: GPL-2.0-or-later
 * See the GNU GPL for copying terms and warranty disclaimer.
 */

#ifndef IMRPHENOMTHM_H
#define IMRPHENOMTHM_H

#include <complex.h>

#include "IMRPhenomT.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IMRPHENOMTHM_MAX_MODES 10

typedef struct
{
    int ell;
    int emm;
} IMRPhenomTHMMode;

typedef struct
{
    double amplitude;
    /*
     * Phase convention: hlm = amplitude*exp(-I*phase).  For (2,2), this is
     * the caller-supplied/integrated stripped IMRPhenomT 22 phase.  For
     * higher modes, the complex PN amplitude phase has already been folded
     * into this scalar phase.
     */
    double phase;
    /* Dimensionless angular frequency d(phase)/d(t/M). */
    double omega;
    double complex hlm;
} IMRPhenomTHMModeSample;

typedef struct
{
    int ell;
    int emm;
    double fac0;
    double tshift;
    double alpha1RD;
    double alpha2RD;
    double alpha21RD;
    double alpha1RD_prec;
    double alpha2RD_prec;
    double alpha21RD_prec;
    double c1;
    double c2;
    double c3;
    double c4;
    double c1_prec;
    double c2_prec;
    double c4_prec;
    double mergerC1;
    double mergerC2;
    double mergerC3;
    double mergerC4;
    double omegaCutPNAMP;
    double phiCutPNAMP;
    double ampN;
    double amp0halfPNreal;
    double amp0halfPNimag;
    double amp1PNreal;
    double amp1PNimag;
    double amp1halfPNreal;
    double amp1halfPNimag;
    double amp2PNreal;
    double amp2PNimag;
    double amp2halfPNreal;
    double amp2halfPNimag;
    double amp3PNreal;
    double amp3PNimag;
    double amp3halfPNreal;
    double amp3halfPNimag;
    double amplog;
    double inspC1;
    double inspC2;
    double inspC3;
} IMRPhenomTHMAmpCoeffs;

typedef struct
{
    int emm;
    double omegaRING;
    double omegaRING_prec;
    double omegaPeak;
    double domegaPeak;
    double alpha1RD;
    double alpha2RD;
    double alpha21RD;
    double alpha1RD_prec;
    double c1;
    double c1_prec;
    double c2;
    double c3;
    double c4;
    double omegaMergerC1;
    double omegaMergerC2;
    double omegaMergerC3;
    double phOffMerger;
    double phOffRD;
    int phase_offsets_set;
} IMRPhenomTHMPhaseCoeffs;

typedef struct
{
    IMRPhenomTHMMode requested_mode;
    int ell;
    int emm;
    int abs_emm;
    int is_negative_m;
    int is_zero_by_symmetry;
    double phoff;
    IMRPhenomTHMAmpCoeffs amp;
    IMRPhenomTHMPhaseCoeffs phase;
} IMRPhenomTHMModeState;

typedef struct
{
    double m1;
    double m2;
    double chi1;
    double chi2;
    double Mtot;
    double eta;
    double delta;
    double Shat;
    double dchi;
    double Mfinal;
    /* Aligned-spin fit retained for diagnostics and comparisons. */
    double afinal_aligned;
    /* Active spin used to reconstruct the co-precessing merger-ringdown. */
    double afinal;
    double afinal_prec;
    /* Set after the higher-mode phase offsets have been anchored at
     * tCUT_Freq.  Partitioned waveform grids may then be evaluated without
     * each individual block having to straddle that internal join. */
    int phase_offsets_initialized;
    struct IMRPhenomT mode22;
    /* Reproduce LAL THM's per-mode (2,2) amplitude instead of the legacy
     * stripped-PhenomT amplitude special case. */
    int use_lal_mode22_amplitude;
    int nmodes;
    IMRPhenomTHMModeState modes[IMRPHENOMTHM_MAX_MODES];
} IMRPhenomTHM;

int IMRPhenomTHMInitialize(IMRPhenomTHM *model,
                           double m1,
                           double m2,
                           double chi1,
                           double chi2,
                           const IMRPhenomTHMMode *modes,
                           int nmodes);

/*
 * TPHM initialization after the precession model has supplied the remnant
 * spin magnitude.  The supplied spin is used consistently in the 22
 * backbone and in every higher-mode QNM coefficient.  The ordinary
 * initializer remains the aligned-spin THM path.
 */
int IMRPhenomTHMInitializeWithFinalSpin(IMRPhenomTHM *model,
                                        double m1,
                                        double m2,
                                        double chi1,
                                        double chi2,
                                        double final_dimensionless_spin,
                                        const IMRPhenomTHMMode *modes,
                                        int nmodes);

int IMRPhenomTHMInitializeWithPrecessingFinalSpin(
    IMRPhenomTHM *model,
    double m1,
    double m2,
    double chi1,
    double chi2,
    double precessing_final_dimensionless_spin,
    const IMRPhenomTHMMode *modes,
    int nmodes);

void IMRPhenomTHMDestroy(IMRPhenomTHM *model);

int IMRPhenomTHMDefaultModes(IMRPhenomTHMMode *modes, int max_modes);
int IMRPhenomTHMModeIndex(const IMRPhenomTHM *model, int ell, int emm);

/* Direct accessors for the common 22 backbone and per-mode ingredients. */
double IMRPhenomTHMOmega22(const IMRPhenomTHM *model, double tau);
double IMRPhenomTHMLALReferenceOmega22(const IMRPhenomTHM *model, double tau);
int IMRPhenomTHMLALReferenceTimeFromFrequency(const IMRPhenomTHM *model,
                                              double f_ref_hz,
                                              double mtot_seconds,
                                              double *tau_ref);
double complex IMRPhenomTHMModeComplexAmplitude(const IMRPhenomTHMModeState *mode,
                                                double tau,
                                                double x22);
double IMRPhenomTHMModeFrequency(const IMRPhenomTHM *model,
                                 int mode_index,
                                 double tau);
/* QNM difference used by the TPHM post-peak Euler-angle continuation. */
double IMRPhenomTHMEulerRingdownSlope(double final_dimensionless_spin,
                                      double final_mass_fraction);

void IMRPhenomTHMSetPhaseOffsets(IMRPhenomTHM *model, double phi22_at_tcut);
int IMRPhenomTHMSetPhaseOffsetsFromGrid(IMRPhenomTHM *model,
                                        int n,
                                        const double *tau,
                                        const double *phi22);
/*
 * Convenience 22 integration.  The stripped PhenomT path used in this project
 * splines/integrates omega22, so callers may pass their own phi22 grid instead
 * if they need a different integration or reference phase convention.
 */
int IMRPhenomTHMBuildPhi22Grid(const IMRPhenomTHM *model,
                               int n,
                               const double *tau,
                               double phi0,
                               double *phi22);

/* LAL-compatible analytic 22 phase, shifted so phi22[0]=phi0. */
int IMRPhenomTHMBuildAnalyticPhi22Grid(const IMRPhenomTHM *model,
                                       int n,
                                       const double *tau,
                                       double phi0,
                                       double *phi22);

int IMRPhenomTHMEvaluateMode(const IMRPhenomTHM *model,
                             int mode_index,
                             double tau,
                             double phi22,
                             IMRPhenomTHMModeSample *out);
int IMRPhenomTHMEvaluateGrid(IMRPhenomTHM *model,
                             int n,
                             const double *tau,
                             const double *phi22_in,
                             /* mode-major storage: samples[mode_index*n + i] */
                             IMRPhenomTHMModeSample *samples);

#ifdef __cplusplus
}
#endif

#endif
