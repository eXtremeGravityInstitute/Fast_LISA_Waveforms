/*
 * Stripped IMRPhenomT 22 interface based on LALSimulation IMRPhenomTHM.
 * Copyright (C) 2020 Hector Estelles
 * Port and subsequent modifications: Copyright (C) 2024, 2026 Neil Cornish
 * SPDX-License-Identifier: GPL-2.0-or-later
 * See the GNU GPL for copying terms and warranty disclaimer.
 */

#ifndef IMRPHENOMT_H
#define IMRPHENOMT_H

#include <ctype.h>
#include <complex.h>

struct IMRPhenomT
{
    double tCut22;
    double omegaRING;
    double omegaRING_prec;
    double omegaPeak;
    double domegaPeak;
    double A0;
    double tshift;
    /*
     * LAL stores these early-inspiral bookkeeping constants in the 22 phase
     * structure.  The default waveform branch still evaluates the inspiral
     * with the production theta=(-eta*t/5)^(-1/8), but LAL's tmin/tRef root
     * finder uses the shifted theta=(eta*(tt0-t)/5)^(-1/8) for t<tEarly.
     * Keeping them here lets the stripped code reproduce the same reference
     * epoch without importing a LAL-generated time.
     */
    double tt0;
    double tEarly;
    double alpha1RD;
    double alpha2RD;
    double alpha21RD;
    double alpha1RD_prec;
    double alpha2RD_prec;
    double alpha21RD_prec;
    double amp_c1_prec;
    double amp_c2_prec;
    double amp_c4_prec;
    double phase_c1_prec;
    double c1;
    double c2;
    double c3;
    double c4;
    double mergerC1;
    double mergerC2;
    double mergerC3;
    double mergerC4;
    double omegaCutPNAMP;
    double phiCutPNAMP;
    double phOffInsp;
    double phOffMerger;
    double phOffRD;
    double *omega;
    double *ampR;
    double *ampI;
    double *carray;
    double *CMarray;
};

#define     tCUT_Amp   -150.0
#define     tCUT_Freq   -150.0
#define     tcpMerger   -25.0

double IMRPhenomTomega22(double t, double eta, struct IMRPhenomT *IMRPT);
/* Derivative of the output 22 phase.  This differs from the orbital driver
 * only in TPHM ringdown, where LAL uses the precessing QNM constants. */
double IMRPhenomTPhaseOmega22(double t, double eta,
                              struct IMRPhenomT *IMRPT);
double IMRPhenomTomega22FromTheta(double t, double theta, double eta, struct IMRPhenomT *IMRPT);
double IMRPhenomTLALReferenceOmega22(double t, double eta, struct IMRPhenomT *IMRPT);
int IMRPhenomTLALReferenceTimeFromMf(struct IMRPhenomT *IMRPT, double eta, double mf_ref, double *tau_ref);
double IMRPhenomTPhase22(double t, double eta, struct IMRPhenomT *IMRPT);
double complex IMRPhenomTHMAmp(double t, double x, struct IMRPhenomT *IMRPT);
void allocate_IMRPhenomT(struct IMRPhenomT *IMRPT);
void InspiralT3(double *params, struct IMRPhenomT *IMRPT);
void InspiralFit(double *params, struct IMRPhenomT *IMRPT);
void MergerRingdownFit(double *params, struct IMRPhenomT *IMRPT);
/*
 * Rebuild the complete 22 merger-ringdown using an externally supplied
 * dimensionless remnant spin.  This is the path needed by TPHM after its
 * precession model has supplied the remnant-spin magnitude.  The remnant
 * mass and all calibrated non-QNM fits remain those of the aligned-spin
 * baseline, matching the default LAL TPHM reconstruction.
 */
void MergerRingdownFitWithFinalSpin(double *params,
                                    double final_dimensionless_spin,
                                    struct IMRPhenomT *IMRPT);
/* Keep the calibrated aligned-spin merger, but use a precessing remnant for
 * the post-peak QNM phase and damping, as in default numerical TPHM. */
void IMRPhenomTSetPrecessingFinalSpin(struct IMRPhenomT *IMRPT,
                                      double final_mass_fraction,
                                      double final_dimensionless_spin);
void InspiralAmpT3(int NS, double *xarray, struct IMRPhenomT *IMRPT, double *AR, double *AI);
double IMRPhenomTTaylorT3(double theta, struct IMRPhenomT *IMRPT);
double InspiralOmegaT3(double u, double fac1, double fac2, struct IMRPhenomT *IMRPT);
double InspiralOmega(double u, double fac1, double fac2, struct IMRPhenomT *IMRPT);
double IMRPhenomT_Inspiral_TaylorT3_t0(double *etapow, double *Spow, double dchi, double delta);
void IMRPhenomT_Inspiral_Fit(double *InspiralFit, double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_Merger_Freq_CP1_22(double *etapow, double *Spow, double dchi, double delta);
double XLALSimIMRPhenomXFinalMass2017(double eta, double chi1L, double chi2L);
double XLALSimIMRPhenomXFinalSpin2017(double eta, double chi1L, double chi2L);
double evaluate_QNMfit_fring22(double finalDimlessSpin);
double evaluate_QNMfit_fdamp22(double finalDimlessSpin);
double evaluate_QNMfit_fdamp22n2(double finalDimlessSpin);
double IMRPhenomT_PeakFrequency_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_RD_Freq_D2_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_RD_Freq_D3_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_Inspiral_Amp_CP1_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_Inspiral_Amp_CP2_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_Inspiral_Amp_CP3_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_Merger_Amp_CP1_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_PeakAmp_22(double *etapow, double *Spow, double dchi, double delta);
double IMRPhenomT_RD_Amp_C3_22(double *etapow, double *Spow);
double IMRPhenomTInspiralOmegaAnsatz22(double theta, struct IMRPhenomT *IMRPT);
double IMRPhenomTInspiralPhaseAnsatz22(double t, double thetabar, double eta, struct IMRPhenomT *IMRPT);
double IMRPhenomTRDOmegaAnsatz22(double t, struct IMRPhenomT *IMRPT);
double IMRPhenomTRDPhaseOmegaAnsatz22(double t,
                                      struct IMRPhenomT *IMRPT);
double IMRPhenomTRDPhaseAnsatz22(double t, struct IMRPhenomT *IMRPT);
double IMRPhenomTMergerOmegaAnsatz22(double t, struct IMRPhenomT *IMRPT);
double IMRPhenomTMergerPhaseAnsatz22(double t, struct IMRPhenomT *IMRPT);
double complex IMRPhenomTInspiralAmpAnsatzHM(double x, struct IMRPhenomT *IMRPT);
double ComplexAmpOrientation(double xref, struct IMRPhenomT *IMRPT);
double IMRPhenomTRDAmpAnsatzHM(double t, struct IMRPhenomT *IMRPT);
double IMRPhenomTMergerAmpAnsatzHM(double t, struct IMRPhenomT *IMRPT);

#endif
