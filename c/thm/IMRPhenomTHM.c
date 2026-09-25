/*
 * IMRPhenomTHM is based on the LALSimulation IMRPhenomTHM implementation.
 * Copyright (C) 2020 Hector Estelles
 * LAL-free port and subsequent modifications:
 * Copyright (C) 2026 Neil Cornish
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * This program is free software under the GNU General Public License,
 * version 2 or (at your option) any later version. It is distributed
 * without any warranty; see the GNU GPL for details.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_spline.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif
#ifndef M_EULER
#define M_EULER 0.577215664901532860606512090082402431
#endif

/*
 * The local LALSimIMRPhenomTHM_fits.c copy has already had the LAL includes
 * removed, but its fits still refer to the usual LAL constants/macros and to
 * module-scope eta/S power arrays.  Keep that "LAL baggage" isolated here:
 * the public code below deals only in the stripped structs in IMRPhenomTHM.h.
 */
#define LAL_PI M_PI
#define LAL_GAMMA M_EULER
#define XLAL_EDOM 1
#define XLAL_ERROR(code, ...) return NAN

static double etapow[16];
static double Spow[16];

double *double_vector(int N);

#define IMRPhenomT_Inspiral_TaylorT3 thm_fit_IMRPhenomT_Inspiral_TaylorT3
#define IMRPhenomT_Merger_Freq_CP1_22 thm_fit_IMRPhenomT_Merger_Freq_CP1_22
#define IMRPhenomT_PeakFrequency_22 thm_fit_IMRPhenomT_PeakFrequency_22
#define IMRPhenomT_RD_Freq_D2_22 thm_fit_IMRPhenomT_RD_Freq_D2_22
#define IMRPhenomT_RD_Freq_D3_22 thm_fit_IMRPhenomT_RD_Freq_D3_22
#define IMRPhenomT_Inspiral_Amp_CP1_22 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_22
#define IMRPhenomT_Inspiral_Amp_CP2_22 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_22
#define IMRPhenomT_Inspiral_Amp_CP3_22 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_22
#define IMRPhenomT_Merger_Amp_CP1_22 thm_fit_IMRPhenomT_Merger_Amp_CP1_22
#define IMRPhenomT_PeakAmp_22 thm_fit_IMRPhenomT_PeakAmp_22
#define IMRPhenomT_RD_Amp_C3_22 thm_fit_IMRPhenomT_RD_Amp_C3_22
#define IMRPhenomT_Merger_Freq_CP1_21 thm_fit_IMRPhenomT_Merger_Freq_CP1_21
#define IMRPhenomT_Merger_Freq_CP1_33 thm_fit_IMRPhenomT_Merger_Freq_CP1_33
#define IMRPhenomT_Merger_Freq_CP1_44 thm_fit_IMRPhenomT_Merger_Freq_CP1_44
#define IMRPhenomT_Merger_Freq_CP1_55 thm_fit_IMRPhenomT_Merger_Freq_CP1_55
#define IMRPhenomT_PeakFrequency_21 thm_fit_IMRPhenomT_PeakFrequency_21
#define IMRPhenomT_PeakFrequency_33 thm_fit_IMRPhenomT_PeakFrequency_33
#define IMRPhenomT_PeakFrequency_44 thm_fit_IMRPhenomT_PeakFrequency_44
#define IMRPhenomT_PeakFrequency_55 thm_fit_IMRPhenomT_PeakFrequency_55
#define IMRPhenomT_RD_Freq_D2_21 thm_fit_IMRPhenomT_RD_Freq_D2_21
#define IMRPhenomT_RD_Freq_D3_21 thm_fit_IMRPhenomT_RD_Freq_D3_21
#define IMRPhenomT_RD_Freq_D2_33 thm_fit_IMRPhenomT_RD_Freq_D2_33
#define IMRPhenomT_RD_Freq_D3_33 thm_fit_IMRPhenomT_RD_Freq_D3_33
#define IMRPhenomT_RD_Freq_D2_44 thm_fit_IMRPhenomT_RD_Freq_D2_44
#define IMRPhenomT_RD_Freq_D3_44 thm_fit_IMRPhenomT_RD_Freq_D3_44
#define IMRPhenomT_RD_Freq_D2_55 thm_fit_IMRPhenomT_RD_Freq_D2_55
#define IMRPhenomT_RD_Freq_D3_55 thm_fit_IMRPhenomT_RD_Freq_D3_55
#define IMRPhenomT_Inspiral_Amp_CP1_21 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_21
#define IMRPhenomT_Inspiral_Amp_CP2_21 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_21
#define IMRPhenomT_Inspiral_Amp_CP3_21 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_21
#define IMRPhenomT_Merger_Amp_CP1_21 thm_fit_IMRPhenomT_Merger_Amp_CP1_21
#define IMRPhenomT_PeakAmp_21 thm_fit_IMRPhenomT_PeakAmp_21
#define IMRPhenomT_RD_Amp_C3_21 thm_fit_IMRPhenomT_RD_Amp_C3_21
#define IMRPhenomT_Inspiral_Amp_CP1_33 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_33
#define IMRPhenomT_Inspiral_Amp_CP2_33 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_33
#define IMRPhenomT_Inspiral_Amp_CP3_33 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_33
#define IMRPhenomT_Merger_Amp_CP1_33 thm_fit_IMRPhenomT_Merger_Amp_CP1_33
#define IMRPhenomT_PeakAmp_33 thm_fit_IMRPhenomT_PeakAmp_33
#define IMRPhenomT_RD_Amp_C3_33 thm_fit_IMRPhenomT_RD_Amp_C3_33
#define IMRPhenomT_Inspiral_Amp_CP1_44 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_44
#define IMRPhenomT_Inspiral_Amp_CP2_44 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_44
#define IMRPhenomT_Inspiral_Amp_CP3_44 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_44
#define IMRPhenomT_Merger_Amp_CP1_44 thm_fit_IMRPhenomT_Merger_Amp_CP1_44
#define IMRPhenomT_PeakAmp_44 thm_fit_IMRPhenomT_PeakAmp_44
#define IMRPhenomT_RD_Amp_C3_44 thm_fit_IMRPhenomT_RD_Amp_C3_44
#define IMRPhenomT_Inspiral_Amp_CP1_55 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_55
#define IMRPhenomT_Inspiral_Amp_CP2_55 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_55
#define IMRPhenomT_Inspiral_Amp_CP3_55 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_55
#define IMRPhenomT_Merger_Amp_CP1_55 thm_fit_IMRPhenomT_Merger_Amp_CP1_55
#define IMRPhenomT_PeakAmp_55 thm_fit_IMRPhenomT_PeakAmp_55
#define IMRPhenomT_RD_Amp_C3_55 thm_fit_IMRPhenomT_RD_Amp_C3_55
#define evaluate_QNMfit_fring21 thm_fit_evaluate_QNMfit_fring21
#define evaluate_QNMfit_fring33 thm_fit_evaluate_QNMfit_fring33
#define evaluate_QNMfit_fring44 thm_fit_evaluate_QNMfit_fring44
#define evaluate_QNMfit_fring55 thm_fit_evaluate_QNMfit_fring55
#define evaluate_QNMfit_fdamp21 thm_fit_evaluate_QNMfit_fdamp21
#define evaluate_QNMfit_fdamp33 thm_fit_evaluate_QNMfit_fdamp33
#define evaluate_QNMfit_fdamp44 thm_fit_evaluate_QNMfit_fdamp44
#define evaluate_QNMfit_fdamp55 thm_fit_evaluate_QNMfit_fdamp55
#define evaluate_QNMfit_fdamp21n2 thm_fit_evaluate_QNMfit_fdamp21n2
#define evaluate_QNMfit_fdamp33n2 thm_fit_evaluate_QNMfit_fdamp33n2
#define evaluate_QNMfit_fdamp44n2 thm_fit_evaluate_QNMfit_fdamp44n2
#define evaluate_QNMfit_fdamp55n2 thm_fit_evaluate_QNMfit_fdamp55n2
#define IMRPhenomT_tshift_21 thm_fit_IMRPhenomT_tshift_21
#define IMRPhenomT_tshift_33 thm_fit_IMRPhenomT_tshift_33
#define IMRPhenomT_tshift_44 thm_fit_IMRPhenomT_tshift_44
#define IMRPhenomT_tshift_55 thm_fit_IMRPhenomT_tshift_55
#define evaluate_QNMfit_fring22 thm_fit_evaluate_QNMfit_fring22
#define evaluate_QNMfit_fdamp22 thm_fit_evaluate_QNMfit_fdamp22
#define evaluate_QNMfit_fdamp22n2 thm_fit_evaluate_QNMfit_fdamp22n2

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#include "LALSimIMRPhenomTHM_fits.c"
#pragma GCC diagnostic pop

#undef XLAL_ERROR
#undef XLAL_EDOM

#undef IMRPhenomT_Inspiral_TaylorT3
#undef IMRPhenomT_Merger_Freq_CP1_22
#undef IMRPhenomT_PeakFrequency_22
#undef IMRPhenomT_RD_Freq_D2_22
#undef IMRPhenomT_RD_Freq_D3_22
#undef IMRPhenomT_Inspiral_Amp_CP1_22
#undef IMRPhenomT_Inspiral_Amp_CP2_22
#undef IMRPhenomT_Inspiral_Amp_CP3_22
#undef IMRPhenomT_Merger_Amp_CP1_22
#undef IMRPhenomT_PeakAmp_22
#undef IMRPhenomT_RD_Amp_C3_22
#undef IMRPhenomT_Merger_Freq_CP1_21
#undef IMRPhenomT_Merger_Freq_CP1_33
#undef IMRPhenomT_Merger_Freq_CP1_44
#undef IMRPhenomT_Merger_Freq_CP1_55
#undef IMRPhenomT_PeakFrequency_21
#undef IMRPhenomT_PeakFrequency_33
#undef IMRPhenomT_PeakFrequency_44
#undef IMRPhenomT_PeakFrequency_55
#undef IMRPhenomT_RD_Freq_D2_21
#undef IMRPhenomT_RD_Freq_D3_21
#undef IMRPhenomT_RD_Freq_D2_33
#undef IMRPhenomT_RD_Freq_D3_33
#undef IMRPhenomT_RD_Freq_D2_44
#undef IMRPhenomT_RD_Freq_D3_44
#undef IMRPhenomT_RD_Freq_D2_55
#undef IMRPhenomT_RD_Freq_D3_55
#undef IMRPhenomT_Inspiral_Amp_CP1_21
#undef IMRPhenomT_Inspiral_Amp_CP2_21
#undef IMRPhenomT_Inspiral_Amp_CP3_21
#undef IMRPhenomT_Merger_Amp_CP1_21
#undef IMRPhenomT_PeakAmp_21
#undef IMRPhenomT_RD_Amp_C3_21
#undef IMRPhenomT_Inspiral_Amp_CP1_33
#undef IMRPhenomT_Inspiral_Amp_CP2_33
#undef IMRPhenomT_Inspiral_Amp_CP3_33
#undef IMRPhenomT_Merger_Amp_CP1_33
#undef IMRPhenomT_PeakAmp_33
#undef IMRPhenomT_RD_Amp_C3_33
#undef IMRPhenomT_Inspiral_Amp_CP1_44
#undef IMRPhenomT_Inspiral_Amp_CP2_44
#undef IMRPhenomT_Inspiral_Amp_CP3_44
#undef IMRPhenomT_Merger_Amp_CP1_44
#undef IMRPhenomT_PeakAmp_44
#undef IMRPhenomT_RD_Amp_C3_44
#undef IMRPhenomT_Inspiral_Amp_CP1_55
#undef IMRPhenomT_Inspiral_Amp_CP2_55
#undef IMRPhenomT_Inspiral_Amp_CP3_55
#undef IMRPhenomT_Merger_Amp_CP1_55
#undef IMRPhenomT_PeakAmp_55
#undef IMRPhenomT_RD_Amp_C3_55
#undef evaluate_QNMfit_fring21
#undef evaluate_QNMfit_fring33
#undef evaluate_QNMfit_fring44
#undef evaluate_QNMfit_fring55
#undef evaluate_QNMfit_fdamp21
#undef evaluate_QNMfit_fdamp33
#undef evaluate_QNMfit_fdamp44
#undef evaluate_QNMfit_fdamp55
#undef evaluate_QNMfit_fdamp21n2
#undef evaluate_QNMfit_fdamp33n2
#undef evaluate_QNMfit_fdamp44n2
#undef evaluate_QNMfit_fdamp55n2
#undef IMRPhenomT_tshift_21
#undef IMRPhenomT_tshift_33
#undef IMRPhenomT_tshift_44
#undef IMRPhenomT_tshift_55
#undef evaluate_QNMfit_fring22
#undef evaluate_QNMfit_fdamp22
#undef evaluate_QNMfit_fdamp22n2

#include "IMRPhenomTHM.h"

#define IMRPhenomT_Merger_Freq_CP1_22 thm_fit_IMRPhenomT_Merger_Freq_CP1_22
#define IMRPhenomT_PeakFrequency_22 thm_fit_IMRPhenomT_PeakFrequency_22
#define IMRPhenomT_RD_Freq_D2_22 thm_fit_IMRPhenomT_RD_Freq_D2_22
#define IMRPhenomT_RD_Freq_D3_22 thm_fit_IMRPhenomT_RD_Freq_D3_22
#define IMRPhenomT_Inspiral_Amp_CP1_22 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_22
#define IMRPhenomT_Inspiral_Amp_CP2_22 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_22
#define IMRPhenomT_Inspiral_Amp_CP3_22 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_22
#define IMRPhenomT_Merger_Amp_CP1_22 thm_fit_IMRPhenomT_Merger_Amp_CP1_22
#define IMRPhenomT_PeakAmp_22 thm_fit_IMRPhenomT_PeakAmp_22
#define IMRPhenomT_RD_Amp_C3_22 thm_fit_IMRPhenomT_RD_Amp_C3_22
#define IMRPhenomT_Merger_Freq_CP1_21 thm_fit_IMRPhenomT_Merger_Freq_CP1_21
#define IMRPhenomT_Merger_Freq_CP1_33 thm_fit_IMRPhenomT_Merger_Freq_CP1_33
#define IMRPhenomT_Merger_Freq_CP1_44 thm_fit_IMRPhenomT_Merger_Freq_CP1_44
#define IMRPhenomT_Merger_Freq_CP1_55 thm_fit_IMRPhenomT_Merger_Freq_CP1_55
#define IMRPhenomT_PeakFrequency_21 thm_fit_IMRPhenomT_PeakFrequency_21
#define IMRPhenomT_PeakFrequency_33 thm_fit_IMRPhenomT_PeakFrequency_33
#define IMRPhenomT_PeakFrequency_44 thm_fit_IMRPhenomT_PeakFrequency_44
#define IMRPhenomT_PeakFrequency_55 thm_fit_IMRPhenomT_PeakFrequency_55
#define IMRPhenomT_RD_Freq_D2_21 thm_fit_IMRPhenomT_RD_Freq_D2_21
#define IMRPhenomT_RD_Freq_D3_21 thm_fit_IMRPhenomT_RD_Freq_D3_21
#define IMRPhenomT_RD_Freq_D2_33 thm_fit_IMRPhenomT_RD_Freq_D2_33
#define IMRPhenomT_RD_Freq_D3_33 thm_fit_IMRPhenomT_RD_Freq_D3_33
#define IMRPhenomT_RD_Freq_D2_44 thm_fit_IMRPhenomT_RD_Freq_D2_44
#define IMRPhenomT_RD_Freq_D3_44 thm_fit_IMRPhenomT_RD_Freq_D3_44
#define IMRPhenomT_RD_Freq_D2_55 thm_fit_IMRPhenomT_RD_Freq_D2_55
#define IMRPhenomT_RD_Freq_D3_55 thm_fit_IMRPhenomT_RD_Freq_D3_55
#define IMRPhenomT_Inspiral_Amp_CP1_21 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_21
#define IMRPhenomT_Inspiral_Amp_CP2_21 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_21
#define IMRPhenomT_Inspiral_Amp_CP3_21 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_21
#define IMRPhenomT_Merger_Amp_CP1_21 thm_fit_IMRPhenomT_Merger_Amp_CP1_21
#define IMRPhenomT_PeakAmp_21 thm_fit_IMRPhenomT_PeakAmp_21
#define IMRPhenomT_RD_Amp_C3_21 thm_fit_IMRPhenomT_RD_Amp_C3_21
#define IMRPhenomT_Inspiral_Amp_CP1_33 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_33
#define IMRPhenomT_Inspiral_Amp_CP2_33 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_33
#define IMRPhenomT_Inspiral_Amp_CP3_33 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_33
#define IMRPhenomT_Merger_Amp_CP1_33 thm_fit_IMRPhenomT_Merger_Amp_CP1_33
#define IMRPhenomT_PeakAmp_33 thm_fit_IMRPhenomT_PeakAmp_33
#define IMRPhenomT_RD_Amp_C3_33 thm_fit_IMRPhenomT_RD_Amp_C3_33
#define IMRPhenomT_Inspiral_Amp_CP1_44 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_44
#define IMRPhenomT_Inspiral_Amp_CP2_44 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_44
#define IMRPhenomT_Inspiral_Amp_CP3_44 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_44
#define IMRPhenomT_Merger_Amp_CP1_44 thm_fit_IMRPhenomT_Merger_Amp_CP1_44
#define IMRPhenomT_PeakAmp_44 thm_fit_IMRPhenomT_PeakAmp_44
#define IMRPhenomT_RD_Amp_C3_44 thm_fit_IMRPhenomT_RD_Amp_C3_44
#define IMRPhenomT_Inspiral_Amp_CP1_55 thm_fit_IMRPhenomT_Inspiral_Amp_CP1_55
#define IMRPhenomT_Inspiral_Amp_CP2_55 thm_fit_IMRPhenomT_Inspiral_Amp_CP2_55
#define IMRPhenomT_Inspiral_Amp_CP3_55 thm_fit_IMRPhenomT_Inspiral_Amp_CP3_55
#define IMRPhenomT_Merger_Amp_CP1_55 thm_fit_IMRPhenomT_Merger_Amp_CP1_55
#define IMRPhenomT_PeakAmp_55 thm_fit_IMRPhenomT_PeakAmp_55
#define IMRPhenomT_RD_Amp_C3_55 thm_fit_IMRPhenomT_RD_Amp_C3_55
#define evaluate_QNMfit_fring21 thm_fit_evaluate_QNMfit_fring21
#define evaluate_QNMfit_fring33 thm_fit_evaluate_QNMfit_fring33
#define evaluate_QNMfit_fring44 thm_fit_evaluate_QNMfit_fring44
#define evaluate_QNMfit_fring55 thm_fit_evaluate_QNMfit_fring55
#define evaluate_QNMfit_fdamp21 thm_fit_evaluate_QNMfit_fdamp21
#define evaluate_QNMfit_fdamp33 thm_fit_evaluate_QNMfit_fdamp33
#define evaluate_QNMfit_fdamp44 thm_fit_evaluate_QNMfit_fdamp44
#define evaluate_QNMfit_fdamp55 thm_fit_evaluate_QNMfit_fdamp55
#define evaluate_QNMfit_fdamp21n2 thm_fit_evaluate_QNMfit_fdamp21n2
#define evaluate_QNMfit_fdamp33n2 thm_fit_evaluate_QNMfit_fdamp33n2
#define evaluate_QNMfit_fdamp44n2 thm_fit_evaluate_QNMfit_fdamp44n2
#define evaluate_QNMfit_fdamp55n2 thm_fit_evaluate_QNMfit_fdamp55n2
#define IMRPhenomT_tshift_21 thm_fit_IMRPhenomT_tshift_21
#define IMRPhenomT_tshift_33 thm_fit_IMRPhenomT_tshift_33
#define IMRPhenomT_tshift_44 thm_fit_IMRPhenomT_tshift_44
#define IMRPhenomT_tshift_55 thm_fit_IMRPhenomT_tshift_55
#define evaluate_QNMfit_fring22 thm_fit_evaluate_QNMfit_fring22
#define evaluate_QNMfit_fdamp22 thm_fit_evaluate_QNMfit_fdamp22
#define evaluate_QNMfit_fdamp22n2 thm_fit_evaluate_QNMfit_fdamp22n2

/*
 * Keep the namespaced fit copy for the higher-mode helpers, but use the
 * corrected stripped IMRPhenomT.c implementations for the 22 QNM frequencies.
 * The 22 amplitude ansatz depends on these damping constants, and routing them
 * through the stale copied fits was a small but visible source of PyCBC/LAL
 * amplitude mismatch near merger.
 */
#undef evaluate_QNMfit_fring22
#undef evaluate_QNMfit_fdamp22
#undef evaluate_QNMfit_fdamp22n2

static const IMRPhenomTHMMode default_modes[] = {
    {2, 2}, {2, -2}, {2, 1}, {2, -1}, {3, 3},
    {3, -3}, {4, 4}, {4, -4}, {5, 5}, {5, -5}
};

static void thm_set_fit_powers(double eta, double S)
{
    int i;

    etapow[0] = 1.0;
    Spow[0] = 1.0;
    etapow[1] = eta;
    Spow[1] = S;
    for(i = 2; i < 16; i++)
    {
        etapow[i] = etapow[i-1]*eta;
        Spow[i] = Spow[i-1]*S;
    }
}

static int thm_supported_mode(int ell, int abs_emm)
{
    return (ell == 2 && (abs_emm == 1 || abs_emm == 2)) ||
           (ell == 3 && abs_emm == 3) ||
           (ell == 4 && abs_emm == 4) ||
           (ell == 5 && abs_emm == 5);
}

static double thm_mode_phase_offset(int ell, int abs_emm)
{
    if(ell == 2 && abs_emm == 1) return 0.5*M_PI;
    if(ell == 3 && abs_emm == 3) return -0.5*M_PI;
    if(ell == 4 && abs_emm == 4) return M_PI;
    if(ell == 5 && abs_emm == 5) return 0.5*M_PI;
    return 0.0;
}

static double thm_principal_angle(double x)
{
    return atan2(sin(x), cos(x));
}

static double thm_sech(double x)
{
    return 1.0/cosh(x);
}

static double thm_mode_scale(const IMRPhenomTHMModeState *mode)
{
    return 0.5*(double)mode->abs_emm;
}

static double thm_fring(int ell, int abs_emm, double afinal)
{
    if(ell == 2 && abs_emm == 1) return evaluate_QNMfit_fring21(afinal);
    if(ell == 2 && abs_emm == 2) return evaluate_QNMfit_fring22(afinal);
    if(ell == 3 && abs_emm == 3) return evaluate_QNMfit_fring33(afinal);
    if(ell == 4 && abs_emm == 4) return evaluate_QNMfit_fring44(afinal);
    if(ell == 5 && abs_emm == 5) return evaluate_QNMfit_fring55(afinal);
    return NAN;
}

double IMRPhenomTHMEulerRingdownSlope(double final_dimensionless_spin,
                                      double final_mass_fraction)
{
    double slope;

    if(!isfinite(final_dimensionless_spin) ||
       final_dimensionless_spin <= -1.0 ||
       final_dimensionless_spin >= 1.0 ||
       !isfinite(final_mass_fraction) || final_mass_fraction <= 0.0)
    {
        return NAN;
    }

    slope = 2.0*M_PI*(
        thm_fring(2, 2, final_dimensionless_spin) -
        thm_fring(2, 1, final_dimensionless_spin))/final_mass_fraction;

    /* LAL's TPHM convention reverses the effective precession for af < 0. */
    return final_dimensionless_spin < 0.0 ? -slope : slope;
}

static double thm_fdamp(int ell, int abs_emm, double afinal)
{
    if(ell == 2 && abs_emm == 1) return evaluate_QNMfit_fdamp21(afinal);
    if(ell == 2 && abs_emm == 2) return evaluate_QNMfit_fdamp22(afinal);
    if(ell == 3 && abs_emm == 3) return evaluate_QNMfit_fdamp33(afinal);
    if(ell == 4 && abs_emm == 4) return evaluate_QNMfit_fdamp44(afinal);
    if(ell == 5 && abs_emm == 5) return evaluate_QNMfit_fdamp55(afinal);
    return NAN;
}

static double thm_fdamp_n2(int ell, int abs_emm, double afinal)
{
    if(ell == 2 && abs_emm == 1) return evaluate_QNMfit_fdamp21n2(afinal);
    if(ell == 2 && abs_emm == 2) return evaluate_QNMfit_fdamp22n2(afinal);
    if(ell == 3 && abs_emm == 3) return evaluate_QNMfit_fdamp33n2(afinal);
    if(ell == 4 && abs_emm == 4) return evaluate_QNMfit_fdamp44n2(afinal);
    if(ell == 5 && abs_emm == 5) return evaluate_QNMfit_fdamp55n2(afinal);
    return NAN;
}

static double thm_merger_freq_cp1(int ell, int abs_emm, double eta, double S, double dchi, double delta)
{
    if(ell == 2 && abs_emm == 1) return IMRPhenomT_Merger_Freq_CP1_21(eta, S, dchi, delta);
    if(ell == 2 && abs_emm == 2) return IMRPhenomT_Merger_Freq_CP1_22(eta, S, dchi, delta);
    if(ell == 3 && abs_emm == 3) return IMRPhenomT_Merger_Freq_CP1_33(eta, S, dchi, delta);
    if(ell == 4 && abs_emm == 4) return IMRPhenomT_Merger_Freq_CP1_44(eta, S, dchi, delta);
    if(ell == 5 && abs_emm == 5) return IMRPhenomT_Merger_Freq_CP1_55(eta, S, dchi, delta);
    return NAN;
}

static double thm_peak_frequency(int ell, int abs_emm, double eta, double S, double dchi, double delta)
{
    if(ell == 2 && abs_emm == 1) return IMRPhenomT_PeakFrequency_21(eta, S, dchi, delta);
    if(ell == 2 && abs_emm == 2) return IMRPhenomT_PeakFrequency_22(eta, S, dchi, delta);
    if(ell == 3 && abs_emm == 3) return IMRPhenomT_PeakFrequency_33(eta, S, dchi);
    if(ell == 4 && abs_emm == 4) return IMRPhenomT_PeakFrequency_44(eta, S, dchi, delta);
    if(ell == 5 && abs_emm == 5) return IMRPhenomT_PeakFrequency_55(eta, S, dchi, delta);
    return NAN;
}

static double thm_rd_freq_d2(int ell, int abs_emm, double eta, double S, double dchi, double delta)
{
    if(ell == 2 && abs_emm == 1) return IMRPhenomT_RD_Freq_D2_21(eta, S, dchi, delta);
    if(ell == 2 && abs_emm == 2) return IMRPhenomT_RD_Freq_D2_22(eta, S, dchi, delta);
    if(ell == 3 && abs_emm == 3) return IMRPhenomT_RD_Freq_D2_33(eta, S, dchi, delta);
    if(ell == 4 && abs_emm == 4) return IMRPhenomT_RD_Freq_D2_44(eta, S, dchi, delta);
    if(ell == 5 && abs_emm == 5) return IMRPhenomT_RD_Freq_D2_55(eta, S, dchi, delta);
    return NAN;
}

static double thm_rd_freq_d3(int ell, int abs_emm, double eta, double S, double dchi, double delta)
{
    if(ell == 2 && abs_emm == 1) return IMRPhenomT_RD_Freq_D3_21(eta, S, dchi, delta);
    if(ell == 2 && abs_emm == 2) return IMRPhenomT_RD_Freq_D3_22(eta, S, dchi, delta);
    if(ell == 3 && abs_emm == 3) return IMRPhenomT_RD_Freq_D3_33(eta, S, dchi, delta);
    if(ell == 4 && abs_emm == 4) return IMRPhenomT_RD_Freq_D3_44(eta, S, dchi, delta);
    if(ell == 5 && abs_emm == 5) return IMRPhenomT_RD_Freq_D3_55(eta, S, dchi, delta);
    return NAN;
}

static double thm_inspiral_amp_cp(int which, int ell, int abs_emm, double eta, double S, double dchi, double delta)
{
    if(ell == 2 && abs_emm == 1)
    {
        if(which == 0) return IMRPhenomT_Inspiral_Amp_CP1_21(eta, S, dchi, delta);
        if(which == 1) return IMRPhenomT_Inspiral_Amp_CP2_21(eta, S, dchi, delta);
        return IMRPhenomT_Inspiral_Amp_CP3_21(eta, S, dchi, delta);
    }
    if(ell == 2 && abs_emm == 2)
    {
        if(which == 0) return IMRPhenomT_Inspiral_Amp_CP1_22(eta, S, dchi, delta);
        if(which == 1) return IMRPhenomT_Inspiral_Amp_CP2_22(eta, S, dchi, delta);
        return IMRPhenomT_Inspiral_Amp_CP3_22(eta, S, dchi, delta);
    }
    if(ell == 3 && abs_emm == 3)
    {
        if(which == 0) return IMRPhenomT_Inspiral_Amp_CP1_33(eta, S, dchi, delta);
        if(which == 1) return IMRPhenomT_Inspiral_Amp_CP2_33(eta, S, dchi, delta);
        return IMRPhenomT_Inspiral_Amp_CP3_33(eta, S, dchi, delta);
    }
    if(ell == 4 && abs_emm == 4)
    {
        if(which == 0) return IMRPhenomT_Inspiral_Amp_CP1_44(eta, S, dchi, delta);
        if(which == 1) return IMRPhenomT_Inspiral_Amp_CP2_44(eta, S, dchi, delta);
        return IMRPhenomT_Inspiral_Amp_CP3_44(eta, S, dchi, delta);
    }
    if(ell == 5 && abs_emm == 5)
    {
        if(which == 0) return IMRPhenomT_Inspiral_Amp_CP1_55(eta, S, dchi, delta);
        if(which == 1) return IMRPhenomT_Inspiral_Amp_CP2_55(eta, S, dchi, delta);
        return IMRPhenomT_Inspiral_Amp_CP3_55(eta, S, dchi, delta);
    }
    return NAN;
}

static double thm_merger_amp_cp1(int ell, int abs_emm, double eta, double S, double dchi, double delta)
{
    if(ell == 2 && abs_emm == 1) return IMRPhenomT_Merger_Amp_CP1_21(eta, S, dchi, delta);
    if(ell == 2 && abs_emm == 2) return IMRPhenomT_Merger_Amp_CP1_22(eta, S, dchi, delta);
    if(ell == 3 && abs_emm == 3) return IMRPhenomT_Merger_Amp_CP1_33(eta, S, dchi, delta);
    if(ell == 4 && abs_emm == 4) return IMRPhenomT_Merger_Amp_CP1_44(eta, S, dchi, delta);
    if(ell == 5 && abs_emm == 5) return IMRPhenomT_Merger_Amp_CP1_55(eta, S, dchi, delta);
    return NAN;
}

static double thm_peak_amp(int ell, int abs_emm, double eta, double S, double dchi, double delta)
{
    if(ell == 2 && abs_emm == 1) return IMRPhenomT_PeakAmp_21(eta, S, dchi, delta);
    if(ell == 2 && abs_emm == 2) return IMRPhenomT_PeakAmp_22(eta, S, dchi, delta);
    if(ell == 3 && abs_emm == 3) return IMRPhenomT_PeakAmp_33(eta, S, dchi, delta);
    if(ell == 4 && abs_emm == 4) return IMRPhenomT_PeakAmp_44(eta, S, dchi, delta);
    if(ell == 5 && abs_emm == 5) return IMRPhenomT_PeakAmp_55(eta, S, dchi, delta);
    return NAN;
}

static double thm_rd_amp_c3(int ell, int abs_emm, double eta, double S, double dchi)
{
    if(ell == 2 && abs_emm == 1) return IMRPhenomT_RD_Amp_C3_21(eta, S, dchi);
    if(ell == 2 && abs_emm == 2) return IMRPhenomT_RD_Amp_C3_22(eta, S);
    if(ell == 3 && abs_emm == 3) return IMRPhenomT_RD_Amp_C3_33(eta, S);
    if(ell == 4 && abs_emm == 4) return IMRPhenomT_RD_Amp_C3_44(eta, S);
    if(ell == 5 && abs_emm == 5) return IMRPhenomT_RD_Amp_C3_55(eta, S, dchi);
    return NAN;
}

double IMRPhenomTHMOmega22(const IMRPhenomTHM *model, double tau)
{
    return IMRPhenomTomega22(tau, model->eta, (struct IMRPhenomT *)&model->mode22);
}

static double thm_phase_omega22(const IMRPhenomTHM *model, double tau)
{
    return IMRPhenomTPhaseOmega22(
        tau, model->eta, (struct IMRPhenomT *)&model->mode22);
}

double IMRPhenomTHMLALReferenceOmega22(const IMRPhenomTHM *model, double tau)
{
    if(model == NULL)
    {
        return NAN;
    }
    return IMRPhenomTLALReferenceOmega22(tau, model->eta,
                                         (struct IMRPhenomT *)&model->mode22);
}

int IMRPhenomTHMLALReferenceTimeFromFrequency(const IMRPhenomTHM *model,
                                              double f_ref_hz,
                                              double mtot_seconds,
                                              double *tau_ref)
{
    if(model == NULL || tau_ref == NULL || f_ref_hz <= 0.0 || mtot_seconds <= 0.0)
    {
        return 1;
    }

    return IMRPhenomTLALReferenceTimeFromMf((struct IMRPhenomT *)&model->mode22,
                                            model->eta,
                                            f_ref_hz*mtot_seconds,
                                            tau_ref);
}

static double complex thm_inspiral_amp(double x, const IMRPhenomTHMAmpCoeffs *amp)
{
    double xhalf = sqrt(x);
    double x1half = x*xhalf;
    double x2 = x*x;
    double x2half = x2*xhalf;
    double x3 = x2*x;
    double x3half = x3*xhalf;
    double x4 = x2*x2;
    double x4half = x4*xhalf;
    double x5 = x3*x2;
    double ampreal;
    double ampimag;

    ampreal = amp->ampN +
              amp->amp0halfPNreal*xhalf +
              amp->amp1PNreal*x +
              amp->amp1halfPNreal*x1half +
              amp->amp2PNreal*x2 +
              amp->amp2halfPNreal*x2half +
              amp->amp3PNreal*x3 +
              amp->amp3halfPNreal*x3half +
              amp->amplog*log(16.0*x)*x3 +
              amp->inspC1*x4 +
              amp->inspC2*x4half +
              amp->inspC3*x5;
    ampimag = amp->amp0halfPNimag*xhalf +
              amp->amp1PNimag*x +
              amp->amp1halfPNimag*x1half +
              amp->amp2PNimag*x2 +
              amp->amp2halfPNimag*x2half +
              amp->amp3PNimag*x3 +
              amp->amp3halfPNimag*x3half;

    return amp->fac0*x*(ampreal + I*ampimag);
}

static double thm_complex_amp_orientation(double x, const IMRPhenomTHMAmpCoeffs *amp)
{
    return carg(thm_inspiral_amp(x, amp));
}

static double thm_merger_amp(double tau, const IMRPhenomTHMAmpCoeffs *amp)
{
    double tpeak = amp->tshift;
    double sech1 = thm_sech(amp->alpha1RD*(tau - tpeak));
    double sech2 = thm_sech(2.0*amp->alpha1RD*(tau - tpeak));

    return amp->mergerC1 +
           amp->mergerC2*sech1 +
           amp->mergerC3*pow(sech2, 1.0/7.0) +
           amp->mergerC4*(tau - tpeak)*(tau - tpeak);
}

static double thm_rd_amp(double tau, const IMRPhenomTHMAmpCoeffs *amp)
{
    double tpeak = amp->tshift;
    double tanhphi = tanh(amp->c2_prec*(tau - tpeak) + amp->c3);
    double exp_alpha = exp(-amp->alpha1RD_prec*(tau - tpeak));

    return exp_alpha*(amp->c1_prec*tanhphi + amp->c4_prec);
}

double complex IMRPhenomTHMModeComplexAmplitude(const IMRPhenomTHMModeState *mode,
                                                double tau,
                                                double x22)
{
    if(tau < tCUT_Amp)
    {
        return thm_inspiral_amp(x22, &mode->amp);
    }
    if(tau > mode->amp.tshift)
    {
        return thm_rd_amp(tau, &mode->amp);
    }
    return thm_merger_amp(tau, &mode->amp);
}

static double thm_merger_omega_bar(double tau, const IMRPhenomTHMPhaseCoeffs *phase)
{
    double x = asinh(phase->alpha1RD*tau);
    double x2 = x*x;
    double x3 = x2*x;
    double x4 = x2*x2;

    return 1.0 - phase->omegaPeak/phase->omegaRING +
           (phase->domegaPeak/phase->alpha1RD)*x +
           phase->omegaMergerC1*x2 +
           phase->omegaMergerC2*x3 +
           phase->omegaMergerC3*x4;
}

static double thm_merger_omega(double tau, const IMRPhenomTHMPhaseCoeffs *phase)
{
    return phase->omegaRING*(1.0 - thm_merger_omega_bar(tau, phase));
}

static double thm_rd_omega(double tau, const IMRPhenomTHMPhaseCoeffs *phase)
{
    double expC = exp(-phase->c2*tau);
    double expC2 = expC*expC;
    double num = phase->c1*(-2.0*phase->c2*phase->c4*expC2 -
                            phase->c2*phase->c3*expC);
    double den = 1.0 + phase->c4*expC2 + phase->c3*expC;

    return num/den + phase->omegaRING;
}

static double thm_merger_phase_no_offset(double tau, const IMRPhenomTHMPhaseCoeffs *phase)
{
    double x = asinh(phase->alpha1RD*tau);
    double alpha1RD = phase->alpha1RD;
    double omegaPeak = phase->omegaPeak;
    double domegaPeak = phase->domegaPeak;
    double omegaRING = phase->omegaRING;
    double cc = phase->omegaMergerC1;
    double dd = phase->omegaMergerC2;
    double ee = phase->omegaMergerC3;
    double root = sqrt(1.0 + alpha1RD*alpha1RD*tau*tau);
    double aux;

    aux = omegaRING*tau - omegaRING*(
        2.0*cc*tau + 24.0*ee*tau + 6.0*dd*tau*x +
        domegaPeak*tau*x/alpha1RD +
        tau*(1.0 - omegaPeak/omegaRING) +
        cc*tau*x*x + 12.0*ee*tau*x*x + dd*tau*x*x*x +
        ee*tau*x*x*x*x -
        domegaPeak*root/(alpha1RD*alpha1RD) -
        6.0*dd*root/alpha1RD -
        2.0*cc*x*root/alpha1RD -
        24.0*ee*x*root/alpha1RD -
        3.0*dd*x*x*root/alpha1RD -
        4.0*ee*x*x*x*root/alpha1RD);

    return aux;
}

static double thm_merger_phase(double tau, const IMRPhenomTHMPhaseCoeffs *phase)
{
    return thm_merger_phase_no_offset(tau, phase) + phase->phOffMerger;
}

static double thm_rd_phase(double tau, const IMRPhenomTHMPhaseCoeffs *phase)
{
    double expC = exp(-phase->c2*tau);
    double expC2 = expC*expC;
    double num = 1.0 + phase->c3*expC + phase->c4*expC2;
    double den = 1.0 + phase->c3 + phase->c4;

    return phase->c1_prec*log(num/den) +
           phase->omegaRING_prec*tau +
           phase->phOffRD;
}

static double thm_carrier_phase(const IMRPhenomTHMModeState *mode,
                                double tau,
                                double phi22)
{
    if(mode->ell == 2 && mode->abs_emm == 2)
    {
        return phi22;
    }
    if(tau < tCUT_Freq)
    {
        return thm_mode_scale(mode)*phi22;
    }
    if(tau > 0.0)
    {
        return thm_rd_phase(tau, &mode->phase) - mode->amp.phiCutPNAMP;
    }
    return thm_merger_phase(tau, &mode->phase) - mode->amp.phiCutPNAMP;
}

static double thm_amp_phase_frequency(const IMRPhenomTHM *model,
                                      const IMRPhenomTHMModeState *mode,
                                      double tau)
{
    const double eps = 1.0e-6;
    double tau0 = tau - eps;
    double omega0 = IMRPhenomTHMOmega22(model, tau0);
    double omega1 = IMRPhenomTHMOmega22(model, tau);
    double x0 = pow(0.5*omega0, 2.0/3.0);
    double x1 = pow(0.5*omega1, 2.0/3.0);
    double arg0 = thm_complex_amp_orientation(x0, &mode->amp);
    double arg1 = thm_complex_amp_orientation(x1, &mode->amp);

    return -thm_principal_angle(arg1 - arg0)/eps;
}

double IMRPhenomTHMModeFrequency(const IMRPhenomTHM *model,
                                 int mode_index,
                                 double tau)
{
    const IMRPhenomTHMModeState *mode;
    double omega;

    if(model == NULL || mode_index < 0 || mode_index >= model->nmodes)
    {
        return NAN;
    }

    mode = &model->modes[mode_index];
    if(mode->is_zero_by_symmetry)
    {
        return 0.0;
    }

    if(mode->ell == 2 && mode->abs_emm == 2)
    {
        omega = IMRPhenomTHMOmega22(model, tau);
    }
    else if(tau < tCUT_Freq)
    {
        omega = thm_mode_scale(mode)*IMRPhenomTHMOmega22(model, tau) +
                thm_amp_phase_frequency(model, mode, tau);
    }
    else if(tau > 0.0)
    {
        omega = thm_rd_omega(tau, &mode->phase);
    }
    else
    {
        omega = thm_merger_omega(tau, &mode->phase);
    }

    return mode->is_negative_m ? -omega : omega;
}

static void thm_set_mode_pn_amplitude_coeffs(int ell,
                                             int abs_emm,
                                             double eta,
                                             double chi1,
                                             double chi2,
                                             double delta,
                                             IMRPhenomTHMAmpCoeffs *amp)
{
    double chi1sq = chi1*chi1;
    double chi2sq = chi2*chi2;

    memset(amp, 0, sizeof(*amp));
    amp->ell = ell;
    amp->emm = abs_emm;
    amp->fac0 = 2.0*eta*sqrt(16.0*M_PI/5.0);

    if(ell == 2 && abs_emm == 2)
    {
        amp->tshift = 0.0;
        amp->ampN = 1.0;
        amp->amp1PNreal = -2.5476190476190474 + (55.0*eta)/42.0;
        amp->amp1halfPNreal = (-2.0*chi1)/3.0 - (2.0*chi2)/3.0 -
                              (2.0*chi1*delta)/3.0 + (2.0*chi2*delta)/3.0 +
                              (2.0*chi1*eta)/3.0 + (2.0*chi2*eta)/3.0 + 2.0*M_PI;
        amp->amp2PNreal = -1.437169312169312 + chi1sq/2.0 + chi2sq/2.0 +
                          (chi1sq*delta)/2.0 - (chi2sq*delta)/2.0 -
                          (1069.0*eta)/216.0 - chi1sq*eta +
                          2.0*chi1*chi2*eta - chi2sq*eta +
                          (2047.0*eta*eta)/1512.0;
        amp->amp2halfPNreal = -(107.0*M_PI)/21.0 + (34.0*eta*M_PI)/21.0;
        amp->amp2halfPNimag = -24.0*eta;
        amp->amp3PNreal = 41.78634662956092 - (278185.0*eta)/33264.0 -
                          (20261.0*eta*eta)/2772.0 +
                          (114635.0*eta*eta*eta)/99792.0 -
                          (856.0*M_EULER)/105.0 + (2.0*M_PI*M_PI)/3.0 +
                          (41.0*eta*M_PI*M_PI)/96.0;
        amp->amp3PNimag = (428.0/105.0)*M_PI;
        amp->amp3halfPNreal = (-2173.0*M_PI)/756.0 -
                              (2495.0*eta*M_PI)/378.0 +
                              (40.0*eta*eta*M_PI)/27.0;
        amp->amp3halfPNimag = (14333.0*eta)/162.0 -
                              (4066.0*eta*eta)/945.0;
        amp->amplog = -428.0/105.0;
    }
    else if(ell == 2 && abs_emm == 1)
    {
        amp->tshift = IMRPhenomT_tshift_21(eta, Spow[1], chi1 - chi2);
        amp->amp0halfPNreal = delta/3.0;
        amp->amp1PNreal = -chi1/4.0 + chi2/4.0 - (chi1*delta)/4.0 - (chi2*delta)/4.0;
        amp->amp1halfPNreal = -17.0*delta/84.0 + (5.0*delta*eta)/21.0;
        amp->amp2PNimag = -delta/6.0 - (2.0*delta*log(2.0))/3.0;
        amp->amp2PNreal = (79.0*chi1)/84.0 - (79.0*chi2)/84.0 +
                          (79.0*chi1*delta)/84.0 + (79.0*chi2*delta)/84.0 -
                          (43.0*chi1)/42.0 + (43.0*chi2)/42.0 -
                          (43.0*chi1*delta)/42.0 - (43.0*chi2*delta)/42.0 -
                          (139.0*chi1*eta)/84.0 + (139.0*chi2*eta)/84.0 -
                          (139.0*chi1*delta*eta)/84.0 -
                          (139.0*chi2*delta*eta)/84.0 +
                          (86.0*chi1*eta)/21.0 - (86.0*chi2*eta)/21.0 +
                          (43.0*chi1*delta*eta)/21.0 +
                          (43.0*chi2*delta*eta)/21.0 + (delta*M_PI)/3.0;
        amp->amp2halfPNreal = (-43.0*delta)/378.0 -
                              (509.0*delta*eta)/378.0 +
                              (79.0*delta*eta*eta)/504.0;
        amp->amp3PNimag = -((-17.0*delta)/168.0 + (353.0*delta*eta)/84.0 -
                            (17.0*delta*log(2.0))/42.0 +
                            (delta*eta*log(2.0))/7.0);
        amp->amp3PNreal = (-17.0*delta*M_PI)/84.0 +
                          (delta*eta*M_PI)/14.0;
    }
    else if(ell == 3 && abs_emm == 3)
    {
        amp->tshift = IMRPhenomT_tshift_33(eta, Spow[1]);
        amp->amp0halfPNreal = 0.7763237542601484*delta;
        amp->amp1halfPNreal = -3.1052950170405937*delta +
                              1.5526475085202969*delta*eta;
        amp->amp2PNimag = -1.371926598204461*delta;
        amp->amp2PNreal = -(-0.5822428156951114*chi1 + 0.5822428156951114*chi2 -
                            7.316679009572791*delta -
                            0.5822428156951114*chi1*delta -
                            0.5822428156951114*chi2*delta +
                            1.3585665699552598*chi1 -
                            1.3585665699552598*chi2 +
                            1.3585665699552598*chi1*delta +
                            1.3585665699552598*chi2*delta +
                            1.7467284470853341*chi1*eta -
                            1.7467284470853341*chi2*eta +
                            1.7467284470853341*chi1*delta*eta +
                            1.7467284470853341*chi2*delta*eta -
                            5.434266279821039*chi1*eta +
                            5.434266279821039*chi2*eta -
                            2.7171331399105196*chi1*delta*eta -
                            2.7171331399105196*chi2*delta*eta);
        amp->amp2halfPNreal = -(-0.08680711070363478*delta +
                                8.647776123213047*delta*eta -
                                2.0866641516022777*delta*eta*eta);
    }
    else if(ell == 4 && abs_emm == 4)
    {
        amp->tshift = IMRPhenomT_tshift_44(eta, Spow[1]);
        amp->amp1PNreal = 0.751248226425348*(1.0 - 3.0*eta);
        amp->amp2PNreal = -4.049910893365739 +
                          14.489984730901032*eta -
                          5.9758381647470875*eta*eta;
        amp->amp2halfPNreal = 0.751248226425348*(4.0*M_PI - 12.0*eta*M_PI);
        amp->amp2halfPNimag = 0.751248226425348*(-2.854822555520438 +
                                                 13.189467666561313*eta);
        amp->amp3PNreal = -(-8.0*sqrt(0.7142857142857143)*
                            (5.338016983016983 -
                             (1088119.0*eta)/28600.0 +
                             (146879.0*eta*eta)/2340.0 -
                             (226097.0*eta*eta*eta)/17160.0))/9.0;
    }
    else if(ell == 5 && abs_emm == 5)
    {
        amp->tshift = IMRPhenomT_tshift_55(eta, Spow[1]);
        amp->amp1halfPNreal = 0.8013768943966973*delta*(1.0 - 2.0*eta);
        amp->amp2halfPNreal = 0.8013768943966973*delta*
                              (-6.743589743589744 +
                               (688.0*eta)/39.0 -
                               (256.0*eta*eta)/39.0);
        amp->amp3PNimag = -3.0177162096765713*delta +
                          12.454250695829877*delta*eta;
        amp->amp3PNreal = 12.58799882096634*delta -
                          25.175997641932675*delta*eta;
    }
}

static int thm_set_amplitude_coefficients(IMRPhenomTHM *model,
                                          IMRPhenomTHMModeState *mode)
{
    IMRPhenomTHMAmpCoeffs *amp = &mode->amp;
    double eta = model->eta;
    double S = model->Shat;
    double dchi = model->dchi;
    double delta = model->delta;
    double tCut = tCUT_Amp;
    double ampInspCP[3];
    double ampMergerCP1;
    double ampPeak;
    double ampRDC3;
    double fDAMP;
    double fDAMPn2;
    double fDAMP_prec;
    double fDAMPn2_prec;
    double coshc3;
    double tanhc3;
    double tinsppoints[3] = {-2000.0, -250.0, -150.0};
    gsl_vector *b;
    gsl_vector *x;
    gsl_matrix *A;
    gsl_permutation *p;
    int s;
    int idx;

    thm_set_mode_pn_amplitude_coeffs(mode->ell, mode->abs_emm, eta,
                                     model->chi1, model->chi2, delta, amp);

    ampInspCP[0] = thm_inspiral_amp_cp(0, mode->ell, mode->abs_emm, eta, S, dchi, delta);
    ampInspCP[1] = thm_inspiral_amp_cp(1, mode->ell, mode->abs_emm, eta, S, dchi, delta);
    ampInspCP[2] = thm_inspiral_amp_cp(2, mode->ell, mode->abs_emm, eta, S, dchi, delta);
    ampMergerCP1 = thm_merger_amp_cp1(mode->ell, mode->abs_emm, eta, S, dchi, delta);
    ampPeak = thm_peak_amp(mode->ell, mode->abs_emm, eta, S, dchi, delta);
    ampRDC3 = thm_rd_amp_c3(mode->ell, mode->abs_emm, eta, S, dchi);

    fDAMP = thm_fdamp(mode->ell, mode->abs_emm, model->afinal)/model->Mfinal;
    fDAMPn2 = thm_fdamp_n2(mode->ell, mode->abs_emm, model->afinal)/model->Mfinal;
    fDAMP_prec = thm_fdamp(mode->ell, mode->abs_emm, model->afinal_prec)/model->Mfinal;
    fDAMPn2_prec = thm_fdamp_n2(mode->ell, mode->abs_emm, model->afinal_prec)/model->Mfinal;

    amp->alpha1RD = 2.0*M_PI*fDAMP;
    amp->alpha2RD = 2.0*M_PI*fDAMPn2;
    amp->alpha21RD = 0.5*(amp->alpha2RD - amp->alpha1RD);
    amp->alpha1RD_prec = 2.0*M_PI*fDAMP_prec;
    amp->alpha2RD_prec = 2.0*M_PI*fDAMPn2_prec;
    amp->alpha21RD_prec = 0.5*(amp->alpha2RD_prec - amp->alpha1RD_prec);

    amp->c3 = ampRDC3;
    amp->c2 = 0.5*(amp->alpha2RD - amp->alpha1RD);
    amp->c2_prec = 0.5*(amp->alpha2RD_prec - amp->alpha1RD_prec);
    coshc3 = cosh(amp->c3);
    tanhc3 = tanh(amp->c3);

    if(fabs(amp->c2) > fabs(0.5*amp->alpha1RD/tanhc3))
    {
        amp->c2 = -0.5*amp->alpha1RD/tanhc3;
    }
    if(fabs(amp->c2_prec) > fabs(0.5*amp->alpha1RD_prec/tanhc3))
    {
        amp->c2_prec = -0.5*amp->alpha1RD_prec/tanhc3;
    }

    amp->c1 = ampPeak*amp->alpha1RD*coshc3*coshc3/amp->c2;
    amp->c1_prec = ampPeak*amp->alpha1RD_prec*coshc3*coshc3/amp->c2_prec;
    amp->c4 = ampPeak - amp->c1*tanhc3;
    amp->c4_prec = ampPeak - amp->c1_prec*tanhc3;

    amp->inspC1 = 0.0;
    amp->inspC2 = 0.0;
    amp->inspC3 = 0.0;

    p = gsl_permutation_alloc(3);
    b = gsl_vector_alloc(3);
    x = gsl_vector_alloc(3);
    A = gsl_matrix_alloc(3, 3);
    if(p == NULL || b == NULL || x == NULL || A == NULL) return 2;

    for(idx = 0; idx < 3; idx++)
    {
        double omega = IMRPhenomTomega22(tinsppoints[idx], eta, &model->mode22);
        double xx = pow(0.5*omega, 2.0/3.0);
        double x4 = xx*xx*xx*xx;
        double x4half = x4*sqrt(xx);
        double x5 = x4*xx;
        double ampoffset = creal(thm_inspiral_amp(xx, amp));
        double bi = (ampInspCP[idx] - ampoffset)/(amp->fac0*xx);

        gsl_vector_set(b, idx, bi);
        gsl_matrix_set(A, idx, 0, x4);
        gsl_matrix_set(A, idx, 1, x4half);
        gsl_matrix_set(A, idx, 2, x5);
    }

    gsl_linalg_LU_decomp(A, p, &s);
    gsl_linalg_LU_solve(A, p, b, x);
    amp->inspC1 = gsl_vector_get(x, 0);
    amp->inspC2 = gsl_vector_get(x, 1);
    amp->inspC3 = gsl_vector_get(x, 2);

    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A);
    gsl_permutation_free(p);

    p = gsl_permutation_alloc(4);
    b = gsl_vector_alloc(4);
    x = gsl_vector_alloc(4);
    A = gsl_matrix_alloc(4, 4);
    if(p == NULL || b == NULL || x == NULL || A == NULL) return 2;

    {
        double omegaCut = IMRPhenomTomega22(tCut, eta, &model->mode22);
        double xx = pow(0.5*omegaCut, 2.0/3.0);
        double ampinsp = copysign(1.0, creal(thm_inspiral_amp(xx, amp)))*
                         cabs(thm_inspiral_amp(xx, amp));
        double sech1 = thm_sech(amp->alpha1RD*(tCut - amp->tshift));
        double sech2 = thm_sech(2.0*amp->alpha1RD*(tCut - amp->tshift));
        double sechcp1 = thm_sech(amp->alpha1RD*(tcpMerger - amp->tshift));
        double sechcp2 = thm_sech(2.0*amp->alpha1RD*(tcpMerger - amp->tshift));
        double tauCut = tCut - amp->tshift;
        double tauCP = tcpMerger - amp->tshift;
        double omega2 = IMRPhenomTomega22(tCut, eta, &model->mode22);
        double omega1 = IMRPhenomTomega22(tCut - 1.0e-6, eta, &model->mode22);
        double x2 = pow(0.5*omega2, 2.0/3.0);
        double x1 = pow(0.5*omega1, 2.0/3.0);
        double dampMECO = copysign(1.0, creal(thm_inspiral_amp(x2, amp)))*
                          (cabs(thm_inspiral_amp(x2, amp)) -
                           cabs(thm_inspiral_amp(x1, amp)))/1.0e-6;
        double tanhCut = tanh(amp->alpha1RD*tauCut);
        double sinh2Cut = sinh(2.0*amp->alpha1RD*tauCut);
        double aux1 = -amp->alpha1RD*sech1*tanhCut;
        double aux2 = (-2.0/7.0)*amp->alpha1RD*sinh2Cut*pow(sech2, 8.0/7.0);
        double aux3 = 2.0*tauCut;

        gsl_vector_set(b, 0, ampinsp);
        gsl_matrix_set(A, 0, 0, 1.0);
        gsl_matrix_set(A, 0, 1, sech1);
        gsl_matrix_set(A, 0, 2, pow(sech2, 1.0/7.0));
        gsl_matrix_set(A, 0, 3, tauCut*tauCut);

        gsl_vector_set(b, 1, ampMergerCP1);
        gsl_matrix_set(A, 1, 0, 1.0);
        gsl_matrix_set(A, 1, 1, sechcp1);
        gsl_matrix_set(A, 1, 2, pow(sechcp2, 1.0/7.0));
        gsl_matrix_set(A, 1, 3, tauCP*tauCP);

        gsl_vector_set(b, 2, ampPeak);
        gsl_matrix_set(A, 2, 0, 1.0);
        gsl_matrix_set(A, 2, 1, 1.0);
        gsl_matrix_set(A, 2, 2, 1.0);
        gsl_matrix_set(A, 2, 3, 0.0);

        gsl_vector_set(b, 3, dampMECO);
        gsl_matrix_set(A, 3, 0, 0.0);
        gsl_matrix_set(A, 3, 1, aux1);
        gsl_matrix_set(A, 3, 2, aux2);
        gsl_matrix_set(A, 3, 3, aux3);

        amp->omegaCutPNAMP = -thm_principal_angle(thm_complex_amp_orientation(x2, amp) -
                                                 thm_complex_amp_orientation(x1, amp))/1.0e-6;
        amp->phiCutPNAMP = carg(thm_inspiral_amp(x2, amp));
        if(copysign(1.0, creal(thm_inspiral_amp(x2, amp))) == -1.0)
        {
            amp->phiCutPNAMP += M_PI;
        }
    }

    gsl_linalg_LU_decomp(A, p, &s);
    gsl_linalg_LU_solve(A, p, b, x);
    amp->mergerC1 = gsl_vector_get(x, 0);
    amp->mergerC2 = gsl_vector_get(x, 1);
    amp->mergerC3 = gsl_vector_get(x, 2);
    amp->mergerC4 = gsl_vector_get(x, 3);

    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A);
    gsl_permutation_free(p);

    return 0;
}

static int thm_set_phase_coefficients(IMRPhenomTHM *model,
                                      IMRPhenomTHMModeState *mode)
{
    IMRPhenomTHMPhaseCoeffs *phase = &mode->phase;
    double eta = model->eta;
    double S = model->Shat;
    double dchi = model->dchi;
    double delta = model->delta;
    double tCut = tCUT_Freq;
    double scale = thm_mode_scale(mode);
    double omegaCut22 = IMRPhenomTomega22(tCut, eta, &model->mode22);
    double omegaCut = scale*omegaCut22;
    double omegaCutPNAMP = mode->amp.omegaCutPNAMP;
    double theta2;
    double omega2;
    double omega1;
    double domegaCut22;
    double domegaCut;
    double omegaCutBar;
    double omegaMergerCP;
    double domegaPeak;
    gsl_vector *b;
    gsl_vector *x;
    gsl_matrix *A;
    gsl_permutation *p;
    int s;

    memset(phase, 0, sizeof(*phase));
    phase->emm = mode->abs_emm;

    phase->omegaRING = 2.0*M_PI*thm_fring(mode->ell, mode->abs_emm, model->afinal)/model->Mfinal;
    phase->alpha1RD = 2.0*M_PI*thm_fdamp(mode->ell, mode->abs_emm, model->afinal)/model->Mfinal;
    phase->alpha2RD = 2.0*M_PI*thm_fdamp_n2(mode->ell, mode->abs_emm, model->afinal)/model->Mfinal;
    phase->alpha21RD = 0.5*(phase->alpha2RD - phase->alpha1RD);
    phase->omegaRING_prec = 2.0*M_PI*thm_fring(mode->ell, mode->abs_emm, model->afinal_prec)/model->Mfinal;
    phase->alpha1RD_prec = 2.0*M_PI*thm_fdamp(mode->ell, mode->abs_emm, model->afinal_prec)/model->Mfinal;

    phase->omegaPeak = thm_peak_frequency(mode->ell, mode->abs_emm, eta, S, dchi, delta);
    phase->c3 = thm_rd_freq_d3(mode->ell, mode->abs_emm, eta, S, dchi, delta);
    phase->c2 = thm_rd_freq_d2(mode->ell, mode->abs_emm, eta, S, dchi, delta);
    phase->c4 = 0.0;
    phase->c1 = (1.0 + phase->c3 + phase->c4)*
                (phase->omegaRING - phase->omegaPeak)/
                phase->c2/(phase->c3 + 2.0*phase->c4);
    phase->c1_prec = (1.0 + phase->c3 + phase->c4)*
                     (phase->omegaRING_prec - phase->omegaPeak)/
                     phase->c2/(phase->c3 + 2.0*phase->c4);

    theta2 = tCut - 1.0e-7;
    omega2 = IMRPhenomTomega22(tCut, eta, &model->mode22);
    omega1 = IMRPhenomTomega22(theta2, eta, &model->mode22);
    domegaCut22 = (omega2 - omega1)/1.0e-7;
    domegaCut = -scale*domegaCut22/phase->omegaRING;

    domegaPeak = -(thm_rd_omega(1.0e-7, phase) - thm_rd_omega(0.0, phase))/
                 1.0e-7/phase->omegaRING;
    phase->domegaPeak = domegaPeak;

    omegaCutBar = 1.0 - (omegaCut + omegaCutPNAMP)/phase->omegaRING;
    omegaMergerCP = 1.0 - thm_merger_freq_cp1(mode->ell, mode->abs_emm,
                                              eta, S, dchi, delta)/phase->omegaRING;

    p = gsl_permutation_alloc(3);
    b = gsl_vector_alloc(3);
    x = gsl_vector_alloc(3);
    A = gsl_matrix_alloc(3, 3);
    if(p == NULL || b == NULL || x == NULL || A == NULL) return 2;

    {
        double ascut = asinh(phase->alpha1RD*tCut);
        double ascut2 = ascut*ascut;
        double ascut3 = ascut2*ascut;
        double ascut4 = ascut2*ascut2;
        double ascp = asinh(phase->alpha1RD*tcpMerger);
        double ascp2 = ascp*ascp;
        double ascp3 = ascp2*ascp;
        double ascp4 = ascp2*ascp2;
        double dencut = sqrt(1.0 + tCut*tCut*phase->alpha1RD*phase->alpha1RD);

        gsl_vector_set(b, 0, omegaCutBar -
                             (1.0 - phase->omegaPeak/phase->omegaRING) -
                             (phase->domegaPeak/phase->alpha1RD)*ascut);
        gsl_matrix_set(A, 0, 0, ascut2);
        gsl_matrix_set(A, 0, 1, ascut3);
        gsl_matrix_set(A, 0, 2, ascut4);

        gsl_vector_set(b, 1, omegaMergerCP -
                             (1.0 - phase->omegaPeak/phase->omegaRING) -
                             (phase->domegaPeak/phase->alpha1RD)*ascp);
        gsl_matrix_set(A, 1, 0, ascp2);
        gsl_matrix_set(A, 1, 1, ascp3);
        gsl_matrix_set(A, 1, 2, ascp4);

        gsl_vector_set(b, 2, domegaCut - phase->domegaPeak/dencut);
        gsl_matrix_set(A, 2, 0, 2.0*phase->alpha1RD*ascut/dencut);
        gsl_matrix_set(A, 2, 1, 3.0*phase->alpha1RD*ascut2/dencut);
        gsl_matrix_set(A, 2, 2, 4.0*phase->alpha1RD*ascut3/dencut);
    }

    gsl_linalg_LU_decomp(A, p, &s);
    gsl_linalg_LU_solve(A, p, b, x);
    phase->omegaMergerC1 = gsl_vector_get(x, 0);
    phase->omegaMergerC2 = gsl_vector_get(x, 1);
    phase->omegaMergerC3 = gsl_vector_get(x, 2);

    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A);
    gsl_permutation_free(p);

    phase->phOffMerger = 0.0;
    phase->phOffRD = 0.0;
    phase->phase_offsets_set = 0;

    return 0;
}

static void thm_set_mode_phase_offset(IMRPhenomTHMModeState *mode,
                                      double phi22_at_tcut)
{
    if(mode->ell == 2 && mode->abs_emm == 2)
    {
        return;
    }

    /*
     * The original LAL code fixes this with its analytic 22 phase.  This port
     * deliberately waits until the caller supplies the integrated/splined 22
     * phase, so the higher-mode merger/RD phase is tied to the same 22 phase
     * convention used by the stripped IMRPhenomT path.
     */
    mode->phase.phOffMerger = thm_mode_scale(mode)*phi22_at_tcut -
                              thm_merger_phase_no_offset(tCUT_Freq, &mode->phase);
    mode->phase.phOffRD = thm_merger_phase(0.0, &mode->phase);
    mode->phase.phase_offsets_set = 1;
}

int IMRPhenomTHMDefaultModes(IMRPhenomTHMMode *modes, int max_modes)
{
    int i;
    int n = (int)(sizeof(default_modes)/sizeof(default_modes[0]));

    if(modes == NULL || max_modes <= 0)
    {
        return n;
    }

    if(max_modes < n)
    {
        n = max_modes;
    }
    for(i = 0; i < n; i++)
    {
        modes[i] = default_modes[i];
    }
    return n;
}

static int thm_initialize(IMRPhenomTHM *model,
                          double m1,
                          double m2,
                          double chi1,
                          double chi2,
                          int final_spin_mode,
                          double final_dimensionless_spin,
                          const IMRPhenomTHMMode *modes,
                          int nmodes)
{
    IMRPhenomTHMMode local_modes[IMRPHENOMTHM_MAX_MODES];
    double params[4];
    double qdelta;
    int i;

    if(model == NULL || m1 <= 0.0 || m2 <= 0.0)
    {
        return 1;
    }
    if(final_spin_mode != 0 &&
       (!isfinite(final_dimensionless_spin) ||
        final_dimensionless_spin <= -1.0 ||
        final_dimensionless_spin >= 1.0))
    {
        return 5;
    }

    memset(model, 0, sizeof(*model));

    if(m2 > m1)
    {
        fprintf(stderr, "Warning: IMRPhenomTHM uses the LAL mass hierarchy m1 >= m2; switching values (m1 <-> m2 and chi1 <-> chi2) to match the waveform default mass hierarchy.\n");
        qdelta = m1;
        m1 = m2;
        m2 = qdelta;
        qdelta = chi1;
        chi1 = chi2;
        chi2 = qdelta;
    }

    model->m1 = m1;
    model->m2 = m2;
    model->chi1 = chi1;
    model->chi2 = chi2;
    model->Mtot = m1 + m2;
    model->delta = fabs((m1 - m2)/model->Mtot);
    model->eta = m1*m2/(model->Mtot*model->Mtot);
    if(model->eta > 0.25) model->eta = 0.25;
    model->Shat = (m1*m1*chi1 + m2*m2*chi2)/(m1*m1 + m2*m2);
    model->dchi = chi1 - chi2;
    model->Mfinal = XLALSimIMRPhenomXFinalMass2017(model->eta, chi1, chi2);
    model->afinal_aligned = XLALSimIMRPhenomXFinalSpin2017(model->eta, chi1, chi2);
    model->afinal = final_spin_mode == 1 ? final_dimensionless_spin :
                                          model->afinal_aligned;
    model->afinal_prec = final_spin_mode != 0 ? final_dimensionless_spin :
                                               model->afinal;

    params[0] = m1;
    params[1] = m2;
    params[2] = chi1;
    params[3] = chi2;

    allocate_IMRPhenomT(&model->mode22);
    if(model->mode22.omega == NULL || model->mode22.ampR == NULL ||
       model->mode22.ampI == NULL || model->mode22.carray == NULL ||
       model->mode22.CMarray == NULL)
    {
        IMRPhenomTHMDestroy(model);
        return 2;
    }
    InspiralT3(params, &model->mode22);
    InspiralFit(params, &model->mode22);
    if(final_spin_mode == 1)
    {
        MergerRingdownFitWithFinalSpin(params, model->afinal, &model->mode22);
    }
    else
    {
        MergerRingdownFit(params, &model->mode22);
    }
    if(final_spin_mode == 2)
    {
        IMRPhenomTSetPrecessingFinalSpin(&model->mode22, model->Mfinal,
                                         model->afinal_prec);
    }

    thm_set_fit_powers(model->eta, model->Shat);

    if(modes == NULL || nmodes <= 0)
    {
        nmodes = IMRPhenomTHMDefaultModes(local_modes, IMRPHENOMTHM_MAX_MODES);
        modes = local_modes;
    }
    if(nmodes > IMRPHENOMTHM_MAX_MODES)
    {
        IMRPhenomTHMDestroy(model);
        return 3;
    }

    for(i = 0; i < nmodes; i++)
    {
        IMRPhenomTHMModeState *mode = &model->modes[i];
        int abs_emm = abs(modes[i].emm);
        int status;

        if(modes[i].emm == 0 || !thm_supported_mode(modes[i].ell, abs_emm))
        {
            IMRPhenomTHMDestroy(model);
            return 4;
        }

        mode->requested_mode = modes[i];
        mode->ell = modes[i].ell;
        mode->emm = modes[i].emm;
        mode->abs_emm = abs_emm;
        mode->is_negative_m = modes[i].emm < 0;
        mode->phoff = thm_mode_phase_offset(mode->ell, mode->abs_emm);
        mode->is_zero_by_symmetry = (abs_emm % 2 != 0 &&
                                     model->delta < 1.0e-10 &&
                                     fabs(model->chi1 - model->chi2) < 1.0e-10);

        if(mode->is_zero_by_symmetry)
        {
            continue;
        }

        status = thm_set_amplitude_coefficients(model, mode);
        if(status != 0)
        {
            IMRPhenomTHMDestroy(model);
            return status;
        }

        if(!(mode->ell == 2 && mode->abs_emm == 2))
        {
            status = thm_set_phase_coefficients(model, mode);
            if(status != 0)
            {
                IMRPhenomTHMDestroy(model);
                return status;
            }
        }
    }

    model->nmodes = nmodes;
    return 0;
}

int IMRPhenomTHMInitialize(IMRPhenomTHM *model,
                           double m1,
                           double m2,
                           double chi1,
                           double chi2,
                           const IMRPhenomTHMMode *modes,
                           int nmodes)
{
    return thm_initialize(model, m1, m2, chi1, chi2, 0, 0.0,
                          modes, nmodes);
}

int IMRPhenomTHMInitializeWithFinalSpin(IMRPhenomTHM *model,
                                        double m1,
                                        double m2,
                                        double chi1,
                                        double chi2,
                                        double final_dimensionless_spin,
                                        const IMRPhenomTHMMode *modes,
                                        int nmodes)
{
    return thm_initialize(model, m1, m2, chi1, chi2, 1,
                          final_dimensionless_spin, modes, nmodes);
}

int IMRPhenomTHMInitializeWithPrecessingFinalSpin(
    IMRPhenomTHM *model,
    double m1,
    double m2,
    double chi1,
    double chi2,
    double precessing_final_dimensionless_spin,
    const IMRPhenomTHMMode *modes,
    int nmodes)
{
    return thm_initialize(model, m1, m2, chi1, chi2, 2,
                          precessing_final_dimensionless_spin,
                          modes, nmodes);
}

void IMRPhenomTHMDestroy(IMRPhenomTHM *model)
{
    if(model == NULL)
    {
        return;
    }

    free(model->mode22.omega);
    free(model->mode22.ampR);
    free(model->mode22.ampI);
    free(model->mode22.carray);
    free(model->mode22.CMarray);
    memset(model, 0, sizeof(*model));
}

int IMRPhenomTHMModeIndex(const IMRPhenomTHM *model, int ell, int emm)
{
    int i;

    if(model == NULL)
    {
        return -1;
    }
    for(i = 0; i < model->nmodes; i++)
    {
        if(model->modes[i].ell == ell && model->modes[i].emm == emm)
        {
            return i;
        }
    }
    return -1;
}

void IMRPhenomTHMSetPhaseOffsets(IMRPhenomTHM *model, double phi22_at_tcut)
{
    int i;

    if(model == NULL)
    {
        return;
    }
    for(i = 0; i < model->nmodes; i++)
    {
        if(!model->modes[i].is_zero_by_symmetry)
        {
            thm_set_mode_phase_offset(&model->modes[i], phi22_at_tcut);
        }
    }
    model->phase_offsets_initialized = 1;
}

int IMRPhenomTHMSetPhaseOffsetsFromGrid(IMRPhenomTHM *model,
                                        int n,
                                        const double *tau,
                                        const double *phi22)
{
    int i;
    double phi_cut;

    if(model == NULL || tau == NULL || phi22 == NULL || n < 1)
    {
        return 1;
    }

    if(n == 1)
    {
        if(fabs(tau[0] - tCUT_Freq) > 0.0) return 2;
        IMRPhenomTHMSetPhaseOffsets(model, phi22[0]);
        return 0;
    }

    for(i = 1; i < n; i++)
    {
        if(tau[i] < tau[i-1])
        {
            return 3;
        }
        if(tau[i-1] <= tCUT_Freq && tCUT_Freq <= tau[i])
        {
            double u = (tCUT_Freq - tau[i-1])/(tau[i] - tau[i-1]);
            phi_cut = phi22[i-1] + u*(phi22[i] - phi22[i-1]);
            IMRPhenomTHMSetPhaseOffsets(model, phi_cut);
            return 0;
        }
    }

    return 2;
}

int IMRPhenomTHMBuildPhi22Grid(const IMRPhenomTHM *model,
                               int n,
                               const double *tau,
                               double phi0,
                               double *phi22)
{
    int i;
    double *omega;
    gsl_interp_accel *acc;
    gsl_spline *spline;

    if(model == NULL || tau == NULL || phi22 == NULL || n < 1)
    {
        return 1;
    }

    phi22[0] = phi0;
    if(n == 1)
    {
        return 0;
    }

    omega = (double *)calloc((size_t)n, sizeof(double));
    if(omega == NULL)
    {
        return 4;
    }
    for(i = 1; i < n; i++)
    {
        if(tau[i] <= tau[i-1])
        {
            free(omega);
            return 2;
        }
    }
    for(i = 0; i < n; i++)
    {
        omega[i] = thm_phase_omega22(model, tau[i]);
    }

    /*
     * Match the stripped 22 code's phase construction: spline omega22 on the
     * adaptive grid and integrate the spline.  The old trapezoid rule was
     * good enough locally, but it accumulated a visible phase drift across
     * the early inspiral and obscured convention checks against PhenomT_TDI.c.
     */
    acc = gsl_interp_accel_alloc();
    spline = gsl_spline_alloc(gsl_interp_cspline, n);
    if(acc == NULL || spline == NULL)
    {
        if(acc != NULL) gsl_interp_accel_free(acc);
        if(spline != NULL) gsl_spline_free(spline);
        free(omega);
        return 4;
    }
    gsl_spline_init(spline, tau, omega, n);
    {
        double integral = 0.0;
        double compensation = 0.0;
        for(i = 1; i < n; i++)
        {
            double increment = gsl_spline_eval_integ(
                spline, tau[i-1], tau[i], acc);
            double corrected = increment-compensation;
            double updated = integral+corrected;
            compensation = (updated-integral)-corrected;
            integral = updated;
            phi22[i] = phi0+integral;
        }
    }

    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    free(omega);

    return 0;
}

int IMRPhenomTHMBuildAnalyticPhi22Grid(const IMRPhenomTHM *model,
                                       int n,
                                       const double *tau,
                                       double phi0,
                                       double *phi22)
{
    double phase_at_start;
    int i;

    if(model == NULL || tau == NULL || phi22 == NULL || n < 1)
        return 1;
    for(i = 1; i < n; i++)
    {
        if(tau[i] <= tau[i-1]) return 2;
    }

    phase_at_start = IMRPhenomTPhase22(
        tau[0], model->eta, (struct IMRPhenomT *)&model->mode22);
    for(i = 0; i < n; i++)
    {
        phi22[i] = phi0 + IMRPhenomTPhase22(
            tau[i], model->eta, (struct IMRPhenomT *)&model->mode22) -
            phase_at_start;
    }
    return 0;
}

int IMRPhenomTHMEvaluateMode(const IMRPhenomTHM *model,
                             int mode_index,
                             double tau,
                             double phi22,
                             IMRPhenomTHMModeSample *out)
{
    const IMRPhenomTHMModeState *mode;
    double omega22;
    double x22;
    double complex amp_complex;
    double amp_abs;
    double phase;

    if(model == NULL || out == NULL || mode_index < 0 || mode_index >= model->nmodes)
    {
        return 1;
    }

    mode = &model->modes[mode_index];
    if(mode->is_zero_by_symmetry)
    {
        out->amplitude = 0.0;
        out->phase = 0.0;
        out->omega = 0.0;
        out->hlm = 0.0;
        return 0;
    }
    if(!(mode->ell == 2 && mode->abs_emm == 2) &&
       tau >= tCUT_Freq &&
       !mode->phase.phase_offsets_set)
    {
        return 3;
    }

    omega22 = IMRPhenomTHMOmega22(model, tau);
    x22 = pow(0.5*omega22, 2.0/3.0);
    if(mode->ell == 2 && mode->abs_emm == 2 &&
       !model->use_lal_mode22_amplitude)
    {
        /*
         * Keep the 22 backbone exactly tied to the stripped IMRPhenomT.c path.
         * The TDI response is a difference of delayed strains, so even tiny
         * 22 amplitude differences are amplified in low-frequency convention
         * checks against PhenomT_TDI.c.
         */
        amp_complex = IMRPhenomTHMAmp(tau, x22, (struct IMRPhenomT *)&model->mode22);
    }
    else
    {
        amp_complex = IMRPhenomTHMModeComplexAmplitude(mode, tau, x22);
    }
    amp_abs = cabs(amp_complex);

    if(mode->ell == 2 && mode->abs_emm == 2)
    {
        /*
         * Match the LAL THM construction: the 22 waveform uses the absolute
         * value of the complex PN amplitude, but the argument of that complex
         * amplitude is not added to the 22 phase because the calibrated 22
         * frequency/phase already contains that behavior.  Higher modes keep
         * the complex inspiral amplitude in LAL's hlm = A_complex exp(-i phi)
         * form; for the real-A/phase API needed by the LISA response this is
         * equivalent to A=|A_complex| and phi -> phi - arg(A_complex).
         */
        phase = phi22;
    }
    else
    {
        phase = thm_carrier_phase(mode, tau, phi22) -
                mode->phoff -
                carg(amp_complex);
    }

    if(mode->is_negative_m)
    {
        phase = -phase - (double)mode->ell*M_PI;
    }
    out->amplitude = amp_abs;
    out->phase = phase;
    out->omega = (mode->ell == 2 && mode->abs_emm == 2) ?
                 thm_phase_omega22(model, tau) :
                 IMRPhenomTHMModeFrequency(model, mode_index, tau);
    out->hlm = amp_abs*cexp(-I*phase);

    return 0;
}

int IMRPhenomTHMEvaluateGrid(IMRPhenomTHM *model,
                             int n,
                             const double *tau,
                             const double *phi22_in,
                             IMRPhenomTHMModeSample *samples)
{
    double *phi22 = NULL;
    int own_phi22 = 0;
    int need_offsets = 0;
    int i;
    int k;
    int status;

    if(model == NULL || tau == NULL || samples == NULL || n < 1)
    {
        return 1;
    }

    if(phi22_in == NULL)
    {
        phi22 = (double *)calloc((size_t)n, sizeof(double));
        if(phi22 == NULL)
        {
            return 2;
        }
        own_phi22 = 1;
        status = IMRPhenomTHMBuildPhi22Grid(model, n, tau, 0.0, phi22);
        if(status != 0)
        {
            free(phi22);
            return status;
        }
    }
    else
    {
        phi22 = (double *)phi22_in;
    }

    for(k = 0; k < model->nmodes; k++)
    {
        if(!(model->modes[k].ell == 2 && model->modes[k].abs_emm == 2) &&
           !model->modes[k].is_zero_by_symmetry)
        {
            need_offsets = 1;
            break;
        }
    }
    if(need_offsets)
    {
        status = IMRPhenomTHMSetPhaseOffsetsFromGrid(model, n, tau, phi22);
        /* A partitioned FFT block need not cross tCUT_Freq.  In that case an
         * explicitly prepared model keeps the fixed global offsets computed
         * at source setup; all other failures remain errors. */
        if(status == 2 && model->phase_offsets_initialized)
        {
            status = 0;
        }
        if(status != 0)
        {
            if(own_phi22) free(phi22);
            return status;
        }
    }

    for(k = 0; k < model->nmodes; k++)
    {
        for(i = 0; i < n; i++)
        {
            status = IMRPhenomTHMEvaluateMode(model, k, tau[i], phi22[i],
                                              &samples[(size_t)k*(size_t)n + (size_t)i]);
            if(status != 0)
            {
                if(own_phi22) free(phi22);
                return status;
            }
        }
    }

    if(own_phi22)
    {
        free(phi22);
    }
    return 0;
}
