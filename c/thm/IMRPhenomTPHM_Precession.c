/*
 * Numerical IMRPhenomTPHM precession, based on the LALSimulation model.
 * Copyright (C) 2020 Hector Estelles
 * LAL-free reconstruction and subsequent modifications:
 * Copyright (C) 2026 Neil Cornish
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * This program is free software under the GNU General Public License,
 * version 2 or (at your option) any later version. It is distributed
 * without any warranty; see the GNU GPL for details.
 */

#include "IMRPhenomTPHM_Precession.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_spline.h>

#define TPHM_STATE_SIZE 10
#define TPHM_DEFAULT_ABS_TOL 1.0e-11
#define TPHM_DEFAULT_REL_TOL 1.0e-11
#define TPHM_LAL_ABS_TOL 1.0e-12
#define TPHM_LAL_REL_TOL 1.0e-12
#define TPHM_DEFAULT_INITIAL_STEP 1.0

typedef struct
{
    const IMRPhenomTHM *carrier;
    double eta;
    double m1M;
    double m2M;
    double spin1_dot3;
    double spin2_dot3;
    double spin1_dot5;
    double spin2_dot5;
    double spin1_dot7;
    double spin2_dot7;
    const gsl_spline *v_spline;
    gsl_interp_accel *v_acc;
    double v_time_offset;
    const IMRPhenomTPHMPrecessionSummary *frame;
    int integrate_gamma;
} TPHMPrecessionRHS;

static double vec_dot(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void vec_cross(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static double vec_norm(const double a[3])
{
    return sqrt(vec_dot(a, a));
}

static int vec_normalize(double a[3])
{
    double norm = vec_norm(a);
    if(!isfinite(norm) || norm <= DBL_MIN) return 1;
    a[0] /= norm;
    a[1] /= norm;
    a[2] /= norm;
    return 0;
}

static double clamp_unit(double x)
{
    if(x > 1.0) return 1.0;
    if(x < -1.0) return -1.0;
    return x;
}

/* Coefficients used by the phenomtp=1, all-spin-orders SpinTaylor branch. */
static double spin_dot_3pn(double mass_fraction)
{
    return 1.5 - mass_fraction - 0.5*mass_fraction*mass_fraction;
}

static double spin_dot_5pn(double x)
{
    double x2 = x*x;
    double x3 = x2*x;
    double x4 = x2*x2;
    return 9.0/8.0 - x/2.0 + 7.0*x2/12.0 - 7.0*x3/6.0 - x4/24.0;
}

static double spin_dot_7pn(double x)
{
    double x2 = x*x;
    double x3 = x2*x;
    double x4 = x2*x2;
    double x5 = x4*x;
    double x6 = x3*x3;
    return x6/48.0 - 3.0*x5/8.0 - 39.0*x4/16.0 -
           23.0*x3/6.0 + 181.0*x2/16.0 - 51.0*x/8.0 + 27.0/16.0;
}

static double orbital_l_magnitude(double eta, double v)
{
    double v2 = v*v;
    double l1 = 1.5 + eta/6.0;
    double l2 = 27.0/8.0 - 19.0*eta/8.0 + eta*eta/24.0;
    return (eta/v)*(1.0 + l1*v2 + l2*v2*v2);
}

static double carrier_v(const IMRPhenomTHM *carrier, double tau)
{
    double omega = IMRPhenomTHMOmega22(carrier, tau);
    if(!isfinite(omega) || omega <= 0.0) return NAN;
    return cbrt(0.5*omega);
}

static double rhs_v(TPHMPrecessionRHS *rhs, double tau)
{
    tau += rhs->v_time_offset;
    if(rhs->v_spline != NULL)
        return gsl_spline_eval(rhs->v_spline, tau, rhs->v_acc);
    return carrier_v(rhs->carrier, tau);
}

/*
 * LAL's numerical TPHM option calls the orbit-averaged SpinTaylor equations
 * with phenomtp=1, all spin orders, lscorr=0, and zero quadrupole parameters.
 * This is that active subset, expressed directly in total-mass units.
 */
static int precession_rhs(double tau,
                          const double state[],
                          double derivative[],
                          void *parameters)
{
    TPHMPrecessionRHS *rhs = (TPHMPrecessionRHS *)parameters;
    const double *lhat = state;
    const double *spin1 = state + 3;
    const double *spin2 = state + 6;
    double lx_s1[3];
    double lx_s2[3];
    double s1_x_s2[3];
    double ds1[3];
    double ds2[3];
    double dl[3];
    double omega[3];
    double dlhat[3];
    double v;
    double v2;
    double v3;
    double v5;
    double v6;
    double v7;
    double v9;
    double l_dot_s1;
    double l_dot_s2;
    double lmag;
    int k;

    v = rhs_v(rhs, tau);
    if(!isfinite(v)) return GSL_EBADFUNC;
    v2 = v*v;
    v3 = v2*v;
    v5 = v3*v2;
    v6 = v3*v3;
    v7 = v6*v;
    v9 = v6*v3;

    vec_cross(lhat, spin1, lx_s1);
    vec_cross(lhat, spin2, lx_s2);
    vec_cross(spin1, spin2, s1_x_s2);
    l_dot_s1 = vec_dot(lhat, spin1);
    l_dot_s2 = vec_dot(lhat, spin2);

    for(k = 0; k < 3; k++)
    {
        /* 1.5PN + orbit-averaged 2PN + 2.5PN + TPHM 3.5PN terms. */
        ds1[k] = rhs->spin1_dot3*v5*lx_s1[k] +
                 v6*(-0.5*s1_x_s2[k] - 1.5*l_dot_s2*lx_s1[k]) +
                 rhs->spin1_dot5*v7*lx_s1[k] +
                 rhs->spin1_dot7*v9*lx_s1[k];
        ds2[k] = rhs->spin2_dot3*v5*lx_s2[k] +
                 v6*(0.5*s1_x_s2[k] - 1.5*l_dot_s1*lx_s2[k]) +
                 rhs->spin2_dot5*v7*lx_s2[k] +
                 rhs->spin2_dot7*v9*lx_s2[k];
        dl[k] = -(ds1[k] + ds2[k]);
    }

    lmag = orbital_l_magnitude(rhs->eta, v);
    if(!isfinite(lmag) || lmag <= 0.0) return GSL_EBADFUNC;
    for(k = 0; k < 3; k++) dl[k] /= lmag;

    /* Project dL/L orthogonal to Lhat, as in SpinDerivativesAvg. */
    vec_cross(lhat, dl, omega);
    vec_cross(omega, lhat, dlhat);

    for(k = 0; k < 3; k++)
    {
        derivative[k] = dlhat[k];
        derivative[3+k] = ds1[k];
        derivative[6+k] = ds2[k];
    }
    if(rhs->integrate_gamma)
    {
        const double lx = vec_dot(lhat, rhs->frame->j_frame_x);
        const double ly = vec_dot(lhat, rhs->frame->j_frame_y);
        const double lz = vec_dot(lhat, rhs->frame->j_frame_z);
        const double dlx = vec_dot(dlhat, rhs->frame->j_frame_x);
        const double dly = vec_dot(dlhat, rhs->frame->j_frame_y);
        const double transverse2 = lx*lx + ly*ly;
        derivative[9] = transverse2 > 1.0e-24 ?
                        -lz*(lx*dly-ly*dlx)/transverse2 : 0.0;
    }
    return GSL_SUCCESS;
}

static void initial_state(const IMRPhenomTHM *carrier,
                          const IMRPhenomTPHMPrecessionConfig *config,
                          double state[TPHM_STATE_SIZE])
{
    double m1M = carrier->m1/carrier->Mtot;
    double m2M = carrier->m2/carrier->Mtot;
    double norm1 = m1M*m1M;
    double norm2 = m2M*m2M;
    int k;

    state[0] = 0.0;
    state[1] = 0.0;
    state[2] = 1.0;
    for(k = 0; k < 3; k++)
    {
        state[3+k] = norm1*config->chi1[k];
        state[6+k] = norm2*config->chi2[k];
    }
    state[9] = 0.0;
}

static gsl_odeiv2_driver *make_driver(gsl_odeiv2_system *system,
                                      double initial_step,
                                      double abs_tol,
                                      double rel_tol,
                                      int direction)
{
    return gsl_odeiv2_driver_alloc_y_new(system,
                                          gsl_odeiv2_step_rk8pd,
                                          direction*initial_step,
                                          abs_tol,
                                          rel_tol);
}

static int evolve_to(gsl_odeiv2_driver *driver,
                     double *current_tau,
                     double target_tau,
                     double state[TPHM_STATE_SIZE])
{
    if(target_tau == *current_tau) return GSL_SUCCESS;
    return gsl_odeiv2_driver_apply(driver, current_tau, target_tau, state);
}

static double lal_x102_reference_l_magnitude(
    const IMRPhenomTHM *carrier,
    const IMRPhenomTPHMPrecessionConfig *config,
    double v)
{
    const double eta = carrier->eta;
    const double delta = (carrier->m1-carrier->m2)/carrier->Mtot;
    const double chi1 = config->chi1[2];
    const double chi2 = config->chi2[2];
    const double v2 = v*v;
    const double v3 = v2*v;
    const double v4 = v2*v2;
    const double v5 = v4*v;
    const double v6 = v3*v3;
    const double l2 = 1.5 + eta/6.0;
    const double l3 = 5.0*(chi1*(-2.0-2.0*delta+eta) +
                           chi2*(-2.0+2.0*delta+eta))/6.0;
    const double l4 = (81.0 + (-57.0+eta)*eta)/24.0;
    const double l5 = -7.0*(
        chi1*(72.0 + delta*(72.0-31.0*eta) + eta*(-121.0+2.0*eta)) +
        chi2*(72.0 + eta*(-121.0+2.0*eta) +
              delta*(-72.0+31.0*eta)))/144.0;
    /*
     * The LAL TPHM route enters IMRPhenomXGetAndSetPrecessionVariables()
     * without first initializing PhenomX's global powers_of_lalpi cache.
     * Consequently powers_of_lalpi.two is zero here, despite the source
     * expression nominally containing 2214*pi^2.  This cadence-independent
     * bug is intentionally reproduced only by the compatibility frame.
     */
    const double l6 = (10935.0 + eta*(-62001.0 + eta*(1674.0+7.0*eta)))/
                      1296.0;

    return (eta/v)*(1.0 + l2*v2 + l3*v3 + l4*v4 + l5*v5 + l6*v6);
}

static void build_j_frame(const IMRPhenomTHM *carrier,
                          const IMRPhenomTPHMPrecessionConfig *config,
                          const double reference_state[TPHM_STATE_SIZE],
                          IMRPhenomTPHMPrecessionSummary *summary,
                          int lal_compatible)
{
    double vref = carrier_v(carrier, config->tau_ref);
    double lmag;
    double projection;

    /*
     * LAL's numerical TPHM branch mixes two angular-momentum models: the
     * SpinTaylor RHS normalizes dL with the nonspinning 2PN expression,
     * while the fixed J frame is initialized with PhenomX version 102's
     * spin-dependent 3PN LRef.  Preserve that historical convention only
     * in the compatibility path.
     */
    lmag = lal_compatible ?
           lal_x102_reference_l_magnitude(carrier, config, vref) :
           orbital_l_magnitude(carrier->eta, vref);

    summary->j_frame_z[0] = reference_state[3] + reference_state[6];
    summary->j_frame_z[1] = reference_state[4] + reference_state[7];
    summary->j_frame_z[2] = lmag + reference_state[5] + reference_state[8];
    vec_normalize(summary->j_frame_z);

    /* Put Lhat(tref) in the positive x-z half-plane of the J frame. */
    projection = summary->j_frame_z[2];
    summary->j_frame_x[0] = -projection*summary->j_frame_z[0];
    summary->j_frame_x[1] = -projection*summary->j_frame_z[1];
    summary->j_frame_x[2] = 1.0 - projection*summary->j_frame_z[2];
    if(vec_normalize(summary->j_frame_x) != 0)
    {
        summary->j_frame_x[0] = 1.0;
        summary->j_frame_x[1] = 0.0;
        summary->j_frame_x[2] = 0.0;
    }
    vec_cross(summary->j_frame_z, summary->j_frame_x,
              summary->j_frame_y);
    vec_normalize(summary->j_frame_y);
}

static void state_to_angles(const double state[TPHM_STATE_SIZE],
                            const IMRPhenomTPHMPrecessionSummary *summary,
                            double *raw_alpha,
                            double *cosbeta)
{
    double lhat[3] = {state[0], state[1], state[2]};
    double norm = vec_norm(lhat);
    double lx;
    double ly;
    double lz;

    if(norm > 0.0)
    {
        lhat[0] /= norm;
        lhat[1] /= norm;
        lhat[2] /= norm;
    }
    lx = vec_dot(lhat, summary->j_frame_x);
    ly = vec_dot(lhat, summary->j_frame_y);
    lz = vec_dot(lhat, summary->j_frame_z);
    *raw_alpha = atan2(ly, lx);
    *cosbeta = clamp_unit(lz);
}

static double unwrap_near(double angle, double reference)
{
    while(angle-reference > M_PI) angle -= 2.0*M_PI;
    while(angle-reference < -M_PI) angle += 2.0*M_PI;
    return angle;
}

static double final_spin_from_state(const IMRPhenomTHM *carrier,
                                    const double peak_state[TPHM_STATE_SIZE],
                                    double *aligned_final_spin)
{
    double lhat[3] = {peak_state[0], peak_state[1], peak_state[2]};
    double spin1[3] = {peak_state[3], peak_state[4], peak_state[5]};
    double spin2[3] = {peak_state[6], peak_state[7], peak_state[8]};
    double spin_perp[3];
    double m1M = carrier->m1/carrier->Mtot;
    double m2M = carrier->m2/carrier->Mtot;
    double norm1 = m1M*m1M;
    double norm2 = m2M*m2M;
    double s1_l;
    double s2_l;
    double afinal;
    int k;

    vec_normalize(lhat);
    s1_l = vec_dot(spin1, lhat);
    s2_l = vec_dot(spin2, lhat);
    for(k = 0; k < 3; k++)
    {
        spin_perp[k] = (spin1[k] - s1_l*lhat[k]) +
                       (spin2[k] - s2_l*lhat[k]);
    }
    *aligned_final_spin = XLALSimIMRPhenomXFinalSpin2017(
        carrier->eta, s1_l/norm1, s2_l/norm2);
    afinal = copysign(hypot(vec_norm(spin_perp), *aligned_final_spin),
                      *aligned_final_spin);
    if(afinal >= 1.0) afinal = 1.0 - 1.0e-12;
    if(afinal <= -1.0) afinal = -1.0 + 1.0e-12;
    return afinal;
}

static int lal_compatible_gamma(int n,
                                const double *tau,
                                const double *alpha,
                                const double *cosbeta,
                                double tau_ref,
                                double *gamma)
{
    gsl_interp_accel *alpha_acc = NULL;
    gsl_interp_accel *cosbeta_acc = NULL;
    gsl_spline *alpha_spline = NULL;
    gsl_spline *cosbeta_spline = NULL;
    double offset;
    int reference_index = 0;
    int i;

    if(n < 3) return 1;
    while(reference_index + 1 < n && tau[reference_index + 1] <= tau_ref)
        reference_index++;

    alpha_acc = gsl_interp_accel_alloc();
    cosbeta_acc = gsl_interp_accel_alloc();
    alpha_spline = gsl_spline_alloc(gsl_interp_cspline, (size_t)n);
    cosbeta_spline = gsl_spline_alloc(gsl_interp_cspline, (size_t)n);
    if(alpha_acc == NULL || cosbeta_acc == NULL || alpha_spline == NULL ||
       cosbeta_spline == NULL)
        goto fail;
    if(gsl_spline_init(alpha_spline, tau, alpha, (size_t)n) != GSL_SUCCESS ||
       gsl_spline_init(cosbeta_spline, tau, cosbeta, (size_t)n) != GSL_SUCCESS)
        goto fail;

    gamma[0] = -alpha[0];
    for(i = 1; i < n; i++)
    {
        const double t1 = tau[i-1];
        const double t2 = tau[i];
        const double tq1 = 0.25*(t1 + 3.0*t2);
        const double tm = 0.5*(t1 + t2);
        const double tq3 = 0.25*(3.0*t1 + t2);
        const double f1 = -gsl_spline_eval_deriv(alpha_spline, t1, alpha_acc)*
                          gsl_spline_eval(cosbeta_spline, t1, cosbeta_acc);
        const double fq1 = -gsl_spline_eval_deriv(alpha_spline, tq1, alpha_acc)*
                           gsl_spline_eval(cosbeta_spline, tq1, cosbeta_acc);
        const double fm = -gsl_spline_eval_deriv(alpha_spline, tm, alpha_acc)*
                          gsl_spline_eval(cosbeta_spline, tm, cosbeta_acc);
        const double fq3 = -gsl_spline_eval_deriv(alpha_spline, tq3, alpha_acc)*
                           gsl_spline_eval(cosbeta_spline, tq3, cosbeta_acc);
        const double f2 = -gsl_spline_eval_deriv(alpha_spline, t2, alpha_acc)*
                          gsl_spline_eval(cosbeta_spline, t2, cosbeta_acc);
        gamma[i] = gamma[i-1] + (t2-t1)*(7.0*f1 + 32.0*fq1 + 12.0*fm +
                                          32.0*fq3 + 7.0*f2)/90.0;
    }

    offset = -gamma[reference_index] - alpha[reference_index];
    /* LAL leaves element zero unshifted. */
    for(i = 1; i < n; i++) gamma[i] += offset;

    gsl_spline_free(cosbeta_spline);
    gsl_spline_free(alpha_spline);
    gsl_interp_accel_free(cosbeta_acc);
    gsl_interp_accel_free(alpha_acc);
    return 0;

fail:
    if(cosbeta_spline != NULL) gsl_spline_free(cosbeta_spline);
    if(alpha_spline != NULL) gsl_spline_free(alpha_spline);
    if(cosbeta_acc != NULL) gsl_interp_accel_free(cosbeta_acc);
    if(alpha_acc != NULL) gsl_interp_accel_free(alpha_acc);
    return 1;
}

static int evolve_euler_angles(
    const IMRPhenomTHM *carrier,
    const IMRPhenomTPHMPrecessionConfig *config,
    int n,
    const double *tau,
    double *alpha,
    double *beta,
    double *gamma,
    IMRPhenomTPHMPrecessionSummary *summary,
    int lal_compatible)
{
    TPHMPrecessionRHS rhs;
    gsl_odeiv2_system system;
    gsl_odeiv2_driver *forward = NULL;
    gsl_odeiv2_driver *backward = NULL;
    gsl_interp_accel *v_acc = NULL;
    gsl_spline *v_spline = NULL;
    double reference_state[TPHM_STATE_SIZE];
    double peak_state[TPHM_STATE_SIZE];
    double state[TPHM_STATE_SIZE];
    double *raw_alpha = NULL;
    double *cosbeta = NULL;
    double *v_samples = NULL;
    double abs_tol;
    double rel_tol;
    double initial_step;
    double current_tau;
    double alpha_ref;
    double cosbeta_ref;
    int first_at_or_after_ref;
    int first_after_peak;
    int reference_index;
    int i;
    int status = 0;

    if(carrier == NULL || config == NULL || tau == NULL || alpha == NULL ||
       beta == NULL || gamma == NULL || summary == NULL || n < 1)
        return 1;
    if(!isfinite(config->tau_ref) || config->tau_ref > 0.0)
        return 2;
    for(i = 0; i < n; i++)
    {
        if(!isfinite(tau[i]) || (i > 0 && tau[i] <= tau[i-1])) return 3;
    }
    for(i = 0; i < 3; i++)
    {
        if(!isfinite(config->chi1[i]) || !isfinite(config->chi2[i])) return 4;
    }
    if(vec_norm(config->chi1) > 1.0 + 1.0e-12 ||
       vec_norm(config->chi2) > 1.0 + 1.0e-12)
        return 5;
    if(fabs(config->chi1[2]-carrier->chi1) > 1.0e-12 ||
       fabs(config->chi2[2]-carrier->chi2) > 1.0e-12)
        return 11;

    memset(summary, 0, sizeof(*summary));
    initial_state(carrier, config, reference_state);
    build_j_frame(carrier, config, reference_state, summary,
                  lal_compatible);
    state_to_angles(reference_state, summary, &alpha_ref, &cosbeta_ref);
    /*
     * LAL fixes the residual azimuthal gauge with the direction of the total
     * in-plane spin.  build_j_frame() makes the raw reference azimuth zero,
     * so this is precisely LAL's alphaOff.  Minimal rotation then gives
     * gamma_ref=-alpha_ref.  Retaining these constants is essential when the
     * co-precessing modes are rotated back to the inertial L0 frame.
     */
    {
        double m1M = carrier->m1/carrier->Mtot;
        double m2M = carrier->m2/carrier->Mtot;
        double sx = m1M*m1M*config->chi1[0] +
                    m2M*m2M*config->chi2[0];
        double sy = m1M*m1M*config->chi1[1] +
                    m2M*m2M*config->chi2[1];
        alpha_ref = atan2(sy, sx) - M_PI;
    }
    summary->alpha_reference = alpha_ref;
    summary->gamma_reference = -alpha_ref;
    reference_state[9] = summary->gamma_reference;

    abs_tol = config->absolute_tolerance > 0.0 ?
              config->absolute_tolerance :
              (lal_compatible ? TPHM_LAL_ABS_TOL : TPHM_DEFAULT_ABS_TOL);
    rel_tol = config->relative_tolerance > 0.0 ?
              config->relative_tolerance :
              (lal_compatible ? TPHM_LAL_REL_TOL : TPHM_DEFAULT_REL_TOL);
    initial_step = config->initial_step > 0.0 ?
                   config->initial_step : TPHM_DEFAULT_INITIAL_STEP;

    rhs.carrier = carrier;
    rhs.eta = carrier->eta;
    rhs.m1M = carrier->m1/carrier->Mtot;
    rhs.m2M = carrier->m2/carrier->Mtot;
    rhs.spin1_dot3 = spin_dot_3pn(rhs.m1M);
    rhs.spin2_dot3 = spin_dot_3pn(rhs.m2M);
    rhs.spin1_dot5 = spin_dot_5pn(rhs.m1M);
    rhs.spin2_dot5 = spin_dot_5pn(rhs.m2M);
    rhs.spin1_dot7 = spin_dot_7pn(rhs.m1M);
    rhs.spin2_dot7 = spin_dot_7pn(rhs.m2M);
    rhs.v_spline = NULL;
    rhs.v_acc = NULL;
    rhs.v_time_offset = 0.0;
    rhs.frame = summary;
    rhs.integrate_gamma = !lal_compatible;

    /*
     * LAL first samples v(t) on the waveform grid and then feeds a cubic
     * spline of those samples to SpinTaylor.  The continuous carrier is the
     * cleaner ODE driver, but preserving the sampled driver here makes the
     * compatibility policy sensitive to the same cadence-dependent detail.
     */
    if(lal_compatible && n >= 3 && tau[0] <= config->tau_ref &&
       tau[n-1] >= 0.0)
    {
        v_samples = (double *)calloc((size_t)n, sizeof(*v_samples));
        v_acc = gsl_interp_accel_alloc();
        v_spline = gsl_spline_alloc(gsl_interp_cspline, (size_t)n);
        if(v_samples == NULL || v_acc == NULL || v_spline == NULL)
        {
            status = 6;
            goto cleanup;
        }
        for(i = 0; i < n; i++) v_samples[i] = carrier_v(carrier, tau[i]);
        if(gsl_spline_init(v_spline, tau, v_samples, (size_t)n) != GSL_SUCCESS)
        {
            status = 6;
            goto cleanup;
        }
        rhs.v_spline = v_spline;
        rhs.v_acc = v_acc;
    }
    system.function = precession_rhs;
    system.jacobian = NULL;
    system.dimension = lal_compatible ? 9 : TPHM_STATE_SIZE;
    system.params = &rhs;

    forward = make_driver(&system, initial_step, abs_tol, rel_tol, 1);
    backward = make_driver(&system, initial_step, abs_tol, rel_tol, -1);
    raw_alpha = (double *)calloc((size_t)n, sizeof(*raw_alpha));
    cosbeta = (double *)calloc((size_t)n, sizeof(*cosbeta));
    if(forward == NULL || backward == NULL || raw_alpha == NULL || cosbeta == NULL)
    {
        status = 6;
        goto cleanup;
    }

    first_at_or_after_ref = 0;
    while(first_at_or_after_ref < n && tau[first_at_or_after_ref] < config->tau_ref)
        first_at_or_after_ref++;
    first_after_peak = 0;
    while(first_after_peak < n && tau[first_after_peak] <= 0.0)
        first_after_peak++;
    reference_index = first_at_or_after_ref - 1;

    if(lal_compatible && reference_index >= 0)
    {
        double dtau = n > 1 ? tau[1]-tau[0] : 0.0;

        /* LAL's Hermite-output bookkeeping places the reference state in two
         * adjacent waveform samples.  The forward branch is anchored at the
         * last sample not later than t_ref; the backward branch is anchored
         * one sample earlier and evaluates v one cadence earlier.  This is a
         * grid-indexing convention rather than precession physics, so it is
         * intentionally confined to the compatibility policy. */
        memcpy(state, reference_state, sizeof(state));
        state_to_angles(state, summary, &raw_alpha[reference_index],
                        &cosbeta[reference_index]);
        raw_alpha[reference_index] += alpha_ref;
        if(reference_index > 0)
        {
            memcpy(state, reference_state, sizeof(state));
            state_to_angles(state, summary, &raw_alpha[reference_index-1],
                            &cosbeta[reference_index-1]);
            raw_alpha[reference_index-1] += alpha_ref;
            current_tau = config->tau_ref;
            rhs.v_time_offset = -dtau;
            for(i = reference_index-2; i >= 0; i--)
            {
                double target_tau = config->tau_ref +
                    tau[i]-tau[reference_index-1];
                if(evolve_to(backward, &current_tau, target_tau, state) !=
                   GSL_SUCCESS)
                {
                    status = 7;
                    goto cleanup;
                }
                state_to_angles(state, summary, &raw_alpha[i], &cosbeta[i]);
                raw_alpha[i] += alpha_ref;
            }
        }

        memcpy(state, reference_state, sizeof(state));
        current_tau = config->tau_ref;
        rhs.v_time_offset = 0.0;
        for(i = reference_index; i < first_after_peak; i++)
        {
            double target_tau = config->tau_ref +
                tau[i]-tau[reference_index];
            if(evolve_to(forward, &current_tau, target_tau, state) !=
               GSL_SUCCESS)
            {
                status = 8;
                goto cleanup;
            }
            state_to_angles(state, summary, &raw_alpha[i], &cosbeta[i]);
            raw_alpha[i] += alpha_ref;
            if(!lal_compatible) gamma[i] = state[9];
        }
    }
    else
    {
        memcpy(state, reference_state, sizeof(state));
        current_tau = config->tau_ref;
        for(i = first_at_or_after_ref-1; i >= 0; i--)
        {
            if(evolve_to(backward, &current_tau, tau[i], state) != GSL_SUCCESS)
            {
                status = 7;
                goto cleanup;
            }
            state_to_angles(state, summary, &raw_alpha[i], &cosbeta[i]);
            raw_alpha[i] += alpha_ref;
            if(!lal_compatible) gamma[i] = state[9];
        }

        memcpy(state, reference_state, sizeof(state));
        current_tau = config->tau_ref;
        for(i = first_at_or_after_ref; i < first_after_peak; i++)
        {
            if(evolve_to(forward, &current_tau, tau[i], state) != GSL_SUCCESS)
            {
                status = 8;
                goto cleanup;
            }
            state_to_angles(state, summary, &raw_alpha[i], &cosbeta[i]);
            raw_alpha[i] += alpha_ref;
            if(!lal_compatible) gamma[i] = state[9];
        }
    }
    rhs.v_time_offset = 0.0;
    if(evolve_to(forward, &current_tau, 0.0, state) != GSL_SUCCESS)
    {
        status = 9;
        goto cleanup;
    }
    memcpy(peak_state, state, sizeof(peak_state));

    summary->final_spin = final_spin_from_state(carrier, peak_state,
                                                 &summary->aligned_final_spin_at_peak);
    summary->ringdown_alpha_slope = IMRPhenomTHMEulerRingdownSlope(
        summary->final_spin, carrier->Mfinal);
    if(!isfinite(summary->ringdown_alpha_slope))
    {
        status = 10;
        goto cleanup;
    }

    /* Unwrap independently away from the reference epoch. */
    {
        double previous = alpha_ref;
        for(i = first_at_or_after_ref; i < first_after_peak; i++)
        {
            raw_alpha[i] = unwrap_near(raw_alpha[i], previous);
            previous = raw_alpha[i];
        }
        previous = alpha_ref;
        for(i = first_at_or_after_ref-1; i >= 0; i--)
        {
            raw_alpha[i] = unwrap_near(raw_alpha[i], previous);
            previous = raw_alpha[i];
        }
    }

    if(lal_compatible && first_after_peak >= 3 && n >= 3)
    {
        int reference_index = 0;
        int continuation_start = first_after_peak - 2;
        int continuation_anchor = continuation_start - 1;
        double alpha_shift;

        while(reference_index + 1 < continuation_start &&
              tau[reference_index + 1] <= config->tau_ref)
            reference_index++;
        alpha_shift = alpha_ref - raw_alpha[reference_index];
        for(i = 0; i < continuation_start; i++) raw_alpha[i] += alpha_shift;

        for(i = continuation_start; i < n; i++)
        {
            raw_alpha[i] = raw_alpha[continuation_anchor] +
                           summary->ringdown_alpha_slope*tau[i];
            cosbeta[i] = cosbeta[continuation_anchor];
        }
        if(lal_compatible_gamma(n, tau, raw_alpha, cosbeta,
                                config->tau_ref, gamma) != 0)
        {
            status = 12;
            goto cleanup;
        }
    }
    else
    {
    /* The adaptive ODE carries the minimal-rotation angle from tau_ref. */
    /* Obtain the peak angles from the evolved peak state. */
    {
        double peak_alpha;
        double peak_cosbeta;
        double peak_gamma = peak_state[9];
        double previous_alpha = alpha_ref;

        state_to_angles(peak_state, summary, &peak_alpha, &peak_cosbeta);
        peak_alpha += alpha_ref;
        if(first_after_peak > first_at_or_after_ref)
        {
            i = first_after_peak-1;
            previous_alpha = raw_alpha[i];
        }
        peak_alpha = unwrap_near(peak_alpha, previous_alpha);

        for(i = first_after_peak; i < n; i++)
        {
            raw_alpha[i] = peak_alpha + summary->ringdown_alpha_slope*tau[i];
            cosbeta[i] = peak_cosbeta;
            gamma[i] = peak_gamma - peak_cosbeta*
                       (raw_alpha[i] - peak_alpha);
        }
    }
    }

    for(i = 0; i < n; i++)
    {
        alpha[i] = raw_alpha[i];
        beta[i] = acos(clamp_unit(cosbeta[i]));
    }

cleanup:
    if(v_spline != NULL) gsl_spline_free(v_spline);
    if(v_acc != NULL) gsl_interp_accel_free(v_acc);
    free(v_samples);
    free(raw_alpha);
    free(cosbeta);
    if(forward != NULL) gsl_odeiv2_driver_free(forward);
    if(backward != NULL) gsl_odeiv2_driver_free(backward);
    return status;
}

int IMRPhenomTPHMEvolveEulerAngles(
    const IMRPhenomTHM *carrier,
    const IMRPhenomTPHMPrecessionConfig *config,
    int n,
    const double *tau,
    double *alpha,
    double *beta,
    double *gamma,
    IMRPhenomTPHMPrecessionSummary *summary)
{
    return evolve_euler_angles(carrier, config, n, tau, alpha, beta, gamma,
                               summary, 0);
}

int IMRPhenomTPHMEvolveEulerAnglesLALCompatible(
    const IMRPhenomTHM *carrier,
    const IMRPhenomTPHMPrecessionConfig *config,
    int n,
    const double *tau,
    double *alpha,
    double *beta,
    double *gamma,
    IMRPhenomTPHMPrecessionSummary *summary)
{
    return evolve_euler_angles(carrier, config, n, tau, alpha, beta, gamma,
                               summary, 1);
}
