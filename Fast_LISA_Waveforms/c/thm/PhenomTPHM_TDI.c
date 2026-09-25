/*
 * LISA response and sparse WDM implementation for IMRPhenomTPHM.
 * Copyright (C) 2026 Neil Cornish
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Free software under GNU GPL version 3 or later, without warranty.
 * The LAL-derived waveform model has separate upstream notices.
 */

#include "PhenomTPHM_TDI.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_integration.h>

#define TPHM_GPC_SECONDS 1.0292712503794875e17

static int tphm_complex_envelope_groups_override = -1;
static int tphm_post_tdi_envelope_groups_override = -1;

typedef struct
{
    const IMRPhenomTHM *carrier;
} TPHMPhaseIntegral;

static int tphm_evaluate_factorized_carrier_polarizations(
    PhenomTPHMTDISource *source,
    int n,
    const double *source_time,
    double complex *hplus,
    double complex *hcross);

static double tphm_phase_integrand(double tau, void *userdata)
{
    const TPHMPhaseIntegral *integral =
        (const TPHMPhaseIntegral *)userdata;
    return IMRPhenomTHMOmega22(integral->carrier, tau);
}

static int tphm_phase_from_anchor(const PhenomTPHMTDISource *source,
                                  double tau,
                                  double *phase)
{
    const double tau_anchor =
        (source->source_time_anchor-source->coalescence_time)/
        source->total_mass;
    TPHMPhaseIntegral integral;
    gsl_integration_workspace *workspace;
    gsl_function function;
    double result = 0.0;
    double error = 0.0;
    int status;

    if(phase == NULL) return 1;
    if(tau == tau_anchor)
    {
        *phase = source->phi22_anchor;
        return 0;
    }

    workspace = gsl_integration_workspace_alloc(2048);
    if(workspace == NULL) return 2;
    integral.carrier = &source->waveform.carrier;
    function.function = tphm_phase_integrand;
    function.params = &integral;
    status = gsl_integration_qag(&function, tau_anchor, tau,
                                 1.0e-10, 1.0e-11, 2048,
                                 GSL_INTEG_GAUSS31, workspace,
                                 &result, &error);
    gsl_integration_workspace_free(workspace);
    if(status != GSL_SUCCESS || !isfinite(result)) return 3;
    *phase = source->phi22_anchor+result;
    return 0;
}

int phenom_tphm_tdi_source_initialize(
    PhenomTPHMTDISource *source,
    double m1_seconds,
    double m2_seconds,
    const IMRPhenomTPHMPrecessionConfig *precession,
    const IMRPhenomTHMMode *carriers,
    int ncarriers,
    double coalescence_time,
    double distance_gpc,
    double source_time_anchor,
    double phi22_anchor,
    double observer_theta,
    double observer_phi,
    double polarization)
{
    double total_mass;
    double tau_anchor;
    double phase_at_cut;
    TPHMPhaseIntegral integral;
    gsl_integration_workspace *integration_workspace = NULL;
    gsl_function function;
    double phase_integral = 0.0;
    double phase_error = 0.0;
    int status;

    if(source == NULL || precession == NULL || m1_seconds <= 0.0 ||
       m2_seconds <= 0.0 || distance_gpc <= 0.0 ||
       !isfinite(coalescence_time) || !isfinite(source_time_anchor))
        return 1;
    memset(source, 0, sizeof(*source));
    status = IMRPhenomTPHMInitialize(&source->waveform,
                                      m1_seconds, m2_seconds,
                                      precession, carriers, ncarriers);
    if(status != 0) return 10+status;

    total_mass = m1_seconds+m2_seconds;
    source->total_mass = total_mass;
    source->eta = m1_seconds*m2_seconds/(total_mass*total_mass);
    source->coalescence_time = coalescence_time;
    source->source_time_anchor = source_time_anchor;
    source->phi22_anchor = phi22_anchor;
    source->observer_theta = observer_theta;
    source->observer_phi = observer_phi;
    source->polarization = polarization;
    source->strain_scale = sqrt(2.0)*source->eta*total_mass/
                           (distance_gpc*TPHM_GPC_SECONDS);

    /* Anchor the higher-mode merger phases once.  Subsequent partitioned
     * grids can lie wholly before or after tCUT_Freq without changing the
     * physical phase convention. */
    tau_anchor = (source_time_anchor-coalescence_time)/total_mass;
    integration_workspace = gsl_integration_workspace_alloc(2048);
    if(integration_workspace == NULL)
    {
        phenom_tphm_tdi_source_destroy(source);
        return 20;
    }
    integral.carrier = &source->waveform.carrier;
    function.function = tphm_phase_integrand;
    function.params = &integral;
    status = gsl_integration_qag(&function, tau_anchor, tCUT_Freq,
                                 1.0e-10, 1.0e-11, 2048,
                                 GSL_INTEG_GAUSS31,
                                 integration_workspace,
                                 &phase_integral, &phase_error);
    gsl_integration_workspace_free(integration_workspace);
    if(status != GSL_SUCCESS || !isfinite(phase_integral))
    {
        phenom_tphm_tdi_source_destroy(source);
        return 21;
    }
    phase_at_cut = phi22_anchor+phase_integral;
    IMRPhenomTHMSetPhaseOffsets(&source->waveform.carrier, phase_at_cut);
    source->initialized = 1;
    return 0;
}

void phenom_tphm_tdi_source_destroy(PhenomTPHMTDISource *source)
{
    if(source == NULL) return;
    IMRPhenomTPHMDestroy(&source->waveform);
    memset(source, 0, sizeof(*source));
}

int phenom_tphm_complex_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    PhenomTPHMTDISource *source = (PhenomTPHMTDISource *)userdata;
    double *tau = NULL;
    double complex *strain = NULL;
    double complex *quadrature = NULL;
    double phi_at_start;
    int i;
    int status;

    if(source == NULL || !source->initialized || n < 1 ||
       source_time == NULL || hplus_analytic == NULL ||
       hcross_analytic == NULL)
        return 1;
    for(i=1; i<n; i++)
    {
        if(source_time[i] <= source_time[i-1]) return 2;
    }

    tau = calloc((size_t)n, sizeof(*tau));
    strain = calloc((size_t)n, sizeof(*strain));
    quadrature = calloc((size_t)n, sizeof(*quadrature));
    if(tau == NULL || strain == NULL || quadrature == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(i=0; i<n; i++)
    {
        tau[i] = (source_time[i]-source->coalescence_time)/
                 source->total_mass;
    }
    status = tphm_phase_from_anchor(source, tau[0], &phi_at_start);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    status = IMRPhenomTPHMEvaluateGridWithQuadrature(
        &source->waveform, n, tau, phi_at_start,
        source->observer_theta, source->observer_phi,
        source->polarization, NULL, NULL, strain, quadrature,
        NULL, NULL, NULL);
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }

    for(i=0; i<n; i++)
    {
        hplus_analytic[i] = source->strain_scale*
            (creal(strain[i])-I*creal(quadrature[i]));
        hcross_analytic[i] = source->strain_scale*
            (-cimag(strain[i])+I*cimag(quadrature[i]));
    }
    status = 0;

cleanup:
    free(quadrature);
    free(strain);
    free(tau);
    return status;
}

static int tphm_evaluate_carrier_polarization_arrays(
    PhenomTPHMTDISource *source,
    int n,
    const double *source_time,
    double complex *hplus,
    double complex *hcross,
    double *omega,
    double *alpha,
    double *beta,
    double *gamma)
{
    double *tau = NULL;
    double complex *carrier_strain = NULL;
    double complex *carrier_quadrature = NULL;
    double phi_at_start;
    int nc, i, k, status;

    if(source == NULL || !source->initialized || n < 1 ||
       source_time == NULL || hplus == NULL || hcross == NULL)
        return 1;
    for(i=1; i<n; i++)
    {
        if(source_time[i] <= source_time[i-1]) return 2;
    }
    nc = source->waveform.ncarriers;
    tau = calloc((size_t)n, sizeof(*tau));
    carrier_strain = calloc((size_t)nc*(size_t)n,
                            sizeof(*carrier_strain));
    carrier_quadrature = calloc((size_t)nc*(size_t)n,
                                sizeof(*carrier_quadrature));
    if(tau == NULL || carrier_strain == NULL || carrier_quadrature == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(i=0; i<n; i++)
        tau[i] = (source_time[i]-source->coalescence_time)/source->total_mass;
    status = tphm_phase_from_anchor(source, tau[0], &phi_at_start);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    status = IMRPhenomTPHMEvaluateCarrierGridWithQuadrature(
        &source->waveform, n, tau, phi_at_start,
        source->observer_theta, source->observer_phi,
        source->polarization, carrier_strain, carrier_quadrature,
        omega, alpha, beta, gamma);
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }
    for(k=0; k<nc; k++)
    {
        for(i=0; i<n; i++)
        {
            size_t q = (size_t)k*(size_t)n+(size_t)i;
            hplus[q] = source->strain_scale*
                (creal(carrier_strain[q])-I*creal(carrier_quadrature[q]));
            hcross[q] = source->strain_scale*
                (-cimag(carrier_strain[q])+I*cimag(carrier_quadrature[q]));
        }
    }
    status = 0;

cleanup:
    free(carrier_quadrature);
    free(carrier_strain);
    free(tau);
    return status;
}

int phenom_tphm_carrier_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    PhenomTPHMCarrierEvaluator *selection =
        (PhenomTPHMCarrierEvaluator *)userdata;
    double *tau = NULL;
    double complex *strain = NULL;
    double complex *quadrature = NULL;
    double phi_at_start;
    int nc, i, status;

    if(selection == NULL || selection->source == NULL || n < 1 ||
       hplus_analytic == NULL || hcross_analytic == NULL)
        return 1;
    nc = selection->source->waveform.ncarriers;
    if(selection->carrier_index < 0 || selection->carrier_index >= nc)
        return 2;
    for(i=1; i<n; i++)
    {
        if(source_time[i] <= source_time[i-1]) return 3;
    }
    tau = calloc((size_t)n, sizeof(*tau));
    strain = calloc((size_t)n, sizeof(*strain));
    quadrature = calloc((size_t)n, sizeof(*quadrature));
    if(tau == NULL || strain == NULL || quadrature == NULL)
    {
        status = 4;
        goto cleanup;
    }
    for(i=0; i<n; i++)
        tau[i] = (source_time[i]-selection->source->coalescence_time)/
                 selection->source->total_mass;
    status = tphm_phase_from_anchor(selection->source, tau[0],
                                    &phi_at_start);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    status = IMRPhenomTPHMEvaluateSelectedCarrierGridWithQuadrature(
        &selection->source->waveform, selection->carrier_index,
        n, tau, phi_at_start, selection->source->observer_theta,
        selection->source->observer_phi, selection->source->polarization,
        strain, quadrature, NULL, NULL, NULL, NULL);
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }
    for(i=0; i<n; i++)
    {
        hplus_analytic[i] = selection->source->strain_scale*
            (creal(strain[i])-I*creal(quadrature[i]));
        hcross_analytic[i] = selection->source->strain_scale*
            (-cimag(strain[i])+I*cimag(quadrature[i]));
    }
    status = 0;

cleanup:
    free(quadrature);
    free(strain);
    free(tau);
    return status;
}

int phenom_tphm_all_carrier_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    int nfamilies,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    PhenomTPHMTDISource *source = (PhenomTPHMTDISource *)userdata;
    if(source == NULL || nfamilies != source->waveform.ncarriers)
        return 1;
    return tphm_evaluate_carrier_polarization_arrays(
        source, n, source_time, hplus_analytic, hcross_analytic,
        NULL, NULL, NULL, NULL);
}

static int tphm_factorized_total_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus,
    double complex *hcross)
{
    PhenomTPHMTDISource *source = (PhenomTPHMTDISource *)userdata;
    double complex *family_plus = NULL, *family_cross = NULL;
    int nc, i, k, status;
    if(source == NULL || n < 1 || hplus == NULL || hcross == NULL) return 1;
    nc = source->waveform.ncarriers;
    family_plus = calloc((size_t)nc*(size_t)n, sizeof(*family_plus));
    family_cross = calloc((size_t)nc*(size_t)n, sizeof(*family_cross));
    if(family_plus == NULL || family_cross == NULL)
    {
        status = 2;
        goto cleanup;
    }
    status = tphm_evaluate_factorized_carrier_polarizations(
        source, n, source_time, family_plus, family_cross);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    for(i=0; i<n; i++)
    {
        hplus[i] = 0.0;
        hcross[i] = 0.0;
        for(k=0; k<nc; k++)
        {
            size_t q = (size_t)k*(size_t)n+(size_t)i;
            hplus[i] += family_plus[q];
            hcross[i] += family_cross[q];
        }
    }
    status = 0;

cleanup:
    free(family_cross);
    free(family_plus);
    return status;
}

int phenom_tphm_evaluate_tdi_context(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    PhenomTPHMTDISource *source,
    int n,
    const double *detector_time,
    double ecliptic_latitude,
    double ecliptic_longitude,
    double complex *X,
    double complex *Y,
    double complex *Z)
{
    return thm_evaluate_complex_tdi_context(
        context, workspace, n, detector_time,
        ecliptic_latitude, ecliptic_longitude,
        phenom_tphm_complex_polarizations, source, X, Y, Z);
}

static int tphm_next_power_of_two(int n)
{
    int p = 1;
    if(n <= 1) return 1;
    while(p < n && p <= (1<<29)) p <<= 1;
    return p;
}

static int tphm_largest_power_of_two(int n)
{
    int p = 1;
    if(n < 1) return 0;
    while(p <= n/2) p <<= 1;
    return p;
}

void phenom_tphm_partition_plan_init(PhenomTPHMPartitionPlan *plan)
{
    if(plan == NULL) return;
    memset(plan, 0, sizeof(*plan));
}

void phenom_tphm_partition_plan_free(PhenomTPHMPartitionPlan *plan)
{
    int i;
    if(plan == NULL) return;
    for(i=0; i<plan->nblocks; i++)
        thm_complex_tdi_fft_block_free(&plan->blocks[i]);
    free(plan->endpoint_wdm_nsize);
    free(plan->endpoint_wdm_nmid);
    free(plan->wdm_nsize);
    free(plan->wdm_nmid);
    free(plan->blocks);
    phenom_tphm_partition_plan_init(plan);
}

void phenom_tphm_set_complex_envelope_groups(int enabled)
{
    tphm_complex_envelope_groups_override = enabled < 0 ? -1 : !!enabled;
}

void phenom_tphm_set_post_tdi_envelope_groups(int enabled)
{
    tphm_post_tdi_envelope_groups_override = enabled < 0 ? -1 : !!enabled;
}

void phenom_tphm_default_endpoint_window(
    const PhenomTPHMTDISource *source,
    double *endpoint_start,
    double *endpoint_rise,
    double *waveform_stop)
{
    double delay_margin;
    if(source == NULL) return;
    delay_margin = 500.0+4.0*THM_TDI_LARM_NOMINAL_SECONDS;
    if(endpoint_start != NULL)
        *endpoint_start = source->coalescence_time-delay_margin-
                          10000.0*source->total_mass;
    if(endpoint_rise != NULL)
        *endpoint_rise = 5000.0*source->total_mass;
    if(waveform_stop != NULL)
        *waveform_stop = source->coalescence_time+500.0+
                         1000.0*source->total_mass;
}

void phenom_tphm_default_endpoint_window_context(
    const THMObservationContext *context,
    const PhenomTPHMTDISource *source,
    double *endpoint_start,
    double *endpoint_rise,
    double *waveform_stop)
{
    phenom_tphm_default_endpoint_window(source, endpoint_start,
                                         endpoint_rise, waveform_stop);
    if(source != NULL && endpoint_start != NULL &&
       thm_observation_tdi_generation(context) == 2)
        *endpoint_start -= 4.0*THM_TDI_LARM_NOMINAL_SECONDS;
}

static int tphm_partition_plan_reserve(PhenomTPHMPartitionPlan *plan,
                                       int needed)
{
    THMComplexTDIFFTBlock *grown;
    int capacity, i;
    if(plan == NULL || needed < 0) return 0;
    if(needed <= plan->capacity) return 1;
    capacity = plan->capacity > 0 ? plan->capacity : 16;
    while(capacity < needed) capacity *= 2;
    grown = realloc(plan->blocks, (size_t)capacity*sizeof(*grown));
    if(grown == NULL) return 0;
    plan->blocks = grown;
    for(i=plan->capacity; i<capacity; i++)
        thm_complex_tdi_fft_block_init(&plan->blocks[i]);
    plan->capacity = capacity;
    return 1;
}

static int tphm_track_range(int n, const double *time,
                            const double *flow, const double *fhigh,
                            double tlo, double thi,
                            double *out_low, double *out_high)
{
    int i, have = 0;
    double lo = HUGE_VAL, hi = -HUGE_VAL;
    if(n < 1 || time == NULL || flow == NULL || fhigh == NULL ||
       out_low == NULL || out_high == NULL || !(thi > tlo))
        return 0;
    for(i=0; i<n; i++)
    {
        if(time[i] < tlo || time[i] > thi) continue;
        if(!isfinite(flow[i]) || !isfinite(fhigh[i])) continue;
        if(flow[i] < lo) lo = flow[i];
        if(fhigh[i] > hi) hi = fhigh[i];
        have = 1;
    }
    if(!have)
    {
        int nearest = 0;
        double center = 0.5*(tlo+thi);
        for(i=1; i<n; i++)
        {
            if(fabs(time[i]-center) < fabs(time[nearest]-center)) nearest = i;
        }
        lo = flow[nearest];
        hi = fhigh[nearest];
        have = isfinite(lo) && isfinite(hi);
    }
    *out_low = lo;
    *out_high = hi;
    return have && hi >= lo;
}

static double tphm_angle_derivative(int n, const double *time,
                                    const double *angle, int i)
{
    if(n < 2) return 0.0;
    if(i <= 0) return (angle[1]-angle[0])/(time[1]-time[0]);
    if(i >= n-1)
        return (angle[n-1]-angle[n-2])/(time[n-1]-time[n-2]);
    return (angle[i+1]-angle[i-1])/(time[i+1]-time[i-1]);
}

static int tphm_build_planning_tracks(
    PhenomTPHMTDISource *source,
    const THMWDMGridInfo *grid,
    double time_start,
    double time_stop,
    int *ntrack_out,
    double **time_out,
    double **fcenter_out,
    double **flow_out,
    double **fhigh_out)
{
    const double seconds_per_year = 31557600.0;
    double *time = NULL, *omega = NULL, *fcenter = NULL, *alpha = NULL;
    double *beta = NULL, *gamma = NULL, *flow = NULL, *fhigh = NULL;
    double complex *hplus = NULL, *hcross = NULL;
    int nc, ntrack, i, k, status;
    double track_start;

    if(source == NULL || grid == NULL || ntrack_out == NULL ||
       time_out == NULL || fcenter_out == NULL || flow_out == NULL ||
       fhigh_out == NULL ||
       !(time_stop > time_start))
        return 1;
    nc = source->waveform.ncarriers;
    track_start = floor(time_start/grid->time_pixel_dt)*grid->time_pixel_dt;
    if(track_start < 0.0) track_start = 0.0;
    ntrack = (int)ceil((time_stop-track_start)/grid->time_pixel_dt)+1;
    if(ntrack < 2) ntrack = 2;
    time = calloc((size_t)ntrack, sizeof(*time));
    omega = calloc((size_t)nc*(size_t)ntrack, sizeof(*omega));
    fcenter = calloc((size_t)nc*(size_t)ntrack, sizeof(*fcenter));
    flow = calloc((size_t)nc*(size_t)ntrack, sizeof(*flow));
    fhigh = calloc((size_t)nc*(size_t)ntrack, sizeof(*fhigh));
    alpha = calloc((size_t)ntrack, sizeof(*alpha));
    beta = calloc((size_t)ntrack, sizeof(*beta));
    gamma = calloc((size_t)ntrack, sizeof(*gamma));
    hplus = calloc((size_t)nc*(size_t)ntrack, sizeof(*hplus));
    hcross = calloc((size_t)nc*(size_t)ntrack, sizeof(*hcross));
    if(time == NULL || omega == NULL || fcenter == NULL ||
       flow == NULL || fhigh == NULL ||
       alpha == NULL || beta == NULL || gamma == NULL || hplus == NULL ||
       hcross == NULL)
    {
        status = 2;
        goto cleanup;
    }
    for(i=0; i<ntrack; i++)
        time[i] = track_start+grid->time_pixel_dt*(double)i;
    status = tphm_evaluate_carrier_polarization_arrays(
        source, ntrack, time, hplus, hcross, omega, alpha, beta, gamma);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    for(i=0; i<ntrack; i++)
    {
        double adot = tphm_angle_derivative(ntrack, time, alpha, i);
        double gdot = tphm_angle_derivative(ntrack, time, gamma, i);
        for(k=0; k<nc; k++)
        {
            int ell = source->waveform.carrier_modes[k].ell;
            int emm = abs(source->waveform.carrier_modes[k].emm);
            size_t q = (size_t)k*(size_t)ntrack+(size_t)i;
            double center = fabs(omega[q])/
                (2.0*M_PI*source->total_mass);
            double sideband = ((double)ell*fabs(adot)+
                               (double)emm*fabs(gdot))/(2.0*M_PI);
            double tdi_margin = 3.0e-4*center+4.0/seconds_per_year;
            fcenter[q] = center;
            flow[q] = fmax(0.0, center-sideband-tdi_margin);
            fhigh[q] = center+sideband+tdi_margin;
        }
    }
    *ntrack_out = ntrack;
    *time_out = time;
    *fcenter_out = fcenter;
    *flow_out = flow;
    *fhigh_out = fhigh;
    time = fcenter = flow = fhigh = NULL;
    status = 0;

cleanup:
    free(hcross);
    free(hplus);
    free(gamma);
    free(beta);
    free(alpha);
    free(fhigh);
    free(flow);
    free(fcenter);
    free(omega);
    free(time);
    return status;
}

/*
 * Convert the time-local precessing carrier bands into one compact packet per
 * WDM frequency layer.  The Euler-angle rates broaden a folded carrier into a
 * narrow sideband family, but that physical broadening is normally much less
 * than the rectangular frequency range assigned to a long FFT block.  Using
 * the block rectangles as the WDM mask therefore retains many pixels that no
 * precessing track ever visits.
 *
 * The endpoint FFT is additive, not a frequency handoff.  Its rising taper
 * produces sidebands below the lowest instantaneous endpoint frequency, so a
 * 3/rise margin is retained, capped at half that frequency as in the aligned
 * THM partitioned transform.  The upper edge follows the largest physical
 * precessing band over the endpoint, with the same taper/Meyer margin.
 */
static void tphm_packet_from_bounds(const THMWDMGridInfo *grid,
                                    int jlo, int jhi,
                                    int *nmid, int *nsize)
{
    int needed, block = 1, center;
    *nmid = -1;
    *nsize = 0;
    if(grid == NULL || jhi < jlo) return;
    needed = jhi-jlo+1;
    while(block < needed+2) block *= 2;
    if(block < 2*grid->packet_time_half_support)
        block = 2*grid->packet_time_half_support;
    if(block > grid->time_pixels) block = grid->time_pixels;
    center = (jlo+jhi+1)/2;
    if(center & 1) center--;
    if(center < block/2) center = block/2;
    if(center+block/2 > grid->time_pixels)
        center = grid->time_pixels-block/2;
    if(center & 1) center--;
    while(center-block/2 > jlo) center -= 2;
    while(center+block/2-1 < jhi) center += 2;
    if(center < block/2) center = block/2;
    if(center+block/2 > grid->time_pixels)
        center = grid->time_pixels-block/2;
    if(center & 1) center--;
    *nmid = center;
    *nsize = block;
}

static int tphm_build_compact_wdm_plan(
    const THMWDMGridInfo *grid,
    int ncarriers,
    int ntrack,
    const double *time,
    const double *fcenter,
    const double *flow,
    const double *fhigh,
    double time_start,
    double time_stop,
    double endpoint_start,
    double endpoint_rise,
    PhenomTPHMPartitionPlan *plan)
{
    int *jmin_layer = NULL, *jmax_layer = NULL;
    int i, k, m, active_layers = 0, volume = 0;
    double early_stop = fmin(time_stop, endpoint_start+endpoint_rise);
    double endpoint_center_low = HUGE_VAL, endpoint_high = -HUGE_VAL;
    double endpoint_margin, endpoint_frequency_start, endpoint_frequency_stop;

    if(grid == NULL || ncarriers < 1 || ntrack < 2 || time == NULL ||
       fcenter == NULL || flow == NULL || fhigh == NULL || plan == NULL ||
       !(time_stop > time_start) || endpoint_rise <= 0.0)
        return 1;

    plan->wdm_nmid = calloc((size_t)grid->frequency_layers,
                            sizeof(*plan->wdm_nmid));
    plan->wdm_nsize = calloc((size_t)grid->frequency_layers,
                             sizeof(*plan->wdm_nsize));
    plan->endpoint_wdm_nmid = calloc(
        (size_t)grid->frequency_layers, sizeof(*plan->endpoint_wdm_nmid));
    plan->endpoint_wdm_nsize = calloc(
        (size_t)grid->frequency_layers, sizeof(*plan->endpoint_wdm_nsize));
    jmin_layer = calloc((size_t)grid->frequency_layers,
                        sizeof(*jmin_layer));
    jmax_layer = calloc((size_t)grid->frequency_layers,
                        sizeof(*jmax_layer));
    if(plan->wdm_nmid == NULL || plan->wdm_nsize == NULL ||
       plan->endpoint_wdm_nmid == NULL ||
       plan->endpoint_wdm_nsize == NULL ||
       jmin_layer == NULL || jmax_layer == NULL)
    {
        free(jmax_layer);
        free(jmin_layer);
        return 2;
    }
    for(m=0; m<grid->frequency_layers; m++)
    {
        plan->wdm_nmid[m] = -1;
        plan->wdm_nsize[m] = 0;
        plan->endpoint_wdm_nmid[m] = -1;
        plan->endpoint_wdm_nsize[m] = 0;
        jmin_layer[m] = grid->time_pixels+1;
        jmax_layer[m] = -1;
    }

    for(k=0; k<ncarriers; k++)
    {
        const double *carrier_low = flow+(size_t)k*(size_t)ntrack;
        const double *carrier_high = fhigh+(size_t)k*(size_t)ntrack;
        for(i=0; i<ntrack-1; i++)
        {
            double segment_start = time[i];
            double segment_stop = time[i+1];
            double flo = fmin(carrier_low[i], carrier_low[i+1]);
            double fhi = fmax(carrier_high[i], carrier_high[i+1]);
            double tlo, thi;
            int jlo, jhi, mlo, mhi;

            if(segment_stop >= endpoint_start &&
               segment_start <= time_stop && isfinite(fhi) && fhi > 0.0)
            {
                double center_lo = fmin(
                    fcenter[(size_t)k*(size_t)ntrack+(size_t)i],
                    fcenter[(size_t)k*(size_t)ntrack+(size_t)(i+1)]);
                if(isfinite(center_lo) && center_lo > 0.0 &&
                   center_lo < endpoint_center_low)
                    endpoint_center_low = center_lo;
                if(fhi > endpoint_high) endpoint_high = fhi;
            }

            if(segment_stop < time_start || segment_start > early_stop)
                continue;
            tlo = fmax(segment_start, time_start);
            thi = fmin(segment_stop, early_stop);
            if(!(thi >= tlo) || !isfinite(flo) ||
               !isfinite(fhi) || fhi <= 0.0)
                continue;
            mlo = (int)ceil((flo-grid->meyer_half_bandwidth)/
                            grid->frequency_pixel_df);
            mhi = (int)floor((fhi+grid->meyer_half_bandwidth)/
                             grid->frequency_pixel_df);
            if(mlo < 1) mlo = 1;
            if(mhi > grid->frequency_layers-1)
                mhi = grid->frequency_layers-1;
            if(mhi < mlo) continue;
            jlo = (int)floor(tlo/grid->time_pixel_dt);
            jhi = (int)ceil(thi/grid->time_pixel_dt);
            if(jlo < 0) jlo = 0;
            if(jhi > grid->time_pixels-1) jhi = grid->time_pixels-1;
            for(m=mlo; m<=mhi; m++)
            {
                if(jlo < jmin_layer[m]) jmin_layer[m] = jlo;
                if(jhi > jmax_layer[m]) jmax_layer[m] = jhi;
            }
        }
    }

    if(!isfinite(endpoint_center_low) || endpoint_center_low == HUGE_VAL ||
       !isfinite(endpoint_high) || endpoint_high <= 0.0)
    {
        free(jmax_layer);
        free(jmin_layer);
        return 3;
    }
    endpoint_margin = 3.0/endpoint_rise+grid->meyer_half_bandwidth;
    if(endpoint_margin > 0.5*endpoint_center_low)
        endpoint_margin = 0.5*endpoint_center_low;
    endpoint_frequency_start =
        fmax(0.0, endpoint_center_low-endpoint_margin);
    endpoint_frequency_stop = endpoint_high+
        3.0/endpoint_rise+grid->meyer_half_bandwidth;
    if(endpoint_frequency_stop >
       (double)(grid->frequency_layers-1)*grid->frequency_pixel_df+
       grid->meyer_half_bandwidth)
        endpoint_frequency_stop =
            (double)(grid->frequency_layers-1)*grid->frequency_pixel_df+
            grid->meyer_half_bandwidth;

    for(m=1; m<grid->frequency_layers; m++)
    {
        if(jmax_layer[m] >= jmin_layer[m])
            tphm_packet_from_bounds(grid, jmin_layer[m], jmax_layer[m],
                                    &plan->wdm_nmid[m],
                                    &plan->wdm_nsize[m]);
    }
    {
        int mlo = (int)ceil((endpoint_frequency_start-
                             grid->meyer_half_bandwidth)/
                            grid->frequency_pixel_df);
        int mhi = (int)floor((endpoint_frequency_stop+
                              grid->meyer_half_bandwidth)/
                             grid->frequency_pixel_df);
        int jlo = (int)floor(endpoint_start/grid->time_pixel_dt);
        int jhi = (int)ceil(time_stop/grid->time_pixel_dt);
        int endpoint_mid, endpoint_size;
        if(mlo < 1) mlo = 1;
        if(mhi > grid->frequency_layers-1)
            mhi = grid->frequency_layers-1;
        if(jlo < 0) jlo = 0;
        if(jhi > grid->time_pixels-1) jhi = grid->time_pixels-1;
        tphm_packet_from_bounds(grid, jlo, jhi,
                                &endpoint_mid, &endpoint_size);
        for(m=mlo; m<=mhi; m++)
        {
            plan->endpoint_wdm_nmid[m] = endpoint_mid;
            plan->endpoint_wdm_nsize[m] = endpoint_size;
        }
    }

    for(m=1; m<grid->frequency_layers; m++)
    {
        int early = plan->wdm_nsize[m];
        int late = plan->endpoint_wdm_nsize[m];
        if(early <= 0 && late <= 0) continue;
        active_layers++;
        if(early > 0 && late > 0)
        {
            int elo = plan->wdm_nmid[m]-early/2;
            int ehi = elo+early;
            int llo = plan->endpoint_wdm_nmid[m]-late/2;
            int lhi = llo+late;
            int overlap = (ehi < lhi ? ehi : lhi)-
                          (elo > llo ? elo : llo);
            if(overlap < 0) overlap = 0;
            volume += early+late-overlap;
        }
        else
        {
            volume += early > 0 ? early : late;
        }
    }
    plan->wdm_active_layers = active_layers;
    plan->wdm_pixels_per_channel = volume;
    plan->endpoint_frequency_start = endpoint_frequency_start;
    plan->endpoint_frequency_stop = endpoint_frequency_stop;
    free(jmax_layer);
    free(jmin_layer);
    return 0;
}

static int tphm_block_local_bins(const THMWDMGridInfo *grid,
                                 double shift, double fstart, double fstop)
{
    double base = fmax(fabs(fstart-shift), fabs(fstop-shift));
    int bins = 1;
    while(0.5*(double)bins/grid->time_pixel_dt <= 1.05*base &&
          bins < grid->frequency_layers)
        bins *= 2;
    if(bins < 2) bins = 2;
    if(bins > grid->frequency_layers) bins = grid->frequency_layers;
    return bins;
}

static double tphm_smooth_step(double t, double t1, double t2)
{
    double x;
    if(t <= t1) return 0.0;
    if(t >= t2) return 1.0;
    x = (t-t1)/(t2-t1);
    return 0.5*(1.0-cos(M_PI*x));
}

typedef struct
{
    int n;
    int ncarriers;
    double *time;
    gsl_spline **real_spline;
    gsl_spline **imag_spline;
    gsl_interp_accel **real_acc;
    gsl_interp_accel **imag_acc;
} TPHMEarlyResponseCache;

typedef struct
{
    int carrier_index;
    int nfft;
    double sample_start;
    double sample_dt;
    double source_sample_dt;
    double heterodyne_frequency;
    double rise_start;
    double rise_end;
    double fall_start;
    double fall_end;
    double frequency_start;
    double frequency_stop;
} TPHMNarrowBlockSpec;

typedef struct
{
    int n;
    double time_start;
    double time_stop;
    gsl_spline *alpha_spline;
    gsl_spline *beta_spline;
    gsl_spline *gamma_spline;
    gsl_interp_accel *alpha_acc;
    gsl_interp_accel *beta_acc;
    gsl_interp_accel *gamma_acc;
} TPHMSharedEulerCache;

/*
 * One complex envelope for each twisted-up parent carrier.  The rapidly
 * varying aligned-spin carrier phase is removed before the four Cartesian
 * envelope components are splined.  Unlike the older response-ratio
 * experiment below, this cache lives upstream of TDI: delayed source times,
 * unequal arms, and the complete TDI sum are still evaluated explicitly.
 */
typedef struct
{
    int n;
    int ncarriers;
    double *time;
    gsl_spline **plus_real_spline;
    gsl_spline **plus_imag_spline;
    gsl_spline **cross_real_spline;
    gsl_spline **cross_imag_spline;
    gsl_interp_accel **plus_real_acc;
    gsl_interp_accel **plus_imag_acc;
    gsl_interp_accel **cross_real_acc;
    gsl_interp_accel **cross_imag_acc;
} TPHMComplexEnvelopeCache;

typedef struct
{
    double total_seconds;
    double phase_anchor_seconds;
    double carrier_seconds;
    long long samples;
    int calls;
} TPHMCachedEvaluationTiming;

typedef struct
{
    PhenomTPHMCarrierEvaluator selection;
    const TPHMSharedEulerCache *euler;
    TPHMCachedEvaluationTiming *timing;
} TPHMCachedCarrierEvaluator;

typedef struct
{
    PhenomTPHMCarrierEvaluator selection;
    const TPHMComplexEnvelopeCache *envelope;
    TPHMCachedEvaluationTiming *timing;
} TPHMEnvelopeCarrierEvaluator;

typedef struct
{
    PhenomTPHMTDISource *source;
    const TPHMComplexEnvelopeCache *envelope;
} TPHMEnvelopeFamilyEvaluator;

static void tphm_shared_euler_cache_free(TPHMSharedEulerCache *cache)
{
    if(cache == NULL) return;
    if(cache->gamma_acc != NULL) gsl_interp_accel_free(cache->gamma_acc);
    if(cache->beta_acc != NULL) gsl_interp_accel_free(cache->beta_acc);
    if(cache->alpha_acc != NULL) gsl_interp_accel_free(cache->alpha_acc);
    if(cache->gamma_spline != NULL) gsl_spline_free(cache->gamma_spline);
    if(cache->beta_spline != NULL) gsl_spline_free(cache->beta_spline);
    if(cache->alpha_spline != NULL) gsl_spline_free(cache->alpha_spline);
    memset(cache, 0, sizeof(*cache));
}

static void tphm_complex_envelope_cache_free(
    TPHMComplexEnvelopeCache *cache)
{
    int k;
    if(cache == NULL) return;
    for(k=0; k<cache->ncarriers; k++)
    {
        if(cache->plus_real_spline != NULL &&
           cache->plus_real_spline[k] != NULL)
            gsl_spline_free(cache->plus_real_spline[k]);
        if(cache->plus_imag_spline != NULL &&
           cache->plus_imag_spline[k] != NULL)
            gsl_spline_free(cache->plus_imag_spline[k]);
        if(cache->cross_real_spline != NULL &&
           cache->cross_real_spline[k] != NULL)
            gsl_spline_free(cache->cross_real_spline[k]);
        if(cache->cross_imag_spline != NULL &&
           cache->cross_imag_spline[k] != NULL)
            gsl_spline_free(cache->cross_imag_spline[k]);
        if(cache->plus_real_acc != NULL && cache->plus_real_acc[k] != NULL)
            gsl_interp_accel_free(cache->plus_real_acc[k]);
        if(cache->plus_imag_acc != NULL && cache->plus_imag_acc[k] != NULL)
            gsl_interp_accel_free(cache->plus_imag_acc[k]);
        if(cache->cross_real_acc != NULL &&
           cache->cross_real_acc[k] != NULL)
            gsl_interp_accel_free(cache->cross_real_acc[k]);
        if(cache->cross_imag_acc != NULL &&
           cache->cross_imag_acc[k] != NULL)
            gsl_interp_accel_free(cache->cross_imag_acc[k]);
    }
    free(cache->cross_imag_acc);
    free(cache->cross_real_acc);
    free(cache->plus_imag_acc);
    free(cache->plus_real_acc);
    free(cache->cross_imag_spline);
    free(cache->cross_real_spline);
    free(cache->plus_imag_spline);
    free(cache->plus_real_spline);
    free(cache->time);
    memset(cache, 0, sizeof(*cache));
}

static int tphm_double_compare(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static int tphm_build_shared_euler_cache(
    const THMObservationContext *context,
    PhenomTPHMTDISource *source,
    int nspec,
    const TPHMNarrowBlockSpec *spec,
    TPHMSharedEulerCache *cache)
{
    const double delay_margin = 500.0+
        (thm_observation_tdi_generation(context) == 2 ? 8.0 : 4.0)*
        THM_TDI_LARM_NOMINAL_SECONDS;
    IMRPhenomTPHMPrecessionSummary summary;
    double *time = NULL, *tau = NULL;
    double *alpha = NULL, *beta = NULL, *gamma = NULL;
    long long capacity64 = 0;
    int capacity, count = 0, unique = 0;
    int b, i, status = 0;

    if(source == NULL || nspec < 1 || spec == NULL || cache == NULL)
        return 1;
    tphm_shared_euler_cache_free(cache);
    for(b=0; b<nspec; b++)
    {
        double dt = spec[b].source_sample_dt;
        double lo = spec[b].sample_start-delay_margin-4.0*dt;
        double hi = spec[b].fall_end+delay_margin+4.0*dt;
        long long n;
        if(!(dt > 0.0) || !(hi > lo)) return 2;
        n = (long long)ceil((hi-lo)/dt)+1;
        capacity64 += n;
        if(capacity64 > 1073741824LL) return 3;
    }
    capacity = (int)capacity64;
    time = calloc((size_t)capacity, sizeof(*time));
    if(time == NULL) return 4;
    for(b=0; b<nspec; b++)
    {
        double dt = spec[b].source_sample_dt;
        double lo = spec[b].sample_start-delay_margin-4.0*dt;
        double hi = spec[b].fall_end+delay_margin+4.0*dt;
        double start = floor(lo/dt)*dt;
        int n = (int)ceil((hi-start)/dt)+1;
        for(i=0; i<n; i++) time[count++] = start+dt*(double)i;
    }
    qsort(time, (size_t)count, sizeof(*time), tphm_double_compare);
    for(i=0; i<count; i++)
    {
        double tolerance = 32.0*DBL_EPSILON*fmax(1.0, fabs(time[i]));
        if(unique == 0 || time[i]-time[unique-1] > tolerance)
            time[unique++] = time[i];
    }
    if(unique < 5)
    {
        status = 5;
        goto cleanup;
    }
    tau = calloc((size_t)unique, sizeof(*tau));
    alpha = calloc((size_t)unique, sizeof(*alpha));
    beta = calloc((size_t)unique, sizeof(*beta));
    gamma = calloc((size_t)unique, sizeof(*gamma));
    if(tau == NULL || alpha == NULL || beta == NULL || gamma == NULL)
    {
        status = 6;
        goto cleanup;
    }
    for(i=0; i<unique; i++)
        tau[i] = (time[i]-source->coalescence_time)/source->total_mass;
    if(source->waveform.merger_reconstruction ==
       IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
        status = IMRPhenomTPHMEvolveEulerAnglesLALCompatible(
            &source->waveform.precession_driver,
            &source->waveform.precession, unique, tau,
            alpha, beta, gamma, &summary);
    else
        status = IMRPhenomTPHMEvolveEulerAngles(
            &source->waveform.precession_driver,
            &source->waveform.precession, unique, tau,
            alpha, beta, gamma, &summary);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    cache->alpha_spline = gsl_spline_alloc(gsl_interp_akima,
                                            (size_t)unique);
    cache->beta_spline = gsl_spline_alloc(gsl_interp_akima,
                                           (size_t)unique);
    cache->gamma_spline = gsl_spline_alloc(gsl_interp_akima,
                                            (size_t)unique);
    cache->alpha_acc = gsl_interp_accel_alloc();
    cache->beta_acc = gsl_interp_accel_alloc();
    cache->gamma_acc = gsl_interp_accel_alloc();
    if(cache->alpha_spline == NULL || cache->beta_spline == NULL ||
       cache->gamma_spline == NULL || cache->alpha_acc == NULL ||
       cache->beta_acc == NULL || cache->gamma_acc == NULL)
    {
        status = 7;
        goto cleanup;
    }
    if(gsl_spline_init(cache->alpha_spline, time, alpha,
                       (size_t)unique) != GSL_SUCCESS ||
       gsl_spline_init(cache->beta_spline, time, beta,
                       (size_t)unique) != GSL_SUCCESS ||
       gsl_spline_init(cache->gamma_spline, time, gamma,
                       (size_t)unique) != GSL_SUCCESS)
    {
        status = 8;
        goto cleanup;
    }
    cache->n = unique;
    cache->time_start = time[0];
    cache->time_stop = time[unique-1];
    status = 0;

cleanup:
    free(gamma);
    free(beta);
    free(alpha);
    free(tau);
    free(time);
    if(status != 0) tphm_shared_euler_cache_free(cache);
    return status;
}

static int tphm_cached_carrier_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    TPHMCachedCarrierEvaluator *cached =
        (TPHMCachedCarrierEvaluator *)userdata;
    PhenomTPHMTDISource *source;
    double *tau = NULL, *alpha = NULL, *beta = NULL, *gamma = NULL;
    double complex *strain = NULL, *quadrature = NULL;
    double phi_at_start;
    clock_t total_start, stage_start;
    int i, status;

    total_start = clock();

    if(cached == NULL || cached->selection.source == NULL ||
       cached->euler == NULL || n < 1 || source_time == NULL ||
       hplus_analytic == NULL || hcross_analytic == NULL)
        return 1;
    source = cached->selection.source;
    if(source_time[0] < cached->euler->time_start ||
       source_time[n-1] > cached->euler->time_stop)
        return 2;
    tau = calloc((size_t)n, sizeof(*tau));
    alpha = calloc((size_t)n, sizeof(*alpha));
    beta = calloc((size_t)n, sizeof(*beta));
    gamma = calloc((size_t)n, sizeof(*gamma));
    strain = calloc((size_t)n, sizeof(*strain));
    quadrature = calloc((size_t)n, sizeof(*quadrature));
    if(tau == NULL || alpha == NULL || beta == NULL || gamma == NULL ||
       strain == NULL || quadrature == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(i=0; i<n; i++)
    {
        tau[i] = (source_time[i]-source->coalescence_time)/
                 source->total_mass;
        alpha[i] = gsl_spline_eval(cached->euler->alpha_spline,
                                   source_time[i],
                                   cached->euler->alpha_acc);
        beta[i] = gsl_spline_eval(cached->euler->beta_spline,
                                  source_time[i],
                                  cached->euler->beta_acc);
        gamma[i] = gsl_spline_eval(cached->euler->gamma_spline,
                                   source_time[i],
                                   cached->euler->gamma_acc);
    }
    stage_start = clock();
    status = tphm_phase_from_anchor(source, tau[0], &phi_at_start);
    if(cached->timing != NULL)
        cached->timing->phase_anchor_seconds +=
            (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    stage_start = clock();
    status = IMRPhenomTPHMEvaluateSelectedCarrierGridWithQuadratureAndEuler(
        &source->waveform, cached->selection.carrier_index,
        n, tau, phi_at_start, source->observer_theta,
        source->observer_phi, source->polarization,
        alpha, beta, gamma, strain, quadrature, NULL);
    if(cached->timing != NULL)
        cached->timing->carrier_seconds +=
            (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }
    for(i=0; i<n; i++)
    {
        hplus_analytic[i] = source->strain_scale*
            (creal(strain[i])-I*creal(quadrature[i]));
        hcross_analytic[i] = source->strain_scale*
            (-cimag(strain[i])+I*cimag(quadrature[i]));
    }
    status = 0;

cleanup:
    free(quadrature);
    free(strain);
    free(gamma);
    free(beta);
    free(alpha);
    free(tau);
    if(cached->timing != NULL)
    {
        cached->timing->total_seconds +=
            (double)(clock()-total_start)/(double)CLOCKS_PER_SEC;
        cached->timing->samples += n;
        cached->timing->calls++;
    }
    return status;
}

static int tphm_cached_all_carrier_polarizations(
    PhenomTPHMTDISource *source,
    const TPHMSharedEulerCache *euler,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    double *tau = NULL, *alpha = NULL, *beta = NULL, *gamma = NULL;
    double complex *strain = NULL, *quadrature = NULL;
    double phi_at_start;
    int nc, i, k, status;

    if(source == NULL || euler == NULL || n < 1 || source_time == NULL ||
       hplus_analytic == NULL || hcross_analytic == NULL)
        return 1;
    if(source_time[0] < euler->time_start ||
       source_time[n-1] > euler->time_stop)
        return 2;
    nc = source->waveform.ncarriers;
    tau = calloc((size_t)n, sizeof(*tau));
    alpha = calloc((size_t)n, sizeof(*alpha));
    beta = calloc((size_t)n, sizeof(*beta));
    gamma = calloc((size_t)n, sizeof(*gamma));
    strain = calloc((size_t)nc*(size_t)n, sizeof(*strain));
    quadrature = calloc((size_t)nc*(size_t)n, sizeof(*quadrature));
    if(tau == NULL || alpha == NULL || beta == NULL || gamma == NULL ||
       strain == NULL || quadrature == NULL)
    {
        status = 3;
        goto cleanup;
    }
    gsl_interp_accel_reset(euler->alpha_acc);
    gsl_interp_accel_reset(euler->beta_acc);
    gsl_interp_accel_reset(euler->gamma_acc);
    for(i=0; i<n; i++)
    {
        tau[i] = (source_time[i]-source->coalescence_time)/
                 source->total_mass;
        alpha[i] = gsl_spline_eval(euler->alpha_spline, source_time[i],
                                   euler->alpha_acc);
        beta[i] = gsl_spline_eval(euler->beta_spline, source_time[i],
                                  euler->beta_acc);
        gamma[i] = gsl_spline_eval(euler->gamma_spline, source_time[i],
                                   euler->gamma_acc);
    }
    status = tphm_phase_from_anchor(source, tau[0], &phi_at_start);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    status = IMRPhenomTPHMEvaluateCarrierGridWithQuadratureAndEuler(
        &source->waveform, n, tau, phi_at_start,
        source->observer_theta, source->observer_phi, source->polarization,
        alpha, beta, gamma, strain, quadrature, NULL);
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }
    for(k=0; k<nc; k++)
    {
        for(i=0; i<n; i++)
        {
            size_t q = (size_t)k*(size_t)n+(size_t)i;
            hplus_analytic[q] = source->strain_scale*
                (creal(strain[q])-I*creal(quadrature[q]));
            hcross_analytic[q] = source->strain_scale*
                (-cimag(strain[q])+I*cimag(quadrature[q]));
        }
    }
    status = 0;

cleanup:
    free(quadrature);
    free(strain);
    free(gamma);
    free(beta);
    free(alpha);
    free(tau);
    return status;
}

static void tphm_early_response_cache_free(TPHMEarlyResponseCache *cache)
{
    int i, count;
    if(cache == NULL) return;
    count = 3*cache->ncarriers;
    for(i=0; i<count; i++)
    {
        if(cache->real_spline != NULL && cache->real_spline[i] != NULL)
            gsl_spline_free(cache->real_spline[i]);
        if(cache->imag_spline != NULL && cache->imag_spline[i] != NULL)
            gsl_spline_free(cache->imag_spline[i]);
        if(cache->real_acc != NULL && cache->real_acc[i] != NULL)
            gsl_interp_accel_free(cache->real_acc[i]);
        if(cache->imag_acc != NULL && cache->imag_acc[i] != NULL)
            gsl_interp_accel_free(cache->imag_acc[i]);
    }
    free(cache->imag_acc);
    free(cache->real_acc);
    free(cache->imag_spline);
    free(cache->real_spline);
    free(cache->time);
    memset(cache, 0, sizeof(*cache));
}

static int tphm_evaluate_carrier_references(
    PhenomTPHMTDISource *source,
    int selected_carrier,
    int n,
    const double *time,
    double complex *reference)
{
    double *tau = NULL, *phi22 = NULL;
    IMRPhenomTHMModeSample *samples = NULL;
    double phi_at_start;
    int nc, i, k, status;

    if(source == NULL || n < 1 || time == NULL || reference == NULL)
        return 1;
    nc = source->waveform.ncarriers;
    if(selected_carrier >= nc) return 2;
    tau = calloc((size_t)n, sizeof(*tau));
    phi22 = calloc((size_t)n, sizeof(*phi22));
    samples = calloc((size_t)(selected_carrier < 0 ? nc : 1)*(size_t)n,
                     sizeof(*samples));
    if(tau == NULL || phi22 == NULL || samples == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(i=0; i<n; i++)
        tau[i] = (time[i]-source->coalescence_time)/source->total_mass;
    status = tphm_phase_from_anchor(source, tau[0], &phi_at_start);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    if(source->waveform.merger_reconstruction ==
       IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
        status = IMRPhenomTHMBuildAnalyticPhi22Grid(
            &source->waveform.carrier, n, tau, phi_at_start, phi22);
    else
        status = IMRPhenomTHMBuildPhi22Grid(
            &source->waveform.carrier, n, tau, phi_at_start, phi22);
    if(status != 0)
    {
        if(getenv("TPHM_TDI_DEBUG") != NULL)
            for(i=1; i<n; i++)
                if(tau[i] <= tau[i-1])
                {
                    fprintf(stderr,
                            "TPHM_TDI_DEBUG carrier phase grid inversion %d/%d t %.17e %.17e tau %.17e %.17e\n",
                            i, n, time[i-1], time[i], tau[i-1], tau[i]);
                    break;
                }
        status = 20+status;
        goto cleanup;
    }
    if(selected_carrier < 0)
    {
        status = IMRPhenomTHMEvaluateGrid(&source->waveform.carrier,
                                          n, tau, phi22, samples);
        if(status != 0)
        {
            status = 30+status;
            goto cleanup;
        }
        for(k=0; k<nc; k++)
        {
            for(i=0; i<n; i++)
            {
                size_t q = (size_t)k*(size_t)n+(size_t)i;
                reference[q] = source->strain_scale*conj(samples[q].hlm);
            }
        }
    }
    else
    {
        for(i=0; i<n; i++)
        {
            status = IMRPhenomTHMEvaluateMode(
                &source->waveform.carrier, selected_carrier,
                tau[i], phi22[i], &samples[i]);
            if(status != 0)
            {
                status = 40+status;
                goto cleanup;
            }
            reference[i] = source->strain_scale*conj(samples[i].hlm);
        }
    }
    status = 0;

cleanup:
    free(samples);
    free(phi22);
    free(tau);
    return status;
}

static int tphm_build_complex_envelope_cache(
    const THMObservationContext *context,
    PhenomTPHMTDISource *source,
    const TPHMSharedEulerCache *euler,
    int nspec,
    const TPHMNarrowBlockSpec *spec,
    TPHMComplexEnvelopeCache *cache)
{
    const double delay_margin = 500.0+
        (thm_observation_tdi_generation(context) == 2 ? 8.0 : 4.0)*
        THM_TDI_LARM_NOMINAL_SECONDS;
    const char *far_dt_text = getenv("TPHM_COMPLEX_ENVELOPE_DT");
    /* The twist-up envelope is much slower than the parent carrier.  This
     * conservative far-inspiral default sits on the accuracy plateau for the
     * one-year reference source; the near-merger cadence remains mass-scaled. */
    double far_dt = 20000.0;
    double near_dt, plunge_dt, zone_near, zone_plunge;
    double lo = HUGE_VAL, hi = -HUGE_VAL, t;
    double complex *plus = NULL, *cross = NULL, *reference = NULL;
    double *value[4] = {NULL, NULL, NULL, NULL};
    int capacity, n = 0, nc, b, i, k, status = 0;

    if(source == NULL || euler == NULL || nspec < 1 || spec == NULL ||
       cache == NULL)
        return 1;
    tphm_complex_envelope_cache_free(cache);
    if(far_dt_text != NULL)
    {
        double requested = strtod(far_dt_text, NULL);
        if(isfinite(requested) && requested > 0.0) far_dt = requested;
    }
    near_dt = fmin(far_dt,
                   fmin(1000.0, fmax(50.0, 32.0*source->total_mass)));
    plunge_dt = fmin(near_dt,
                     fmin(50.0, fmax(0.5, 2.0*source->total_mass)));
    zone_near = source->coalescence_time-1.0e5;
    zone_plunge = source->coalescence_time-2000.0*source->total_mass;
    if(zone_plunge < zone_near) zone_plunge = zone_near;
    for(b=0; b<nspec; b++)
    {
        double block_lo = spec[b].sample_start-delay_margin-
                          4.0*spec[b].source_sample_dt;
        double block_hi = spec[b].fall_end+delay_margin+
                          4.0*spec[b].source_sample_dt;
        if(block_lo < lo) lo = block_lo;
        if(block_hi > hi) hi = block_hi;
    }
    if(lo < euler->time_start) lo = euler->time_start;
    if(hi > euler->time_stop) hi = euler->time_stop;
    if(!(hi > lo)) return 2;
    capacity =
        (int)ceil(fmax(0.0, fmin(hi, zone_near)-lo)/far_dt)+
        (int)ceil(fmax(0.0, fmin(hi, zone_plunge)-
                       fmax(lo, zone_near))/near_dt)+
        (int)ceil(fmax(0.0, hi-fmax(lo, zone_plunge))/plunge_dt)+16;
    if(capacity < 16) capacity = 16;
    cache->time = calloc((size_t)capacity, sizeof(*cache->time));
    if(cache->time == NULL) return 3;
    t = lo;
    while(t < hi && n < capacity-1)
    {
        double step;
        cache->time[n++] = t;
        if(t < zone_near) step = far_dt;
        else if(t < zone_plunge) step = near_dt;
        else step = plunge_dt;
        t += step;
        if(cache->time[n-1] < zone_near && t > zone_near) t = zone_near;
        if(cache->time[n-1] < zone_plunge && t > zone_plunge)
            t = zone_plunge;
    }
    cache->time[n++] = hi;
    if(n < 5)
    {
        status = 4;
        goto cleanup;
    }
    nc = source->waveform.ncarriers;
    cache->n = n;
    cache->ncarriers = nc;
    cache->plus_real_spline = calloc((size_t)nc,
                                      sizeof(*cache->plus_real_spline));
    cache->plus_imag_spline = calloc((size_t)nc,
                                      sizeof(*cache->plus_imag_spline));
    cache->cross_real_spline = calloc((size_t)nc,
                                       sizeof(*cache->cross_real_spline));
    cache->cross_imag_spline = calloc((size_t)nc,
                                       sizeof(*cache->cross_imag_spline));
    cache->plus_real_acc = calloc((size_t)nc,
                                  sizeof(*cache->plus_real_acc));
    cache->plus_imag_acc = calloc((size_t)nc,
                                  sizeof(*cache->plus_imag_acc));
    cache->cross_real_acc = calloc((size_t)nc,
                                   sizeof(*cache->cross_real_acc));
    cache->cross_imag_acc = calloc((size_t)nc,
                                   sizeof(*cache->cross_imag_acc));
    plus = calloc((size_t)nc*(size_t)n, sizeof(*plus));
    cross = calloc((size_t)nc*(size_t)n, sizeof(*cross));
    reference = calloc((size_t)nc*(size_t)n, sizeof(*reference));
    for(i=0; i<4; i++) value[i] = calloc((size_t)n, sizeof(*value[i]));
    if(cache->plus_real_spline == NULL ||
       cache->plus_imag_spline == NULL ||
       cache->cross_real_spline == NULL ||
       cache->cross_imag_spline == NULL ||
       cache->plus_real_acc == NULL || cache->plus_imag_acc == NULL ||
       cache->cross_real_acc == NULL || cache->cross_imag_acc == NULL ||
       plus == NULL || cross == NULL || reference == NULL ||
       value[0] == NULL || value[1] == NULL ||
       value[2] == NULL || value[3] == NULL)
    {
        status = 5;
        goto cleanup;
    }

    status = tphm_cached_all_carrier_polarizations(
        source, euler, n, cache->time, plus, cross);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    status = tphm_evaluate_carrier_references(
        source, -1, n, cache->time, reference);
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }

    for(k=0; k<nc; k++)
    {
        for(i=0; i<n; i++)
        {
            size_t q = (size_t)k*(size_t)n+(size_t)i;
            double magnitude = cabs(reference[q]);
            double complex phasor = magnitude > 1.0e-300 ?
                                    reference[q]/magnitude : 1.0;
            double complex plus_envelope = plus[q]*conj(phasor);
            double complex cross_envelope = cross[q]*conj(phasor);
            value[0][i] = creal(plus_envelope);
            value[1][i] = cimag(plus_envelope);
            value[2][i] = creal(cross_envelope);
            value[3][i] = cimag(cross_envelope);
        }
        cache->plus_real_spline[k] = gsl_spline_alloc(
            gsl_interp_akima, (size_t)n);
        cache->plus_imag_spline[k] = gsl_spline_alloc(
            gsl_interp_akima, (size_t)n);
        cache->cross_real_spline[k] = gsl_spline_alloc(
            gsl_interp_akima, (size_t)n);
        cache->cross_imag_spline[k] = gsl_spline_alloc(
            gsl_interp_akima, (size_t)n);
        cache->plus_real_acc[k] = gsl_interp_accel_alloc();
        cache->plus_imag_acc[k] = gsl_interp_accel_alloc();
        cache->cross_real_acc[k] = gsl_interp_accel_alloc();
        cache->cross_imag_acc[k] = gsl_interp_accel_alloc();
        if(cache->plus_real_spline[k] == NULL ||
           cache->plus_imag_spline[k] == NULL ||
           cache->cross_real_spline[k] == NULL ||
           cache->cross_imag_spline[k] == NULL ||
           cache->plus_real_acc[k] == NULL ||
           cache->plus_imag_acc[k] == NULL ||
           cache->cross_real_acc[k] == NULL ||
           cache->cross_imag_acc[k] == NULL ||
           gsl_spline_init(cache->plus_real_spline[k], cache->time,
                           value[0], (size_t)n) != GSL_SUCCESS ||
           gsl_spline_init(cache->plus_imag_spline[k], cache->time,
                           value[1], (size_t)n) != GSL_SUCCESS ||
           gsl_spline_init(cache->cross_real_spline[k], cache->time,
                           value[2], (size_t)n) != GSL_SUCCESS ||
           gsl_spline_init(cache->cross_imag_spline[k], cache->time,
                           value[3], (size_t)n) != GSL_SUCCESS)
        {
            status = 6;
            goto cleanup;
        }
    }
    status = 0;

cleanup:
    for(i=0; i<4; i++) free(value[i]);
    free(reference);
    free(cross);
    free(plus);
    if(status != 0) tphm_complex_envelope_cache_free(cache);
    return status;
}

static int tphm_envelope_carrier_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    TPHMEnvelopeCarrierEvaluator *cached =
        (TPHMEnvelopeCarrierEvaluator *)userdata;
    const TPHMComplexEnvelopeCache *envelope;
    PhenomTPHMTDISource *source;
    double complex *reference = NULL;
    clock_t start;
    int i, k, status;

    start = clock();
    if(cached == NULL || cached->selection.source == NULL ||
       cached->envelope == NULL || n < 1 || source_time == NULL ||
       hplus_analytic == NULL || hcross_analytic == NULL)
        return 1;
    source = cached->selection.source;
    envelope = cached->envelope;
    k = cached->selection.carrier_index;
    if(k < 0 || k >= envelope->ncarriers ||
       source_time[0] < envelope->time[0] ||
       source_time[n-1] > envelope->time[envelope->n-1])
        return 2;
    reference = calloc((size_t)n, sizeof(*reference));
    if(reference == NULL) return 3;
    status = tphm_evaluate_carrier_references(
        source, k, n, source_time, reference);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    gsl_interp_accel_reset(envelope->plus_real_acc[k]);
    gsl_interp_accel_reset(envelope->plus_imag_acc[k]);
    gsl_interp_accel_reset(envelope->cross_real_acc[k]);
    gsl_interp_accel_reset(envelope->cross_imag_acc[k]);
    for(i=0; i<n; i++)
    {
        double magnitude = cabs(reference[i]);
        double complex phasor = magnitude > 1.0e-300 ?
                                reference[i]/magnitude : 1.0;
        double complex plus_envelope =
            gsl_spline_eval(envelope->plus_real_spline[k], source_time[i],
                            envelope->plus_real_acc[k])+
            I*gsl_spline_eval(envelope->plus_imag_spline[k], source_time[i],
                              envelope->plus_imag_acc[k]);
        double complex cross_envelope =
            gsl_spline_eval(envelope->cross_real_spline[k], source_time[i],
                            envelope->cross_real_acc[k])+
            I*gsl_spline_eval(envelope->cross_imag_spline[k], source_time[i],
                              envelope->cross_imag_acc[k]);
        hplus_analytic[i] = phasor*plus_envelope;
        hcross_analytic[i] = phasor*cross_envelope;
    }
    status = 0;

cleanup:
    free(reference);
    if(cached->timing != NULL)
    {
        double elapsed = (double)(clock()-start)/(double)CLOCKS_PER_SEC;
        cached->timing->total_seconds += elapsed;
        cached->timing->carrier_seconds += elapsed;
        cached->timing->samples += n;
        cached->timing->calls++;
    }
    return status;
}

static int tphm_all_envelope_carrier_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    int nfamilies,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    TPHMEnvelopeFamilyEvaluator *cached =
        (TPHMEnvelopeFamilyEvaluator *)userdata;
    const TPHMComplexEnvelopeCache *envelope;
    double complex *reference = NULL;
    int i, k, status;

    if(cached == NULL || cached->source == NULL ||
       cached->envelope == NULL || n < 1 || source_time == NULL ||
       hplus_analytic == NULL || hcross_analytic == NULL)
        return 1;
    envelope = cached->envelope;
    if(nfamilies != envelope->ncarriers ||
       source_time[0] < envelope->time[0] ||
       source_time[n-1] > envelope->time[envelope->n-1])
    {
        if(getenv("TPHM_TDI_DEBUG") != NULL)
            fprintf(stderr,
                    "TPHM_TDI_DEBUG envelope source [%.15e, %.15e] cache [%.15e, %.15e] families %d/%d\n",
                    source_time[0], source_time[n-1], envelope->time[0],
                    envelope->time[envelope->n-1], nfamilies,
                    envelope->ncarriers);
        return 2;
    }
    reference = calloc((size_t)nfamilies*(size_t)n, sizeof(*reference));
    if(reference == NULL) return 3;
    status = tphm_evaluate_carrier_references(
        cached->source, -1, n, source_time, reference);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    for(k=0; k<nfamilies; k++)
    {
        gsl_interp_accel_reset(envelope->plus_real_acc[k]);
        gsl_interp_accel_reset(envelope->plus_imag_acc[k]);
        gsl_interp_accel_reset(envelope->cross_real_acc[k]);
        gsl_interp_accel_reset(envelope->cross_imag_acc[k]);
        for(i=0; i<n; i++)
        {
            size_t q = (size_t)k*(size_t)n+(size_t)i;
            double magnitude = cabs(reference[q]);
            double complex phasor = magnitude > 1.0e-300 ?
                                    reference[q]/magnitude : 1.0;
            double complex plus_envelope =
                gsl_spline_eval(envelope->plus_real_spline[k],
                                source_time[i],
                                envelope->plus_real_acc[k])+
                I*gsl_spline_eval(envelope->plus_imag_spline[k],
                                  source_time[i],
                                  envelope->plus_imag_acc[k]);
            double complex cross_envelope =
                gsl_spline_eval(envelope->cross_real_spline[k],
                                source_time[i],
                                envelope->cross_real_acc[k])+
                I*gsl_spline_eval(envelope->cross_imag_spline[k],
                                  source_time[i],
                                  envelope->cross_imag_acc[k]);
            hplus_analytic[q] = phasor*plus_envelope;
            hcross_analytic[q] = phasor*cross_envelope;
        }
    }
    status = 0;

cleanup:
    free(reference);
    return status;
}

static int tphm_interpolate_carrier_references(
    PhenomTPHMTDISource *source,
    int ns,
    const double *sparse_time,
    int n,
    const double *time,
    double complex *sparse_reference,
    double complex *reference)
{
    double *tau = NULL, *phi22 = NULL, *logamp = NULL, *phase_residual = NULL;
    IMRPhenomTHMModeSample *samples = NULL;
    gsl_spline *amp_spline = NULL, *phase_spline = NULL;
    gsl_interp_accel *amp_acc = NULL, *phase_acc = NULL;
    double phi_at_start;
    int nc, i, k, status;

    if(source == NULL || ns < 5 || sparse_time == NULL || n < 1 ||
       time == NULL || sparse_reference == NULL || reference == NULL)
        return 1;
    nc = source->waveform.ncarriers;
    tau = calloc((size_t)ns, sizeof(*tau));
    phi22 = calloc((size_t)ns, sizeof(*phi22));
    logamp = calloc((size_t)ns, sizeof(*logamp));
    phase_residual = calloc((size_t)ns, sizeof(*phase_residual));
    samples = calloc((size_t)nc*(size_t)ns, sizeof(*samples));
    if(tau == NULL || phi22 == NULL || logamp == NULL ||
       phase_residual == NULL || samples == NULL)
    {
        status = 2;
        goto cleanup;
    }
    for(i=0; i<ns; i++)
        tau[i] = (sparse_time[i]-source->coalescence_time)/source->total_mass;
    status = tphm_phase_from_anchor(source, tau[0], &phi_at_start);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    if(source->waveform.merger_reconstruction ==
       IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
        status = IMRPhenomTHMBuildAnalyticPhi22Grid(
            &source->waveform.carrier, ns, tau, phi_at_start, phi22);
    else
        status = IMRPhenomTHMBuildPhi22Grid(
            &source->waveform.carrier, ns, tau, phi_at_start, phi22);
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }
    status = IMRPhenomTHMEvaluateGrid(&source->waveform.carrier,
                                      ns, tau, phi22, samples);
    if(status != 0)
    {
        status = 30+status;
        goto cleanup;
    }
    amp_spline = gsl_spline_alloc(gsl_interp_akima, (size_t)ns);
    phase_spline = gsl_spline_alloc(gsl_interp_akima, (size_t)ns);
    amp_acc = gsl_interp_accel_alloc();
    phase_acc = gsl_interp_accel_alloc();
    if(amp_spline == NULL || phase_spline == NULL ||
       amp_acc == NULL || phase_acc == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(k=0; k<nc; k++)
    {
        double phase0 = samples[(size_t)k*(size_t)ns].phase;
        double slope = (samples[(size_t)k*(size_t)ns+(size_t)(ns-1)].phase-
                        phase0)/(sparse_time[ns-1]-sparse_time[0]);
        for(i=0; i<ns; i++)
        {
            size_t q = (size_t)k*(size_t)ns+(size_t)i;
            double amp = fmax(samples[q].amplitude, 1.0e-300);
            double phase = samples[q].phase;
            logamp[i] = log(amp);
            phase_residual[i] = phase-phase0-
                                slope*(sparse_time[i]-sparse_time[0]);
            sparse_reference[q] = source->strain_scale*amp*
                                   (cos(phase)+I*sin(phase));
        }
        if(gsl_spline_init(amp_spline, sparse_time, logamp,
                           (size_t)ns) != GSL_SUCCESS ||
           gsl_spline_init(phase_spline, sparse_time, phase_residual,
                           (size_t)ns) != GSL_SUCCESS)
        {
            status = 4;
            goto cleanup;
        }
        gsl_interp_accel_reset(amp_acc);
        gsl_interp_accel_reset(phase_acc);
        for(i=0; i<n; i++)
        {
            size_t q = (size_t)k*(size_t)n+(size_t)i;
            double amp = source->strain_scale*exp(
                gsl_spline_eval(amp_spline, time[i], amp_acc));
            double phase = phase0+slope*(time[i]-sparse_time[0])+
                gsl_spline_eval(phase_spline, time[i], phase_acc);
            reference[q] = amp*(cos(phase)+I*sin(phase));
        }
    }
    status = 0;

cleanup:
    if(phase_acc != NULL) gsl_interp_accel_free(phase_acc);
    if(amp_acc != NULL) gsl_interp_accel_free(amp_acc);
    if(phase_spline != NULL) gsl_spline_free(phase_spline);
    if(amp_spline != NULL) gsl_spline_free(amp_spline);
    free(samples);
    free(phase_residual);
    free(logamp);
    free(phi22);
    free(tau);
    return status;
}

static int tphm_evaluate_factorized_carrier_polarizations(
    PhenomTPHMTDISource *source,
    int n,
    const double *source_time,
    double complex *hplus,
    double complex *hcross)
{
    double *sparse_time = NULL;
    double complex *sparse_plus = NULL, *sparse_cross = NULL;
    double complex *sparse_reference = NULL, *dense_reference = NULL;
    double *pr = NULL, *pi = NULL, *cr = NULL, *ci = NULL;
    gsl_spline *spr = NULL, *spi = NULL, *scr = NULL, *sci = NULL;
    gsl_interp_accel *apr = NULL, *api = NULL, *acr = NULL, *aci = NULL;
    double zone_near, zone_plunge, near_dt, plunge_dt;
    double t, stop;
    int capacity, ns, nc, i, k, status = 0;

    if(source == NULL || n < 1 || source_time == NULL ||
       hplus == NULL || hcross == NULL)
        return 1;
    for(i=1; i<n; i++)
    {
        if(source_time[i] <= source_time[i-1]) return 2;
    }
    /* Small grids are already efficient and remain the exact regression path. */
    if(n <= 4096)
        return tphm_evaluate_carrier_polarization_arrays(
            source, n, source_time, hplus, hcross,
            NULL, NULL, NULL, NULL);

    nc = source->waveform.ncarriers;
    zone_near = source->coalescence_time-1.0e5;
    zone_plunge = source->coalescence_time-2000.0*source->total_mass;
    if(zone_plunge < zone_near) zone_plunge = zone_near;
    near_dt = fmin(1000.0, fmax(50.0, 32.0*source->total_mass));
    plunge_dt = fmin(50.0, fmax(0.5, 2.0*source->total_mass));
    stop = source_time[n-1];
    capacity = (int)ceil(fmax(0.0, fmin(stop, zone_near)-source_time[0])/
                         1000.0)+
               (int)ceil(fmax(0.0, fmin(stop, zone_plunge)-
                              fmax(source_time[0], zone_near))/near_dt)+
               (int)ceil(fmax(0.0, stop-fmax(source_time[0], zone_plunge))/
                         plunge_dt)+16;
    if(capacity < 16) capacity = 16;
    sparse_time = calloc((size_t)capacity, sizeof(*sparse_time));
    if(sparse_time == NULL) return 3;
    ns = 0;
    t = source_time[0];
    while(t < stop && ns < capacity-1)
    {
        double step;
        sparse_time[ns++] = t;
        if(t < zone_near) step = 1000.0;
        else if(t < zone_plunge) step = near_dt;
        else step = plunge_dt;
        t += step;
        if(sparse_time[ns-1] < zone_near && t > zone_near) t = zone_near;
        if(sparse_time[ns-1] < zone_plunge && t > zone_plunge)
            t = zone_plunge;
    }
    sparse_time[ns++] = stop;
    if(ns < 5)
    {
        status = tphm_evaluate_carrier_polarization_arrays(
            source, n, source_time, hplus, hcross,
            NULL, NULL, NULL, NULL);
        goto cleanup;
    }

    sparse_plus = calloc((size_t)nc*(size_t)ns, sizeof(*sparse_plus));
    sparse_cross = calloc((size_t)nc*(size_t)ns, sizeof(*sparse_cross));
    sparse_reference = calloc((size_t)nc*(size_t)ns,
                              sizeof(*sparse_reference));
    dense_reference = calloc((size_t)nc*(size_t)n,
                             sizeof(*dense_reference));
    pr = calloc((size_t)ns, sizeof(*pr));
    pi = calloc((size_t)ns, sizeof(*pi));
    cr = calloc((size_t)ns, sizeof(*cr));
    ci = calloc((size_t)ns, sizeof(*ci));
    if(sparse_plus == NULL || sparse_cross == NULL ||
       sparse_reference == NULL || dense_reference == NULL ||
       pr == NULL || pi == NULL || cr == NULL || ci == NULL)
    {
        status = 4;
        goto cleanup;
    }
    status = tphm_evaluate_carrier_polarization_arrays(
        source, ns, sparse_time, sparse_plus, sparse_cross,
        NULL, NULL, NULL, NULL);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    status = tphm_interpolate_carrier_references(
        source, ns, sparse_time, n, source_time,
        sparse_reference, dense_reference);
    if(status != 0)
    {
        status = 20+status;
        goto cleanup;
    }
    spr = gsl_spline_alloc(gsl_interp_akima, (size_t)ns);
    spi = gsl_spline_alloc(gsl_interp_akima, (size_t)ns);
    scr = gsl_spline_alloc(gsl_interp_akima, (size_t)ns);
    sci = gsl_spline_alloc(gsl_interp_akima, (size_t)ns);
    apr = gsl_interp_accel_alloc();
    api = gsl_interp_accel_alloc();
    acr = gsl_interp_accel_alloc();
    aci = gsl_interp_accel_alloc();
    if(spr == NULL || spi == NULL || scr == NULL || sci == NULL ||
       apr == NULL || api == NULL || acr == NULL || aci == NULL)
    {
        status = 5;
        goto cleanup;
    }
    for(k=0; k<nc; k++)
    {
        for(i=0; i<ns; i++)
        {
            size_t q = (size_t)k*(size_t)ns+(size_t)i;
            double complex plus_factor = 0.0, cross_factor = 0.0;
            if(cabs(sparse_reference[q]) > 1.0e-300)
            {
                plus_factor = sparse_plus[q]/sparse_reference[q];
                cross_factor = sparse_cross[q]/sparse_reference[q];
            }
            pr[i] = creal(plus_factor);
            pi[i] = cimag(plus_factor);
            cr[i] = creal(cross_factor);
            ci[i] = cimag(cross_factor);
        }
        if(gsl_spline_init(spr, sparse_time, pr, (size_t)ns) != GSL_SUCCESS ||
           gsl_spline_init(spi, sparse_time, pi, (size_t)ns) != GSL_SUCCESS ||
           gsl_spline_init(scr, sparse_time, cr, (size_t)ns) != GSL_SUCCESS ||
           gsl_spline_init(sci, sparse_time, ci, (size_t)ns) != GSL_SUCCESS)
        {
            status = 6;
            goto cleanup;
        }
        gsl_interp_accel_reset(apr);
        gsl_interp_accel_reset(api);
        gsl_interp_accel_reset(acr);
        gsl_interp_accel_reset(aci);
        for(i=0; i<n; i++)
        {
            size_t q = (size_t)k*(size_t)n+(size_t)i;
            double complex plus_factor =
                gsl_spline_eval(spr, source_time[i], apr)+
                I*gsl_spline_eval(spi, source_time[i], api);
            double complex cross_factor =
                gsl_spline_eval(scr, source_time[i], acr)+
                I*gsl_spline_eval(sci, source_time[i], aci);
            hplus[q] = dense_reference[q]*plus_factor;
            hcross[q] = dense_reference[q]*cross_factor;
        }
    }
    status = 0;

cleanup:
    if(aci != NULL) gsl_interp_accel_free(aci);
    if(acr != NULL) gsl_interp_accel_free(acr);
    if(api != NULL) gsl_interp_accel_free(api);
    if(apr != NULL) gsl_interp_accel_free(apr);
    if(sci != NULL) gsl_spline_free(sci);
    if(scr != NULL) gsl_spline_free(scr);
    if(spi != NULL) gsl_spline_free(spi);
    if(spr != NULL) gsl_spline_free(spr);
    free(ci);
    free(cr);
    free(pi);
    free(pr);
    free(dense_reference);
    free(sparse_reference);
    free(sparse_cross);
    free(sparse_plus);
    free(sparse_time);
    return status;
}

static int tphm_build_early_response_cache(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    PhenomTPHMTDISource *source,
    const THMWDMGridInfo *grid,
    double ecliptic_latitude,
    double ecliptic_longitude,
    double time_start,
    double endpoint_start,
    double endpoint_rise,
    const TPHMComplexEnvelopeCache *envelope,
    TPHMEarlyResponseCache *cache)
{
    TPHMEnvelopeFamilyEvaluator envelope_evaluator;
    THMComplexPolarizationFamilyEvaluator family_evaluator;
    void *family_userdata;
    double complex *channel[3] = {NULL, NULL, NULL};
    double complex *reference = NULL;
    double *real_value = NULL, *imag_value = NULL;
    const char *coarse_dt_text = getenv("TPHM_POST_TDI_ENVELOPE_DT");
    const char *fine_dt_text = getenv("TPHM_POST_TDI_ENVELOPE_FINE_DT");
    double coarse_dt, fine_start, fine_dt, t;
    int capacity, n, nc, i, k, ch, status = 0;

    if(context == NULL || workspace == NULL || source == NULL ||
       grid == NULL || cache == NULL)
        return 1;
    memset(cache, 0, sizeof(*cache));
    nc = source->waveform.ncarriers;
    coarse_dt = 10000.0;
    if(coarse_dt_text != NULL)
    {
        double requested = strtod(coarse_dt_text, NULL);
        if(isfinite(requested) && requested > 0.0) coarse_dt = requested;
    }
    fine_start = fmax(time_start, endpoint_start-2.0*endpoint_rise);
    fine_dt = fmin(coarse_dt, endpoint_rise/64.0);
    if(fine_dt_text != NULL)
    {
        double requested = strtod(fine_dt_text, NULL);
        if(isfinite(requested) && requested > 0.0)
            fine_dt = fmin(coarse_dt, requested);
    }
    if(fine_dt < grid->sample_dt) fine_dt = grid->sample_dt;
    capacity = (int)ceil((fine_start-time_start)/coarse_dt)+
               (int)ceil((endpoint_start+endpoint_rise-fine_start)/fine_dt)+8;
    if(capacity < 8) capacity = 8;
    cache->time = calloc((size_t)capacity, sizeof(*cache->time));
    if(cache->time == NULL) return 2;
    n = 0;
    t = time_start;
    while(t < endpoint_start+endpoint_rise && n < capacity-1)
    {
        cache->time[n++] = t;
        t += t < fine_start ? coarse_dt : fine_dt;
        if(t > fine_start && cache->time[n-1] < fine_start) t = fine_start;
    }
    cache->time[n++] = endpoint_start+endpoint_rise;
    if(n < 5)
    {
        status = 3;
        goto cleanup;
    }
    cache->n = n;
    cache->ncarriers = nc;
    envelope_evaluator.source = source;
    envelope_evaluator.envelope = envelope;
    family_evaluator = envelope != NULL ?
        tphm_all_envelope_carrier_polarizations :
        phenom_tphm_all_carrier_polarizations;
    family_userdata = envelope != NULL ?
        (void *)&envelope_evaluator : (void *)source;
    for(ch=0; ch<3; ch++)
        channel[ch] = calloc((size_t)nc*(size_t)n, sizeof(*channel[ch]));
    reference = calloc((size_t)nc*(size_t)n, sizeof(*reference));
    real_value = calloc((size_t)n, sizeof(*real_value));
    imag_value = calloc((size_t)n, sizeof(*imag_value));
    cache->real_spline = calloc((size_t)(3*nc), sizeof(*cache->real_spline));
    cache->imag_spline = calloc((size_t)(3*nc), sizeof(*cache->imag_spline));
    cache->real_acc = calloc((size_t)(3*nc), sizeof(*cache->real_acc));
    cache->imag_acc = calloc((size_t)(3*nc), sizeof(*cache->imag_acc));
    if(channel[0] == NULL || channel[1] == NULL || channel[2] == NULL ||
       reference == NULL || real_value == NULL || imag_value == NULL ||
       cache->real_spline == NULL || cache->imag_spline == NULL ||
       cache->real_acc == NULL || cache->imag_acc == NULL)
    {
        status = 4;
        goto cleanup;
    }
    status = thm_evaluate_complex_tdi_families_context(
        context, workspace, n, cache->time,
        ecliptic_latitude, ecliptic_longitude, nc,
        family_evaluator, family_userdata,
        channel[0], channel[1], channel[2]);
    if(status != 0)
    {
        if(getenv("TPHM_TDI_DEBUG") != NULL)
            fprintf(stderr, "TPHM_TDI_DEBUG early response TDI status %d time [%.15e, %.15e]\n",
                    status, cache->time[0], cache->time[n-1]);
        status = 10+status;
        goto cleanup;
    }
    status = tphm_evaluate_carrier_references(source, -1, n,
                                               cache->time, reference);
    if(status != 0)
    {
        if(getenv("TPHM_TDI_DEBUG") != NULL)
            fprintf(stderr, "TPHM_TDI_DEBUG early carrier reference status %d time [%.15e, %.15e]\n",
                    status, cache->time[0], cache->time[n-1]);
        status = 20+status;
        goto cleanup;
    }
    for(ch=0; ch<3; ch++)
    {
        for(k=0; k<nc; k++)
        {
            int q = ch*nc+k;
            for(i=0; i<n; i++)
            {
                size_t index = (size_t)k*(size_t)n+(size_t)i;
                double magnitude = cabs(reference[index]);
                double complex phasor = magnitude > 1.0e-300 ?
                    reference[index]/magnitude : 1.0;
                double complex response_envelope =
                    channel[ch][index]*conj(phasor);
                real_value[i] = creal(response_envelope);
                imag_value[i] = cimag(response_envelope);
            }
            cache->real_spline[q] = gsl_spline_alloc(gsl_interp_akima,
                                                       (size_t)n);
            cache->imag_spline[q] = gsl_spline_alloc(gsl_interp_akima,
                                                       (size_t)n);
            cache->real_acc[q] = gsl_interp_accel_alloc();
            cache->imag_acc[q] = gsl_interp_accel_alloc();
            if(cache->real_spline[q] == NULL ||
               cache->imag_spline[q] == NULL ||
               cache->real_acc[q] == NULL || cache->imag_acc[q] == NULL ||
               gsl_spline_init(cache->real_spline[q], cache->time,
                               real_value, (size_t)n) != GSL_SUCCESS ||
               gsl_spline_init(cache->imag_spline[q], cache->time,
                               imag_value, (size_t)n) != GSL_SUCCESS)
            {
                status = 5;
                goto cleanup;
            }
        }
    }
    status = 0;

cleanup:
    free(imag_value);
    free(real_value);
    free(reference);
    for(ch=0; ch<3; ch++) free(channel[ch]);
    if(status != 0) tphm_early_response_cache_free(cache);
    return status;
}

static int tphm_build_fast_carrier_block(
    PhenomTPHMTDISource *source,
    const TPHMEarlyResponseCache *cache,
    int carrier_index,
    int nfft,
    double sample_start,
    double sample_dt,
    double heterodyne_frequency,
    double rise_start,
    double rise_end,
    double fall_start,
    double fall_end,
    THMComplexTDIFFTBlock *block)
{
    double *time = NULL;
    double complex *reference = NULL;
    int eval_stop, neval, i, ch, status;

    if(source == NULL || cache == NULL || block == NULL || nfft < 2 ||
       carrier_index < 0 || carrier_index >= cache->ncarriers)
        return 1;
    eval_stop = nfft;
    if(fall_end > fall_start)
    {
        eval_stop = (int)ceil((fall_end-sample_start)/sample_dt)+1;
        if(eval_stop > nfft) eval_stop = nfft;
    }
    if(eval_stop < 2) return 2;
    neval = eval_stop;
    time = calloc((size_t)neval, sizeof(*time));
    reference = calloc((size_t)neval, sizeof(*reference));
    thm_complex_tdi_fft_block_free(block);
    for(ch=0; ch<3; ch++)
        block->fft[ch] = calloc((size_t)(2*nfft), sizeof(*block->fft[ch]));
    if(time == NULL || reference == NULL || block->fft[0] == NULL ||
       block->fft[1] == NULL || block->fft[2] == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(i=0; i<neval; i++) time[i] = sample_start+sample_dt*(double)i;
    status = tphm_evaluate_carrier_references(source, carrier_index,
                                               neval, time, reference);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    for(i=0; i<nfft; i++)
    {
        double t = sample_start+sample_dt*(double)i;
        double weight = 1.0;
        double phase = 2.0*M_PI*heterodyne_frequency*(t-sample_start);
        double complex demod = cos(phase)-I*sin(phase);
        if(rise_end > rise_start)
            weight *= tphm_smooth_step(t, rise_start, rise_end);
        if(fall_end > fall_start)
            weight *= 1.0-tphm_smooth_step(t, fall_start, fall_end);
        for(ch=0; ch<3; ch++)
        {
            double complex value = 0.0;
            if(i < neval && t >= cache->time[0] &&
               t <= cache->time[cache->n-1])
            {
                int q = ch*cache->ncarriers+carrier_index;
                double re = gsl_spline_eval(cache->real_spline[q], t,
                                             cache->real_acc[q]);
                double im = gsl_spline_eval(cache->imag_spline[q], t,
                                             cache->imag_acc[q]);
                double magnitude = cabs(reference[i]);
                double complex phasor = magnitude > 1.0e-300 ?
                    reference[i]/magnitude : 1.0;
                value = weight*phasor*(re+I*im)*demod;
            }
            block->fft[ch][2*i] = creal(value);
            block->fft[ch][2*i+1] = cimag(value);
        }
    }
    for(ch=0; ch<3; ch++)
    {
        gsl_fft_complex_radix2_forward(block->fft[ch], 1, (size_t)nfft);
        for(i=0; i<nfft; i++)
        {
            block->fft[ch][2*i] *= sample_dt;
            block->fft[ch][2*i+1] *= sample_dt;
        }
    }
    block->nfft = nfft;
    block->sample_start = sample_start;
    block->sample_dt = sample_dt;
    block->heterodyne_frequency = heterodyne_frequency;
    block->nonzero_start = rise_end > rise_start ? rise_start : sample_start;
    block->nonzero_end = fall_end > fall_start ? fall_end :
                         sample_start+sample_dt*(double)(nfft-1);
    status = 0;

cleanup:
    free(reference);
    free(time);
    if(status != 0) thm_complex_tdi_fft_block_free(block);
    return status;
}

int phenom_tphm_build_narrow_partition_context(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    PhenomTPHMTDISource *source,
    double ecliptic_latitude,
    double ecliptic_longitude,
    double time_start,
    double time_stop,
    double endpoint_start,
    double endpoint_rise,
    double bandwidth_hz,
    double block_roll_seconds,
    PhenomTPHMPartitionPlan *plan)
{
    THMWDMGridInfo grid;
    TPHMEarlyResponseCache response_cache;
    TPHMSharedEulerCache euler_cache;
    TPHMComplexEnvelopeCache envelope_cache;
    TPHMCachedEvaluationTiming evaluation_timing;
    TPHMNarrowBlockSpec *specs = NULL;
    double *track_time = NULL, *fcenter = NULL, *flow = NULL, *fhigh = NULL;
    int ntrack = 0, nc, k, status = 0;
    const char *failure_stage = "initialization";
    int nspec = 0, spec_capacity = 0;
    int start_pixel, endpoint_pixel, roll_pixels;
    clock_t stage_start;
    const char *post_tdi_envelope_option =
        getenv("TPHM_POST_TDI_ENVELOPE_GROUPS");
    const int use_factorized_early_response =
        tphm_post_tdi_envelope_groups_override >= 0 ?
        tphm_post_tdi_envelope_groups_override :
        (post_tdi_envelope_option == NULL ||
         atoi(post_tdi_envelope_option) != 0);
    const char *envelope_option = getenv("TPHM_COMPLEX_ENVELOPE_GROUPS");
    const int use_complex_envelope_groups =
        tphm_complex_envelope_groups_override >= 0 ?
        tphm_complex_envelope_groups_override :
        (envelope_option != NULL && atoi(envelope_option) != 0);

    memset(&response_cache, 0, sizeof(response_cache));
    memset(&euler_cache, 0, sizeof(euler_cache));
    memset(&envelope_cache, 0, sizeof(envelope_cache));
    memset(&evaluation_timing, 0, sizeof(evaluation_timing));

    if(context == NULL || workspace == NULL || source == NULL ||
       !source->initialized || plan == NULL || !(time_stop > time_start) ||
       !(endpoint_start > time_start) || !(time_stop > endpoint_start) ||
       endpoint_rise <= 0.0 || bandwidth_hz <= 0.0 ||
       block_roll_seconds <= 0.0)
        return 1;
    if(thm_observation_wdm_grid_info(context, &grid) != 0) return 2;
    if(time_start < 0.0 || time_stop >
       grid.time_pixel_dt*(double)grid.time_pixels)
        return 3;
    phenom_tphm_partition_plan_free(plan);
    nc = source->waveform.ncarriers;
    stage_start = clock();
    failure_stage = "planning tracks";
    status = tphm_build_planning_tracks(source, &grid, time_start, time_stop,
                                        &ntrack, &track_time,
                                        &fcenter, &flow, &fhigh);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    plan->planning_seconds =
        (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    failure_stage = "WDM support";
    status = tphm_build_compact_wdm_plan(
        &grid, nc, ntrack, track_time, fcenter, flow, fhigh,
        time_start, time_stop, endpoint_start, endpoint_rise, plan);
    if(status != 0)
    {
        status = 30+status;
        goto cleanup;
    }
    start_pixel = (int)floor(time_start/grid.time_pixel_dt);
    endpoint_pixel = (int)floor(endpoint_start/grid.time_pixel_dt);
    if(endpoint_pixel <= start_pixel)
    {
        status = 4;
        goto cleanup;
    }
    roll_pixels = (int)ceil(block_roll_seconds/grid.time_pixel_dt);
    if(roll_pixels < 1) roll_pixels = 1;
    if(endpoint_pixel-start_pixel > 0 &&
       nc <= 1073741824/(endpoint_pixel-start_pixel))
        spec_capacity = nc*(endpoint_pixel-start_pixel);
    if(spec_capacity < nc) spec_capacity = nc;
    specs = calloc((size_t)spec_capacity, sizeof(*specs));
    if(specs == NULL)
    {
        status = 5;
        goto cleanup;
    }

    stage_start = clock();
    for(k=0; k<nc; k++)
    {
        const double *carrier_low = &flow[(size_t)k*(size_t)ntrack];
        const double *carrier_high = &fhigh[(size_t)k*(size_t)ntrack];
        double *boundary = NULL, *boundary_roll = NULL;
        int max_blocks = endpoint_pixel-start_pixel+1;
        int nblocks = 0, tile_lo = start_pixel, block;

        boundary = calloc((size_t)max_blocks, sizeof(*boundary));
        boundary_roll = calloc((size_t)max_blocks,
                               sizeof(*boundary_roll));
        if(boundary == NULL || boundary_roll == NULL)
        {
            free(boundary_roll);
            free(boundary);
            status = 5;
            goto cleanup;
        }
        while(tile_lo < endpoint_pixel)
        {
            int remaining = endpoint_pixel-tile_lo;
            int width = tphm_largest_power_of_two(remaining);
            int best = 0;
            while(width >= 1)
            {
                double tlo = (double)(tile_lo-roll_pixels)*
                             grid.time_pixel_dt;
                double thi = (double)(tile_lo+width+roll_pixels)*
                             grid.time_pixel_dt;
                double lo, hi;
                double pad = grid.meyer_half_bandwidth+
                             3.0/block_roll_seconds;
                if(tlo < time_start) tlo = time_start;
                if(thi > endpoint_start+endpoint_rise)
                    thi = endpoint_start+endpoint_rise;
                if(!tphm_track_range(ntrack, track_time,
                                     carrier_low, carrier_high,
                                     tlo, thi, &lo, &hi) ||
                   hi-lo+2.0*pad <= bandwidth_hz)
                {
                    best = width;
                    break;
                }
                width /= 2;
            }
            if(best < 1) best = 1;
            tile_lo += best;
            if(tile_lo > endpoint_pixel) tile_lo = endpoint_pixel;
            boundary[nblocks++] = (double)tile_lo*grid.time_pixel_dt;
        }
        for(block=0; block<nblocks; block++)
            boundary_roll[block] = block_roll_seconds;
        for(block=0; block<nblocks-1; block++)
        {
            double available = boundary[block+1]-boundary[block];
            if(boundary_roll[block] > available)
                boundary_roll[block] = available;
            if(boundary_roll[block] < grid.time_pixel_dt)
                boundary_roll[block] = grid.time_pixel_dt;
        }

        for(block=0; block<nblocks; block++)
        {
            TPHMNarrowBlockSpec *block_spec;
            double nonzero_start = block == 0 ? time_start : boundary[block-1];
            double nonzero_stop = block < nblocks-1 ?
                boundary[block]+boundary_roll[block] :
                endpoint_start+endpoint_rise;
            double rise_start = block == 0 ? time_start : boundary[block-1];
            double rise_end = block == 0 ?
                fmin(time_start+block_roll_seconds, nonzero_stop) :
                boundary[block-1]+boundary_roll[block-1];
            double fall_start = block < nblocks-1 ?
                boundary[block] : endpoint_start;
            double fall_end = block < nblocks-1 ?
                boundary[block]+boundary_roll[block] :
                endpoint_start+endpoint_rise;
            double lo, hi, pad, fstart, fstop, center, shift;
            double sample_start;
            int support_pixels, K, local_bins, nfft;
            long long nfft64;

            if(!tphm_track_range(ntrack, track_time,
                                 carrier_low, carrier_high,
                                 nonzero_start, nonzero_stop, &lo, &hi))
            {
                status = 6;
                free(boundary_roll);
                free(boundary);
                goto cleanup;
            }
            pad = grid.meyer_half_bandwidth;
            if(rise_end > rise_start)
                pad = fmax(pad, grid.meyer_half_bandwidth+
                            3.0/(rise_end-rise_start));
            if(fall_end > fall_start)
                pad = fmax(pad, grid.meyer_half_bandwidth+
                            3.0/(fall_end-fall_start));
            fstart = fmax(0.0, lo-pad);
            fstop = hi+pad;
            center = 0.5*(fstart+fstop);
            shift = 2.0*nearbyint(center/(2.0*grid.frequency_pixel_df))*
                    grid.frequency_pixel_df;
            local_bins = tphm_block_local_bins(&grid, shift, fstart, fstop);
            sample_start = floor(nonzero_start/grid.time_pixel_dt)*
                           grid.time_pixel_dt;
            support_pixels = (int)ceil((nonzero_stop-sample_start)/
                                       grid.time_pixel_dt);
            if(support_pixels < 1) support_pixels = 1;
            /* Meyer packets extend mult WDM pixels on each side.  Zero-pad
             * the block to that packet-compatible duration so every packet
             * frequency is an exact FFT bin rather than silently sampling
             * only every K/Ntx bin. */
            K = tphm_next_power_of_two(
                support_pixels+2*grid.packet_time_half_support);
            if(K > grid.time_pixels) K = grid.time_pixels;
            nfft64 = (long long)K*(long long)local_bins;
            if(nfft64 > 1073741824LL)
            {
                status = 7;
                free(boundary_roll);
                free(boundary);
                goto cleanup;
            }
            nfft = (int)nfft64;
            if(nspec >= spec_capacity)
            {
                status = 8;
                free(boundary_roll);
                free(boundary);
                goto cleanup;
            }
            block_spec = &specs[nspec++];
            block_spec->carrier_index = k;
            block_spec->nfft = nfft;
            block_spec->sample_start = sample_start;
            block_spec->sample_dt =
                grid.time_pixel_dt/(double)local_bins;
            block_spec->source_sample_dt = 0.5*block_spec->sample_dt;
            block_spec->heterodyne_frequency = shift;
            block_spec->rise_start = rise_start;
            block_spec->rise_end = rise_end;
            block_spec->fall_start = fall_start;
            block_spec->fall_end = fall_end;
            block_spec->frequency_start = fstart;
            block_spec->frequency_stop = fstop;
        }
        free(boundary_roll);
        free(boundary);
    }

    /*
     * A packet containing N WDM time pixels requires Fourier samples spaced
     * by 1/(N DT).  Zero-pad every contributing heterodyned block to at least
     * that duration.  Without this step the exact-bin sampler would quietly
     * return zero between the coarser block FFT bins.
     */
    for(k=0; k<nspec; k++)
    {
        TPHMNarrowBlockSpec *target = &specs[k];
        int local_bins = (int)llround(grid.time_pixel_dt/target->sample_dt);
        int K = local_bins > 0 ? target->nfft/local_bins : 0;
        int m;
        if(local_bins < 1 || K < 1)
        {
            status = 11;
            goto cleanup;
        }
        for(m=1; m<grid.frequency_layers; m++)
        {
            double layer_low, layer_high, packet_start, packet_stop;
            int Ntx;
            if(plan->wdm_nmid[m] < 0 || plan->wdm_nsize[m] <= 0)
                continue;
            layer_low = (double)m*grid.frequency_pixel_df-
                        grid.meyer_half_bandwidth;
            layer_high = (double)m*grid.frequency_pixel_df+
                         grid.meyer_half_bandwidth;
            if(layer_high < target->frequency_start ||
               layer_low > target->frequency_stop)
                continue;
            Ntx = plan->wdm_nsize[m];
            packet_start = (double)(plan->wdm_nmid[m]-Ntx/2)*
                           grid.time_pixel_dt;
            packet_stop = packet_start+(double)Ntx*grid.time_pixel_dt;
            if(!(packet_stop > target->rise_start &&
                 packet_start < target->fall_end))
                continue;
            if(Ntx > K) K = Ntx;
        }
        if((long long)K*(long long)local_bins > 1073741824LL)
        {
            status = 12;
            goto cleanup;
        }
        target->nfft = K*local_bins;
    }
    plan->layout_seconds =
        (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;

    stage_start = clock();
    failure_stage = "shared Euler cache";
    status = tphm_build_shared_euler_cache(context, source, nspec, specs,
                                            &euler_cache);
    if(status != 0)
    {
        status = 60+status;
        goto cleanup;
    }
    plan->shared_dynamics_samples = euler_cache.n;
    plan->shared_dynamics_seconds =
        (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;

    if(use_complex_envelope_groups || use_factorized_early_response)
    {
        stage_start = clock();
        failure_stage = "complex envelope cache";
        status = tphm_build_complex_envelope_cache(
            context, source, &euler_cache, nspec, specs, &envelope_cache);
        if(status != 0)
        {
            status = 70+status;
            goto cleanup;
        }
        plan->envelope_cache_seconds =
            (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
        plan->envelope_cache_samples = envelope_cache.n;
        plan->used_complex_envelope_groups = 1;
    }

    if(use_factorized_early_response)
    {
        stage_start = clock();
        failure_stage = "early response cache";
        status = tphm_build_early_response_cache(
            context, workspace, source, &grid,
            ecliptic_latitude, ecliptic_longitude,
            time_start, endpoint_start, endpoint_rise,
            &envelope_cache, &response_cache);
        if(status != 0)
        {
            status = 50+status;
            goto cleanup;
        }
        plan->used_post_tdi_envelope_groups = 1;
        plan->post_tdi_envelope_samples = response_cache.n;
        plan->response_seconds =
            (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    }

    stage_start = clock();
    failure_stage = "early FFT blocks";
    for(k=0; k<nspec; k++)
    {
        const TPHMNarrowBlockSpec *block_spec = &specs[k];
        THMComplexTDIFFTBlock built;
        TPHMCachedCarrierEvaluator cached;
        TPHMEnvelopeCarrierEvaluator envelope_cached;
        thm_complex_tdi_fft_block_init(&built);
        cached.selection.source = source;
        cached.selection.carrier_index = block_spec->carrier_index;
        cached.euler = &euler_cache;
        cached.timing = &evaluation_timing;
        envelope_cached.selection.source = source;
        envelope_cached.selection.carrier_index =
            block_spec->carrier_index;
        envelope_cached.envelope = &envelope_cache;
        envelope_cached.timing = &evaluation_timing;
        if(use_factorized_early_response)
        {
            status = tphm_build_fast_carrier_block(
                source, &response_cache, block_spec->carrier_index,
                block_spec->nfft, block_spec->sample_start,
                block_spec->sample_dt,
                block_spec->heterodyne_frequency,
                block_spec->rise_start, block_spec->rise_end,
                block_spec->fall_start, block_spec->fall_end, &built);
        }
        else if(use_complex_envelope_groups)
        {
            status = thm_build_complex_tdi_fft_block_interpolated_context(
                context, workspace, tphm_envelope_carrier_polarizations,
                &envelope_cached, ecliptic_latitude, ecliptic_longitude,
                block_spec->nfft, block_spec->sample_start,
                block_spec->sample_dt, block_spec->source_sample_dt,
                block_spec->heterodyne_frequency,
                block_spec->rise_start, block_spec->rise_end,
                block_spec->fall_start, block_spec->fall_end, &built);
        }
        else
        {
            status = thm_build_complex_tdi_fft_block_interpolated_context(
                context, workspace, tphm_cached_carrier_polarizations,
                &cached, ecliptic_latitude, ecliptic_longitude,
                block_spec->nfft, block_spec->sample_start,
                block_spec->sample_dt, block_spec->source_sample_dt,
                block_spec->heterodyne_frequency,
                block_spec->rise_start, block_spec->rise_end,
                block_spec->fall_start, block_spec->fall_end, &built);
        }
        if(status != 0)
        {
            thm_complex_tdi_fft_block_free(&built);
            status = 100+status;
            goto cleanup;
        }
        built.frequency_start = block_spec->frequency_start;
        built.frequency_stop = block_spec->frequency_stop;
        built.carrier_index = block_spec->carrier_index;
        built.is_endpoint = 0;
        if(!tphm_partition_plan_reserve(plan, plan->nblocks+1))
        {
            thm_complex_tdi_fft_block_free(&built);
            status = 8;
            goto cleanup;
        }
        plan->blocks[plan->nblocks++] = built;
        plan->narrow_blocks++;
        plan->total_fft_samples += block_spec->nfft;
    }
    plan->early_blocks_seconds =
        (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    plan->early_source_seconds =
        evaluation_timing.total_seconds;
    plan->early_phase_anchor_seconds =
        evaluation_timing.phase_anchor_seconds;
    plan->early_carrier_seconds =
        evaluation_timing.carrier_seconds;
    plan->early_source_samples = evaluation_timing.samples;
    plan->early_source_calls = evaluation_timing.calls;

    {
        THMComplexTDIFFTBlock endpoint;
        failure_stage = "endpoint FFT";
        double endpoint_sample_start = floor(endpoint_start/
            grid.time_pixel_dt)*grid.time_pixel_dt;
        double physical_waveform_stop = source->coalescence_time+500.0+
                                        1000.0*source->total_mass;
        /* A physical THM ringdown closes itself.  Retain the ordinary long
         * falling taper only for diagnostic intervals that stop early. */
        int natural_ringdown_endpoint =
            fabs(time_stop-physical_waveform_stop) <= grid.sample_dt;
        double fall_start = natural_ringdown_endpoint ? time_stop :
            fmax(endpoint_start+endpoint_rise,
                 time_stop-block_roll_seconds);
        double fall_end = natural_ringdown_endpoint ?
                          time_stop+grid.sample_dt : time_stop;
        int support_pixels = (int)ceil((time_stop-endpoint_sample_start)/
                                      grid.time_pixel_dt);
        int K, nfft, m;
        if(support_pixels < 1) support_pixels = 1;
        K = tphm_next_power_of_two(
            support_pixels+2*grid.packet_time_half_support);
        for(m=1; m<grid.frequency_layers; m++)
        {
            double layer_low = (double)m*grid.frequency_pixel_df-
                               grid.meyer_half_bandwidth;
            double layer_high = (double)m*grid.frequency_pixel_df+
                                grid.meyer_half_bandwidth;
            if(plan->endpoint_wdm_nmid[m] < 0 ||
               plan->endpoint_wdm_nsize[m] <= 0 ||
               layer_high < plan->endpoint_frequency_start ||
               layer_low > plan->endpoint_frequency_stop)
                continue;
            if(plan->endpoint_wdm_nsize[m] > K)
                K = plan->endpoint_wdm_nsize[m];
        }
        if(K > grid.time_pixels) K = grid.time_pixels;
        if((long long)K*(long long)grid.frequency_layers > 1073741824LL)
        {
            status = 9;
            goto cleanup;
        }
        nfft = K*grid.frequency_layers;
        thm_complex_tdi_fft_block_init(&endpoint);
        stage_start = clock();
        status = thm_build_complex_tdi_fft_block_interpolated_context(
            context, workspace,
            phenom_tphm_complex_polarizations,
            source,
            ecliptic_latitude, ecliptic_longitude, nfft,
            endpoint_sample_start, grid.sample_dt, 0.5*grid.sample_dt, 0.0,
            endpoint_start, endpoint_start+endpoint_rise,
            fall_start, fall_end, &endpoint);
        if(status != 0)
        {
            status = 200+status;
            goto cleanup;
        }
        endpoint.frequency_start = plan->endpoint_frequency_start;
        endpoint.frequency_stop = plan->endpoint_frequency_stop;
        endpoint.carrier_index = -1;
        endpoint.is_endpoint = 1;
        if(!tphm_partition_plan_reserve(plan, plan->nblocks+1))
        {
            thm_complex_tdi_fft_block_free(&endpoint);
            status = 10;
            goto cleanup;
        }
        plan->blocks[plan->nblocks++] = endpoint;
        plan->endpoint_blocks = 1;
        plan->total_fft_samples += nfft;
        plan->endpoint_seconds =
            (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    }
    status = 0;

cleanup:
    if(status != 0 && getenv("TPHM_TDI_DEBUG") != NULL)
        fprintf(stderr, "TPHM_TDI_DEBUG plan failed in %s: %d\n",
                failure_stage, status);
    tphm_complex_envelope_cache_free(&envelope_cache);
    tphm_shared_euler_cache_free(&euler_cache);
    tphm_early_response_cache_free(&response_cache);
    free(specs);
    free(fhigh);
    free(flow);
    free(fcenter);
    free(track_time);
    if(status != 0) phenom_tphm_partition_plan_free(plan);
    return status;
}

static int tphm_add_sparse_channel(THMSparseWDMChannel *target,
                                   const THMSparseWDMChannel *add)
{
    int *new_n = NULL, *new_m = NULL;
    double *new_value = NULL;
    int ia = 0, ib = 0, out = 0;
    int capacity;

    if(target == NULL || add == NULL) return 0;
    capacity = target->npixels+add->npixels;
    if(capacity > 0)
    {
        new_n = malloc((size_t)capacity*sizeof(*new_n));
        new_m = malloc((size_t)capacity*sizeof(*new_m));
        new_value = malloc((size_t)capacity*sizeof(*new_value));
        if(new_n == NULL || new_m == NULL || new_value == NULL)
        {
            free(new_value);
            free(new_m);
            free(new_n);
            return 0;
        }
    }
    while(ia < target->npixels || ib < add->npixels)
    {
        int take_target = 0, take_add = 0;
        if(ib >= add->npixels)
            take_target = 1;
        else if(ia >= target->npixels)
            take_add = 1;
        else if(target->m[ia] < add->m[ib] ||
                (target->m[ia] == add->m[ib] &&
                 target->n[ia] < add->n[ib]))
            take_target = 1;
        else if(add->m[ib] < target->m[ia] ||
                (add->m[ib] == target->m[ia] &&
                 add->n[ib] < target->n[ia]))
            take_add = 1;
        else
            take_target = take_add = 1;

        if(take_target)
        {
            new_n[out] = target->n[ia];
            new_m[out] = target->m[ia];
            new_value[out] = target->value[ia];
            ia++;
        }
        if(take_add)
        {
            if(!take_target)
            {
                new_n[out] = add->n[ib];
                new_m[out] = add->m[ib];
                new_value[out] = 0.0;
            }
            new_value[out] += add->value[ib];
            ib++;
        }
        out++;
    }
    free(target->value);
    free(target->m);
    free(target->n);
    target->n = new_n;
    target->m = new_m;
    target->value = new_value;
    target->npixels = out;
    target->capacity = capacity;
    return 1;
}

int phenom_tphm_partition_to_wdm_context(
    const THMObservationContext *context,
    const PhenomTPHMPartitionPlan *plan,
    THMSparseWDMTriplet *out_tracks)
{
    THMSparseWDMTriplet endpoint_tracks;
    int ch, status;

    if(plan == NULL || plan->nblocks < 1 || plan->wdm_nmid == NULL ||
       plan->wdm_nsize == NULL || plan->endpoint_wdm_nmid == NULL ||
       plan->endpoint_wdm_nsize == NULL || out_tracks == NULL ||
       plan->narrow_blocks < 1 || plan->endpoint_blocks != 1 ||
       plan->narrow_blocks+plan->endpoint_blocks != plan->nblocks)
        return 1;
    thm_sparse_wdm_triplet_init(&endpoint_tracks);
    status = thm_complex_fft_blocks_to_sparse_wdm_plan_context(
        context, plan->narrow_blocks, plan->blocks,
        plan->wdm_nmid, plan->wdm_nsize, out_tracks);
    if(status != 0) goto cleanup;
    status = thm_complex_fft_blocks_to_sparse_wdm_plan_context(
        context, 1, &plan->blocks[plan->narrow_blocks],
        plan->endpoint_wdm_nmid, plan->endpoint_wdm_nsize,
        &endpoint_tracks);
    if(status != 0) goto cleanup;
    for(ch=0; ch<3; ch++)
    {
        if(!tphm_add_sparse_channel(&out_tracks->channel[ch],
                                    &endpoint_tracks.channel[ch]))
        {
            status = 2;
            goto cleanup;
        }
    }

cleanup:
    thm_sparse_wdm_triplet_free(&endpoint_tracks);
    return status;
}

int phenom_tphm_partition_to_conservative_wdm_context(
    const THMObservationContext *context,
    const PhenomTPHMPartitionPlan *plan,
    THMSparseWDMTriplet *out_tracks)
{
    THMSparseWDMTriplet block_tracks;
    int block, ch, status = 0;

    if(context == NULL || plan == NULL || plan->nblocks < 1 ||
       plan->blocks == NULL || out_tracks == NULL)
        return 1;
    for(ch=0; ch<3; ch++) out_tracks->channel[ch].npixels = 0;
    thm_sparse_wdm_triplet_init(&block_tracks);
    for(block=0; block<plan->nblocks; block++)
    {
        status = thm_complex_fft_blocks_to_planned_sparse_wdm_context(
            context, 1, &plan->blocks[block], &block_tracks);
        if(status != 0) goto cleanup;
        for(ch=0; ch<3; ch++)
        {
            if(!tphm_add_sparse_channel(&out_tracks->channel[ch],
                                        &block_tracks.channel[ch]))
            {
                status = 2;
                goto cleanup;
            }
        }
    }

cleanup:
    thm_sparse_wdm_triplet_free(&block_tracks);
    return status;
}
