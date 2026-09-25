/*
 * LISA response and sparse WDM interface for IMRPhenomTPHM.
 * Copyright (C) 2026 Neil Cornish
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Free software under GNU GPL version 3 or later, without warranty.
 * The LAL-derived waveform model has separate upstream notices.
 */

#ifndef PHENOMTPHM_TDI_H
#define PHENOMTPHM_TDI_H

#include <complex.h>

#include "IMRPhenomTPHM.h"
#include "PhenomTHM_TDI.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Intrinsic TPHM state plus the fixed source-frame projection needed by TDI.
 * Masses and all times are in seconds.  distance_gpc supplies the physical
 * strain normalization used by PhenomTHM_TDI.c.
 *
 * phi22_anchor is the co-precessing 22 phase at source_time_anchor.  Keeping
 * this anchor fixed makes independently generated partitioned-FFT blocks
 * phase coherent.
 */
typedef struct
{
    IMRPhenomTPHM waveform;
    double total_mass;
    double eta;
    double coalescence_time;
    double source_time_anchor;
    double phi22_anchor;
    double observer_theta;
    double observer_phi;
    double polarization;
    double strain_scale;
    int initialized;
} PhenomTPHMTDISource;

typedef struct
{
    PhenomTPHMTDISource *source;
    int carrier_index;
} PhenomTPHMCarrierEvaluator;

typedef struct
{
    int nblocks;
    int capacity;
    int narrow_blocks;
    int endpoint_blocks;
    long long total_fft_samples;
    int shared_dynamics_samples;
    double planning_seconds;
    double layout_seconds;
    double shared_dynamics_seconds;
    double envelope_cache_seconds;
    double response_seconds;
    double early_blocks_seconds;
    double early_source_seconds;
    double early_phase_anchor_seconds;
    double early_carrier_seconds;
    long long early_source_samples;
    int early_source_calls;
    int envelope_cache_samples;
    int used_complex_envelope_groups;
    int post_tdi_envelope_samples;
    int used_post_tdi_envelope_groups;
    double endpoint_seconds;
    int wdm_active_layers;
    int wdm_pixels_per_channel;
    double endpoint_frequency_start;
    double endpoint_frequency_stop;
    int *wdm_nmid;
    int *wdm_nsize;
    int *endpoint_wdm_nmid;
    int *endpoint_wdm_nsize;
    THMComplexTDIFFTBlock *blocks;
} PhenomTPHMPartitionPlan;

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
    double polarization);

void phenom_tphm_tdi_source_destroy(PhenomTPHMTDISource *source);

/* THMComplexPolarizationEvaluator-compatible callback. */
int phenom_tphm_complex_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic);

/* Selected folded-carrier version of the complex polarization callback. */
int phenom_tphm_carrier_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic);

/* All configured folded carriers in carrier-major storage. */
int phenom_tphm_all_carrier_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    int nfamilies,
    double complex *hplus_analytic,
    double complex *hcross_analytic);

/* Apply unequal-arm TDI-on-the-fly to the complete twisted-up waveform. */
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
    double complex *Z);

void phenom_tphm_partition_plan_init(PhenomTPHMPartitionPlan *plan);
void phenom_tphm_partition_plan_free(PhenomTPHMPartitionPlan *plan);

/* Parent-carrier complex-envelope controls.  The post-TDI grouped envelope is
 * the default; forcing it off restores blockwise sparse TDI.  A negative
 * value restores environment-controlled/default behavior, while zero and one
 * force a path off/on.  The upstream-only cache remains a convergence option. */
void phenom_tphm_set_complex_envelope_groups(int enabled);
void phenom_tphm_set_post_tdi_envelope_groups(int enabled);

/* Mass-scaled default for the complete-carrier endpoint transform. */
void phenom_tphm_default_endpoint_window(
    const PhenomTPHMTDISource *source,
    double *endpoint_start,
    double *endpoint_rise,
    double *waveform_stop);
/* Use this version when the observation context selects TDI-2: its endpoint
 * must cover the full second-generation retarded-delay reach. */
void phenom_tphm_default_endpoint_window_context(
    const THMObservationContext *context,
    const PhenomTPHMTDISource *source,
    double *endpoint_start,
    double *endpoint_rise,
    double *waveform_stop);

/*
 * Build the efficient no-SPA TPHM transform.  Before endpoint_start, each
 * folded carrier is assigned independent power-of-two time blocks whose
 * duration is limited by bandwidth_hz.  Every block is shifted by an even
 * multiple of DF and sampled only fast enough for its complex baseband.
 * The early blocks fall while one complete, full-cadence endpoint block rises,
 * so their sum is an exact partition of the time-domain waveform.
 *
 * Times are detector/WDM times.  endpoint_start is the beginning of the
 * complementary early/late taper and endpoint_rise is its duration.
 */
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
    PhenomTPHMPartitionPlan *plan);

int phenom_tphm_partition_to_wdm_context(
    const THMObservationContext *context,
    const PhenomTPHMPartitionPlan *plan,
    THMSparseWDMTriplet *out_tracks);

/* Broad block-rectangle reference used to audit compact TPHM support. */
int phenom_tphm_partition_to_conservative_wdm_context(
    const THMObservationContext *context,
    const PhenomTPHMPartitionPlan *plan,
    THMSparseWDMTriplet *out_tracks);

#ifdef __cplusplus
}
#endif

#endif
