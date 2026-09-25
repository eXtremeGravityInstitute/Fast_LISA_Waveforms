/*
 * LISA response and sparse WDM interface for IMRPhenomTHM.
 * Copyright (C) 2025, 2026 Neil Cornish
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Free software under GNU GPL version 3 or later, without warranty.
 * The LAL-derived waveform model has separate upstream notices.
 */

#ifndef PHENOMTHM_TDI_H
#define PHENOMTHM_TDI_H

#include <complex.h>
#include <gsl/gsl_spline.h>

#include "IMRPhenomTHM.h"

#define THM_TDI_PI 3.141592653589793238462643383279502884
#define THM_TDI_AU_METERS 1.4959787e11
#define THM_TDI_CLIGHT 2.99792458e8
#define THM_TDI_SQRT3 1.73205080756887729352744634150587237
#define THM_TDI_ORBIT_ECCENTRICITY 0.0048241852175
#define THM_TDI_LARM_NOMINAL_SECONDS \
    (2.0*THM_TDI_SQRT3*THM_TDI_AU_METERS*THM_TDI_ORBIT_ECCENTRICITY/THM_TDI_CLIGHT)
#define THM_TDI_LARM_NOMINAL_METERS \
    (THM_TDI_LARM_NOMINAL_SECONDS*THM_TDI_CLIGHT)
#define THM_TDI_FSTAR_HZ \
    (1.0/(2.0*THM_TDI_PI*THM_TDI_LARM_NOMINAL_SECONDS))

typedef struct
{
    int npixels;
    int capacity;
    int *n;
    int *m;
    double *value;
} THMSparseWDMChannel;

typedef struct
{
    THMSparseWDMChannel channel[3];
} THMSparseWDMTriplet;

/* One complex heterodyned time block and its three channel FFTs. */
typedef struct
{
    int nfft;
    double sample_start;
    double sample_dt;
    double heterodyne_frequency;
    double nonzero_start;
    double nonzero_end;
    double frequency_start;
    double frequency_stop;
    int carrier_index;
    int is_endpoint;
    double *fft[3]; /* interleaved complex GSL storage, length 2*nfft */
} THMComplexTDIFFTBlock;

typedef struct
{
    double sample_dt;
    double time_pixel_dt;
    double frequency_pixel_df;
    double meyer_half_bandwidth;
    int time_pixels;
    int frequency_layers;
    int packet_time_half_support;
} THMWDMGridInfo;

/*
 * Reusable state for likelihood and MCMC applications.
 *
 * THMObservationContext is immutable after construction and owns quantities
 * fixed by the observing setup: WDM geometry and the sampled LISA
 * constellation splines.  It may be shared by independent sampler workers.
 * THMWorkerWorkspace owns mutable GSL accelerators and large scratch buffers;
 * create one workspace per concurrent caller and do not share it concurrently.
 * Runtime switches are currently process-global and must remain fixed while
 * contexts are in use; process-based sampler parallelism is the simplest safe
 * configuration until those switches are also moved into the context.
 */
typedef struct THMObservationContext THMObservationContext;
typedef struct THMWorkerWorkspace THMWorkerWorkspace;

/*
 * Batch evaluator for complex polarization carriers.  source_time contains
 * monotonically increasing SSB source times.  The returned convention is
 *
 *   hplus_analytic = hplus - i hplus_quadrature,
 *   hcross_analytic = hcross - i hcross_quadrature.
 *
 * The callback may evaluate a complete multi-harmonic signal; no individual
 * frequency tracks are required by the TDI layer.
 */
typedef int (*THMComplexPolarizationEvaluator)(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic);

/* Family-major extension used to share delay geometry and one intrinsic call. */
typedef int (*THMComplexPolarizationFamilyEvaluator)(
    void *userdata,
    int n,
    const double *source_time,
    int nfamilies,
    double complex *hplus_analytic,
    double complex *hcross_analytic);

/* One-way GW fractional-frequency links for reference comparisons.  Arrays
 * use receiver/emitter order 12, 13, 21, 23, 31, 32.  Positions and LTTs
 * are evaluated at each reception time in the context's TCB coordinate.
 */
int thm_evaluate_complex_eta_context(
    const THMObservationContext *context, THMWorkerWorkspace *workspace,
    int n, const double *detector_time, double ecliptic_latitude,
    double ecliptic_longitude, THMComplexPolarizationEvaluator evaluator,
    void *userdata, double complex *eta[6], double *ltt[6],
    double *ltt_derivative[6]);

/*
 * Coherent positive-frequency X/Y/Z spectra on a common uniform grid.
 * The stored convention is the one-sided real-signal convention used by the
 * fast transform: h(f)=A(f) exp(i phi(f)), with the factor of two already
 * included.  Thus a pure-instrument inner product is a frequency integral of
 * h^dagger C_inst^{-1} h, without an additional factor of four.
 */
typedef struct
{
    int n;
    double df;
    double fmax;
    double blend_center[3];
    double blend_half_width;
    double endpoint_dt;
    double endpoint_start;
    double endpoint_duration;
    int endpoint_samples;
    int endpoint_driver_channel;
    int endpoint_driver_ell;
    int endpoint_driver_abs_m;
    int endpoint_local_spa_planner_used;
    int endpoint_trigger_channel;
    int endpoint_trigger_ell;
    int endpoint_trigger_abs_m;
    double endpoint_join_time;
    double endpoint_rise;
    double endpoint_trigger_frequency;
    double endpoint_trigger_epsilon_f;
    double endpoint_trigger_epsilon_a;
    int direct22_split_spa_used;
    int direct_response_grid_intrinsic_used;
    int fixed_response_grid_used;
    int ap_response_samples;
    int ap_model_samples;
    int ap_interpolated_samples;
    int ap_exact_samples;
    int ap_detector_planned_samples;
    int request_full_fft;
    int has_full_fft;
    double full_fft_tukey_alpha;
    double *re[3];
    double *im[3];
    double *full_re[3];
    double *full_im[3];
} THMFourierTriplet;

void thm_sparse_wdm_triplet_init(THMSparseWDMTriplet *tracks);
void thm_sparse_wdm_triplet_free(THMSparseWDMTriplet *tracks);
int write_thm_sparse_wdm_channel(const char *filename, const THMSparseWDMChannel *track);
void thm_fourier_triplet_init(THMFourierTriplet *spectrum);
void thm_fourier_triplet_free(THMFourierTriplet *spectrum);

/*
 * Set the initial guiding-center orbital phase used by the constellation.
 * Advancing kappa0 by pi/2 is equivalent to advancing the mission epoch by
 * approximately one quarter year.  The galaxy modulation file must be
 * generated with the same value for a physically consistent epoch scan.
 */
void thm_set_orbit_phase(double kappa0_value);
/* Select the Michelson generation captured by subsequently created contexts.
 * Generation 2 uses the SGS/PyTDI X2/Y2/Z2 eta convention, with ordered
 * directed light-travel times and Doppler-shifted frequency delays.
 * The existing generation-1 response remains the default.  Select before
 * creating the context and do not change it while generating waveforms from
 * that context: the adaptive grid planner uses the selected generation too.
 * TDI-2 currently supports the direct complex partitioned-FFT WDM path;
 * Fourier/SPA diagnostics and noise likelihoods still assume TDI-1.
 */
void thm_set_tdi_generation(int generation);
int thm_observation_tdi_generation(const THMObservationContext *context);
/* Enabling this selects the retarded delay chain with output-time
 * projections; it is a diagnostic distinct from the fast TDI-2 default. */
void thm_set_tdi2_frozen_projection(int enabled);
int thm_observation_tdi2_frozen_projection(
    const THMObservationContext *context);

/* Expand every directed delay word about the detector output time using
 * one reference light time and derivative per directed link.  This also
 * freezes each one-link projection at that reference time.  Enabling this
 * selects the combined fast TDI-2 path. */
void thm_set_tdi2_chain_taylor(int enabled);
int thm_observation_tdi2_chain_taylor(
    const THMObservationContext *context);

/* The TDI-2 default combines the Taylor delay chain, reference-time
 * projections, and quadratic one-link light times.  Pass 1 before context
 * creation to select the fully retarded, iterative numerical reference;
 * pass 0 to restore the fast default. */
void thm_set_tdi2_full_numerical(int enabled);

/* Select the one-link solver before context creation.  With the Taylor
 * chain, it is called only for six reference links per output time. */
enum {
    THM_TDI2_LIGHT_TIME_ITERATIVE = 0,
    THM_TDI2_LIGHT_TIME_QUADRATIC = 1,
    THM_TDI2_LIGHT_TIME_TAYLOR2 = 2
};
void thm_set_tdi2_light_time_solver(int solver);
int thm_observation_tdi2_light_time_solver(
    const THMObservationContext *context);

/*
 * Set the folded-carrier order that defines the clean endpoint FFT threshold.
 * The default is 3, matching the historical all-mode THM path; setting this
 * to 2 extends the direct endpoint FFT down to the 22 handoff.
 */
void thm_set_endpoint_min_abs_m(int min_abs_m);
void thm_set_wdm_join_override(int enabled, double join_time, double rise_seconds);
void thm_set_wdm_partition_endpoint(int enabled);
void thm_set_wdm_blend_endpoint(int enabled, double half_width_layers);
/*
 * Evaluate the full delayed TDI response directly at every short endpoint FFT
 * time and use it in place of the sparse post-TDI AP reconstruction.  This is
 * the default because it removes the most delicate merger interpolation while
 * retaining fixed endpoint array sizes and cadence.  Disable it only for
 * regression comparisons; the independent dense-TDI diagnostic is unchanged.
 */
void thm_set_wdm_direct_endpoint_tdi(int enabled);
/*
 * Diagnostic cap on the detector-time AP spacing around intrinsic merger.
 * Passing non-positive values disables it.  This is used to test convergence
 * of the sparse post-TDI endpoint response against dense direct TDI.
 */
void thm_set_merger_grid_cap(double max_step_seconds,
                             double half_width_seconds);
/*
 * Select the no-SPA, multi-region WDM engine for in-memory waveform calls.
 * Each folded carrier is split into bandwidth-limited, even-DF heterodyned
 * early blocks; all carriers share one full-rate endpoint FFT.  This is the
 * production default.  Pass zero to select the faster, less accurate
 * leading-SPA plus endpoint-FFT engine.  The default bandwidth is 0.016 Hz
 * and the default endpoint taper margin is 3 cycles.
 */
void thm_set_wdm_split_early_fft(int enabled, double bandwidth_hz,
                                 double endpoint_margin_cycles);
/* Retain the old channel-by-channel split FFT only as a regression reference. */
void thm_set_wdm_fast_complex_partition(int enabled);
/*
 * Use the sparse post-TDI complex-envelope construction for the early
 * partitioned FFT blocks.  This is the default.  Passing zero restores the
 * older blockwise delayed-TDI evaluation for regression comparisons.
 */
void thm_set_wdm_complex_envelope(int enabled);
/* Expensive untapered full-observation FFT reference for WDM diagnostics. */
void thm_set_wdm_full_fft_reference(int enabled);
/*
 * Enable the first post-adiabatic SPA phase correction built from the stable
 * T1+T2 subset of Delta1.  This affects only the SPA side of the hybrid; the
 * endpoint FFT and its coefficient-space blend are unchanged.
 */
void thm_set_spa_t1t2_correction(int enabled);
/*
 * Fourier-only diagnostic for the single folded 22 carrier.  It evaluates the
 * large intrinsic carrier phase/frequency directly from IMRPhenomT and takes
 * numerical derivatives only of the small post-TDI residual.  The WDM output
 * path is deliberately unchanged.
 */
void thm_set_fourier_direct22_split_spa(int enabled);
/*
 * Fourier/WDM interpolation diagnostic.  Keep the production adaptive
 * response times, but evaluate every intrinsic mode directly at their mapped
 * source times before constructing the splines consumed by fast TDI.  This
 * isolates errors introduced by interpolating from the sparse intrinsic grid.
 */
void thm_set_fourier_direct_response_grid_intrinsic(int enabled);
/*
 * Continuity diagnostic shared by the Fourier and sparse-WDM generators.  The
 * first generated waveform plans and caches its detector-time AP grid; all
 * later waveforms reuse those exact knot times and reevaluate the intrinsic
 * modes after remapping to source time.
 */
void thm_set_fixed_response_grid(int enabled);
/* Backward-compatible name retained for older Fourier diagnostic drivers. */
void thm_set_fourier_fixed_response_grid(int enabled);
/*
 * Write the final pre-TDI intrinsic AP spline knots and a 1 ks dense sampling
 * for the next Fourier waveform call.  Passing NULL or an empty string disables
 * the dump.  This is intended for adaptive-grid continuity diagnostics.
 */
void thm_set_fourier_ap_grid_diagnostic(const char *prefix);
/*
 * Experimental common endpoint planner.  Intrinsic chirp curvature identifies
 * the approach to the endpoint; once it is close to tolerance, post-TDI
 * frequency curvature and signed-amplitude variation may advance the common
 * handoff.  The taper spans leakage_cycles stationary-phase widths.
 */
void thm_set_endpoint_local_spa_planner(int enabled,
                                        double epsilon_f_tolerance,
                                        double epsilon_a_tolerance,
                                        double leakage_cycles);
void thm_set_wdm_instrument_prewhiten(int enabled, int average_halfwidth);
void thm_set_wdm_instrument_prewhiten_smooth_halfwidth(double halfwidth_hz);
void thm_set_wdm_instrument_prewhiten_smooth_fraction(double frac,
                                                      double floor_hz);

/*
 * Build reusable observation/worker state.  kappa0 fixes the LISA orbital
 * phase for this context, independently of later calls to
 * thm_set_orbit_phase().  The current compile-time WDM grid is captured when
 * the context is created.
 */
THMObservationContext *thm_observation_context_create(double kappa0);
/* Time-major positions in metres: position_m[(sample*3+spacecraft)*3+axis].
 * The orbit and source direction must already use the same Cartesian frame. */
THMObservationContext *thm_observation_context_create_from_orbit(
    int samples, const double *time, const double *position_m);
void thm_observation_context_destroy(THMObservationContext *context);
THMWorkerWorkspace *thm_worker_workspace_create(
    const THMObservationContext *context);
void thm_worker_workspace_destroy(THMWorkerWorkspace *workspace);
int thm_observation_wdm_grid_info(const THMObservationContext *context,
                                  THMWDMGridInfo *info);

/*
 * Apply the selected X/Y/Z Michelson generation to a complex analytic source
 * waveform.  Retarded source times are merged, sorted, and deduplicated before
 * each bounded callback batch.  This is the common
 * entry point for the TPHM partitioned-FFT path: a complete precessing signal
 * is evaluated in one batch instead of being unfolded into inertial sidebands.
 *
 * detector_time is in the SSB output/data coordinate used by the WDM grid;
 * source_time passed to the callback includes the spacecraft projection and
 * arm delays.  The output convention is X-iXq, Y-iYq, Z-iZq, matching the
 * positive-phase complex carrier convention used by the existing
 * partitioned-FFT machinery.
 */
int thm_evaluate_complex_tdi_context(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    int n,
    const double *detector_time,
    double ecliptic_latitude,
    double ecliptic_longitude,
    THMComplexPolarizationEvaluator evaluator,
    void *userdata,
    double complex *X,
    double complex *Y,
    double complex *Z);

/* X/Y/Z use family-major storage, output[family*n+sample]. */
int thm_evaluate_complex_tdi_families_context(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    int n,
    const double *detector_time,
    double ecliptic_latitude,
    double ecliptic_longitude,
    int nfamilies,
    THMComplexPolarizationFamilyEvaluator evaluator,
    void *userdata,
    double complex *X,
    double complex *Y,
    double complex *Z);

void thm_complex_tdi_fft_block_init(THMComplexTDIFFTBlock *block);
void thm_complex_tdi_fft_block_free(THMComplexTDIFFTBlock *block);

/*
 * Evaluate, window, heterodyne, and FFT one TDI block.  The rise and fall
 * intervals are independent smooth steps.  Adjacent blocks form a partition
 * of unity when one uses a falling step and the next uses its complementary
 * rising step over the same interval.
 */
int thm_build_complex_tdi_fft_block_context(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    THMComplexPolarizationEvaluator evaluator,
    void *userdata,
    double ecliptic_latitude,
    double ecliptic_longitude,
    int nfft,
    double sample_start,
    double sample_dt,
    double heterodyne_frequency,
    double rise_start,
    double rise_end,
    double fall_start,
    double fall_end,
    THMComplexTDIFFTBlock *block);

/*
 * Efficient variant for planned blocks.  The polarization callback is first
 * evaluated on a uniform source-time lattice with spacing source_sample_dt.
 * Its complex output is demodulated by heterodyne_frequency, interpolated as
 * a slow envelope to the exact TDI retarded times, and remodulated there.
 * This avoids evaluating the intrinsic model separately at all 24 delayed
 * times per detector sample without approximating the delay phase.
 */
int thm_build_complex_tdi_fft_block_interpolated_context(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    THMComplexPolarizationEvaluator evaluator,
    void *userdata,
    double ecliptic_latitude,
    double ecliptic_longitude,
    int nfft,
    double sample_start,
    double sample_dt,
    double source_sample_dt,
    double heterodyne_frequency,
    double rise_start,
    double rise_end,
    double fall_start,
    double fall_end,
    THMComplexTDIFFTBlock *block);

/*
 * Assemble Meyer WDM packets over a rectangular diagnostic support directly
 * from a sum of partitioned complex FFT blocks.  This is the low-level bridge
 * used while the TPHM block planner is being developed; the planner will
 * replace the rectangle by a sparse union following the precessing bands.
 */
int thm_complex_fft_blocks_to_sparse_wdm_context(
    const THMObservationContext *context,
    int nblocks,
    const THMComplexTDIFFTBlock *blocks,
    double time_start,
    double time_stop,
    double frequency_start,
    double frequency_stop,
    THMSparseWDMTriplet *out_tracks);

/*
 * Sparse counterpart driven by the physical band and nonzero time support
 * stored in each block.  Each Meyer layer is transformed only when it
 * overlaps at least one block, and its packet length covers only the relevant
 * time intervals.  An unset frequency interval retains full-band behavior.
 */
int thm_complex_fft_blocks_to_planned_sparse_wdm_context(
    const THMObservationContext *context,
    int nblocks,
    const THMComplexTDIFFTBlock *blocks,
    THMSparseWDMTriplet *out_tracks);

/*
 * Assemble the same block sum on an explicit, common sparse packet plan.
 * nmid[m] and nsize[m] give the center and power-of-two time extent of layer
 * m; an unset layer has nmid < 0 or nsize == 0.  This entry point is used by
 * TPHM, whose precessing carrier bands define a substantially tighter support
 * than the rectangular frequency bounds of its heterodyned FFT blocks.
 */
int thm_complex_fft_blocks_to_sparse_wdm_plan_context(
    const THMObservationContext *context,
    int nblocks,
    const THMComplexTDIFFTBlock *blocks,
    const int *nmid,
    const int *nsize,
    THMSparseWDMTriplet *out_tracks);

/*
 * Persistent-context versions of the in-memory generators.  Repeated calls
 * reuse constellation interpolation state and the large WDM work arrays.
 */
int generate_thm_tdi_wdm_context(const THMObservationContext *context,
                                 THMWorkerWorkspace *workspace,
                                 const double params_in[11],
                                 const char *mode_spec,
                                 THMSparseWDMTriplet *out_tracks);
int generate_thm_tdi_fourier_context(const THMObservationContext *context,
                                     THMWorkerWorkspace *workspace,
                                     const double params_in[11],
                                     const char *mode_spec,
                                     double df_hz,
                                     double fmax_hz,
                                     THMFourierTriplet *out_spectrum);

/*
 * In-memory THM TDI+WDM generator for Fisher/MCMC drivers.
 *
 * params_in:
 *   [0] m1 seconds, [1] m2 seconds, [2] chi1, [3] chi2,
 *   [4] phi0, [5] tc seconds, [6] ln(DL/Gpc),
 *   [7] ecliptic colatitude, [8] ecliptic longitude,
 *   [9] polarization, [10] cos(inclination).
 *
 * out_tracks must be initialized with thm_sparse_wdm_triplet_init().
 * The function reuses/overwrites its buffers and returns 0 on success.
 */
int generate_thm_tdi_wdm(const double params_in[11], const char *mode_spec,
                         THMSparseWDMTriplet *out_tracks);

/*
 * Diagnostic Fourier-domain counterpart to generate_thm_tdi_wdm().  It uses
 * the same adaptive THM and fast-TDI construction, sums the folded-carrier SPA
 * spectra, and blends them into the single combined endpoint FFT in each
 * channel.  This bypasses WDM packet construction and support bookkeeping.
 *
 * df_hz selects the common output spacing.  fmax_hz <= 0 requests the full
 * frequency range supported by the common endpoint FFT.  out_spectrum must be
 * initialized with thm_fourier_triplet_init(); its storage is reused.
 */
int generate_thm_tdi_fourier(const double params_in[11], const char *mode_spec,
                             double df_hz, double fmax_hz,
                             THMFourierTriplet *out_spectrum);

/*
 * Same output format as generate_thm_tdi_wdm(), but the returned sparse WDM
 * samples are central finite-difference derivatives with respect to the
 * caller's coordinate step.  The differencing is done on the smooth per-carrier
 * post-TDI amplitude/phase tracks, then pushed through the fixed base WDM
 * plan.  This is a Fisher-diagnostic path motivated by the amplitude/phase
 * derivative form of Eq. (9) in arXiv:1007.4820.
 */
int generate_thm_tdi_wdm_ap_derivative(const double params0_in[11],
                                       const double params_plus_in[11],
                                       const double params_minus_in[11],
                                       const char *mode_spec,
                                       double coord_step,
                                       THMSparseWDMTriplet *out_tracks);

/*
 * Source-time-only adaptive intrinsic THM sampler.  This deliberately excludes
 * LISA/TDI transfer-zero guards: it is meant to evaluate the expensive
 * IMRPhenomTHM mode amplitudes, phases, and frequencies on a compact intrinsic
 * grid before later interpolation onto a detector/TDI grid.
 */
int PhenomTHM_AP_IntrinsicAdaptive(double *params,
                                   IMRPhenomTHM *model,
                                   IMRPhenomTHMMode *modes,
                                   int *nmodes,
                                   const IMRPhenomTHMMode *requested_modes,
                                   int requested_nmodes,
                                   int Nsmax,
                                   double *TS,
                                   double *tspace,
                                   double **mode_amp,
                                   double **mode_phase,
                                   double **mode_freq,
                                   double observation_tmin,
                                   double observation_tmax);

/*
 * Production AP sampler for the current THM TDI+WDM path.  The expensive THM
 * modes are evaluated on an intrinsic source-time grid.  The response grid uses
 * the 10 ks TDI/orbit cadence while the intrinsic grid is smooth, then switches
 * to the full detector-aware planner near transfer-crossing/merger structure.
 * Sparse intervals are interpolated to the response grid; dense tail response
 * samples are inserted into the model grid and copied through exactly.
 */
int PhenomTHM_AP_IntrinsicTDIAdaptive(double *params,
                                      IMRPhenomTHM *model,
                                      IMRPhenomTHMMode *modes,
                                      int *nmodes,
                                      const IMRPhenomTHMMode *requested_modes,
                                      int requested_nmodes,
                                      int Nsmax,
                                      double *TS,
                                      double *tspace,
                                      double **mode_amp,
                                      double **mode_phase,
                                      double **mode_freq,
                                      gsl_interp_accel **SPacc,
                                      gsl_spline **SPspline,
                                      double constellation_tmin,
                                      double constellation_tmax,
                                      double observation_tmin,
                                      double observation_tmax);

#endif
