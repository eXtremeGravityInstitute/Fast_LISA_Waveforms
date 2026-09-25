/*******************************************************************************************

Copyright (c) 2025, 2026 Neil Cornish
SPDX-License-Identifier: GPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

**********************************************************************************************/

/*
 Fast-WDM diagnostic workflow
 ----------------------------
 PhenomT_TDI writes several WDM products used to check the fast transform:

   wtranfast.dat / BinaryFast.dat
       SPA-based fast WDM diagnostic.  The production THM path now uses the
       complex partitioned FFT; the SPA plus endpoint FFT remains available
       as the lower-latency approximation.

   wtranfft.dat / BinaryFFT.dat
       Same active pixels and local WDM machinery, but using a direct FFT
       spectrum instead of the SPA spectrum. This isolates SPA errors.

   wtranfft_spacut.dat
       Same as wtranfft.dat, but with the FFT packet samples zeroed outside
       the SPA frequency support. This tests low-frequency support-edge
       effects.

   wtranfast_mergerdirect.dat / BinaryFastMergerDirect.dat
       Diagnostic hybrid path. It starts from wtranfast.dat, then replaces the
       taper-safe merger/ringdown frequency layers with WDM coefficients built
       directly from the tapered short merger time segment used by ftran(),
       bypassing the intermediate A(f), phi(f) splines.

   wtranfast_splitfft.dat / BinaryFastSplitFFT.dat
       Experimental split-FFT path. The time-domain waveform is split with a
       partition-of-unity set of tapers. The inspiral side is transformed with
       a long, low-sample-rate FFT; the bend near the chirp turn-up is handled
       by a short intermediate-rate FFT; the merger/ringdown is handled by a
       16-DT full-rate FFT. The WDM grid DT/DF is kept identical to the
       production grid, and the block spectra are added before wdmtranF.

   Xtime.dat
       Raw time-domain X-channel waveform.

   Xtime_tukey.dat
       Same waveform with a Tukey roll-on/off applied. Use this for a more
       physical finite-observation reference, e.g.
           ./wd_viafreq Xtime_tukey.dat 0 meyer

	   PhenomTHM_TDI_dense_check.dat / PhenomTHM_TDI_dense_check_summary.dat
	       THM time-domain TDI diagnostic.  The production-like path evaluates the
	       TDI response only on the adaptive, mode-frequency-controlled grid.  The
	       dense check calls the same delayed-time TDI evaluator on short 1-second
	       chunks, then compares those dense samples to direct interpolation of the
	       sparse channel outputs.  This does not independently test the TDI delay
	       algebra, but it does test whether the sparse response grid is adequate.

	   --split-tdi-response
	       Experimental THM WDM path.  Each folded harmonic is split into the
	       prompt and delayed Michelson-like groups of the TDI-X/Y/Z response,
	       those smoother pieces are passed separately through AP extraction and
	       the SPA+endpoint WDM machinery, and the WDM coefficients are added at
	       the end.  This also selects --no-tdi-delay-zero-grid by default, because
	       the individual pieces should not require dense sampling solely to
	       resolve the full TDI transfer-function nulls.

	   Xchan_AP_notukey.dat
       FFT-vs-SPA amplitude/phase diagnostic for the untapered X channel.
       Column 3 is the phase comparison column. Near TDI response nodes the
       complex amplitude can be effectively zero, making atan2 phase extraction
       numerically meaningless; those bins are flagged in column 10 and column
       3 is interpolated through them. Column 11 keeps the raw direct phase.

   wdm_match_info.dat
       Key/value metadata read by match.c and match_track.c. The match tools
       use it to exclude WDM time pixels whose windows overlap the Tukey
       roll-on/off regions.

 Typical checks:
   ./wd_viafreq Xtime_tukey.dat 0 meyer
   ./match_track track_pixels.dat 8 256 1 4096 wtran.dat wtranfast.dat
   ./match_track track_pixels_fft.dat 8 256 1 4096 wtran.dat wtranfft.dat
 */

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <complex.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_errno.h>

#include "wdm.h"
#include "IMRPhenomTHM.h"
#include "PhenomTHM_TDI.h"

#define TSUN 4.925490947641267e-6
#define GPSEC 1.0292712503794875e17 // Gpc to seconds
#define Tpad 1.0e6
#define PDfref 0.0
#define h22fac  0.31539156525252
#define IRTWO 0.707106781186547549

#define AU THM_TDI_AU_METERS      // Astronomical Unit in meters
#define SECSYR 3.15581498e7
#define SECSDAY 86400.0
#define PARSEC 3.08568025e16    // Parsec in meters
#define CLIGHT THM_TDI_CLIGHT     // Speed of light in m/s
#define sq3 THM_TDI_SQRT3
#define ec THM_TDI_ORBIT_ECCENTRICITY // LISA orbital eccentricity
#define fm 3.168753575e-8  // LISA modulation frequency
#define fstar THM_TDI_FSTAR_HZ // one-radian delay phase scale for the first-order constellation
#define THM_LAV_SECONDS THM_TDI_LARM_NOMINAL_SECONDS
#define THM_LAM_METERS THM_TDI_LARM_NOMINAL_METERS
#define THM_SPOS0 2.25e-22
#define THM_SACC0 9.0e-30

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

#ifndef WDM_USE_GSL_COMPLEX_WAVETABLE
#define WDM_USE_GSL_COMPLEX_WAVETABLE 1
#endif

typedef struct
{
    int Ntx;
    double dfx;
    double scale;
    double *phif;
    double *data;
    double *wdmout;
    double *DX;
#if WDM_USE_GSL_COMPLEX_WAVETABLE
    gsl_fft_complex_wavetable *wavetable;
    gsl_fft_complex_workspace *workspace;
#endif
} WDMPacketFFTPlan;

struct wdmshape
{
    double DT;
    double DF;
    double A;
    double B;
    double OM;
    double DOM;
    double insDOM;
    double FB;
    double DFA;
    double Tobs;
    double Tfilt;
};

typedef struct
{
    int N;
    int nts_capacity;
    int ns_capacity;
    int *nmid;
    int *nsize;
    int *nmid_endpoint;
    int *nsize_endpoint;
    int *listn;
    int *listm;
    double *wdmwave;
    double **wdm;
    double *freq;
    double *phase;
    double *amp;
    double *ataper;
    double *hfull;
} THMReusableWDMWork;

struct THMObservationContext
{
    struct wdmshape wdms;
    double kappa0;
    int tdi_generation;
    int tdi2_frozen_projection;
    int tdi2_chain_taylor;
    int tdi2_light_time_solver;
    double constellation_tmin;
    double constellation_tmax;
    int Nc;
    double **Larray;
    double ***Parray;
    double ***Varray;
    double *tarray;
    gsl_spline *SLspline[3];
    gsl_spline *SPspline[9];
    gsl_spline *SVspline[9];
};

struct THMWorkerWorkspace
{
    const THMObservationContext *context;
    int Nsmax;
    double *TS;
    double *tspace;
    double *tc_tau;
    double **mode_amp;
    double **mode_phase;
    double **mode_freq;
    gsl_interp_accel *SLacc[3];
    gsl_interp_accel *SPacc[9];
    gsl_interp_accel *SVacc[9];
    THMReusableWDMWork wdm_work;
};

typedef enum
{
    THM_AP_INTERVAL_OK = 0,
    THM_AP_FAIL_FREQ_STEP = 1,
    THM_AP_FAIL_DELAY_ZERO = 2,
    THM_AP_FAIL_PHASE_CURVATURE = 3,
    THM_AP_FAIL_REL_CURVATURE = 4,
    THM_AP_FAIL_AMP_STEP = 5,
    THM_AP_FAIL_AMP_CURVATURE = 6
} THMAPIntervalFailReason;

typedef struct
{
    double proposed_step;
    double capped_step;
    double accepted_step;
    double max_omega;
    int dtmax_cap;
    int phase_step_cap;
    int tstop_cap;
    int shrink_count;
    int fail_freq_step;
    int fail_delay_zero;
    int fail_phase_curvature;
    int fail_rel_curvature;
    int fail_amp_step;
    int fail_amp_curvature;
    int last_fail_reason;
    int last_fail_mode;
    int last_fail_probe;
} THMAPStepDiagnostic;

#define dTmax 2.0e5  // maximum time step for TDI extraction
#define THM_DTM_MAX 1.0e4 // tighter shared AP grid for higher-mode TDI/WDM diagnostics
#define dTmin 1.0 // minimum time step for TDI extraction
#define THM_AP_MAX_PHASE_STEP 100.0 // loose phase-step ceiling for the highest active mode on the new AP grid
#define THM_AP_PHASE_CURVATURE_TOL 0.25 // radians, midpoint phase-curvature interval test
#define THM_AP_FREQ_REL_CURVATURE_TOL 2.0e-2 // relative midpoint frequency-curvature interval test
#define THM_AP_FREQ_STEP_REL_TOL 5.0e-2 // maximum endpoint fractional frequency change on one AP interval
#define THM_AP_TRANSFER_FREQ_WINDOW_LOW_REL 3.5e-1 // legacy guard, not used by the detector-adaptive delay-zero test
#define THM_AP_TRANSFER_FREQ_WINDOW_HIGH_REL 1.5e-1 // refine just after crossings without forcing the ringdown plateau
#define THM_AP_TRANSFER_PHASE_STEP 0.1 // target intrinsic phase step near equal-arm TDI delay zeros
#define THM_AP_STEP_GROWTH_LIMIT 1.2 // adjacent accepted AP intervals may grow by at most this factor
#define THM_AP_STEP_SHRINK_FACTOR 0.8 // failed trial intervals are reduced smoothly rather than halved
#define THM_AP_STEP_BRACKET_ITERATIONS 9 // refine passing/failing detector-step brackets to avoid discrete 0.8^n jumps
#define THM_AP_SPLINE_TYPE gsl_interp_akima // shape-preserving AP splines reduce merger/transfer-function ringing
#define THM_AP_WRITE_SPACING_DIAGNOSTIC 1 // write PhenomTHM_AP_spacing_reasons.dat for AP-grid tuning
#define THM_INTRINSIC_DTM_MAX 2.0e5 // maximum source-time step for intrinsic AP waveform sampling
#define THM_INTRINSIC_MAX_PHASE_STEP 100.0 // conservative carrier phase-step cap for spline-safe production phases
#define THM_INTRINSIC_PHASE_CURVATURE_TOL 1.0e-2 // radians, midpoint phase-curvature interval test
#define THM_INTRINSIC_FREQ_REL_CURVATURE_TOL 5.0e-3 // relative midpoint frequency-curvature interval test
#define THM_INTRINSIC_FREQ_STEP_REL_TOL 5.0e-2 // maximum endpoint fractional frequency change on one intrinsic interval
#define THM_INTRINSIC_AMP_STEP_REL_TOL 2.5e-1 // maximum endpoint fractional amplitude change on one intrinsic interval
#define THM_INTRINSIC_AMP_REL_CURVATURE_TOL 2.0e-2 // relative midpoint amplitude-curvature interval test
#define THM_INTRINSIC_STEP_GROWTH_LIMIT 1.5 // intrinsic grid can grow more quickly away from merger
#define THM_INTRINSIC_STEP_SHRINK_FACTOR 0.8
#define THM_INTRINSIC_TDI_SWITCH_TRANSFER_FRACTION 0.5 // begin full TDI-safe planning before the first equal-arm delay-zero family
#define THM_INTRINSIC_TDI_SWITCH_TAIL_SECONDS (2.0*THM_INTRINSIC_DTM_MAX) // fallback detector-planned tail if no transfer crossing is reached
#define THM_INTRINSIC_TDI_SWITCH_GUARD_SECONDS (THM_DTM_MAX+CONSTELLATION_LIGHT_TIME_SECONDS)
#define NPTSPLINE 100000 // coarse diagnostic samples for PTspline.dat
#define WRITE_FFT_SPECTRUM_DIAGNOSTIC 1 // write Xchan_freq_notukey.dat for SPA-vs-FFT checks
/*
 The FFT phase diagnostic treats bins near TDI response nodes as unreliable.
 The threshold must be relative to the local amplitude scale rather than a
 fixed strain value, because source mass, distance, orientation, and TDI
 geometry can change the absolute amplitude by many orders of magnitude.
 */
#define FFT_DIAGNOSTIC_NODE_FLOOR_FRACTION 5.0e-2
#define SHORTFFT_MERGER_TAPER_MARGIN_SECONDS 0.0 // extra time after the short-FFT roll-on before using direct merger WDM layers
#define SPLIT_FFT_ROLL_SECONDS 1.0e5 // time-domain roll length used between adjacent split-FFT blocks
#define SPLIT_FFT_LOW_DECIMATION 128 // dt_low = SPLIT_FFT_LOW_DECIMATION*dt, with Nf_low=Nf/R
#define SPLIT_FFT_BEND_DECIMATION 32 // intermediate bend block cadence; R=32 covers higher-mode inspiral before the final full-rate block
#define SPLIT_FFT_BEND_NTILES 64 // conservative bend-block duration in WDM time pixels
#define SPLIT_FFT_HIGH_NTILES 16 // full-rate merger/ringdown block duration in WDM time pixels
#define SPLIT_FFT_PLAN_MAX_BANDWIDTHS 16
#define SPLIT_FFT_PLAN_MIN_SHIFTED_LAYER 2 // keep the heterodyned band away from the special DC layer
#define SPLIT_FFT_PLAN_NYQUIST_GUARD_LAYERS 2
#define SPLIT_FFT_ENDPOINT_MARGIN_CYCLES_DEFAULT 3.0 // endpoint taper sideband margin in units of 1/rise
#define SPLIT_FFT_ENDPOINT_MARGIN_FRACTION_MAX 0.5 // never extend endpoint taper margin by more than this fraction of f_endpoint
#ifndef THM_PARTITION_TAPER_BAND_CYCLES
#define THM_PARTITION_TAPER_BAND_CYCLES 3.0
#endif
#define REFERENCE_TUKEY_ROLL_SECONDS 1.0e5 // roll-on/off length for Xtime_tukey.dat
#define WAVEFORM_PRE_PADDING_SECONDS 0.0 // waveform support before the observation start
#define SPA_T1T2_MAX_PHASE_CORRECTION 0.25 // radians; outside this range the post-adiabatic series is not perturbative
#define CONSTELLATION_LIGHT_TIME_SECONDS 500.0 // one AU light travel time, rounded up
#define CONSTELLATION_MIN_PADDING_SECONDS CONSTELLATION_LIGHT_TIME_SECONDS
/*
 * A TDI-1 link samples h(t-k.r-Delta L).  The sky-dependent k.r term can
 * advance or retard a feature by almost one AU light-travel time, while the
 * longest path in TDI_spline_thm_piece() has Delta L = 2 L_b + 2 L_c.
 * Consequently the early response margin is 500 s, whereas the late margin
 * also includes four nominal LISA arm light times.
 */
#define THM_TDI_ARM_DELAY_SAFETY_FACTOR 1.02
#define THM_TDI_MAX_PATH_DELAY_SECONDS \
    (4.0*THM_TDI_ARM_DELAY_SAFETY_FACTOR*THM_LAV_SECONDS)
#define THM_RESPONSE_EARLY_MARGIN_SECONDS CONSTELLATION_LIGHT_TIME_SECONDS
#define THM_RESPONSE_LATE_MARGIN_SECONDS \
    (CONSTELLATION_LIGHT_TIME_SECONDS+THM_TDI_MAX_PATH_DELAY_SECONDS)
#define THM_TDI_DENSE_CHECK_DT_SECONDS 1.0 // high-cadence reference chunks
#define THM_TDI_DENSE_CHECK_CHUNK_SECONDS 16384.0 // 2^14 seconds per diagnostic chunk
#define THM_TDI_DENSE_CHECK_EARLY_FRACTION 0.25 // place early chunk this far from start to merger
#ifndef THM_WDM_TIMESCAN_PAD_PIXELS
#define THM_WDM_TIMESCAN_PAD_PIXELS 0
#endif
#ifndef THM_WDM_TIMESCAN_PAD_LAYERS
#define THM_WDM_TIMESCAN_PAD_LAYERS 0
#endif
/*
 * Fold each aligned-spin +m/-m pair into one carrier for the TDI response.
 * This is the fast default for THM.  Set to 0 when developing TPHM/twist-up
 * logic, where the explicit mode basis is often the safer object to inspect.
 */
#ifndef THM_TDI_USE_FOLDED_CARRIERS
#define THM_TDI_USE_FOLDED_CARRIERS 1
#endif

#define THM_TDI_PIECE_FULL 0
#define THM_TDI_PIECE_DELAYED_MICHELSON 1
#define THM_TDI_PIECE_PROMPT_MICHELSON 2

/* Initial orientation of the LISA constellation */
#define lambda0 0.0

// Compile with
// gcc -I/opt/homebrew/include -L/opt/homebrew/lib -o PhenomTHM_TDI PhenomTHM_TDI.c IMRPhenomTHM.c IMRPhenomT.c -lgsl -lgslcblas -lm
// For a Fisher driver, compile this file without its executable main:
// gcc -DPHENOMTHM_TDI_LIBRARY -I/opt/homebrew/include -c PhenomTHM_TDI.c

typedef struct
{
    int ell;
    int emm;
    double plus_cos;
    double plus_sin;
    double cross_cos;
    double cross_sin;
} THMProjection;

typedef struct
{
    int ell;
    int emm;
    int mode_index;
    int paired_mode_index;
    int is_pair;
    double hp_cos;
    double hp_sin;
    double hc_cos;
    double hc_sin;
    double hpf_cos;
    double hpf_sin;
    double hcf_cos;
    double hcf_sin;
} THMFoldedCarrier;

typedef struct
{
    const char *mode_spec;
    int tdi_generation;
    int tdi2_frozen_projection;
    int tdi2_chain_taylor;
    int tdi2_exact;
    int tdi2_custom_option;
    int tdi2_light_time_solver;
    int show_help;
    int invalid;
    int wdm_mode_enabled;
    int wdm_all_enabled;
    int split_tdi_response;
    int no_tdi_delay_zero_grid;
    int wdm_ell;
    int wdm_abs_emm;
    int wdm_join_time_enabled;
    double wdm_join_time;
    double wdm_rise_seconds;
    int compare_ap_grid;
    int compare_per_mode_grid;
    int use_intrinsic_tdi_ap_grid;
    int use_detector_ap_grid;
    int wdm_spline_endpoint;
    int wdm_partition_endpoint;
    int wdm_blend_endpoint;
    int direct_endpoint_tdi;
    double wdm_blend_half_width_layers;
    int spa_t1t2_correction;
    int prewhiten_instrument;
    int prewhiten_average_halfwidth;
    double prewhiten_smooth_halfwidth_hz;
    double prewhiten_smooth_frac;
    double prewhiten_smooth_floor_hz;
    int wdm_split_fft;
    int wdm_split_early_fft;
    int wdm_spa_endpoint_fft;
    int blockwise_sparse_tdi;
    int wdm_split_plan;
    int wdm_split_plan_nband;
    double wdm_split_plan_bandwidth[SPLIT_FFT_PLAN_MAX_BANDWIDTHS];
    double wdm_split_endpoint_margin_cycles;
    int wdm_endpoint_min_abs_m;
    int diagnostics_enabled;
    int write_time_domain_files;
    int write_sparse_files;
    int write_intrinsic_ap_grid;
    int tc_override_enabled;
    double tc_override;
    int kappa0_override_enabled;
    double kappa0_override;
} THMRunOptions;

typedef struct
{
    int tile_lo;
    int tile_hi;
    int support_lo;
    int support_hi;
    int R;
    int m_shift;
    int shifted_lo;
    int shifted_hi;
    int local_bins;
    int Nfft;
    int Kfft;
    double sample_dt;
    double shift_frequency;
    double block_start;
    double nonzero_start;
    double nonzero_end;
    double fmin;
    double fmax;
    double fpad;
    double build_time;
    double *hfft;
} THMEarlySplitFFTBlock;

/*
 * Production partitioned-FFT blocks use the same complex, three-channel
 * construction as the TPHM path.  The older THMEarlySplitFFTBlock machinery
 * is retained below for diagnostics, but it rebuilds packets channel by
 * channel and is much slower on year-long observations.
 */
typedef struct
{
    int carrier_index;
    int nfft;
    double sample_start;
    double sample_dt;
    double heterodyne_frequency;
    double rise_start;
    double rise_end;
    double fall_start;
    double fall_end;
    double frequency_start;
    double frequency_stop;
} THMFastPartitionBlockSpec;

typedef struct
{
    int nblocks;
    int capacity;
    long long total_fft_samples;
    THMComplexTDIFFTBlock *blocks;
} THMFastPartitionPlan;

typedef struct
{
    int ncarriers;
    int selected_carrier;
    const THMFoldedCarrier *carrier;
    gsl_interp_accel **Aacc;
    gsl_spline **Aspline;
    gsl_interp_accel **Pacc;
    gsl_spline **Pspline;
    double time_start;
    double time_stop;
} THMFastPartitionSource;

typedef struct
{
    int n;
    int ncarriers;
    double *time;
    gsl_spline **real_spline;
    gsl_spline **imag_spline;
    gsl_interp_accel **real_acc;
    gsl_interp_accel **imag_acc;
} THMFastResponseEnvelopeCache;

typedef struct
{
    double params[11];
    int Ns;
    int nmodes;
    int ncarriers;
    IMRPhenomTHM model;
    IMRPhenomTHMMode modes[IMRPHENOMTHM_MAX_MODES];
    THMProjection projection[IMRPHENOMTHM_MAX_MODES];
    THMFoldedCarrier carrier[IMRPHENOMTHM_MAX_MODES];
    double *TS;
    double *tspace;
    double **mode_amp;
    double **mode_phase;
    double **mode_freq;
    gsl_interp_accel **Aacc;
    gsl_interp_accel **Pacc;
    gsl_spline **Aspline;
    gsl_spline **Pspline;
} THMAPContext;

typedef struct
{
    int valid;
    int trigger_channel;
    int trigger_stream;
    double join_time;
    double rise;
    double trigger_frequency;
    double trigger_epsilon_f;
    double trigger_epsilon_a;
    double max_stationary_width;
} THMLocalSPAEndpointPlan;

static double thm_response_spline_tmin = 0.0;
static double thm_response_spline_tmax = 0.0;
static const char *thm_ap_spacing_diagnostic_filename = "PhenomTHM_AP_spacing_reasons.dat";
static const char *thm_intrinsic_ap_spacing_diagnostic_filename = "PhenomTHM_intrinsic_AP_spacing_reasons.dat";
static const char *thm_intrinsic_tdi_ap_spacing_diagnostic_filename = "PhenomTHM_intrinsic_TDI_AP_spacing_reasons.dat";
static int thm_diagnostics_enabled = 0;
static int thm_summary_output_enabled = 1;
static int thm_split_fft_plan_enabled = 0;
static int thm_split_fft_plan_nband = 0;
static double thm_split_fft_plan_bandwidth[SPLIT_FFT_PLAN_MAX_BANDWIDTHS];
static double thm_split_fft_endpoint_margin_cycles =
    SPLIT_FFT_ENDPOINT_MARGIN_CYCLES_DEFAULT;
static int thm_endpoint_min_abs_m = 3;
static int thm_wdm_join_override_enabled = 0;
static double thm_wdm_join_override_time = 0.0;
static double thm_wdm_join_override_rise = 0.0;
static int thm_wdm_partition_endpoint_enabled = 0;
static int thm_wdm_blend_endpoint_enabled = 0;
static double thm_wdm_blend_half_width_layers = 1.0;
/*
 * The merger/ringdown endpoint is short enough that the cleanest construction
 * is to evaluate the full delayed TDI response directly at every endpoint FFT
 * sample.  This avoids a second interpolation through the rapidly varying,
 * sparsely sampled post-TDI amplitude and phase.  The setter can still disable
 * this path for regression comparisons with the historical AP reconstruction.
 */
static int thm_wdm_direct_endpoint_tdi_enabled = 1;
static double thm_merger_grid_max_step_seconds = 0.0;
static double thm_merger_grid_half_width_seconds = 0.0;
/*
 * The partitioned complex-FFT engine is the production default.  It removes
 * the leading-SPA accuracy floor at only a modest cost.  Passing zero to
 * thm_set_wdm_split_early_fft() restores the faster SPA plus endpoint FFT.
 */
static int thm_wdm_split_early_fft_enabled = 1;
static double thm_wdm_split_early_fft_bandwidth_hz = 0.016;
static int thm_wdm_fast_complex_partition_enabled = 1;
/* Build one sparse post-TDI complex response envelope per folded carrier and
 * channel, then remodulate it by the intrinsic carrier on each FFT lattice.
 * Disable only to reproduce the older blockwise delayed-response path. */
static int thm_wdm_complex_envelope_enabled = 1;
static int thm_tdi_generation = 1;
static int thm_tdi2_frozen_projection = 0;
static int thm_tdi2_chain_taylor = 1;
static int thm_tdi2_light_time_solver = THM_TDI2_LIGHT_TIME_QUADRATIC;
static int thm_wdm_full_fft_reference_enabled = 0;
static int thm_spa_t1t2_correction_enabled = 0;
static int thm_fourier_direct22_split_spa_enabled = 0;
static int thm_fourier_direct_response_grid_intrinsic_enabled = 0;
#define THM_FOURIER_FIXED_RESPONSE_GRID_CAPACITY 10000
static int thm_fourier_fixed_response_grid_enabled = 0;
static int thm_fourier_fixed_response_grid_samples = 0;
static int thm_fourier_fixed_response_output_samples = 0;
static int thm_fourier_fixed_response_detector_planned_samples = 0;
static double thm_fourier_fixed_response_grid[
    THM_FOURIER_FIXED_RESPONSE_GRID_CAPACITY];
static char thm_fourier_ap_grid_diagnostic_prefix[512] = "";
static int thm_endpoint_local_spa_planner_enabled = 0;
static double thm_endpoint_local_spa_epsilon_f_tolerance = 1.0;
static double thm_endpoint_local_spa_epsilon_a_tolerance = 1.0;
static double thm_endpoint_local_spa_leakage_cycles = 2.0;
static int thm_wdm_instrument_prewhiten_enabled = 0;
static int thm_wdm_instrument_prewhiten_average_halfwidth = 2;
static double thm_wdm_instrument_prewhiten_smooth_halfwidth_hz = 1.0e-5;
static double thm_wdm_instrument_prewhiten_smooth_frac = 0.0;
static double thm_wdm_instrument_prewhiten_smooth_floor_hz = 1.0e-5;
static int thm_intrinsic_tdi_include_delay_zero_grid = 1;
/* Runtime initial azimuthal position of the guiding center. */
static double thm_orbit_phase_offset = 0.0;
static int thm_last_intrinsic_tdi_model_samples = 0;
static int thm_last_intrinsic_tdi_interpolated_samples = 0;
static int thm_last_intrinsic_tdi_exact_samples = 0;
static int thm_last_intrinsic_tdi_detector_planned_samples = 0;
static double thm_last_intrinsic_tdi_switch_source_time = 0.0;
static double thm_last_intrinsic_tdi_switch_detector_time = 0.0;

static int thm_near_delay_zero(double omega, double omstar);
static int thm_delay_zero_interval_reason(double f0, double fmid, double f1, double width, double fmode_max);
static double thm_ssb_output_time_from_center_source_time(
    double center_source_time, double *params,
    gsl_interp_accel **SPacc, gsl_spline **SPspline);
static double thm_reference_phase_eval(const gsl_spline *spline,
                                       double time,
                                       gsl_interp_accel *acc);

/*
 * extractAP() removes a carrier evaluated at detector/reference time, while
 * the intrinsic AP spline knots carry barycentric source-time labels.  Their
 * endpoints can differ by up to one AU light-travel time.  Continue the phase
 * linearly over that small endpoint interval instead of clamping it or asking
 * GSL to extrapolate.  The reference phase only guides branch tracking and is
 * added back before the SPA derivatives are formed.
 */
static double thm_reference_phase_eval(const gsl_spline *spline,
                                       double time,
                                       gsl_interp_accel *acc)
{
    static int warned = 0;
    size_t i0, i1;
    double x0, x1, y0, y1, extension;

    if(spline == NULL || spline->size < 2 || acc == NULL) return NAN;
    if(time >= spline->x[0] && time <= spline->x[spline->size-1])
    {
        return gsl_spline_eval(spline, time, acc);
    }

    if(time < spline->x[0])
    {
        i0 = 0;
        i1 = 1;
        extension = spline->x[0]-time;
    }
    else
    {
        i0 = spline->size-2;
        i1 = spline->size-1;
        extension = time-spline->x[spline->size-1];
    }
    x0 = spline->x[i0];
    x1 = spline->x[i1];
    y0 = spline->y[i0];
    y1 = spline->y[i1];
    if(x1 <= x0) return NAN;

    if(extension > CONSTELLATION_LIGHT_TIME_SECONDS+1.0 && !warned)
    {
        fprintf(stderr,
                "Warning: detector-time carrier phase extends %.6e s beyond "
                "the barycentric AP spline, exceeding the nominal %.6e s "
                "light-time margin.\n",
                extension, (double)CONSTELLATION_LIGHT_TIME_SECONDS);
        warned = 1;
    }
    return y0+(time-x0)*(y1-y0)/(x1-x0);
}

void thm_set_orbit_phase(double kappa0_value)
{
    if(!isfinite(kappa0_value)) kappa0_value = 0.0;
    thm_orbit_phase_offset = fmod(kappa0_value, 2.0*M_PI);
    if(thm_orbit_phase_offset < 0.0) thm_orbit_phase_offset += 2.0*M_PI;
}

void thm_set_tdi_generation(int generation)
{
    if(generation == 1 || generation == 2)
        thm_tdi_generation = generation;
}

int thm_observation_tdi_generation(const THMObservationContext *context)
{
    return context != NULL ? context->tdi_generation : 0;
}

void thm_set_tdi2_frozen_projection(int enabled)
{
    thm_tdi2_frozen_projection = enabled != 0;
    if(enabled) thm_tdi2_chain_taylor = 0;
}

int thm_observation_tdi2_frozen_projection(
    const THMObservationContext *context)
{
    return context != NULL ? context->tdi2_frozen_projection : 0;
}

void thm_set_tdi2_chain_taylor(int enabled)
{
    thm_tdi2_chain_taylor = enabled != 0;
    if(enabled) thm_tdi2_frozen_projection = 0;
}

int thm_observation_tdi2_chain_taylor(
    const THMObservationContext *context)
{
    return context != NULL ? context->tdi2_chain_taylor : 0;
}

void thm_set_tdi2_full_numerical(int enabled)
{
    thm_tdi2_chain_taylor = enabled == 0;
    thm_tdi2_frozen_projection = 0;
    thm_tdi2_light_time_solver = enabled ?
        THM_TDI2_LIGHT_TIME_ITERATIVE : THM_TDI2_LIGHT_TIME_QUADRATIC;
}

void thm_set_tdi2_light_time_solver(int solver)
{
    if(solver >= THM_TDI2_LIGHT_TIME_ITERATIVE &&
       solver <= THM_TDI2_LIGHT_TIME_TAYLOR2)
        thm_tdi2_light_time_solver = solver;
}

int thm_observation_tdi2_light_time_solver(
    const THMObservationContext *context)
{
    return context != NULL ? context->tdi2_light_time_solver : -1;
}

void thm_set_endpoint_min_abs_m(int min_abs_m)
{
    if(min_abs_m < 1) min_abs_m = 1;
    if(min_abs_m > 5) min_abs_m = 5;
    thm_endpoint_min_abs_m = min_abs_m;
}

void thm_set_wdm_join_override(int enabled, double join_time, double rise_seconds)
{
    if(enabled && isfinite(join_time) && join_time > 0.0)
    {
        thm_wdm_join_override_enabled = 1;
        thm_wdm_join_override_time = join_time;
        thm_wdm_join_override_rise =
            (isfinite(rise_seconds) && rise_seconds > 0.0) ? rise_seconds : 0.0;
    }
    else
    {
        thm_wdm_join_override_enabled = 0;
        thm_wdm_join_override_time = 0.0;
        thm_wdm_join_override_rise = 0.0;
    }
}

void thm_set_wdm_partition_endpoint(int enabled)
{
    thm_wdm_partition_endpoint_enabled = enabled ? 1 : 0;
}

void thm_set_wdm_blend_endpoint(int enabled, double half_width_layers)
{
    thm_wdm_blend_endpoint_enabled = enabled ? 1 : 0;
    if(isfinite(half_width_layers) && half_width_layers > 0.0)
    {
        thm_wdm_blend_half_width_layers = half_width_layers;
    }
}

void thm_set_wdm_direct_endpoint_tdi(int enabled)
{
    thm_wdm_direct_endpoint_tdi_enabled = enabled ? 1 : 0;
}

void thm_set_merger_grid_cap(double max_step_seconds,
                             double half_width_seconds)
{
    if(isfinite(max_step_seconds) && max_step_seconds > 0.0 &&
       isfinite(half_width_seconds) && half_width_seconds > 0.0)
    {
        thm_merger_grid_max_step_seconds = max_step_seconds;
        thm_merger_grid_half_width_seconds = half_width_seconds;
    }
    else
    {
        thm_merger_grid_max_step_seconds = 0.0;
        thm_merger_grid_half_width_seconds = 0.0;
    }
}

void thm_set_wdm_split_early_fft(int enabled, double bandwidth_hz,
                                 double endpoint_margin_cycles)
{
    thm_wdm_split_early_fft_enabled = enabled ? 1 : 0;
    if(isfinite(bandwidth_hz) && bandwidth_hz > 0.0)
    {
        thm_wdm_split_early_fft_bandwidth_hz = bandwidth_hz;
    }
    if(isfinite(endpoint_margin_cycles) && endpoint_margin_cycles >= 0.0)
    {
        thm_split_fft_endpoint_margin_cycles = endpoint_margin_cycles;
    }
}

void thm_set_wdm_fast_complex_partition(int enabled)
{
    thm_wdm_fast_complex_partition_enabled = enabled ? 1 : 0;
}

void thm_set_wdm_complex_envelope(int enabled)
{
    thm_wdm_complex_envelope_enabled = enabled ? 1 : 0;
}

void thm_set_wdm_full_fft_reference(int enabled)
{
    thm_wdm_full_fft_reference_enabled = enabled ? 1 : 0;
}

void thm_set_spa_t1t2_correction(int enabled)
{
    thm_spa_t1t2_correction_enabled = enabled ? 1 : 0;
}

void thm_set_fourier_direct22_split_spa(int enabled)
{
    thm_fourier_direct22_split_spa_enabled = enabled ? 1 : 0;
}

void thm_set_fourier_direct_response_grid_intrinsic(int enabled)
{
    thm_fourier_direct_response_grid_intrinsic_enabled = enabled ? 1 : 0;
}

void thm_set_fixed_response_grid(int enabled)
{
    thm_fourier_fixed_response_grid_enabled = enabled ? 1 : 0;
    thm_fourier_fixed_response_grid_samples = 0;
    thm_fourier_fixed_response_output_samples = 0;
    thm_fourier_fixed_response_detector_planned_samples = 0;
}

void thm_set_fourier_fixed_response_grid(int enabled)
{
    thm_set_fixed_response_grid(enabled);
}

void thm_set_fourier_ap_grid_diagnostic(const char *prefix)
{
    if(prefix == NULL || prefix[0] == '\0')
    {
        thm_fourier_ap_grid_diagnostic_prefix[0] = '\0';
        return;
    }
    snprintf(thm_fourier_ap_grid_diagnostic_prefix,
             sizeof(thm_fourier_ap_grid_diagnostic_prefix), "%s", prefix);
}

void thm_set_endpoint_local_spa_planner(int enabled,
                                        double epsilon_f_tolerance,
                                        double epsilon_a_tolerance,
                                        double leakage_cycles)
{
    thm_endpoint_local_spa_planner_enabled = enabled ? 1 : 0;
    if(isfinite(epsilon_f_tolerance) && epsilon_f_tolerance > 0.0)
    {
        thm_endpoint_local_spa_epsilon_f_tolerance = epsilon_f_tolerance;
    }
    if(isfinite(epsilon_a_tolerance) && epsilon_a_tolerance > 0.0)
    {
        thm_endpoint_local_spa_epsilon_a_tolerance = epsilon_a_tolerance;
    }
    if(isfinite(leakage_cycles) && leakage_cycles > 0.0)
    {
        thm_endpoint_local_spa_leakage_cycles = leakage_cycles;
    }
}

void thm_set_wdm_instrument_prewhiten(int enabled, int average_halfwidth)
{
    thm_wdm_instrument_prewhiten_enabled = enabled ? 1 : 0;
    if(average_halfwidth < 0) average_halfwidth = 0;
    thm_wdm_instrument_prewhiten_average_halfwidth = average_halfwidth;
}

void thm_set_wdm_instrument_prewhiten_smooth_halfwidth(double halfwidth_hz)
{
    if(isfinite(halfwidth_hz) && halfwidth_hz >= 0.0)
    {
        thm_wdm_instrument_prewhiten_smooth_halfwidth_hz = halfwidth_hz;
    }
}

void thm_set_wdm_instrument_prewhiten_smooth_fraction(double frac,
                                                      double floor_hz)
{
    if(isfinite(frac) && frac >= 0.0)
    {
        thm_wdm_instrument_prewhiten_smooth_frac = frac;
    }
    if(isfinite(floor_hz) && floor_hz >= 0.0)
    {
        thm_wdm_instrument_prewhiten_smooth_floor_hz = floor_hz;
    }
}

void wdmvalues(struct wdmshape *wdms);
void tukey(double *data, double alpha, int N);
void spacecraft(double t, double *x, double *y, double *z);
void spacraft_loc(int i, double *x, double *y, double *z, double sa, double ca, double sb, double cb);
void constellation(int Ns, double *tarray, double **Larray, double ***Parray, double ***Varray);
void detector_time(double *tarray, double *tspace, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int N);
void barycenter_time(double *tarray, double *tspace, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int N);
void hphc(double t, gsl_interp_accel *ATacc, gsl_spline *ATspline, gsl_interp_accel *PTacc, gsl_spline *PTspline, double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf);
void TDI_spline(double *M, double *Mf, int a, int b, int c, double* tarray, int n, gsl_interp_accel *ATacc, gsl_spline *ATspline, gsl_interp_accel *PTacc, gsl_spline *PTspline, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
void fast_response(double *tarray, int N, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel *ATacc, gsl_spline *ATspline, gsl_interp_accel *PTacc, gsl_spline *PTspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf);
int PhenomTHM_AP(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax);
int PhenomTHM_AP_DetectorAdaptive(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax, double observation_tmin, double observation_tmax);
int PhenomTHM_AP_IntrinsicAdaptive(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, double observation_tmin, double observation_tmax);
int PhenomTHM_AP_IntrinsicTDIAdaptive(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax, double observation_tmin, double observation_tmax);
int PhenomTHM_AP_OnDetectorGrid(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Ns, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline);
void thm_projection_coefficients(int ell, int emm, double cosi, double orbital_azimuth, double psi, THMProjection *projection);
int thm_build_folded_carriers(int nmodes, const IMRPhenomTHMMode *modes, const THMProjection *projection, THMFoldedCarrier *carrier, int max_carriers);
void hphc_thm(double t, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *hp, double *hc, double *hpf, double *hcf);
void TDI_spline_thm(double *M, double *Mf, int a, int b, int c, double* tarray, int n, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
void TDI_spline_thm_piece(double *M, double *Mf, int a, int b, int c, double* tarray, int n, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm, int piece);
void fast_response_thm(double *tarray, int N, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf);
void fast_response_thm_piece(double *tarray, int N, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf, int piece);
void write_thm_mode_ap(const char *filename, int Ns, int nmodes, IMRPhenomTHMMode *modes, double *TS, double *tc_tau, double **mode_amp, double **mode_phase, double **mode_freq);
void write_thm_tdi_time(const char *filename, int Ns, double *TS, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf);
void write_thm_tdi_dense_check(const char *data_filename, const char *summary_filename, double dense_dt, double chunk_seconds, double tc, int Ns_sparse, double *TS_sparse, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline);
void direct_response_full_waveform_thm(double *tarray, int N, double dense_dt, double delay_pad, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *X, double *Y, double *Z);
void write_thm_single_carrier_wdm_diagnostic(int carrier_index, int Ns, double *response_time, double *plan_time, double **mode_freq, double *params, const IMRPhenomTHM *model, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, struct wdmshape *wdms, int use_join_override, double join_time_override, double rise_override);
void thm_sparse_wdm_triplet_init(THMSparseWDMTriplet *tracks);
void thm_sparse_wdm_triplet_free(THMSparseWDMTriplet *tracks);
int write_thm_sparse_wdm_channel(const char *filename, const THMSparseWDMChannel *track);
int generate_thm_tdi_wdm(const double params_in[11], const char *mode_spec, THMSparseWDMTriplet *out_tracks);
void thm_fourier_triplet_init(THMFourierTriplet *spectrum);
void thm_fourier_triplet_free(THMFourierTriplet *spectrum);
int generate_thm_tdi_fourier(const double params_in[11], const char *mode_spec, double df_hz, double fmax_hz, THMFourierTriplet *out_spectrum);
int generate_thm_tdi_wdm_ap_derivative(const double params0_in[11], const double params_plus_in[11], const double params_minus_in[11], const char *mode_spec, double coord_step, THMSparseWDMTriplet *out_tracks);
static int count_wdm_nonzero_pixels(double **wdm);
static int thm_sparse_wdm_channel_from_dense(THMSparseWDMChannel *track, double **wdm);
static int thm_sparse_wdm_channel_reserve(THMSparseWDMChannel *track, int capacity);
static int thm_fourier_triplet_reserve(THMFourierTriplet *spectrum, int n);
static void thm_ap_context_init(THMAPContext *ctx);
static void thm_ap_context_free(THMAPContext *ctx);
static int thm_ap_context_alloc(THMAPContext *ctx, int Nsmax);
static int thm_ap_context_build_splines_and_carriers(THMAPContext *ctx);
static double thm_unwrapped_phase_difference(double raw, double previous, int have_previous);
static int thm_keep_increasing_frequency_samples(int n, double *freq, double *phase, double *amp);
static int thm_parse_double_list(const char *text, double *values, int max_values, int *nvalues);
static int thm_next_power_of_two_int(int n);
static int thm_largest_power_of_two_leq_int(int n);
static int thm_first_upward_crossing_time(int n, const double *time,
                                          const double *frequency,
                                          double threshold,
                                          double stop_time,
                                          double *crossing_time);
static void thm_adjust_endpoint_taper_flat_time(double setup[7],
                                                 double target_flat_time);
static void thm_set_default_split_fft_plan_bandwidths(const struct wdmshape *wdms, THMRunOptions *options);
void generate_thm_all_carrier_wdm_combined_fft(int Ns, double *response_time, double *plan_time, double **mode_freq, double *params, const IMRPhenomTHM *model, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, struct wdmshape *wdms, int use_join_override, double join_time_override, double rise_override, int use_spline_endpoint, int use_partition_endpoint, int use_split_fft, int use_split_early_fft, int use_split_tdi_response, THMSparseWDMTriplet *out_tracks, THMFourierTriplet *out_fourier, double fourier_df, double fourier_fmax, int write_sparse_files, int write_time_domain_files, THMReusableWDMWork *reusable_work, const THMObservationContext *observation_context, THMWorkerWorkspace *worker_workspace);
void write_thm_all_carrier_wdm_combined_fft(int Ns, double *response_time, double *plan_time, double **mode_freq, double *params, const IMRPhenomTHM *model, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, struct wdmshape *wdms, int use_join_override, double join_time_override, double rise_override);
int find_thm_carrier(int ncarriers, const THMFoldedCarrier *carrier, int ell, int abs_emm);
void extractAP(int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
void unwrap(int Ns, double *phi);
double nonuniform_phase_derivative(int i, int N, const double *t, const double *phase);
void build_nonuniform_phase_derivatives(int N, double *t, gsl_interp_accel *Pacc, gsl_spline *Pspline, double *phase_grid, double *freq_grid, double *fdot_grid);
static int thm_build_local_spa_endpoint_plan(int N, const double *response_time,
                                             int nstreams,
                                             const THMFoldedCarrier *carrier,
                                             double **mode_freq,
                                             double ***Achan,
                                             double ***freq_track,
                                             double tc_output,
                                             THMLocalSPAEndpointPlan *plan);
void build_split_reference_frequency_track(int N, double *response_time, double *plan_time, double *mode_freq, double *phase_correction, double *phase_total, double *freq_ref_track, double *plan_jacobian, double *freq_correction, double *freq_split, double *fdot_split, double *freq_fullphase, double *fdot_fullphase);
int PhenomT_AP(double *params, int Nsmax, double *TS, double *tspace, double *AS, double *PS, double *FS, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax, double *setup);
static void enforce_phenom_mass_hierarchy(double *m1, double *m2, double *chi1, double *chi2);

/* tc_output and t22_output must already be mapped to SSB output time. */
void transformplan(double Mc, double Mtot, double tc_output, double *response_time, double *omegat, double t22_output, double fring, double fdamp, double tmax, double *setup);
void transformplan_custom(double Mc, double Mtot, double tc_output, double *response_time, double *omegat, double t22_output, double fring, double fdamp, double tmax, double *setup, int use_join_override, double join_time_override, double rise_override);

int ftran(double *setup, double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *freq, double *phase, double *Amp, double *short_htime);
int ftran_spa_only(double *setup, double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *freq, double *phase, double *Amp);
int ftran_endpoint_fft(double *setup, double Tobs, double *short_htime, double f_start, double *freq, double *phase, double *Amp);
int build_full_spa_samples(double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *freq, double *phase, double *Amp, double *tstat, double *fdot, int *valid_single_branch);
int build_full_spa_samples_from_tracks(double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, const double *freq_grid, const double *fdot_grid, double *freq, double *phase, double *Amp, double *tstat, double *fdot, int *valid_single_branch);
int build_first_order_spa_samples(double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double correction_sign, int correction_mask, double *freq, double *phase, double *Amp, double *tstat, double *fdot, double *delta1, double *delta1_over_A, int *valid_single_branch);

void WDMpixels(int Ns, double *TF, double *FF, int *nmid, int *nsize, int N, struct wdmshape *wdms);
void WDMpixelsTimeScan(int Ns, double *TF, double *FF, int *nmid, int *nsize, int N, struct wdmshape *wdms);
void WDMpixelsTimeScanRange(int Ns, double *TF, double *FF, double tmin, double tmax, int *nmid, int *nsize, int N, struct wdmshape *wdms);
void WDMpixelsAddMergerFrequencyTail(int *nmid, int *nsize, double fmax_spectrum, double tail_time, struct wdmshape *wdms);
void WDMmergePixelPlans(int *nmid_total, int *nsize_total, int *nmid_add, int *nsize_add);
void WDMtrack(double *wdmwave, int *listn, int *listm, int *Np, int Ns, int N, int *nmid, int *nsize, double tc, double *FF, gsl_interp_accel *AFacc, gsl_spline *AFspline, gsl_interp_accel *PFacc, gsl_spline *PFspline, struct wdmshape *wdms);
void WDMtrackAPProductDerivative(double *wdmwave, int *listn, int *listm, int *Np, int Ns, int N, int *nmid, int *nsize, double *FF, gsl_interp_accel *PFacc, gsl_spline *PFspline, gsl_interp_accel *DAFacc, gsl_spline *DAFspline, gsl_interp_accel *ADPFacc, gsl_spline *ADPFspline, struct wdmshape *wdms);
void WDMtrackFFT(double *wdmwave, int *listn, int *listm, int *Np, int N, int *nmid, int *nsize, double *hfft, struct wdmshape *wdms, int clip_to_spa_band, double fmin, double fmax);
void WDMreplaceWithShortFFTThreshold(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, double f_replace_start, double f_replace_stop, struct wdmshape *wdms, const char *layers_filename, int *layers_replaced, int *pixels_replaced);
void WDMaddWithShortFFTThreshold(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, double f_replace_start, double f_replace_stop, struct wdmshape *wdms, const char *layers_filename, int *layers_replaced, int *pixels_replaced);
void WDMblendWithShortFFTThreshold(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, double f_replace_start, double f_replace_stop, double blend_half_width, struct wdmshape *wdms, const char *layers_filename, int *layers_replaced, int *pixels_replaced);
void WDMreplaceMergerShortWindow(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, int Ns, double *TF, double *FF, struct wdmshape *wdms, int *layers_replaced, int *pixels_replaced);
void WDMtrackSplitFFT(double **wdm, int *nmid, int *nsize, int N, double tmax, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, struct wdmshape *wdms, int *layers_used, int *pixels_used);
void WDMtrackSplitFFTSummed(double **wdm, int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline, struct wdmshape *wdms, const char *label, int *layers_used, int *pixels_used);
void WDMtrackEarlySplitFFTSummed(double **wdm, int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline, struct wdmshape *wdms, double *endpoint_setup, double bandwidth, const char *label, int *layers_used, int *pixels_used);
void WDMtrackEarlySplitFFTPerCarrierEndpoint(double **wdm, int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline, struct wdmshape *wdms, double *endpoint_setup, double bandwidth, const char *label, int *layers_used, int *pixels_used);
void WDMbuildTHMUnionPixelPlan(int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, struct wdmshape *wdms, int include_merger_tail, int *nmid, int *nsize);
void WDMwriteSplitFFTBandwidthPlan(int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, struct wdmshape *wdms, const char *label, double plan_stop_time, int nband, const double *bandwidth);
void WDMwriteSplitFFTTrackBandwidthPlan(int Ns, double *response_time, int ncarriers, int channel, double ***freq_track, const THMFoldedCarrier *carrier, struct wdmshape *wdms, const char *label, double plan_stop_time, int nband, const double *bandwidth);
static void wdm_packet_fft_plan_init(WDMPacketFFTPlan *plan);
static void wdm_packet_fft_plan_free(WDMPacketFFTPlan *plan);
static int wdm_packet_fft_plan_prepare(WDMPacketFFTPlan *plan, int Ntx, struct wdmshape *wdms);
static void wdmtranF_plan(int m, WDMPacketFFTPlan *plan);
static double thm_instrument_prewhiten_factor(double f);
double phitilde(double om, double insDOM, double A, double B);
void wdmtranF(int m, int bw, double scale, double *phihf, double *data, double *wdmout);
void build_direct_fft_spectrum(double *hfft, int N, double tmax, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline);
void build_direct_fft_spectrum_tukey(double *hfft, int N, double tmax, double alpha, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline);
double tukey_weight_at_time(double t, double alpha, int N);
void sample_direct_fft_spectrum(double *hfft, int N, double Tobs, double f, double *re, double *im);
void sample_short_window_spectrum(double *h, int N, double dte, double f, double *re, double *im);
void sample_real_fft_exact(double *hfft, int N, double sample_dt, double f, double *re, double *im);
void sample_complex_fft_exact(double *zfft, int N, double sample_dt, double f, double *re, double *im);
double split_smooth_step(double t, double t1, double t2);
double linear_interp_clamped(int N, const double *x, const double *y, double xq);
void split_fft_weights(double t, double bend_start, double high_start, double roll, double *wlow, double *wbend, double *whigh);
void build_split_low_fft(double *hfft, int Nlow, double dtlow, double tmax, double bend_start, double high_start, double roll, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline);
void build_split_local_fft(double *hfft, int Nblock, double sample_dt, double block_start, double tmax, double bend_start, double high_start, double roll, int block_id, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline);
void build_split_low_fft_sum(double *hfft, int Nlow, double dtlow, double tmin, double tmax, double bend_start, double high_start, double roll, int ncarriers, int channel, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline);
void build_split_local_fft_sum(double *hfft, int Nblock, double sample_dt, double block_start, double tmin, double tmax, double bend_start, double high_start, double roll, int block_id, int ncarriers, int channel, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline);
void write_fft_amp_phase_diagnostic(const char *filename, double *hfft, int N, double Tobs, double *freq, int Nts, gsl_interp_accel *AFacc, gsl_spline *AFspline, gsl_interp_accel *PFacc, gsl_spline *PFspline);
void write_full_spa_fft_amp_phase_diagnostic(const char *filename, double *hfft, int N, double Tobs, int Nspa, double *time_spa, double *freq_spa, double *fdot_spa, int *valid_spa, double *Amp_spa, double *phase_spa);
void write_dense_spa_fft_amp_phase_diagnostic(const char *filename, double *hfft, int N, double Tobs, int Ns, double *TS, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double fmin, double fmax, double df, double spa_tukey_alpha);
void write_dense_spa_fft_amp_phase_diagnostic_from_tracks(const char *filename, double *hfft, int N, double Tobs, int Ns, double *TS, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, const double *freq_grid, const double *fdot_grid, double fmin, double fmax, double df, double spa_tukey_alpha, const char *track_description);
void write_fourier_band_comparison(const char *filename, double *hfft, int N, double Tobs, gsl_interp_accel *AFacc, gsl_spline *AFspline, gsl_interp_accel *PFacc, gsl_spline *PFspline, double fmin, double fmax, const char *test_name, struct wdmshape *wdms);
void write_first_order_spa_fourier_diagnostic(const char *label, const char *suffix, const char *description, double correction_sign, int correction_mask, int Ns, double *TS, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *phase0_all, double *hfft, double *hfft_tukey, int N, struct wdmshape *wdms);
void write_wdm_layer_comparison(const char *filename, double **reference, double **test, const char *reference_name, const char *test_name, struct wdmshape *wdms, int jmin, int jmax);
void write_spline_frequency_diagnostic(const char *label, double *setup, double *TS, int Ns, double *freq_track, gsl_interp_accel *PSacc, gsl_spline *PSspline);
void write_intrinsic_spline_frequency_diagnostic(const char *label, double *setup, double *TS, int Ns, double *freq_mode, gsl_interp_accel *PSacc, gsl_spline *PSspline);
void write_short_fft_spa_complex_diagnostic(const char *label, double *setup, double Tobs, double *TS, int Ns, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *short_htime);
void unpack_wdm_track(double **wdm, int *listn, int *listm, double *wdmwave, int Np);
int write_wdm_nonzero_track_pixels(const char *filename, double **wdm);
void write_track_pixels(const char *filename, int *listn, int *listm, double *wdmwave, int Np);
void write_wdm_matrix(const char *filename, double **wdm);
void write_wdm_binary(const char *filename, double **wdm, struct wdmshape *wdms);
void write_match_info(const char *filename, struct wdmshape *wdms, double tukey_alpha, double roll_time);

int *int_vector(int N);
void free_int_vector(int *v);
double *double_vector(int N);
void free_double_vector(double *v);
double **double_matrix(int N, int M);
void free_double_matrix(double **m, int N);
double ***double_tensor(int N, int M, int L);
void free_double_tensor(double ***t, int N, int M);
int **int_matrix(int N, int M);
void free_int_matrix(int **m, int N);
double ****double_quad(int N, int M, int L, int K);
void free_double_quad(double ****t, int N, int M, int L);

static void enforce_phenom_mass_hierarchy(double *m1, double *m2, double *chi1, double *chi2)
{
    double tmp;

    if (*m2 > *m1)
    {
        fprintf(stderr, "Warning: IMRPhenomT uses the LAL mass hierarchy m1 >= m2; switching values (m1 <-> m2 and chi1 <-> chi2) to match the waveform default mass hierarchy.\n");

        tmp = *m1;
        *m1 = *m2;
        *m2 = tmp;

        tmp = *chi1;
        *chi1 = *chi2;
        *chi2 = tmp;
    }
}

static int thm_parse_double_list(const char *text, double *values, int max_values, int *nvalues)
{
    const char *p;
    char *endptr;
    double value;
    int count;

    if(nvalues != NULL) *nvalues = 0;
    if(text == NULL || values == NULL || nvalues == NULL || max_values < 1)
    {
        return 0;
    }

    p = text;
    count = 0;
    while(*p != '\0')
    {
        while(*p == ' ' || *p == '\t' || *p == ',') p++;
        if(*p == '\0') break;
        if(count >= max_values) return 0;

        value = strtod(p, &endptr);
        if(endptr == p || !isfinite(value) || value <= 0.0)
        {
            return 0;
        }
        values[count++] = value;
        p = endptr;

        while(*p == ' ' || *p == '\t') p++;
        if(*p != '\0' && *p != ',')
        {
            return 0;
        }
    }

    if(count < 1) return 0;
    *nvalues = count;
    return 1;
}

static int thm_next_power_of_two_int(int n)
{
    int p;

    if(n <= 1) return 1;
    p = 1;
    while(p < n && p <= INT_MAX/2) p *= 2;
    return p;
}

static int thm_largest_power_of_two_leq_int(int n)
{
    int p;

    if(n <= 1) return 1;
    p = 1;
    while(p <= n/2) p *= 2;
    return p;
}

static int thm_first_upward_crossing_time(int n, const double *time,
                                          const double *frequency,
                                          double threshold,
                                          double stop_time,
                                          double *crossing_time)
{
    int i, first;

    if(n < 1 || time == NULL || frequency == NULL ||
       crossing_time == NULL || !isfinite(threshold) || threshold <= 0.0)
    {
        return 0;
    }

    first = -1;
    for(i=0; i<n; i++)
    {
        if(time[i] > stop_time) break;
        if(isfinite(time[i]) && isfinite(frequency[i]))
        {
            first = i;
            break;
        }
    }
    if(first < 0) return 0;
    if(frequency[first] >= threshold)
    {
        *crossing_time = time[first];
        return 1;
    }

    for(i=first; i<n-1; i++)
    {
        double fraction;

        if(time[i+1] > stop_time) break;
        if(!isfinite(time[i]) || !isfinite(time[i+1]) ||
           !isfinite(frequency[i]) || !isfinite(frequency[i+1]))
        {
            continue;
        }
        if(frequency[i] < threshold && threshold <= frequency[i+1] &&
           frequency[i+1] > frequency[i])
        {
            fraction = (threshold-frequency[i])/
                       (frequency[i+1]-frequency[i]);
            *crossing_time = time[i]+fraction*(time[i+1]-time[i]);
            return 1;
        }
    }
    return 0;
}

static void thm_adjust_endpoint_taper_flat_time(double setup[7],
                                                 double target_flat_time)
{
    double dte, old_start, old_stop, original_rise, available_rise;
    double requested_start, old_frequency_stop;
    int old_samples, required_samples, expanded_samples;

    if(setup == NULL || !isfinite(target_flat_time)) return;
    dte = setup[0];
    old_samples = (int)setup[1];
    old_start = setup[2];
    original_rise = setup[3];
    if(!isfinite(dte) || dte <= 0.0 || old_samples < 1 ||
       !isfinite(old_start) || !isfinite(original_rise) ||
       original_rise <= 0.0)
    {
        return;
    }

    target_flat_time -= SHORTFFT_MERGER_TAPER_MARGIN_SECONDS;
    available_rise = target_flat_time-old_start;
    if(available_rise >= dte)
    {
        if(available_rise < original_rise) setup[3] = available_rise;
        return;
    }

    old_stop = old_start+dte*(double)old_samples;
    requested_start = target_flat_time-original_rise;
    required_samples = (int)ceil((old_stop-requested_start)/dte);
    if(required_samples < old_samples) required_samples = old_samples;
    expanded_samples = thm_next_power_of_two_int(required_samples);
    if(expanded_samples < old_samples) return;

    old_frequency_stop = setup[6]/(dte*(double)old_samples);
    setup[1] = (double)expanded_samples;
    setup[2] = dte*rint(old_stop/dte)-dte*(double)expanded_samples;
    setup[3] = fmin(original_rise, target_flat_time-setup[2]);
    setup[6] = ceil(old_frequency_stop*dte*(double)expanded_samples);
}

static void thm_set_default_split_fft_plan_bandwidths(const struct wdmshape *wdms, THMRunOptions *options)
{
    int i;
    const int default_layers[5] = {16, 32, 64, 128, 256};

    if(wdms == NULL || options == NULL) return;
    if(options->wdm_split_plan_nband > 0) return;

    /* The production transform uses the benchmarked 0.016-Hz block budget.
     * The layer-spaced list below is only for the standalone planner
     * diagnostic that compares several candidate bandwidths. */
    if(options->wdm_split_early_fft)
    {
        options->wdm_split_plan_nband = 1;
        options->wdm_split_plan_bandwidth[0] =
            thm_wdm_split_early_fft_bandwidth_hz;
        return;
    }

    options->wdm_split_plan_nband = 5;
    for(i=0; i<options->wdm_split_plan_nband; i++)
    {
        options->wdm_split_plan_bandwidth[i] =
            ((double)default_layers[i])*wdms->DF;
    }
}

static void print_thm_mode_usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s [default|22pair|MODE[,MODE,...]] [--wdm-mode MODE] [--wdm-all]\n"
            "       %s --modes default [--wdm-mode 55|--wdm-all] [--wdm-join-time SECONDS] [--wdm-rise SECONDS]\n"
            "       %s --modes 22pair [--wdm-mode 22]\n"
            "       %s --modes 22,2-2,33,3-3 [--wdm-mode 33]\n"
            "       %s --modes 22pair --compare-ap-grid\n"
            "       %s --modes 22pair --compare-per-mode-grid --wdm-mode 22\n"
            "       %s --modes 22pair --legacy-ap-grid --wdm-mode 22\n"
            "       %s --modes default --wdm-all --tdi-generation 2 [--tc SECONDS] [--kappa0 RADIANS]\n"
            "       TDI-2 defaults to Taylor-expanded delay chains, reference-time projections, and quadratic one-link solves\n"
            "       add --tdi2-exact for fully retarded projections and iterative light-cone solves\n"
            "       add --tdi2-retarded-chain to retain nested retarded delays with the selected one-link solver\n"
            "       add --tdi2-frozen-projection to freeze projections while retaining retarded delay chains\n"
            "       add --tdi2-light-time iterative|quadratic|taylor2 to select the one-link solver\n"
            "       default AP grid is --intrinsic-tdi-ap-grid; use --detector-ap-grid for the old direct detector-adaptive AP evaluation\n"
            "       add --wdm-partition-endpoint to try the experimental falling-SPA plus summed endpoint FFT path\n"
            "       add --wdm-blend-endpoint [--wdm-blend-half-width-layers L] to smoothly blend clean SPA and endpoint WDM coefficients\n"
            "       exact-delay endpoint TDI is the default; add --no-direct-endpoint-tdi for the historical sparse AP reconstruction\n"
            "       partitioned complex FFT is the WDM default; add --spa-endpoint-fft for the faster leading-SPA plus endpoint approximation\n"
            "       with --spa-endpoint-fft, add --spa-t1t2 to try the optional T1+T2 post-adiabatic phase correction\n"
            "       add --prewhiten-instrument [--prewhiten-average-halfwidth N] to ASD-whiten local Fourier samples before WDM\n"
            "       add --prewhiten-smooth-halfwidth-mhz W to use a physical triangular ASD half-width [0.01]\n"
            "       add --prewhiten-smooth-frac E [--prewhiten-smooth-floor-mhz W] to use max(W,E*f) smoothing\n"
            "       add --wdm-spline-endpoint only for diagnostics of the old per-carrier A/P spline bridge\n"
            "       add --wdm-split-fft to write an experimental no-SPA split-FFT WDM diagnostic\n"
            "       --wdm-split-early-fft explicitly selects the default bandwidth-planned partition FFTs [--wdm-split-bandwidths B]\n"
            "       complex response envelopes are the partitioned-FFT default; add --blockwise-sparse-tdi for the older per-block delayed response\n"
            "       add --wdm-split-endpoint-margin-cycles C to prune endpoint layers below f_endpoint-C/rise\n"
            "       add --wdm-endpoint-min-abs-m M to start the clean endpoint FFT from the first |m|>=M join [3]\n"
            "       add --wdm-split-plan [--wdm-split-bandwidths B1,B2,...] to write bandwidth-driven split-FFT block layouts\n"
            "       add --split-tdi-response to WDM-transform the two Michelson-like TDI pieces separately, then subtract/add coefficients\n"
            "       add --no-tdi-delay-zero-grid to skip response-grid refinement forced only by equal-arm delay zeros\n"
            "       add --diagnostics to write dense/debug output files; default WDM output is sparse track_pixels_THM_X/Y/Z.dat\n"
            "       add --write-time-domain to write THM_all_X/Y/Ztime(.tukey).dat without dense WDM plotting diagnostics\n"
            "       add --intrinsic-ap-grid to write source-time AP samples before the TDI grid interpolation\n"
            "       add --no-sparse-files to exercise the in-memory WDM path without writing track_pixels files\n"
            "       add --tc SECONDS to override the default merger time\n"
            "       add --kappa0 RADIANS to shift the initial LISA guiding-center orbital phase\n"
            "Modes are written as ellm for positive m and ell-m for negative m.\n",
            program, program, program, program, program, program, program,
            program);
}

static int parse_thm_mode_token(const char *token, int *ell, int *emm)
{
    size_t len;

    if(token == NULL || ell == NULL || emm == NULL)
    {
        return 0;
    }
    len = strlen(token);
    if(len < 2)
    {
        return 0;
    }

    if(isdigit((unsigned char)token[0]) &&
       (token[1] == '-' || token[1] == '+') &&
       len >= 3)
    {
        char *endptr = NULL;
        *ell = token[0]-'0';
        *emm = (int)strtol(token+1, &endptr, 10);
        return endptr != NULL && *endptr == '\0';
    }
    if(isdigit((unsigned char)token[0]) &&
       (token[1] == ':' || token[1] == '_') &&
       len >= 3)
    {
        char *endptr = NULL;
        *ell = token[0]-'0';
        *emm = (int)strtol(token+2, &endptr, 10);
        return endptr != NULL && *endptr == '\0';
    }
    if(len == 2 && isdigit((unsigned char)token[0]) && isdigit((unsigned char)token[1]))
    {
        *ell = token[0]-'0';
        *emm = token[1]-'0';
        return 1;
    }

    return 0;
}

static int parse_thm_mode_selection(const char *spec, IMRPhenomTHMMode *modes, int max_modes)
{
    char clean[512];
    char *token;
    size_t i, j;
    int nmodes;

    if(modes == NULL || max_modes <= 0)
    {
        return -1;
    }
    if(spec == NULL || strcmp(spec, "default") == 0 || strcmp(spec, "all") == 0 ||
       strcmp(spec, "full") == 0 || strcmp(spec, "thm") == 0)
    {
        return IMRPhenomTHMDefaultModes(modes, max_modes);
    }
    if(strcmp(spec, "22") == 0 || strcmp(spec, "22pair") == 0 ||
       strcmp(spec, "dominant") == 0)
    {
        if(max_modes < 2) return -1;
        modes[0].ell = 2;
        modes[0].emm = 2;
        modes[1].ell = 2;
        modes[1].emm = -2;
        return 2;
    }

    j = 0;
    for(i=0; spec[i] != '\0' && j+1 < sizeof(clean); i++)
    {
        char c = spec[i];
        if(isspace((unsigned char)c) || c == '(' || c == ')' ||
           c == '[' || c == ']')
        {
            continue;
        }
        clean[j++] = c;
    }
    clean[j] = '\0';

    nmodes = 0;
    token = strtok(clean, ",");
    while(token != NULL)
    {
        int ell, emm;
        if(nmodes >= max_modes || !parse_thm_mode_token(token, &ell, &emm))
        {
            return -1;
        }
        modes[nmodes].ell = ell;
        modes[nmodes].emm = emm;
        nmodes++;
        token = strtok(NULL, ",");
    }

    return nmodes > 0 ? nmodes : -1;
}

static void thm_default_options(THMRunOptions *options)
{
    memset(options, 0, sizeof(*options));
    options->tdi_generation = 1;
    options->tdi2_chain_taylor = 1;
    options->tdi2_light_time_solver = THM_TDI2_LIGHT_TIME_QUADRATIC;
    options->use_intrinsic_tdi_ap_grid = 1;
    options->use_detector_ap_grid = 0;
    options->write_sparse_files = 1;
    options->wdm_endpoint_min_abs_m = 3;
    options->wdm_blend_half_width_layers = 1.0;
    options->direct_endpoint_tdi = 1;
    options->wdm_split_early_fft = 1;
    options->spa_t1t2_correction = 0;
    options->prewhiten_average_halfwidth = 2;
    options->prewhiten_smooth_halfwidth_hz = 1.0e-5;
    options->prewhiten_smooth_frac = 0.0;
    options->prewhiten_smooth_floor_hz = 1.0e-5;
    options->wdm_split_endpoint_margin_cycles =
        SPLIT_FFT_ENDPOINT_MARGIN_CYCLES_DEFAULT;
}

static int parse_thm_run_options(int argc, char **argv, THMRunOptions *options)
{
    int i;

    if(options == NULL)
    {
        return 0;
    }
    thm_default_options(options);

    for(i=1; i<argc; i++)
    {
        if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            options->show_help = 1;
            return 1;
        }
        else if(strcmp(argv[i], "--modes") == 0)
        {
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            options->mode_spec = argv[++i];
        }
        else if(strcmp(argv[i], "--wdm-mode") == 0)
        {
            int ell, emm;
            if(i+1 >= argc || !parse_thm_mode_token(argv[i+1], &ell, &emm))
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_mode_enabled = 1;
            options->wdm_ell = ell;
            options->wdm_abs_emm = abs(emm);
            i++;
        }
        else if(strcmp(argv[i], "--tdi-generation") == 0)
        {
            if(i+1 >= argc ||
               (strcmp(argv[i+1], "1") != 0 &&
                strcmp(argv[i+1], "2") != 0))
            {
                options->invalid = 1;
                return 0;
            }
            options->tdi_generation = argv[++i][0]-'0';
        }
        else if(strcmp(argv[i], "--tdi2-frozen-projection") == 0)
        {
            options->tdi2_frozen_projection = 1;
            options->tdi2_chain_taylor = 0;
            options->tdi2_custom_option = 1;
        }
        else if(strcmp(argv[i], "--tdi2-chain-taylor") == 0)
        {
            options->tdi2_chain_taylor = 1;
            options->tdi2_frozen_projection = 0;
            options->tdi2_custom_option = 1;
        }
        else if(strcmp(argv[i], "--tdi2-retarded-chain") == 0)
        {
            options->tdi2_chain_taylor = 0;
            options->tdi2_frozen_projection = 0;
            options->tdi2_custom_option = 1;
        }
        else if(strcmp(argv[i], "--tdi2-exact") == 0)
        {
            options->tdi2_exact = 1;
        }
        else if(strcmp(argv[i], "--tdi2-light-time") == 0)
        {
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            i++;
            if(strcmp(argv[i], "iterative") == 0)
                options->tdi2_light_time_solver =
                    THM_TDI2_LIGHT_TIME_ITERATIVE;
            else if(strcmp(argv[i], "quadratic") == 0)
                options->tdi2_light_time_solver =
                    THM_TDI2_LIGHT_TIME_QUADRATIC;
            else if(strcmp(argv[i], "taylor2") == 0)
                options->tdi2_light_time_solver =
                    THM_TDI2_LIGHT_TIME_TAYLOR2;
            else
            {
                options->invalid = 1;
                return 0;
            }
            options->tdi2_custom_option = 1;
        }
        else if(strcmp(argv[i], "--wdm-all") == 0 ||
                strcmp(argv[i], "--wdm-combined") == 0)
        {
            options->wdm_all_enabled = 1;
        }
        else if(strcmp(argv[i], "--spa-t1t2") == 0)
        {
            options->spa_t1t2_correction = 1;
        }
        else if(strcmp(argv[i], "--no-spa-t1t2") == 0)
        {
            options->spa_t1t2_correction = 0;
        }
        else if(strcmp(argv[i], "--split-tdi-response") == 0 ||
                strcmp(argv[i], "--wdm-split-tdi-response") == 0)
        {
            options->split_tdi_response = 1;
            options->no_tdi_delay_zero_grid = 1;
        }
        else if(strcmp(argv[i], "--no-tdi-delay-zero-grid") == 0)
        {
            options->no_tdi_delay_zero_grid = 1;
        }
        else if(strcmp(argv[i], "--tdi-delay-zero-grid") == 0)
        {
            options->no_tdi_delay_zero_grid = 0;
        }
        else if(strcmp(argv[i], "--wdm-join-time") == 0)
        {
            char *endptr = NULL;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_join_time = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' || !isfinite(options->wdm_join_time))
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_join_time_enabled = 1;
        }
        else if(strcmp(argv[i], "--wdm-rise") == 0)
        {
            char *endptr = NULL;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_rise_seconds = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' ||
               !isfinite(options->wdm_rise_seconds) ||
               options->wdm_rise_seconds <= 0.0)
            {
                options->invalid = 1;
                return 0;
            }
        }
        else if(strcmp(argv[i], "--tc") == 0 ||
                strcmp(argv[i], "--merger-time") == 0)
        {
            char *endptr = NULL;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            options->tc_override = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' ||
               !isfinite(options->tc_override) ||
               options->tc_override <= 0.0)
            {
                options->invalid = 1;
                return 0;
            }
            options->tc_override_enabled = 1;
        }
        else if(strcmp(argv[i], "--kappa0") == 0)
        {
            char *endptr = NULL;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            options->kappa0_override = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' ||
               !isfinite(options->kappa0_override))
            {
                options->invalid = 1;
                return 0;
            }
            options->kappa0_override_enabled = 1;
        }
        else if(strcmp(argv[i], "--wdm-partition-endpoint") == 0)
        {
            options->wdm_partition_endpoint = 1;
        }
        else if(strcmp(argv[i], "--wdm-blend-endpoint") == 0)
        {
            options->wdm_blend_endpoint = 1;
        }
        else if(strcmp(argv[i], "--direct-endpoint-tdi") == 0 ||
                strcmp(argv[i], "--wdm-direct-endpoint-tdi") == 0)
        {
            options->direct_endpoint_tdi = 1;
        }
        else if(strcmp(argv[i], "--no-direct-endpoint-tdi") == 0 ||
                strcmp(argv[i], "--no-wdm-direct-endpoint-tdi") == 0)
        {
            options->direct_endpoint_tdi = 0;
        }
        else if(strcmp(argv[i], "--wdm-blend-half-width-layers") == 0)
        {
            char *endptr = NULL;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_blend_half_width_layers = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' ||
               !isfinite(options->wdm_blend_half_width_layers) ||
               options->wdm_blend_half_width_layers <= 0.0)
            {
                options->invalid = 1;
                return 0;
            }
        }
        else if(strcmp(argv[i], "--prewhiten-instrument") == 0 ||
                strcmp(argv[i], "--wdm-prewhiten-instrument") == 0)
        {
            options->prewhiten_instrument = 1;
        }
        else if(strcmp(argv[i], "--prewhiten-average-halfwidth") == 0 ||
                strcmp(argv[i], "--prewhiten-avg-halfwidth") == 0 ||
                strcmp(argv[i], "--wdm-prewhiten-average-halfwidth") == 0 ||
                strcmp(argv[i], "--wdm-prewhiten-avg-halfwidth") == 0)
        {
            char *endptr = NULL;
            long value;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            value = strtol(argv[++i], &endptr, 10);
            if(endptr == argv[i] || *endptr != '\0' || value < 0)
            {
                options->invalid = 1;
                return 0;
            }
            options->prewhiten_average_halfwidth = (int)value;
        }
        else if(strcmp(argv[i], "--prewhiten-smooth-halfwidth-mhz") == 0 ||
                strcmp(argv[i], "--prewhiten-smooth-width-mhz") == 0 ||
                strcmp(argv[i], "--wdm-prewhiten-smooth-halfwidth-mhz") == 0)
        {
            char *endptr = NULL;
            double value;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            value = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' || !isfinite(value) ||
               value < 0.0)
            {
                options->invalid = 1;
                return 0;
            }
            options->prewhiten_smooth_halfwidth_hz = value*1.0e-3;
        }
        else if(strcmp(argv[i], "--prewhiten-smooth-frac") == 0 ||
                strcmp(argv[i], "--wdm-prewhiten-smooth-frac") == 0)
        {
            char *endptr = NULL;
            double value;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            value = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' || !isfinite(value) ||
               value < 0.0)
            {
                options->invalid = 1;
                return 0;
            }
            options->prewhiten_smooth_frac = value;
        }
        else if(strcmp(argv[i], "--prewhiten-smooth-floor-mhz") == 0 ||
                strcmp(argv[i], "--wdm-prewhiten-smooth-floor-mhz") == 0)
        {
            char *endptr = NULL;
            double value;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            value = strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' || !isfinite(value) ||
               value < 0.0)
            {
                options->invalid = 1;
                return 0;
            }
            options->prewhiten_smooth_floor_hz = value*1.0e-3;
        }
        else if(strcmp(argv[i], "--wdm-spline-endpoint") == 0)
        {
            options->wdm_spline_endpoint = 1;
        }
        else if(strcmp(argv[i], "--wdm-split-fft") == 0)
        {
            options->wdm_split_fft = 1;
        }
        else if(strcmp(argv[i], "--wdm-split-early-fft") == 0)
        {
            options->wdm_split_early_fft = 1;
            options->wdm_split_plan = 1;
        }
        else if(strcmp(argv[i], "--spa-endpoint-fft") == 0 ||
                strcmp(argv[i], "--wdm-spa-endpoint-fft") == 0)
        {
            options->wdm_spa_endpoint_fft = 1;
            options->wdm_split_early_fft = 0;
            options->wdm_split_plan = 0;
        }
        else if(strcmp(argv[i], "--blockwise-sparse-tdi") == 0 ||
                strcmp(argv[i], "--wdm-blockwise-sparse-tdi") == 0)
        {
            options->blockwise_sparse_tdi = 1;
        }
        else if(strcmp(argv[i], "--complex-envelope") == 0 ||
                strcmp(argv[i], "--wdm-complex-envelope") == 0)
        {
            options->blockwise_sparse_tdi = 0;
        }
        else if(strcmp(argv[i], "--wdm-split-plan") == 0)
        {
            options->wdm_split_plan = 1;
        }
        else if(strcmp(argv[i], "--wdm-split-bandwidths") == 0)
        {
            if(i+1 >= argc ||
               !thm_parse_double_list(argv[i+1],
                                      options->wdm_split_plan_bandwidth,
                                      SPLIT_FFT_PLAN_MAX_BANDWIDTHS,
                                      &options->wdm_split_plan_nband))
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_split_plan = 1;
            i++;
        }
        else if(strcmp(argv[i], "--wdm-split-endpoint-margin-cycles") == 0)
        {
            char *endptr = NULL;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_split_endpoint_margin_cycles =
                strtod(argv[++i], &endptr);
            if(endptr == argv[i] || *endptr != '\0' ||
               !isfinite(options->wdm_split_endpoint_margin_cycles) ||
               options->wdm_split_endpoint_margin_cycles < 0.0)
            {
                options->invalid = 1;
                return 0;
            }
        }
        else if(strcmp(argv[i], "--wdm-endpoint-min-abs-m") == 0 ||
                strcmp(argv[i], "--endpoint-min-abs-m") == 0)
        {
            char *endptr = NULL;
            long value;
            if(i+1 >= argc)
            {
                options->invalid = 1;
                return 0;
            }
            value = strtol(argv[++i], &endptr, 10);
            if(endptr == argv[i] || *endptr != '\0' || value < 1 || value > 5)
            {
                options->invalid = 1;
                return 0;
            }
            options->wdm_endpoint_min_abs_m = (int)value;
        }
        else if(strcmp(argv[i], "--compare-ap-grid") == 0)
        {
            options->compare_ap_grid = 1;
        }
        else if(strcmp(argv[i], "--compare-per-mode-grid") == 0)
        {
            options->compare_per_mode_grid = 1;
        }
        else if(strcmp(argv[i], "--shared-ap-grid") == 0)
        {
            options->use_intrinsic_tdi_ap_grid = 0;
            options->use_detector_ap_grid = 1;
        }
        else if(strcmp(argv[i], "--detector-ap-grid") == 0)
        {
            options->use_intrinsic_tdi_ap_grid = 0;
            options->use_detector_ap_grid = 1;
        }
        else if(strcmp(argv[i], "--intrinsic-tdi-ap-grid") == 0 ||
                strcmp(argv[i], "--production-ap-grid") == 0)
        {
            options->use_intrinsic_tdi_ap_grid = 1;
            options->use_detector_ap_grid = 0;
        }
        else if(strcmp(argv[i], "--legacy-ap-grid") == 0)
        {
            options->use_intrinsic_tdi_ap_grid = 0;
            options->use_detector_ap_grid = 0;
        }
        else if(strcmp(argv[i], "--diagnostics") == 0)
        {
            options->diagnostics_enabled = 1;
        }
        else if(strcmp(argv[i], "--no-diagnostics") == 0)
        {
            options->diagnostics_enabled = 0;
        }
        else if(strcmp(argv[i], "--write-time-domain") == 0)
        {
            options->write_time_domain_files = 1;
        }
        else if(strcmp(argv[i], "--no-write-time-domain") == 0)
        {
            options->write_time_domain_files = 0;
        }
        else if(strcmp(argv[i], "--intrinsic-ap-grid") == 0 ||
                strcmp(argv[i], "--write-intrinsic-ap-grid") == 0)
        {
            options->write_intrinsic_ap_grid = 1;
        }
        else if(strcmp(argv[i], "--no-sparse-files") == 0)
        {
            options->write_sparse_files = 0;
        }
        else if(strcmp(argv[i], "--sparse-files") == 0)
        {
            options->write_sparse_files = 1;
        }
        else if(argv[i][0] != '-' && options->mode_spec == NULL)
        {
            options->mode_spec = argv[i];
        }
        else
        {
            options->invalid = 1;
            return 0;
        }
    }

    return 1;
}

#ifndef PHENOMTHM_TDI_LIBRARY
int main(int argc, char **argv)
{
    const double phi0 = 0.0;
    double m1, m2, chi1, chi2, distance, Mtot, eta, tc, Tobs;
    double *params;
    double *TS, *tspace, *tc_tau;
    double *response_time, *trim_time;
    double **mode_amp, **mode_phase, **mode_freq;
    double *X, *Y, *Z, *Xf, *Yf, *Zf;
    int Ns, Nsmax, Ns_response, nmodes;
    int i, j, k, N, Nc;
    double dtx, dtc;
    double constellation_tmin, constellation_tmax;
    double **Larray, *tarray;
    double ***Parray, ***Varray;
    clock_t start, end;
    double adaptive_ap_time, tdi_time;
    double max_mode_frequency_hz;
    IMRPhenomTHM model;
    IMRPhenomTHMMode modes[IMRPHENOMTHM_MAX_MODES];
    IMRPhenomTHMMode requested_modes[IMRPHENOMTHM_MAX_MODES];
    THMProjection projection[IMRPHENOMTHM_MAX_MODES];
    THMFoldedCarrier carrier[IMRPHENOMTHM_MAX_MODES];
    THMRunOptions options;
    THMSparseWDMTriplet wdm_tracks;
    const char *mode_spec;
    const char *ap_grid_label;
    int requested_nmodes, ncarriers;
    int wdm_carrier_index;
    gsl_interp_accel **Aacc, **Pacc;
    gsl_spline **Aspline, **Pspline;
    struct wdmshape *wdms = malloc(sizeof(struct wdmshape));

    if(wdms == NULL)
    {
        fprintf(stderr, "allocation failure\n");
        return 1;
    }
    wdmvalues(wdms);
    thm_sparse_wdm_triplet_init(&wdm_tracks);
    N = Nt*Nf;
    Tobs = wdms->Tobs;

    if(!parse_thm_run_options(argc, argv, &options))
    {
        print_thm_mode_usage(argv[0]);
        return 1;
    }
    if(options.show_help)
    {
        print_thm_mode_usage(argv[0]);
        return 0;
    }
    if(options.invalid)
    {
        print_thm_mode_usage(argv[0]);
        return 1;
    }
    if((options.tdi2_custom_option || options.tdi2_exact) &&
       options.tdi_generation != 2)
    {
        fprintf(stderr,
                "TDI-2 response options require --tdi-generation 2.\n");
        return 1;
    }
    if(options.tdi2_exact && options.tdi2_custom_option)
    {
        fprintf(stderr, "--tdi2-exact cannot be combined with other TDI-2 response options.\n");
        return 1;
    }
    if(options.tdi2_exact)
    {
        options.tdi2_chain_taylor = 0;
        options.tdi2_frozen_projection = 0;
        options.tdi2_light_time_solver = THM_TDI2_LIGHT_TIME_ITERATIVE;
    }
    thm_set_orbit_phase(options.kappa0_override_enabled ?
                        options.kappa0_override : 0.0);
    thm_set_tdi_generation(options.tdi_generation);
    thm_set_tdi2_frozen_projection(options.tdi2_frozen_projection);
    thm_set_tdi2_chain_taylor(options.tdi2_chain_taylor);
    thm_set_tdi2_light_time_solver(options.tdi2_light_time_solver);
    mode_spec = options.mode_spec;
    requested_nmodes = parse_thm_mode_selection(mode_spec, requested_modes,
                                                IMRPHENOMTHM_MAX_MODES);
    if(requested_nmodes < 1)
    {
        print_thm_mode_usage(argv[0]);
        return 1;
    }
    thm_diagnostics_enabled = options.diagnostics_enabled;
    if(options.wdm_split_plan || options.wdm_split_early_fft)
    {
        thm_set_default_split_fft_plan_bandwidths(wdms, &options);
    }
    thm_split_fft_plan_enabled = options.wdm_split_plan || options.wdm_split_early_fft;
    thm_split_fft_plan_nband = options.wdm_split_plan_nband;
    for(i=0; i<thm_split_fft_plan_nband; i++)
    {
        thm_split_fft_plan_bandwidth[i] = options.wdm_split_plan_bandwidth[i];
    }
    thm_split_fft_endpoint_margin_cycles =
        options.wdm_split_endpoint_margin_cycles;
    thm_set_endpoint_min_abs_m(options.wdm_endpoint_min_abs_m);
    thm_set_wdm_blend_endpoint(options.wdm_blend_endpoint,
                               options.wdm_blend_half_width_layers);
    thm_set_wdm_direct_endpoint_tdi(options.direct_endpoint_tdi);
    thm_set_wdm_complex_envelope(!options.blockwise_sparse_tdi);
    thm_set_spa_t1t2_correction(options.spa_t1t2_correction);
    thm_set_wdm_instrument_prewhiten(options.prewhiten_instrument,
                                     options.prewhiten_average_halfwidth);
    thm_set_wdm_instrument_prewhiten_smooth_halfwidth(options.prewhiten_smooth_halfwidth_hz);
    thm_set_wdm_instrument_prewhiten_smooth_fraction(options.prewhiten_smooth_frac,
                                                     options.prewhiten_smooth_floor_hz);
    thm_intrinsic_tdi_include_delay_zero_grid =
        options.no_tdi_delay_zero_grid ? 0 : 1;

    /*
     * This THM driver is intentionally limited to time-domain waveform
     * construction.  PhenomT_TDI.c remains the 22 fast-WDM reference.  Here we
     * share the constellation/TDI geometry across all modes, then sum the
     * per-mode quadratures after applying spin-weighted-harmonic and
     * polarization factors.
     */
    m1 = 2.0e5*TSUN;
    m2 = 1.0e5*TSUN;
    chi1 = 0.42;
    chi2 = 0.85;
    distance = 1.0; /* Gpc */
    tc = 3.0e7;
    if(options.tc_override_enabled) tc = options.tc_override;
    enforce_phenom_mass_hierarchy(&m1, &m2, &chi1, &chi2);

    Mtot = m1 + m2;
    eta = (m1*m2)/(Mtot*Mtot);

    params = (double *)malloc(sizeof(double)*11);
    if(params == NULL)
    {
        fprintf(stderr, "allocation failure\n");
        return 1;
    }
    params[0] = m1;
    params[1] = m2;
    params[2] = chi1;
    params[3] = chi2;
    params[4] = phi0;
    params[5] = tc;
    params[6] = log(distance);
    params[7] = 2.31; /* EclipticCoLatitude */
    params[8] = 0.57; /* EclipticLongitude */
    params[9] = 0.4;  /* polarization */
    params[10] = 0.3; /* cos inclination */

    if(options.tdi_generation == 2)
    {
        THMObservationContext *context = NULL;
        THMWorkerWorkspace *workspace = NULL;
        const char *channel_name[3] = {"X2", "Y2", "Z2"};
        char output_name[80];
        int status = 1;

        if(!options.wdm_all_enabled || options.wdm_mode_enabled ||
           options.diagnostics_enabled || options.write_time_domain_files ||
           options.prewhiten_instrument || options.wdm_partition_endpoint ||
           options.wdm_spline_endpoint || options.wdm_split_fft ||
           options.split_tdi_response || options.compare_ap_grid ||
           options.compare_per_mode_grid || options.wdm_split_plan ||
           options.spa_t1t2_correction || options.wdm_spa_endpoint_fft ||
           options.wdm_blend_endpoint || options.wdm_join_time_enabled ||
           options.write_intrinsic_ap_grid ||
           options.use_detector_ap_grid ||
           !options.use_intrinsic_tdi_ap_grid ||
           !options.direct_endpoint_tdi)
        {
            fprintf(stderr, "TDI-2 currently requires --wdm-all and the direct complex partition path; legacy diagnostics and prewhitening are not supported.\n");
            goto tdi2_cleanup;
        }
        context = thm_observation_context_create(
            options.kappa0_override_enabled ? options.kappa0_override : 0.0);
        if(context == NULL) goto tdi2_cleanup;
        workspace = thm_worker_workspace_create(context);
        if(workspace == NULL) goto tdi2_cleanup;
        status = generate_thm_tdi_wdm_context(context, workspace, params,
                                               mode_spec, &wdm_tracks);
        if(status == 0)
        {
            for(i=0; i<3; i++)
            {
                printf("%s active_pixels %d\n", channel_name[i],
                       wdm_tracks.channel[i].npixels);
                if(options.write_sparse_files)
                {
                    snprintf(output_name, sizeof(output_name),
                             "track_pixels_THM_%s.dat", channel_name[i]);
                    if(write_thm_sparse_wdm_channel(
                           output_name, &wdm_tracks.channel[i]) !=
                       wdm_tracks.channel[i].npixels)
                        status = 3;
                }
            }
        }
        else
            fprintf(stderr, "TDI-2 THM WDM generation failed: %d\n", status);

tdi2_cleanup:
        thm_worker_workspace_destroy(workspace);
        thm_observation_context_destroy(context);
        thm_sparse_wdm_triplet_free(&wdm_tracks);
        free(params);
        free(wdms);
        return status != 0;
    }

    Nc = (int)(200.0*Tobs/SECSYR);
    if(Nc < 20) Nc = 20;
    dtx = Tobs/(double)(Nc-1);
    if(dtx < CONSTELLATION_MIN_PADDING_SECONDS) dtx = CONSTELLATION_MIN_PADDING_SECONDS;
    dtc = (Tobs+2.0*dtx)/(double)(Nc-1);

    printf("THM TDI-WDM run diagnostics %s\n",
           thm_diagnostics_enabled ? "on" : "off");
    printf("experimental_split_tdi_response %d delay_zero_grid %d\n",
           options.split_tdi_response,
           thm_intrinsic_tdi_include_delay_zero_grid);
    printf("spa_t1t2_correction %d max_abs_phase_correction_rad %.6e\n",
           thm_spa_t1t2_correction_enabled,
           (double)SPA_T1T2_MAX_PHASE_CORRECTION);
    printf("prewhiten_instrument %d prewhiten_average_halfwidth %d prewhiten_smooth_halfwidth_hz %.15e prewhiten_smooth_frac %.15e prewhiten_smooth_floor_hz %.15e\n",
           options.prewhiten_instrument,
           options.prewhiten_average_halfwidth,
           options.prewhiten_smooth_halfwidth_hz,
           options.prewhiten_smooth_frac,
           options.prewhiten_smooth_floor_hz);
    printf("wdm_grid nt %d nf %d dt %.15e Tobs %.15e\n", Nt, Nf, (double)dt, Tobs);
    printf("constellation_kappa0 %.15e equivalent_shift_days %.9f\n",
           thm_orbit_phase_offset,
           thm_orbit_phase_offset/(2.0*M_PI)*SECSYR/86400.0);
    printf("constellation padding %e\n", dtx);
    printf("m1 %.15e m2 %.15e eta %.15e tc %.15e\n", m1/TSUN, m2/TSUN, eta, tc);
    printf("mode_selection %s nmodes %d", mode_spec == NULL ? "default" : mode_spec, requested_nmodes);
    for(i=0; i<requested_nmodes; i++)
    {
        printf(" (%d,%+d)", requested_modes[i].ell, requested_modes[i].emm);
    }
    printf("\n");

    Larray = double_matrix(3, Nc);
    Parray = double_tensor(3, 3, Nc);
    Varray = double_tensor(3, 3, Nc);
    tarray = double_vector(Nc);
    for(i=0; i<Nc; i++) tarray[i] = -dtx + dtc*(double)i;
    constellation_tmin = tarray[0];
    constellation_tmax = tarray[Nc-1];
    constellation(Nc, tarray, Larray, Parray, Varray);

    gsl_interp_accel **SLacc = malloc(3*sizeof(gsl_interp_accel *));
    gsl_spline **SLspline = malloc(3*sizeof(gsl_spline *));
    gsl_interp_accel **SPacc = malloc(9*sizeof(gsl_interp_accel *));
    gsl_spline **SPspline = malloc(9*sizeof(gsl_spline *));
    gsl_interp_accel **SVacc = malloc(9*sizeof(gsl_interp_accel *));
    gsl_spline **SVspline = malloc(9*sizeof(gsl_spline *));
    if(SLacc == NULL || SLspline == NULL || SPacc == NULL || SPspline == NULL ||
       SVacc == NULL || SVspline == NULL)
    {
        fprintf(stderr, "allocation failure\n");
        return 1;
    }
    for(i=0; i<3; i++)
    {
        SLacc[i] = gsl_interp_accel_alloc();
        SLspline[i] = gsl_spline_alloc(gsl_interp_cspline, Nc);
        gsl_spline_init(SLspline[i], tarray, Larray[i], Nc);
    }
    for(i=0; i<9; i++)
    {
        SPacc[i] = gsl_interp_accel_alloc();
        SPspline[i] = gsl_spline_alloc(gsl_interp_cspline, Nc);
        SVacc[i] = gsl_interp_accel_alloc();
        SVspline[i] = gsl_spline_alloc(gsl_interp_cspline, Nc);
    }
    for(i=0; i<3; i++)
    {
        for(j=0; j<3; j++)
        {
            k = j+i*3;
            gsl_spline_init(SPspline[k], tarray, Parray[i][j], Nc);
            gsl_spline_init(SVspline[k], tarray, Varray[i][j], Nc);
        }
    }

    Nsmax = 10000;
    TS = double_vector(Nsmax);
    tspace = double_vector(Nsmax);
    tc_tau = double_vector(Nsmax);
    mode_amp = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
    mode_phase = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
    mode_freq = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);

    memset(&model, 0, sizeof(model));
    start = clock();
    if(options.use_intrinsic_tdi_ap_grid)
    {
        Ns = PhenomTHM_AP_IntrinsicTDIAdaptive(params, &model, modes, &nmodes,
                                               requested_modes, requested_nmodes,
                                               Nsmax, TS, tspace,
                                               mode_amp, mode_phase, mode_freq,
                                               SPacc, SPspline,
                                               constellation_tmin, constellation_tmax,
                                               0.0, Tobs);
        ap_grid_label = thm_intrinsic_tdi_include_delay_zero_grid ?
                        "intrinsic-tdi" : "intrinsic-tdi-no-delay-zero";
    }
    else if(options.use_detector_ap_grid)
    {
        Ns = PhenomTHM_AP_DetectorAdaptive(params, &model, modes, &nmodes,
                                           requested_modes, requested_nmodes,
                                           Nsmax, TS, tspace,
                                           mode_amp, mode_phase, mode_freq,
                                           SPacc, SPspline,
                                           constellation_tmin, constellation_tmax,
                                           0.0, Tobs);
        ap_grid_label = "detector";
    }
    else
    {
        Ns = PhenomTHM_AP(params, &model, modes, &nmodes,
                          requested_modes, requested_nmodes,
                          Nsmax, TS, tspace,
                          mode_amp, mode_phase, mode_freq,
                          SPacc, SPspline, constellation_tmin, constellation_tmax);
        ap_grid_label = "legacy";
    }
    end = clock();
    adaptive_ap_time = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("samples %d modes %d adaptive_ap %.6f grid %s\n",
           Ns, nmodes, adaptive_ap_time, ap_grid_label);
    if(options.use_intrinsic_tdi_ap_grid)
    {
        printf("intrinsic_tdi_ap model_samples %d response_samples %d interpolated_samples %d exact_samples %d detector_planned_samples %d switch_source %.6e switch_detector %.6e\n",
               thm_last_intrinsic_tdi_model_samples, Ns,
               thm_last_intrinsic_tdi_interpolated_samples,
               thm_last_intrinsic_tdi_exact_samples,
               thm_last_intrinsic_tdi_detector_planned_samples,
               thm_last_intrinsic_tdi_switch_source_time,
               thm_last_intrinsic_tdi_switch_detector_time);
    }

    max_mode_frequency_hz = 0.0;
    for(i=0; i<Ns; i++)
    {
        for(k=0; k<nmodes; k++)
        {
            if(fabs(mode_freq[k][i]) > max_mode_frequency_hz)
            {
                max_mode_frequency_hz = fabs(mode_freq[k][i]);
            }
        }
    }
    printf("max_mode_frequency %.15e dense_check_nyquist %.15e\n",
           max_mode_frequency_hz, 0.5/THM_TDI_DENSE_CHECK_DT_SECONDS);
    if(max_mode_frequency_hz > 0.5/THM_TDI_DENSE_CHECK_DT_SECONDS)
    {
        fprintf(stderr, "Warning: dense TDI check cadence %.15e undersamples the maximum mode frequency %.15e Hz.\n",
                THM_TDI_DENSE_CHECK_DT_SECONDS, max_mode_frequency_hz);
    }

    for(i=0; i<Ns; i++) tc_tau[i] = (TS[i]-tc)/Mtot;
    if(thm_diagnostics_enabled)
    {
        write_thm_mode_ap("PhenomTHM_modes_AP.dat", Ns, nmodes, modes, TS, tc_tau,
                          mode_amp, mode_phase, mode_freq);
    }

    if(options.compare_ap_grid)
    {
        IMRPhenomTHM model_detector;
        IMRPhenomTHMMode modes_detector[IMRPHENOMTHM_MAX_MODES];
        int nmodes_detector = 0;
        int Ns_detector;
        double *TS_detector = double_vector(Nsmax);
        double *tspace_detector = double_vector(Nsmax);
        double *tc_tau_detector = double_vector(Nsmax);
        double **mode_amp_detector = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
        double **mode_phase_detector = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
        double **mode_freq_detector = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
        clock_t compare_start, compare_end;
        double detector_ap_time;

        if(TS_detector == NULL || tspace_detector == NULL ||
           tc_tau_detector == NULL || mode_amp_detector == NULL ||
           mode_phase_detector == NULL || mode_freq_detector == NULL)
        {
            fprintf(stderr, "allocation failure in AP grid comparison\n");
            return 1;
        }

        memset(&model_detector, 0, sizeof(model_detector));
        compare_start = clock();
        Ns_detector = PhenomTHM_AP_DetectorAdaptive(params, &model_detector,
                                                    modes_detector, &nmodes_detector,
                                                    requested_modes, requested_nmodes,
                                                    Nsmax, TS_detector, tspace_detector,
                                                    mode_amp_detector, mode_phase_detector,
                                                    mode_freq_detector,
                                                    SPacc, SPspline,
                                                    constellation_tmin, constellation_tmax,
                                                    0.0, Tobs);
        compare_end = clock();
        detector_ap_time = ((double)(compare_end-compare_start))/CLOCKS_PER_SEC;

        for(i=0; i<Ns_detector; i++) tc_tau_detector[i] = (TS_detector[i]-tc)/Mtot;
        write_thm_mode_ap("PhenomTHM_modes_AP_detector.dat",
                          Ns_detector, nmodes_detector, modes_detector,
                          TS_detector, tc_tau_detector,
                          mode_amp_detector, mode_phase_detector,
                          mode_freq_detector);
        {
            FILE *grid = fopen("PhenomTHM_AP_detector_grid.dat", "w");
            if(grid != NULL)
            {
                fprintf(grid, "# i detector_time_s source_time_s tau dt_detector_s dt_source_s\n");
                for(i=0; i<Ns_detector; i++)
                {
                    double dt_detector = (i > 0) ? tspace_detector[i]-tspace_detector[i-1] : 0.0;
                    double dt_source = (i > 0) ? TS_detector[i]-TS_detector[i-1] : 0.0;
                    fprintf(grid, "%d %.15e %.15e %.15e %.15e %.15e\n",
                            i, tspace_detector[i], TS_detector[i],
                            tc_tau_detector[i], dt_detector, dt_source);
                }
                fclose(grid);
            }
            grid = fopen("PhenomTHM_AP_primary_grid.dat", "w");
            if(grid != NULL)
            {
                fprintf(grid, "# i primary_grid_time_s source_time_s tau dt_primary_s dt_source_s\n");
                for(i=0; i<Ns; i++)
                {
                    double dt_seed = (i > 0) ? tspace[i]-tspace[i-1] : 0.0;
                    double dt_source = (i > 0) ? TS[i]-TS[i-1] : 0.0;
                    fprintf(grid, "%d %.15e %.15e %.15e %.15e %.15e\n",
                            i, tspace[i], TS[i], tc_tau[i], dt_seed, dt_source);
                }
                fclose(grid);
            }
        }

        printf("ap_grid_compare modes %d primary_samples %d detector_samples %d ratio %.6f detector_ap %.6f\n",
               nmodes_detector, Ns, Ns_detector,
               Ns > 0 ? ((double)Ns_detector)/((double)Ns) : 0.0,
               detector_ap_time);
        printf("wrote PhenomTHM_modes_AP_detector.dat PhenomTHM_AP_detector_grid.dat PhenomTHM_AP_primary_grid.dat\n");

        IMRPhenomTHMDestroy(&model_detector);
        free_double_vector(TS_detector);
        free_double_vector(tspace_detector);
        free_double_vector(tc_tau_detector);
        free_double_matrix(mode_amp_detector, IMRPHENOMTHM_MAX_MODES);
        free_double_matrix(mode_phase_detector, IMRPHENOMTHM_MAX_MODES);
        free_double_matrix(mode_freq_detector, IMRPHENOMTHM_MAX_MODES);
    }

    if(options.compare_per_mode_grid)
    {
        FILE *grid_summary = fopen("PhenomTHM_AP_per_mode_grid_summary.dat", "w");
        FILE *grid_points = fopen("PhenomTHM_AP_per_mode_grids.dat", "w");

        if(grid_summary != NULL)
        {
            fprintf(grid_summary, "# mode_index ell m samples shared_samples ratio min_dt_detector max_dt_detector min_dt_source max_dt_source detector_ap_seconds\n");
        }
        if(grid_points != NULL)
        {
            fprintf(grid_points, "# mode_index ell m sample_index detector_time_s source_time_s dt_detector_s dt_source_s\n");
        }

        for(k=0; k<requested_nmodes; k++)
        {
            IMRPhenomTHM model_single;
            IMRPhenomTHMMode mode_single_request[1];
            IMRPhenomTHMMode modes_single[IMRPHENOMTHM_MAX_MODES];
            int nmodes_single = 0;
            int Ns_single;
            double *TS_single = double_vector(Nsmax);
            double *tspace_single = double_vector(Nsmax);
            double **mode_amp_single = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
            double **mode_phase_single = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
            double **mode_freq_single = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
            double min_dt_detector = HUGE_VAL;
            double max_dt_detector = 0.0;
            double min_dt_source = HUGE_VAL;
            double max_dt_source = 0.0;
            double detector_ap_seconds;
            char spacing_filename[128];
            clock_t single_start, single_end;
            int q;

            if(TS_single == NULL || tspace_single == NULL ||
               mode_amp_single == NULL || mode_phase_single == NULL ||
               mode_freq_single == NULL)
            {
                fprintf(stderr, "allocation failure in per-mode AP grid comparison\n");
                return 1;
            }

            mode_single_request[0] = requested_modes[k];
            memset(&model_single, 0, sizeof(model_single));
            snprintf(spacing_filename, sizeof(spacing_filename),
                     "PhenomTHM_AP_spacing_reasons_mode%d%+d.dat",
                     requested_modes[k].ell, requested_modes[k].emm);
            thm_ap_spacing_diagnostic_filename = spacing_filename;
            single_start = clock();
            Ns_single = PhenomTHM_AP_DetectorAdaptive(params, &model_single,
                                                      modes_single, &nmodes_single,
                                                      mode_single_request, 1,
                                                      Nsmax, TS_single, tspace_single,
                                                      mode_amp_single, mode_phase_single,
                                                      mode_freq_single,
                                                      SPacc, SPspline,
                                                      constellation_tmin, constellation_tmax,
                                                      0.0, Tobs);
            single_end = clock();
            thm_ap_spacing_diagnostic_filename = "PhenomTHM_AP_spacing_reasons.dat";
            detector_ap_seconds = ((double)(single_end-single_start))/CLOCKS_PER_SEC;

            for(q=1; q<Ns_single; q++)
            {
                double dt_detector = tspace_single[q]-tspace_single[q-1];
                double dt_source = TS_single[q]-TS_single[q-1];
                if(dt_detector < min_dt_detector) min_dt_detector = dt_detector;
                if(dt_detector > max_dt_detector) max_dt_detector = dt_detector;
                if(dt_source < min_dt_source) min_dt_source = dt_source;
                if(dt_source > max_dt_source) max_dt_source = dt_source;
            }
            if(Ns_single < 2)
            {
                min_dt_detector = 0.0;
                min_dt_source = 0.0;
            }

            if(grid_summary != NULL)
            {
                fprintf(grid_summary, "%d %d %+d %d %d %.15e %.15e %.15e %.15e %.15e %.15e\n",
                        k, requested_modes[k].ell, requested_modes[k].emm,
                        Ns_single, Ns,
                        Ns > 0 ? ((double)Ns_single)/((double)Ns) : 0.0,
                        min_dt_detector, max_dt_detector,
                        min_dt_source, max_dt_source,
                        detector_ap_seconds);
            }
            if(grid_points != NULL)
            {
                for(q=0; q<Ns_single; q++)
                {
                    double dt_detector = (q > 0) ? tspace_single[q]-tspace_single[q-1] : 0.0;
                    double dt_source = (q > 0) ? TS_single[q]-TS_single[q-1] : 0.0;
                    fprintf(grid_points, "%d %d %+d %d %.15e %.15e %.15e %.15e\n",
                            k, requested_modes[k].ell, requested_modes[k].emm,
                            q, tspace_single[q], TS_single[q],
                            dt_detector, dt_source);
                }
            }

            printf("per_mode_ap_grid (%d,%+d) samples %d shared_samples %d ratio %.6f detector_ap %.6f\n",
                   requested_modes[k].ell, requested_modes[k].emm, Ns_single, Ns,
                   Ns > 0 ? ((double)Ns_single)/((double)Ns) : 0.0,
                   detector_ap_seconds);

            IMRPhenomTHMDestroy(&model_single);
            free_double_vector(TS_single);
            free_double_vector(tspace_single);
            free_double_matrix(mode_amp_single, IMRPHENOMTHM_MAX_MODES);
            free_double_matrix(mode_phase_single, IMRPHENOMTHM_MAX_MODES);
            free_double_matrix(mode_freq_single, IMRPHENOMTHM_MAX_MODES);
        }

        if(grid_summary != NULL) fclose(grid_summary);
        if(grid_points != NULL) fclose(grid_points);
        printf("wrote PhenomTHM_AP_per_mode_grid_summary.dat PhenomTHM_AP_per_mode_grids.dat\n");
    }

    if(options.write_intrinsic_ap_grid)
    {
        IMRPhenomTHM model_intrinsic;
        IMRPhenomTHMMode modes_intrinsic[IMRPHENOMTHM_MAX_MODES];
        int nmodes_intrinsic = 0;
        int Ns_intrinsic;
        double *TS_intrinsic = double_vector(Nsmax);
        double *tspace_intrinsic = double_vector(Nsmax);
        double *tc_tau_intrinsic = double_vector(Nsmax);
        double **mode_amp_intrinsic = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
        double **mode_phase_intrinsic = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
        double **mode_freq_intrinsic = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
        clock_t intrinsic_start, intrinsic_end;
        double intrinsic_ap_time;

        if(TS_intrinsic == NULL || tspace_intrinsic == NULL ||
           tc_tau_intrinsic == NULL || mode_amp_intrinsic == NULL ||
           mode_phase_intrinsic == NULL || mode_freq_intrinsic == NULL)
        {
            fprintf(stderr, "allocation failure in intrinsic AP grid diagnostic\n");
            return 1;
        }

        memset(&model_intrinsic, 0, sizeof(model_intrinsic));
        intrinsic_start = clock();
        Ns_intrinsic = PhenomTHM_AP_IntrinsicAdaptive(params, &model_intrinsic,
                                                      modes_intrinsic,
                                                      &nmodes_intrinsic,
                                                      requested_modes,
                                                      requested_nmodes,
                                                      Nsmax, TS_intrinsic,
                                                      tspace_intrinsic,
                                                      mode_amp_intrinsic,
                                                      mode_phase_intrinsic,
                                                      mode_freq_intrinsic,
                                                      0.0, Tobs);
        intrinsic_end = clock();
        intrinsic_ap_time = ((double)(intrinsic_end-intrinsic_start))/CLOCKS_PER_SEC;

        for(i=0; i<Ns_intrinsic; i++)
        {
            tc_tau_intrinsic[i] = (TS_intrinsic[i]-tc)/Mtot;
        }
        write_thm_mode_ap("PhenomTHM_modes_AP_intrinsic.dat",
                          Ns_intrinsic, nmodes_intrinsic, modes_intrinsic,
                          TS_intrinsic, tc_tau_intrinsic,
                          mode_amp_intrinsic, mode_phase_intrinsic,
                          mode_freq_intrinsic);
        {
            FILE *grid = fopen("PhenomTHM_AP_intrinsic_grid.dat", "w");
            if(grid != NULL)
            {
                fprintf(grid, "# i source_time_s tau dt_source_s\n");
                for(i=0; i<Ns_intrinsic; i++)
                {
                    double dt_source = (i > 0) ? TS_intrinsic[i]-TS_intrinsic[i-1] : 0.0;
                    fprintf(grid, "%d %.15e %.15e %.15e\n",
                            i, TS_intrinsic[i], tc_tau_intrinsic[i], dt_source);
                }
                fclose(grid);
            }
        }

        printf("intrinsic_ap_grid modes %d samples %d detector_samples %d ratio %.6f intrinsic_ap %.6f\n",
               nmodes_intrinsic, Ns_intrinsic, Ns,
               Ns > 0 ? ((double)Ns_intrinsic)/((double)Ns) : 0.0,
               intrinsic_ap_time);
        printf("wrote PhenomTHM_modes_AP_intrinsic.dat PhenomTHM_AP_intrinsic_grid.dat PhenomTHM_intrinsic_AP_spacing_reasons.dat\n");

        IMRPhenomTHMDestroy(&model_intrinsic);
        free_double_vector(TS_intrinsic);
        free_double_vector(tspace_intrinsic);
        free_double_vector(tc_tau_intrinsic);
        free_double_matrix(mode_amp_intrinsic, IMRPHENOMTHM_MAX_MODES);
        free_double_matrix(mode_phase_intrinsic, IMRPHENOMTHM_MAX_MODES);
        free_double_matrix(mode_freq_intrinsic, IMRPHENOMTHM_MAX_MODES);
    }

    Aacc = malloc((size_t)nmodes*sizeof(gsl_interp_accel *));
    Pacc = malloc((size_t)nmodes*sizeof(gsl_interp_accel *));
    Aspline = malloc((size_t)nmodes*sizeof(gsl_spline *));
    Pspline = malloc((size_t)nmodes*sizeof(gsl_spline *));
    if(Aacc == NULL || Pacc == NULL || Aspline == NULL || Pspline == NULL)
    {
        fprintf(stderr, "allocation failure\n");
        return 1;
    }

    /*
     * LAL's SimAddMode comparison used phi=pi/2 for phiRef=0.  Keep that
     * convention here.  The coalescence/reference phase is already in the mode
     * phases, so changing phiRef later should be done in one place only.
     */
    for(k=0; k<nmodes; k++)
    {
        thm_projection_coefficients(modes[k].ell, modes[k].emm, params[10],
                                    0.5*M_PI, params[9], &projection[k]);
        Aacc[k] = gsl_interp_accel_alloc();
        Pacc[k] = gsl_interp_accel_alloc();
        Aspline[k] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
        Pspline[k] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
        gsl_spline_init(Aspline[k], TS, mode_amp[k], Ns);
        gsl_spline_init(Pspline[k], TS, mode_phase[k], Ns);
    }
    ncarriers = thm_build_folded_carriers(nmodes, modes, projection,
                                          carrier, IMRPHENOMTHM_MAX_MODES);
    if(ncarriers < 1)
    {
        fprintf(stderr, "Error: no THM carriers available for TDI response.\n");
        return 1;
    }
#if THM_TDI_USE_FOLDED_CARRIERS
    printf("tdi_carriers %d folded_from_modes %d\n", ncarriers, nmodes);
#else
    printf("tdi_carriers %d explicit_modes %d\n", ncarriers, nmodes);
#endif

    /*
     * tspace is the detector-center retarded waveform time
     * u=t_SSB-k.r_0(t_SSB).  barycenter_time() inverts that relation to obtain
     * the public SSB output timestamps TS.  The TDI response is evaluated at
     * TS and applies the individual spacecraft projections and arm delays
     * internally.  Keep these two coordinates distinct: the physical response
     * merger is near u=tc, while an intrinsic spline knot labelled TS reaches
     * tc at a generally different grid point.
     */
    response_time = TS;
    trim_time = tspace;
    Ns_response = Ns;
    i = 0;
    do
    {
        i++;
    } while(i < Ns && trim_time[Ns-i] > TS[Ns-1]);
    Ns_response = Ns-i;
    if(Ns_response < 1)
    {
        fprintf(stderr, "Error: no TDI-safe THM samples remain after delayed-time trimming.\n");
        return 1;
    }
    printf("tdi_safe_samples %d trimmed %d last_spline_time %.15e last_response_time %.15e\n",
           Ns_response, Ns-Ns_response, TS[Ns-1], response_time[Ns_response-1]);
    thm_response_spline_tmin = TS[0];
    thm_response_spline_tmax = TS[Ns-1];

    X = double_vector(Ns_response);
    Y = double_vector(Ns_response);
    Z = double_vector(Ns_response);
    Xf = double_vector(Ns_response);
    Yf = double_vector(Ns_response);
    Zf = double_vector(Ns_response);

    start = clock();
    fast_response_thm(response_time, Ns_response, params, ncarriers, carrier,
                      SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                      Aacc, Aspline, Pacc, Pspline,
                      X, Y, Z, Xf, Yf, Zf);
    end = clock();
    tdi_time = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("multi_mode_tdi %.6f\n", tdi_time);
    if(thm_diagnostics_enabled)
    {
        write_thm_tdi_time("PhenomTHM_TDI_time.dat", Ns_response, response_time, X, Y, Z, Xf, Yf, Zf);
    }
    if(options.wdm_mode_enabled)
    {
        wdm_carrier_index = find_thm_carrier(ncarriers, carrier,
                                             options.wdm_ell,
                                             options.wdm_abs_emm);
        if(wdm_carrier_index < 0)
        {
            fprintf(stderr, "Error: requested WDM diagnostic mode (%d,%d) is not present in the folded carrier list.\n",
                    options.wdm_ell, options.wdm_abs_emm);
            return 1;
        }
        write_thm_single_carrier_wdm_diagnostic(wdm_carrier_index, Ns_response,
                                                response_time, tspace,
                                                mode_freq,
                                                params, &model, carrier,
                                                SLacc, SLspline, SPacc, SPspline,
                                                SVacc, SVspline,
                                                Aacc, Aspline, Pacc, Pspline,
                                                wdms,
                                                options.wdm_join_time_enabled,
                                                options.wdm_join_time,
                                                options.wdm_rise_seconds);
    }
    if(options.wdm_all_enabled)
    {
        generate_thm_all_carrier_wdm_combined_fft(Ns_response,
                                                  response_time, tspace,
                                                  mode_freq,
                                                  params, &model, ncarriers, carrier,
                                                  SLacc, SLspline, SPacc, SPspline,
                                                  SVacc, SVspline,
                                                  Aacc, Aspline, Pacc, Pspline,
                                                  wdms,
                                                  options.wdm_join_time_enabled,
                                                  options.wdm_join_time,
                                                  options.wdm_rise_seconds,
                                                  options.wdm_spline_endpoint,
                                                  options.wdm_partition_endpoint,
                                                  options.wdm_split_fft,
                                                  options.wdm_split_early_fft,
                                                  options.split_tdi_response,
                                                  &wdm_tracks,
                                                  NULL, 0.0, 0.0,
                                                  options.write_sparse_files,
                                                  options.write_time_domain_files,
                                                  NULL, NULL, NULL);
    }
    if(thm_diagnostics_enabled)
    {
        write_thm_tdi_dense_check("PhenomTHM_TDI_dense_check.dat",
                                  "PhenomTHM_TDI_dense_check_summary.dat",
                                  THM_TDI_DENSE_CHECK_DT_SECONDS,
                                  THM_TDI_DENSE_CHECK_CHUNK_SECONDS,
                                  tc, Ns_response, response_time,
                                  params, ncarriers, carrier,
                                  SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                                  Aacc, Aspline, Pacc, Pspline);
        printf("diagnostic_outputs wrote PhenomTHM_modes_AP.dat PhenomTHM_TDI_time.dat PhenomTHM_TDI_dense_check.dat PhenomTHM_TDI_dense_check_summary.dat\n");
    }

    for(k=0; k<nmodes; k++)
    {
        gsl_spline_free(Aspline[k]);
        gsl_spline_free(Pspline[k]);
        gsl_interp_accel_free(Aacc[k]);
        gsl_interp_accel_free(Pacc[k]);
    }
    free(Aacc);
    free(Pacc);
    free(Aspline);
    free(Pspline);

    for(i=0; i<3; i++)
    {
        gsl_spline_free(SLspline[i]);
        gsl_interp_accel_free(SLacc[i]);
    }
    for(i=0; i<9; i++)
    {
        gsl_spline_free(SPspline[i]);
        gsl_interp_accel_free(SPacc[i]);
        gsl_spline_free(SVspline[i]);
        gsl_interp_accel_free(SVacc[i]);
    }
    free(SLacc);
    free(SLspline);
    free(SPacc);
    free(SPspline);
    free(SVacc);
    free(SVspline);

    free_double_vector(X);
    free_double_vector(Y);
    free_double_vector(Z);
    free_double_vector(Xf);
    free_double_vector(Yf);
    free_double_vector(Zf);
    free_double_vector(TS);
    free_double_vector(tspace);
    free_double_vector(tc_tau);
    free_double_matrix(mode_amp, IMRPHENOMTHM_MAX_MODES);
    free_double_matrix(mode_phase, IMRPHENOMTHM_MAX_MODES);
    free_double_matrix(mode_freq, IMRPHENOMTHM_MAX_MODES);
    free_double_matrix(Larray, 3);
    free_double_tensor(Parray, 3, 3);
    free_double_tensor(Varray, 3, 3);
    free_double_vector(tarray);
    free(params);
    thm_sparse_wdm_triplet_free(&wdm_tracks);
    free(wdms);
    IMRPhenomTHMDestroy(&model);

    (void)N;
    return 0;
}
#endif

static void thm_write_fourier_ap_grid_diagnostic(
    const char *prefix,
    double *params,
    int nmodes,
    const IMRPhenomTHMMode *modes,
    int Ns,
    int Ns_response,
    const double *TS,
    const double *tspace,
    double **mode_amp,
    double **mode_phase,
    double **mode_freq,
    gsl_interp_accel **SPacc,
    gsl_spline **SPspline,
    gsl_interp_accel **Aacc,
    gsl_spline **Aspline,
    gsl_interp_accel **Pacc,
    gsl_spline **Pspline)
{
    const double dense_dt = 1000.0;
    char path[640];
    FILE *out;
    gsl_interp_accel *Facc;
    gsl_spline *Fspline;
    int i, k22;

    if(prefix == NULL || prefix[0] == '\0' || Ns < 4 || Ns_response < 2)
    {
        return;
    }

    k22 = -1;
    for(i=0; i<nmodes; i++)
    {
        if(modes[i].ell == 2 && abs(modes[i].emm) == 2)
        {
            k22 = i;
            break;
        }
    }
    if(k22 < 0) return;

    snprintf(path, sizeof(path), "%s_knots.dat", prefix);
    out = fopen(path, "w");
    if(out != NULL)
    {
        fprintf(out, "# Final pre-TDI intrinsic 22 AP spline knots.\n");
        fprintf(out, "# columns: i detector_time_s detector_time_minus_tc_s source_time_s source_time_minus_tc_s dt_detector_s dt_source_s amplitude phase_rad frequency_Hz\n");
        for(i=0; i<Ns; i++)
        {
            double dtd = i > 0 ? tspace[i]-tspace[i-1] : 0.0;
            double dts = i > 0 ? TS[i]-TS[i-1] : 0.0;
            fprintf(out,
                    "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    i, tspace[i], tspace[i]-params[5], TS[i],
                    TS[i]-params[5], dtd, dts, mode_amp[k22][i],
                    mode_phase[k22][i], mode_freq[k22][i]);
        }
        fclose(out);
    }

    Facc = gsl_interp_accel_alloc();
    Fspline = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
    if(Facc == NULL || Fspline == NULL)
    {
        if(Fspline != NULL) gsl_spline_free(Fspline);
        if(Facc != NULL) gsl_interp_accel_free(Facc);
        return;
    }
    gsl_spline_init(Fspline, TS, mode_freq[k22], Ns);
    gsl_interp_accel_reset(Aacc[k22]);
    gsl_interp_accel_reset(Pacc[k22]);

    snprintf(path, sizeof(path), "%s_dense.dat", prefix);
    out = fopen(path, "w");
    if(out != NULL)
    {
        double detector_start = tspace[0];
        double detector_stop = tspace[Ns_response-1];
        int ndense = (int)floor((detector_stop-detector_start)/dense_dt)+1;

        fprintf(out, "# Pre-TDI intrinsic 22 AP splines evaluated every %.15e s in detector time.\n",
                dense_dt);
        fprintf(out, "# columns: detector_time_s detector_time_minus_tc_s source_time_s source_time_minus_tc_s amplitude_spline phase_spline_rad frequency_spline_Hz\n");
        for(i=0; i<ndense; i++)
        {
            double td = detector_start+dense_dt*(double)i;
            double ts = thm_ssb_output_time_from_center_source_time(
                td, params, SPacc, SPspline);
            double amp, phase, freq;

            if(ts < TS[0] || ts > TS[Ns-1]) continue;
            amp = gsl_spline_eval(Aspline[k22], ts, Aacc[k22]);
            phase = gsl_spline_eval(Pspline[k22], ts, Pacc[k22]);
            freq = gsl_spline_eval(Fspline, ts, Facc);
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    td, td-params[5], ts, ts-params[5], amp, phase, freq);
        }
        fclose(out);
    }

    snprintf(path, sizeof(path), "%s_merger_dense.dat", prefix);
    out = fopen(path, "w");
    if(out != NULL)
    {
        const double zoom_dt = 1.0;
        double detector_start = fmax(tspace[0], params[5]-2000.0);
        double detector_stop = fmin(tspace[Ns_response-1], params[5]+2000.0);
        int ndense = (int)floor(
            (detector_stop-detector_start)/zoom_dt)+1;

        fprintf(out, "# Pre-TDI intrinsic 22 AP merger zoom evaluated every %.15e s in detector time.\n",
                zoom_dt);
        fprintf(out, "# columns: detector_time_s detector_time_minus_tc_s source_time_s source_time_minus_tc_s amplitude_spline phase_spline_rad frequency_spline_Hz\n");
        for(i=0; i<ndense; i++)
        {
            double td = detector_start+zoom_dt*(double)i;
            double ts = thm_ssb_output_time_from_center_source_time(
                td, params, SPacc, SPspline);
            double amp, phase, freq;

            if(ts < TS[0] || ts > TS[Ns-1]) continue;
            amp = gsl_spline_eval(Aspline[k22], ts, Aacc[k22]);
            phase = gsl_spline_eval(Pspline[k22], ts, Pacc[k22]);
            freq = gsl_spline_eval(Fspline, ts, Facc);
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    td, td-params[5], ts, ts-params[5], amp, phase, freq);
        }
        fclose(out);
    }

    gsl_spline_free(Fspline);
    gsl_interp_accel_free(Facc);
}

static void thm_reusable_wdm_work_release(THMReusableWDMWork *work)
{
    if(work == NULL) return;
    free_int_vector(work->nmid);
    free_int_vector(work->nsize);
    free_int_vector(work->nmid_endpoint);
    free_int_vector(work->nsize_endpoint);
    free_int_vector(work->listn);
    free_int_vector(work->listm);
    free_double_vector(work->wdmwave);
    if(work->wdm != NULL) free_double_matrix(work->wdm, Nt);
    free_double_vector(work->freq);
    free_double_vector(work->phase);
    free_double_vector(work->amp);
    free_double_vector(work->ataper);
    free_double_vector(work->hfull);
    memset(work, 0, sizeof(*work));
}

static int thm_reusable_wdm_work_prepare(THMReusableWDMWork *work,
                                         int N, int nts_needed, int ns_needed)
{
    if(work == NULL || N < 1 || nts_needed < 1 || ns_needed < 1) return 0;

    if(work->N != N || work->nmid == NULL || work->nsize == NULL ||
       work->nmid_endpoint == NULL || work->nsize_endpoint == NULL ||
       work->listn == NULL || work->listm == NULL || work->wdmwave == NULL ||
       work->wdm == NULL || work->hfull == NULL)
    {
        thm_reusable_wdm_work_release(work);
        work->nmid = int_vector(Nf);
        work->nsize = int_vector(Nf);
        work->nmid_endpoint = int_vector(Nf);
        work->nsize_endpoint = int_vector(Nf);
        work->listn = int_vector(N);
        work->listm = int_vector(N);
        work->wdmwave = double_vector(N);
        work->wdm = double_matrix(Nt, Nf+1);
        work->hfull = double_vector(N);
        work->N = N;
        if(work->nmid == NULL || work->nsize == NULL ||
           work->nmid_endpoint == NULL || work->nsize_endpoint == NULL ||
           work->listn == NULL || work->listm == NULL ||
           work->wdmwave == NULL || work->wdm == NULL || work->hfull == NULL)
        {
            thm_reusable_wdm_work_release(work);
            return 0;
        }
    }

    if(work->nts_capacity < nts_needed || work->freq == NULL ||
       work->phase == NULL || work->amp == NULL)
    {
        int new_capacity = nts_needed;
        double *freq = double_vector(new_capacity);
        double *phase = double_vector(new_capacity);
        double *amp = double_vector(new_capacity);

        if(freq == NULL || phase == NULL || amp == NULL)
        {
            free_double_vector(freq);
            free_double_vector(phase);
            free_double_vector(amp);
            return 0;
        }
        free_double_vector(work->freq);
        free_double_vector(work->phase);
        free_double_vector(work->amp);
        work->freq = freq;
        work->phase = phase;
        work->amp = amp;
        work->nts_capacity = new_capacity;
    }

    if(work->ns_capacity < ns_needed || work->ataper == NULL)
    {
        double *ataper = double_vector(ns_needed);
        if(ataper == NULL) return 0;
        free_double_vector(work->ataper);
        work->ataper = ataper;
        work->ns_capacity = ns_needed;
    }

    return 1;
}

static int thm_observation_context_build_splines(THMObservationContext *context)
{
    int i, j, k;

    for(i=0; i<3; i++)
    {
        context->SLspline[i] = gsl_spline_alloc(gsl_interp_cspline,
                                                context->Nc);
        if(context->SLspline[i] == NULL ||
           gsl_spline_init(context->SLspline[i], context->tarray,
                           context->Larray[i], context->Nc) != GSL_SUCCESS)
            return 1;
    }
    for(i=0; i<3; i++)
        for(j=0; j<3; j++)
        {
            k = j+i*3;
            context->SPspline[k] = gsl_spline_alloc(gsl_interp_cspline,
                                                    context->Nc);
            context->SVspline[k] = gsl_spline_alloc(gsl_interp_cspline,
                                                    context->Nc);
            if(context->SPspline[k] == NULL || context->SVspline[k] == NULL ||
               gsl_spline_init(context->SPspline[k], context->tarray,
                               context->Parray[i][j], context->Nc) != GSL_SUCCESS ||
               gsl_spline_init(context->SVspline[k], context->tarray,
                               context->Varray[i][j], context->Nc) != GSL_SUCCESS)
                return 1;
        }
    return 0;
}

THMObservationContext *thm_observation_context_create(double kappa0)
{
    THMObservationContext *context;
    double Tobs, dtx, dtc, saved_kappa0;
    int i;

    if(!isfinite(kappa0)) return NULL;
    context = calloc(1, sizeof(*context));
    if(context == NULL) return NULL;

    wdmvalues(&context->wdms);
    Tobs = context->wdms.Tobs;
    context->kappa0 = fmod(kappa0, 2.0*M_PI);
    context->tdi_generation = thm_tdi_generation;
    context->tdi2_frozen_projection = thm_tdi2_frozen_projection;
    context->tdi2_chain_taylor = thm_tdi2_chain_taylor;
    context->tdi2_light_time_solver = thm_tdi2_light_time_solver;
    if(context->kappa0 < 0.0) context->kappa0 += 2.0*M_PI;
    context->Nc = (int)(200.0*Tobs/SECSYR);
    if(context->Nc < 20) context->Nc = 20;
    dtx = Tobs/(double)(context->Nc-1);
    if(dtx < CONSTELLATION_MIN_PADDING_SECONDS)
        dtx = CONSTELLATION_MIN_PADDING_SECONDS;
    dtc = (Tobs+2.0*dtx)/(double)(context->Nc-1);

    context->Larray = double_matrix(3, context->Nc);
    context->Parray = double_tensor(3, 3, context->Nc);
    context->Varray = double_tensor(3, 3, context->Nc);
    context->tarray = double_vector(context->Nc);
    if(context->Larray == NULL || context->Parray == NULL ||
       context->Varray == NULL || context->tarray == NULL)
    {
        thm_observation_context_destroy(context);
        return NULL;
    }

    for(i=0; i<context->Nc; i++) context->tarray[i] = -dtx+dtc*(double)i;
    context->constellation_tmin = context->tarray[0];
    context->constellation_tmax = context->tarray[context->Nc-1];

    /* spacecraft() uses the legacy process-global phase; capture it here. */
    saved_kappa0 = thm_orbit_phase_offset;
    thm_orbit_phase_offset = context->kappa0;
    constellation(context->Nc, context->tarray, context->Larray,
                  context->Parray, context->Varray);
    thm_orbit_phase_offset = saved_kappa0;

    if(thm_observation_context_build_splines(context) != 0)
    {
        thm_observation_context_destroy(context);
        return NULL;
    }
    return context;
}

THMObservationContext *thm_observation_context_create_from_orbit(
    int samples, const double *time, const double *position_m)
{
    THMObservationContext *context;
    int i, j, k;

    if(samples < 4 || time == NULL || position_m == NULL) return NULL;
    for(i=0; i<samples; i++)
    {
        if(!isfinite(time[i]) || (i > 0 && time[i] <= time[i-1])) return NULL;
        for(j=0; j<9; j++)
            if(!isfinite(position_m[9*i+j])) return NULL;
    }
    context = calloc(1, sizeof(*context));
    if(context == NULL) return NULL;
    wdmvalues(&context->wdms);
    context->tdi_generation = thm_tdi_generation;
    context->tdi2_frozen_projection = thm_tdi2_frozen_projection;
    context->tdi2_chain_taylor = thm_tdi2_chain_taylor;
    context->tdi2_light_time_solver = thm_tdi2_light_time_solver;
    context->Nc = samples;
    context->tarray = double_vector(samples);
    context->Larray = double_matrix(3, samples);
    context->Parray = double_tensor(3, 3, samples);
    context->Varray = double_tensor(3, 3, samples);
    if(context->tarray == NULL || context->Larray == NULL ||
       context->Parray == NULL || context->Varray == NULL)
        goto fail;
    for(i=0; i<samples; i++)
    {
        context->tarray[i] = time[i];
        for(j=0; j<3; j++)
            for(k=0; k<3; k++)
                context->Parray[j][k][i] =
                    position_m[9*i+3*j+k]/CLIGHT;
        for(k=0; k<3; k++)
        {
            context->Varray[0][k][i] =
                context->Parray[1][k][i]-context->Parray[2][k][i];
            context->Varray[1][k][i] =
                context->Parray[2][k][i]-context->Parray[0][k][i];
            context->Varray[2][k][i] =
                context->Parray[0][k][i]-context->Parray[1][k][i];
        }
        for(j=0; j<3; j++)
        {
            double length2 = 0.0;
            for(k=0; k<3; k++)
                length2 += context->Varray[j][k][i]*
                           context->Varray[j][k][i];
            context->Larray[j][i] = sqrt(length2);
            if(!(context->Larray[j][i] > 0.0)) goto fail;
            for(k=0; k<3; k++)
                context->Varray[j][k][i] /= context->Larray[j][i];
        }
    }
    context->constellation_tmin = time[0];
    context->constellation_tmax = time[samples-1];
    if(thm_observation_context_build_splines(context) != 0) goto fail;
    return context;

fail:
    thm_observation_context_destroy(context);
    return NULL;
}

void thm_observation_context_destroy(THMObservationContext *context)
{
    int i;

    if(context == NULL) return;
    for(i=0; i<3; i++)
    {
        if(context->SLspline[i] != NULL) gsl_spline_free(context->SLspline[i]);
    }
    for(i=0; i<9; i++)
    {
        if(context->SPspline[i] != NULL) gsl_spline_free(context->SPspline[i]);
        if(context->SVspline[i] != NULL) gsl_spline_free(context->SVspline[i]);
    }
    if(context->Larray != NULL) free_double_matrix(context->Larray, 3);
    if(context->Parray != NULL) free_double_tensor(context->Parray, 3, 3);
    if(context->Varray != NULL) free_double_tensor(context->Varray, 3, 3);
    free_double_vector(context->tarray);
    free(context);
}

int thm_observation_wdm_grid_info(const THMObservationContext *context,
                                  THMWDMGridInfo *info)
{
    if(context == NULL || info == NULL) return 1;
    info->sample_dt = dt;
    info->time_pixel_dt = context->wdms.DT;
    info->frequency_pixel_df = context->wdms.DF;
    info->meyer_half_bandwidth = context->wdms.FB;
    info->time_pixels = Nt;
    info->frequency_layers = Nf;
    info->packet_time_half_support = mult;
    return 0;
}

THMWorkerWorkspace *thm_worker_workspace_create(
    const THMObservationContext *context)
{
    THMWorkerWorkspace *workspace;
    int i;

    if(context == NULL) return NULL;
    workspace = calloc(1, sizeof(*workspace));
    if(workspace == NULL) return NULL;
    workspace->context = context;
    workspace->Nsmax = 10000;
    workspace->TS = double_vector(workspace->Nsmax);
    workspace->tspace = double_vector(workspace->Nsmax);
    workspace->tc_tau = double_vector(workspace->Nsmax);
    workspace->mode_amp = double_matrix(IMRPHENOMTHM_MAX_MODES,
                                        workspace->Nsmax);
    workspace->mode_phase = double_matrix(IMRPHENOMTHM_MAX_MODES,
                                          workspace->Nsmax);
    workspace->mode_freq = double_matrix(IMRPHENOMTHM_MAX_MODES,
                                         workspace->Nsmax);
    if(workspace->TS == NULL || workspace->tspace == NULL ||
       workspace->tc_tau == NULL || workspace->mode_amp == NULL ||
       workspace->mode_phase == NULL || workspace->mode_freq == NULL)
    {
        thm_worker_workspace_destroy(workspace);
        return NULL;
    }

    for(i=0; i<3; i++)
    {
        workspace->SLacc[i] = gsl_interp_accel_alloc();
        if(workspace->SLacc[i] == NULL)
        {
            thm_worker_workspace_destroy(workspace);
            return NULL;
        }
    }
    for(i=0; i<9; i++)
    {
        workspace->SPacc[i] = gsl_interp_accel_alloc();
        workspace->SVacc[i] = gsl_interp_accel_alloc();
        if(workspace->SPacc[i] == NULL || workspace->SVacc[i] == NULL)
        {
            thm_worker_workspace_destroy(workspace);
            return NULL;
        }
    }
    return workspace;
}

void thm_worker_workspace_destroy(THMWorkerWorkspace *workspace)
{
    int i;

    if(workspace == NULL) return;
    for(i=0; i<3; i++)
    {
        if(workspace->SLacc[i] != NULL) gsl_interp_accel_free(workspace->SLacc[i]);
    }
    for(i=0; i<9; i++)
    {
        if(workspace->SPacc[i] != NULL) gsl_interp_accel_free(workspace->SPacc[i]);
        if(workspace->SVacc[i] != NULL) gsl_interp_accel_free(workspace->SVacc[i]);
    }
    free_double_vector(workspace->TS);
    free_double_vector(workspace->tspace);
    free_double_vector(workspace->tc_tau);
    if(workspace->mode_amp != NULL)
        free_double_matrix(workspace->mode_amp, IMRPHENOMTHM_MAX_MODES);
    if(workspace->mode_phase != NULL)
        free_double_matrix(workspace->mode_phase, IMRPHENOMTHM_MAX_MODES);
    if(workspace->mode_freq != NULL)
        free_double_matrix(workspace->mode_freq, IMRPHENOMTHM_MAX_MODES);
    thm_reusable_wdm_work_release(&workspace->wdm_work);
    free(workspace);
}

typedef struct
{
    double source_time;
    double plus_coefficient;
    double cross_coefficient;
    int output_index;
    int unique_index;
} THMComplexTDITerm;

static int thm_complex_tdi_term_compare(const void *left, const void *right)
{
    const THMComplexTDITerm *a = (const THMComplexTDITerm *)left;
    const THMComplexTDITerm *b = (const THMComplexTDITerm *)right;

    if(a->source_time < b->source_time) return -1;
    if(a->source_time > b->source_time) return 1;
    return 0;
}

typedef struct
{
    int receiver, emitter;
    int sign;
    int ndelays;
    int delay[7];
} THMTDI2EtaTerm;

/* SGS conventions, Eq. (38b), arXiv:2603.22377.  The first eight entries
 * are X1; the remaining eight are the second-generation extension.  A link
 * ij receives at i and emits at j.  Y2 and Z2 cyclically rotate the labels. */
static const THMTDI2EtaTerm thm_tdi2_x_terms[16] = {
    {1,3, 1,0,{0}},
    {3,1, 1,1,{13}},
    {1,2, 1,2,{13,31}},
    {2,1, 1,3,{13,31,12}},
    {1,2,-1,0,{0}},
    {2,1,-1,1,{12}},
    {1,3,-1,2,{12,21}},
    {3,1,-1,3,{12,21,13}},
    {1,2, 1,4,{13,31,12,21}},
    {2,1, 1,5,{13,31,12,21,12}},
    {1,3, 1,6,{13,31,12,21,12,21}},
    {3,1, 1,7,{13,31,12,21,12,21,13}},
    {1,3,-1,4,{12,21,13,31}},
    {3,1,-1,5,{12,21,13,31,13}},
    {1,2,-1,6,{12,21,13,31,13,31}},
    {2,1,-1,7,{12,21,13,31,13,31,12}}
};

typedef struct
{
    int parent, link;
    double time, lag, jacobian;
} THMTDI2DelayNode;

typedef struct
{
    double light_time, light_time_derivative;
    double source_time_receiver, source_time_emitter;
    double receiver_source_rate, emitter_source_rate;
    double plus_projection, cross_projection;
} THMTDI2Link;

static int thm_tdi2_rotated_spacecraft(int spacecraft, int rotation)
{
    return (spacecraft-1+rotation)%3;
}

static void thm_tdi2_polarization_basis(double ecliptic_latitude,
                                        double ecliptic_longitude,
                                        double kv[3], double eplus[3][3],
                                        double ecross[3][3])
{
    double u[3], v[3];
    double costh = sin(ecliptic_latitude);
    double sinth = sqrt(fmax(0.0, 1.0-costh*costh));
    double cosph = cos(ecliptic_longitude);
    double sinph = sin(ecliptic_longitude);
    int i, j;
    /* u = e_latitude and v = -e_longitude; their relative sign fixes h_cross. */
    u[0] = -costh*cosph; u[1] = -costh*sinph; u[2] = sinth;
    v[0] = sinph; v[1] = -cosph; v[2] = 0.0;
    kv[0] = -sinth*cosph; kv[1] = -sinth*sinph; kv[2] = -costh;
    for(i=0; i<3; i++)
        for(j=0; j<3; j++)
        {
            eplus[i][j] = v[i]*v[j]-u[i]*u[j];
            ecross[i][j] = u[i]*v[j]+v[i]*u[j];
        }
}

static int thm_tdi2_position(const THMObservationContext *context,
                             THMWorkerWorkspace *workspace, int spacecraft,
                             double time, double position[3],
                             double velocity[3])
{
    int k;
    if(time < context->constellation_tmin ||
       time > context->constellation_tmax || !isfinite(time)) return 1;
    for(k=0; k<3; k++)
    {
        int q = 3*spacecraft+k;
        position[k] = gsl_spline_eval(context->SPspline[q], time,
                                      workspace->SPacc[q]);
        if(velocity != NULL)
            velocity[k] = gsl_spline_eval_deriv(context->SPspline[q], time,
                                                workspace->SPacc[q]);
    }
    return 0;
}

/* The six directed projections vary on the orbital time scale.  Their
 * reference epoch is the detector output time, not a nested delay time. */
static int thm_tdi2_frozen_link_projections(
    const THMObservationContext *context, THMWorkerWorkspace *workspace,
    double time, const double kv[3], const double eplus[3][3],
    const double ecross[3][3], double plus[3][3], double cross[3][3])
{
    double position[3][3];
    int receiver, emitter, i, j;

    for(i=0; i<3; i++)
        if(thm_tdi2_position(context, workspace, i, time,
                              position[i], NULL) != 0) return 1;
    for(receiver=0; receiver<3; receiver++)
        for(emitter=0; emitter<3; emitter++)
        {
            double arm[3], length = 0.0, kdot = 0.0;
            double pp = 0.0, pc = 0.0;
            if(receiver == emitter) continue;
            for(i=0; i<3; i++)
            {
                arm[i] = position[receiver][i]-position[emitter][i];
                length += arm[i]*arm[i];
            }
            length = sqrt(length);
            if(!(length > 0.0)) return 2;
            for(i=0; i<3; i++)
            {
                arm[i] /= length;
                kdot += kv[i]*arm[i];
            }
            for(i=0; i<3; i++)
                for(j=0; j<3; j++)
                {
                    pp += arm[i]*arm[j]*eplus[i][j];
                    pc += arm[i]*arm[j]*ecross[i][j];
                }
            if(fabs(1.0-kdot) < 1.0e-12) return 3;
            plus[receiver][emitter] = -0.5*pp/(1.0-kdot);
            cross[receiver][emitter] = -0.5*pc/(1.0-kdot);
        }
    return 0;
}

static int thm_tdi2_link(const THMObservationContext *context,
                         THMWorkerWorkspace *workspace, int receiver,
                         int emitter, double reception_time,
                         const double kv[3], const double eplus[3][3],
                         const double ecross[3][3], int project,
                         THMTDI2Link *link)
{
    double xr[3], xe[3], vr[3], ve[3], ve0[3], arm[3];
    double light_time = 0.0, denominator, numerator = 0.0;
    double plus = 0.0, cross = 0.0, kdot = 0.0;
    int k, l, iteration;
    int solver = context->tdi2_light_time_solver;

    if(thm_tdi2_position(context, workspace, receiver, reception_time,
                         xr, vr) != 0 ||
       thm_tdi2_position(context, workspace, emitter, reception_time,
                         xe, solver == THM_TDI2_LIGHT_TIME_ITERATIVE ?
                             NULL : ve0) != 0) return 1;
    for(k=0; k<3; k++) light_time += (xr[k]-xe[k])*(xr[k]-xe[k]);
    light_time = sqrt(light_time);
    if(solver == THM_TDI2_LIGHT_TIME_QUADRATIC ||
       solver == THM_TDI2_LIGHT_TIME_TAYLOR2)
    {
        double rv = 0.0, v2 = 0.0;
        double radius = light_time;
        if(!(radius > 0.0)) return 3;
        for(k=0; k<3; k++)
        {
            rv += (xr[k]-xe[k])*ve0[k];
            v2 += ve0[k]*ve0[k];
        }
        if(!(v2 < 1.0)) return 3;
        /* With a constant emitter velocity over one light flight,
         * L = |R + v_e L| has this positive quadratic root. */
        if(solver == THM_TDI2_LIGHT_TIME_QUADRATIC)
            light_time = (rv+sqrt(rv*rv+(1.0-v2)*radius*radius)) /
                         (1.0-v2);
        else
        {
            double mu = rv/radius;
            light_time = radius*(1.0+mu+0.5*(v2+mu*mu));
        }
    }
    else
    {
        /* Reference moving-endpoint light-cone solve. */
        for(iteration=0; iteration<6; iteration++)
        {
            double next = 0.0;
            if(thm_tdi2_position(context, workspace, emitter,
                                 reception_time-light_time, xe, NULL) != 0)
                return 2;
            for(k=0; k<3; k++) next += (xr[k]-xe[k])*(xr[k]-xe[k]);
            next = sqrt(next);
            if(fabs(next-light_time) < 2.0e-13)
            {
                light_time = next;
                break;
            }
            light_time = next;
        }
    }
    if(!(light_time > 0.0) ||
       thm_tdi2_position(context, workspace, emitter,
                         reception_time-light_time, xe, ve) != 0) return 3;
    /* The approximate solve still uses the actual retarded emitter position
     * for the photon direction, Doppler time shift, and delay Jacobian. */
    denominator = light_time;
    if(solver != THM_TDI2_LIGHT_TIME_ITERATIVE)
    {
        denominator = 0.0;
        for(k=0; k<3; k++)
            denominator += (xr[k]-xe[k])*(xr[k]-xe[k]);
        denominator = sqrt(denominator);
    }
    if(!(denominator > 0.0)) return 3;
    for(k=0; k<3; k++)
    {
        arm[k] = (xr[k]-xe[k])/denominator;
        if(project) kdot += kv[k]*arm[k];
    }
    if(project) for(k=0; k<3; k++)
    {
        for(l=0; l<3; l++)
        {
            plus += arm[k]*arm[l]*eplus[k][l];
            cross += arm[k]*arm[l]*ecross[k][l];
        }
    }
    if(project && fabs(1.0-kdot) < 1.0e-12) return 4;
    denominator = 1.0;
    for(k=0; k<3; k++)
    {
        denominator -= arm[k]*ve[k];
        numerator += arm[k]*(vr[k]-ve[k]);
    }
    if(fabs(denominator) < 1.0e-12) return 5;
    link->light_time = light_time;
    link->light_time_derivative = numerator/denominator;
    link->source_time_receiver = reception_time;
    link->source_time_emitter = reception_time-light_time;
    link->receiver_source_rate = 1.0;
    link->emitter_source_rate = 1.0-link->light_time_derivative;
    for(k=0; k<3; k++)
    {
        link->source_time_receiver -= kv[k]*xr[k];
        link->source_time_emitter -= kv[k]*xe[k];
        link->receiver_source_rate -= kv[k]*vr[k];
        link->emitter_source_rate -=
            (1.0-link->light_time_derivative)*kv[k]*ve[k];
    }
    link->plus_projection = project ? -0.5*plus/(1.0-kdot) : 0.0;
    link->cross_projection = project ? -0.5*cross/(1.0-kdot) : 0.0;
    return 0;
}

int thm_evaluate_complex_eta_context(
    const THMObservationContext *context, THMWorkerWorkspace *workspace,
    int n, const double *detector_time, double ecliptic_latitude,
    double ecliptic_longitude, THMComplexPolarizationEvaluator evaluator,
    void *userdata, double complex *eta[6], double *ltt[6],
    double *ltt_derivative[6])
{
    const int links[6] = {12, 13, 21, 23, 31, 32};
    const int chunk_size = 4096;
    THMComplexTDITerm *terms = NULL;
    double *unique_time = NULL;
    double complex *hplus = NULL, *hcross = NULL;
    double kv[3], eplus[3][3], ecross[3][3];
    int i, j, offset, status = 0;

    if(context == NULL || workspace == NULL || workspace->context != context ||
       n < 1 || detector_time == NULL || evaluator == NULL ||
       eta == NULL || ltt == NULL || ltt_derivative == NULL)
        return 1;
    for(j=0; j<6; j++)
        if(eta[j] == NULL || ltt[j] == NULL || ltt_derivative[j] == NULL)
            return 1;
    for(i=1; i<n; i++)
        if(detector_time[i] <= detector_time[i-1]) return 2;
    if(detector_time[0] < context->constellation_tmin ||
       detector_time[n-1] > context->constellation_tmax) return 3;
    terms = malloc((size_t)chunk_size*12*sizeof(*terms));
    unique_time = double_vector(chunk_size*12);
    hplus = malloc((size_t)chunk_size*12*sizeof(*hplus));
    hcross = malloc((size_t)chunk_size*12*sizeof(*hcross));
    if(terms == NULL || unique_time == NULL || hplus == NULL ||
       hcross == NULL)
    {
        status = 4;
        goto cleanup;
    }
    thm_tdi2_polarization_basis(ecliptic_latitude, ecliptic_longitude,
                                 kv, eplus, ecross);
    for(offset=0; offset<n; offset+=chunk_size)
    {
        int count = n-offset < chunk_size ? n-offset : chunk_size;
        int nterms = 0, nunique = 0;
        for(i=0; i<count; i++)
            for(j=0; j<6; j++)
            {
                THMTDI2Link link;
                int receiver = links[j]/10-1;
                int emitter = links[j]%10-1;
                if(thm_tdi2_link(context, workspace, receiver, emitter,
                                 detector_time[offset+i], kv, eplus,
                                 ecross, 1, &link) != 0)
                {
                    status = 5;
                    goto cleanup;
                }
                ltt[j][offset+i] = link.light_time;
                ltt_derivative[j][offset+i] = link.light_time_derivative;
                terms[nterms].source_time = link.source_time_receiver;
                terms[nterms].plus_coefficient = link.plus_projection;
                terms[nterms].cross_coefficient = link.cross_projection;
                terms[nterms].output_index = 6*i+j;
                nterms++;
                terms[nterms].source_time = link.source_time_emitter;
                terms[nterms].plus_coefficient = -link.plus_projection;
                terms[nterms].cross_coefficient = -link.cross_projection;
                terms[nterms].output_index = 6*i+j;
                nterms++;
            }
        qsort(terms, (size_t)nterms, sizeof(*terms),
              thm_complex_tdi_term_compare);
        for(i=0; i<nterms; i++)
        {
            /* The source callback may convert t to (t-tc)/M.  Distinct
             * retarded words separated only at the precision of the large
             * observation/merger epoch collapse to an identical tau. */
            double tolerance = 16.0*DBL_EPSILON*
                fmax(context->wdms.Tobs, fmax(fabs(terms[i].source_time),
                    nunique > 0 ? fabs(unique_time[nunique-1]) : 0.0));
            if(nunique == 0 ||
               fabs(terms[i].source_time-unique_time[nunique-1]) > tolerance)
                unique_time[nunique++] = terms[i].source_time;
            terms[i].unique_index = nunique-1;
        }
        status = evaluator(userdata, nunique, unique_time, hplus, hcross);
        if(status != 0)
        {
            status += 100;
            goto cleanup;
        }
        for(i=0; i<count; i++)
            for(j=0; j<6; j++) eta[j][offset+i] = 0.0;
        for(i=0; i<nterms; i++)
        {
            int target = terms[i].output_index;
            eta[target%6][offset+target/6] +=
                terms[i].plus_coefficient*hplus[terms[i].unique_index] +
                terms[i].cross_coefficient*hcross[terms[i].unique_index];
        }
    }

cleanup:
    free(hcross);
    free(hplus);
    free_double_vector(unique_time);
    free(terms);
    return status;
}

static int thm_evaluate_complex_tdi2(
    const THMObservationContext *context, THMWorkerWorkspace *workspace,
    int n, const double *detector_time, double ecliptic_latitude,
    double ecliptic_longitude, int nfamilies,
    THMComplexPolarizationFamilyEvaluator evaluator, void *userdata,
    double complex *X, double complex *Y, double complex *Z)
{
    const int chunk_size = 4096;
    const int terms_per_sample = 96;
    THMComplexTDITerm *terms = NULL;
    double *unique_time = NULL;
    double complex *hplus = NULL, *hcross = NULL;
    double complex *output[3] = {X, Y, Z};
    double kv[3], eplus[3][3], ecross[3][3];
    int offset, i, j, k, ch, family, status = 0;

    terms = malloc((size_t)chunk_size*terms_per_sample*sizeof(*terms));
    unique_time = double_vector(chunk_size*terms_per_sample);
    hplus = malloc((size_t)chunk_size*terms_per_sample*(size_t)nfamilies*
                   sizeof(*hplus));
    hcross = malloc((size_t)chunk_size*terms_per_sample*(size_t)nfamilies*
                    sizeof(*hcross));
    if(terms == NULL || unique_time == NULL || hplus == NULL ||
       hcross == NULL)
    {
        status = 5;
        goto cleanup;
    }
    thm_tdi2_polarization_basis(ecliptic_latitude, ecliptic_longitude,
                                 kv, eplus, ecross);

    for(offset=0; offset<n; offset+=chunk_size)
    {
        int count = n-offset < chunk_size ? n-offset : chunk_size;
        int nterms = 0, nunique = 0;
        for(i=0; i<count; i++)
        {
            THMTDI2DelayNode nodes[128];
            THMTDI2Link reference[3][3];
            double frozen_plus[3][3], frozen_cross[3][3];
            int nnodes = 1;
            if(context->tdi2_chain_taylor)
            {
                /* One retarded one-link geometry per directed arm at the
                 * output epoch supplies every occurrence in the delay tree. */
                int receiver, emitter;
                for(receiver=0; receiver<3; receiver++)
                    for(emitter=0; emitter<3; emitter++)
                        if(receiver != emitter &&
                           thm_tdi2_link(context, workspace, receiver,
                                         emitter, detector_time[offset+i],
                                         kv, eplus, ecross, 1,
                                         &reference[receiver][emitter]) != 0)
                        {
                            status = 9;
                            goto cleanup;
                        }
            }
            else if(context->tdi2_frozen_projection &&
               thm_tdi2_frozen_link_projections(
                   context, workspace, detector_time[offset+i], kv,
                   eplus, ecross, frozen_plus, frozen_cross) != 0)
            {
                status = 8;
                goto cleanup;
            }
            nodes[0].parent = -1;
            nodes[0].link = 0;
            nodes[0].time = detector_time[offset+i];
            nodes[0].lag = 0.0;
            nodes[0].jacobian = 1.0;
            for(ch=0; ch<3; ch++)
            {
                for(j=0; j<16; j++)
                {
                    const THMTDI2EtaTerm *spec = &thm_tdi2_x_terms[j];
                    THMTDI2Link link;
                    int node = 0;
                    int receiver = thm_tdi2_rotated_spacecraft(
                        spec->receiver, ch);
                    int emitter = thm_tdi2_rotated_spacecraft(
                        spec->emitter, ch);
                    for(k=0; k<spec->ndelays; k++)
                    {
                        int label = spec->delay[k];
                        int dr = thm_tdi2_rotated_spacecraft(label/10, ch);
                        int de = thm_tdi2_rotated_spacecraft(label%10, ch);
                        int directed = 3*dr+de;
                        int child;
                        for(child=1; child<nnodes; child++)
                            if(nodes[child].parent == node &&
                               nodes[child].link == directed) break;
                        if(child == nnodes)
                        {
                            if(nnodes >= (int)(sizeof(nodes)/sizeof(nodes[0])))
                            {
                                status = 6;
                                goto cleanup;
                            }
                            if(context->tdi2_chain_taylor)
                            {
                                link = reference[dr][de];
                                /* Delta' = Delta + L(t-Delta), with the
                                 * first-order evolution L(t-Delta). */
                                nodes[child].lag = nodes[node].lag +
                                    link.light_time-
                                    link.light_time_derivative*nodes[node].lag;
                                nodes[child].time = detector_time[offset+i]-
                                    nodes[child].lag;
                                if(nodes[child].time < context->constellation_tmin ||
                                   nodes[child].time > context->constellation_tmax)
                                {
                                    status = 6;
                                    goto cleanup;
                                }
                            }
                            else
                            {
                                if(thm_tdi2_link(context, workspace, dr, de,
                                                nodes[node].time, kv, eplus,
                                                ecross, 0, &link) != 0)
                                {
                                    status = 6;
                                    goto cleanup;
                                }
                                nodes[child].time = nodes[node].time-
                                    link.light_time;
                            }
                            nodes[child].parent = node;
                            nodes[child].link = directed;
                            nodes[child].jacobian = nodes[node].jacobian*
                                (1.0-link.light_time_derivative);
                            nnodes++;
                        }
                        node = child;
                    }
                    if(context->tdi2_chain_taylor)
                    {
                        link = reference[receiver][emitter];
                        if(nodes[node].time-link.light_time+
                           link.light_time_derivative*nodes[node].lag <
                           context->constellation_tmin)
                        {
                            status = 7;
                            goto cleanup;
                        }
                    }
                    else if(thm_tdi2_link(context, workspace, receiver,
                                         emitter, nodes[node].time, kv,
                                         eplus, ecross,
                                         !context->tdi2_frozen_projection,
                                         &link) != 0)
                    {
                        status = 7;
                        goto cleanup;
                    }
                    terms[nterms].source_time =
                        context->tdi2_chain_taylor ?
                        link.source_time_receiver-
                        link.receiver_source_rate*nodes[node].lag :
                        link.source_time_receiver;
                    terms[nterms].plus_coefficient = spec->sign*
                        nodes[node].jacobian*
                        (context->tdi2_chain_taylor ?
                         link.plus_projection :
                         context->tdi2_frozen_projection ?
                         frozen_plus[receiver][emitter] :
                         link.plus_projection);
                    terms[nterms].cross_coefficient = spec->sign*
                        nodes[node].jacobian*
                        (context->tdi2_chain_taylor ?
                         link.cross_projection :
                         context->tdi2_frozen_projection ?
                         frozen_cross[receiver][emitter] :
                         link.cross_projection);
                    terms[nterms].output_index = 3*(offset+i)+ch;
                    nterms++;
                    terms[nterms].source_time =
                        context->tdi2_chain_taylor ?
                        link.source_time_emitter-
                        link.emitter_source_rate*nodes[node].lag :
                        link.source_time_emitter;
                    terms[nterms].plus_coefficient = -terms[nterms-1].plus_coefficient;
                    terms[nterms].cross_coefficient = -terms[nterms-1].cross_coefficient;
                    terms[nterms].output_index = 3*(offset+i)+ch;
                    nterms++;
                }
            }
        }
        qsort(terms, (size_t)nterms, sizeof(*terms),
              thm_complex_tdi_term_compare);
        for(i=0; i<nterms; i++)
        {
            /* Source models may form (t-tc)/M; nearly coincident delay
             * words must remain distinct only if that coordinate can. */
            double tolerance = 16.0*DBL_EPSILON*
                fmax(context->wdms.Tobs, fmax(fabs(terms[i].source_time),
                    nunique > 0 ? fabs(unique_time[nunique-1]) : 0.0));
            if(nunique == 0 ||
               fabs(terms[i].source_time-unique_time[nunique-1]) > tolerance)
                unique_time[nunique++] = terms[i].source_time;
            terms[i].unique_index = nunique-1;
        }
        status = evaluator(userdata, nunique, unique_time, nfamilies,
                           hplus, hcross);
        if(status != 0)
        {
            status += 100;
            goto cleanup;
        }
        for(family=0; family<nfamilies; family++)
            for(i=offset; i<offset+count; i++)
                for(ch=0; ch<3; ch++)
                    output[ch][(size_t)family*(size_t)n+(size_t)i] = 0.0;
        for(family=0; family<nfamilies; family++)
            for(i=0; i<nterms; i++)
            {
                int target = terms[i].output_index;
                int q = terms[i].unique_index;
                size_t source = (size_t)family*(size_t)nunique+(size_t)q;
                size_t destination =
                    (size_t)family*(size_t)n+(size_t)(target/3);
                output[target%3][destination] +=
                    terms[i].plus_coefficient*hplus[source] +
                    terms[i].cross_coefficient*hcross[source];
            }
    }

cleanup:
    free(hcross);
    free(hplus);
    free_double_vector(unique_time);
    free(terms);
    return status;
}

static int thm_evaluate_complex_tdi_families_core(
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
    double complex *Z)
{
    const int terms_per_sample = 24;
    THMComplexTDITerm *terms = NULL;
    double *unique_time = NULL;
    double complex *hplus = NULL;
    double complex *hcross = NULL;
    double complex *output[3] = {X, Y, Z};
    double u[3], v[3], kv[3], eplus[3][3], ecross[3][3];
    double costh, sinth, cosph, sinph;
    int nterms, nunique, i, j, k, ch, family, term_index;
    int status = 0;

    if(context == NULL || workspace == NULL || workspace->context != context ||
       n < 1 || detector_time == NULL || evaluator == NULL ||
       X == NULL || Y == NULL || Z == NULL || nfamilies < 1)
        return 1;
    for(i=1; i<n; i++)
    {
        if(detector_time[i] <= detector_time[i-1]) return 2;
    }
    if(detector_time[0] < context->constellation_tmin ||
       detector_time[n-1] > context->constellation_tmax)
        return 3;
    if(context->tdi_generation == 2)
        return thm_evaluate_complex_tdi2(context, workspace, n,
                   detector_time, ecliptic_latitude, ecliptic_longitude,
                   nfamilies, evaluator, userdata, X, Y, Z);
    if((size_t)n > ((size_t)-1)/(size_t)terms_per_sample) return 4;

    nterms = terms_per_sample*n;
    terms = calloc((size_t)nterms, sizeof(*terms));
    unique_time = double_vector(nterms);
    hplus = calloc((size_t)nterms*(size_t)nfamilies, sizeof(*hplus));
    hcross = calloc((size_t)nterms*(size_t)nfamilies, sizeof(*hcross));
    if(terms == NULL || unique_time == NULL || hplus == NULL || hcross == NULL)
    {
        status = 5;
        goto cleanup;
    }

    costh = sin(ecliptic_latitude);
    sinth = sqrt(fmax(0.0, 1.0-costh*costh));
    cosph = cos(ecliptic_longitude);
    sinph = sin(ecliptic_longitude);
    u[0] = -costh*cosph; u[1] = -costh*sinph; u[2] = sinth;
    v[0] = sinph; v[1] = -cosph; v[2] = 0.0;
    kv[0] = -sinth*cosph;
    kv[1] = -sinth*sinph;
    kv[2] = -costh;
    for(i=0; i<3; i++)
    {
        for(j=0; j<3; j++)
        {
            eplus[i][j] = v[i]*v[j]-u[i]*u[j];
            ecross[i][j] = u[i]*v[j]+v[i]*u[j];
        }
    }
    for(family=0; family<nfamilies; family++)
    {
        for(i=0; i<n; i++)
        {
            X[(size_t)family*(size_t)n+(size_t)i] = 0.0;
            Y[(size_t)family*(size_t)n+(size_t)i] = 0.0;
            Z[(size_t)family*(size_t)n+(size_t)i] = 0.0;
        }
    }

    term_index = 0;
    for(i=0; i<n; i++)
    {
        double t = detector_time[i];
        double Larm[3], position[3][3], arm[3][3];
        double kr[3], kn[3], App[3], Apm[3], Acp[3], Acm[3];

        for(j=0; j<3; j++)
            Larm[j] = gsl_spline_eval(context->SLspline[j], t,
                                      workspace->SLacc[j]);
        for(j=0; j<3; j++)
        {
            for(k=0; k<3; k++)
            {
                int q = k+3*j;
                position[j][k] = gsl_spline_eval(context->SPspline[q], t,
                                                 workspace->SPacc[q]);
                arm[j][k] = gsl_spline_eval(context->SVspline[q], t,
                                            workspace->SVacc[q]);
            }
        }
        for(j=0; j<3; j++)
        {
            double plus = 0.0;
            double cross = 0.0;
            kr[j] = 0.0;
            kn[j] = 0.0;
            for(k=0; k<3; k++)
            {
                int l;
                kr[j] += position[j][k]*kv[k];
                kn[j] += arm[j][k]*kv[k];
                for(l=0; l<3; l++)
                {
                    plus += arm[j][k]*arm[j][l]*eplus[k][l];
                    cross += arm[j][k]*arm[j][l]*ecross[k][l];
                }
            }
            App[j] = 0.5*plus/(1.0+kn[j]);
            Apm[j] = 0.5*plus/(1.0-kn[j]);
            Acp[j] = 0.5*cross/(1.0+kn[j]);
            Acm[j] = 0.5*cross/(1.0-kn[j]);
        }

        for(ch=0; ch<3; ch++)
        {
            int a = ch;
            int b = (ch+1)%3;
            int c = (ch+2)%3;
            double delay[8];
            double plus[8];
            double cross[8];

            delay[0] = kr[a]+2.0*Larm[c]+2.0*Larm[b];
            plus[0] = App[c]-Apm[b]; cross[0] = Acp[c]-Acm[b];
            delay[1] = kr[b]+Larm[c]+2.0*Larm[b];
            plus[1] = -App[c]+Apm[c]; cross[1] = -Acp[c]+Acm[c];
            delay[2] = kr[c]+Larm[b]+2.0*Larm[c];
            plus[2] = Apm[b]-App[b]; cross[2] = Acm[b]-Acp[b];
            delay[3] = kr[a]+2.0*Larm[b];
            plus[3] = -Apm[c]+Apm[b]; cross[3] = -Acm[c]+Acm[b];
            delay[4] = kr[a]+2.0*Larm[c];
            plus[4] = App[b]-App[c]; cross[4] = Acp[b]-Acp[c];
            delay[5] = kr[c]+Larm[b];
            plus[5] = -Apm[b]+App[b]; cross[5] = -Acm[b]+Acp[b];
            delay[6] = kr[b]+Larm[c];
            plus[6] = App[c]-Apm[c]; cross[6] = Acp[c]-Acm[c];
            delay[7] = kr[a];
            plus[7] = -App[b]+Apm[c]; cross[7] = -Acp[b]+Acm[c];

            for(j=0; j<8; j++)
            {
                terms[term_index].source_time = t-delay[j];
                terms[term_index].plus_coefficient = plus[j];
                terms[term_index].cross_coefficient = cross[j];
                terms[term_index].output_index = 3*i+ch;
                term_index++;
            }
        }
    }

    qsort(terms, (size_t)nterms, sizeof(*terms),
          thm_complex_tdi_term_compare);
    nunique = 0;
    for(i=0; i<nterms; i++)
    {
        double tolerance = 16.0*DBL_EPSILON*
            fmax(1.0, fmax(fabs(terms[i].source_time),
                           nunique > 0 ? fabs(unique_time[nunique-1]) : 0.0));
        if(nunique == 0 ||
           fabs(terms[i].source_time-unique_time[nunique-1]) > tolerance)
        {
            unique_time[nunique] = terms[i].source_time;
            nunique++;
        }
        terms[i].unique_index = nunique-1;
    }

    status = evaluator(userdata, nunique, unique_time, nfamilies,
                       hplus, hcross);
    if(status != 0)
    {
        status = 100+status;
        goto cleanup;
    }
    for(family=0; family<nfamilies; family++)
    {
        for(i=0; i<nterms; i++)
        {
            int target = terms[i].output_index;
            int channel = target%3;
            int sample = target/3;
            int q = terms[i].unique_index;
            size_t source_index =
                (size_t)family*(size_t)nunique+(size_t)q;
            size_t output_index =
                (size_t)family*(size_t)n+(size_t)sample;
            output[channel][output_index] +=
                terms[i].plus_coefficient*hplus[source_index] +
                terms[i].cross_coefficient*hcross[source_index];
        }
    }

cleanup:
    free(hcross);
    free(hplus);
    free_double_vector(unique_time);
    free(terms);
    return status;
}

typedef struct
{
    THMComplexPolarizationEvaluator evaluator;
    void *userdata;
} THMSinglePolarizationAdapter;

static int thm_single_polarization_family_adapter(
    void *userdata,
    int n,
    const double *source_time,
    int nfamilies,
    double complex *hplus,
    double complex *hcross)
{
    THMSinglePolarizationAdapter *adapter =
        (THMSinglePolarizationAdapter *)userdata;
    if(adapter == NULL || adapter->evaluator == NULL || nfamilies != 1)
        return 1;
    return adapter->evaluator(adapter->userdata, n, source_time,
                              hplus, hcross);
}

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
    double complex *Z)
{
    THMSinglePolarizationAdapter adapter;
    adapter.evaluator = evaluator;
    adapter.userdata = userdata;
    return thm_evaluate_complex_tdi_families_core(
        context, workspace, n, detector_time,
        ecliptic_latitude, ecliptic_longitude, 1,
        thm_single_polarization_family_adapter, &adapter, X, Y, Z);
}

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
    double complex *Z)
{
    return thm_evaluate_complex_tdi_families_core(
        context, workspace, n, detector_time,
        ecliptic_latitude, ecliptic_longitude, nfamilies,
        evaluator, userdata, X, Y, Z);
}

void thm_complex_tdi_fft_block_init(THMComplexTDIFFTBlock *block)
{
    if(block == NULL) return;
    memset(block, 0, sizeof(*block));
}

void thm_complex_tdi_fft_block_free(THMComplexTDIFFTBlock *block)
{
    int ch;

    if(block == NULL) return;
    for(ch=0; ch<3; ch++) free_double_vector(block->fft[ch]);
    thm_complex_tdi_fft_block_init(block);
}

typedef struct
{
    THMComplexPolarizationEvaluator evaluator;
    void *userdata;
    double sample_dt;
    double heterodyne_frequency;
} THMInterpolatedPolarizationEvaluator;

static double complex thm_uniform_complex_cubic(
    const double complex *value, int n, double x)
{
    int i = (int)floor(x);
    double u = x-(double)i;
    if(i < 1)
    {
        if(i < 0) return value[0];
        return (1.0-u)*value[i]+u*value[i+1];
    }
    if(i >= n-2)
    {
        if(i >= n-1) return value[n-1];
        return (1.0-u)*value[i]+u*value[i+1];
    }
    return 0.5*((2.0*value[i])+
                (-value[i-1]+value[i+1])*u+
                (2.0*value[i-1]-5.0*value[i]+4.0*value[i+1]-
                 value[i+2])*u*u+
                (-value[i-1]+3.0*value[i]-3.0*value[i+1]+
                 value[i+2])*u*u*u);
}

static int thm_interpolated_polarization_callback(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus,
    double complex *hcross)
{
    THMInterpolatedPolarizationEvaluator *interpolator =
        (THMInterpolatedPolarizationEvaluator *)userdata;
    double *grid_time = NULL;
    double complex *grid_plus = NULL, *grid_cross = NULL;
    double grid_start, span, phase;
    int ngrid, i, status;

    if(interpolator == NULL || interpolator->evaluator == NULL || n < 1 ||
       source_time == NULL || hplus == NULL || hcross == NULL ||
       interpolator->sample_dt <= 0.0)
        return 1;
    grid_start = floor(source_time[0]/interpolator->sample_dt)*
                 interpolator->sample_dt-2.0*interpolator->sample_dt;
    span = source_time[n-1]-grid_start;
    if(span < 0.0 || span/interpolator->sample_dt > 1073741820.0)
        return 2;
    ngrid = (int)ceil(span/interpolator->sample_dt)+4;
    if(ngrid < 6) ngrid = 6;
    grid_time = double_vector(ngrid);
    grid_plus = calloc((size_t)ngrid, sizeof(*grid_plus));
    grid_cross = calloc((size_t)ngrid, sizeof(*grid_cross));
    if(grid_time == NULL || grid_plus == NULL || grid_cross == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(i=0; i<ngrid; i++)
        grid_time[i] = grid_start+interpolator->sample_dt*(double)i;
    status = interpolator->evaluator(interpolator->userdata, ngrid,
                                     grid_time, grid_plus, grid_cross);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    for(i=0; i<ngrid; i++)
    {
        double complex demod;
        phase = 2.0*M_PI*interpolator->heterodyne_frequency*
                (grid_time[i]-grid_start);
        demod = cos(phase)-I*sin(phase);
        grid_plus[i] *= demod;
        grid_cross[i] *= demod;
    }
    for(i=0; i<n; i++)
    {
        double x = (source_time[i]-grid_start)/interpolator->sample_dt;
        double complex remod;
        phase = 2.0*M_PI*interpolator->heterodyne_frequency*
                (source_time[i]-grid_start);
        remod = cos(phase)+I*sin(phase);
        hplus[i] = remod*thm_uniform_complex_cubic(grid_plus, ngrid, x);
        hcross[i] = remod*thm_uniform_complex_cubic(grid_cross, ngrid, x);
    }
    status = 0;

cleanup:
    free(grid_cross);
    free(grid_plus);
    free_double_vector(grid_time);
    return status;
}

static int thm_build_complex_tdi_fft_block_common(
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
    THMComplexTDIFFTBlock *block)
{
    double *time = NULL;
    double complex *channel[3] = {NULL, NULL, NULL};
    double sample_stop;
    int eval_start, eval_stop, neval;
    int i, ch;
    int status;
    THMInterpolatedPolarizationEvaluator interpolator;
    THMComplexPolarizationEvaluator effective_evaluator = evaluator;
    void *effective_userdata = userdata;

    if(context == NULL || workspace == NULL || evaluator == NULL ||
       block == NULL || nfft < 2 || (nfft & (nfft-1)) != 0 ||
       !isfinite(sample_start) || !isfinite(sample_dt) || sample_dt <= 0.0 ||
       !isfinite(heterodyne_frequency))
        return 1;
    sample_stop = sample_start+sample_dt*(double)(nfft-1);
    if(sample_start < context->constellation_tmin) return 2;

    eval_start = 0;
    eval_stop = nfft;
    /* Keep the leading zero-weight taper samples in the waveform batch.  The
     * stripped THM phase is obtained by integrating frequency on the supplied
     * grid, so retaining the block's common lattice origin gives overlapping
     * blocks identical phase histories.  The long trailing zero pad carries
     * no such information and is safely omitted below. */
    if(fall_end > fall_start)
    {
        eval_stop = (int)ceil((fall_end-sample_start)/sample_dt)+1;
        if(eval_stop < 0) eval_stop = 0;
        if(eval_stop > nfft) eval_stop = nfft;
    }
    if(eval_stop < eval_start) eval_stop = eval_start;
    neval = eval_stop-eval_start;
    if(neval < 2) return 3;
    if(sample_start+sample_dt*(double)(eval_stop-1) >
       context->constellation_tmax)
        return 2;

    thm_complex_tdi_fft_block_free(block);
    time = double_vector(neval);
    for(ch=0; ch<3; ch++)
    {
        channel[ch] = calloc((size_t)neval, sizeof(*channel[ch]));
        block->fft[ch] = double_vector(2*nfft);
    }
    if(time == NULL || channel[0] == NULL || channel[1] == NULL ||
       channel[2] == NULL || block->fft[0] == NULL ||
       block->fft[1] == NULL || block->fft[2] == NULL)
    {
        status = 4;
        goto cleanup;
    }
    for(i=0; i<neval; i++)
        time[i] = sample_start+sample_dt*(double)(i+eval_start);
    if(source_sample_dt > 0.0)
    {
        interpolator.evaluator = evaluator;
        interpolator.userdata = userdata;
        interpolator.sample_dt = source_sample_dt;
        interpolator.heterodyne_frequency = heterodyne_frequency;
        effective_evaluator = thm_interpolated_polarization_callback;
        effective_userdata = &interpolator;
    }
    status = thm_evaluate_complex_tdi_context(
        context, workspace, neval, time, ecliptic_latitude,
        ecliptic_longitude, effective_evaluator, effective_userdata,
        channel[0], channel[1], channel[2]);
    if(status != 0)
    {
        status = 100+status;
        goto cleanup;
    }

    for(i=0; i<nfft; i++)
    {
        double t = sample_start+sample_dt*(double)i;
        double tau = t-sample_start;
        double weight = 1.0;
        double phase = 2.0*M_PI*heterodyne_frequency*tau;
        double complex heterodyne = cos(phase)-I*sin(phase);

        if(rise_end > rise_start)
            weight *= split_smooth_step(t, rise_start, rise_end);
        if(fall_end > fall_start)
            weight *= 1.0-split_smooth_step(t, fall_start, fall_end);
        for(ch=0; ch<3; ch++)
        {
            double complex value = 0.0;
            if(i >= eval_start && i < eval_stop)
                value = weight*channel[ch][i-eval_start]*heterodyne;
            REAL(block->fft[ch], i) = creal(value);
            IMAG(block->fft[ch], i) = cimag(value);
        }
    }
    for(ch=0; ch<3; ch++)
    {
        gsl_fft_complex_radix2_forward(block->fft[ch], 1, (size_t)nfft);
        for(i=0; i<nfft; i++)
        {
            REAL(block->fft[ch], i) *= sample_dt;
            IMAG(block->fft[ch], i) *= sample_dt;
        }
    }
    block->nfft = nfft;
    block->sample_start = sample_start;
    block->sample_dt = sample_dt;
    block->heterodyne_frequency = heterodyne_frequency;
    block->nonzero_start = rise_end > rise_start ? rise_start : sample_start;
    block->nonzero_end = fall_end > fall_start ? fall_end : sample_stop;
    status = 0;

cleanup:
    for(ch=0; ch<3; ch++) free(channel[ch]);
    free_double_vector(time);
    if(status != 0) thm_complex_tdi_fft_block_free(block);
    return status;
}

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
    THMComplexTDIFFTBlock *block)
{
    return thm_build_complex_tdi_fft_block_common(
        context, workspace, evaluator, userdata,
        ecliptic_latitude, ecliptic_longitude,
        nfft, sample_start, sample_dt, 0.0, heterodyne_frequency,
        rise_start, rise_end, fall_start, fall_end, block);
}

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
    THMComplexTDIFFTBlock *block)
{
    if(!isfinite(source_sample_dt) || source_sample_dt <= 0.0) return 1;
    return thm_build_complex_tdi_fft_block_common(
        context, workspace, evaluator, userdata,
        ecliptic_latitude, ecliptic_longitude,
        nfft, sample_start, sample_dt, source_sample_dt,
        heterodyne_frequency, rise_start, rise_end,
        fall_start, fall_end, block);
}

int thm_complex_fft_blocks_to_sparse_wdm_context(
    const THMObservationContext *context,
    int nblocks,
    const THMComplexTDIFFTBlock *blocks,
    double time_start,
    double time_stop,
    double frequency_start,
    double frequency_stop,
    THMSparseWDMTriplet *out_tracks)
{
    struct wdmshape *wdms;
    WDMPacketFFTPlan plan;
    int nlo, nhi, width, Ntx, nmid, mlo, mhi;
    int ch, m, j, i, block_index;
    int capacity;
    double t0, fcenter, f, fbase, re, im, phase, c, s, white;

    if(context == NULL || blocks == NULL || nblocks < 1 ||
       out_tracks == NULL || !isfinite(time_start) ||
       !isfinite(time_stop) || time_stop <= time_start ||
       !isfinite(frequency_start) || !isfinite(frequency_stop) ||
       frequency_stop <= frequency_start)
        return 1;
    if(context->tdi_generation == 2 &&
       thm_wdm_instrument_prewhiten_enabled) return 5;
    wdms = (struct wdmshape *)&context->wdms;
    nlo = (int)floor(time_start/wdms->DT)-mult;
    nhi = (int)ceil(time_stop/wdms->DT)+mult;
    if(nlo < 0) nlo = 0;
    if(nhi > Nt) nhi = Nt;
    if(nhi <= nlo) return 2;
    width = nhi-nlo;
    Ntx = thm_next_power_of_two_int(width);
    if(Ntx > Nt) Ntx = Nt;
    nmid = (nlo+nhi)/2;
    if(nmid & 1) nmid--;
    if(nmid-Ntx/2 < 0) nmid = Ntx/2;
    if(nmid+Ntx/2 > Nt) nmid = Nt-Ntx/2;
    t0 = ((double)(nmid-Ntx/2))*wdms->DT;

    mlo = (int)ceil((frequency_start-wdms->FB)/wdms->DF);
    mhi = (int)floor((frequency_stop+wdms->FB)/wdms->DF);
    if(mlo < 1) mlo = 1;
    if(mhi > Nf-1) mhi = Nf-1;
    if(mhi < mlo) return 3;
    capacity = (mhi-mlo+1)*Ntx;
    for(ch=0; ch<3; ch++)
    {
        out_tracks->channel[ch].npixels = 0;
        if(!thm_sparse_wdm_channel_reserve(&out_tracks->channel[ch],
                                           capacity))
            return 4;
    }

    wdm_packet_fft_plan_init(&plan);
    if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms))
    {
        wdm_packet_fft_plan_free(&plan);
        return 5;
    }
    for(ch=0; ch<3; ch++)
    {
        THMSparseWDMChannel *track = &out_tracks->channel[ch];

        for(m=mlo; m<=mhi; m++)
        {
            fcenter = ((double)m)*wdms->DF;
            memset(plan.data, 0, (size_t)(2*Ntx)*sizeof(*plan.data));
            for(j=-Ntx/2; j<Ntx/2; j++)
            {
                int jj = j+Ntx/2;
                if(jj <= 0) continue;
                f = ((double)j)*plan.dfx+fcenter;
                if(f <= 0.0) continue;
                re = 0.0;
                im = 0.0;
                for(block_index=0; block_index<nblocks; block_index++)
                {
                    const THMComplexTDIFFTBlock *block =
                        &blocks[block_index];
                    double bre, bim;
                    double nyquist;

                    if(block->fft[ch] == NULL || block->nfft < 2) continue;
                    fbase = f-block->heterodyne_frequency;
                    nyquist = 0.5/block->sample_dt;
                    if(fabs(fbase) >= nyquist) continue;
                    sample_complex_fft_exact(block->fft[ch], block->nfft,
                                             block->sample_dt, fbase,
                                             &bre, &bim);
                    phase = 2.0*M_PI*f*(t0-block->sample_start);
                    c = cos(phase);
                    s = sin(phase);
                    re += bre*c-bim*s;
                    im += bre*s+bim*c;
                }
                white = thm_instrument_prewhiten_factor(f);
                plan.data[jj] = white*re;
                plan.data[2*Ntx-jj] = white*im;
            }
            wdmtranF_plan(m, &plan);
            for(i=0; i<Ntx; i++)
            {
                int outn = i+nmid-Ntx/2;
                int q;
                if(outn < 0 || outn >= Nt) continue;
                q = track->npixels++;
                track->n[q] = outn;
                track->m[q] = m;
                track->value[q] = plan.wdmout[i];
            }
        }
    }
    wdm_packet_fft_plan_free(&plan);
    return 0;
}

static int thm_complex_block_overlaps_layer(
    const THMComplexTDIFFTBlock *block,
    double layer_low,
    double layer_high)
{
    if(block == NULL) return 0;
    if(!(block->frequency_stop > block->frequency_start)) return 1;
    return layer_high >= block->frequency_start &&
           layer_low <= block->frequency_stop;
}

int thm_complex_fft_blocks_to_planned_sparse_wdm_context(
    const THMObservationContext *context,
    int nblocks,
    const THMComplexTDIFFTBlock *blocks,
    THMSparseWDMTriplet *out_tracks)
{
    struct wdmshape *wdms;
    WDMPacketFFTPlan plan;
    int global_mlo = Nf-1;
    int global_mhi = 1;
    int have_band = 0;
    int ch, m, j, i, block_index;

    if(context == NULL || blocks == NULL || nblocks < 1 ||
       out_tracks == NULL)
        return 1;
    wdms = (struct wdmshape *)&context->wdms;
    for(block_index=0; block_index<nblocks; block_index++)
    {
        int mlo, mhi;
        const THMComplexTDIFFTBlock *block = &blocks[block_index];
        if(!(block->frequency_stop > block->frequency_start))
        {
            global_mlo = 1;
            global_mhi = Nf-1;
            have_band = 1;
            break;
        }
        mlo = (int)ceil((block->frequency_start-wdms->FB)/wdms->DF);
        mhi = (int)floor((block->frequency_stop+wdms->FB)/wdms->DF);
        if(mlo < 1) mlo = 1;
        if(mhi > Nf-1) mhi = Nf-1;
        if(mhi < mlo) continue;
        if(mlo < global_mlo) global_mlo = mlo;
        if(mhi > global_mhi) global_mhi = mhi;
        have_band = 1;
    }
    if(!have_band) return 2;
    for(ch=0; ch<3; ch++) out_tracks->channel[ch].npixels = 0;

    wdm_packet_fft_plan_init(&plan);
    for(m=global_mlo; m<=global_mhi; m++)
    {
        double fcenter = ((double)m)*wdms->DF;
        double layer_low = fcenter-wdms->FB;
        double layer_high = fcenter+wdms->FB;
        double support_start = HUGE_VAL;
        double support_stop = -HUGE_VAL;
        double t0;
        int nlo, nhi, width, Ntx, nmid;
        int relevant = 0;

        for(block_index=0; block_index<nblocks; block_index++)
        {
            const THMComplexTDIFFTBlock *block = &blocks[block_index];
            if(!thm_complex_block_overlaps_layer(block,
                                                  layer_low, layer_high))
                continue;
            if(block->nonzero_start < support_start)
                support_start = block->nonzero_start;
            if(block->nonzero_end > support_stop)
                support_stop = block->nonzero_end;
            relevant = 1;
        }
        if(!relevant || !(support_stop > support_start)) continue;

        nlo = (int)floor(support_start/wdms->DT)-mult;
        nhi = (int)ceil(support_stop/wdms->DT)+mult;
        if(nlo < 0) nlo = 0;
        if(nhi > Nt) nhi = Nt;
        if(nhi <= nlo) continue;
        width = nhi-nlo;
        Ntx = thm_next_power_of_two_int(width);
        if(Ntx > Nt) Ntx = Nt;
        nmid = (nlo+nhi)/2;
        if(nmid & 1) nmid--;
        if(nmid-Ntx/2 < 0) nmid = Ntx/2;
        if(nmid+Ntx/2 > Nt) nmid = Nt-Ntx/2;
        t0 = ((double)(nmid-Ntx/2))*wdms->DT;

        if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms))
        {
            wdm_packet_fft_plan_free(&plan);
            return 3;
        }
        for(ch=0; ch<3; ch++)
        {
            THMSparseWDMChannel *track = &out_tracks->channel[ch];
            if(!thm_sparse_wdm_channel_reserve(track,
                                               track->npixels+Ntx))
            {
                wdm_packet_fft_plan_free(&plan);
                return 4;
            }
            memset(plan.data, 0, (size_t)(2*Ntx)*sizeof(*plan.data));
            for(j=-Ntx/2; j<Ntx/2; j++)
            {
                int jj = j+Ntx/2;
                double f, re, im, white;
                if(jj <= 0) continue;
                f = ((double)j)*plan.dfx+fcenter;
                if(f <= 0.0) continue;
                re = 0.0;
                im = 0.0;
                for(block_index=0; block_index<nblocks; block_index++)
                {
                    const THMComplexTDIFFTBlock *block =
                        &blocks[block_index];
                    double fbase, nyquist, bre, bim, phase, c, s;
                    if(!thm_complex_block_overlaps_layer(
                           block, layer_low, layer_high) ||
                       block->fft[ch] == NULL || block->nfft < 2)
                        continue;
                    fbase = f-block->heterodyne_frequency;
                    nyquist = 0.5/block->sample_dt;
                    if(fabs(fbase) >= nyquist) continue;
                    sample_complex_fft_exact(block->fft[ch], block->nfft,
                                             block->sample_dt, fbase,
                                             &bre, &bim);
                    phase = 2.0*M_PI*f*(t0-block->sample_start);
                    c = cos(phase);
                    s = sin(phase);
                    re += bre*c-bim*s;
                    im += bre*s+bim*c;
                }
                white = thm_instrument_prewhiten_factor(f);
                plan.data[jj] = white*re;
                plan.data[2*Ntx-jj] = white*im;
            }
            wdmtranF_plan(m, &plan);
            for(i=0; i<Ntx; i++)
            {
                int outn = i+nmid-Ntx/2;
                int q;
                if(outn < 0 || outn >= Nt) continue;
                q = track->npixels++;
                track->n[q] = outn;
                track->m[q] = m;
                track->value[q] = plan.wdmout[i];
            }
        }
    }
    wdm_packet_fft_plan_free(&plan);
    return 0;
}

/*
 * Assemble the shared THM carrier blocks on the exact sparse packet plan used
 * by the established split-FFT implementation.  The block spectra are shared
 * across channels, but X, Y, and Z retain their own active packet durations.
 * This avoids inferring a packet's duration from conservative taper-sideband
 * bounds, which can make a short endpoint block incompatible with an
 * unnecessarily year-long Fourier grid.
 */
static int thm_complex_fft_blocks_to_thm_plan_sparse_wdm_context(
    const THMObservationContext *context,
    int nblocks,
    const THMComplexTDIFFTBlock *blocks,
    int **nmid_plan,
    int **nsize_plan,
    THMSparseWDMTriplet *out_tracks)
{
    struct wdmshape *wdms;
    WDMPacketFFTPlan plan;
    int ch, m, j, i, block_index;

    if(context == NULL || blocks == NULL || nblocks < 1 ||
       nmid_plan == NULL || nsize_plan == NULL || out_tracks == NULL)
        return 1;
    if(context->tdi_generation == 2 &&
       thm_wdm_instrument_prewhiten_enabled) return 6;
    wdms = (struct wdmshape *)&context->wdms;
    for(ch=0; ch<3; ch++)
    {
        int capacity = 0;
        out_tracks->channel[ch].npixels = 0;
        for(m=1; m<Nf; m++)
        {
            if(nmid_plan[ch][m] >= 0 && nsize_plan[ch][m] > 0)
                capacity += nsize_plan[ch][m];
        }
        if(!thm_sparse_wdm_channel_reserve(&out_tracks->channel[ch],
                                           capacity))
            return 2;
    }

    wdm_packet_fft_plan_init(&plan);
    for(ch=0; ch<3; ch++)
    {
        THMSparseWDMChannel *track = &out_tracks->channel[ch];
        for(m=1; m<Nf; m++)
        {
            double fcenter, layer_low, layer_high, packet_start, packet_end;
            double t0;
            int Ntx, n;

            if(nmid_plan[ch][m] < 0 || nsize_plan[ch][m] <= 0) continue;
            Ntx = nsize_plan[ch][m];
            n = nmid_plan[ch][m];
            t0 = (double)(n-Ntx/2)*wdms->DT;
            packet_start = t0;
            packet_end = t0+(double)Ntx*wdms->DT;
            fcenter = (double)m*wdms->DF;
            layer_low = fcenter-wdms->FB;
            layer_high = fcenter+wdms->FB;
            if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms))
            {
                wdm_packet_fft_plan_free(&plan);
                return 3;
            }
            memset(plan.data, 0, (size_t)(2*Ntx)*sizeof(*plan.data));
            for(j=-Ntx/2; j<Ntx/2; j++)
            {
                int jj = j+Ntx/2;
                double f, re = 0.0, im = 0.0, white;
                if(jj <= 0) continue;
                f = (double)j*plan.dfx+fcenter;
                if(f <= 0.0) continue;
                for(block_index=0; block_index<nblocks; block_index++)
                {
                    const THMComplexTDIFFTBlock *block =
                        &blocks[block_index];
                    double fbase, nyquist, bre, bim, phase, c, s;
                    if(!thm_complex_block_overlaps_layer(
                           block, layer_low, layer_high) ||
                       !(packet_end > block->nonzero_start &&
                         packet_start < block->nonzero_end) ||
                       block->fft[ch] == NULL || block->nfft < 2)
                        continue;
                    fbase = f-block->heterodyne_frequency;
                    nyquist = 0.5/block->sample_dt;
                    if(fabs(fbase) >= nyquist) continue;
                    sample_complex_fft_exact(block->fft[ch], block->nfft,
                                             block->sample_dt, fbase,
                                             &bre, &bim);
                    phase = 2.0*M_PI*f*(t0-block->sample_start);
                    c = cos(phase);
                    s = sin(phase);
                    re += bre*c-bim*s;
                    im += bre*s+bim*c;
                }
                white = thm_instrument_prewhiten_factor(f);
                plan.data[jj] = white*re;
                plan.data[2*Ntx-jj] = white*im;
            }
            wdmtranF_plan(m, &plan);
            for(i=0; i<Ntx; i++)
            {
                int outn = i+n-Ntx/2;
                int q;
                if(outn < 0 || outn >= Nt) continue;
                q = track->npixels++;
                track->n[q] = outn;
                track->m[q] = m;
                track->value[q] = plan.wdmout[i];
            }
        }
    }
    wdm_packet_fft_plan_free(&plan);
    return 0;
}

int thm_complex_fft_blocks_to_sparse_wdm_plan_context(
    const THMObservationContext *context,
    int nblocks,
    const THMComplexTDIFFTBlock *blocks,
    const int *nmid,
    const int *nsize,
    THMSparseWDMTriplet *out_tracks)
{
    int *nmid_plan[3];
    int *nsize_plan[3];
    int ch;

    if(nmid == NULL || nsize == NULL) return 1;
    for(ch=0; ch<3; ch++)
    {
        nmid_plan[ch] = (int *)nmid;
        nsize_plan[ch] = (int *)nsize;
    }
    return thm_complex_fft_blocks_to_thm_plan_sparse_wdm_context(
        context, nblocks, blocks, nmid_plan, nsize_plan, out_tracks);
}

static void thm_fast_partition_plan_init(THMFastPartitionPlan *plan)
{
    if(plan != NULL) memset(plan, 0, sizeof(*plan));
}

static void thm_fast_partition_plan_free(THMFastPartitionPlan *plan)
{
    int i;
    if(plan == NULL) return;
    for(i=0; i<plan->nblocks; i++)
        thm_complex_tdi_fft_block_free(&plan->blocks[i]);
    free(plan->blocks);
    thm_fast_partition_plan_init(plan);
}

static int thm_fast_partition_plan_reserve(THMFastPartitionPlan *plan,
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

static int thm_fast_partition_track_range(int n, const double *time,
                                          const double *flow,
                                          const double *fhigh,
                                          double tlo, double thi,
                                          double *out_low,
                                          double *out_high)
{
    double lo = HUGE_VAL, hi = -HUGE_VAL;
    int i, have = 0;

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

static int thm_fast_partition_local_bins(const THMWDMGridInfo *grid,
                                         double shift,
                                         double fstart,
                                         double fstop)
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

static void thm_fast_partition_packet_from_bounds(
    const THMWDMGridInfo *grid,
    int jlo,
    int jhi,
    int *nmid,
    int *nsize)
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

/*
 * Build the compact, common packet support used by the direct complex THM
 * blocks.  The nonprecessing carrier is broadened only by a small allowance
 * for the time-dependent detector response.  The endpoint is kept separate:
 * it overlaps the final early block in time through complementary tapers, but
 * it does not force every earlier layer onto the endpoint packet duration.
 */
static int thm_fast_partition_build_compact_plan(
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
    int *early_nmid,
    int *early_nsize,
    int *endpoint_nmid,
    int *endpoint_nsize,
    double *endpoint_frequency_start,
    double *endpoint_frequency_stop)
{
    int *jmin_layer = NULL, *jmax_layer = NULL;
    int i, k, m;
    double early_stop = fmin(time_stop, endpoint_start+endpoint_rise);
    double endpoint_center_low = HUGE_VAL, endpoint_high = -HUGE_VAL;
    double endpoint_margin;
    int status = 1;

    if(grid == NULL || ncarriers < 1 || ntrack < 2 || time == NULL ||
       fcenter == NULL || flow == NULL || fhigh == NULL ||
       early_nmid == NULL || early_nsize == NULL ||
       endpoint_nmid == NULL || endpoint_nsize == NULL ||
       endpoint_frequency_start == NULL ||
       endpoint_frequency_stop == NULL || !(time_stop > time_start) ||
       endpoint_rise <= 0.0)
        return 1;

    jmin_layer = calloc((size_t)grid->frequency_layers,
                        sizeof(*jmin_layer));
    jmax_layer = calloc((size_t)grid->frequency_layers,
                        sizeof(*jmax_layer));
    if(jmin_layer == NULL || jmax_layer == NULL)
    {
        status = 2;
        goto cleanup;
    }
    for(m=0; m<grid->frequency_layers; m++)
    {
        early_nmid[m] = -1;
        early_nsize[m] = 0;
        endpoint_nmid[m] = -1;
        endpoint_nsize[m] = 0;
        jmin_layer[m] = grid->time_pixels+1;
        jmax_layer[m] = -1;
    }

    for(k=0; k<ncarriers; k++)
    {
        const double *carrier_center =
            fcenter+(size_t)k*(size_t)ntrack;
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
                double center_lo = fmin(carrier_center[i],
                                        carrier_center[i+1]);
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

    if(!isfinite(endpoint_center_low) ||
       endpoint_center_low == HUGE_VAL || !isfinite(endpoint_high) ||
       endpoint_high <= 0.0)
    {
        status = 3;
        goto cleanup;
    }
    endpoint_margin = 3.0/endpoint_rise+grid->meyer_half_bandwidth;
    if(endpoint_margin > 0.5*endpoint_center_low)
        endpoint_margin = 0.5*endpoint_center_low;
    *endpoint_frequency_start =
        fmax(0.0, endpoint_center_low-endpoint_margin);
    *endpoint_frequency_stop = endpoint_high+3.0/endpoint_rise+
                               grid->meyer_half_bandwidth;
    if(*endpoint_frequency_stop >
       (double)(grid->frequency_layers-1)*grid->frequency_pixel_df+
       grid->meyer_half_bandwidth)
        *endpoint_frequency_stop =
            (double)(grid->frequency_layers-1)*grid->frequency_pixel_df+
            grid->meyer_half_bandwidth;

    for(m=1; m<grid->frequency_layers; m++)
    {
        if(jmax_layer[m] >= jmin_layer[m])
            thm_fast_partition_packet_from_bounds(
                grid, jmin_layer[m], jmax_layer[m],
                &early_nmid[m], &early_nsize[m]);
    }
    {
        int mlo = (int)ceil((*endpoint_frequency_start-
                             grid->meyer_half_bandwidth)/
                            grid->frequency_pixel_df);
        int mhi = (int)floor((*endpoint_frequency_stop+
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
        thm_fast_partition_packet_from_bounds(
            grid, jlo, jhi, &endpoint_mid, &endpoint_size);
        for(m=mlo; m<=mhi; m++)
        {
            endpoint_nmid[m] = endpoint_mid;
            endpoint_nsize[m] = endpoint_size;
        }
    }
    status = 0;

cleanup:
    free(jmax_layer);
    free(jmin_layer);
    return status;
}

static int thm_fast_partition_add_sparse_channel(
    THMSparseWDMChannel *target,
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
        int n, m;
        double value = 0.0;
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
            n = target->n[ia];
            m = target->m[ia];
            value += target->value[ia++];
        }
        else
        {
            n = add->n[ib];
            m = add->m[ib];
        }
        if(take_add)
        {
            n = add->n[ib];
            m = add->m[ib];
            value += add->value[ib++];
        }
        new_n[out] = n;
        new_m[out] = m;
        new_value[out] = value;
        out++;
    }
    free(target->n);
    free(target->m);
    free(target->value);
    target->n = new_n;
    target->m = new_m;
    target->value = new_value;
    target->npixels = out;
    target->capacity = capacity;
    return 1;
}

/*
 * Complex analytic polarization for one folded carrier, or for their sum
 * when selected_carrier is negative.  This is the nonprecessing counterpart
 * of the selected-carrier callback used by the TPHM block planner.
 */
static int thm_fast_partition_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    THMFastPartitionSource *source = (THMFastPartitionSource *)userdata;
    int i, k, first, stop;

    if(source == NULL || n < 1 || source_time == NULL ||
       hplus_analytic == NULL || hcross_analytic == NULL ||
       source->ncarriers < 1 || source->carrier == NULL)
        return 1;
    first = source->selected_carrier >= 0 ? source->selected_carrier : 0;
    stop = source->selected_carrier >= 0 ? first+1 : source->ncarriers;
    if(first < 0 || stop > source->ncarriers) return 2;

    for(i=0; i<n; i++)
    {
        double hp = 0.0, hc = 0.0, hpq = 0.0, hcq = 0.0;
        if(source_time[i] >= source->time_start &&
           source_time[i] <= source->time_stop)
        {
            for(k=first; k<stop; k++)
            {
                const THMFoldedCarrier *c = &source->carrier[k];
                int q = c->mode_index;
                double amplitude, phase, cp, sp;
                if(q < 0 || source->Aspline[q] == NULL ||
                   source->Pspline[q] == NULL)
                    continue;
                amplitude = gsl_spline_eval(source->Aspline[q], source_time[i],
                                            source->Aacc[q]);
                phase = gsl_spline_eval(source->Pspline[q], source_time[i],
                                        source->Pacc[q]);
                cp = cos(phase);
                sp = sin(phase);
                hp += amplitude*(c->hp_cos*cp+c->hp_sin*sp);
                hc += amplitude*(c->hc_cos*cp+c->hc_sin*sp);
                /*
                 * The complex block represents the analytic signal of the
                 * already-folded +m/-m carrier.  Its quadrature is therefore
                 * obtained by a pi/2 rotation of that folded carrier's real
                 * cosine/sine coefficients.  c->hpf_* and c->hcf_* retain
                 * the separate mode-wise auxiliary quadratures used by other
                 * diagnostics; for a folded pair those are not, in general,
                 * the Hilbert quadrature of hp and hc.
                 */
                hpq += amplitude*(c->hp_sin*cp-c->hp_cos*sp);
                hcq += amplitude*(c->hc_sin*cp-c->hc_cos*sp);
            }
        }
        hplus_analytic[i] = hp-I*hpq;
        hcross_analytic[i] = hc-I*hcq;
    }
    return 0;
}

static int thm_fast_partition_family_polarizations(
    void *userdata,
    int n,
    const double *source_time,
    int nfamilies,
    double complex *hplus_analytic,
    double complex *hcross_analytic)
{
    THMFastPartitionSource *source = (THMFastPartitionSource *)userdata;
    int i, k;

    if(source == NULL || n < 1 || source_time == NULL ||
       hplus_analytic == NULL || hcross_analytic == NULL ||
       nfamilies != source->ncarriers || source->carrier == NULL)
        return 1;
    for(k=0; k<nfamilies; k++)
    {
        const THMFoldedCarrier *c = &source->carrier[k];
        int q = c->mode_index;
        if(q < 0 || source->Aspline[q] == NULL ||
           source->Pspline[q] == NULL)
            return 2;
        for(i=0; i<n; i++)
        {
            size_t index = (size_t)k*(size_t)n+(size_t)i;
            double complex hp = 0.0, hc = 0.0;
            if(source_time[i] >= source->time_start &&
               source_time[i] <= source->time_stop)
            {
                double amplitude = gsl_spline_eval(
                    source->Aspline[q], source_time[i], source->Aacc[q]);
                double phase = gsl_spline_eval(
                    source->Pspline[q], source_time[i], source->Pacc[q]);
                double complex phasor = cos(phase)+I*sin(phase);
                hp = amplitude*(c->hp_cos-I*c->hp_sin)*phasor;
                hc = amplitude*(c->hc_cos-I*c->hc_sin)*phasor;
            }
            hplus_analytic[index] = hp;
            hcross_analytic[index] = hc;
        }
    }
    return 0;
}

static void thm_fast_response_envelope_cache_free(
    THMFastResponseEnvelopeCache *cache)
{
    int q, count;
    if(cache == NULL) return;
    count = 3*cache->ncarriers;
    for(q=0; q<count; q++)
    {
        if(cache->real_spline != NULL && cache->real_spline[q] != NULL)
            gsl_spline_free(cache->real_spline[q]);
        if(cache->imag_spline != NULL && cache->imag_spline[q] != NULL)
            gsl_spline_free(cache->imag_spline[q]);
        if(cache->real_acc != NULL && cache->real_acc[q] != NULL)
            gsl_interp_accel_free(cache->real_acc[q]);
        if(cache->imag_acc != NULL && cache->imag_acc[q] != NULL)
            gsl_interp_accel_free(cache->imag_acc[q]);
    }
    free(cache->imag_acc);
    free(cache->real_acc);
    free(cache->imag_spline);
    free(cache->real_spline);
    free(cache->time);
    memset(cache, 0, sizeof(*cache));
}

/*
 * Factor each sparse detector response as
 *
 *     H_{c k}(t) = exp[i Phi_k(t)] R_{c k}(t).
 *
 * R contains the folded-pair projection, delayed unequal-arm TDI response,
 * and intrinsic amplitude.  Its Cartesian components remain regular through
 * response zeros, unlike an amplitude/phase representation.  The adaptive
 * response grid already contains the transfer-crossing and merger
 * refinements, so it is also the natural envelope grid.
 */
static int thm_build_fast_response_envelope_cache(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    THMFastPartitionSource *source,
    int n,
    const double *detector_time,
    double ecliptic_latitude,
    double ecliptic_longitude,
    THMFastResponseEnvelopeCache *cache)
{
    double complex *channel[3] = {NULL, NULL, NULL};
    double *real_value = NULL, *imag_value = NULL;
    int i, k, ch, status = 0;

    if(context == NULL || workspace == NULL || source == NULL || n < 5 ||
       detector_time == NULL || cache == NULL)
        return 1;
    thm_fast_response_envelope_cache_free(cache);
    cache->n = n;
    cache->ncarriers = source->ncarriers;
    cache->time = calloc((size_t)n, sizeof(*cache->time));
    cache->real_spline = calloc((size_t)(3*source->ncarriers),
                                sizeof(*cache->real_spline));
    cache->imag_spline = calloc((size_t)(3*source->ncarriers),
                                sizeof(*cache->imag_spline));
    cache->real_acc = calloc((size_t)(3*source->ncarriers),
                             sizeof(*cache->real_acc));
    cache->imag_acc = calloc((size_t)(3*source->ncarriers),
                             sizeof(*cache->imag_acc));
    real_value = calloc((size_t)n, sizeof(*real_value));
    imag_value = calloc((size_t)n, sizeof(*imag_value));
    for(ch=0; ch<3; ch++)
        channel[ch] = calloc((size_t)source->ncarriers*(size_t)n,
                             sizeof(*channel[ch]));
    if(cache->time == NULL || cache->real_spline == NULL ||
       cache->imag_spline == NULL || cache->real_acc == NULL ||
       cache->imag_acc == NULL || real_value == NULL ||
       imag_value == NULL || channel[0] == NULL || channel[1] == NULL ||
       channel[2] == NULL)
    {
        status = 2;
        goto cleanup;
    }
    memcpy(cache->time, detector_time, (size_t)n*sizeof(*cache->time));
    status = thm_evaluate_complex_tdi_families_context(
        context, workspace, n, detector_time,
        ecliptic_latitude, ecliptic_longitude, source->ncarriers,
        thm_fast_partition_family_polarizations, source,
        channel[0], channel[1], channel[2]);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    for(ch=0; ch<3; ch++)
    {
        for(k=0; k<source->ncarriers; k++)
        {
            int mode = source->carrier[k].mode_index;
            int q = ch*source->ncarriers+k;
            for(i=0; i<n; i++)
            {
                size_t index = (size_t)k*(size_t)n+(size_t)i;
                double phase = gsl_spline_eval(
                    source->Pspline[mode], detector_time[i],
                    source->Pacc[mode]);
                double complex phasor = cos(phase)+I*sin(phase);
                double complex envelope = channel[ch][index]*conj(phasor);
                real_value[i] = creal(envelope);
                imag_value[i] = cimag(envelope);
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
                status = 3;
                goto cleanup;
            }
        }
    }

cleanup:
    for(ch=0; ch<3; ch++) free(channel[ch]);
    free(imag_value);
    free(real_value);
    if(status != 0) thm_fast_response_envelope_cache_free(cache);
    return status;
}

static int thm_build_fast_response_envelope_block(
    THMFastPartitionSource *source,
    THMFastResponseEnvelopeCache *cache,
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
    int mode, eval_stop, i, ch, status = 0;

    if(source == NULL || cache == NULL || block == NULL || nfft < 2 ||
       carrier_index < 0 || carrier_index >= cache->ncarriers)
        return 1;
    mode = source->carrier[carrier_index].mode_index;
    eval_stop = nfft;
    if(fall_end > fall_start)
    {
        eval_stop = (int)ceil((fall_end-sample_start)/sample_dt)+1;
        if(eval_stop > nfft) eval_stop = nfft;
    }
    if(eval_stop < 2) return 2;
    thm_complex_tdi_fft_block_free(block);
    for(ch=0; ch<3; ch++)
    {
        int q = ch*cache->ncarriers+carrier_index;
        block->fft[ch] = double_vector(2*nfft);
        gsl_interp_accel_reset(cache->real_acc[q]);
        gsl_interp_accel_reset(cache->imag_acc[q]);
    }
    if(block->fft[0] == NULL || block->fft[1] == NULL ||
       block->fft[2] == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(i=0; i<nfft; i++)
    {
        double t = sample_start+sample_dt*(double)i;
        double weight = 1.0;
        double shift_phase = 2.0*M_PI*heterodyne_frequency*
                             (t-sample_start);
        double complex demod = cos(shift_phase)-I*sin(shift_phase);
        if(rise_end > rise_start)
            weight *= split_smooth_step(t, rise_start, rise_end);
        if(fall_end > fall_start)
            weight *= 1.0-split_smooth_step(t, fall_start, fall_end);
        for(ch=0; ch<3; ch++)
        {
            double complex value = 0.0;
            if(i < eval_stop && t >= cache->time[0] &&
               t <= cache->time[cache->n-1])
            {
                int q = ch*cache->ncarriers+carrier_index;
                double carrier_phase = gsl_spline_eval(
                    source->Pspline[mode], t, source->Pacc[mode]);
                double complex phasor = cos(carrier_phase)+
                                        I*sin(carrier_phase);
                double re = gsl_spline_eval(cache->real_spline[q], t,
                                             cache->real_acc[q]);
                double im = gsl_spline_eval(cache->imag_spline[q], t,
                                             cache->imag_acc[q]);
                value = weight*phasor*(re+I*im)*demod;
            }
            REAL(block->fft[ch], i) = creal(value);
            IMAG(block->fft[ch], i) = cimag(value);
        }
    }
    for(ch=0; ch<3; ch++)
    {
        gsl_fft_complex_radix2_forward(block->fft[ch], 1, (size_t)nfft);
        for(i=0; i<nfft; i++)
        {
            REAL(block->fft[ch], i) *= sample_dt;
            IMAG(block->fft[ch], i) *= sample_dt;
        }
    }
    block->nfft = nfft;
    block->sample_start = sample_start;
    block->sample_dt = sample_dt;
    block->heterodyne_frequency = heterodyne_frequency;
    block->nonzero_start = rise_end > rise_start ? rise_start : sample_start;
    block->nonzero_end = fall_end > fall_start ? fall_end :
                         sample_start+sample_dt*(double)(nfft-1);
    return 0;

cleanup:
    thm_complex_tdi_fft_block_free(block);
    return status;
}

/*
 * Production THM partitioned transform.  The default evaluates all folded
 * carrier families through TDI once on the common adaptive detector grid,
 * removes each rapid intrinsic carrier phase, and splines the resulting
 * Cartesian complex response envelopes.  Blocks restore the cheap carrier
 * phasor on their own FFT lattices.  The older blockwise delayed-TDI route is
 * retained below as a regression option.  This is the aligned-spin
 * specialization of the TPHM grouped-envelope construction.
 */
static int thm_direct_partitioned_wdm_context(
    const THMObservationContext *context,
    THMWorkerWorkspace *workspace,
    int ns,
    const double *response_time,
    double **mode_frequency,
    const double *params,
    int ncarriers,
    const THMFoldedCarrier *carrier,
    gsl_interp_accel **Aacc,
    gsl_spline **Aspline,
    gsl_interp_accel **Pacc,
    gsl_spline **Pspline,
    double time_start,
    double time_stop,
    double endpoint_start,
    double endpoint_rise,
    double bandwidth_hz,
    double block_roll_seconds,
    THMSparseWDMTriplet *out_tracks,
    long long *fft_samples_out,
    int *block_count_out)
{
    const double seconds_per_year = 31557600.0;
    THMWDMGridInfo grid;
    THMFastPartitionPlan plan;
    THMFastPartitionBlockSpec *spec = NULL;
    THMFastPartitionSource source;
    THMFastResponseEnvelopeCache response_envelope;
    THMSparseWDMTriplet endpoint_tracks;
    double *track_time = NULL, *fcenter = NULL;
    double *flow = NULL, *fhigh = NULL;
    int *early_nmid = NULL, *early_nsize = NULL;
    int *endpoint_nmid = NULL, *endpoint_nsize = NULL;
    double endpoint_frequency_start = 0.0;
    double endpoint_frequency_stop = 0.0;
    int ntrack, spec_capacity, nspec = 0;
    int start_pixel, endpoint_pixel, roll_pixels;
    int i, k, ch, status = 0;
    clock_t stage_start, total_start;
    double planning_seconds = 0.0, layout_seconds = 0.0;
    double envelope_seconds = 0.0, blocks_seconds = 0.0;
    double endpoint_seconds = 0.0;
    double wdm_seconds = 0.0;
    double source_sample_fraction = 0.0;
    const char *source_sample_fraction_text =
        getenv("THM_PARTITION_SOURCE_SAMPLE_FRACTION");
    const char *complex_envelope_text =
        getenv("THM_COMPLEX_ENVELOPE_GROUPS");
    const char *blockwise_tdi_text = getenv("THM_BLOCKWISE_SPARSE_TDI");
    int use_complex_envelope = thm_wdm_complex_envelope_enabled;

    thm_fast_partition_plan_init(&plan);
    thm_sparse_wdm_triplet_init(&endpoint_tracks);
    memset(&source, 0, sizeof(source));
    memset(&response_envelope, 0, sizeof(response_envelope));
    if(fft_samples_out != NULL) *fft_samples_out = 0;
    if(block_count_out != NULL) *block_count_out = 0;
    if(context == NULL || workspace == NULL || ns < 2 ||
       response_time == NULL || mode_frequency == NULL || params == NULL ||
       ncarriers < 1 || carrier == NULL || Aacc == NULL ||
       Aspline == NULL || Pacc == NULL || Pspline == NULL ||
       out_tracks == NULL || !(time_stop > time_start) ||
       !(endpoint_start > time_start) || !(time_stop > endpoint_start) ||
       endpoint_rise <= 0.0 || bandwidth_hz <= 0.0 ||
       block_roll_seconds <= 0.0)
        return 1;
    if(thm_observation_wdm_grid_info(context, &grid) != 0) return 2;
    if(source_sample_fraction_text != NULL)
    {
        source_sample_fraction = strtod(source_sample_fraction_text, NULL);
        if(!isfinite(source_sample_fraction) || source_sample_fraction < 0.0)
            source_sample_fraction = 0.0;
    }
    if(complex_envelope_text != NULL)
        use_complex_envelope = atoi(complex_envelope_text) != 0;
    if(blockwise_tdi_text != NULL && atoi(blockwise_tdi_text) != 0)
        use_complex_envelope = 0;
    if(time_start < 0.0 ||
       time_stop > grid.time_pixel_dt*(double)grid.time_pixels)
        return 3;
    total_start = clock();

    stage_start = clock();
    ntrack = (int)ceil((time_stop-time_start)/grid.time_pixel_dt)+1;
    if(ntrack < 2) ntrack = 2;
    track_time = calloc((size_t)ntrack, sizeof(*track_time));
    fcenter = calloc((size_t)ncarriers*(size_t)ntrack, sizeof(*fcenter));
    flow = calloc((size_t)ncarriers*(size_t)ntrack, sizeof(*flow));
    fhigh = calloc((size_t)ncarriers*(size_t)ntrack, sizeof(*fhigh));
    early_nmid = calloc((size_t)grid.frequency_layers,
                        sizeof(*early_nmid));
    early_nsize = calloc((size_t)grid.frequency_layers,
                         sizeof(*early_nsize));
    endpoint_nmid = calloc((size_t)grid.frequency_layers,
                           sizeof(*endpoint_nmid));
    endpoint_nsize = calloc((size_t)grid.frequency_layers,
                            sizeof(*endpoint_nsize));
    if(track_time == NULL || fcenter == NULL || flow == NULL ||
       fhigh == NULL || early_nmid == NULL || early_nsize == NULL ||
       endpoint_nmid == NULL || endpoint_nsize == NULL)
    {
        status = 4;
        goto cleanup;
    }
    for(i=0; i<ntrack; i++)
    {
        double t = time_start+grid.time_pixel_dt*(double)i;
        if(t > time_stop) t = time_stop;
        track_time[i] = t;
    }
    for(k=0; k<ncarriers; k++)
    {
        int q = carrier[k].mode_index;
        if(q < 0 || mode_frequency[q] == NULL)
        {
            status = 5;
            goto cleanup;
        }
        for(i=0; i<ntrack; i++)
        {
            size_t index = (size_t)k*(size_t)ntrack+(size_t)i;
            double center = fabs(linear_interp_clamped(
                ns, response_time, mode_frequency[q], track_time[i]));
            double margin = 3.0e-4*center+4.0/seconds_per_year;
            fcenter[index] = center;
            flow[index] = fmax(0.0, center-margin);
            fhigh[index] = center+margin;
        }
    }
    status = thm_fast_partition_build_compact_plan(
        &grid, ncarriers, ntrack, track_time, fcenter, flow, fhigh,
        time_start, time_stop, endpoint_start, endpoint_rise,
        early_nmid, early_nsize, endpoint_nmid, endpoint_nsize,
        &endpoint_frequency_start, &endpoint_frequency_stop);
    if(status != 0)
    {
        status = 10+status;
        goto cleanup;
    }
    planning_seconds = (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;

    start_pixel = (int)floor(time_start/grid.time_pixel_dt);
    endpoint_pixel = (int)floor(endpoint_start/grid.time_pixel_dt);
    if(start_pixel < 0) start_pixel = 0;
    if(endpoint_pixel > grid.time_pixels)
        endpoint_pixel = grid.time_pixels;
    if(endpoint_pixel <= start_pixel)
    {
        status = 6;
        goto cleanup;
    }
    roll_pixels = (int)ceil(block_roll_seconds/grid.time_pixel_dt);
    if(roll_pixels < 1) roll_pixels = 1;
    spec_capacity = ncarriers*(endpoint_pixel-start_pixel);
    if(spec_capacity < ncarriers) spec_capacity = ncarriers;
    spec = calloc((size_t)spec_capacity, sizeof(*spec));
    if(spec == NULL)
    {
        status = 7;
        goto cleanup;
    }

    stage_start = clock();
    for(k=0; k<ncarriers; k++)
    {
        const double *carrier_low = flow+(size_t)k*(size_t)ntrack;
        const double *carrier_high = fhigh+(size_t)k*(size_t)ntrack;
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
            status = 8;
            goto cleanup;
        }
        while(tile_lo < endpoint_pixel)
        {
            int remaining = endpoint_pixel-tile_lo;
            int width = thm_largest_power_of_two_leq_int(remaining);
            int best = 0;
            while(width >= 1)
            {
                double tlo = (double)(tile_lo-roll_pixels)*
                             grid.time_pixel_dt;
                double thi = (double)(tile_lo+width+roll_pixels)*
                             grid.time_pixel_dt;
                double lo, hi;
                double pad = grid.meyer_half_bandwidth+
                             THM_PARTITION_TAPER_BAND_CYCLES/
                             block_roll_seconds;
                if(tlo < time_start) tlo = time_start;
                if(thi > endpoint_start+endpoint_rise)
                    thi = endpoint_start+endpoint_rise;
                if(!thm_fast_partition_track_range(
                       ntrack, track_time, carrier_low, carrier_high,
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
            THMFastPartitionBlockSpec *block_spec;
            double nonzero_start = block == 0 ? time_start :
                                   boundary[block-1];
            double nonzero_stop = block < nblocks-1 ?
                                  boundary[block]+boundary_roll[block] :
                                  endpoint_start+endpoint_rise;
            double rise_start = block == 0 ? time_start :
                                boundary[block-1];
            double fall_start = block < nblocks-1 ? boundary[block] :
                                endpoint_start;
            double fall_end = block < nblocks-1 ?
                              boundary[block]+boundary_roll[block] :
                              endpoint_start+endpoint_rise;
            /* No complementary block exists to the left of the observation
             * boundary.  The first block therefore owns the signal from the
             * first sample and must not apply an unpaired rising taper. */
            double rise_end = block == 0 ? rise_start :
                              boundary[block-1]+boundary_roll[block-1];
            double lo, hi, pad, fstart, fstop, center, shift;
            double sample_start;
            int support_pixels, K, local_bins;
            long long nfft64;

            if(!thm_fast_partition_track_range(
                   ntrack, track_time, carrier_low, carrier_high,
                   nonzero_start, nonzero_stop, &lo, &hi))
            {
                free(boundary_roll);
                free(boundary);
                status = 9;
                goto cleanup;
            }
            pad = grid.meyer_half_bandwidth;
            if(rise_end > rise_start)
                pad = fmax(pad, grid.meyer_half_bandwidth+
                            THM_PARTITION_TAPER_BAND_CYCLES/
                            (rise_end-rise_start));
            if(fall_end > fall_start)
                pad = fmax(pad, grid.meyer_half_bandwidth+
                            THM_PARTITION_TAPER_BAND_CYCLES/
                            (fall_end-fall_start));
            fstart = fmax(0.0, lo-pad);
            fstop = hi+pad;
            center = 0.5*(fstart+fstop);
            shift = 2.0*nearbyint(
                center/(2.0*grid.frequency_pixel_df))*
                grid.frequency_pixel_df;
            local_bins = thm_fast_partition_local_bins(
                &grid, shift, fstart, fstop);
            sample_start = floor(nonzero_start/grid.time_pixel_dt)*
                           grid.time_pixel_dt;
            support_pixels = (int)ceil((nonzero_stop-sample_start)/
                                       grid.time_pixel_dt);
            if(support_pixels < 1) support_pixels = 1;
            K = thm_next_power_of_two_int(
                support_pixels+2*grid.packet_time_half_support);
            if(K > grid.time_pixels) K = grid.time_pixels;
            nfft64 = (long long)K*(long long)local_bins;
            if(nfft64 > 1073741824LL || nspec >= spec_capacity)
            {
                free(boundary_roll);
                free(boundary);
                status = 11;
                goto cleanup;
            }
            block_spec = &spec[nspec++];
            block_spec->carrier_index = k;
            block_spec->nfft = (int)nfft64;
            block_spec->sample_start = sample_start;
            block_spec->sample_dt =
                grid.time_pixel_dt/(double)local_bins;
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

    for(k=0; k<nspec; k++)
    {
        THMFastPartitionBlockSpec *target = &spec[k];
        int local_bins = (int)llround(
            grid.time_pixel_dt/target->sample_dt);
        int K = local_bins > 0 ? target->nfft/local_bins : 0;
        int m;
        if(local_bins < 1 || K < 1)
        {
            status = 12;
            goto cleanup;
        }
        for(m=1; m<grid.frequency_layers; m++)
        {
            double layer_low, layer_high, packet_start, packet_stop;
            int Ntx;
            if(early_nmid[m] < 0 || early_nsize[m] <= 0) continue;
            layer_low = (double)m*grid.frequency_pixel_df-
                        grid.meyer_half_bandwidth;
            layer_high = (double)m*grid.frequency_pixel_df+
                         grid.meyer_half_bandwidth;
            if(layer_high < target->frequency_start ||
               layer_low > target->frequency_stop)
                continue;
            Ntx = early_nsize[m];
            packet_start = (double)(early_nmid[m]-Ntx/2)*
                           grid.time_pixel_dt;
            packet_stop = packet_start+(double)Ntx*grid.time_pixel_dt;
            if(!(packet_stop > target->rise_start &&
                 packet_start < target->fall_end))
                continue;
            if(Ntx > K) K = Ntx;
        }
        if((long long)K*(long long)local_bins > 1073741824LL)
        {
            status = 13;
            goto cleanup;
        }
        target->nfft = K*local_bins;
    }
    if(getenv("THM_PARTITION_DEBUG_BLOCKS") != NULL)
    {
        for(k=0; k<nspec; k++)
        {
            const THMFastPartitionBlockSpec *target = &spec[k];
            fprintf(stderr,
                    "THM_PARTITION_BLOCK index %d carrier %d nfft %d dt %.15e shift %.15e rise %.15e %.15e fall %.15e %.15e band %.15e %.15e\n",
                    k, target->carrier_index, target->nfft,
                    target->sample_dt, target->heterodyne_frequency,
                    target->rise_start, target->rise_end,
                    target->fall_start, target->fall_end,
                    target->frequency_start, target->frequency_stop);
        }
    }
    layout_seconds = (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;

    source.ncarriers = ncarriers;
    source.carrier = carrier;
    source.Aacc = Aacc;
    source.Aspline = Aspline;
    source.Pacc = Pacc;
    source.Pspline = Pspline;
    /*
     * response_time is truncated to the detector-output samples that can be
     * formed safely from the constellation grid.  The intrinsic AP splines,
     * however, deliberately extend beyond that last output knot to supply
     * the delayed endpoint waveform through merger and ringdown.  Clipping
     * the direct complex evaluator to response_time[ns-1] silently removed
     * that tail from the common endpoint FFT.  Use the actual spline domain,
     * just as the full-cadence reference does.
     */
    source.time_start = thm_response_spline_tmin;
    source.time_stop = thm_response_spline_tmax;
    if(getenv("THM_PARTITION_DEBUG") != NULL)
        fprintf(stderr,
                "THM_PARTITION_DEBUG intrinsic_domain %.15e %.15e response_domain %.15e %.15e endpoint %.15e rise %.15e waveform_stop %.15e\n",
                source.time_start, source.time_stop,
                response_time[0], response_time[ns-1],
                endpoint_start, endpoint_rise, time_stop);

    if(use_complex_envelope)
    {
        if(response_time[0] > time_start ||
           response_time[ns-1] < endpoint_start+endpoint_rise)
        {
            status = 16;
            goto cleanup;
        }
        stage_start = clock();
        status = thm_build_fast_response_envelope_cache(
            context, workspace, &source, ns, response_time,
            params[7], params[8], &response_envelope);
        if(status != 0)
        {
            status = 50+status;
            goto cleanup;
        }
        envelope_seconds =
            (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    }

    stage_start = clock();
    /*
     * The default blocks restore the rapid carrier phasor on their local
     * lattices and interpolate the shared complex response envelope.  The
     * fallback below reevaluates delayed TDI separately for every block.  Its
     * optional private source interpolation is retained only for convergence
     * tests because different block cadences can spoil complementary overlaps.
     */
    for(k=0; k<nspec; k++)
    {
        THMFastPartitionBlockSpec *block_spec = &spec[k];
        THMComplexTDIFFTBlock built;
        thm_complex_tdi_fft_block_init(&built);
        source.selected_carrier = block_spec->carrier_index;
        if(use_complex_envelope)
            status = thm_build_fast_response_envelope_block(
                &source, &response_envelope, block_spec->carrier_index,
                block_spec->nfft, block_spec->sample_start,
                block_spec->sample_dt, block_spec->heterodyne_frequency,
                block_spec->rise_start, block_spec->rise_end,
                block_spec->fall_start, block_spec->fall_end, &built);
        else if(source_sample_fraction > 0.0)
            status = thm_build_complex_tdi_fft_block_interpolated_context(
                context, workspace, thm_fast_partition_polarizations, &source,
                params[7], params[8], block_spec->nfft,
                block_spec->sample_start, block_spec->sample_dt,
                source_sample_fraction*block_spec->sample_dt,
                block_spec->heterodyne_frequency,
                block_spec->rise_start, block_spec->rise_end,
                block_spec->fall_start, block_spec->fall_end, &built);
        else
            status = thm_build_complex_tdi_fft_block_context(
                context, workspace, thm_fast_partition_polarizations, &source,
                params[7], params[8], block_spec->nfft,
                block_spec->sample_start, block_spec->sample_dt,
                block_spec->heterodyne_frequency,
                block_spec->rise_start, block_spec->rise_end,
                block_spec->fall_start, block_spec->fall_end, &built);
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
        if(!thm_fast_partition_plan_reserve(&plan, plan.nblocks+1))
        {
            thm_complex_tdi_fft_block_free(&built);
            status = 14;
            goto cleanup;
        }
        plan.blocks[plan.nblocks++] = built;
        plan.total_fft_samples += block_spec->nfft;
    }

    blocks_seconds = (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;

    {
        THMComplexTDIFFTBlock endpoint;
        double endpoint_sample_start = floor(endpoint_start/
            grid.time_pixel_dt)*grid.time_pixel_dt;
        int support_pixels = (int)ceil((time_stop-endpoint_sample_start)/
                                      grid.time_pixel_dt);
        int K, nfft, m;
        if(support_pixels < 1) support_pixels = 1;
        K = thm_next_power_of_two_int(
            support_pixels+2*grid.packet_time_half_support);
        for(m=1; m<grid.frequency_layers; m++)
        {
            double layer_low = (double)m*grid.frequency_pixel_df-
                               grid.meyer_half_bandwidth;
            double layer_high = (double)m*grid.frequency_pixel_df+
                                grid.meyer_half_bandwidth;
            if(endpoint_nmid[m] < 0 || endpoint_nsize[m] <= 0 ||
               layer_high < endpoint_frequency_start ||
               layer_low > endpoint_frequency_stop)
                continue;
            if(endpoint_nsize[m] > K) K = endpoint_nsize[m];
        }
        if(K > grid.time_pixels) K = grid.time_pixels;
        if((long long)K*(long long)grid.frequency_layers > 1073741824LL)
        {
            status = 15;
            goto cleanup;
        }
        nfft = K*grid.frequency_layers;
        thm_complex_tdi_fft_block_init(&endpoint);
        source.selected_carrier = -1;
        stage_start = clock();
        /* The physical ringdown is already negligible at time_stop.  Use a
         * one-sample numerical close rather than the ordinary 100 ks block
         * taper, which would otherwise begin before merger for this compact
         * endpoint interval and suppress the signal being transformed. */
        if(source_sample_fraction > 0.0)
            status = thm_build_complex_tdi_fft_block_interpolated_context(
                context, workspace, thm_fast_partition_polarizations, &source,
                params[7], params[8], nfft, endpoint_sample_start,
                grid.sample_dt, source_sample_fraction*grid.sample_dt, 0.0,
                endpoint_start, endpoint_start+endpoint_rise,
                time_stop, time_stop+grid.sample_dt, &endpoint);
        else
            status = thm_build_complex_tdi_fft_block_context(
                context, workspace, thm_fast_partition_polarizations, &source,
                params[7], params[8], nfft, endpoint_sample_start,
                grid.sample_dt, 0.0,
                endpoint_start, endpoint_start+endpoint_rise,
                time_stop, time_stop+grid.sample_dt, &endpoint);
        if(status != 0)
        {
            thm_complex_tdi_fft_block_free(&endpoint);
            status = 200+status;
            goto cleanup;
        }
        {
            double total = 0.0, tail = 0.0;
            double spectral_stop = endpoint_frequency_stop;
            int bin, old_mhi = -1, new_mhi;
            for(bin=0; bin<=nfft/2; bin++)
            {
                for(ch=0; ch<3; ch++)
                    total += REAL(endpoint.fft[ch], bin)*REAL(endpoint.fft[ch], bin)+
                             IMAG(endpoint.fft[ch], bin)*IMAG(endpoint.fft[ch], bin);
            }
            if(total > 0.0)
            {
                for(bin=nfft/2; bin>=0; bin--)
                {
                    for(ch=0; ch<3; ch++)
                        tail += REAL(endpoint.fft[ch], bin)*REAL(endpoint.fft[ch], bin)+
                                IMAG(endpoint.fft[ch], bin)*IMAG(endpoint.fft[ch], bin);
                    if(tail > 1.0e-8*total)
                    {
                        spectral_stop = (double)bin/((double)nfft*grid.sample_dt)+
                                        grid.meyer_half_bandwidth;
                        break;
                    }
                }
                if(spectral_stop > 0.5/grid.sample_dt)
                    spectral_stop = 0.5/grid.sample_dt;
            }
            for(m=1; m<grid.frequency_layers; m++)
                if(endpoint_nmid[m] >= 0 && endpoint_nsize[m] > 0)
                    old_mhi = m;
            new_mhi = (int)floor((spectral_stop+
                                 grid.meyer_half_bandwidth)/grid.frequency_pixel_df);
            if(new_mhi >= grid.frequency_layers)
                new_mhi = grid.frequency_layers-1;
            if(old_mhi >= 1 && new_mhi > old_mhi)
            {
                for(m=old_mhi+1; m<=new_mhi; m++)
                {
                    endpoint_nmid[m] = endpoint_nmid[old_mhi];
                    endpoint_nsize[m] = endpoint_nsize[old_mhi];
                }
                endpoint_frequency_stop = spectral_stop;
            }
        }
        if(getenv("THM_PARTITION_DEBUG_ENDPOINT") != NULL)
        {
            double *td = double_vector(nfft);
            double *xd = double_vector(nfft);
            double *yd = double_vector(nfft);
            double *zd = double_vector(nfft);
            double *copy[3] = {NULL, NULL, NULL};
            double error[3] = {0.0, 0.0, 0.0};
            double norm[3] = {0.0, 0.0, 0.0};
            double fast_norm[3] = {0.0, 0.0, 0.0};
            double cross_norm[3] = {0.0, 0.0, 0.0};
            int q, dch;
            for(dch=0; dch<3; dch++)
            {
                copy[dch] = double_vector(2*nfft);
                if(copy[dch] != NULL)
                {
                    memcpy(copy[dch], endpoint.fft[dch],
                           (size_t)(2*nfft)*sizeof(*copy[dch]));
                    gsl_fft_complex_radix2_backward(copy[dch], 1,
                                                    (size_t)nfft);
                }
            }
            if(td != NULL && xd != NULL && yd != NULL && zd != NULL &&
               copy[0] != NULL && copy[1] != NULL && copy[2] != NULL)
            {
                for(q=0; q<nfft; q++)
                    td[q] = endpoint_sample_start+grid.sample_dt*(double)q;
                direct_response_full_waveform_thm(
                    td, nfft, grid.sample_dt,
                    500.0+4.0*THM_TDI_LARM_NOMINAL_SECONDS,
                    (double *)params, ncarriers, carrier,
                    workspace->SLacc, (gsl_spline **)context->SLspline,
                    workspace->SPacc, (gsl_spline **)context->SPspline,
                    workspace->SVacc, (gsl_spline **)context->SVspline,
                    Aacc, Aspline, Pacc, Pspline, xd, yd, zd);
                for(q=0; q<nfft; q++)
                {
                    double weight = split_smooth_step(
                        td[q], endpoint_start,
                        endpoint_start+endpoint_rise);
                    weight *= 1.0-split_smooth_step(
                        td[q], time_stop, time_stop+grid.sample_dt);
                    double direct[3] = {weight*xd[q], weight*yd[q],
                                        weight*zd[q]};
                    for(dch=0; dch<3; dch++)
                    {
                        double fast = REAL(copy[dch], q)/
                                      ((double)nfft*grid.sample_dt);
                        double delta = fast-direct[dch];
                        error[dch] += delta*delta;
                        norm[dch] += direct[dch]*direct[dch];
                        fast_norm[dch] += fast*fast;
                        cross_norm[dch] += fast*direct[dch];
                    }
                }
                for(dch=0; dch<3; dch++)
                    fprintf(stderr,
                            "THM_PARTITION_DEBUG endpoint_time_channel %d rel_l2 %.15e amplitude_ratio %.15e projection_scale %.15e match %.15e\n",
                            dch, norm[dch] > 0.0 ?
                            sqrt(error[dch]/norm[dch]) : 0.0,
                            norm[dch] > 0.0 ?
                            sqrt(fast_norm[dch]/norm[dch]) : 0.0,
                            norm[dch] > 0.0 ?
                            cross_norm[dch]/norm[dch] : 0.0,
                            norm[dch] > 0.0 && fast_norm[dch] > 0.0 ?
                            cross_norm[dch]/sqrt(norm[dch]*fast_norm[dch]) :
                            0.0);
            }
            for(dch=0; dch<3; dch++) free_double_vector(copy[dch]);
            free_double_vector(zd);
            free_double_vector(yd);
            free_double_vector(xd);
            free_double_vector(td);
        }
        endpoint.frequency_start = endpoint_frequency_start;
        endpoint.frequency_stop = endpoint_frequency_stop;
        endpoint.carrier_index = -1;
        endpoint.is_endpoint = 1;
        if(!thm_fast_partition_plan_reserve(&plan, plan.nblocks+1))
        {
            thm_complex_tdi_fft_block_free(&endpoint);
            status = 16;
            goto cleanup;
        }
        plan.blocks[plan.nblocks++] = endpoint;
        plan.total_fft_samples += nfft;
        endpoint_seconds =
            (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;
    }

    stage_start = clock();
    status = thm_complex_fft_blocks_to_sparse_wdm_plan_context(
        context, plan.nblocks-1, plan.blocks,
        early_nmid, early_nsize, out_tracks);
    if(status != 0)
    {
        status = 300+status;
        goto cleanup;
    }
    status = thm_complex_fft_blocks_to_sparse_wdm_plan_context(
        context, 1, &plan.blocks[plan.nblocks-1],
        endpoint_nmid, endpoint_nsize, &endpoint_tracks);
    if(status != 0)
    {
        status = 400+status;
        goto cleanup;
    }
    for(ch=0; ch<3; ch++)
    {
        if(!thm_fast_partition_add_sparse_channel(
               &out_tracks->channel[ch], &endpoint_tracks.channel[ch]))
        {
            status = 500;
            goto cleanup;
        }
    }
    wdm_seconds = (double)(clock()-stage_start)/(double)CLOCKS_PER_SEC;

    if(fft_samples_out != NULL) *fft_samples_out = plan.total_fft_samples;
    if(block_count_out != NULL) *block_count_out = plan.nblocks;
    if(getenv("THM_PARTITION_DEBUG") != NULL)
    {
        int pixels = out_tracks->channel[0].npixels+
                     out_tracks->channel[1].npixels+
                     out_tracks->channel[2].npixels;
        fprintf(stderr,
                "THM_PARTITION_DEBUG direct blocks %d fft_samples %lld pixels %d planning %.6f layout %.6f envelope %.6f early_blocks %.6f endpoint %.6f wdm %.6f total %.6f endpoint_band %.15e %.15e method %s\n",
                plan.nblocks, plan.total_fft_samples, pixels,
                planning_seconds, layout_seconds, envelope_seconds,
                blocks_seconds,
                endpoint_seconds, wdm_seconds,
                (double)(clock()-total_start)/(double)CLOCKS_PER_SEC,
                endpoint_frequency_start, endpoint_frequency_stop,
                use_complex_envelope ? "complex_envelope" :
                                       "blockwise_sparse_tdi");
    }
    status = 0;

cleanup:
    thm_fast_response_envelope_cache_free(&response_envelope);
    thm_sparse_wdm_triplet_free(&endpoint_tracks);
    thm_fast_partition_plan_free(&plan);
    free(spec);
    free(endpoint_nsize);
    free(endpoint_nmid);
    free(early_nsize);
    free(early_nmid);
    free(fhigh);
    free(flow);
    free(fcenter);
    free(track_time);
    return status;
}

/*
 * Build one THM partition block from the established post-TDI amplitude and
 * phase splines.  The old, well-tested split-FFT path transforms precisely
 * this interpolant, so using it here is important: reevaluating the delayed
 * complex response on each block's private lattice produces a different
 * waveform between adaptive response knots.
 *
 * A non-negative selected_carrier builds its positive-frequency analytic
 * carrier A exp(i Phi).  A negative selected_carrier builds the real sum used
 * by the common merger/ringdown endpoint; its FFT is multiplied by 2 dt, as in
 * the real-FFT reference, so both frequency mirrors are retained exactly.
 */
static int thm_build_posttdi_ap_fft_block(
    int ncarriers,
    int selected_carrier,
    gsl_interp_accel **ATacc,
    gsl_spline **ATspline,
    gsl_interp_accel **PTacc,
    gsl_spline **PTspline,
    double valid_start,
    double valid_stop,
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
    double sample_stop, fft_scale;
    int i, ch, first, stop;

    if(ncarriers < 1 || ATacc == NULL || ATspline == NULL ||
       PTacc == NULL || PTspline == NULL || block == NULL || nfft < 2 ||
       (nfft & (nfft-1)) != 0 || !isfinite(sample_start) ||
       !isfinite(sample_dt) || sample_dt <= 0.0 ||
       !isfinite(heterodyne_frequency) || !(valid_stop > valid_start))
        return 1;
    if(selected_carrier >= ncarriers) return 2;

    first = selected_carrier >= 0 ? selected_carrier : 0;
    stop = selected_carrier >= 0 ? first+1 : ncarriers;
    sample_stop = sample_start+sample_dt*(double)(nfft-1);
    thm_complex_tdi_fft_block_free(block);
    for(ch=0; ch<3; ch++)
    {
        int k;
        block->fft[ch] = double_vector(2*nfft);
        if(block->fft[ch] == NULL)
        {
            thm_complex_tdi_fft_block_free(block);
            return 3;
        }
        for(k=first; k<stop; k++)
        {
            int idx = ch*ncarriers+k;
            if(ATspline[idx] != NULL) gsl_interp_accel_reset(ATacc[idx]);
            if(PTspline[idx] != NULL) gsl_interp_accel_reset(PTacc[idx]);
        }
    }

    for(i=0; i<nfft; i++)
    {
        double t = sample_start+sample_dt*(double)i;
        double tau = t-sample_start;
        double weight = 1.0;
        double shift_phase = 2.0*M_PI*heterodyne_frequency*tau;
        double complex heterodyne = cos(shift_phase)-I*sin(shift_phase);

        if(rise_end > rise_start)
            weight *= split_smooth_step(t, rise_start, rise_end);
        if(fall_end > fall_start)
            weight *= 1.0-split_smooth_step(t, fall_start, fall_end);
        if(t < valid_start || t > valid_stop) weight = 0.0;

        for(ch=0; ch<3; ch++)
        {
            double complex value = 0.0;
            int k;
            if(weight != 0.0)
            {
                for(k=first; k<stop; k++)
                {
                    int idx = ch*ncarriers+k;
                    double amplitude, phase;
                    if(ATspline[idx] == NULL || PTspline[idx] == NULL)
                        continue;
                    amplitude = gsl_spline_eval(ATspline[idx], t, ATacc[idx]);
                    phase = gsl_spline_eval(PTspline[idx], t, PTacc[idx]);
                    if(selected_carrier >= 0)
                        value += amplitude*(cos(phase)+I*sin(phase));
                    else
                        value += amplitude*cos(phase);
                }
                value *= weight*heterodyne;
            }
            REAL(block->fft[ch], i) = creal(value);
            IMAG(block->fft[ch], i) = cimag(value);
        }
    }

    fft_scale = (selected_carrier >= 0 ? 1.0 : 2.0)*sample_dt;
    for(ch=0; ch<3; ch++)
    {
        gsl_fft_complex_radix2_forward(block->fft[ch], 1, (size_t)nfft);
        for(i=0; i<nfft; i++)
        {
            REAL(block->fft[ch], i) *= fft_scale;
            IMAG(block->fft[ch], i) *= fft_scale;
        }
    }
    block->nfft = nfft;
    block->sample_start = sample_start;
    block->sample_dt = sample_dt;
    block->heterodyne_frequency = heterodyne_frequency;
    block->nonzero_start = rise_end > rise_start ? rise_start : valid_start;
    block->nonzero_end = fall_end > fall_start ? fall_end :
                         fmin(valid_stop, sample_stop);
    return 0;
}

/*
 * Fast THM partitioned transform.  This deliberately mirrors the TPHM
 * narrow-block implementation: all three TDI channels are made together,
 * every folded carrier has its own bandwidth-limited complex heterodyne, and
 * one full-band endpoint block contains the sum of all carriers.
 */
static int thm_fast_partitioned_wdm_context(
    const THMObservationContext *context,
    int ns,
    const double *response_time,
    int ncarriers,
    double ***Achan,
    double ***frequency_track,
    double ***setup_carrier,
    gsl_interp_accel **ATacc,
    gsl_spline **ATspline,
    gsl_interp_accel **PTacc,
    gsl_spline **PTspline,
    double endpoint_start,
    double endpoint_rise,
    double waveform_stop,
    double endpoint_frequency_stop,
    double bandwidth_hz,
    double block_roll_seconds,
    THMSparseWDMTriplet *out_tracks,
    long long *fft_samples_out,
    int *block_count_out)
{
    const double seconds_per_year = 31557600.0;
    THMWDMGridInfo grid;
    THMFastPartitionPlan plan;
    THMFastPartitionBlockSpec *spec = NULL;
    int **nmid_plan = NULL, **nsize_plan = NULL;
    double *flow = NULL, *fhigh = NULL;
    double endpoint_frequency_start = 0.0;
    int spec_capacity, nspec = 0;
    int start_pixel, endpoint_pixel, roll_pixels;
    int i, k, ch, status = 0;

    thm_fast_partition_plan_init(&plan);
    if(fft_samples_out != NULL) *fft_samples_out = 0;
    if(block_count_out != NULL) *block_count_out = 0;
    if(context == NULL || ns < 2 ||
       response_time == NULL || ncarriers < 1 ||
       Achan == NULL || frequency_track == NULL || setup_carrier == NULL ||
       ATacc == NULL || ATspline == NULL ||
       PTacc == NULL || PTspline == NULL || out_tracks == NULL ||
       !(endpoint_start > response_time[0]) ||
       !(waveform_stop > endpoint_start) || endpoint_rise <= 0.0 ||
       bandwidth_hz <= 0.0 || block_roll_seconds <= 0.0 ||
       endpoint_frequency_stop <= 0.0)
        return 1;
    if(thm_observation_wdm_grid_info(context, &grid) != 0) return 2;

    nmid_plan = int_matrix(3, Nf);
    nsize_plan = int_matrix(3, Nf);
    if(nmid_plan == NULL || nsize_plan == NULL)
    {
        status = 3;
        goto cleanup;
    }
    for(ch=0; ch<3; ch++)
    {
        WDMbuildTHMUnionPixelPlan(ns, (double *)response_time, ncarriers,
                                  ch, Achan, frequency_track, setup_carrier,
                                  (struct wdmshape *)&context->wdms, 1,
                                  nmid_plan[ch], nsize_plan[ch]);
    }

    flow = calloc((size_t)ncarriers*(size_t)ns, sizeof(*flow));
    fhigh = calloc((size_t)ncarriers*(size_t)ns, sizeof(*fhigh));
    if(flow == NULL || fhigh == NULL)
    {
        status = 4;
        goto cleanup;
    }
    for(k=0; k<ncarriers; k++)
    {
        for(i=0; i<ns; i++)
        {
            double lo = HUGE_VAL, hi = -HUGE_VAL, center, margin;
            for(ch=0; ch<3; ch++)
            {
                double f = frequency_track[ch][k][i];
                if(!isfinite(f) || f < 0.0) continue;
                if(f < lo) lo = f;
                if(f > hi) hi = f;
            }
            if(!isfinite(lo) || !isfinite(hi)) lo = hi = 0.0;
            center = 0.5*(lo+hi);
            margin = 3.0e-4*center+4.0/seconds_per_year;
            flow[(size_t)k*(size_t)ns+(size_t)i] = fmax(0.0, lo-margin);
            fhigh[(size_t)k*(size_t)ns+(size_t)i] = hi+margin;
        }
    }

    {
        double f_late_start = HUGE_VAL;
        double endpoint_margin = 0.0;
        for(k=0; k<ncarriers; k++)
        {
            for(ch=0; ch<3; ch++)
            {
                double f = linear_interp_clamped(
                    ns, response_time, frequency_track[ch][k],
                    endpoint_start);
                if(isfinite(f) && f > 0.0 && f < f_late_start)
                    f_late_start = f;
            }
        }
        if(isfinite(f_late_start) && f_late_start < HUGE_VAL &&
           thm_split_fft_endpoint_margin_cycles > 0.0)
        {
            endpoint_margin =
                thm_split_fft_endpoint_margin_cycles/endpoint_rise+
                grid.meyer_half_bandwidth;
            if(endpoint_margin >
               SPLIT_FFT_ENDPOINT_MARGIN_FRACTION_MAX*f_late_start)
                endpoint_margin =
                    SPLIT_FFT_ENDPOINT_MARGIN_FRACTION_MAX*f_late_start;
            endpoint_frequency_start = f_late_start-endpoint_margin;
            if(endpoint_frequency_start < 0.0)
                endpoint_frequency_start = 0.0;
        }
        if(getenv("THM_PARTITION_DEBUG") != NULL)
            fprintf(stderr,
                    "THM_PARTITION_DEBUG fast_endpoint_frequency_start %.15e f_late_start %.15e margin %.15e\n",
                    endpoint_frequency_start, f_late_start, endpoint_margin);
    }

    start_pixel = (int)floor(response_time[0]/grid.time_pixel_dt);
    endpoint_pixel = (int)floor(endpoint_start/grid.time_pixel_dt);
    if(start_pixel < 0) start_pixel = 0;
    if(endpoint_pixel > grid.time_pixels) endpoint_pixel = grid.time_pixels;
    if(endpoint_pixel <= start_pixel)
    {
        status = 4;
        goto cleanup;
    }
    roll_pixels = (int)ceil(block_roll_seconds/grid.time_pixel_dt);
    if(roll_pixels < 1) roll_pixels = 1;
    spec_capacity = ncarriers*(endpoint_pixel-start_pixel);
    if(spec_capacity < ncarriers) spec_capacity = ncarriers;
    spec = calloc((size_t)spec_capacity, sizeof(*spec));
    if(spec == NULL)
    {
        status = 5;
        goto cleanup;
    }

    for(k=0; k<ncarriers; k++)
    {
        const double *carrier_low = &flow[(size_t)k*(size_t)ns];
        const double *carrier_high = &fhigh[(size_t)k*(size_t)ns];
        double *boundary = NULL, *boundary_roll = NULL;
        int max_blocks = endpoint_pixel-start_pixel+1;
        int nblocks = 0, tile_lo = start_pixel, block;

        boundary = calloc((size_t)max_blocks, sizeof(*boundary));
        boundary_roll = calloc((size_t)max_blocks, sizeof(*boundary_roll));
        if(boundary == NULL || boundary_roll == NULL)
        {
            free(boundary_roll);
            free(boundary);
            status = 6;
            goto cleanup;
        }
        while(tile_lo < endpoint_pixel)
        {
            int remaining = endpoint_pixel-tile_lo;
            int width = thm_largest_power_of_two_leq_int(remaining);
            int best = 0;
            while(width >= 1)
            {
                double tlo = (double)(tile_lo-roll_pixels)*grid.time_pixel_dt;
                double thi = (double)(tile_lo+width+roll_pixels)*
                             grid.time_pixel_dt;
                double lo, hi;
                double pad = grid.meyer_half_bandwidth+3.0/block_roll_seconds;
                if(tlo < response_time[0]) tlo = response_time[0];
                if(thi > endpoint_start+endpoint_rise)
                    thi = endpoint_start+endpoint_rise;
                if(!thm_fast_partition_track_range(
                       ns, response_time, carrier_low, carrier_high,
                       tlo, thi, &lo, &hi) || hi-lo+2.0*pad <= bandwidth_hz)
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
            THMFastPartitionBlockSpec *block_spec;
            double nonzero_start = block == 0 ? response_time[0] :
                                   boundary[block-1];
            double nonzero_stop = block < nblocks-1 ?
                                  boundary[block]+boundary_roll[block] :
                                  endpoint_start+endpoint_rise;
            double rise_start = block == 0 ? response_time[0] :
                                boundary[block-1];
            /* The first block already owns the left observation boundary.
             * With no preceding block there is no complementary rising
             * taper, so applying one here would remove physical signal and
             * violate the partition of unity. */
            double rise_end = block == 0 ? rise_start :
                              boundary[block-1]+boundary_roll[block-1];
            double fall_start = block < nblocks-1 ? boundary[block] :
                                endpoint_start;
            double fall_end = block < nblocks-1 ?
                              boundary[block]+boundary_roll[block] :
                              endpoint_start+endpoint_rise;
            double lo, hi, pad, fstart, fstop, center, shift, sample_start;
            int support_pixels, K, local_bins, nfft;
            long long nfft64;

            if(!thm_fast_partition_track_range(
                   ns, response_time, carrier_low, carrier_high,
                   nonzero_start, nonzero_stop, &lo, &hi))
            {
                status = 7;
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
            local_bins = thm_fast_partition_local_bins(&grid, shift,
                                                       fstart, fstop);
            sample_start = floor(nonzero_start/grid.time_pixel_dt)*
                           grid.time_pixel_dt;
            support_pixels = (int)ceil((nonzero_stop-sample_start)/
                                       grid.time_pixel_dt);
            if(support_pixels < 1) support_pixels = 1;
            K = thm_next_power_of_two_int(
                support_pixels+2*grid.packet_time_half_support);
            if(K > grid.time_pixels) K = grid.time_pixels;
            nfft64 = (long long)K*(long long)local_bins;
            if(nfft64 > 1073741824LL)
            {
                status = 8;
                free(boundary_roll);
                free(boundary);
                goto cleanup;
            }
            if(nspec >= spec_capacity)
            {
                status = 9;
                free(boundary_roll);
                free(boundary);
                goto cleanup;
            }
            nfft = (int)nfft64;
            block_spec = &spec[nspec++];
            block_spec->carrier_index = k;
            block_spec->nfft = nfft;
            block_spec->sample_start = sample_start;
            block_spec->sample_dt = grid.time_pixel_dt/(double)local_bins;
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
     * A packet of Ntx WDM time pixels samples the local spectrum on the
     * 1/(Ntx*DT) frequency grid.  Every partition block contributing to that
     * packet must have at least the same zero-padded duration.  Use the actual
     * track-pixel plan here, rather than the much wider taper-sideband bounds,
     * so the padding agrees exactly with the established implementation.
     */
    for(k=0; k<nspec; k++)
    {
        THMFastPartitionBlockSpec *target = &spec[k];
        int local_bins = (int)llround(grid.time_pixel_dt/target->sample_dt);
        int K = local_bins > 0 ? target->nfft/local_bins : 0;
        int m;
        if(local_bins < 1 || K < 1)
        {
            status = 13;
            goto cleanup;
        }
        for(m=1; m<grid.frequency_layers; m++)
        {
            double layer_low = (double)m*grid.frequency_pixel_df-
                               grid.meyer_half_bandwidth;
            double layer_high = (double)m*grid.frequency_pixel_df+
                                grid.meyer_half_bandwidth;
            int Ntx = 0;

            if(layer_high < target->frequency_start ||
               layer_low > target->frequency_stop)
                continue;
            for(ch=0; ch<3; ch++)
            {
                if(nmid_plan[ch][m] >= 0 &&
                   nsize_plan[ch][m] > Ntx)
                    Ntx = nsize_plan[ch][m];
            }
            if(Ntx > K) K = Ntx;
        }
        if((long long)K*(long long)local_bins > 1073741824LL)
        {
            status = 14;
            goto cleanup;
        }
        target->nfft = K*local_bins;
        if(getenv("THM_PARTITION_DEBUG") != NULL)
            fprintf(stderr,
                    "THM_PARTITION_DEBUG fast_block %d carrier %d nfft %d K %d local_bins %d dt %.15e shift %.15e time %.15e %.15e band %.15e %.15e\n",
                    k, target->carrier_index, target->nfft, K, local_bins,
                    target->sample_dt, target->heterodyne_frequency,
                    target->rise_start, target->fall_end,
                    target->frequency_start, target->frequency_stop);
    }

    for(k=0; k<nspec; k++)
    {
        THMComplexTDIFFTBlock built;
        THMFastPartitionBlockSpec *block_spec = &spec[k];
        thm_complex_tdi_fft_block_init(&built);
        status = thm_build_posttdi_ap_fft_block(
            ncarriers, block_spec->carrier_index,
            ATacc, ATspline, PTacc, PTspline,
            response_time[0], response_time[ns-1],
            block_spec->nfft, block_spec->sample_start,
            block_spec->sample_dt, block_spec->heterodyne_frequency,
            block_spec->rise_start, block_spec->rise_end,
            block_spec->fall_start, block_spec->fall_end, &built);
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
        if(!thm_fast_partition_plan_reserve(&plan, plan.nblocks+1))
        {
            thm_complex_tdi_fft_block_free(&built);
            status = 10;
            goto cleanup;
        }
        plan.blocks[plan.nblocks++] = built;
        plan.total_fft_samples += block_spec->nfft;
    }

    {
        THMComplexTDIFFTBlock endpoint;
        double endpoint_sample_start = floor(endpoint_start/
            grid.time_pixel_dt)*grid.time_pixel_dt;
        int support_pixels = (int)ceil((waveform_stop-endpoint_sample_start)/
                                      grid.time_pixel_dt);
        int K, nfft, m;
        if(support_pixels < 1) support_pixels = 1;
        K = thm_next_power_of_two_int(
            support_pixels+2*grid.packet_time_half_support);
        for(m=1; m<grid.frequency_layers; m++)
        {
            double layer_low = (double)m*grid.frequency_pixel_df-
                               grid.meyer_half_bandwidth;
            double layer_high = (double)m*grid.frequency_pixel_df+
                                grid.meyer_half_bandwidth;
            if(layer_high < endpoint_frequency_start ||
               layer_low > endpoint_frequency_stop)
                continue;
            for(ch=0; ch<3; ch++)
            {
                if(nmid_plan[ch][m] >= 0 && nsize_plan[ch][m] > K)
                    K = nsize_plan[ch][m];
            }
        }
        if(K > grid.time_pixels) K = grid.time_pixels;
        if((long long)K*(long long)grid.frequency_layers > 1073741824LL)
        {
            status = 11;
            goto cleanup;
        }
        nfft = K*grid.frequency_layers;
        thm_complex_tdi_fft_block_init(&endpoint);
        status = thm_build_posttdi_ap_fft_block(
            ncarriers, -1, ATacc, ATspline, PTacc, PTspline,
            response_time[0], waveform_stop, nfft,
            endpoint_sample_start, grid.sample_dt, 0.0,
            endpoint_start, endpoint_start+endpoint_rise,
            waveform_stop, waveform_stop, &endpoint);
        if(status != 0)
        {
            thm_complex_tdi_fft_block_free(&endpoint);
            status = 200+status;
            goto cleanup;
        }
        {
            double total = 0.0, tail = 0.0;
            double spectral_stop = endpoint_frequency_stop;
            int bin, old_mhi, new_mhi;
            for(bin=0; bin<=nfft/2; bin++)
            {
                for(ch=0; ch<3; ch++)
                    total += REAL(endpoint.fft[ch], bin)*REAL(endpoint.fft[ch], bin)+
                             IMAG(endpoint.fft[ch], bin)*IMAG(endpoint.fft[ch], bin);
            }
            if(total > 0.0)
            {
                for(bin=nfft/2; bin>=0; bin--)
                {
                    for(ch=0; ch<3; ch++)
                        tail += REAL(endpoint.fft[ch], bin)*REAL(endpoint.fft[ch], bin)+
                                IMAG(endpoint.fft[ch], bin)*IMAG(endpoint.fft[ch], bin);
                    if(tail > 1.0e-8*total)
                    {
                        spectral_stop = (double)bin/((double)nfft*grid.sample_dt)+
                                        grid.meyer_half_bandwidth;
                        break;
                    }
                }
                if(spectral_stop > 0.5/grid.sample_dt)
                    spectral_stop = 0.5/grid.sample_dt;
            }
            if(spectral_stop > endpoint_frequency_stop)
            {
                int endpoint_lo = (int)floor(endpoint.nonzero_start/grid.time_pixel_dt)-
                                  grid.packet_time_half_support;
                int endpoint_hi = (int)ceil(endpoint.nonzero_end/grid.time_pixel_dt)+
                                  grid.packet_time_half_support;
                if(endpoint_lo < 0) endpoint_lo = 0;
                if(endpoint_hi > grid.time_pixels) endpoint_hi = grid.time_pixels;
                old_mhi = (int)floor((endpoint_frequency_stop+
                                     grid.meyer_half_bandwidth)/grid.frequency_pixel_df);
                new_mhi = (int)floor((spectral_stop+
                                     grid.meyer_half_bandwidth)/grid.frequency_pixel_df);
                if(new_mhi >= grid.frequency_layers)
                    new_mhi = grid.frequency_layers-1;
                for(m=old_mhi+1; m<=new_mhi; m++)
                {
                    for(ch=0; ch<3; ch++)
                    {
                        int lo = endpoint_lo, hi = endpoint_hi;
                        int width, middle;
                        if(nmid_plan[ch][m] >= 0 && nsize_plan[ch][m] > 0)
                        {
                            int old_lo = nmid_plan[ch][m]-nsize_plan[ch][m]/2;
                            int old_hi = old_lo+nsize_plan[ch][m];
                            if(old_lo < lo) lo = old_lo;
                            if(old_hi > hi) hi = old_hi;
                        }
                        width = thm_next_power_of_two_int(hi-lo);
                        if(width > grid.time_pixels) width = grid.time_pixels;
                        middle = (lo+hi)/2;
                        if(middle & 1) middle--;
                        if(middle-width/2 < 0) middle = width/2;
                        if(middle+width/2 > grid.time_pixels)
                            middle = grid.time_pixels-width/2;
                        nmid_plan[ch][m] = middle;
                        nsize_plan[ch][m] = width;
                    }
                }
                endpoint_frequency_stop = spectral_stop;
            }
        }
        endpoint.frequency_start = endpoint_frequency_start;
        endpoint.frequency_stop = endpoint_frequency_stop;
        endpoint.carrier_index = -1;
        endpoint.is_endpoint = 1;
        if(!thm_fast_partition_plan_reserve(&plan, plan.nblocks+1))
        {
            thm_complex_tdi_fft_block_free(&endpoint);
            status = 12;
            goto cleanup;
        }
        plan.blocks[plan.nblocks++] = endpoint;
        plan.total_fft_samples += nfft;
    }

    status = thm_complex_fft_blocks_to_thm_plan_sparse_wdm_context(
        context, plan.nblocks, plan.blocks, nmid_plan, nsize_plan,
        out_tracks);
    if(status != 0) status = 300+status;
    if(status == 0)
    {
        if(fft_samples_out != NULL) *fft_samples_out = plan.total_fft_samples;
        if(block_count_out != NULL) *block_count_out = plan.nblocks;
    }

cleanup:
    if(nsize_plan != NULL) free_int_matrix(nsize_plan, 3);
    if(nmid_plan != NULL) free_int_matrix(nmid_plan, 3);
    free(spec);
    free(fhigh);
    free(flow);
    thm_fast_partition_plan_free(&plan);
    return status;
}

/*
 * High-level in-memory THM TDI+WDM generator for Fisher/MCMC drivers.
 *
 * params_in uses the same convention as the executable:
 *   [0] m1 seconds, [1] m2 seconds, [2] chi1, [3] chi2,
 *   [4] phi0, [5] tc seconds, [6] ln(DL/Gpc),
 *   [7] ecliptic colatitude, [8] ecliptic longitude,
 *   [9] polarization, [10] cos(inclination).
 *
 * out_tracks must have been initialized with thm_sparse_wdm_triplet_init().
 * Its existing buffers are reused and overwritten, so a Fisher loop can keep
 * one workspace per finite-difference waveform without going through files.
 */
static int generate_thm_tdi_products(const THMObservationContext *context,
                                     THMWorkerWorkspace *workspace,
                                     const double params_in[11], const char *mode_spec,
                                     THMSparseWDMTriplet *out_tracks,
                                     THMFourierTriplet *out_fourier,
                                     double fourier_df,
                                     double fourier_fmax)
{
    double params[11];
    double *TS, *tspace;
    double **mode_amp, **mode_phase, **mode_freq;
    double *tc_tau;
    double Tobs, constellation_tmin, constellation_tmax;
    double max_mode_frequency_hz;
    int Ns, Nsmax, Ns_response, nmodes, requested_nmodes, ncarriers;
    int i, k, N;
    int status, old_diagnostics, old_summary_output;
    int direct_response_grid_intrinsic_used;
    int fixed_response_grid_used;
    clock_t partition_profile_start, partition_profile_ap_stop;
    clock_t partition_profile_transform_start, partition_profile_stop;
    IMRPhenomTHM model;
    IMRPhenomTHMMode modes[IMRPHENOMTHM_MAX_MODES];
    IMRPhenomTHMMode requested_modes[IMRPHENOMTHM_MAX_MODES];
    THMProjection projection[IMRPHENOMTHM_MAX_MODES];
    THMFoldedCarrier carrier[IMRPHENOMTHM_MAX_MODES];
    gsl_interp_accel **SLacc, **SPacc, **SVacc;
    gsl_spline **SLspline, **SPspline, **SVspline;
    gsl_interp_accel **Aacc, **Pacc;
    gsl_spline **Aspline, **Pspline;
    struct wdmshape *wdms;

    if(context == NULL || workspace == NULL || workspace->context != context ||
       params_in == NULL || (out_tracks == NULL && out_fourier == NULL)) return 1;
    /* Only the direct complex partition route uses the selected TDI stencil.
     * The older AP/SPA and Fourier fallbacks still construct TDI-1 explicitly. */
    if(context->tdi_generation == 2 &&
       (thm_tdi_generation != 2 || out_fourier != NULL || out_tracks == NULL ||
        !thm_wdm_fast_complex_partition_enabled ||
        !thm_wdm_split_early_fft_enabled ||
        thm_wdm_full_fft_reference_enabled ||
        thm_wdm_instrument_prewhiten_enabled))
        return 2;

    TS = NULL;
    tspace = NULL;
    tc_tau = NULL;
    mode_amp = NULL;
    mode_phase = NULL;
    mode_freq = NULL;
    SLacc = workspace->SLacc;
    SPacc = workspace->SPacc;
    SVacc = workspace->SVacc;
    SLspline = (gsl_spline **)context->SLspline;
    SPspline = (gsl_spline **)context->SPspline;
    SVspline = (gsl_spline **)context->SVspline;
    Aacc = NULL;
    Pacc = NULL;
    Aspline = NULL;
    Pspline = NULL;
    wdms = (struct wdmshape *)&context->wdms;
    status = 1;
    Ns = 0;
    nmodes = 0;
    requested_nmodes = 0;
    ncarriers = 0;
    direct_response_grid_intrinsic_used = 0;
    fixed_response_grid_used = 0;
    old_diagnostics = thm_diagnostics_enabled;
    old_summary_output = thm_summary_output_enabled;
    thm_diagnostics_enabled = 0;
    thm_summary_output_enabled = 0;
    memset(&model, 0, sizeof(model));
    partition_profile_start = clock();
    partition_profile_ap_stop = partition_profile_start;
    partition_profile_transform_start = partition_profile_start;
    partition_profile_stop = partition_profile_start;

    for(i=0; i<11; i++) params[i] = params_in[i];
    enforce_phenom_mass_hierarchy(&params[0], &params[1], &params[2], &params[3]);

    requested_nmodes = parse_thm_mode_selection(mode_spec, requested_modes,
                                                IMRPHENOMTHM_MAX_MODES);
    if(requested_nmodes < 1) goto cleanup;

    N = Nt*Nf;
    Tobs = wdms->Tobs;
    constellation_tmin = context->constellation_tmin;
    constellation_tmax = context->constellation_tmax;
    Nsmax = workspace->Nsmax;
    TS = workspace->TS;
    tspace = workspace->tspace;
    tc_tau = workspace->tc_tau;
    mode_amp = workspace->mode_amp;
    mode_phase = workspace->mode_phase;
    mode_freq = workspace->mode_freq;
    for(i=0; i<3; i++) gsl_interp_accel_reset(SLacc[i]);
    for(i=0; i<9; i++)
    {
        gsl_interp_accel_reset(SPacc[i]);
        gsl_interp_accel_reset(SVacc[i]);
    }

    if(thm_fourier_fixed_response_grid_enabled &&
       thm_fourier_fixed_response_grid_samples > 0)
    {
        Ns = thm_fourier_fixed_response_grid_samples;
        if(Ns > Nsmax) goto cleanup;
        for(i=0; i<Ns; i++)
        {
            tspace[i] = thm_fourier_fixed_response_grid[i];
        }
        Ns = PhenomTHM_AP_OnDetectorGrid(params, &model, modes, &nmodes,
                                         requested_modes, requested_nmodes,
                                         Ns, TS, tspace,
                                         mode_amp, mode_phase, mode_freq,
                                         SPacc, SPspline);
        if(Ns < 4 || nmodes < 1) goto cleanup;
        thm_last_intrinsic_tdi_model_samples = Ns;
        thm_last_intrinsic_tdi_interpolated_samples = 0;
        thm_last_intrinsic_tdi_exact_samples = Ns;
        thm_last_intrinsic_tdi_detector_planned_samples =
            thm_fourier_fixed_response_detector_planned_samples;
        direct_response_grid_intrinsic_used = 1;
        fixed_response_grid_used = 1;
    }
    else
    {
        Ns = PhenomTHM_AP_IntrinsicTDIAdaptive(params, &model, modes, &nmodes,
                                               requested_modes, requested_nmodes,
                                               Nsmax, TS, tspace,
                                               mode_amp, mode_phase, mode_freq,
                                               SPacc, SPspline,
                                               constellation_tmin,
                                               constellation_tmax,
                                               0.0, Tobs);
        if(Ns < 4 || nmodes < 1) goto cleanup;

        if(thm_fourier_fixed_response_grid_enabled)
        {
            if(Ns > THM_FOURIER_FIXED_RESPONSE_GRID_CAPACITY) goto cleanup;
            thm_fourier_fixed_response_grid_samples = Ns;
            thm_fourier_fixed_response_detector_planned_samples =
                thm_last_intrinsic_tdi_detector_planned_samples;
            for(i=0; i<Ns; i++)
            {
                thm_fourier_fixed_response_grid[i] = tspace[i];
            }
            fixed_response_grid_used = 1;
        }
    }

    if(thm_fourier_direct_response_grid_intrinsic_enabled ||
       (thm_fourier_fixed_response_grid_enabled &&
        !direct_response_grid_intrinsic_used))
    {
        int direct_ns;

        /*
         * Diagnostic separation of the intrinsic carrier from the detector
         * grid.  The adaptive planner above still supplies exactly the same
         * response times.  Rebuild the intrinsic phase integral and evaluate
         * every requested mode directly at the corresponding source times,
         * before fast_response_thm_piece() sees the AP splines.  Consequently
         * any change in the full-FFT likelihood comes only from bypassing the
         * sparse-intrinsic-to-response-grid interpolation.
         */
        IMRPhenomTHMDestroy(&model);
        direct_ns = PhenomTHM_AP_OnDetectorGrid(params, &model, modes, &nmodes,
                                                requested_modes,
                                                requested_nmodes,
                                                Ns, TS, tspace,
                                                mode_amp, mode_phase, mode_freq,
                                                SPacc, SPspline);
        if(direct_ns != Ns || nmodes < 1) goto cleanup;
        direct_response_grid_intrinsic_used = 1;
        if(fixed_response_grid_used)
        {
            thm_last_intrinsic_tdi_model_samples = Ns;
            thm_last_intrinsic_tdi_interpolated_samples = 0;
            thm_last_intrinsic_tdi_exact_samples = Ns;
        }
    }

    max_mode_frequency_hz = 0.0;
    for(i=0; i<Ns; i++)
    {
        tc_tau[i] = (TS[i]-params[5])/(params[0]+params[1]);
        for(k=0; k<nmodes; k++)
        {
            if(fabs(mode_freq[k][i]) > max_mode_frequency_hz)
            {
                max_mode_frequency_hz = fabs(mode_freq[k][i]);
            }
        }
    }
    partition_profile_ap_stop = clock();

    Aacc = malloc((size_t)nmodes*sizeof(gsl_interp_accel *));
    Pacc = malloc((size_t)nmodes*sizeof(gsl_interp_accel *));
    Aspline = malloc((size_t)nmodes*sizeof(gsl_spline *));
    Pspline = malloc((size_t)nmodes*sizeof(gsl_spline *));
    if(Aacc == NULL || Pacc == NULL || Aspline == NULL || Pspline == NULL) goto cleanup;
    for(k=0; k<nmodes; k++)
    {
        Aacc[k] = NULL;
        Pacc[k] = NULL;
        Aspline[k] = NULL;
        Pspline[k] = NULL;
    }
    for(k=0; k<nmodes; k++)
    {
        Aacc[k] = gsl_interp_accel_alloc();
        Pacc[k] = gsl_interp_accel_alloc();
        Aspline[k] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
        Pspline[k] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
        if(Aacc[k] == NULL || Pacc[k] == NULL ||
           Aspline[k] == NULL || Pspline[k] == NULL) goto cleanup;
        gsl_spline_init(Aspline[k], TS, mode_amp[k], Ns);
        gsl_spline_init(Pspline[k], TS, mode_phase[k], Ns);
        thm_projection_coefficients(modes[k].ell, modes[k].emm, params[10],
                                    0.5*M_PI, params[9], &projection[k]);
    }

    ncarriers = thm_build_folded_carriers(nmodes, modes, projection,
                                          carrier, IMRPHENOMTHM_MAX_MODES);
    if(ncarriers < 1) goto cleanup;

    if(fixed_response_grid_used &&
       thm_fourier_fixed_response_output_samples > 0)
    {
        Ns_response = thm_fourier_fixed_response_output_samples;
    }
    else
    {
        Ns_response = Ns;
        i = 0;
        do
        {
            i++;
        } while(i < Ns && tspace[Ns-i] > TS[Ns-1]);
        Ns_response = Ns-i;
        if(fixed_response_grid_used)
        {
            thm_fourier_fixed_response_output_samples = Ns_response;
        }
    }
    if(Ns_response < 4) goto cleanup;

    thm_response_spline_tmin = TS[0];
    thm_response_spline_tmax = TS[Ns-1];

    if(out_fourier != NULL &&
       thm_fourier_ap_grid_diagnostic_prefix[0] != '\0')
    {
        thm_write_fourier_ap_grid_diagnostic(
            thm_fourier_ap_grid_diagnostic_prefix,
            params, nmodes, modes, Ns, Ns_response, TS, tspace,
            mode_amp, mode_phase, mode_freq, SPacc, SPspline,
            Aacc, Aspline, Pacc, Pspline);
    }

    (void)max_mode_frequency_hz;
    partition_profile_transform_start = clock();

    /*
     * The production partitioned-FFT route works directly with the complex
     * intrinsic carriers.  Do this before the historical combined routine,
     * because that routine first evaluates every carrier on the adaptive TDI
     * grid, extracts signed amplitude and phase separately in X/Y/Z, and
     * resplines those quantities even though a complex block needs none of
     * that intermediate representation.
     */
    if(thm_wdm_fast_complex_partition_enabled &&
       thm_wdm_split_early_fft_enabled &&
       !thm_wdm_full_fft_reference_enabled && out_tracks != NULL &&
       out_fourier == NULL && !thm_diagnostics_enabled)
    {
        double Mtot = params[0]+params[1];
        double delay_margin = 500.0+
            (context->tdi_generation == 2 ? 8.0 : 4.0)*
            THM_TDI_LARM_NOMINAL_SECONDS;
        double endpoint_start = params[5]-delay_margin-10000.0*Mtot;
        double endpoint_rise = 5000.0*Mtot;
        double waveform_stop = params[5]+500.0+1000.0*Mtot;
        double split_bandwidth = thm_wdm_split_early_fft_bandwidth_hz;
        long long direct_fft_samples = 0;
        int direct_blocks = 0;
        int direct_status;

        if(thm_wdm_join_override_enabled)
        {
            endpoint_start = thm_wdm_join_override_time;
            if(thm_wdm_join_override_rise > 0.0)
                endpoint_rise = thm_wdm_join_override_rise;
        }
        if(thm_split_fft_plan_nband > 0 &&
           isfinite(thm_split_fft_plan_bandwidth[0]) &&
           thm_split_fft_plan_bandwidth[0] > 0.0)
            split_bandwidth = thm_split_fft_plan_bandwidth[0];
        if(waveform_stop > Tobs) waveform_stop = Tobs;
        if(endpoint_start <= 0.0)
            endpoint_start = fmin(0.25*waveform_stop,
                                  fmax(wdms->DT, waveform_stop-endpoint_rise));
        if(endpoint_start+endpoint_rise > waveform_stop)
            endpoint_rise = waveform_stop-endpoint_start;

        /* Detector-guided AP placement need not begin at SSB output t=0.
         * Start the first partition where the response spline is defined. */
        direct_status = thm_direct_partitioned_wdm_context(
            context, workspace, Ns_response, TS, mode_freq, params,
            ncarriers, carrier, Aacc, Aspline, Pacc, Pspline,
            fmax(0.0, TS[0]), waveform_stop, endpoint_start, endpoint_rise,
            split_bandwidth, SPLIT_FFT_ROLL_SECONDS, out_tracks,
            &direct_fft_samples, &direct_blocks);
        if(direct_status == 0)
        {
            partition_profile_stop = clock();
            if(getenv("THM_PARTITION_DEBUG") != NULL)
            {
                fprintf(stderr,
                        "THM_PARTITION_DEBUG direct_top_level blocks %d fft_samples %lld intrinsic_ap %.6f spline_setup %.6f response_and_wdm %.6f total %.6f\n",
                        direct_blocks, direct_fft_samples,
                        (double)(partition_profile_ap_stop-
                                 partition_profile_start)/
                            (double)CLOCKS_PER_SEC,
                        (double)(partition_profile_transform_start-
                                 partition_profile_ap_stop)/
                            (double)CLOCKS_PER_SEC,
                        (double)(partition_profile_stop-
                                 partition_profile_transform_start)/
                            (double)CLOCKS_PER_SEC,
                        (double)(partition_profile_stop-
                                 partition_profile_start)/
                            (double)CLOCKS_PER_SEC);
            }
            status = 0;
            goto cleanup;
        }
        for(i=0; i<3; i++) out_tracks->channel[i].npixels = 0;
        fprintf(stderr,
                "Warning: direct complex THM partition failed (%d); using the post-TDI AP path.\n",
                direct_status);
        if(context->tdi_generation == 2)
        {
            status = 100+direct_status;
            goto cleanup;
        }
    }

    generate_thm_all_carrier_wdm_combined_fft(Ns_response,
                                              TS, tspace,
                                              mode_freq,
                                              params, &model,
                                              ncarriers, carrier,
                                              SLacc, SLspline, SPacc, SPspline,
                                              SVacc, SVspline,
                                              Aacc, Aspline, Pacc, Pspline,
                                              wdms,
                                              thm_wdm_join_override_enabled,
                                              thm_wdm_join_override_time,
                                              thm_wdm_join_override_rise,
                                              0, thm_wdm_partition_endpoint_enabled, 0,
                                              thm_wdm_split_early_fft_enabled &&
                                              !thm_wdm_full_fft_reference_enabled,
                                              0,
                                              out_tracks,
                                              out_fourier,
                                              fourier_df,
                                              fourier_fmax,
                                              0, 0,
                                              &workspace->wdm_work,
                                              context, workspace);
    partition_profile_stop = clock();
    if(getenv("THM_PARTITION_DEBUG") != NULL)
    {
        fprintf(stderr,
                "THM_PARTITION_DEBUG top_level intrinsic_ap %.6f spline_setup %.6f response_and_wdm %.6f total %.6f\n",
                (double)(partition_profile_ap_stop-partition_profile_start)/
                    (double)CLOCKS_PER_SEC,
                (double)(partition_profile_transform_start-
                         partition_profile_ap_stop)/(double)CLOCKS_PER_SEC,
                (double)(partition_profile_stop-
                         partition_profile_transform_start)/
                    (double)CLOCKS_PER_SEC,
                (double)(partition_profile_stop-partition_profile_start)/
                    (double)CLOCKS_PER_SEC);
    }
    if(out_fourier != NULL)
    {
        out_fourier->ap_response_samples = Ns_response;
        out_fourier->ap_model_samples = thm_last_intrinsic_tdi_model_samples;
        out_fourier->ap_interpolated_samples =
            thm_last_intrinsic_tdi_interpolated_samples;
        out_fourier->ap_exact_samples = thm_last_intrinsic_tdi_exact_samples;
        out_fourier->ap_detector_planned_samples =
            thm_last_intrinsic_tdi_detector_planned_samples;
        out_fourier->direct_response_grid_intrinsic_used =
            direct_response_grid_intrinsic_used;
        out_fourier->fixed_response_grid_used = fixed_response_grid_used;
    }
    status = 0;

cleanup:
    if(Aacc != NULL)
    {
        for(k=0; k<nmodes; k++)
        {
            if(Aspline != NULL && Aspline[k] != NULL) gsl_spline_free(Aspline[k]);
            if(Pspline != NULL && Pspline[k] != NULL) gsl_spline_free(Pspline[k]);
            if(Aacc[k] != NULL) gsl_interp_accel_free(Aacc[k]);
            if(Pacc != NULL && Pacc[k] != NULL) gsl_interp_accel_free(Pacc[k]);
        }
    }
    free(Aacc);
    free(Pacc);
    free(Aspline);
    free(Pspline);

    IMRPhenomTHMDestroy(&model);
    thm_diagnostics_enabled = old_diagnostics;
    thm_summary_output_enabled = old_summary_output;

    (void)N;
    return status;
}

int generate_thm_tdi_wdm(const double params_in[11], const char *mode_spec,
                         THMSparseWDMTriplet *out_tracks)
{
    THMObservationContext *context;
    THMWorkerWorkspace *workspace;
    int status;

    context = thm_observation_context_create(thm_orbit_phase_offset);
    if(context == NULL) return 1;
    workspace = thm_worker_workspace_create(context);
    if(workspace == NULL)
    {
        thm_observation_context_destroy(context);
        return 1;
    }
    status = generate_thm_tdi_wdm_context(context, workspace, params_in,
                                          mode_spec, out_tracks);
    thm_worker_workspace_destroy(workspace);
    thm_observation_context_destroy(context);
    return status;
}

int generate_thm_tdi_wdm_context(const THMObservationContext *context,
                                 THMWorkerWorkspace *workspace,
                                 const double params_in[11],
                                 const char *mode_spec,
                                 THMSparseWDMTriplet *out_tracks)
{
    return generate_thm_tdi_products(context, workspace, params_in, mode_spec,
                                     out_tracks, NULL, 0.0, 0.0);
}

int generate_thm_tdi_fourier(const double params_in[11], const char *mode_spec,
                             double df_hz, double fmax_hz,
                             THMFourierTriplet *out_spectrum)
{
    THMObservationContext *context;
    THMWorkerWorkspace *workspace;
    int status;

    if(out_spectrum == NULL || !isfinite(df_hz) || df_hz <= 0.0) return 1;
    context = thm_observation_context_create(thm_orbit_phase_offset);
    if(context == NULL) return 1;
    workspace = thm_worker_workspace_create(context);
    if(workspace == NULL)
    {
        thm_observation_context_destroy(context);
        return 1;
    }
    status = generate_thm_tdi_fourier_context(context, workspace, params_in,
                                              mode_spec, df_hz, fmax_hz,
                                              out_spectrum);
    thm_worker_workspace_destroy(workspace);
    thm_observation_context_destroy(context);
    return status;
}

int generate_thm_tdi_fourier_context(const THMObservationContext *context,
                                     THMWorkerWorkspace *workspace,
                                     const double params_in[11],
                                     const char *mode_spec,
                                     double df_hz,
                                     double fmax_hz,
                                     THMFourierTriplet *out_spectrum)
{
    if(out_spectrum == NULL || !isfinite(df_hz) || df_hz <= 0.0) return 1;
    return generate_thm_tdi_products(context, workspace, params_in, mode_spec,
                                     NULL, out_spectrum, df_hz, fmax_hz);
}

int generate_thm_tdi_wdm_ap_derivative(const double params0_in[11], const double params_plus_in[11], const double params_minus_in[11], const char *mode_spec, double coord_step, THMSparseWDMTriplet *out_tracks)
{
    const int Nvar = 3;
    const int Nchan = 3;
    THMAPContext ctx[3];
    IMRPhenomTHMMode requested_modes[IMRPHENOMTHM_MAX_MODES];
    THMFoldedCarrier ap_carrier;
    const IMRPhenomTHMModeState *mode_state;
    double **Larray, *tarray;
    double ***Parray, ***Varray;
    gsl_interp_accel **SLacc, **SPacc, **SVacc;
    gsl_spline **SLspline, **SPspline, **SVspline;
    struct wdmshape *wdms;
    double ****Apost, ****Ppost;
    double ***freq_track, ***setup_carrier;
    double *Xmode, *Ymode, *Zmode, *Xfmode, *Yfmode, *Zfmode;
    double *phi_ref, *phi_res, *omega_plan;
    double *freq0, *phase0, *Amp0;
    double *freqp, *phasep, *Ampp;
    double *freqm, *phasem, *Ampm;
    double *freqd, *phased, *Ampd, *dAmpd, *dPhased;
    double *short_tmp;
    double *wdmwave, *short_hsum;
    int *nmid, *nsize, *nmid_endpoint, *nsize_endpoint;
    int *listn, *listm;
    double **wdm;
    int Nsmax, Ns_response, requested_nmodes, ncarriers;
    int status, old_diagnostics, old_summary_output;
    int i, j, k, ch, v, idx, N, Nc, m, p, Np, final_pixels;
    int Ntsmax, N0, NpF, NmF, Nd;
    int driver_index, driver_channel;
    int endpoint_mlo, endpoint_mmax, endpoint_direct_layers, endpoint_direct_pixels;
    int min_block, center;
    double Tobs, dtx, dtc, constellation_tmin, constellation_tmax;
    double Mtot, Mc, tc, t_transition, fring, fdamp, fjoin;
    double score, driver_score, setup_driver[7], setup_spa[7], f_replace_start[3], f_replace_pair_start[3], f_replace_fallback[3];
    double t, A0v, P0v, Apv, Ppv, Amv, Pmv, dAv, dPv, hval;
    double f_endpoint_start, f_endpoint_max, tail_amp, tail_time;
    double blend_half_width;
    double rawp, rawm, diffp, diffm, prevp, prevm;
    gsl_interp_accel **TDAacc, **TDPacc;
    gsl_spline **TDAspline, **TDPspline;
    gsl_interp_accel *AFacc, *PFacc, *DAFacc, *DPFacc;
    gsl_spline *AFspline, *PFspline, *DAFspline, *DPFspline;
    gsl_interp_accel *PAAacc, *PPPacc, *MAAacc, *MPPacc;
    gsl_spline *PAAspline, *PPPspline, *MAAspline, *MPPspline;

    if(params0_in == NULL || params_plus_in == NULL || params_minus_in == NULL ||
       out_tracks == NULL || coord_step == 0.0)
    {
        return 1;
    }

    for(v=0; v<Nvar; v++) thm_ap_context_init(&ctx[v]);
    Larray = NULL;
    tarray = NULL;
    Parray = NULL;
    Varray = NULL;
    SLacc = SPacc = SVacc = NULL;
    SLspline = SPspline = SVspline = NULL;
    wdms = NULL;
    Apost = Ppost = NULL;
    freq_track = setup_carrier = NULL;
    Xmode = Ymode = Zmode = Xfmode = Yfmode = Zfmode = NULL;
    phi_ref = phi_res = omega_plan = NULL;
    freq0 = phase0 = Amp0 = NULL;
    freqp = phasep = Ampp = NULL;
    freqm = phasem = Ampm = NULL;
    freqd = phased = Ampd = dAmpd = dPhased = NULL;
    short_tmp = NULL;
    wdmwave = short_hsum = NULL;
    nmid = nsize = nmid_endpoint = nsize_endpoint = NULL;
    listn = listm = NULL;
    wdm = NULL;
    TDAacc = TDPacc = NULL;
    TDAspline = TDPspline = NULL;
    AFacc = PFacc = DAFacc = DPFacc = NULL;
    AFspline = PFspline = DAFspline = DPFspline = NULL;
    PAAacc = PPPacc = MAAacc = MPPacc = NULL;
    PAAspline = PPPspline = MAAspline = MPPspline = NULL;

    status = 1;
    old_diagnostics = thm_diagnostics_enabled;
    old_summary_output = thm_summary_output_enabled;
    thm_diagnostics_enabled = 0;
    thm_summary_output_enabled = 0;

    for(i=0; i<11; i++)
    {
        ctx[0].params[i] = params0_in[i];
        ctx[1].params[i] = params_plus_in[i];
        ctx[2].params[i] = params_minus_in[i];
    }
    for(v=0; v<Nvar; v++)
    {
        enforce_phenom_mass_hierarchy(&ctx[v].params[0], &ctx[v].params[1],
                                      &ctx[v].params[2], &ctx[v].params[3]);
    }

    requested_nmodes = parse_thm_mode_selection(mode_spec, requested_modes,
                                                IMRPHENOMTHM_MAX_MODES);
    if(requested_nmodes < 1) goto cleanup;

    wdms = malloc(sizeof(struct wdmshape));
    if(wdms == NULL) goto cleanup;
    wdmvalues(wdms);
    N = Nt*Nf;
    blend_half_width = thm_wdm_blend_half_width_layers*
                       (1.0/(2.0*(double)Nf*dt));
    if(!isfinite(blend_half_width) || blend_half_width <= 0.0)
    {
        blend_half_width = 1.0/(2.0*(double)Nf*dt);
    }
    Tobs = wdms->Tobs;

    Nc = (int)(200.0*Tobs/SECSYR);
    if(Nc < 20) Nc = 20;
    dtx = Tobs/(double)(Nc-1);
    if(dtx < CONSTELLATION_MIN_PADDING_SECONDS) dtx = CONSTELLATION_MIN_PADDING_SECONDS;
    dtc = (Tobs+2.0*dtx)/(double)(Nc-1);

    Larray = double_matrix(3, Nc);
    Parray = double_tensor(3, 3, Nc);
    Varray = double_tensor(3, 3, Nc);
    tarray = double_vector(Nc);
    if(Larray == NULL || Parray == NULL || Varray == NULL || tarray == NULL) goto cleanup;

    for(i=0; i<Nc; i++) tarray[i] = -dtx+dtc*(double)i;
    constellation_tmin = tarray[0];
    constellation_tmax = tarray[Nc-1];
    constellation(Nc, tarray, Larray, Parray, Varray);

    SLacc = calloc(3, sizeof(gsl_interp_accel *));
    SLspline = calloc(3, sizeof(gsl_spline *));
    SPacc = calloc(9, sizeof(gsl_interp_accel *));
    SPspline = calloc(9, sizeof(gsl_spline *));
    SVacc = calloc(9, sizeof(gsl_interp_accel *));
    SVspline = calloc(9, sizeof(gsl_spline *));
    if(SLacc == NULL || SLspline == NULL || SPacc == NULL || SPspline == NULL ||
       SVacc == NULL || SVspline == NULL) goto cleanup;

    for(i=0; i<3; i++)
    {
        SLacc[i] = gsl_interp_accel_alloc();
        SLspline[i] = gsl_spline_alloc(gsl_interp_cspline, Nc);
        if(SLacc[i] == NULL || SLspline[i] == NULL) goto cleanup;
        gsl_spline_init(SLspline[i], tarray, Larray[i], Nc);
    }
    for(i=0; i<9; i++)
    {
        SPacc[i] = gsl_interp_accel_alloc();
        SPspline[i] = gsl_spline_alloc(gsl_interp_cspline, Nc);
        SVacc[i] = gsl_interp_accel_alloc();
        SVspline[i] = gsl_spline_alloc(gsl_interp_cspline, Nc);
        if(SPacc[i] == NULL || SPspline[i] == NULL ||
           SVacc[i] == NULL || SVspline[i] == NULL) goto cleanup;
    }
    for(i=0; i<3; i++)
    {
        for(j=0; j<3; j++)
        {
            k = j+i*3;
            gsl_spline_init(SPspline[k], tarray, Parray[i][j], Nc);
            gsl_spline_init(SVspline[k], tarray, Varray[i][j], Nc);
        }
    }

    Nsmax = 10000;
    for(v=0; v<Nvar; v++)
    {
        if(!thm_ap_context_alloc(&ctx[v], Nsmax)) goto cleanup;
    }

    ctx[0].Ns = PhenomTHM_AP_DetectorAdaptive(ctx[0].params, &ctx[0].model,
                                              ctx[0].modes, &ctx[0].nmodes,
                                              requested_modes, requested_nmodes,
                                              Nsmax, ctx[0].TS, ctx[0].tspace,
                                              ctx[0].mode_amp, ctx[0].mode_phase,
                                              ctx[0].mode_freq,
                                              SPacc, SPspline,
                                              constellation_tmin, constellation_tmax,
                                              0.0, Tobs);
    if(ctx[0].Ns < 4 || ctx[0].nmodes < 1) goto cleanup;

    for(v=1; v<Nvar; v++)
    {
        for(i=0; i<ctx[0].Ns; i++) ctx[v].tspace[i] = ctx[0].tspace[i];
        ctx[v].Ns = PhenomTHM_AP_OnDetectorGrid(ctx[v].params, &ctx[v].model,
                                                ctx[v].modes, &ctx[v].nmodes,
                                                requested_modes, requested_nmodes,
                                                ctx[0].Ns, ctx[v].TS, ctx[v].tspace,
                                                ctx[v].mode_amp, ctx[v].mode_phase,
                                                ctx[v].mode_freq,
                                                SPacc, SPspline);
        if(ctx[v].Ns != ctx[0].Ns || ctx[v].nmodes != ctx[0].nmodes) goto cleanup;
    }

    for(v=0; v<Nvar; v++)
    {
        if(!thm_ap_context_build_splines_and_carriers(&ctx[v])) goto cleanup;
        if(v > 0 && ctx[v].ncarriers != ctx[0].ncarriers) goto cleanup;
    }
    ncarriers = ctx[0].ncarriers;
    if(ncarriers < 1) goto cleanup;

    Ns_response = ctx[0].Ns;
    i = 0;
    do
    {
        i++;
    } while(i < ctx[0].Ns && ctx[0].tspace[ctx[0].Ns-i] > ctx[0].TS[ctx[0].Ns-1]);
    Ns_response = ctx[0].Ns-i;
    if(Ns_response < 4) goto cleanup;

    Apost = double_quad(Nvar, Nchan, ncarriers, Ns_response);
    Ppost = double_quad(Nvar, Nchan, ncarriers, Ns_response);
    freq_track = double_tensor(Nchan, ncarriers, Ns_response);
    setup_carrier = double_tensor(Nchan, ncarriers, 7);
    Xmode = double_vector(Ns_response);
    Ymode = double_vector(Ns_response);
    Zmode = double_vector(Ns_response);
    Xfmode = double_vector(Ns_response);
    Yfmode = double_vector(Ns_response);
    Zfmode = double_vector(Ns_response);
    phi_ref = double_vector(Ns_response);
    phi_res = double_vector(Ns_response);
    omega_plan = double_vector(Ns_response);
    if(Apost == NULL || Ppost == NULL || freq_track == NULL ||
       setup_carrier == NULL || Xmode == NULL ||
       Ymode == NULL || Zmode == NULL || Xfmode == NULL || Yfmode == NULL ||
       Zfmode == NULL || phi_ref == NULL || phi_res == NULL ||
       omega_plan == NULL) goto cleanup;

    Mtot = ctx[0].params[0]+ctx[0].params[1];
    Mc = pow(ctx[0].params[0]*ctx[0].params[1], 3.0/5.0)/pow(Mtot, 1.0/5.0);
    tc = ctx[0].params[5];
    t_transition = tc+Mtot*tCUT_Freq;
    driver_index = -1;
    driver_channel = -1;
    driver_score = -1.0;
    for(ch=0; ch<Nchan; ch++)
    {
        f_replace_start[ch] = HUGE_VAL;
        f_replace_pair_start[ch] = HUGE_VAL;
        f_replace_fallback[ch] = 0.0;
    }

    for(v=0; v<Nvar; v++)
    {
        for(k=0; k<ncarriers; k++)
        {
            int mode_index = ctx[v].carrier[k].mode_index;
            double *Mchan[3], *Mfchan[3];

            if(mode_index < 0 || mode_index >= ctx[v].model.nmodes) goto cleanup;

            ap_carrier = ctx[v].carrier[k];
            ap_carrier.hpf_cos = ap_carrier.hp_sin;
            ap_carrier.hpf_sin = -ap_carrier.hp_cos;
            ap_carrier.hcf_cos = ap_carrier.hc_sin;
            ap_carrier.hcf_sin = -ap_carrier.hc_cos;

            thm_response_spline_tmin = ctx[v].TS[0];
            thm_response_spline_tmax = ctx[v].TS[ctx[v].Ns-1];
            fast_response_thm(ctx[0].TS, Ns_response, ctx[v].params, 1, &ap_carrier,
                              SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                              ctx[v].Aacc, ctx[v].Aspline, ctx[v].Pacc, ctx[v].Pspline,
                              Xmode, Ymode, Zmode, Xfmode, Yfmode, Zfmode);

            Mchan[0] = Xmode;
            Mchan[1] = Ymode;
            Mchan[2] = Zmode;
            Mfchan[0] = Xfmode;
            Mfchan[1] = Yfmode;
            Mfchan[2] = Zfmode;

            for(i=0; i<Ns_response; i++)
            {
                phi_ref[i] = thm_reference_phase_eval(
                    ctx[v].Pspline[mode_index], ctx[0].tspace[i],
                    ctx[v].Pacc[mode_index]);
            }

            for(ch=0; ch<Nchan; ch++)
            {
                extractAP(Ns_response, Apost[v][ch][k], phi_res,
                          Mchan[ch], Mfchan[ch], phi_ref);
                unwrap(Ns_response, phi_res);
                for(i=0; i<Ns_response; i++)
                {
                    Ppost[v][ch][k][i] = phi_ref[i]+phi_res[i];
                }

                if(v == 0)
                {
                    for(i=0; i<Ns_response; i++)
                    {
                        double dphidt = nonuniform_phase_derivative(i, Ns_response,
                                                                    ctx[0].TS,
                                                                    Ppost[0][ch][k]);
                        freq_track[ch][k][i] = fabs(dphidt)/(2.0*M_PI);
                        if(!isfinite(freq_track[ch][k][i]) ||
                           freq_track[ch][k][i] <= 0.0)
                        {
                            freq_track[ch][k][i] = fabs(ctx[0].mode_freq[mode_index][i]);
                        }
                        omega_plan[i] = 2.0*M_PI*freq_track[ch][k][i];
                    }
                    {
                        mode_state = &ctx[0].model.modes[mode_index];
                        if(ctx[0].carrier[k].ell == 2 && abs(ctx[0].carrier[k].emm) == 2)
                        {
                            fring = ctx[0].model.mode22.omegaRING/(2.0*M_PI*Mtot);
                            fdamp = ctx[0].model.mode22.alpha1RD/Mtot;
                        }
                        else
                        {
                            fring = mode_state->phase.omegaRING/(2.0*M_PI*Mtot);
                            fdamp = mode_state->phase.alpha1RD/Mtot;
                        }
                        /*
                         * Do not force an artificial monotone planning ramp
                         * here: endpoint and support decisions need the
                         * physical post-TDI carrier frequency.
                         */
                        transformplan_custom(Mc, Mtot,
                                             linear_interp_clamped(Ns_response,
                                                                   ctx[0].tspace,
                                                                   ctx[0].TS, tc),
                                             ctx[0].TS, omega_plan,
                                             linear_interp_clamped(Ns_response,
                                                                   ctx[0].tspace,
                                                                   ctx[0].TS,
                                                                   t_transition),
                                             fring, fdamp,
                                             ctx[0].TS[Ns_response-1],
                                             setup_carrier[ch][k],
                                             thm_wdm_join_override_enabled,
                                             thm_wdm_join_override_time,
                                             thm_wdm_join_override_rise);
                        {
                            double join_time_cont = setup_carrier[ch][k][5];
                            if(isfinite(join_time_cont))
                            {
                                fjoin = linear_interp_clamped(Ns_response,
                                                              ctx[0].TS,
                                                              freq_track[ch][k],
                                                              join_time_cont);
                            }
                            else
                            {
                                fjoin = freq_track[ch][k][(int)setup_carrier[ch][k][4]];
                            }
                        }
                        if(isfinite(fjoin) && fjoin > 0.0)
                        {
                            if(fjoin > f_replace_fallback[ch])
                            {
                                f_replace_fallback[ch] = fjoin;
                            }
                            /*
                             * Shared endpoint rule used by the in-memory Fisher
                             * path.  The historical all-mode choice was the
                             * first |m|>=3 handoff; diagnostics can lower this
                             * to |m|>=2 to test whether the last 22-owned SPA
                             * layers are contaminating likelihood smoothness.
                             * A run with no selected carrier above the threshold
                             * falls back to the first |m|>=2 handoff.
                             */
                            if(abs(ctx[0].carrier[k].emm) >= 2 && fjoin < f_replace_pair_start[ch])
                            {
                                f_replace_pair_start[ch] = fjoin;
                            }
                            if(abs(ctx[0].carrier[k].emm) >= thm_endpoint_min_abs_m &&
                               fjoin < f_replace_start[ch])
                            {
                                f_replace_start[ch] = fjoin;
                            }
                        }
                        score = setup_carrier[ch][k][6]/
                                (setup_carrier[ch][k][0]*setup_carrier[ch][k][1]);
                        if(driver_index < 0 || score > driver_score)
                        {
                            driver_index = k;
                            driver_channel = ch;
                            driver_score = score;
                            for(i=0; i<7; i++) setup_driver[i] = setup_carrier[ch][k][i];
                        }
                    }
                }
            }
        }
    }

    if(driver_index < 0 || driver_channel < 0 || driver_score <= 0.0) goto cleanup;
    for(ch=0; ch<Nchan; ch++)
    {
        if(!isfinite(f_replace_start[ch]) || f_replace_start[ch] <= 0.0)
        {
            f_replace_start[ch] = f_replace_pair_start[ch];
        }
        if(!isfinite(f_replace_start[ch]) || f_replace_start[ch] <= 0.0)
        {
            f_replace_start[ch] = f_replace_fallback[ch];
        }
    }

    nmid = int_vector(Nf);
    nsize = int_vector(Nf);
    nmid_endpoint = int_vector(Nf);
    nsize_endpoint = int_vector(Nf);
    listn = int_vector(N);
    listm = int_vector(N);
    wdmwave = double_vector(N);
    wdm = double_matrix(Nt, Nf+1);
    if(nmid == NULL || nsize == NULL || nmid_endpoint == NULL ||
       nsize_endpoint == NULL || listn == NULL || listm == NULL ||
       wdmwave == NULL || wdm == NULL) goto cleanup;

    Ntsmax = Ns_response+32;
    for(ch=0; ch<Nchan; ch++)
    {
        for(k=0; k<ncarriers; k++)
        {
            int needed = (int)(setup_carrier[ch][k][4]+setup_carrier[ch][k][6])+32;
            if(needed > Ntsmax) Ntsmax = needed;
        }
    }

    freq0 = double_vector(Ntsmax);
    phase0 = double_vector(Ntsmax);
    Amp0 = double_vector(Ntsmax);
    freqp = double_vector(Ntsmax);
    phasep = double_vector(Ntsmax);
    Ampp = double_vector(Ntsmax);
    freqm = double_vector(Ntsmax);
    phasem = double_vector(Ntsmax);
    Ampm = double_vector(Ntsmax);
    freqd = double_vector(Ntsmax);
    phased = double_vector(Ntsmax);
    Ampd = double_vector(Ntsmax);
    dAmpd = double_vector(Ntsmax);
    dPhased = double_vector(Ntsmax);
    AFacc = gsl_interp_accel_alloc();
    PFacc = gsl_interp_accel_alloc();
    DAFacc = gsl_interp_accel_alloc();
    DPFacc = gsl_interp_accel_alloc();
    PAAacc = gsl_interp_accel_alloc();
    PPPacc = gsl_interp_accel_alloc();
    MAAacc = gsl_interp_accel_alloc();
    MPPacc = gsl_interp_accel_alloc();
    if(freq0 == NULL || phase0 == NULL || Amp0 == NULL ||
       freqp == NULL || phasep == NULL || Ampp == NULL ||
       freqm == NULL || phasem == NULL || Ampm == NULL ||
       freqd == NULL || phased == NULL || Ampd == NULL ||
       dAmpd == NULL || dPhased == NULL || AFacc == NULL ||
       PFacc == NULL || DAFacc == NULL || DPFacc == NULL ||
       PAAacc == NULL || PPPacc == NULL || MAAacc == NULL || MPPacc == NULL)
    {
        goto cleanup;
    }

    for(ch=0; ch<Nchan; ch++)
    {
        int ntimespl = Nvar*ncarriers;

        for(i=0; i<Nt; i++)
        {
            for(m=0; m<=Nf; m++) wdm[i][m] = 0.0;
        }
        for(m=0; m<Nf; m++)
        {
            nmid_endpoint[m] = -1;
            nsize_endpoint[m] = 0;
        }

        TDAacc = calloc((size_t)ntimespl, sizeof(gsl_interp_accel *));
        TDPacc = calloc((size_t)ntimespl, sizeof(gsl_interp_accel *));
        TDAspline = calloc((size_t)ntimespl, sizeof(gsl_spline *));
        TDPspline = calloc((size_t)ntimespl, sizeof(gsl_spline *));
        if(TDAacc == NULL || TDPacc == NULL || TDAspline == NULL ||
           TDPspline == NULL) goto cleanup;

        for(v=0; v<Nvar; v++)
        {
            for(k=0; k<ncarriers; k++)
            {
                idx = v*ncarriers+k;
                TDAacc[idx] = gsl_interp_accel_alloc();
                TDPacc[idx] = gsl_interp_accel_alloc();
                TDAspline[idx] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns_response);
                TDPspline[idx] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns_response);
                if(TDAacc[idx] == NULL || TDPacc[idx] == NULL ||
                   TDAspline[idx] == NULL || TDPspline[idx] == NULL) goto cleanup;
                gsl_spline_init(TDAspline[idx], ctx[0].TS, Apost[v][ch][k], Ns_response);
                gsl_spline_init(TDPspline[idx], ctx[0].TS, Ppost[v][ch][k], Ns_response);
            }
        }

        for(k=0; k<ncarriers; k++)
        {
            int q;
            double f_spa_stop, spa_support_tmax;

            idx = k;

            for(q=0; q<7; q++) setup_spa[q] = setup_carrier[ch][k][q];
            /*
             * Match the production SPA+endpoint-FFT engine used by
             * generate_thm_tdi_wdm(): the AP-derivative spectrum is pure SPA
             * on the low-frequency side.  A hard replacement needs only one
             * Meyer half-band of SPA overlap below the endpoint boundary.  A
             * coefficient-space blend needs more: both the SPA and the summed
             * endpoint FFT must exist across the whole blend interval, or the
             * nominally smooth handoff still has a hidden layer/source
             * discontinuity.  The old ftran() carrier-local FFT bridge is
             * avoided here for the same reason it is avoided in the waveform
             * path: it can imprint small spline-bridge scars right where Fisher
             * derivatives are most sensitive.
             */
            f_spa_stop = f_replace_start[ch]+wdms->FB;
            if(thm_wdm_blend_endpoint_enabled)
            {
                f_spa_stop += blend_half_width;
            }
            if(isfinite(f_spa_stop) && f_spa_stop > 0.0)
            {
                int jstop = (int)setup_spa[4];
                if(jstop < 1) jstop = 1;
                if(jstop > Ns_response-1) jstop = Ns_response-1;
                /*
                 * Extend the SPA samples using the actual post-TDI carrier
                 * frequency.  Low-|m| carriers can still be at a much lower
                 * physical frequency at the shared endpoint time; using an
                 * artificial monotone planning ramp here creates an upper
                 * frequency edge that can move across a WDM layer under small
                 * parameter changes.
                 */
                while(jstop+1 < Ns_response &&
                      freq_track[ch][k][jstop] < f_spa_stop)
                {
                    jstop++;
                }
                if((double)jstop > setup_spa[4]) setup_spa[4] = (double)jstop;
            }
            {
                int jstop = (int)setup_spa[4];
                if(jstop < 1) jstop = 1;
                if(jstop > Ns_response-1) jstop = Ns_response-1;
                spa_support_tmax = ctx[0].TS[jstop];
            }

            N0 = ftran_spa_only(setup_spa, ctx[0].TS, Ns_response,
                                Tobs, TDAacc[idx], TDAspline[idx],
                                TDPacc[idx], TDPspline[idx],
                                freq0, phase0, Amp0);

            NpF = ftran_spa_only(setup_spa, ctx[0].TS, Ns_response,
                                 Tobs, TDAacc[ncarriers+k],
                                 TDAspline[ncarriers+k],
                                 TDPacc[ncarriers+k],
                                 TDPspline[ncarriers+k],
                                 freqp, phasep, Ampp);

            NmF = ftran_spa_only(setup_spa, ctx[0].TS, Ns_response,
                                 Tobs, TDAacc[2*ncarriers+k],
                                 TDAspline[2*ncarriers+k],
                                 TDPacc[2*ncarriers+k],
                                 TDPspline[2*ncarriers+k],
                                 freqm, phasem, Ampm);
            N0 = thm_keep_increasing_frequency_samples(N0, freq0, phase0, Amp0);
            NpF = thm_keep_increasing_frequency_samples(NpF, freqp, phasep, Ampp);
            NmF = thm_keep_increasing_frequency_samples(NmF, freqm, phasem, Ampm);
            if(N0 < 4 || NpF < 4 || NmF < 4) continue;

            PAAspline = gsl_spline_alloc(gsl_interp_akima, NpF);
            PPPspline = gsl_spline_alloc(gsl_interp_cspline, NpF);
            MAAspline = gsl_spline_alloc(gsl_interp_akima, NmF);
            MPPspline = gsl_spline_alloc(gsl_interp_cspline, NmF);
            if(PAAspline == NULL || PPPspline == NULL ||
               MAAspline == NULL || MPPspline == NULL) goto cleanup;
            gsl_spline_init(PAAspline, freqp, Ampp, NpF);
            gsl_spline_init(PPPspline, freqp, phasep, NpF);
            gsl_spline_init(MAAspline, freqm, Ampm, NmF);
            gsl_spline_init(MPPspline, freqm, phasem, NmF);
            gsl_interp_accel_reset(PAAacc);
            gsl_interp_accel_reset(PPPacc);
            gsl_interp_accel_reset(MAAacc);
            gsl_interp_accel_reset(MPPacc);

            Nd = 0;
            prevp = 0.0;
            prevm = 0.0;
            for(i=0; i<N0; i++)
            {
                double f = freq0[i];

                if(f < freqp[0] || f > freqp[NpF-1] ||
                   f < freqm[0] || f > freqm[NmF-1])
                {
                    continue;
                }

                Apv = gsl_spline_eval(PAAspline, f, PAAacc);
                Ppv = gsl_spline_eval(PPPspline, f, PPPacc);
                Amv = gsl_spline_eval(MAAspline, f, MAAacc);
                Pmv = gsl_spline_eval(MPPspline, f, MPPacc);
                freqd[Nd] = f;
                phased[Nd] = phase0[i];
                Ampd[Nd] = Amp0[i];
                {
                    double dzre, dzim, c0, s0;

                    /*
                     * Work in the base carrier frame:
                     *   dz e^{-i Phi0} = dA + i A dPhi.
                     * The two stored derivative functions are therefore slow
                     * even though the underlying complex Fourier sample is
                     * rapidly rotating.
                     */
                    dzre = (Apv*cos(Ppv)-Amv*cos(Pmv))/(2.0*coord_step);
                    dzim = (Apv*sin(Ppv)-Amv*sin(Pmv))/(2.0*coord_step);
                    c0 = cos(phase0[i]);
                    s0 = sin(phase0[i]);
                    dAmpd[Nd] = dzre*c0 + dzim*s0;
                    dPhased[Nd] = dzim*c0 - dzre*s0;
                }
                Nd++;
            }

            gsl_spline_free(PAAspline); PAAspline = NULL;
            gsl_spline_free(PPPspline); PPPspline = NULL;
            gsl_spline_free(MAAspline); MAAspline = NULL;
            gsl_spline_free(MPPspline); MPPspline = NULL;

            if(Nd < 4) continue;

            if(getenv("THM_AP_DERIV_DEBUG") != NULL)
            {
                double num_a = 0.0, den_a = 0.0, num_p = 0.0;
                for(i=0; i<Nd; i++)
                {
                    num_a += (dAmpd[i]+Ampd[i])*(dAmpd[i]+Ampd[i]);
                    den_a += Ampd[i]*Ampd[i];
                    num_p += dPhased[i]*dPhased[i];
                }
                fprintf(stderr,
                        "APDERIV_DEBUG channel %d carrier %d Nd %d rel_dA_plus_A %.15e rms_AdPhi_over_A %.15e\n",
                        ch, k, Nd,
                        den_a > 0.0 ? sqrt(num_a/den_a) : 0.0,
                        den_a > 0.0 ? sqrt(num_p/den_a) : 0.0);
            }

            PFspline = gsl_spline_alloc(gsl_interp_cspline, Nd);
            DAFspline = gsl_spline_alloc(gsl_interp_akima, Nd);
            DPFspline = gsl_spline_alloc(gsl_interp_akima, Nd);
            if(PFspline == NULL || DAFspline == NULL || DPFspline == NULL) goto cleanup;
            gsl_spline_init(PFspline, freqd, phased, Nd);
            gsl_spline_init(DAFspline, freqd, dAmpd, Nd);
            gsl_spline_init(DPFspline, freqd, dPhased, Nd);
            gsl_interp_accel_reset(PFacc);
            gsl_interp_accel_reset(DAFacc);
            gsl_interp_accel_reset(DPFacc);

            /* Match the production mask: the SPA owns only its early branch. */
            WDMpixelsTimeScanRange(Ns_response, ctx[0].TS,
                                   freq_track[ch][k], ctx[0].TS[0],
                                   spa_support_tmax, nmid, nsize, N, wdms);
            tail_time = tc;
            tail_amp = -1.0;
            for(i=0; i<Ns_response; i++)
            {
                double amp_abs = fabs(Apost[0][ch][k][i]);
                if(isfinite(amp_abs) && amp_abs > tail_amp)
                {
                    tail_amp = amp_abs;
                    tail_time = ctx[0].TS[i];
                }
            }
            WDMpixelsAddMergerFrequencyTail(nmid, nsize, freqd[Nd-1],
                                            tail_time, wdms);

            WDMtrackAPProductDerivative(wdmwave, listn, listm, &Np, Nd, N,
                                        nmid, nsize, freqd,
                                        PFacc, PFspline,
                                        DAFacc, DAFspline,
                                        DPFacc, DPFspline,
                                        wdms);
            for(p=0; p<Np; p++) wdm[listn[p]][listm[p]] += wdmwave[p];

            /*
             * Endpoint derivative replacement uses the same time-scan support
             * as the production waveform path, but applied to the finite
             * difference waveform d h(t)/d lambda.  This keeps low-frequency
             * packets on the SPA derivative and replaces only packets whose
             * time/frequency support belongs to the direct short endpoint FFT.
             */
            {
                double direct_start_time, carrier_join_time, endpoint_stop_time;
                double fmax_spectrum;

                direct_start_time = setup_driver[2]+setup_driver[3]+
                                    SHORTFFT_MERGER_TAPER_MARGIN_SECONDS;
                carrier_join_time = setup_carrier[ch][k][5];
                if(!isfinite(carrier_join_time))
                {
                    carrier_join_time = ctx[0].TS[(int)setup_carrier[ch][k][4]];
                }
                /*
                 * A hard replacement should not ask the short FFT to replace
                 * packets before the carrier's own SPA/FFT join.  A blended
                 * endpoint is different: the direct FFT must be available on
                 * the low-frequency side of the transition as well.  Its real
                 * validity condition is the common endpoint taper-safe time;
                 * the frequency blend weight decides how much of that
                 * coefficient is used.
                 */
                if(!thm_wdm_blend_endpoint_enabled &&
                   direct_start_time < carrier_join_time)
                {
                    direct_start_time = carrier_join_time;
                }
                endpoint_stop_time = setup_driver[2]+setup_driver[0]*setup_driver[1];

                WDMpixelsTimeScanRange(Ns_response, ctx[0].TS,
                                       freq_track[ch][k],
                                       direct_start_time,
                                       endpoint_stop_time,
                                       nmid, nsize, N, wdms);

                tail_time = tc;
                tail_amp = -1.0;
                for(i=0; i<Ns_response; i++)
                {
                    double amp_abs = fabs(Apost[0][ch][k][i]);
                    if(isfinite(amp_abs) && amp_abs > tail_amp)
                    {
                        tail_amp = amp_abs;
                        tail_time = ctx[0].TS[i];
                    }
                }
                fmax_spectrum = setup_carrier[ch][k][6]/
                                (setup_carrier[ch][k][0]*setup_carrier[ch][k][1]);
                WDMpixelsAddMergerFrequencyTail(nmid, nsize, fmax_spectrum,
                                                tail_time, wdms);
                WDMmergePixelPlans(nmid_endpoint, nsize_endpoint,
                                   nmid, nsize);
            }

            gsl_spline_free(PFspline); PFspline = NULL;
            gsl_spline_free(DAFspline); DAFspline = NULL;
            gsl_spline_free(DPFspline); DPFspline = NULL;
        }

        short_hsum = double_vector((int)setup_driver[1]);
        if(short_hsum == NULL) goto cleanup;
        for(i=0; i<(int)setup_driver[1]; i++)
        {
            t = setup_driver[2]+(double)i*setup_driver[0];
            hval = 0.0;
            if(t >= ctx[0].TS[0] && t <= ctx[0].TS[Ns_response-1])
            {
                for(k=0; k<ncarriers; k++)
                {
                    A0v = gsl_spline_eval(TDAspline[k], t, TDAacc[k]);
                    P0v = gsl_spline_eval(TDPspline[k], t, TDPacc[k]);
                    Apv = gsl_spline_eval(TDAspline[ncarriers+k], t,
                                          TDAacc[ncarriers+k]);
                    Ppv = gsl_spline_eval(TDPspline[ncarriers+k], t,
                                          TDPacc[ncarriers+k]);
                    Amv = gsl_spline_eval(TDAspline[2*ncarriers+k], t,
                                          TDAacc[2*ncarriers+k]);
                    Pmv = gsl_spline_eval(TDPspline[2*ncarriers+k], t,
                                          TDPacc[2*ncarriers+k]);
                    (void)A0v;
                    (void)P0v;
                    hval += (Apv*cos(Ppv)-Amv*cos(Pmv))/(2.0*coord_step);
                }
            }
            if(t-setup_driver[2] < setup_driver[3])
            {
                hval *= 0.5*(1.0-cos(M_PI*(t-setup_driver[2])/setup_driver[3]));
            }
            short_hsum[i] = hval;
        }

        f_endpoint_start = f_replace_start[ch];
        if(!isfinite(f_endpoint_start) || f_endpoint_start < 0.0) f_endpoint_start = 0.0;

        f_endpoint_max = setup_driver[6]/(setup_driver[0]*setup_driver[1]);
        if(f_endpoint_max > 0.5/setup_driver[0]) f_endpoint_max = 0.5/setup_driver[0];
        if(thm_wdm_blend_endpoint_enabled)
        {
            endpoint_mlo = (int)floor((f_endpoint_start-
                                      blend_half_width-wdms->FB)/wdms->DF);
        }
        else
        {
            endpoint_mlo = (int)ceil((f_endpoint_start+wdms->FB)/wdms->DF);
        }
        endpoint_mmax = (int)floor((f_endpoint_max-wdms->FB)/wdms->DF);
        if(endpoint_mlo < 1) endpoint_mlo = 1;
        if(endpoint_mmax > Nf-1) endpoint_mmax = Nf-1;
        if(endpoint_mmax >= endpoint_mlo)
        {
            if(thm_wdm_blend_endpoint_enabled)
            {
                WDMblendWithShortFFTThreshold(wdm, nmid_endpoint, nsize_endpoint,
                                              short_hsum, setup_driver,
                                              f_endpoint_start, f_endpoint_max,
                                              blend_half_width, wdms, NULL,
                                              &endpoint_direct_layers,
                                              &endpoint_direct_pixels);
            }
            else
            {
                WDMreplaceWithShortFFTThreshold(wdm, nmid_endpoint, nsize_endpoint,
                                                short_hsum, setup_driver,
                                                f_endpoint_start, f_endpoint_max,
                                                wdms, NULL,
                                                &endpoint_direct_layers,
                                                &endpoint_direct_pixels);
            }
        }

        final_pixels = thm_sparse_wdm_channel_from_dense(&out_tracks->channel[ch], wdm);
        if(final_pixels < 0) goto cleanup;

        free_double_vector(short_hsum); short_hsum = NULL;
        for(i=0; i<ntimespl; i++)
        {
            if(TDAspline[i] != NULL) gsl_spline_free(TDAspline[i]);
            if(TDPspline[i] != NULL) gsl_spline_free(TDPspline[i]);
            if(TDAacc[i] != NULL) gsl_interp_accel_free(TDAacc[i]);
            if(TDPacc[i] != NULL) gsl_interp_accel_free(TDPacc[i]);
        }
        free(TDAacc); TDAacc = NULL;
        free(TDPacc); TDPacc = NULL;
        free(TDAspline); TDAspline = NULL;
        free(TDPspline); TDPspline = NULL;
    }

    status = 0;

cleanup:
    if(TDAspline != NULL || TDPspline != NULL || TDAacc != NULL || TDPacc != NULL)
    {
        int ntimespl = (ctx[0].ncarriers > 0) ? Nvar*ctx[0].ncarriers : 0;
        for(i=0; i<ntimespl; i++)
        {
            if(TDAspline != NULL && TDAspline[i] != NULL) gsl_spline_free(TDAspline[i]);
            if(TDPspline != NULL && TDPspline[i] != NULL) gsl_spline_free(TDPspline[i]);
            if(TDAacc != NULL && TDAacc[i] != NULL) gsl_interp_accel_free(TDAacc[i]);
            if(TDPacc != NULL && TDPacc[i] != NULL) gsl_interp_accel_free(TDPacc[i]);
        }
    }
    free(TDAacc);
    free(TDPacc);
    free(TDAspline);
    free(TDPspline);

    if(AFspline != NULL) gsl_spline_free(AFspline);
    if(PFspline != NULL) gsl_spline_free(PFspline);
    if(DAFspline != NULL) gsl_spline_free(DAFspline);
    if(DPFspline != NULL) gsl_spline_free(DPFspline);
    if(PAAspline != NULL) gsl_spline_free(PAAspline);
    if(PPPspline != NULL) gsl_spline_free(PPPspline);
    if(MAAspline != NULL) gsl_spline_free(MAAspline);
    if(MPPspline != NULL) gsl_spline_free(MPPspline);
    if(AFacc != NULL) gsl_interp_accel_free(AFacc);
    if(PFacc != NULL) gsl_interp_accel_free(PFacc);
    if(DAFacc != NULL) gsl_interp_accel_free(DAFacc);
    if(DPFacc != NULL) gsl_interp_accel_free(DPFacc);
    if(PAAacc != NULL) gsl_interp_accel_free(PAAacc);
    if(PPPacc != NULL) gsl_interp_accel_free(PPPacc);
    if(MAAacc != NULL) gsl_interp_accel_free(MAAacc);
    if(MPPacc != NULL) gsl_interp_accel_free(MPPacc);

    if(SLacc != NULL)
    {
        for(i=0; i<3; i++)
        {
            if(SLspline != NULL && SLspline[i] != NULL) gsl_spline_free(SLspline[i]);
            if(SLacc[i] != NULL) gsl_interp_accel_free(SLacc[i]);
        }
    }
    if(SPacc != NULL)
    {
        for(i=0; i<9; i++)
        {
            if(SPspline != NULL && SPspline[i] != NULL) gsl_spline_free(SPspline[i]);
            if(SPacc[i] != NULL) gsl_interp_accel_free(SPacc[i]);
            if(SVspline != NULL && SVspline[i] != NULL) gsl_spline_free(SVspline[i]);
            if(SVacc != NULL && SVacc[i] != NULL) gsl_interp_accel_free(SVacc[i]);
        }
    }
    free(SLacc);
    free(SLspline);
    free(SPacc);
    free(SPspline);
    free(SVacc);
    free(SVspline);

    for(v=0; v<Nvar; v++) thm_ap_context_free(&ctx[v]);
    if(Apost != NULL) free_double_quad(Apost, Nvar, Nchan, ncarriers);
    if(Ppost != NULL) free_double_quad(Ppost, Nvar, Nchan, ncarriers);
    if(freq_track != NULL) free_double_tensor(freq_track, Nchan, ncarriers);
    if(setup_carrier != NULL) free_double_tensor(setup_carrier, Nchan, ncarriers);
    if(Xmode != NULL) free_double_vector(Xmode);
    if(Ymode != NULL) free_double_vector(Ymode);
    if(Zmode != NULL) free_double_vector(Zmode);
    if(Xfmode != NULL) free_double_vector(Xfmode);
    if(Yfmode != NULL) free_double_vector(Yfmode);
    if(Zfmode != NULL) free_double_vector(Zfmode);
    if(phi_ref != NULL) free_double_vector(phi_ref);
    if(phi_res != NULL) free_double_vector(phi_res);
    if(omega_plan != NULL) free_double_vector(omega_plan);
    if(freq0 != NULL) free_double_vector(freq0);
    if(phase0 != NULL) free_double_vector(phase0);
    if(Amp0 != NULL) free_double_vector(Amp0);
    if(freqp != NULL) free_double_vector(freqp);
    if(phasep != NULL) free_double_vector(phasep);
    if(Ampp != NULL) free_double_vector(Ampp);
    if(freqm != NULL) free_double_vector(freqm);
    if(phasem != NULL) free_double_vector(phasem);
    if(Ampm != NULL) free_double_vector(Ampm);
    if(freqd != NULL) free_double_vector(freqd);
    if(phased != NULL) free_double_vector(phased);
    if(Ampd != NULL) free_double_vector(Ampd);
    if(dAmpd != NULL) free_double_vector(dAmpd);
    if(dPhased != NULL) free_double_vector(dPhased);
    if(wdmwave != NULL) free_double_vector(wdmwave);
    if(short_hsum != NULL) free_double_vector(short_hsum);
    if(short_tmp != NULL) free_double_vector(short_tmp);
    if(nmid != NULL) free_int_vector(nmid);
    if(nsize != NULL) free_int_vector(nsize);
    if(nmid_endpoint != NULL) free_int_vector(nmid_endpoint);
    if(nsize_endpoint != NULL) free_int_vector(nsize_endpoint);
    if(listn != NULL) free_int_vector(listn);
    if(listm != NULL) free_int_vector(listm);
    if(wdm != NULL) free_double_matrix(wdm, Nt);
    if(Larray != NULL) free_double_matrix(Larray, 3);
    if(Parray != NULL) free_double_tensor(Parray, 3, 3);
    if(Varray != NULL) free_double_tensor(Varray, 3, 3);
    if(tarray != NULL) free_double_vector(tarray);
    if(wdms != NULL) free(wdms);
    thm_diagnostics_enabled = old_diagnostics;
    thm_summary_output_enabled = old_summary_output;

    return status;
}

int phenomt_tdi_legacy_main(void) {
    const double phi0 = 0.0;
    double deltaF;
    double m1, m2;
    double chi1;
    double chi2;
    double distance; // Gpc
    double Mtot, Mc, eta, tx, alpha;
    double f, t, cp, sp, p, tshift, ftrans, tseg, Tobs;
    double cv, sv, u, v, A, dtm, tc, t0, tp, ts, fs;
    double *params;
    double *TS, *AS, *PS, *FS;
    double *h, *phs;
    double pf, pw, pold;
    int Ns, Nsmax, Nsam, Nts, Ntsmax;
    int ret, flag;
    double t22, fring, fdamp;
    double *setup;
    
    FILE *out;
    
    double x, y;
    int i, ii, j, k, is, N, Nx, Nc;
    
    clock_t start, end;
    clock_t timing_start, timing_end;
    double cpu_time_used;
    double spa_freq_time, spa_track_time;
    double adaptive_ap_time, fast_tdi_time, fast_wdm_time;
    double fft_build_time, fft_track_time, fft_spectrum_write_time;
    
    struct wdmshape *wdms  = malloc(sizeof(struct wdmshape));
    
    wdmvalues(wdms);

    N = Nt*Nf;
    Tobs = wdms->Tobs;
    
    
    // We start by setting up a coarse sampling of the LISA orbit and
    // spline the various terms we need to compute the TDI response
    // This only has to be done once. These same splines are used for
    // galactic bianries, EMRIs etc.
    double dtx, dtc;
    double constellation_tmin, constellation_tmax;
    double **Larray, *tarray, *times;
    double ***Parray, ***Varray;
    
    

    Nc = (int)(200.0*(Tobs)/SECSYR);
    //printf("%d\n", Nc);
    if (Nc < 20) Nc = 20;
    dtx = (Tobs)/(double)(Nc-1);
    if(dtx < CONSTELLATION_MIN_PADDING_SECONDS) dtx = CONSTELLATION_MIN_PADDING_SECONDS;
    dtc = (Tobs+2.0*dtx)/(double)(Nc-1);  // tarray extends beyond observation time by dtx on each end to allow for interpolation
    
    printf("constellation padding %e\n", dtx);
    
    // The constellation orientation can be computed once and stored
    Larray = double_matrix(3,Nc);  // armlengths
    Parray = double_tensor(3,3,Nc);  // spacecraft positions
    Varray = double_tensor(3,3,Nc);  // arm vectors
    
    tarray = double_vector(Nc);
    for(i=0; i<Nc ;i++) tarray[i] = -dtx + dtc*(double)(i);   // coarse time sampling
    constellation_tmin = tarray[0];
    constellation_tmax = tarray[Nc-1];

    constellation(Nc, tarray, Larray, Parray, Varray);
    
    // now we spline the orbits. These can be used by all waveform types and only have to be computed and stored once. Different waveform types will want to use different time spacings for their interpolation, so we need to be able to access the orbi  information at any random time. For example, MBHBs will use time arrays that are non-uniform (closer spaced near merger) and may terminate before the observation time if the system merges.
    
    // Probably makes sense to put these splines into a spacecraft structure
 
    gsl_interp_accel **SLacc = malloc(3 * sizeof(gsl_interp_accel *));
    gsl_spline **SLspline = malloc(3 *sizeof(gsl_spline *));
    for(i=0; i<3; i++)
    {
        SLacc[i] = gsl_interp_accel_alloc();
        SLspline[i] = gsl_spline_alloc (gsl_interp_cspline, Nc);
    }
    
    for(i = 0 ; i< 3; i++)
    {
        gsl_spline_init(SLspline[i], tarray, Larray[i], Nc);
    }
    
    gsl_interp_accel **SPacc = malloc(9 * sizeof(gsl_interp_accel *));
    gsl_spline **SPspline = malloc(9 *sizeof(gsl_spline *));
    gsl_interp_accel **SVacc = malloc(9 * sizeof(gsl_interp_accel *));
    gsl_spline **SVspline = malloc(9 *sizeof(gsl_spline *));
    for(i=0; i<9; i++)
    {
        SPacc[i] = gsl_interp_accel_alloc();
        SPspline[i] = gsl_spline_alloc (gsl_interp_cspline, Nc);
        SVacc[i] = gsl_interp_accel_alloc();
        SVspline[i] = gsl_spline_alloc (gsl_interp_cspline, Nc);
    }
    
    for(i = 0 ; i< 3; i++)
    {
        for(j = 0 ; j< 3; j++)
        {
            k = j+i*3;
            gsl_spline_init(SPspline[k], tarray, Parray[i][j], Nc);
            gsl_spline_init(SVspline[k], tarray, Varray[i][j], Nc);
        }
    }
    
    // now that the constellation has been set up, time to compute the IMRPhenomT amplitude and phase as a function of time
   
    m1 = 2.0e5*TSUN; // seconds, heavier mass by LAL convention
    m2 = 1.0e5*TSUN; // seconds
    // [0] m1  [1] m2  [2] Spin1 [3] Spin2 [4] phic [5] tc [6] ln(distance)
    // [7] EclipticCoLatitude, [8] EclipticLongitude  [9] polarization, [10] inclination
    
    chi1 = 0.42;
    chi2 = 0.85;
    distance = 1.0; // Gpc
    tc = 3.0e7;

    enforce_phenom_mass_hierarchy(&m1, &m2, &chi1, &chi2);
    
    Mtot = (m1 + m2);  // total mass in seconds
    eta = (m1*m2)/(Mtot*Mtot);
    Mc = pow(m1*m2,3.0/5.0)/pow(Mtot, 1.0/5.0);
    
    params = (double*)malloc(sizeof(double)* (11));
    
    // set the source parameters
    params[0] = m1;
    params[1] = m2;
    params[2] = chi1;
    params[3] = chi2;
    params[4] = phi0;
    params[5] = tc;
    params[6] = log(distance);
    params[7] = 2.31;       // EclipticCoLatitude
    params[8] = 0.57;            // EclipticLongitude
    params[9] = 0.4;                // polarization
    params[10] = 0.3;               // cos inclination
    
    // used to construct full time domain signal. Not needed with fast likelihood methods.
    h = double_vector(N);
    phs = double_vector(N);
    
    Nsmax = 10000;
    
   // printf("%f\n", Tobs/dTmax);
    
    double *tspace;

    TS = double_vector(Nsmax);
    tspace = double_vector(Nsmax);
    AS = double_vector(Nsmax);
    PS = double_vector(Nsmax);
    FS = double_vector(Nsmax);
    
    start = clock();
    
    // this extracts the time domain amplitude and phase on a coarse grid.
    // The spacing is chosen so that the maximum spacing between points is
    // dTmax 1.0e5 seconds (roughly a day). The spacing gets finer around merger.
    
    // holds information needed to do fast time -> frequency mapping
    setup = double_vector(7);
    
    Ns = PhenomT_AP(params, Nsmax, TS, tspace, AS, PS, FS, SPacc, SPspline, constellation_tmin, constellation_tmax, setup);
    end = clock();
    adaptive_ap_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("samples = %d\n", Ns);
    
    /*
     ftran recomputes the join frequency from the TDI-channel phase derivative.
     setup[5] stores the continuous requested join time; setup[4] is the
     corresponding AP-grid index used for array slicing.  Allocate against the
     full FFT frequency range rather than any join-local frequency estimate.
     */
    Ntsmax = (int)(setup[4]+setup[6])+32;
    
    printf("max transform samples = %d\n", Ntsmax);
    
    
    out = fopen("times.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%.15e %.15e\n", TS[i], tspace[i]);
    }
    fclose(out);
    
    
    out = fopen("ft.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%.15e %.15e %.15e\n", FS[i], TS[i], tspace[i]);
    }
    
    
    double *freq, *phase, *Amp;
    
    freq = double_vector(Ntsmax);
    phase = double_vector(Ntsmax);
    Amp = double_vector(Ntsmax);
    
    fast_tdi_time = 0.0;
    timing_start = clock();

    // set up splines for A(t), Phi(t)
    gsl_interp_accel *PSacc = gsl_interp_accel_alloc();
    gsl_spline *PSspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *ASacc = gsl_interp_accel_alloc();
    gsl_spline *ASspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_spline_init(PSspline, TS, PS, Ns);
    gsl_spline_init(ASspline, TS, AS, Ns);
 
    free(PS);
    free(AS);

    // we have to allow for the time delay across the LISA orbit
    // otherwise the interpolation can go out of range
    
    i = 0;
    do
    {
        i++;
    }while(tspace[Ns-i] > TS[Ns-1]);
    Ns = Ns-i;
    printf("%d %e %e\n", Ns, TS[Ns+i-1], tspace[Ns-1]);
     
    
    double *phiR, *phiB;
    // coarse sampled phase at reference spacecraft
    phiR = double_vector(Ns);
    phiB = double_vector(Ns);
    timing_end = clock();
    fast_tdi_time += ((double) (timing_end - timing_start)) / CLOCKS_PER_SEC;
    
    // the TDI phase extraction is done using the time samples define by the waveform model, not the constellation sampling rate
    
    // shift reference times from Barycenter to spacecraft 0
   // detector_time(TS, tspace, params, SPacc, SPspline, Ns);
    
    /*
    // check the inverse mapping from spacecraft to baycenter
    double *tb;
    tb = double_vector(Ns);
    barycenter_time(tb, tspace, params, SPacc, SPspline, Ns);
    out = fopen("times.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%.15e %.15e %.15e\n", TS[i], tspace[i], tb[i]);
    }
    fclose(out);
    */
    
    out = fopen("PTspline.dat","w");
    fprintf(out, "# time_s frequency_Hz amplitude\n");
    fprintf(out, "# WARNING: frequency_Hz is from a spline derivative and is diagnostic only; spline derivatives can ring on nonuniform grids near TDI transfer features.\n");
    for(i=0; i< NPTSPLINE; i++)
    {
        t = Tobs*(double)(i)/(double)(NPTSPLINE-1);
        if(t >= TS[0] && t <= TS[Ns-1]) fprintf(out,"%e %e %e\n", t, gsl_spline_eval_deriv (PSspline, t, PSacc)/(2.0*M_PI), gsl_spline_eval (ASspline, t, ASacc));
    }
    fclose(out);
    
    timing_start = clock();
    
    for (i=0; i< Ns; i++)
    {
        phiR[i] = gsl_spline_eval (PSspline, tspace[i], PSacc);
        phiB[i] = gsl_spline_eval (PSspline, TS[i], PSacc);
    }
    
    // set up TDI responses. These are done in phase quadratures (for example, X and Xf are phased by pi/2. We do this so that we can extract the amplitude and phase
    
    double *X, *Y, *Z, *Xf, *Yf, *Zf;
    double *phiX, *AX, *phiY, *AY, *phiZ, *AZ;
    
    X = double_vector(Ns);
    Y = double_vector(Ns);
    Z = double_vector(Ns);
    Xf = double_vector(Ns);
    Yf = double_vector(Ns);
    Zf = double_vector(Ns);
    
    phiX = double_vector(Ns);
    AX = double_vector(Ns);
    phiY = double_vector(Ns);
    AY = double_vector(Ns);
    phiZ = double_vector(Ns);
    AZ = double_vector(Ns);
    
    // here we compute the TDI response on the coarse waveform grid
    fast_response(TS, Ns, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, ASacc, ASspline, PSacc, PSspline, X, Y, Z, Xf, Yf, Zf);
    
    // next we extract the amplitude and phase and remove any 2 pi phase wraps
    
    extractAP(Ns, AX, phiX, X, Xf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiX);
    
    extractAP(Ns, AY, phiY, Y, Yf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiY);
    
    extractAP(Ns, AZ, phiZ, Z, Zf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiZ);
    
    end = clock();
    fast_tdi_time += ((double) (end - timing_start)) / CLOCKS_PER_SEC;
    cpu_time_used = adaptive_ap_time + fast_tdi_time;
    printf("TDI calculation took %f seconds\n", cpu_time_used);
    
    
    // This is the final result. The full phase in each channel is found by adding phiR to phiX etc. AX etc arw the complete TDI amplitudes
    out = fopen("PhenomT_TDI.dat","w");
    for (i = 0; i < Ns; ++i)
    {
        fprintf(out,"%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n", TS[i], phiR[i], phiB[i], phiX[i], phiY[i], phiZ[i], AX[i], AY[i], AZ[i]);
    }
    fclose(out);
    
    int *nmid, *nsize;
    nmid = int_vector(Nf);
    nsize = int_vector(Nf);
	    double **wdm;
	    double **wdmfft;
	    double **wdmfftcut;
	    double **wdmshort;
	    double **wdmsplit;
    
    // holds the time (n) and frequency (m) labels of the active pixels
    // this list could be about 5% smaller if we cut the boundaries closer
    // but not really worth the bookeeping effort
    int Np;  // total pixels used
    int *listn;
    int *listm;
	    double *wdmwave;
	    double *wdmwavefft;
	    double *wdmwavefftcut;
	    double *hfft;
	    double *short_htime;
	    int NpFFT;
	    int NpFFTCut;
	    int merger_direct_layers;
	    int merger_direct_pixels;
	    int split_fft_layers;
	    int split_fft_pixels;
    
    listn = int_vector(N);
    listm = int_vector(N);
	    wdmwave = double_vector(N);
	    wdmwavefft = double_vector(N);
	    wdmwavefftcut = double_vector(N);
    hfft = double_vector(N);
    short_htime = double_vector((int)(setup[1]));
    
    fast_wdm_time = 0.0;
    timing_start = clock();

    // We use the reference (pre-TDI) time frequency mapping to set up the transform
    // Use this for X, Y and Z (all will have the same pixel list)
    WDMpixels(Ns, TS, FS, nmid, nsize, N, wdms);
    
    //for (i=0; i< Nf; i++) printf("%d %d %d\n", i, nmid[i], nsize[i]);
    
    // splines of time domain phase and amplitude
    gsl_interp_accel *Pacc = gsl_interp_accel_alloc();
    gsl_spline *Pspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *Aacc = gsl_interp_accel_alloc();
    gsl_spline *Aspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    
    // X channel time domain phase and amplitude
    for (i = 0; i < Ns; ++i)
    {
        PS[i] = phiR[i]+phiX[i];
        AS[i] = AX[i];
    }
    gsl_spline_init(Pspline, TS, PS, Ns);
    gsl_spline_init(Aspline, TS, AS, Ns);
    timing_end = clock();
    fast_wdm_time += ((double) (timing_end - timing_start)) / CLOCKS_PER_SEC;
    
    double fdot;
    
    out = fopen("test.dat","w");
    fprintf(out, "# response_time_s detector_time_s plan_freq_Hz spline_freq_Hz spline_fdot_Hz_per_s amplitude\n");
    fprintf(out, "# WARNING: spline_freq_Hz and spline_fdot_Hz_per_s are spline-derivative diagnostics only; production SPA should use local nonuniform derivatives.\n");
    for (i = 0; i < Ns; ++i)
    {
        f = gsl_spline_eval_deriv (Pspline, TS[i], Pacc)/(2.0*M_PI);
        fdot = gsl_spline_eval_deriv2 (Pspline, TS[i], Pacc)/(2.0*M_PI);
        fprintf(out,"%.15e %.15e %.15e %.15e %.15e %.15e\n", TS[i], tspace[i], FS[i], f, fdot, AX[i]);
    }
    fclose(out);
    
    /*
     Production fast path:
     ftran maps the coarse time-domain amplitude/phase into a sparse
     frequency-domain amplitude/phase representation. During the inspiral this
     uses the stationary phase approximation (SPA); the resulting splines are
     then sampled by WDMtrack on each active local WDM packet.

     Low-frequency edge note:
     For a finite observation, the raw FFT reference contains start-edge
     leakage below the instantaneous frequency at t=0. The SPA path has no
     stationary point below its first frequency sample and WDMtrack zero-fills
     those packet samples. The lowest active Meyer layers, overlap that support edge. Use the
     tapered reference Xtime_tukey.dat plus the wdm_match_info.dat time cuts
     when checking the physical interior of the observation.
     */
    start = clock();
	    Nts = ftran(setup, TS, Ns, Tobs, Aacc, Aspline, Pacc, Pspline, freq, phase, Amp, short_htime);
    
    end = clock();
    spa_freq_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    fast_wdm_time += spa_freq_time;
    printf("SPA time to frequency %f seconds\n", spa_freq_time);
    
    printf("%d %d\n", Nts, Ntsmax);
    
    //2.999947e+07 2.999968e+07
    
    out = fopen("ch.dat","w");
    for (i=0; i< Nts; i++)
    {
        fprintf(out, "%e %e %e\n", freq[i], phase[i], Amp[i]);
    }
    fclose(out);
    
    start = clock();
    
    // splines of frequency domain phase and amplitude
    gsl_interp_accel *PFacc = gsl_interp_accel_alloc();
    gsl_spline *PFspline = gsl_spline_alloc (gsl_interp_cspline, Nts);
    gsl_interp_accel *AFacc = gsl_interp_accel_alloc();
    /*
     * The signed SPA/FFT Fourier amplitude can be sparsely sampled near TDI
     * transfer features. Akima interpolation preserves the local shape better
     * than a natural cubic, which can overshoot and make fake WDM power.
     */
    gsl_spline *AFspline = gsl_spline_alloc (gsl_interp_akima, Nts);
    gsl_spline_init(PFspline, freq, phase, Nts);
    gsl_spline_init(AFspline, freq, Amp, Nts);
    
    WDMtrack(wdmwave, listn, listm, &Np, Nts, N, nmid, nsize, tc, freq, AFacc, AFspline, PFacc, PFspline, wdms);
    
    end = clock();
    spa_track_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    fast_wdm_time += spa_track_time;
    printf("SPA frequency to wdm took %f seconds\n", spa_track_time);
    printf("timing_seconds\n");
    printf("adaptive_ap %.6f\n", adaptive_ap_time);
    printf("fast_tdi %.6f\n", fast_tdi_time);
    printf("fast_wdm %.6f\n", fast_wdm_time);
    
    wdm = double_matrix(Nt,Nf+1);
    unpack_wdm_track(wdm, listn, listm, wdmwave, Np);
    
    printf("# wavelet pixels used %d\n", Np);

	    write_track_pixels("track_pixels.dat", listn, listm, wdmwave, Np);
	    write_wdm_matrix("wtranfast.dat", wdm);
	    write_wdm_binary("BinaryFast.dat", wdm, wdms);

	    /*
	     Simple merger-direct WDM diagnostic:
	     Start from the standard fast SPA/short-FFT-spline result, then replace
	     only those merger/ringdown layers whose full Meyer frequency support maps
	     to times after the short-FFT roll-on has flattened. This tests whether
	     evaluating the local WDM packet directly from the short merger time
	     segment avoids errors introduced by unwrapping/splining A(f), phi(f) in the transition
	     region, without changing the inspiral SPA path.
	     */
	    start = clock();
	    wdmshort = double_matrix(Nt,Nf+1);
	    for(i=0; i<Nt; i++)
	    {
	        for(j=0; j<=Nf; j++) wdmshort[i][j] = wdm[i][j];
	    }
	    WDMreplaceMergerShortWindow(wdmshort, nmid, nsize, short_htime, setup, Ns, TS, FS, wdms, &merger_direct_layers, &merger_direct_pixels);
	    end = clock();
	    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	    printf("short merger window direct WDM replaced %d layers / %d pixels in %f seconds\n",
	           merger_direct_layers, merger_direct_pixels, cpu_time_used);
	    write_wdm_matrix("wtranfast_mergerdirect.dat", wdmshort);
	    write_wdm_binary("BinaryFastMergerDirect.dat", wdmshort, wdms);

	    /*
	     Experimental split-FFT diagnostic:
	     Split h(t) with a three-piece partition of unity. The low-frequency
	     inspiral side uses a full-duration R=128 FFT; the bend uses a local
	     R=64, 64-DT FFT; the vertical merger/ringdown uses a local full-rate,
	     16-DT FFT whose upper edge covers the generated ringdown tail. The
	     taper from the bend block to the high block is placed so the bend taper
	     has flattened before its Nyquist frequency is reached.
	     */
	    start = clock();
	    wdmsplit = double_matrix(Nt,Nf+1);
	    WDMtrackSplitFFT(wdmsplit, nmid, nsize, N, TS[Ns-1], Aacc, Aspline, Pacc, Pspline, wdms, &split_fft_layers, &split_fft_pixels);
	    end = clock();
	    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	    printf("split FFT WDM used %d layers / %d pixels in %f seconds\n",
	           split_fft_layers, split_fft_pixels, cpu_time_used);
	    write_wdm_matrix("wtranfast_splitfft.dat", wdmsplit);
	    write_wdm_binary("BinaryFastSplitFFT.dat", wdmsplit, wdms);

	    /*
	     Diagnostic FFT-input path:
     This intentionally keeps WDMpixels, the active pixel list, phase
     recentering, and wdmtranF the same as the SPA path. The only change is
     that WDMtrackFFT samples a direct full-cadence FFT of the same X-channel
     time-domain waveform instead of the SPA frequency splines. Comparing
     wtranfft.dat to wtranfast.dat isolates errors in the SPA frequency-domain
     construction from errors in the local WDM machinery. The companion
     wtranfft_spacut.dat uses the FFT spectrum but applies the same frequency
     support cut as the SPA splines; it tests whether low-frequency packet
     samples hitting zeros are enough to reproduce the SPA mismatch.
     */
    start = clock();
    build_direct_fft_spectrum(hfft, N, TS[Ns-1], Aacc, Aspline, Pacc, Pspline);
    end = clock();
    fft_build_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("direct FFT spectrum build took %f seconds\n", fft_build_time);

#if WRITE_FFT_SPECTRUM_DIAGNOSTIC
    start = clock();
    out = fopen("Xchan_freq_notukey.dat","w");
    for(i=1; i<N/2; i++)
    {
        f = (double)(i)/Tobs;
        fprintf(out, "%e %e %e\n", f, hfft[i], hfft[N-i]);
    }
    fclose(out);
    write_fft_amp_phase_diagnostic("Xchan_AP_notukey.dat", hfft, N, Tobs, freq, Nts, AFacc, AFspline, PFacc, PFspline);
    end = clock();
    fft_spectrum_write_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("FFT spectrum diagnostic write took %f seconds\n", fft_spectrum_write_time);
#else
    fft_spectrum_write_time = 0.0;
#endif

    start = clock();
    WDMtrackFFT(wdmwavefft, listn, listm, &NpFFT, N, nmid, nsize, hfft, wdms, 0, 0.0, 0.0);
    end = clock();
    fft_track_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("FFT spectrum to wdm took %f seconds\n", fft_track_time);
    printf("Timing summary: SPA freq+WDM %f seconds, FFT build+WDM %f seconds, FFT diagnostic I/O %f seconds\n",
           spa_freq_time+spa_track_time, fft_build_time+fft_track_time, fft_spectrum_write_time);

    wdmfft = double_matrix(Nt,Nf+1);
    unpack_wdm_track(wdmfft, listn, listm, wdmwavefft, NpFFT);

    printf("# FFT wavelet pixels used %d\n", NpFFT);

    write_track_pixels("track_pixels_fft.dat", listn, listm, wdmwavefft, NpFFT);
    write_wdm_matrix("wtranfft.dat", wdmfft);
    write_wdm_binary("BinaryFFT.dat", wdmfft, wdms);

    start = clock();
    WDMtrackFFT(wdmwavefftcut, listn, listm, &NpFFTCut, N, nmid, nsize, hfft, wdms, 1, freq[0], freq[Nts-1]);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("FFT spectrum with SPA band cut to wdm took %f seconds\n", cpu_time_used);

    wdmfftcut = double_matrix(Nt,Nf+1);
    unpack_wdm_track(wdmfftcut, listn, listm, wdmwavefftcut, NpFFTCut);

    printf("# FFT SPA-cut wavelet pixels used %d\n", NpFFTCut);

    write_track_pixels("track_pixels_fft_spacut.dat", listn, listm, wdmwavefftcut, NpFFTCut);
    write_wdm_matrix("wtranfft_spacut.dat", wdmfftcut);
    
    out = fopen("Xtime.dat","w");
    for (i=0; i< N; i++)
    {
        t = (double)(i)*dt;
        h[i] = 0.0;
        if(t < TS[Ns-1])
        {
            h[i] = gsl_spline_eval (Aspline, t, Aacc)*cos(gsl_spline_eval (Pspline, t, Pacc));
        }
        fprintf(out, "%e %e\n", t, h[i]);
    }
    fclose(out);

    alpha = 2.0*REFERENCE_TUKEY_ROLL_SECONDS/Tobs;
    if(alpha > 1.0) alpha = 1.0;
    if(alpha < 0.0) alpha = 0.0;

    /*
     The raw finite observation has a hard rectangular start. Its FFT contains
     endpoint leakage below the instantaneous start frequency, while the SPA
     path has no stationary point there and zero-fills. Xtime_tukey.dat is the
     finite-observation reference to use when testing the fast SPA transform
     away from the artificial start/end boundaries.
     */
    tukey(h, alpha, N);
    out = fopen("Xtime_tukey.dat","w");
    for (i=0; i< N; i++)
    {
        t = (double)(i)*dt;
        fprintf(out, "%e %e\n", t, h[i]);
    }
    fclose(out);

    write_match_info("wdm_match_info.dat", wdms, alpha, 0.5*alpha*Tobs);
    
    // and now the full frequency domain signal

    gsl_fft_real_radix2_transform(h, 1, N);
    
    for (i=0; i< N; i++) h[i] *= (2.0*dt);
    
    i = 1;
    pw = 0.0;
    pold = atan2(h[N-i],h[i]);
    for (i=1; i < N/2; i++)
    {
        f = (double)(i)/Tobs;
        pf = atan2(h[N-i],h[i]);
        if(pold -pf > 5.0) pw += 2.0*M_PI;
        if(pf - pold > 5.0) pw -= 2.0*M_PI;
        pold = pf;
        phs[i] = ((pf+pw));
    }
    
    out = fopen("Xfreq.dat","w");
    for (i=1; i< N/2; i++)
    {
        f = (double)(i)/Tobs;
        fprintf(out, "%e %e %e\n", f, phs[i], sqrt(h[i]*h[i]+h[N-i]*h[N-i]));
    }
    fclose(out);
    
    out = fopen("Xchan_freq.dat","w");
    for (i=1; i< N/2; i++)
    {
        f = (double)(i)/Tobs;
        fprintf(out, "%e %e %e\n", f, h[i], h[N-i]);
    }
    fclose(out);
    
    out = fopen("Xf.dat","w");
    h[0] =0.0;
    for (i=1; i< N/2; i++)
    {
        f = (double)(i)/Tobs;
        A = 0.0;
        x = 0.0;
        if(f > freq[0] && f < freq[Nts-1])
        {
            A = gsl_spline_eval (AFspline, f, AFacc);
            x = gsl_spline_eval (PFspline, f, PFacc);
        }
        
        h[i] = A*cos(x);
        h[N-i] = A*sin(x);
        
        fprintf(out, "%e %e %e\n", f, h[i], h[N-i]);
    }
    fclose(out);
    
    
    gsl_fft_halfcomplex_radix2_inverse(h, 1, N);
    for (i=0; i< N; i++) h[i] /= (2.0*dt);
    
    out = fopen("Xt.dat","w");
    for (i=0; i< N; i++)
    {
        t = (double)(i)*dt;
        fprintf(out, "%e %e\n", t, h[i]);
    }
    fclose(out);
    
     
    return 0;
    
}

void transformplan(double Mc, double Mtot, double tc_output, double *response_time, double *omegat, double t22_output, double fring, double fdamp, double tmax, double *setup)
{
    transformplan_custom(Mc, Mtot, tc_output, response_time, omegat, t22_output, fring, fdamp,
                         tmax, setup, 0, 0.0, 0.0);
}

void transformplan_custom(double Mc, double Mtot, double tc_output, double *response_time, double *omegat, double t22_output, double fring, double fdamp, double tmax, double *setup, int use_join_override, double join_time_override, double rise_override)
{
    double tes, tee, Tspan, rise, dte, damping_rate, padded_end;
    double t;
    double tjoin, tjoin_target, fjoin, x;
    double *hend;
    int i, j, k, Nend;
    
    FILE *out;
    
    // covers out to 1.5 fring
    dte = 1.0/(6.0*fring);
    if(dte < dt) dte = dt;
    
    // sample rate
    setup[0] = dt*floor(dte/dt);
    dte = setup[0];
    
    rise = 1000.0*Mc;
    if(rise_override > 0.0) rise = rise_override;

    if(use_join_override)
    {
        tjoin_target = join_time_override;
        tjoin = dte*rint(tjoin_target/dte);
        /*
         * Preserve the old 22 geometry: the short FFT roll-on ends well before
         * the SPA/FFT join.  The power-of-two block rounding below may move the
         * actual segment start earlier, which is conservative for leakage.
         */
        tes = tjoin-1.75*rise;
    }
    else
    {
        tes = (t22_output-2.0*rise);
        /*
         * t22_output has already been mapped from the intrinsic transition
         * event to the SSB output/data coordinate.  Require the endpoint taper
         * to be fully flat before the earliest delayed response arrival.  The
         * historical two-rise geometry is retained whenever it is already more
         * conservative.
         */
        if(tes > t22_output-THM_RESPONSE_EARLY_MARGIN_SECONDS-rise)
        {
            tes = t22_output-THM_RESPONSE_EARLY_MARGIN_SECONDS-rise;
        }
        tjoin_target = t22_output-0.25*rise;
        tjoin = dte*rint(tjoin_target/dte);
    }
    /*
     The short FFT used for the merger-ringdown replacement must include the
     same post-merger padding used when building the waveform grid.  The late
     response margin covers the solar-barycenter/LISA projection plus the
     longest four-arm TDI-1 path, while 1000 Mtot gives the ringdown tail room
     to decay. The 10/alpha damping horizon is kept only as an additional lower
     bound; use fabs because some stripped-down fit conventions can carry a
     signed damping rate.
     */
    damping_rate = fabs(fdamp);
    padded_end = tc_output+THM_RESPONSE_LATE_MARGIN_SECONDS+1000.0*Mtot;
    tee = padded_end;
    if(damping_rate > 0.0 && tee < tc_output+10.0/damping_rate) tee = tc_output+10.0/damping_rate;
    if(tee < tmax) tee = tmax;
    
    x = floor(pow(2.0, floor(log2((tee-tes)/dte))+1.0));
    
    // number of samples in FFT
    setup[1] = x;
    
    Tspan = x*dte;
    
    tes = dte*rint(tee/dte)-Tspan;
    
    // start location of merger segment
    setup[2] = tes;
    
    setup[3] = rise;
    
    // where we join the SPA to the FFT
    // mapped to integer entry in array
    if(use_join_override && thm_diagnostics_enabled)
    {
        printf("transformplan_override join_time %e rounded_join_time %e rise %e\n",
               tjoin_target, tjoin, rise);
    }
    
    i = -1;
    do
    {
        i++;
        t = response_time[i];
    }
    while(t < tjoin_target);
    
    // location of tjoin mapped to sample in TS sample grid
    setup[4] = (double)(i);
    
    /*
     * Continuous requested join time.  setup[4] remains the first AP-grid
     * sample after this time for array slicing, while the short FFT still keeps
     * its internally rounded sample cadence.  Endpoint guards should use this
     * unsnapped time to define the SPA/FFT frequency boundary.  Near merger,
     * one adaptive-grid or endpoint-FFT sample step can move several WDM
     * layers, which shows up as coherent likelihood jumps when source
     * parameters are varied.
     */
    setup[5] = tjoin_target;
    
    if(thm_diagnostics_enabled)
    {
        printf("transformplan_join_sample %d %.15e %.15e requested_join %.15e rounded_join %.15e\n",
               i, response_time[i], omegat[i]/(2.0*M_PI), setup[5], tjoin);
    }
    
    // final frequency bin we need
    setup[6] = ceil(1.5*fring*Tspan);
    

}
 

int ftran(double *setup, double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *freq, double *phase, double *Amp, double *short_htime)
{
    
    double Tspan, dte, tes, tjoin, fjoin, pjoin;
    double rise;
    double t, f, A, x;
    double f1, f2, p1, p2;
    int i, j, k;
    int ii, jj, Nend, Nx, imx, istart;
    double dA1, dA2, dA3;
    double *hend, *flip, *pjump;
    double *phase_grid, *freq_grid, *fdot_grid;
    double fdot_val;
    
    FILE *out;
    
    dte = setup[0];
    Nend = (int)(setup[1]);
    Tspan = setup[1]*dte;
    tes = setup[2];
    rise = setup[3];
    jj = (int)(setup[4]);
    imx = (int)(setup[6]);
    if(imx > Nend/2)
    {
        imx = Nend/2;
    }

    phase_grid = double_vector(Ns);
    freq_grid = double_vector(Ns);
    fdot_grid = double_vector(Ns);
    build_nonuniform_phase_derivatives(Ns, TS, PSacc, PSspline,
                                       phase_grid, freq_grid, fdot_grid);
    
    tjoin = TS[jj];
    fjoin = freq_grid[jj];
    if(!isfinite(fjoin) || fjoin <= 0.0)
    {
        fjoin = gsl_spline_eval_deriv(PSspline, tjoin, PSacc)/(2.0*M_PI);
    }
    ii = (int)(ceil(fjoin*Tspan));
    
    if(thm_diagnostics_enabled)
    {
        printf("tjoin %e fjoin %e sample %d requested_join %e\n",
               tjoin, fjoin, ii, setup[5]);
    }
    
    Nx = jj+(imx-ii)+1;
    
    // first point is special, can't compute derivatives there, so offset derivative by 1
    i = 0;
    t = TS[i];
    f = freq_grid[i];
    phase[i] = gsl_spline_eval (PSspline, t, PSacc) - 2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
    fdot_val = fabs(fdot_grid[i]);
    if(!isfinite(fdot_val) || fdot_val <= 0.0)
    {
        fdot_val = fabs(gsl_spline_eval_deriv2(PSspline, TS[1], PSacc)/(2.0*M_PI));
    }
    x = sqrt(1.0/fdot_val);
    Amp[i] = (x*gsl_spline_eval (ASspline, t, ASacc));
    freq[i] = f;
    
    for (i=1; i <= jj; i++)
    {
        t = TS[i];
        f = freq_grid[i];
        if(!isfinite(f) || f <= 0.0)
        {
            f = gsl_spline_eval_deriv(PSspline, t, PSacc)/(2.0*M_PI);
        }
        phase[i] = gsl_spline_eval (PSspline, t, PSacc) - 2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        fdot_val = fabs(fdot_grid[i]);
        if(!isfinite(fdot_val) || fdot_val <= 0.0)
        {
            fdot_val = fabs(gsl_spline_eval_deriv2(PSspline, t, PSacc)/(2.0*M_PI));
        }
        x = sqrt(1.0/fdot_val);
        Amp[i] = (x*gsl_spline_eval (ASspline, t, ASacc));
        freq[i] = f;
    }
    
    f1 = freq[jj-1];
    f2 = freq[jj];
    p1 = phase[jj-1];
    p2 = phase[jj];
    
    hend = double_vector(Nend);
    
    
   // out = fopen("hend_t.dat","w");
    for (i=0; i< Nend; i++)
    {
        t =  tes + (double)(i)*dte;
        hend[i] = 0.0;
        A = 0.0;
        if(t < TS[Ns-1])
        {
            A = gsl_spline_eval (ASspline, t, ASacc);
            if(t-tes < rise)
            {
                A *= 0.5*(1.0-cos(M_PI*(t-tes)/rise));
            }
	            hend[i] = A*cos(gsl_spline_eval (PSspline, t, PSacc));
	        }
	        if(short_htime != NULL) short_htime[i] = hend[i];
	       // fprintf(out, "%e %e %e\n", t, hend[i], A);
	    }
	   // fclose(out);
	    
	    gsl_fft_real_radix2_transform(hend, 1, Nend);
	    
	    for (i=0; i< Nend; i++) hend[i] *= (2.0*dte);
    
    double pold, pf, pw, fft_phase_shift, fft_join_phase;
    double fft_reference_time, fft_reference_offset, theta, raw_phase, delta;
    
    f = (double)(ii)/Tspan;
    //printf("%e %e %e %e %e\n", f1, f, f2, p1, p2);
    // linear interpolation to get the join phase
    pjoin = p1 + (p2-p1)/(f2-f1)*(f-f1);
    //printf("pjoin %f\n", pjoin);
    /*
     The FFT is taken over h(tes + tau). Before unwrapping, shift the numerical
     phase to a local reference time inside the short FFT block. This removes
     the large linear phase ramp from the atan2 phase and makes 2*pi wrap
     detection much less brittle. The complementary linear phase is restored
     below through fft_phase_shift, leaving the physical Fourier phase
     unchanged.
     */
    fft_reference_time = tes+0.5*Tspan;
    fft_reference_offset = fft_reference_time-tes;
    fft_phase_shift = Tobs-fft_reference_time;
    raw_phase = atan2(hend[Nend-ii],hend[ii]);
    theta = 2.0*M_PI*f*fft_reference_offset;
    pold = atan2(sin(raw_phase+theta), cos(raw_phase+theta));
    fft_join_phase = pold+2.0*M_PI*f*fft_phase_shift;
    pw = 2.0*M_PI*rint((pjoin-fft_join_phase)/(2.0*M_PI));
    k = jj;
   // out = fopen("x.dat","w");
    for (i=ii; i < imx; i++)
    {
        f = (double)(i)/Tspan;
        raw_phase = atan2(hend[Nend-i],hend[i]);
        theta = 2.0*M_PI*f*fft_reference_offset;
        pf = atan2(sin(raw_phase+theta), cos(raw_phase+theta));
        delta = pf-pold;
        if(delta > M_PI) pw -= 2.0*M_PI;
        if(delta < -M_PI) pw += 2.0*M_PI;
        pold = pf;
       // fprintf(out,"%e %e\n", f, pf+pw);
        k++;
        phase[k] = (pf+pw)+2.0*M_PI*f*fft_phase_shift;
        Amp[k] = sqrt(hend[i]*hend[i]+hend[Nend-i]*hend[Nend-i]);
        freq[k] = f;
    }
    //fclose(out);

    /*
     A near-zero in the complex short-FFT spectrum makes the positive
     amplitude / atan2 phase representation ill-conditioned: the physical
     Fourier-domain waveform is smooth, but the phase can jump by roughly pi.
     Use the same convention as extractAP() does for the TDI time-domain
     quadratures: let the amplitude become signed and compensate with a pi
     phase shift across zero-like local minima. Here the zero test uses the
     local shape of the amplitude samples, not an absolute amplitude scale, so
     it remains meaningful when the overall source amplitude changes.
     */
    flip = double_vector(Nx);
    pjump = double_vector(Nx);
    flip[0] = 1.0;
    pjump[0] = 0.0;
    istart = jj+1;
    if(istart < 1) istart = 1;
    i = 1;
    while(i < Nx-1)
    {
        flip[i] = flip[i-1];
        pjump[i] = pjump[i-1];

        if(i >= istart && (Amp[i] < Amp[i-1]) && (Amp[i] < Amp[i+1]))
        {
            dA1 = Amp[i+1]+Amp[i-1]-2.0*Amp[i];
            dA2 = -Amp[i+1]+Amp[i-1]-2.0*Amp[i];
            dA3 = -Amp[i+1]+Amp[i-1]+2.0*Amp[i];

            if(fabs(dA1) > 0.0 && fabs(dA2/dA1) < 0.1)
            {
                flip[i+1] = -flip[i];
                pjump[i+1] = pjump[i]+M_PI;
                i++;
            }
            else if(fabs(dA1) > 0.0 && fabs(dA3/dA1) < 0.1)
            {
                flip[i] = -flip[i-1];
                pjump[i] = pjump[i-1]+M_PI;
            }
        }

        i++;
    }
    flip[Nx-1] = flip[Nx-2];
    pjump[Nx-1] = pjump[Nx-2];

    for(i=istart; i<Nx; i++)
    {
        Amp[i] *= flip[i];
        phase[i] += pjump[i];
    }
    
    /*
    out = fopen("hend_f.dat","w");
    for (i=0; i < imx; i++)
    {
        f = (double)(i)/Tspan;
        fprintf(out,"%e %e\n", f, sqrt(hend[i]*hend[i]+hend[Nend-i]*hend[Nend-i]));
    }
    fclose(out);
    */
    
    //printf("%d %d\n", k, Nx);
    
    /*
    out = fopen("h_f.dat","w");
    for (i=0; i< Nx; i++)
    {
        
        fprintf(out, "%.15e %.15e %.15e\n", freq[i], phase[i], Amp[i]);
    }
    fclose(out);
    */
    
    free(hend);
    free(flip);
    free(pjump);
    free_double_vector(phase_grid);
    free_double_vector(freq_grid);
    free_double_vector(fdot_grid);
    
    return(Nx);
    
}

/*
 * SPA-only companion to ftran().
 *
 * The ordinary fast path uses the stationary-phase approximation through the
 * inspiral/bend and then appends a short FFT of the merger-ringdown signal for
 * the high-frequency end.  The THM combined-merger path keeps the first piece
 * per folded carrier, but performs the final endpoint FFT once on the summed
 * carrier waveform.  This helper therefore stops at the same join sample used
 * by ftran(), using the same nonuniform finite-difference derivatives and
 * phase convention, but intentionally does not build a carrier-local short FFT.
 */
int ftran_spa_only(double *setup, double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *freq, double *phase, double *Amp)
{
    double t, f, x, fdot_val, A0, D1, psi2, psi3, delta_phase;
    double *phase_grid, *freq_grid, *fdot_grid;
    double *fddot_grid, *A_grid, *A1_grid, *A2_grid;
    int i, jj;

    if(setup == NULL || TS == NULL || ASacc == NULL || ASspline == NULL ||
       PSacc == NULL || PSspline == NULL || freq == NULL ||
       phase == NULL || Amp == NULL || Ns < 2)
    {
        return 0;
    }

    jj = (int)(setup[4]);
    if(jj < 1) jj = 1;
    if(jj > Ns-1) jj = Ns-1;

    phase_grid = double_vector(Ns);
    freq_grid = double_vector(Ns);
    fdot_grid = double_vector(Ns);
    fddot_grid = NULL;
    A_grid = NULL;
    A1_grid = NULL;
    A2_grid = NULL;
    build_nonuniform_phase_derivatives(Ns, TS, PSacc, PSspline,
                                       phase_grid, freq_grid, fdot_grid);

    /*
     * The first post-adiabatic SPA term is imaginary relative to the leading
     * bracket and is therefore primarily a phase correction.  The complete
     * Delta1 also needs Phi'''' and proved too noisy on the adaptive grid.
     * Retain only the empirically stable T1+T2 subset,
     *
     *   D1 = A''/(2 Phi'') - A' Phi'''/(2 Phi''^2),
     *
     * using the same local nonuniform derivatives as the leading SPA track.
     * The signed-amplitude representation below keeps post-TDI response zeros
     * from introducing pi jumps into the phase spline.
     */
    if(thm_spa_t1t2_correction_enabled)
    {
        fddot_grid = double_vector(Ns);
        A_grid = double_vector(Ns);
        A1_grid = double_vector(Ns);
        A2_grid = double_vector(Ns);
        for(i=0; i<Ns; i++)
        {
            A_grid[i] = gsl_spline_eval(ASspline, TS[i], ASacc);
        }
        for(i=0; i<Ns; i++)
        {
            fddot_grid[i] = nonuniform_phase_derivative(i, Ns, TS,
                                                        fdot_grid);
            A1_grid[i] = nonuniform_phase_derivative(i, Ns, TS, A_grid);
        }
        for(i=0; i<Ns; i++)
        {
            A2_grid[i] = nonuniform_phase_derivative(i, Ns, TS, A1_grid);
        }
    }

    for(i=0; i<=jj; i++)
    {
        t = TS[i];
        f = freq_grid[i];
        if(!isfinite(f) || f <= 0.0)
        {
            f = gsl_spline_eval_deriv(PSspline, t, PSacc)/(2.0*M_PI);
        }
        fdot_val = fabs(fdot_grid[i]);
        if(!isfinite(fdot_val) || fdot_val <= 0.0)
        {
            fdot_val = fabs(gsl_spline_eval_deriv2(PSspline, t, PSacc)/(2.0*M_PI));
        }
        if(!isfinite(fdot_val) || fdot_val <= 0.0) fdot_val = 1.0;

        x = sqrt(1.0/fdot_val);
        A0 = thm_spa_t1t2_correction_enabled ? A_grid[i] :
             gsl_spline_eval(ASspline, t, ASacc);
        phase[i] = gsl_spline_eval(PSspline, t, PSacc) -
                   2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        Amp[i] = x*A0;
        if(thm_spa_t1t2_correction_enabled && fdot_grid[i] > 0.0 &&
           isfinite(A0) && A0 != 0.0 && isfinite(fddot_grid[i]) &&
           isfinite(A1_grid[i]) && isfinite(A2_grid[i]))
        {
            psi2 = 2.0*M_PI*fdot_grid[i];
            psi3 = 2.0*M_PI*fddot_grid[i];
            D1 = A2_grid[i]/(2.0*psi2) -
                 A1_grid[i]*psi3/(2.0*psi2*psi2);
            delta_phase = atan(D1/A0);
            if(isfinite(D1) && isfinite(delta_phase) &&
               fabs(delta_phase) <= SPA_T1T2_MAX_PHASE_CORRECTION)
            {
                Amp[i] = copysign(hypot(A0, D1), A0)*x;
                phase[i] += delta_phase;
            }
        }
        freq[i] = f;
    }

    free_double_vector(phase_grid);
    free_double_vector(freq_grid);
    free_double_vector(fdot_grid);
    free_double_vector(fddot_grid);
    free_double_vector(A_grid);
    free_double_vector(A1_grid);
    free_double_vector(A2_grid);

    return jj+1;
}

/*
 * Full-SPA diagnostic sampler.
 *
 * The production ftran() intentionally switches to a short FFT before the SPA
 * becomes awkward.  For debugging we sometimes want to see what the SPA itself
 * is trying to do all the way through the available post-TDI time samples.
 * This routine uses the same local nonuniform finite-difference derivatives as
 * ftran(), then writes one stationary-point estimate per time sample.  When
 * df/dt <= 0 the single-valued chirp SPA has formally failed; we still report
 * the current code's absolute-fdot amplitude convention so the blow-up/turnover
 * is visible in plots, but valid_single_branch is set to zero.
 */
int build_full_spa_samples(double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *freq, double *phase, double *Amp, double *tstat, double *fdot, int *valid_single_branch)
{
    double *phase_grid, *freq_grid, *fdot_grid;
    double t, f, fdot_val, fdot_abs, A;
    int i;

    if(TS == NULL || ASacc == NULL || ASspline == NULL ||
       PSacc == NULL || PSspline == NULL || freq == NULL ||
       phase == NULL || Amp == NULL || tstat == NULL || fdot == NULL ||
       valid_single_branch == NULL || Ns < 2)
    {
        return 0;
    }

    phase_grid = double_vector(Ns);
    freq_grid = double_vector(Ns);
    fdot_grid = double_vector(Ns);
    build_nonuniform_phase_derivatives(Ns, TS, PSacc, PSspline,
                                       phase_grid, freq_grid, fdot_grid);

    for(i=0; i<Ns; i++)
    {
        t = TS[i];
        f = freq_grid[i];
        if(!isfinite(f) || f <= 0.0)
        {
            f = gsl_spline_eval_deriv(PSspline, t, PSacc)/(2.0*M_PI);
        }

        fdot_val = fdot_grid[i];
        if(!isfinite(fdot_val) || fdot_val == 0.0)
        {
            fdot_val = gsl_spline_eval_deriv2(PSspline, t, PSacc)/(2.0*M_PI);
        }
        fdot_abs = fabs(fdot_val);

        tstat[i] = t;
        freq[i] = f;
        fdot[i] = fdot_val;
        valid_single_branch[i] = (isfinite(f) && f > 0.0 &&
                                  isfinite(fdot_val) && fdot_val > 0.0);

        if(!isfinite(f) || f <= 0.0 || !isfinite(fdot_abs) || fdot_abs <= 0.0)
        {
            phase[i] = NAN;
            Amp[i] = NAN;
            valid_single_branch[i] = 0;
            continue;
        }

        A = gsl_spline_eval(ASspline, t, ASacc);
        phase[i] = gsl_spline_eval(PSspline, t, PSacc) -
                   2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        Amp[i] = A*sqrt(1.0/fdot_abs);
    }

    free_double_vector(phase_grid);
    free_double_vector(freq_grid);
    free_double_vector(fdot_grid);

    return Ns;
}

/*
 * Same leading-order SPA diagnostic as build_full_spa_samples(), but with the
 * stationary frequency track supplied by the caller.  This is useful for tests
 * where the total phase is still the physical post-TDI phase, but the first
 * derivative is assembled from a better-conditioned decomposition.
 */
int build_full_spa_samples_from_tracks(double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, const double *freq_grid, const double *fdot_grid, double *freq, double *phase, double *Amp, double *tstat, double *fdot, int *valid_single_branch)
{
    double t, f, fdot_val, fdot_abs, A;
    int i;

    if(TS == NULL || ASacc == NULL || ASspline == NULL ||
       PSacc == NULL || PSspline == NULL || freq_grid == NULL ||
       fdot_grid == NULL || freq == NULL || phase == NULL ||
       Amp == NULL || tstat == NULL || fdot == NULL ||
       valid_single_branch == NULL || Ns < 2)
    {
        return 0;
    }

    for(i=0; i<Ns; i++)
    {
        t = TS[i];
        f = freq_grid[i];
        fdot_val = fdot_grid[i];
        fdot_abs = fabs(fdot_val);

        tstat[i] = t;
        freq[i] = f;
        fdot[i] = fdot_val;
        valid_single_branch[i] = (isfinite(f) && f > 0.0 &&
                                  isfinite(fdot_val) && fdot_val > 0.0);

        if(!isfinite(f) || f <= 0.0 || !isfinite(fdot_abs) || fdot_abs <= 0.0)
        {
            phase[i] = NAN;
            Amp[i] = NAN;
            valid_single_branch[i] = 0;
            continue;
        }

        A = gsl_spline_eval(ASspline, t, ASacc);
        phase[i] = gsl_spline_eval(PSspline, t, PSacc) -
                   2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        Amp[i] = A*sqrt(1.0/fdot_abs);
    }

    return Ns;
}

/*
 * First post-adiabatic SPA correction diagnostic.
 *
 * The leading SPA keeps only A(t_f).  The next term expands both the amplitude
 * and phase around the stationary point:
 *
 *   h(f) = exp(i[psi0+pi/4]) / sqrt(fdot)
 *          [ A0 + i Delta1 + ... ],
 *
 * with psi2 = Phi'' = 2*pi*fdot, psi3 = Phi''', psi4 = Phi'''', and
 *
 *   Delta1 = A2/(2 psi2)
 *          - A1 psi3/(2 psi2^2)
 *          - A0 psi4/(8 psi2^2)
 *          + 5 A0 psi3^2/(24 psi2^3).
 *
 * correction_mask selects cumulative/individual experiments:
 *   bit 0: A2/(2 psi2)
 *   bit 1: -A1 psi3/(2 psi2^2)
 *   bit 2: -A0 psi4/(8 psi2^2)
 *   bit 3: 5 A0 psi3^2/(24 psi2^3)
 * A non-positive mask uses all four terms.  This lets us test whether the
 * derivative hierarchy is useful before promoting any correction to ftran().
 *
 * The stable T1+T2 subset is used by ftran_spa_only() when enabled.  This
 * full-mask routine remains diagnostic because T3+T4 require still higher
 * derivatives, whose noise can overwhelm the Fourier-band improvement.
 */
int build_first_order_spa_samples(double *TS, int Ns, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double correction_sign, int correction_mask, double *freq, double *phase, double *Amp, double *tstat, double *fdot, double *delta1, double *delta1_over_A, int *valid_single_branch)
{
    double *phase_grid, *freq_grid, *fdot_grid, *fddot_grid, *f3_grid;
    double *A_grid, *A1_grid, *A2_grid;
    double t, f, fdot_val, psi2, psi3, psi4, A0, A1, A2, D1, phase0;
    int i;

    if(TS == NULL || ASacc == NULL || ASspline == NULL ||
       PSacc == NULL || PSspline == NULL || freq == NULL ||
       phase == NULL || Amp == NULL || tstat == NULL || fdot == NULL ||
       delta1 == NULL || delta1_over_A == NULL ||
       valid_single_branch == NULL || Ns < 4)
    {
        return 0;
    }

    if(correction_sign == 0.0) correction_sign = 1.0;
    if(correction_mask <= 0) correction_mask = 15;

    phase_grid = double_vector(Ns);
    freq_grid = double_vector(Ns);
    fdot_grid = double_vector(Ns);
    fddot_grid = double_vector(Ns);
    f3_grid = double_vector(Ns);
    A_grid = double_vector(Ns);
    A1_grid = double_vector(Ns);
    A2_grid = double_vector(Ns);

    build_nonuniform_phase_derivatives(Ns, TS, PSacc, PSspline,
                                       phase_grid, freq_grid, fdot_grid);
    for(i=0; i<Ns; i++)
    {
        A_grid[i] = gsl_spline_eval(ASspline, TS[i], ASacc);
    }
    for(i=0; i<Ns; i++)
    {
        fddot_grid[i] = nonuniform_phase_derivative(i, Ns, TS, fdot_grid);
        A1_grid[i] = nonuniform_phase_derivative(i, Ns, TS, A_grid);
    }
    for(i=0; i<Ns; i++)
    {
        f3_grid[i] = nonuniform_phase_derivative(i, Ns, TS, fddot_grid);
        A2_grid[i] = nonuniform_phase_derivative(i, Ns, TS, A1_grid);
    }

    for(i=0; i<Ns; i++)
    {
        t = TS[i];
        f = freq_grid[i];
        fdot_val = fdot_grid[i];
        A0 = A_grid[i];
        A1 = A1_grid[i];
        A2 = A2_grid[i];
        psi2 = 2.0*M_PI*fdot_val;
        psi3 = 2.0*M_PI*fddot_grid[i];
        psi4 = 2.0*M_PI*f3_grid[i];

        tstat[i] = t;
        freq[i] = f;
        fdot[i] = fdot_val;
        delta1[i] = NAN;
        delta1_over_A[i] = NAN;
        valid_single_branch[i] = 0;

        if(!isfinite(f) || f <= 0.0 ||
           !isfinite(fdot_val) || fdot_val <= 0.0 ||
           !isfinite(A0) || !isfinite(psi2) || psi2 == 0.0)
        {
            phase[i] = NAN;
            Amp[i] = NAN;
            continue;
        }

        D1 = 0.0;
        if(correction_mask & 1)
        {
            if(!isfinite(A2))
            {
                phase[i] = NAN;
                Amp[i] = NAN;
                continue;
            }
            D1 += A2/(2.0*psi2);
        }
        if(correction_mask & 2)
        {
            if(!isfinite(A1) || !isfinite(psi3))
            {
                phase[i] = NAN;
                Amp[i] = NAN;
                continue;
            }
            D1 += - A1*psi3/(2.0*psi2*psi2);
        }
        if(correction_mask & 4)
        {
            if(!isfinite(psi4))
            {
                phase[i] = NAN;
                Amp[i] = NAN;
                continue;
            }
            D1 += - A0*psi4/(8.0*psi2*psi2);
        }
        if(correction_mask & 8)
        {
            if(!isfinite(psi3))
            {
                phase[i] = NAN;
                Amp[i] = NAN;
                continue;
            }
            D1 += 5.0*A0*psi3*psi3/(24.0*psi2*psi2*psi2);
        }
        D1 *= correction_sign;
        if(!isfinite(D1))
        {
            phase[i] = NAN;
            Amp[i] = NAN;
            continue;
        }

        phase0 = gsl_spline_eval(PSspline, t, PSacc) -
                 2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        Amp[i] = sqrt(A0*A0+D1*D1)*sqrt(1.0/fdot_val);
        phase[i] = phase0+atan2(D1, A0);
        delta1[i] = D1;
        if(A0 != 0.0) delta1_over_A[i] = D1/A0;
        valid_single_branch[i] = 1;
    }

    free_double_vector(phase_grid);
    free_double_vector(freq_grid);
    free_double_vector(fdot_grid);
    free_double_vector(fddot_grid);
    free_double_vector(f3_grid);
    free_double_vector(A_grid);
    free_double_vector(A1_grid);
    free_double_vector(A2_grid);

    return Ns;
}

/*
 * Convert one tapered endpoint time segment into signed A(f), phi(f) samples.
 *
 * This is the FFT half of ftran(), separated so the THM path can sum all
 * folded carriers in the time domain and transform the merger-ringdown once.
 * The phase is first referenced to the center of the short block to make
 * unwrapping local, then shifted back into the same Tobs convention used by
 * WDMtrack.  Near local spectral zeros the amplitude is allowed to become
 * signed, matching the extractAP()/ftran convention that keeps the phase
 * smooth enough for interpolation.
 */
int ftran_endpoint_fft(double *setup, double Tobs, double *short_htime, double f_start, double *freq, double *phase, double *Amp)
{
    double dte, Tspan, tes, fft_reference_time, fft_reference_offset;
    double fft_phase_shift, f, raw_phase, theta, pf, pold, delta, pw;
    double dA1, dA2, dA3;
    double *hend, *flip, *pjump;
    int Nend, imx, ii, i, k;

    if(setup == NULL || short_htime == NULL || freq == NULL ||
       phase == NULL || Amp == NULL)
    {
        return 0;
    }

    dte = setup[0];
    Nend = (int)(setup[1]);
    if(Nend < 8 || dte <= 0.0) return 0;

    Tspan = (double)Nend*dte;
    tes = setup[2];
    imx = (int)(setup[6]);
    if(imx > Nend/2) imx = Nend/2;
    ii = (int)ceil(f_start*Tspan);
    if(ii < 1) ii = 1;
    if(ii >= imx) return 0;

    hend = double_vector(Nend);
    for(i=0; i<Nend; i++) hend[i] = short_htime[i];

    gsl_fft_real_radix2_transform(hend, 1, Nend);
    for(i=0; i<Nend; i++) hend[i] *= (2.0*dte);

    fft_reference_time = tes+0.5*Tspan;
    fft_reference_offset = fft_reference_time-tes;
    fft_phase_shift = Tobs-fft_reference_time;

    f = (double)ii/Tspan;
    raw_phase = atan2(hend[Nend-ii], hend[ii]);
    theta = 2.0*M_PI*f*fft_reference_offset;
    pold = atan2(sin(raw_phase+theta), cos(raw_phase+theta));
    pw = 0.0;
    k = 0;

    for(i=ii; i<imx; i++)
    {
        f = (double)i/Tspan;
        raw_phase = atan2(hend[Nend-i], hend[i]);
        theta = 2.0*M_PI*f*fft_reference_offset;
        pf = atan2(sin(raw_phase+theta), cos(raw_phase+theta));
        delta = pf-pold;
        if(delta > M_PI) pw -= 2.0*M_PI;
        if(delta < -M_PI) pw += 2.0*M_PI;
        pold = pf;

        freq[k] = f;
        phase[k] = (pf+pw)+2.0*M_PI*f*fft_phase_shift;
        Amp[k] = sqrt(hend[i]*hend[i]+hend[Nend-i]*hend[Nend-i]);
        k++;
    }

    if(k > 2)
    {
        flip = double_vector(k);
        pjump = double_vector(k);
        flip[0] = 1.0;
        pjump[0] = 0.0;
        i = 1;
        while(i < k-1)
        {
            flip[i] = flip[i-1];
            pjump[i] = pjump[i-1];

            if((Amp[i] < Amp[i-1]) && (Amp[i] < Amp[i+1]))
            {
                dA1 = Amp[i+1]+Amp[i-1]-2.0*Amp[i];
                dA2 = -Amp[i+1]+Amp[i-1]-2.0*Amp[i];
                dA3 = -Amp[i+1]+Amp[i-1]+2.0*Amp[i];

                if(fabs(dA1) > 0.0 && fabs(dA2/dA1) < 0.1)
                {
                    flip[i+1] = -flip[i];
                    pjump[i+1] = pjump[i]+M_PI;
                    i++;
                }
                else if(fabs(dA1) > 0.0 && fabs(dA3/dA1) < 0.1)
                {
                    flip[i] = -flip[i-1];
                    pjump[i] = pjump[i-1]+M_PI;
                }
            }
            i++;
        }
        flip[k-1] = flip[k-2];
        pjump[k-1] = pjump[k-2];

        for(i=1; i<k; i++)
        {
            Amp[i] *= flip[i];
            phase[i] += pjump[i];
        }
        free_double_vector(flip);
        free_double_vector(pjump);
    }

    free_double_vector(hend);

    return k;
}

static void wdm_packet_fft_plan_init(WDMPacketFFTPlan *plan)
{
    if(plan == NULL) return;
    plan->Ntx = 0;
    plan->dfx = 0.0;
    plan->scale = 0.0;
    plan->phif = NULL;
    plan->data = NULL;
    plan->wdmout = NULL;
    plan->DX = NULL;
#if WDM_USE_GSL_COMPLEX_WAVETABLE
    plan->wavetable = NULL;
    plan->workspace = NULL;
#endif
}

static void wdm_packet_fft_plan_free(WDMPacketFFTPlan *plan)
{
    if(plan == NULL) return;
    free(plan->phif);
    free(plan->data);
    free(plan->wdmout);
    free(plan->DX);
#if WDM_USE_GSL_COMPLEX_WAVETABLE
    if(plan->wavetable != NULL) gsl_fft_complex_wavetable_free(plan->wavetable);
    if(plan->workspace != NULL) gsl_fft_complex_workspace_free(plan->workspace);
#endif
    wdm_packet_fft_plan_init(plan);
}

static int wdm_packet_fft_plan_prepare(WDMPacketFFTPlan *plan, int Ntx, struct wdmshape *wdms)
{
    int i;
    double dom, om;

    if(plan == NULL || wdms == NULL || Ntx < 2) return 0;
    if(plan->Ntx == Ntx) return 1;

    wdm_packet_fft_plan_free(plan);

    plan->Ntx = Ntx;
    plan->phif = double_vector(Ntx/2+1);
    plan->data = double_vector(2*Ntx);
    plan->wdmout = double_vector(Ntx);
    plan->DX = double_vector(2*Ntx);
#if WDM_USE_GSL_COMPLEX_WAVETABLE
    plan->wavetable = gsl_fft_complex_wavetable_alloc((size_t)Ntx);
    plan->workspace = gsl_fft_complex_workspace_alloc((size_t)Ntx);
#endif

    if(plan->phif == NULL || plan->data == NULL || plan->wdmout == NULL ||
       plan->DX == NULL
#if WDM_USE_GSL_COMPLEX_WAVETABLE
       || plan->wavetable == NULL || plan->workspace == NULL
#endif
       )
    {
        fprintf(stderr, "allocation failure preparing WDM packet FFT plan for Ntx=%d\n", Ntx);
        wdm_packet_fft_plan_free(plan);
        return 0;
    }

    plan->dfx = 1.0/((double)Ntx*wdms->DT);
    dom = 2.0*M_PI/((double)Ntx*wdms->DT);
    plan->scale = sqrt(8.0*M_PI/15.0)/((double)Ntx*wdms->DT);

    for(i=0; i<=Ntx/2; i++)
    {
        om = (double)i*dom;
        plan->phif[i] = phitilde(om, wdms->insDOM, wdms->A, wdms->B);
    }

    return 1;
}

static double thm_noise_spos(double f)
{
    double x;

    if(f <= 0.0) return HUGE_VAL;
    x = 2.0e-3/f;
    x *= x;
    x *= x;
    return THM_SPOS0*(1.0+x);
}

static double thm_noise_sacc(double f)
{
    double x, y;

    if(f <= 0.0) return HUGE_VAL;
    x = 4.0e-4/f;
    y = f/8.0e-3;
    x *= x;
    y *= y;
    y *= y;
    return THM_SACC0*(1.0+x)*(1.0+y);
}

static double thm_instrument_x_auto_psd(double f)
{
    double fonfs, lc, omf, s2, c2, sps, sac;

    if(f <= 0.0) return HUGE_VAL;
    fonfs = 2.0*M_PI*f*THM_LAV_SECONDS;
    lc = fonfs*fonfs/(4.0*THM_LAM_METERS*THM_LAM_METERS);
    omf = 2.0*M_PI*f;
    s2 = sin(fonfs);
    c2 = cos(fonfs);
    sps = thm_noise_spos(f);
    sac = thm_noise_sacc(f)/(omf*omf*omf*omf);
    return lc*s2*s2*(16.0*sps+32.0*sac*(1.0+c2*c2));
}

static double thm_instrument_prewhiten_psd(double f)
{
    int dm, halfwidth;
    double df, ff, weight, norm, sum, psd, width;

    width = thm_wdm_instrument_prewhiten_smooth_halfwidth_hz;
    if(thm_wdm_instrument_prewhiten_smooth_frac > 0.0)
    {
        double fwidth, floor_width;

        fwidth = thm_wdm_instrument_prewhiten_smooth_frac*f;
        floor_width = thm_wdm_instrument_prewhiten_smooth_floor_hz;
        if(isfinite(fwidth) && fwidth > width) width = fwidth;
        if(isfinite(floor_width) && floor_width > width) width = floor_width;
    }
    if(width > 0.0)
    {
        int q, nquad;

        /*
         * Smooth the pure-instrument ASD on the Fourier side before the WDM
         * packet is formed.  The equal-arm PSD has exact delay zeros, while a
         * finite-window chirping TDI signal only approximately shares those
         * zeros.  A narrow triangular average prevents a single frequency
         * sample from receiving infinite weight.  If the fractional smoothing
         * option is enabled, the half-width grows as max(floor, frac*f), a
         * phenomenological stand-in for time-varying unequal-arm null blurring.
         * The fixed half-width remains a lower bound so old runs are
         * reproduced when frac=0.
         */
        nquad = 65;
        norm = 0.0;
        sum = 0.0;
        for(q=0; q<nquad; q++)
        {
            double u;

            u = -1.0+2.0*((double)q)/((double)(nquad-1));
            ff = f+u*width;
            if(ff <= 0.0) continue;
            weight = 1.0-fabs(u);
            psd = thm_instrument_x_auto_psd(ff);
            if(!isfinite(psd) || psd <= 0.0) continue;
            sum += weight*psd;
            norm += weight;
        }
        if(norm > 0.0) return sum/norm;
    }

    halfwidth = thm_wdm_instrument_prewhiten_average_halfwidth;
    if(halfwidth <= 0) return thm_instrument_x_auto_psd(f);

    df = 1.0/(2.0*dt*(double)Nf);
    norm = 0.0;
    sum = 0.0;
    for(dm=-halfwidth; dm<=halfwidth; dm++)
    {
        ff = f+(double)dm*df;
        if(ff <= 0.0) continue;
        weight = (double)(halfwidth+1-abs(dm));
        psd = thm_instrument_x_auto_psd(ff);
        if(!isfinite(psd) || psd <= 0.0) continue;
        sum += weight*psd;
        norm += weight;
    }
    if(norm <= 0.0) return thm_instrument_x_auto_psd(f);
    return sum/norm;
}

static double thm_instrument_prewhiten_factor(double f)
{
    double psd;

    if(!thm_wdm_instrument_prewhiten_enabled) return 1.0;
    psd = thm_instrument_prewhiten_psd(f);
    if(!isfinite(psd) || psd <= 0.0) return 0.0;
    return 1.0/sqrt(psd);
}

static void wdmtranF_plan(int m, WDMPacketFFTPlan *plan)
{
    int n, i, j, jj, Ntx;

    if(plan == NULL || plan->Ntx < 2) return;
    Ntx = plan->Ntx;

    /*
     * The WDM packet transform is called thousands of times with the same
     * small set of packet lengths.  Keep the complex FFT scratch array, Meyer
     * window, and optional GSL wavetable/workspace in WDMPacketFFTPlan instead
     * of allocating them for every frequency layer.  This is intentionally
     * local to each WDM assembly call, not a static global, so independent
     * waveform evaluations remain thread-safe.
     */
    for(j=-Ntx/2; j<Ntx/2; j++)
    {
        i = j+Ntx/2;

        REAL(plan->DX,i) = 0.0;
        IMAG(plan->DX,i) = 0.0;

        jj = j+Ntx/2;
        if(i > 0 && i < Ntx)
        {
            REAL(plan->DX,i) = plan->data[jj]*plan->phif[abs(j)];
            IMAG(plan->DX,i) = plan->data[2*Ntx-jj]*plan->phif[abs(j)];
        }
    }

#if WDM_USE_GSL_COMPLEX_WAVETABLE
    gsl_fft_complex_backward(plan->DX, 1, (size_t)Ntx,
                             plan->wavetable, plan->workspace);
#else
    gsl_fft_complex_radix2_backward(plan->DX, 1, (size_t)Ntx);
#endif

    for(n=0; n<Ntx; n++)
    {
        if(m%2 == 0)
        {
            if((n+m)%2 == 0)
            {
                plan->wdmout[n] = plan->scale*REAL(plan->DX,n);
            }
            else
            {
                plan->wdmout[n] = plan->scale*IMAG(plan->DX,n);
            }
        }
        else
        {
            if((n+m)%2 == 0)
            {
                plan->wdmout[n] = plan->scale*REAL(plan->DX,n);
            }
            else
            {
                plan->wdmout[n] = -plan->scale*IMAG(plan->DX,n);
            }
        }
    }
}

void WDMtrack(double *wdmwave, int *listn, int *listm, int *Np, int Ns, int N, int *nmid, int *nsize, double tc, double *FF, gsl_interp_accel *AFacc, gsl_spline *AFspline, gsl_interp_accel *PFacc, gsl_spline *PFspline, struct wdmshape *wdms)
    {
        int i, j, jj;
        double AA, PP, f, white;
        int n, m, Ntx;
        int cnt;
        WDMPacketFFTPlan plan;
    
    wdm_packet_fft_plan_init(&plan);
    
    cnt = 0;
    
        
  for(m=1; m< Nf; m++)
    {
       if(nmid[m] > 0)
       {
           
           Ntx = nsize[m];
           n = nmid[m];
           
           if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;
           plan.data[0] = 0.0;
           
           for(j=-Ntx/2; j < Ntx/2; j++)
           {
               f = (double)(j)*plan.dfx+(double)(m)*wdms->DF;
               
               jj = j + Ntx/2;
               
               if(jj > 0)
               {
                   plan.data[jj] = 0.0;
                   plan.data[2*Ntx-jj] = 0.0;
                   
                   if(f > FF[0] && f < FF[Ns-1])
                   {
                       int status_a, status_p;

                       status_a = gsl_spline_eval_e(AFspline, f, AFacc, &AA);
                       status_p = gsl_spline_eval_e(PFspline, f, PFacc, &PP);
                       if(status_a != GSL_SUCCESS || status_p != GSL_SUCCESS)
                       {
                           continue;
                       }
                       
                       PP += 2.0*M_PI*f*((double)(n-Ntx/2)*wdms->DT);
                       white = thm_instrument_prewhiten_factor(f);
                       AA *= white;
                       
                       plan.data[jj] = AA*cos(PP);
                       plan.data[2*Ntx-jj] = AA*sin(PP);
                       
                   }
               }
               
           }
           
           wdmtranF_plan(m, &plan);
           
           for (i = 0; i < Ntx; ++i)
           {
               j = (i+n-Ntx/2);
               if(j > -1 && j < Nt)
               {
                   listn[cnt] = j;
                   listm[cnt] = m;
                   wdmwave[cnt]=plan.wdmout[i];
                   cnt++;
               }
           }
           
         }
       }
    
    *Np = cnt;
    
            wdm_packet_fft_plan_free(&plan);
	        
	}

void WDMtrackAPProductDerivative(double *wdmwave, int *listn, int *listm, int *Np, int Ns, int N, int *nmid, int *nsize, double *FF, gsl_interp_accel *PFacc, gsl_spline *PFspline, gsl_interp_accel *DAFacc, gsl_spline *DAFspline, gsl_interp_accel *ADPFacc, gsl_spline *ADPFspline, struct wdmshape *wdms)
{
    int i, j, jj;
    double P, dA, AdP, theta, cth, sth, f, white;
    int n, m, Ntx;
    int cnt;
    WDMPacketFFTPlan plan;

    wdm_packet_fft_plan_init(&plan);
    cnt = 0;

    for(m=1; m<Nf; m++)
    {
        if(nmid[m] > 0)
        {
            Ntx = nsize[m];
            n = nmid[m];

            if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;
            plan.data[0] = 0.0;

            for(j=-Ntx/2; j<Ntx/2; j++)
            {
                f = (double)(j)*plan.dfx+(double)(m)*wdms->DF;
                jj = j+Ntx/2;

                if(jj > 0)
                {
                    plan.data[jj] = 0.0;
                    plan.data[2*Ntx-jj] = 0.0;

                    if(f > FF[0] && f < FF[Ns-1])
                    {
                        int status_p = gsl_spline_eval_e(PFspline, f, PFacc, &P);
                        int status_da = gsl_spline_eval_e(DAFspline, f, DAFacc, &dA);
                        int status_adp = gsl_spline_eval_e(ADPFspline, f, ADPFacc, &AdP);
                        if(status_p != GSL_SUCCESS || status_da != GSL_SUCCESS ||
                           status_adp != GSL_SUCCESS)
                        {
                            continue;
                        }

                        theta = P+2.0*M_PI*f*((double)(n-Ntx/2)*wdms->DT);
                        cth = cos(theta);
                        sth = sin(theta);
                        white = thm_instrument_prewhiten_factor(f);
                        dA *= white;
                        AdP *= white;

                        plan.data[jj] = dA*cth - AdP*sth;
                        plan.data[2*Ntx-jj] = dA*sth + AdP*cth;
                    }
                }
            }

            wdmtranF_plan(m, &plan);

            for(i=0; i<Ntx; ++i)
            {
                j = i+n-Ntx/2;
                if(j > -1 && j < Nt)
                {
                    listn[cnt] = j;
                    listm[cnt] = m;
                    wdmwave[cnt] = plan.wdmout[i];
                    cnt++;
                }
            }
        }
    }

    *Np = cnt;
    wdm_packet_fft_plan_free(&plan);
}

/*
 Diagnostic companion to WDMtrack.

 WDMtrack samples the SPA amplitude/phase splines to build each local
 frequency packet before calling wdmtranF. WDMtrackFFT builds the same local
 packets from a direct FFT spectrum instead. Everything downstream of the
 spectrum source is deliberately kept the same: active pixels from WDMpixels,
 packet sizes, recentering phase, Meyer window, normalization, and wdmtranF.
 This makes wtranfft.dat a check of the SPA frequency-domain input rather than
 a separate WDM algorithm. When clip_to_spa_band is true, samples outside
 fmin..fmax are zeroed to match the frequency-support guard used by WDMtrack.
 */
void WDMtrackFFT(double *wdmwave, int *listn, int *listm, int *Np, int N, int *nmid, int *nsize, double *hfft, struct wdmshape *wdms, int clip_to_spa_band, double fmin, double fmax)
	    {
        int i, j, jj;
        double re, im, phase, c, s, f, white;
        int n, m, Ntx;
        int cnt;
        WDMPacketFFTPlan plan;
    
    wdm_packet_fft_plan_init(&plan);
    
    cnt = 0;
    
        
  for(m=1; m< Nf; m++)
    {
       if(nmid[m] > 0)
       {
           
           Ntx = nsize[m];
           n = nmid[m];
           
           if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;
           plan.data[0] = 0.0;
           
           for(j=-Ntx/2; j < Ntx/2; j++)
           {
               f = (double)(j)*plan.dfx+(double)(m)*wdms->DF;
               
               jj = j + Ntx/2;
               
               if(jj > 0)
               {
                   plan.data[jj] = 0.0;
                   plan.data[2*Ntx-jj] = 0.0;
                   
                   if(!clip_to_spa_band || (f > fmin && f < fmax))
                   {
                       sample_direct_fft_spectrum(hfft, N, wdms->Tobs, f, &re, &im);
                       white = thm_instrument_prewhiten_factor(f);
                       re *= white;
                       im *= white;
                       phase = 2.0*M_PI*f*((double)(n-Ntx/2)*wdms->DT);
                       c = cos(phase);
                       s = sin(phase);
                       
                       plan.data[jj] = re*c-im*s;
                       plan.data[2*Ntx-jj] = re*s+im*c;
                   }
               }
               
           }
           
           wdmtranF_plan(m, &plan);
           
           for (i = 0; i < Ntx; ++i)
           {
               j = (i+n-Ntx/2);
               if(j > -1 && j < Nt)
               {
                   listn[cnt] = j;
                   listm[cnt] = m;
                   wdmwave[cnt]=plan.wdmout[i];
                   cnt++;
               }
           }
           
         }
       }
    
    *Np = cnt;
    
            wdm_packet_fft_plan_free(&plan);
	        
	}

/*
 * Direct endpoint-FFT WDM replacement.
 *
 * This is the no-interpolation merger path.  The input short_htime is a
 * tapered endpoint waveform segment sampled at setup[0] and starting at
 * setup[2].  For each WDM frequency layer we zero-pad that same segment to
 * the local packet duration Ntx*DT, FFT it once per packet size, and read the
 * exact Fourier bins needed by wdmtranF.  The only phase operation is the
 * deterministic time-origin rotation from the local FFT coordinate
 * h(tes+tau) to the global convention used by WDMtrack.
 */
static void WDMapplyShortFFTThreshold(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, double f_replace_start, double f_replace_stop, double blend_half_width, struct wdmshape *wdms, const char *layers_filename, int *layers_replaced, int *pixels_replaced, int output_mode)
{
    enum
    {
        SHORT_FFT_CACHE_MAX = 16,
        SHORT_FFT_REPLACE = 0,
        SHORT_FFT_ADD = 1,
        SHORT_FFT_BLEND = 2
    };
    int i, j, jj, flag;
    int n, m, Ntx, Nshort, Nblock, Ncopy, ci, cache_count;
    int layer_count, pixel_count;
    double re, im, c, s, f, fcenter, flower, fupper, phase, white;
    double blend_weight, blend_lo, blend_hi;
    double t0, tes, dte;
    double *short_hfft, *short_hfft_cache[SHORT_FFT_CACHE_MAX];
    int nblock_cache[SHORT_FFT_CACHE_MAX];
    WDMPacketFFTPlan plan;
    FILE *out;

    if(layers_replaced != NULL) *layers_replaced = 0;
    if(pixels_replaced != NULL) *pixels_replaced = 0;
    if(wdm == NULL || nmid == NULL || nsize == NULL || short_htime == NULL ||
       setup == NULL || wdms == NULL)
    {
        return;
    }

    dte = setup[0];
    Nshort = (int)setup[1];
    tes = setup[2];
    if(dte <= 0.0 || Nshort < 2) return;
    if(!isfinite(f_replace_stop) || f_replace_stop <= 0.0)
    {
        f_replace_stop = 0.5/dte;
    }
    if(output_mode == SHORT_FFT_BLEND)
    {
        if(!isfinite(blend_half_width) || blend_half_width <= 0.0)
        {
            blend_half_width = wdms->FB;
        }
    }

    wdm_packet_fft_plan_init(&plan);
    short_hfft = NULL;
    cache_count = 0;
    for(i=0; i<SHORT_FFT_CACHE_MAX; i++)
    {
        short_hfft_cache[i] = NULL;
        nblock_cache[i] = 0;
    }
    layer_count = 0;
    pixel_count = 0;

    out = NULL;
    if(layers_filename != NULL)
    {
        out = fopen(layers_filename, "w");
    }
    if(out != NULL)
    {
        fprintf(out, "# m f_center f_lower_support f_upper_support nmid nsize nblock replaced\n");
    }

    for(m=1; m<Nf; m++)
    {
        int replace_layer;

        if(nmid[m] < 0 || nsize[m] <= 0) continue;

        fcenter = (double)m*wdms->DF;
        flower = fcenter-wdms->FB;
        fupper = fcenter+wdms->FB;
        /*
         * Replacement and additive endpoint pieces have different support
         * logic.  A replacement must be conservative: only overwrite layers
         * whose full Meyer support lies inside the direct-FFT band, otherwise
         * valid SPA content near the boundary is lost.  An additive endpoint or
         * split-FFT piece is different.  It represents a windowed time-domain
         * contribution, so any layer whose Meyer support overlaps the window's
         * frequency support should receive that contribution.  The packet mask
         * in nmid/nsize still controls the time support.  The blend mode is a
         * coefficient-space diagnostic: layers below the boundary keep the SPA
         * value, layers above it use the endpoint FFT, and a narrow transition
         * band uses a smooth frequency weight to avoid an all-or-nothing layer
         * flip as source parameters move.
         */
        blend_weight = 1.0;
        if(output_mode == SHORT_FFT_ADD)
        {
            replace_layer = (fupper >= f_replace_start && flower <= f_replace_stop);
        }
        else if(output_mode == SHORT_FFT_BLEND)
        {
            blend_lo = f_replace_start-blend_half_width;
            blend_hi = f_replace_start+blend_half_width;
            if(blend_lo < 0.0) blend_lo = 0.0;
            replace_layer = (fupper >= blend_lo && flower <= f_replace_stop);
            if(fcenter <= blend_lo)
            {
                blend_weight = 0.0;
            }
            else if(fcenter >= blend_hi)
            {
                blend_weight = 1.0;
            }
            else
            {
                blend_weight = split_smooth_step(fcenter, blend_lo, blend_hi);
            }
            if(blend_weight <= 0.0) replace_layer = 0;
        }
        else
        {
            replace_layer = (flower >= f_replace_start && fupper <= f_replace_stop);
        }

        Ntx = nsize[m];
        Nblock = (int)llround(((double)Ntx)*wdms->DT/dte);
        if(Nblock < 2 || (Nblock & (Nblock-1)) != 0) replace_layer = 0;
        if(Nblock < Nshort) replace_layer = 0;

        if(out != NULL)
        {
            fprintf(out, "%d %.15e %.15e %.15e %d %d %d %d\n",
                    m, fcenter, flower, fupper, nmid[m], nsize[m],
                    Nblock, replace_layer);
        }

        if(!replace_layer) continue;

        flag = (Ntx != plan.Ntx);
        n = nmid[m];

        if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;

        short_hfft = NULL;
        for(ci=0; ci<cache_count; ci++)
        {
            if(nblock_cache[ci] == Nblock)
            {
                short_hfft = short_hfft_cache[ci];
                break;
            }
        }

        if(short_hfft == NULL)
        {
            short_hfft = double_vector(Nblock);
            if(short_hfft == NULL)
            {
                wdm_packet_fft_plan_free(&plan);
                continue;
            }

            for(i=0; i<Nblock; i++) short_hfft[i] = 0.0;
            Ncopy = Nshort;
            if(Ncopy > Nblock) Ncopy = Nblock;
            for(i=0; i<Ncopy; i++) short_hfft[i] = short_htime[i];

            gsl_fft_real_radix2_transform(short_hfft, 1, Nblock);
            for(i=0; i<Nblock; i++) short_hfft[i] *= (2.0*dte);

            if(cache_count < SHORT_FFT_CACHE_MAX)
            {
                short_hfft_cache[cache_count] = short_hfft;
                nblock_cache[cache_count] = Nblock;
                cache_count++;
            }
        }
        (void)flag;

        t0 = ((double)(n-Ntx/2))*wdms->DT;
        plan.data[0] = 0.0;

        for(j=-Ntx/2; j<Ntx/2; j++)
        {
            f = (double)j/((double)Ntx*wdms->DT)+fcenter;
            jj = j+Ntx/2;

            if(jj > 0)
            {
                plan.data[jj] = 0.0;
                plan.data[2*Ntx-jj] = 0.0;

                sample_short_window_spectrum(short_hfft, Nblock, dte, f, &re, &im);
                white = thm_instrument_prewhiten_factor(f);
                re *= white;
                im *= white;
                if(re != 0.0 || im != 0.0)
                {
                    phase = 2.0*M_PI*f*(wdms->Tobs-tes+t0);
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] = re*c-im*s;
                    plan.data[2*Ntx-jj] = re*s+im*c;
                }
            }
        }

        wdmtranF_plan(m, &plan);

        for(i=0; i<Ntx; i++)
        {
            j = i+n-Ntx/2;
            if(j > -1 && j < Nt)
            {
                if(output_mode == SHORT_FFT_ADD)
                {
                    wdm[j][m] += plan.wdmout[i];
                }
                else if(output_mode == SHORT_FFT_BLEND)
                {
                    wdm[j][m] = (1.0-blend_weight)*wdm[j][m]+
                                blend_weight*plan.wdmout[i];
                }
                else
                {
                    wdm[j][m] = plan.wdmout[i];
                }
                pixel_count++;
            }
        }

        layer_count++;
    }

    if(out != NULL) fclose(out);
    wdm_packet_fft_plan_free(&plan);
    for(ci=0; ci<cache_count; ci++)
    {
        free(short_hfft_cache[ci]);
    }

    if(layers_replaced != NULL) *layers_replaced = layer_count;
    if(pixels_replaced != NULL) *pixels_replaced = pixel_count;
}

void WDMreplaceWithShortFFTThreshold(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, double f_replace_start, double f_replace_stop, struct wdmshape *wdms, const char *layers_filename, int *layers_replaced, int *pixels_replaced)
{
    WDMapplyShortFFTThreshold(wdm, nmid, nsize, short_htime, setup,
                              f_replace_start, f_replace_stop, 0.0,
                              wdms, layers_filename, layers_replaced,
                              pixels_replaced, 0);
}

void WDMaddWithShortFFTThreshold(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, double f_replace_start, double f_replace_stop, struct wdmshape *wdms, const char *layers_filename, int *layers_replaced, int *pixels_replaced)
{
    WDMapplyShortFFTThreshold(wdm, nmid, nsize, short_htime, setup,
                              f_replace_start, f_replace_stop, 0.0,
                              wdms, layers_filename, layers_replaced,
                              pixels_replaced, 1);
}

void WDMblendWithShortFFTThreshold(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, double f_replace_start, double f_replace_stop, double blend_half_width, struct wdmshape *wdms, const char *layers_filename, int *layers_replaced, int *pixels_replaced)
{
    WDMapplyShortFFTThreshold(wdm, nmid, nsize, short_htime, setup,
                              f_replace_start, f_replace_stop,
                              blend_half_width, wdms, layers_filename,
                              layers_replaced, pixels_replaced, 2);
}

/*
 Diagnostic hybrid merger path.

 The production fast path converts the short merger FFT into A(f), phi(f),
 splines those quantities, and then lets WDMtrack sample the splines. This
 diagnostic skips that intermediate representation for merger/ringdown layers:
 it zero-pads the tapered short time segment to the local WDM packet duration
 Ntx*DT, FFTs that padded segment, and reads exact FFT bins at the local packet
 frequencies. No complex-spectrum interpolation is used. The resulting samples
 are sent to the same wdmtranF routine used by WDMtrack.

 The input wdm matrix should already contain the standard SPA/short-FFT-spline
 result. This routine overwrites only layers whose full Meyer support begins
 after both the short-FFT roll-on has flattened and the production SPA/FFT join
 has been reached. The layer test is local in frequency: the lower support edge
 m*DF-FB is mapped to time using t(f), and the layer is accepted only if that
 time is later than max(tes+rise+margin, tjoin). Layers touching the taper or
 the pre-join SPA region are deliberately left on the standard path.
 */
void WDMreplaceMergerShortWindow(double **wdm, int *nmid, int *nsize, double *short_htime, double *setup, int Ns, double *TF, double *FF, struct wdmshape *wdms, int *layers_replaced, int *pixels_replaced)
{
    int i, j, jj, flag;
    int n, m, Ntx, Nshort, Nblock, Ncopy;
    int layer_count, pixel_count, replace_layer;
    double re, im, c, s, f, fcenter, flower, fupper, phase, white;
    double t0, tedge, taper_flat_time, join_time, direct_start_time;
    double tes, rise, dte;
    double *short_hfft;
    WDMPacketFFTPlan plan;
    FILE *out;
    
    gsl_interp_accel *TFacc;
    gsl_spline *TFspline;
    
    dte = setup[0];
    Nshort = (int)(setup[1]);
    tes = setup[2];
    rise = setup[3];
    taper_flat_time = tes+rise+SHORTFFT_MERGER_TAPER_MARGIN_SECONDS;
    join_time = TF[(int)(setup[4])];
    direct_start_time = taper_flat_time;
    if(direct_start_time < join_time) direct_start_time = join_time;
    
    TFacc = gsl_interp_accel_alloc();
    TFspline = gsl_spline_alloc(gsl_interp_cspline, Ns);
    gsl_spline_init(TFspline, FF, TF, Ns);
    
    wdm_packet_fft_plan_init(&plan);
    short_hfft = NULL;
    layer_count = 0;
    pixel_count = 0;
    
    out = fopen("merger_direct_layers.dat","w");
    fprintf(out, "# m f_center f_lower_support f_upper_support t_lower_support taper_flat_time join_time direct_start_time nmid nsize nblock replaced\n");
    
    for(m=1; m<Nf; m++)
    {
        if(nmid[m] <= 0) continue;
        
        fcenter = (double)(m)*wdms->DF;
        flower = fcenter-wdms->FB;
        fupper = fcenter+wdms->FB;
        tedge = NAN;
        replace_layer = 0;
        
        Ntx = nsize[m];
        Nblock = (int)llround(((double)Ntx)*wdms->DT/dte);
        
        if(flower > FF[0] && flower < FF[Ns-1])
        {
            tedge = gsl_spline_eval(TFspline, flower, TFacc);
            if(tedge > direct_start_time && fupper < 0.5/dte) replace_layer = 1;
        }
        if(Nblock < 2 || (Nblock & (Nblock-1)) != 0) replace_layer = 0;
        
        fprintf(out, "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %d %d %d %d\n",
                m, fcenter, flower, fupper, tedge, taper_flat_time, join_time, direct_start_time,
                nmid[m], nsize[m], Nblock, replace_layer);
        
        if(!replace_layer) continue;
        
        flag = (Ntx != plan.Ntx);
        n = nmid[m];

        if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;

        if(flag == 1)
        {
            free(short_hfft);

            short_hfft = double_vector(Nblock);
            if(short_hfft == NULL)
            {
                wdm_packet_fft_plan_free(&plan);
                continue;
            }

            for(i=0; i<Nblock; i++) short_hfft[i] = 0.0;
            Ncopy = Nshort;
            if(Ncopy > Nblock) Ncopy = Nblock;
            for(i=0; i<Ncopy; i++) short_hfft[i] = short_htime[i];
            
            gsl_fft_real_radix2_transform(short_hfft, 1, Nblock);
            for(i=0; i<Nblock; i++) short_hfft[i] *= (2.0*dte);
        }
        
        t0 = ((double)(n-Ntx/2))*wdms->DT;
        plan.data[0] = 0.0;
        
        for(j=-Ntx/2; j < Ntx/2; j++)
        {
            f = (double)(j)*plan.dfx+fcenter;
            jj = j+Ntx/2;
            
            if(jj > 0)
            {
                plan.data[jj] = 0.0;
                plan.data[2*Ntx-jj] = 0.0;
                
                sample_short_window_spectrum(short_hfft, Nblock, dte, f, &re, &im);
                white = thm_instrument_prewhiten_factor(f);
                re *= white;
                im *= white;
                if(re != 0.0 || im != 0.0)
                {
                    /*
                     The zero-padded FFT is taken over h(tes+tau), the same
                     tapered segment whose short FFT is used by ftran's A/phi
                     path. Its frequency spacing is 1/(Ntx*DT), so the WDM
                     packet frequencies are exact FFT bins. Convert the result
                     to the SPA phase convention with
                     exp[i 2*pi*f*(Tobs-tes)], then apply WDMtrack's local
                     packet recentering exp[i 2*pi*f*t0].
                     */
                    phase = 2.0*M_PI*f*(wdms->Tobs-tes+t0);
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] = re*c-im*s;
                    plan.data[2*Ntx-jj] = re*s+im*c;
                }
            }
        }
        
        wdmtranF_plan(m, &plan);
        
        for(i=0; i<Ntx; i++)
        {
            j = i+n-Ntx/2;
            if(j > -1 && j < Nt)
            {
                wdm[j][m] = plan.wdmout[i];
                pixel_count++;
            }
        }
        
        layer_count++;
    }
    
    fclose(out);
    gsl_spline_free(TFspline);
    gsl_interp_accel_free(TFacc);
    
    wdm_packet_fft_plan_free(&plan);
    free(short_hfft);
    
    *layers_replaced = layer_count;
    *pixels_replaced = pixel_count;
}

/*
 Experimental no-SPA split FFT path.

 The waveform is split in time using a smooth partition of unity:

   h_low(t)  = [1-s_bend(t)] h(t)
   h_bend(t) = s_bend(t) [1-s_high(t)] h(t)
   h_high(t) = s_high(t) h(t)

 The low part is smooth and restricted to early inspiral frequencies, so it is
 transformed once on a coarse full-observation grid with dt_low=128*dt. The
 bend is transformed in a conservative 64-DT block at R=64, and the final
 vertical merger/ringdown track is transformed in a 16-DT full-rate block. The
 decimations preserve the same WDM DT/DF grid. A block contributes to a packet
 only when that block duration contains the packet frequency grid exactly; this
 avoids interpolation in the diagnostic while keeping the number of FFT setups
 small.
 */
void WDMtrackSplitFFT(double **wdm, int *nmid, int *nsize, int N, double tmax, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, struct wdmshape *wdms, int *layers_used, int *pixels_used)
{
    int i, j, jj, m, n, outn;
    int Ntx, Nblock, Nlow, Nbend, Nhigh;
    int Rlow, Rbend, Kbend, Khigh;
    int use_low, use_bend, use_high;
    int layer_count, pixel_count;
    double dtlow, dtbend, fcenter, f, t0;
    double bend_start, bend_end, high_start, high_end, high_taper_end, low_end;
    double packet_start, packet_end;
    double re_low, im_low, re_bend, im_bend, re_high, im_high;
    double phase, c, s, white;
    double low_fft_time, bend_fft_time, high_fft_time, packet_time;
    double *low_hfft, *bend_hfft, *high_hfft;
    WDMPacketFFTPlan plan;
    clock_t timer_start, timer_end;
    FILE *out;
    
    Rlow = SPLIT_FFT_LOW_DECIMATION;
    Rbend = SPLIT_FFT_BEND_DECIMATION;
    Kbend = SPLIT_FFT_BEND_NTILES;
    Khigh = SPLIT_FFT_HIGH_NTILES;
    Nlow = N/Rlow;
    Nbend = Kbend*(Nf/Rbend);
    Nhigh = Khigh*Nf;
    dtlow = dt*(double)Rlow;
    dtbend = dt*(double)Rbend;
    
    for(i=0; i<Nt; i++)
    {
        for(j=0; j<=Nf; j++) wdm[i][j] = 0.0;
    }
    
    *layers_used = 0;
    *pixels_used = 0;
    
    if(Rlow < 1 || N%Rlow != 0 || Nf%Rlow != 0 || (Nlow & (Nlow-1)) != 0)
    {
        fprintf(stderr, "Error: SPLIT_FFT_LOW_DECIMATION=%d must divide N and Nf and leave a power-of-two low-rate FFT length.\n", Rlow);
        return;
    }
    
    if(Rbend < 1 || Nf%Rbend != 0 || Nbend < 2 || (Nbend & (Nbend-1)) != 0)
    {
        fprintf(stderr, "Error: SPLIT_FFT_BEND_DECIMATION=%d and SPLIT_FFT_BEND_NTILES=%d must define a power-of-two bend FFT length.\n", Rbend, Kbend);
        return;
    }
    
    if(Khigh < 2 || Nhigh < 2 || (Nhigh & (Nhigh-1)) != 0)
    {
        fprintf(stderr, "Error: SPLIT_FFT_HIGH_NTILES=%d must define a power-of-two high-rate FFT length.\n", Khigh);
        return;
    }
    
    if(SPLIT_FFT_ROLL_SECONDS <= 0.0 || SPLIT_FFT_ROLL_SECONDS >= (double)Khigh*wdms->DT || 2.0*SPLIT_FFT_ROLL_SECONDS >= (double)Kbend*wdms->DT)
    {
        fprintf(stderr, "Error: SPLIT_FFT_ROLL_SECONDS=%e is incompatible with the bend/high block durations.\n", SPLIT_FFT_ROLL_SECONDS);
        return;
    }
    
    /*
     Anchor the short high-rate block to the WDM time grid and put its upper
     edge just after the generated ringdown tail. Its roll-on begins one block
     length earlier; the intermediate bend block extends through the end of that
     roll-on, so the bend contribution is already tapered away before the track
     reaches the bend block's R=64 Nyquist.
     */
    high_end = wdms->DT*ceil(tmax/wdms->DT);
    high_start = high_end-(double)Nhigh*dt;
    high_taper_end = high_start+SPLIT_FFT_ROLL_SECONDS;
    bend_end = high_taper_end;
    bend_start = bend_end-(double)Nbend*dtbend;
    low_end = bend_start+SPLIT_FFT_ROLL_SECONDS;
    
    if(high_start < 0.0 || bend_start < 0.0)
    {
        fprintf(stderr, "Error: split FFT blocks would start before the observation: bend_start=%e high_start=%e.\n", bend_start, high_start);
        return;
    }
    
    low_hfft = double_vector(Nlow);
    timer_start = clock();
    build_split_low_fft(low_hfft, Nlow, dtlow, tmax, bend_start, high_start, SPLIT_FFT_ROLL_SECONDS, ASacc, ASspline, PSacc, PSspline);
    timer_end = clock();
    low_fft_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;
    
    /*
     The bend and high blocks are local FFTs of the middle and final pieces of
     the same time-domain partition. The high block is intentionally only 16 DT
     wide and ends just after the generated ringdown tail; the bend block keeps
     the transition below its R=64 Nyquist until the high taper has flattened.
     */
    bend_hfft = double_vector(Nbend);
    timer_start = clock();
    build_split_local_fft(bend_hfft, Nbend, dtbend, bend_start, tmax, bend_start, high_start, SPLIT_FFT_ROLL_SECONDS, 1, ASacc, ASspline, PSacc, PSspline);
    timer_end = clock();
    bend_fft_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;
    
    high_hfft = double_vector(Nhigh);
    timer_start = clock();
    build_split_local_fft(high_hfft, Nhigh, dt, high_start, tmax, bend_start, high_start, SPLIT_FFT_ROLL_SECONDS, 2, ASacc, ASspline, PSacc, PSspline);
    timer_end = clock();
    high_fft_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;
    
    wdm_packet_fft_plan_init(&plan);
    layer_count = 0;
    pixel_count = 0;
    
    out = fopen("splitfft_layers.dat","w");
    fprintf(out, "# m f_center nmid nsize nblock use_low use_bend use_high dt_low low_nyquist dt_bend bend_nyquist bend_start bend_end high_start high_end\n");
    
    timer_start = clock();
    for(m=1; m<Nf; m++)
    {
        if(nmid[m] <= 0) continue;
        
        Ntx = nsize[m];
        n = nmid[m];
        t0 = ((double)(n-Ntx/2))*wdms->DT;
        Nblock = (int)llround(((double)Ntx)*wdms->DT/dt);
        fcenter = (double)m*wdms->DF;
        packet_start = t0;
        packet_end = t0+(double)Ntx*wdms->DT;
        /*
         A block is allowed to contribute only when two conditions hold:
         (1) its FFT duration contains the local WDM frequency grid exactly, so
             no interpolation is needed, and
         (2) its nonzero time window overlaps the WDM packet support. This keeps
             short late-time blocks from injecting low-frequency leakage into
             purely early-time packets.
         */
        use_low = (Nt%Ntx == 0 && packet_start < low_end && packet_end > 0.0);
        use_bend = (Kbend%Ntx == 0 && packet_start < bend_end && packet_end > bend_start);
        use_high = (Khigh%Ntx == 0 && packet_start < high_end && packet_end > high_start);
        
        if(Nblock < 2 || (Nblock & (Nblock-1)) != 0)
        {
            fprintf(out, "%d %.15e %d %d %d %d %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    m, fcenter, n, Ntx, Nblock, use_low, use_bend, use_high,
                    dtlow, 0.5/dtlow, dtbend, 0.5/dtbend, bend_start, bend_end, high_start, high_end);
            continue;
        }
        
        if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;
        
        fprintf(out, "%d %.15e %d %d %d %d %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                m, fcenter, n, Ntx, Nblock, use_low, use_bend, use_high,
                dtlow, 0.5/dtlow, dtbend, 0.5/dtbend, bend_start, bend_end, high_start, high_end);
        
        plan.data[0] = 0.0;
        for(j=-Ntx/2; j<Ntx/2; j++)
        {
            f = (double)j*plan.dfx+fcenter;
            jj = j+Ntx/2;
            
            if(jj > 0)
            {
                plan.data[jj] = 0.0;
                plan.data[2*Ntx-jj] = 0.0;
                
                if(use_low)
                {
                    sample_real_fft_exact(low_hfft, Nlow, dtlow, f, &re_low, &im_low);
                    phase = 2.0*M_PI*f*t0;
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] += re_low*c-im_low*s;
                    plan.data[2*Ntx-jj] += re_low*s+im_low*c;
                }
                
                if(use_bend)
                {
                    sample_real_fft_exact(bend_hfft, Nbend, dtbend, f, &re_bend, &im_bend);
                    phase = 2.0*M_PI*f*(t0-bend_start);
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] += re_bend*c-im_bend*s;
                    plan.data[2*Ntx-jj] += re_bend*s+im_bend*c;
                }
                
                if(use_high)
                {
                    sample_real_fft_exact(high_hfft, Nhigh, dt, f, &re_high, &im_high);
                    phase = 2.0*M_PI*f*(t0-high_start);
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] += re_high*c-im_high*s;
                    plan.data[2*Ntx-jj] += re_high*s+im_high*c;
                }

                white = thm_instrument_prewhiten_factor(f);
                plan.data[jj] *= white;
                plan.data[2*Ntx-jj] *= white;
            }
        }
        
        wdmtranF_plan(m, &plan);
        
        for(i=0; i<Ntx; i++)
        {
            outn = i+n-Ntx/2;
            if(outn > -1 && outn < Nt)
            {
                wdm[outn][m] = plan.wdmout[i];
                pixel_count++;
            }
        }
        
        layer_count++;
    }
    timer_end = clock();
    packet_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;
    
    printf("split FFT low block N=%d dt=%e duration=%e Nyquist=%e volume=%d time=%f seconds\n",
           Nlow, dtlow, (double)Nlow*dtlow, 0.5/dtlow, (Nf/Rlow)*Nt, low_fft_time);
    printf("split FFT bend block N=%d dt=%e duration=%e Nyquist=%e start=%e end=%e volume=%d time=%f seconds\n",
           Nbend, dtbend, (double)Nbend*dtbend, 0.5/dtbend, bend_start, bend_end, (Nf/Rbend)*Kbend, bend_fft_time);
    printf("split FFT high block N=%d dt=%e duration=%e Nyquist=%e start=%e end=%e volume=%d time=%f seconds\n",
           Nhigh, dt, (double)Nhigh*dt, 0.5/dt, high_start, high_end, Nf*Khigh, high_fft_time);
    printf("split FFT packet assembly/WDM time=%f seconds\n", packet_time);
    
    fclose(out);
    free(low_hfft);
    free(bend_hfft);
    free(high_hfft);
    wdm_packet_fft_plan_free(&plan);
    
    *layers_used = layer_count;
    *pixels_used = pixel_count;
}

void WDMbuildTHMUnionPixelPlan(int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, struct wdmshape *wdms, int include_merger_tail, int *nmid, int *nsize)
{
    int i, k, m, N;
    int *nmid_tmp, *nsize_tmp;
    double tail_time, tail_amp, fmax_spectrum;

    if(nmid == NULL || nsize == NULL || response_time == NULL ||
       Achan == NULL || freq_track == NULL || setup_carrier == NULL ||
       wdms == NULL || Ns < 2 || ncarriers < 1)
    {
        return;
    }

    for(m=0; m<=Nf; m++)
    {
        nmid[m] = -1;
        nsize[m] = 0;
    }

    nmid_tmp = int_vector(Nf);
    nsize_tmp = int_vector(Nf);
    if(nmid_tmp == NULL || nsize_tmp == NULL)
    {
        free_int_vector(nmid_tmp);
        free_int_vector(nsize_tmp);
        return;
    }

    N = Nt*Nf;
    for(k=0; k<ncarriers; k++)
    {
        WDMpixelsTimeScan(Ns, response_time, freq_track[channel][k],
                          nmid_tmp, nsize_tmp, N, wdms);
        if(include_merger_tail)
        {
            tail_time = response_time[Ns-1];
            tail_amp = -1.0;
            for(i=0; i<Ns; i++)
            {
                double amp_abs = fabs(Achan[channel][k][i]);
                if(isfinite(amp_abs) && amp_abs > tail_amp)
                {
                    tail_amp = amp_abs;
                    tail_time = response_time[i];
                }
            }
            fmax_spectrum = setup_carrier[channel][k][6]/
                            (setup_carrier[channel][k][0]*setup_carrier[channel][k][1]);
            WDMpixelsAddMergerFrequencyTail(nmid_tmp, nsize_tmp, fmax_spectrum,
                                            tail_time, wdms);
        }
        WDMmergePixelPlans(nmid, nsize, nmid_tmp, nsize_tmp);
    }

    free_int_vector(nmid_tmp);
    free_int_vector(nsize_tmp);
}

static int split_plan_active_time_range(const int *nmid, const int *nsize, int *active_lo, int *active_hi, int *active_layers, int *active_volume)
{
    int m, lo, hi, layers, volume;

    if(active_lo != NULL) *active_lo = Nt;
    if(active_hi != NULL) *active_hi = 0;
    if(active_layers != NULL) *active_layers = 0;
    if(active_volume != NULL) *active_volume = 0;
    if(nmid == NULL || nsize == NULL) return 0;

    layers = 0;
    volume = 0;
    for(m=1; m<Nf; m++)
    {
        if(nmid[m] < 0 || nsize[m] <= 0) continue;
        lo = nmid[m]-nsize[m]/2;
        hi = lo+nsize[m];
        if(lo < 0) lo = 0;
        if(hi > Nt) hi = Nt;
        if(hi <= lo) continue;

        if(active_lo != NULL && lo < *active_lo) *active_lo = lo;
        if(active_hi != NULL && hi > *active_hi) *active_hi = hi;
        layers++;
        volume += hi-lo;
    }

    if(active_layers != NULL) *active_layers = layers;
    if(active_volume != NULL) *active_volume = volume;
    return layers > 0;
}

static int split_plan_support_layers(const int *nmid, const int *nsize, int support_lo, int support_hi, int *mlo, int *mhi, int *layers, int *overlap_volume)
{
    int m, lo, hi, olo, ohi, count, volume;

    if(mlo != NULL) *mlo = Nf;
    if(mhi != NULL) *mhi = -1;
    if(layers != NULL) *layers = 0;
    if(overlap_volume != NULL) *overlap_volume = 0;
    if(nmid == NULL || nsize == NULL) return 0;

    if(support_lo < 0) support_lo = 0;
    if(support_hi > Nt) support_hi = Nt;
    if(support_hi <= support_lo) return 0;

    count = 0;
    volume = 0;
    for(m=1; m<Nf; m++)
    {
        if(nmid[m] < 0 || nsize[m] <= 0) continue;
        lo = nmid[m]-nsize[m]/2;
        hi = lo+nsize[m];
        if(lo < 0) lo = 0;
        if(hi > Nt) hi = Nt;
        if(hi <= support_lo || lo >= support_hi) continue;

        if(mlo != NULL && m < *mlo) *mlo = m;
        if(mhi != NULL && m > *mhi) *mhi = m;
        olo = lo > support_lo ? lo : support_lo;
        ohi = hi < support_hi ? hi : support_hi;
        if(ohi > olo) volume += ohi-olo;
        count++;
    }

    if(layers != NULL) *layers = count;
    if(overlap_volume != NULL) *overlap_volume = volume;
    return count > 0;
}

static int split_plan_decimation_for_shifted_band(int mhi_shifted, int support_pixels)
{
    int R, bins, guard;

    if(support_pixels < 1) support_pixels = 1;
    guard = SPLIT_FFT_PLAN_NYQUIST_GUARD_LAYERS;
    for(R=Nf; R>=1; R/=2)
    {
        if(Nf%R != 0) continue;
        bins = Nf/R;
        if(bins > mhi_shifted+guard && support_pixels*bins >= 2)
        {
            return R;
        }
    }
    return 1;
}

static int split_plan_centered_even_shift(int mlo, int mhi)
{
    int m_shift;
    double center;

    if(mhi < mlo) return 0;

    /*
     * The split-FFT blocks are complex heterodyned signals, so their baseband
     * support may straddle zero.  Centering the padded layer interval around
     * zero keeps the sample cadence tied to the bandwidth of the block, not to
     * the carrier's absolute frequency.  The shift is kept to an even multiple
     * of DF so that the WDM layer remapping preserves the Meyer/Wilson packet
     * symmetries used elsewhere in the code.
     */
    center = 0.5*((double)mlo+(double)mhi);
    m_shift = 2*(int)llround(0.5*center);
    if(m_shift < 0) m_shift = 0;
    return m_shift;
}

void WDMwriteSplitFFTBandwidthPlan(int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, struct wdmshape *wdms, const char *label, double plan_stop_time, int nband, const double *bandwidth)
{
    int b, block, width, best_width, remaining, max_width;
    int active_lo, active_hi, active_layers, active_volume, stop_hi;
    int tile_lo, tile_hi, support_lo, support_hi, roll_pix;
    int mlo, mhi, layers, overlap_volume, pad_layers, m_needed_lo, m_needed_hi;
    int shifted_lo, shifted_hi, m_shift, target_layers, span_layers, over_band;
    int R, local_bins, min_fft_samples, fft_samples, tf_volume;
    int total_blocks, total_fft_samples, total_tf_volume, total_active_overlap;
    int total_over_band, min_R, max_R, max_span_layers;
    double cost, total_cost, duration, min_duration, max_duration;
    int *nmid, *nsize;
    FILE *out, *summary;
    char filename[160], summary_filename[160];

    if(response_time == NULL || Achan == NULL || freq_track == NULL ||
       setup_carrier == NULL || wdms == NULL || nband < 1 ||
       bandwidth == NULL || Ns < 2 || ncarriers < 1)
    {
        return;
    }

    nmid = int_vector(Nf);
    nsize = int_vector(Nf);
    if(nmid == NULL || nsize == NULL)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    WDMbuildTHMUnionPixelPlan(Ns, response_time, ncarriers, channel,
                              Achan, freq_track, setup_carrier, wdms,
                              0, nmid, nsize);

    if(!split_plan_active_time_range(nmid, nsize, &active_lo, &active_hi,
                                     &active_layers, &active_volume))
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }
    stop_hi = active_hi;
    if(isfinite(plan_stop_time) && plan_stop_time > 0.0)
    {
        stop_hi = (int)floor(plan_stop_time/wdms->DT);
        if(stop_hi > Nt) stop_hi = Nt;
        if(stop_hi < active_hi) active_hi = stop_hi;
    }
    if(active_hi <= active_lo)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    snprintf(filename, sizeof(filename), "splitfft_plan_THM_%s.dat",
             label != NULL ? label : "channel");
    snprintf(summary_filename, sizeof(summary_filename),
             "splitfft_plan_summary_THM_%s.dat",
             label != NULL ? label : "channel");
    out = fopen(filename, "w");
    summary = fopen(summary_filename, "w");
    if(out == NULL && summary == NULL)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    roll_pix = (int)ceil(SPLIT_FFT_ROLL_SECONDS/wdms->DT);
    if(roll_pix < 1) roll_pix = 1;
    pad_layers = (int)ceil(wdms->FB/wdms->DF);
    if(pad_layers < 1) pad_layers = 1;

    if(out != NULL)
    {
        fprintf(out, "# channel %s active_time_pixels [%d,%d) plan_stop_time %.15e active_layers %d active_volume %d DT %.15e DF %.15e FB %.15e roll_seconds %.15e roll_pixels %d\n",
                label != NULL ? label : "channel", active_lo, active_hi,
                plan_stop_time, active_layers, active_volume, wdms->DT,
                wdms->DF, wdms->FB, SPLIT_FFT_ROLL_SECONDS, roll_pix);
        fprintf(out, "# bandwidth_index bandwidth_hz bandwidth_layers block tile_lo tile_hi support_lo support_hi t_tile_start t_tile_end t_support_start t_support_end mlo mhi m_needed_lo m_needed_hi m_shift shifted_lo shifted_hi span_layers over_band R dt_block nyquist_hz local_bins min_fft_samples fft_samples support_duration tf_volume active_overlap est_nlog2n\n");
    }
    if(summary != NULL)
    {
        fprintf(summary, "# channel %s active_time_pixels [%d,%d) plan_stop_time %.15e active_layers %d active_volume %d DT %.15e DF %.15e FB %.15e roll_seconds %.15e roll_pixels %d\n",
                label != NULL ? label : "channel", active_lo, active_hi,
                plan_stop_time, active_layers, active_volume, wdms->DT,
                wdms->DF, wdms->FB, SPLIT_FFT_ROLL_SECONDS, roll_pix);
        fprintf(summary, "# bandwidth_index bandwidth_hz bandwidth_layers blocks total_fft_samples total_tf_volume total_active_overlap total_est_nlog2n over_band_blocks min_R max_R min_duration max_duration max_span_layers\n");
    }

    for(b=0; b<nband; b++)
    {
        if(!isfinite(bandwidth[b]) || bandwidth[b] <= 0.0) continue;
        target_layers = (int)ceil(bandwidth[b]/wdms->DF);
        if(target_layers < 1) target_layers = 1;

        tile_lo = active_lo;
        block = 0;
        total_blocks = 0;
        total_fft_samples = 0;
        total_tf_volume = 0;
        total_active_overlap = 0;
        total_over_band = 0;
        min_R = Nf;
        max_R = 1;
        max_span_layers = 0;
        min_duration = HUGE_VAL;
        max_duration = 0.0;
        total_cost = 0.0;

        while(tile_lo < active_hi)
        {
            remaining = active_hi-tile_lo;
            max_width = thm_largest_power_of_two_leq_int(remaining);
            best_width = 0;

            for(width=max_width; width>=1; width/=2)
            {
                support_lo = tile_lo-roll_pix;
                support_hi = tile_lo+width+roll_pix;
                if(support_lo < 0) support_lo = 0;
                if(support_hi > Nt) support_hi = Nt;
                if(support_hi > active_hi) support_hi = active_hi;

                if(!split_plan_support_layers(nmid, nsize, support_lo,
                                              support_hi, &mlo, &mhi, &layers,
                                              &overlap_volume))
                {
                    best_width = width;
                    break;
                }

                m_needed_lo = mlo-pad_layers;
                m_needed_hi = mhi+pad_layers;
                if(m_needed_lo < 1) m_needed_lo = 1;
                if(m_needed_hi > Nf-1) m_needed_hi = Nf-1;
                span_layers = m_needed_hi-m_needed_lo;
                if(span_layers <= target_layers)
                {
                    best_width = width;
                    break;
                }
            }

            over_band = 0;
            if(best_width < 1)
            {
                best_width = 1;
                over_band = 1;
            }

            tile_hi = tile_lo+best_width;
            if(tile_hi > active_hi) tile_hi = active_hi;
            support_lo = tile_lo-roll_pix;
            support_hi = tile_hi+roll_pix;
            if(support_lo < 0) support_lo = 0;
            if(support_hi > Nt) support_hi = Nt;
            if(support_hi > active_hi) support_hi = active_hi;

            if(!split_plan_support_layers(nmid, nsize, support_lo, support_hi,
                                          &mlo, &mhi, &layers,
                                          &overlap_volume))
            {
                mlo = mhi = m_needed_lo = m_needed_hi = 0;
                shifted_lo = shifted_hi = m_shift = 0;
                span_layers = 0;
                R = Nf;
                local_bins = 1;
            }
            else
            {
                m_needed_lo = mlo-pad_layers;
                m_needed_hi = mhi+pad_layers;
                if(m_needed_lo < 1) m_needed_lo = 1;
                if(m_needed_hi > Nf-1) m_needed_hi = Nf-1;
                span_layers = m_needed_hi-m_needed_lo;
                if(span_layers > target_layers) over_band = 1;

                if(m_needed_lo > SPLIT_FFT_PLAN_MIN_SHIFTED_LAYER)
                {
                    m_shift = 2*((m_needed_lo-SPLIT_FFT_PLAN_MIN_SHIFTED_LAYER)/2);
                }
                else
                {
                    m_shift = 0;
                }
                if(m_shift < 0) m_shift = 0;
                shifted_lo = m_needed_lo-m_shift;
                shifted_hi = m_needed_hi-m_shift;

                R = split_plan_decimation_for_shifted_band(shifted_hi,
                                                           support_hi-support_lo);
                local_bins = Nf/R;
            }

            min_fft_samples = (support_hi-support_lo)*local_bins;
            if(min_fft_samples < 2) min_fft_samples = 2;
            fft_samples = thm_next_power_of_two_int(min_fft_samples);
            duration = ((double)(support_hi-support_lo))*wdms->DT;
            tf_volume = (support_hi-support_lo)*local_bins;
            cost = ((double)fft_samples)*(log((double)fft_samples)/log(2.0));

            if(out != NULL)
            {
                fprintf(out, "%d %.15e %d %d %d %d %d %d %.15e %.15e %.15e %.15e %d %d %d %d %d %d %d %d %d %d %.15e %.15e %d %d %d %.15e %d %d %.15e\n",
                        b, bandwidth[b], target_layers, block, tile_lo,
                        tile_hi, support_lo, support_hi,
                        ((double)tile_lo)*wdms->DT,
                        ((double)tile_hi)*wdms->DT,
                        ((double)support_lo)*wdms->DT,
                        ((double)support_hi)*wdms->DT,
                        mlo, mhi, m_needed_lo, m_needed_hi, m_shift,
                        shifted_lo, shifted_hi, span_layers, over_band,
                        R, ((double)R)*dt, 0.5/(((double)R)*dt),
                        local_bins, min_fft_samples, fft_samples, duration,
                        tf_volume, overlap_volume, cost);
            }

            total_blocks++;
            total_fft_samples += fft_samples;
            total_tf_volume += tf_volume;
            total_active_overlap += overlap_volume;
            total_cost += cost;
            if(over_band) total_over_band++;
            if(R < min_R) min_R = R;
            if(R > max_R) max_R = R;
            if(duration < min_duration) min_duration = duration;
            if(duration > max_duration) max_duration = duration;
            if(span_layers > max_span_layers) max_span_layers = span_layers;

            tile_lo = tile_hi;
            block++;
            if(block > 4*Nt)
            {
                fprintf(stderr, "Warning: split FFT planner stopped after too many blocks for channel %s bandwidth %.15e.\n",
                        label != NULL ? label : "channel", bandwidth[b]);
                break;
            }
        }

        if(summary != NULL)
        {
            if(total_blocks == 0) min_duration = 0.0;
            fprintf(summary, "%d %.15e %d %d %d %d %d %.15e %d %d %d %.15e %.15e %d\n",
                    b, bandwidth[b], target_layers, total_blocks,
                    total_fft_samples, total_tf_volume, total_active_overlap,
                    total_cost, total_over_band, min_R, max_R, min_duration,
                    max_duration, max_span_layers);
        }

        printf("split FFT early packet planner THM %s bandwidth %.6e Hz (%d layers): blocks %d fft_samples %d tf_volume %d active_overlap %d over_band %d est_nlog2n %.6e\n",
               label != NULL ? label : "channel", bandwidth[b],
               target_layers, total_blocks, total_fft_samples, total_tf_volume,
               total_active_overlap, total_over_band, total_cost);
    }

    if(out != NULL) fclose(out);
    if(summary != NULL) fclose(summary);
    free_int_vector(nmid);
    free_int_vector(nsize);
}

static int split_plan_track_frequency_range(int Ns, const double *response_time, const double *freq, double tlo, double thi, double *fmin, double *fmax)
{
    int i;
    double flo, fhi, f;

    if(fmin != NULL) *fmin = 0.0;
    if(fmax != NULL) *fmax = 0.0;
    if(Ns < 2 || response_time == NULL || freq == NULL ||
       fmin == NULL || fmax == NULL || thi <= tlo)
    {
        return 0;
    }

    if(tlo < response_time[0]) tlo = response_time[0];
    if(thi > response_time[Ns-1]) thi = response_time[Ns-1];
    if(thi <= tlo) return 0;

    flo = linear_interp_clamped(Ns, response_time, freq, tlo);
    fhi = linear_interp_clamped(Ns, response_time, freq, thi);
    *fmin = flo < fhi ? flo : fhi;
    *fmax = flo > fhi ? flo : fhi;

    for(i=0; i<Ns; i++)
    {
        if(response_time[i] <= tlo || response_time[i] >= thi) continue;
        f = freq[i];
        if(!isfinite(f)) continue;
        if(f < *fmin) *fmin = f;
        if(f > *fmax) *fmax = f;
    }

    return isfinite(*fmin) && isfinite(*fmax) && *fmax >= *fmin;
}

static int split_plan_track_width_ok(int Ns, double *response_time, int ncarriers, int channel, double ***freq_track, struct wdmshape *wdms, int tile_lo, int width, int roll_pix, int max_support_hi, double bandwidth, double *max_span)
{
    int k, support_lo, support_hi;
    double tlo, thi, fmin, fmax, span;

    if(max_span != NULL) *max_span = 0.0;
    if(response_time == NULL || freq_track == NULL || wdms == NULL ||
       ncarriers < 1 || width < 1 || bandwidth <= 0.0)
    {
        return 0;
    }

    support_lo = tile_lo-roll_pix;
    support_hi = tile_lo+width+roll_pix;
    if(support_lo < 0) support_lo = 0;
    if(support_hi > Nt) support_hi = Nt;
    if(max_support_hi > 0 && support_hi > max_support_hi) support_hi = max_support_hi;
    tlo = ((double)support_lo)*wdms->DT;
    thi = ((double)support_hi)*wdms->DT;
    if(tlo < response_time[0]) tlo = response_time[0];
    if(thi > response_time[Ns-1]) thi = response_time[Ns-1];
    if(thi <= tlo) return 1;

    for(k=0; k<ncarriers; k++)
    {
        if(!split_plan_track_frequency_range(Ns, response_time,
                                             freq_track[channel][k], tlo, thi,
                                             &fmin, &fmax))
        {
            continue;
        }
        span = (fmax-fmin)+2.0*wdms->FB;
        if(max_span != NULL && span > *max_span) *max_span = span;
        if(span > bandwidth) return 0;
    }

    return 1;
}

static int split_plan_single_track_width_ok(int Ns, double *response_time, const double *freq, struct wdmshape *wdms, int tile_lo, int width, int roll_pix, int max_support_hi, double bandwidth, double *max_span)
{
    int support_lo, support_hi;
    double tlo, thi, fmin, fmax, span;

    if(max_span != NULL) *max_span = 0.0;
    if(response_time == NULL || freq == NULL || wdms == NULL ||
       width < 1 || bandwidth <= 0.0)
    {
        return 0;
    }

    support_lo = tile_lo-roll_pix;
    support_hi = tile_lo+width+roll_pix;
    if(support_lo < 0) support_lo = 0;
    if(support_hi > Nt) support_hi = Nt;
    if(max_support_hi > 0 && support_hi > max_support_hi)
    {
        support_hi = max_support_hi;
    }

    tlo = ((double)support_lo)*wdms->DT;
    thi = ((double)support_hi)*wdms->DT;
    if(tlo < response_time[0]) tlo = response_time[0];
    if(thi > response_time[Ns-1]) thi = response_time[Ns-1];
    if(thi <= tlo) return 1;

    if(!split_plan_track_frequency_range(Ns, response_time, freq,
                                         tlo, thi, &fmin, &fmax))
    {
        return 1;
    }

    span = (fmax-fmin)+2.0*wdms->FB;
    if(max_span != NULL) *max_span = span;

    return span <= bandwidth;
}

void WDMwriteSplitFFTTrackBandwidthPlan(int Ns, double *response_time, int ncarriers, int channel, double ***freq_track, const THMFoldedCarrier *carrier, struct wdmshape *wdms, const char *label, double plan_stop_time, int nband, const double *bandwidth)
{
    int b, k, block, width, best_width, remaining, max_width;
    int active_lo, active_hi, tile_lo, tile_hi, support_lo, support_hi, roll_pix;
    int target_layers, m_needed_lo, m_needed_hi, shifted_lo, shifted_hi, m_shift;
    int span_layers, over_band, R, local_bins, min_fft_samples, fft_samples, tf_volume;
    int total_blocks, total_carrier_blocks, total_fft_samples, total_tf_volume;
    int total_over_band, min_R, max_R, max_span_layers;
    double tlo, thi, fmin, fmax, span, max_span, duration, min_duration, max_duration;
    double cost, total_cost;
    FILE *out, *summary;
    char filename[160], summary_filename[160];

    if(response_time == NULL || freq_track == NULL || wdms == NULL ||
       ncarriers < 1 || nband < 1 || bandwidth == NULL || Ns < 2)
    {
        return;
    }

    active_lo = 0;
    active_hi = (int)ceil(response_time[Ns-1]/wdms->DT);
    if(active_hi > Nt) active_hi = Nt;
    if(active_hi <= active_lo) active_hi = Nt;
    if(isfinite(plan_stop_time) && plan_stop_time > 0.0)
    {
        int stop_hi = (int)floor(plan_stop_time/wdms->DT);
        if(stop_hi > Nt) stop_hi = Nt;
        if(stop_hi < active_hi) active_hi = stop_hi;
    }
    if(active_hi <= active_lo) return;

    snprintf(filename, sizeof(filename), "splitfft_track_plan_THM_%s.dat",
             label != NULL ? label : "channel");
    snprintf(summary_filename, sizeof(summary_filename),
             "splitfft_track_plan_summary_THM_%s.dat",
             label != NULL ? label : "channel");
    out = fopen(filename, "w");
    summary = fopen(summary_filename, "w");
    if(out == NULL && summary == NULL) return;

    roll_pix = (int)ceil(SPLIT_FFT_ROLL_SECONDS/wdms->DT);
    if(roll_pix < 1) roll_pix = 1;

    if(out != NULL)
    {
        fprintf(out, "# channel %s active_time_pixels [%d,%d) plan_stop_time %.15e DT %.15e DF %.15e FB %.15e roll_seconds %.15e roll_pixels %d\n",
                label != NULL ? label : "channel", active_lo, active_hi,
                plan_stop_time, wdms->DT, wdms->DF, wdms->FB,
                SPLIT_FFT_ROLL_SECONDS, roll_pix);
        fprintf(out, "# bandwidth_index bandwidth_hz bandwidth_layers time_block carrier ell abs_m tile_lo tile_hi support_lo support_hi t_tile_start t_tile_end t_support_start t_support_end fmin fmax span_hz m_needed_lo m_needed_hi m_shift shifted_lo shifted_hi span_layers over_band R dt_block nyquist_hz local_bins min_fft_samples fft_samples support_duration tf_volume est_nlog2n\n");
    }
    if(summary != NULL)
    {
        fprintf(summary, "# channel %s active_time_pixels [%d,%d) plan_stop_time %.15e DT %.15e DF %.15e FB %.15e roll_seconds %.15e roll_pixels %d\n",
                label != NULL ? label : "channel", active_lo, active_hi,
                plan_stop_time, wdms->DT, wdms->DF, wdms->FB,
                SPLIT_FFT_ROLL_SECONDS, roll_pix);
        fprintf(summary, "# bandwidth_index bandwidth_hz bandwidth_layers time_blocks carrier_blocks total_fft_samples total_tf_volume total_est_nlog2n over_band_carrier_blocks min_R max_R min_duration max_duration max_span_layers\n");
    }

    for(b=0; b<nband; b++)
    {
        if(!isfinite(bandwidth[b]) || bandwidth[b] <= 0.0) continue;
        target_layers = (int)ceil(bandwidth[b]/wdms->DF);
        if(target_layers < 1) target_layers = 1;

        tile_lo = active_lo;
        block = 0;
        total_blocks = 0;
        total_carrier_blocks = 0;
        total_fft_samples = 0;
        total_tf_volume = 0;
        total_over_band = 0;
        min_R = Nf;
        max_R = 1;
        max_span_layers = 0;
        min_duration = HUGE_VAL;
        max_duration = 0.0;
        total_cost = 0.0;

        while(tile_lo < active_hi)
        {
            remaining = active_hi-tile_lo;
            max_width = thm_largest_power_of_two_leq_int(remaining);
            best_width = 0;
            for(width=max_width; width>=1; width/=2)
            {
                if(split_plan_track_width_ok(Ns, response_time, ncarriers,
                                             channel, freq_track, wdms,
                                             tile_lo, width, roll_pix, active_hi,
                                             bandwidth[b], &max_span))
                {
                    best_width = width;
                    break;
                }
            }
            if(best_width < 1) best_width = 1;

            tile_hi = tile_lo+best_width;
            if(tile_hi > active_hi) tile_hi = active_hi;
            support_lo = tile_lo-roll_pix;
            support_hi = tile_hi+roll_pix;
            if(support_lo < 0) support_lo = 0;
            if(support_hi > Nt) support_hi = Nt;
            if(support_hi > active_hi) support_hi = active_hi;
            tlo = ((double)support_lo)*wdms->DT;
            thi = ((double)support_hi)*wdms->DT;
            if(tlo < response_time[0]) tlo = response_time[0];
            if(thi > response_time[Ns-1]) thi = response_time[Ns-1];
            duration = thi-tlo;
            if(duration <= 0.0) duration = ((double)(support_hi-support_lo))*wdms->DT;

            for(k=0; k<ncarriers; k++)
            {
                if(!split_plan_track_frequency_range(Ns, response_time,
                                                     freq_track[channel][k],
                                                     tlo, thi, &fmin, &fmax))
                {
                    continue;
                }

                span = (fmax-fmin)+2.0*wdms->FB;
                m_needed_lo = (int)floor((fmin-wdms->FB)/wdms->DF);
                m_needed_hi = (int)ceil((fmax+wdms->FB)/wdms->DF);
                if(m_needed_lo < 1) m_needed_lo = 1;
                if(m_needed_hi > Nf-1) m_needed_hi = Nf-1;
                span_layers = m_needed_hi-m_needed_lo;
                over_band = span > bandwidth[b];
                if(m_needed_lo > SPLIT_FFT_PLAN_MIN_SHIFTED_LAYER)
                {
                    m_shift = 2*((m_needed_lo-SPLIT_FFT_PLAN_MIN_SHIFTED_LAYER)/2);
                }
                else
                {
                    m_shift = 0;
                }
                shifted_lo = m_needed_lo-m_shift;
                shifted_hi = m_needed_hi-m_shift;
                R = split_plan_decimation_for_shifted_band(shifted_hi,
                                                           support_hi-support_lo);
                local_bins = Nf/R;
                min_fft_samples = (support_hi-support_lo)*local_bins;
                if(min_fft_samples < 2) min_fft_samples = 2;
                fft_samples = thm_next_power_of_two_int(min_fft_samples);
                tf_volume = (support_hi-support_lo)*local_bins;
                cost = ((double)fft_samples)*(log((double)fft_samples)/log(2.0));

                if(out != NULL)
                {
                    fprintf(out, "%d %.15e %d %d %d %d %d %d %d %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %d %d %d %d %d %d %d %d %.15e %.15e %d %d %d %.15e %d %.15e\n",
                            b, bandwidth[b], target_layers, block, k,
                            carrier != NULL ? carrier[k].ell : 0,
                            carrier != NULL ? abs(carrier[k].emm) : 0,
                            tile_lo, tile_hi, support_lo, support_hi,
                            ((double)tile_lo)*wdms->DT,
                            ((double)tile_hi)*wdms->DT, tlo, thi,
                            fmin, fmax, span, m_needed_lo, m_needed_hi,
                            m_shift, shifted_lo, shifted_hi, span_layers,
                            over_band, R, ((double)R)*dt,
                            0.5/(((double)R)*dt), local_bins,
                            min_fft_samples, fft_samples, duration, tf_volume,
                            cost);
                }

                total_carrier_blocks++;
                total_fft_samples += fft_samples;
                total_tf_volume += tf_volume;
                total_cost += cost;
                if(over_band) total_over_band++;
                if(R < min_R) min_R = R;
                if(R > max_R) max_R = R;
                if(span_layers > max_span_layers) max_span_layers = span_layers;
            }

            if(duration < min_duration) min_duration = duration;
            if(duration > max_duration) max_duration = duration;
            total_blocks++;
            tile_lo = tile_hi;
            block++;
            if(block > 4*Nt)
            {
                fprintf(stderr, "Warning: track split FFT planner stopped after too many blocks for channel %s bandwidth %.15e.\n",
                        label != NULL ? label : "channel", bandwidth[b]);
                break;
            }
        }

        if(summary != NULL)
        {
            if(total_blocks == 0) min_duration = 0.0;
            fprintf(summary, "%d %.15e %d %d %d %d %d %.15e %d %d %d %.15e %.15e %d\n",
                    b, bandwidth[b], target_layers, total_blocks,
                    total_carrier_blocks, total_fft_samples, total_tf_volume,
                    total_cost, total_over_band, min_R, max_R, min_duration,
                    max_duration, max_span_layers);
        }

        printf("split FFT early track planner THM %s bandwidth %.6e Hz (%d layers): time_blocks %d carrier_blocks %d fft_samples %d tf_volume %d over_band %d est_nlog2n %.6e\n",
               label != NULL ? label : "channel", bandwidth[b],
               target_layers, total_blocks, total_carrier_blocks,
               total_fft_samples, total_tf_volume, total_over_band,
               total_cost);
    }

    if(out != NULL) fclose(out);
    if(summary != NULL) fclose(summary);
}

static double thm_channel_sum_from_ap(double t, int ncarriers, int channel, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline)
{
    int k, idx;
    double h, A, P;

    h = 0.0;
    for(k=0; k<ncarriers; k++)
    {
        idx = channel*ncarriers+k;
        if(ATspline[idx] == NULL || PTspline[idx] == NULL) continue;
        A = gsl_spline_eval(ATspline[idx], t, ATacc[idx]);
        P = gsl_spline_eval(PTspline[idx], t, PTacc[idx]);
        h += A*cos(P);
    }

    return h;
}

static int thm_time_intervals_overlap(double alo, double ahi, double blo, double bhi)
{
    return (ahi > blo && alo < bhi);
}

static double thm_early_split_partition_weight(double t, int block, int nblocks, const double *boundary, const double *boundary_roll, double endpoint_start, double endpoint_rise, double roll)
{
    double w, r;

    w = 1.0;
    if(block > 0)
    {
        r = boundary_roll != NULL ? boundary_roll[block-1] : roll;
        if(r <= 0.0) r = roll;
        w *= split_smooth_step(t, boundary[block-1], boundary[block-1]+r);
    }
    if(block < nblocks-1)
    {
        r = boundary_roll != NULL ? boundary_roll[block] : roll;
        if(r <= 0.0) r = roll;
        w *= 1.0-split_smooth_step(t, boundary[block], boundary[block]+r);
    }
    else
    {
        w *= 1.0-split_smooth_step(t, endpoint_start,
                                   endpoint_start+endpoint_rise);
    }

    return w;
}

static void WDMaddEarlySplitFFTCarrier(double **wdm, int Ns, double *response_time, int ncarriers, int carrier_index, int channel, double ***freq_track, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline, struct wdmshape *wdms, const int *nmid_eval, const int *nsize_eval, double endpoint_start, double endpoint_rise, double bandwidth, const char *label, int *layers_used, int *pixels_used, double *early_build_time_out, double *packet_time_out)
{
    int i, j, jj, m, block, idx, outn;
    int tile_lo, tile_hi, active_hi, endpoint_support_hi;
    int remaining, max_width, width, best_width;
    int roll_pix, nblocks, max_blocks, support_pixels, max_packet_nsize;
    int m_needed_lo, m_needed_hi, m_shift, shifted_lo, shifted_hi, shifted_extent;
    int R, local_bins, Kfft, Nfft, Ntx, n, layer_count, pixel_count;
    int *nmid, *nsize;
    double *boundary, *boundary_roll;
    double t, tau, t0, packet_start, packet_end, fcenter, f, fbase, phase;
    double layer_low, layer_high, block_low, block_high;
    double fmin, fmax, re, im, c, s, weight, A, P, shift_phase, white;
    double early_build_time, packet_time;
    THMEarlySplitFFTBlock *blocks;
    WDMPacketFFTPlan plan;
    FILE *out;
    char filename[180];
    clock_t timer_start, timer_end;

    if(layers_used != NULL) *layers_used = 0;
    if(pixels_used != NULL) *pixels_used = 0;
    if(early_build_time_out != NULL) *early_build_time_out = 0.0;
    if(packet_time_out != NULL) *packet_time_out = 0.0;

    if(wdm == NULL || response_time == NULL || freq_track == NULL ||
       ATacc == NULL || ATspline == NULL || PTacc == NULL ||
       PTspline == NULL || wdms == NULL || Ns < 2 ||
       nmid_eval == NULL || nsize_eval == NULL ||
       carrier_index < 0 || carrier_index >= ncarriers)
    {
        return;
    }

    idx = channel*ncarriers+carrier_index;
    if(ATspline[idx] == NULL || PTspline[idx] == NULL) return;

    nmid = int_vector(Nf);
    nsize = int_vector(Nf);
    if(nmid == NULL || nsize == NULL)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    WDMpixelsTimeScanRange(Ns, response_time, freq_track[channel][carrier_index],
                           response_time[0], endpoint_start+endpoint_rise,
                           nmid, nsize, Nt*Nf, wdms);

    if(!isfinite(bandwidth) || bandwidth <= 0.0) bandwidth = 0.016;
    roll_pix = (int)ceil(SPLIT_FFT_ROLL_SECONDS/wdms->DT);
    if(roll_pix < 1) roll_pix = 1;
    active_hi = (int)floor(endpoint_start/wdms->DT);
    if(active_hi > Nt) active_hi = Nt;
    endpoint_support_hi = (int)ceil((endpoint_start+endpoint_rise)/wdms->DT);
    if(endpoint_support_hi > Nt) endpoint_support_hi = Nt;
    if(endpoint_support_hi < active_hi) endpoint_support_hi = active_hi;
    if(active_hi < 1)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    max_blocks = Nt+1;
    blocks = calloc((size_t)max_blocks, sizeof(*blocks));
    boundary = double_vector(max_blocks);
    boundary_roll = double_vector(max_blocks);
    if(blocks == NULL || boundary == NULL || boundary_roll == NULL)
    {
        free(blocks);
        free_double_vector(boundary);
        free_double_vector(boundary_roll);
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    tile_lo = 0;
    nblocks = 0;
    while(tile_lo < active_hi && nblocks < max_blocks)
    {
        remaining = active_hi-tile_lo;
        max_width = thm_largest_power_of_two_leq_int(remaining);
        best_width = 0;
        for(width=max_width; width>=1; width/=2)
        {
            /*
             * This routine is already called one folded carrier at a time.  The
             * block layout must therefore be set by this carrier's baseband
             * sweep, not by the union of all harmonics.  Otherwise a high-m
             * harmonic can force unnecessarily fine sampling for every carrier.
             */
            if(split_plan_single_track_width_ok(Ns, response_time,
                                                freq_track[channel][carrier_index],
                                                wdms, tile_lo, width, roll_pix,
                                                endpoint_support_hi, bandwidth,
                                                NULL))
            {
                best_width = width;
                break;
            }
        }
        if(best_width < 1) best_width = 1;

        tile_hi = tile_lo+best_width;
        if(tile_hi > active_hi) tile_hi = active_hi;

        blocks[nblocks].tile_lo = tile_lo;
        blocks[nblocks].tile_hi = tile_hi;
        boundary[nblocks] = ((double)tile_hi)*wdms->DT;

        tile_lo = tile_hi;
        nblocks++;
    }

    for(block=0; block<nblocks; block++)
    {
        boundary_roll[block] = SPLIT_FFT_ROLL_SECONDS;
    }
    for(block=0; block<nblocks-1; block++)
    {
        double next_width, endpoint_gap, r;

        r = SPLIT_FFT_ROLL_SECONDS;
        next_width = boundary[block+1]-boundary[block];
        endpoint_gap = endpoint_start-boundary[block];
        if(next_width > 0.0 && r > next_width) r = next_width;
        if(endpoint_gap > 0.0 && r > endpoint_gap) r = endpoint_gap;
        if(r <= 0.0) r = wdms->DT;
        boundary_roll[block] = r;
    }

    out = NULL;
    if(thm_diagnostics_enabled)
    {
        snprintf(filename, sizeof(filename),
                 "splitfft_early_blocks_THM_%s_carrier%d.dat",
                 label != NULL ? label : "channel", carrier_index);
        out = fopen(filename, "w");
    }
    if(out != NULL)
    {
        fprintf(out, "# channel %s carrier %d bandwidth %.15e endpoint_start %.15e endpoint_rise %.15e DT %.15e DF %.15e FB %.15e roll_seconds %.15e\n",
                label != NULL ? label : "channel", carrier_index, bandwidth,
                endpoint_start, endpoint_rise, wdms->DT, wdms->DF, wdms->FB,
                SPLIT_FFT_ROLL_SECONDS);
        fprintf(out, "# block tile_lo tile_hi nonzero_start nonzero_end support_lo support_hi fmin fmax fpad m_needed_lo m_needed_hi m_shift shifted_lo shifted_hi f_shift R dt_block nyquist_hz local_bins Kfft Nfft max_packet_nsize local_roll_s build_time\n");
    }

    early_build_time = 0.0;
    for(block=0; block<nblocks; block++)
    {
        blocks[block].nonzero_start = block == 0 ? response_time[0] : boundary[block-1];
        if(block < nblocks-1)
        {
            blocks[block].nonzero_end = boundary[block]+boundary_roll[block];
        }
        else
        {
            blocks[block].nonzero_end = endpoint_start+endpoint_rise;
        }
        if(blocks[block].nonzero_end > response_time[Ns-1])
        {
            blocks[block].nonzero_end = response_time[Ns-1];
        }
        if(blocks[block].nonzero_end <= blocks[block].nonzero_start)
        {
            blocks[block].nonzero_end = blocks[block].nonzero_start+wdms->DT;
        }

        blocks[block].support_lo = (int)floor(blocks[block].nonzero_start/wdms->DT);
        if(blocks[block].support_lo < 0) blocks[block].support_lo = 0;
        blocks[block].support_hi = (int)ceil(blocks[block].nonzero_end/wdms->DT);
        if(blocks[block].support_hi > Nt) blocks[block].support_hi = Nt;
        if(blocks[block].support_hi <= blocks[block].support_lo)
        {
            blocks[block].support_hi = blocks[block].support_lo+1;
        }
        blocks[block].block_start = ((double)blocks[block].support_lo)*wdms->DT;
        support_pixels = blocks[block].support_hi-blocks[block].support_lo;

        if(!split_plan_track_frequency_range(Ns, response_time,
                                             freq_track[channel][carrier_index],
                                             blocks[block].nonzero_start,
                                             blocks[block].nonzero_end,
                                             &fmin, &fmax))
        {
            fmin = 0.0;
            fmax = wdms->DF;
        }
        blocks[block].fmin = fmin;
        blocks[block].fmax = fmax;

        /*
         * A partitioned time block has Fourier sidebands from the rising and
         * falling tapers.  Keep only the WDM layers that can overlap the
         * carrier sweep after padding by the Meyer half-band and a few inverse
         * taper times.  This is deliberately conservative: it keeps the
         * additive partition accurate while preventing every carrier block from
         * filling every active layer.
         */
        blocks[block].fpad = wdms->FB;
        if(thm_split_fft_endpoint_margin_cycles > 0.0)
        {
            if(block > 0 && boundary_roll[block-1] > 0.0)
            {
                double fpad_rise;
                fpad_rise = wdms->FB+
                    thm_split_fft_endpoint_margin_cycles/boundary_roll[block-1];
                if(fpad_rise > blocks[block].fpad)
                {
                    blocks[block].fpad = fpad_rise;
                }
            }
            if(block < nblocks-1 && boundary_roll[block] > 0.0)
            {
                double fpad_fall;
                fpad_fall = wdms->FB+
                    thm_split_fft_endpoint_margin_cycles/boundary_roll[block];
                if(fpad_fall > blocks[block].fpad)
                {
                    blocks[block].fpad = fpad_fall;
                }
            }
            if(block == nblocks-1 && endpoint_rise > 0.0)
            {
                double fpad_endpoint;
                fpad_endpoint = wdms->FB+
                    thm_split_fft_endpoint_margin_cycles/endpoint_rise;
                if(fpad_endpoint > blocks[block].fpad)
                {
                    blocks[block].fpad = fpad_endpoint;
                }
            }
        }

        m_needed_lo = (int)floor((fmin-blocks[block].fpad)/wdms->DF);
        m_needed_hi = (int)ceil((fmax+blocks[block].fpad)/wdms->DF);
        if(m_needed_lo < 1) m_needed_lo = 1;
        if(m_needed_hi < 1) m_needed_hi = 1;
        if(m_needed_hi > Nf-1) m_needed_hi = Nf-1;
        m_shift = split_plan_centered_even_shift(m_needed_lo, m_needed_hi);
        shifted_lo = m_needed_lo-m_shift;
        shifted_hi = m_needed_hi-m_shift;
        shifted_extent = abs(shifted_lo);
        if(abs(shifted_hi) > shifted_extent) shifted_extent = abs(shifted_hi);
        if(shifted_extent < 1) shifted_extent = 1;

        R = split_plan_decimation_for_shifted_band(shifted_extent,
                                                   support_pixels);
        if(R < 1) R = 1;
        /*
         * The last early block is multiplied by the relatively short falling
         * endpoint taper.  Its Fourier support is therefore wider than the
         * intrinsic track band used to choose shifted_hi above.  Keep one extra
         * octave of baseband headroom so the complementary rising/falling
         * endpoint tapers can cancel cleanly in the overlap.
         */
        if(block == nblocks-1 && R > 1) R /= 2;
        local_bins = Nf/R;

        max_packet_nsize = 1;
        for(m=1; m<Nf; m++)
        {
            if(nmid_eval[m] < 0 || nsize_eval[m] <= 0) continue;
            layer_low = ((double)m)*wdms->DF-wdms->FB;
            layer_high = ((double)m)*wdms->DF+wdms->FB;
            block_low = blocks[block].fmin-blocks[block].fpad;
            block_high = blocks[block].fmax+blocks[block].fpad;
            if(layer_high < block_low || layer_low > block_high) continue;

            packet_start = ((double)(nmid_eval[m]-nsize_eval[m]/2))*wdms->DT;
            packet_end = packet_start+((double)nsize_eval[m])*wdms->DT;
            /*
             * Size the local FFT for active WDM packets that overlap the block
             * in both time and padded frequency support.  The padding above is
             * the safety margin for taper sidebands.
             */
            if(thm_time_intervals_overlap(packet_start, packet_end,
                                          blocks[block].nonzero_start,
                                          blocks[block].nonzero_end) &&
               nsize_eval[m] > max_packet_nsize)
            {
                max_packet_nsize = nsize_eval[m];
            }
        }

        Kfft = thm_next_power_of_two_int(support_pixels);
        if(Kfft < max_packet_nsize)
        {
            Kfft = thm_next_power_of_two_int(max_packet_nsize);
        }
        if(Kfft < 1) Kfft = 1;
        Nfft = Kfft*local_bins;
        if(Nfft < 2) Nfft = 2;

        blocks[block].R = R;
        blocks[block].m_shift = m_shift;
        blocks[block].shifted_lo = shifted_lo;
        blocks[block].shifted_hi = shifted_hi;
        blocks[block].local_bins = local_bins;
        blocks[block].Kfft = Kfft;
        blocks[block].Nfft = Nfft;
        blocks[block].sample_dt = ((double)R)*dt;
        blocks[block].shift_frequency = ((double)m_shift)*wdms->DF;
        blocks[block].hfft = double_vector(2*Nfft);
        if(blocks[block].hfft == NULL) continue;

        timer_start = clock();
        for(i=0; i<Nfft; i++)
        {
            t = blocks[block].block_start+((double)i)*blocks[block].sample_dt;
            REAL(blocks[block].hfft,i) = 0.0;
            IMAG(blocks[block].hfft,i) = 0.0;
            if(t >= response_time[0] && t <= response_time[Ns-1])
            {
                weight = thm_early_split_partition_weight(t, block, nblocks,
                                                          boundary,
                                                          boundary_roll,
                                                          endpoint_start,
                                                          endpoint_rise,
                                                          SPLIT_FFT_ROLL_SECONDS);
                if(weight != 0.0)
                {
                    A = gsl_spline_eval(ATspline[idx], t, ATacc[idx]);
                    P = gsl_spline_eval(PTspline[idx], t, PTacc[idx]);
                    tau = t-blocks[block].block_start;
                    shift_phase = 2.0*M_PI*blocks[block].shift_frequency*tau;
                    REAL(blocks[block].hfft,i) = weight*A*cos(P-shift_phase);
                    IMAG(blocks[block].hfft,i) = weight*A*sin(P-shift_phase);
                }
            }
        }
        gsl_fft_complex_radix2_forward(blocks[block].hfft, 1,
                                       (size_t)Nfft);
        for(i=0; i<Nfft; i++)
        {
            REAL(blocks[block].hfft,i) *= blocks[block].sample_dt;
            IMAG(blocks[block].hfft,i) *= blocks[block].sample_dt;
        }
        timer_end = clock();
        blocks[block].build_time =
            ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;
        early_build_time += blocks[block].build_time;

        if(out != NULL)
        {
            fprintf(out, "%d %d %d %.15e %.15e %d %d %.15e %.15e %.15e %d %d %d %d %d %.15e %d %.15e %.15e %d %d %d %d %.15e %.15e\n",
                    block, blocks[block].tile_lo, blocks[block].tile_hi,
                    blocks[block].nonzero_start, blocks[block].nonzero_end,
                    blocks[block].support_lo, blocks[block].support_hi,
                    blocks[block].fmin, blocks[block].fmax,
                    blocks[block].fpad,
                    m_needed_lo, m_needed_hi, blocks[block].m_shift,
                    blocks[block].shifted_lo, blocks[block].shifted_hi,
                    blocks[block].shift_frequency, blocks[block].R,
                    blocks[block].sample_dt, 0.5/blocks[block].sample_dt,
                    blocks[block].local_bins, blocks[block].Kfft,
                    blocks[block].Nfft, max_packet_nsize,
                    block < nblocks-1 ? boundary_roll[block] : endpoint_rise,
                    blocks[block].build_time);
        }
    }
    if(out != NULL) fclose(out);

    wdm_packet_fft_plan_init(&plan);
    layer_count = 0;
    pixel_count = 0;
    timer_start = clock();
    for(m=1; m<Nf; m++)
    {
        if(nmid_eval[m] <= 0 || nsize_eval[m] <= 0) continue;
        Ntx = nsize_eval[m];
        n = nmid_eval[m];
        t0 = ((double)(n-Ntx/2))*wdms->DT;
        packet_start = t0;
        packet_end = t0+((double)Ntx)*wdms->DT;
        fcenter = ((double)m)*wdms->DF;
        layer_low = fcenter-wdms->FB;
        layer_high = fcenter+wdms->FB;

        for(block=0; block<nblocks; block++)
        {
            block_low = blocks[block].fmin-blocks[block].fpad;
            block_high = blocks[block].fmax+blocks[block].fpad;
            if(layer_high < block_low || layer_low > block_high) continue;
            if(thm_time_intervals_overlap(packet_start, packet_end,
                                          blocks[block].nonzero_start,
                                          blocks[block].nonzero_end))
            {
                break;
            }
        }
        if(block >= nblocks) continue;

        if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;
        plan.data[0] = 0.0;

        for(j=-Ntx/2; j<Ntx/2; j++)
        {
            f = ((double)j)*plan.dfx+fcenter;
            jj = j+Ntx/2;
            if(jj > 0)
            {
                plan.data[jj] = 0.0;
                plan.data[2*Ntx-jj] = 0.0;

                for(block=0; block<nblocks; block++)
                {
                    if(blocks[block].hfft == NULL) continue;
                    if(blocks[block].Kfft%Ntx != 0) continue;
                    block_low = blocks[block].fmin-blocks[block].fpad;
                    block_high = blocks[block].fmax+blocks[block].fpad;
                    if(layer_high < block_low || layer_low > block_high)
                    {
                        continue;
                    }
                    if(!thm_time_intervals_overlap(packet_start, packet_end,
                                                   blocks[block].nonzero_start,
                                                   blocks[block].nonzero_end))
                    {
                        continue;
                    }

                    fbase = f-blocks[block].shift_frequency;
                    sample_complex_fft_exact(blocks[block].hfft,
                                             blocks[block].Nfft,
                                             blocks[block].sample_dt,
                                             fbase, &re, &im);
                    if(re != 0.0 || im != 0.0)
                    {
                        phase = 2.0*M_PI*f*(t0-blocks[block].block_start);
                        c = cos(phase);
                        s = sin(phase);
                        plan.data[jj] += re*c-im*s;
                        plan.data[2*Ntx-jj] += re*s+im*c;
                    }
                }

                white = thm_instrument_prewhiten_factor(f);
                plan.data[jj] *= white;
                plan.data[2*Ntx-jj] *= white;
            }
        }

        wdmtranF_plan(m, &plan);

        for(i=0; i<Ntx; i++)
        {
            outn = i+n-Ntx/2;
            if(outn > -1 && outn < Nt)
            {
                wdm[outn][m] += plan.wdmout[i];
                pixel_count++;
            }
        }
        layer_count++;
    }
    timer_end = clock();
    packet_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;

    wdm_packet_fft_plan_free(&plan);
    for(block=0; block<nblocks; block++)
    {
        free_double_vector(blocks[block].hfft);
    }
    free(blocks);
    free_double_vector(boundary);
    free_double_vector(boundary_roll);
    free_int_vector(nmid);
    free_int_vector(nsize);

    if(layers_used != NULL) *layers_used = layer_count;
    if(pixels_used != NULL) *pixels_used = pixel_count;
    if(early_build_time_out != NULL) *early_build_time_out = early_build_time;
    if(packet_time_out != NULL) *packet_time_out = packet_time;
}

/*
 * Intended no-SPA early FFT path for THM.
 *
 * The early/inspiral signal is transformed one folded carrier at a time.  Each
 * carrier gets its own bandwidth-driven time blocks and its own even-DF
 * heterodyne, so the baseband bandwidth is compact.  The endpoint remains one
 * summed channel FFT because all carrier tracks are nearly vertical there and
 * the single transform is both cheaper and closer to the direct time-domain
 * construction.
 */
void WDMtrackEarlySplitFFTPerCarrierEndpoint(double **wdm, int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline, struct wdmshape *wdms, double *endpoint_setup, double bandwidth, const char *label, int *layers_used, int *pixels_used)
{
    int i, k, m, idx;
    int early_layers, early_pixels, total_early_layers, total_early_pixels;
    int endpoint_layers, endpoint_pixels;
    int *nmid_common, *nsize_common;
    double endpoint_dt, endpoint_start, endpoint_rise, endpoint_end;
    double f_late_start, fk, f_endpoint_start, f_endpoint_max;
    double endpoint_margin;
    double *endpoint_base;
    double t, weight, hval, build_time, packet_time;
    double total_build_time, total_packet_time, endpoint_build_time, endpoint_wdm_time;
    clock_t start, end;

    if(layers_used != NULL) *layers_used = 0;
    if(pixels_used != NULL) *pixels_used = 0;
    if(wdm == NULL || response_time == NULL || ncarriers < 1 ||
       Achan == NULL || freq_track == NULL || setup_carrier == NULL ||
       ATacc == NULL || ATspline == NULL || PTacc == NULL ||
       PTspline == NULL || wdms == NULL || endpoint_setup == NULL ||
       Ns < 2)
    {
        return;
    }

    for(i=0; i<Nt; i++)
    {
        for(m=0; m<=Nf; m++) wdm[i][m] = 0.0;
    }

    if(!isfinite(bandwidth) || bandwidth <= 0.0) bandwidth = 0.016;

    endpoint_dt = endpoint_setup[0];
    endpoint_start = endpoint_setup[2];
    endpoint_rise = endpoint_setup[3];
    if(endpoint_dt <= 0.0 || endpoint_setup[1] < 2.0 ||
       !isfinite(endpoint_start) || !isfinite(endpoint_rise) ||
       endpoint_rise <= 0.0)
    {
        fprintf(stderr, "Warning: per-carrier split-early FFT THM %s has invalid endpoint setup.\n",
                label != NULL ? label : "channel");
        return;
    }
    endpoint_end = endpoint_start+endpoint_dt*endpoint_setup[1];

    nmid_common = int_vector(Nf);
    nsize_common = int_vector(Nf);
    endpoint_base = double_vector((int)endpoint_setup[1]);
    if(nmid_common == NULL || nsize_common == NULL ||
       endpoint_base == NULL)
    {
        free_int_vector(nmid_common);
        free_int_vector(nsize_common);
        free_double_vector(endpoint_base);
        return;
    }
    WDMbuildTHMUnionPixelPlan(Ns, response_time, ncarriers, channel,
                              Achan, freq_track, setup_carrier, wdms,
                              1, nmid_common, nsize_common);

    total_early_layers = 0;
    total_early_pixels = 0;
    total_build_time = 0.0;
    total_packet_time = 0.0;
    for(k=0; k<ncarriers; k++)
    {
        build_time = 0.0;
        packet_time = 0.0;
        early_layers = 0;
        early_pixels = 0;
        WDMaddEarlySplitFFTCarrier(wdm, Ns, response_time, ncarriers, k,
                                   channel, freq_track, ATacc, ATspline,
                                   PTacc, PTspline, wdms,
                                   nmid_common, nsize_common,
                                   endpoint_start, endpoint_rise,
                                   bandwidth, label,
                                   &early_layers, &early_pixels,
                                   &build_time, &packet_time);
        total_early_layers += early_layers;
        total_early_pixels += early_pixels;
        total_build_time += build_time;
        total_packet_time += packet_time;
    }

    start = clock();
    for(i=0; i<(int)endpoint_setup[1]; i++)
    {
        t = endpoint_start+((double)i)*endpoint_dt;
        endpoint_base[i] = 0.0;
        if(t >= response_time[0] && t <= response_time[Ns-1])
        {
            weight = split_smooth_step(t, endpoint_start,
                                       endpoint_start+endpoint_rise);
            if(weight != 0.0)
            {
                hval = 0.0;
                for(k=0; k<ncarriers; k++)
                {
                    idx = channel*ncarriers+k;
                    if(ATspline[idx] == NULL || PTspline[idx] == NULL) continue;
                    hval += gsl_spline_eval(ATspline[idx], t, ATacc[idx])*
                            cos(gsl_spline_eval(PTspline[idx], t, PTacc[idx]));
                }
                endpoint_base[i] = weight*hval;
            }
        }
    }
    end = clock();
    endpoint_build_time = ((double)(end-start))/CLOCKS_PER_SEC;

    /*
     * The split-early FFT path is a true time-domain partition of unity:
     *
     *     h(t) = sum_b w_b(t) h_b(t) + w_endpoint(t) h(t).
     *
     * The endpoint FFT is therefore an additive piece, not a frequency-layer
     * handoff.  In particular, low-frequency Meyer packets centered near the
     * endpoint can overlap the endpoint rising taper even when their center
     * frequency lies below the instantaneous endpoint frequency.  The pruning
     * below keeps those taper sidebands by starting several inverse-rise-times
     * below the first carrier frequency at the endpoint.  For low-frequency
     * carriers the nominal C/rise sideband can be wider than the carrier
     * frequency itself; cap the margin at a fraction of f_endpoint so those
     * modes do not force the summed endpoint block all the way to DC.  Set the
     * margin to zero from the command line for the exact unpruned diagnostic.
     */
    f_late_start = HUGE_VAL;
    for(k=0; k<ncarriers; k++)
    {
        fk = linear_interp_clamped(Ns, response_time,
                                   freq_track[channel][k],
                                   endpoint_start);
        if(isfinite(fk) && fk > 0.0 && fk < f_late_start)
        {
            f_late_start = fk;
        }
    }
    endpoint_margin = 0.0;
    if(thm_split_fft_endpoint_margin_cycles > 0.0 &&
       isfinite(f_late_start) && f_late_start < HUGE_VAL &&
       endpoint_rise > 0.0)
    {
        endpoint_margin =
            thm_split_fft_endpoint_margin_cycles/endpoint_rise+wdms->FB;
        if(endpoint_margin >
           SPLIT_FFT_ENDPOINT_MARGIN_FRACTION_MAX*f_late_start)
        {
            endpoint_margin =
                SPLIT_FFT_ENDPOINT_MARGIN_FRACTION_MAX*f_late_start;
        }
        f_endpoint_start = f_late_start-endpoint_margin;
        if(f_endpoint_start < 0.0) f_endpoint_start = 0.0;
    }
    else
    {
        f_endpoint_start = 0.0;
    }
    f_endpoint_max = endpoint_setup[6]/(endpoint_setup[0]*endpoint_setup[1]);
    if(!isfinite(f_endpoint_max) || f_endpoint_max <= 0.0)
    {
        f_endpoint_max = 0.5/endpoint_setup[0];
    }
    if(f_endpoint_max > 0.5/endpoint_setup[0])
    {
        f_endpoint_max = 0.5/endpoint_setup[0];
    }

    start = clock();
    WDMaddWithShortFFTThreshold(wdm, nmid_common, nsize_common,
                                endpoint_base, endpoint_setup,
                                f_endpoint_start, f_endpoint_max,
                                wdms, NULL,
                                &endpoint_layers, &endpoint_pixels);
    end = clock();
    endpoint_wdm_time = ((double)(end-start))/CLOCKS_PER_SEC;

    if(thm_summary_output_enabled || getenv("THM_PARTITION_DEBUG") != NULL)
    {
        printf("split early per-carrier FFT THM %s bandwidth %.6e carriers %d early_build %.6f early_packet_wdm %.6f endpoint_build %.6f endpoint_wdm %.6f early_layers %d early_pixels %d endpoint_layers %d endpoint_pixels %d f_endpoint_start %.6e f_endpoint_max %.6e endpoint_margin_cycles %.6e endpoint_margin_hz %.6e\n",
               label != NULL ? label : "channel", bandwidth, ncarriers,
               total_build_time, total_packet_time, endpoint_build_time,
               endpoint_wdm_time, total_early_layers, total_early_pixels,
               endpoint_layers, endpoint_pixels, f_endpoint_start, f_endpoint_max,
               thm_split_fft_endpoint_margin_cycles, endpoint_margin);
    }

    if(layers_used != NULL) *layers_used = total_early_layers+endpoint_layers;
    if(pixels_used != NULL) *pixels_used = total_early_pixels+endpoint_pixels;

    free_int_vector(nmid_common);
    free_int_vector(nsize_common);
    free_double_vector(endpoint_base);
}

/*
 * Experimental bandwidth-planned split FFT for the early part of the THM TDI
 * waveform.  This is meant to test replacing the SPA side of the production
 * WDM path with a sum of direct FFTs:
 *
 *     h(t) = sum_b w_b(t) h(t) + w_end(t) h(t)
 *
 * where the early block weights w_b form a partition of unity up to the start
 * of the existing merger/ringdown endpoint FFT.  Each early block is
 * heterodyned by an even multiple of DF so the baseband packet can be sampled
 * at a lower cadence, then added back to the original WDM layer.  The blocks
 * are built from the positive-frequency analytic carriers A_k exp(i phi_k);
 * heterodyning the real cosine waveform would also shift the negative
 * frequency mirror and can alias it into the baseband after decimation.  The
 * even shift preserves the Wilson/Meyer layer parity used by wdmtranF_plan().
 * The endpoint block keeps the usual short full-rate real time window.
 */
void WDMtrackEarlySplitFFTSummed(double **wdm, int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline, struct wdmshape *wdms, double *endpoint_setup, double bandwidth, const char *label, int *layers_used, int *pixels_used)
{
    int i, j, jj, k, m, n, outn, block, idx;
    int tile_lo, tile_hi, active_hi, remaining, max_width, width, best_width;
    int roll_pix, nblocks, max_blocks, support_pixels, max_packet_nsize;
    int m_needed_lo, m_needed_hi, m_shift, shifted_lo, shifted_hi;
    int R, local_bins, Kfft, Nfft;
    int Ntx, Nblock, Ncopy, endpoint_Nblock, endpoint_samples;
    int layer_count, pixel_count, endpoint_layers, endpoint_pixels;
    int *nmid, *nsize;
    double endpoint_dt, endpoint_start, endpoint_rise, endpoint_end;
    double t, tau, t0, packet_start, packet_end, fcenter, f, fbase, phase;
    double fmin, fmax, fkmin, fkmax, re, im, c, s, weight, hval, shift_phase;
    double white;
    double A, P, zre, zim;
    double early_build_time, endpoint_build_time, endpoint_fft_time, packet_time;
    double *boundary, *endpoint_base, *endpoint_hfft;
    THMEarlySplitFFTBlock *blocks;
    WDMPacketFFTPlan plan;
    FILE *out;
    char filename[160];
    clock_t timer_start, timer_end;

    if(layers_used != NULL) *layers_used = 0;
    if(pixels_used != NULL) *pixels_used = 0;
    if(wdm == NULL || response_time == NULL || ncarriers < 1 ||
       freq_track == NULL || setup_carrier == NULL || ATacc == NULL ||
       ATspline == NULL || PTacc == NULL || PTspline == NULL ||
       wdms == NULL || endpoint_setup == NULL || Ns < 2)
    {
        return;
    }

    for(i=0; i<Nt; i++)
    {
        for(j=0; j<=Nf; j++) wdm[i][j] = 0.0;
    }

    if(!isfinite(bandwidth) || bandwidth <= 0.0)
    {
        bandwidth = 0.016;
    }

    endpoint_dt = endpoint_setup[0];
    endpoint_samples = (int)endpoint_setup[1];
    endpoint_start = endpoint_setup[2];
    endpoint_rise = endpoint_setup[3];
    if(endpoint_dt <= 0.0 || endpoint_samples < 2 ||
       !isfinite(endpoint_start) || !isfinite(endpoint_rise) ||
       endpoint_rise <= 0.0)
    {
        fprintf(stderr, "Warning: split-early FFT THM %s has invalid endpoint setup.\n",
                label != NULL ? label : "channel");
        return;
    }
    endpoint_end = endpoint_start+(double)endpoint_samples*endpoint_dt;

    nmid = int_vector(Nf);
    nsize = int_vector(Nf);
    if(nmid == NULL || nsize == NULL)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }
    WDMbuildTHMUnionPixelPlan(Ns, response_time, ncarriers, channel,
                              Achan, freq_track, setup_carrier, wdms,
                              1, nmid, nsize);

    roll_pix = (int)ceil(SPLIT_FFT_ROLL_SECONDS/wdms->DT);
    if(roll_pix < 1) roll_pix = 1;
    active_hi = (int)floor(endpoint_start/wdms->DT);
    if(active_hi > Nt) active_hi = Nt;
    if(active_hi < 1)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    max_blocks = Nt+1;
    blocks = calloc((size_t)max_blocks, sizeof(*blocks));
    boundary = double_vector(max_blocks);
    if(blocks == NULL || boundary == NULL)
    {
        free(blocks);
        free_double_vector(boundary);
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    tile_lo = 0;
    nblocks = 0;
    while(tile_lo < active_hi && nblocks < max_blocks)
    {
        remaining = active_hi-tile_lo;
        max_width = thm_largest_power_of_two_leq_int(remaining);
        best_width = 0;
        for(width=max_width; width>=1; width/=2)
        {
            if(split_plan_track_width_ok(Ns, response_time, ncarriers,
                                         channel, freq_track, wdms,
                                         tile_lo, width, roll_pix, active_hi,
                                         bandwidth, NULL))
            {
                best_width = width;
                break;
            }
        }
        if(best_width < 1) best_width = 1;

        tile_hi = tile_lo+best_width;
        if(tile_hi > active_hi) tile_hi = active_hi;

        blocks[nblocks].tile_lo = tile_lo;
        blocks[nblocks].tile_hi = tile_hi;
        boundary[nblocks] = ((double)tile_hi)*wdms->DT;

        tile_lo = tile_hi;
        nblocks++;
    }

    if(nblocks < 1)
    {
        free(blocks);
        free_double_vector(boundary);
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    snprintf(filename, sizeof(filename), "splitfft_early_blocks_THM_%s.dat",
             label != NULL ? label : "channel");
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# channel %s bandwidth %.15e endpoint_start %.15e endpoint_rise %.15e endpoint_end %.15e DT %.15e DF %.15e FB %.15e roll_seconds %.15e\n",
                label != NULL ? label : "channel", bandwidth, endpoint_start,
                endpoint_rise, endpoint_end, wdms->DT, wdms->DF, wdms->FB,
                SPLIT_FFT_ROLL_SECONDS);
        fprintf(out, "# block tile_lo tile_hi nonzero_start nonzero_end support_lo support_hi fmin fmax m_needed_lo m_needed_hi m_shift shifted_lo shifted_hi f_shift R dt_block nyquist_hz local_bins Kfft Nfft max_packet_nsize build_time\n");
    }

    early_build_time = 0.0;
    for(block=0; block<nblocks; block++)
    {
        blocks[block].nonzero_start = block == 0 ? 0.0 : boundary[block-1];
        if(block < nblocks-1)
        {
            blocks[block].nonzero_end = boundary[block]+SPLIT_FFT_ROLL_SECONDS;
        }
        else
        {
            blocks[block].nonzero_end = endpoint_start+endpoint_rise;
        }
        if(blocks[block].nonzero_end > response_time[Ns-1])
        {
            blocks[block].nonzero_end = response_time[Ns-1];
        }
        if(blocks[block].nonzero_end <= blocks[block].nonzero_start)
        {
            blocks[block].nonzero_end = blocks[block].nonzero_start+wdms->DT;
        }

        blocks[block].support_lo = (int)floor(blocks[block].nonzero_start/wdms->DT);
        if(blocks[block].support_lo < 0) blocks[block].support_lo = 0;
        blocks[block].support_hi = (int)ceil(blocks[block].nonzero_end/wdms->DT);
        if(blocks[block].support_hi > Nt) blocks[block].support_hi = Nt;
        if(blocks[block].support_hi <= blocks[block].support_lo)
        {
            blocks[block].support_hi = blocks[block].support_lo+1;
        }
        blocks[block].block_start = ((double)blocks[block].support_lo)*wdms->DT;
        support_pixels = blocks[block].support_hi-blocks[block].support_lo;

        fmin = HUGE_VAL;
        fmax = 0.0;
        for(k=0; k<ncarriers; k++)
        {
            if(split_plan_track_frequency_range(Ns, response_time,
                                                freq_track[channel][k],
                                                blocks[block].nonzero_start,
                                                blocks[block].nonzero_end,
                                                &fkmin, &fkmax))
            {
                if(fkmin < fmin) fmin = fkmin;
                if(fkmax > fmax) fmax = fkmax;
            }
        }
        if(!isfinite(fmin) || !isfinite(fmax) || fmax <= 0.0)
        {
            fmin = 0.0;
            fmax = wdms->DF;
        }
        blocks[block].fmin = fmin;
        blocks[block].fmax = fmax;

        m_needed_lo = (int)floor((fmin-wdms->FB)/wdms->DF);
        m_needed_hi = (int)ceil((fmax+wdms->FB)/wdms->DF);
        if(m_needed_lo < 1) m_needed_lo = 1;
        if(m_needed_hi < 1) m_needed_hi = 1;
        if(m_needed_hi > Nf-1) m_needed_hi = Nf-1;
        if(m_needed_lo > SPLIT_FFT_PLAN_MIN_SHIFTED_LAYER)
        {
            m_shift = 2*((m_needed_lo-SPLIT_FFT_PLAN_MIN_SHIFTED_LAYER)/2);
        }
        else
        {
            m_shift = 0;
        }
        if(m_shift < 0) m_shift = 0;
        shifted_lo = m_needed_lo-m_shift;
        shifted_hi = m_needed_hi-m_shift;
        if(shifted_lo < 1) shifted_lo = 1;
        if(shifted_hi < shifted_lo) shifted_hi = shifted_lo;

        R = split_plan_decimation_for_shifted_band(shifted_hi,
                                                   support_pixels);
        if(R < 1) R = 1;
        local_bins = Nf/R;

        max_packet_nsize = 1;
        for(m=1; m<Nf; m++)
        {
            if(nmid[m] < 0 || nsize[m] <= 0) continue;
            packet_start = ((double)(nmid[m]-nsize[m]/2))*wdms->DT;
            packet_end = packet_start+((double)nsize[m])*wdms->DT;
            if(thm_time_intervals_overlap(packet_start, packet_end,
                                          blocks[block].nonzero_start,
                                          blocks[block].nonzero_end) &&
               nsize[m] > max_packet_nsize)
            {
                max_packet_nsize = nsize[m];
            }
        }

        Kfft = thm_next_power_of_two_int(support_pixels);
        if(Kfft < max_packet_nsize)
        {
            Kfft = thm_next_power_of_two_int(max_packet_nsize);
        }
        if(Kfft < 1) Kfft = 1;
        Nfft = Kfft*local_bins;
        if(Nfft < 2) Nfft = 2;

        blocks[block].R = R;
        blocks[block].m_shift = m_shift;
        blocks[block].shifted_lo = shifted_lo;
        blocks[block].shifted_hi = shifted_hi;
        blocks[block].local_bins = local_bins;
        blocks[block].Kfft = Kfft;
        blocks[block].Nfft = Nfft;
        blocks[block].sample_dt = ((double)R)*dt;
        blocks[block].shift_frequency = ((double)m_shift)*wdms->DF;
        blocks[block].hfft = double_vector(2*Nfft);
        if(blocks[block].hfft == NULL) continue;

        timer_start = clock();
        for(i=0; i<Nfft; i++)
        {
            t = blocks[block].block_start+((double)i)*blocks[block].sample_dt;
            REAL(blocks[block].hfft,i) = 0.0;
            IMAG(blocks[block].hfft,i) = 0.0;
            if(t >= response_time[0] && t <= response_time[Ns-1])
            {
                weight = thm_early_split_partition_weight(t, block, nblocks,
                                                          boundary,
                                                          NULL,
                                                          endpoint_start,
                                                          endpoint_rise,
                                                          SPLIT_FFT_ROLL_SECONDS);
                if(weight != 0.0)
                {
                    zre = 0.0;
                    zim = 0.0;
                    for(k=0; k<ncarriers; k++)
                    {
                        idx = channel*ncarriers+k;
                        if(ATspline[idx] == NULL || PTspline[idx] == NULL)
                        {
                            continue;
                        }
                        A = gsl_spline_eval(ATspline[idx], t, ATacc[idx]);
                        P = gsl_spline_eval(PTspline[idx], t, PTacc[idx]);
                        zre += A*cos(P);
                        zim += A*sin(P);
                    }
                    tau = t-blocks[block].block_start;
                    shift_phase = 2.0*M_PI*blocks[block].shift_frequency*tau;
                    c = cos(shift_phase);
                    s = sin(shift_phase);
                    REAL(blocks[block].hfft,i) = weight*(zre*c+zim*s);
                    IMAG(blocks[block].hfft,i) = weight*(zim*c-zre*s);
                }
            }
        }
        gsl_fft_complex_radix2_forward(blocks[block].hfft, 1,
                                       (size_t)Nfft);
        for(i=0; i<Nfft; i++)
        {
            REAL(blocks[block].hfft,i) *= blocks[block].sample_dt;
            IMAG(blocks[block].hfft,i) *= blocks[block].sample_dt;
        }
        timer_end = clock();
        blocks[block].build_time =
            ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;
        early_build_time += blocks[block].build_time;

        if(out != NULL)
        {
            fprintf(out, "%d %d %d %.15e %.15e %d %d %.15e %.15e %d %d %d %d %d %.15e %d %.15e %.15e %d %d %d %d %.15e\n",
                    block, blocks[block].tile_lo, blocks[block].tile_hi,
                    blocks[block].nonzero_start, blocks[block].nonzero_end,
                    blocks[block].support_lo, blocks[block].support_hi,
                    blocks[block].fmin, blocks[block].fmax,
                    m_needed_lo, m_needed_hi, blocks[block].m_shift,
                    blocks[block].shifted_lo, blocks[block].shifted_hi,
                    blocks[block].shift_frequency, blocks[block].R,
                    blocks[block].sample_dt, 0.5/blocks[block].sample_dt,
                    blocks[block].local_bins, blocks[block].Kfft,
                    blocks[block].Nfft,
                    max_packet_nsize, blocks[block].build_time);
        }
    }
    if(out != NULL) fclose(out);

    endpoint_base = double_vector(endpoint_samples);
    endpoint_hfft = NULL;
    endpoint_Nblock = 0;
    endpoint_build_time = 0.0;
    endpoint_fft_time = 0.0;
    if(endpoint_base != NULL)
    {
        timer_start = clock();
        for(i=0; i<endpoint_samples; i++)
        {
            t = endpoint_start+((double)i)*endpoint_dt;
            endpoint_base[i] = 0.0;
            if(t >= response_time[0] && t <= response_time[Ns-1])
            {
                weight = split_smooth_step(t, endpoint_start,
                                           endpoint_start+endpoint_rise);
                if(weight != 0.0)
                {
                    endpoint_base[i] =
                        weight*thm_channel_sum_from_ap(t, ncarriers, channel,
                                                       ATacc, ATspline,
                                                       PTacc, PTspline);
                }
            }
        }
        timer_end = clock();
        endpoint_build_time =
            ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;
    }

    wdm_packet_fft_plan_init(&plan);
    layer_count = 0;
    pixel_count = 0;
    endpoint_layers = 0;
    endpoint_pixels = 0;

    timer_start = clock();
    for(m=1; m<Nf; m++)
    {
        if(nmid[m] <= 0 || nsize[m] <= 0) continue;

        Ntx = nsize[m];
        n = nmid[m];
        t0 = ((double)(n-Ntx/2))*wdms->DT;
        packet_start = t0;
        packet_end = t0+((double)Ntx)*wdms->DT;
        fcenter = ((double)m)*wdms->DF;

        if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;
        plan.data[0] = 0.0;

        for(j=-Ntx/2; j<Ntx/2; j++)
        {
            f = ((double)j)*plan.dfx+fcenter;
            jj = j+Ntx/2;
            if(jj > 0)
            {
                plan.data[jj] = 0.0;
                plan.data[2*Ntx-jj] = 0.0;

                for(block=0; block<nblocks; block++)
                {
                    if(blocks[block].hfft == NULL) continue;
                    if(blocks[block].Kfft%Ntx != 0) continue;
                    if(!thm_time_intervals_overlap(packet_start, packet_end,
                                                   blocks[block].nonzero_start,
                                                   blocks[block].nonzero_end))
                    {
                        continue;
                    }
                    fbase = f-blocks[block].shift_frequency;
                    sample_complex_fft_exact(blocks[block].hfft,
                                             blocks[block].Nfft,
                                             blocks[block].sample_dt,
                                             fbase, &re, &im);
                    if(re != 0.0 || im != 0.0)
                    {
                        phase = 2.0*M_PI*f*(t0-blocks[block].block_start);
                        c = cos(phase);
                        s = sin(phase);
                        plan.data[jj] += re*c-im*s;
                        plan.data[2*Ntx-jj] += re*s+im*c;
                    }
                }

                if(endpoint_base != NULL &&
                   thm_time_intervals_overlap(packet_start, packet_end,
                                              endpoint_start, endpoint_end))
                {
                    Nblock = (int)llround(((double)Ntx)*wdms->DT/endpoint_dt);
                    if(Nblock >= 2 && (Nblock & (Nblock-1)) == 0)
                    {
                        if(Nblock != endpoint_Nblock)
                        {
                            clock_t e_start, e_end;

                            free(endpoint_hfft);
                            endpoint_hfft = double_vector(Nblock);
                            endpoint_Nblock = 0;
                            if(endpoint_hfft != NULL)
                            {
                                e_start = clock();
                                for(i=0; i<Nblock; i++) endpoint_hfft[i] = 0.0;
                                Ncopy = endpoint_samples;
                                if(Ncopy > Nblock) Ncopy = Nblock;
                                for(i=0; i<Ncopy; i++)
                                {
                                    endpoint_hfft[i] = endpoint_base[i];
                                }
                                gsl_fft_real_radix2_transform(endpoint_hfft,
                                                              1, Nblock);
                                for(i=0; i<Nblock; i++)
                                {
                                    endpoint_hfft[i] *= (2.0*endpoint_dt);
                                }
                                e_end = clock();
                                endpoint_fft_time +=
                                    ((double)(e_end-e_start))/CLOCKS_PER_SEC;
                                endpoint_Nblock = Nblock;
                            }
                        }
                        if(endpoint_hfft != NULL)
                        {
                            sample_short_window_spectrum(endpoint_hfft,
                                                         Nblock, endpoint_dt,
                                                         f, &re, &im);
                            if(re != 0.0 || im != 0.0)
                            {
                                phase = 2.0*M_PI*f*(t0-endpoint_start);
                                c = cos(phase);
                                s = sin(phase);
                                plan.data[jj] += re*c-im*s;
                                plan.data[2*Ntx-jj] += re*s+im*c;
                            }
                        }
                    }
                }

                white = thm_instrument_prewhiten_factor(f);
                plan.data[jj] *= white;
                plan.data[2*Ntx-jj] *= white;
            }
        }

        wdmtranF_plan(m, &plan);

        for(i=0; i<Ntx; i++)
        {
            outn = i+n-Ntx/2;
            if(outn > -1 && outn < Nt)
            {
                wdm[outn][m] = plan.wdmout[i];
                pixel_count++;
                if(thm_time_intervals_overlap(packet_start, packet_end,
                                              endpoint_start, endpoint_end))
                {
                    endpoint_pixels++;
                }
            }
        }
        layer_count++;
        if(thm_time_intervals_overlap(packet_start, packet_end,
                                      endpoint_start, endpoint_end))
        {
            endpoint_layers++;
        }
    }
    timer_end = clock();
    packet_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;

    printf("split early FFT THM %s bandwidth %.6e blocks %d early_build %.6f endpoint_build %.6f endpoint_ffts %.6f packet_wdm %.6f layers %d pixels %d endpoint_layers %d endpoint_pixels %d\n",
           label != NULL ? label : "channel", bandwidth, nblocks,
           early_build_time, endpoint_build_time, endpoint_fft_time,
           packet_time, layer_count, pixel_count, endpoint_layers,
           endpoint_pixels);

    wdm_packet_fft_plan_free(&plan);
    free(endpoint_hfft);
    free_double_vector(endpoint_base);
    for(block=0; block<nblocks; block++)
    {
        free_double_vector(blocks[block].hfft);
    }
    free(blocks);
    free_double_vector(boundary);
    free_int_vector(nmid);
    free_int_vector(nsize);

    if(layers_used != NULL) *layers_used = layer_count;
    if(pixels_used != NULL) *pixels_used = pixel_count;
}

/*
 * Experimental no-SPA split FFT for the summed THM TDI channel.
 *
 * This is the multi-carrier companion to WDMtrackSplitFFT().  It is meant as
 * a diagnostic for replacing the SPA side of the fast WDM calculation with a
 * small number of time-windowed FFTs.  The channel waveform is constructed as
 * sum_k A_k(t) cos(phi_k(t)) from the post-TDI folded-carrier AP splines, then
 * split by a smooth partition of unity.  No heterodyning is applied in this
 * first version; the block cadences and durations are the same as the old 22
 * diagnostic.
 */
void WDMtrackSplitFFTSummed(double **wdm, int Ns, double *response_time, int ncarriers, int channel, double ***Achan, double ***freq_track, double ***setup_carrier, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline, struct wdmshape *wdms, const char *label, int *layers_used, int *pixels_used)
{
    int i, j, jj, m, n, outn;
    int N, Ntx, Nblock, Nlow, Nbend, Nhigh;
    int Rlow, Rbend, Kbend, Khigh;
    int use_low, use_bend, use_high;
    int layer_count, pixel_count;
    int *nmid, *nsize;
    double dtlow, dtbend, fcenter, f, t0;
    double bend_start, bend_end, high_start, high_end, high_taper_end, low_end;
    double packet_start, packet_end;
    double re_low, im_low, re_bend, im_bend, re_high, im_high;
    double phase, c, s, tmin, tmax, white;
    double low_fft_time, bend_fft_time, high_fft_time, packet_time;
    double *low_hfft, *bend_hfft, *high_hfft;
    WDMPacketFFTPlan plan;
    clock_t timer_start, timer_end;
    FILE *out;
    char filename[160];

    if(layers_used != NULL) *layers_used = 0;
    if(pixels_used != NULL) *pixels_used = 0;
    if(wdm == NULL || response_time == NULL || ncarriers < 1 ||
       Achan == NULL || freq_track == NULL || setup_carrier == NULL ||
       ATacc == NULL || ATspline == NULL || PTacc == NULL ||
       PTspline == NULL || wdms == NULL || Ns < 2)
    {
        return;
    }

    N = Nt*Nf;
    Rlow = SPLIT_FFT_LOW_DECIMATION;
    Rbend = SPLIT_FFT_BEND_DECIMATION;
    Kbend = SPLIT_FFT_BEND_NTILES;
    Khigh = SPLIT_FFT_HIGH_NTILES;
    Nlow = N/Rlow;
    Nbend = Kbend*(Nf/Rbend);
    Nhigh = Khigh*Nf;
    dtlow = dt*(double)Rlow;
    dtbend = dt*(double)Rbend;
    tmin = response_time[0];
    tmax = response_time[Ns-1];

    for(i=0; i<Nt; i++)
    {
        for(j=0; j<=Nf; j++) wdm[i][j] = 0.0;
    }

    if(Rlow < 1 || N%Rlow != 0 || Nf%Rlow != 0 || (Nlow & (Nlow-1)) != 0)
    {
        fprintf(stderr, "Error: SPLIT_FFT_LOW_DECIMATION=%d must divide N and Nf and leave a power-of-two low-rate FFT length.\n", Rlow);
        return;
    }

    if(Rbend < 1 || Nf%Rbend != 0 || Nbend < 2 || (Nbend & (Nbend-1)) != 0)
    {
        fprintf(stderr, "Error: SPLIT_FFT_BEND_DECIMATION=%d and SPLIT_FFT_BEND_NTILES=%d must define a power-of-two bend FFT length.\n", Rbend, Kbend);
        return;
    }

    if(Khigh < 2 || Nhigh < 2 || (Nhigh & (Nhigh-1)) != 0)
    {
        fprintf(stderr, "Error: SPLIT_FFT_HIGH_NTILES=%d must define a power-of-two high-rate FFT length.\n", Khigh);
        return;
    }

    if(SPLIT_FFT_ROLL_SECONDS <= 0.0 ||
       SPLIT_FFT_ROLL_SECONDS >= (double)Khigh*wdms->DT ||
       2.0*SPLIT_FFT_ROLL_SECONDS >= (double)Kbend*wdms->DT)
    {
        fprintf(stderr, "Error: SPLIT_FFT_ROLL_SECONDS=%e is incompatible with the bend/high block durations.\n", SPLIT_FFT_ROLL_SECONDS);
        return;
    }

    nmid = int_vector(Nf);
    nsize = int_vector(Nf);
    if(nmid == NULL || nsize == NULL)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }
    WDMbuildTHMUnionPixelPlan(Ns, response_time, ncarriers, channel,
                              Achan, freq_track, setup_carrier, wdms,
                              1, nmid, nsize);

    high_end = wdms->DT*ceil(tmax/wdms->DT);
    high_start = high_end-(double)Nhigh*dt;
    high_taper_end = high_start+SPLIT_FFT_ROLL_SECONDS;
    bend_end = high_taper_end;
    bend_start = bend_end-(double)Nbend*dtbend;
    low_end = bend_start+SPLIT_FFT_ROLL_SECONDS;

    if(high_start < tmin || bend_start < tmin)
    {
        fprintf(stderr, "Error: split FFT blocks would start before available response samples: bend_start=%e high_start=%e tmin=%e.\n",
                bend_start, high_start, tmin);
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    low_hfft = double_vector(Nlow);
    bend_hfft = double_vector(Nbend);
    high_hfft = double_vector(Nhigh);
    if(low_hfft == NULL || bend_hfft == NULL || high_hfft == NULL)
    {
        free_double_vector(low_hfft);
        free_double_vector(bend_hfft);
        free_double_vector(high_hfft);
        free_int_vector(nmid);
        free_int_vector(nsize);
        return;
    }

    timer_start = clock();
    build_split_low_fft_sum(low_hfft, Nlow, dtlow, tmin, tmax, bend_start,
                            high_start, SPLIT_FFT_ROLL_SECONDS, ncarriers,
                            channel, ATacc, ATspline, PTacc, PTspline);
    timer_end = clock();
    low_fft_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;

    timer_start = clock();
    build_split_local_fft_sum(bend_hfft, Nbend, dtbend, bend_start, tmin, tmax,
                              bend_start, high_start, SPLIT_FFT_ROLL_SECONDS,
                              1, ncarriers, channel, ATacc, ATspline,
                              PTacc, PTspline);
    timer_end = clock();
    bend_fft_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;

    timer_start = clock();
    build_split_local_fft_sum(high_hfft, Nhigh, dt, high_start, tmin, tmax,
                              bend_start, high_start, SPLIT_FFT_ROLL_SECONDS,
                              2, ncarriers, channel, ATacc, ATspline,
                              PTacc, PTspline);
    timer_end = clock();
    high_fft_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;

    wdm_packet_fft_plan_init(&plan);
    layer_count = 0;
    pixel_count = 0;

    snprintf(filename, sizeof(filename), "splitfft_layers_THM_%s.dat",
             label != NULL ? label : "channel");
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# m f_center nmid nsize nblock use_low use_bend use_high dt_low low_nyquist dt_bend bend_nyquist bend_start bend_end high_start high_end\n");
    }

    timer_start = clock();
    for(m=1; m<Nf; m++)
    {
        if(nmid[m] <= 0) continue;

        Ntx = nsize[m];
        n = nmid[m];
        t0 = ((double)(n-Ntx/2))*wdms->DT;
        Nblock = (int)llround(((double)Ntx)*wdms->DT/dt);
        fcenter = (double)m*wdms->DF;
        packet_start = t0;
        packet_end = t0+(double)Ntx*wdms->DT;

        use_low = (Nt%Ntx == 0 && packet_start < low_end && packet_end > 0.0);
        use_bend = (Kbend%Ntx == 0 && packet_start < bend_end && packet_end > bend_start);
        use_high = (Khigh%Ntx == 0 && packet_start < high_end && packet_end > high_start);

        if(out != NULL)
        {
            fprintf(out, "%d %.15e %d %d %d %d %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    m, fcenter, n, Ntx, Nblock, use_low, use_bend, use_high,
                    dtlow, 0.5/dtlow, dtbend, 0.5/dtbend, bend_start,
                    bend_end, high_start, high_end);
        }

        if(Nblock < 2 || (Nblock & (Nblock-1)) != 0) continue;
        if(!wdm_packet_fft_plan_prepare(&plan, Ntx, wdms)) continue;

        plan.data[0] = 0.0;
        for(j=-Ntx/2; j<Ntx/2; j++)
        {
            f = (double)j*plan.dfx+fcenter;
            jj = j+Ntx/2;

            if(jj > 0)
            {
                plan.data[jj] = 0.0;
                plan.data[2*Ntx-jj] = 0.0;

                if(use_low)
                {
                    sample_real_fft_exact(low_hfft, Nlow, dtlow, f,
                                          &re_low, &im_low);
                    phase = 2.0*M_PI*f*t0;
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] += re_low*c-im_low*s;
                    plan.data[2*Ntx-jj] += re_low*s+im_low*c;
                }

                if(use_bend)
                {
                    sample_real_fft_exact(bend_hfft, Nbend, dtbend, f,
                                          &re_bend, &im_bend);
                    phase = 2.0*M_PI*f*(t0-bend_start);
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] += re_bend*c-im_bend*s;
                    plan.data[2*Ntx-jj] += re_bend*s+im_bend*c;
                }

                if(use_high)
                {
                    sample_real_fft_exact(high_hfft, Nhigh, dt, f,
                                          &re_high, &im_high);
                    phase = 2.0*M_PI*f*(t0-high_start);
                    c = cos(phase);
                    s = sin(phase);
                    plan.data[jj] += re_high*c-im_high*s;
                    plan.data[2*Ntx-jj] += re_high*s+im_high*c;
                }

                white = thm_instrument_prewhiten_factor(f);
                plan.data[jj] *= white;
                plan.data[2*Ntx-jj] *= white;
            }
        }

        wdmtranF_plan(m, &plan);

        for(i=0; i<Ntx; i++)
        {
            outn = i+n-Ntx/2;
            if(outn > -1 && outn < Nt)
            {
                wdm[outn][m] = plan.wdmout[i];
                pixel_count++;
            }
        }

        layer_count++;
    }
    timer_end = clock();
    packet_time = ((double)(timer_end-timer_start))/CLOCKS_PER_SEC;

    if(out != NULL) fclose(out);

    printf("split FFT THM %s low block N=%d dt=%e duration=%e Nyquist=%e volume=%d time=%f seconds\n",
           label != NULL ? label : "channel", Nlow, dtlow,
           (double)Nlow*dtlow, 0.5/dtlow, (Nf/Rlow)*Nt,
           low_fft_time);
    printf("split FFT THM %s bend block N=%d dt=%e duration=%e Nyquist=%e start=%e end=%e volume=%d time=%f seconds\n",
           label != NULL ? label : "channel", Nbend, dtbend,
           (double)Nbend*dtbend, 0.5/dtbend, bend_start, bend_end,
           (Nf/Rbend)*Kbend, bend_fft_time);
    printf("split FFT THM %s high block N=%d dt=%e duration=%e Nyquist=%e start=%e end=%e volume=%d time=%f seconds\n",
           label != NULL ? label : "channel", Nhigh, (double)dt,
           (double)Nhigh*dt, 0.5/dt, high_start, high_end,
           Nf*Khigh, high_fft_time);
    printf("split FFT THM %s packet assembly/WDM time=%f seconds\n",
           label != NULL ? label : "channel", packet_time);

    wdm_packet_fft_plan_free(&plan);
    free_double_vector(low_hfft);
    free_double_vector(bend_hfft);
    free_double_vector(high_hfft);
    free_int_vector(nmid);
    free_int_vector(nsize);

    if(layers_used != NULL) *layers_used = layer_count;
    if(pixels_used != NULL) *pixels_used = pixel_count;
}

/*
 Build the direct FFT spectrum used by WDMtrackFFT. This intentionally matches
 Xtime.dat: no Tukey taper is applied here. That keeps the diagnostic reference
 aligned with wd_viafreq Xtime.dat 0 meyer; edge/leakage effects should be
 controlled by match_track time/frequency cuts.
 */
void build_direct_fft_spectrum(double *hfft, int N, double tmax, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline)
{
    int i;
    double t;
    
    for(i=0; i<N; i++)
    {
        t = (double)(i)*dt;
        hfft[i] = 0.0;
        if(t < tmax) hfft[i] = gsl_spline_eval(ASspline, t, ASacc)*cos(gsl_spline_eval(PSspline, t, PSacc));
    }
    
    gsl_fft_real_radix2_transform(hfft, 1, N);
    for(i=0; i<N; i++) hfft[i] *= (2.0*dt);
}

/*
 * Tukey-windowed companion to build_direct_fft_spectrum().
 *
 * This is a diagnostic reference for the Fourier-band and WDM comparisons.
 * The no-Tukey reference exposes the exact finite observation used by the fast
 * path, but its sharp turn-on at t=0 leaks into the low-frequency spectrum.
 * Applying the same Tukey window used for THM_modeXX_Xtime_tukey.dat makes it
 * easier to tell real SPA error from start-edge FFT leakage.
 */
void build_direct_fft_spectrum_tukey(double *hfft, int N, double tmax, double alpha, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline)
{
    int i;
    double t;

    for(i=0; i<N; i++)
    {
        t = (double)(i)*dt;
        hfft[i] = 0.0;
        if(t < tmax)
        {
            hfft[i] = gsl_spline_eval(ASspline, t, ASacc)*
                      cos(gsl_spline_eval(PSspline, t, PSacc));
        }
    }

    tukey(hfft, alpha, N);
    gsl_fft_real_radix2_transform(hfft, 1, N);
    for(i=0; i<N; i++) hfft[i] *= (2.0*dt);
}

/*
 Sample the half-complex GSL real FFT at an arbitrary positive frequency.
 The current WDM settings usually land on integer global FFT bins; the linear
 fallback is only there to make the diagnostic robust if nsize choices change.
 */
void sample_direct_fft_spectrum(double *hfft, int N, double Tobs, double f, double *re, double *im)
{
    int k;
    double bin, frac;
    double re0, im0, re1, im1;
    
    *re = 0.0;
    *im = 0.0;
    
    if(f <= 0.0 || f >= 0.5/dt) return;
    
    bin = f*Tobs;
    k = (int)floor(bin);
    frac = bin-(double)(k);
    
    if(frac > 0.5)
    {
        k++;
        frac = bin-(double)(k);
    }
    
    if(k <= 0 || k >= N/2) return;
    
    if(fabs(frac) < 1.0e-8 || k+1 >= N/2)
    {
        *re = hfft[k];
        *im = hfft[N-k];
        return;
    }
    
    if(frac < 0.0)
    {
        re0 = hfft[k-1];
        im0 = hfft[N-(k-1)];
        re1 = hfft[k];
        im1 = hfft[N-k];
        frac += 1.0;
    }
    else
    {
        re0 = hfft[k];
        im0 = hfft[N-k];
        re1 = hfft[k+1];
        im1 = hfft[N-(k+1)];
    }
    
    *re = (1.0-frac)*re0+frac*re1;
    *im = (1.0-frac)*im0+frac*im1;
}

void write_full_spa_fft_amp_phase_diagnostic(const char *filename, double *hfft, int N, double Tobs, int Nspa, double *time_spa, double *freq_spa, double *fdot_spa, int *valid_spa, double *Amp_spa, double *phase_spa)
{
    int i;
    double f, re_fft, im_fft, amp_fft, phi_fft_raw, phi_spa_eff;
    double phi_fft_aligned, phase_diff, amp_ratio;
    double re_spa, im_spa;
    FILE *out;

    if(filename == NULL || hfft == NULL || Nspa < 1 || time_spa == NULL ||
       freq_spa == NULL || fdot_spa == NULL || valid_spa == NULL ||
       Amp_spa == NULL || phase_spa == NULL)
    {
        return;
    }

    out = fopen(filename, "w");
    if(out == NULL) return;

    fprintf(out, "# f_Hz t_s fdot_Hz_per_s spa_valid_single_branch A_SPA_signed phi_SPA re_SPA im_SPA A_FFT_abs phi_FFT_aligned re_FFT im_FFT phase_SPA_effective_minus_FFT amp_abs_SPA_over_FFT\n");
    fprintf(out, "# The SPA columns use the same signed-amplitude, +pi/4, abs(fdot) convention as ftran().\n");
    fprintf(out, "# spa_valid_single_branch=0 marks samples where the usual single-valued chirp SPA has failed, usually because df/dt <= 0.\n");
    fprintf(out, "# phi_FFT_aligned is the full-observation FFT atan2 phase shifted by 2*pi to the nearest branch of phi_SPA, including a pi shift when A_SPA is negative.\n");

    for(i=0; i<Nspa; i++)
    {
        f = freq_spa[i];
        if(!isfinite(f) || f <= 0.0 || !isfinite(Amp_spa[i]) ||
           !isfinite(phase_spa[i]))
        {
            continue;
        }

        sample_direct_fft_spectrum(hfft, N, Tobs, f, &re_fft, &im_fft);
        amp_fft = sqrt(re_fft*re_fft+im_fft*im_fft);
        phi_fft_raw = atan2(im_fft, re_fft);
        phi_spa_eff = phase_spa[i];
        if(Amp_spa[i] < 0.0) phi_spa_eff += M_PI;
        phi_fft_aligned = phi_fft_raw +
                          2.0*M_PI*rint((phi_spa_eff-phi_fft_raw)/(2.0*M_PI));
        phase_diff = phi_spa_eff-phi_fft_aligned;
        amp_ratio = NAN;
        if(amp_fft > 0.0) amp_ratio = fabs(Amp_spa[i])/amp_fft;
        re_spa = Amp_spa[i]*cos(phase_spa[i]);
        im_spa = Amp_spa[i]*sin(phase_spa[i]);

        fprintf(out, "%.15e %.15e %.15e %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                f, time_spa[i], fdot_spa[i], valid_spa[i],
                Amp_spa[i], phase_spa[i], re_spa, im_spa,
                amp_fft, phi_fft_aligned, re_fft, im_fft,
                phase_diff, amp_ratio);
    }

    fclose(out);
}

void write_dense_spa_fft_amp_phase_diagnostic(const char *filename, double *hfft, int N, double Tobs, int Ns, double *TS, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double fmin, double fmax, double df, double spa_tukey_alpha)
{
    int i, j;
    double f, t, u, f0, f1, fdot_val, fdot_abs, A, phase, Amp, win;
    double re_fft, im_fft, amp_fft, phi_fft_raw, phi_spa_eff;
    double phi_fft_aligned, phase_diff, amp_ratio;
    double re_spa, im_spa;
    double *phase_grid, *freq_grid, *fdot_grid;
    FILE *out;

    if(filename == NULL || hfft == NULL || TS == NULL || ASacc == NULL ||
       ASspline == NULL || PSacc == NULL || PSspline == NULL || Ns < 2 ||
       fmax <= fmin)
    {
        return;
    }

    if(df <= 0.0) df = 1.0/Tobs;

    phase_grid = double_vector(Ns);
    freq_grid = double_vector(Ns);
    fdot_grid = double_vector(Ns);
    if(phase_grid == NULL || freq_grid == NULL || fdot_grid == NULL)
    {
        if(phase_grid != NULL) free_double_vector(phase_grid);
        if(freq_grid != NULL) free_double_vector(freq_grid);
        if(fdot_grid != NULL) free_double_vector(fdot_grid);
        return;
    }

    build_nonuniform_phase_derivatives(Ns, TS, PSacc, PSspline,
                                       phase_grid, freq_grid, fdot_grid);

    out = fopen(filename, "w");
    if(out == NULL)
    {
        free_double_vector(phase_grid);
        free_double_vector(freq_grid);
        free_double_vector(fdot_grid);
        return;
    }

    fprintf(out, "# f_Hz t_s fdot_Hz_per_s spa_valid_single_branch A_SPA_signed phi_SPA re_SPA im_SPA A_FFT_abs phi_FFT_aligned re_FFT im_FFT phase_SPA_effective_minus_FFT amp_abs_SPA_over_FFT\n");
    fprintf(out, "# Dense SPA diagnostic: stationary points are linearly interpolated from the local nonuniform finite-difference frequency track.\n");
    fprintf(out, "# The frequency step is %.15e Hz; for the current run this is usually the native full-FFT bin spacing 1/Tobs.\n", df);
    fprintf(out, "# This is diagnostic only: it intentionally evaluates the leading SPA between AP samples to expose smooth structure hidden by sparse frequency sampling.\n");
    if(spa_tukey_alpha > 0.0)
    {
        fprintf(out, "# The SPA amplitude is multiplied by the same Tukey time-domain window used for the FFT reference, evaluated at the stationary time.\n");
    }

    j = 0;
    for(f=fmin; f <= fmax+0.5*df; f += df)
    {
        int valid;

        while(j < Ns-2 &&
              (!isfinite(freq_grid[j]) || !isfinite(freq_grid[j+1]) ||
               freq_grid[j+1] <= freq_grid[j] || freq_grid[j+1] < f))
        {
            j++;
        }

        if(j >= Ns-1 || !isfinite(freq_grid[j]) || !isfinite(freq_grid[j+1]) ||
           freq_grid[j+1] <= freq_grid[j] || f < freq_grid[j] ||
           f > freq_grid[j+1])
        {
            continue;
        }

        f0 = freq_grid[j];
        f1 = freq_grid[j+1];
        u = (f-f0)/(f1-f0);
        if(u < 0.0) u = 0.0;
        if(u > 1.0) u = 1.0;

        t = TS[j]+u*(TS[j+1]-TS[j]);
        fdot_val = fdot_grid[j]+u*(fdot_grid[j+1]-fdot_grid[j]);
        fdot_abs = fabs(fdot_val);
        valid = (isfinite(fdot_val) && fdot_val > 0.0);
        if(!isfinite(fdot_abs) || fdot_abs <= 0.0) continue;

        A = gsl_spline_eval(ASspline, t, ASacc);
        win = tukey_weight_at_time(t, spa_tukey_alpha, N);
        phase = gsl_spline_eval(PSspline, t, PSacc) -
                2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        Amp = win*A*sqrt(1.0/fdot_abs);

        sample_direct_fft_spectrum(hfft, N, Tobs, f, &re_fft, &im_fft);
        amp_fft = sqrt(re_fft*re_fft+im_fft*im_fft);
        phi_fft_raw = atan2(im_fft, re_fft);
        phi_spa_eff = phase;
        if(Amp < 0.0) phi_spa_eff += M_PI;
        phi_fft_aligned = phi_fft_raw +
                          2.0*M_PI*rint((phi_spa_eff-phi_fft_raw)/(2.0*M_PI));
        phase_diff = phi_spa_eff-phi_fft_aligned;
        amp_ratio = NAN;
        if(amp_fft > 0.0) amp_ratio = fabs(Amp)/amp_fft;
        re_spa = Amp*cos(phase);
        im_spa = Amp*sin(phase);

        fprintf(out, "%.15e %.15e %.15e %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                f, t, fdot_val, valid, Amp, phase, re_spa, im_spa,
                amp_fft, phi_fft_aligned, re_fft, im_fft,
                phase_diff, amp_ratio);
    }

    fclose(out);
    free_double_vector(phase_grid);
    free_double_vector(freq_grid);
    free_double_vector(fdot_grid);
}

void write_dense_spa_fft_amp_phase_diagnostic_from_tracks(const char *filename, double *hfft, int N, double Tobs, int Ns, double *TS, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, const double *freq_grid, const double *fdot_grid, double fmin, double fmax, double df, double spa_tukey_alpha, const char *track_description)
{
    int j;
    double f, t, u, f0, f1, fdot_val, fdot_abs, A, phase, Amp, win;
    double re_fft, im_fft, amp_fft, phi_fft_raw, phi_spa_eff;
    double phi_fft_aligned, phase_diff, amp_ratio;
    double re_spa, im_spa;
    FILE *out;

    if(filename == NULL || hfft == NULL || TS == NULL || ASacc == NULL ||
       ASspline == NULL || PSacc == NULL || PSspline == NULL ||
       freq_grid == NULL || fdot_grid == NULL || Ns < 2 || fmax <= fmin)
    {
        return;
    }

    if(df <= 0.0) df = 1.0/Tobs;

    out = fopen(filename, "w");
    if(out == NULL)
    {
        return;
    }

    fprintf(out, "# f_Hz t_s fdot_Hz_per_s spa_valid_single_branch A_SPA_signed phi_SPA re_SPA im_SPA A_FFT_abs phi_FFT_aligned re_FFT im_FFT phase_SPA_effective_minus_FFT amp_abs_SPA_over_FFT\n");
    fprintf(out, "# Dense SPA diagnostic using caller-supplied frequency and fdot tracks.\n");
    if(track_description != NULL) fprintf(out, "# track: %s\n", track_description);
    fprintf(out, "# The frequency step is %.15e Hz; for the current run this is usually the native full-FFT bin spacing 1/Tobs.\n", df);
    if(spa_tukey_alpha > 0.0)
    {
        fprintf(out, "# The SPA amplitude is multiplied by the same Tukey time-domain window used for the FFT reference, evaluated at the stationary time.\n");
    }

    j = 0;
    for(f=fmin; f <= fmax+0.5*df; f += df)
    {
        int valid;

        while(j < Ns-2 &&
              (!isfinite(freq_grid[j]) || !isfinite(freq_grid[j+1]) ||
               freq_grid[j+1] <= freq_grid[j] || freq_grid[j+1] < f))
        {
            j++;
        }

        if(j >= Ns-1 || !isfinite(freq_grid[j]) || !isfinite(freq_grid[j+1]) ||
           freq_grid[j+1] <= freq_grid[j] || f < freq_grid[j] ||
           f > freq_grid[j+1])
        {
            continue;
        }

        f0 = freq_grid[j];
        f1 = freq_grid[j+1];
        u = (f-f0)/(f1-f0);
        if(u < 0.0) u = 0.0;
        if(u > 1.0) u = 1.0;

        t = TS[j]+u*(TS[j+1]-TS[j]);
        fdot_val = fdot_grid[j]+u*(fdot_grid[j+1]-fdot_grid[j]);
        fdot_abs = fabs(fdot_val);
        valid = (isfinite(fdot_val) && fdot_val > 0.0);
        if(!isfinite(fdot_abs) || fdot_abs <= 0.0) continue;

        A = gsl_spline_eval(ASspline, t, ASacc);
        win = tukey_weight_at_time(t, spa_tukey_alpha, N);
        phase = gsl_spline_eval(PSspline, t, PSacc) -
                2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        Amp = win*A*sqrt(1.0/fdot_abs);

        sample_direct_fft_spectrum(hfft, N, Tobs, f, &re_fft, &im_fft);
        amp_fft = sqrt(re_fft*re_fft+im_fft*im_fft);
        phi_fft_raw = atan2(im_fft, re_fft);
        phi_spa_eff = phase;
        if(Amp < 0.0) phi_spa_eff += M_PI;
        phi_fft_aligned = phi_fft_raw +
                          2.0*M_PI*rint((phi_spa_eff-phi_fft_raw)/(2.0*M_PI));
        phase_diff = phi_spa_eff-phi_fft_aligned;
        amp_ratio = NAN;
        if(amp_fft > 0.0) amp_ratio = fabs(Amp)/amp_fft;
        re_spa = Amp*cos(phase);
        im_spa = Amp*sin(phase);

        fprintf(out, "%.15e %.15e %.15e %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                f, t, fdot_val, valid, Amp, phase, re_spa, im_spa,
                amp_fft, phi_fft_aligned, re_fft, im_fft,
                phase_diff, amp_ratio);
    }

    fclose(out);
}

void write_fourier_band_comparison(const char *filename, double *hfft, int N, double Tobs, gsl_interp_accel *AFacc, gsl_spline *AFspline, gsl_interp_accel *PFacc, gsl_spline *PFspline, double fmin, double fmax, const char *test_name, struct wdmshape *wdms)
{
    int i, k, m, nbins;
    int *bins, *support_bins;
    double f, A, P, re_ref, im_ref, re_test, im_test;
    double *ref_power, *test_power, *cross;
    double total_ref, total_test, total_cross, total_residual;
    FILE *out;

    if(filename == NULL || hfft == NULL || AFacc == NULL || AFspline == NULL ||
       PFacc == NULL || PFspline == NULL || wdms == NULL || N < 4 ||
       Tobs <= 0.0 || fmax <= fmin)
    {
        return;
    }

    ref_power = double_vector(Nf+1);
    test_power = double_vector(Nf+1);
    cross = double_vector(Nf+1);
    bins = int_vector(Nf+1);
    support_bins = int_vector(Nf+1);

    for(i=0; i<=Nf; i++)
    {
        ref_power[i] = 0.0;
        test_power[i] = 0.0;
        cross[i] = 0.0;
        bins[i] = 0;
        support_bins[i] = 0;
    }

    nbins = N/2;
    total_ref = 0.0;
    total_test = 0.0;
    total_cross = 0.0;
    for(k=1; k<nbins; k++)
    {
        f = (double)k/Tobs;
        m = (int)floor(f/wdms->DF+0.5);
        if(m < 1 || m >= Nf) continue;

        re_ref = hfft[k];
        im_ref = hfft[N-k];
        re_test = 0.0;
        im_test = 0.0;
        if(f > fmin && f < fmax)
        {
            A = gsl_spline_eval(AFspline, f, AFacc);
            P = gsl_spline_eval(PFspline, f, PFacc);
            if(isfinite(A) && isfinite(P))
            {
                re_test = A*cos(P);
                im_test = A*sin(P);
                support_bins[m]++;
            }
        }

        bins[m]++;
        ref_power[m] += re_ref*re_ref+im_ref*im_ref;
        test_power[m] += re_test*re_test+im_test*im_test;
        cross[m] += re_ref*re_test+im_ref*im_test;
    }

    for(m=1; m<Nf; m++)
    {
        total_ref += ref_power[m];
        total_test += test_power[m];
        total_cross += cross[m];
    }
    total_residual = total_ref+total_test-2.0*total_cross;

    out = fopen(filename, "w");
    if(out == NULL)
    {
        free_double_vector(ref_power);
        free_double_vector(test_power);
        free_double_vector(cross);
        free_int_vector(bins);
        free_int_vector(support_bins);
        return;
    }

    fprintf(out, "# reference full_observation_fft\n");
    fprintf(out, "# test %s\n", test_name == NULL ? "frequency_domain_input" : test_name);
    fprintf(out, "# support_frequency_range %.15e %.15e\n", fmin, fmax);
    fprintf(out, "# strict WDM-layer bands are [(m-0.5)*DF,(m+0.5)*DF); Meyer filters extend beyond these bands.\n");
    fprintf(out, "# DF %.15e FB %.15e global_df %.15e bins_per_DF %.15e\n",
            wdms->DF, wdms->FB, 1.0/Tobs, wdms->DF*Tobs);
    if(total_ref > 0.0 && total_test > 0.0)
    {
        double match = total_cross/sqrt(total_ref*total_test);
        fprintf(out, "# total_ref_norm %.15e total_test_norm %.15e total_match %.15e total_mismatch %.15e total_residual_fraction %.15e\n",
                sqrt(total_ref), sqrt(total_test), match, 1.0-match,
                total_residual/total_ref);
    }
    else
    {
        fprintf(out, "# total_ref_norm %.15e total_test_norm %.15e total_match nan total_mismatch nan total_residual_fraction nan\n",
                sqrt(total_ref), sqrt(total_test));
    }
    fprintf(out, "# m f_low_Hz f_center_Hz f_high_Hz bins support_bins ref_norm test_norm match mismatch ref_power_fraction test_power_fraction residual_power_fraction cross\n");

    for(m=1; m<Nf; m++)
    {
        double match, mismatch, residual;

        match = NAN;
        mismatch = NAN;
        if(ref_power[m] > 0.0 && test_power[m] > 0.0)
        {
            match = cross[m]/sqrt(ref_power[m]*test_power[m]);
            mismatch = 1.0-match;
        }
        residual = ref_power[m]+test_power[m]-2.0*cross[m];

        fprintf(out, "%d %.15e %.15e %.15e %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                m, ((double)m-0.5)*wdms->DF, (double)m*wdms->DF,
                ((double)m+0.5)*wdms->DF, bins[m], support_bins[m],
                sqrt(ref_power[m]), sqrt(test_power[m]), match, mismatch,
                total_ref > 0.0 ? ref_power[m]/total_ref : NAN,
                total_test > 0.0 ? test_power[m]/total_test : NAN,
                total_ref > 0.0 ? residual/total_ref : NAN,
                cross[m]);
    }

    fclose(out);

    free_double_vector(ref_power);
    free_double_vector(test_power);
    free_double_vector(cross);
    free_int_vector(bins);
    free_int_vector(support_bins);
}

void write_first_order_spa_fourier_diagnostic(const char *label, const char *suffix, const char *description, double correction_sign, int correction_mask, int Ns, double *TS, double Tobs, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *phase0_all, double *hfft, double *hfft_tukey, int N, struct wdmshape *wdms)
{
    double *freq_all, *phase_all, *Amp_all, *time_all, *fdot_all, *delta_all, *rel_all;
    double *freq, *phase, *Amp, *delta, *rel;
    int *valid_all;
    int Nraw, Nout, i;
    double theta_prev, rel_abs_max;
    int have_theta, rel_gt_1, rel_gt_0p1, rel_gt_0p01;
    char filename[256];
    FILE *out;
    gsl_interp_accel *AFacc, *PFacc;
    gsl_spline *AFspline, *PFspline;

    if(label == NULL || suffix == NULL || Ns < 4 || TS == NULL ||
       ASacc == NULL || ASspline == NULL || PSacc == NULL ||
       PSspline == NULL || phase0_all == NULL || hfft == NULL ||
       wdms == NULL || N < 4)
    {
        return;
    }

    freq_all = double_vector(Ns);
    phase_all = double_vector(Ns);
    Amp_all = double_vector(Ns);
    time_all = double_vector(Ns);
    fdot_all = double_vector(Ns);
    delta_all = double_vector(Ns);
    rel_all = double_vector(Ns);
    valid_all = int_vector(Ns);
    freq = double_vector(Ns);
    phase = double_vector(Ns);
    Amp = double_vector(Ns);
    delta = double_vector(Ns);
    rel = double_vector(Ns);

    Nraw = build_first_order_spa_samples(TS, Ns, Tobs,
                                         ASacc, ASspline, PSacc, PSspline,
                                         correction_sign, correction_mask,
                                         freq_all, phase_all, Amp_all,
                                         time_all, fdot_all, delta_all,
                                         rel_all, valid_all);

    Nout = 0;
    theta_prev = 0.0;
    have_theta = 0;
    rel_abs_max = 0.0;
    rel_gt_1 = 0;
    rel_gt_0p1 = 0;
    rel_gt_0p01 = 0;
    for(i=0; i<Nraw; i++)
    {
        if(valid_all[i] &&
           isfinite(freq_all[i]) &&
           isfinite(phase_all[i]) &&
           isfinite(Amp_all[i]) &&
           isfinite(phase0_all[i]) &&
           (Nout == 0 || freq_all[i] > freq[Nout-1]))
        {
            double theta, rel_abs;

            theta = phase_all[i]-phase0_all[i];
            if(!isfinite(theta)) continue;
            if(have_theta)
            {
                while(theta-theta_prev > M_PI) theta -= 2.0*M_PI;
                while(theta-theta_prev < -M_PI) theta += 2.0*M_PI;
            }

            freq[Nout] = freq_all[i];
            phase[Nout] = phase0_all[i]+theta;
            Amp[Nout] = Amp_all[i];
            delta[Nout] = delta_all[i];
            rel[Nout] = rel_all[i];

            if(isfinite(rel_all[i]))
            {
                rel_abs = fabs(rel_all[i]);
                if(rel_abs > rel_abs_max) rel_abs_max = rel_abs;
                if(rel_abs > 1.0) rel_gt_1++;
                if(rel_abs > 0.1) rel_gt_0p1++;
                if(rel_abs > 0.01) rel_gt_0p01++;
            }

            theta_prev = theta;
            have_theta = 1;
            Nout++;
        }
    }

    snprintf(filename, sizeof(filename), "ch_THM_%s_%s.dat", label, suffix);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# f_Hz phase amplitude delta1 delta1_over_A\n");
        fprintf(out, "# %s\n", description == NULL ? suffix : description);
        fprintf(out, "# correction_mask %d correction_sign %.15e\n",
                correction_mask, correction_sign);
        for(i=0; i<Nout; i++)
        {
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e\n",
                    freq[i], phase[i], Amp[i], delta[i], rel[i]);
        }
        fclose(out);
    }

    printf("spa_delta1_diagnostic mode %s suffix %s mask %d sign %.1f samples_all %d monotone_valid %d rel_abs_max %.6e rel_gt_1 %d rel_gt_0p1 %d rel_gt_0p01 %d\n",
           label, suffix, correction_mask, correction_sign, Nraw, Nout,
           rel_abs_max, rel_gt_1, rel_gt_0p1, rel_gt_0p01);

    if(Nout >= 4)
    {
        AFacc = gsl_interp_accel_alloc();
        PFacc = gsl_interp_accel_alloc();
        AFspline = gsl_spline_alloc(gsl_interp_akima, Nout);
        PFspline = gsl_spline_alloc(gsl_interp_cspline, Nout);
        gsl_spline_init(AFspline, freq, Amp, Nout);
        gsl_spline_init(PFspline, freq, phase, Nout);

        snprintf(filename, sizeof(filename), "THM_mode%s_%s_fourier_band_match_X.dat",
                 label, suffix);
        write_fourier_band_comparison(filename, hfft, N, Tobs,
                                      AFacc, AFspline, PFacc, PFspline,
                                      freq[0], freq[Nout-1],
                                      description == NULL ? suffix : description,
                                      wdms);
        if(hfft_tukey != NULL)
        {
            snprintf(filename, sizeof(filename), "THM_mode%s_%s_fourier_band_match_tukeyref_X.dat",
                     label, suffix);
            write_fourier_band_comparison(filename, hfft_tukey, N, Tobs,
                                          AFacc, AFspline, PFacc, PFspline,
                                          freq[0], freq[Nout-1],
                                          description == NULL ? suffix : description,
                                          wdms);
        }

        gsl_spline_free(AFspline);
        gsl_spline_free(PFspline);
        gsl_interp_accel_free(AFacc);
        gsl_interp_accel_free(PFacc);
    }
    else
    {
        fprintf(stderr, "Warning: SPA correction diagnostic %s for mode %s has only %d monotone samples; skipping Fourier-band output.\n",
                suffix, label, Nout);
    }

    free_double_vector(freq_all);
    free_double_vector(phase_all);
    free_double_vector(Amp_all);
    free_double_vector(time_all);
    free_double_vector(fdot_all);
    free_double_vector(delta_all);
    free_double_vector(rel_all);
    free_int_vector(valid_all);
    free_double_vector(freq);
    free_double_vector(phase);
    free_double_vector(Amp);
    free_double_vector(delta);
    free_double_vector(rel);
}

void write_wdm_layer_comparison(const char *filename, double **reference, double **test, const char *reference_name, const char *test_name, struct wdmshape *wdms, int jmin, int jmax)
{
    int i, j;
    double ref_total, test_total, cross_total;
    FILE *out;

    if(filename == NULL || reference == NULL || test == NULL || wdms == NULL)
    {
        return;
    }

    if(jmin < 0) jmin = 0;
    if(jmax > Nt) jmax = Nt;
    if(jmax <= jmin) return;

    ref_total = 0.0;
    test_total = 0.0;
    cross_total = 0.0;
    for(j=jmin; j<jmax; j++)
    {
        for(i=1; i<Nf; i++)
        {
            ref_total += reference[j][i]*reference[j][i];
            test_total += test[j][i]*test[j][i];
            cross_total += reference[j][i]*test[j][i];
        }
    }

    out = fopen(filename, "w");
    if(out == NULL) return;

    fprintf(out, "# reference %s\n", reference_name == NULL ? "reference" : reference_name);
    fprintf(out, "# test %s\n", test_name == NULL ? "test" : test_name);
    fprintf(out, "# time_pixel_range [%d,%d)\n", jmin, jmax);
    if(ref_total > 0.0 && test_total > 0.0)
    {
        double match = cross_total/sqrt(ref_total*test_total);
        fprintf(out, "# total_ref_norm %.15e total_test_norm %.15e total_match %.15e total_mismatch %.15e\n",
                sqrt(ref_total), sqrt(test_total), match, 1.0-match);
    }
    else
    {
        fprintf(out, "# total_ref_norm %.15e total_test_norm %.15e total_match nan total_mismatch nan\n",
                sqrt(ref_total), sqrt(test_total));
    }
    fprintf(out, "# m f_Hz ref_norm test_norm match mismatch ref_power_fraction test_power_fraction cross\n");

    for(i=1; i<Nf; i++)
    {
        double x, y, z, match, mismatch;

        x = 0.0;
        y = 0.0;
        z = 0.0;
        for(j=jmin; j<jmax; j++)
        {
            x += reference[j][i]*reference[j][i];
            y += test[j][i]*test[j][i];
            z += reference[j][i]*test[j][i];
        }

        match = NAN;
        mismatch = NAN;
        if(x > 0.0 && y > 0.0)
        {
            match = z/sqrt(x*y);
            mismatch = 1.0-match;
        }

        fprintf(out, "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                i, (double)i*wdms->DF, sqrt(x), sqrt(y), match, mismatch,
                ref_total > 0.0 ? x/ref_total : NAN,
                test_total > 0.0 ? y/test_total : NAN,
                z);
    }

    fclose(out);
}

double split_smooth_step(double t, double t1, double t2)
{
    double x;

    if(t <= t1) return 0.0;
    if(t >= t2) return 1.0;
    
    x = (t-t1)/(t2-t1);
    return 0.5*(1.0-cos(M_PI*x));
}

double linear_interp_clamped(int N, const double *x, const double *y, double xq)
{
    int lo, hi, mid;
    double dx, u;

    if(N < 1 || x == NULL || y == NULL) return NAN;
    if(N == 1 || xq <= x[0]) return y[0];
    if(xq >= x[N-1]) return y[N-1];

    lo = 0;
    hi = N-1;
    while(hi-lo > 1)
    {
        mid = lo+(hi-lo)/2;
        if(x[mid] <= xq) lo = mid;
        else hi = mid;
    }

    dx = x[hi]-x[lo];
    if(dx <= 0.0) return y[lo];
    u = (xq-x[lo])/dx;
    return (1.0-u)*y[lo]+u*y[hi];
}

void split_fft_weights(double t, double bend_start, double high_start, double roll, double *wlow, double *wbend, double *whigh)
{
    double sbend, shigh;
    
    sbend = split_smooth_step(t, bend_start, bend_start+roll);
    shigh = split_smooth_step(t, high_start, high_start+roll);
    
    *wlow = 1.0-sbend;
    *wbend = sbend*(1.0-shigh);
    *whigh = shigh;
}

void build_split_low_fft(double *hfft, int Nlow, double dtlow, double tmax, double bend_start, double high_start, double roll, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline)
{
    int i;
    double t, wlow, wbend, whigh;
    
    for(i=0; i<Nlow; i++)
    {
        t = (double)i*dtlow;
        hfft[i] = 0.0;
        if(t >= 0.0 && t < tmax)
        {
            split_fft_weights(t, bend_start, high_start, roll, &wlow, &wbend, &whigh);
            if(wlow > 0.0) hfft[i] = wlow*gsl_spline_eval(ASspline, t, ASacc)*cos(gsl_spline_eval(PSspline, t, PSacc));
        }
    }
    
    gsl_fft_real_radix2_transform(hfft, 1, Nlow);
    for(i=0; i<Nlow; i++) hfft[i] *= (2.0*dtlow);
}

void build_split_local_fft(double *hfft, int Nblock, double sample_dt, double block_start, double tmax, double bend_start, double high_start, double roll, int block_id, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline)
{
    int i;
    double t, wlow, wbend, whigh, weight;
    
    for(i=0; i<Nblock; i++)
    {
        t = block_start+(double)i*sample_dt;
        hfft[i] = 0.0;
        if(t >= 0.0 && t < tmax)
        {
            split_fft_weights(t, bend_start, high_start, roll, &wlow, &wbend, &whigh);
            weight = 0.0;
            if(block_id == 1) weight = wbend;
            if(block_id == 2) weight = whigh;
            if(weight > 0.0)
            {
                hfft[i] = weight*gsl_spline_eval(ASspline, t, ASacc)*cos(gsl_spline_eval(PSspline, t, PSacc));
            }
        }
    }
    
    gsl_fft_real_radix2_transform(hfft, 1, Nblock);
    for(i=0; i<Nblock; i++) hfft[i] *= (2.0*sample_dt);
}

void build_split_low_fft_sum(double *hfft, int Nlow, double dtlow, double tmin, double tmax, double bend_start, double high_start, double roll, int ncarriers, int channel, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline)
{
    int i, k, idx;
    double t, h, wlow, wbend, whigh;

    for(i=0; i<Nlow; i++)
    {
        t = (double)i*dtlow;
        hfft[i] = 0.0;
        if(t >= tmin && t < tmax)
        {
            split_fft_weights(t, bend_start, high_start, roll,
                              &wlow, &wbend, &whigh);
            if(wlow > 0.0)
            {
                h = 0.0;
                for(k=0; k<ncarriers; k++)
                {
                    idx = channel*ncarriers+k;
                    if(ATspline[idx] == NULL || PTspline[idx] == NULL) continue;
                    h += gsl_spline_eval(ATspline[idx], t, ATacc[idx])*
                         cos(gsl_spline_eval(PTspline[idx], t, PTacc[idx]));
                }
                hfft[i] = wlow*h;
            }
        }
    }

    gsl_fft_real_radix2_transform(hfft, 1, Nlow);
    for(i=0; i<Nlow; i++) hfft[i] *= (2.0*dtlow);
}

void build_split_local_fft_sum(double *hfft, int Nblock, double sample_dt, double block_start, double tmin, double tmax, double bend_start, double high_start, double roll, int block_id, int ncarriers, int channel, gsl_interp_accel **ATacc, gsl_spline **ATspline, gsl_interp_accel **PTacc, gsl_spline **PTspline)
{
    int i, k, idx;
    double t, h, wlow, wbend, whigh, weight;

    for(i=0; i<Nblock; i++)
    {
        t = block_start+(double)i*sample_dt;
        hfft[i] = 0.0;
        if(t >= tmin && t < tmax)
        {
            split_fft_weights(t, bend_start, high_start, roll,
                              &wlow, &wbend, &whigh);
            weight = 0.0;
            if(block_id == 1) weight = wbend;
            if(block_id == 2) weight = whigh;
            if(weight > 0.0)
            {
                h = 0.0;
                for(k=0; k<ncarriers; k++)
                {
                    idx = channel*ncarriers+k;
                    if(ATspline[idx] == NULL || PTspline[idx] == NULL) continue;
                    h += gsl_spline_eval(ATspline[idx], t, ATacc[idx])*
                         cos(gsl_spline_eval(PTspline[idx], t, PTacc[idx]));
                }
                hfft[i] = weight*h;
            }
        }
    }

    gsl_fft_real_radix2_transform(hfft, 1, Nblock);
    for(i=0; i<Nblock; i++) hfft[i] *= (2.0*sample_dt);
}

void sample_real_fft_exact(double *hfft, int N, double sample_dt, double f, double *re, double *im)
{
    int k;
    double Tspan, bin;
    
    *re = 0.0;
    *im = 0.0;
    
    if(f <= 0.0 || f >= 0.5/sample_dt) return;
    
    Tspan = (double)N*sample_dt;
    bin = f*Tspan;
    k = (int)llround(bin);
    
    if(k <= 0 || k >= N/2) return;
    if(fabs(bin-(double)k) > 1.0e-6) return;
    
    *re = hfft[k];
    *im = hfft[N-k];
}

void sample_complex_fft_exact(double *zfft, int N, double sample_dt, double f, double *re, double *im)
{
    int k;
    double Tspan, bin, nyquist;

    *re = 0.0;
    *im = 0.0;

    if(zfft == NULL || N < 2 || sample_dt <= 0.0) return;

    nyquist = 0.5/sample_dt;
    if(f <= -nyquist || f >= nyquist) return;

    Tspan = (double)N*sample_dt;
    bin = f*Tspan;
    k = (int)llround(bin);
    if(fabs(bin-(double)k) > 1.0e-6) return;

    while(k < 0) k += N;
    while(k >= N) k -= N;

    *re = REAL(zfft,k);
    *im = IMAG(zfft,k);
}

void sample_short_window_spectrum(double *hfft, int N, double dte, double f, double *re, double *im)
{
    int k;
    double Tspan, bin;
    
    *re = 0.0;
    *im = 0.0;
    
    if(f <= 0.0 || f >= 0.5/dte) return;
    
    Tspan = (double)(N)*dte;
    bin = f*Tspan;
    k = (int)llround(bin);
    
    if(k <= 0 || k >= N/2) return;
    if(fabs(bin-(double)k) > 1.0e-8) return;
    
    *re = hfft[k];
    *im = hfft[N-k];
}

void write_fft_amp_phase_diagnostic(const char *filename, double *hfft, int N, double Tobs, double *freq, int Nts, gsl_interp_accel *AFacc, gsl_spline *AFspline, gsl_interp_accel *PFacc, gsl_spline *PFspline)
{
    int i, j, nbins, ilo, ihi, istart, iend, prev, next, sign_change;
    double f, re, im;
    double amp_fft, phi_raw, phi_raw_prev, phi_unwrapped, phi_unwrapped_signed, phi_compare;
    double phi_spa, amp_spa;
    double phase_offset, delta, local_scale, floor, diff_prev, diff_next, diff, frac;
    double *fout, *amp_fft_out, *phi_branch_out, *phi_branch_direct;
    double *phi_unwrapped_arr, *phi_raw_out, *phi_spa_out, *amp_spa_out;
    double *re_out, *im_out;
    int *has_spa, *phase_valid;
    FILE *out;
    
    /*
     Columns:
       f, A_FFT, phi_FFT_branch, phi_FFT_nearest_unwrap,
       phi_FFT_raw, phi_SPA, A_SPA, Re_FFT, Im_FFT,
       phase_direct_valid, phi_FFT_branch_direct

     The primary phase column uses the same signed-amplitude convention as the
     SPA/short-FFT path: if A_SPA is negative, A_FFT is reported negative and
     pi is added to the FFT phase before choosing the 2*pi branch closest to
     phi_SPA. This is the frequency-domain analogue of the sign-flip treatment
     in extractAP(), where the TDI time-domain quadratures are allowed to have
     signed amplitudes and the phase is shifted by pi across amplitude zeros.
     Outside the SPA frequency range it falls back to nearest-neighbor
     unwrapping.

     At TDI response nodes the complex Fourier amplitude passes very close to
     zero, so the raw atan2 phase is ill-conditioned: tiny numerical changes in
     Re/Im can move the phase by an order-one amount without representing a
     physical phase error. Those bins are identified using a local
     signed-amplitude floor and the primary phase is interpolated through the
     node. The direct/non-interpolated phase is retained in the last column, and
     phase_direct_valid is 0 for interpolated node bins.

     The nearest-neighbor unwrap is also printed because it is independent of
     the SPA phase, but it can miss a 2*pi branch when the Fourier phase
     advances by more than pi per FFT bin.
     */
    
    nbins = N/2;
    fout = double_vector(nbins);
    amp_fft_out = double_vector(nbins);
    phi_branch_out = double_vector(nbins);
    phi_branch_direct = double_vector(nbins);
    phi_unwrapped_arr = double_vector(nbins);
    phi_raw_out = double_vector(nbins);
    phi_spa_out = double_vector(nbins);
    amp_spa_out = double_vector(nbins);
    re_out = double_vector(nbins);
    im_out = double_vector(nbins);
    has_spa = int_vector(nbins);
    phase_valid = int_vector(nbins);
    
    phase_offset = 0.0;
    phi_raw_prev = atan2(hfft[N-1], hfft[1]);
    
    for(i=0; i<nbins; i++)
    {
        fout[i] = NAN;
        amp_fft_out[i] = NAN;
        phi_branch_out[i] = NAN;
        phi_branch_direct[i] = NAN;
        phi_unwrapped_arr[i] = NAN;
        phi_raw_out[i] = NAN;
        phi_spa_out[i] = NAN;
        amp_spa_out[i] = NAN;
        re_out[i] = NAN;
        im_out[i] = NAN;
        has_spa[i] = 0;
        phase_valid[i] = 1;
    }
    
    for(i=1; i<nbins; i++)
    {
        f = (double)(i)/Tobs;
        re = hfft[i];
        im = hfft[N-i];
        amp_fft = sqrt(re*re+im*im);
        
        phi_raw = atan2(im, re);
        delta = phi_raw-phi_raw_prev;
        if(delta > M_PI) phase_offset -= 2.0*M_PI;
        if(delta < -M_PI) phase_offset += 2.0*M_PI;
        phi_raw_prev = phi_raw;
        phi_unwrapped = phi_raw+phase_offset;
        
        phi_spa = NAN;
        amp_spa = NAN;
        phi_unwrapped_signed = phi_unwrapped;
        phi_compare = phi_raw;
        if(f > freq[0] && f < freq[Nts-1])
        {
            phi_spa = gsl_spline_eval(PFspline, f, PFacc);
            amp_spa = gsl_spline_eval(AFspline, f, AFacc);
            has_spa[i] = 1;
            if(amp_spa < 0.0)
            {
                amp_fft *= -1.0;
                phi_compare += M_PI;
                phi_unwrapped_signed += M_PI;
            }
            phi_branch_direct[i] = phi_compare+2.0*M_PI*round((phi_spa-phi_compare)/(2.0*M_PI));
        }
        else
        {
            phi_branch_direct[i] = phi_unwrapped;
        }
        
        fout[i] = f;
        amp_fft_out[i] = amp_fft;
        phi_branch_out[i] = phi_branch_direct[i];
        phi_unwrapped_arr[i] = phi_unwrapped_signed;
        phi_raw_out[i] = phi_raw;
        phi_spa_out[i] = phi_spa;
        amp_spa_out[i] = amp_spa;
        re_out[i] = re;
        im_out[i] = im;
    }
    
    /*
     Near a signed-amplitude node, atan2 is dominated by tiny numerical
     differences in the FFT real/imaginary parts. This mirrors the issue handled
     in extractAP() for the TDI time-domain amplitude/phase extraction.

     The notion of "small amplitude" is intentionally local. We look in a
     frequency window around each candidate point, find the typical nearby
     signed-amplitude scale from max(|A_SPA|), and flag only samples below
     FFT_DIAGNOSTIC_NODE_FLOOR_FRACTION of that local scale when there is an
     actual sign change in the window. A fixed absolute amplitude threshold
     would fail as soon as the source distance, masses, orientation, or TDI
     response changed.

     For flagged bins, interpolate the plotted phase residual through the node.
     This keeps column 3 useful for phase comparisons while preserving the raw
     direct branch in the final column.
     */
    for(i=1; i<nbins-1; i++)
    {
        if(!has_spa[i]) continue;
        
        ilo = i-32;
        ihi = i+32;
        if(ilo < 1) ilo = 1;
        if(ihi > nbins-1) ihi = nbins-1;
        
        local_scale = 0.0;
        sign_change = 0;
        for(j=ilo; j<=ihi; j++)
        {
            if(has_spa[j] && fabs(amp_spa_out[j]) > local_scale) local_scale = fabs(amp_spa_out[j]);
            if(j < ihi && has_spa[j] && has_spa[j+1] && amp_spa_out[j]*amp_spa_out[j+1] <= 0.0) sign_change = 1;
        }
        
        floor = FFT_DIAGNOSTIC_NODE_FLOOR_FRACTION*local_scale;
        if(sign_change && local_scale > 0.0 && fabs(amp_spa_out[i]) < floor)
        {
            phase_valid[i] = 0;
        }
    }
    
    i = 1;
    while(i < nbins)
    {
        if(phase_valid[i] || !has_spa[i])
        {
            i++;
            continue;
        }
        
        istart = i;
        while(i < nbins && !phase_valid[i] && has_spa[i]) i++;
        iend = i-1;
        
        prev = istart-1;
        while(prev >= 1 && (!phase_valid[prev] || !has_spa[prev] || !isfinite(phi_branch_out[prev]))) prev--;
        next = iend+1;
        while(next < nbins && (!phase_valid[next] || !has_spa[next] || !isfinite(phi_branch_out[next]))) next++;
        
        if(prev >= 1 && next < nbins && fout[next] != fout[prev])
        {
            diff_prev = phi_branch_out[prev]-phi_spa_out[prev];
            diff_next = phi_branch_out[next]-phi_spa_out[next];
            diff_next += 2.0*M_PI*round((diff_prev-diff_next)/(2.0*M_PI));
            for(j=istart; j<=iend; j++)
            {
                frac = (fout[j]-fout[prev])/(fout[next]-fout[prev]);
                diff = (1.0-frac)*diff_prev+frac*diff_next;
                phi_branch_out[j] = phi_spa_out[j]+diff;
            }
        }
        else
        {
            for(j=istart; j<=iend; j++) phi_branch_out[j] = NAN;
        }
    }
    
    out = fopen(filename, "w");
    
    for(i=1; i<nbins; i++)
    {
        
        fprintf(out, "%e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %d %.15e\n",
                fout[i], amp_fft_out[i], phi_branch_out[i], phi_unwrapped_arr[i],
                phi_raw_out[i], phi_spa_out[i], amp_spa_out[i], re_out[i], im_out[i],
                phase_valid[i], phi_branch_direct[i]);
    }
    
    fclose(out);
    
    free_double_vector(fout);
    free_double_vector(amp_fft_out);
    free_double_vector(phi_branch_out);
    free_double_vector(phi_branch_direct);
    free_double_vector(phi_unwrapped_arr);
    free_double_vector(phi_raw_out);
    free_double_vector(phi_spa_out);
    free_double_vector(amp_spa_out);
    free_double_vector(re_out);
    free_double_vector(im_out);
    free_int_vector(has_spa);
    free_int_vector(phase_valid);
}

static void write_spline_frequency_diagnostic_files(const char *dense_filename, const char *points_filename, const char *raw_description, double *setup, double *TS, int Ns, double *freq_track, gsl_interp_accel *PSacc, gsl_spline *PSspline)
{
    int i, j, Nend, Ndiag;
    double dte, tes, t0, t1, t, f_spline, fdot_spline;
    double f_raw_linear, diff, frac;
    FILE *out;

    if(dense_filename == NULL || points_filename == NULL || setup == NULL ||
       TS == NULL || freq_track == NULL || PSacc == NULL || PSspline == NULL ||
       Ns < 2)
    {
        return;
    }
    if(raw_description == NULL) raw_description = "raw frequency samples";

    dte = setup[0];
    Nend = (int)(setup[1]);
    tes = setup[2];
    t0 = tes;
    t1 = tes+(double)Nend*dte;
    if(t0 < TS[0]) t0 = TS[0];
    if(t1 > TS[Ns-1]) t1 = TS[Ns-1];
    if(t1 <= t0) return;

    /*
     The SPA machinery uses the derivative of a cubic spline through phase.
     This diagnostic makes that distinction visible: freq_spline/dfdt_spline
     come from P'(t)/(2*pi) and P''(t)/(2*pi), while freq_raw_linear is only a
     piecewise linear interpolation of the supplied raw samples.  For the
     post-TDI file those raw samples are finite-difference estimates from the
     extracted total TDI phase.  For the intrinsic file they are the model's
     mode frequency samples before the TDI response is applied.
     */
    out = fopen(dense_filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# t_s freq_spline_Hz dfdt_spline_Hz freq_raw_linear_Hz freq_spline_minus_raw_Hz\n");
        fprintf(out, "# raw_frequency_source: %s\n", raw_description);
        fprintf(out, "# WARNING: freq_spline_Hz and dfdt_spline_Hz are spline-derivative diagnostics only; they can ring near TDI transfer nodes or on strongly nonuniform grids.\n");
        fprintf(out, "# Dense diagnostic over the short-FFT time span, clipped to the valid adaptive phase spline.\n");
        fprintf(out, "# t_start_s %.15e t_stop_s %.15e dt_s %.15e samples %d\n",
                t0, t1, dte, (int)floor((t1-t0)/dte)+1);

        j = 0;
        Ndiag = (int)floor((t1-t0)/dte)+1;
        for(i=0; i<Ndiag; i++)
        {
            t = t0+(double)i*dte;
            if(t > t1) t = t1;

            while(j < Ns-2 && TS[j+1] < t) j++;
            f_raw_linear = NAN;
            if(t >= TS[j] && t <= TS[j+1] && TS[j+1] > TS[j])
            {
                frac = (t-TS[j])/(TS[j+1]-TS[j]);
                f_raw_linear = (1.0-frac)*freq_track[j]+frac*freq_track[j+1];
            }

            f_spline = gsl_spline_eval_deriv(PSspline, t, PSacc)/(2.0*M_PI);
            fdot_spline = gsl_spline_eval_deriv2(PSspline, t, PSacc)/(2.0*M_PI);
            diff = f_spline-f_raw_linear;

            fprintf(out, "%.15e %.15e %.15e %.15e %.15e\n",
                    t, f_spline, fdot_spline, f_raw_linear, diff);
        }
        fclose(out);
    }

    out = fopen(points_filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# t_s freq_raw_Hz freq_spline_Hz dfdt_spline_Hz freq_spline_minus_raw_Hz\n");
        fprintf(out, "# raw_frequency_source: %s\n", raw_description);
        fprintf(out, "# WARNING: freq_spline_Hz and dfdt_spline_Hz are spline-derivative diagnostics only; use raw/local finite-difference frequencies for SPA stability checks.\n");
        for(i=0; i<Ns; i++)
        {
            t = TS[i];
            f_spline = gsl_spline_eval_deriv(PSspline, t, PSacc)/(2.0*M_PI);
            fdot_spline = gsl_spline_eval_deriv2(PSspline, t, PSacc)/(2.0*M_PI);
            diff = f_spline-freq_track[i];
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e\n",
                    t, freq_track[i], f_spline, fdot_spline, diff);
        }
        fclose(out);
    }

}

void write_spline_frequency_diagnostic(const char *label, double *setup, double *TS, int Ns, double *freq_track, gsl_interp_accel *PSacc, gsl_spline *PSspline)
{
    char dense_filename[256];
    char points_filename[256];

    snprintf(dense_filename, sizeof(dense_filename), "THM_mode%s_spline_frequency_dense.dat", label);
    snprintf(points_filename, sizeof(points_filename), "THM_mode%s_frequency_adaptive_points.dat", label);
    write_spline_frequency_diagnostic_files(dense_filename, points_filename,
                                            "post-TDI finite-difference phase frequency",
                                            setup, TS, Ns, freq_track,
                                            PSacc, PSspline);
}

void write_intrinsic_spline_frequency_diagnostic(const char *label, double *setup, double *TS, int Ns, double *freq_mode, gsl_interp_accel *PSacc, gsl_spline *PSspline)
{
    char dense_filename[256];
    char points_filename[256];

    snprintf(dense_filename, sizeof(dense_filename), "THM_mode%s_intrinsic_spline_frequency_dense.dat", label);
    snprintf(points_filename, sizeof(points_filename), "THM_mode%s_intrinsic_frequency_adaptive_points.dat", label);
    write_spline_frequency_diagnostic_files(dense_filename, points_filename,
                                            "pre-TDI IMRPhenomTHM mode frequency",
                                            setup, TS, Ns, freq_mode,
                                            PSacc, PSspline);
}

void write_short_fft_spa_complex_diagnostic(const char *label, double *setup, double Tobs, double *TS, int Ns, gsl_interp_accel *ASacc, gsl_spline *ASspline, gsl_interp_accel *PSacc, gsl_spline *PSspline, double *short_htime)
{
    int i, k;
    int Nend, imx, jjoin, nmono, lo, hi, mid;
    int found_turn;
    double dte, Tspan, tes, tjoin, tstop, fstop, fmin, fmax;
    double t, f, dfdt, ta, tb, frac;
    double A, phase, x, re_spa, im_spa, re_fft, im_fft, re_raw, im_raw;
    double phase_shift, c, s;
    double *short_hfft, *tmono, *fmono, *fdot_mono;
    double *phase_grid, *freq_grid, *fdot_grid;
    char filename[256];
    FILE *out;

    if(short_htime == NULL) return;

    dte = setup[0];
    Nend = (int)(setup[1]);
    Tspan = setup[1]*dte;
    tes = setup[2];
    jjoin = (int)(setup[4]);
    imx = (int)(setup[6]);
    if(jjoin < 1) jjoin = 1;
    if(jjoin >= Ns) jjoin = Ns-1;
    if(imx > Nend/2) imx = Nend/2;

    tjoin = TS[jjoin];

    snprintf(filename, sizeof(filename), "THM_mode%s_short_window_t.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# t_s h_windowed\n");
        fprintf(out, "# Short merger/ringdown FFT input after the ftran cosine roll-on window.\n");
        fprintf(out, "# t_start_s %.15e dt_s %.15e samples %d join_time_s %.15e\n",
                tes, dte, Nend, tjoin);
        for(i=0; i<Nend; i++)
        {
            t = tes+(double)i*dte;
            fprintf(out, "%.15e %.15e\n", t, short_htime[i]);
        }
        fclose(out);
    }

    short_hfft = double_vector(Nend);
    for(i=0; i<Nend; i++) short_hfft[i] = short_htime[i];
    gsl_fft_real_radix2_transform(short_hfft, 1, Nend);
    for(i=0; i<Nend; i++) short_hfft[i] *= (2.0*dte);

    /*
     This diagnostic deliberately lets the SPA branch run past the production
     handoff.  Starting from the requested join index, find the first point
     where df/dt reaches zero; this is the formal local stationary-phase
     turning point.  Use the same local nonuniform AP-grid derivatives as the
     production ftran() path, not spline second derivatives.  That keeps this
     diagnostic from reintroducing the transfer-node ringing artifact it is
     meant to help identify.
     */
    phase_grid = double_vector(Ns);
    freq_grid = double_vector(Ns);
    fdot_grid = double_vector(Ns);
    build_nonuniform_phase_derivatives(Ns, TS, PSacc, PSspline,
                                       phase_grid, freq_grid, fdot_grid);

    found_turn = 0;
    tstop = TS[Ns-1];
    fstop = freq_grid[Ns-1];
    for(i=jjoin+1; i<Ns; i++)
    {
        if(fdot_grid[i-1] > 0.0 && fdot_grid[i] <= 0.0)
        {
            frac = fdot_grid[i-1]/(fdot_grid[i-1]-fdot_grid[i]);
            if(!isfinite(frac)) frac = 0.5;
            if(frac < 0.0) frac = 0.0;
            if(frac > 1.0) frac = 1.0;
            tstop = TS[i-1]+frac*(TS[i]-TS[i-1]);
            fstop = freq_grid[i-1]+frac*(freq_grid[i]-freq_grid[i-1]);
            found_turn = 1;
            break;
        }
    }
    if(!isfinite(fstop) || fstop <= 0.0) fstop = freq_grid[jjoin];

    tmono = double_vector(Ns+2);
    fmono = double_vector(Ns+2);
    fdot_mono = double_vector(Ns+2);
    nmono = 0;
    for(i=0; i<Ns && TS[i] < tstop; i++)
    {
        f = freq_grid[i];
        if(!isfinite(f) || f <= 0.0) continue;
        if(nmono == 0 || f > fmono[nmono-1])
        {
            tmono[nmono] = TS[i];
            fmono[nmono] = f;
            fdot_mono[nmono] = fdot_grid[i];
            nmono++;
        }
    }
    if(isfinite(fstop) && fstop > 0.0 && (nmono == 0 || fstop > fmono[nmono-1]))
    {
        tmono[nmono] = tstop;
        fmono[nmono] = fstop;
        fdot_mono[nmono] = found_turn ? 0.0 : fdot_grid[Ns-1];
        nmono++;
    }

    fmin = (nmono > 0) ? fmono[0] : NAN;
    fmax = (nmono > 0) ? fmono[nmono-1] : NAN;

    snprintf(filename, sizeof(filename), "THM_mode%s_spa_fft_complex.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# f_Hz realSPA imaginarySPA realFFT imaginaryFFT\n");
        fprintf(out, "# SPA is extended beyond the production join and stops at the first df/dt=0 after the join.\n");
        fprintf(out, "# WARNING: SPA columns use local nonuniform AP-grid derivatives; older spline-derivative diagnostics can ring near TDI transfer nodes.\n");
        fprintf(out, "# join_time_s %.15e spa_stop_time_s %.15e spa_stop_frequency_Hz %.15e found_dfdt_turn %d\n",
                tjoin, tstop, fstop, found_turn);
        fprintf(out, "# FFT columns are from the same tapered short-time waveform as ftran, rotated by exp[i 2*pi*f*(Tobs-t_start)].\n");

        for(k=1; k<imx; k++)
        {
            f = (double)k/Tspan;

            re_raw = short_hfft[k];
            im_raw = short_hfft[Nend-k];
            phase_shift = 2.0*M_PI*f*(Tobs-tes);
            c = cos(phase_shift);
            s = sin(phase_shift);
            re_fft = re_raw*c-im_raw*s;
            im_fft = re_raw*s+im_raw*c;

            re_spa = NAN;
            im_spa = NAN;
            if(nmono >= 2 && f > fmin && f < fmax && f < fstop)
            {
                lo = 0;
                hi = nmono-1;
                while(hi-lo > 1)
                {
                    mid = (lo+hi)/2;
                    if(fmono[mid] <= f) lo = mid;
                    else hi = mid;
                }

                ta = tmono[lo];
                tb = tmono[hi];
                if(fmono[hi] > fmono[lo])
                {
                    frac = (f-fmono[lo])/(fmono[hi]-fmono[lo]);
                }
                else
                {
                    frac = 0.0;
                }
                if(frac < 0.0) frac = 0.0;
                if(frac > 1.0) frac = 1.0;

                t = ta+frac*(tb-ta);
                dfdt = fdot_mono[lo]+frac*(fdot_mono[hi]-fdot_mono[lo]);
                if(dfdt > 0.0 && isfinite(dfdt))
                {
                    A = gsl_spline_eval(ASspline, t, ASacc);
                    phase = gsl_spline_eval(PSspline, t, PSacc)-2.0*M_PI*f*(t-Tobs)+M_PI/4.0;
                    x = sqrt(1.0/dfdt);
                    re_spa = x*A*cos(phase);
                    im_spa = x*A*sin(phase);
                }
            }

            fprintf(out, "%.15e %.15e %.15e %.15e %.15e\n",
                    f, re_spa, im_spa, re_fft, im_fft);
        }
        fclose(out);
    }

    free_double_vector(short_hfft);
    free_double_vector(tmono);
    free_double_vector(fmono);
    free_double_vector(fdot_mono);
    free_double_vector(phase_grid);
    free_double_vector(freq_grid);
    free_double_vector(fdot_grid);
}

void unpack_wdm_track(double **wdm, int *listn, int *listm, double *wdmwave, int Np)
{
    int i, j;
    
    for(i=0; i<Nt; i++)
    {
        for(j=0; j<=Nf; j++) wdm[i][j] = 0.0;
    }
    
    for(i=0; i<Np; i++) wdm[listn[i]][listm[i]] = wdmwave[i];
}

static void thm_sparse_wdm_channel_init(THMSparseWDMChannel *track)
{
    if(track == NULL) return;
    memset(track, 0, sizeof(*track));
}

static void thm_sparse_wdm_channel_free(THMSparseWDMChannel *track)
{
    if(track == NULL) return;
    free(track->n);
    free(track->m);
    free(track->value);
    thm_sparse_wdm_channel_init(track);
}

void thm_sparse_wdm_triplet_init(THMSparseWDMTriplet *tracks)
{
    int ch;

    if(tracks == NULL) return;
    for(ch=0; ch<3; ch++) thm_sparse_wdm_channel_init(&tracks->channel[ch]);
}

void thm_sparse_wdm_triplet_free(THMSparseWDMTriplet *tracks)
{
    int ch;

    if(tracks == NULL) return;
    for(ch=0; ch<3; ch++) thm_sparse_wdm_channel_free(&tracks->channel[ch]);
}

void thm_fourier_triplet_init(THMFourierTriplet *spectrum)
{
    if(spectrum == NULL) return;
    memset(spectrum, 0, sizeof(*spectrum));
}

void thm_fourier_triplet_free(THMFourierTriplet *spectrum)
{
    int ch;

    if(spectrum == NULL) return;
    for(ch=0; ch<3; ch++)
    {
        free(spectrum->re[ch]);
        free(spectrum->im[ch]);
        free(spectrum->full_re[ch]);
        free(spectrum->full_im[ch]);
    }
    thm_fourier_triplet_init(spectrum);
}

static int thm_fourier_triplet_reserve(THMFourierTriplet *spectrum, int n)
{
    int ch;
    double *rnew, *inew;

    if(spectrum == NULL || n < 2) return 0;
    if(spectrum->n == n)
    {
        int complete = 1;
        for(ch=0; ch<3; ch++)
        {
            if(spectrum->re[ch] == NULL || spectrum->im[ch] == NULL)
            {
                complete = 0;
            }
        }
        if(spectrum->request_full_fft)
        {
            for(ch=0; ch<3; ch++)
            {
                if(spectrum->full_re[ch] == NULL || spectrum->full_im[ch] == NULL)
                {
                    complete = 0;
                }
            }
        }
        if(complete) return 1;
    }

    for(ch=0; ch<3; ch++)
    {
        rnew = (double *)realloc(spectrum->re[ch], (size_t)n*sizeof(double));
        if(rnew == NULL) return 0;
        spectrum->re[ch] = rnew;
        inew = (double *)realloc(spectrum->im[ch], (size_t)n*sizeof(double));
        if(inew == NULL) return 0;
        spectrum->im[ch] = inew;
        if(spectrum->request_full_fft)
        {
            rnew = (double *)realloc(spectrum->full_re[ch], (size_t)n*sizeof(double));
            if(rnew == NULL) return 0;
            spectrum->full_re[ch] = rnew;
            inew = (double *)realloc(spectrum->full_im[ch], (size_t)n*sizeof(double));
            if(inew == NULL) return 0;
            spectrum->full_im[ch] = inew;
        }
    }
    spectrum->n = n;
    return 1;
}

static int thm_sparse_wdm_channel_reserve(THMSparseWDMChannel *track, int capacity)
{
    int *nnew, *mnew;
    double *vnew;

    if(track == NULL || capacity < 0) return 0;
    if(capacity <= track->capacity) return 1;

    nnew = (int *)realloc(track->n, (size_t)capacity*sizeof(int));
    if(nnew == NULL) return 0;
    track->n = nnew;

    mnew = (int *)realloc(track->m, (size_t)capacity*sizeof(int));
    if(mnew == NULL) return 0;
    track->m = mnew;

    vnew = (double *)realloc(track->value, (size_t)capacity*sizeof(double));
    if(vnew == NULL) return 0;
    track->value = vnew;

    track->capacity = capacity;
    return 1;
}

static int count_wdm_nonzero_pixels(double **wdm)
{
    int i, j, count;

    if(wdm == NULL) return 0;

    count = 0;
    for(i=0; i<Nt; i++)
    {
        for(j=1; j<Nf; j++)
        {
            if(wdm[i][j] != 0.0) count++;
        }
    }
    return count;
}

static int thm_sparse_wdm_channel_from_dense(THMSparseWDMChannel *track, double **wdm)
{
    int i, j, count;

    if(track == NULL || wdm == NULL) return -1;

    count = count_wdm_nonzero_pixels(wdm);

    if(!thm_sparse_wdm_channel_reserve(track, count)) return -1;

    count = 0;
    for(i=0; i<Nt; i++)
    {
        for(j=1; j<Nf; j++)
        {
            if(wdm[i][j] != 0.0)
            {
                track->n[count] = i;
                track->m[count] = j;
                track->value[count] = wdm[i][j];
                count++;
            }
        }
    }
    track->npixels = count;
    return count;
}

int write_thm_sparse_wdm_channel(const char *filename, const THMSparseWDMChannel *track)
{
    int i;
    FILE *out;

    if(filename == NULL || track == NULL) return 0;

    out = fopen(filename, "w");
    if(out == NULL) return 0;

    fprintf(out, "# n m wdm\n");
    for(i=0; i<track->npixels; i++)
    {
        fprintf(out, "%d %d %.14e\n", track->n[i], track->m[i], track->value[i]);
    }

    fclose(out);
    return track->npixels;
}

static void thm_ap_context_init(THMAPContext *ctx)
{
    if(ctx == NULL) return;
    memset(ctx, 0, sizeof(*ctx));
}

static void thm_ap_context_free(THMAPContext *ctx)
{
    int k;

    if(ctx == NULL) return;
    if(ctx->Aacc != NULL)
    {
        for(k=0; k<ctx->nmodes; k++)
        {
            if(ctx->Aspline != NULL && ctx->Aspline[k] != NULL) gsl_spline_free(ctx->Aspline[k]);
            if(ctx->Pspline != NULL && ctx->Pspline[k] != NULL) gsl_spline_free(ctx->Pspline[k]);
            if(ctx->Aacc[k] != NULL) gsl_interp_accel_free(ctx->Aacc[k]);
            if(ctx->Pacc != NULL && ctx->Pacc[k] != NULL) gsl_interp_accel_free(ctx->Pacc[k]);
        }
    }
    free(ctx->Aacc);
    free(ctx->Pacc);
    free(ctx->Aspline);
    free(ctx->Pspline);

    if(ctx->TS != NULL) free_double_vector(ctx->TS);
    if(ctx->tspace != NULL) free_double_vector(ctx->tspace);
    if(ctx->mode_amp != NULL) free_double_matrix(ctx->mode_amp, IMRPHENOMTHM_MAX_MODES);
    if(ctx->mode_phase != NULL) free_double_matrix(ctx->mode_phase, IMRPHENOMTHM_MAX_MODES);
    if(ctx->mode_freq != NULL) free_double_matrix(ctx->mode_freq, IMRPHENOMTHM_MAX_MODES);
    IMRPhenomTHMDestroy(&ctx->model);
    thm_ap_context_init(ctx);
}

static int thm_ap_context_alloc(THMAPContext *ctx, int Nsmax)
{
    if(ctx == NULL || Nsmax < 2) return 0;

    ctx->TS = double_vector(Nsmax);
    ctx->tspace = double_vector(Nsmax);
    ctx->mode_amp = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
    ctx->mode_phase = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);
    ctx->mode_freq = double_matrix(IMRPHENOMTHM_MAX_MODES, Nsmax);

    return ctx->TS != NULL && ctx->tspace != NULL &&
           ctx->mode_amp != NULL && ctx->mode_phase != NULL &&
           ctx->mode_freq != NULL;
}

static int thm_ap_context_build_splines_and_carriers(THMAPContext *ctx)
{
    int k;

    if(ctx == NULL || ctx->Ns < 4 || ctx->nmodes < 1) return 0;

    ctx->Aacc = malloc((size_t)ctx->nmodes*sizeof(gsl_interp_accel *));
    ctx->Pacc = malloc((size_t)ctx->nmodes*sizeof(gsl_interp_accel *));
    ctx->Aspline = malloc((size_t)ctx->nmodes*sizeof(gsl_spline *));
    ctx->Pspline = malloc((size_t)ctx->nmodes*sizeof(gsl_spline *));
    if(ctx->Aacc == NULL || ctx->Pacc == NULL ||
       ctx->Aspline == NULL || ctx->Pspline == NULL)
    {
        return 0;
    }

    for(k=0; k<ctx->nmodes; k++)
    {
        ctx->Aacc[k] = NULL;
        ctx->Pacc[k] = NULL;
        ctx->Aspline[k] = NULL;
        ctx->Pspline[k] = NULL;
    }

    for(k=0; k<ctx->nmodes; k++)
    {
        ctx->Aacc[k] = gsl_interp_accel_alloc();
        ctx->Pacc[k] = gsl_interp_accel_alloc();
        ctx->Aspline[k] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, ctx->Ns);
        ctx->Pspline[k] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, ctx->Ns);
        if(ctx->Aacc[k] == NULL || ctx->Pacc[k] == NULL ||
           ctx->Aspline[k] == NULL || ctx->Pspline[k] == NULL)
        {
            return 0;
        }
        gsl_spline_init(ctx->Aspline[k], ctx->TS, ctx->mode_amp[k], ctx->Ns);
        gsl_spline_init(ctx->Pspline[k], ctx->TS, ctx->mode_phase[k], ctx->Ns);
        thm_projection_coefficients(ctx->modes[k].ell, ctx->modes[k].emm,
                                    ctx->params[10], 0.5*M_PI,
                                    ctx->params[9], &ctx->projection[k]);
    }

    ctx->ncarriers = thm_build_folded_carriers(ctx->nmodes, ctx->modes,
                                               ctx->projection,
                                               ctx->carrier,
                                               IMRPHENOMTHM_MAX_MODES);
    return ctx->ncarriers > 0;
}

static double thm_unwrapped_phase_difference(double raw, double previous, int have_previous)
{
    double x;

    x = raw;
    if(!have_previous)
    {
        return atan2(sin(x), cos(x));
    }
    while(x-previous > M_PI) x -= 2.0*M_PI;
    while(x-previous < -M_PI) x += 2.0*M_PI;
    return x;
}

static int thm_keep_increasing_frequency_samples(int n, double *freq, double *phase, double *amp)
{
    int i, kept;

    if(n < 1 || freq == NULL || phase == NULL || amp == NULL) return 0;

    kept = 1;
    for(i=1; i<n; i++)
    {
        if(freq[i] > freq[kept-1])
        {
            if(kept != i)
            {
                freq[kept] = freq[i];
                phase[kept] = phase[i];
                amp[kept] = amp[i];
            }
            kept++;
        }
    }
    return kept;
}

int write_wdm_nonzero_track_pixels(const char *filename, double **wdm)
{
    int i, j, count;
    FILE *out;

    out = fopen(filename, "w");
    if(out == NULL) return 0;

    fprintf(out, "# n m wdm\n");
    count = 0;
    for(i=0; i<Nt; i++)
    {
        for(j=1; j<Nf; j++)
        {
            if(wdm[i][j] != 0.0)
            {
                fprintf(out, "%d %d %.14e\n", i, j, wdm[i][j]);
                count++;
            }
        }
    }

    fclose(out);
    return count;
}

void write_track_pixels(const char *filename, int *listn, int *listm, double *wdmwave, int Np)
{
    int i;
    FILE *out;
    
    out = fopen(filename, "w");
    fprintf(out, "# n m wdm\n");
    for(i=0; i<Np; i++) fprintf(out, "%d %d %.14e\n", listn[i], listm[i], wdmwave[i]);
    fclose(out);
}

void write_wdm_matrix(const char *filename, double **wdm)
{
    int i, j;
    FILE *out;
    
    out = fopen(filename, "w");
    for(j=0; j<Nt; j++)
    {
        for(i=0; i<=Nf; i++) fprintf(out, "%.14e ", wdm[j][i]);
        fprintf(out, "\n");
    }
    fclose(out);
}

void write_wdm_binary(const char *filename, double **wdm, struct wdmshape *wdms)
{
    int i, j;
    double f;
    FILE *out;
    
    out = fopen(filename, "w");
    for(i=0; i<=Nf; i++)
    {
        if(i == 0) f = 0.25*wdms->DF;
        else if(i == Nf) f = ((double)Nf-0.25)*wdms->DF;
        else f = (double)(i)*wdms->DF;
        
        for(j=0; j<Nt; j++)
        {
            if((i == 0 || i == Nf) && (j+i)%2 != 0) continue;
            fprintf(out, "%e %e %.14e\n", (double)(j)*wdms->DT, f, wdm[j][i]);
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

void write_match_info(const char *filename, struct wdmshape *wdms, double tukey_alpha, double roll_time)
{
    int roll_samples, roll_jmin, roll_jmax;
    int match_jmin, match_jmax;
    double match_padding;
    FILE *out;

    roll_samples = (int)(tukey_alpha*((double)(Nt*Nf-1))/2.0);
    roll_jmin = (int)ceil(roll_time/wdms->DT);
    roll_jmax = (int)floor((wdms->Tobs-roll_time)/wdms->DT)+1;

    /*
     WDM pixels have finite time support. Use a half-filter padding so the
     recommended match window excludes pixels whose windows overlap the Tukey
     roll-on/off. The raw roll_jmin/roll_jmax values are also written for less
     conservative checks.
     */
    match_padding = roll_time+0.5*wdms->Tfilt;
    match_jmin = (int)ceil(match_padding/wdms->DT);
    match_jmax = (int)floor((wdms->Tobs-match_padding)/wdms->DT)+1;

    if(roll_jmin < 0) roll_jmin = 0;
    if(roll_jmax > Nt) roll_jmax = Nt;
    if(match_jmin < 0) match_jmin = 0;
    if(match_jmax > Nt) match_jmax = Nt;

    out = fopen(filename, "w");
    fprintf(out, "# WDM match metadata written by PhenomT_TDI.c\n");
    fprintf(out, "# match.c and match_track.c intersect requested time limits with match_jmin/match_jmax.\n");
    fprintf(out, "Nt %d\n", Nt);
    fprintf(out, "Nf %d\n", Nf);
    fprintf(out, "dt %.15e\n", dt);
    fprintf(out, "DT %.15e\n", wdms->DT);
    fprintf(out, "DF %.15e\n", wdms->DF);
    fprintf(out, "Tobs %.15e\n", wdms->Tobs);
    fprintf(out, "tukey_alpha %.15e\n", tukey_alpha);
    fprintf(out, "tukey_roll_time %.15e\n", roll_time);
    fprintf(out, "tukey_roll_samples %d\n", roll_samples);
    fprintf(out, "wdm_filter_half_time %.15e\n", 0.5*wdms->Tfilt);
    fprintf(out, "roll_jmin %d\n", roll_jmin);
    fprintf(out, "roll_jmax %d\n", roll_jmax);
    fprintf(out, "match_jmin %d\n", match_jmin);
    fprintf(out, "match_jmax %d\n", match_jmax);
    fclose(out);
}

void WDMpixels(int Ns, double *TF, double *FF, int *nmid, int *nsize, int N, struct wdmshape *wdms)
{
    double flower, fupper, tupper, tlower;
    double f, t;
    int mmin, mmax;
    int i, j, k, ii, jj, kk;
    
    FILE *out;

        for(i=0; i<Nf; i++)
        {
            nmid[i] = -1;
            nsize[i] = 0;
        }
        for(i=1; i<Ns; i++)
        {
            if(FF[i] <= FF[i-1])
            {
                fprintf(stderr,
                        "Warning: WDMpixels uses an inverse spline t(f) and requires strictly increasing frequencies; use WDMpixelsTimeScan for nonmonotonic or TDI-distorted tracks.\n");
                return;
            }
        }
        
        // this spline gives us t(f)
        gsl_interp_accel *TFacc;
        gsl_spline *TFspline;
        TFacc = gsl_interp_accel_alloc();
        TFspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
        gsl_spline_init(TFspline, FF, TF, Ns);

  
        mmin = (int)(floor((FF[0]-wdms->FB)/wdms->DF));
        mmax = (int)(floor(FF[Ns-1]/wdms->DF));
        if(mmin < 1) mmin = 1;
        if(mmax > Nf) mmax = Nf;
    
       printf("fmax %e\n", FF[Ns-1]);
        
         //printf("lowest frequency layer needed %d\n", mmin);
       // printf("highest frequency layer needed %d\n", mmax);
        
        // get t_min and t_max boundaries for each frequency layer
        //out = fopen("boundaries.dat","w");
        k = 0;
        kk = 0;
        for(i=0; i < mmin; ++i) nmid[i] = -1;
        for(i=mmax; i < Nf; ++i) nmid[i] = -1;
        for (i = mmin; i < mmax; ++i)
        {
            f = (double)(i)*wdms->DF;
            flower = f-wdms->FB;
            fupper = f+wdms->FB;
            tlower = 0.0;
            if(flower > FF[0] && flower < FF[Ns-1]) tlower = gsl_spline_eval (TFspline, flower, TFacc);
            if(tlower < 0.0) tlower = 0.0;
            tupper = 0.0;
            if(fupper > FF[0] && fupper < FF[Ns-1]) tupper = gsl_spline_eval (TFspline, fupper, TFacc);
            if(tupper > wdms->Tobs) tupper = wdms->Tobs;
            t = 0.5*(tupper+tlower); // midpoint of band
            ii = (int)((t/wdms->DT));
            if(ii%2 != 0) ii--; // need this to be even so as not to mess up the transform
            nmid[i] = ii;
            
            j = (int)(ceil(tupper/wdms->DT)-floor(tlower/wdms->DT))+2*mult-1;
            jj = (int)(pow(2.0,floor(log2((double)j))));
            if(jj < (j-2)) jj *= 2;  // willing to miss the two end pixels in time (very small values)
            
            
            nsize[i] = jj;
            
            if((ii - jj/2) < 0) nmid[i] = jj/2;
                
            //fprintf(out, "%f %d %f %f\n", f/wdms->DF, nmid[i], floor(tlower/wdms->DT)-mult, ceil(tupper/wdms->DT)+mult);
            
            // once jj hits 2*mult it stays at that value
            // a bit further along in frequency the time boundaries also take fixed values
             kk += jj;
             k += j;
            // printf("%d %d %d %d %d\n", i, j, k, jj, kk);
        }
        //fclose(out);
        
        // 6% waste for the 1e5 case
        // 4% waste for the 1e6 case
    
   // printf("pixels %d %d\n", k, kk);
    
    gsl_spline_free(TFspline);
    gsl_interp_accel_free(TFacc);
	        
	}

void WDMpixelsTimeScan(int Ns, double *TF, double *FF, int *nmid, int *nsize, int N, struct wdmshape *wdms)
{
    int *jmin_layer, *jmax_layer;
    int i, m;
    int time_pad, layer_pad;
    int active_layers, volume;
    double fmax_seen;

    (void)N;

    jmin_layer = int_vector(Nf);
    jmax_layer = int_vector(Nf);

    for(m=0; m<=Nf; m++)
    {
        nmid[m] = -1;
        nsize[m] = 0;
        jmin_layer[m] = Nt+1;
        jmax_layer[m] = -1;
    }

    fmax_seen = 0.0;
    time_pad = THM_WDM_TIMESCAN_PAD_PIXELS;
    if(time_pad < 0) time_pad = 0;
    layer_pad = THM_WDM_TIMESCAN_PAD_LAYERS;
    if(layer_pad < 0) layer_pad = 0;

    /*
     * Forward-scan the time-frequency track.  Unlike WDMpixels(), this never
     * inverts f(t), so it remains well-defined when the TDI-extracted frequency
     * is flat, nearly vertical in the WDM plane, or locally nonmonotonic.
     *
     * Each adaptive waveform interval contributes the frequency range it sweeps,
     * expanded by the Meyer half-bandwidth FB, to all overlapping WDM layers.
     * The time range is padded more conservatively than the inverse-spline
     * finder.  Near merger the track can bend through a frequency layer quickly,
     * and the finite packet response leaves visible support just outside the
     * instantaneous f(t) crossing interval.  Over-covering here is intentional:
     * this diagnostic should first identify all active pixels, then later we can
     * trim the block layout for speed.
     */
    for(i=0; i<Ns-1; i++)
    {
        double tlo, thi, flo, fhi;
        int jlo, jhi, mlo, mhi;

        if(!isfinite(TF[i]) || !isfinite(TF[i+1]) ||
           !isfinite(FF[i]) || !isfinite(FF[i+1]))
        {
            continue;
        }

        tlo = TF[i] < TF[i+1] ? TF[i] : TF[i+1];
        thi = TF[i] > TF[i+1] ? TF[i] : TF[i+1];
        if(thi < 0.0 || tlo > wdms->Tobs) continue;
        if(tlo < 0.0) tlo = 0.0;
        if(thi > wdms->Tobs) thi = wdms->Tobs;

        flo = fabs(FF[i]);
        fhi = fabs(FF[i+1]);
        if(flo > fmax_seen) fmax_seen = flo;
        if(fhi > fmax_seen) fmax_seen = fhi;
        if(flo > fhi)
        {
            double tmp = flo;
            flo = fhi;
            fhi = tmp;
        }

        flo -= wdms->FB;
        fhi += wdms->FB;
        if(fhi <= 0.0) continue;

        mlo = (int)ceil(flo/wdms->DF);
        mhi = (int)floor(fhi/wdms->DF);
        mlo -= layer_pad;
        mhi += layer_pad;
        if(mlo < 1) mlo = 1;
        if(mhi > Nf-1) mhi = Nf-1;
        if(mhi < mlo) continue;

        jlo = (int)floor(tlo/wdms->DT)-time_pad;
        jhi = (int)ceil(thi/wdms->DT)+time_pad;
        if(jlo < 0) jlo = 0;
        if(jhi > Nt-1) jhi = Nt-1;
        if(jhi < jlo) continue;

        for(m=mlo; m<=mhi; m++)
        {
            if(jlo < jmin_layer[m]) jmin_layer[m] = jlo;
            if(jhi > jmax_layer[m]) jmax_layer[m] = jhi;
        }
    }

    active_layers = 0;
    volume = 0;
    for(m=1; m<Nf; m++)
    {
        int jlo, jhi, needed, block, center;

        if(jmax_layer[m] < jmin_layer[m]) continue;

        jlo = jmin_layer[m];
        jhi = jmax_layer[m];
        needed = jhi-jlo+1;
        if(needed < 1) needed = 1;

        /*
         * WDMtrack accepts one power-of-two time block per frequency layer.
         * Use the smallest conservative block that covers the marked pixels,
         * with a tiny guard because these are raw time-scan bounds.
         */
        block = 1;
        while(block < needed+2) block *= 2;
        if(block < 2*mult) block = 2*mult;

        center = (jlo+jhi+1)/2;
        if(center%2 != 0) center--;
        if(center < block/2) center = block/2;
        if(center+block/2 > Nt) center = Nt-block/2;
        if(center%2 != 0) center--;

        while(center-block/2 > jlo) center -= 2;
        while(center+block/2-1 < jhi) center += 2;

        if(center < block/2) center = block/2;
        if(center+block/2 > Nt) center = Nt-block/2;
        if(center%2 != 0) center--;

        nmid[m] = center;
        nsize[m] = block;
        active_layers++;
        volume += block;
    }

    if(thm_diagnostics_enabled)
    {
        printf("time-scan WDM pixels fmax %e active_layers %d volume %d\n",
               fmax_seen, active_layers, volume);
    }

    free_int_vector(jmin_layer);
    free_int_vector(jmax_layer);
}

void WDMpixelsTimeScanRange(int Ns, double *TF, double *FF, double tmin, double tmax, int *nmid, int *nsize, int N, struct wdmshape *wdms)
{
    int *jmin_layer, *jmax_layer;
    int i, m;
    int time_pad, layer_pad;
    int active_layers, volume;
    double fmax_seen;

    (void)N;

    jmin_layer = int_vector(Nf);
    jmax_layer = int_vector(Nf);

    for(m=0; m<=Nf; m++)
    {
        nmid[m] = -1;
        nsize[m] = 0;
        jmin_layer[m] = Nt+1;
        jmax_layer[m] = -1;
    }

    if(TF == NULL || FF == NULL || wdms == NULL || Ns < 2 ||
       !isfinite(tmin) || !isfinite(tmax) || tmax <= tmin)
    {
        free_int_vector(jmin_layer);
        free_int_vector(jmax_layer);
        return;
    }

    if(tmin < 0.0) tmin = 0.0;
    if(tmax > wdms->Tobs) tmax = wdms->Tobs;
    if(tmax <= tmin)
    {
        free_int_vector(jmin_layer);
        free_int_vector(jmax_layer);
        return;
    }

    fmax_seen = 0.0;
    time_pad = THM_WDM_TIMESCAN_PAD_PIXELS;
    if(time_pad < 0) time_pad = 0;
    layer_pad = THM_WDM_TIMESCAN_PAD_LAYERS;
    if(layer_pad < 0) layer_pad = 0;

    /*
     * Windowed companion to WDMpixelsTimeScan().  It clips each adaptive
     * time-frequency segment to [tmin,tmax] before assigning WDM packets.  The
     * combined THM endpoint FFT uses this to remain local in time: the endpoint
     * waveform is nonzero only in the short tapered merger/ringdown segment, so
     * its packet mask should not inherit the long inspiral blocks used by the
     * SPA piece at the same frequency layer.
     */
    for(i=0; i<Ns-1; i++)
    {
        double ta, tb, fa, fb;
        double seglo, seghi, ua, ub;
        double tlo, thi, flo, fhi;
        int jlo, jhi, mlo, mhi;

        if(!isfinite(TF[i]) || !isfinite(TF[i+1]) ||
           !isfinite(FF[i]) || !isfinite(FF[i+1]))
        {
            continue;
        }

        ta = TF[i];
        tb = TF[i+1];
        fa = FF[i];
        fb = FF[i+1];

        if(tb == ta) continue;
        if(tb < ta)
        {
            double tmp;
            tmp = ta; ta = tb; tb = tmp;
            tmp = fa; fa = fb; fb = tmp;
        }

        if(tb < tmin || ta > tmax) continue;
        seglo = ta > tmin ? ta : tmin;
        seghi = tb < tmax ? tb : tmax;
        if(seghi < seglo) continue;

        ua = (seglo-ta)/(tb-ta);
        ub = (seghi-ta)/(tb-ta);
        tlo = seglo;
        thi = seghi;
        flo = fabs(fa+ua*(fb-fa));
        fhi = fabs(fa+ub*(fb-fa));

        if(flo > fmax_seen) fmax_seen = flo;
        if(fhi > fmax_seen) fmax_seen = fhi;
        if(flo > fhi)
        {
            double tmp = flo;
            flo = fhi;
            fhi = tmp;
        }

        flo -= wdms->FB;
        fhi += wdms->FB;
        if(fhi <= 0.0) continue;

        mlo = (int)ceil(flo/wdms->DF);
        mhi = (int)floor(fhi/wdms->DF);
        mlo -= layer_pad;
        mhi += layer_pad;
        if(mlo < 1) mlo = 1;
        if(mhi > Nf-1) mhi = Nf-1;
        if(mhi < mlo) continue;

        jlo = (int)floor(tlo/wdms->DT)-time_pad;
        jhi = (int)ceil(thi/wdms->DT)+time_pad;
        if(jlo < 0) jlo = 0;
        if(jhi > Nt-1) jhi = Nt-1;
        if(jhi < jlo) continue;

        for(m=mlo; m<=mhi; m++)
        {
            if(jlo < jmin_layer[m]) jmin_layer[m] = jlo;
            if(jhi > jmax_layer[m]) jmax_layer[m] = jhi;
        }
    }

    active_layers = 0;
    volume = 0;
    for(m=1; m<Nf; m++)
    {
        int jlo, jhi, needed, block, center;

        if(jmax_layer[m] < jmin_layer[m]) continue;

        jlo = jmin_layer[m];
        jhi = jmax_layer[m];
        needed = jhi-jlo+1;
        if(needed < 1) needed = 1;

        block = 1;
        while(block < needed+2) block *= 2;
        if(block < 2*mult) block = 2*mult;

        center = (jlo+jhi+1)/2;
        if(center%2 != 0) center--;
        if(center < block/2) center = block/2;
        if(center+block/2 > Nt) center = Nt-block/2;
        if(center%2 != 0) center--;

        while(center-block/2 > jlo) center -= 2;
        while(center+block/2-1 < jhi) center += 2;

        if(center < block/2) center = block/2;
        if(center+block/2 > Nt) center = Nt-block/2;
        if(center%2 != 0) center--;

        nmid[m] = center;
        nsize[m] = block;
        active_layers++;
        volume += block;
    }

    if(thm_diagnostics_enabled)
    {
        printf("time-scan WDM pixels range [%e,%e] fmax %e active_layers %d volume %d\n",
               tmin, tmax, fmax_seen, active_layers, volume);
    }

    free_int_vector(jmin_layer);
    free_int_vector(jmax_layer);
}

void WDMpixelsAddMergerFrequencyTail(int *nmid, int *nsize, double fmax_spectrum, double tail_time, struct wdmshape *wdms)
{
    int m, active_max, mmax_tail, block, center;
    int layers_added, layers_present;

    if(nmid == NULL || nsize == NULL || wdms == NULL ||
       !isfinite(fmax_spectrum) || fmax_spectrum <= 0.0 ||
       !isfinite(tail_time))
    {
        return;
    }

    active_max = 0;
    for(m=1; m<Nf; m++)
    {
        if(nmid[m] >= 0 && nsize[m] > 0) active_max = m;
    }

    /*
     * The time-scan mask follows the instantaneous TDI frequency track.  The
     * merger/ringdown end of ftran(), however, is supplied by a short FFT and
     * can have real Fourier support above the last instantaneous frequency,
     * especially for the lower harmonics.  Add only that high-frequency tail:
     * one minimal WDM packet per extra layer, centered near the carrier's peak
     * response.  This replaces the old broad time padding, which inflated every
     * layer but did not address the actual missing support.
     */
    mmax_tail = (int)floor((fmax_spectrum+wdms->FB)/wdms->DF);
    if(mmax_tail > Nf-1) mmax_tail = Nf-1;
    if(mmax_tail <= active_max) return;

    block = 1;
    while(block < 2*mult) block *= 2;
    if(block > Nt) block = Nt;

    center = 2*(int)floor(tail_time/(2.0*wdms->DT)+0.5);
    if(center < block/2) center = block/2;
    if(center+block/2 > Nt) center = Nt-block/2;
    if(center%2 != 0) center--;
    if(center < block/2) center = block/2;

    layers_added = 0;
    layers_present = 0;
    for(m=active_max+1; m<=mmax_tail; m++)
    {
        if(nmid[m] >= 0 && nsize[m] > 0)
        {
            layers_present++;
            continue;
        }
        nmid[m] = center;
        nsize[m] = block;
        layers_added++;
    }

    if(thm_diagnostics_enabled)
    {
        printf("time-scan merger-frequency tail fmax_spectrum %e active_max_layer %d tail_max_layer %d added_layers %d present_layers %d center_time_pixel %d block %d\n",
               fmax_spectrum, active_max, mmax_tail, layers_added, layers_present,
               center, block);
    }
}

void WDMmergePixelPlans(int *nmid_total, int *nsize_total, int *nmid_add, int *nsize_add)
{
    int m;

    if(nmid_total == NULL || nsize_total == NULL ||
       nmid_add == NULL || nsize_add == NULL)
    {
        return;
    }

    for(m=1; m<Nf; m++)
    {
        int lo_total, hi_total, lo_add, hi_add;
        int lo, hi, needed, block, center;

        if(nmid_add[m] < 0 || nsize_add[m] <= 0) continue;

        if(nmid_total[m] < 0 || nsize_total[m] <= 0)
        {
            nmid_total[m] = nmid_add[m];
            nsize_total[m] = nsize_add[m];
            continue;
        }

        lo_total = nmid_total[m]-nsize_total[m]/2;
        hi_total = nmid_total[m]+nsize_total[m]/2-1;
        lo_add = nmid_add[m]-nsize_add[m]/2;
        hi_add = nmid_add[m]+nsize_add[m]/2-1;

        lo = lo_total < lo_add ? lo_total : lo_add;
        hi = hi_total > hi_add ? hi_total : hi_add;
        if(lo < 0) lo = 0;
        if(hi > Nt-1) hi = Nt-1;
        needed = hi-lo+1;
        if(needed < 1) needed = 1;

        /*
         * The inputs are already WDM packet blocks, not raw marked intervals.
         * Cover the union of their existing supports without adding the raw
         * time-scan guard again; otherwise repeated merges of identical
         * 16-pixel blocks inflate to 32, 64, ... with each added harmonic.
         */
        block = 1;
        while(block < needed) block *= 2;
        if(block < 2*mult) block = 2*mult;
        if(block > Nt) block = Nt;

        center = (lo+hi+1)/2;
        if(center%2 != 0) center--;
        if(center < block/2) center = block/2;
        if(center+block/2 > Nt) center = Nt-block/2;
        if(center%2 != 0) center--;

        while(center-block/2 > lo) center -= 2;
        while(center+block/2-1 < hi) center += 2;

        if(center < block/2) center = block/2;
        if(center+block/2 > Nt) center = Nt-block/2;
        if(center%2 != 0) center--;

        nmid_total[m] = center;
        nsize_total[m] = block;
    }
}

double phitilde(double om, double insDOM, double A, double B)
{
    double x, y, z;
    
       z = 0.0;
       
       if(fabs(om) >= A && fabs(om) < A+B)
       {
           x = (fabs(om)-A)/B;
           y = gsl_sf_beta_inc(nx, nx, x);
           z = insDOM*cos(y*M_PI/2.0);
       }
       
       if(fabs(om) < A) z = insDOM;
    
    return(z);
    
}

void wdmtranF(int m, int Ntx, double scale, double *phihf, double *data, double *wdmout)
{
    double *DX;
    int n, i, j, jj;
    
    DX = double_vector(2*Ntx);

         
        for(j=-Ntx/2; j< Ntx/2; j++)
        {
            i = j+Ntx/2;
            
            REAL(DX,i) = 0.0;
            IMAG(DX,i) = 0.0;
            
            jj = j + Ntx/2;
            
            if(i > 0 && i < Ntx)
            {
                    REAL(DX,i) = data[jj]*phihf[abs(j)];
                    IMAG(DX,i) = data[2*Ntx-jj]*phihf[abs(j)];
            }
            
                
        }
         
         gsl_fft_complex_radix2_backward(DX, 1, Ntx);
         
        
         for(n=0; n < Ntx; n++)
         {
             
             if(m%2 == 0)
             {
                 
                 if((n+m)%2 ==0)
                 {
                      wdmout[n] = scale*REAL(DX,n);
                 }
                 else
                 {
                    wdmout[n] = scale*IMAG(DX,n);
                 }
                 
             }
             else
             {
                 if((n+m)%2 ==0)
                 {
                      wdmout[n] = scale*REAL(DX,n);
                 }
                 else
                 {
                    wdmout[n] = -scale*IMAG(DX,n);
                 }
                 
             }
            
         }
    
    free(DX);
    
    
}



void wdmvalues(struct wdmshape *wdms)
{
    
    wdms->DT = dt*(double)(Nf);           // width of wavelet pixel in time
    wdms->DF = 1.0/(2.0*dt*(double)(Nf));   // width of wavelet pixel in frequency
    
    wdms->OM = M_PI/dt; // angular Nyquist frequency
    
    wdms->DOM = wdms->OM/(double)(Nf);
    
    wdms->B = Bfrac*wdms->DOM;
    
    wdms->A = (wdms->DOM-wdms->B)/2.0;
    
    wdms->insDOM = 1.0/sqrt(wdms->DOM);
    
    wdms->FB = (wdms->A+wdms->B)/(2.0*M_PI); // extent
    
    wdms->DFA =  wdms->A/(2.0*M_PI); // half-width of non-overlapping frequency region
    
    wdms->Tfilt = dt*(double)(mult*2*Nf);
    
    wdms->Tobs = dt*(double)(Nt*Nf);
    
}


double tukey_weight_at_time(double t, double alpha, int N)
{
    double x, imin, imax, nwin, filter;

    if(alpha <= 0.0 || N <= 1) return 1.0;

    x = t/dt;
    imin = alpha*(double)(N-1)/2.0;
    imax = (double)(N-1)*(1.0-alpha/2.0);
    nwin = (double)N-imax;
    filter = 1.0;

    if(x < 0.0 || x > (double)(N-1)) return 0.0;
    if(imin > 0.0 && x < imin)
    {
        filter = 0.5*(1.0+cos(M_PI*(x/imin-1.0)));
    }
    if(nwin > 0.0 && x > imax)
    {
        filter = 0.5*(1.0+cos(M_PI*((x-imax)/nwin)));
    }

    return filter;
}

void tukey(double *data, double alpha, int N)
{
  int i, imin, imax;
  double filter;
  
  imin = (int)(alpha*(double)(N-1)/2.0);
  imax = (int)((double)(N-1)*(1.0-alpha/2.0));
  
    int Nwin = N-imax;

 for(i=0; i< N; i++)
  {
    filter = 1.0;
    if(i<imin) filter = 0.5*(1.0+cos(M_PI*( (double)(i)/(double)(imin)-1.0 )));
    if(i>imax) filter = 0.5*(1.0+cos(M_PI*( (double)(i-imax)/(double)(Nwin))));
    data[i] *= filter;
  }
  
}

int PhenomT_AP(double *params, int Nsmax, double *TS, double *tspace, double *AS, double *PS, double *FS, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax, double *setup)
    {
    int ii, i, j, Ns;
    double complex H22;
    double t, tnext, tstart, tstart_requested, tstart_constellation, f, x, y, om, A;
    double m1, m2, Mtot, eta, tc, Mc;
    double dPhase, delT, dPhstar, omstar;
    double tend;
    double t22, fring, fdamp;
    double fny, fmx;
    int Ntc;
    double *tarrayB, *omegatB;
    double *omegat;
    
    struct IMRPhenomT *IMRPT  = malloc(sizeof(struct IMRPhenomT));
    
    FILE *out;
    
    omstar = 2.0*M_PI*fstar;  // reference angular transfer frequency
    
    enforce_phenom_mass_hierarchy(&params[0], &params[1], &params[2], &params[3]);

    m1 = params[0];
    m2 = params[1];
    tc = params[5];
    
    Mtot = (m1 + m2);  // total mass in seconds
    eta = (m1*m2)/(Mtot*Mtot);
    Mc = pow(m1*m2,3.0/5.0)/pow(Mtot, 1.0/5.0);
        
        // set up the IMRPhenomT model
        allocate_IMRPhenomT(IMRPT);
        // PN terms
        InspiralT3(params, IMRPT);
        // Beyond PN fit
        InspiralFit(params, IMRPT);
        // Merger-Ringdown fitting parameters
        MergerRingdownFit(params, IMRPT);
    
    t22 = tc+Mtot*IMRPT->tCut22;
    fring = IMRPT->omegaRING/(2.*M_PI*Mtot);
    fdamp = IMRPT->alpha1RD/Mtot;
    
    if(thm_diagnostics_enabled) printf("fring = %.15e\n", fring);
    
    // unsure why sqrt(2). Thought it should be 2
    A = sqrt(2.0)*eta*Mtot/(exp(params[6])*GPSEC);
        
        // just for testing
        
        /*out = fopen("PT.dat","w");
        for(i=0; i< 10000; i++)
        {
            t = 3.15e7*(double)(i)/(double)(10000);
            om = IMRPhenomTomega22((t-tc)/Mtot, eta, IMRPT)/Mtot;
            x = pow(0.5*om*Mtot,2./3.);
            H22 = IMRPhenomTHMAmp((t-tc)/Mtot, x, IMRPT);
            fprintf(out,"%e %e %e\n", t, om/(2.0*M_PI), A*cabs(H22));
        }
        fclose(out); */
        
        
        // allow for light travel time and TDI time delays
        tend = THM_RESPONSE_LATE_MARGIN_SECONDS+1000.0*Mtot;
        if(tc+tend > constellation_tmax-CONSTELLATION_LIGHT_TIME_SECONDS)
        {
            double requested_tend = tend;
            tend = constellation_tmax-CONSTELLATION_LIGHT_TIME_SECONDS-tc;
            if(tend <= 0.0)
            {
                fprintf(stderr, "Error: merger time %.15e leaves no post-merger waveform support inside the constellation-safe range.\n", tc);
                exit(1);
            }
            fprintf(stderr, "Warning: requested post-merger padding %.15e exceeds the constellation-safe range; using %.15e instead.\n", requested_tend, tend);
        }
        
       // printf("Mass (s) = %e\n", Mtot);
        
        tarrayB = double_vector(Nsmax);
        omegatB = double_vector(Nsmax);
        omegat = double_vector(Nsmax);
        
       dPhstar = 0.1; // phase increment when passing through transfer frequency harmonics
        dPhase = 0.5;
        
        
        /*
         Start at peak and work backwards.

         The requested pre-observation waveform padding is measured in the
         observation-time coordinate, while t is measured relative to merger.
         By default this padding is zero, so the first waveform-grid point is
         clamped exactly to the observation start t = 0.
         The constellation splines are only valid on their own tarray domain,
         and barycenter_time evaluates the spacecraft position once at tspace
         and once after a k.r shift. Keep one rounded-AU light-time margin so
         both evaluations stay inside the constellation interpolation range.
         */
        tstart_requested = -WAVEFORM_PRE_PADDING_SECONDS;
        tstart_constellation = constellation_tmin+CONSTELLATION_LIGHT_TIME_SECONDS;
        tstart = tstart_requested;
        if(tstart < tstart_constellation)
        {
            tstart = tstart_constellation;
            fprintf(stderr, "Warning: requested waveform start %.15e is outside the constellation-safe range; using %.15e instead.\n", tstart_requested, tstart);
        }
        // start at peak and work backwards
        // The IMRPhenomT model defined merger at t=0
        // we have to offset this by the merger time tc
        // in the physical time array
        // The spacing in time is designed to give a maximum
        // phase shift from the previous point of dPhase radians
        // This is computed by dividing dPhase by the angular
        // frequency omega. Note that IMRPhenomT uses units where
        // the total mass = 1, so we have to divide by Mtotal in seconds
        // If deltaT = dPhase/omega exceeds dTmax (usually set to about a
        // day) then it gets set to dTmax. Note that once we get more than
        // t = 100 Mtotal from merger we start making dPhase larger to take
        // bigger steps. We work back from merger and forward from merger.
        j = 0;
        t = 0.0;
        while(1)
        {
            if(j >= Nsmax)
            {
                fprintf(stderr, "Error: PhenomT_AP exceeded Nsmax while building the backward waveform grid.\n");
                exit(1);
            }

            om = IMRPhenomTomega22(t/Mtot, eta, IMRPT)/Mtot;
            
            tarrayB[j] = t+tc;
            omegatB[j] = om;
            j++;

            if(t+tc <= tstart) break;

            delT = dPhase/om;
            if(thm_near_delay_zero(om, omstar))
            {
                delT = dPhstar/om;
            }
            if(delT > dTmax) delT = dTmax;
            if(delT < dTmin) delT = dTmin;
            tnext = t-delT;
            if(tnext+tc < tstart) tnext = tstart-tc;
            t = tnext;
            if(t < -100.0*Mtot) dPhase *= 1.1;
        }
        
        Ntc = j-1;
        Ns = j;
        
        // flip to increasing time direction
        for(i=0; i< Ns; i++)
        {
            //printf("%d %d\n", i, Ns-1-i);
            tspace[i] = tarrayB[Ns-1-i];
            omegat[i] = omegatB[Ns-1-i];
        }
        
        //printf("%e\n", TS[Ntc]);
        
        // continue from peak and work forwards
        dPhase = 0.5;
        j = 0;
        t = dPhase/omegat[Ns-1];
        do
        {
            om = IMRPhenomTomega22(t/Mtot, eta, IMRPT)/Mtot;
            tspace[Ns+j] = t+tc;
            omegat[Ns+j] = om;
            //printf("%e %e\n", t, om);
            delT = dPhase/om;
            if(thm_near_delay_zero(om, omstar))
            {
                delT = dPhstar/om;
            }
            if(delT > dTmax) delT = dTmax;
            if(delT < dTmin) delT = dTmin;
            t += delT;
            if(t > 10.0*Mtot) dPhase *= 1.1;
            j++;
        }while(t-delT < tend);
        
        Ns += j;
        
        //printf("%d\n", Ns);

        

    
        barycenter_time(TS, tspace, params, SPacc, SPspline, Ns);
      
        // Barycenter angular frequency using Barycenter time
         for(i=0; i< Ns; i++)
         {
             omegatB[i] = IMRPhenomTomega22((TS[i]-tc)/Mtot, eta, IMRPT)/Mtot;
         }
    
       // Get the amplitude as a function of time
        for(i=0; i< Ns; i++)
        {
            x = pow(0.5*omegatB[i]*Mtot,2./3.);  // PN velocity
            H22 = IMRPhenomTHMAmp((TS[i]-tc)/Mtot, x, IMRPT);
            AS[i] = A*cabs(H22);
            FS[i] = omegatB[i]/(2.0*M_PI);
        }
    
      // catch where frequency freezes
       
      y = fring/1.0e2;
      
      ii = 0;
       do
       {
           ii++;
           x = fabs(FS[ii]-fring);
       }while(x > y);
    
       ii -= 1;
    
    
    
       fny = 1.0/(2.0*dt);
       fmx = 1.6*fring;
       if(fmx > fny) fmx = fny;
    
        //printf("%e %e %e\n", FS[ii], fring, fny);
    
       x = (fmx-FS[ii])/(double)(Ns-ii);
    
        // keep the frequency growing (for t-f mapping)
       for(i=ii; i< Ns; i++) FS[i] = FS[i-1]+x;
    
     
       // integrate the angular frequency to find the phase
        gsl_interp_accel *acc = gsl_interp_accel_alloc();
        gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, Ns);
        gsl_spline_init(spline, TS, omegatB, Ns);
        
        PS[0] = 0.0;  // will end up setting so that phase = phi0 at t=tc
        for(i=1; i< Ns; i++)
        {
            PS[i] = PS[i-1]+gsl_spline_eval_integ(spline, TS[i-1], TS[i], acc);
        }
        
        // fix merger phase to equal phi_c
        x = params[4]-PS[Ntc];
        for(i=0; i< Ns; i++) PS[i] += x;
    
        /*
         * tc and t22 are source-waveform events, while TS is the SSB
         * output/data coordinate.  Map the events through the paired
         * guiding-center source coordinate tspace before laying out the
         * endpoint FFT on TS.
         */
        transformplan_custom(Mc, Mtot,
                             linear_interp_clamped(Ns, tspace, TS, tc),
                             TS, omegat,
                             linear_interp_clamped(Ns, tspace, TS, t22),
                             fring, fdamp, TS[Ns-1], setup, 0, 0.0, 0.0);

  // printf("%e %e %e %e %e\n", setup[0], setup[1], setup[2], setup[3], setup[4]);
    
    /*
    out = fopen("PTS.dat","w");
    for(i=0; i< Ns; i++)
    {
        fprintf(out,"%.15e %.15e %.15e %.15e %.15e %.15e\n", TS[i], tspace[i], FS[i], omegat[i]/(2.0*M_PI), PS[i], AS[i]);
    }
    fclose(out);
    */
    
    free(tarrayB);
    free(omegatB);
    free(omegat);
    
  return Ns;
        
}

static double thm_factorial_int(int n)
{
    double out = 1.0;
    int i;

    for(i=2; i<=n; i++) out *= (double)i;
    return out;
}

static double complex spin_weighted_y_minus2(int ell, int emm, double theta, double phi)
{
    const int n = 2; /* d^l_{m,-s} with s=-2 */
    double c = cos(0.5*theta);
    double s = sin(0.5*theta);
    double pref, dpref, dsum;
    int k;

    pref = sqrt((2.0*(double)ell+1.0)/(4.0*M_PI));
    dpref = sqrt(thm_factorial_int(ell+emm)*
                 thm_factorial_int(ell-emm)*
                 thm_factorial_int(ell+n)*
                 thm_factorial_int(ell-n));
    dsum = 0.0;
    for(k=0; k<=2*ell; k++)
    {
        int a = ell+emm-k;
        int b = k;
        int cden = n-emm+k;
        int d = ell-n-k;
        int cpow_exp;
        int sexp;
        double term;

        if(a < 0 || b < 0 || cden < 0 || d < 0) continue;
        cpow_exp = 2*ell+emm-n-2*k;
        sexp = n-emm+2*k;
        if(cpow_exp < 0 || sexp < 0) continue;

        term = dpref/(thm_factorial_int(a)*
                      thm_factorial_int(b)*
                      thm_factorial_int(cden)*
                      thm_factorial_int(d));
        if(k%2) term = -term;
        term *= pow(c, (double)cpow_exp)*pow(s, (double)sexp);
        dsum += term;
    }

    return pref*dsum*cexp(I*(double)emm*phi);
}

void thm_projection_coefficients(int ell, int emm, double cosi, double orbital_azimuth, double psi, THMProjection *projection)
{
    double theta;
    double complex ylm;
    double yr, yi;

    if(projection == NULL) return;
    if(cosi > 1.0) cosi = 1.0;
    if(cosi < -1.0) cosi = -1.0;
    theta = acos(cosi);

    /*
     * The mode convention is h_lm=A_lm exp(-i phi_lm), and the complex strain
     * is H=hplus-i hcross=sum h_lm * _{-2}Y_lm.  Polarization is applied as
     * H -> exp(2 i psi) H, matching the older 22 hphc() convention.
     */
    ylm = spin_weighted_y_minus2(ell, emm, theta, orbital_azimuth)*cexp(2.0*I*psi);
    yr = creal(ylm);
    yi = cimag(ylm);

    projection->ell = ell;
    projection->emm = emm;
    projection->plus_cos = yr;
    projection->plus_sin = yi;
    projection->cross_cos = -yi;
    projection->cross_sin = yr;
}

int thm_build_folded_carriers(int nmodes, const IMRPhenomTHMMode *modes, const THMProjection *projection, THMFoldedCarrier *carrier, int max_carriers)
{
#if !THM_TDI_USE_FOLDED_CARRIERS
    int i;

    if(modes == NULL || projection == NULL || carrier == NULL || max_carriers <= 0)
    {
        return 0;
    }
    if(nmodes > max_carriers) nmodes = max_carriers;
    for(i=0; i<nmodes; i++)
    {
        const THMProjection *p = &projection[i];
        THMFoldedCarrier *c = &carrier[i];

        memset(c, 0, sizeof(*c));
        c->ell = modes[i].ell;
        c->emm = modes[i].emm;
        c->mode_index = i;
        c->paired_mode_index = -1;
        c->is_pair = 0;
        c->hp_cos = p->plus_cos;
        c->hp_sin = p->plus_sin;
        c->hc_cos = p->cross_cos;
        c->hc_sin = p->cross_sin;
        c->hpf_cos = p->plus_sin;
        c->hpf_sin = -p->plus_cos;
        c->hcf_cos = p->cross_sin;
        c->hcf_sin = -p->cross_cos;
    }
    return nmodes;
#else
    int used[IMRPHENOMTHM_MAX_MODES];
    int i, j, ncarriers;

    if(modes == NULL || projection == NULL || carrier == NULL || max_carriers <= 0)
    {
        return 0;
    }
    if(nmodes > IMRPHENOMTHM_MAX_MODES) nmodes = IMRPHENOMTHM_MAX_MODES;
    for(i=0; i<IMRPHENOMTHM_MAX_MODES; i++) used[i] = 0;

    ncarriers = 0;
    for(i=0; i<nmodes; i++)
    {
        int pair = -1;
        int pos = i;
        int neg = -1;
        int q;
        const THMProjection *pp;
        const THMProjection *pn;
        THMFoldedCarrier *c;

        if(used[i]) continue;
        if(ncarriers >= max_carriers)
        {
            fprintf(stderr, "Error: too many folded THM carriers.\n");
            exit(1);
        }

        if(modes[i].emm != 0)
        {
            for(j=i+1; j<nmodes; j++)
            {
                if(!used[j] && modes[j].ell == modes[i].ell &&
                   modes[j].emm == -modes[i].emm)
                {
                    pair = j;
                    break;
                }
            }
        }

        c = &carrier[ncarriers];
        memset(c, 0, sizeof(*c));
        c->ell = modes[i].ell;
        c->emm = modes[i].emm;
        c->mode_index = i;
        c->paired_mode_index = -1;
        c->is_pair = 0;

        if(pair >= 0)
        {
            if(modes[i].emm > 0)
            {
                pos = i;
                neg = pair;
            }
            else
            {
                pos = pair;
                neg = i;
            }

            /*
             * IMRPhenomTHM returns phi_-m = -phi_+m - ell*pi for these
             * aligned-spin pairs, and the amplitudes are identical.  Fold the
             * pair into one positive-m carrier while preserving the current
             * mode-by-mode quadrature convention:
             *
             *   h_pair^quad = h_+m(phi_+m+pi/2) + h_-m(phi_-m+pi/2)
             *
             * This is not the same auxiliary quadrature as shifting the
             * already-folded real pair waveform, which is what the legacy
             * single-22 code used for extractAP().
             */
            q = (modes[pos].ell % 2 == 0) ? 1 : -1;
            pp = &projection[pos];
            pn = &projection[neg];

            c->ell = modes[pos].ell;
            c->emm = modes[pos].emm;
            c->mode_index = pos;
            c->paired_mode_index = neg;
            c->is_pair = 1;
            c->hp_cos = pp->plus_cos + q*pn->plus_cos;
            c->hp_sin = pp->plus_sin - q*pn->plus_sin;
            c->hc_cos = pp->cross_cos + q*pn->cross_cos;
            c->hc_sin = pp->cross_sin - q*pn->cross_sin;
            c->hpf_cos = pp->plus_sin + q*pn->plus_sin;
            c->hpf_sin = -pp->plus_cos + q*pn->plus_cos;
            c->hcf_cos = pp->cross_sin + q*pn->cross_sin;
            c->hcf_sin = -pp->cross_cos + q*pn->cross_cos;

            used[pos] = 1;
            used[neg] = 1;
        }
        else
        {
            pp = &projection[i];
            c->hp_cos = pp->plus_cos;
            c->hp_sin = pp->plus_sin;
            c->hc_cos = pp->cross_cos;
            c->hc_sin = pp->cross_sin;
            c->hpf_cos = pp->plus_sin;
            c->hpf_sin = -pp->plus_cos;
            c->hcf_cos = pp->cross_sin;
            c->hcf_sin = -pp->cross_cos;
            used[i] = 1;
        }

        ncarriers++;
    }

    return ncarriers;
#endif
}

static double thm_max_mode_omega_seconds(const IMRPhenomTHM *model, double tau, double mtot)
{
    double max_omega = 0.0;
    int k;

    for(k=0; k<model->nmodes; k++)
    {
        double omega = fabs(IMRPhenomTHMModeFrequency(model, k, tau))/mtot;
        if(omega > max_omega) max_omega = omega;
    }
    return max_omega;
}

int PhenomTHM_AP(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax)
{
    double m1, m2, chi1, chi2, Mtot, eta, tc, distance;
    double t, tnext, tstart, tstart_requested, tstart_constellation, tend;
    double om, delT, dPhase, dPhstar, omstar;
    double strain_scale;
    double *tarrayB, *omegatB, *omegat;
    double *tau, *phi22;
    int status;
    int i, j, k, Ns, Ntc;

    if(params == NULL || model == NULL || modes == NULL || nmodes == NULL ||
       TS == NULL || tspace == NULL || mode_amp == NULL || mode_phase == NULL ||
       mode_freq == NULL)
    {
        return 0;
    }

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    tc = params[5];
    distance = exp(params[6]);
    Mtot = m1 + m2;
    eta = (m1*m2)/(Mtot*Mtot);

    if(requested_modes != NULL && requested_nmodes > 0)
    {
        if(requested_nmodes > IMRPHENOMTHM_MAX_MODES)
        {
            fprintf(stderr, "Error: requested %d THM modes, but only %d are supported by this driver.\n",
                    requested_nmodes, IMRPHENOMTHM_MAX_MODES);
            exit(1);
        }
        *nmodes = requested_nmodes;
        for(i=0; i<*nmodes; i++) modes[i] = requested_modes[i];
    }
    else
    {
        *nmodes = IMRPhenomTHMDefaultModes(modes, IMRPHENOMTHM_MAX_MODES);
    }
    status = IMRPhenomTHMInitialize(model, m1, m2, chi1, chi2, modes, *nmodes);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMInitialize failed: %d\n", status);
        exit(1);
    }

    tarrayB = double_vector(Nsmax);
    omegatB = double_vector(Nsmax);
    omegat = double_vector(Nsmax);
    tau = double_vector(Nsmax);
    phi22 = double_vector(Nsmax);

    omstar = 2.0*M_PI*fstar;
    dPhstar = 0.1;
    dPhase = 0.5;

    tend = THM_RESPONSE_LATE_MARGIN_SECONDS+1000.0*Mtot;
    if(tc+tend > constellation_tmax-CONSTELLATION_LIGHT_TIME_SECONDS)
    {
        double requested_tend = tend;
        tend = constellation_tmax-CONSTELLATION_LIGHT_TIME_SECONDS-tc;
        if(tend <= 0.0)
        {
            fprintf(stderr, "Error: merger time %.15e leaves no post-merger waveform support inside the constellation-safe range.\n", tc);
            exit(1);
        }
        fprintf(stderr, "Warning: requested post-merger padding %.15e exceeds the constellation-safe range; using %.15e instead.\n", requested_tend, tend);
    }

    tstart_requested = -WAVEFORM_PRE_PADDING_SECONDS;
    tstart_constellation = constellation_tmin+CONSTELLATION_LIGHT_TIME_SECONDS;
    tstart = tstart_requested;
    if(tstart < tstart_constellation)
    {
        tstart = tstart_constellation;
        fprintf(stderr, "Warning: requested waveform start %.15e is outside the constellation-safe range; using %.15e instead.\n", tstart_requested, tstart);
    }

    /*
     * Build one adaptive detector-time grid for all modes.  The step size is
     * controlled by the largest absolute mode frequency, so the 55 mode drives
     * the sampling when it is present.  This is deliberately conservative for
     * the first multi-mode TDI reference path.
     */
    j = 0;
    t = 0.0;
    while(1)
    {
        if(j >= Nsmax)
        {
            fprintf(stderr, "Error: PhenomTHM_AP exceeded Nsmax while building the backward waveform grid.\n");
            exit(1);
        }

        om = thm_max_mode_omega_seconds(model, t/Mtot, Mtot);
        tarrayB[j] = t+tc;
        omegatB[j] = om;
        j++;

        if(t+tc <= tstart) break;
        delT = dPhase/om;
        if(thm_near_delay_zero(om, omstar)) delT = dPhstar/om;
        if(delT > THM_DTM_MAX) delT = THM_DTM_MAX;
        if(delT < dTmin) delT = dTmin;
        tnext = t-delT;
        if(tnext+tc < tstart) tnext = tstart-tc;
        t = tnext;
        if(t < -100.0*Mtot) dPhase *= 1.1;
    }

    Ntc = j-1;
    Ns = j;
    for(i=0; i<Ns; i++)
    {
        tspace[i] = tarrayB[Ns-1-i];
        omegat[i] = omegatB[Ns-1-i];
    }

    dPhase = 0.5;
    j = 0;
    t = dPhase/omegat[Ns-1];
    do
    {
        if(Ns+j >= Nsmax)
        {
            fprintf(stderr, "Error: PhenomTHM_AP exceeded Nsmax while building the forward waveform grid.\n");
            exit(1);
        }
        om = thm_max_mode_omega_seconds(model, t/Mtot, Mtot);
        tspace[Ns+j] = t+tc;
        omegat[Ns+j] = om;
        delT = dPhase/om;
        if(thm_near_delay_zero(om, omstar)) delT = dPhstar/om;
        if(delT > THM_DTM_MAX) delT = THM_DTM_MAX;
        if(delT < dTmin) delT = dTmin;
        t += delT;
        if(t > 10.0*Mtot) dPhase *= 1.1;
        j++;
    } while(t-delT < tend);
    Ns += j;

    barycenter_time(TS, tspace, params, SPacc, SPspline, Ns);
    for(i=0; i<Ns; i++) tau[i] = (TS[i]-tc)/Mtot;

    status = IMRPhenomTHMBuildPhi22Grid(model, Ns, tau, 0.0, phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMBuildPhi22Grid failed: %d\n", status);
        exit(1);
    }
    {
        double shift = params[4]-phi22[Ntc];
        for(i=0; i<Ns; i++) phi22[i] += shift;
    }
    status = IMRPhenomTHMSetPhaseOffsetsFromGrid(model, Ns, tau, phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMSetPhaseOffsetsFromGrid failed: %d\n", status);
        exit(1);
    }

    strain_scale = sqrt(2.0)*eta*Mtot/(distance*GPSEC);
    for(i=0; i<Ns; i++)
    {
        for(k=0; k<*nmodes; k++)
        {
            IMRPhenomTHMModeSample sample;
            status = IMRPhenomTHMEvaluateMode(model, k, tau[i], phi22[i], &sample);
            if(status != 0)
            {
                fprintf(stderr, "IMRPhenomTHMEvaluateMode failed at mode %d sample %d: %d\n", k, i, status);
                exit(1);
            }
            mode_amp[k][i] = strain_scale*sample.amplitude;
            mode_phase[k][i] = sample.phase;
            mode_freq[k][i] = sample.omega/(2.0*M_PI*Mtot);
        }
    }
    /*
     * Do not call the legacy residual-phase unwrap() here.  These are full
     * mode phases, and adjacent adaptive samples can differ by many radians in
     * the early inspiral.  Folding those jumps into [-pi,pi] leaves the node
     * values equivalent modulo 2pi, but gives the phase spline the wrong path
     * between nodes and spoils delayed TDI evaluations.  IMRPhenomTHM.c returns
     * phases on a physical continuous branch; keep that branch intact.
     */

    free_double_vector(tarrayB);
    free_double_vector(omegatB);
    free_double_vector(omegat);
    free_double_vector(tau);
    free_double_vector(phi22);

    return Ns;
}

static double thm_ssb_output_time_from_center_source_time(double center_source_time, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline)
{
    double costh, phi, sinth, cosph, sinph;
    double kv[3], x[3];
    double kr, output_time;
    int j;

    costh = sin(params[7]);
    phi = params[8];
    sinth = sqrt(1.0-costh*costh);
    cosph = cos(phi);
    sinph = sin(phi);

    kv[0] = -sinth*cosph;
    kv[1] = -sinth*sinph;
    kv[2] = -costh;

    for(j=0; j<3; j++)
    {
        x[j] = gsl_spline_eval(SPspline[j], center_source_time, SPacc[j]);
    }
    kr = 0.0;
    for(j=0; j<3; j++) kr += x[j]*kv[j];
    output_time = center_source_time+kr;

    for(j=0; j<3; j++)
    {
        x[j] = gsl_spline_eval(SPspline[j], output_time, SPacc[j]);
    }
    kr = 0.0;
    for(j=0; j<3; j++) kr += x[j]*kv[j];

    return center_source_time+kr;
}

static double thm_abs_mode_frequency_hz_at_detector_time(const IMRPhenomTHM *model, int mode_index, double detector_time, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline)
{
    double tc, mtot;
    double source_time;
    double tau;
    double omega;

    tc = params[5];
    mtot = params[0]+params[1];
    source_time = thm_ssb_output_time_from_center_source_time(detector_time, params, SPacc, SPspline);
    tau = (source_time-tc)/mtot;
    omega = IMRPhenomTHMModeFrequency(model, mode_index, tau);
    if(!isfinite(omega)) return 0.0;
    return fabs(omega)/(2.0*M_PI*mtot);
}

static double thm_abs_mode_frequency_hz_at_source_time(const IMRPhenomTHM *model, int mode_index, double source_time, double *params)
{
    double tc, mtot;
    double tau;
    double omega;

    tc = params[5];
    mtot = params[0]+params[1];
    tau = (source_time-tc)/mtot;
    omega = IMRPhenomTHMModeFrequency(model, mode_index, tau);
    if(!isfinite(omega)) return 0.0;
    return fabs(omega)/(2.0*M_PI*mtot);
}

static double thm_mode_amplitude_unscaled(const IMRPhenomTHM *model, int mode_index, double tau)
{
    const IMRPhenomTHMModeState *mode;
    double omega22, x22;
    double complex amp_complex;

    if(model == NULL || mode_index < 0 || mode_index >= model->nmodes)
    {
        return 0.0;
    }

    mode = &model->modes[mode_index];
    if(mode->is_zero_by_symmetry)
    {
        return 0.0;
    }

    omega22 = IMRPhenomTHMOmega22(model, tau);
    x22 = pow(0.5*omega22, 2.0/3.0);
    if(mode->ell == 2 && mode->abs_emm == 2)
    {
        amp_complex = IMRPhenomTHMAmp(tau, x22,
                                      (struct IMRPhenomT *)&model->mode22);
    }
    else
    {
        amp_complex = IMRPhenomTHMModeComplexAmplitude(mode, tau, x22);
    }

    if(!isfinite(creal(amp_complex)) || !isfinite(cimag(amp_complex)))
    {
        return 0.0;
    }

    return cabs(amp_complex);
}

static double thm_max_omega_source(const IMRPhenomTHM *model, double source_time, double *params)
{
    double max_omega;
    int k;

    max_omega = 0.0;
    for(k=0; k<model->nmodes; k++)
    {
        double f = thm_abs_mode_frequency_hz_at_source_time(model, k,
                                                            source_time,
                                                            params);
        double omega = 2.0*M_PI*f;
        if(isfinite(omega) && omega > max_omega) max_omega = omega;
    }

    return max_omega;
}

static double thm_mode_ring_frequency_hz(const IMRPhenomTHM *model, int mode_index, double Mtot)
{
    const IMRPhenomTHMModeState *mode;

    if(model == NULL || mode_index < 0 || mode_index >= model->nmodes ||
       Mtot <= 0.0)
    {
        return 0.0;
    }

    mode = &model->modes[mode_index];
    if(mode->ell == 2 && mode->abs_emm == 2 &&
       model->mode22.omegaRING > 0.0)
    {
        return fabs(model->mode22.omegaRING)/(2.0*M_PI*Mtot);
    }
    if(mode->phase.omegaRING > 0.0)
    {
        return fabs(mode->phase.omegaRING)/(2.0*M_PI*Mtot);
    }

    return 0.0;
}

static int thm_intrinsic_frequency_interval_reason(double f0, double fmid, double f1, double width, double fmode_max, int include_delay_zeros)
{
    double ferr;
    double fscale;
    double fend;
    double phase_err;
    int reason;

    f0 = fabs(f0);
    fmid = fabs(fmid);
    f1 = fabs(f1);
    ferr = fabs(fmid-0.5*(f0+f1));
    fscale = fmid;
    if(fscale < f0) fscale = f0;
    if(fscale < f1) fscale = f1;
    if(fscale < 1.0e-8) fscale = 1.0e-8;

    fend = fabs(f1-f0)/fscale;
    if(fend > THM_INTRINSIC_FREQ_STEP_REL_TOL) return THM_AP_FAIL_FREQ_STEP;

    if(include_delay_zeros)
    {
        reason = thm_delay_zero_interval_reason(f0, fmid, f1, width,
                                                fmode_max);
        if(reason != THM_AP_INTERVAL_OK) return reason;
    }

    phase_err = 0.5*M_PI*ferr*width;
    if(phase_err > THM_INTRINSIC_PHASE_CURVATURE_TOL)
    {
        return THM_AP_FAIL_PHASE_CURVATURE;
    }
    if(ferr/fscale > THM_INTRINSIC_FREQ_REL_CURVATURE_TOL)
    {
        return THM_AP_FAIL_REL_CURVATURE;
    }

    return THM_AP_INTERVAL_OK;
}

static int thm_intrinsic_amplitude_interval_reason(double a0, double amid, double a1)
{
    double ascale;
    double astep;
    double aerr;

    a0 = fabs(a0);
    amid = fabs(amid);
    a1 = fabs(a1);
    ascale = a0;
    if(ascale < amid) ascale = amid;
    if(ascale < a1) ascale = a1;
    if(ascale <= 0.0 || !isfinite(ascale))
    {
        return THM_AP_INTERVAL_OK;
    }

    astep = fabs(a1-a0)/ascale;
    if(astep > THM_INTRINSIC_AMP_STEP_REL_TOL)
    {
        return THM_AP_FAIL_AMP_STEP;
    }

    aerr = fabs(amid-0.5*(a0+a1))/ascale;
    if(aerr > THM_INTRINSIC_AMP_REL_CURVATURE_TOL)
    {
        return THM_AP_FAIL_AMP_CURVATURE;
    }

    return THM_AP_INTERVAL_OK;
}

static int thm_intrinsic_interval_reason(const IMRPhenomTHM *model, double t0, double t1, double *params, int include_delay_zeros, int *fail_mode)
{
    double width, tmid;
    double Mtot, tc;
    int k;

    width = t1-t0;
    if(fail_mode != NULL) *fail_mode = -1;
    if(width <= dTmin) return THM_AP_INTERVAL_OK;
    tmid = 0.5*(t0+t1);
    Mtot = params[0]+params[1];
    tc = params[5];

    for(k=0; k<model->nmodes; k++)
    {
        double f0, f1, fmid;
        double fmode_max;
        double tau0, tau1, taumid;
        double a0, a1, amid;
        int reason;

        f0 = thm_abs_mode_frequency_hz_at_source_time(model, k, t0, params);
        f1 = thm_abs_mode_frequency_hz_at_source_time(model, k, t1, params);
        fmid = thm_abs_mode_frequency_hz_at_source_time(model, k, tmid, params);
        fmode_max = thm_mode_ring_frequency_hz(model, k, Mtot);
        reason = thm_intrinsic_frequency_interval_reason(f0, fmid, f1,
                                                         width, fmode_max,
                                                         include_delay_zeros);
        if(reason != THM_AP_INTERVAL_OK)
        {
            if(fail_mode != NULL) *fail_mode = k;
            return reason;
        }

        if(include_delay_zeros)
        {
            double Lnom = 1.0/(2.0*M_PI*fstar);
            double offsets[4];
            int noffsets;
            int p;

            offsets[0] = -Lnom;
            offsets[1] = -2.0*Lnom;
            offsets[2] = -4.0*Lnom;
            offsets[3] = -8.0*Lnom;
            noffsets = thm_tdi_generation == 2 ? 4 : 3;
            for(p=0; p<noffsets; p++)
            {
                f0 = thm_abs_mode_frequency_hz_at_source_time(model, k,
                                                               t0+offsets[p],
                                                               params);
                f1 = thm_abs_mode_frequency_hz_at_source_time(model, k,
                                                               t1+offsets[p],
                                                               params);
                fmid = thm_abs_mode_frequency_hz_at_source_time(model, k,
                                                                 tmid+offsets[p],
                                                                 params);
                reason = thm_intrinsic_frequency_interval_reason(f0, fmid, f1,
                                                                 width,
                                                                 fmode_max, 1);
                if(reason != THM_AP_INTERVAL_OK)
                {
                    if(fail_mode != NULL) *fail_mode = k;
                    return reason;
                }
            }
        }

        tau0 = (t0-tc)/Mtot;
        tau1 = (t1-tc)/Mtot;
        taumid = (tmid-tc)/Mtot;
        a0 = thm_mode_amplitude_unscaled(model, k, tau0);
        a1 = thm_mode_amplitude_unscaled(model, k, tau1);
        amid = thm_mode_amplitude_unscaled(model, k, taumid);
        reason = thm_intrinsic_amplitude_interval_reason(a0, amid, a1);
        if(reason != THM_AP_INTERVAL_OK)
        {
            if(fail_mode != NULL) *fail_mode = k;
            return reason;
        }
    }

    return THM_AP_INTERVAL_OK;
}

static int thm_intrinsic_interval_ok(const IMRPhenomTHM *model, double t0, double t1, double *params, int include_delay_zeros)
{
    return thm_intrinsic_interval_reason(model, t0, t1, params,
                                         include_delay_zeros, NULL) == THM_AP_INTERVAL_OK;
}

static double thm_intrinsic_adaptive_step(const IMRPhenomTHM *model, double t, double tstop, double proposed_step, double *params, int include_delay_zeros, THMAPStepDiagnostic *diag)
{
    double max_omega, delT;

    if(diag != NULL)
    {
        memset(diag, 0, sizeof(*diag));
        diag->proposed_step = proposed_step;
        diag->last_fail_reason = THM_AP_INTERVAL_OK;
        diag->last_fail_mode = -1;
        diag->last_fail_probe = -1;
    }

    delT = proposed_step;
    if(!isfinite(delT) || delT <= 0.0) delT = THM_INTRINSIC_DTM_MAX;
    if(delT > THM_INTRINSIC_DTM_MAX)
    {
        delT = THM_INTRINSIC_DTM_MAX;
        if(diag != NULL) diag->dtmax_cap = 1;
    }
    max_omega = thm_max_omega_source(model, t, params);
    if(diag != NULL) diag->max_omega = max_omega;
    if(max_omega > 0.0 && delT > THM_INTRINSIC_MAX_PHASE_STEP/max_omega)
    {
        delT = THM_INTRINSIC_MAX_PHASE_STEP/max_omega;
        if(diag != NULL) diag->phase_step_cap = 1;
    }
    if(delT < dTmin) delT = dTmin;
    if(t+delT > tstop)
    {
        delT = tstop-t;
        if(diag != NULL) diag->tstop_cap = 1;
    }
    if(delT < dTmin) delT = dTmin;
    if(diag != NULL) diag->capped_step = delT;

    while(delT > dTmin &&
          t+delT <= tstop &&
          !thm_intrinsic_interval_ok(model, t, t+delT, params,
                                     include_delay_zeros))
    {
        int fail_mode = -1;
        int reason = thm_intrinsic_interval_reason(model, t, t+delT,
                                                   params, include_delay_zeros,
                                                   &fail_mode);
        if(diag != NULL)
        {
            diag->shrink_count++;
            diag->last_fail_reason = reason;
            diag->last_fail_mode = fail_mode;
            if(reason == THM_AP_FAIL_FREQ_STEP) diag->fail_freq_step++;
            else if(reason == THM_AP_FAIL_DELAY_ZERO) diag->fail_delay_zero++;
            else if(reason == THM_AP_FAIL_PHASE_CURVATURE) diag->fail_phase_curvature++;
            else if(reason == THM_AP_FAIL_REL_CURVATURE) diag->fail_rel_curvature++;
            else if(reason == THM_AP_FAIL_AMP_STEP) diag->fail_amp_step++;
            else if(reason == THM_AP_FAIL_AMP_CURVATURE) diag->fail_amp_curvature++;
        }
        delT *= THM_INTRINSIC_STEP_SHRINK_FACTOR;
        if(delT < dTmin) delT = dTmin;
    }

    if(diag != NULL) diag->accepted_step = delT;

    return delT;
}

static double thm_max_omega_detector(const IMRPhenomTHM *model, double detector_time, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline)
{
    double max_omega;
    double source_time;
    double Lnom;
    double offsets[5];
    int noffsets;
    int k;

    max_omega = 0.0;
    source_time = thm_ssb_output_time_from_center_source_time(
        detector_time, params, SPacc, SPspline);
    Lnom = 1.0/(2.0*M_PI*fstar);
    offsets[0] = 0.0;
    offsets[1] = -Lnom;
    offsets[2] = -2.0*Lnom;
    offsets[3] = -4.0*Lnom;
    offsets[4] = -8.0*Lnom;
    noffsets = thm_tdi_generation == 2 ? 5 : 4;
    for(k=0; k<model->nmodes; k++)
    {
        double f;
        double omega;
        int p;

        /*
         * The AP knots are labelled by the SSB output time returned by
         * barycenter_time(), so retain a phase-step guard at that time.
         */
        f = thm_abs_mode_frequency_hz_at_source_time(
            model, k, source_time, params);
        omega = 2.0*M_PI*f;
        if(isfinite(omega) && omega > max_omega) max_omega = omega;

        /*
         * The TDI response at that output time samples the waveform near the
         * detector-center retarded time, with additional arm delays.  This is
         * a distinct coordinate family and can reach merger hundreds of
         * seconds before the AP-knot time does.
         */
        for(p=0; p<noffsets; p++)
        {
            f = thm_abs_mode_frequency_hz_at_source_time(
                model, k, detector_time+offsets[p], params);
            omega = 2.0*M_PI*f;
            if(isfinite(omega) && omega > max_omega) max_omega = omega;
        }
    }

    return max_omega;
}

static int thm_near_delay_zero(double omega, double omstar)
{
    double omega_abs, omstar_abs, delay_zero_omega;

    omega_abs = fabs(omega);
    omstar_abs = fabs(omstar);
    if(!isfinite(omega_abs) || !isfinite(omstar_abs) ||
       omega_abs <= 0.0 || omstar_abs <= 0.0)
    {
        return 0;
    }

    /*
     * fstar = 1/(2*pi*L), so omstar = 2*pi*fstar = 1/L and omega*L =
     * omega/omstar = f/fstar.  fstar itself is only the one-radian delay
     * phase scale.  The leading equal-arm X1 factor has zeros at
     * omega = n*pi*omstar.  X2 adds the four-arm commutator factor, with
     * additional nominal zeros halfway between these locations.
     */
    delay_zero_omega = (thm_tdi_generation == 2 ? 0.5 : 1.0)*
                       M_PI*omstar_abs;
    if(fabs(remainder(omega_abs, delay_zero_omega))/omega_abs < 5.0e-2) return 1;

    return 0;
}

static int thm_delay_zero_family_interval_reason(double fmin, double fmax, double width, double fmode_max, double spacing)
{
    int nmin, nmax, n;

    if(width <= dTmin || spacing <= 0.0)
    {
        return THM_AP_INTERVAL_OK;
    }

    if(!isfinite(fmin) || !isfinite(fmax) || fmax <= 0.0)
    {
        return THM_AP_INTERVAL_OK;
    }

    if(!isfinite(fmode_max) || fmode_max <= 0.0)
    {
        fmode_max = fmax;
    }

    nmin = (int)floor(fmin/spacing)-1;
    nmax = (int)floor(fmode_max/spacing);
    if(nmin < 1) nmin = 1;
    if(nmax < nmin) return THM_AP_INTERVAL_OK;

    for(n=nmin; n<=nmax; n++)
    {
        double fc = (double)n*spacing;

        if(fmin <= fc && fmax >= fc)
        {
            double fscale = fmax;
            double max_width;

            if(fscale < fc) fscale = fc;
            max_width = THM_AP_TRANSFER_PHASE_STEP/(2.0*M_PI*fscale);
            if(max_width < dTmin) max_width = dTmin;
            if(width > max_width)
            {
                return THM_AP_FAIL_DELAY_ZERO;
            }
        }
    }

    return THM_AP_INTERVAL_OK;
}

static int thm_delay_zero_interval_reason(double f0, double fmid, double f1, double width, double fmode_max)
{
    double fmin, fmax;

    if(width <= dTmin || fstar <= 0.0)
    {
        return THM_AP_INTERVAL_OK;
    }

    f0 = fabs(f0);
    fmid = fabs(fmid);
    f1 = fabs(f1);
    fmin = f0;
    fmax = f0;
    if(fmid < fmin) fmin = fmid;
    if(f1 < fmin) fmin = f1;
    if(fmid > fmax) fmax = fmid;
    if(f1 > fmax) fmax = f1;

    return thm_delay_zero_family_interval_reason(
        fmin, fmax, width, fmode_max,
        (thm_tdi_generation == 2 ? 0.5 : 1.0)*M_PI*fstar);
}

static int thm_frequency_interval_reason(double f0, double fmid, double f1, double width, double fmode_max, int include_delay_zeros)
{
    double ferr;
    double fscale;
    double fend;
    double phase_err;
    int reason;

    f0 = fabs(f0);
    fmid = fabs(fmid);
    f1 = fabs(f1);
    ferr = fabs(fmid-0.5*(f0+f1));
    fscale = fmid;
    if(fscale < f0) fscale = f0;
    if(fscale < f1) fscale = f1;
    if(fscale < 1.0e-8) fscale = 1.0e-8;

    /*
     * The midpoint curvature test alone can miss intervals where the
     * frequency changes almost linearly but very rapidly.  That leaves
     * too few samples for a stable phase/frequency spline through the
     * merger bend.  Keep this criterion dimensionless so it follows the
     * source and mode: each accepted interval must have a small
     * fractional endpoint frequency change for every active track.
    */
    fend = fabs(f1-f0)/fscale;
    if(fend > THM_AP_FREQ_STEP_REL_TOL) return THM_AP_FAIL_FREQ_STEP;

    /*
     * fstar is the one-radian delay-phase scale: omega*L = f/fstar.  There is
     * not generally a TDI node at f=fstar.  The equal-arm delay zeros shared by
     * the response and the pure-instrument PSD are at sin(omega*L)=0, i.e.
     * f ~= n*pi*fstar.  Force extra samples only when an interval crosses that
     * delay-zero family; the smooth variation between zeros is covered by the
     * endpoint and curvature tests above/below.
     */
    if(include_delay_zeros)
    {
        reason = thm_delay_zero_interval_reason(f0, fmid, f1, width, fmode_max);
        if(reason != THM_AP_INTERVAL_OK) return reason;
    }

    phase_err = 0.5*M_PI*ferr*width;
    if(phase_err > THM_AP_PHASE_CURVATURE_TOL) return THM_AP_FAIL_PHASE_CURVATURE;
    if(ferr/fscale > THM_AP_FREQ_REL_CURVATURE_TOL) return THM_AP_FAIL_REL_CURVATURE;

    return THM_AP_INTERVAL_OK;
}

static int thm_detector_interval_reason(const IMRPhenomTHM *model, double t0, double t1, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int include_delay_zeros, int *fail_mode, int *fail_probe)
{
    double width, tmid;
    int k;

    width = t1-t0;
    if(fail_mode != NULL) *fail_mode = -1;
    if(fail_probe != NULL) *fail_probe = -1;
    if(width <= dTmin) return THM_AP_INTERVAL_OK;
    tmid = 0.5*(t0+t1);

    for(k=0; k<model->nmodes; k++)
    {
        double fmode_max = 0.0;
        double f0, f1, fmid;

        {
            double Mtot = params[0]+params[1];

            if(Mtot > 0.0 &&
               model->modes[k].ell == 2 &&
               model->modes[k].abs_emm == 2 &&
               model->mode22.omegaRING > 0.0)
            {
                fmode_max = fabs(model->mode22.omegaRING)/(2.0*M_PI*Mtot);
            }
            else if(Mtot > 0.0 && model->modes[k].phase.omegaRING > 0.0)
            {
                fmode_max = fabs(model->modes[k].phase.omegaRING)/(2.0*M_PI*Mtot);
            }
        }

        /*
         * This is a derivative-free curvature test.  The midpoint deviation of
         * f(t) from the linear interpolation of the endpoints is proportional to
         * d^2 f/dt^2 times dt^2.  Multiplying that frequency deviation by dt
         * gives a local phase-error scale.
         *
         * There are two separate time-coordinate requirements.  The returned
         * AP spline knots are labelled by the SSB output time obtained from
         * barycenter_time(), so the first probe protects interpolation of the
         * intrinsic arrays on that knot grid.  The TDI formula, however, samples
         * h at t-k.r-Delta L.  The second probe family below therefore follows
         * the detector-center retarded time and its nominal arm delays.  These
         * merger neighborhoods differ by up to one AU light-travel time and
         * must not be conflated.
         */
        f0 = thm_abs_mode_frequency_hz_at_detector_time(model, k, t0, params, SPacc, SPspline);
        f1 = thm_abs_mode_frequency_hz_at_detector_time(model, k, t1, params, SPacc, SPspline);
        fmid = thm_abs_mode_frequency_hz_at_detector_time(model, k, tmid, params, SPacc, SPspline);
        {
            int reason = thm_frequency_interval_reason(f0, fmid, f1, width, fmode_max, include_delay_zeros);
            if(reason != THM_AP_INTERVAL_OK)
            {
                if(fail_mode != NULL) *fail_mode = k;
                if(fail_probe != NULL) *fail_probe = 0;
                return reason;
            }
        }

        {
            double Lnom = 1.0/(2.0*M_PI*fstar);
            double offsets[5];
            int noffsets;
            int p;

            offsets[0] = 0.0;
            offsets[1] = -Lnom;
            offsets[2] = -2.0*Lnom;
            offsets[3] = -4.0*Lnom;
            offsets[4] = -8.0*Lnom;
            noffsets = thm_tdi_generation == 2 ? 5 : 4;

            for(p=0; p<noffsets; p++)
            {
                f0 = thm_abs_mode_frequency_hz_at_source_time(
                    model, k, t0+offsets[p], params);
                f1 = thm_abs_mode_frequency_hz_at_source_time(
                    model, k, t1+offsets[p], params);
                fmid = thm_abs_mode_frequency_hz_at_source_time(
                    model, k, tmid+offsets[p], params);
                {
                    int reason = thm_frequency_interval_reason(f0, fmid, f1, width, fmode_max, include_delay_zeros);
                    if(reason != THM_AP_INTERVAL_OK)
                    {
                        if(fail_mode != NULL) *fail_mode = k;
                        if(fail_probe != NULL) *fail_probe = p+1;
                        return reason;
                    }
                }
            }
        }
    }

    return THM_AP_INTERVAL_OK;
}

static int thm_detector_interval_ok(const IMRPhenomTHM *model, double t0, double t1, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int include_delay_zeros)
{
    return thm_detector_interval_reason(model, t0, t1, params, SPacc, SPspline, include_delay_zeros, NULL, NULL) == THM_AP_INTERVAL_OK;
}

static double thm_detector_step_proposal(double last_step,
                                         double previous_step,
                                         int history_count)
{
    double ratio;

    if(!isfinite(last_step) || last_step <= 0.0) return THM_DTM_MAX;

    /*
     * Follow a decreasing detector-grid trend into merger instead of first
     * trying to grow every interval by 20 percent.  With the old geometric
     * rejection, one failed trial only tightened the previous spacing by
     * 1.2*0.8 = 0.96 and an occasional second rejection caused an abrupt
     * 20-percent knot displacement.  Growing regions retain the 1.2 cap.
     */
    if(history_count >= 2 && isfinite(previous_step) && previous_step > 0.0 &&
       last_step < previous_step)
    {
        ratio = last_step/previous_step;
        if(ratio < 1.0/THM_AP_STEP_GROWTH_LIMIT)
        {
            ratio = 1.0/THM_AP_STEP_GROWTH_LIMIT;
        }
        if(ratio > 1.0) ratio = 1.0;
        return last_step*ratio;
    }

    return THM_AP_STEP_GROWTH_LIMIT*last_step;
}

static void thm_record_detector_step_failure(THMAPStepDiagnostic *diag,
                                             int reason,
                                             int fail_mode,
                                             int fail_probe)
{
    if(diag == NULL) return;

    diag->shrink_count++;
    diag->last_fail_reason = reason;
    diag->last_fail_mode = fail_mode;
    diag->last_fail_probe = fail_probe;
    if(reason == THM_AP_FAIL_FREQ_STEP) diag->fail_freq_step++;
    else if(reason == THM_AP_FAIL_DELAY_ZERO) diag->fail_delay_zero++;
    else if(reason == THM_AP_FAIL_PHASE_CURVATURE) diag->fail_phase_curvature++;
    else if(reason == THM_AP_FAIL_REL_CURVATURE) diag->fail_rel_curvature++;
}

static double thm_detector_adaptive_step(const IMRPhenomTHM *model, double t, double tstop, double proposed_step, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int include_delay_zeros, THMAPStepDiagnostic *diag)
{
    double max_omega, delT;

    if(diag != NULL)
    {
        memset(diag, 0, sizeof(*diag));
        diag->proposed_step = proposed_step;
        diag->last_fail_reason = THM_AP_INTERVAL_OK;
        diag->last_fail_mode = -1;
        diag->last_fail_probe = -1;
    }

    delT = proposed_step;
    if(!isfinite(delT) || delT <= 0.0) delT = THM_DTM_MAX;
    if(delT > THM_DTM_MAX)
    {
        delT = THM_DTM_MAX;
        if(diag != NULL) diag->dtmax_cap = 1;
    }
    max_omega = thm_max_omega_detector(model, t, params, SPacc, SPspline);
    if(diag != NULL) diag->max_omega = max_omega;
    if(max_omega > 0.0 && delT > THM_AP_MAX_PHASE_STEP/max_omega)
    {
        delT = THM_AP_MAX_PHASE_STEP/max_omega;
        if(diag != NULL) diag->phase_step_cap = 1;
    }
    if(delT < dTmin) delT = dTmin;
    if(t+delT > tstop)
    {
        delT = tstop-t;
        if(diag != NULL) diag->tstop_cap = 1;
    }
    if(delT < dTmin) delT = dTmin;
    if(diag != NULL) diag->capped_step = delT;

    if(delT > dTmin && t+delT <= tstop &&
       !thm_detector_interval_ok(model, t, t+delT, params, SPacc, SPspline,
                                 include_delay_zeros))
    {
        double failed_step = delT;
        double passing_step;
        int fail_mode;
        int fail_probe;
        int reason;
        int q;

        /* Find a passing lower bracket using the old robust contraction. */
        do
        {
            fail_mode = -1;
            fail_probe = -1;
            reason = thm_detector_interval_reason(
                model, t, t+failed_step, params, SPacc, SPspline,
                include_delay_zeros, &fail_mode, &fail_probe);
            thm_record_detector_step_failure(diag, reason, fail_mode,
                                             fail_probe);
            passing_step = failed_step*THM_AP_STEP_SHRINK_FACTOR;
            if(passing_step < dTmin) passing_step = dTmin;
            if(thm_detector_interval_ok(model, t, t+passing_step, params,
                                        SPacc, SPspline,
                                        include_delay_zeros))
            {
                break;
            }
            failed_step = passing_step;
        } while(passing_step > dTmin);

        /*
         * Refine the admissible-step boundary rather than retaining one of the
         * discrete 0.8^n values.  The lower endpoint is explicitly tested and
         * passing, so no separate safety offset is needed.
         */
        for(q=0; q<THM_AP_STEP_BRACKET_ITERATIONS &&
                 failed_step-passing_step > 1.0e-6*fmax(1.0, passing_step);
            q++)
        {
            double trial_step = 0.5*(passing_step+failed_step);

            if(thm_detector_interval_ok(model, t, t+trial_step, params,
                                        SPacc, SPspline,
                                        include_delay_zeros))
            {
                passing_step = trial_step;
            }
            else
            {
                fail_mode = -1;
                fail_probe = -1;
                reason = thm_detector_interval_reason(
                    model, t, t+trial_step, params, SPacc, SPspline,
                    include_delay_zeros, &fail_mode, &fail_probe);
                thm_record_detector_step_failure(diag, reason, fail_mode,
                                                 fail_probe);
                failed_step = trial_step;
            }
        }
        delT = passing_step;
    }

    /*
     * Optional convergence diagnostic for the rapidly varying TDI response.
     * Here t is the detector-planning coordinate: the prompt TDI term samples
     * the intrinsic waveform near t, while the remaining terms extend back by
     * up to roughly 4L.  Land exactly on the capped region and then enforce a
     * uniform upper bound there.  The cap may be below the production dTmin;
     * this is intentional for the 0.5 s convergence test.
     */
    if(thm_merger_grid_max_step_seconds > 0.0 &&
       thm_merger_grid_half_width_seconds > 0.0)
    {
        double cap_start = params[5]-thm_merger_grid_half_width_seconds;
        double cap_stop = params[5]+thm_merger_grid_half_width_seconds;

        if(t < cap_start && t+delT > cap_start)
        {
            delT = cap_start-t;
        }
        else if(t >= cap_start && t < cap_stop)
        {
            if(delT > thm_merger_grid_max_step_seconds)
            {
                delT = thm_merger_grid_max_step_seconds;
            }
            if(t+delT > cap_stop) delT = cap_stop-t;
        }
    }

    if(diag != NULL) diag->accepted_step = delT;

    return delT;
}

int PhenomTHM_AP_DetectorAdaptive(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax, double observation_tmin, double observation_tmax)
{
    double m1, m2, chi1, chi2, Mtot, eta, tc, distance;
    double t, tnext, tstart, tstop, post_merger_stop, source_tmin, source_tmax;
    double constellation_margin;
    double last_step, previous_step;
    double strain_scale;
    double *tau, *phi22;
    FILE *spacing_diag;
    int status;
    int i, k, Ns;

    if(params == NULL || model == NULL || modes == NULL || nmodes == NULL ||
       TS == NULL || tspace == NULL || mode_amp == NULL || mode_phase == NULL ||
       mode_freq == NULL)
    {
        return 0;
    }

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    tc = params[5];
    distance = exp(params[6]);
    Mtot = m1 + m2;
    eta = (m1*m2)/(Mtot*Mtot);

    if(requested_modes != NULL && requested_nmodes > 0)
    {
        if(requested_nmodes > IMRPHENOMTHM_MAX_MODES)
        {
            fprintf(stderr, "Error: requested %d THM modes, but only %d are supported by this driver.\n",
                    requested_nmodes, IMRPHENOMTHM_MAX_MODES);
            exit(1);
        }
        *nmodes = requested_nmodes;
        for(i=0; i<*nmodes; i++) modes[i] = requested_modes[i];
    }
    else
    {
        *nmodes = IMRPhenomTHMDefaultModes(modes, IMRPHENOMTHM_MAX_MODES);
    }
    status = IMRPhenomTHMInitialize(model, m1, m2, chi1, chi2, modes, *nmodes);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMInitialize failed in detector AP grid: %d\n", status);
        exit(1);
    }

    /*
     * New detector-time grid: march forward only over the portion of the
     * observation that the response can touch.  The constellation/orbit splines
     * are built on their own slow grid and can be evaluated at these detector
     * times; the one-AU margin only keeps the two-pass detector-to-barycenter
     * mapping inside the constellation spline domain.  It is not a waveform
     * spacing guard.
     */
    constellation_margin = CONSTELLATION_LIGHT_TIME_SECONDS;
    tstart = observation_tmin;
    if(tstart < constellation_tmin+constellation_margin)
    {
        tstart = constellation_tmin+constellation_margin;
    }
    if(tstart < 0.0) tstart = 0.0;

    post_merger_stop = tc+THM_RESPONSE_LATE_MARGIN_SECONDS+1000.0*Mtot;
    tstop = observation_tmax;
    if(post_merger_stop < tstop) tstop = post_merger_stop;
    if(tstop > constellation_tmax-constellation_margin)
    {
        tstop = constellation_tmax-constellation_margin;
    }
    if(tstop <= tstart)
    {
        fprintf(stderr, "Error: detector AP grid has empty support [%.15e, %.15e].\n", tstart, tstop);
        exit(1);
    }

    Ns = 0;
    t = tstart;
    last_step = THM_DTM_MAX;
    previous_step = THM_DTM_MAX;
    spacing_diag = NULL;
#if THM_AP_WRITE_SPACING_DIAGNOSTIC
    if(thm_diagnostics_enabled)
    {
        spacing_diag = fopen(thm_ap_spacing_diagnostic_filename, "w");
        if(spacing_diag != NULL)
        {
            fprintf(spacing_diag, "# Adaptive AP spacing diagnostic for PhenomTHM_AP_DetectorAdaptive\n");
            fprintf(spacing_diag, "# probe labels: 0 detector-guiding-center barycenter map, 1 source t, 2 source t-L, 3 source t-2L, 4 source t-4L with L=1/(2*pi*fstar)\n");
            fprintf(spacing_diag, "# equal-arm delay-zero response-grid guard %s\n",
                    thm_intrinsic_tdi_include_delay_zero_grid ? "enabled" : "disabled");
            fprintf(spacing_diag, "# reason labels: 0 ok, 1 endpoint_fractional_frequency_step, 2 equal_arm_delay_zero_crossing, 3 phase_curvature, 4 relative_frequency_curvature\n");
            fprintf(spacing_diag, "# fstar is the delay phase scale 1/(2*pi*L); equal-arm delay zeros are f = n*pi*fstar\n");
            fprintf(spacing_diag, "# columns: interval_index center_source_time_u_s ssb_output_time_s accepted_du_s proposed_du_s capped_du_s max_freq_hz dtmax_cap phase_step_cap tstop_cap shrink_count fail_freq_step fail_delay_zero fail_phase_curvature fail_rel_curvature last_reason last_mode last_probe\n");
        }
    }
#endif
    while(1)
    {
        THMAPStepDiagnostic step_diag;
        double source_t;
        double delT;

        if(Ns >= Nsmax)
        {
            fprintf(stderr, "Error: PhenomTHM_AP_DetectorAdaptive exceeded Nsmax while building the detector-time grid.\n");
            if(spacing_diag != NULL) fclose(spacing_diag);
            exit(1);
        }
        tspace[Ns] = t;
        Ns++;
        if(t >= tstop) break;

        delT = thm_detector_adaptive_step(model, t, tstop,
                                          thm_detector_step_proposal(
                                              last_step, previous_step, Ns-1),
                                          params, SPacc, SPspline,
                                          thm_intrinsic_tdi_include_delay_zero_grid,
                                          &step_diag);
        source_t = thm_ssb_output_time_from_center_source_time(t, params, SPacc, SPspline);
        if(spacing_diag != NULL)
        {
            fprintf(spacing_diag, "%d %.15e %.15e %.15e %.15e %.15e %.15e %d %d %d %d %d %d %d %d %d %d %d\n",
                    Ns-1, t, source_t, step_diag.accepted_step,
                    step_diag.proposed_step, step_diag.capped_step,
                    step_diag.max_omega/(2.0*M_PI),
                    step_diag.dtmax_cap, step_diag.phase_step_cap,
                    step_diag.tstop_cap, step_diag.shrink_count,
                    step_diag.fail_freq_step, step_diag.fail_delay_zero,
                    step_diag.fail_phase_curvature, step_diag.fail_rel_curvature,
                    step_diag.last_fail_reason, step_diag.last_fail_mode,
                    step_diag.last_fail_probe);
        }

        tnext = t+delT;
        if(tnext <= t) tnext = t+dTmin;
        if(tnext > tstop) tnext = tstop;
        previous_step = last_step;
        last_step = tnext-t;
        t = tnext;
    }
    if(spacing_diag != NULL) fclose(spacing_diag);

    /*
     * Keep the waveform/source-time coordinate separate from the detector-time
     * sampling coordinate.  For this first comparison routine TS is the
     * barycenter/source time used by the mode AP splines; tspace is the
     * detector-time grid that was actually adapted.
     */
    barycenter_time(TS, tspace, params, SPacc, SPspline, Ns);
    tau = double_vector(Nsmax);
    phi22 = double_vector(Nsmax);
    for(i=0; i<Ns; i++) tau[i] = (TS[i]-tc)/Mtot;

    source_tmin = tau[0];
    source_tmax = tau[Ns-1];
    if(source_tmax <= source_tmin)
    {
        fprintf(stderr, "Error: detector AP grid produced non-increasing source times.\n");
        exit(1);
    }

    status = IMRPhenomTHMBuildPhi22Grid(model, Ns, tau,
                                        IMRPhenomTPhase22(source_tmin, eta, &model->mode22),
                                        phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMBuildPhi22Grid failed in detector AP grid: %d\n", status);
        exit(1);
    }
    {
        double shift = params[4]-IMRPhenomTPhase22(0.0, eta, &model->mode22);
        for(i=0; i<Ns; i++) phi22[i] += shift;
    }
    status = IMRPhenomTHMSetPhaseOffsetsFromGrid(model, Ns, tau, phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMSetPhaseOffsetsFromGrid failed in detector AP grid: %d\n", status);
        exit(1);
    }

    strain_scale = sqrt(2.0)*eta*Mtot/(distance*GPSEC);
    for(i=0; i<Ns; i++)
    {
        for(k=0; k<*nmodes; k++)
        {
            IMRPhenomTHMModeSample sample;
            status = IMRPhenomTHMEvaluateMode(model, k, tau[i], phi22[i], &sample);
            if(status != 0)
            {
                fprintf(stderr, "IMRPhenomTHMEvaluateMode failed in detector AP grid at mode %d sample %d: %d\n", k, i, status);
                exit(1);
            }
            mode_amp[k][i] = strain_scale*sample.amplitude;
            mode_phase[k][i] = sample.phase;
            mode_freq[k][i] = sample.omega/(2.0*M_PI*Mtot);
        }
    }

    free_double_vector(tau);
    free_double_vector(phi22);

    return Ns;
}

int PhenomTHM_AP_IntrinsicAdaptive(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, double observation_tmin, double observation_tmax)
{
    double m1, m2, chi1, chi2, Mtot, eta, tc, distance;
    double t, tnext, tstart, tstop, post_merger_stop, source_tmin, source_tmax;
    double last_step;
    double strain_scale;
    double *tau, *phi22;
    FILE *spacing_diag;
    int status;
    int i, k, Ns;

    if(params == NULL || model == NULL || modes == NULL || nmodes == NULL ||
       TS == NULL || tspace == NULL || mode_amp == NULL || mode_phase == NULL ||
       mode_freq == NULL)
    {
        return 0;
    }

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    tc = params[5];
    distance = exp(params[6]);
    Mtot = m1 + m2;
    eta = (m1*m2)/(Mtot*Mtot);

    if(requested_modes != NULL && requested_nmodes > 0)
    {
        if(requested_nmodes > IMRPHENOMTHM_MAX_MODES)
        {
            fprintf(stderr, "Error: requested %d THM modes, but only %d are supported by this driver.\n",
                    requested_nmodes, IMRPHENOMTHM_MAX_MODES);
            exit(1);
        }
        *nmodes = requested_nmodes;
        for(i=0; i<*nmodes; i++) modes[i] = requested_modes[i];
    }
    else
    {
        *nmodes = IMRPhenomTHMDefaultModes(modes, IMRPHENOMTHM_MAX_MODES);
    }
    status = IMRPhenomTHMInitialize(model, m1, m2, chi1, chi2, modes, *nmodes);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMInitialize failed in intrinsic AP grid: %d\n", status);
        exit(1);
    }

    /*
     * This source-time grid is intentionally independent of the LISA orbit and
     * the TDI transfer-function zeros.  It is a first diagnostic for a future
     * two-stage construction: evaluate the expensive intrinsic THM modes on this
     * sparse grid, then spline A(t), phi(t), and f(t) onto the detector/TDI grid.
     */
    tstart = observation_tmin;
    if(tstart < 0.0) tstart = 0.0;
    post_merger_stop = tc+THM_RESPONSE_LATE_MARGIN_SECONDS+1000.0*Mtot;
    tstop = observation_tmax;
    if(post_merger_stop < tstop) tstop = post_merger_stop;
    if(tstop <= tstart)
    {
        fprintf(stderr, "Error: intrinsic AP grid has empty support [%.15e, %.15e].\n",
                tstart, tstop);
        exit(1);
    }

    Ns = 0;
    t = tstart;
    last_step = THM_INTRINSIC_DTM_MAX;
    spacing_diag = fopen(thm_intrinsic_ap_spacing_diagnostic_filename, "w");
    if(spacing_diag != NULL)
    {
        fprintf(spacing_diag, "# Intrinsic AP spacing diagnostic for PhenomTHM_AP_IntrinsicAdaptive\n");
        fprintf(spacing_diag, "# This grid is source/barycenter time only; it deliberately omits TDI delay-zero guards.\n");
        fprintf(spacing_diag, "# reason labels: 0 ok, 1 endpoint_fractional_frequency_step, 3 phase_curvature, 4 relative_frequency_curvature, 5 endpoint_fractional_amplitude_step, 6 relative_amplitude_curvature\n");
        fprintf(spacing_diag, "# columns: interval_index source_time_s tau accepted_dt_s proposed_dt_s capped_dt_s max_freq_hz dtmax_cap phase_step_cap tstop_cap shrink_count fail_freq_step fail_phase_curvature fail_rel_curvature fail_amp_step fail_amp_curvature last_reason last_mode\n");
    }

    while(1)
    {
        THMAPStepDiagnostic step_diag;
        double delT;

        if(Ns >= Nsmax)
        {
            fprintf(stderr, "Error: PhenomTHM_AP_IntrinsicAdaptive exceeded Nsmax while building the source-time grid.\n");
            if(spacing_diag != NULL) fclose(spacing_diag);
            exit(1);
        }
        TS[Ns] = t;
        tspace[Ns] = t;
        Ns++;
        if(t >= tstop) break;

        delT = thm_intrinsic_adaptive_step(model, t, tstop,
                                           THM_INTRINSIC_STEP_GROWTH_LIMIT*last_step,
                                           params, 0, &step_diag);
        if(spacing_diag != NULL)
        {
            fprintf(spacing_diag, "%d %.15e %.15e %.15e %.15e %.15e %.15e %d %d %d %d %d %d %d %d %d %d %d\n",
                    Ns-1, t, (t-tc)/Mtot, step_diag.accepted_step,
                    step_diag.proposed_step, step_diag.capped_step,
                    step_diag.max_omega/(2.0*M_PI),
                    step_diag.dtmax_cap, step_diag.phase_step_cap,
                    step_diag.tstop_cap, step_diag.shrink_count,
                    step_diag.fail_freq_step, step_diag.fail_phase_curvature,
                    step_diag.fail_rel_curvature, step_diag.fail_amp_step,
                    step_diag.fail_amp_curvature,
                    step_diag.last_fail_reason, step_diag.last_fail_mode);
        }

        tnext = t+delT;
        if(tnext <= t) tnext = t+dTmin;
        if(tnext > tstop) tnext = tstop;
        last_step = tnext-t;
        t = tnext;
    }
    if(spacing_diag != NULL) fclose(spacing_diag);

    tau = double_vector(Ns);
    phi22 = double_vector(Ns);
    if(tau == NULL || phi22 == NULL)
    {
        fprintf(stderr, "allocation failure in PhenomTHM_AP_IntrinsicAdaptive\n");
        exit(1);
    }
    for(i=0; i<Ns; i++) tau[i] = (TS[i]-tc)/Mtot;

    source_tmin = tau[0];
    source_tmax = tau[Ns-1];
    if(source_tmax <= source_tmin)
    {
        fprintf(stderr, "Error: intrinsic AP grid produced non-increasing source times.\n");
        exit(1);
    }

    status = IMRPhenomTHMBuildPhi22Grid(model, Ns, tau,
                                        IMRPhenomTPhase22(source_tmin, eta, &model->mode22),
                                        phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMBuildPhi22Grid failed in intrinsic AP grid: %d\n",
                status);
        exit(1);
    }
    {
        double shift = params[4]-IMRPhenomTPhase22(0.0, eta, &model->mode22);
        for(i=0; i<Ns; i++) phi22[i] += shift;
    }
    status = IMRPhenomTHMSetPhaseOffsetsFromGrid(model, Ns, tau, phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMSetPhaseOffsetsFromGrid failed in intrinsic AP grid: %d\n",
                status);
        exit(1);
    }

    strain_scale = sqrt(2.0)*eta*Mtot/(distance*GPSEC);
    for(i=0; i<Ns; i++)
    {
        for(k=0; k<*nmodes; k++)
        {
            IMRPhenomTHMModeSample sample;
            status = IMRPhenomTHMEvaluateMode(model, k, tau[i], phi22[i], &sample);
            if(status != 0)
            {
                fprintf(stderr, "IMRPhenomTHMEvaluateMode failed in intrinsic AP grid at mode %d sample %d: %d\n",
                        k, i, status);
                exit(1);
            }
            mode_amp[k][i] = strain_scale*sample.amplitude;
            mode_phase[k][i] = sample.phase;
            mode_freq[k][i] = sample.omega/(2.0*M_PI*Mtot);
        }
    }

    free_double_vector(tau);
    free_double_vector(phi22);

    return Ns;
}

static int thm_append_source_model_time(double *times, int *N, int Nmax, double t)
{
    if(times == NULL || N == NULL || Nmax < 1 || !isfinite(t))
    {
        return 0;
    }
    if(*N >= Nmax)
    {
        return 0;
    }
    times[*N] = t;
    (*N)++;
    return 1;
}

static int thm_response_sample_should_be_exact(const double *tspace, int Ns, int i)
{
    double left_dt, right_dt;
    double eps;

    if(tspace == NULL || Ns < 1 || i < 0 || i >= Ns)
    {
        return 0;
    }
    if(i == 0 || i == Ns-1)
    {
        return 1;
    }

    eps = 1.0e-10*THM_DTM_MAX;
    left_dt = tspace[i]-tspace[i-1];
    right_dt = tspace[i+1]-tspace[i];
    if(left_dt < THM_DTM_MAX-eps || right_dt < THM_DTM_MAX-eps)
    {
        return 1;
    }

    return 0;
}

static int thm_find_source_model_time(const double *times, int N, double t)
{
    int lo, hi;
    double tol;

    if(times == NULL || N < 1 || !isfinite(t))
    {
        return -1;
    }

    tol = 1.0e-7 + 1.0e-13*fabs(t);
    lo = 0;
    hi = N-1;
    while(lo <= hi)
    {
        int mid = lo + (hi-lo)/2;
        double diff = times[mid]-t;
        if(fabs(diff) <= tol)
        {
            return mid;
        }
        if(diff < 0.0)
        {
            lo = mid+1;
        }
        else
        {
            hi = mid-1;
        }
    }

    if(lo < N && fabs(times[lo]-t) <= tol) return lo;
    if(hi >= 0 && fabs(times[hi]-t) <= tol) return hi;

    return -1;
}

static double thm_max_mode_frequency_hz_at_source_time(const IMRPhenomTHM *model,
                                                       double source_time,
                                                       double *params)
{
    double omega = thm_max_omega_source(model, source_time, params);

    if(!isfinite(omega) || omega <= 0.0)
    {
        return 0.0;
    }

    return omega/(2.0*M_PI);
}

static double thm_intrinsic_tdi_switch_source_time(const IMRPhenomTHM *model,
                                                  double *params,
                                                  const double *TS_model,
                                                  int Ns_model,
                                                  double source_grid_start,
                                                  double source_grid_stop,
                                                  int include_delay_zero_trigger)
{
    double f_trigger;
    double switch_time, tail_switch;
    double fprev;
    int i;

    if(model == NULL || params == NULL || TS_model == NULL || Ns_model < 2)
    {
        return source_grid_start;
    }

    /*
     * The production grid has two jobs that should not be conflated:
     *
     *   1. sample the slowly varying LISA response at roughly THM_DTM_MAX;
     *   2. add genuinely dense samples only when the waveform/TDI structure
     *      needs them.
     *
     * A full detector-aware interval test everywhere is safe but expensive,
     * because it probes every active mode at every 10 ks interval over a
     * one-year observation.  Instead, locate the onset of the difficult region
     * on the sparse intrinsic source-time grid, then switch the response grid
     * to the full detector-aware planner only for the tail.
     */
    f_trigger = include_delay_zero_trigger ?
                THM_INTRINSIC_TDI_SWITCH_TRANSFER_FRACTION*
                (thm_tdi_generation == 2 ? 0.5 : 1.0)*M_PI*fstar : 0.0;
    switch_time = HUGE_VAL;

    fprev = thm_max_mode_frequency_hz_at_source_time(model, TS_model[0], params);
    for(i=1; i<Ns_model; i++)
    {
        double fnow;
        double dt_model;

        dt_model = TS_model[i]-TS_model[i-1];
        if(dt_model > 0.0 && dt_model < THM_DTM_MAX)
        {
            switch_time = TS_model[i-1];
            break;
        }

        fnow = thm_max_mode_frequency_hz_at_source_time(model, TS_model[i],
                                                        params);
        if(f_trigger > 0.0 && isfinite(fprev) && isfinite(fnow) &&
           ((fprev < f_trigger && fnow >= f_trigger) ||
            (fprev >= f_trigger && fnow > 0.0)))
        {
            if(fnow != fprev)
            {
                double u = (f_trigger-fprev)/(fnow-fprev);
                if(u < 0.0) u = 0.0;
                if(u > 1.0) u = 1.0;
                switch_time = TS_model[i-1]+u*(TS_model[i]-TS_model[i-1]);
            }
            else
            {
                switch_time = TS_model[i-1];
            }
            break;
        }
        fprev = fnow;
    }

    /*
     * If the system never reaches the transfer-zero trigger, still give the
     * merger/ringdown tail to the detector-aware planner.  That keeps low-mass
     * or low-frequency examples from relying on a sparse intrinsic interpolation
     * exactly where the endpoint FFT and final response are most sensitive.
     */
    tail_switch = source_grid_stop-THM_INTRINSIC_TDI_SWITCH_TAIL_SECONDS;
    if(!isfinite(switch_time) || tail_switch < switch_time)
    {
        switch_time = tail_switch;
    }

    if(!isfinite(switch_time)) switch_time = source_grid_start;
    if(switch_time < source_grid_start) switch_time = source_grid_start;
    if(switch_time > source_grid_stop) switch_time = source_grid_stop;

    return switch_time;
}

int PhenomTHM_AP_IntrinsicTDIAdaptive(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Nsmax, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline, double constellation_tmin, double constellation_tmax, double observation_tmin, double observation_tmax)
{
    double m1, m2, chi1, chi2, Mtot, eta, tc, distance;
    double t, tnext, tstart, tstop, post_merger_stop;
    double source_grid_start, source_grid_stop, source_tmin, source_tmax;
    double response_switch_source_time, response_switch_detector_time;
    double constellation_margin;
    double last_response_step, previous_response_step, last_model_step;
    double strain_scale;
    double *TS_model, *tau_model, *phi22;
    double **amp_model, **phase_model, **freq_model;
    int *source_index;
    FILE *spacing_diag;
    FILE *response_diag;
    char response_diag_path[640];
    int model_capacity;
    int status;
    int i, k, Ns_model, Ns_response;
    int detector_planned_samples, detector_step_history;

    if(params == NULL || model == NULL || modes == NULL || nmodes == NULL ||
       TS == NULL || tspace == NULL || mode_amp == NULL || mode_phase == NULL ||
       mode_freq == NULL || SPacc == NULL || SPspline == NULL)
    {
        return 0;
    }

    thm_last_intrinsic_tdi_model_samples = 0;
    thm_last_intrinsic_tdi_interpolated_samples = 0;
    thm_last_intrinsic_tdi_exact_samples = 0;
    thm_last_intrinsic_tdi_detector_planned_samples = 0;
    thm_last_intrinsic_tdi_switch_source_time = 0.0;
    thm_last_intrinsic_tdi_switch_detector_time = 0.0;

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    tc = params[5];
    distance = exp(params[6]);
    Mtot = m1 + m2;
    eta = (m1*m2)/(Mtot*Mtot);

    if(requested_modes != NULL && requested_nmodes > 0)
    {
        if(requested_nmodes > IMRPHENOMTHM_MAX_MODES)
        {
            fprintf(stderr, "Error: requested %d THM modes, but only %d are supported by this driver.\n",
                    requested_nmodes, IMRPHENOMTHM_MAX_MODES);
            exit(1);
        }
        *nmodes = requested_nmodes;
        for(i=0; i<*nmodes; i++) modes[i] = requested_modes[i];
    }
    else
    {
        *nmodes = IMRPhenomTHMDefaultModes(modes, IMRPHENOMTHM_MAX_MODES);
    }
    status = IMRPhenomTHMInitialize(model, m1, m2, chi1, chi2, modes, *nmodes);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMInitialize failed in intrinsic/TDI AP grid: %d\n",
                status);
        exit(1);
    }

    /*
     * Hybrid production grid:
     *
     *   - The expensive IMRPhenomTHM calls are first planned on a sparse
     *     source-time intrinsic grid.
     *   - The public response grid is kept at the slow TDI/orbit cadence while
     *     that intrinsic grid is smooth, and the AP values are interpolated onto
     *     those 10 ks response samples.
     *   - Once the intrinsic grid itself wants sub-10 ks spacing, or the highest
     *     active mode approaches the first equal-arm transfer-zero family, the
     *     response grid switches to the full detector-aware planner.  Those dense
     *     tail response samples are inserted into the source grid and copied
     *     exactly instead of splined.
     *
     * This keeps the WDM planning safe near merger without paying for the full
     * detector-aware waveform probes over the quiet inspiral.
     */
    constellation_margin = CONSTELLATION_LIGHT_TIME_SECONDS;
    tstart = observation_tmin;
    if(tstart < constellation_tmin+constellation_margin)
    {
        tstart = constellation_tmin+constellation_margin;
    }
    if(tstart < 0.0) tstart = 0.0;

    post_merger_stop = tc+THM_RESPONSE_LATE_MARGIN_SECONDS+1000.0*Mtot;
    tstop = observation_tmax;
    if(post_merger_stop < tstop) tstop = post_merger_stop;
    if(tstop > constellation_tmax-constellation_margin)
    {
        tstop = constellation_tmax-constellation_margin;
    }
    if(tstop <= tstart)
    {
        fprintf(stderr, "Error: intrinsic/TDI AP grid has empty detector-time support [%.15e, %.15e].\n",
                tstart, tstop);
        exit(1);
    }

    source_grid_start = thm_ssb_output_time_from_center_source_time(tstart, params,
                                                               SPacc, SPspline);
    source_grid_stop = thm_ssb_output_time_from_center_source_time(tstop, params,
                                                              SPacc, SPspline);
    if(source_grid_stop <= source_grid_start)
    {
        fprintf(stderr, "Error: intrinsic/TDI AP grid produced non-increasing source-time support.\n");
        exit(1);
    }

    model_capacity = 2*Nsmax+4;
    TS_model = double_vector(model_capacity);
    tau_model = double_vector(model_capacity);
    phi22 = double_vector(model_capacity);
    amp_model = double_matrix(IMRPHENOMTHM_MAX_MODES, model_capacity);
    phase_model = double_matrix(IMRPHENOMTHM_MAX_MODES, model_capacity);
    freq_model = double_matrix(IMRPHENOMTHM_MAX_MODES, model_capacity);
    source_index = int_vector(Nsmax);
    if(TS_model == NULL || tau_model == NULL || phi22 == NULL ||
       amp_model == NULL || phase_model == NULL || freq_model == NULL ||
       source_index == NULL)
    {
        fprintf(stderr, "allocation failure in PhenomTHM_AP_IntrinsicTDIAdaptive\n");
        exit(1);
    }

    Ns_model = 0;
    t = source_grid_start;
    last_model_step = THM_INTRINSIC_DTM_MAX;
    spacing_diag = NULL;
#if THM_AP_WRITE_SPACING_DIAGNOSTIC
    if(thm_diagnostics_enabled)
    {
        spacing_diag = fopen(thm_intrinsic_tdi_ap_spacing_diagnostic_filename, "w");
        if(spacing_diag != NULL)
        {
            fprintf(spacing_diag, "# Intrinsic/TDI AP spacing diagnostic for PhenomTHM_AP_IntrinsicTDIAdaptive\n");
            fprintf(spacing_diag, "# The expensive THM seed grid is source time and tracks smooth intrinsic AP structure.\n");
            fprintf(spacing_diag, "# The response grid later uses 10 ks detector-time cadence before the source-aware switch, then the full detector/TDI delay-zero planner in the tail.\n");
            fprintf(spacing_diag, "# Response samples from detector intervals denser than THM_DTM_MAX are inserted into the model grid and copied exactly.\n");
            fprintf(spacing_diag, "# equal-arm delay-zero response-grid guard %s\n",
                    thm_intrinsic_tdi_include_delay_zero_grid ? "enabled" : "disabled");
            fprintf(spacing_diag, "# reason labels: 0 ok, 1 endpoint_fractional_frequency_step, 2 equal_arm_delay_zero_crossing, 3 phase_curvature, 4 relative_frequency_curvature, 5 endpoint_fractional_amplitude_step, 6 relative_amplitude_curvature\n");
            fprintf(spacing_diag, "# fstar is the delay phase scale 1/(2*pi*L); equal-arm delay zeros are f = n*pi*fstar\n");
            fprintf(spacing_diag, "# columns: seed_interval_index source_time_s tau accepted_dt_s proposed_dt_s capped_dt_s max_freq_hz dtmax_cap phase_step_cap tstop_cap shrink_count fail_freq_step fail_delay_zero fail_phase_curvature fail_rel_curvature fail_amp_step fail_amp_curvature last_reason last_mode\n");
        }
    }
#endif

    while(1)
    {
        THMAPStepDiagnostic step_diag;
        double delT;

        if(Ns_model >= model_capacity)
        {
            fprintf(stderr, "Error: PhenomTHM_AP_IntrinsicTDIAdaptive exceeded Nsmax while building the source-time model grid.\n");
            if(spacing_diag != NULL) fclose(spacing_diag);
            exit(1);
        }
        if(!thm_append_source_model_time(TS_model, &Ns_model,
                                         model_capacity, t))
        {
            fprintf(stderr, "Error: failed to append intrinsic/TDI source-time model sample.\n");
            if(spacing_diag != NULL) fclose(spacing_diag);
            exit(1);
        }
        if(t >= source_grid_stop) break;

        delT = thm_intrinsic_adaptive_step(model, t, source_grid_stop,
                                           THM_INTRINSIC_STEP_GROWTH_LIMIT*last_model_step,
                                           params, 0, &step_diag);
        if(spacing_diag != NULL)
        {
            fprintf(spacing_diag, "%d %.15e %.15e %.15e %.15e %.15e %.15e %d %d %d %d %d %d %d %d %d %d %d %d\n",
                    Ns_model-1, t, (t-tc)/Mtot, step_diag.accepted_step,
                    step_diag.proposed_step, step_diag.capped_step,
                    step_diag.max_omega/(2.0*M_PI),
                    step_diag.dtmax_cap, step_diag.phase_step_cap,
                    step_diag.tstop_cap, step_diag.shrink_count,
                    step_diag.fail_freq_step, step_diag.fail_delay_zero,
                    step_diag.fail_phase_curvature, step_diag.fail_rel_curvature,
                    step_diag.fail_amp_step, step_diag.fail_amp_curvature,
                    step_diag.last_fail_reason, step_diag.last_fail_mode);
        }

        tnext = t+delT;
        if(tnext <= t) tnext = t+dTmin;
        if(tnext > source_grid_stop) tnext = source_grid_stop;
        last_model_step = tnext-t;
        t = tnext;
    }
    if(spacing_diag != NULL) fclose(spacing_diag);

    gsl_sort(TS_model, 1, (size_t)Ns_model);
    {
        int q, Nu;
        double last_t;

        Nu = 0;
        last_t = 0.0;
        for(q=0; q<Ns_model; q++)
        {
            double tq = TS_model[q];
            double tol = 1.0e-7 + 1.0e-13*fabs(tq);

            if(Nu == 0 || fabs(tq-last_t) > tol)
            {
                TS_model[Nu] = tq;
                last_t = tq;
                Nu++;
            }
        }
        Ns_model = Nu;
    }

    response_switch_source_time =
        thm_intrinsic_tdi_switch_source_time(model, params, TS_model, Ns_model,
                                             source_grid_start,
                                             source_grid_stop,
                                             thm_intrinsic_tdi_include_delay_zero_grid);
    response_switch_detector_time =
        response_switch_source_time-THM_INTRINSIC_TDI_SWITCH_GUARD_SECONDS;
    if(response_switch_detector_time < tstart)
    {
        response_switch_detector_time = tstart;
    }
    if(response_switch_detector_time > tstop)
    {
        response_switch_detector_time = tstop;
    }

    Ns_response = 0;
    detector_planned_samples = 0;
    detector_step_history = 0;
    t = tstart;
    last_response_step = THM_DTM_MAX;
    previous_response_step = THM_DTM_MAX;
    response_diag = NULL;
    if(thm_fourier_ap_grid_diagnostic_prefix[0] != '\0')
    {
        snprintf(response_diag_path, sizeof(response_diag_path),
                 "%s_response_spacing.dat",
                 thm_fourier_ap_grid_diagnostic_prefix);
        response_diag = fopen(response_diag_path, "w");
        if(response_diag != NULL)
        {
            fprintf(response_diag, "# Hybrid detector-time response-grid spacing diagnostic.\n");
            fprintf(response_diag, "# reason labels: 0 ok, 1 endpoint_fractional_frequency_step, 2 equal_arm_delay_zero_crossing, 3 phase_curvature, 4 relative_frequency_curvature\n");
            fprintf(response_diag, "# probe labels: 0 SSB-labelled intrinsic knot, 1 center source u, 2 u-L, 3 u-2L, 4 u-4L\n");
            fprintf(response_diag, "# columns: interval_index center_source_time_u_s u_minus_tc_s ssb_output_time_s accepted_du_s proposed_du_s capped_du_s max_freq_Hz shrink_count fail_freq_step fail_delay_zero fail_phase_curvature fail_rel_curvature last_reason last_mode last_probe detector_planner\n");
        }
    }
    while(1)
    {
        THMAPStepDiagnostic step_diag;
        double delT;
        int used_detector_planner;

        memset(&step_diag, 0, sizeof(step_diag));
        step_diag.last_fail_reason = THM_AP_INTERVAL_OK;
        step_diag.last_fail_mode = -1;
        step_diag.last_fail_probe = -1;
        used_detector_planner = 0;

        if(Ns_response >= Nsmax)
        {
            fprintf(stderr, "Error: PhenomTHM_AP_IntrinsicTDIAdaptive exceeded Nsmax while building the hybrid detector-time response grid.\n");
            exit(1);
        }
        tspace[Ns_response] = t;
        Ns_response++;
        if(t >= tstop) break;

        if(t < response_switch_detector_time)
        {
            delT = THM_DTM_MAX;
            if(t+delT > response_switch_detector_time)
            {
                delT = response_switch_detector_time-t;
            }
            if(delT < dTmin) delT = dTmin;
            step_diag.proposed_step = THM_DTM_MAX;
            step_diag.capped_step = delT;
            step_diag.accepted_step = delT;
        }
        else
        {
            delT = thm_detector_adaptive_step(model, t, tstop,
                                              thm_detector_step_proposal(
                                                  last_response_step,
                                                  previous_response_step,
                                                  detector_step_history),
                                              params, SPacc, SPspline,
                                              thm_intrinsic_tdi_include_delay_zero_grid,
                                              &step_diag);
            detector_planned_samples++;
            detector_step_history++;
            used_detector_planner = 1;
        }

        if(response_diag != NULL)
        {
            double source_t = thm_ssb_output_time_from_center_source_time(
                t, params, SPacc, SPspline);
            fprintf(response_diag,
                    "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %d %d %d %d %d %d %d %d %d\n",
                    Ns_response-1, t, t-tc, source_t, delT,
                    step_diag.proposed_step, step_diag.capped_step,
                    step_diag.max_omega/(2.0*M_PI),
                    step_diag.shrink_count, step_diag.fail_freq_step,
                    step_diag.fail_delay_zero,
                    step_diag.fail_phase_curvature,
                    step_diag.fail_rel_curvature,
                    step_diag.last_fail_reason,
                    step_diag.last_fail_mode,
                    step_diag.last_fail_probe,
                    used_detector_planner);
        }

        tnext = t+delT;
        if(tnext <= t) tnext = t+dTmin;
        if(tnext > tstop) tnext = tstop;
        if(used_detector_planner)
        {
            previous_response_step = last_response_step;
            last_response_step = tnext-t;
        }
        t = tnext;
    }
    if(response_diag != NULL) fclose(response_diag);

    barycenter_time(TS, tspace, params, SPacc, SPspline, Ns_response);
    thm_last_intrinsic_tdi_switch_source_time = response_switch_source_time;
    thm_last_intrinsic_tdi_switch_detector_time = response_switch_detector_time;
    thm_last_intrinsic_tdi_detector_planned_samples = detector_planned_samples;

    if(TS[0] < source_grid_start)
    {
        source_grid_start = TS[0];
        if(!thm_append_source_model_time(TS_model, &Ns_model,
                                         model_capacity, TS[0]))
        {
            fprintf(stderr, "Error: intrinsic/TDI source-time model grid exceeded capacity while adding the first response sample.\n");
            exit(1);
        }
    }
    if(TS[Ns_response-1] > source_grid_stop)
    {
        source_grid_stop = TS[Ns_response-1];
        if(!thm_append_source_model_time(TS_model, &Ns_model,
                                         model_capacity, TS[Ns_response-1]))
        {
            fprintf(stderr, "Error: intrinsic/TDI source-time model grid exceeded capacity while adding the final response sample.\n");
            exit(1);
        }
    }

    gsl_sort(TS_model, 1, (size_t)Ns_model);
    {
        int q, Nu;
        double last_t;

        Nu = 0;
        last_t = 0.0;
        for(q=0; q<Ns_model; q++)
        {
            double tq = TS_model[q];
            double tol = 1.0e-7 + 1.0e-13*fabs(tq);

            if(Nu == 0 || fabs(tq-last_t) > tol)
            {
                TS_model[Nu] = tq;
                last_t = tq;
                Nu++;
            }
        }
        Ns_model = Nu;
    }

    for(i=0; i<Ns_response; i++)
    {
        source_index[i] = -1;
        if(thm_response_sample_should_be_exact(tspace, Ns_response, i))
        {
            if(!thm_append_source_model_time(TS_model, &Ns_model,
                                             model_capacity, TS[i]))
            {
                fprintf(stderr, "Error: intrinsic/TDI source-time model grid exceeded capacity while adding exact response samples.\n");
                exit(1);
            }
        }
    }

    gsl_sort(TS_model, 1, (size_t)Ns_model);
    {
        int q, Nu;
        double last_t;

        Nu = 0;
        last_t = 0.0;
        for(q=0; q<Ns_model; q++)
        {
            double tq = TS_model[q];
            double tol = 1.0e-7 + 1.0e-13*fabs(tq);

            if(Nu == 0 || fabs(tq-last_t) > tol)
            {
                TS_model[Nu] = tq;
                last_t = tq;
                Nu++;
            }
        }
        Ns_model = Nu;
    }

    for(i=0; i<Ns_response; i++)
    {
        source_index[i] = thm_find_source_model_time(TS_model, Ns_model, TS[i]);
    }

    for(i=0; i<Ns_model; i++) tau_model[i] = (TS_model[i]-tc)/Mtot;
    source_tmin = tau_model[0];
    source_tmax = tau_model[Ns_model-1];
    if(source_tmax <= source_tmin)
    {
        fprintf(stderr, "Error: intrinsic/TDI AP grid produced non-increasing source times.\n");
        exit(1);
    }

    status = IMRPhenomTHMBuildPhi22Grid(model, Ns_model, tau_model,
                                        IMRPhenomTPhase22(source_tmin, eta, &model->mode22),
                                        phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMBuildPhi22Grid failed in intrinsic/TDI AP grid: %d\n",
                status);
        exit(1);
    }
    {
        double shift = params[4]-IMRPhenomTPhase22(0.0, eta, &model->mode22);
        for(i=0; i<Ns_model; i++) phi22[i] += shift;
    }
    status = IMRPhenomTHMSetPhaseOffsetsFromGrid(model, Ns_model,
                                                 tau_model, phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMSetPhaseOffsetsFromGrid failed in intrinsic/TDI AP grid: %d\n",
                status);
        exit(1);
    }

    strain_scale = sqrt(2.0)*eta*Mtot/(distance*GPSEC);
    for(i=0; i<Ns_model; i++)
    {
        for(k=0; k<*nmodes; k++)
        {
            IMRPhenomTHMModeSample sample;
            status = IMRPhenomTHMEvaluateMode(model, k, tau_model[i],
                                              phi22[i], &sample);
            if(status != 0)
            {
                fprintf(stderr, "IMRPhenomTHMEvaluateMode failed in intrinsic/TDI AP grid at mode %d sample %d: %d\n",
                        k, i, status);
                exit(1);
            }
            amp_model[k][i] = strain_scale*sample.amplitude;
            phase_model[k][i] = sample.phase;
            freq_model[k][i] = sample.omega/(2.0*M_PI*Mtot);
        }
    }

    if(thm_diagnostics_enabled)
    {
        write_thm_mode_ap("PhenomTHM_modes_AP_intrinsic_model.dat",
                          Ns_model, *nmodes, modes, TS_model, tau_model,
                          amp_model, phase_model, freq_model);
        {
            FILE *grid = fopen("PhenomTHM_AP_intrinsic_model_grid.dat", "w");
            if(grid != NULL)
            {
                fprintf(grid, "# i source_time_s tau dt_source_s\n");
                for(i=0; i<Ns_model; i++)
                {
                    double dt_source = (i > 0) ? TS_model[i]-TS_model[i-1] : 0.0;
                    fprintf(grid, "%d %.15e %.15e %.15e\n",
                            i, TS_model[i], tau_model[i], dt_source);
                }
                fclose(grid);
            }
        }
        {
            FILE *grid = fopen("PhenomTHM_AP_intrinsic_TDI_response_grid.dat", "w");
            if(grid != NULL)
            {
                fprintf(grid, "# switch_source_time_s %.15e switch_detector_time_s %.15e detector_planned_samples %d\n",
                        response_switch_source_time, response_switch_detector_time,
                        detector_planned_samples);
                fprintf(grid, "# i center_source_time_u_s ssb_output_time_s tau dt_u_s dt_ssb_s source_model_index exact_model_sample\n");
                for(i=0; i<Ns_response; i++)
                {
                    double dt_detector = (i > 0) ? tspace[i]-tspace[i-1] : 0.0;
                    double dt_source = (i > 0) ? TS[i]-TS[i-1] : 0.0;
                    fprintf(grid, "%d %.15e %.15e %.15e %.15e %.15e %d %d\n",
                            i, tspace[i], TS[i], (TS[i]-tc)/Mtot,
                            dt_detector, dt_source, source_index[i],
                            source_index[i] >= 0 ? 1 : 0);
                }
                fclose(grid);
            }
        }
    }

    {
        const gsl_interp_type *interp_type = (Ns_model >= 5) ? THM_AP_SPLINE_TYPE : gsl_interp_linear;
        gsl_interp_accel *Aacc_local = gsl_interp_accel_alloc();
        gsl_interp_accel *Pacc_local = gsl_interp_accel_alloc();
        gsl_interp_accel *Facc_local = gsl_interp_accel_alloc();
        gsl_spline *Aspline_local = gsl_spline_alloc(interp_type, Ns_model);
        gsl_spline *Pspline_local = gsl_spline_alloc(interp_type, Ns_model);
        gsl_spline *Fspline_local = gsl_spline_alloc(interp_type, Ns_model);

        if(Aacc_local == NULL || Pacc_local == NULL || Facc_local == NULL ||
           Aspline_local == NULL || Pspline_local == NULL ||
           Fspline_local == NULL)
        {
            fprintf(stderr, "allocation failure while interpolating intrinsic/TDI AP grid\n");
            exit(1);
        }

        thm_last_intrinsic_tdi_model_samples = Ns_model;
        for(k=0; k<*nmodes; k++)
        {
            gsl_spline_init(Aspline_local, TS_model, amp_model[k], Ns_model);
            gsl_spline_init(Pspline_local, TS_model, phase_model[k], Ns_model);
            gsl_spline_init(Fspline_local, TS_model, freq_model[k], Ns_model);
            gsl_interp_accel_reset(Aacc_local);
            gsl_interp_accel_reset(Pacc_local);
            gsl_interp_accel_reset(Facc_local);

            for(i=0; i<Ns_response; i++)
            {
                int q = source_index[i];
                if(q >= 0)
                {
                    mode_amp[k][i] = amp_model[k][q];
                    mode_phase[k][i] = phase_model[k][q];
                    mode_freq[k][i] = freq_model[k][q];
                    if(k == 0) thm_last_intrinsic_tdi_exact_samples++;
                }
                else
                {
                    mode_amp[k][i] = gsl_spline_eval(Aspline_local, TS[i],
                                                      Aacc_local);
                    mode_phase[k][i] = gsl_spline_eval(Pspline_local, TS[i],
                                                        Pacc_local);
                    mode_freq[k][i] = gsl_spline_eval(Fspline_local, TS[i],
                                                       Facc_local);
                    if(k == 0) thm_last_intrinsic_tdi_interpolated_samples++;
                }
            }
        }

        gsl_spline_free(Aspline_local);
        gsl_spline_free(Pspline_local);
        gsl_spline_free(Fspline_local);
        gsl_interp_accel_free(Aacc_local);
        gsl_interp_accel_free(Pacc_local);
        gsl_interp_accel_free(Facc_local);
    }

    free_double_vector(TS_model);
    free_double_vector(tau_model);
    free_double_vector(phi22);
    free_double_matrix(amp_model, IMRPHENOMTHM_MAX_MODES);
    free_double_matrix(phase_model, IMRPHENOMTHM_MAX_MODES);
    free_double_matrix(freq_model, IMRPHENOMTHM_MAX_MODES);
    free_int_vector(source_index);

    return Ns_response;
}

int PhenomTHM_AP_OnDetectorGrid(double *params, IMRPhenomTHM *model, IMRPhenomTHMMode *modes, int *nmodes, const IMRPhenomTHMMode *requested_modes, int requested_nmodes, int Ns, double *TS, double *tspace, double **mode_amp, double **mode_phase, double **mode_freq, gsl_interp_accel **SPacc, gsl_spline **SPspline)
{
    double m1, m2, chi1, chi2, Mtot, eta, tc, distance;
    double strain_scale;
    double *tau, *phi22;
    double source_tmin, source_tmax;
    int status;
    int i, k;

    if(params == NULL || model == NULL || modes == NULL || nmodes == NULL ||
       TS == NULL || tspace == NULL || mode_amp == NULL || mode_phase == NULL ||
       mode_freq == NULL || Ns < 2)
    {
        return 0;
    }

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    tc = params[5];
    distance = exp(params[6]);
    Mtot = m1 + m2;
    eta = (m1*m2)/(Mtot*Mtot);

    if(requested_modes != NULL && requested_nmodes > 0)
    {
        if(requested_nmodes > IMRPHENOMTHM_MAX_MODES)
        {
            fprintf(stderr, "Error: requested %d THM modes, but only %d are supported by this driver.\n",
                    requested_nmodes, IMRPHENOMTHM_MAX_MODES);
            exit(1);
        }
        *nmodes = requested_nmodes;
        for(i=0; i<*nmodes; i++) modes[i] = requested_modes[i];
    }
    else
    {
        *nmodes = IMRPhenomTHMDefaultModes(modes, IMRPHENOMTHM_MAX_MODES);
    }

    status = IMRPhenomTHMInitialize(model, m1, m2, chi1, chi2, modes, *nmodes);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMInitialize failed on fixed detector grid: %d\n", status);
        exit(1);
    }

    /*
     * The Fisher AP-derivative path keeps one detector-time grid fixed at the
     * base source parameters.  For each perturbation, remap that same detector
     * grid to barycenter/source time before evaluating the intrinsic mode AP.
     * This lets sky-position derivatives include the time-delay geometry while
     * avoiding finite-difference noise from moving adaptive sample locations.
     */
    barycenter_time(TS, tspace, params, SPacc, SPspline, Ns);

    tau = double_vector(Ns);
    phi22 = double_vector(Ns);
    if(tau == NULL || phi22 == NULL)
    {
        fprintf(stderr, "allocation failure in PhenomTHM_AP_OnDetectorGrid\n");
        exit(1);
    }

    for(i=0; i<Ns; i++) tau[i] = (TS[i]-tc)/Mtot;

    source_tmin = tau[0];
    source_tmax = tau[Ns-1];
    if(source_tmax <= source_tmin)
    {
        fprintf(stderr, "Error: fixed detector grid produced non-increasing source times.\n");
        exit(1);
    }

    status = IMRPhenomTHMBuildPhi22Grid(model, Ns, tau,
                                        IMRPhenomTPhase22(source_tmin, eta, &model->mode22),
                                        phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMBuildPhi22Grid failed on fixed detector grid: %d\n", status);
        exit(1);
    }
    {
        double shift = params[4]-IMRPhenomTPhase22(0.0, eta, &model->mode22);
        for(i=0; i<Ns; i++) phi22[i] += shift;
    }
    status = IMRPhenomTHMSetPhaseOffsetsFromGrid(model, Ns, tau, phi22);
    if(status != 0)
    {
        fprintf(stderr, "IMRPhenomTHMSetPhaseOffsetsFromGrid failed on fixed detector grid: %d\n", status);
        exit(1);
    }

    strain_scale = sqrt(2.0)*eta*Mtot/(distance*GPSEC);
    for(i=0; i<Ns; i++)
    {
        for(k=0; k<*nmodes; k++)
        {
            IMRPhenomTHMModeSample sample;
            status = IMRPhenomTHMEvaluateMode(model, k, tau[i], phi22[i], &sample);
            if(status != 0)
            {
                fprintf(stderr, "IMRPhenomTHMEvaluateMode failed on fixed detector grid at mode %d sample %d: %d\n",
                        k, i, status);
                exit(1);
            }
            mode_amp[k][i] = strain_scale*sample.amplitude;
            mode_phase[k][i] = sample.phase;
            mode_freq[k][i] = sample.omega/(2.0*M_PI*Mtot);
        }
    }

    free_double_vector(tau);
    free_double_vector(phi22);

    return Ns;
}

void hphc_thm(double t, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *hp, double *hc, double *hpf, double *hcf)
{
    int k;

    *hp = 0.0;
    *hc = 0.0;
    *hpf = 0.0;
    *hcf = 0.0;
    if(t < thm_response_spline_tmin || t > thm_response_spline_tmax)
    {
        return;
    }
    for(k=0; k<ncarriers; k++)
    {
        int idx = carrier[k].mode_index;
        double A = gsl_spline_eval(Aspline[idx], t, Aacc[idx]);
        double phase = gsl_spline_eval(Pspline[idx], t, Pacc[idx]);
        double cp = cos(phase);
        double sp = sin(phase);

        *hp += A*(carrier[k].hp_cos*cp + carrier[k].hp_sin*sp);
        *hc += A*(carrier[k].hc_cos*cp + carrier[k].hc_sin*sp);
        *hpf += A*(carrier[k].hpf_cos*cp + carrier[k].hpf_sin*sp);
        *hcf += A*(carrier[k].hcf_cos*cp + carrier[k].hcf_sin*sp);
    }
}

void TDI_spline_thm_piece(double *M, double *Mf, int a, int b, int c, double* tarray, int n, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm, int piece)
{
    double t, hp, hc, hpf, hcf;

    M[n] = 0.0;
    Mf[n] = 0.0;

    if(piece != THM_TDI_PIECE_FULL &&
       piece != THM_TDI_PIECE_DELAYED_MICHELSON &&
       piece != THM_TDI_PIECE_PROMPT_MICHELSON)
    {
        piece = THM_TDI_PIECE_FULL;
    }

    /*
     * Experimental split-TDI grouping.  Each waveform evaluation contributes two
     * projected link terms, so the four evaluations below are the eight terms of
     * the delayed Michelson-like copy in Eq. (3) of arXiv:2506.08093.  The next
     * four evaluations are the corresponding prompt Michelson-like response.  The
     * delayed and prompt pieces sum to the standard link-stream Michelson sign.
     */
    if(piece == THM_TDI_PIECE_FULL ||
       piece == THM_TDI_PIECE_DELAYED_MICHELSON)
    {
        t = tarray[n] - kr[a]-2.0*Larm[c]-2.0*Larm[b];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] += hp*App[c]+hc*Acp[c];
        M[n] -= hp*Apm[b]+hc*Acm[b];
        Mf[n] += hpf*App[c]+hcf*Acp[c];
        Mf[n] -= hpf*Apm[b]+hcf*Acm[b];

        t = tarray[n] - kr[b]-Larm[c]-2.0*Larm[b];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] -= hp*App[c]+hc*Acp[c];
        M[n] += hp*Apm[c]+hc*Acm[c];
        Mf[n] -= hpf*App[c]+hcf*Acp[c];
        Mf[n] += hpf*Apm[c]+hcf*Acm[c];

        t = tarray[n] - kr[c]-Larm[b]-2.0*Larm[c];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] += hp*Apm[b]+hc*Acm[b];
        M[n] -= hp*App[b]+hc*Acp[b];
        Mf[n] += hpf*Apm[b]+hcf*Acm[b];
        Mf[n] -= hpf*App[b]+hcf*Acp[b];

        t = tarray[n] - kr[a]-2.0*Larm[b];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] -= hp*Apm[c]+hc*Acm[c];
        M[n] += hp*Apm[b]+hc*Acm[b];
        Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
        Mf[n] += hpf*Apm[b]+hcf*Acm[b];
    }

    if(piece == THM_TDI_PIECE_FULL ||
       piece == THM_TDI_PIECE_PROMPT_MICHELSON)
    {
        t = tarray[n] - kr[a]-2.0*Larm[c];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] += hp*App[b]+hc*Acp[b];
        M[n] -= hp*App[c]+hc*Acp[c];
        Mf[n] += hpf*App[b]+hcf*Acp[b];
        Mf[n] -= hpf*App[c]+hcf*Acp[c];

        t = tarray[n] - kr[c]- Larm[b];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] -= hp*Apm[b]+hc*Acm[b];
        M[n] += hp*App[b]+hc*Acp[b];
        Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
        Mf[n] += hpf*App[b]+hcf*Acp[b];

        t = tarray[n] - kr[b]- Larm[c];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] += hp*App[c]+hc*Acp[c];
        M[n] -= hp*Apm[c]+hc*Acm[c];
        Mf[n] += hpf*App[c]+hcf*Acp[c];
        Mf[n] -= hpf*Apm[c]+hcf*Acm[c];

        t = tarray[n] - kr[a];
        hphc_thm(t, ncarriers, carrier, Aacc, Aspline, Pacc, Pspline, &hp, &hc, &hpf, &hcf);
        M[n] -= hp*App[b]+hc*Acp[b];
        M[n] += hp*Apm[c]+hc*Acm[c];
        Mf[n] -= hpf*App[b]+hcf*Acp[b];
        Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    }

}

void TDI_spline_thm(double *M, double *Mf, int a, int b, int c, double* tarray, int n, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    TDI_spline_thm_piece(M, Mf, a, b, c, tarray, n, ncarriers, carrier,
                         Aacc, Aspline, Pacc, Pspline,
                         App, Apm, Acp, Acm, kr, Larm,
                         THM_TDI_PIECE_FULL);
}

void fast_response_thm_piece(double *tarray, int N, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf, int piece)
{
    int i, j, k, n;
    double *u,*v,*kv;
    double *kr, *kn;
    double *plus, *cross;
    double **eplus, **ecross;
    double costh, phi, sinth, cosph, sinph;
    double *App, *Apm, *Acp, *Acm;
    double *Larm;
    double **Parray, **Varray;

    Larm = double_vector(3);
    Parray = double_matrix(3,3);
    Varray = double_matrix(3,3);
    u = double_vector(3);
    v = double_vector(3);
    kv = double_vector(3);
    eplus = double_matrix(3,3);
    ecross = double_matrix(3,3);
    kr = double_vector(3);
    kn = double_vector(3);
    plus = double_vector(3);
    cross = double_vector(3);
    App = double_vector(3);
    Apm = double_vector(3);
    Acp = double_vector(3);
    Acm = double_vector(3);

    costh = sin(params[7]);
    phi = params[8];
    sinth = sqrt(1.0-costh*costh);
    cosph = cos(phi);
    sinph = sin(phi);

    u[0] = -costh*cosph;  u[1] = -costh*sinph;  u[2] =  sinth;
    v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.0;
    kv[0] = -sinth*cosph;
    kv[1] = -sinth*sinph;
    kv[2] = -costh;

    for(i=0; i<3; i++)
    {
        for(j=0; j<3; j++)
        {
            eplus[i][j] = v[i]*v[j] - u[i]*u[j];
            ecross[i][j] = u[i]*v[j] + v[i]*u[j];
        }
    }

    for(n=0; n<N; n++)
    {
        double t = tarray[n];
        for(i=0; i<3; i++) Larm[i] = gsl_spline_eval(SLspline[i], t, SLacc[i]);
        for(i=0; i<3; i++)
        {
            for(j=0; j<3; j++)
            {
                k = j+3*i;
                Parray[i][j] = gsl_spline_eval(SPspline[k], t, SPacc[k]);
                Varray[i][j] = gsl_spline_eval(SVspline[k], t, SVacc[k]);
            }
        }
        for(i=0; i<3; i++)
        {
            kr[i] = 0.0;
            kn[i] = 0.0;
            for(j=0; j<3; j++)
            {
                kr[i] += Parray[i][j]*kv[j];
                kn[i] += Varray[i][j]*kv[j];
            }
        }
        for(i=0; i<3; i++)
        {
            plus[i] = 0.0;
            cross[i] = 0.0;
            for(j=0; j<3; j++)
            {
                for(k=0; k<3; k++)
                {
                    plus[i] += Varray[i][j]*Varray[i][k]*eplus[j][k];
                    cross[i] += Varray[i][j]*Varray[i][k]*ecross[j][k];
                }
            }
            App[i] = 0.5*plus[i]/(1.0+kn[i]);
            Apm[i] = 0.5*plus[i]/(1.0-kn[i]);
            Acp[i] = 0.5*cross[i]/(1.0+kn[i]);
            Acm[i] = 0.5*cross[i]/(1.0-kn[i]);
        }

        TDI_spline_thm_piece(X, Xf, 0, 1, 2, tarray, n, ncarriers, carrier,
                             Aacc, Aspline, Pacc, Pspline, App, Apm, Acp, Acm,
                             kr, Larm, piece);
        TDI_spline_thm_piece(Y, Yf, 1, 2, 0, tarray, n, ncarriers, carrier,
                             Aacc, Aspline, Pacc, Pspline, App, Apm, Acp, Acm,
                             kr, Larm, piece);
        TDI_spline_thm_piece(Z, Zf, 2, 0, 1, tarray, n, ncarriers, carrier,
                             Aacc, Aspline, Pacc, Pspline, App, Apm, Acp, Acm,
                             kr, Larm, piece);
    }

    free_double_vector(Larm);
    free_double_matrix(Parray,3);
    free_double_matrix(Varray,3);
    free_double_vector(u);
    free_double_vector(v);
    free_double_vector(kv);
    free_double_matrix(eplus,3);
    free_double_matrix(ecross,3);
    free_double_vector(kr);
    free_double_vector(kn);
    free_double_vector(plus);
    free_double_vector(cross);
    free_double_vector(App);
    free_double_vector(Apm);
    free_double_vector(Acp);
    free_double_vector(Acm);
}

void fast_response_thm(double *tarray, int N, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf)
{
    fast_response_thm_piece(tarray, N, params, ncarriers, carrier,
                            SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                            Aacc, Aspline, Pacc, Pspline,
                            X, Y, Z, Xf, Yf, Zf,
                            THM_TDI_PIECE_FULL);
}

void write_thm_mode_ap(const char *filename, int Ns, int nmodes, IMRPhenomTHMMode *modes, double *TS, double *tc_tau, double **mode_amp, double **mode_phase, double **mode_freq)
{
    FILE *out = fopen(filename, "w");
    int i, k;

    if(out == NULL)
    {
        perror(filename);
        return;
    }
    fprintf(out, "# columns: ssb_reference_source_time_s tau");
    for(k=0; k<nmodes; k++)
    {
        fprintf(out, " A%d_%+d phase%d_%+d freq%d_%+d_Hz",
                modes[k].ell, modes[k].emm,
                modes[k].ell, modes[k].emm,
                modes[k].ell, modes[k].emm);
    }
    fprintf(out, "\n");
    for(i=0; i<Ns; i++)
    {
        fprintf(out, "%.15e %.15e", TS[i], tc_tau[i]);
        for(k=0; k<nmodes; k++)
        {
            fprintf(out, " %.15e %.15e %.15e",
                    mode_amp[k][i], mode_phase[k][i], mode_freq[k][i]);
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

void write_thm_tdi_time(const char *filename, int Ns, double *TS, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf)
{
    FILE *out = fopen(filename, "w");
    int i;

    if(out == NULL)
    {
        perror(filename);
        return;
    }
    fprintf(out, "# columns: ssb_output_time_s X Y Z X_quadrature Y_quadrature Z_quadrature\n");
    for(i=0; i<Ns; i++)
    {
        fprintf(out, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                TS[i], X[i], Y[i], Z[i], Xf[i], Yf[i], Zf[i]);
    }
    fclose(out);
}

static double thm_bounded_chunk_start(double start, double duration, double safe_start, double safe_end)
{
    if(start < safe_start) start = safe_start;
    if(start+duration > safe_end) start = safe_end-duration;
    if(start < safe_start) start = safe_start;
    return start;
}

static void hphc_dense_full_waveform(double t, gsl_interp_accel *HPacc, gsl_spline *HPspline, gsl_interp_accel *HCacc, gsl_spline *HCspline, double tmin, double tmax, double *hp, double *hc)
{
    if(t < tmin || t > tmax)
    {
        *hp = 0.0;
        *hc = 0.0;
        return;
    }
    *hp = gsl_spline_eval(HPspline, t, HPacc);
    *hc = gsl_spline_eval(HCspline, t, HCacc);
}

static void TDI_full_waveform_dense(double *M, int a, int b, int c, double *tarray, int n, gsl_interp_accel *HPacc, gsl_spline *HPspline, gsl_interp_accel *HCacc, gsl_spline *HCspline, double h_tmin, double h_tmax, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    double t, hp, hc;

    M[n] = 0.0;

    t = tarray[n] - kr[a]-2.0*Larm[c]-2.0*Larm[b];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] += hp*App[c]+hc*Acp[c];
    M[n] -= hp*Apm[b]+hc*Acm[b];

    t = tarray[n] - kr[b]-Larm[c]-2.0*Larm[b];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] -= hp*App[c]+hc*Acp[c];
    M[n] += hp*Apm[c]+hc*Acm[c];

    t = tarray[n] - kr[c]-Larm[b]-2.0*Larm[c];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] += hp*Apm[b]+hc*Acm[b];
    M[n] -= hp*App[b]+hc*Acp[b];

    t = tarray[n] - kr[a]-2.0*Larm[b];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] -= hp*Apm[c]+hc*Acm[c];
    M[n] += hp*Apm[b]+hc*Acm[b];

    t = tarray[n] - kr[a]-2.0*Larm[c];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] += hp*App[b]+hc*Acp[b];
    M[n] -= hp*App[c]+hc*Acp[c];

    t = tarray[n] - kr[c]- Larm[b];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] -= hp*Apm[b]+hc*Acm[b];
    M[n] += hp*App[b]+hc*Acp[b];

    t = tarray[n] - kr[b]- Larm[c];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] += hp*App[c]+hc*Acp[c];
    M[n] -= hp*Apm[c]+hc*Acm[c];

    t = tarray[n] - kr[a];
    hphc_dense_full_waveform(t, HPacc, HPspline, HCacc, HCspline, h_tmin, h_tmax, &hp, &hc);
    M[n] -= hp*App[b]+hc*Acp[b];
    M[n] += hp*Apm[c]+hc*Acm[c];

}

void direct_response_full_waveform_thm(double *tarray, int N, double dense_dt, double delay_pad, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, double *X, double *Y, double *Z)
{
    int i, j, k, n;
    int Nh;
    double h_tmin, h_tmax;
    double *htime, *hplus, *hcross;
    gsl_interp_accel *HPacc, *HCacc;
    gsl_spline *HPspline, *HCspline;
    double *u, *v, *kv;
    double *kr, *kn;
    double *plus, *cross;
    double **eplus, **ecross;
    double costh, phi, sinth, cosph, sinph;
    double *App, *Apm, *Acp, *Acm;
    double *Larm;
    double **Parray, **Varray;

    if(N < 1)
    {
        return;
    }

    h_tmin = tarray[0]-delay_pad;
    h_tmax = tarray[N-1]+delay_pad;
    if(h_tmin < thm_response_spline_tmin) h_tmin = thm_response_spline_tmin;
    if(h_tmax > thm_response_spline_tmax) h_tmax = thm_response_spline_tmax;
    if(h_tmax-h_tmin < dense_dt)
    {
        for(n=0; n<N; n++)
        {
            X[n] = 0.0;
            Y[n] = 0.0;
            Z[n] = 0.0;
        }
        return;
    }

    Nh = (int)floor((h_tmax-h_tmin)/dense_dt)+1;
    if(Nh < 4) Nh = 4;
    htime = double_vector(Nh);
    hplus = double_vector(Nh);
    hcross = double_vector(Nh);

    for(i=0; i<Nh; i++)
    {
        double t = h_tmin+(double)i*dense_dt;
        if(t > h_tmax) t = h_tmax;
        htime[i] = t;
    }

    /*
     * Build the dense total waveform before applying any TDI delays.  The AP
     * splines define the THM waveform for this driver; this diagnostic samples
     * that waveform at full cadence, sums all modes into h_+(t), h_x(t), and
     * then gives the TDI code only those total waveform arrays.  That ordering
     * is the direct Eq. (3) check we want: the delays act on the entire signal,
     * not on separately delayed mode carriers.
     */
    for(i=0; i<Nh; i++)
    {
        double hpf, hcf;
        hphc_thm(htime[i], ncarriers, carrier, Aacc, Aspline, Pacc, Pspline,
                 &hplus[i], &hcross[i], &hpf, &hcf);
    }

    HPacc = gsl_interp_accel_alloc();
    HCacc = gsl_interp_accel_alloc();
    HPspline = gsl_spline_alloc(gsl_interp_cspline, Nh);
    HCspline = gsl_spline_alloc(gsl_interp_cspline, Nh);
    gsl_spline_init(HPspline, htime, hplus, Nh);
    gsl_spline_init(HCspline, htime, hcross, Nh);

    Larm = double_vector(3);
    Parray = double_matrix(3,3);
    Varray = double_matrix(3,3);
    u = double_vector(3);
    v = double_vector(3);
    kv = double_vector(3);
    eplus = double_matrix(3,3);
    ecross = double_matrix(3,3);
    kr = double_vector(3);
    kn = double_vector(3);
    plus = double_vector(3);
    cross = double_vector(3);
    App = double_vector(3);
    Apm = double_vector(3);
    Acp = double_vector(3);
    Acm = double_vector(3);

    costh = sin(params[7]);
    phi = params[8];
    sinth = sqrt(1.0-costh*costh);
    cosph = cos(phi);
    sinph = sin(phi);

    u[0] = -costh*cosph;  u[1] = -costh*sinph;  u[2] =  sinth;
    v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.0;
    kv[0] = -sinth*cosph;
    kv[1] = -sinth*sinph;
    kv[2] = -costh;

    for(i=0; i<3; i++)
    {
        for(j=0; j<3; j++)
        {
            eplus[i][j] = v[i]*v[j] - u[i]*u[j];
            ecross[i][j] = u[i]*v[j] + v[i]*u[j];
        }
    }

    for(n=0; n<N; n++)
    {
        double t = tarray[n];
        for(i=0; i<3; i++) Larm[i] = gsl_spline_eval(SLspline[i], t, SLacc[i]);
        for(i=0; i<3; i++)
        {
            for(j=0; j<3; j++)
            {
                k = j+3*i;
                Parray[i][j] = gsl_spline_eval(SPspline[k], t, SPacc[k]);
                Varray[i][j] = gsl_spline_eval(SVspline[k], t, SVacc[k]);
            }
        }
        for(i=0; i<3; i++)
        {
            kr[i] = 0.0;
            kn[i] = 0.0;
            for(j=0; j<3; j++)
            {
                kr[i] += Parray[i][j]*kv[j];
                kn[i] += Varray[i][j]*kv[j];
            }
        }
        for(i=0; i<3; i++)
        {
            plus[i] = 0.0;
            cross[i] = 0.0;
            for(j=0; j<3; j++)
            {
                for(k=0; k<3; k++)
                {
                    plus[i] += Varray[i][j]*Varray[i][k]*eplus[j][k];
                    cross[i] += Varray[i][j]*Varray[i][k]*ecross[j][k];
                }
            }
            App[i] = 0.5*plus[i]/(1.0+kn[i]);
            Apm[i] = 0.5*plus[i]/(1.0-kn[i]);
            Acp[i] = 0.5*cross[i]/(1.0+kn[i]);
            Acm[i] = 0.5*cross[i]/(1.0-kn[i]);
        }

        TDI_full_waveform_dense(X, 0, 1, 2, tarray, n, HPacc, HPspline, HCacc, HCspline,
                                htime[0], htime[Nh-1], App, Apm, Acp, Acm, kr, Larm);
        TDI_full_waveform_dense(Y, 1, 2, 0, tarray, n, HPacc, HPspline, HCacc, HCspline,
                                htime[0], htime[Nh-1], App, Apm, Acp, Acm, kr, Larm);
        TDI_full_waveform_dense(Z, 2, 0, 1, tarray, n, HPacc, HPspline, HCacc, HCspline,
                                htime[0], htime[Nh-1], App, Apm, Acp, Acm, kr, Larm);
    }

    gsl_spline_free(HPspline);
    gsl_spline_free(HCspline);
    gsl_interp_accel_free(HPacc);
    gsl_interp_accel_free(HCacc);
    free_double_vector(htime);
    free_double_vector(hplus);
    free_double_vector(hcross);
    free_double_vector(Larm);
    free_double_matrix(Parray,3);
    free_double_matrix(Varray,3);
    free_double_vector(u);
    free_double_vector(v);
    free_double_vector(kv);
    free_double_matrix(eplus,3);
    free_double_matrix(ecross,3);
    free_double_vector(kr);
    free_double_vector(kn);
    free_double_vector(plus);
    free_double_vector(cross);
    free_double_vector(App);
    free_double_vector(Apm);
    free_double_vector(Acp);
    free_double_vector(Acm);
}

void write_thm_tdi_dense_check(const char *data_filename, const char *summary_filename, double dense_dt, double chunk_seconds, double tc, int Ns_sparse, double *TS_sparse, double *params, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline)
{
    FILE *data, *summary;
    const char *names[2] = {"early", "merger"};
    double starts[2];
    double safe_start, safe_end, duration, delay_pad;
    double early_anchor, merger_anchor;
    int chunk;

    if(dense_dt <= 0.0 || chunk_seconds <= 0.0 || Ns_sparse < 4)
    {
        fprintf(stderr, "Warning: skipping THM dense TDI check because the grid parameters are invalid.\n");
        return;
    }

    delay_pad = 2.0*CONSTELLATION_LIGHT_TIME_SECONDS;

    /*
     * Direct THM TDI diagnostic:
     *   1. Build h_+(t), h_x(t) on a dense time grid for the whole chunk plus
     *      a light-time delay margin.
     *   2. Spline only those already-summed waveforms.
     *   3. Apply the Eq. (3) delayed TDI combination to that full waveform.
     *
     * The mode-delay columns are retained as a local comparison with the
     * sparse AP machinery, but the direct columns are the independent
     * reference for testing the fast response.
     */
    safe_start = TS_sparse[0]+delay_pad;
    safe_end = thm_response_spline_tmax-delay_pad;
    if(safe_end > TS_sparse[Ns_sparse-1]) safe_end = TS_sparse[Ns_sparse-1];
    if(safe_end-safe_start < dense_dt)
    {
        safe_start = TS_sparse[0];
        safe_end = TS_sparse[Ns_sparse-1];
    }
    if(safe_end-safe_start < dense_dt)
    {
        fprintf(stderr, "Warning: skipping THM dense TDI check because the response span is too short.\n");
        return;
    }

    duration = chunk_seconds;
    if(duration > safe_end-safe_start)
    {
        duration = safe_end-safe_start;
    }

    early_anchor = safe_start;
    if(tc > safe_start)
    {
        double early_end = tc;
        if(early_end > safe_end) early_end = safe_end;
        early_anchor = safe_start+THM_TDI_DENSE_CHECK_EARLY_FRACTION*(early_end-safe_start);
    }
    merger_anchor = tc-0.5*duration;
    starts[0] = thm_bounded_chunk_start(early_anchor, duration, safe_start, safe_end);
    starts[1] = thm_bounded_chunk_start(merger_anchor, duration, safe_start, safe_end);

    data = fopen(data_filename, "w");
    if(data == NULL)
    {
        perror(data_filename);
        return;
    }
    summary = fopen(summary_filename, "w");
    if(summary == NULL)
    {
        perror(summary_filename);
        fclose(data);
        return;
    }

    fprintf(data, "# columns: chunk_id t_s X_direct Y_direct Z_direct X_mode_delay Y_mode_delay Z_mode_delay dX dY dZ\n");
    fprintf(summary, "# chunk_id name start_s end_s dt_s samples direct_vs_mode_match rel_l2 max_abs_diff max_abs_direct direct_eval_seconds mode_delay_eval_seconds\n");

    for(chunk=0; chunk<2; chunk++)
    {
        int i, Nchunk;
        double t0, t1;
        double *Tdense, *Xdirect, *Ydirect, *Zdirect;
        double *Xmode, *Ymode, *Zmode, *Xfmode, *Yfmode, *Zfmode;
        double power_direct, power_mode, dot, diff_power;
        double max_abs_diff, max_abs_direct;
        double match, rel_l2;
        clock_t start_clock, end_clock;
        double direct_seconds, mode_seconds;

        t0 = starts[chunk];
        t1 = t0+duration;
        if(t1 > safe_end) t1 = safe_end;
        Nchunk = (int)floor((t1-t0)/dense_dt)+1;
        if(Nchunk < 2) continue;

        Tdense = double_vector(Nchunk);
        Xdirect = double_vector(Nchunk);
        Ydirect = double_vector(Nchunk);
        Zdirect = double_vector(Nchunk);
        Xmode = double_vector(Nchunk);
        Ymode = double_vector(Nchunk);
        Zmode = double_vector(Nchunk);
        Xfmode = double_vector(Nchunk);
        Yfmode = double_vector(Nchunk);
        Zfmode = double_vector(Nchunk);

        for(i=0; i<Nchunk; i++)
        {
            double t = t0+(double)i*dense_dt;
            if(t > t1) t = t1;
            Tdense[i] = t;
        }

        start_clock = clock();
        direct_response_full_waveform_thm(Tdense, Nchunk, dense_dt, delay_pad,
                                          params, ncarriers, carrier,
                                          SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                                          Aacc, Aspline, Pacc, Pspline,
                                          Xdirect, Ydirect, Zdirect);
        end_clock = clock();
        direct_seconds = ((double)(end_clock-start_clock))/CLOCKS_PER_SEC;

        start_clock = clock();
        fast_response_thm(Tdense, Nchunk, params, ncarriers, carrier,
                          SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                          Aacc, Aspline, Pacc, Pspline,
                          Xmode, Ymode, Zmode, Xfmode, Yfmode, Zfmode);
        end_clock = clock();
        mode_seconds = ((double)(end_clock-start_clock))/CLOCKS_PER_SEC;

        power_direct = 0.0;
        power_mode = 0.0;
        dot = 0.0;
        diff_power = 0.0;
        max_abs_diff = 0.0;
        max_abs_direct = 0.0;

        for(i=0; i<Nchunk; i++)
        {
            double t = Tdense[i];
            double dX = Xdirect[i]-Xmode[i];
            double dY = Ydirect[i]-Ymode[i];
            double dZ = Zdirect[i]-Zmode[i];

            power_direct += Xdirect[i]*Xdirect[i]+Ydirect[i]*Ydirect[i]+Zdirect[i]*Zdirect[i];
            power_mode += Xmode[i]*Xmode[i]+Ymode[i]*Ymode[i]+Zmode[i]*Zmode[i];
            dot += Xdirect[i]*Xmode[i]+Ydirect[i]*Ymode[i]+Zdirect[i]*Zmode[i];
            diff_power += dX*dX+dY*dY+dZ*dZ;

            if(fabs(dX) > max_abs_diff) max_abs_diff = fabs(dX);
            if(fabs(dY) > max_abs_diff) max_abs_diff = fabs(dY);
            if(fabs(dZ) > max_abs_diff) max_abs_diff = fabs(dZ);
            if(fabs(Xdirect[i]) > max_abs_direct) max_abs_direct = fabs(Xdirect[i]);
            if(fabs(Ydirect[i]) > max_abs_direct) max_abs_direct = fabs(Ydirect[i]);
            if(fabs(Zdirect[i]) > max_abs_direct) max_abs_direct = fabs(Zdirect[i]);

            fprintf(data, "%d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    chunk, t,
                    Xdirect[i], Ydirect[i], Zdirect[i],
                    Xmode[i], Ymode[i], Zmode[i],
                    dX, dY, dZ);
        }

        match = 1.0;
        if(power_direct > 0.0 && power_mode > 0.0)
        {
            match = dot/sqrt(power_direct*power_mode);
        }
        rel_l2 = 0.0;
        if(power_direct > 0.0) rel_l2 = sqrt(diff_power/power_direct);

        fprintf(summary, "%d %s %.15e %.15e %.15e %d %.15e %.15e %.15e %.15e %.15e %.15e\n",
                chunk, names[chunk], t0, Tdense[Nchunk-1], dense_dt, Nchunk,
                match, rel_l2, max_abs_diff, max_abs_direct,
                direct_seconds, mode_seconds);
        printf("dense_tdi_direct_check %s samples %d start %.6e end %.6e direct_vs_mode_match %.12f rel_l2 %.6e direct_time %.6f mode_time %.6f\n",
               names[chunk], Nchunk, t0, Tdense[Nchunk-1], match, rel_l2,
               direct_seconds, mode_seconds);

        free_double_vector(Tdense);
        free_double_vector(Xdirect);
        free_double_vector(Ydirect);
        free_double_vector(Zdirect);
        free_double_vector(Xmode);
        free_double_vector(Ymode);
        free_double_vector(Zmode);
        free_double_vector(Xfmode);
        free_double_vector(Yfmode);
        free_double_vector(Zfmode);
    }
    fclose(data);
    fclose(summary);
}

int find_thm_carrier(int ncarriers, const THMFoldedCarrier *carrier, int ell, int abs_emm)
{
    int i;

    if(carrier == NULL) return -1;
    for(i=0; i<ncarriers; i++)
    {
        if(carrier[i].ell == ell && abs(carrier[i].emm) == abs_emm)
        {
            return i;
        }
    }
    return -1;
}

double nonuniform_phase_derivative(int i, int N, const double *t, const double *phase)
{
    double x0, x1, x2;
    double y0, y1, y2;
    double d0, d1, d2;
    double c0, c1, c2;

    if(t == NULL || phase == NULL || N < 2)
    {
        return 0.0;
    }
    if(i <= 0)
    {
        if(t[1] == t[0]) return 0.0;
        return (phase[1]-phase[0])/(t[1]-t[0]);
    }
    if(i >= N-1)
    {
        if(t[N-1] == t[N-2]) return 0.0;
        return (phase[N-1]-phase[N-2])/(t[N-1]-t[N-2]);
    }

    /*
     * The AP grid can tighten by large factors near TDI transfer-function
     * crossings. A simple (phase[i+1]-phase[i-1])/(t[i+1]-t[i-1]) secant then
     * estimates the derivative at the midpoint of the two neighbors, not at
     * t[i], and it produces an artificial jump just as dense sampling starts.
     * Differentiate the local quadratic interpolant instead; this is still a
     * local diagnostic estimate, but it behaves correctly on a nonuniform grid.
     */
    x0 = t[i-1];
    x1 = t[i];
    x2 = t[i+1];
    y0 = phase[i-1];
    y1 = phase[i];
    y2 = phase[i+1];

    d0 = (x0-x1)*(x0-x2);
    d1 = (x1-x0)*(x1-x2);
    d2 = (x2-x0)*(x2-x1);
    if(d0 == 0.0 || d1 == 0.0 || d2 == 0.0)
    {
        if(x2 == x0) return 0.0;
        return (y2-y0)/(x2-x0);
    }

    c0 = (x1-x2)/d0;
    c1 = (2.0*x1-x0-x2)/d1;
    c2 = (x1-x0)/d2;

    return c0*y0+c1*y1+c2*y2;
}

/*
 * Experimental physical endpoint planner.
 *
 * For Phi'' = 2 pi fdot, the stationary-phase time width is proportional to
 * |fdot|^{-1/2}.  The two dimensionless local variation measures used here are
 *
 *   epsilon_f = |fddot|/|fdot|^(3/2),
 *   epsilon_A = |Adot|/(|A| sqrt(|fdot|)).
 *
 * The intrinsic epsilon_f first identifies the physical approach to merger.
 * Once it reaches 80 percent of tolerance, the post-TDI epsilon_f and
 * epsilon_A indicators may advance the handoff.  This gate prevents annual
 * detector modulation from demanding an impractically long endpoint FFT in
 * the early inspiral.  The first order-unity crossing over all folded carriers
 * and channels defines the common handoff.
 *
 * The existing endpoint geometry leaves 0.75*T_rise between the end of the
 * cosine taper and that handoff.  For a locally linear chirp, requiring
 * T_rise*Delta f_gap >= C therefore gives
 *
 *   T_rise >= sqrt(C/0.75) / sqrt(|fdot|).
 *
 * Signed AP amplitudes pass smoothly through response zeros.  A small local
 * envelope floor in epsilon_A avoids dividing by roundoff exactly at a zero
 * while retaining the rapid relative variation that makes the SPA unsafe.
 */
static int thm_build_local_spa_endpoint_plan(int N, const double *response_time,
                                             int nstreams,
                                             const THMFoldedCarrier *carrier,
                                             double **mode_freq,
                                             double ***Achan,
                                             double ***freq_track,
                                             double tc_output,
                                             THMLocalSPAEndpointPlan *plan)
{
    const int Nchan = 3;
    const double taper_to_join_fraction = 0.75;
    const double amplitude_floor_fraction = 2.0e-2;
    const double response_frequency_gate = 0.8;
    double *fdot, *fddot, *post_fdot, *post_fddot, *adot, *metric;
    double earliest_time, max_stationary_width;
    int ch, k, i, j;

    if(plan == NULL) return 0;
    memset(plan, 0, sizeof(*plan));
    if(N < 7 || response_time == NULL || nstreams < 1 ||
       carrier == NULL || mode_freq == NULL ||
       Achan == NULL || freq_track == NULL)
    {
        return 0;
    }

    fdot = double_vector(N);
    fddot = double_vector(N);
    post_fdot = double_vector(N);
    post_fddot = double_vector(N);
    adot = double_vector(N);
    metric = double_vector(N);
    if(fdot == NULL || fddot == NULL || post_fdot == NULL ||
       post_fddot == NULL || adot == NULL || metric == NULL)
    {
        free_double_vector(fdot);
        free_double_vector(fddot);
        free_double_vector(post_fdot);
        free_double_vector(post_fddot);
        free_double_vector(adot);
        free_double_vector(metric);
        return 0;
    }

    earliest_time = HUGE_VAL;
    for(ch=0; ch<Nchan; ch++)
    {
        for(k=0; k<nstreams; k++)
        {
            int mode_index = carrier[k].mode_index;
            if(mode_index < 0) continue;
            for(i=0; i<N; i++)
            {
                fdot[i] = nonuniform_phase_derivative(i, N, response_time,
                                                       mode_freq[mode_index]);
                post_fdot[i] = nonuniform_phase_derivative(i, N,
                                                            response_time,
                                                            freq_track[ch][k]);
            }
            for(i=0; i<N; i++)
            {
                double local_amp, amp_den, ef, ef_post, ea, mf, mr, ma;

                fddot[i] = nonuniform_phase_derivative(i, N, response_time,
                                                        fdot);
                post_fddot[i] = nonuniform_phase_derivative(i, N,
                                                             response_time,
                                                             post_fdot);
                adot[i] = nonuniform_phase_derivative(i, N, response_time,
                                                       Achan[ch][k]);
                metric[i] = 0.0;
                /* Two numerical derivatives plus the three-point metric
                 * smoothing contaminate several samples at either endpoint. */
                if(i < 6 || i > N-7 || response_time[i] >= tc_output ||
                   !isfinite(freq_track[ch][k][i]) ||
                   freq_track[ch][k][i] <= 0.0)
                {
                    continue;
                }

                if(!isfinite(fdot[i]) || fdot[i] <= 0.0)
                {
                    continue;
                }

                local_amp = 0.0;
                for(j=i-2; j<=i+2; j++)
                {
                    double aj = fabs(Achan[ch][k][j]);
                    if(isfinite(aj) && aj > local_amp) local_amp = aj;
                }
                amp_den = fabs(Achan[ch][k][i]);
                if(amp_den < amplitude_floor_fraction*local_amp)
                {
                    amp_den = amplitude_floor_fraction*local_amp;
                }

                ef = fabs(fddot[i])/pow(fdot[i], 1.5);
                ef_post = 0.0;
                if(isfinite(post_fdot[i]) && post_fdot[i] > 0.0 &&
                   isfinite(post_fddot[i]))
                {
                    ef_post = fabs(post_fddot[i])/pow(post_fdot[i], 1.5);
                }
                ea = 0.0;
                if(isfinite(amp_den) && amp_den > 0.0)
                {
                    ea = fabs(adot[i])/(amp_den*sqrt(fdot[i]));
                }
                mf = isfinite(ef) ?
                     ef/thm_endpoint_local_spa_epsilon_f_tolerance : 0.0;
                /*
                 * Annual Doppler/polarization modulation can make the post-TDI
                 * indicators order unity in the early inspiral.  A compact
                 * endpoint FFT cannot cure that regime.  Let response-induced
                 * curvature and amplitude variation advance the handoff only
                 * after the intrinsic chirp reaches 80 percent of its local
                 * frequency-curvature tolerance.
                 */
                mr = (isfinite(ef_post) && mf >= response_frequency_gate) ?
                     ef_post/thm_endpoint_local_spa_epsilon_f_tolerance : 0.0;
                ma = (isfinite(ea) && mf >= response_frequency_gate) ?
                     ea/thm_endpoint_local_spa_epsilon_a_tolerance : 0.0;
                metric[i] = fmax(mf, fmax(mr, ma));
                if(metric[i] > 4.0) metric[i] = 4.0;
            }

            for(i=7; i<N-7; i++)
            {
                double smooth_metric, previous_metric, u, crossing_time;

                smooth_metric = 0.25*metric[i-1]+0.5*metric[i]+0.25*metric[i+1];
                if(smooth_metric < 1.0) continue;
                previous_metric = 0.25*metric[i-2]+0.5*metric[i-1]+
                                  0.25*metric[i];
                u = 1.0;
                if(previous_metric < 1.0 && smooth_metric > previous_metric)
                {
                    u = (1.0-previous_metric)/(smooth_metric-previous_metric);
                    if(u < 0.0) u = 0.0;
                    if(u > 1.0) u = 1.0;
                }
                crossing_time = response_time[i-1]+u*(response_time[i]-response_time[i-1]);
                if(crossing_time < earliest_time)
                {
                    earliest_time = crossing_time;
                    plan->trigger_channel = ch;
                    plan->trigger_stream = k;
                    plan->trigger_frequency =
                        freq_track[ch][k][i-1]+u*(freq_track[ch][k][i]-
                                                  freq_track[ch][k][i-1]);
                    plan->trigger_epsilon_f = fmax(
                        fabs(fddot[i])/pow(fabs(fdot[i]), 1.5),
                        (isfinite(post_fdot[i]) && post_fdot[i] > 0.0) ?
                        fabs(post_fddot[i])/pow(post_fdot[i], 1.5) : 0.0);
                    if(isfinite(Achan[ch][k][i]) && fabs(Achan[ch][k][i]) > 0.0 &&
                       isfinite(fdot[i]) && fdot[i] > 0.0)
                    {
                        double local_amp = 0.0;
                        double amp_den;
                        for(j=i-2; j<=i+2; j++)
                        {
                            double aj = fabs(Achan[ch][k][j]);
                            if(isfinite(aj) && aj > local_amp) local_amp = aj;
                        }
                        amp_den = fabs(Achan[ch][k][i]);
                        if(amp_den < amplitude_floor_fraction*local_amp)
                        {
                            amp_den = amplitude_floor_fraction*local_amp;
                        }
                        plan->trigger_epsilon_a = amp_den > 0.0 ?
                            fabs(adot[i])/(amp_den*sqrt(fdot[i])) : 0.0;
                    }
                    else
                    {
                        plan->trigger_epsilon_a = HUGE_VAL;
                    }
                }
                break;
            }
        }
    }

    if(!isfinite(earliest_time) || earliest_time == HUGE_VAL)
    {
        free_double_vector(fdot);
        free_double_vector(fddot);
        free_double_vector(post_fdot);
        free_double_vector(post_fddot);
        free_double_vector(adot);
        free_double_vector(metric);
        return 0;
    }

    max_stationary_width = 0.0;
    for(ch=0; ch<Nchan; ch++)
    {
        for(k=0; k<nstreams; k++)
        {
            double fdot_join;
            int mode_index = carrier[k].mode_index;
            if(mode_index < 0) continue;

            for(i=0; i<N; i++)
            {
                fdot[i] = nonuniform_phase_derivative(i, N, response_time,
                                                       mode_freq[mode_index]);
            }
            fdot_join = linear_interp_clamped(N, response_time, fdot,
                                               earliest_time);
            if(isfinite(fdot_join) && fdot_join > 0.0)
            {
                double width = 1.0/sqrt(fdot_join);
                if(width > max_stationary_width) max_stationary_width = width;
            }
        }
    }

    if(!isfinite(max_stationary_width) || max_stationary_width <= 0.0)
    {
        free_double_vector(fdot);
        free_double_vector(fddot);
        free_double_vector(post_fdot);
        free_double_vector(post_fddot);
        free_double_vector(adot);
        free_double_vector(metric);
        return 0;
    }

    plan->valid = 1;
    plan->join_time = earliest_time;
    plan->max_stationary_width = max_stationary_width;
    plan->rise = sqrt(thm_endpoint_local_spa_leakage_cycles/
                      taper_to_join_fraction)*max_stationary_width;
    if(plan->rise < 32.0*dt) plan->rise = 32.0*dt;

    free_double_vector(fdot);
    free_double_vector(fddot);
    free_double_vector(post_fdot);
    free_double_vector(post_fddot);
    free_double_vector(adot);
    free_double_vector(metric);
    return 1;
}

void build_nonuniform_phase_derivatives(int N, double *t, gsl_interp_accel *Pacc, gsl_spline *Pspline, double *phase_grid, double *freq_grid, double *fdot_grid)
{
    int i;

    if(t == NULL || Pacc == NULL || Pspline == NULL ||
       phase_grid == NULL || freq_grid == NULL || fdot_grid == NULL ||
       N < 2)
    {
        return;
    }

    /*
     * The SPA needs f = Phi'/(2*pi) and fdot = f'.  Differentiating the phase
     * spline twice is fragile on the detector-adaptive, nonuniform grid: near
     * TDI transfer-function structure the spline can ring and drive fdot close
     * to zero even when the waveform amplitude and phase are smooth.  Use local
     * nonuniform finite derivatives on the sampled phase instead.  The GSL
     * derivative calls below are only last-ditch fallbacks for invalid local
     * estimates.
     */
    for(i=0; i<N; i++)
    {
        phase_grid[i] = gsl_spline_eval(Pspline, t[i], Pacc);
    }
    for(i=0; i<N; i++)
    {
        freq_grid[i] = nonuniform_phase_derivative(i, N, t, phase_grid)/(2.0*M_PI);
        if(!isfinite(freq_grid[i]) || freq_grid[i] <= 0.0)
        {
            freq_grid[i] = gsl_spline_eval_deriv(Pspline, t[i], Pacc)/(2.0*M_PI);
        }
    }
    for(i=0; i<N; i++)
    {
        fdot_grid[i] = nonuniform_phase_derivative(i, N, t, freq_grid);
        if(!isfinite(fdot_grid[i]) || fdot_grid[i] == 0.0)
        {
            fdot_grid[i] = gsl_spline_eval_deriv2(Pspline, t[i], Pacc)/(2.0*M_PI);
        }
    }
}

/*
 * Diagnostic frequency decomposition for the post-TDI SPA.
 *
 * The default path estimates f(t) by differentiating the full post-TDI phase,
 * Phi_TDI = Phi_mode + DeltaPhi_TDI.  Most of Phi_mode is known analytically by
 * the THM model, so this diagnostic keeps that exact model frequency and takes
 * a numerical derivative only of the comparatively small TDI phase correction:
 *
 *     f_split = f_mode(plan_time) d(plan_time)/d(response_time)
 *               + (1/2pi) d DeltaPhi_TDI/d(response_time).
 *
 * The full phase is still used later in the SPA phase itself.  This isolates
 * whether noisy first derivatives of the large carrier phase are responsible
 * for the low-level SPA/FFT ripples.
 */
void build_split_reference_frequency_track(int N, double *response_time, double *plan_time, double *mode_freq, double *phase_correction, double *phase_total, double *freq_ref_track, double *plan_jacobian, double *freq_correction, double *freq_split, double *fdot_split, double *freq_fullphase, double *fdot_fullphase)
{
    const gsl_interp_type *interp_type;
    gsl_interp_accel *Facc;
    gsl_spline *Fspline;
    double tmin, tmax;
    int i;

    if(N < 2 || response_time == NULL || plan_time == NULL ||
       mode_freq == NULL || phase_correction == NULL ||
       phase_total == NULL || freq_ref_track == NULL ||
       plan_jacobian == NULL || freq_correction == NULL ||
       freq_split == NULL || fdot_split == NULL ||
       freq_fullphase == NULL || fdot_fullphase == NULL)
    {
        return;
    }

    interp_type = (N >= 5) ? THM_AP_SPLINE_TYPE : gsl_interp_linear;
    Facc = gsl_interp_accel_alloc();
    Fspline = gsl_spline_alloc(interp_type, N);
    if(Facc == NULL || Fspline == NULL)
    {
        if(Facc != NULL) gsl_interp_accel_free(Facc);
        if(Fspline != NULL) gsl_spline_free(Fspline);
        return;
    }
    gsl_spline_init(Fspline, response_time, mode_freq, N);
    tmin = response_time[0];
    tmax = response_time[N-1];

    for(i=0; i<N; i++)
    {
        double tq, jac, fref, fcorr, fsplit, ffull;

        tq = plan_time[i];
        if(tq < tmin) tq = tmin;
        if(tq > tmax) tq = tmax;
        fref = gsl_spline_eval(Fspline, tq, Facc);
        jac = nonuniform_phase_derivative(i, N, response_time, plan_time);
        if(!isfinite(jac) || fabs(jac) < 1.0e-12) jac = 1.0;
        fcorr = nonuniform_phase_derivative(i, N, response_time, phase_correction)/(2.0*M_PI);
        ffull = nonuniform_phase_derivative(i, N, response_time, phase_total)/(2.0*M_PI);

        if(!isfinite(fref) || fref <= 0.0) fref = fabs(ffull);
        if(!isfinite(fcorr)) fcorr = 0.0;
        if(!isfinite(ffull)) ffull = fref*jac+fcorr;

        fsplit = fref*jac+fcorr;
        if(!isfinite(fsplit) || fsplit <= 0.0) fsplit = fabs(ffull);
        if(!isfinite(fsplit) || fsplit <= 0.0) fsplit = fabs(fref);

        freq_ref_track[i] = fref*jac;
        plan_jacobian[i] = jac;
        freq_correction[i] = fcorr;
        freq_split[i] = fabs(fsplit);
        freq_fullphase[i] = fabs(ffull);
    }

    for(i=0; i<N; i++)
    {
        fdot_split[i] = nonuniform_phase_derivative(i, N, response_time, freq_split);
        fdot_fullphase[i] = nonuniform_phase_derivative(i, N, response_time, freq_fullphase);
    }

    gsl_spline_free(Fspline);
    gsl_interp_accel_free(Facc);
}

/*
 * Direct 22 carrier used by the diagnostic split-carrier SPA below.  The
 * stripped THM driver sets phi22(tc)=params[4], so the analytic PhenomT phase
 * needs the same constant shift as the integrated AP phase grid.
 */
static double thm_direct_22_phase_seconds(const IMRPhenomTHM *model,
                                          const double *params,
                                          double source_time)
{
    double tau, phase0;

    if(model == NULL || params == NULL || !isfinite(source_time) ||
       !isfinite(model->Mtot) || model->Mtot <= 0.0)
    {
        return NAN;
    }

    tau = (source_time-params[5])/model->Mtot;
    phase0 = IMRPhenomTPhase22(0.0, model->eta,
                               (struct IMRPhenomT *)&model->mode22);
    return IMRPhenomTPhase22(tau, model->eta,
                             (struct IMRPhenomT *)&model->mode22) +
           params[4]-phase0;
}

static double thm_direct_mode_frequency_hz(const IMRPhenomTHM *model,
                                           int mode_index,
                                           const double *params,
                                           double source_time)
{
    double tau, omega;

    if(model == NULL || params == NULL || mode_index < 0 ||
       mode_index >= model->nmodes || !isfinite(source_time) ||
       !isfinite(model->Mtot) || model->Mtot <= 0.0)
    {
        return NAN;
    }

    tau = (source_time-params[5])/model->Mtot;
    omega = IMRPhenomTHMModeFrequency(model, mode_index, tau);
    return omega/(2.0*M_PI*model->Mtot);
}

/*
 * Differentiate a smooth interpolation without asking GSL for a spline
 * derivative.  Centered five-point stencils are used in the interior and
 * one-sided five-point stencils at the observation boundaries.  This helper
 * is only applied to the slowly varying detector-time map and TDI residual,
 * never to the rapidly accumulating carrier phase.
 */
static int thm_spline_five_point_derivatives(gsl_interp_accel *acc,
                                             gsl_spline *spline,
                                             double x,
                                             double xmin,
                                             double xmax,
                                             double h_hint,
                                             double *d1,
                                             double *d2)
{
    double h, y0, y1, y2, y3, y4;

    if(acc == NULL || spline == NULL || d1 == NULL || d2 == NULL ||
       !isfinite(x) || !isfinite(xmin) || !isfinite(xmax) || xmax <= xmin)
    {
        return 0;
    }

    h = fabs(h_hint);
    if(!isfinite(h) || h <= 0.0) h = 1.0;
    if(h > 0.125*(xmax-xmin)) h = 0.125*(xmax-xmin);
    if(h <= 0.0) return 0;

    if(x-2.0*h >= xmin && x+2.0*h <= xmax)
    {
        y0 = gsl_spline_eval(spline, x-2.0*h, acc);
        y1 = gsl_spline_eval(spline, x-h, acc);
        y2 = gsl_spline_eval(spline, x, acc);
        y3 = gsl_spline_eval(spline, x+h, acc);
        y4 = gsl_spline_eval(spline, x+2.0*h, acc);
        *d1 = (y0-8.0*y1+8.0*y3-y4)/(12.0*h);
        *d2 = (-y0+16.0*y1-30.0*y2+16.0*y3-y4)/(12.0*h*h);
    }
    else if(x+4.0*h <= xmax)
    {
        y0 = gsl_spline_eval(spline, x, acc);
        y1 = gsl_spline_eval(spline, x+h, acc);
        y2 = gsl_spline_eval(spline, x+2.0*h, acc);
        y3 = gsl_spline_eval(spline, x+3.0*h, acc);
        y4 = gsl_spline_eval(spline, x+4.0*h, acc);
        *d1 = (-25.0*y0+48.0*y1-36.0*y2+16.0*y3-3.0*y4)/(12.0*h);
        *d2 = (35.0*y0-104.0*y1+114.0*y2-56.0*y3+11.0*y4)/(12.0*h*h);
    }
    else if(x-4.0*h >= xmin)
    {
        y0 = gsl_spline_eval(spline, x, acc);
        y1 = gsl_spline_eval(spline, x-h, acc);
        y2 = gsl_spline_eval(spline, x-2.0*h, acc);
        y3 = gsl_spline_eval(spline, x-3.0*h, acc);
        y4 = gsl_spline_eval(spline, x-4.0*h, acc);
        *d1 = (25.0*y0-48.0*y1+36.0*y2-16.0*y3+3.0*y4)/(12.0*h);
        *d2 = (35.0*y0-104.0*y1+114.0*y2-56.0*y3+11.0*y4)/(12.0*h*h);
    }
    else
    {
        return 0;
    }

    return isfinite(*d1) && isfinite(*d2);
}

/*
 * Construct the genuinely split 22 SPA ingredients.
 *
 * Let t label the post-TDI response samples and u(t) be the detector/reference
 * time at which extractAP() removed the carrier.  At the AP nodes we write
 *
 *   Phi_TDI(t) = Phi_22^direct(u(t)) + R(t).
 *
 * R includes both the physical TDI phase correction and the tiny difference
 * between the old interpolated carrier and the direct analytic carrier.  This
 * makes the split reconstruct the sampled phase exactly at every AP node.  The
 * frequency and chirp rate then follow from
 *
 *   f = f22(u) u' + R'/(2 pi),
 *   fdot = (df22/du) u'^2 + f22(u) u'' + R''/(2 pi).
 *
 * Only df22/du requires new waveform evaluations.  They are inexpensive
 * intrinsic frequency calls on a local five-point stencil; no extra TDI
 * responses are evaluated.
 */
static int build_direct22_residual_spa_tracks(int N,
                                              const double *response_time,
                                              const double *plan_time,
                                              const double *phase_reference,
                                              const double *phase_correction,
                                              const double *phase_total,
                                              const double *params,
                                              const IMRPhenomTHM *model,
                                              int mode_index,
                                              double *carrier_phase,
                                              double *residual_phase,
                                              double *carrier_frequency,
                                              double *carrier_dfdu,
                                              double *plan_d1,
                                              double *plan_d2,
                                              double *residual_d1,
                                              double *residual_d2,
                                              double *freq_split,
                                              double *fdot_split)
{
    const gsl_interp_type *interp_type;
    gsl_interp_accel *Uacc, *Racc;
    gsl_spline *Uspline, *Rspline;
    double tmin, tmax;
    int i, ok;

    if(N < 5 || response_time == NULL || plan_time == NULL ||
       phase_reference == NULL || phase_correction == NULL ||
       phase_total == NULL || params == NULL || model == NULL ||
       mode_index < 0 || mode_index >= model->nmodes ||
       carrier_phase == NULL || residual_phase == NULL ||
       carrier_frequency == NULL || carrier_dfdu == NULL ||
       plan_d1 == NULL || plan_d2 == NULL || residual_d1 == NULL ||
       residual_d2 == NULL || freq_split == NULL || fdot_split == NULL)
    {
        return 0;
    }
    if(model->modes[mode_index].ell != 2 ||
       model->modes[mode_index].abs_emm != 2)
    {
        return 0;
    }

    interp_type = (N >= 5) ? THM_AP_SPLINE_TYPE : gsl_interp_linear;
    Uacc = gsl_interp_accel_alloc();
    Racc = gsl_interp_accel_alloc();
    Uspline = gsl_spline_alloc(interp_type, N);
    Rspline = gsl_spline_alloc(interp_type, N);
    if(Uacc == NULL || Racc == NULL || Uspline == NULL || Rspline == NULL)
    {
        if(Uacc != NULL) gsl_interp_accel_free(Uacc);
        if(Racc != NULL) gsl_interp_accel_free(Racc);
        if(Uspline != NULL) gsl_spline_free(Uspline);
        if(Rspline != NULL) gsl_spline_free(Rspline);
        return 0;
    }

    tmin = response_time[0];
    tmax = response_time[N-1];
    for(i=0; i<N; i++)
    {
        carrier_phase[i] = thm_direct_22_phase_seconds(model, params,
                                                       plan_time[i]);
        /* Preserve the exact AP-node total phase while changing reference. */
        residual_phase[i] = phase_correction[i] + phase_reference[i] -
                            carrier_phase[i];
        if(!isfinite(residual_phase[i]))
        {
            residual_phase[i] = phase_total[i]-carrier_phase[i];
        }
    }
    unwrap(N, residual_phase);
    gsl_spline_init(Uspline, response_time, plan_time, N);
    gsl_spline_init(Rspline, response_time, residual_phase, N);

    ok = 1;
    for(i=0; i<N; i++)
    {
        double left, right, local_step, h, hc;
        double fm2, fm1, fp1, fp2;

        left = (i > 0) ? response_time[i]-response_time[i-1] :
                         response_time[1]-response_time[0];
        right = (i < N-1) ? response_time[i+1]-response_time[i] :
                            response_time[N-1]-response_time[N-2];
        local_step = fmin(fabs(left), fabs(right));
        h = 0.1*local_step;
        if(h > 100.0) h = 100.0;
        if(h < 1.0) h = 1.0;

        if(!thm_spline_five_point_derivatives(Uacc, Uspline,
                                              response_time[i], tmin, tmax, h,
                                              &plan_d1[i], &plan_d2[i]))
        {
            plan_d1[i] = nonuniform_phase_derivative(i, N, response_time,
                                                     plan_time);
            plan_d2[i] = NAN;
        }
        if(!thm_spline_five_point_derivatives(Racc, Rspline,
                                              response_time[i], tmin, tmax, h,
                                              &residual_d1[i], &residual_d2[i]))
        {
            residual_d1[i] = nonuniform_phase_derivative(i, N, response_time,
                                                         residual_phase);
            residual_d2[i] = NAN;
        }

        carrier_frequency[i] = thm_direct_mode_frequency_hz(model, mode_index,
                                                             params,
                                                             plan_time[i]);
        hc = h*fabs(plan_d1[i]);
        if(!isfinite(hc) || hc < 1.0) hc = 1.0;
        if(hc > 100.0) hc = 100.0;
        fm2 = thm_direct_mode_frequency_hz(model, mode_index, params,
                                           plan_time[i]-2.0*hc);
        fm1 = thm_direct_mode_frequency_hz(model, mode_index, params,
                                           plan_time[i]-hc);
        fp1 = thm_direct_mode_frequency_hz(model, mode_index, params,
                                           plan_time[i]+hc);
        fp2 = thm_direct_mode_frequency_hz(model, mode_index, params,
                                           plan_time[i]+2.0*hc);
        carrier_dfdu[i] = (fm2-8.0*fm1+8.0*fp1-fp2)/(12.0*hc);

        freq_split[i] = carrier_frequency[i]*plan_d1[i] +
                        residual_d1[i]/(2.0*M_PI);
        fdot_split[i] = carrier_dfdu[i]*plan_d1[i]*plan_d1[i] +
                        carrier_frequency[i]*plan_d2[i] +
                        residual_d2[i]/(2.0*M_PI);
        if(!isfinite(freq_split[i]) || !isfinite(fdot_split[i])) ok = 0;
    }

    gsl_spline_free(Uspline);
    gsl_spline_free(Rspline);
    gsl_interp_accel_free(Uacc);
    gsl_interp_accel_free(Racc);
    return ok;
}

/*
 * Add the leading-order split-carrier SPA directly on the caller's uniform
 * Fourier grid.  This avoids reconstructing the rapidly accumulating carrier
 * phase with a phase spline: only the detector-time map and the small TDI
 * residual are interpolated.  The stationary time is found by linearly
 * inverting the locally monotone split frequency track, matching the dense
 * diagnostic used to validate this construction.
 */
static int accumulate_direct22_split_spa_fourier(
        int Ns,
        const double *response_time,
        const double *plan_time,
        const double *residual_phase,
        const double *freq_grid,
        const double *fdot_grid,
        int last_index,
        double Tobs,
        gsl_interp_accel *ASacc,
        gsl_spline *ASspline,
        const double *params,
        const IMRPhenomTHM *model,
        double fourier_df,
        int fourier_n,
        double *out_re,
        double *out_im)
{
    const gsl_interp_type *interp_type;
    gsl_interp_accel *Uacc, *Racc;
    gsl_spline *Uspline, *Rspline;
    int fq, j, used;

    if(Ns < 5 || response_time == NULL || plan_time == NULL ||
       residual_phase == NULL || freq_grid == NULL || fdot_grid == NULL ||
       ASacc == NULL || ASspline == NULL || params == NULL || model == NULL ||
       !isfinite(fourier_df) || fourier_df <= 0.0 || fourier_n < 2 ||
       out_re == NULL || out_im == NULL)
    {
        return 0;
    }
    if(last_index > Ns-1) last_index = Ns-1;
    if(last_index < 1) return 0;

    interp_type = (Ns >= 5) ? THM_AP_SPLINE_TYPE : gsl_interp_linear;
    Uacc = gsl_interp_accel_alloc();
    Racc = gsl_interp_accel_alloc();
    Uspline = gsl_spline_alloc(interp_type, Ns);
    Rspline = gsl_spline_alloc(interp_type, Ns);
    if(Uacc == NULL || Racc == NULL || Uspline == NULL || Rspline == NULL)
    {
        if(Uacc != NULL) gsl_interp_accel_free(Uacc);
        if(Racc != NULL) gsl_interp_accel_free(Racc);
        if(Uspline != NULL) gsl_spline_free(Uspline);
        if(Rspline != NULL) gsl_spline_free(Rspline);
        return 0;
    }
    gsl_spline_init(Uspline, response_time, plan_time, Ns);
    gsl_spline_init(Rspline, response_time, residual_phase, Ns);

    j = 0;
    used = 0;
    for(fq=1; fq<fourier_n; fq++)
    {
        double f, f0, f1, frac, t, u, fdot, A, p;

        f = (double)fq*fourier_df;
        while(j < last_index-1 &&
              (!isfinite(freq_grid[j]) || !isfinite(freq_grid[j+1]) ||
               freq_grid[j+1] <= freq_grid[j] || freq_grid[j+1] < f))
        {
            j++;
        }
        if(j >= last_index || !isfinite(freq_grid[j]) ||
           !isfinite(freq_grid[j+1]) || freq_grid[j+1] <= freq_grid[j] ||
           f < freq_grid[j] || f > freq_grid[j+1])
        {
            continue;
        }

        f0 = freq_grid[j];
        f1 = freq_grid[j+1];
        frac = (f-f0)/(f1-f0);
        if(frac < 0.0) frac = 0.0;
        if(frac > 1.0) frac = 1.0;
        t = response_time[j]+frac*(response_time[j+1]-response_time[j]);
        fdot = fdot_grid[j]+frac*(fdot_grid[j+1]-fdot_grid[j]);
        if(!isfinite(fdot) || fdot <= 0.0) continue;

        u = gsl_spline_eval(Uspline, t, Uacc);
        A = gsl_spline_eval(ASspline, t, ASacc)/sqrt(fdot);
        p = thm_direct_22_phase_seconds(model, params, u) +
            gsl_spline_eval(Rspline, t, Racc) -
            2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        if(!isfinite(A) || !isfinite(p)) continue;
        out_re[fq] += A*cos(p);
        out_im[fq] += A*sin(p);
        used++;
    }

    gsl_spline_free(Uspline);
    gsl_spline_free(Rspline);
    gsl_interp_accel_free(Uacc);
    gsl_interp_accel_free(Racc);
    return used;
}

static void build_direct22_residual_fft_spectrum(double *hfft,
                                                 int N,
                                                 double tmax,
                                                 double tukey_alpha,
                                                 int Ns,
                                                 const double *response_time,
                                                 const double *plan_time,
                                                 const double *residual_phase,
                                                 gsl_interp_accel *ASacc,
                                                 gsl_spline *ASspline,
                                                 const double *params,
                                                 const IMRPhenomTHM *model)
{
    const gsl_interp_type *interp_type;
    gsl_interp_accel *Uacc, *Racc;
    gsl_spline *Uspline, *Rspline;
    double tmin, tstop;
    int i;

    if(hfft == NULL || N < 2 || Ns < 5 || response_time == NULL ||
       plan_time == NULL || residual_phase == NULL || ASacc == NULL ||
       ASspline == NULL || params == NULL || model == NULL)
    {
        return;
    }

    interp_type = (Ns >= 5) ? THM_AP_SPLINE_TYPE : gsl_interp_linear;
    Uacc = gsl_interp_accel_alloc();
    Racc = gsl_interp_accel_alloc();
    Uspline = gsl_spline_alloc(interp_type, Ns);
    Rspline = gsl_spline_alloc(interp_type, Ns);
    if(Uacc == NULL || Racc == NULL || Uspline == NULL || Rspline == NULL)
    {
        if(Uacc != NULL) gsl_interp_accel_free(Uacc);
        if(Racc != NULL) gsl_interp_accel_free(Racc);
        if(Uspline != NULL) gsl_spline_free(Uspline);
        if(Rspline != NULL) gsl_spline_free(Rspline);
        return;
    }
    gsl_spline_init(Uspline, response_time, plan_time, Ns);
    gsl_spline_init(Rspline, response_time, residual_phase, Ns);
    tmin = response_time[0];
    tstop = fmin(tmax, response_time[Ns-1]);

    for(i=0; i<N; i++)
    {
        double t = (double)i*dt;
        hfft[i] = 0.0;
        if(t >= tmin && t < tstop)
        {
            double u = gsl_spline_eval(Uspline, t, Uacc);
            double r = gsl_spline_eval(Rspline, t, Racc);
            double p = thm_direct_22_phase_seconds(model, params, u)+r;
            hfft[i] = gsl_spline_eval(ASspline, t, ASacc)*cos(p);
        }
    }
    if(tukey_alpha > 0.0) tukey(hfft, tukey_alpha, N);
    gsl_fft_real_radix2_transform(hfft, 1, N);
    for(i=0; i<N; i++) hfft[i] *= 2.0*dt;

    gsl_spline_free(Uspline);
    gsl_spline_free(Rspline);
    gsl_interp_accel_free(Uacc);
    gsl_interp_accel_free(Racc);
}

/*
 * Dense Fourier diagnostic for the complete split-carrier construction.  Two
 * references are shown side by side: the existing FFT of the full AP phase
 * spline and an FFT reconstructed as direct carrier plus residual spline.
 * The latter is the internally consistent reference for this experiment.
 */
static void write_dense_direct22_residual_spa_fft_diagnostic(
        const char *filename,
        const double *hfft_full_phase,
        const double *hfft_split_reconstruction,
        int N,
        double Tobs,
        int Ns,
        const double *response_time,
        const double *plan_time,
        const double *residual_phase,
        gsl_interp_accel *ASacc,
        gsl_spline *ASspline,
        const double *params,
        const IMRPhenomTHM *model,
        const double *freq_grid,
        const double *fdot_grid,
        double fmin,
        double fmax,
        double df,
        double spa_tukey_alpha)
{
    const gsl_interp_type *interp_type;
    gsl_interp_accel *Uacc, *Racc;
    gsl_spline *Uspline, *Rspline;
    FILE *out;
    int j, nused;
    double f;
    double spa_power, full_power, split_power, cross_full, cross_split;

    if(filename == NULL || hfft_full_phase == NULL ||
       hfft_split_reconstruction == NULL || N < 2 || Tobs <= 0.0 ||
       Ns < 5 || response_time == NULL || plan_time == NULL ||
       residual_phase == NULL || ASacc == NULL || ASspline == NULL ||
       params == NULL || model == NULL || freq_grid == NULL ||
       fdot_grid == NULL || fmax <= fmin)
    {
        return;
    }
    if(df <= 0.0) df = 1.0/Tobs;

    interp_type = (Ns >= 5) ? THM_AP_SPLINE_TYPE : gsl_interp_linear;
    Uacc = gsl_interp_accel_alloc();
    Racc = gsl_interp_accel_alloc();
    Uspline = gsl_spline_alloc(interp_type, Ns);
    Rspline = gsl_spline_alloc(interp_type, Ns);
    if(Uacc == NULL || Racc == NULL || Uspline == NULL || Rspline == NULL)
    {
        if(Uacc != NULL) gsl_interp_accel_free(Uacc);
        if(Racc != NULL) gsl_interp_accel_free(Racc);
        if(Uspline != NULL) gsl_spline_free(Uspline);
        if(Rspline != NULL) gsl_spline_free(Rspline);
        return;
    }
    gsl_spline_init(Uspline, response_time, plan_time, Ns);
    gsl_spline_init(Rspline, response_time, residual_phase, Ns);

    out = fopen(filename, "w");
    if(out == NULL)
    {
        gsl_spline_free(Uspline);
        gsl_spline_free(Rspline);
        gsl_interp_accel_free(Uacc);
        gsl_interp_accel_free(Racc);
        return;
    }
    fprintf(out, "# f_Hz t_response_s t_plan_s fdot_Hz_per_s valid A_SPA_signed phi_SPA A_FFT_fullphase phi_FFT_fullphase_aligned phase_SPA_minus_fullphase amp_SPA_over_fullphase A_FFT_splitrecon phi_FFT_splitrecon_aligned phase_SPA_minus_splitrecon amp_SPA_over_splitrecon\n");
    fprintf(out, "# SPA phase uses Phi22_direct(plan_time)+residual(response_time)-2*pi*f*(response_time-Tobs)+pi/4; the rapidly varying full phase is never splined.\n");
    fprintf(out, "# fullphase FFT uses the existing spline of Phi_TDI; splitrecon FFT uses the same direct carrier plus residual representation as the SPA.\n");
    fprintf(out, "# carrier df/du uses five direct intrinsic frequency calls per AP node; no extra TDI response calls are made.\n");

    j = 0;
    nused = 0;
    spa_power = 0.0;
    full_power = 0.0;
    split_power = 0.0;
    cross_full = 0.0;
    cross_split = 0.0;
    for(f=fmin; f <= fmax+0.5*df; f += df)
    {
        double f0, f1, frac, t, u, fdot, A, win, phase, Aspa;
        double re_spa, im_spa;
        double re_full, im_full, amp_full, phi_full, phi_full_aligned;
        double re_split, im_split, amp_split, phi_split, phi_split_aligned;
        double phase_eff;
        int valid;

        while(j < Ns-2 &&
              (!isfinite(freq_grid[j]) || !isfinite(freq_grid[j+1]) ||
               freq_grid[j+1] <= freq_grid[j] || freq_grid[j+1] < f))
        {
            j++;
        }
        if(j >= Ns-1 || !isfinite(freq_grid[j]) ||
           !isfinite(freq_grid[j+1]) || freq_grid[j+1] <= freq_grid[j] ||
           f < freq_grid[j] || f > freq_grid[j+1])
        {
            continue;
        }

        f0 = freq_grid[j];
        f1 = freq_grid[j+1];
        frac = (f-f0)/(f1-f0);
        if(frac < 0.0) frac = 0.0;
        if(frac > 1.0) frac = 1.0;
        t = response_time[j]+frac*(response_time[j+1]-response_time[j]);
        fdot = fdot_grid[j]+frac*(fdot_grid[j+1]-fdot_grid[j]);
        valid = isfinite(fdot) && fdot > 0.0;
        if(!isfinite(fdot) || fdot == 0.0) continue;

        u = gsl_spline_eval(Uspline, t, Uacc);
        phase = thm_direct_22_phase_seconds(model, params, u) +
                gsl_spline_eval(Rspline, t, Racc) -
                2.0*M_PI*f*(t-Tobs) + M_PI/4.0;
        A = gsl_spline_eval(ASspline, t, ASacc);
        win = tukey_weight_at_time(t, spa_tukey_alpha, N);
        Aspa = win*A/sqrt(fabs(fdot));
        re_spa = Aspa*cos(phase);
        im_spa = Aspa*sin(phase);
        phase_eff = phase + ((Aspa < 0.0) ? M_PI : 0.0);

        sample_direct_fft_spectrum((double *)hfft_full_phase, N, Tobs, f,
                                   &re_full, &im_full);
        sample_direct_fft_spectrum((double *)hfft_split_reconstruction, N,
                                   Tobs, f, &re_split, &im_split);
        amp_full = hypot(re_full, im_full);
        amp_split = hypot(re_split, im_split);
        phi_full = atan2(im_full, re_full);
        phi_split = atan2(im_split, re_split);
        phi_full_aligned = phi_full +
            2.0*M_PI*rint((phase_eff-phi_full)/(2.0*M_PI));
        phi_split_aligned = phi_split +
            2.0*M_PI*rint((phase_eff-phi_split)/(2.0*M_PI));

        fprintf(out, "%.15e %.15e %.15e %.15e %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                f, t, u, fdot, valid, Aspa, phase,
                amp_full, phi_full_aligned, phase_eff-phi_full_aligned,
                (amp_full > 0.0) ? fabs(Aspa)/amp_full : NAN,
                amp_split, phi_split_aligned, phase_eff-phi_split_aligned,
                (amp_split > 0.0) ? fabs(Aspa)/amp_split : NAN);

        spa_power += re_spa*re_spa+im_spa*im_spa;
        full_power += re_full*re_full+im_full*im_full;
        split_power += re_split*re_split+im_split*im_split;
        cross_full += re_spa*re_full+im_spa*im_full;
        cross_split += re_spa*re_split+im_spa*im_split;
        nused++;
    }
    fclose(out);

    if(nused > 0 && spa_power > 0.0 && full_power > 0.0 && split_power > 0.0)
    {
        double match_full = cross_full/sqrt(spa_power*full_power);
        double match_split = cross_split/sqrt(spa_power*split_power);
        printf("directcarrier_split_spa_fourier_match file %s bins %d fullphase %.15e mismatch %.15e splitrecon %.15e mismatch %.15e\n",
               filename, nused, match_full, 1.0-match_full,
               match_split, 1.0-match_split);
    }

    gsl_spline_free(Uspline);
    gsl_spline_free(Rspline);
    gsl_interp_accel_free(Uacc);
    gsl_interp_accel_free(Racc);
}

static void thm_write_fourier_posttdi_ap_grid_diagnostic(
    const char *prefix,
    const double *params,
    int Ns,
    int nstreams,
    const THMFoldedCarrier *stream_carrier,
    const double *response_time,
    double ***Achan,
    double ***phase_td,
    double ***freq_track,
    gsl_interp_accel **ATacc,
    gsl_spline **ATspline,
    gsl_interp_accel **PTacc,
    gsl_spline **PTspline)
{
    const double dense_dt = 1000.0;
    const char *label[3] = {"X", "Y", "Z"};
    char path[640];
    FILE *out;
    int ch, i, stream;

    if(prefix == NULL || prefix[0] == '\0' || Ns < 4 || nstreams < 1)
    {
        return;
    }
    stream = -1;
    for(i=0; i<nstreams; i++)
    {
        if(stream_carrier[i].ell == 2 && abs(stream_carrier[i].emm) == 2)
        {
            stream = i;
            break;
        }
    }
    if(stream < 0) return;

    snprintf(path, sizeof(path), "%s_posttdi_knots.dat", prefix);
    out = fopen(path, "w");
    if(out != NULL)
    {
        fprintf(out, "# Final post-TDI 22 AP spline knots for X/Y/Z.\n");
        fprintf(out, "# columns: i ssb_output_time_s output_time_minus_numeric_tc_s dt_output_s");
        for(ch=0; ch<3; ch++)
        {
            fprintf(out, " A_%s phase_%s_rad frequency_%s_Hz",
                    label[ch], label[ch], label[ch]);
        }
        fprintf(out, "\n");
        for(i=0; i<Ns; i++)
        {
            double dtr = i > 0 ? response_time[i]-response_time[i-1] : 0.0;

            fprintf(out, "%d %.15e %.15e %.15e",
                    i, response_time[i], response_time[i]-params[5], dtr);
            for(ch=0; ch<3; ch++)
            {
                fprintf(out, " %.15e %.15e %.15e",
                        Achan[ch][stream][i], phase_td[ch][stream][i],
                        freq_track[ch][stream][i]);
            }
            fprintf(out, "\n");
        }
        fclose(out);
    }

    snprintf(path, sizeof(path), "%s_posttdi_dense.dat", prefix);
    out = fopen(path, "w");
    if(out != NULL)
    {
        int ndense = (int)floor(
            (response_time[Ns-1]-response_time[0])/dense_dt)+1;

        fprintf(out, "# Post-TDI 22 AP splines evaluated every %.15e s.\n",
                dense_dt);
        fprintf(out, "# columns: response_time_s response_time_minus_tc_s");
        for(ch=0; ch<3; ch++)
        {
            fprintf(out, " A_%s phase_%s_rad", label[ch], label[ch]);
        }
        fprintf(out, "\n");
        for(i=0; i<ndense; i++)
        {
            double t = response_time[0]+dense_dt*(double)i;

            fprintf(out, "%.15e %.15e", t, t-params[5]);
            for(ch=0; ch<3; ch++)
            {
                int idx = ch*nstreams+stream;
                double amp = gsl_spline_eval(ATspline[idx], t, ATacc[idx]);
                double phase = gsl_spline_eval(PTspline[idx], t, PTacc[idx]);

                fprintf(out, " %.15e %.15e", amp, phase);
            }
            fprintf(out, "\n");
        }
        fclose(out);
    }

    snprintf(path, sizeof(path), "%s_posttdi_merger_dense.dat", prefix);
    out = fopen(path, "w");
    if(out != NULL)
    {
        const double zoom_dt = 1.0;
        double zoom_start = fmax(response_time[0], params[5]-2000.0);
        double zoom_stop = fmin(response_time[Ns-1], params[5]+2000.0);
        int ndense = (int)floor((zoom_stop-zoom_start)/zoom_dt)+1;

        fprintf(out, "# Post-TDI 22 AP merger zoom evaluated every %.15e s.\n",
                zoom_dt);
        fprintf(out, "# columns: response_time_s response_time_minus_tc_s");
        for(ch=0; ch<3; ch++)
        {
            fprintf(out, " A_%s phase_%s_rad", label[ch], label[ch]);
        }
        fprintf(out, "\n");
        for(i=0; i<ndense; i++)
        {
            double t = zoom_start+zoom_dt*(double)i;

            fprintf(out, "%.15e %.15e", t, t-params[5]);
            for(ch=0; ch<3; ch++)
            {
                int idx = ch*nstreams+stream;
                double amp = gsl_spline_eval(ATspline[idx], t, ATacc[idx]);
                double phase = gsl_spline_eval(PTspline[idx], t, PTacc[idx]);

                fprintf(out, " %.15e %.15e", amp, phase);
            }
            fprintf(out, "\n");
        }
        fclose(out);
    }
}

void generate_thm_all_carrier_wdm_combined_fft(int Ns, double *response_time, double *plan_time, double **mode_freq, double *params, const IMRPhenomTHM *model, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, struct wdmshape *wdms, int use_join_override, double join_time_override, double rise_override, int use_spline_endpoint, int use_partition_endpoint, int use_split_fft, int use_split_early_fft, int use_split_tdi_response, THMSparseWDMTriplet *out_tracks, THMFourierTriplet *out_fourier, double fourier_df, double fourier_fmax, int write_sparse_files, int write_time_domain_files, THMReusableWDMWork *reusable_work, const THMObservationContext *observation_context, THMWorkerWorkspace *worker_workspace)
{
    const int Nchan = 3;
    const char *chan_label[3] = {"X", "Y", "Z"};
    THMFoldedCarrier ap_carrier;
    THMFoldedCarrier *stream_carrier;
    THMLocalSPAEndpointPlan local_spa_plan;
    const IMRPhenomTHMModeState *mode_state;
    int N, Ntsmax, Nts, Nshortfft, Nspa_all, Nspa, Nspa1p_all, Nspa1m_all, Nspa1p, Nspa1m;
    int i, k, s, m, p, ch, idx, fq, fourier_n;
    int mode_index, driver_index, driver_channel;
    int stream_capacity, nstreams, npieces, piece_index, response_piece;
    int *stream_parent, *stream_piece;
    int *nmid, *nsize, *nmid_endpoint, *nsize_endpoint;
    int *listn, *listm;
    int Np, final_pixels;
    int endpoint_layers, endpoint_volume, endpoint_mlo, endpoint_mmax;
    int endpoint_plan_layers, endpoint_plan_volume;
    int endpoint_direct_layers, endpoint_direct_pixels;
    int split_fft_layers, split_fft_pixels;
    double Mtot, Mc, tc, Tobs;
    double t_transition, fring, fdamp;
    double fjoin, f_endpoint_start, f_endpoint_max, f_late_start, driver_score, score;
    double fout, fourier_stop, fourier_weight, endpoint_re, endpoint_im;
    double blend_half_width;
    double tail_time, tail_amp;
    double alpha, t, A, hval, wlate, wearly;
    double tc_output, endpoint_target_flat;
    double setup_driver[7], setup_spa[7];
    double f_replace_start[3], f_replace_pair_start[3], f_replace_fallback[3];
    double f_clean_endpoint_start[3], f_clean_endpoint_taper_start[3];
    double f_desired_endpoint_start[3], f_blend_lower[3];
    double channel_ap_time, carrier_tdi_time;
    double spa_freq_time[3], spa_wdm_time[3], endpoint_build_time[3], endpoint_wdm_time[3], split_fft_time[3];
    double total_spa_freq, total_spa_wdm, total_endpoint_build, total_endpoint_wdm, total_split_fft;
    double ***Achan, ***phase_td, ***freq_track, ***setup_carrier;
    double *Xmode, *Ymode, *Zmode, *Xfmode, *Yfmode, *Zfmode;
    double *phi_ref, *phi_res, *omega_plan;
    double *freq, *phase, *Amp, *Ataper;
    double *short_hsum, *short_tmp;
    double *direct_endpoint_time;
    double **direct_endpoint_wave, **direct_endpoint_quadrature;
    double *direct_full_reference_y, *direct_full_reference_z;
    double *wdmwave, *hfull;
    double **wdm, **wdm_split;
    double *Mchan[3], *Mfchan[3];
    gsl_interp_accel **ATacc, **PTacc;
    gsl_spline **ATspline, **PTspline;
    gsl_interp_accel *AFacc, *PFacc, *TAacc;
    gsl_spline *AFspline, *PFspline, *TAspline;
    FILE *out;
    clock_t start, end;
    char filename[160];
    int use_clean_endpoint, use_blend_endpoint, use_direct22_split_spa;
    int use_direct_endpoint_tdi, direct_endpoint_samples;
    int *direct22_split_valid;
    double ***direct22_residual, ***direct22_frequency, ***direct22_fdot;
    double *direct22_carrier_phase, *direct22_carrier_frequency;
    double *direct22_carrier_dfdu, *direct22_plan_d1, *direct22_plan_d2;
    double *direct22_residual_d1, *direct22_residual_d2;
    int use_local_spa_endpoint_plan;
    int owns_wdm_work;
    int direct_full_reference_ready;
    int endpoint_debug, carrier_pixel_debug;
    int debug_nlo, debug_nhi, debug_mlo, debug_mhi;
    const char *debug_env;

    if(Ns < 4 || ncarriers < 1 || carrier == NULL || model == NULL ||
       wdms == NULL)
    {
        fprintf(stderr, "Warning: skipping combined THM WDM path because the inputs are invalid.\n");
        return;
    }
    if(out_fourier != NULL && (!isfinite(fourier_df) || fourier_df <= 0.0))
    {
        fprintf(stderr, "Warning: skipping combined THM Fourier output because df is invalid.\n");
        return;
    }

    use_clean_endpoint = (!use_spline_endpoint && !use_partition_endpoint);
    use_blend_endpoint = thm_wdm_blend_endpoint_enabled && use_clean_endpoint;
    use_direct_endpoint_tdi = thm_wdm_direct_endpoint_tdi_enabled &&
                              !use_split_tdi_response;
    direct_endpoint_samples = 0;
    direct_endpoint_time = NULL;
    direct_endpoint_wave = NULL;
    direct_endpoint_quadrature = NULL;
    direct_full_reference_y = NULL;
    direct_full_reference_z = NULL;
    direct_full_reference_ready = 0;
    use_direct22_split_spa =
        thm_fourier_direct22_split_spa_enabled && out_fourier != NULL &&
        !use_split_tdi_response && ncarriers == 1 &&
        carrier[0].ell == 2 && abs(carrier[0].emm) == 2;
    direct22_split_valid = NULL;
    direct22_residual = NULL;
    direct22_frequency = NULL;
    direct22_fdot = NULL;
    direct22_carrier_phase = NULL;
    direct22_carrier_frequency = NULL;
    direct22_carrier_dfdu = NULL;
    direct22_plan_d1 = NULL;
    direct22_plan_d2 = NULL;
    direct22_residual_d1 = NULL;
    direct22_residual_d2 = NULL;
    use_local_spa_endpoint_plan = 0;
    memset(&local_spa_plan, 0, sizeof(local_spa_plan));
    endpoint_debug = (getenv("THM_ENDPOINT_DEBUG") != NULL);
    carrier_pixel_debug = (getenv("THM_CARRIER_PIXEL_DEBUG") != NULL);
    debug_nlo = 0;
    debug_nhi = Nt-1;
    debug_mlo = 1;
    debug_mhi = Nf-1;
    debug_env = getenv("THM_DEBUG_NLO");
    if(debug_env != NULL) debug_nlo = atoi(debug_env);
    debug_env = getenv("THM_DEBUG_NHI");
    if(debug_env != NULL) debug_nhi = atoi(debug_env);
    debug_env = getenv("THM_DEBUG_MLO");
    if(debug_env != NULL) debug_mlo = atoi(debug_env);
    debug_env = getenv("THM_DEBUG_MHI");
    if(debug_env != NULL) debug_mhi = atoi(debug_env);
    if(debug_nlo < 0) debug_nlo = 0;
    if(debug_nhi > Nt-1) debug_nhi = Nt-1;
    if(debug_mlo < 1) debug_mlo = 1;
    if(debug_mhi > Nf-1) debug_mhi = Nf-1;
    blend_half_width = thm_wdm_blend_half_width_layers*wdms->DF;
    if(!isfinite(blend_half_width) || blend_half_width <= 0.0)
    {
        blend_half_width = wdms->DF;
    }

    N = Nt*Nf;
    Tobs = wdms->Tobs;
    Mtot = params[0]+params[1];
    Mc = pow(params[0]*params[1], 3.0/5.0)/pow(Mtot, 1.0/5.0);
    tc = params[5];
    tc_output = linear_interp_clamped(Ns, plan_time, response_time, tc);

    stream_capacity = use_split_tdi_response ? 2*ncarriers : ncarriers;
    nstreams = stream_capacity;
    stream_carrier = malloc((size_t)stream_capacity*sizeof(THMFoldedCarrier));
    stream_parent = int_vector(stream_capacity);
    stream_piece = int_vector(stream_capacity);
    Achan = double_tensor(Nchan, stream_capacity, Ns);
    phase_td = double_tensor(Nchan, stream_capacity, Ns);
    freq_track = double_tensor(Nchan, stream_capacity, Ns);
    setup_carrier = double_tensor(Nchan, stream_capacity, 7);
    ATacc = malloc((size_t)Nchan*(size_t)stream_capacity*sizeof(gsl_interp_accel *));
    PTacc = malloc((size_t)Nchan*(size_t)stream_capacity*sizeof(gsl_interp_accel *));
    ATspline = malloc((size_t)Nchan*(size_t)stream_capacity*sizeof(gsl_spline *));
    PTspline = malloc((size_t)Nchan*(size_t)stream_capacity*sizeof(gsl_spline *));
    Xmode = double_vector(Ns);
    Ymode = double_vector(Ns);
    Zmode = double_vector(Ns);
    Xfmode = double_vector(Ns);
    Yfmode = double_vector(Ns);
    Zfmode = double_vector(Ns);
    phi_ref = double_vector(Ns);
    phi_res = double_vector(Ns);
    omega_plan = double_vector(Ns);

    if(use_direct22_split_spa)
    {
        direct22_split_valid = int_vector(Nchan*stream_capacity);
        direct22_residual = double_tensor(Nchan, stream_capacity, Ns);
        direct22_frequency = double_tensor(Nchan, stream_capacity, Ns);
        direct22_fdot = double_tensor(Nchan, stream_capacity, Ns);
        direct22_carrier_phase = double_vector(Ns);
        direct22_carrier_frequency = double_vector(Ns);
        direct22_carrier_dfdu = double_vector(Ns);
        direct22_plan_d1 = double_vector(Ns);
        direct22_plan_d2 = double_vector(Ns);
        direct22_residual_d1 = double_vector(Ns);
        direct22_residual_d2 = double_vector(Ns);
    }

    if(Achan == NULL || phase_td == NULL || freq_track == NULL ||
       stream_carrier == NULL || stream_parent == NULL || stream_piece == NULL ||
       setup_carrier == NULL || ATacc == NULL ||
       PTacc == NULL || ATspline == NULL || PTspline == NULL ||
       Xmode == NULL || Ymode == NULL || Zmode == NULL ||
       Xfmode == NULL || Yfmode == NULL || Zfmode == NULL ||
       phi_ref == NULL || phi_res == NULL || omega_plan == NULL ||
       (use_direct22_split_spa &&
        (direct22_split_valid == NULL || direct22_residual == NULL ||
         direct22_frequency == NULL || direct22_fdot == NULL ||
         direct22_carrier_phase == NULL ||
         direct22_carrier_frequency == NULL ||
         direct22_carrier_dfdu == NULL || direct22_plan_d1 == NULL ||
         direct22_plan_d2 == NULL || direct22_residual_d1 == NULL ||
         direct22_residual_d2 == NULL)))
    {
        fprintf(stderr, "allocation failure in combined THM WDM path\n");
        return;
    }

    for(idx=0; idx<Nchan*stream_capacity; idx++)
    {
        ATacc[idx] = NULL;
        PTacc[idx] = NULL;
        ATspline[idx] = NULL;
        PTspline[idx] = NULL;
        if(direct22_split_valid != NULL) direct22_split_valid[idx] = 0;
    }
    for(ch=0; ch<Nchan; ch++)
    {
        f_replace_start[ch] = HUGE_VAL;
        f_replace_pair_start[ch] = HUGE_VAL;
        f_replace_fallback[ch] = 0.0;
        f_clean_endpoint_taper_start[ch] = 0.0;
        spa_freq_time[ch] = 0.0;
        spa_wdm_time[ch] = 0.0;
        endpoint_build_time[ch] = 0.0;
        endpoint_wdm_time[ch] = 0.0;
        split_fft_time[ch] = 0.0;
    }

    carrier_tdi_time = 0.0;
    channel_ap_time = 0.0;
    driver_index = -1;
    driver_channel = -1;
    driver_score = -1.0;

    s = 0;
    for(k=0; k<ncarriers; k++)
    {
        const THMFoldedCarrier *c = &carrier[k];

        mode_index = c->mode_index;
        if(mode_index < 0 || mode_index >= model->nmodes)
        {
            fprintf(stderr, "Warning: skipping invalid THM carrier %d in combined WDM path.\n", k);
            continue;
        }

        ap_carrier = *c;
        /*
         * Folded-pair TDI extraction uses the quadrature of the folded track,
         * not the explicit +m and -m quadratures.  This is the same convention
         * as the single-carrier WDM diagnostic and is the object whose phase is
         * later passed to the SPA/WDM machinery.
         */
        ap_carrier.hpf_cos = ap_carrier.hp_sin;
        ap_carrier.hpf_sin = -ap_carrier.hp_cos;
        ap_carrier.hcf_cos = ap_carrier.hc_sin;
        ap_carrier.hcf_sin = -ap_carrier.hc_cos;

        for(i=0; i<Ns; i++)
        {
            phi_ref[i] = thm_reference_phase_eval(
                Pspline[mode_index], plan_time[i], Pacc[mode_index]);
        }

        mode_state = &model->modes[mode_index];
        t_transition = tc+Mtot*tCUT_Freq;
        if(c->ell == 2 && abs(c->emm) == 2)
        {
            fring = model->mode22.omegaRING/(2.0*M_PI*Mtot);
            fdamp = model->mode22.alpha1RD/Mtot;
        }
        else
        {
            fring = mode_state->phase.omegaRING/(2.0*M_PI*Mtot);
            fdamp = mode_state->phase.alpha1RD/Mtot;
        }

        npieces = use_split_tdi_response ? 2 : 1;
        for(piece_index=0; piece_index<npieces; piece_index++)
        {
            if(s >= stream_capacity)
            {
                fprintf(stderr, "Warning: split-TDI stream capacity exceeded in combined WDM path.\n");
                break;
            }

            response_piece = THM_TDI_PIECE_FULL;
            if(use_split_tdi_response)
            {
                response_piece = (piece_index == 0) ?
                    THM_TDI_PIECE_PROMPT_MICHELSON :
                    THM_TDI_PIECE_DELAYED_MICHELSON;
            }
            stream_carrier[s] = *c;
            stream_parent[s] = k;
            stream_piece[s] = response_piece;

            start = clock();
            fast_response_thm_piece(response_time, Ns, params, 1, &ap_carrier,
                                    SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                                    Aacc, Aspline, Pacc, Pspline,
                                    Xmode, Ymode, Zmode, Xfmode, Yfmode, Zfmode,
                                    response_piece);
            end = clock();
            carrier_tdi_time += ((double)(end-start))/CLOCKS_PER_SEC;

            Mchan[0] = Xmode;
            Mchan[1] = Ymode;
            Mchan[2] = Zmode;
            Mfchan[0] = Xfmode;
            Mfchan[1] = Yfmode;
            Mfchan[2] = Zfmode;

            for(ch=0; ch<Nchan; ch++)
            {
                start = clock();
                extractAP(Ns, Achan[ch][s], phi_res, Mchan[ch], Mfchan[ch], phi_ref);
                unwrap(Ns, phi_res);

                for(i=0; i<Ns; i++)
                {
                    phase_td[ch][s][i] = phi_ref[i]+phi_res[i];
                }
                idx = ch*stream_capacity+s;
                if(use_direct22_split_spa)
                {
                    direct22_split_valid[idx] =
                        build_direct22_residual_spa_tracks(
                            Ns, response_time, plan_time, phi_ref, phi_res,
                            phase_td[ch][s], params, model, mode_index,
                            direct22_carrier_phase,
                            direct22_residual[ch][s],
                            direct22_carrier_frequency,
                            direct22_carrier_dfdu,
                            direct22_plan_d1, direct22_plan_d2,
                            direct22_residual_d1, direct22_residual_d2,
                            direct22_frequency[ch][s],
                            direct22_fdot[ch][s]);
                    if(!direct22_split_valid[idx])
                    {
                        fprintf(stderr,
                                "Warning: direct-carrier split SPA setup failed for channel %s; using the ordinary full-phase SPA.\n",
                                chan_label[ch]);
                    }
                }
                for(i=0; i<Ns; i++)
                {
                    double dphidt;

                    dphidt = nonuniform_phase_derivative(i, Ns, response_time, phase_td[ch][s]);
                    freq_track[ch][s][i] = fabs(dphidt)/(2.0*M_PI);
                    if(!isfinite(freq_track[ch][s][i]) || freq_track[ch][s][i] <= 0.0)
                    {
                        freq_track[ch][s][i] = fabs(mode_freq[mode_index][i]);
                    }
                    omega_plan[i] = 2.0*M_PI*freq_track[ch][s][i];
                }
                /*
                 * No artificial monotone planning ramp is built here.  The
                 * physical post-TDI carrier frequency is the only frequency
                 * track allowed to control endpoint bookkeeping and WDM support.
                 */
                end = clock();
                channel_ap_time += ((double)(end-start))/CLOCKS_PER_SEC;

                if(thm_diagnostics_enabled)
                {
                    const char *piece_label = "full";

                    if(response_piece == THM_TDI_PIECE_PROMPT_MICHELSON)
                    {
                        piece_label = "prompt";
                    }
                    else if(response_piece == THM_TDI_PIECE_DELAYED_MICHELSON)
                    {
                        piece_label = "delayed";
                    }

                    snprintf(filename, sizeof(filename),
                             "THM_%s_mode%d%d_%s_TDI_AP.dat",
                             chan_label[ch], c->ell, abs(c->emm), piece_label);
                    out = fopen(filename, "w");
                    if(out != NULL)
                    {
                        fprintf(out, "# t_s phase_total amplitude_signed freq_track_raw_Hz freq_track_copy_Hz h h_quadrature h_reconstructed\n");
                        for(i=0; i<Ns; i++)
                        {
                            double hrec = Achan[ch][s][i]*cos(phase_td[ch][s][i]);

                            fprintf(out,
                                    "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                                    response_time[i], phase_td[ch][s][i],
                                    Achan[ch][s][i], freq_track[ch][s][i],
                                    freq_track[ch][s][i], Mchan[ch][i],
                                    Mfchan[ch][i], hrec);
                        }
                        fclose(out);
                    }
                }

                transformplan_custom(Mc, Mtot,
                                     linear_interp_clamped(Ns, plan_time,
                                                           response_time, tc),
                                     response_time, omega_plan,
                                     linear_interp_clamped(Ns, plan_time,
                                                           response_time,
                                                           t_transition),
                                     fring, fdamp, response_time[Ns-1],
                                     setup_carrier[ch][s],
                                     use_join_override, join_time_override,
                                     rise_override);

                {
                    int join_index = (int)setup_carrier[ch][s][4];
                    if(join_index < 0) join_index = 0;
                    if(join_index > Ns-1) join_index = Ns-1;
                    if(isfinite(setup_carrier[ch][s][5]))
                    {
                        fjoin = linear_interp_clamped(Ns, response_time,
                                                     freq_track[ch][s],
                                                     setup_carrier[ch][s][5]);
                    }
                    else
                    {
                        fjoin = freq_track[ch][s][join_index];
                    }
                    if(!isfinite(fjoin) || fjoin <= 0.0)
                    {
                        fjoin = NAN;
                    }
                }
                if(isfinite(fjoin) && fjoin > 0.0)
                {
                    if(endpoint_debug)
                    {
                        fprintf(stderr,
                                "ENDPOINT_DEBUG carrier_join channel %s mode %d%d stream %d abs_m %d tjoin %.15e fjoin %.15e start %.15e rise %.15e dt %.15e samples %.0f\n",
                                chan_label[ch], c->ell, abs(c->emm), s,
                                abs(c->emm), setup_carrier[ch][s][5],
                                fjoin, setup_carrier[ch][s][2],
                                setup_carrier[ch][s][3],
                                setup_carrier[ch][s][0],
                                setup_carrier[ch][s][1]);
                    }
                    /*
                     * Keep a representative historical SPA/FFT join frequency for
                     * diagnostics and as a fallback if the taper-start frequency
                     * cannot be inferred.  The partition-of-unity path below uses
                     * the start of the late time taper, not this later join, as the
                     * lower edge of the additive endpoint FFT.
                     */
                    if(fjoin > f_replace_fallback[ch])
                    {
                        f_replace_fallback[ch] = fjoin;
                    }
                    if(abs(c->emm) >= 2 && fjoin < f_replace_pair_start[ch])
                    {
                        f_replace_pair_start[ch] = fjoin;
                    }
                    if(abs(c->emm) >= thm_endpoint_min_abs_m &&
                       fjoin < f_replace_start[ch])
                    {
                        f_replace_start[ch] = fjoin;
                    }
                }
                score = setup_carrier[ch][s][6]/
                        (setup_carrier[ch][s][0]*setup_carrier[ch][s][1]);
                if(driver_index < 0 || score > driver_score)
                {
                    driver_index = s;
                    driver_channel = ch;
                    driver_score = score;
                    for(i=0; i<7; i++) setup_driver[i] = setup_carrier[ch][s][i];
                }

                if(thm_diagnostics_enabled)
                {
                    const char *piece_label = "full";
                    if(response_piece == THM_TDI_PIECE_PROMPT_MICHELSON) piece_label = "prompt";
                    else if(response_piece == THM_TDI_PIECE_DELAYED_MICHELSON) piece_label = "delayed";
                    printf("combined_carrier_short_fft_plan channel %s mode %d%d carrier %d stream %d piece %s start %.15e end %.15e duration %.15e dt %.15e samples %.0f rise %.15e join %.15e fjoin %.15e nyquist %.15e fmax_bin %.15e\n",
                           chan_label[ch], c->ell, abs(c->emm), k, s,
                           piece_label,
                           setup_carrier[ch][s][2],
                           setup_carrier[ch][s][2]+setup_carrier[ch][s][0]*setup_carrier[ch][s][1],
                           setup_carrier[ch][s][0]*setup_carrier[ch][s][1],
                           setup_carrier[ch][s][0], setup_carrier[ch][s][1],
                           setup_carrier[ch][s][3],
                           response_time[(int)setup_carrier[ch][s][4]], fjoin,
                           0.5/setup_carrier[ch][s][0], score);
                }

                ATacc[idx] = gsl_interp_accel_alloc();
                PTacc[idx] = gsl_interp_accel_alloc();
                ATspline[idx] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
                PTspline[idx] = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
                gsl_spline_init(ATspline[idx], response_time, Achan[ch][s], Ns);
                gsl_spline_init(PTspline[idx], response_time, phase_td[ch][s], Ns);
            }
            s++;
        }
    }
    nstreams = s;
    if(nstreams < 1)
    {
        fprintf(stderr, "Warning: combined THM WDM path produced no valid response streams.\n");
        return;
    }
    if(nstreams != stream_capacity)
    {
        fprintf(stderr, "Warning: combined THM WDM path does not support skipped carriers in split-stream storage yet.\n");
        return;
    }

    if(out_fourier != NULL &&
       thm_fourier_ap_grid_diagnostic_prefix[0] != '\0')
    {
        thm_write_fourier_posttdi_ap_grid_diagnostic(
            thm_fourier_ap_grid_diagnostic_prefix, params, Ns, nstreams,
            stream_carrier, response_time, Achan, phase_td, freq_track,
            ATacc, ATspline, PTacc, PTspline);
    }

    /*
     * Trial common endpoint planner.  The response tracks must all exist before
     * the earliest local-SPA failure can be identified, so the ordinary
     * per-carrier setup above is deliberately recalculated here only when the
     * experiment is enabled.  Production behavior is unchanged otherwise.
     */
    if(thm_endpoint_local_spa_planner_enabled && use_clean_endpoint &&
       !use_join_override &&
       thm_build_local_spa_endpoint_plan(Ns, response_time, nstreams,
                                         stream_carrier, mode_freq,
                                         Achan, freq_track,
                                         linear_interp_clamped(Ns, plan_time,
                                                               response_time,
                                                               tc),
                                         &local_spa_plan))
    {
        use_local_spa_endpoint_plan = 1;
        driver_index = -1;
        driver_channel = -1;
        driver_score = -1.0;
        for(ch=0; ch<Nchan; ch++)
        {
            f_replace_start[ch] = HUGE_VAL;
            f_replace_pair_start[ch] = HUGE_VAL;
            f_replace_fallback[ch] = 0.0;
        }

        for(k=0; k<nstreams; k++)
        {
            mode_index = stream_carrier[k].mode_index;
            if(mode_index < 0 || mode_index >= model->nmodes) continue;
            mode_state = &model->modes[mode_index];
            t_transition = tc+Mtot*tCUT_Freq;
            if(stream_carrier[k].ell == 2 &&
               abs(stream_carrier[k].emm) == 2)
            {
                fring = model->mode22.omegaRING/(2.0*M_PI*Mtot);
                fdamp = model->mode22.alpha1RD/Mtot;
            }
            else
            {
                fring = mode_state->phase.omegaRING/(2.0*M_PI*Mtot);
                fdamp = mode_state->phase.alpha1RD/Mtot;
            }

            for(ch=0; ch<Nchan; ch++)
            {
                for(i=0; i<Ns; i++)
                {
                    omega_plan[i] = 2.0*M_PI*freq_track[ch][k][i];
                }
                transformplan_custom(Mc, Mtot,
                                     linear_interp_clamped(Ns, plan_time,
                                                           response_time, tc),
                                     response_time, omega_plan,
                                     linear_interp_clamped(Ns, plan_time,
                                                           response_time,
                                                           t_transition),
                                     fring, fdamp,
                                     response_time[Ns-1],
                                     setup_carrier[ch][k], 1,
                                     local_spa_plan.join_time,
                                     local_spa_plan.rise);

                fjoin = linear_interp_clamped(Ns, response_time,
                                              freq_track[ch][k],
                                              local_spa_plan.join_time);
                if(isfinite(fjoin) && fjoin > 0.0)
                {
                    if(fjoin > f_replace_fallback[ch])
                    {
                        f_replace_fallback[ch] = fjoin;
                    }
                    if(abs(stream_carrier[k].emm) >= 2 &&
                       fjoin < f_replace_pair_start[ch])
                    {
                        f_replace_pair_start[ch] = fjoin;
                    }
                    if(abs(stream_carrier[k].emm) >= thm_endpoint_min_abs_m &&
                       fjoin < f_replace_start[ch])
                    {
                        f_replace_start[ch] = fjoin;
                    }
                }

                score = setup_carrier[ch][k][6]/
                        (setup_carrier[ch][k][0]*setup_carrier[ch][k][1]);
                if(driver_index < 0 || score > driver_score)
                {
                    driver_index = k;
                    driver_channel = ch;
                    driver_score = score;
                    for(i=0; i<7; i++) setup_driver[i] = setup_carrier[ch][k][i];
                }
            }
        }

        if(thm_diagnostics_enabled || endpoint_debug)
        {
            fprintf(stderr,
                    "LOCAL_SPA_ENDPOINT trigger_channel %d trigger_stream %d join %.15e rise %.15e f_trigger %.15e epsilon_f %.15e epsilon_A %.15e stationary_width %.15e\n",
                    local_spa_plan.trigger_channel,
                    local_spa_plan.trigger_stream,
                    local_spa_plan.join_time, local_spa_plan.rise,
                    local_spa_plan.trigger_frequency,
                    local_spa_plan.trigger_epsilon_f,
                    local_spa_plan.trigger_epsilon_a,
                    local_spa_plan.max_stationary_width);
        }
    }

    if(driver_index < 0 || !isfinite(driver_score) || driver_score <= 0.0)
    {
        fprintf(stderr, "Warning: combined THM WDM path could not determine a valid endpoint FFT driver.\n");
        return;
    }
    /*
     * Preserve the established frequency handoff, but make the common
     * endpoint taper flat at its lower blend edge.  Moving the blend upward
     * hides taper suppression by leaving the SPA in charge later than
     * intended.  The physical alternative is to move taper-flat earlier.
     */
    for(ch=0; ch<Nchan; ch++)
    {
        double original_flat_time;
        double original_taper_guard;

        if(!isfinite(f_replace_start[ch]) || f_replace_start[ch] <= 0.0)
        {
            f_replace_start[ch] = f_replace_pair_start[ch];
        }
        if(!isfinite(f_replace_start[ch]) || f_replace_start[ch] <= 0.0)
        {
            f_replace_start[ch] = f_replace_fallback[ch];
        }
        f_clean_endpoint_start[ch] = f_replace_start[ch];
        if(!isfinite(f_clean_endpoint_start[ch]) ||
           f_clean_endpoint_start[ch] <= 0.0)
        {
            f_clean_endpoint_start[ch] = f_replace_pair_start[ch];
        }
        if(!isfinite(f_clean_endpoint_start[ch]) ||
           f_clean_endpoint_start[ch] <= 0.0)
        {
            f_clean_endpoint_start[ch] = f_replace_fallback[ch];
        }

        original_flat_time = setup_driver[2]+setup_driver[3]+
                             SHORTFFT_MERGER_TAPER_MARGIN_SECONDS;
        original_taper_guard = 0.0;
        if(use_clean_endpoint)
        {
            for(k=0; k<nstreams; k++)
            {
                double fk = linear_interp_clamped(
                    Ns, response_time, freq_track[ch][k], original_flat_time);

                if(isfinite(fk) && fk > original_taper_guard)
                {
                    original_taper_guard = fk;
                }
            }
        }
        f_desired_endpoint_start[ch] = f_clean_endpoint_start[ch];
        if(original_taper_guard > f_desired_endpoint_start[ch])
        {
            f_desired_endpoint_start[ch] = original_taper_guard;
        }
        f_blend_lower[ch] = f_desired_endpoint_start[ch]-
                            (use_blend_endpoint ? blend_half_width : 0.0);
        if(f_blend_lower[ch] < 0.0) f_blend_lower[ch] = 0.0;
    }

    endpoint_target_flat = HUGE_VAL;
    if(use_clean_endpoint)
    {
        for(ch=0; ch<Nchan; ch++)
        {
            for(k=0; k<nstreams; k++)
            {
                double crossing_time;

                if(thm_first_upward_crossing_time(
                       Ns, response_time, freq_track[ch][k],
                       f_blend_lower[ch], tc_output, &crossing_time) &&
                   crossing_time < endpoint_target_flat)
                {
                    endpoint_target_flat = crossing_time;
                }
            }
        }
        if(isfinite(endpoint_target_flat))
        {
            thm_adjust_endpoint_taper_flat_time(setup_driver,
                                                endpoint_target_flat);
        }
    }

    for(ch=0; ch<Nchan; ch++)
    {
        double endpoint_flat_time;
        double f_taper_guard;

        f_clean_endpoint_start[ch] = f_desired_endpoint_start[ch];
        endpoint_flat_time = setup_driver[2]+setup_driver[3]+
                             SHORTFFT_MERGER_TAPER_MARGIN_SECONDS;
        f_taper_guard = 0.0;
        if(use_clean_endpoint)
        {
            for(k=0; k<nstreams; k++)
            {
                double fk = linear_interp_clamped(
                    Ns, response_time, freq_track[ch][k], endpoint_flat_time);

                if(isfinite(fk) && fk > f_taper_guard) f_taper_guard = fk;
            }
            if(f_taper_guard > 0.0)
            {
                double safe_center = f_taper_guard+
                    (use_blend_endpoint ? blend_half_width : 0.0);

                f_clean_endpoint_taper_start[ch] = f_taper_guard;
                if(safe_center > f_clean_endpoint_start[ch])
                {
                    f_clean_endpoint_start[ch] = safe_center;
                }
            }
        }
        if(endpoint_debug)
        {
            fprintf(stderr,
                    "ENDPOINT_DEBUG channel_boundary %s f_replace %.15e f_pair %.15e f_fallback %.15e f_taper_guard %.15e f_clean %.15e blend %d blend_width %.15e taper_flat %.15e target_flat %.15e\n",
                    chan_label[ch], f_replace_start[ch],
                    f_replace_pair_start[ch], f_replace_fallback[ch],
                    f_clean_endpoint_taper_start[ch],
                    f_clean_endpoint_start[ch], use_blend_endpoint,
                    blend_half_width, endpoint_flat_time,
                    endpoint_target_flat);
        }
    }
    if(thm_diagnostics_enabled)
    {
        printf("combined_wdm_endpoint_driver channel %s carrier %d mode %d%d fmax_bin %.15e f_endpoint_base_X %.15e f_endpoint_base_Y %.15e f_endpoint_base_Z %.15e f_clean_X %.15e f_clean_Y %.15e f_clean_Z %.15e\n",
               chan_label[driver_channel], driver_index,
               stream_carrier[driver_index].ell,
               abs(stream_carrier[driver_index].emm),
               driver_score, f_replace_start[0], f_replace_start[1],
               f_replace_start[2], f_clean_endpoint_start[0],
               f_clean_endpoint_start[1], f_clean_endpoint_start[2]);
    }

    if(use_direct_endpoint_tdi)
    {
        direct_endpoint_samples = (int)setup_driver[1];
        direct_endpoint_time = double_vector(direct_endpoint_samples);
        direct_endpoint_wave = double_matrix(Nchan, direct_endpoint_samples);
        direct_endpoint_quadrature = double_matrix(Nchan, direct_endpoint_samples);
        if(direct_endpoint_time == NULL || direct_endpoint_wave == NULL ||
           direct_endpoint_quadrature == NULL)
        {
            fprintf(stderr,
                    "Warning: could not allocate the exact-delay endpoint TDI response; using post-TDI AP splines.\n");
            free_double_vector(direct_endpoint_time);
            if(direct_endpoint_wave != NULL)
                free_double_matrix(direct_endpoint_wave, Nchan);
            if(direct_endpoint_quadrature != NULL)
                free_double_matrix(direct_endpoint_quadrature, Nchan);
            direct_endpoint_time = NULL;
            direct_endpoint_wave = NULL;
            direct_endpoint_quadrature = NULL;
            use_direct_endpoint_tdi = 0;
        }
        else
        {
            for(i=0; i<direct_endpoint_samples; i++)
            {
                direct_endpoint_time[i] =
                    setup_driver[2]+(double)i*setup_driver[0];
            }
            /*
             * Evaluate every delayed carrier directly from the intrinsic AP
             * splines at the endpoint FFT output times.  Unlike the dense
             * diagnostic path, this introduces no intermediate uniformly
             * sampled h(t) and therefore no cadence-limited interpolation of
             * the rapidly evolving merger signal.
             */
            fast_response_thm(
                direct_endpoint_time, direct_endpoint_samples,
                params, ncarriers, carrier,
                SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                Aacc, Aspline, Pacc, Pspline,
                direct_endpoint_wave[0], direct_endpoint_wave[1],
                direct_endpoint_wave[2],
                direct_endpoint_quadrature[0], direct_endpoint_quadrature[1],
                direct_endpoint_quadrature[2]);
        }
    }

    fourier_n = 0;
    if(out_fourier != NULL)
    {
        fourier_stop = fourier_fmax;
        if(!isfinite(fourier_stop) || fourier_stop <= 0.0)
        {
            fourier_stop = driver_score;
        }
        if(fourier_stop > 0.5/dt) fourier_stop = 0.5/dt;
        if(!isfinite(fourier_stop) || fourier_stop <= fourier_df)
        {
            fprintf(stderr, "Warning: combined THM Fourier output has an invalid frequency range.\n");
            return;
        }
        fourier_n = (int)floor(fourier_stop/fourier_df)+1;
        if(fourier_n < 2 || !thm_fourier_triplet_reserve(out_fourier, fourier_n))
        {
            fprintf(stderr, "Warning: could not allocate combined THM Fourier output.\n");
            return;
        }
        out_fourier->df = fourier_df;
        out_fourier->fmax = (double)(fourier_n-1)*fourier_df;
        out_fourier->blend_half_width = blend_half_width;
        out_fourier->endpoint_dt = setup_driver[0];
        out_fourier->endpoint_samples = (int)setup_driver[1];
        out_fourier->endpoint_start = setup_driver[2];
        out_fourier->endpoint_duration = setup_driver[0]*setup_driver[1];
        out_fourier->endpoint_driver_channel = driver_channel;
        out_fourier->endpoint_driver_ell = stream_carrier[driver_index].ell;
        out_fourier->endpoint_driver_abs_m = abs(stream_carrier[driver_index].emm);
        out_fourier->direct22_split_spa_used = use_direct22_split_spa;
        if(use_direct22_split_spa)
        {
            for(ch=0; ch<Nchan; ch++)
            {
                if(!direct22_split_valid[ch*stream_capacity])
                {
                    out_fourier->direct22_split_spa_used = 0;
                }
            }
        }
        out_fourier->endpoint_local_spa_planner_used = use_local_spa_endpoint_plan;
        if(use_local_spa_endpoint_plan)
        {
            out_fourier->endpoint_trigger_channel = local_spa_plan.trigger_channel;
            out_fourier->endpoint_trigger_ell =
                stream_carrier[local_spa_plan.trigger_stream].ell;
            out_fourier->endpoint_trigger_abs_m =
                abs(stream_carrier[local_spa_plan.trigger_stream].emm);
            out_fourier->endpoint_join_time = local_spa_plan.join_time;
            out_fourier->endpoint_rise = local_spa_plan.rise;
            out_fourier->endpoint_trigger_frequency =
                local_spa_plan.trigger_frequency;
            out_fourier->endpoint_trigger_epsilon_f =
                local_spa_plan.trigger_epsilon_f;
            out_fourier->endpoint_trigger_epsilon_a =
                local_spa_plan.trigger_epsilon_a;
        }
        out_fourier->has_full_fft = 0;
        out_fourier->full_fft_tukey_alpha = 0.0;
        for(ch=0; ch<Nchan; ch++)
        {
            out_fourier->blend_center[ch] = f_clean_endpoint_start[ch];
            for(fq=0; fq<fourier_n; fq++)
            {
                out_fourier->re[ch][fq] = 0.0;
                out_fourier->im[ch][fq] = 0.0;
                if(out_fourier->request_full_fft)
                {
                    out_fourier->full_re[ch][fq] = 0.0;
                    out_fourier->full_im[ch][fq] = 0.0;
                }
            }
        }
    }

    Ntsmax = 32;
    for(ch=0; ch<Nchan; ch++)
    {
        for(k=0; k<nstreams; k++)
        {
            int needed = Ns+(int)setup_carrier[ch][k][6]+32;
            if(needed > Ntsmax) Ntsmax = needed;
        }
    }
    owns_wdm_work = 1;
    if(reusable_work != NULL &&
       thm_reusable_wdm_work_prepare(reusable_work, N, Ntsmax, Ns))
    {
        nmid = reusable_work->nmid;
        nsize = reusable_work->nsize;
        nmid_endpoint = reusable_work->nmid_endpoint;
        nsize_endpoint = reusable_work->nsize_endpoint;
        listn = reusable_work->listn;
        listm = reusable_work->listm;
        wdmwave = reusable_work->wdmwave;
        wdm = reusable_work->wdm;
        freq = reusable_work->freq;
        phase = reusable_work->phase;
        Amp = reusable_work->amp;
        Ataper = reusable_work->ataper;
        hfull = reusable_work->hfull;
        owns_wdm_work = 0;
    }
    else
    {
        nmid = int_vector(Nf);
        nsize = int_vector(Nf);
        nmid_endpoint = int_vector(Nf);
        nsize_endpoint = int_vector(Nf);
        listn = int_vector(N);
        listm = int_vector(N);
        wdmwave = double_vector(N);
        wdm = double_matrix(Nt, Nf+1);
        freq = double_vector(Ntsmax);
        phase = double_vector(Ntsmax);
        Amp = double_vector(Ntsmax);
        Ataper = double_vector(Ns);
        hfull = double_vector(N);
    }
    wdm_split = NULL;
    AFacc = gsl_interp_accel_alloc();
    PFacc = gsl_interp_accel_alloc();
    TAacc = gsl_interp_accel_alloc();
    TAspline = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);

    alpha = 2.0*REFERENCE_TUKEY_ROLL_SECONDS/Tobs;
    if(alpha > 1.0) alpha = 1.0;
    if(alpha < 0.0) alpha = 0.0;
    if(out_fourier != NULL) out_fourier->full_fft_tukey_alpha = alpha;

    /*
     * Production no-SPA route.  The original split-early implementation below
     * is useful for dense diagnostics, but it repeats both the FFT packet work
     * and the carrier assembly for X, Y, and Z.  Use the same complex block
     * engine as TPHM whenever the caller requests the in-memory sparse result.
     */
    if(thm_wdm_fast_complex_partition_enabled && use_split_early_fft &&
       out_tracks != NULL && out_fourier == NULL &&
       !thm_diagnostics_enabled && !write_time_domain_files && !use_split_fft &&
       !use_split_tdi_response && !thm_wdm_full_fft_reference_enabled)
    {
        const THMObservationContext *fast_context = observation_context;
        THMWorkerWorkspace *fast_workspace = worker_workspace;
        THMObservationContext *owned_context = NULL;
        THMWorkerWorkspace *owned_workspace = NULL;
        double split_bandwidth = thm_wdm_split_early_fft_bandwidth_hz;
        double waveform_stop = setup_driver[2]+setup_driver[0]*setup_driver[1];
        long long fast_fft_samples = 0;
        int fast_blocks = 0;
        int fast_status;

        if(thm_split_fft_plan_nband > 0 &&
           isfinite(thm_split_fft_plan_bandwidth[0]) &&
           thm_split_fft_plan_bandwidth[0] > 0.0)
            split_bandwidth = thm_split_fft_plan_bandwidth[0];
        if(waveform_stop > response_time[Ns-1])
            waveform_stop = response_time[Ns-1];
        if(fast_context == NULL)
        {
            owned_context = thm_observation_context_create(
                thm_orbit_phase_offset);
            fast_context = owned_context;
        }
        if(fast_context != NULL && fast_workspace == NULL)
        {
            owned_workspace = thm_worker_workspace_create(fast_context);
            fast_workspace = owned_workspace;
        }
        if(fast_context != NULL && fast_workspace != NULL &&
           getenv("THM_PARTITION_DEBUG") != NULL)
        {
            const int nd = 257;
            double *td = double_vector(nd);
            double *xr = double_vector(nd), *yr = double_vector(nd);
            double *zr = double_vector(nd), *xq = double_vector(nd);
            double *yq = double_vector(nd), *zq = double_vector(nd);
            double complex *xc = calloc((size_t)nd, sizeof(*xc));
            double complex *yc = calloc((size_t)nd, sizeof(*yc));
            double complex *zc = calloc((size_t)nd, sizeof(*zc));
            THMFastPartitionSource debug_source;
            if(td != NULL && xr != NULL && yr != NULL && zr != NULL &&
               xq != NULL && yq != NULL && zq != NULL && xc != NULL &&
               yc != NULL && zc != NULL)
            {
                double error_minus[3] = {0.0, 0.0, 0.0};
                double error_plus[3] = {0.0, 0.0, 0.0};
                double error_ap[3] = {0.0, 0.0, 0.0};
                double norm[3] = {0.0, 0.0, 0.0};
                double span = fmin(2.0e6, waveform_stop-response_time[0]);
                int q;
                memset(&debug_source, 0, sizeof(debug_source));
                debug_source.ncarriers = ncarriers;
                debug_source.selected_carrier = -1;
                debug_source.carrier = carrier;
                debug_source.Aacc = Aacc;
                debug_source.Aspline = Aspline;
                debug_source.Pacc = Pacc;
                debug_source.Pspline = Pspline;
                debug_source.time_start = response_time[0];
                debug_source.time_stop = response_time[Ns-1];
                for(q=0; q<nd; q++)
                    td[q] = waveform_stop-span+
                            span*(double)q/(double)(nd-1);
                fast_response_thm(td, nd, params, ncarriers, carrier,
                                  SLacc, SLspline, SPacc, SPspline,
                                  SVacc, SVspline, Aacc, Aspline,
                                  Pacc, Pspline, xr, yr, zr, xq, yq, zq);
                if(thm_evaluate_complex_tdi_context(
                       fast_context, fast_workspace, nd, td,
                       params[7], params[8],
                       thm_fast_partition_polarizations, &debug_source,
                       xc, yc, zc) == 0)
                {
                    for(q=0; q<nd; q++)
                    {
                        const double real_value[3] = {xr[q], yr[q], zr[q]};
                        const double quadrature[3] = {xq[q], yq[q], zq[q]};
                        const double complex value[3] = {xc[q], yc[q], zc[q]};
                        int dch;
                        for(dch=0; dch<3; dch++)
                        {
                            double complex ap_value = 0.0;
                            double complex minus = real_value[dch]-
                                                   I*quadrature[dch];
                            double complex plus = real_value[dch]+
                                                  I*quadrature[dch];
                            int ds;
                            for(ds=0; ds<nstreams; ds++)
                            {
                                int didx = dch*stream_capacity+ds;
                                double amplitude;
                                double phase;

                                if(ATspline[didx] == NULL ||
                                   PTspline[didx] == NULL)
                                    continue;
                                amplitude = gsl_spline_eval(
                                    ATspline[didx], td[q], ATacc[didx]);
                                phase = gsl_spline_eval(
                                    PTspline[didx], td[q], PTacc[didx]);
                                ap_value += amplitude*(cos(phase)+I*sin(phase));
                            }
                            error_minus[dch] += pow(cabs(value[dch]-minus), 2.0);
                            error_plus[dch] += pow(cabs(value[dch]-plus), 2.0);
                            error_ap[dch] += pow(cabs(value[dch]-ap_value), 2.0);
                            norm[dch] += pow(cabs(value[dch]), 2.0);
                        }
                    }
                    for(q=0; q<3; q++)
                    {
                        fprintf(stderr,
                                "THM_PARTITION_DEBUG channel %d rel_l2_X_minus_iXq %.15e rel_l2_X_plus_iXq %.15e\n",
                                q, sqrt(error_minus[q]/norm[q]),
                                sqrt(error_plus[q]/norm[q]));
                        fprintf(stderr,
                                "THM_PARTITION_DEBUG channel %d rel_l2_complex_TDI_minus_postTDI_AP_spline %.15e\n",
                                q, sqrt(error_ap[q]/norm[q]));
                    }
                }
            }
            free(zc); free(yc); free(xc);
            free_double_vector(zq); free_double_vector(yq);
            free_double_vector(xq); free_double_vector(zr);
            free_double_vector(yr); free_double_vector(xr);
            free_double_vector(td);
        }
        fast_status = thm_fast_partitioned_wdm_context(
            fast_context, Ns, response_time, ncarriers,
            Achan, freq_track, setup_carrier,
            ATacc, ATspline, PTacc, PTspline,
            setup_driver[2], setup_driver[3],
            waveform_stop,
            setup_driver[6]/(setup_driver[0]*setup_driver[1]),
            split_bandwidth, SPLIT_FFT_ROLL_SECONDS,
            out_tracks, &fast_fft_samples, &fast_blocks);
        if(owned_workspace != NULL)
            thm_worker_workspace_destroy(owned_workspace);
        if(owned_context != NULL)
            thm_observation_context_destroy(owned_context);
        if(fast_status == 0)
        {
            if(write_sparse_files)
            {
                for(ch=0; ch<3; ch++)
                {
                    snprintf(filename, sizeof(filename),
                             "track_pixels_THM_%s.dat", chan_label[ch]);
                    write_thm_sparse_wdm_channel(filename,
                                                 &out_tracks->channel[ch]);
                }
            }
            if(thm_summary_output_enabled)
            {
                printf("split early FFT THM complex blocks %d fft_samples %lld total_channel_pixels %d\n",
                       fast_blocks, fast_fft_samples,
                       out_tracks->channel[0].npixels+
                       out_tracks->channel[1].npixels+
                       out_tracks->channel[2].npixels);
            }
            goto combined_cleanup;
        }
        fprintf(stderr,
                "Warning: fast complex THM partition failed (%d); using the legacy channel-by-channel branch.\n",
                fast_status);
    }

    if(thm_wdm_full_fft_reference_enabled)
    {
        double delay_pad = 500.0+4.0*THM_TDI_LARM_NOMINAL_SECONDS;
        clock_t reference_start = clock();

        direct_full_reference_y = double_vector(N);
        direct_full_reference_z = double_vector(N);
        if(direct_full_reference_y == NULL ||
           direct_full_reference_z == NULL)
        {
            fprintf(stderr,
                    "Warning: could not allocate the direct full-cadence THM reference.\n");
        }
        else
        {
            for(i=0; i<N; i++) wdmwave[i] = (double)i*dt;
            direct_response_full_waveform_thm(
                wdmwave, N, dt, delay_pad, params, ncarriers, carrier,
                SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                Aacc, Aspline, Pacc, Pspline,
                hfull, direct_full_reference_y, direct_full_reference_z);
            direct_full_reference_ready = 1;
            if(getenv("THM_PARTITION_DEBUG") != NULL)
                fprintf(stderr,
                        "THM_PARTITION_DEBUG direct_full_cadence_tdi_seconds %.6f samples %d\n",
                        (double)(clock()-reference_start)/
                            (double)CLOCKS_PER_SEC, N);
        }
    }

    /*
     * Combined THM WDM path, run for X, Y, and Z.  The default is the clean
     * endpoint construction: each folded carrier contributes SPA-only WDM
     * packets, and endpoint-time packets are then replaced by one direct WDM
     * transform of the summed short FFT per channel.  This avoids the old
     * ftran() bridge that appended short-FFT A(f),phi(f) samples to the SPA
     * samples and then splined across the interface.
     *
     * --wdm-spline-endpoint is retained only as a diagnostic of the old
     * per-carrier hybrid ftran() path.  That path can hide some SPA error but
     * also creates channel-dependent spline-bridge scars, so it is not the
     * default.  --wdm-partition-endpoint keeps the experimental
     * partition-of-unity taper/add path.
     */
    for(ch=0; ch<Nchan; ch++)
    {
        for(i=0; i<Nt; i++)
        {
            for(m=0; m<=Nf; m++) wdm[i][m] = 0.0;
        }
        for(m=0; m<=Nf; m++)
        {
            nmid_endpoint[m] = -1;
            nsize_endpoint[m] = 0;
        }

        for(k=0; k<nstreams; k++)
        {
            double spa_support_tmax;

            idx = ch*stream_capacity+k;
            if(ATspline[idx] == NULL || PTspline[idx] == NULL) continue;
            spa_support_tmax = response_time[Ns-1];

            start = clock();
            if(use_partition_endpoint)
            {
                for(i=0; i<Ns; i++)
                {
                    wlate = split_smooth_step(response_time[i],
                                              setup_driver[2],
                                              setup_driver[2]+setup_driver[3]);
                    wearly = 1.0-wlate;
                    Ataper[i] = wearly*Achan[ch][k][i];
                }
                gsl_spline_init(TAspline, response_time, Ataper, Ns);
                gsl_interp_accel_reset(TAacc);
                Nts = ftran_spa_only(setup_carrier[ch][k], response_time, Ns, Tobs,
                                     TAacc, TAspline,
                                     PTacc[idx], PTspline[idx],
                                     freq, phase, Amp);
            }
            else if(use_spline_endpoint)
            {
                short_tmp = double_vector((int)setup_carrier[ch][k][1]);
                if(short_tmp == NULL)
                {
                    end = clock();
                    spa_freq_time[ch] += ((double)(end-start))/CLOCKS_PER_SEC;
                    continue;
                }
                Nts = ftran(setup_carrier[ch][k], response_time, Ns, Tobs,
                            ATacc[idx], ATspline[idx],
                            PTacc[idx], PTspline[idx],
                            freq, phase, Amp, short_tmp);
                free_double_vector(short_tmp);
            }
            else
            {
                int q;
                double f_spa_stop;

                for(q=0; q<7; q++) setup_spa[q] = setup_carrier[ch][k][q];

                if(use_clean_endpoint)
                {
                    /*
                     * Clean endpoint replacement only overwrites layers whose
                     * full Meyer support lies above f_clean_endpoint_start.
                     * The layer immediately below that boundary still samples
                     * up to f_clean_endpoint_start+FB, so the SPA-only input
                     * must deliberately overlap that support.  If a smooth
                     * coefficient-space blend is requested, extend the SPA
                     * samples through the full upper side of the blend.  The
                     * endpoint FFT is a single summed transform for the whole
                     * channel, so both representations must be available across
                     * the blend band; otherwise the transition reintroduces a
                     * discrete per-layer handoff.  This avoids the old ftran()
                     * A(f),phi(f) bridge while keeping the two frequency-domain
                     * pieces continuous in the WDM sense.
                     */
                    f_spa_stop = f_clean_endpoint_start[ch]+wdms->FB;
                    if(use_blend_endpoint || out_fourier != NULL)
                    {
                        f_spa_stop += blend_half_width;
                    }
                    if(isfinite(f_spa_stop) && f_spa_stop > 0.0)
                    {
                        int jstop;

                        jstop = (int)setup_spa[4];
                        if(jstop < 1) jstop = 1;
                        if(jstop > Ns-1) jstop = Ns-1;
                        /*
                         * The overlap is a requirement on the physical
                         * post-TDI carrier spectrum.  Use freq_track so every
                         * carrier supplies SPA samples through the frequency
                         * support still assigned to the SPA side.
                         */
                        while(jstop+1 < Ns &&
                              freq_track[ch][k][jstop] < f_spa_stop)
                        {
                            jstop++;
                        }
                        if((double)jstop > setup_spa[4])
                        {
                            if(thm_diagnostics_enabled)
                            {
                                printf("combined_wdm clean SPA overlap channel %s carrier %d mode %d%d old_join_index %.0f new_join_index %d f_stop %.15e old_f %.15e new_f %.15e\n",
                                       chan_label[ch], k,
                                       stream_carrier[k].ell,
                                       abs(stream_carrier[k].emm), setup_spa[4],
                                       jstop, f_spa_stop,
                                       freq_track[ch][k][(int)setup_spa[4]],
                                       freq_track[ch][k][jstop]);
                            }
                            setup_spa[4] = (double)jstop;
                        }
                    }

                    /*
                     * The SPA spectrum above represents only the early,
                     * single-valued branch through setup_spa[4].  Do not let
                     * a later nonmonotonic post-TDI frequency excursion add
                     * late-time pixels to this early spectrum.  In particular,
                     * amplitude/transfer-function structure after merger can
                     * drive the extracted frequency back through low layers;
                     * assigning the inspiral SPA coefficients to those pixels
                     * creates discontinuous sparse support as parameters move.
                     * The common endpoint FFT builds its own late-time mask
                     * below and remains responsible for that branch.
                     */
                    {
                        int jstop = (int)setup_spa[4];
                        if(jstop < 1) jstop = 1;
                        if(jstop > Ns-1) jstop = Ns-1;
                        spa_support_tmax = response_time[jstop];
                    }
                }

                Nts = ftran_spa_only(setup_spa, response_time, Ns, Tobs,
                                     ATacc[idx], ATspline[idx],
                                     PTacc[idx], PTspline[idx],
                                     freq, phase, Amp);
            }
            end = clock();
            spa_freq_time[ch] += ((double)(end-start))/CLOCKS_PER_SEC;

            {
                int kept = 1;
                int dropped = 0;
                for(i=1; i<Nts; i++)
                {
                    if(freq[i] > freq[kept-1])
                    {
                        if(kept != i)
                        {
                            freq[kept] = freq[i];
                            phase[kept] = phase[i];
                            Amp[kept] = Amp[i];
                        }
                        kept++;
                    }
                    else
                    {
                        dropped++;
                    }
                }
                if(dropped > 0 && thm_diagnostics_enabled)
                {
                    printf("combined_wdm dropped %d non-increasing SPA samples for channel %s carrier %d mode %d%d\n",
                           dropped, chan_label[ch], k,
                           stream_carrier[k].ell,
                           abs(stream_carrier[k].emm));
                }
                Nts = kept;
            }
            if(Nts < 4) continue;

            AFspline = gsl_spline_alloc(gsl_interp_akima, Nts);
            PFspline = gsl_spline_alloc(gsl_interp_cspline, Nts);
            gsl_spline_init(AFspline, freq, Amp, Nts);
            gsl_spline_init(PFspline, freq, phase, Nts);
            gsl_interp_accel_reset(AFacc);
            gsl_interp_accel_reset(PFacc);

            /*
             * The Fourier diagnostic accumulates every folded carrier before
             * applying the common endpoint blend.  Keeping the sum complex is
             * essential: inclination/polarization factors and the relative
             * phases between harmonics must be present before the likelihood
             * is formed.
             */
            if(out_fourier != NULL)
            {
                int direct_samples;

                direct_samples = 0;
                if(use_direct22_split_spa && direct22_split_valid[idx])
                {
                    direct_samples = accumulate_direct22_split_spa_fourier(
                        Ns, response_time, plan_time,
                        direct22_residual[ch][k],
                        direct22_frequency[ch][k], direct22_fdot[ch][k],
                        (int)setup_spa[4], Tobs,
                        ATacc[idx], ATspline[idx], params, model,
                        fourier_df, fourier_n,
                        out_fourier->re[ch], out_fourier->im[ch]);
                }
                if(direct_samples == 0)
                {
                    for(fq=1; fq<fourier_n; fq++)
                    {
                        fout = (double)fq*fourier_df;
                        if(fout <= freq[0] || fout >= freq[Nts-1]) continue;
                        A = gsl_spline_eval(AFspline, fout, AFacc);
                        t = gsl_spline_eval(PFspline, fout, PFacc);
                        if(!isfinite(A) || !isfinite(t)) continue;
                        out_fourier->re[ch][fq] += A*cos(t);
                        out_fourier->im[ch][fq] += A*sin(t);
                    }
                }
                gsl_interp_accel_reset(AFacc);
                gsl_interp_accel_reset(PFacc);
            }

            if(use_clean_endpoint)
            {
                WDMpixelsTimeScanRange(Ns, response_time, freq_track[ch][k],
                                       response_time[0], spa_support_tmax,
                                       nmid, nsize, N, wdms);
            }
            else
            {
                WDMpixelsTimeScan(Ns, response_time, freq_track[ch][k],
                                  nmid, nsize, N, wdms);
            }
            if(use_spline_endpoint)
            {
                tail_time = tc;
                tail_amp = -1.0;
                for(i=0; i<Ns; i++)
                {
                    double amp_abs = fabs(Achan[ch][k][i]);
                    if(isfinite(amp_abs) && amp_abs > tail_amp)
                    {
                        tail_amp = amp_abs;
                        tail_time = response_time[i];
                    }
                }
                WDMpixelsAddMergerFrequencyTail(nmid, nsize, freq[Nts-1],
                                                tail_time, wdms);
            }

            start = clock();
            WDMtrack(wdmwave, listn, listm, &Np, Nts, N, nmid, nsize, tc,
                     freq, AFacc, AFspline, PFacc, PFspline, wdms);
            end = clock();
            spa_wdm_time[ch] += ((double)(end-start))/CLOCKS_PER_SEC;

            if(carrier_pixel_debug)
            {
                for(p=0; p<Np; p++)
                {
                    if(listn[p] >= debug_nlo && listn[p] <= debug_nhi &&
                       listm[p] >= debug_mlo && listm[p] <= debug_mhi)
                    {
                        printf("CARRIER_PIXEL_DEBUG channel %s carrier %d mode %d%d n %d m %d value %.15e Nts %d f_first %.15e f_last %.15e f_clean %.15e\n",
                               chan_label[ch], k, stream_carrier[k].ell,
                               abs(stream_carrier[k].emm), listn[p], listm[p],
                               wdmwave[p], Nts, freq[0], freq[Nts-1],
                               f_clean_endpoint_start[ch]);
                    }
                }
            }

            for(p=0; p<Np; p++) wdm[listn[p]][listm[p]] += wdmwave[p];

            if(use_partition_endpoint)
            {
                /*
                 * Build the additive endpoint mask from the same time-frequency
                 * track, but clipped to the short late-time FFT interval.  This
                 * is the important difference from the ordinary SPA mask: low
                 * frequency layers keep their long inspiral blocks for the SPA
                 * piece, while the endpoint FFT is only asked to populate
                 * packets whose time support actually overlaps w_late(t).
                 */
                WDMpixelsTimeScanRange(Ns, response_time, freq_track[ch][k],
                                       setup_driver[2],
                                       setup_driver[2]+setup_driver[0]*setup_driver[1],
                                       nmid, nsize, N, wdms);
                WDMmergePixelPlans(nmid_endpoint, nsize_endpoint, nmid, nsize);
            }
            else if(use_clean_endpoint)
            {
                double direct_start_time;
                double carrier_join_time;
                double fmax_spectrum;

                direct_start_time = setup_driver[2]+setup_driver[3]+
                                    SHORTFFT_MERGER_TAPER_MARGIN_SECONDS;
                carrier_join_time = setup_carrier[ch][k][5];
                if(!isfinite(carrier_join_time))
                {
                    carrier_join_time = response_time[(int)setup_carrier[ch][k][4]];
                }
                /*
                 * In the hard-replacement ordinary path the endpoint
                 * replacement starts no earlier than the per-carrier SPA/FFT
                 * join.  A blended endpoint, like the split-TDI experiment, is
                 * gated by the common taper-flat time and the frequency guard
                 * above.  Pushing that mask out to each stream-local join would
                 * remove the low-frequency side of the blend, which is exactly
                 * the layer-scale discontinuity the blend is meant to remove.
                 */
                if(!use_split_tdi_response && !use_blend_endpoint &&
                   direct_start_time < carrier_join_time)
                {
                    direct_start_time = carrier_join_time;
                }

                WDMpixelsTimeScanRange(Ns, response_time, freq_track[ch][k],
                                       direct_start_time,
                                       setup_driver[2]+setup_driver[0]*setup_driver[1],
                                       nmid, nsize, N, wdms);

                tail_time = tc;
                tail_amp = -1.0;
                for(i=0; i<Ns; i++)
                {
                    double amp_abs = fabs(Achan[ch][k][i]);
                    if(isfinite(amp_abs) && amp_abs > tail_amp)
                    {
                        tail_amp = amp_abs;
                        tail_time = response_time[i];
                    }
                }
                fmax_spectrum = setup_carrier[ch][k][6]/
                                (setup_carrier[ch][k][0]*setup_carrier[ch][k][1]);
                WDMpixelsAddMergerFrequencyTail(nmid, nsize, fmax_spectrum,
                                                tail_time, wdms);
                WDMmergePixelPlans(nmid_endpoint, nsize_endpoint, nmid, nsize);
            }

            gsl_spline_free(AFspline);
            gsl_spline_free(PFspline);
        }

        endpoint_wdm_time[ch] = 0.0;
        endpoint_direct_layers = 0;
        endpoint_direct_pixels = 0;
        endpoint_plan_layers = 0;
        endpoint_plan_volume = 0;
        endpoint_layers = 0;
        endpoint_volume = 0;

        if(use_partition_endpoint || use_clean_endpoint)
        {
            short_hsum = double_vector((int)setup_driver[1]);
            start = clock();
            for(i=0; i<(int)setup_driver[1]; i++)
            {
                t = setup_driver[2]+(double)i*setup_driver[0];
                if(use_direct_endpoint_tdi && direct_endpoint_wave != NULL)
                {
                    hval = direct_endpoint_wave[ch][i];
                }
                else
                {
                    hval = 0.0;
                    if(t >= response_time[0] && t <= response_time[Ns-1])
                    {
                        for(k=0; k<nstreams; k++)
                        {
                            idx = ch*stream_capacity+k;
                            A = gsl_spline_eval(ATspline[idx], t, ATacc[idx]);
                            hval += A*cos(gsl_spline_eval(PTspline[idx], t,
                                                         PTacc[idx]));
                        }
                    }
                }
                if(t-setup_driver[2] < setup_driver[3])
                {
                    hval *= 0.5*(1.0-cos(M_PI*(t-setup_driver[2])/setup_driver[3]));
                }
                short_hsum[i] = hval;
            }
            end = clock();
            endpoint_build_time[ch] = ((double)(end-start))/CLOCKS_PER_SEC;

            if(thm_diagnostics_enabled)
            {
                snprintf(filename, sizeof(filename), "THM_%s_short_window_t.dat", chan_label[ch]);
                out = fopen(filename, "w");
                if(out != NULL)
                {
                    fprintf(out, "# t_s h_%s_sum_tapered\n", chan_label[ch]);
                    for(i=0; i<(int)setup_driver[1]; i++)
                    {
                        t = setup_driver[2]+(double)i*setup_driver[0];
                        fprintf(out, "%.15e %.15e\n", t, short_hsum[i]);
                    }
                    fclose(out);
                }
                if(ch == 0)
                {
                    out = fopen("THM_all_short_window_t.dat", "w");
                    if(out != NULL)
                    {
                        fprintf(out, "# t_s h_X_sum_tapered\n");
                        for(i=0; i<(int)setup_driver[1]; i++)
                        {
                            t = setup_driver[2]+(double)i*setup_driver[0];
                            fprintf(out, "%.15e %.15e\n", t, short_hsum[i]);
                        }
                        fclose(out);
                    }
                }
            }

            if(out_fourier != NULL)
            {
                gsl_spline *EAspline = NULL;
                gsl_spline *EPspline = NULL;

                Nshortfft = ftran_endpoint_fft(setup_driver, Tobs, short_hsum,
                                               0.0, freq, phase, Amp);
                if(Nshortfft >= 4)
                {
                    EAspline = gsl_spline_alloc(gsl_interp_akima, Nshortfft);
                    EPspline = gsl_spline_alloc(gsl_interp_cspline, Nshortfft);
                }
                if(EAspline != NULL && EPspline != NULL)
                {
                    gsl_spline_init(EAspline, freq, Amp, Nshortfft);
                    gsl_spline_init(EPspline, freq, phase, Nshortfft);
                    gsl_interp_accel_reset(AFacc);
                    gsl_interp_accel_reset(PFacc);

                    for(fq=1; fq<fourier_n; fq++)
                    {
                        fout = (double)fq*fourier_df;
                        fourier_weight = split_smooth_step(
                            fout,
                            f_clean_endpoint_start[ch]-blend_half_width,
                            f_clean_endpoint_start[ch]+blend_half_width);
                        endpoint_re = 0.0;
                        endpoint_im = 0.0;
                        if(fout > freq[0] && fout < freq[Nshortfft-1])
                        {
                            A = gsl_spline_eval(EAspline, fout, AFacc);
                            t = gsl_spline_eval(EPspline, fout, PFacc);
                            if(isfinite(A) && isfinite(t))
                            {
                                endpoint_re = A*cos(t);
                                endpoint_im = A*sin(t);
                            }
                        }
                        out_fourier->re[ch][fq] =
                            (1.0-fourier_weight)*out_fourier->re[ch][fq]
                            +fourier_weight*endpoint_re;
                        out_fourier->im[ch][fq] =
                            (1.0-fourier_weight)*out_fourier->im[ch][fq]
                            +fourier_weight*endpoint_im;
                    }
                }
                else
                {
                    fprintf(stderr,
                            "Warning: channel %s endpoint Fourier blend has too few FFT samples.\n",
                            chan_label[ch]);
                }
                if(EAspline != NULL) gsl_spline_free(EAspline);
                if(EPspline != NULL) gsl_spline_free(EPspline);
                gsl_interp_accel_reset(AFacc);
                gsl_interp_accel_reset(PFacc);
            }

            if(use_partition_endpoint)
            {
                /*
                 * Partition-of-unity endpoint: add WDM[w_late h] to the
                 * falling-SPA contribution.  Unlike the clean replacement
                 * path, this is not a hand-off between layers: both tapered
                 * pieces are real parts of the waveform and must be allowed to
                 * overlap where the late-time WDM packets are safely described
                 * by the compact endpoint FFT.  Very low layers have long time
                 * support and are not reliably supplied by this short block,
                 * so keep the lower validity guard one Meyer half-band below
                 * the start of the rising endpoint taper.
                 */
                f_late_start = HUGE_VAL;
                for(k=0; k<nstreams; k++)
                {
                    double fk = linear_interp_clamped(Ns, response_time,
                                                     freq_track[ch][k],
                                                     setup_driver[2]);
                    if(isfinite(fk) && fk > 0.0 && fk < f_late_start)
                    {
                        f_late_start = fk;
                    }
                }
                if(isfinite(f_late_start) && f_late_start > 0.0)
                {
                    f_endpoint_start = f_late_start-wdms->FB;
                }
                else
                {
                    f_endpoint_start = f_replace_start[ch]-wdms->FB;
                }
            }
            else
            {
                /*
                 * Clean endpoint replacement is time/support selected through
                 * nmid_endpoint/nsize_endpoint, but still needs a frequency
                 * guard.  Low-frequency WDM packets have very long time
                 * support, so a compact endpoint FFT cannot replace them even
                 * when the track touches the endpoint range.  By default this
                 * guard uses the common first |m|>=3 folded-carrier join,
                 * falling back to the |m|>=2 join if no higher carrier is
                 * present.  Diagnostics can lower thm_endpoint_min_abs_m to 2
                 * to test whether the last 22-owned SPA layers are already too
                 * inaccurate for likelihood work.
                 */
                f_late_start = use_split_tdi_response ?
                    f_clean_endpoint_taper_start[ch] : 0.0;
                f_endpoint_start = f_clean_endpoint_start[ch];
            }
            if(!isfinite(f_endpoint_start) || f_endpoint_start < 0.0) f_endpoint_start = 0.0;

            f_endpoint_max = setup_driver[6]/(setup_driver[0]*setup_driver[1]);
            if(f_endpoint_max > 0.5/setup_driver[0]) f_endpoint_max = 0.5/setup_driver[0];
            if(use_blend_endpoint)
            {
                endpoint_mlo = (int)floor((f_endpoint_start-
                                           blend_half_width-wdms->FB)/wdms->DF);
            }
            else
            {
                endpoint_mlo = (int)ceil((f_endpoint_start+wdms->FB)/wdms->DF);
            }
            endpoint_mmax = (int)floor((f_endpoint_max-wdms->FB)/wdms->DF);
            if(endpoint_mlo < 1) endpoint_mlo = 1;
            if(endpoint_mmax > Nf-1) endpoint_mmax = Nf-1;

            if(endpoint_mmax >= endpoint_mlo)
            {
                for(m=endpoint_mlo; m<=endpoint_mmax; m++)
                {
                    if(nmid_endpoint[m] >= 0 && nsize_endpoint[m] > 0)
                    {
                        endpoint_plan_layers++;
                        endpoint_plan_volume += nsize_endpoint[m];
                        endpoint_layers++;
                        endpoint_volume += nsize_endpoint[m];
                    }
                }

                if(thm_diagnostics_enabled)
                {
                    snprintf(filename, sizeof(filename), "combined_endpoint_direct_layers_%s.dat", chan_label[ch]);
                }
                start = clock();
                if(use_partition_endpoint)
                {
                    WDMaddWithShortFFTThreshold(wdm, nmid_endpoint, nsize_endpoint,
                                                short_hsum, setup_driver,
                                                f_endpoint_start, f_endpoint_max,
                                                wdms, thm_diagnostics_enabled ? filename : NULL,
                                                &endpoint_direct_layers,
                                                &endpoint_direct_pixels);
                }
                else if(use_blend_endpoint)
                {
                    WDMblendWithShortFFTThreshold(wdm, nmid_endpoint, nsize_endpoint,
                                                  short_hsum, setup_driver,
                                                  f_endpoint_start, f_endpoint_max,
                                                  blend_half_width,
                                                  wdms, thm_diagnostics_enabled ? filename : NULL,
                                                  &endpoint_direct_layers,
                                                  &endpoint_direct_pixels);
                }
                else
                {
                    WDMreplaceWithShortFFTThreshold(wdm, nmid_endpoint, nsize_endpoint,
                                                    short_hsum, setup_driver,
                                                    f_endpoint_start, f_endpoint_max,
                                                    wdms, thm_diagnostics_enabled ? filename : NULL,
                                                    &endpoint_direct_layers,
                                                    &endpoint_direct_pixels);
                }
                end = clock();
                endpoint_wdm_time[ch] = ((double)(end-start))/CLOCKS_PER_SEC;
            }
            else
            {
                fprintf(stderr, "Warning: channel %s combined endpoint FFT has no fully supported WDM layers between %.15e and %.15e Hz.\n",
                        chan_label[ch], f_endpoint_start, f_endpoint_max);
            }

            if(thm_diagnostics_enabled)
            {
                printf("combined_wdm_endpoint_layers channel %s mode %s mlo %d mmax %d layers %d volume %d plan_layers %d plan_volume %d direct_layers %d direct_pixels %d fjoin_ref %.15e ftaper_start %.15e fapply_start %.15e fmax %.15e\n",
                       chan_label[ch],
                       use_partition_endpoint ? "partition_add" :
                       (use_blend_endpoint ? "blend_replace" : "clean_replace"),
                       endpoint_mlo, endpoint_mmax, endpoint_layers,
                       endpoint_volume, endpoint_plan_layers, endpoint_plan_volume,
                       endpoint_direct_layers,
                       endpoint_direct_pixels,
                       use_partition_endpoint ? f_replace_start[ch] :
                                                f_clean_endpoint_start[ch],
                       f_late_start,
                       f_endpoint_start, f_endpoint_max);
            }

            free_double_vector(short_hsum);
        }

        if(out_fourier != NULL && out_fourier->request_full_fft)
        {
            /*
             * Expensive reference used only by the Fourier diagnostic.  It is
             * intentionally constructed from the same post-TDI carrier
             * splines, but then follows the ordinary full-cadence FFT route.
             * The start/end Tukey keeps finite-observation leakage from being
             * mistaken for a hybridization error.
             */
            for(i=0; i<N; i++)
            {
                t = (double)i*dt;
                hval = 0.0;
                if(t >= response_time[0] && t <= response_time[Ns-1])
                {
                    for(k=0; k<nstreams; k++)
                    {
                        idx = ch*stream_capacity+k;
                        hval += gsl_spline_eval(ATspline[idx], t, ATacc[idx])*
                                cos(gsl_spline_eval(PTspline[idx], t, PTacc[idx]));
                    }
                }
                hfull[i] = hval;
            }
            tukey(hfull, alpha, N);
            gsl_fft_real_radix2_transform(hfull, 1, N);
            for(i=0; i<N; i++) hfull[i] *= 2.0*dt;
            for(fq=1; fq<fourier_n; fq++)
            {
                fout = (double)fq*fourier_df;
                sample_direct_fft_spectrum(hfull, N, Tobs, fout,
                                           &out_fourier->full_re[ch][fq],
                                           &out_fourier->full_im[ch][fq]);
            }
            if(ch == Nchan-1) out_fourier->has_full_fft = 1;
        }

        if(thm_wdm_full_fft_reference_enabled)
        {
            int Np_full;

            /*
             * Gold-standard reference for likelihood/debug work.  The full
             * multi-harmonic h_+,h_x waveform was sampled at dt above and the
             * TDI delays were then applied directly at every cadence sample.
             * In particular, this is not a dense resampling of the sparse
             * post-TDI amplitude/phase representation and is independent of
             * every fast transform under test.
             */
            if(direct_full_reference_ready)
            {
                if(ch == 1)
                    memcpy(hfull, direct_full_reference_y,
                           (size_t)N*sizeof(*hfull));
                else if(ch == 2)
                    memcpy(hfull, direct_full_reference_z,
                           (size_t)N*sizeof(*hfull));
            }
            else
            {
                for(i=0; i<N; i++) hfull[i] = 0.0;
            }
            gsl_fft_real_radix2_transform(hfull, 1, N);
            for(i=0; i<N; i++) hfull[i] *= 2.0*dt;

            WDMbuildTHMUnionPixelPlan(Ns, response_time, nstreams, ch,
                                      Achan, freq_track, setup_carrier, wdms,
                                      1, nmid, nsize);
            WDMtrackFFT(wdmwave, listn, listm, &Np_full, N, nmid, nsize,
                        hfull, wdms, 0, 0.0, 0.0);
            for(i=0; i<Nt; i++)
            {
                for(m=0; m<=Nf; m++) wdm[i][m] = 0.0;
            }
            unpack_wdm_track(wdm, listn, listm, wdmwave, Np_full);
        }

        if(out_tracks != NULL)
        {
            final_pixels = thm_sparse_wdm_channel_from_dense(&out_tracks->channel[ch], wdm);
            if(final_pixels < 0)
            {
                fprintf(stderr, "Warning: could not collect sparse WDM pixels for channel %s.\n",
                        chan_label[ch]);
                final_pixels = 0;
            }
        }
        else
        {
            final_pixels = count_wdm_nonzero_pixels(wdm);
        }
        if(write_sparse_files)
        {
            snprintf(filename, sizeof(filename), "track_pixels_THM_%s.dat", chan_label[ch]);
            if(out_tracks != NULL)
            {
                write_thm_sparse_wdm_channel(filename, &out_tracks->channel[ch]);
            }
            else
            {
                final_pixels = write_wdm_nonzero_track_pixels(filename, wdm);
            }
        }
        if(thm_diagnostics_enabled)
        {
            if(ch == 0) write_wdm_nonzero_track_pixels("track_pixels_THM_all.dat", wdm);

            snprintf(filename, sizeof(filename), "wtranfast_THM_%s.dat", chan_label[ch]);
            write_wdm_matrix(filename, wdm);
            if(ch == 0) write_wdm_matrix("wtranfast_THM_all.dat", wdm);

            snprintf(filename, sizeof(filename), "BinaryFastTHM_%s.dat", chan_label[ch]);
            write_wdm_binary(filename, wdm, wdms);
            if(ch == 0) write_wdm_binary("BinaryFastTHM_all.dat", wdm, wdms);
        }

        if(thm_diagnostics_enabled || write_time_domain_files)
        {
            snprintf(filename, sizeof(filename), "THM_all_%stime.dat", chan_label[ch]);
            out = fopen(filename, "w");
            for(i=0; i<N; i++)
            {
                t = (double)i*dt;
                hval = 0.0;
                if(t >= response_time[0] && t <= response_time[Ns-1])
                {
                    for(k=0; k<nstreams; k++)
                    {
                        idx = ch*stream_capacity+k;
                        hval += gsl_spline_eval(ATspline[idx], t, ATacc[idx])*
                                cos(gsl_spline_eval(PTspline[idx], t, PTacc[idx]));
                    }
                }
                hfull[i] = hval;
                if(out != NULL) fprintf(out, "%.15e %.15e\n", t, hval);
            }
            if(out != NULL) fclose(out);

            tukey(hfull, alpha, N);
            snprintf(filename, sizeof(filename), "THM_all_%stime_tukey.dat", chan_label[ch]);
            out = fopen(filename, "w");
            if(out != NULL)
            {
                for(i=0; i<N; i++)
                {
                    t = (double)i*dt;
                    fprintf(out, "%.15e %.15e\n", t, hfull[i]);
                }
                fclose(out);
            }
        }

        if(thm_split_fft_plan_enabled && thm_split_fft_plan_nband > 0)
        {
            WDMwriteSplitFFTBandwidthPlan(Ns, response_time, nstreams, ch,
                                          Achan, freq_track, setup_carrier,
                                          wdms, chan_label[ch],
                                          setup_driver[2],
                                          thm_split_fft_plan_nband,
                                          thm_split_fft_plan_bandwidth);
            WDMwriteSplitFFTTrackBandwidthPlan(Ns, response_time, nstreams,
                                                ch, freq_track,
                                                stream_carrier, wdms,
                                                chan_label[ch],
                                                setup_driver[2],
                                                thm_split_fft_plan_nband,
                                                thm_split_fft_plan_bandwidth);
        }

        if(use_split_fft)
        {
            wdm_split = double_matrix(Nt, Nf+1);
            if(wdm_split == NULL)
            {
                fprintf(stderr, "Warning: could not allocate split-FFT WDM matrix for channel %s.\n",
                        chan_label[ch]);
            }
            else
            {
                start = clock();
                WDMtrackSplitFFTSummed(wdm_split, Ns, response_time, nstreams,
                                       ch, Achan, freq_track, setup_carrier,
                                       ATacc, ATspline, PTacc, PTspline,
                                       wdms, chan_label[ch],
                                       &split_fft_layers, &split_fft_pixels);
                end = clock();
                split_fft_time[ch] = ((double)(end-start))/CLOCKS_PER_SEC;

                snprintf(filename, sizeof(filename), "track_pixels_splitfft_THM_%s.dat",
                         chan_label[ch]);
                write_wdm_nonzero_track_pixels(filename, wdm_split);

                snprintf(filename, sizeof(filename), "wtranfast_splitfft_THM_%s.dat",
                         chan_label[ch]);
                write_wdm_matrix(filename, wdm_split);

                if(thm_diagnostics_enabled)
                {
                    snprintf(filename, sizeof(filename), "BinaryFastSplitFFTTHM_%s.dat",
                             chan_label[ch]);
                    write_wdm_binary(filename, wdm_split, wdms);
                }

                printf("split FFT THM %s used %d layers / %d pixels in %f seconds\n",
                       chan_label[ch], split_fft_layers, split_fft_pixels,
                       split_fft_time[ch]);
                free_double_matrix(wdm_split, Nt);
                wdm_split = NULL;
            }
        }

        if(use_split_early_fft)
        {
            double split_bandwidth;

            wdm_split = double_matrix(Nt, Nf+1);
            if(wdm_split == NULL)
            {
                fprintf(stderr, "Warning: could not allocate split-early FFT WDM matrix for channel %s.\n",
                        chan_label[ch]);
            }
            else
            {
                split_bandwidth = thm_wdm_split_early_fft_bandwidth_hz;
                if(thm_split_fft_plan_nband > 0 &&
                   isfinite(thm_split_fft_plan_bandwidth[0]) &&
                   thm_split_fft_plan_bandwidth[0] > 0.0)
                {
                    split_bandwidth = thm_split_fft_plan_bandwidth[0];
                }

                start = clock();
                WDMtrackEarlySplitFFTPerCarrierEndpoint(wdm_split, Ns,
                                                        response_time,
                                                        nstreams, ch, Achan,
                                                        freq_track,
                                                        setup_carrier,
                                                        ATacc, ATspline,
                                                        PTacc, PTspline,
                                                        wdms, setup_driver,
                                                        split_bandwidth,
                                                        chan_label[ch],
                                                        &split_fft_layers,
                                                        &split_fft_pixels);
                end = clock();
                split_fft_time[ch] += ((double)(end-start))/CLOCKS_PER_SEC;

                /*
                 * In library mode the split branch is the requested waveform,
                 * not an auxiliary diagnostic.  Overwrite the earlier hybrid
                 * sparse result before releasing the dense split workspace.
                 */
                if(out_tracks != NULL)
                {
                    final_pixels = thm_sparse_wdm_channel_from_dense(
                        &out_tracks->channel[ch], wdm_split);
                    if(final_pixels < 0)
                    {
                        fprintf(stderr,
                                "Warning: could not collect split-early sparse WDM pixels for channel %s.\n",
                                chan_label[ch]);
                        final_pixels = 0;
                    }
                }

                snprintf(filename, sizeof(filename),
                         "track_pixels_splitearly_THM_%s.dat",
                         chan_label[ch]);
                write_wdm_nonzero_track_pixels(filename, wdm_split);

                snprintf(filename, sizeof(filename),
                         "wtranfast_splitearly_THM_%s.dat",
                         chan_label[ch]);
                write_wdm_matrix(filename, wdm_split);

                if(thm_diagnostics_enabled)
                {
                    snprintf(filename, sizeof(filename),
                             "BinaryFastSplitEarlyTHM_%s.dat",
                             chan_label[ch]);
                    write_wdm_binary(filename, wdm_split, wdms);
                }

                if(thm_summary_output_enabled)
                {
                    printf("split early FFT THM %s used %d layers / %d pixels in %f seconds\n",
                           chan_label[ch], split_fft_layers, split_fft_pixels,
                           ((double)(end-start))/CLOCKS_PER_SEC);
                }
                free_double_matrix(wdm_split, Nt);
                wdm_split = NULL;
            }
        }

        if(write_sparse_files || thm_diagnostics_enabled || write_time_domain_files ||
           use_split_fft || use_split_early_fft)
        {
            snprintf(filename, sizeof(filename), "wdm_match_info_THM_%s.dat", chan_label[ch]);
            write_match_info(filename, wdms, alpha, 0.5*alpha*Tobs);
            if(ch == 0)
            {
                write_match_info("wdm_match_info_THM_all.dat", wdms, alpha, 0.5*alpha*Tobs);
                write_match_info("wdm_match_info.dat", wdms, alpha, 0.5*alpha*Tobs);
            }
        }

        if(thm_diagnostics_enabled)
        {
            printf("combined_wdm_channel_timing channel %s spa_freq %.6f spa_wdm %.6f endpoint_build %.6f endpoint_direct_wdm %.6f final_nonzero %d\n",
                   chan_label[ch], spa_freq_time[ch], spa_wdm_time[ch],
                   endpoint_build_time[ch], endpoint_wdm_time[ch], final_pixels);
        }
    }

    total_spa_freq = 0.0;
    total_spa_wdm = 0.0;
    total_endpoint_build = 0.0;
    total_endpoint_wdm = 0.0;
    total_split_fft = 0.0;
    for(ch=0; ch<Nchan; ch++)
    {
        total_spa_freq += spa_freq_time[ch];
        total_spa_wdm += spa_wdm_time[ch];
        total_endpoint_build += endpoint_build_time[ch];
        total_endpoint_wdm += endpoint_wdm_time[ch];
        total_split_fft += split_fft_time[ch];
    }

    if(thm_summary_output_enabled)
    {
        printf("combined_wdm_timing_total channels XYZ carrier_tdi %.6f channel_ap %.6f spa_freq %.6f spa_wdm %.6f endpoint_build %.6f endpoint_direct_wdm %.6f split_fft %.6f\n",
               carrier_tdi_time, channel_ap_time, total_spa_freq, total_spa_wdm,
               total_endpoint_build, total_endpoint_wdm, total_split_fft);
        if(thm_diagnostics_enabled)
        {
            if(write_sparse_files)
            {
                printf("combined_wdm_outputs wrote sparse track_pixels_THM_X/Y/Z.dat plus dense WDM, Binary, and time-domain diagnostic files\n");
            }
            else
            {
                printf("combined_wdm_outputs kept sparse track pixels in memory and wrote dense/debug diagnostic files\n");
            }
        }
        else if(write_time_domain_files)
        {
            if(write_sparse_files)
            {
                printf("combined_wdm_outputs wrote sparse track_pixels_THM_X/Y/Z.dat plus THM_all_X/Y/Ztime(.tukey).dat\n");
            }
            else
            {
                printf("combined_wdm_outputs wrote THM_all_X/Y/Ztime(.tukey).dat\n");
            }
        }
        else
        {
            if(write_sparse_files)
            {
                printf("combined_wdm_outputs wrote sparse track_pixels_THM_X/Y/Z.dat; rerun with --diagnostics for dense/plotting files\n");
            }
            else
            {
                printf("combined_wdm_outputs kept sparse track pixels in memory; rerun without --no-sparse-files to write track_pixels_THM_X/Y/Z.dat\n");
            }
        }
    }

combined_cleanup:
    for(idx=0; idx<Nchan*stream_capacity; idx++)
    {
        if(ATspline[idx] != NULL) gsl_spline_free(ATspline[idx]);
        if(PTspline[idx] != NULL) gsl_spline_free(PTspline[idx]);
        if(ATacc[idx] != NULL) gsl_interp_accel_free(ATacc[idx]);
        if(PTacc[idx] != NULL) gsl_interp_accel_free(PTacc[idx]);
    }
    free(ATacc);
    free(PTacc);
    free(ATspline);
    free(PTspline);
    gsl_interp_accel_free(AFacc);
    gsl_interp_accel_free(PFacc);
    gsl_interp_accel_free(TAacc);
    gsl_spline_free(TAspline);
    free_double_tensor(Achan, Nchan, stream_capacity);
    free_double_tensor(phase_td, Nchan, stream_capacity);
    free_double_tensor(freq_track, Nchan, stream_capacity);
    free_double_tensor(setup_carrier, Nchan, stream_capacity);
    if(direct22_residual != NULL)
        free_double_tensor(direct22_residual, Nchan, stream_capacity);
    if(direct22_frequency != NULL)
        free_double_tensor(direct22_frequency, Nchan, stream_capacity);
    if(direct22_fdot != NULL)
        free_double_tensor(direct22_fdot, Nchan, stream_capacity);
    free_int_vector(direct22_split_valid);
    free_double_vector(direct22_carrier_phase);
    free_double_vector(direct22_carrier_frequency);
    free_double_vector(direct22_carrier_dfdu);
    free_double_vector(direct22_plan_d1);
    free_double_vector(direct22_plan_d2);
    free_double_vector(direct22_residual_d1);
    free_double_vector(direct22_residual_d2);
    free(stream_carrier);
    free_int_vector(stream_parent);
    free_int_vector(stream_piece);
    free_double_vector(Xmode);
    free_double_vector(Ymode);
    free_double_vector(Zmode);
    free_double_vector(Xfmode);
    free_double_vector(Yfmode);
    free_double_vector(Zfmode);
    free_double_vector(phi_ref);
    free_double_vector(phi_res);
    free_double_vector(omega_plan);
    free_double_vector(direct_endpoint_time);
    free_double_vector(direct_full_reference_y);
    free_double_vector(direct_full_reference_z);
    if(direct_endpoint_wave != NULL)
        free_double_matrix(direct_endpoint_wave, Nchan);
    if(direct_endpoint_quadrature != NULL)
        free_double_matrix(direct_endpoint_quadrature, Nchan);
    if(owns_wdm_work)
    {
        free_int_vector(nmid);
        free_int_vector(nsize);
        free_int_vector(nmid_endpoint);
        free_int_vector(nsize_endpoint);
        free_int_vector(listn);
        free_int_vector(listm);
        free_double_vector(wdmwave);
        free_double_matrix(wdm, Nt);
        free_double_vector(freq);
        free_double_vector(phase);
        free_double_vector(Amp);
        free_double_vector(Ataper);
        free_double_vector(hfull);
    }
}

void write_thm_all_carrier_wdm_combined_fft(int Ns, double *response_time, double *plan_time, double **mode_freq, double *params, const IMRPhenomTHM *model, int ncarriers, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, struct wdmshape *wdms, int use_join_override, double join_time_override, double rise_override)
{
    generate_thm_all_carrier_wdm_combined_fft(Ns, response_time, plan_time,
                                              mode_freq, params, model,
                                              ncarriers, carrier,
                                              SLacc, SLspline, SPacc, SPspline,
                                              SVacc, SVspline,
                                              Aacc, Aspline, Pacc, Pspline,
                                              wdms, use_join_override,
                                              join_time_override,
                                              rise_override, 0, 0, 0, 0,
                                              0,
                                              NULL, NULL, 0.0, 0.0, 1, 0,
                                              NULL, NULL, NULL);
}

void write_thm_single_carrier_wdm_diagnostic(int carrier_index, int Ns, double *response_time, double *plan_time, double **mode_freq, double *params, const IMRPhenomTHM *model, const THMFoldedCarrier *carrier, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel **Aacc, gsl_spline **Aspline, gsl_interp_accel **Pacc, gsl_spline **Pspline, struct wdmshape *wdms, int use_join_override, double join_time_override, double rise_override)
{
    const THMFoldedCarrier *c;
    THMFoldedCarrier ap_carrier;
    const IMRPhenomTHMModeState *mode_state;
    char label[32];
    char filename[128];
    int N, Ntsmax, Nts, Nshortfft, Nspa_all, Nspa, Nspa1p_all, Nspa1m_all, Nspa1p, Nspa1m;
    int mode_index;
    int i, j;
    int *nmid, *nsize, *nmid_inverse, *nsize_inverse, *listn, *listm;
    int Np, NpFFT, NpFFTTukey, NpSPA, clean_layers, clean_pixels;
    double Mtot, Mc, tc, Tobs;
    double t_transition, fring, fdamp;
    double alpha, t, hval;
    double setup[7];
    double *Xmode, *Ymode, *Zmode, *Xfmode, *Yfmode, *Zfmode;
    double *AX, *phiX, *phi_ref, *phase_td, *omega_plan, *freq_track;
    double *freq_ref_track, *plan_jacobian, *freq_correction, *freq_split_track;
    double *fdot_split_track, *freq_fullphase_track, *fdot_fullphase_track;
    double *carrier_phase_direct, *residual_phase_direct;
    double *carrier_frequency_direct, *carrier_dfdu_direct;
    double *plan_d1_direct, *plan_d2_direct;
    double *residual_d1_direct, *residual_d2_direct;
    double *freq_split_direct, *fdot_split_direct;
    double *freq, *phase, *Amp, *short_htime;
    double *freq_shortfft, *phase_shortfft, *Amp_shortfft;
    double *freq_spa_all, *phase_spa_all, *Amp_spa_all, *time_spa_all, *fdot_spa_all;
    double *freq_spa, *phase_spa, *Amp_spa;
    double *freq_spa1p_all, *phase_spa1p_all, *Amp_spa1p_all, *time_spa1p_all, *fdot_spa1p_all, *delta1p_all, *rel1p_all;
    double *freq_spa1m_all, *phase_spa1m_all, *Amp_spa1m_all, *time_spa1m_all, *fdot_spa1m_all, *delta1m_all, *rel1m_all;
    double *freq_spa1p, *phase_spa1p, *Amp_spa1p;
    double *freq_spa1m, *phase_spa1m, *Amp_spa1m;
    double *wdmwave, *wdmwavefft, *wdmwavefft_tukey, *wdmwavespa, *hfft, *hfft_tukey, *hfft_split, *hfull;
    double **wdm, **wdmfft, **wdmfft_tukey, **wdmspa, **wdmclean;
    int *valid_spa_all, *valid_spa1p_all, *valid_spa1m_all;
    int direct_split_ok;
    gsl_interp_accel *ATacc, *PTacc, *AFacc, *PFacc, *AFspaacc, *PFspaacc;
    gsl_interp_accel *AFspa1pacc, *PFspa1pacc, *AFspa1macc, *PFspa1macc;
    gsl_interp_accel *AFshortfftacc, *PFshortfftacc;
    gsl_spline *ATspline, *PTspline, *AFspline, *PFspline, *AFspaspline, *PFspaspline;
    gsl_spline *AFspa1pspline, *PFspa1pspline, *AFspa1mspline, *PFspa1mspline;
    gsl_spline *AFshortfftspline, *PFshortfftspline;
    FILE *out;
    clock_t start, end;
    double single_tdi_time, spa_freq_time, spa_wdm_time, fft_build_time, fft_tukey_build_time, fft_wdm_time, fft_tukey_wdm_time, spaonly_wdm_time, clean_wdm_time;
    double direct_split_setup_time, direct_split_fft_time;

    if(carrier == NULL || model == NULL || wdms == NULL ||
       response_time == NULL || plan_time == NULL ||
       carrier_index < 0 || Ns < 4)
    {
        fprintf(stderr, "Warning: skipping THM WDM diagnostic because the carrier request is invalid.\n");
        return;
    }

    Nspa_all = 0;
    Nspa = 0;
    Nspa1p_all = 0;
    Nspa1m_all = 0;
    Nspa1p = 0;
    Nspa1m = 0;
    Nshortfft = 0;
    NpSPA = 0;
    NpFFTTukey = 0;
    clean_layers = 0;
    clean_pixels = 0;
    spaonly_wdm_time = 0.0;
    clean_wdm_time = 0.0;
    fft_tukey_build_time = 0.0;
    fft_tukey_wdm_time = 0.0;
    direct_split_ok = 0;
    direct_split_setup_time = 0.0;
    direct_split_fft_time = 0.0;
    carrier_phase_direct = NULL;
    residual_phase_direct = NULL;
    carrier_frequency_direct = NULL;
    carrier_dfdu_direct = NULL;
    plan_d1_direct = NULL;
    plan_d2_direct = NULL;
    residual_d1_direct = NULL;
    residual_d2_direct = NULL;
    freq_split_direct = NULL;
    fdot_split_direct = NULL;
    hfft_split = NULL;
    wdmspa = NULL;
    wdmclean = NULL;
    wdmfft_tukey = NULL;
    AFspaacc = NULL;
    PFspaacc = NULL;
    AFspaspline = NULL;
    PFspaspline = NULL;
    AFspa1pacc = NULL;
    PFspa1pacc = NULL;
    AFspa1macc = NULL;
    PFspa1macc = NULL;
    AFshortfftacc = NULL;
    PFshortfftacc = NULL;
    AFspa1pspline = NULL;
    PFspa1pspline = NULL;
    AFspa1mspline = NULL;
    PFspa1mspline = NULL;
    AFshortfftspline = NULL;
    PFshortfftspline = NULL;

    c = &carrier[carrier_index];
    mode_index = c->mode_index;
    if(mode_index < 0 || mode_index >= model->nmodes)
    {
        fprintf(stderr, "Warning: skipping THM WDM diagnostic because carrier %d has invalid mode index %d.\n",
                carrier_index, mode_index);
        return;
    }
    mode_state = &model->modes[mode_index];
    ap_carrier = *c;
    /*
     * The main THM time-domain diagnostic keeps a mode-by-mode quadrature,
     * h_+m(phi_+m+pi/2)+h_-m(phi_-m+pi/2), because that explicit basis is useful
     * for future TPHM/twist-up work.  For the folded-carrier WDM path, however,
     * extractAP() needs the quadrature of the already-folded track:
     *
     *     h_pair(phi+pi/2) for h_pair = A[C cos(phi)+S sin(phi)].
     *
     * Use that local convention here without changing the public Xf/Yf/Zf
     * diagnostics written by the main THM path.
     */
    ap_carrier.hpf_cos = ap_carrier.hp_sin;
    ap_carrier.hpf_sin = -ap_carrier.hp_cos;
    ap_carrier.hcf_cos = ap_carrier.hc_sin;
    ap_carrier.hcf_sin = -ap_carrier.hc_cos;

    N = Nt*Nf;
    Tobs = wdms->Tobs;
    Mtot = params[0]+params[1];
    Mc = pow(params[0]*params[1], 3.0/5.0)/pow(Mtot, 1.0/5.0);
    tc = params[5];
    snprintf(label, sizeof(label), "%d%d", c->ell, abs(c->emm));

    Xmode = double_vector(Ns);
    Ymode = double_vector(Ns);
    Zmode = double_vector(Ns);
    Xfmode = double_vector(Ns);
    Yfmode = double_vector(Ns);
    Zfmode = double_vector(Ns);
    AX = double_vector(Ns);
    phiX = double_vector(Ns);
    phi_ref = double_vector(Ns);
    phase_td = double_vector(Ns);
    omega_plan = double_vector(Ns);
    freq_track = double_vector(Ns);
    freq_ref_track = double_vector(Ns);
    plan_jacobian = double_vector(Ns);
    freq_correction = double_vector(Ns);
    freq_split_track = double_vector(Ns);
    fdot_split_track = double_vector(Ns);
    freq_fullphase_track = double_vector(Ns);
    fdot_fullphase_track = double_vector(Ns);

    start = clock();
    fast_response_thm(response_time, Ns, params, 1, &ap_carrier,
                      SLacc, SLspline, SPacc, SPspline, SVacc, SVspline,
                      Aacc, Aspline, Pacc, Pspline,
                      Xmode, Ymode, Zmode, Xfmode, Yfmode, Zfmode);
    end = clock();
    single_tdi_time = ((double)(end-start))/CLOCKS_PER_SEC;

    for(i=0; i<Ns; i++)
    {
        /*
         * Match the original 22 TDI convention: the response itself is
         * evaluated on the barycenter/source-time labels response_time, but
         * extractAP() removes the fast carrier phase at the detector/reference
         * time plan_time.  Using mode_phase[mode_index][i] here instead removes
         * the source-time carrier and leaves an artificial residual ramp of
         * roughly the light-travel time times df/dt.  That ramp is especially
         * ugly near the transfer-frequency notches, where the amplitude is
         * small and the phase branch is already delicate.
         */
        phi_ref[i] = thm_reference_phase_eval(
            Pspline[mode_index], plan_time[i], Pacc[mode_index]);
    }
    extractAP(Ns, AX, phiX, Xmode, Xfmode, phi_ref);
    unwrap(Ns, phiX);
    for(i=0; i<Ns; i++)
    {
        phase_td[i] = phi_ref[i]+phiX[i];
    }
    for(i=0; i<Ns; i++)
    {
        double dphidt;

        dphidt = nonuniform_phase_derivative(i, Ns, response_time, phase_td);
        freq_track[i] = fabs(dphidt)/(2.0*M_PI);
        if(!isfinite(freq_track[i]) || freq_track[i] <= 0.0)
        {
            freq_track[i] = fabs(mode_freq[mode_index][i]);
        }
        omega_plan[i] = 2.0*M_PI*freq_track[i];
    }
    build_split_reference_frequency_track(Ns, response_time, plan_time,
                                          mode_freq[mode_index],
                                          phiX, phase_td,
                                          freq_ref_track,
                                          plan_jacobian,
                                          freq_correction,
                                          freq_split_track,
                                          fdot_split_track,
                                          freq_fullphase_track,
                                          fdot_fullphase_track);

    if(c->ell == 2 && abs(c->emm) == 2)
    {
        carrier_phase_direct = double_vector(Ns);
        residual_phase_direct = double_vector(Ns);
        carrier_frequency_direct = double_vector(Ns);
        carrier_dfdu_direct = double_vector(Ns);
        plan_d1_direct = double_vector(Ns);
        plan_d2_direct = double_vector(Ns);
        residual_d1_direct = double_vector(Ns);
        residual_d2_direct = double_vector(Ns);
        freq_split_direct = double_vector(Ns);
        fdot_split_direct = double_vector(Ns);

        start = clock();
        direct_split_ok = build_direct22_residual_spa_tracks(
                Ns, response_time, plan_time, phi_ref, phiX, phase_td,
                params, model, mode_index,
                carrier_phase_direct, residual_phase_direct,
                carrier_frequency_direct, carrier_dfdu_direct,
                plan_d1_direct, plan_d2_direct,
                residual_d1_direct, residual_d2_direct,
                freq_split_direct, fdot_split_direct);
        end = clock();
        direct_split_setup_time = ((double)(end-start))/CLOCKS_PER_SEC;
        printf("directcarrier_split_setup mode %s seconds %.9f valid %d intrinsic_frequency_calls %d extra_tdi_calls 0\n",
               label, direct_split_setup_time, direct_split_ok, 5*Ns);

        snprintf(filename, sizeof(filename),
                 "THM_mode%s_TDI_AP_directcarrier_split.dat", label);
        out = fopen(filename, "w");
        if(out != NULL)
        {
            fprintf(out, "# t_response_s t_plan_s carrier_phase_direct carrier_phase_spline carrier_phase_direct_minus_spline residual_direct phase_total reconstructed_phase reconstruction_error amplitude_signed carrier_f_Hz carrier_dfdu_Hz_per_s plan_d1 plan_d2_per_s residual_d1_rad_per_s residual_d2_rad_per_s2 freq_split_Hz fdot_split_Hz_per_s freq_fullphase_Hz fdot_fullphase_Hz_per_s stencil_hint_max_s\n");
            fprintf(out, "# Phi_TDI = Phi22_direct(plan_time)+residual_direct; the split is exact at each AP node before interpolating the small residual.\n");
            fprintf(out, "# carrier_dfdu is evaluated from direct intrinsic frequency calls on a local five-point stencil; no additional TDI calls are used.\n");
            for(i=0; i<Ns; i++)
            {
                double reconstructed = carrier_phase_direct[i]+
                                       residual_phase_direct[i];
                fprintf(out, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                        response_time[i], plan_time[i],
                        carrier_phase_direct[i], phi_ref[i],
                        carrier_phase_direct[i]-phi_ref[i],
                        residual_phase_direct[i], phase_td[i], reconstructed,
                        reconstructed-phase_td[i], AX[i],
                        carrier_frequency_direct[i], carrier_dfdu_direct[i],
                        plan_d1_direct[i], plan_d2_direct[i],
                        residual_d1_direct[i], residual_d2_direct[i],
                        freq_split_direct[i], fdot_split_direct[i],
                        freq_fullphase_track[i], fdot_fullphase_track[i],
                        100.0);
            }
            fclose(out);
        }
    }

    /*
     * Reuse the 22 SPA+short-FFT WDM machinery one folded carrier at a time.
     * The positive-m mode phase is the carrier phase.  extractAP() turns the
     * X-channel TDI quadratures into a signed amplitude and a residual phase,
     * so the total phase passed to ftran() is phi_mode + phi_X.
     *
     * freq_track is the raw local frequency inferred from the TDI-extracted
     * phase, and is what the WDM pixel finder should cover.  The diagnostic
     * file below keeps a compatibility column, but it is now just a copy of
     * freq_track rather than a synthetic monotone proxy.
     */
    t_transition = tc+Mtot*tCUT_Freq;
    if(c->ell == 2 && abs(c->emm) == 2)
    {
        fring = model->mode22.omegaRING/(2.0*M_PI*Mtot);
        fdamp = model->mode22.alpha1RD/Mtot;
    }
    else
    {
        fring = mode_state->phase.omegaRING/(2.0*M_PI*Mtot);
        fdamp = mode_state->phase.alpha1RD/Mtot;
    }
    snprintf(filename, sizeof(filename), "THM_mode%s_TDI_AP.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# t_s phi_ref phi_residual phase_total amplitude_signed freq_track_raw_Hz freq_track_copy_Hz X X_quadrature X_reconstructed\n");
        for(i=0; i<Ns; i++)
        {
            double xrec = AX[i]*cos(phase_td[i]);
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    response_time[i], phi_ref[i], phiX[i], phase_td[i], AX[i],
                    freq_track[i], freq_track[i], Xmode[i], Xfmode[i], xrec);
        }
        fclose(out);
    }
    snprintf(filename, sizeof(filename), "THM_mode%s_TDI_AP_refsplit.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# t_response_s t_plan_s phi_ref phi_residual phase_total amplitude_signed freq_ref_times_jac_Hz plan_jacobian freq_residual_Hz freq_refsplit_Hz freq_fullphase_Hz freq_refsplit_minus_fullphase_Hz fdot_refsplit_Hz_per_s fdot_fullphase_Hz_per_s\n");
        fprintf(out, "# Diagnostic: freq_refsplit = f_intrinsic(plan_time)*d(plan_time)/d(response_time) + (1/2pi)*d(phi_residual)/d(response_time).\n");
        fprintf(out, "# The full post-TDI phase is still used for the SPA phase; only the stationary-frequency track is decomposed.\n");
        for(i=0; i<Ns; i++)
        {
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                    response_time[i], plan_time[i], phi_ref[i], phiX[i],
                    phase_td[i], AX[i],
                    freq_ref_track[i], plan_jacobian[i], freq_correction[i],
                    freq_split_track[i], freq_fullphase_track[i],
                    freq_split_track[i]-freq_fullphase_track[i],
                    fdot_split_track[i], fdot_fullphase_track[i]);
        }
        fclose(out);
    }
    transformplan_custom(Mc, Mtot,
                         linear_interp_clamped(Ns, plan_time,
                                               response_time, tc),
                         response_time, omega_plan,
                         linear_interp_clamped(Ns, plan_time,
                                               response_time, t_transition),
                         fring, fdamp, response_time[Ns-1], setup,
                         use_join_override, join_time_override, rise_override);
    printf("single_carrier_short_fft_plan mode %s start %.15e end %.15e duration %.15e dt %.15e samples %.0f rise %.15e join %.15e nyquist %.15e fmax_bin %.15e fring %.15e fdamp %.15e\n",
           label, setup[2], setup[2]+setup[0]*setup[1], setup[0]*setup[1],
           setup[0], setup[1], setup[3], response_time[(int)setup[4]],
           0.5/setup[0], setup[6]/(setup[0]*setup[1]), fring, fdamp);
    write_intrinsic_spline_frequency_diagnostic(label, setup, response_time, Ns,
                                                mode_freq[mode_index],
                                                Pacc[mode_index],
                                                Pspline[mode_index]);
    Ntsmax = (int)(setup[4]+setup[6])+32;
    if(Ntsmax < 32) Ntsmax = 32;

    freq = double_vector(Ntsmax);
    phase = double_vector(Ntsmax);
    Amp = double_vector(Ntsmax);
    short_htime = double_vector((int)(setup[1]));
    freq_shortfft = double_vector(Ntsmax);
    phase_shortfft = double_vector(Ntsmax);
    Amp_shortfft = double_vector(Ntsmax);
    freq_spa_all = double_vector(Ns);
    phase_spa_all = double_vector(Ns);
    Amp_spa_all = double_vector(Ns);
    time_spa_all = double_vector(Ns);
    fdot_spa_all = double_vector(Ns);
    valid_spa_all = int_vector(Ns);
    freq_spa = double_vector(Ns);
    phase_spa = double_vector(Ns);
    Amp_spa = double_vector(Ns);
    freq_spa1p_all = double_vector(Ns);
    phase_spa1p_all = double_vector(Ns);
    Amp_spa1p_all = double_vector(Ns);
    time_spa1p_all = double_vector(Ns);
    fdot_spa1p_all = double_vector(Ns);
    delta1p_all = double_vector(Ns);
    rel1p_all = double_vector(Ns);
    valid_spa1p_all = int_vector(Ns);
    freq_spa1m_all = double_vector(Ns);
    phase_spa1m_all = double_vector(Ns);
    Amp_spa1m_all = double_vector(Ns);
    time_spa1m_all = double_vector(Ns);
    fdot_spa1m_all = double_vector(Ns);
    delta1m_all = double_vector(Ns);
    rel1m_all = double_vector(Ns);
    valid_spa1m_all = int_vector(Ns);
    freq_spa1p = double_vector(Ns);
    phase_spa1p = double_vector(Ns);
    Amp_spa1p = double_vector(Ns);
    freq_spa1m = double_vector(Ns);
    phase_spa1m = double_vector(Ns);
    Amp_spa1m = double_vector(Ns);

    ATacc = gsl_interp_accel_alloc();
    PTacc = gsl_interp_accel_alloc();
    ATspline = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
    PTspline = gsl_spline_alloc(THM_AP_SPLINE_TYPE, Ns);
    gsl_spline_init(ATspline, response_time, AX, Ns);
    gsl_spline_init(PTspline, response_time, phase_td, Ns);

    Nspa_all = build_full_spa_samples(response_time, Ns, Tobs,
                                      ATacc, ATspline, PTacc, PTspline,
                                      freq_spa_all, phase_spa_all,
                                      Amp_spa_all, time_spa_all,
                                      fdot_spa_all, valid_spa_all);
    Nspa = 0;
    for(i=0; i<Nspa_all; i++)
    {
        if(valid_spa_all[i] &&
           isfinite(freq_spa_all[i]) &&
           isfinite(phase_spa_all[i]) &&
           isfinite(Amp_spa_all[i]) &&
           (Nspa == 0 || freq_spa_all[i] > freq_spa[Nspa-1]))
        {
            freq_spa[Nspa] = freq_spa_all[i];
            phase_spa[Nspa] = phase_spa_all[i];
            Amp_spa[Nspa] = Amp_spa_all[i];
            Nspa++;
        }
    }
    snprintf(filename, sizeof(filename), "ch_THM_%s_spaonly.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# f_Hz phase_SPA amplitude_SPA_signed\n");
        fprintf(out, "# Pure-SPA, monotone single-branch subset used for wtran_spaonly_THM_%s.dat.\n", label);
        for(i=0; i<Nspa; i++) fprintf(out, "%.15e %.15e %.15e\n", freq_spa[i], phase_spa[i], Amp_spa[i]);
        fclose(out);
    }
    printf("single_carrier_full_spa mode %s samples_all %d monotone_valid %d dropped_or_invalid %d\n",
           label, Nspa_all, Nspa, Nspa_all-Nspa);

    Nspa1p_all = build_first_order_spa_samples(response_time, Ns, Tobs,
                                               ATacc, ATspline, PTacc, PTspline,
                                               1.0, 15,
                                               freq_spa1p_all, phase_spa1p_all,
                                               Amp_spa1p_all, time_spa1p_all,
                                               fdot_spa1p_all, delta1p_all,
                                               rel1p_all, valid_spa1p_all);
    Nspa1p = 0;
    {
        double theta_prev = 0.0;
        int have_theta = 0;
    for(i=0; i<Nspa1p_all; i++)
    {
        if(valid_spa1p_all[i] &&
           isfinite(freq_spa1p_all[i]) &&
           isfinite(phase_spa1p_all[i]) &&
           isfinite(Amp_spa1p_all[i]) &&
           isfinite(phase_spa_all[i]) &&
           (Nspa1p == 0 || freq_spa1p_all[i] > freq_spa1p[Nspa1p-1]))
        {
            double theta;

            theta = phase_spa1p_all[i]-phase_spa_all[i];
            if(have_theta)
            {
                while(theta-theta_prev > M_PI) theta -= 2.0*M_PI;
                while(theta-theta_prev < -M_PI) theta += 2.0*M_PI;
            }
            freq_spa1p[Nspa1p] = freq_spa1p_all[i];
            phase_spa1p[Nspa1p] = phase_spa_all[i]+theta;
            Amp_spa1p[Nspa1p] = Amp_spa1p_all[i];
            theta_prev = theta;
            have_theta = 1;
            Nspa1p++;
        }
    }
    }
    snprintf(filename, sizeof(filename), "ch_THM_%s_spa1plus.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# f_Hz phase_SPA1_plus amplitude_SPA1 delta1 delta1_over_A\n");
        fprintf(out, "# First post-adiabatic SPA correction using bracket A+i*Delta1.\n");
        for(i=0; i<Nspa1p; i++)
        {
            int q;
            q = -1;
            for(j=0; j<Nspa1p_all; j++)
            {
                if(freq_spa1p_all[j] == freq_spa1p[i])
                {
                    q = j;
                    break;
                }
            }
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e\n",
                    freq_spa1p[i], phase_spa1p[i], Amp_spa1p[i],
                    q >= 0 ? delta1p_all[q] : NAN,
                    q >= 0 ? rel1p_all[q] : NAN);
        }
        fclose(out);
    }

    Nspa1m_all = build_first_order_spa_samples(response_time, Ns, Tobs,
                                               ATacc, ATspline, PTacc, PTspline,
                                               -1.0, 15,
                                               freq_spa1m_all, phase_spa1m_all,
                                               Amp_spa1m_all, time_spa1m_all,
                                               fdot_spa1m_all, delta1m_all,
                                               rel1m_all, valid_spa1m_all);
    Nspa1m = 0;
    {
        double theta_prev = 0.0;
        int have_theta = 0;
    for(i=0; i<Nspa1m_all; i++)
    {
        if(valid_spa1m_all[i] &&
           isfinite(freq_spa1m_all[i]) &&
           isfinite(phase_spa1m_all[i]) &&
           isfinite(Amp_spa1m_all[i]) &&
           isfinite(phase_spa_all[i]) &&
           (Nspa1m == 0 || freq_spa1m_all[i] > freq_spa1m[Nspa1m-1]))
        {
            double theta;

            theta = phase_spa1m_all[i]-phase_spa_all[i];
            if(have_theta)
            {
                while(theta-theta_prev > M_PI) theta -= 2.0*M_PI;
                while(theta-theta_prev < -M_PI) theta += 2.0*M_PI;
            }
            freq_spa1m[Nspa1m] = freq_spa1m_all[i];
            phase_spa1m[Nspa1m] = phase_spa_all[i]+theta;
            Amp_spa1m[Nspa1m] = Amp_spa1m_all[i];
            theta_prev = theta;
            have_theta = 1;
            Nspa1m++;
        }
    }
    }
    snprintf(filename, sizeof(filename), "ch_THM_%s_spa1minus.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# f_Hz phase_SPA1_minus amplitude_SPA1 delta1 delta1_over_A\n");
        fprintf(out, "# Opposite-sign diagnostic using bracket A-i*Delta1, included to catch Fourier-convention sign mistakes.\n");
        for(i=0; i<Nspa1m; i++)
        {
            int q;
            q = -1;
            for(j=0; j<Nspa1m_all; j++)
            {
                if(freq_spa1m_all[j] == freq_spa1m[i])
                {
                    q = j;
                    break;
                }
            }
            fprintf(out, "%.15e %.15e %.15e %.15e %.15e\n",
                    freq_spa1m[i], phase_spa1m[i], Amp_spa1m[i],
                    q >= 0 ? delta1m_all[q] : NAN,
                    q >= 0 ? rel1m_all[q] : NAN);
        }
        fclose(out);
    }
    printf("single_carrier_spa1 mode %s plus_valid %d minus_valid %d\n",
           label, Nspa1p, Nspa1m);

    start = clock();
    Nts = ftran(setup, response_time, Ns, Tobs, ATacc, ATspline, PTacc, PTspline,
                freq, phase, Amp, short_htime);
    end = clock();
    spa_freq_time = ((double)(end-start))/CLOCKS_PER_SEC;
    {
        int kept = 1;
        int dropped = 0;
        for(i=1; i<Nts; i++)
        {
            if(freq[i] > freq[kept-1])
            {
                if(kept != i)
                {
                    freq[kept] = freq[i];
                    phase[kept] = phase[i];
                    Amp[kept] = Amp[i];
                }
                kept++;
            }
            else
            {
                dropped++;
            }
        }
        if(dropped > 0)
        {
            printf("single_carrier_wdm dropped %d non-increasing ftran samples for mode %s\n",
                   dropped, label);
        }
        Nts = kept;
        if(Nts < 4)
        {
            fprintf(stderr, "Warning: skipping THM WDM diagnostic because ftran returned too few monotonic samples.\n");
            return;
        }
    }

    AFacc = gsl_interp_accel_alloc();
    PFacc = gsl_interp_accel_alloc();
    /*
     * The frequency-domain SPA samples can be sparse just below a TDI transfer
     * feature.  A natural cubic spline through the signed Fourier amplitude can
     * overshoot badly in those intervals even when the time-domain waveform and
     * the SPA samples themselves are benign.  Use Akima for the amplitude to
     * preserve the local shape; keep the phase cubic, where the smooth unwrap is
     * the quantity we want.
     */
    AFspline = gsl_spline_alloc(gsl_interp_akima, Nts);
    PFspline = gsl_spline_alloc(gsl_interp_cspline, Nts);
    gsl_spline_init(AFspline, freq, Amp, Nts);
    gsl_spline_init(PFspline, freq, phase, Nts);

    write_spline_frequency_diagnostic(label, setup, response_time, Ns, freq_track,
                                      PTacc, PTspline);
    write_short_fft_spa_complex_diagnostic(label, setup, Tobs, response_time, Ns,
                                           ATacc, ATspline, PTacc, PTspline,
                                           short_htime);

    nmid = int_vector(Nf);
    nsize = int_vector(Nf);
    nmid_inverse = int_vector(Nf);
    nsize_inverse = int_vector(Nf);
    listn = int_vector(N);
    listm = int_vector(N);
    wdmwave = double_vector(N);
    wdmwavefft = double_vector(N);
    wdmwavefft_tukey = double_vector(N);
    wdmwavespa = double_vector(N);
    hfft = double_vector(N);
    hfft_tukey = double_vector(N);
    if(direct_split_ok) hfft_split = double_vector(N);

    for(i=0; i<Nf; i++)
    {
        nmid_inverse[i] = -1;
        nsize_inverse[i] = 0;
    }
    {
        int strictly_increasing = 1;
        for(i=1; i<Ns; i++)
        {
            if(freq_track[i] <= freq_track[i-1])
            {
                strictly_increasing = 0;
                break;
            }
        }
        if(strictly_increasing)
        {
            WDMpixels(Ns, response_time, freq_track, nmid_inverse,
                      nsize_inverse, N, wdms);
        }
    }
    WDMpixelsTimeScan(Ns, response_time, freq_track, nmid, nsize, N, wdms);
    {
        double fmax_spectrum = freq[Nts-1];
        double tail_time = tc;
        double tail_amp = -1.0;

        for(i=0; i<Nts; i++)
        {
            if(isfinite(freq[i]) && freq[i] > fmax_spectrum) fmax_spectrum = freq[i];
        }
        for(i=0; i<Ns; i++)
        {
            double amp_abs = fabs(AX[i]);
            if(isfinite(amp_abs) && amp_abs > tail_amp)
            {
                tail_amp = amp_abs;
                tail_time = response_time[i];
            }
        }
        WDMpixelsAddMergerFrequencyTail(nmid, nsize, fmax_spectrum, tail_time, wdms);
    }
    {
        int inverse_layers = 0;
        int inverse_volume = 0;
        int scan_layers = 0;
        int scan_volume = 0;
        int mm;

        for(mm=1; mm<Nf; mm++)
        {
            if(nmid_inverse[mm] >= 0 && nsize_inverse[mm] > 0)
            {
                inverse_layers++;
                inverse_volume += nsize_inverse[mm];
            }
            if(nmid[mm] >= 0 && nsize[mm] > 0)
            {
                scan_layers++;
                scan_volume += nsize[mm];
            }
        }
        printf("single_carrier_wdm_pixel_volume inverse_spline layers %d volume %d time_scan layers %d volume %d\n",
               inverse_layers, inverse_volume, scan_layers, scan_volume);
    }

    start = clock();
    WDMtrack(wdmwave, listn, listm, &Np, Nts, N, nmid, nsize, tc,
             freq, AFacc, AFspline, PFacc, PFspline, wdms);
    end = clock();
    spa_wdm_time = ((double)(end-start))/CLOCKS_PER_SEC;

    wdm = double_matrix(Nt, Nf+1);
    unpack_wdm_track(wdm, listn, listm, wdmwave, Np);

    snprintf(filename, sizeof(filename), "track_pixels_THM_%s.dat", label);
    write_track_pixels(filename, listn, listm, wdmwave, Np);
    snprintf(filename, sizeof(filename), "wtranfast_THM_%s.dat", label);
    write_wdm_matrix(filename, wdm);
    snprintf(filename, sizeof(filename), "BinaryFastTHM_%s.dat", label);
    write_wdm_binary(filename, wdm, wdms);

    snprintf(filename, sizeof(filename), "ch_THM_%s.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# f_Hz phase amplitude\n");
        for(i=0; i<Nts; i++) fprintf(out, "%.15e %.15e %.15e\n", freq[i], phase[i], Amp[i]);
        fclose(out);
    }

    alpha = 2.0*REFERENCE_TUKEY_ROLL_SECONDS/Tobs;
    if(alpha > 1.0) alpha = 1.0;
    if(alpha < 0.0) alpha = 0.0;

    start = clock();
    build_direct_fft_spectrum(hfft, N, response_time[Ns-1], ATacc, ATspline, PTacc, PTspline);
    end = clock();
    fft_build_time = ((double)(end-start))/CLOCKS_PER_SEC;

    start = clock();
    build_direct_fft_spectrum_tukey(hfft_tukey, N, response_time[Ns-1], alpha,
                                    ATacc, ATspline, PTacc, PTspline);
    end = clock();
    fft_tukey_build_time = ((double)(end-start))/CLOCKS_PER_SEC;

    /*
     * Short-FFT-only diagnostic.  This deliberately ignores the SPA samples and
     * asks how well the tapered endpoint FFT, by itself, matches the
     * full-observation FFT in each WDM frequency band.  If a feature appears in
     * this file before it appears in the hybrid WDM output, the issue is in the
     * window/interface Fourier input rather than in the Meyer-WDM machinery.
     */
    Nshortfft = ftran_endpoint_fft(setup, Tobs, short_htime, 0.0,
                                   freq_shortfft, phase_shortfft, Amp_shortfft);
    snprintf(filename, sizeof(filename), "ch_THM_%s_shortfft.dat", label);
    out = fopen(filename, "w");
    if(out != NULL)
    {
        fprintf(out, "# f_Hz phase_short_fft amplitude_short_fft_signed\n");
        fprintf(out, "# Endpoint short FFT only; no SPA samples are included.\n");
        fprintf(out, "# t_start_s %.15e t_end_s %.15e rise_s %.15e dt_s %.15e samples %.0f\n",
                setup[2], setup[2]+setup[0]*setup[1], setup[3],
                setup[0], setup[1]);
        for(i=0; i<Nshortfft; i++)
        {
            fprintf(out, "%.15e %.15e %.15e\n",
                    freq_shortfft[i], phase_shortfft[i], Amp_shortfft[i]);
        }
        fclose(out);
    }
    if(Nshortfft >= 4)
    {
        AFshortfftacc = gsl_interp_accel_alloc();
        PFshortfftacc = gsl_interp_accel_alloc();
        AFshortfftspline = gsl_spline_alloc(gsl_interp_akima, Nshortfft);
        PFshortfftspline = gsl_spline_alloc(gsl_interp_cspline, Nshortfft);
        gsl_spline_init(AFshortfftspline, freq_shortfft, Amp_shortfft, Nshortfft);
        gsl_spline_init(PFshortfftspline, freq_shortfft, phase_shortfft, Nshortfft);

        snprintf(filename, sizeof(filename), "THM_mode%s_shortfft_fourier_band_match_X.dat", label);
        write_fourier_band_comparison(filename, hfft, N, Tobs,
                                      AFshortfftacc, AFshortfftspline,
                                      PFshortfftacc, PFshortfftspline,
                                      freq_shortfft[0], freq_shortfft[Nshortfft-1],
                                      "endpoint_short_fft_only",
                                      wdms);
        snprintf(filename, sizeof(filename), "THM_mode%s_shortfft_fourier_band_match_tukeyref_X.dat", label);
        write_fourier_band_comparison(filename, hfft_tukey, N, Tobs,
                                      AFshortfftacc, AFshortfftspline,
                                      PFshortfftacc, PFshortfftspline,
                                      freq_shortfft[0], freq_shortfft[Nshortfft-1],
                                      "endpoint_short_fft_only",
                                      wdms);
    }
    else
    {
        fprintf(stderr, "Warning: endpoint short-FFT diagnostic for mode %s has only %d samples; skipping Fourier-band output.\n",
                label, Nshortfft);
    }

    /*
     * The formal first post-adiabatic SPA correction has a natural hierarchy.
     * The first contribution, A''/(2 psi2), only needs the second derivative
     * of the slowly varying signed amplitude.  The remaining pieces use psi3
     * and psi4 and are much more exposed to derivative noise on an adaptive
     * grid.  Keep both signs as diagnostics until the Fourier convention is
     * settled; these files are not used by the production WDM path.
     */
    write_first_order_spa_fourier_diagnostic(label, "spa1term1plus",
                                             "first_order_SPA_term1_A2_over_2psi2_plus",
                                             1.0, 1,
                                             Ns, response_time, Tobs,
                                             ATacc, ATspline, PTacc, PTspline,
                                             phase_spa_all, hfft, hfft_tukey,
                                             N, wdms);
    write_first_order_spa_fourier_diagnostic(label, "spa1term1minus",
                                             "first_order_SPA_term1_A2_over_2psi2_minus",
                                             -1.0, 1,
                                             Ns, response_time, Tobs,
                                             ATacc, ATspline, PTacc, PTspline,
                                             phase_spa_all, hfft, hfft_tukey,
                                             N, wdms);
    write_first_order_spa_fourier_diagnostic(label, "spa1term12plus",
                                             "first_order_SPA_terms12_A2_and_A1psi3_plus",
                                             1.0, 3,
                                             Ns, response_time, Tobs,
                                             ATacc, ATspline, PTacc, PTspline,
                                             phase_spa_all, hfft, hfft_tukey,
                                             N, wdms);
    write_first_order_spa_fourier_diagnostic(label, "spa1term12minus",
                                             "first_order_SPA_terms12_A2_and_A1psi3_minus",
                                             -1.0, 3,
                                             Ns, response_time, Tobs,
                                             ATacc, ATspline, PTacc, PTspline,
                                             phase_spa_all, hfft, hfft_tukey,
                                             N, wdms);
    write_first_order_spa_fourier_diagnostic(label, "spa1term124plus",
                                             "first_order_SPA_terms124_no_psi4_plus",
                                             1.0, 11,
                                             Ns, response_time, Tobs,
                                             ATacc, ATspline, PTacc, PTspline,
                                             phase_spa_all, hfft, hfft_tukey,
                                             N, wdms);
    write_first_order_spa_fourier_diagnostic(label, "spa1term124minus",
                                             "first_order_SPA_terms124_no_psi4_minus",
                                             -1.0, 11,
                                             Ns, response_time, Tobs,
                                             ATacc, ATspline, PTacc, PTspline,
                                             phase_spa_all, hfft, hfft_tukey,
                                             N, wdms);

	snprintf(filename, sizeof(filename), "THM_mode%s_full_spa_vs_full_fft_X.dat", label);
	write_full_spa_fft_amp_phase_diagnostic(filename, hfft, N, Tobs,
	                                            Nspa_all, time_spa_all,
	                                            freq_spa_all, fdot_spa_all,
	                                            valid_spa_all, Amp_spa_all,
	                                            phase_spa_all);
	snprintf(filename, sizeof(filename), "THM_mode%s_dense_spa_vs_full_fft_X_1mHz_2mHz.dat", label);
	write_dense_spa_fft_amp_phase_diagnostic(filename, hfft, N, Tobs,
	                                         Ns, response_time,
	                                         ATacc, ATspline, PTacc, PTspline,
	                                         1.0e-3, 2.0e-3, 1.0/Tobs, 0.0);
    {
        int jj_join = (int)setup[4];
        double fdiag_min = 3.5e-4;
        double fdiag_max = 2.0e-2;

        if(jj_join < 0) jj_join = 0;
        if(jj_join >= Ns) jj_join = Ns-1;
        if(isfinite(freq_track[jj_join]) && freq_track[jj_join] > 0.0)
        {
            fdiag_max = 2.0*freq_track[jj_join];
        }
        if(fdiag_max <= fdiag_min) fdiag_max = fdiag_min + 1.0e-3;

        /*
         * Dense full-FFT diagnostic for plotting the handoff region.  The
         * older full_spa_vs_full_fft file samples the full FFT only at SPA
         * stationary-point frequencies, which makes the full FFT look more
         * sparsely sampled than the endpoint FFT.  This file uses the native
         * full-observation frequency spacing, 1/Tobs, up to twice the current
         * SPA/endpoint join frequency.
         */
        snprintf(filename, sizeof(filename), "THM_mode%s_dense_spa_vs_full_fft_X_to2join.dat", label);
        write_dense_spa_fft_amp_phase_diagnostic(filename, hfft, N, Tobs,
                                                 Ns, response_time,
                                                 ATacc, ATspline, PTacc, PTspline,
                                                 fdiag_min, fdiag_max, 1.0/Tobs, 0.0);
        snprintf(filename, sizeof(filename), "THM_mode%s_dense_spa_vs_full_fft_X_tukeyref_to2join.dat", label);
        write_dense_spa_fft_amp_phase_diagnostic(filename, hfft_tukey, N, Tobs,
                                                 Ns, response_time,
                                                 ATacc, ATspline, PTacc, PTspline,
                                                 fdiag_min, fdiag_max, 1.0/Tobs, alpha);
        snprintf(filename, sizeof(filename), "THM_mode%s_dense_refsplit_spa_vs_full_fft_X_to2join.dat", label);
        write_dense_spa_fft_amp_phase_diagnostic_from_tracks(filename, hfft, N, Tobs,
                                                             Ns, response_time,
                                                             ATacc, ATspline,
                                                             PTacc, PTspline,
                                                             freq_split_track,
                                                             fdot_split_track,
                                                             fdiag_min, fdiag_max,
                                                             1.0/Tobs, 0.0,
                                                             "intrinsic carrier frequency plus finite-difference derivative of the TDI residual phase");
        snprintf(filename, sizeof(filename), "THM_mode%s_dense_refsplit_spa_vs_full_fft_X_tukeyref_to2join.dat", label);
        write_dense_spa_fft_amp_phase_diagnostic_from_tracks(filename, hfft_tukey, N, Tobs,
                                                             Ns, response_time,
                                                             ATacc, ATspline,
                                                             PTacc, PTspline,
                                                             freq_split_track,
                                                             fdot_split_track,
                                                             fdiag_min, fdiag_max,
                                                             1.0/Tobs, alpha,
                                                             "intrinsic carrier frequency plus finite-difference derivative of the TDI residual phase; SPA amplitude also carries the reference Tukey window");

        if(direct_split_ok && hfft_split != NULL)
        {
            double split_fft_start;

            split_fft_start = (double)clock();
            build_direct22_residual_fft_spectrum(
                    hfft_split, N, response_time[Ns-1], 0.0,
                    Ns, response_time, plan_time, residual_phase_direct,
                    ATacc, ATspline, params, model);
            snprintf(filename, sizeof(filename),
                     "THM_mode%s_dense_directcarrier_split_spa_vs_fft_X_to2join.dat",
                     label);
            write_dense_direct22_residual_spa_fft_diagnostic(
                    filename, hfft, hfft_split, N, Tobs,
                    Ns, response_time, plan_time, residual_phase_direct,
                    ATacc, ATspline, params, model,
                    freq_split_direct, fdot_split_direct,
                    fdiag_min, fdiag_max, 1.0/Tobs, 0.0);

            build_direct22_residual_fft_spectrum(
                    hfft_split, N, response_time[Ns-1], alpha,
                    Ns, response_time, plan_time, residual_phase_direct,
                    ATacc, ATspline, params, model);
            snprintf(filename, sizeof(filename),
                     "THM_mode%s_dense_directcarrier_split_spa_vs_fft_X_tukeyref_to2join.dat",
                     label);
            write_dense_direct22_residual_spa_fft_diagnostic(
                    filename, hfft_tukey, hfft_split, N, Tobs,
                    Ns, response_time, plan_time, residual_phase_direct,
                    ATacc, ATspline, params, model,
                    freq_split_direct, fdot_split_direct,
                    fdiag_min, fdiag_max, 1.0/Tobs, alpha);
            direct_split_fft_time = ((double)clock()-split_fft_start)/
                                    CLOCKS_PER_SEC;
            printf("directcarrier_split_fft_diagnostic mode %s seconds %.9f transforms 2\n",
                   label, direct_split_fft_time);
        }
    }
	snprintf(filename, sizeof(filename), "THM_mode%s_hybrid_fourier_band_match_X.dat", label);
	write_fourier_band_comparison(filename, hfft, N, Tobs,
	                                  AFacc, AFspline, PFacc, PFspline,
                                  freq[0], freq[Nts-1],
                                  "hybrid_ftran_input_to_WDM",
                                  wdms);
    snprintf(filename, sizeof(filename), "THM_mode%s_hybrid_fourier_band_match_tukeyref_X.dat", label);
    write_fourier_band_comparison(filename, hfft_tukey, N, Tobs,
                                  AFacc, AFspline, PFacc, PFspline,
                                  freq[0], freq[Nts-1],
                                  "hybrid_ftran_input_to_WDM",
                                  wdms);

    start = clock();
    WDMtrackFFT(wdmwavefft, listn, listm, &NpFFT, N, nmid, nsize, hfft, wdms, 0, 0.0, 0.0);
    end = clock();
    fft_wdm_time = ((double)(end-start))/CLOCKS_PER_SEC;

    start = clock();
    WDMtrackFFT(wdmwavefft_tukey, listn, listm, &NpFFTTukey, N, nmid, nsize, hfft_tukey, wdms, 0, 0.0, 0.0);
    end = clock();
    fft_tukey_wdm_time = ((double)(end-start))/CLOCKS_PER_SEC;

    wdmfft = double_matrix(Nt, Nf+1);
    unpack_wdm_track(wdmfft, listn, listm, wdmwavefft, NpFFT);
    snprintf(filename, sizeof(filename), "track_pixels_fft_THM_%s.dat", label);
    write_track_pixels(filename, listn, listm, wdmwavefft, NpFFT);
    snprintf(filename, sizeof(filename), "wtranfft_THM_%s.dat", label);
    write_wdm_matrix(filename, wdmfft);
    snprintf(filename, sizeof(filename), "BinaryFFTTHM_%s.dat", label);
    write_wdm_binary(filename, wdmfft, wdms);
    snprintf(filename, sizeof(filename), "THM_mode%s_hybrid_layer_match_X.dat", label);
    write_wdm_layer_comparison(filename, wdmfft, wdm,
                               "wtranfft_THM", "wtranfast_THM",
                               wdms, 0, Nt);

    wdmfft_tukey = double_matrix(Nt, Nf+1);
    unpack_wdm_track(wdmfft_tukey, listn, listm, wdmwavefft_tukey, NpFFTTukey);
    snprintf(filename, sizeof(filename), "track_pixels_fft_tukeyref_THM_%s.dat", label);
    write_track_pixels(filename, listn, listm, wdmwavefft_tukey, NpFFTTukey);
    snprintf(filename, sizeof(filename), "wtranfft_tukeyref_THM_%s.dat", label);
    write_wdm_matrix(filename, wdmfft_tukey);
    snprintf(filename, sizeof(filename), "BinaryFFTTukeyRefTHM_%s.dat", label);
    write_wdm_binary(filename, wdmfft_tukey, wdms);
    snprintf(filename, sizeof(filename), "THM_mode%s_hybrid_layer_match_tukeyref_X.dat", label);
    write_wdm_layer_comparison(filename, wdmfft_tukey, wdm,
                               "wtranfft_tukeyref_THM", "wtranfast_THM",
                               wdms, 0, Nt);

    if(Nspa >= 4)
    {
        AFspaacc = gsl_interp_accel_alloc();
        PFspaacc = gsl_interp_accel_alloc();
        AFspaspline = gsl_spline_alloc(gsl_interp_akima, Nspa);
        PFspaspline = gsl_spline_alloc(gsl_interp_cspline, Nspa);
        gsl_spline_init(AFspaspline, freq_spa, Amp_spa, Nspa);
        gsl_spline_init(PFspaspline, freq_spa, phase_spa, Nspa);

        snprintf(filename, sizeof(filename), "THM_mode%s_spaonly_fourier_band_match_X.dat", label);
        write_fourier_band_comparison(filename, hfft, N, Tobs,
                                      AFspaacc, AFspaspline,
                                      PFspaacc, PFspaspline,
                                      freq_spa[0], freq_spa[Nspa-1],
                                      "pure_SPA_input_to_WDM",
                                      wdms);
        snprintf(filename, sizeof(filename), "THM_mode%s_spaonly_fourier_band_match_tukeyref_X.dat", label);
        write_fourier_band_comparison(filename, hfft_tukey, N, Tobs,
                                      AFspaacc, AFspaspline,
                                      PFspaacc, PFspaspline,
                                      freq_spa[0], freq_spa[Nspa-1],
                                      "pure_SPA_input_to_WDM",
                                      wdms);

        if(Nspa1p >= 4)
        {
            AFspa1pacc = gsl_interp_accel_alloc();
            PFspa1pacc = gsl_interp_accel_alloc();
            AFspa1pspline = gsl_spline_alloc(gsl_interp_akima, Nspa1p);
            PFspa1pspline = gsl_spline_alloc(gsl_interp_cspline, Nspa1p);
            gsl_spline_init(AFspa1pspline, freq_spa1p, Amp_spa1p, Nspa1p);
            gsl_spline_init(PFspa1pspline, freq_spa1p, phase_spa1p, Nspa1p);

            snprintf(filename, sizeof(filename), "THM_mode%s_spa1plus_fourier_band_match_X.dat", label);
            write_fourier_band_comparison(filename, hfft, N, Tobs,
                                          AFspa1pacc, AFspa1pspline,
                                          PFspa1pacc, PFspa1pspline,
                                          freq_spa1p[0], freq_spa1p[Nspa1p-1],
                                          "first_order_SPA_A_plus_iDelta1",
                                          wdms);
            snprintf(filename, sizeof(filename), "THM_mode%s_spa1plus_fourier_band_match_tukeyref_X.dat", label);
            write_fourier_band_comparison(filename, hfft_tukey, N, Tobs,
                                          AFspa1pacc, AFspa1pspline,
                                          PFspa1pacc, PFspa1pspline,
                                          freq_spa1p[0], freq_spa1p[Nspa1p-1],
                                          "first_order_SPA_A_plus_iDelta1",
                                          wdms);
        }
        else
        {
            fprintf(stderr, "Warning: first-order SPA plus diagnostic for mode %s has only %d monotone samples; skipping Fourier-band output.\n",
                    label, Nspa1p);
        }

        if(Nspa1m >= 4)
        {
            AFspa1macc = gsl_interp_accel_alloc();
            PFspa1macc = gsl_interp_accel_alloc();
            AFspa1mspline = gsl_spline_alloc(gsl_interp_akima, Nspa1m);
            PFspa1mspline = gsl_spline_alloc(gsl_interp_cspline, Nspa1m);
            gsl_spline_init(AFspa1mspline, freq_spa1m, Amp_spa1m, Nspa1m);
            gsl_spline_init(PFspa1mspline, freq_spa1m, phase_spa1m, Nspa1m);

            snprintf(filename, sizeof(filename), "THM_mode%s_spa1minus_fourier_band_match_X.dat", label);
            write_fourier_band_comparison(filename, hfft, N, Tobs,
                                          AFspa1macc, AFspa1mspline,
                                          PFspa1macc, PFspa1mspline,
                                          freq_spa1m[0], freq_spa1m[Nspa1m-1],
                                          "first_order_SPA_A_minus_iDelta1",
                                          wdms);
            snprintf(filename, sizeof(filename), "THM_mode%s_spa1minus_fourier_band_match_tukeyref_X.dat", label);
            write_fourier_band_comparison(filename, hfft_tukey, N, Tobs,
                                          AFspa1macc, AFspa1mspline,
                                          PFspa1macc, PFspa1mspline,
                                          freq_spa1m[0], freq_spa1m[Nspa1m-1],
                                          "first_order_SPA_A_minus_iDelta1",
                                          wdms);
        }
        else
        {
            fprintf(stderr, "Warning: first-order SPA minus diagnostic for mode %s has only %d monotone samples; skipping Fourier-band output.\n",
                    label, Nspa1m);
        }

        start = clock();
        WDMtrack(wdmwavespa, listn, listm, &NpSPA, Nspa, N, nmid, nsize, tc,
                 freq_spa, AFspaacc, AFspaspline, PFspaacc, PFspaspline, wdms);
        end = clock();
        spaonly_wdm_time = ((double)(end-start))/CLOCKS_PER_SEC;

        wdmspa = double_matrix(Nt, Nf+1);
        unpack_wdm_track(wdmspa, listn, listm, wdmwavespa, NpSPA);
        snprintf(filename, sizeof(filename), "track_pixels_spaonly_THM_%s.dat", label);
        write_track_pixels(filename, listn, listm, wdmwavespa, NpSPA);
        snprintf(filename, sizeof(filename), "wtran_spaonly_THM_%s.dat", label);
        write_wdm_matrix(filename, wdmspa);
        snprintf(filename, sizeof(filename), "BinarySPAOnlyTHM_%s.dat", label);
        write_wdm_binary(filename, wdmspa, wdms);
        snprintf(filename, sizeof(filename), "THM_mode%s_spaonly_layer_match_X.dat", label);
        write_wdm_layer_comparison(filename, wdmfft, wdmspa,
                                   "wtranfft_THM", "wtran_spaonly_THM",
                                   wdms, 0, Nt);
        snprintf(filename, sizeof(filename), "THM_mode%s_spaonly_layer_match_tukeyref_X.dat", label);
        write_wdm_layer_comparison(filename, wdmfft_tukey, wdmspa,
                                   "wtranfft_tukeyref_THM", "wtran_spaonly_THM",
                                   wdms, 0, Nt);

        /*
         * Clean hybrid diagnostic: keep the pre-join WDM packets on the pure
         * SPA spline, but replace fully post-join layers with native short-FFT
         * packet coefficients.  This avoids the old ftran() operation that
         * appends short-FFT A(f),phi(f) samples to the SPA samples and then
         * splines across the interface.  Boundary layers whose Meyer support
         * straddles the join are deliberately left on the SPA side for this
         * first conservative test.
         */
        wdmclean = double_matrix(Nt, Nf+1);
        for(j=0; j<Nt; j++)
        {
            for(i=0; i<=Nf; i++) wdmclean[j][i] = wdmspa[j][i];
        }
        {
            int jj_join = (int)setup[4];
            double f_clean_start;
            double f_clean_stop;

            if(jj_join < 0) jj_join = 0;
            if(jj_join >= Ns) jj_join = Ns-1;
            f_clean_start = freq_track[jj_join];
            if(!isfinite(f_clean_start) || f_clean_start <= 0.0)
            {
                f_clean_start = freq_spa[Nspa-1];
            }
            f_clean_stop = setup[6]/(setup[0]*setup[1]);
            if(f_clean_stop > 0.5/setup[0]) f_clean_stop = 0.5/setup[0];

            snprintf(filename, sizeof(filename), "cleanhybrid_direct_layers_THM_%s.dat", label);
            start = clock();
            WDMreplaceWithShortFFTThreshold(wdmclean, nmid, nsize, short_htime,
                                            setup, f_clean_start, f_clean_stop,
                                            wdms, filename,
                                            &clean_layers, &clean_pixels);
            end = clock();
            clean_wdm_time = ((double)(end-start))/CLOCKS_PER_SEC;

            printf("single_carrier_cleanhybrid mode %s f_replace_start %.15e f_replace_stop %.15e layers %d pixels %d\n",
                   label, f_clean_start, f_clean_stop,
                   clean_layers, clean_pixels);
        }
        snprintf(filename, sizeof(filename), "track_pixels_cleanhybrid_THM_%s.dat", label);
        write_wdm_nonzero_track_pixels(filename, wdmclean);
        snprintf(filename, sizeof(filename), "wtran_cleanhybrid_THM_%s.dat", label);
        write_wdm_matrix(filename, wdmclean);
        snprintf(filename, sizeof(filename), "BinaryCleanHybridTHM_%s.dat", label);
        write_wdm_binary(filename, wdmclean, wdms);
        snprintf(filename, sizeof(filename), "THM_mode%s_cleanhybrid_layer_match_X.dat", label);
        write_wdm_layer_comparison(filename, wdmfft, wdmclean,
                                   "wtranfft_THM", "wtran_cleanhybrid_THM",
                                   wdms, 0, Nt);
        snprintf(filename, sizeof(filename), "THM_mode%s_cleanhybrid_layer_match_tukeyref_X.dat", label);
        write_wdm_layer_comparison(filename, wdmfft_tukey, wdmclean,
                                   "wtranfft_tukeyref_THM", "wtran_cleanhybrid_THM",
                                   wdms, 0, Nt);
    }
    else
    {
        fprintf(stderr, "Warning: pure-SPA WDM diagnostic for mode %s has only %d monotone samples; skipping wtran_spaonly output.\n",
                label, Nspa);
    }

    hfull = double_vector(N);
    snprintf(filename, sizeof(filename), "THM_mode%s_Xtime.dat", label);
    out = fopen(filename, "w");
    for(i=0; i<N; i++)
    {
        t = (double)i*dt;
        hval = 0.0;
        if(t >= response_time[0] && t <= response_time[Ns-1])
        {
            hval = gsl_spline_eval(ATspline, t, ATacc)*
                   cos(gsl_spline_eval(PTspline, t, PTacc));
        }
        hfull[i] = hval;
        if(out != NULL) fprintf(out, "%.15e %.15e\n", t, hval);
    }
    if(out != NULL) fclose(out);

    alpha = 2.0*REFERENCE_TUKEY_ROLL_SECONDS/Tobs;
    if(alpha > 1.0) alpha = 1.0;
    if(alpha < 0.0) alpha = 0.0;
    tukey(hfull, alpha, N);
    snprintf(filename, sizeof(filename), "THM_mode%s_Xtime_tukey.dat", label);
    out = fopen(filename, "w");
    for(i=0; i<N; i++)
    {
        t = (double)i*dt;
        if(out != NULL) fprintf(out, "%.15e %.15e\n", t, hfull[i]);
    }
    if(out != NULL) fclose(out);

    snprintf(filename, sizeof(filename), "wdm_match_info_THM_%s.dat", label);
    write_match_info(filename, wdms, alpha, 0.5*alpha*Tobs);

    printf("single_carrier_wdm mode (%d,%+d)%s carrier %d samples %d\n",
           c->ell, c->emm, c->is_pair ? " folded_pair" : "", carrier_index, Ns);
    printf("single_carrier_wdm_timing tdi %.6f spa_freq %.6f spa_wdm %.6f fft_build %.6f fft_wdm %.6f fft_tukey_build %.6f fft_tukey_wdm %.6f spaonly_wdm %.6f cleanhybrid_replace %.6f\n",
           single_tdi_time, spa_freq_time, spa_wdm_time, fft_build_time,
           fft_wdm_time, fft_tukey_build_time, fft_tukey_wdm_time,
           spaonly_wdm_time, clean_wdm_time);
    if(direct_split_ok)
    {
        printf("single_carrier_directcarrier_split_timing setup %.6f two_reference_ffts_and_files %.6f\n",
               direct_split_setup_time, direct_split_fft_time);
    }
    printf("single_carrier_wdm_pixels fast %d fft %d fft_tukey %d spaonly %d nts %d ntsmax %d spaonly_samples %d\n",
           Np, NpFFT, NpFFTTukey, NpSPA, Nts, Ntsmax, Nspa);
    printf("wrote THM_mode%s_Xtime.dat THM_mode%s_Xtime_tukey.dat wtranfast_THM_%s.dat wtranfft_THM_%s.dat wtranfft_tukeyref_THM_%s.dat wtran_spaonly_THM_%s.dat\n",
           label, label, label, label, label, label);
	printf("wrote THM_mode%s_full_spa_vs_full_fft_X.dat THM_mode%s_dense_spa_vs_full_fft_X_1mHz_2mHz.dat THM_mode%s_dense_spa_vs_full_fft_X_to2join.dat THM_mode%s_dense_spa_vs_full_fft_X_tukeyref_to2join.dat THM_mode%s_spaonly_layer_match_X.dat\n",
	       label, label, label, label, label);
    printf("wrote THM_mode%s_TDI_AP_refsplit.dat THM_mode%s_dense_refsplit_spa_vs_full_fft_X_to2join.dat THM_mode%s_dense_refsplit_spa_vs_full_fft_X_tukeyref_to2join.dat\n",
           label, label, label);
    if(direct_split_ok)
    {
        printf("wrote THM_mode%s_TDI_AP_directcarrier_split.dat THM_mode%s_dense_directcarrier_split_spa_vs_fft_X_to2join.dat THM_mode%s_dense_directcarrier_split_spa_vs_fft_X_tukeyref_to2join.dat\n",
               label, label, label);
    }
    printf("wrote THM_mode%s_hybrid_layer_match_X.dat\n", label);
    printf("wrote THM_mode%s_cleanhybrid_layer_match_X.dat wtran_cleanhybrid_THM_%s.dat\n",
           label, label);
    printf("wrote THM_mode%s_hybrid_fourier_band_match_X.dat THM_mode%s_spaonly_fourier_band_match_X.dat\n",
           label, label);
    printf("wrote THM_mode%s_hybrid_fourier_band_match_tukeyref_X.dat THM_mode%s_spaonly_fourier_band_match_tukeyref_X.dat\n",
           label, label);
    printf("wrote THM_mode%s_shortfft_fourier_band_match_X.dat THM_mode%s_shortfft_fourier_band_match_tukeyref_X.dat\n",
           label, label);
    printf("wrote THM_mode%s_spa1plus_fourier_band_match_X.dat THM_mode%s_spa1minus_fourier_band_match_X.dat\n",
           label, label);
    printf("wrote THM_mode%s_spa1term1plus_fourier_band_match_X.dat THM_mode%s_spa1term1minus_fourier_band_match_X.dat\n",
           label, label);
    printf("wrote THM_mode%s_spa1term12plus_fourier_band_match_X.dat THM_mode%s_spa1term124plus_fourier_band_match_X.dat\n",
           label, label);
    printf("wrote THM_mode%s_spa1term12minus_fourier_band_match_X.dat THM_mode%s_spa1term124minus_fourier_band_match_X.dat\n",
           label, label);

    free_double_vector(Xmode);
    free_double_vector(Ymode);
    free_double_vector(Zmode);
    free_double_vector(Xfmode);
    free_double_vector(Yfmode);
    free_double_vector(Zfmode);
    free_double_vector(AX);
    free_double_vector(phiX);
    free_double_vector(phi_ref);
    free_double_vector(phase_td);
    free_double_vector(omega_plan);
    free_double_vector(freq_track);
    free_double_vector(freq_ref_track);
    free_double_vector(plan_jacobian);
    free_double_vector(freq_correction);
    free_double_vector(freq_split_track);
    free_double_vector(fdot_split_track);
    free_double_vector(freq_fullphase_track);
    free_double_vector(fdot_fullphase_track);
    free_double_vector(carrier_phase_direct);
    free_double_vector(residual_phase_direct);
    free_double_vector(carrier_frequency_direct);
    free_double_vector(carrier_dfdu_direct);
    free_double_vector(plan_d1_direct);
    free_double_vector(plan_d2_direct);
    free_double_vector(residual_d1_direct);
    free_double_vector(residual_d2_direct);
    free_double_vector(freq_split_direct);
    free_double_vector(fdot_split_direct);
    free_double_vector(freq);
    free_double_vector(phase);
    free_double_vector(Amp);
    free_double_vector(short_htime);
    free_double_vector(freq_shortfft);
    free_double_vector(phase_shortfft);
    free_double_vector(Amp_shortfft);
    free_double_vector(freq_spa_all);
    free_double_vector(phase_spa_all);
    free_double_vector(Amp_spa_all);
    free_double_vector(time_spa_all);
    free_double_vector(fdot_spa_all);
    free_int_vector(valid_spa_all);
    free_double_vector(freq_spa);
    free_double_vector(phase_spa);
    free_double_vector(Amp_spa);
    free_double_vector(freq_spa1p_all);
    free_double_vector(phase_spa1p_all);
    free_double_vector(Amp_spa1p_all);
    free_double_vector(time_spa1p_all);
    free_double_vector(fdot_spa1p_all);
    free_double_vector(delta1p_all);
    free_double_vector(rel1p_all);
    free_int_vector(valid_spa1p_all);
    free_double_vector(freq_spa1m_all);
    free_double_vector(phase_spa1m_all);
    free_double_vector(Amp_spa1m_all);
    free_double_vector(time_spa1m_all);
    free_double_vector(fdot_spa1m_all);
    free_double_vector(delta1m_all);
    free_double_vector(rel1m_all);
    free_int_vector(valid_spa1m_all);
    free_double_vector(freq_spa1p);
    free_double_vector(phase_spa1p);
    free_double_vector(Amp_spa1p);
    free_double_vector(freq_spa1m);
    free_double_vector(phase_spa1m);
    free_double_vector(Amp_spa1m);
    free_int_vector(nmid);
    free_int_vector(nsize);
    free_int_vector(nmid_inverse);
    free_int_vector(nsize_inverse);
    free_int_vector(listn);
    free_int_vector(listm);
    free_double_vector(wdmwave);
    free_double_vector(wdmwavefft);
    free_double_vector(wdmwavefft_tukey);
    free_double_vector(wdmwavespa);
    free_double_vector(hfft);
    free_double_vector(hfft_tukey);
    free_double_vector(hfft_split);
    free_double_vector(hfull);
    free_double_matrix(wdm, Nt);
    free_double_matrix(wdmfft, Nt);
    if(wdmfft_tukey != NULL) free_double_matrix(wdmfft_tukey, Nt);
    if(wdmspa != NULL) free_double_matrix(wdmspa, Nt);
    if(wdmclean != NULL) free_double_matrix(wdmclean, Nt);
    gsl_spline_free(ATspline);
    gsl_spline_free(PTspline);
    gsl_spline_free(AFspline);
    gsl_spline_free(PFspline);
    if(AFspaspline != NULL) gsl_spline_free(AFspaspline);
    if(PFspaspline != NULL) gsl_spline_free(PFspaspline);
    if(AFspa1pspline != NULL) gsl_spline_free(AFspa1pspline);
    if(PFspa1pspline != NULL) gsl_spline_free(PFspa1pspline);
    if(AFspa1mspline != NULL) gsl_spline_free(AFspa1mspline);
    if(PFspa1mspline != NULL) gsl_spline_free(PFspa1mspline);
    if(AFshortfftspline != NULL) gsl_spline_free(AFshortfftspline);
    if(PFshortfftspline != NULL) gsl_spline_free(PFshortfftspline);
    gsl_interp_accel_free(ATacc);
    gsl_interp_accel_free(PTacc);
    gsl_interp_accel_free(AFacc);
    gsl_interp_accel_free(PFacc);
    if(AFspaacc != NULL) gsl_interp_accel_free(AFspaacc);
    if(PFspaacc != NULL) gsl_interp_accel_free(PFspaacc);
    if(AFspa1pacc != NULL) gsl_interp_accel_free(AFspa1pacc);
    if(PFspa1pacc != NULL) gsl_interp_accel_free(PFspa1pacc);
    if(AFspa1macc != NULL) gsl_interp_accel_free(AFspa1macc);
    if(PFspa1macc != NULL) gsl_interp_accel_free(PFspa1macc);
    if(AFshortfftacc != NULL) gsl_interp_accel_free(AFshortfftacc);
    if(PFshortfftacc != NULL) gsl_interp_accel_free(PFshortfftacc);
}


void extractAP(int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR)
{

    int i;
    double raw, prev;

    if(Ns <= 0) return;

    /*
     * The old sign handling looked for a sharp local minimum in |A| and then
     * inserted a pi phase jump.  That misses transfer-function crossings where
     * the complex TDI response loops close to zero instead of passing through a
     * clean V-shaped real amplitude.  Here we use the exact degeneracy
     *
     *     A cos(phi) = (-A) cos(phi + pi)
     *
     * directly.  First remove the fast carrier branch from atan2, then unwrap
     * the residual modulo pi rather than modulo 2pi.  Odd multiples of pi are
     * recorded as signed-amplitude flips.  This keeps the residual phase smooth
     * through transfer-frequency nodes while leaving M and Mf reconstructed
     * exactly from the signed amplitude and total phase.
     */
    raw = -atan2(Mf[0], M[0])-remainder(phiR[0], 2.0*M_PI);
    As[0] = sqrt(M[0]*M[0]+Mf[0]*Mf[0]);
    Dphi[0] = raw;
    prev = Dphi[0];

    for(i=1; i<Ns; i++)
    {
        double amp = sqrt(M[i]*M[i]+Mf[i]*Mf[i]);
        long q;

        raw = -atan2(Mf[i], M[i])-remainder(phiR[i], 2.0*M_PI);
        q = lround((prev-raw)/M_PI);
        Dphi[i] = raw+(double)q*M_PI;
        As[i] = (labs(q)%2 == 0) ? amp : -amp;
        prev = Dphi[i];
    }
    
    
}

void unwrap(int Ns, double *phi)
{
    double u, v, q;
    int i;
    
    v = phi[0];
    for(i=0; i<Ns ;i++)
    {
        u = phi[i];
        q = rint(fabs(u-v)/(2.0*M_PI));
        if(q > 0.0)
        {
           if(v > u)
           {
               u += q*2.0*M_PI;
           }
           else
           {
               u -= q*2.0*M_PI;
           }
        }
        v = u;
        phi[i] = u;
    }
    
}
 


void fast_response(double *tarray, int N, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel *ATacc, gsl_spline *ATspline, gsl_interp_accel *PTacc, gsl_spline *PTspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf)
{
    
    /*   Indicies    */
    int i,j, k, n, m, a, M;
    
    /*   Gravitational Wave basis vectors   */
    double *u,*v,*kv;
    
    double *kr, *kn;
    double *plus, *cross;
    
    /*   Polarization basis tensors   */
    double **eplus, **ecross;
    
    double phi, cosi, psi;
    double costh, sinth, cosph, sinph;
    double cos2psi, sin2psi;
    double f, t, f0, fdot, fddot, Amp, phi0;
    double AA, Aplus, Across;
    double *App, *Apm, *Acp, *Acm;
    double *Larm;
    
    double **Parray, **Varray;
    
    double hp, hc;
    
    /*   Allocating Arrays   */
    
    Larm = double_vector(3);
    Parray  = double_matrix(3,3); Varray = double_matrix(3,3);
    
    u = double_vector(3); v = double_vector(3); kv = double_vector(3);
    
    eplus  = double_matrix(3,3); ecross = double_matrix(3,3);
    
    kr = double_vector(3); kn = double_vector(3);
    
    plus = double_vector(3); cross = double_vector(3);
    
    App = double_vector(3); Apm = double_vector(3);
    Acp = double_vector(3); Acm = double_vector(3);
    
    
    
    // [7] EclipticCoLatitude, [8] EclipticLongitude  [9] polarization, [10] inclination

    costh = sin(params[7]);      // costh
    phi = params[8];      // phi
    cosi = params[10];    // cosi
    psi = params[9];   // psi
    
    cos2psi = cos(2.*psi);  sin2psi = sin(2.*psi);
    
    Aplus = 0.5*(1.+cosi*cosi);
    Across = -cosi;

    
    //Calculate cos and sin of sky position
    sinth = sqrt(1.0-costh*costh);
    cosph = cos(phi);     sinph = sin(phi);
    
    u[0] = -costh*cosph;  u[1] = -costh*sinph;  u[2] =  sinth;
    v[0] =  sinph;        v[1] = -cosph;        v[2] =  0.;
    
    kv[0] = -sinth*cosph;
    kv[1] = -sinth*sinph;
    kv[2] = -costh;
    
    for(i=0;i<3;i++)
    {
        for(j=0;j<3;j++)
        {
            eplus[i][j]  = v[i]*v[j] - u[i]*u[j];
            ecross[i][j] = u[i]*v[j] + v[i]*u[j];
        }
    }
    
      for(n=0; n< N; n++)
      {
          
          t = tarray[n];
          
          //printf("%d %e\n", n, t);
          
          for(i=0;i<3;i++)
          {
              Larm[i] = gsl_spline_eval(SLspline[i], t, SLacc[i]);
          }
          
          //printf("%e %e\n", Larm[0], 1.0/(2.0*M_PI*Larm[0]));
          
          for(i=0;i<3;i++)
          {
              for(j=0;j<3;j++)
              {
                  k = j+3*i;
                  Parray[i][j] = gsl_spline_eval(SPspline[k], t, SPacc[k]);
                  Varray[i][j] = gsl_spline_eval(SVspline[k], t, SVacc[k]);
              }
          }
              
          // k dot r_i
          for(i=0;i<3;i++)
          {
              kr[i] = 0.0;
              for(j=0;j<3;j++)
              {
                  kr[i] += Parray[i][j]*kv[j];
              }
          }
          
          //printf("%e\n", kr[0]);
          
          // k dot n_i
          for(i=0;i<3;i++)
          {
              kn[i] = 0.0;
              for(j=0;j<3;j++)
              {
                  kn[i] += Varray[i][j]*kv[j];
              }
          }
          
          
          //Antenna primitives
          for(i=0; i<3; i++)
          {
              plus[i] = 0.0;
              cross[i] = 0.0;
              for(j=0; j<3; j++)
              {
                  for(k=0; k<3; k++)
                  {
                      plus[i]  += (Varray[i][j]*Varray[i][k])*eplus[j][k];
                      cross[i] += (Varray[i][j]*Varray[i][k])*ecross[j][k];
                  }
              }
           }
          
          //Full Antenna patterns
          for(i=0; i<3; i++)
          {
              App[i] = 0.5*plus[i]/(1.0+kn[i]);
              Apm[i] = 0.5*plus[i]/(1.0-kn[i]);
              Acp[i] = 0.5*cross[i]/(1.0+kn[i]);
              Acm[i] = 0.5*cross[i]/(1.0-kn[i]);
          }
          
          // build X, Y, Z responses
          TDI_spline(X, Xf, 0, 1, 2, tarray, n, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kr, Larm);
          TDI_spline(Y, Yf, 1, 2, 0, tarray, n, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kr, Larm);
          TDI_spline(Z, Zf, 2, 0, 1, tarray, n, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kr, Larm);

          
      }
    
    free_double_vector(Larm);
    free_double_vector(u); free_double_vector(v); free_double_vector(kv);
    free_double_matrix(eplus,3); free_double_matrix(ecross,3);
    free_double_matrix(Parray,3); free_double_matrix(Varray,3);
    free_double_vector(kr); free_double_vector(kn);
    free_double_vector(plus); free_double_vector(cross);
    free_double_vector(App); free_double_vector(Apm);
    free_double_vector(Acp); free_double_vector(Acm);

    
    return;
}


void TDI_spline(double *M, double *Mf, int a, int b, int c, double* tarray, int n, gsl_interp_accel *ATacc, gsl_spline *ATspline, gsl_interp_accel *PTacc, gsl_spline *PTspline, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    double t, hp, hc, hpf, hcf;
    
    char name[1024];

              M[n] = 0.0;
              Mf[n] = 0.0;
              
              t = tarray[n] - kr[a]-2.0*Larm[c]-2.0*Larm[b];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[c]+hc*Acp[c];
              M[n] -= hp*Apm[b]+hc*Acm[b];
              Mf[n] += hpf*App[c]+hcf*Acp[c];
              Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
              
              t = tarray[n] - kr[b]-Larm[c]-2.0*Larm[b];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*App[c]+hc*Acp[c];
              M[n] += hp*Apm[c]+hc*Acm[c];
              Mf[n] -= hpf*App[c]+hcf*Acp[c];
              Mf[n] += hpf*Apm[c]+hcf*Acm[c];
              
              t = tarray[n] - kr[c]-Larm[b]-2.0*Larm[c];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*Apm[b]+hc*Acm[b];
              M[n] -= hp*App[b]+hc*Acp[b];
              Mf[n] += hpf*Apm[b]+hcf*Acm[b];
              Mf[n] -= hpf*App[b]+hcf*Acp[b];
              
              t = tarray[n] - kr[a]-2.0*Larm[b];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*Apm[c]+hc*Acm[c];
              M[n] += hp*Apm[b]+hc*Acm[b];
              Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
              Mf[n] += hpf*Apm[b]+hcf*Acm[b];
              
              t = tarray[n] - kr[a]-2.0*Larm[c];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[b]+hc*Acp[b];
              M[n] -= hp*App[c]+hc*Acp[c];
              Mf[n] += hpf*App[b]+hcf*Acp[b];
              Mf[n] -= hpf*App[c]+hcf*Acp[c];
              
              t = tarray[n] - kr[c]- Larm[b];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*Apm[b]+hc*Acm[b];
              M[n] += hp*App[b]+hc*Acp[b];
              Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
              Mf[n] += hpf*App[b]+hcf*Acp[b];
              
              t = tarray[n] - kr[b]- Larm[c];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[c]+hc*Acp[c];
              M[n] -= hp*Apm[c]+hc*Acm[c];
              Mf[n] += hpf*App[c]+hcf*Acp[c];
              Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
              
              t = tarray[n] - kr[a];
              hphc(t, ATacc, ATspline, PTacc, PTspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*App[b]+hc*Acp[b];
              M[n] += hp*Apm[c]+hc*Acm[c];
              Mf[n] -= hpf*App[b]+hcf*Acp[b];
              Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    
}

void hphc(double t, gsl_interp_accel *ATacc, gsl_spline *ATspline, gsl_interp_accel *PTacc, gsl_spline *PTspline, double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf)
{
    double phase, A;
    double cp, sp;
    
    phase = gsl_spline_eval(PTspline, t, PTacc);
    A = gsl_spline_eval(ATspline, t, ATacc);
    
    cp = cos(phase);
    sp = sin(phase);
    *hp = A*(Aplus*cos2psi*cp+Across*sin2psi*sp);
    *hc = A*(Across*cos2psi*sp-Aplus*sin2psi*cp);
    
    *hpf = A*(-Aplus*cos2psi*sp+Across*sin2psi*cp);
    *hcf = A*(Across*cos2psi*cp+Aplus*sin2psi*sp);
              
}

void barycenter_time(double *tarray, double *tspace, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int N)
{
    
    
    /*   Indicies    */
    int j, n;
    
    double t, kr;
    
    // sky location of source
    double costh, phi, sinth, cosph, sinph;
    
    // location of soacecraft 0
    double *x;
    
    // poistion vector of source
    double *kv;
    
    x = double_vector(3);
    kv = double_vector(3);

    costh = sin(params[7]);      // costh
    phi = params[8];      // phi
   
    //Calculate cos and sin of sky position
    sinth = sqrt(1.0-costh*costh);
    cosph = cos(phi);     sinph = sin(phi);
    
    kv[0] = -sinth*cosph;
    kv[1] = -sinth*sinph;
    kv[2] = -costh;
    
     // start with initial guess of barycneter time being spacecraft time, then improve
    
      for(n=0; n< N; n++)
      {
    
          // spline the location of spacecraft 0
          for(j=0;j<3;j++)
          {
              x[j] = gsl_spline_eval(SPspline[j], tspace[n], SPacc[j]);
          }
          
          
          // k dot r_0
          kr = 0.0;
          for(j=0;j<3;j++)
          {
              kr += x[j]*kv[j];
          }
          
          // shift time reference from spacecraft 0 to Barycenter
          tarray[n] = tspace[n] + kr;
          
          // use intial guess of barycneter time to refine the mapping
          for(j=0;j<3;j++)
          {
              x[j] = gsl_spline_eval(SPspline[j], tarray[n], SPacc[j]);
          }
          
          // k dot r_0
          kr = 0.0;
          for(j=0;j<3;j++)
          {
              kr += x[j]*kv[j];
          }
          
          // shift time reference from spacecraft 0 to Barycenter
          tarray[n] = tspace[n] + kr;
          
      }
    
    free(x);
    free(kv);
         
          
}


void detector_time(double *tarray, double *tspace, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int N)
{
    
    
    /*   Indicies    */
    int j, n;
    
    double t, kr;
    
    // sky location of source
    double costh, phi, sinth, cosph, sinph;
    
    // location of soacecraft 0
    double *x;
    
    // poistion vector of source
    double *kv;
    
    x = double_vector(3);
    kv = double_vector(3);

    costh = sin(params[7]);      // costh
    phi = params[8];      // phi
   
    //Calculate cos and sin of sky position
    sinth = sqrt(1.0-costh*costh);
    cosph = cos(phi);     sinph = sin(phi);
    
    kv[0] = -sinth*cosph;
    kv[1] = -sinth*sinph;
    kv[2] = -costh;
    
      for(n=0; n< N; n++)
      {
          t = tarray[n];
    
          // spline the location of spacecraft 0
          for(j=0;j<3;j++)
          {
              x[j] = gsl_spline_eval(SPspline[j], t, SPacc[j]);
          }
          
          
          // k dot r_0
          kr = 0.0;
          for(j=0;j<3;j++)
          {
              kr += x[j]*kv[j];
          }
          
          // shift time reference to spacecraft 0
          tspace[n] = t - kr;
          
      }
    
    free(x);
    free(kv);
         
          
}


void spacecraft(double t, double *x, double *y, double *z)
{

  double alpha;
  double beta1, beta2, beta3;
  double sa, sb, ca, cb;
  double si, ci;
 
  alpha = 2.*M_PI*t*fm + thm_orbit_phase_offset;

  beta1 = 0. + lambda0;
  beta2 = 2.*M_PI/3. + lambda0;
  beta3 = 4.*M_PI/3. + lambda0;

    
  sa = sin(alpha);
  ca = cos(alpha);

  sb = sin(beta1);
  cb = cos(beta1);
  spacraft_loc(0, x, y, z, sa, ca, sb, cb);
 
  sb = sin(beta2);
  cb = cos(beta2);
  spacraft_loc(1, x, y, z, sa, ca, sb, cb);

  sb = sin(beta3);
  cb = cos(beta3);
  spacraft_loc(2, x, y, z, sa, ca, sb, cb);

    
}

void spacraft_loc(int i, double *x, double *y, double *z, double sa, double ca, double sb, double cb)
{
      
      x[i] = AU*ca + AU*ec*(sa*ca*sb - (1. + sa*sa)*cb);
      y[i] = AU*sa + AU*ec*(sa*ca*cb - (1. + ca*ca)*sb);
      z[i] = -sq3*AU*ec*(ca*cb + sa*sb);
      
      // next order corrections (no longer equal arm)
      
       /*
      x[i] += 0.125*ec*ec*AU*(-10.*ca - 5.*ca*cb*cb + 3.*ca*ca*ca*cb*cb - 9.*ca*cb*cb*sa*sa - 10.*cb*sa*sb + 18.*ca*ca*cb*sa*sb - 6.*cb*sa*sa*sa*sb + 5.*ca*sb*sb - 3.*ca*ca*ca*sb*sb + 9.*ca*sa*sa*sb*sb);
      y[i] += 0.125*ec*ec*AU*(-10.*sa + 5.*cb*cb*sa + 9.*ca*ca*cb*cb*sa - 3.*cb*cb*sa*sa*sa - 10.*ca*cb*sb - 6.*ca*ca*ca*cb*sb + 18.*ca*cb*sa*sa*sb - 5.*sa*sb*sb - 9.*ca*ca*sa*sb*sb + 3.*sa*sa*sa*sb*sb);
      z[i] += sq3*AU*ec*ec*(1. + (sa*cb - sb*ca)*(sa*cb - sb*ca));
     */
      
    
}

void constellation(int Ns, double *tarray, double **Larray, double ***Parray, double ***Varray)
{
    
    /*   Indicies    */
    int i,j, k, n;
    
    /*   Spacecraft position   */
    double *x, *y, *z;
    
    x = double_vector(3); y = double_vector(3); z = double_vector(3);
    
    for(n=0; n< Ns; n++)
    {
        
        spacecraft(tarray[n], x, y, z);
        
        //Parray indexed by spacecraft, coordinate, time
        
        // position vectors for each spacecraft (converted to seconds)
        for(i=0;i<3;i++)
        {
            Parray[i][0][n]  = x[i]/CLIGHT;
            Parray[i][1][n]  = y[i]/CLIGHT;
            Parray[i][2][n]  = z[i]/CLIGHT;
        }
        
        // Parray indexed by opposite vertex spacecraft, coordinate, time
        
        // arm vectors
        for(i=0;i<3;i++)
        {
            Varray[0][i][n] = Parray[1][i][n]-Parray[2][i][n];
            Varray[1][i][n] = Parray[2][i][n]-Parray[0][i][n];
            Varray[2][i][n] = Parray[0][i][n]-Parray[1][i][n];
        }
        
        //  Larray indexed by opposite vertex spacecraft, time
        
        // arm lengths
        for(i=0;i<3;i++)
        {
            Larray[i][n] = 0.0;
            for(j=0;j<3;j++)
            {
                Larray[i][n] += Varray[i][j][n]*Varray[i][j][n];
            }
            Larray[i][n] = sqrt(Larray[i][n]);
            
        }
        
        // turn arm vectors into unit vectors
        for(i=0;i<3;i++)
        {
            for(j=0;j<3;j++) Varray[i][j][n] /= Larray[i][n];
        }
        
        
    }
    
    free(x);
    free(y);
    free(z);
    
    
}



int *int_vector(int N)
{
    return malloc( (N+1) * sizeof(int) );
}

void free_int_vector(int *v)
{
    free(v);
}

int **int_matrix(int N, int M)
{
    int i;
    int **m = malloc( (N+1) * sizeof(int *));
    
    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(int));
    }
    
    return m;
}

void free_int_matrix(int **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_int_vector(m[i]);
    free(m);
}

double *double_vector(int N)
{
    return malloc( (N+1) * sizeof(double) );
}

void free_double_vector(double *v)
{
    free(v);
}

double **double_matrix(int N, int M)
{
    int i;
    double **m = malloc( (N+1) * sizeof(double *));
    
    for(i=0; i<N+1; i++)
    {
        m[i] = malloc( (M+1) * sizeof(double));
    }
    
    return m;
}

void free_double_matrix(double **m, int N)
{
    int i;
    for(i=0; i<N+1; i++) free_double_vector(m[i]);
    free(m);
}

double ***double_tensor(int N, int M, int L)
{
    int i,j;
    
    double ***t = malloc( (N+1) * sizeof(double **));
    for(i=0; i<N+1; i++)
    {
        t[i] = malloc( (M+1) * sizeof(double *));
        for(j=0; j<M+1; j++)
        {
            t[i][j] = malloc( (L+1) * sizeof(double));
        }
    }
    
    return t;
}

void free_double_tensor(double ***t, int N, int M)
{
    int i;
    
    for(i=0; i<N+1; i++) free_double_matrix(t[i],M);
    
    free(t);
}

double ****double_quad(int N, int M, int L, int K)
{
    int i,j,k;
    
    double ****t = malloc( (N+1) * sizeof(double **));
    for(i=0; i<N+1; i++)
    {
        t[i] = malloc( (M+1) * sizeof(double *));
        for(j=0; j<M+1; j++)
        {
            t[i][j] = malloc( (L+1) * sizeof(double));
            for(k=0; k<L+1; k++)
            {
                       t[i][j][k] = malloc( (K+1) * sizeof(double));
            }
        }
    }
    
    return t;
}

void free_double_quad(double ****t, int N, int M, int L)
{
    int i;
    
    for(i=0; i<N+1; i++) free_double_tensor(t[i],M,L);
    
    free(t);
}
