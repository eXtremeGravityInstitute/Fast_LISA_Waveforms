/* Copyright (C) 2026 Neil Cornish.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "PhenomTPHM_TDI.h"

#define TSUN 4.925490947641267e-6

#define TPHM_YEAR_WDM_DUMP_MAGIC UINT64_C(0x5450484d57444d31)

static int write_tphm_year_wdm_dump(
    const char *filename, const THMSparseWDMTriplet *wdm)
{
    FILE *out;
    const uint64_t magic = TPHM_YEAR_WDM_DUMP_MAGIC;
    int ch;
    if(filename == NULL || wdm == NULL) return 0;
    out = fopen(filename, "wb");
    if(out == NULL) return 0;
    if(fwrite(&magic, sizeof(magic), 1, out) != 1) goto fail;
    for(ch=0; ch<3; ch++)
    {
        const THMSparseWDMChannel *pixel = &wdm->channel[ch];
        size_t count;
        if(pixel->npixels < 0) goto fail;
        count = (size_t)pixel->npixels;
        if(fwrite(&pixel->npixels, sizeof(pixel->npixels), 1, out) != 1 ||
           fwrite(pixel->n, sizeof(*pixel->n), count, out) != count ||
           fwrite(pixel->m, sizeof(*pixel->m), count, out) != count ||
           fwrite(pixel->value, sizeof(*pixel->value), count, out) != count)
            goto fail;
    }
    return fclose(out) == 0;
fail:
    fclose(out);
    return 0;
}

#define THM_FULL_CADENCE_CACHE_MAGIC UINT64_C(0x54484d57444d5246)
#define THM_FULL_CADENCE_CACHE_VERSION UINT64_C(2026092201)

static int write_thm_full_cadence_cache(
    const char *filename, const double params[11],
    const THMSparseWDMTriplet *reference)
{
    FILE *out;
    uint64_t magic = THM_FULL_CADENCE_CACHE_MAGIC;
    uint64_t version = THM_FULL_CADENCE_CACHE_VERSION;
    int ch;

    if(filename == NULL || params == NULL || reference == NULL) return 0;
    out = fopen(filename, "wb");
    if(out == NULL) return 0;
    if(fwrite(&magic, sizeof(magic), 1, out) != 1 ||
       fwrite(&version, sizeof(version), 1, out) != 1 ||
       fwrite(params, sizeof(*params), 11, out) != 11)
    {
        fclose(out);
        return 0;
    }
    for(ch=0; ch<3; ch++)
    {
        const THMSparseWDMChannel *track = &reference->channel[ch];
        if(track->npixels < 0 ||
           fwrite(&track->npixels, sizeof(track->npixels), 1, out) != 1 ||
           fwrite(track->n, sizeof(*track->n), (size_t)track->npixels, out) !=
               (size_t)track->npixels ||
           fwrite(track->m, sizeof(*track->m), (size_t)track->npixels, out) !=
               (size_t)track->npixels ||
           fwrite(track->value, sizeof(*track->value),
                  (size_t)track->npixels, out) != (size_t)track->npixels)
        {
            fclose(out);
            return 0;
        }
    }
    fclose(out);
    return 1;
}

static int read_thm_full_cadence_cache(
    const char *filename, const double params[11],
    THMSparseWDMTriplet *reference)
{
    FILE *in;
    uint64_t magic, version;
    double cached_params[11];
    int ch;

    if(filename == NULL || params == NULL || reference == NULL) return 0;
    in = fopen(filename, "rb");
    if(in == NULL) return 0;
    if(fread(&magic, sizeof(magic), 1, in) != 1 ||
       fread(&version, sizeof(version), 1, in) != 1 ||
       fread(cached_params, sizeof(*cached_params), 11, in) != 11 ||
       magic != THM_FULL_CADENCE_CACHE_MAGIC ||
       version != THM_FULL_CADENCE_CACHE_VERSION ||
       memcmp(cached_params, params, sizeof(cached_params)) != 0)
    {
        fclose(in);
        return 0;
    }
    for(ch=0; ch<3; ch++)
    {
        THMSparseWDMChannel *track = &reference->channel[ch];
        int npixels;
        if(fread(&npixels, sizeof(npixels), 1, in) != 1 ||
           npixels < 0 || npixels > 100000000)
            goto failure;
        track->n = malloc((size_t)npixels*sizeof(*track->n));
        track->m = malloc((size_t)npixels*sizeof(*track->m));
        track->value = malloc((size_t)npixels*sizeof(*track->value));
        if((npixels > 0) &&
           (track->n == NULL || track->m == NULL || track->value == NULL))
            goto failure;
        if(fread(track->n, sizeof(*track->n), (size_t)npixels, in) !=
               (size_t)npixels ||
           fread(track->m, sizeof(*track->m), (size_t)npixels, in) !=
               (size_t)npixels ||
           fread(track->value, sizeof(*track->value), (size_t)npixels, in) !=
               (size_t)npixels)
            goto failure;
        track->npixels = npixels;
        track->capacity = npixels;
    }
    fclose(in);
    return 1;

failure:
    fclose(in);
    thm_sparse_wdm_triplet_free(reference);
    thm_sparse_wdm_triplet_init(reference);
    return 0;
}

static double relative_error(double complex a, double complex b)
{
    double scale = fmax(fmax(cabs(a), cabs(b)), 1.0e-30);
    return cabs(a-b)/scale;
}

static double complex_mismatch(int n, const double complex *a,
                               const double complex *b)
{
    double aa = 0.0, bb = 0.0;
    double complex ab = 0.0;
    int i;
    for(i=0; i<n; i++)
    {
        aa += creal(conj(a[i])*a[i]);
        bb += creal(conj(b[i])*b[i]);
        ab += conj(a[i])*b[i];
    }
    if(!(aa > 0.0) || !(bb > 0.0)) return 1.0;
    return fmax(0.0, 1.0-cabs(ab)/sqrt(aa*bb));
}

typedef struct
{
    int n;
    int m;
    double value;
} SparseWDMEntry;

static int sparse_wdm_entry_compare(const void *left, const void *right)
{
    const SparseWDMEntry *a = (const SparseWDMEntry *)left;
    const SparseWDMEntry *b = (const SparseWDMEntry *)right;
    if(a->m != b->m) return a->m < b->m ? -1 : 1;
    if(a->n != b->n) return a->n < b->n ? -1 : 1;
    return 0;
}

static double sparse_wdm_union_mismatch_band(const THMSparseWDMChannel *a,
                                             const THMSparseWDMChannel *b,
                                             int mlo, int mhi)
{
    SparseWDMEntry *sa = NULL, *sb = NULL;
    double aa = 0.0, bb = 0.0, ab = 0.0;
    double mismatch = 1.0;
    int ia = 0, ib = 0, i;

    if(a == NULL || b == NULL) return 1.0;
    sa = calloc((size_t)a->npixels, sizeof(*sa));
    sb = calloc((size_t)b->npixels, sizeof(*sb));
    if(sa == NULL || sb == NULL) goto cleanup;
    for(i=0; i<a->npixels; i++)
    {
        sa[i].n = a->n[i];
        sa[i].m = a->m[i];
        sa[i].value = a->value[i];
    }
    for(i=0; i<b->npixels; i++)
    {
        sb[i].n = b->n[i];
        sb[i].m = b->m[i];
        sb[i].value = b->value[i];
    }
    qsort(sa, (size_t)a->npixels, sizeof(*sa), sparse_wdm_entry_compare);
    qsort(sb, (size_t)b->npixels, sizeof(*sb), sparse_wdm_entry_compare);
    while(ia < a->npixels || ib < b->npixels)
    {
        int take_a = 0, take_b = 0;
        double av = 0.0, bv = 0.0;
        if(ib >= b->npixels)
            take_a = 1;
        else if(ia >= a->npixels)
            take_b = 1;
        else if(sa[ia].m < sb[ib].m ||
                (sa[ia].m == sb[ib].m && sa[ia].n < sb[ib].n))
            take_a = 1;
        else if(sb[ib].m < sa[ia].m ||
                (sb[ib].m == sa[ia].m && sb[ib].n < sa[ia].n))
            take_b = 1;
        else
            take_a = take_b = 1;
        if(take_a) av = sa[ia++].value;
        if(take_b) bv = sb[ib++].value;
        {
            int m = take_a ? sa[ia-1].m : sb[ib-1].m;
            if(m >= mlo && m <= mhi)
            {
                aa += av*av;
                bb += bv*bv;
                ab += av*bv;
            }
        }
    }
    if(aa > 0.0 && bb > 0.0)
        mismatch = fmax(0.0, 1.0-ab/sqrt(aa*bb));

cleanup:
    free(sb);
    free(sa);
    return mismatch;
}

static double sparse_wdm_union_mismatch(const THMSparseWDMChannel *a,
                                        const THMSparseWDMChannel *b)
{
    return sparse_wdm_union_mismatch_band(a, b, -2147483647, 2147483647);
}

/* Compare candidate a only where reference b was actually evaluated. */
static double sparse_wdm_reference_support_mismatch_band(
    const THMSparseWDMChannel *a, const THMSparseWDMChannel *b,
    int mlo, int mhi)
{
    SparseWDMEntry *sa = NULL, *sb = NULL;
    double aa = 0.0, bb = 0.0, ab = 0.0;
    double mismatch = 1.0;
    int ia = 0, ib = 0, i;

    if(a == NULL || b == NULL) return 1.0;
    sa = calloc((size_t)a->npixels, sizeof(*sa));
    sb = calloc((size_t)b->npixels, sizeof(*sb));
    if(sa == NULL || sb == NULL) goto cleanup;
    for(i=0; i<a->npixels; i++)
    {
        sa[i].n = a->n[i];
        sa[i].m = a->m[i];
        sa[i].value = a->value[i];
    }
    for(i=0; i<b->npixels; i++)
    {
        sb[i].n = b->n[i];
        sb[i].m = b->m[i];
        sb[i].value = b->value[i];
    }
    qsort(sa, (size_t)a->npixels, sizeof(*sa), sparse_wdm_entry_compare);
    qsort(sb, (size_t)b->npixels, sizeof(*sb), sparse_wdm_entry_compare);
    while(ib < b->npixels)
    {
        double av = 0.0;
        while(ia < a->npixels &&
              (sa[ia].m < sb[ib].m ||
               (sa[ia].m == sb[ib].m && sa[ia].n < sb[ib].n)))
            ia++;
        if(ia < a->npixels && sa[ia].m == sb[ib].m &&
           sa[ia].n == sb[ib].n)
            av = sa[ia].value;
        if(sb[ib].m >= mlo && sb[ib].m <= mhi)
        {
            aa += av*av;
            bb += sb[ib].value*sb[ib].value;
            ab += av*sb[ib].value;
        }
        ib++;
    }
    if(aa > 0.0 && bb > 0.0)
        mismatch = fmax(0.0, 1.0-ab/sqrt(aa*bb));

cleanup:
    free(sb);
    free(sa);
    return mismatch;
}

static double sparse_wdm_reference_support_mismatch(
    const THMSparseWDMChannel *a, const THMSparseWDMChannel *b)
{
    return sparse_wdm_reference_support_mismatch_band(
        a, b, -2147483647, 2147483647);
}

typedef struct
{
    double mismatch;
    double candidate_power;
    double reference_power;
    double reference_total_power;
    int common_pixels;
    int candidate_pixels;
    int reference_pixels;
} SparseWDMIntersectionMetrics;

/*
 * Separate transform error from support error.  Only coordinates present in
 * both sparse channels enter the match; reference_total_power still includes
 * every reference coordinate in the requested band, so its captured fraction
 * measures what the candidate mask omitted.
 */
static SparseWDMIntersectionMetrics sparse_wdm_intersection_metrics_region(
    const THMSparseWDMChannel *candidate,
    const THMSparseWDMChannel *reference,
    int mlo, int mhi, int nlo, int nhi)
{
    SparseWDMIntersectionMetrics result = {1.0, 0.0, 0.0, 0.0, 0, 0, 0};
    SparseWDMEntry *sc = NULL, *sr = NULL;
    double cross = 0.0;
    int ic = 0, ir = 0, i;

    if(candidate == NULL || reference == NULL) return result;
    sc = calloc((size_t)candidate->npixels, sizeof(*sc));
    sr = calloc((size_t)reference->npixels, sizeof(*sr));
    if(sc == NULL || sr == NULL) goto cleanup;
    for(i=0; i<candidate->npixels; i++)
    {
        sc[i].n = candidate->n[i];
        sc[i].m = candidate->m[i];
        sc[i].value = candidate->value[i];
        if(sc[i].m >= mlo && sc[i].m <= mhi &&
           sc[i].n >= nlo && sc[i].n <= nhi)
            result.candidate_pixels++;
    }
    for(i=0; i<reference->npixels; i++)
    {
        sr[i].n = reference->n[i];
        sr[i].m = reference->m[i];
        sr[i].value = reference->value[i];
        if(sr[i].m >= mlo && sr[i].m <= mhi &&
           sr[i].n >= nlo && sr[i].n <= nhi)
        {
            result.reference_total_power += sr[i].value*sr[i].value;
            result.reference_pixels++;
        }
    }
    qsort(sc, (size_t)candidate->npixels, sizeof(*sc),
          sparse_wdm_entry_compare);
    qsort(sr, (size_t)reference->npixels, sizeof(*sr),
          sparse_wdm_entry_compare);
    while(ic < candidate->npixels && ir < reference->npixels)
    {
        if(sc[ic].m < sr[ir].m ||
           (sc[ic].m == sr[ir].m && sc[ic].n < sr[ir].n))
        {
            ic++;
            continue;
        }
        if(sr[ir].m < sc[ic].m ||
           (sr[ir].m == sc[ic].m && sr[ir].n < sc[ic].n))
        {
            ir++;
            continue;
        }
        if(sc[ic].m >= mlo && sc[ic].m <= mhi &&
           sc[ic].n >= nlo && sc[ic].n <= nhi)
        {
            result.candidate_power += sc[ic].value*sc[ic].value;
            result.reference_power += sr[ir].value*sr[ir].value;
            cross += sc[ic].value*sr[ir].value;
            result.common_pixels++;
        }
        ic++;
        ir++;
    }
    if(result.candidate_power > 0.0 && result.reference_power > 0.0)
        result.mismatch = fmax(
            0.0, 1.0-cross/sqrt(result.candidate_power*result.reference_power));

cleanup:
    free(sr);
    free(sc);
    return result;
}

static SparseWDMIntersectionMetrics sparse_wdm_intersection_metrics_band(
    const THMSparseWDMChannel *candidate,
    const THMSparseWDMChannel *reference,
    int mlo, int mhi)
{
    return sparse_wdm_intersection_metrics_region(
        candidate, reference, mlo, mhi, -2147483647, 2147483647);
}

static void write_sparse_wdm_layer_comparison(
    const char *filename,
    const THMSparseWDMChannel *candidate,
    const THMSparseWDMChannel *reference,
    int layer)
{
    SparseWDMEntry *sc = NULL, *sr = NULL;
    FILE *out = NULL;
    int ic = 0, ir = 0, i;

    if(filename == NULL || candidate == NULL || reference == NULL) return;
    sc = calloc((size_t)candidate->npixels, sizeof(*sc));
    sr = calloc((size_t)reference->npixels, sizeof(*sr));
    if(sc == NULL || sr == NULL) goto cleanup;
    for(i=0; i<candidate->npixels; i++)
    {
        sc[i].n = candidate->n[i];
        sc[i].m = candidate->m[i];
        sc[i].value = candidate->value[i];
    }
    for(i=0; i<reference->npixels; i++)
    {
        sr[i].n = reference->n[i];
        sr[i].m = reference->m[i];
        sr[i].value = reference->value[i];
    }
    qsort(sc, (size_t)candidate->npixels, sizeof(*sc),
          sparse_wdm_entry_compare);
    qsort(sr, (size_t)reference->npixels, sizeof(*sr),
          sparse_wdm_entry_compare);
    out = fopen(filename, "w");
    if(out == NULL) goto cleanup;
    fprintf(out, "# n fast reference residual\n");
    while(ic < candidate->npixels || ir < reference->npixels)
    {
        double cv = 0.0, rv = 0.0;
        int n, m, take_c = 0, take_r = 0;
        if(ir >= reference->npixels)
            take_c = 1;
        else if(ic >= candidate->npixels)
            take_r = 1;
        else if(sc[ic].m < sr[ir].m ||
                (sc[ic].m == sr[ir].m && sc[ic].n < sr[ir].n))
            take_c = 1;
        else if(sr[ir].m < sc[ic].m ||
                (sr[ir].m == sc[ic].m && sr[ir].n < sc[ic].n))
            take_r = 1;
        else
            take_c = take_r = 1;
        m = take_c ? sc[ic].m : sr[ir].m;
        n = take_c ? sc[ic].n : sr[ir].n;
        if(take_c) cv = sc[ic++].value;
        if(take_r) rv = sr[ir++].value;
        if(m == layer)
            fprintf(out, "%d %.17e %.17e %.17e\n", n, cv, rv, cv-rv);
    }

cleanup:
    if(out != NULL) fclose(out);
    free(sr);
    free(sc);
}

static double sparse_wdm_power(const THMSparseWDMChannel *track)
{
    double power = 0.0;
    int i;
    if(track == NULL) return 0.0;
    for(i=0; i<track->npixels; i++)
        power += track->value[i]*track->value[i];
    return power;
}

static double sparse_wdm_power_band(const THMSparseWDMChannel *track,
                                    int mlo, int mhi)
{
    double power = 0.0;
    int i;
    if(track == NULL) return 0.0;
    for(i=0; i<track->npixels; i++)
    {
        if(track->m[i] >= mlo && track->m[i] <= mhi)
            power += track->value[i]*track->value[i];
    }
    return power;
}


static void print_endpoint_spectral_support(
    const THMComplexTDIFFTBlock *block, int channel)
{
    static const double quantile[] = {
        1.0e-10, 1.0e-8, 1.0e-6, 1.0-1.0e-6,
        1.0-1.0e-8, 1.0-1.0e-10
    };
    double total = 0.0, cumulative = 0.0;
    int k, q = 0;
    if(block == NULL || channel < 0 || channel > 2 ||
       block->fft[channel] == NULL || block->nfft < 2)
        return;
    for(k=1; k<block->nfft/2; k++)
    {
        double re = block->fft[channel][2*k];
        double im = block->fft[channel][2*k+1];
        total += re*re+im*im;
    }
    if(!(total > 0.0)) return;
    for(k=1; k<block->nfft/2 && q<(int)(sizeof(quantile)/sizeof(quantile[0])); k++)
    {
        double re = block->fft[channel][2*k];
        double im = block->fft[channel][2*k+1];
        cumulative += re*re+im*im;
        while(q<(int)(sizeof(quantile)/sizeof(quantile[0])) &&
              cumulative/total >= quantile[q])
        {
            printf("tphm_endpoint_spectrum channel %d quantile %.10e frequency %.10e\n",
                   channel, quantile[q],
                   (double)k/((double)block->nfft*block->sample_dt));
            q++;
        }
    }
}

static int benchmark_intrinsic_thm_tphm(
    PhenomTPHMTDISource *precessing_source,
    const IMRPhenomTHMMode *carriers,
    int ncarriers)
{
    const int n = 32768;
    const int nnegative = n-1024;
    IMRPhenomTPHMPrecessionConfig aligned_config;
    IMRPhenomTPHM aligned;
    IMRPhenomTHM thm;
    double *tau = NULL, *phi22 = NULL;
    double complex *precessing_strain = NULL;
    double complex *precessing_quadrature = NULL;
    double complex *aligned_strain = NULL;
    double complex *aligned_quadrature = NULL;
    double complex *thm_strain = NULL;
    IMRPhenomTHMModeSample *samples = NULL;
    double tau_start, log_start, log_stop;
    clock_t start, stop;
    double precessing_seconds, aligned_seconds, thm_seconds;
    double mismatch;
    int i, k, ell, status = 0;

    if(precessing_source == NULL || carriers == NULL || ncarriers < 1)
        return 1;
    memset(&aligned_config, 0, sizeof(aligned_config));
    aligned_config.chi1[2] = precessing_source->waveform.precession.chi1[2];
    aligned_config.chi2[2] = precessing_source->waveform.precession.chi2[2];
    aligned_config.tau_ref = precessing_source->waveform.precession.tau_ref;
    status = IMRPhenomTPHMInitialize(
        &aligned, precessing_source->waveform.carrier.m1,
        precessing_source->waveform.carrier.m2,
        &aligned_config, carriers, ncarriers);
    if(status != 0) return 10+status;
    status = IMRPhenomTHMInitialize(
        &thm, precessing_source->waveform.carrier.m1,
        precessing_source->waveform.carrier.m2,
        aligned_config.chi1[2], aligned_config.chi2[2],
        carriers, ncarriers);
    if(status != 0)
    {
        IMRPhenomTPHMDestroy(&aligned);
        return 20+status;
    }
    tau = calloc((size_t)n, sizeof(*tau));
    phi22 = calloc((size_t)n, sizeof(*phi22));
    precessing_strain = calloc((size_t)n, sizeof(*precessing_strain));
    precessing_quadrature = calloc((size_t)n,
                                   sizeof(*precessing_quadrature));
    aligned_strain = calloc((size_t)n, sizeof(*aligned_strain));
    aligned_quadrature = calloc((size_t)n, sizeof(*aligned_quadrature));
    thm_strain = calloc((size_t)n, sizeof(*thm_strain));
    samples = calloc((size_t)n*(size_t)ncarriers, sizeof(*samples));
    if(tau == NULL || phi22 == NULL || precessing_strain == NULL ||
       precessing_quadrature == NULL || aligned_strain == NULL ||
       aligned_quadrature == NULL || thm_strain == NULL || samples == NULL)
    {
        status = 2;
        goto cleanup;
    }
    tau_start = -31557600.0/precessing_source->total_mass;
    log_start = log(-tau_start);
    log_stop = log(1.0e-3);
    for(i=0; i<nnegative; i++)
    {
        double u = (double)i/(double)(nnegative-1);
        tau[i] = -exp(log_start+u*(log_stop-log_start));
    }
    for(i=nnegative; i<n; i++)
        tau[i] = 1000.0*(double)(i-nnegative)/(double)(n-nnegative-1);

    start = clock();
    status = IMRPhenomTPHMEvaluateGridWithQuadrature(
        &precessing_source->waveform, n, tau, 0.2,
        precessing_source->observer_theta,
        precessing_source->observer_phi,
        precessing_source->polarization,
        NULL, NULL, precessing_strain, precessing_quadrature,
        NULL, NULL, NULL);
    stop = clock();
    if(status != 0)
    {
        status = 30+status;
        goto cleanup;
    }
    precessing_seconds = (double)(stop-start)/(double)CLOCKS_PER_SEC;

    start = clock();
    status = IMRPhenomTPHMEvaluateGridWithQuadrature(
        &aligned, n, tau, 0.2,
        precessing_source->observer_theta,
        precessing_source->observer_phi,
        precessing_source->polarization,
        NULL, NULL, aligned_strain, aligned_quadrature,
        NULL, NULL, NULL);
    stop = clock();
    if(status != 0)
    {
        status = 40+status;
        goto cleanup;
    }
    aligned_seconds = (double)(stop-start)/(double)CLOCKS_PER_SEC;

    start = clock();
    status = IMRPhenomTHMBuildPhi22Grid(&thm, n, tau, 0.2, phi22);
    if(status == 0)
        status = IMRPhenomTHMEvaluateGrid(&thm, n, tau, phi22, samples);
    if(status != 0)
    {
        status = 50+status;
        goto cleanup;
    }
    for(i=0; i<n; i++)
    {
        double complex total = 0.0;
        for(ell=IMRPHENOMTPHM_MIN_ELL;
            ell<=IMRPHENOMTPHM_MAX_ELL; ell++)
        {
            double complex multipole[IMRPHENOMTPHM_MAX_M_COUNT] = {0.0};
            double complex ell_strain;
            int populated = 0;
            for(k=0; k<ncarriers; k++)
            {
                if(carriers[k].ell == ell)
                {
                    int emm = abs(carriers[k].emm);
                    double parity = (ell & 1) ? -1.0 : 1.0;
                    double complex hlm = samples[(size_t)k*(size_t)n+
                                                  (size_t)i].hlm;
                    multipole[emm+ell] = hlm;
                    multipole[-emm+ell] = parity*conj(hlm);
                    populated = 1;
                }
            }
            if(populated)
            {
                status = IMRPhenomTPHMProjectMultipole(
                    multipole, ell, precessing_source->observer_theta,
                    precessing_source->observer_phi,
                    precessing_source->polarization, &ell_strain);
                if(status != 0)
                {
                    status = 60+status;
                    goto cleanup;
                }
                total += ell_strain;
            }
        }
        thm_strain[i] = total;
    }
    stop = clock();
    thm_seconds = (double)(stop-start)/(double)CLOCKS_PER_SEC;
    mismatch = complex_mismatch(n, aligned_strain, thm_strain);
    printf("tphm_intrinsic_year_grid_samples %d\n", n);
    printf("tphm_intrinsic_seconds precessing %.6f aligned %.6f thm_projected %.6f\n",
           precessing_seconds, aligned_seconds, thm_seconds);
    printf("tphm_aligned_thm_intrinsic_match %.15f mismatch %.6e\n",
           1.0-mismatch, mismatch);
    if(mismatch > 5.0e-12) status = 70;

    if(status == 0)
    {
        const double scale[] = {1.0, 0.5, 0.25, 0.125, 0.0};
        const int nscale = (int)(sizeof(scale)/sizeof(scale[0]));
        for(k=0; k<nscale; k++)
        {
            IMRPhenomTPHM scaled;
            IMRPhenomTPHMPrecessionConfig config =
                precessing_source->waveform.precession;
            double complex *scaled_strain = precessing_strain;
            int own_scaled = 0;
            int j;

            if(scale[k] == 0.0)
            {
                scaled_strain = aligned_strain;
            }
            else if(scale[k] != 1.0)
            {
                for(j=0; j<2; j++)
                {
                    config.chi1[j] *= scale[k];
                    config.chi2[j] *= scale[k];
                }
                status = IMRPhenomTPHMInitialize(
                    &scaled, precessing_source->waveform.carrier.m1,
                    precessing_source->waveform.carrier.m2,
                    &config, carriers, ncarriers);
                if(status != 0)
                {
                    status = 80+status;
                    goto cleanup;
                }
                scaled_strain = calloc((size_t)n, sizeof(*scaled_strain));
                if(scaled_strain == NULL)
                {
                    IMRPhenomTPHMDestroy(&scaled);
                    status = 81;
                    goto cleanup;
                }
                own_scaled = 1;
                status = IMRPhenomTPHMEvaluateGridWithQuadrature(
                    &scaled, n, tau, 0.2,
                    precessing_source->observer_theta,
                    precessing_source->observer_phi,
                    precessing_source->polarization,
                    NULL, NULL, scaled_strain, aligned_quadrature,
                    NULL, NULL, NULL);
                IMRPhenomTPHMDestroy(&scaled);
                if(status != 0)
                {
                    free(scaled_strain);
                    status = 90+status;
                    goto cleanup;
                }
            }
            mismatch = complex_mismatch(n, scaled_strain, thm_strain);
            printf("tphm_to_thm_inplane_scale %.6f match %.15f mismatch %.6e\n",
                   scale[k], 1.0-mismatch, mismatch);
            if(own_scaled) free(scaled_strain);
        }
    }

cleanup:
    free(samples);
    free(thm_strain);
    free(aligned_quadrature);
    free(aligned_strain);
    free(precessing_quadrature);
    free(precessing_strain);
    free(phi22);
    free(tau);
    IMRPhenomTHMDestroy(&thm);
    IMRPhenomTPHMDestroy(&aligned);
    return status;
}

int main(void)
{
    const int test_tdi2 = getenv("TPHM_TEST_TDI2") != NULL;
    const IMRPhenomTHMMode carriers[] = {
        {2, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}
    };
    const int n = 257;
    const int overlap_offset = 64;
    const double m1 = 2.0e5*TSUN;
    const double m2 = 1.0e5*TSUN;
    const double mtot = m1+m2;
    const double tc = 3.0e7;
    const double t0 = tc-8192.0;
    const double sample_dt = 16.0;
    IMRPhenomTPHMPrecessionConfig precession;
    PhenomTPHMTDISource source;
    THMObservationContext *context = NULL;
    THMWorkerWorkspace *workspace = NULL;
    double *time_a = NULL;
    double *time_b = NULL;
    double complex *hp_a = NULL, *hc_a = NULL;
    double complex *hp_b = NULL, *hc_b = NULL;
    double complex *X = NULL, *Y = NULL, *Z = NULL;
    THMComplexTDIFFTBlock full_block;
    THMComplexTDIFFTBlock partition[2];
    THMComplexTDIFFTBlock efficient_reference;
    PhenomTPHMPartitionPlan efficient_plan;
    THMSparseWDMTriplet wdm_full;
    THMSparseWDMTriplet wdm_partition;
    THMSparseWDMTriplet wdm_efficient_reference;
    THMSparseWDMTriplet wdm_efficient;
    double worst_overlap = 0.0;
    double max_tdi = 0.0;
    double worst_wdm_mismatch = 0.0;
    clock_t start, stop, setup_start;
    int i;
    int status = 0;

    thm_complex_tdi_fft_block_init(&full_block);
    thm_complex_tdi_fft_block_init(&partition[0]);
    thm_complex_tdi_fft_block_init(&partition[1]);
    thm_complex_tdi_fft_block_init(&efficient_reference);
    phenom_tphm_partition_plan_init(&efficient_plan);
    thm_sparse_wdm_triplet_init(&wdm_full);
    thm_sparse_wdm_triplet_init(&wdm_partition);
    thm_sparse_wdm_triplet_init(&wdm_efficient_reference);
    thm_sparse_wdm_triplet_init(&wdm_efficient);

    memset(&precession, 0, sizeof(precession));
    precession.chi1[0] = 0.35;
    precession.chi1[1] = 0.10;
    precession.chi1[2] = 0.30;
    precession.chi2[0] = -0.20;
    precession.chi2[1] = 0.25;
    precession.chi2[2] = -0.20;
    precession.tau_ref = -5000.0;

    setup_start = clock();
    status = phenom_tphm_tdi_source_initialize(
        &source, m1, m2, &precession, carriers, 5,
        tc, 1.0, t0-2048.0, 0.0,
        acos(0.3), 0.5*M_PI, 0.4);
    if(status != 0)
    {
        fprintf(stderr, "TPHM TDI source initialization failed: %d\n", status);
        return 1;
    }
    printf("tphm_source_setup_seconds %.6f\n",
           (double)(clock()-setup_start)/(double)CLOCKS_PER_SEC);

    time_a = calloc((size_t)n, sizeof(*time_a));
    time_b = calloc((size_t)n, sizeof(*time_b));
    hp_a = calloc((size_t)n, sizeof(*hp_a));
    hc_a = calloc((size_t)n, sizeof(*hc_a));
    hp_b = calloc((size_t)n, sizeof(*hp_b));
    hc_b = calloc((size_t)n, sizeof(*hc_b));
    X = calloc((size_t)n, sizeof(*X));
    Y = calloc((size_t)n, sizeof(*Y));
    Z = calloc((size_t)n, sizeof(*Z));
    if(time_a == NULL || time_b == NULL || hp_a == NULL || hc_a == NULL ||
       hp_b == NULL || hc_b == NULL || X == NULL || Y == NULL || Z == NULL)
    {
        status = 2;
        goto cleanup;
    }
    for(i=0; i<n; i++)
    {
        time_a[i] = t0+sample_dt*(double)i;
        time_b[i] = t0+sample_dt*(double)(i+overlap_offset);
    }

    status = phenom_tphm_complex_polarizations(
        &source, n, time_a, hp_a, hc_a);
    if(status != 0) goto cleanup;
    status = phenom_tphm_complex_polarizations(
        &source, n, time_b, hp_b, hc_b);
    if(status != 0) goto cleanup;
    for(i=overlap_offset; i<n; i++)
    {
        int j = i-overlap_offset;
        worst_overlap = fmax(worst_overlap,
                              relative_error(hp_a[i], hp_b[j]));
        worst_overlap = fmax(worst_overlap,
                              relative_error(hc_a[i], hc_b[j]));
    }

    setup_start = clock();
    if(test_tdi2) thm_set_tdi_generation(2);
    thm_set_tdi2_full_numerical(
        test_tdi2 && getenv("TPHM_TEST_TDI2_EXACT") != NULL);
    if(test_tdi2 && getenv("TPHM_TEST_TDI2_FROZEN_PROJECTION") != NULL)
    {
        thm_set_tdi2_chain_taylor(0);
        thm_set_tdi2_frozen_projection(1);
    }
    if(test_tdi2 && getenv("TPHM_TEST_TDI2_CHAIN_TAYLOR") != NULL)
    {
        thm_set_tdi2_chain_taylor(1);
        thm_set_tdi2_frozen_projection(0);
    }
    if(test_tdi2 && getenv("TPHM_TEST_TDI2_LIGHT_TIME") != NULL)
        thm_set_tdi2_light_time_solver(
            atoi(getenv("TPHM_TEST_TDI2_LIGHT_TIME")));
    context = thm_observation_context_create(0.0);
    workspace = thm_worker_workspace_create(context);
    if(context == NULL || workspace == NULL)
    {
        status = 3;
        goto cleanup;
    }
    printf("tphm_observation_setup_seconds %.6f\n",
           (double)(clock()-setup_start)/(double)CLOCKS_PER_SEC);
    start = clock();
    status = phenom_tphm_evaluate_tdi_context(
        context, workspace, &source, n, time_a,
        2.31, 0.57, X, Y, Z);
    stop = clock();
    if(status != 0) goto cleanup;
    for(i=0; i<n; i++)
    {
        if(!isfinite(creal(X[i])) || !isfinite(cimag(X[i])) ||
           !isfinite(creal(Y[i])) || !isfinite(cimag(Y[i])) ||
           !isfinite(creal(Z[i])) || !isfinite(cimag(Z[i])))
        {
            status = 4;
            goto cleanup;
        }
        max_tdi = fmax(max_tdi, cabs(X[i]));
        max_tdi = fmax(max_tdi, cabs(Y[i]));
        max_tdi = fmax(max_tdi, cabs(Z[i]));
    }

    printf("tphm_partition_overlap_max_relative_error %.6e\n",
           worst_overlap);
    printf("tphm_batched_tdi_samples %d unique_input_upper_bound %d\n",
           n, (test_tdi2 ? 96 : 24)*n);
    printf("tphm_batched_tdi_max_abs %.6e\n", max_tdi);
    printf("tphm_batched_tdi_seconds %.6f\n",
           (double)(stop-start)/(double)CLOCKS_PER_SEC);
    if(worst_overlap > 1.0e-6 || max_tdi <= 0.0)
    {
        status = 5;
        goto cleanup;
    }

    /* Two complementary blocks must reproduce a single tapered block.  The
     * blocks deliberately use different even-DF-compatible heterodynes. */
    {
        const int block_nfft = 4096;
        const double block_dt = 60.0;
        const double total_end = t0+8160.0;
        const double wdm_df = 1.0/(2.0*1.875*4096.0);
        const double shift0 = 300.0*wdm_df;
        const double shift1 = shift0;
        const double edge_roll = 480.0;
        const double boundary = t0+3840.0;
        const double boundary_end = boundary+480.0;
        const double second_start = boundary-480.0;
        int ch;

        status = thm_build_complex_tdi_fft_block_context(
            context, workspace, phenom_tphm_complex_polarizations, &source,
            2.31, 0.57, block_nfft, t0, block_dt, shift0,
            t0, t0+edge_roll, total_end-edge_roll, total_end,
            &full_block);
        if(status != 0) goto cleanup;
        status = thm_build_complex_tdi_fft_block_context(
            context, workspace, phenom_tphm_complex_polarizations, &source,
            2.31, 0.57, block_nfft, t0, block_dt, shift0,
            t0, t0+edge_roll, boundary, boundary_end,
            &partition[0]);
        if(status != 0) goto cleanup;
        status = thm_build_complex_tdi_fft_block_context(
            context, workspace, phenom_tphm_complex_polarizations, &source,
            2.31, 0.57, block_nfft, second_start, block_dt, shift1,
            boundary, boundary_end, total_end-edge_roll, total_end,
            &partition[1]);
        if(status != 0) goto cleanup;

        status = thm_complex_fft_blocks_to_sparse_wdm_context(
            context, 1, &full_block, t0, total_end,
            0.012, 0.028, &wdm_full);
        if(status != 0) goto cleanup;
        status = thm_complex_fft_blocks_to_sparse_wdm_context(
            context, 2, partition, t0, total_end,
            0.012, 0.028, &wdm_partition);
        if(status != 0) goto cleanup;

        for(ch=0; ch<3; ch++)
        {
            THMSparseWDMChannel *a = &wdm_full.channel[ch];
            THMSparseWDMChannel *b = &wdm_partition.channel[ch];
            double aa = 0.0, bb = 0.0, ab = 0.0;
            double match, mismatch;

            if(a->npixels != b->npixels)
            {
                status = 6;
                goto cleanup;
            }
            for(i=0; i<a->npixels; i++)
            {
                if(a->n[i] != b->n[i] || a->m[i] != b->m[i])
                {
                    status = 7;
                    goto cleanup;
                }
                aa += a->value[i]*a->value[i];
                bb += b->value[i]*b->value[i];
                ab += a->value[i]*b->value[i];
            }
            match = ab/sqrt(aa*bb);
            mismatch = 1.0-match;
            if(mismatch > worst_wdm_mismatch)
                worst_wdm_mismatch = mismatch;
            printf("tphm_partition_wdm_channel %d pixels %d match %.12f mismatch %.6e\n",
                   ch, a->npixels, match, mismatch);
        }
    }
    if(worst_wdm_mismatch > 2.0e-5) status = 8;
    if(status != 0) goto cleanup;

    /* Automatic carrier-family blocks use independent narrow heterodynes and
     * then hand the exact complementary taper to one complete endpoint FFT. */
    {
        const double pixel_dt = 4096.0*1.875;
        const double efficient_start = 3902.0*pixel_dt;
        const double efficient_endpoint = 3904.0*pixel_dt;
        const double efficient_stop = 3906.0*pixel_dt;
        const int reference_nfft = 32*4096;
        clock_t plan_start, plan_stop;
        int ch;

        plan_start = clock();
        status = phenom_tphm_build_narrow_partition_context(
            context, workspace, &source, 2.31, 0.57,
            efficient_start, efficient_stop, efficient_endpoint,
            pixel_dt, 0.016, pixel_dt, &efficient_plan);
        if(status != 0)
        {
            fprintf(stderr, "TPHM narrow partition planner failed: %d\n",
                    status);
            goto cleanup;
        }
        plan_stop = clock();
        printf("tphm_narrow_plan_build_seconds %.6f\n",
               (double)(plan_stop-plan_start)/(double)CLOCKS_PER_SEC);
        printf("tphm_narrow_plan_stage_seconds planning %.6f layout %.6f shared_dynamics %.6f envelope_cache %.6f response %.6f early_blocks %.6f endpoint %.6f\n",
               efficient_plan.planning_seconds,
               efficient_plan.layout_seconds,
               efficient_plan.shared_dynamics_seconds,
               efficient_plan.envelope_cache_seconds,
               efficient_plan.response_seconds,
               efficient_plan.early_blocks_seconds,
               efficient_plan.endpoint_seconds);
        printf("tphm_narrow_source_seconds total %.6f phase_anchor %.6f carrier %.6f calls %d samples %lld dynamics_samples %d envelope_groups %d envelope_samples %d post_tdi_groups %d post_tdi_samples %d\n",
               efficient_plan.early_source_seconds,
               efficient_plan.early_phase_anchor_seconds,
               efficient_plan.early_carrier_seconds,
               efficient_plan.early_source_calls,
               efficient_plan.early_source_samples,
               efficient_plan.shared_dynamics_samples,
               efficient_plan.used_complex_envelope_groups,
               efficient_plan.envelope_cache_samples,
               efficient_plan.used_post_tdi_envelope_groups,
               efficient_plan.post_tdi_envelope_samples);
        status = thm_build_complex_tdi_fft_block_interpolated_context(
            context, workspace, phenom_tphm_complex_polarizations, &source,
            2.31, 0.57, reference_nfft, efficient_start, 1.875,
            0.9375, 0.0,
            efficient_start, efficient_start+pixel_dt,
            efficient_stop-pixel_dt, efficient_stop,
            &efficient_reference);
        if(status != 0) goto cleanup;
        efficient_reference.frequency_start = 0.0;
        efficient_reference.frequency_stop = 4095.0/(2.0*4096.0*1.875);
        efficient_reference.carrier_index = -1;
        efficient_reference.is_endpoint = 1;

        status = phenom_tphm_partition_to_wdm_context(
            context, &efficient_plan, &wdm_efficient);
        if(status != 0) goto cleanup;
        status = thm_complex_fft_blocks_to_planned_sparse_wdm_context(
            context, 1, &efficient_reference, &wdm_efficient_reference);
        if(status != 0) goto cleanup;

        printf("tphm_narrow_plan_blocks %d narrow %d endpoint %d fft_samples %lld reference_fft_samples %d wdm_layers %d pixels_per_channel %d endpoint_band %.6e %.6e\n",
               efficient_plan.nblocks, efficient_plan.narrow_blocks,
               efficient_plan.endpoint_blocks,
               efficient_plan.total_fft_samples, reference_nfft,
               efficient_plan.wdm_active_layers,
               efficient_plan.wdm_pixels_per_channel,
               efficient_plan.endpoint_frequency_start,
               efficient_plan.endpoint_frequency_stop);
        for(i=0; i<efficient_plan.nblocks; i++)
        {
            const THMComplexTDIFFTBlock *b = &efficient_plan.blocks[i];
            printf("tphm_narrow_block %d carrier %d endpoint %d nfft %d dt %.6e shift %.6e band %.6e %.6e time %.6e %.6e\n",
                   i, b->carrier_index, b->is_endpoint, b->nfft,
                   b->sample_dt, b->heterodyne_frequency,
                   b->frequency_start, b->frequency_stop,
                   b->nonzero_start, b->nonzero_end);
        }
        for(ch=0; ch<3; ch++)
        {
            THMSparseWDMChannel *b = &wdm_efficient.channel[ch];
            THMSparseWDMChannel *a = &wdm_efficient_reference.channel[ch];
            double mismatch = sparse_wdm_union_mismatch(b, a);
            double match = 1.0-mismatch;
            printf("tphm_narrow_wdm_channel %d pixels %d match %.12f mismatch %.6e\n",
                   ch, b->npixels, match, mismatch);
            if(mismatch > 5.0e-5) status = 11;
            printf("tphm_compact_support_channel %d compact_pixels %d conservative_pixels %d captured_power %.12f conservative_match %.12f\n",
                   ch, b->npixels, a->npixels,
                   sparse_wdm_power(b)/sparse_wdm_power(a),
                   1.0-sparse_wdm_reference_support_mismatch(
                       b, a));
        }
    }
    if(status == 0 && getenv("TPHM_FULL_YEAR_BENCHMARK") != NULL)
    {
        PhenomTPHMPartitionPlan year_plan;
        THMSparseWDMTriplet year_wdm;
        THMSparseWDMTriplet year_conservative;
        double year_stop, year_endpoint, year_rise;
        clock_t year_start, year_finish;
        int year_pixels;
        phenom_tphm_partition_plan_init(&year_plan);
        thm_sparse_wdm_triplet_init(&year_wdm);
        thm_sparse_wdm_triplet_init(&year_conservative);
        phenom_tphm_default_endpoint_window_context(
            context, &source, &year_endpoint, &year_rise, &year_stop);
        year_start = clock();
        status = phenom_tphm_build_narrow_partition_context(
            context, workspace, &source, 2.31, 0.57,
            0.0, year_stop, year_endpoint, year_rise,
            0.016, 1.0e5, &year_plan);
        year_finish = clock();
        printf("tphm_year_plan_status %d blocks %d narrow %d endpoint %d fft_samples %lld seconds %.6f planning %.6f layout %.6f shared_dynamics %.6f envelope_cache %.6f response %.6f early_blocks %.6f endpoint_time %.6f wdm_layers %d pixels_per_channel %d endpoint_band %.6e %.6e\n",
               status, year_plan.nblocks, year_plan.narrow_blocks,
               year_plan.endpoint_blocks, year_plan.total_fft_samples,
               (double)(year_finish-year_start)/(double)CLOCKS_PER_SEC,
               year_plan.planning_seconds, year_plan.layout_seconds,
               year_plan.shared_dynamics_seconds,
               year_plan.envelope_cache_seconds,
               year_plan.response_seconds,
               year_plan.early_blocks_seconds, year_plan.endpoint_seconds,
               year_plan.wdm_active_layers,
               year_plan.wdm_pixels_per_channel,
               year_plan.endpoint_frequency_start,
               year_plan.endpoint_frequency_stop);
        printf("tphm_year_source_seconds total %.6f phase_anchor %.6f carrier %.6f calls %d samples %lld dynamics_samples %d envelope_groups %d envelope_samples %d post_tdi_groups %d post_tdi_samples %d\n",
               year_plan.early_source_seconds,
               year_plan.early_phase_anchor_seconds,
               year_plan.early_carrier_seconds,
               year_plan.early_source_calls,
               year_plan.early_source_samples,
               year_plan.shared_dynamics_samples,
               year_plan.used_complex_envelope_groups,
               year_plan.envelope_cache_samples,
               year_plan.used_post_tdi_envelope_groups,
               year_plan.post_tdi_envelope_samples);
        if(status == 0 && year_plan.nblocks > 0)
        {
            const THMComplexTDIFFTBlock *endpoint =
                &year_plan.blocks[year_plan.nblocks-1];
            for(i=0; i<3; i++) print_endpoint_spectral_support(endpoint, i);
        }
        if(status == 0)
        {
            year_start = clock();
            status = phenom_tphm_partition_to_wdm_context(
                context, &year_plan, &year_wdm);
            year_finish = clock();
            year_pixels = year_wdm.channel[0].npixels+
                          year_wdm.channel[1].npixels+
                          year_wdm.channel[2].npixels;
            printf("tphm_year_wdm_status %d total_channel_pixels %d seconds %.6f\n",
                   status, year_pixels,
                   (double)(year_finish-year_start)/(double)CLOCKS_PER_SEC);
            if(status == 0 && getenv("TPHM_YEAR_WDM_DUMP") != NULL)
            {
                const char *filename = getenv("TPHM_YEAR_WDM_DUMP");
                if(!write_tphm_year_wdm_dump(filename, &year_wdm))
                {
                    fprintf(stderr, "Could not write TPHM year WDM dump: %s\n", filename);
                    status = 12;
                }
                else
                    printf("tphm_year_wdm_dump %s\n", filename);
            }
            if(status == 0)
            {
                status = phenom_tphm_partition_to_conservative_wdm_context(
                    context, &year_plan, &year_conservative);
            }
            if(status == 0)
            {
                int ch;
                for(ch=0; ch<3; ch++)
                {
                    double mismatch = sparse_wdm_union_mismatch(
                        &year_wdm.channel[ch],
                        &year_conservative.channel[ch]);
                    printf("tphm_year_compact_channel %d compact_pixels %d conservative_pixels %d captured_power %.12f match %.12f mismatch %.6e\n",
                           ch, year_wdm.channel[ch].npixels,
                           year_conservative.channel[ch].npixels,
                           sparse_wdm_power(&year_wdm.channel[ch])/
                           sparse_wdm_power(&year_conservative.channel[ch]),
                           1.0-mismatch, mismatch);
                }
            }
            if(status == 0 &&
               getenv("TPHM_COMPLEX_ENVELOPE_COMPARE") != NULL)
            {
                PhenomTPHMPartitionPlan direct_plan, envelope_plan;
                PhenomTPHMPartitionPlan post_tdi_plan;
                THMSparseWDMTriplet direct_wdm, envelope_wdm, post_tdi_wdm;
                clock_t direct_start, direct_stop;
                clock_t envelope_start, envelope_stop;
                clock_t post_tdi_start, post_tdi_stop;
                int compare_status = 0, ch;
                phenom_tphm_partition_plan_init(&direct_plan);
                phenom_tphm_partition_plan_init(&envelope_plan);
                phenom_tphm_partition_plan_init(&post_tdi_plan);
                thm_sparse_wdm_triplet_init(&direct_wdm);
                thm_sparse_wdm_triplet_init(&envelope_wdm);
                thm_sparse_wdm_triplet_init(&post_tdi_wdm);
                phenom_tphm_set_complex_envelope_groups(0);
                phenom_tphm_set_post_tdi_envelope_groups(0);
                direct_start = clock();
                compare_status = phenom_tphm_build_narrow_partition_context(
                    context, workspace, &source, 2.31, 0.57,
                    0.0, year_stop, year_endpoint, year_rise,
                    0.016, 1.0e5, &direct_plan);
                if(compare_status == 0)
                    compare_status = phenom_tphm_partition_to_wdm_context(
                        context, &direct_plan, &direct_wdm);
                direct_stop = clock();
                phenom_tphm_set_complex_envelope_groups(1);
                envelope_start = clock();
                if(compare_status == 0)
                    compare_status = phenom_tphm_build_narrow_partition_context(
                        context, workspace, &source, 2.31, 0.57,
                        0.0, year_stop, year_endpoint, year_rise,
                        0.016, 1.0e5, &envelope_plan);
                if(compare_status == 0)
                    compare_status = phenom_tphm_partition_to_wdm_context(
                        context, &envelope_plan, &envelope_wdm);
                envelope_stop = clock();
                phenom_tphm_set_complex_envelope_groups(0);
                phenom_tphm_set_post_tdi_envelope_groups(1);
                post_tdi_start = clock();
                if(compare_status == 0)
                    compare_status = phenom_tphm_build_narrow_partition_context(
                        context, workspace, &source, 2.31, 0.57,
                        0.0, year_stop, year_endpoint, year_rise,
                        0.016, 1.0e5, &post_tdi_plan);
                if(compare_status == 0)
                    compare_status = phenom_tphm_partition_to_wdm_context(
                        context, &post_tdi_plan, &post_tdi_wdm);
                post_tdi_stop = clock();
                phenom_tphm_set_complex_envelope_groups(-1);
                phenom_tphm_set_post_tdi_envelope_groups(-1);
                printf("tphm_envelope_compare_status %d direct_seconds %.6f envelope_seconds %.6f envelope_cache_seconds %.6f envelope_samples %d post_tdi_seconds %.6f post_tdi_cache_seconds %.6f post_tdi_samples %d\n",
                       compare_status,
                       (double)(direct_stop-direct_start)/
                           (double)CLOCKS_PER_SEC,
                       (double)(envelope_stop-envelope_start)/
                           (double)CLOCKS_PER_SEC,
                       envelope_plan.envelope_cache_seconds,
                       envelope_plan.envelope_cache_samples,
                       (double)(post_tdi_stop-post_tdi_start)/
                           (double)CLOCKS_PER_SEC,
                       post_tdi_plan.response_seconds,
                       post_tdi_plan.post_tdi_envelope_samples);
                if(compare_status == 0)
                {
                    for(ch=0; ch<3; ch++)
                    {
                        double mismatch = sparse_wdm_union_mismatch(
                            &direct_wdm.channel[ch],
                            &envelope_wdm.channel[ch]);
                        double direct_power = sparse_wdm_power(
                            &direct_wdm.channel[ch]);
                        double envelope_power = sparse_wdm_power(
                            &envelope_wdm.channel[ch]);
                        printf("tphm_envelope_compare_channel %d match %.15f mismatch %.6e power_ratio %.15e direct_pixels %d envelope_pixels %d\n",
                               ch, 1.0-mismatch, mismatch,
                               envelope_power/direct_power,
                               direct_wdm.channel[ch].npixels,
                               envelope_wdm.channel[ch].npixels);
                        mismatch = sparse_wdm_union_mismatch(
                            &direct_wdm.channel[ch],
                            &post_tdi_wdm.channel[ch]);
                        envelope_power = sparse_wdm_power(
                            &post_tdi_wdm.channel[ch]);
                        printf("tphm_post_tdi_envelope_compare_channel %d match %.15f mismatch %.6e power_ratio %.15e direct_pixels %d envelope_pixels %d\n",
                               ch, 1.0-mismatch, mismatch,
                               envelope_power/direct_power,
                               direct_wdm.channel[ch].npixels,
                               post_tdi_wdm.channel[ch].npixels);
                    }
                }
                thm_sparse_wdm_triplet_free(&post_tdi_wdm);
                thm_sparse_wdm_triplet_free(&envelope_wdm);
                thm_sparse_wdm_triplet_free(&direct_wdm);
                phenom_tphm_partition_plan_free(&post_tdi_plan);
                phenom_tphm_partition_plan_free(&envelope_plan);
                phenom_tphm_partition_plan_free(&direct_plan);
                if(compare_status != 0) status = 20+compare_status;
            }
            if(status == 0)
            {
                IMRPhenomTPHMPrecessionConfig aligned_precession;
                PhenomTPHMTDISource aligned_source;
                PhenomTPHMPartitionPlan aligned_plan;
                memset(&aligned_precession, 0, sizeof(aligned_precession));
                memset(&aligned_source, 0, sizeof(aligned_source));
                aligned_precession.chi1[2] = precession.chi1[2];
                aligned_precession.chi2[2] = precession.chi2[2];
                aligned_precession.tau_ref = precession.tau_ref;
                phenom_tphm_partition_plan_init(&aligned_plan);
                status = phenom_tphm_tdi_source_initialize(
                    &aligned_source, m1, m2, &aligned_precession,
                    carriers, 5, tc, 1.0, t0-2048.0, 0.0,
                    acos(0.3), 0.5*M_PI, 0.4);
                if(status == 0)
                    status = phenom_tphm_build_narrow_partition_context(
                        context, workspace, &aligned_source, 2.31, 0.57,
                        0.0, year_stop, year_endpoint, year_rise,
                        0.016, 1.0e5, &aligned_plan);
                if(status == 0)
                    printf("tphm_year_aligned_support layers %d pixels_per_channel %d precessing_to_aligned_ratio %.6f fft_samples %lld\n",
                           aligned_plan.wdm_active_layers,
                           aligned_plan.wdm_pixels_per_channel,
                           (double)year_plan.wdm_pixels_per_channel/
                           (double)aligned_plan.wdm_pixels_per_channel,
                           aligned_plan.total_fft_samples);
                phenom_tphm_partition_plan_free(&aligned_plan);
                phenom_tphm_tdi_source_destroy(&aligned_source);
            }
        }
        if(status == 0)
            status = benchmark_intrinsic_thm_tphm(
                &source, carriers,
                (int)(sizeof(carriers)/sizeof(carriers[0])));
        if(status == 0 && !test_tdi2)
        {
            THMSparseWDMTriplet thm_year;
            THMSparseWDMTriplet thm_year_fast;
            THMSparseWDMTriplet thm_year_reference;
            double params[11];
            clock_t thm_start, thm_finish;
            int ch;
            thm_sparse_wdm_triplet_init(&thm_year);
            thm_sparse_wdm_triplet_init(&thm_year_fast);
            thm_sparse_wdm_triplet_init(&thm_year_reference);
            params[0] = m1;
            params[1] = m2;
            params[2] = precession.chi1[2];
            params[3] = precession.chi2[2];
            params[4] = 0.0;
            params[5] = tc;
            params[6] = 0.0;
            params[7] = 0.5*M_PI-2.31;
            params[8] = 0.57;
            params[9] = source.polarization;
            params[10] = cos(source.observer_theta);
            thm_set_wdm_split_early_fft(1, 0.016, 3.0);
            thm_start = clock();
            status = generate_thm_tdi_wdm_context(
                context, workspace, params, "default", &thm_year);
            thm_finish = clock();
            printf("thm_year_split_fft_status %d seconds %.6f total_channel_pixels %d\n",
                   status,
                   (double)(thm_finish-thm_start)/(double)CLOCKS_PER_SEC,
                   thm_year.channel[0].npixels+
                   thm_year.channel[1].npixels+
                   thm_year.channel[2].npixels);
            if(status == 0)
            {
                thm_year_fast = thm_year;
                thm_sparse_wdm_triplet_init(&thm_year);
            }
            else
            {
                thm_sparse_wdm_triplet_free(&thm_year);
                thm_sparse_wdm_triplet_init(&thm_year);
            }
            thm_start = clock();
            if(getenv("THM_REBUILD_FULL_CADENCE_REFERENCE") == NULL &&
               read_thm_full_cadence_cache(
                   "thm_full_cadence_reference_default.bin", params,
                   &thm_year_reference))
            {
                status = 0;
                thm_finish = clock();
                printf("thm_year_full_fft_reference_cache loaded seconds %.6f\n",
                       (double)(thm_finish-thm_start)/(double)CLOCKS_PER_SEC);
            }
            else
            {
                thm_set_wdm_full_fft_reference(1);
                status = generate_thm_tdi_wdm_context(
                    context, workspace, params, "default",
                    &thm_year_reference);
                thm_set_wdm_full_fft_reference(0);
                thm_finish = clock();
                if(status == 0 && !write_thm_full_cadence_cache(
                       "thm_full_cadence_reference_default.bin", params,
                       &thm_year_reference))
                    fprintf(stderr,
                            "Warning: could not write the full-cadence THM reference cache.\n");
                else if(status == 0)
                    printf("thm_year_full_fft_reference_cache wrote thm_full_cadence_reference_default.bin\n");
            }
            printf("thm_year_full_fft_reference_status %d seconds %.6f total_channel_pixels %d\n",
                   status,
                   (double)(thm_finish-thm_start)/(double)CLOCKS_PER_SEC,
                   thm_year_reference.channel[0].npixels+
                   thm_year_reference.channel[1].npixels+
                   thm_year_reference.channel[2].npixels);
            if(status == 0)
            {
                static const int band_edge[] = {
                    1, 32, 64, 80, 96, 112, 128, 160, 192, 224, 256,
                    512, 1024, 2048, 4096
                };
                const int nband = (int)(sizeof(band_edge)/sizeof(band_edge[0]));
                for(ch=0; ch<3; ch++)
                {
                    SparseWDMIntersectionMetrics common;
                    double mismatch = sparse_wdm_union_mismatch(
                        &thm_year_fast.channel[ch],
                        &thm_year_reference.channel[ch]);
                    double reference_mismatch =
                        sparse_wdm_reference_support_mismatch(
                            &thm_year_fast.channel[ch],
                            &thm_year_reference.channel[ch]);
                    printf("thm_fast_full_fft_partition_channel %d match %.15f mismatch %.6e fast_pixels %d reference_pixels %d\n",
                           ch, 1.0-mismatch, mismatch,
                           thm_year_fast.channel[ch].npixels,
                           thm_year_reference.channel[ch].npixels);
                    printf("thm_fast_full_fft_reference_support_channel %d match %.15f mismatch %.6e\n",
                           ch, 1.0-reference_mismatch,
                           reference_mismatch);
                    printf("thm_fast_full_fft_power_channel %d fast %.15e reference %.15e ratio %.15e\n",
                           ch,
                           sparse_wdm_power(&thm_year_fast.channel[ch]),
                           sparse_wdm_power(&thm_year_reference.channel[ch]),
                           sparse_wdm_power(&thm_year_fast.channel[ch])/
                           sparse_wdm_power(&thm_year_reference.channel[ch]));
                    common = sparse_wdm_intersection_metrics_band(
                        &thm_year_fast.channel[ch],
                        &thm_year_reference.channel[ch], 1, 4095);
                    printf("thm_fast_full_fft_intersection_channel %d match %.15f mismatch %.6e common_pixels %d fast_pixels %d reference_pixels %d captured_reference_power %.15e\n",
                           ch, 1.0-common.mismatch, common.mismatch,
                           common.common_pixels, common.candidate_pixels,
                           common.reference_pixels,
                           common.reference_total_power > 0.0 ?
                           common.reference_power/common.reference_total_power :
                           0.0);
                    for(i=0; i<nband-1; i++)
                    {
                        mismatch = sparse_wdm_union_mismatch_band(
                            &thm_year_fast.channel[ch],
                            &thm_year_reference.channel[ch],
                            band_edge[i], band_edge[i+1]-1);
                        printf("thm_fast_full_fft_band channel %d m %d %d match %.15f mismatch %.6e\n",
                               ch, band_edge[i], band_edge[i+1]-1,
                               1.0-mismatch, mismatch);
                        printf("thm_fast_full_fft_band_power channel %d m %d %d fast %.15e reference %.15e ratio %.15e\n",
                               ch, band_edge[i], band_edge[i+1]-1,
                               sparse_wdm_power_band(
                                   &thm_year_fast.channel[ch],
                                   band_edge[i], band_edge[i+1]-1),
                               sparse_wdm_power_band(
                                   &thm_year_reference.channel[ch],
                                   band_edge[i], band_edge[i+1]-1),
                               sparse_wdm_power_band(
                                   &thm_year_fast.channel[ch],
                                   band_edge[i], band_edge[i+1]-1)/
                               sparse_wdm_power_band(
                                   &thm_year_reference.channel[ch],
                                   band_edge[i], band_edge[i+1]-1));
                        mismatch = sparse_wdm_reference_support_mismatch_band(
                            &thm_year_fast.channel[ch],
                            &thm_year_reference.channel[ch],
                            band_edge[i], band_edge[i+1]-1);
                        printf("thm_fast_full_fft_reference_band channel %d m %d %d match %.15f mismatch %.6e\n",
                               ch, band_edge[i], band_edge[i+1]-1,
                               1.0-mismatch, mismatch);
                        common = sparse_wdm_intersection_metrics_band(
                            &thm_year_fast.channel[ch],
                            &thm_year_reference.channel[ch],
                            band_edge[i], band_edge[i+1]-1);
                        printf("thm_fast_full_fft_intersection_band channel %d m %d %d match %.15f mismatch %.6e common_pixels %d fast_pixels %d reference_pixels %d captured_reference_power %.15e\n",
                               ch, band_edge[i], band_edge[i+1]-1,
                               1.0-common.mismatch, common.mismatch,
                               common.common_pixels,
                               common.candidate_pixels,
                               common.reference_pixels,
                               common.reference_total_power > 0.0 ?
                               common.reference_power/
                               common.reference_total_power : 0.0);
                    }
                    if(getenv("THM_PARTITION_LAYER_DEBUG") != NULL)
                    {
                        static const int diagnostic_layer[] = {
                            6, 7, 12, 13, 14, 15, 16, 17, 40, 41
                        };
                        char layer_filename[128];
                        int diagnostic_index;
                        for(i=1; i<=127; i++)
                        {
                            common = sparse_wdm_intersection_metrics_band(
                                &thm_year_fast.channel[ch],
                                &thm_year_reference.channel[ch], i, i);
                            printf("thm_fast_full_fft_layer channel %d m %d match %.15f mismatch %.6e common_pixels %d fast_pixels %d reference_pixels %d captured_reference_power %.15e reference_power %.15e\n",
                                   ch, i, 1.0-common.mismatch,
                                   common.mismatch, common.common_pixels,
                                   common.candidate_pixels,
                                   common.reference_pixels,
                                   common.reference_total_power > 0.0 ?
                                   common.reference_power/
                                   common.reference_total_power : 0.0,
                                   common.reference_total_power);
                        }
                        for(diagnostic_index=0;
                            diagnostic_index<(int)(sizeof(diagnostic_layer)/
                                                   sizeof(diagnostic_layer[0]));
                            diagnostic_index++)
                        {
                            snprintf(layer_filename,
                                     sizeof(layer_filename),
                                     "/tmp/thm_layer_%d_ch%d.dat",
                                     diagnostic_layer[diagnostic_index], ch);
                            write_sparse_wdm_layer_comparison(
                                layer_filename,
                                &thm_year_fast.channel[ch],
                                &thm_year_reference.channel[ch],
                                diagnostic_layer[diagnostic_index]);
                        }
                    }
                }
            }
            thm_sparse_wdm_triplet_free(&thm_year_reference);
            thm_set_wdm_split_early_fft(0, 0.016, 3.0);
            thm_sparse_wdm_triplet_init(&thm_year);
            thm_start = clock();
            status = generate_thm_tdi_wdm_context(
                context, workspace, params, "default", &thm_year);
            thm_finish = clock();
            printf("thm_year_spa_fft_status %d seconds %.6f total_channel_pixels %d\n",
                   status,
                   (double)(thm_finish-thm_start)/(double)CLOCKS_PER_SEC,
                   thm_year.channel[0].npixels+
                   thm_year.channel[1].npixels+
                   thm_year.channel[2].npixels);
            thm_sparse_wdm_triplet_free(&thm_year);
            thm_sparse_wdm_triplet_free(&thm_year_fast);
        }
        thm_sparse_wdm_triplet_free(&year_wdm);
        thm_sparse_wdm_triplet_free(&year_conservative);
        phenom_tphm_partition_plan_free(&year_plan);
    }

cleanup:
    if(test_tdi2) thm_set_tdi_generation(1);
    thm_set_tdi2_full_numerical(0);
    thm_sparse_wdm_triplet_free(&wdm_efficient);
    thm_sparse_wdm_triplet_free(&wdm_efficient_reference);
    phenom_tphm_partition_plan_free(&efficient_plan);
    thm_complex_tdi_fft_block_free(&efficient_reference);
    thm_sparse_wdm_triplet_free(&wdm_partition);
    thm_sparse_wdm_triplet_free(&wdm_full);
    thm_complex_tdi_fft_block_free(&partition[1]);
    thm_complex_tdi_fft_block_free(&partition[0]);
    thm_complex_tdi_fft_block_free(&full_block);
    thm_worker_workspace_destroy(workspace);
    thm_observation_context_destroy(context);
    free(Z);
    free(Y);
    free(X);
    free(hc_b);
    free(hp_b);
    free(hc_a);
    free(hp_a);
    free(time_b);
    free(time_a);
    phenom_tphm_tdi_source_destroy(&source);
    if(status != 0)
    {
        fprintf(stderr, "PhenomTPHM TDI test failed: %d\n", status);
        return 1;
    }
    printf("PhenomTPHM TDI test passed\n");
    (void)mtot;
    return 0;
}
