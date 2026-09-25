/*
 * IMRPhenomTPHM is based on the LALSimulation IMRPhenomTPHM implementation.
 * Copyright (C) 2020 Hector Estelles
 * LAL-free reconstruction and subsequent modifications:
 * Copyright (C) 2026 Neil Cornish
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * This program is free software under the GNU General Public License,
 * version 2 or (at your option) any later version. It is distributed
 * without any warranty; see the GNU GPL for details.
 */

#include "IMRPhenomTPHM.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const IMRPhenomTHMMode tphm_default_carriers[] = {
    {2, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}
};

static double tphm_clamp_unit(double value)
{
    if(value > 1.0) return 1.0;
    if(value < -1.0) return -1.0;
    return value;
}

static double tphm_lal_initial_precessing_final_spin(
    double m1,
    double m2,
    const IMRPhenomTPHMPrecessionConfig *config)
{
    double M = m1 + m2;
    double m1M = m1/M;
    double m2M = m2/M;
    double eta = m1M*m2M;
    double s_perp_x = m1M*m1M*config->chi1[0] +
                      m2M*m2M*config->chi2[0];
    double s_perp_y = m1M*m1M*config->chi1[1] +
                      m2M*m2M*config->chi2[1];
    double chi_tot_perp = hypot(s_perp_x, s_perp_y)/(m1M*m1M);
    double a_parallel = XLALSimIMRPhenomXFinalSpin2017(
        eta, config->chi1[2], config->chi2[2]);
    double s_perp = chi_tot_perp*m1M*m1M;
    double afinal = copysign(hypot(a_parallel, s_perp), a_parallel);

    /* PhenomX final-spin version 2, selected internally by numerical TPHM,
     * caps unphysical fit extrapolations before constructing the waveform. */
    return tphm_clamp_unit(afinal);
}

static int tphm_supported_carrier(int ell, int abs_emm)
{
    return (ell == 2 && (abs_emm == 1 || abs_emm == 2)) ||
           (ell == 3 && abs_emm == 3) ||
           (ell == 4 && abs_emm == 4) ||
           (ell == 5 && abs_emm == 5);
}

static int tphm_canonical_modes(IMRPhenomTHMMode *output,
                                const IMRPhenomTHMMode *input,
                                int nmodes)
{
    int i;
    int count = 0;

    if(input == NULL || nmodes <= 0)
    {
        nmodes = (int)(sizeof(tphm_default_carriers)/
                       sizeof(tphm_default_carriers[0]));
        input = tphm_default_carriers;
    }
    if(nmodes > IMRPHENOMTHM_MAX_MODES) return -1;

    for(i = 0; i < nmodes; i++)
    {
        int abs_emm = abs(input[i].emm);
        int j;
        int duplicate = 0;

        if(!tphm_supported_carrier(input[i].ell, abs_emm)) return -1;
        for(j = 0; j < count; j++)
        {
            if(output[j].ell == input[i].ell && output[j].emm == abs_emm)
            {
                duplicate = 1;
                break;
            }
        }
        if(duplicate) continue;
        if(count >= IMRPHENOMTPHM_MAX_CARRIERS) return -1;
        output[count].ell = input[i].ell;
        output[count].emm = abs_emm;
        count++;
    }
    return count;
}

int IMRPhenomTPHMInertialModeIndex(int ell, int emm)
{
    if(ell < IMRPHENOMTPHM_MIN_ELL || ell > IMRPHENOMTPHM_MAX_ELL ||
       emm < -ell || emm > ell)
        return -1;
    return ell*ell - 4 + emm + ell;
}

static int tphm_find_reference_rotation(
    const IMRPhenomTPHMPrecessionSummary *summary,
    int ell,
    IMRPhenomTPHMRotation *j_to_l0)
{
    IMRPhenomTPHMRotation l0_to_j;
    double beta_ref;
    int status;

    /* Undo the dynamic Euler rotation at the reference epoch, including the
     * in-plane-spin gauge used by native numerical TPHM. */
    beta_ref = acos(tphm_clamp_unit(summary->j_frame_z[2]));
    status = IMRPhenomTPHMBuildRotation(ell,
                                        summary->alpha_reference,
                                        beta_ref,
                                        summary->gamma_reference,
                                        &l0_to_j);
    if(status != 0) return status;
    return IMRPhenomTPHMAdjointRotation(&l0_to_j, j_to_l0);
}

static int tphm_initialize(IMRPhenomTPHM *model,
                           double m1,
                           double m2,
                           const IMRPhenomTPHMPrecessionConfig *precession,
                           const IMRPhenomTHMMode *modes,
                           int nmodes,
                           IMRPhenomTPHMMergerReconstruction reconstruction)
{
    const IMRPhenomTHMMode driver_mode = {2, 2};
    IMRPhenomTPHMPrecessionConfig config;
    double spin_swap[3];
    double tau_peak[2];
    double alpha_peak[2];
    double beta_peak[2];
    double gamma_peak[2];
    double initial_precessing_final_spin;
    int ntau_peak;
    int i;
    int status;

    if(model == NULL || precession == NULL || m1 <= 0.0 || m2 <= 0.0)
        return 1;
    if(reconstruction != IMRPHENOMTPHM_RECONSTRUCTION_CONSISTENT &&
       reconstruction != IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
        return 1;
    memset(model, 0, sizeof(*model));
    config = *precession;
    model->merger_reconstruction = reconstruction;

    if(m2 > m1)
    {
        double mass_swap = m1;
        fprintf(stderr,
                "Warning: IMRPhenomTPHM uses m1 >= m2; switching "
                "m1 <-> m2 and chi1 <-> chi2.\n");
        m1 = m2;
        m2 = mass_swap;
        for(i = 0; i < 3; i++)
        {
            spin_swap[i] = config.chi1[i];
            config.chi1[i] = config.chi2[i];
            config.chi2[i] = spin_swap[i];
        }
    }

    model->ncarriers = tphm_canonical_modes(model->carrier_modes,
                                             modes, nmodes);
    if(model->ncarriers <= 0) return 2;
    model->precession = config;

    if(reconstruction == IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
    {
        /* LAL's numerical branch first drives the SpinTaylor evolution with
         * the PhenomX geometric final-spin estimate.  After the evolution it
         * replaces only the co-precessing merger spin by the evolved result.
         * Preserve that two-stage convention solely in compatibility mode. */
        initial_precessing_final_spin =
            tphm_lal_initial_precessing_final_spin(m1, m2, &config);
        status = IMRPhenomTHMInitializeWithPrecessingFinalSpin(
            &model->precession_driver, m1, m2,
            config.chi1[2], config.chi2[2],
            initial_precessing_final_spin, &driver_mode, 1);
    }
    else
    {
        status = IMRPhenomTHMInitialize(&model->precession_driver,
                                        m1, m2,
                                        config.chi1[2], config.chi2[2],
                                        &driver_mode, 1);
    }
    if(status != 0) goto fail;

    if(config.tau_ref < 0.0)
    {
        tau_peak[0] = config.tau_ref;
        tau_peak[1] = 0.0;
        ntau_peak = 2;
    }
    else
    {
        tau_peak[0] = 0.0;
        ntau_peak = 1;
    }
    if(reconstruction == IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
    {
        status = IMRPhenomTPHMEvolveEulerAnglesLALCompatible(
            &model->precession_driver, &model->precession,
            ntau_peak, tau_peak, alpha_peak, beta_peak, gamma_peak,
            &model->precession_summary);
    }
    else
    {
        status = IMRPhenomTPHMEvolveEulerAngles(
            &model->precession_driver, &model->precession,
            ntau_peak, tau_peak, alpha_peak, beta_peak, gamma_peak,
            &model->precession_summary);
    }
    if(status != 0) goto fail;

    if(reconstruction == IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
    {
        status = IMRPhenomTHMInitializeWithPrecessingFinalSpin(
            &model->carrier, m1, m2,
            config.chi1[2], config.chi2[2],
            model->precession_summary.final_spin,
            model->carrier_modes, model->ncarriers);
    }
    else
    {
        status = IMRPhenomTHMInitializeWithFinalSpin(
            &model->carrier, m1, m2,
            config.chi1[2], config.chi2[2],
            model->precession_summary.final_spin,
            model->carrier_modes, model->ncarriers);
    }
    if(status != 0) goto fail;
    if(reconstruction == IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
        model->carrier.use_lal_mode22_amplitude = 1;

    for(i = 0; i < model->ncarriers; i++)
    {
        int ell = model->carrier_modes[i].ell;
        model->active_ell[ell] = 1;
    }
    for(i = IMRPHENOMTPHM_MIN_ELL; i <= IMRPHENOMTPHM_MAX_ELL; i++)
    {
        if(!model->active_ell[i]) continue;
        status = tphm_find_reference_rotation(&model->precession_summary,
                                               i,
                                               &model->j_to_l0[i]);
        if(status != 0) goto fail;
    }

    model->initialized = 1;
    return 0;

fail:
    IMRPhenomTPHMDestroy(model);
    return status != 0 ? status : 3;
}

int IMRPhenomTPHMInitialize(IMRPhenomTPHM *model,
                            double m1,
                            double m2,
                            const IMRPhenomTPHMPrecessionConfig *precession,
                            const IMRPhenomTHMMode *modes,
                            int nmodes)
{
    return tphm_initialize(model, m1, m2, precession, modes, nmodes,
                           IMRPHENOMTPHM_RECONSTRUCTION_CONSISTENT);
}

int IMRPhenomTPHMInitializeWithReconstruction(
    IMRPhenomTPHM *model,
    double m1,
    double m2,
    const IMRPhenomTPHMPrecessionConfig *precession,
    const IMRPhenomTHMMode *modes,
    int nmodes,
    IMRPhenomTPHMMergerReconstruction reconstruction)
{
    return tphm_initialize(model, m1, m2, precession, modes, nmodes,
                           reconstruction);
}

void IMRPhenomTPHMDestroy(IMRPhenomTPHM *model)
{
    if(model == NULL) return;
    IMRPhenomTHMDestroy(&model->carrier);
    IMRPhenomTHMDestroy(&model->precession_driver);
    memset(model, 0, sizeof(*model));
}

static int tphm_evaluate_grid(
    IMRPhenomTPHM *model,
    int n,
    const double *tau,
    double phi22_at_start,
    double observer_theta,
    double observer_phi,
    double polarization,
    double complex *j_modes,
    double complex *l0_modes,
    double complex *strain,
    double complex *strain_quadrature,
    double complex *carrier_strain,
    double complex *carrier_strain_quadrature,
    double *carrier_omega,
    int selected_carrier,
    int compact_carrier_output,
    const double *external_alpha,
    const double *external_beta,
    const double *external_gamma,
    double *alpha_out,
    double *beta_out,
    double *gamma_out)
{
    IMRPhenomTPHMPrecessionSummary summary;
    IMRPhenomTHMModeSample *carrier_samples = NULL;
    double *phi22 = NULL;
    double *alpha = NULL;
    double *beta = NULL;
    double *gamma = NULL;
    double complex j_projection[IMRPHENOMTPHM_MAX_ELL+1]
                               [IMRPHENOMTPHM_MAX_M_COUNT];
    int i;
    int k;
    int ell;
    int status = 0;

    if(model == NULL || !model->initialized || tau == NULL || n < 1)
        return 1;
    if(j_modes == NULL && l0_modes == NULL && strain == NULL &&
       strain_quadrature == NULL && carrier_strain == NULL)
        return 2;
    if((carrier_strain == NULL) != (carrier_strain_quadrature == NULL))
        return 2;
    if((external_alpha == NULL) != (external_beta == NULL) ||
       (external_alpha == NULL) != (external_gamma == NULL))
        return 2;
    if(selected_carrier >= model->ncarriers) return 2;
    for(i = 1; i < n; i++)
    {
        if(tau[i] <= tau[i-1]) return 3;
    }

    phi22 = (double *)calloc((size_t)n, sizeof(*phi22));
    alpha = (double *)calloc((size_t)n, sizeof(*alpha));
    beta = (double *)calloc((size_t)n, sizeof(*beta));
    gamma = (double *)calloc((size_t)n, sizeof(*gamma));
    carrier_samples = (IMRPhenomTHMModeSample *)calloc(
        (size_t)model->ncarriers*(size_t)n, sizeof(*carrier_samples));
    if(phi22 == NULL || alpha == NULL || beta == NULL || gamma == NULL ||
       carrier_samples == NULL)
    {
        status = 4;
        goto cleanup;
    }

    if(external_alpha != NULL)
    {
        for(i = 0; i < n; i++)
        {
            if(!isfinite(external_alpha[i]) ||
               !isfinite(external_beta[i]) ||
               !isfinite(external_gamma[i]))
            {
                status = 6;
                goto cleanup;
            }
            alpha[i] = external_alpha[i];
            beta[i] = external_beta[i];
            gamma[i] = external_gamma[i];
        }
    }
    else if(model->merger_reconstruction ==
            IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
    {
        status = IMRPhenomTPHMEvolveEulerAnglesLALCompatible(
            &model->precession_driver, &model->precession,
            n, tau, alpha, beta, gamma, &summary);
    }
    else
    {
        status = IMRPhenomTPHMEvolveEulerAngles(
            &model->precession_driver, &model->precession,
            n, tau, alpha, beta, gamma, &summary);
    }
    if(status != 0) goto cleanup;
    if(model->merger_reconstruction ==
       IMRPHENOMTPHM_RECONSTRUCTION_LAL_COMPATIBLE)
    {
        status = IMRPhenomTHMBuildAnalyticPhi22Grid(
            &model->carrier, n, tau, phi22_at_start, phi22);
    }
    else
    {
        status = IMRPhenomTHMBuildPhi22Grid(&model->carrier, n, tau,
                                            phi22_at_start, phi22);
    }
    if(status != 0) goto cleanup;
    if(selected_carrier < 0)
    {
        status = IMRPhenomTHMEvaluateGrid(&model->carrier, n, tau, phi22,
                                          carrier_samples);
    }
    else
    {
        for(i=0; i<n; i++)
        {
            status = IMRPhenomTHMEvaluateMode(
                &model->carrier, selected_carrier, tau[i], phi22[i],
                &carrier_samples[(size_t)selected_carrier*(size_t)n+
                                 (size_t)i]);
            if(status != 0) break;
        }
    }
    if(status != 0) goto cleanup;

    if(j_modes != NULL)
        memset(j_modes, 0, (size_t)n*IMRPHENOMTPHM_INERTIAL_MODE_COUNT*
                           sizeof(*j_modes));
    if(l0_modes != NULL)
        memset(l0_modes, 0, (size_t)n*IMRPHENOMTPHM_INERTIAL_MODE_COUNT*
                            sizeof(*l0_modes));
    if(carrier_strain != NULL)
    {
        size_t carrier_count = compact_carrier_output ?
            (size_t)n : (size_t)model->ncarriers*(size_t)n;
        memset(carrier_strain, 0, carrier_count*sizeof(*carrier_strain));
        memset(carrier_strain_quadrature, 0,
               carrier_count*sizeof(*carrier_strain_quadrature));
    }
    if(carrier_omega != NULL)
    {
        size_t carrier_count = compact_carrier_output ?
            (size_t)n : (size_t)model->ncarriers*(size_t)n;
        memset(carrier_omega, 0, carrier_count*sizeof(*carrier_omega));
    }
    memset(j_projection, 0, sizeof(j_projection));
    if(strain != NULL || strain_quadrature != NULL ||
       carrier_strain != NULL)
    {
        for(ell = IMRPHENOMTPHM_MIN_ELL;
            ell <= IMRPHENOMTPHM_MAX_ELL; ell++)
        {
            if(!model->active_ell[ell]) continue;
            status = IMRPhenomTPHMBuildProjection(
                &model->j_to_l0[ell], observer_theta, observer_phi,
                polarization, j_projection[ell]);
            if(status != 0) goto cleanup;
        }
    }

    for(i = 0; i < n; i++)
    {
        double complex total_strain = 0.0;
        double complex total_quadrature = 0.0;

        for(ell = IMRPHENOMTPHM_MIN_ELL;
            ell <= IMRPHENOMTPHM_MAX_ELL; ell++)
        {
            double complex cop[IMRPHENOMTPHM_MAX_M_COUNT] = {0.0};
            double complex jframe[IMRPHENOMTPHM_MAX_M_COUNT] = {0.0};
            double complex l0frame[IMRPHENOMTPHM_MAX_M_COUNT] = {0.0};
            double complex positive_mode[IMRPHENOMTPHM_MAX_CARRIERS];
            int pair_abs_m[IMRPHENOMTPHM_MAX_CARRIERS];
            IMRPhenomTPHMRotation cop_to_j;
            int npairs = 0;
            int emm;

            if(!model->active_ell[ell]) continue;
            for(k = 0; k < model->ncarriers; k++)
            {
                const IMRPhenomTHMModeSample *sample;
                int abs_emm;
                double parity;

                if(model->carrier_modes[k].ell != ell ||
                   (selected_carrier >= 0 && k != selected_carrier))
                    continue;
                abs_emm = model->carrier_modes[k].emm;
                sample = &carrier_samples[(size_t)k*(size_t)n + (size_t)i];
                parity = (ell & 1) ? -1.0 : 1.0;
                cop[abs_emm+ell] = sample->hlm;
                cop[-abs_emm+ell] = parity*conj(sample->hlm);
                pair_abs_m[npairs] = abs_emm;
                positive_mode[npairs] = sample->hlm;
                if(carrier_omega != NULL)
                {
                    size_t output_index = compact_carrier_output ?
                        (size_t)i : (size_t)k*(size_t)n+(size_t)i;
                    carrier_omega[output_index] = sample->omega;
                }
                npairs++;
            }

            if(j_modes != NULL || l0_modes != NULL)
            {
                status = IMRPhenomTPHMBuildRotation(
                    ell, alpha[i], beta[i], gamma[i], &cop_to_j);
                if(status != 0) goto cleanup;
                status = IMRPhenomTPHMRotateMultipole(&cop_to_j, cop, jframe);
                if(status != 0) goto cleanup;
                if(l0_modes != NULL)
                {
                    status = IMRPhenomTPHMRotateMultipole(
                        &model->j_to_l0[ell], jframe, l0frame);
                    if(status != 0) goto cleanup;
                }

                for(emm = -ell; emm <= ell; emm++)
                {
                    int index = IMRPhenomTPHMInertialModeIndex(ell, emm);
                    size_t output_index =
                        (size_t)i*IMRPHENOMTPHM_INERTIAL_MODE_COUNT +
                        (size_t)index;
                    if(j_modes != NULL)
                        j_modes[output_index] = jframe[emm+ell];
                    if(l0_modes != NULL)
                        l0_modes[output_index] = l0frame[emm+ell];
                }
            }
            if(strain != NULL || strain_quadrature != NULL ||
               carrier_strain != NULL)
            {
                double complex ell_strain;
                if(carrier_strain == NULL)
                {
                    status = IMRPhenomTPHMProjectFoldedPairsFast(
                        ell, alpha[i], beta[i], gamma[i], j_projection[ell],
                        npairs, pair_abs_m, positive_mode, &ell_strain);
                    if(status != 0) goto cleanup;
                    total_strain += ell_strain;

                    if(strain_quadrature != NULL)
                    {
                        /* phase -> phase+pi/2 sends h_{l,+m} -> -i h_{l,+m}.
                         * The folded-pair projector reconstructs the conjugate
                         * -m partner with the corresponding opposite shift. */
                        for(k = 0; k < npairs; k++) positive_mode[k] *= -I;
                        status = IMRPhenomTPHMProjectFoldedPairsFast(
                            ell, alpha[i], beta[i], gamma[i],
                            j_projection[ell], npairs, pair_abs_m,
                            positive_mode, &ell_strain);
                        if(status != 0) goto cleanup;
                        total_quadrature += ell_strain;
                    }
                }
                else
                {
                    int p;
                    for(p = 0; p < npairs; p++)
                    {
                        int global_carrier = -1;
                        int one_m = pair_abs_m[p];
                        double complex one_mode = positive_mode[p];
                        double complex one_quadrature;

                        for(k = 0; k < model->ncarriers; k++)
                        {
                            if(model->carrier_modes[k].ell == ell &&
                               model->carrier_modes[k].emm == one_m)
                            {
                                global_carrier = k;
                                break;
                            }
                        }
                        if(global_carrier < 0)
                        {
                            status = 5;
                            goto cleanup;
                        }
                        status = IMRPhenomTPHMProjectFoldedPairsFast(
                            ell, alpha[i], beta[i], gamma[i],
                            j_projection[ell], 1, &one_m, &one_mode,
                            &ell_strain);
                        if(status != 0) goto cleanup;
                        {
                            size_t output_index = compact_carrier_output ?
                                (size_t)i :
                                (size_t)global_carrier*(size_t)n+(size_t)i;
                            carrier_strain[output_index] = ell_strain;
                        }
                        total_strain += ell_strain;

                        one_mode *= -I;
                        status = IMRPhenomTPHMProjectFoldedPairsFast(
                            ell, alpha[i], beta[i], gamma[i],
                            j_projection[ell], 1, &one_m, &one_mode,
                            &one_quadrature);
                        if(status != 0) goto cleanup;
                        {
                            size_t output_index = compact_carrier_output ?
                                (size_t)i :
                                (size_t)global_carrier*(size_t)n+(size_t)i;
                            carrier_strain_quadrature[output_index] =
                                one_quadrature;
                        }
                        total_quadrature += one_quadrature;
                    }
                }
            }
        }
        if(strain != NULL) strain[i] = total_strain;
        if(strain_quadrature != NULL)
            strain_quadrature[i] = total_quadrature;
    }

    if(alpha_out != NULL) memcpy(alpha_out, alpha, (size_t)n*sizeof(*alpha));
    if(beta_out != NULL) memcpy(beta_out, beta, (size_t)n*sizeof(*beta));
    if(gamma_out != NULL) memcpy(gamma_out, gamma, (size_t)n*sizeof(*gamma));

cleanup:
    free(carrier_samples);
    free(gamma);
    free(beta);
    free(alpha);
    free(phi22);
    return status;
}

int IMRPhenomTPHMEvaluateGrid(
    IMRPhenomTPHM *model,
    int n,
    const double *tau,
    double phi22_at_start,
    double observer_theta,
    double observer_phi,
    double polarization,
    double complex *j_modes,
    double complex *l0_modes,
    double complex *strain,
    double *alpha_out,
    double *beta_out,
    double *gamma_out)
{
    return tphm_evaluate_grid(model, n, tau, phi22_at_start,
                              observer_theta, observer_phi, polarization,
                              j_modes, l0_modes, strain, NULL,
                              NULL, NULL, NULL,
                              -1, 0,
                              NULL, NULL, NULL,
                              alpha_out, beta_out, gamma_out);
}

int IMRPhenomTPHMEvaluateGridWithQuadrature(
    IMRPhenomTPHM *model,
    int n,
    const double *tau,
    double phi22_at_start,
    double observer_theta,
    double observer_phi,
    double polarization,
    double complex *j_modes,
    double complex *l0_modes,
    double complex *strain,
    double complex *strain_quadrature,
    double *alpha_out,
    double *beta_out,
    double *gamma_out)
{
    if(strain == NULL || strain_quadrature == NULL) return 2;
    return tphm_evaluate_grid(model, n, tau, phi22_at_start,
                              observer_theta, observer_phi, polarization,
                              j_modes, l0_modes, strain, strain_quadrature,
                              NULL, NULL, NULL,
                              -1, 0,
                              NULL, NULL, NULL,
                              alpha_out, beta_out, gamma_out);
}

int IMRPhenomTPHMEvaluateCarrierGridWithQuadrature(
    IMRPhenomTPHM *model,
    int n,
    const double *tau,
    double phi22_at_start,
    double observer_theta,
    double observer_phi,
    double polarization,
    double complex *carrier_strain,
    double complex *carrier_strain_quadrature,
    double *carrier_omega,
    double *alpha_out,
    double *beta_out,
    double *gamma_out)
{
    if(carrier_strain == NULL || carrier_strain_quadrature == NULL) return 2;
    return tphm_evaluate_grid(model, n, tau, phi22_at_start,
                              observer_theta, observer_phi, polarization,
                              NULL, NULL, NULL, NULL,
                              carrier_strain, carrier_strain_quadrature,
                              carrier_omega,
                              -1, 0,
                              NULL, NULL, NULL,
                              alpha_out, beta_out, gamma_out);
}

int IMRPhenomTPHMEvaluateCarrierGridWithQuadratureAndEuler(
    IMRPhenomTPHM *model,
    int n,
    const double *tau,
    double phi22_at_start,
    double observer_theta,
    double observer_phi,
    double polarization,
    const double *alpha,
    const double *beta,
    const double *gamma,
    double complex *carrier_strain,
    double complex *carrier_strain_quadrature,
    double *carrier_omega)
{
    if(model == NULL || carrier_strain == NULL ||
       carrier_strain_quadrature == NULL || alpha == NULL || beta == NULL ||
       gamma == NULL)
        return 2;
    return tphm_evaluate_grid(model, n, tau, phi22_at_start,
                              observer_theta, observer_phi, polarization,
                              NULL, NULL, NULL, NULL,
                              carrier_strain, carrier_strain_quadrature,
                              carrier_omega, -1, 0,
                              alpha, beta, gamma,
                              NULL, NULL, NULL);
}

int IMRPhenomTPHMEvaluateSelectedCarrierGridWithQuadrature(
    IMRPhenomTPHM *model,
    int carrier_index,
    int n,
    const double *tau,
    double phi22_at_start,
    double observer_theta,
    double observer_phi,
    double polarization,
    double complex *carrier_strain,
    double complex *carrier_strain_quadrature,
    double *carrier_omega,
    double *alpha_out,
    double *beta_out,
    double *gamma_out)
{
    if(model == NULL || carrier_index < 0 ||
       carrier_index >= model->ncarriers || carrier_strain == NULL ||
       carrier_strain_quadrature == NULL)
        return 2;
    return tphm_evaluate_grid(model, n, tau, phi22_at_start,
                              observer_theta, observer_phi, polarization,
                              NULL, NULL, NULL, NULL,
                              carrier_strain, carrier_strain_quadrature,
                              carrier_omega, carrier_index, 1,
                              NULL, NULL, NULL,
                              alpha_out, beta_out, gamma_out);
}

int IMRPhenomTPHMEvaluateSelectedCarrierGridWithQuadratureAndEuler(
    IMRPhenomTPHM *model,
    int carrier_index,
    int n,
    const double *tau,
    double phi22_at_start,
    double observer_theta,
    double observer_phi,
    double polarization,
    const double *alpha,
    const double *beta,
    const double *gamma,
    double complex *carrier_strain,
    double complex *carrier_strain_quadrature,
    double *carrier_omega)
{
    if(model == NULL || carrier_index < 0 ||
       carrier_index >= model->ncarriers || carrier_strain == NULL ||
       carrier_strain_quadrature == NULL || alpha == NULL ||
       beta == NULL || gamma == NULL)
        return 2;
    return tphm_evaluate_grid(model, n, tau, phi22_at_start,
                              observer_theta, observer_phi, polarization,
                              NULL, NULL, NULL, NULL,
                              carrier_strain, carrier_strain_quadrature,
                              carrier_omega, carrier_index, 1,
                              alpha, beta, gamma,
                              NULL, NULL, NULL);
}
