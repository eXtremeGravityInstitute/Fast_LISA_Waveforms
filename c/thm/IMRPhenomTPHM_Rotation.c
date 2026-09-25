/*
 * IMRPhenomTPHM mode rotations, based on LALSimulation frame rotations.
 * Copyright (C) 2020 Hector Estelles
 * LAL-free reconstruction and subsequent modifications:
 * Copyright (C) 2026 Neil Cornish
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * This program is free software under the GNU General Public License,
 * version 2 or (at your option) any later version. It is distributed
 * without any warranty; see the GNU GPL for details.
 */

#include "IMRPhenomTPHM_Rotation.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static const double tphm_factorial[11] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0,
    40320.0, 362880.0, 3628800.0
};

typedef struct
{
    double c[11];
    double s[11];
} TPHMHalfAnglePowers;

static void tphm_half_angle_powers(double beta,
                                   TPHMHalfAnglePowers *powers)
{
    double c = cos(0.5*beta);
    double s = sin(0.5*beta);
    int p;

    powers->c[0] = 1.0;
    powers->s[0] = 1.0;
    for(p=1; p<=10; p++)
    {
        powers->c[p] = powers->c[p-1]*c;
        powers->s[p] = powers->s[p-1]*s;
    }
}

static void tphm_negative_phase_table(int ell,
                                      double angle,
                                      double complex *phase)
{
    double complex step = cos(angle)-I*sin(angle);
    int emm;

    phase[ell] = 1.0;
    for(emm=1; emm<=ell; emm++)
    {
        phase[ell+emm] = phase[ell+emm-1]*step;
        phase[ell-emm] = conj(phase[ell+emm]);
    }
}

/*
 * This is d^ell_{mp,m}(beta) in the convention used by the explicit LAL
 * IMRPhenomTPHM polynomials.  For example, the mp=ell row starts with
 * sin(beta/2)^(2 ell) at m=-ell and ends with cos(beta/2)^(2 ell) at m=ell.
 */
static double tphm_wigner_d_from_powers(
    int ell,
    int mp,
    int emm,
    const TPHMHalfAnglePowers *powers)
{
    double prefactor;
    double sum = 0.0;
    int k;

    prefactor = sqrt(tphm_factorial[ell+emm]*
                     tphm_factorial[ell-emm]*
                     tphm_factorial[ell+mp]*
                     tphm_factorial[ell-mp]);

    for(k=0; k<=2*ell; k++)
    {
        int a = ell+emm-k;
        int b = k;
        int cden = mp-emm+k;
        int d = ell-mp-k;
        int cpower = 2*ell+emm-mp-2*k;
        int spower = mp-emm+2*k;
        double term;

        if(a < 0 || b < 0 || cden < 0 || d < 0 ||
           cpower < 0 || spower < 0)
        {
            continue;
        }

        term = prefactor/(tphm_factorial[a]*tphm_factorial[b]*
                          tphm_factorial[cden]*tphm_factorial[d]);
        if(k & 1) term = -term;
        term *= powers->c[cpower]*powers->s[spower];
        sum += term;
    }

    return sum;
}

static double tphm_wigner_d(int ell, int mp, int emm, double beta)
{
    TPHMHalfAnglePowers powers;

    tphm_half_angle_powers(beta, &powers);
    return tphm_wigner_d_from_powers(ell, mp, emm, &powers);
}

int IMRPhenomTPHMBuildRotation(int ell,
                               double alpha,
                               double beta,
                               double gamma,
                               IMRPhenomTPHMRotation *rotation)
{
    TPHMHalfAnglePowers powers;
    double complex exp_alpha[IMRPHENOMTPHM_MAX_M_COUNT];
    double complex exp_gamma[IMRPHENOMTPHM_MAX_M_COUNT];
    int emm;
    int mp;

    if(rotation == NULL || ell < IMRPHENOMTPHM_MIN_ELL ||
       ell > IMRPHENOMTPHM_MAX_ELL)
    {
        return 1;
    }

    memset(rotation, 0, sizeof(*rotation));
    rotation->ell = ell;
    tphm_half_angle_powers(beta, &powers);
    tphm_negative_phase_table(ell, alpha, exp_alpha);
    tphm_negative_phase_table(ell, gamma, exp_gamma);
    for(emm=-ell; emm<=ell; emm++)
    {
        for(mp=-ell; mp<=ell; mp++)
        {
            rotation->element[emm+ell][mp+ell] =
                exp_alpha[emm+ell]*
                tphm_wigner_d_from_powers(ell, mp, emm, &powers)*
                exp_gamma[mp+ell];
        }
    }

    return 0;
}

int IMRPhenomTPHMComposeRotations(const IMRPhenomTPHMRotation *left,
                                  const IMRPhenomTPHMRotation *right,
                                  IMRPhenomTPHMRotation *out)
{
    IMRPhenomTPHMRotation product;
    int ell;
    int i;
    int j;
    int k;
    int count;

    if(left == NULL || right == NULL || out == NULL ||
       left->ell != right->ell ||
       left->ell < IMRPHENOMTPHM_MIN_ELL ||
       left->ell > IMRPHENOMTPHM_MAX_ELL)
    {
        return 1;
    }

    ell = left->ell;
    count = 2*ell+1;
    memset(&product, 0, sizeof(product));
    product.ell = ell;
    for(i=0; i<count; i++)
    {
        for(j=0; j<count; j++)
        {
            for(k=0; k<count; k++)
            {
                product.element[i][j] +=
                    left->element[i][k]*right->element[k][j];
            }
        }
    }

    *out = product;
    return 0;
}

int IMRPhenomTPHMAdjointRotation(const IMRPhenomTPHMRotation *rotation,
                                 IMRPhenomTPHMRotation *adjoint)
{
    IMRPhenomTPHMRotation result;
    int i;
    int j;
    int count;

    if(rotation == NULL || adjoint == NULL ||
       rotation->ell < IMRPHENOMTPHM_MIN_ELL ||
       rotation->ell > IMRPHENOMTPHM_MAX_ELL)
    {
        return 1;
    }

    memset(&result, 0, sizeof(result));
    result.ell = rotation->ell;
    count = 2*rotation->ell+1;
    for(i=0; i<count; i++)
    {
        for(j=0; j<count; j++)
        {
            result.element[i][j] = conj(rotation->element[j][i]);
        }
    }

    *adjoint = result;
    return 0;
}

int IMRPhenomTPHMRotateMultipole(const IMRPhenomTPHMRotation *rotation,
                                 const double complex *input,
                                 double complex *output)
{
    double complex result[IMRPHENOMTPHM_MAX_M_COUNT];
    int i;
    int j;
    int count;

    if(rotation == NULL || input == NULL || output == NULL ||
       rotation->ell < IMRPHENOMTPHM_MIN_ELL ||
       rotation->ell > IMRPHENOMTPHM_MAX_ELL)
    {
        return 1;
    }

    count = 2*rotation->ell+1;
    memset(result, 0, sizeof(result));
    for(i=0; i<count; i++)
    {
        for(j=0; j<count; j++)
        {
            result[i] += rotation->element[i][j]*input[j];
        }
    }
    for(i=0; i<count; i++) output[i] = result[i];
    return 0;
}

double complex IMRPhenomTPHMSpinWeightedYMinus2(int ell,
                                                int emm,
                                                double theta,
                                                double phi)
{
    const int mp = 2;
    double prefactor;

    if(ell < IMRPHENOMTPHM_MIN_ELL || ell > IMRPHENOMTPHM_MAX_ELL ||
       emm < -ell || emm > ell)
    {
        return NAN + I*NAN;
    }

    prefactor = sqrt((2.0*(double)ell+1.0)/(4.0*M_PI));
    return prefactor*tphm_wigner_d(ell, mp, emm, theta)*
           cexp(I*(double)emm*phi);
}

int IMRPhenomTPHMBuildProjection(const IMRPhenomTPHMRotation *rotation,
                                 double theta,
                                 double phi,
                                 double psi,
                                 double complex *coefficient)
{
    int ell;
    int emm;
    int mp;

    if(rotation == NULL || coefficient == NULL ||
       rotation->ell < IMRPHENOMTPHM_MIN_ELL ||
       rotation->ell > IMRPHENOMTPHM_MAX_ELL)
    {
        return 1;
    }

    ell = rotation->ell;
    for(mp=-ell; mp<=ell; mp++) coefficient[mp+ell] = 0.0;
    for(emm=-ell; emm<=ell; emm++)
    {
        double complex ylm =
            IMRPhenomTPHMSpinWeightedYMinus2(ell, emm, theta, phi)*
            cexp(2.0*I*psi);
        for(mp=-ell; mp<=ell; mp++)
        {
            coefficient[mp+ell] +=
                ylm*rotation->element[emm+ell][mp+ell];
        }
    }

    return 0;
}

int IMRPhenomTPHMProjectMultipole(const double complex *modes,
                                  int ell,
                                  double theta,
                                  double phi,
                                  double psi,
                                  double complex *strain)
{
    int emm;

    if(modes == NULL || strain == NULL ||
       ell < IMRPHENOMTPHM_MIN_ELL || ell > IMRPHENOMTPHM_MAX_ELL)
    {
        return 1;
    }

    *strain = 0.0;
    for(emm=-ell; emm<=ell; emm++)
    {
        *strain += modes[emm+ell]*
            IMRPhenomTPHMSpinWeightedYMinus2(ell, emm, theta, phi)*
            cexp(2.0*I*psi);
    }
    return 0;
}

int IMRPhenomTPHMBuildFoldedProjection(
    const IMRPhenomTPHMRotation *rotation,
    int abs_emm,
    double theta,
    double phi,
    double psi,
    IMRPhenomTPHMFoldedProjection *projection)
{
    double complex coefficient[IMRPHENOMTPHM_MAX_M_COUNT];
    int ell;
    int status;

    if(rotation == NULL || projection == NULL)
    {
        return 1;
    }
    ell = rotation->ell;
    if(abs_emm <= 0 || abs_emm > ell)
    {
        return 2;
    }

    status = IMRPhenomTPHMBuildProjection(rotation, theta, phi, psi,
                                          coefficient);
    if(status != 0) return status;

    projection->ell = ell;
    projection->abs_emm = abs_emm;
    projection->positive = coefficient[abs_emm+ell];
    projection->negative = coefficient[-abs_emm+ell];
    return 0;
}

double complex IMRPhenomTPHMEvaluateFoldedPair(
    const IMRPhenomTPHMFoldedProjection *projection,
    double amplitude,
    double phase)
{
    double parity;
    double complex positive_mode;

    if(projection == NULL) return NAN + I*NAN;
    parity = (projection->ell & 1) ? -1.0 : 1.0;
    positive_mode = amplitude*cexp(-I*phase);
    return projection->positive*positive_mode +
           parity*projection->negative*conj(positive_mode);
}

int IMRPhenomTPHMProjectFoldedPairsFast(
    int ell,
    double alpha,
    double beta,
    double gamma,
    const double complex *j_projection,
    int npairs,
    const int *abs_m,
    const double complex *positive_mode,
    double complex *strain)
{
    TPHMHalfAnglePowers powers;
    double complex exp_alpha[IMRPHENOMTPHM_MAX_M_COUNT];
    double complex exp_gamma[IMRPHENOMTPHM_MAX_M_COUNT];
    double parity;
    int p;

    if(j_projection == NULL || abs_m == NULL || positive_mode == NULL ||
       strain == NULL || ell < IMRPHENOMTPHM_MIN_ELL ||
       ell > IMRPHENOMTPHM_MAX_ELL || npairs < 0 || npairs > ell)
    {
        return 1;
    }

    tphm_half_angle_powers(beta, &powers);
    tphm_negative_phase_table(ell, alpha, exp_alpha);
    tphm_negative_phase_table(ell, gamma, exp_gamma);
    parity = (ell & 1) ? -1.0 : 1.0;
    *strain = 0.0;

    for(p=0; p<npairs; p++)
    {
        double complex coefficient_positive = 0.0;
        double complex coefficient_negative = 0.0;
        int mp = abs_m[p];
        int emm;

        if(mp <= 0 || mp > ell) return 2;
        for(emm=-ell; emm<=ell; emm++)
        {
            double complex projected_row =
                j_projection[emm+ell]*exp_alpha[emm+ell];
            coefficient_positive += projected_row*
                tphm_wigner_d_from_powers(ell, mp, emm, &powers);
            coefficient_negative += projected_row*
                tphm_wigner_d_from_powers(ell, -mp, emm, &powers);
        }
        coefficient_positive *= exp_gamma[mp+ell];
        coefficient_negative *= exp_gamma[-mp+ell];
        *strain += coefficient_positive*positive_mode[p] +
                   parity*coefficient_negative*conj(positive_mode[p]);
    }

    return 0;
}
