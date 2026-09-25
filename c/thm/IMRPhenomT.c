/*
 * Stripped IMRPhenomT 22 backbone based on LALSimulation IMRPhenomTHM.
 * Copyright (C) 2020 Hector Estelles
 * LAL-free port and subsequent modifications:
 * Copyright (C) 2024, 2026 Neil Cornish
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * This program is free software under the GNU General Public License,
 * version 2 or (at your option) any later version. It is distributed
 * without any warranty; see the GNU GPL for details.
 */

#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <complex.h>
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

#include "IMRPhenomT.h"


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


void allocate_IMRPhenomT(struct IMRPhenomT *IMRPT)
{
    IMRPT->omega = double_vector(14);
    IMRPT->ampR = double_vector(14);
    IMRPT->ampI = double_vector(14);
    IMRPT->carray = double_vector(5);
    IMRPT->CMarray = double_vector(4);
    IMRPT->phOffInsp = 0.0;
    IMRPT->phOffMerger = 0.0;
    IMRPT->phOffRD = 0.0;
    IMRPT->tt0 = 0.0;
    IMRPT->tEarly = 0.0;
}

double complex IMRPhenomTInspiralAmpAnsatzHM(double x, struct IMRPhenomT *IMRPT)
{
        int i;
        double fac = IMRPT->A0*x;
 
        double *xarray;
        double xhalf = sqrt(x);
        double ampreal, ampimag;
        double complex amp;
    
        xarray = double_vector(11);
    
        xarray[0] = 1.0;
        for(i=1; i< 11; i++) xarray[i] = xarray[i-1]*xhalf;
    
        ampreal = 0.0;
        ampimag = 0.0;
        for(i=0; i< 11; i++)
        {
            ampreal += IMRPT->ampR[i]*xarray[i];
            ampimag += IMRPT->ampI[i]*xarray[i];
        }
    
        // log term
        ampreal -= 428./105.*log(16.*x)*xarray[6];
    
        amp = ampreal + I*ampimag;
    
       // amp = crect(ampreal, ampimag);
        
        free(xarray);
    
        return fac*amp;
}

// taken from my LISA_BH code. Doesn't have all the higher PN spin terms, but used here as a check
/* double f_at_t(double m1, double m2, double chi1, double chi2, double tc, double t)
{
    // 3PN f(t)
    int i;
    double f, fr, fny, af, M, eta, chi, theta;
    double PN1, PN15, PN2, PN25, PN3, PN35;
    double theta2, theta3, theta4, theta5, theta6, theta7;
    
    M = m1+m2;
    eta = m1*m2/(M*M);
    chi = (m1*chi1+m2*chi2)/M;
    
    // Taylor T3
    

        theta = pow(eta*(tc-t)/(5.0*M),-1.0/8.0);
        theta2 = theta*theta;
        theta3 = theta2*theta;
        theta4 = theta2*theta2;
        theta5 = theta2*theta3;
        theta6 = theta3*theta3;
        theta7 = theta3*theta4;
        
        PN1 = (11.0/32.0*eta+743.0/2688.0)*theta2;
        PN15 = -3.0*M_PI/10.0*theta3 + (1.0/160.0)*(113.0*chi-38.0*eta*(chi1+chi2))*theta3;
        PN2 = (1855099.0/14450688.0+56975.0/258048.0*eta+371.0/2048.0*eta*eta)*theta4 + (1.0/14450688.0)*(-3386880.0*chi*chi+1512.0*chi1*chi2)*theta4;
        PN25 = -(7729.0/21504.0-13.0/256.0*eta)*M_PI*theta5;
        PN3 = (-720817631400877.0/288412611379200.0+53.0/200.0*M_PI*M_PI+107.0/280.0*M_EULER
               +(25302017977.0/4161798144.0-451.0/2048.0*M_PI*M_PI)*eta-30913.0/1835008.0*eta*eta+235925.0/1769472.0*eta*eta*eta +107.0/280.0*log(2.0*theta))*theta6;
        PN35 = (141769.0/1290240.0*eta*eta - 97765.0/258048.0*eta - 188516689.0/433520640.0)*M_PI*theta7;
        
        //printf("%f %f %e %e %e %e %e\n", t, theta3/(8.0*M*M_PI), PN1, PN15, PN2, PN25, PN3);
        
        f = theta3/(8.0*M*M_PI)*(1.0 + PN1 + PN15 + PN2 + PN25 + PN3 + PN35);
        

    return(f);
    
} */

/*
double t_at_f(double M, double tc, double eta, double chi1, double chi2, double f)
{
    int i;
    double t, x;
    double dt, fring, chi, theta;
    double gamma_E=0.5772156649; //Euler's Constant-- shows up in 3PN term
    
    chi = (m1*chi1+m2*chi2)/M;
    
    x = pow(PI*M*f,2.0/3.0);
    
    t =  tc-pow(x,-4.0)*5.0*M/(256.0*eta)*(1.0      // 0 PN
                        +((1/252.0)*(743.0+924.0*eta)*x)  // 1 PN
                        +(2.0/15.0)*(-48.0*PI+113.0*chi-38.0*eta*(chi1+chi2))*pow(x,3.0/2.0)    // 1.5 PN
                        +(1.0/508032.0)*(3058673.0+5472432.0*eta+4353552.0*eta*eta-5080320.0*chi*chi+127008.0*eta*chi1*chi2)*pow(x,4.0/2.0) // 2 PN
                        +(1.0/756.0)*(3.0*PI*(-7729.0+1092.0*eta)+1512.0*chi*chi*chi-56.0*eta*(1285.0+153.0*eta)*(chi1+chi2)+chi*(147101.0+504.0*eta*(13.0-9.0*chi1*chi2)))*pow(x,5.0/2.0)// 2.5 PN
                        +((6848.0*gamma_E)/105.0+PI*PI/12.0*(512.0-451.0*eta)+(25565.0*eta*eta*eta)/1296.0+(-10052469856691.0+24236159077900.0*eta)/23471078400.0+(35.0*chi*chi)/3.0+eta*eta*(-(15211.0/1728.0)+19.0*chi1*chi1+(245.0*chi1*chi2)/6.0+19.0*chi2*chi2)+8.0*PI/3.0*(-73.0*chi+28.0*eta*(chi1+chi2))-eta/336.0*(26992.0*chi*chi-995.0*chi1*chi2+30688.0*chi*(chi1+chi2))+3424.0/105.0*log(16.0*x))*pow(x,6.0/2.0));  // 3PN
    
    return(t);
    
}

*/



/* ***** Functions for obtaining 22 phase and frequency for the whole time series **** */
/* These are needed as separate functions from the general (l,m) piecewise functions because
   22 phase and frequency need to be precomputed before each mode structure is invoked. */
 
double IMRPhenomTomega22FromTheta(double t, double theta, double eta, struct IMRPhenomT *IMRPT)
{
  double w;

    (void)eta;
    
    // Only using the default 4 region version of IMRPhenomT
 
    if(t < IMRPT->tCut22) // For times earlier than the inspiral-merger boundary, computes inspiral frequency
    {
        w = IMRPhenomTInspiralOmegaAnsatz22(theta, IMRPT);
    }
    else if(t > 0.) // For times later than the 22 peak amplitude time, computes ringdown frequencies
    {
        w = IMRPhenomTRDOmegaAnsatz22(t, IMRPT);
    }
    else // Remaining thing is to compute merger frequency
    {
        w = IMRPhenomTMergerOmegaAnsatz22(t, IMRPT);
        w = IMRPT->omegaRING*(1. - w); // This is done for correcting the rescaling in the merger, as explained in IMRPhenomTMergerOmegaAnsatz22
    }
 
  return w;
}

double IMRPhenomTomega22(double t, double eta, struct IMRPhenomT *IMRPT)
{
    double theta;

    if(t < IMRPT->tCut22)
    {
        theta = pow(-eta*t/5.0, -1.0/8.0);
    }
    else
    {
        theta = 0.0;
    }

    return IMRPhenomTomega22FromTheta(t, theta, eta, IMRPT);
}

double IMRPhenomTPhaseOmega22(double t, double eta,
                              struct IMRPhenomT *IMRPT)
{
    if(t > 0.0) return IMRPhenomTRDPhaseOmegaAnsatz22(t, IMRPT);
    return IMRPhenomTomega22(t, eta, IMRPT);
}

double IMRPhenomTLALReferenceOmega22(double t, double eta, struct IMRPhenomT *IMRPT)
{
    double theta;

    /*
     * This mirrors LAL's GetTimeOfFreq helper for tmin/tRef.  It is not the
     * same operation as asking for the production waveform frequency at t:
     * below tEarly, LAL feeds the shifted TaylorT3 theta into the same default
     * inspiral ansatz even when PhenomTHMInspiralVersion=0.  Using this only
     * for the reference-frequency root keeps the stripped waveform itself
     * unchanged while matching LAL's phase reference convention.
     */
    if(t < IMRPT->tEarly)
    {
        theta = pow(eta*(IMRPT->tt0 - t)/5.0, -1.0/8.0);
    }
    else if(t < IMRPT->tCut22)
    {
        theta = pow(-eta*t/5.0, -1.0/8.0);
    }
    else
    {
        theta = 0.0;
    }

    return IMRPhenomTomega22FromTheta(t, theta, eta, IMRPT);
}

int IMRPhenomTLALReferenceTimeFromMf(struct IMRPhenomT *IMRPT, double eta, double mf_ref, double *tau_ref)
{
    const double target_omega = 2.0*M_PI*mf_ref;
    double lo = -1000000000.0;
    double hi = 0.0;
    double flo;
    double fhi;
    int i;

    if(IMRPT == NULL || tau_ref == NULL || eta <= 0.0 || mf_ref <= 0.0)
    {
        return 1;
    }

    flo = target_omega - IMRPhenomTLALReferenceOmega22(lo, eta, IMRPT);
    fhi = target_omega - IMRPhenomTLALReferenceOmega22(hi, eta, IMRPT);
    if(!isfinite(flo) || !isfinite(fhi))
    {
        return 2;
    }
    if(flo == 0.0)
    {
        *tau_ref = lo;
        return 0;
    }
    if(fhi == 0.0)
    {
        *tau_ref = hi;
        return 0;
    }
    if(flo*fhi > 0.0)
    {
        return 3;
    }

    for(i = 0; i < 220; i++)
    {
        double mid = 0.5*(lo + hi);
        double fmid = target_omega - IMRPhenomTLALReferenceOmega22(mid, eta, IMRPT);

        if(!isfinite(fmid))
        {
            return 4;
        }
        if(fmid == 0.0)
        {
            *tau_ref = mid;
            return 0;
        }
        if((flo < 0.0 && fmid < 0.0) || (flo > 0.0 && fmid > 0.0))
        {
            lo = mid;
            flo = fmid;
        }
        else
        {
            hi = mid;
            fhi = fmid;
        }

        if(fabs(hi - lo) < 1.0e-10)
        {
            break;
        }
    }

    (void)fhi;
    *tau_ref = 0.5*(lo + hi);
    return 0;
}

double IMRPhenomTPhase22(double t, double eta, struct IMRPhenomT *IMRPT)
{
    double phase;
    double thetabar;

    /*
     * This is the LAL-style analytic antiderivative of the same omega22
     * ansatz evaluated by IMRPhenomTomega22.  The production 22 path in this
     * project usually splines/integrates omega22 directly; this function is
     * kept as a cross-check and as a reference for the higher-mode phase
     * construction.
     */
    if(t < IMRPT->tCut22)
    {
        thetabar = pow(-eta*t, -1.0/8.0);
        phase = IMRPhenomTInspiralPhaseAnsatz22(t, thetabar, eta, IMRPT);
    }
    else if(t > 0.0)
    {
        phase = IMRPhenomTRDPhaseAnsatz22(t, IMRPT);
    }
    else
    {
        phase = IMRPhenomTMergerPhaseAnsatz22(t, IMRPT);
    }

    return phase;
}

double complex IMRPhenomTHMAmp(double t, double x, struct IMRPhenomT *IMRPT)
{
    double complex amp;
 
    if(t < tCUT_Amp)
        {
          amp = IMRPhenomTInspiralAmpAnsatzHM(x, IMRPT);
        }
        else if(t > IMRPT->tshift)
        {
          amp = IMRPhenomTRDAmpAnsatzHM(t, IMRPT);
        }
        else
        {
          amp = IMRPhenomTMergerAmpAnsatzHM(t, IMRPT);
        }
 
    return amp;
}

// This subroutine computes the coefficients used to correct the T3 inspiral model
void InspiralFit(double *params, struct IMRPhenomT *IMRPT)
{
    int i;
    double m1, m2, M, delta, eta, eta2, eta3;
    double chi1, chi2;
    double chi1sq, chi2sq;
    double dchi, S, u;
    
    double *etapow;
    double *Spow;
    
    etapow = double_vector(9);
    Spow = double_vector(7);

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    
    M = m1+m2;
    
    delta = (m1-m2)/M;
    eta = m1*m2/(M*M);
    
    /* Spin parameterisations for calling the calibrated fits*/
    S     = (m1*m1*chi1 + m2*m2*chi2)/(m1*m1 + m2*m2);
    dchi  = chi1 - chi2;
    
    etapow[1] = eta;
    for(i=2; i< 9; i++) etapow[i] = etapow[i-1]*eta;
    Spow[1] = S;
    for(i=2; i< 7; i++) Spow[i] = Spow[i-1]*S;
    
    /* Set collocation points */
  
    double *thetapoints; // Collocation point times as defined in Eq. 11 of PhenomTHM paper https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf */
    double *omegainsppoints;
    
    thetapoints = double_vector(6);
    omegainsppoints = double_vector(6);
    
    thetapoints[0] = 0.33;
    thetapoints[1] = 0.45;
    thetapoints[2] = 0.55;
    thetapoints[3] = 0.65;
    thetapoints[4] = 0.75;
    thetapoints[5] = 0.82;
    
    /* Calibrated value of TaylorT3 t0 parameter for matching frequency at TaylorT3 theta=0.33.
    If 4 regions are selected for reconstruction, first inspiral region will be employ TaylorT3 with this value of t0.
    With this, it is ensured that the early inspiral is described by the proper TaylorT3 PN description.
    If 3 regions are selected, this will be still employed for computing the value of TaylorT3 at theta=0.33 for setting the first collocation point.*/
    
    double tt0 = IMRPhenomT_Inspiral_TaylorT3_t0(etapow, Spow, dchi, delta);
    IMRPT->tt0 = tt0;
    
    // theta value corresponding to this time
    u = pow(-tt0*eta/5.,-1.0/8.0);
    
   // printf("theta = %f\n", u);
  
     /* Boundary time between early and late inspiral regions (only needed if non-default 4 region reconstruction is selected). Boundary is fixed at theta=0.33 for all cases. Inverting the definition of theta, the boundary time is obtained.
     Each inspiral region (if non-default reconstruction is selected) has a different parameterization. For the early inspiral region, theta is defined employing the calibrated quantity t0 (tt0 in the code),
     while in the late inspiral region t0 is set to 0 for avoiding singularities. */
     double tEarly = -5.0/(eta*pow(thetapoints[0],8.0));
     IMRPT->tEarly = tEarly;
     double thetaini = pow(eta*(tt0 - tEarly)/5.,-1./8.0); // This thetaini is the reparameterization of theta=0.33 for the early inspiral region theta definition.
    
   // printf("%e %e\n", tt0, tEarly);
  
     /* initialize collocation point values for solving inspiral coefficient system */
     omegainsppoints[0] = IMRPhenomTTaylorT3(thetaini, IMRPT);
    
    /* IMRPhenomT_Inspiral_Fit does the work of
    IMRPhenomT_Inspiral_Freq_CP1_22
    IMRPhenomT_Inspiral_Freq_CP2_22
    IMRPhenomT_Inspiral_Freq_CP3_22
    IMRPhenomT_Inspiral_Freq_CP4_22
    IMRPhenomT_Inspiral_Freq_CP5_22 */
    
     IMRPhenomT_Inspiral_Fit(omegainsppoints, etapow, Spow, dchi, delta);
    
    /* GSL objects for solving system of equations via LU decomposition */
    gsl_vector *b, *x;
    gsl_matrix *A;
    gsl_permutation *p;
    int s; /* Sign of permutation */
    
         /*Set linear system, which is rank 6 */
         p = gsl_permutation_alloc(6);
         b = gsl_vector_alloc(6);
         x = gsl_vector_alloc(6);
         A = gsl_matrix_alloc(6,6);
  
         /*Set A matrix and b vector*/
  
         double theta, theta8, theta9, theta10, theta11, theta12, theta13, T3offset; // Initialize theta powers the diagonal A matrix
         /* theta is a weighted dimensionless time parameter defined in Eq. 315 of Blanchet 2014 (https://arxiv.org/abs/1310.1528).
            Notice however that here is defined at the (-1/8) power, in order to represent the Post-Newtonian order (1/c). */
  
          int idx;
         /* Set up inspiral coefficient system.
         This system of equations is explained in eq. 10 of THM paper https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf */
         for (idx=0; idx<6; idx++)
         {
                 /* Needed powers of theta */
                 theta = thetapoints[idx];
                 theta8 = pow(theta,8.0);
                 theta9 = theta*theta8;
                 theta10 = theta*theta9;
                 theta11 = theta*theta10;
                 theta12 = theta*theta11;
                 theta13 = theta*theta12;
  
                 T3offset = IMRPhenomTTaylorT3(theta, IMRPT); // TaylorT3 value at the specific collocation point time */
  
                 gsl_vector_set(b,idx,(4./theta/theta/theta)*(omegainsppoints[idx] - T3offset));
  
                 gsl_matrix_set(A,idx,0,theta8);
                 gsl_matrix_set(A,idx,1,theta9);
                 gsl_matrix_set(A,idx,2,theta10);
                 gsl_matrix_set(A,idx,3,theta11);
                 gsl_matrix_set(A,idx,4,theta12);
                 gsl_matrix_set(A,idx,5,theta13);
  
         }
  
         /* We now solve the system A x = b via an LU decomposition */
         gsl_linalg_LU_decomp(A,p,&s);
         gsl_linalg_LU_solve(A,p,b,x);
  
         /* Set inspiral phenomenological coefficients from solution to A x = b */
        for (idx=0; idx<6; idx++) IMRPT->omega[8+idx] = gsl_vector_get(x,idx);
    
    
    /* Deallocate the gsl linear system objects */
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A);
    gsl_permutation_free(p);
    


 
    
    free(thetapoints);
    free(omegainsppoints);
    free(etapow);
    free(Spow);
    
}

static void MergerRingdownFitInternal(double *params,
                                      int override_final_spin,
                                      double final_dimensionless_spin,
                                      struct IMRPhenomT *IMRPT)
{
    int i, idx;
    double m1, m2, M, delta, eta, eta2, eta3;
    double chi1, chi2;
    double chi1sq, chi2sq;
    double dchi, S, u;
    double Mfinal, afinal;

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    
    M = m1+m2;
    
    delta = (m1-m2)/M;
    eta = m1*m2/(M*M);
    
    /* Spin parameterisations for calling the calibrated fits*/
    S     = (m1*m1*chi1 + m2*m2*chi2)/(m1*m1 + m2*m2);
    dchi  = chi1 - chi2;
    
    double *etapow;
    double *Spow;
    
    etapow = double_vector(9);
    Spow = double_vector(7);
    
    etapow[1] = eta;
    for(i=2; i< 9; i++) etapow[i] = etapow[i-1]*eta;
    Spow[1] = S;
    for(i=2; i< 7; i++) Spow[i] = Spow[i-1]*S;
    
    /* Final Mass and Spin (we employ the XLAL functions of PhenomX) */
    // These are in M=1 units
    Mfinal    = XLALSimIMRPhenomXFinalMass2017(eta,chi1,chi2);
    afinal    = override_final_spin ? final_dimensionless_spin :
                                      XLALSimIMRPhenomXFinalSpin2017(eta,chi1,chi2);
    
    // Despite the name, afinal is already the dimensionless final spin chi_f.
    // The QNM fit takes chi_f as input, then the frequency is scaled by Mfinal.
    double fRING     = evaluate_QNMfit_fring22(afinal) / (Mfinal); // 22 mode ringdown frequency
    double fDAMP     = evaluate_QNMfit_fdamp22(afinal) / (Mfinal); //damping frequency of 122 QNM
    double fDAMPn2   = evaluate_QNMfit_fdamp22n2(afinal) / (Mfinal); //damping frequency of 222 QNM

    /* Angular ringdown and damping frequencies (omega =2*pi*f) */
    double alpha1RD  = 2.*M_PI*fDAMP;
    double omegaRING = 2.*M_PI*fRING;
    
    IMRPT->omegaRING = omegaRING;
    IMRPT->alpha1RD = alpha1RD;
    
    /* ********************************** */
    /* *** RINGDOWN COEFFICIENTS ******** */
    /* ********************************** */

    /* For phase and frequency ringdown ansatz, coefficients are obtained from ringdown and damping frequencies, frequency at peak amplitude
       and phenomenological fits of two free coefficients. No linear system solving is needed.
       Coefficient c1 is defined in Eq. [9] of Damour&Nagar 2014 (10.1103/PhysRevD.90.024054, https://arxiv.org/abs/1406.0401).
       Coefficients c2 and c3 are calibrated to NR/Teukolsky data.
       Coefficient c4 is set to zero.
       Explained also in Sec II.C, eq. 26 of https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf */
    
    double omegaPeak;
    


    omegaPeak = IMRPhenomT_PeakFrequency_22(etapow, Spow, dchi, delta); // 22 Frequency at the peak 22 amplitude time (t=0 by definition)
    IMRPT->omegaPeak = omegaPeak;

    IMRPT->carray[3] = IMRPhenomT_RD_Freq_D3_22(etapow, Spow, dchi, delta);
    IMRPT->carray[2] = IMRPhenomT_RD_Freq_D2_22(etapow, Spow, dchi, delta);
    IMRPT->carray[4] = 0.0;
    IMRPT->carray[1] = (1. + IMRPT->carray[3] + IMRPT->carray[4])*(omegaRING - omegaPeak)/IMRPT->carray[2]/(IMRPT->carray[3] + 2.*IMRPT->carray[4]);
    
    /* GSL objects for solving system of equations via LU decomposition */
    gsl_vector *b, *x;
    gsl_matrix *A;
    gsl_permutation *p;
    int s; /* Sign of permutation */
    
    /*Set linear system for solving merger coefficients*/
    p = gsl_permutation_alloc(3);
    b = gsl_vector_alloc(3);
    x = gsl_vector_alloc(3);
    A = gsl_matrix_alloc(3,3);

    /* ********************************** */
    /* *** MERGER COEFFICIENTS ********** */
    /* ********************************** */

    /* In order to obtain the value of the 3 free coefficients of the ansatz defined in IMRPhenomTMergerOmegaAnsatz22, we need to solve a linear system
  where the ansatz is imposed to match a collocation point value at theta(t)=0.95 and to satisfy continuity with the inspiral ansatz and differentiability with the ringdown ansatz.
  Notice that the ansatz definition IMRPhenomTMergerOmegaAnsatz22 differs from eq [27-26] of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf)
  in that here 2 of the coefficients are analytically solved for imposing continuity with the ringdown ansatz and differentiability with the inspiral ansatz.

  Reminder: the ansatz described in IMRPhenomTMergerOmegaAnsatz22 corresponds to the rescaled frequency \bar{\omega}=1 - (\omega / \omega_ring) (eq. 27 of https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf),
  so the system is solved for the rescaled frequency. For the analytical phase of IMRPhenomTMergerPhaseAnsatz22, which is the quantity employed
  in the waveform construction, the factors are already included to produce the correct phase. */
    
    /* Boundary time between late inspiral and merger regions.
    This is selected to correspond to theta=0.81, earlier than the last collocation point at theta=0.82, because in this way
    the derivative at the boundary with the merger region, needed for the merger reconstruction, is less forced. */
    double tCut = -5.0/(eta*pow(0.81,8));
    IMRPT->tCut22 = tCut;
    double omegaCut =  IMRPhenomTInspiralOmegaAnsatz22(0.81, IMRPT);// 22 frequency value at tCut


    double tMerger22 = -5.0/(eta*pow(0.95,8)); // Collocation point time of the merger region. This is placed at a fixed position theta=0.95.
    double omegaMergerCP = 1. - IMRPhenomT_Merger_Freq_CP1_22(etapow, Spow, dchi, delta)/omegaRING; // Collocation point.
    double omegaCutBar = 1. - omegaCut/omegaRING; // Boundary frequency between inspiral and merger ringdown, rescaled.
    
    //printf("tCut %e tMerger22 %e\n", tCut, tMerger22);

    /* Now we need the derivative values at the boundaries.
       Being the inspiral and ringdown ansatz differentiable analytical functions, the analytical derivative could be computed.
       However, for code saving, it is enough to compute a numerical derivative at the boundary. Since the expressions are differentiable,
       the numerical derivative is clean from any noise, and using a first order finite difference scheme is enough to obtain the derivative
       at sufficient precission. */

    double theta2 = pow(-eta*tCut/5.,-1./8);
    double theta1 = pow(-eta*(tCut-0.0000001)/5,-1./8);
    double domegaCut = -(IMRPhenomTInspiralOmegaAnsatz22(theta2, IMRPT) - IMRPhenomTInspiralOmegaAnsatz22(theta1, IMRPT))/(0.0000001)/omegaRING; // Derivative of rescale frequency at the inspiral boundary
    double domegaPeak = -(IMRPhenomTRDOmegaAnsatz22(0.0000001, IMRPT) - IMRPhenomTRDOmegaAnsatz22(0., IMRPT))/(0.0000001)/omegaRING; // Derivative of rescale frequency at the ringdown boundary
    
    IMRPT->domegaPeak = domegaPeak;

    /* Now we set the linear system for solving merger ansatz coefficients of IMRPhenomTMergerOmegaAnsatz22,
    as explained in eq. 30 of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf)
    Reminder: Here two coefficients are already analytically solved, so the system contains only three equations for the three remaining coefficients. */

    /* A_{0,i} and b_{0}. Here we impose continuity with the inspiral region.
     The value of the solution vector coefficient b_{0} is the rescaled frequency at the inspiral boundary, minus the terms of the ansatz which are already fixed analytically.
     The value of the first row of basis matrix are the arcsinh powers of the ansatz evaluated at the boundary time tCut. */

    gsl_complex phi = gsl_complex_rect(alpha1RD*tCut,0);
    gsl_complex arcsinhphi = gsl_complex_arcsinh(phi);
    double ascut = GSL_REAL(arcsinhphi);
    double ascut2 = ascut*ascut;
    double ascut3 = ascut*ascut2;
    double ascut4 = ascut*ascut3;

    gsl_vector_set(b,0,omegaCutBar - (1. - omegaPeak/omegaRING) - (domegaPeak/alpha1RD)*ascut);

    gsl_matrix_set(A,0,0,ascut2);
    gsl_matrix_set(A,0,1,ascut3);
    gsl_matrix_set(A,0,2,ascut4);

    /* A_{1,i} and b_{1}. Here we impose the collocation point at theta=0.95.
     The value of the solution vector coefficient b_{1} is the rescaled value of the collocation point, minus the terms of the ansatz which are already fixed analytically.
     The value of the second row of basis matrix are the arcsinh powers of the ansatz evaluated at the collocation point time tMerger22 (determined from theta(t)=0.95). */

    phi = gsl_complex_rect(alpha1RD*tMerger22,0);
    arcsinhphi = gsl_complex_arcsinh(phi);
    double as025cut = GSL_REAL(arcsinhphi);
    double as025cut2 = as025cut*as025cut;
    double as025cut3 = as025cut*as025cut2;
    double as025cut4 = as025cut*as025cut3;

    gsl_vector_set(b,1,omegaMergerCP - (1. - omegaPeak/omegaRING) - (domegaPeak/alpha1RD)*as025cut);

    gsl_matrix_set(A,1,0,as025cut2);
    gsl_matrix_set(A,1,1,as025cut3);
    gsl_matrix_set(A,1,2,as025cut4);

    /* A_{2,i} and b_{2}. Here we impose differentiability with the ringdown region.
     The value of the solution vector coefficient b_{2} is the rescaled frequency derivative at the ringdown boundary, minus the terms of the ansatz derivative which are already fixed analytically.
     The value of the third row of basis matrix are the arcsinh powers of the ansatz evaluated at the boundary time tCut. */

    double dencut = sqrt(1.0 + tCut*tCut*alpha1RD*alpha1RD); // Factor that appears from the derivative of the ansatz

    gsl_matrix_set(A,2,0,2.0*alpha1RD*ascut/dencut);
    gsl_matrix_set(A,2,1,3.0*alpha1RD*ascut2/dencut);
    gsl_matrix_set(A,2,2,4.0*alpha1RD*ascut3/dencut);

    gsl_vector_set(b,2,domegaCut - domegaPeak/dencut);

    /* We now solve the system A x = b via an LU decomposition */
    gsl_linalg_LU_decomp(A,p,&s);
    gsl_linalg_LU_solve(A,p,b,x);

    /* Set merger phenomenological coefficients from solution to A x = b */
    IMRPT->CMarray[1] = gsl_vector_get(x,0);
    IMRPT->CMarray[2] = gsl_vector_get(x,1);
    IMRPT->CMarray[3] = gsl_vector_get(x,2);

    // Free the gsl objects employed in solving the coefficient system
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A);
    gsl_permutation_free(p);

    /*
     * Analytic 22 phase offsets.  These are only used by the LAL-style
     * analytic phase cross-check path; the main project path still obtains
     * phi22 by splining/integrating omega22.  The offsets make the analytic
     * inspiral, merger, and ringdown antiderivatives continuous.
     */
    IMRPT->phOffInsp = 0.0;
    IMRPT->phOffMerger = 0.0;
    IMRPT->phOffRD = 0.0;
    {
        double thetabarCut = pow(-eta*IMRPT->tCut22, -1.0/8.0);
        double phMECOinsp = IMRPhenomTInspiralPhaseAnsatz22(IMRPT->tCut22, thetabarCut, eta, IMRPT);
        double phMECOmerger = IMRPhenomTMergerPhaseAnsatz22(IMRPT->tCut22, IMRPT);

        IMRPT->phOffMerger = phMECOinsp - phMECOmerger;
        IMRPT->phOffRD = IMRPhenomTMergerPhaseAnsatz22(0.0, IMRPT);
    }
    
    // Amplitude Fit
    // uses a differtn tCut than the frequency fit
    tCut  = tCUT_Amp;  // tCUT_Amp = -150
    double *ampInspCP; // Inspiral collocation point values.
    double ampMergerCP1, ampPeak;    // Merger collocation point value and peak amplitude
    double ampRDC3, coshc3, tanhc3; // Ringdown ansatz c3 free coefficient, damping frequencies and needed quantities to compute ringdown ansatz coefficients.
    /* Ringdown ansatz coefficients c_{1,2,3,4} as defined in eq. [6-8] of Damour&Nagar 2014 (https://arxiv.org/pdf/1406.0401.pdf), also explained in eq.26 of THM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf)
     and alpha_1, alpha_2, alpha_21 as defined in Sec IV of the same reference are stored in the amplitude struct since they are directly passed to the ringdown ansatz.
     tshift, also passed to the amplitude struct, corresponds to the peak amplitude time of the (l,m) mode (0 for the 22 by construction). */

    /* The PN amplitude coefficients of the inspiral ansatz are defined in Appendix A of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf).
       They are constructed from:
       - 3PN expressions from Blanchet et al. 2008 (Class.Quant.Grav.25:165003,2008, https://arxiv.org/abs/0802.1249)
       - 2PN spin corrections from Buonanno et al. 2013 (Phys. Rev D87, 044009, 2013, https://arxiv.org/abs/1209.6349)
       - 1.5PN contributions from Arun et al. 2008 (Phys.Rev.D79:104023,2009; Erratum-ibid.D84:049901,2011, https://arxiv.org/abs/0810.5336) */
    
    ampInspCP = double_vector(3);
    
    /* Needed fits for collocation points and ringdown coefficient */
    ampInspCP[0] = IMRPhenomT_Inspiral_Amp_CP1_22(etapow, Spow, dchi, delta);
    ampInspCP[1] = IMRPhenomT_Inspiral_Amp_CP2_22(etapow, Spow, dchi, delta);
    ampInspCP[2] = IMRPhenomT_Inspiral_Amp_CP3_22(etapow, Spow, dchi, delta);
    ampMergerCP1 = IMRPhenomT_Merger_Amp_CP1_22(etapow, Spow, dchi, delta);
    ampPeak = IMRPhenomT_PeakAmp_22(etapow, Spow, dchi, delta);
    ampRDC3 = IMRPhenomT_RD_Amp_C3_22(etapow, Spow);

    
    IMRPT->tshift = 0.0; //Peak time, by construction 0 for l=2, m=2
    
    /***********************************************************/
    /************** RINGDOWN ANSATZ COEFFICIENTS ***************/
    /***********************************************************/

    /* Ringdown ansatz coefficients as defined in in eq. [6-8] of Damour&Nagar 2014 (https://arxiv.org/pdf/1406.0401.pdf).
    See also eq.26c-e of THM paper: https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf
    Essentially, c3 is the only calibrated coefficient. c2 accounts for the effect of the first overtone in the early ringdown, and c1 and c4 fix the peak amplitude and null derivative at peak */
    
    IMRPT->alpha1RD = 2.*M_PI*fDAMP;
    IMRPT->alpha2RD = 2.*M_PI*fDAMPn2;
    IMRPT->alpha21RD = 0.5*(IMRPT->alpha2RD - IMRPT->alpha1RD); //Coefficient c_2 of ringdown amplitude ansatz as defined in equation 7 of Damour&Nagar 2014 (https://arxiv.org/pdf/1406.0401.pdf) and eq.26d of https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf

    IMRPT->c3 = ampRDC3;
    IMRPT->c2 = 0.5*(IMRPT->alpha2RD - IMRPT->alpha1RD);

    phi = gsl_complex_rect(IMRPT->c3,0); // Needed complex parameter for gsl hyperbolic functions
    gsl_complex coshphi = gsl_complex_cosh(phi);
    coshc3 =  GSL_REAL(coshphi);
    gsl_complex tanhphi = gsl_complex_tanh(phi);
    tanhc3 =  GSL_REAL(tanhphi);

    /* This condition ensures that the second derivative of the amplitude at the mode peak is always zero or negative, not producing then a second peak in the ringdown */
    if(fabs(IMRPT->c2) > fabs(0.5*IMRPT->alpha1RD/tanhc3))
    {
            IMRPT->c2 = -0.5*IMRPT->alpha1RD/tanhc3;
    }



    IMRPT->c1 = ampPeak*IMRPT->alpha1RD*coshc3*coshc3/IMRPT->c2;
    IMRPT->c4 = ampPeak - IMRPT->c1*tanhc3;

    /***********************************************************/
    /************** INSPIRAL COEFFICIENTS SOLUTION *************/
    /***********************************************************/

    /* In order to obtain the value of the 3 unknown extra coefficients of the inspiral amplitude ansatz, we need to solve a linear system
  where the ansatz is imposed to match collocation point values at the corresponding collocation point times.
  See equation 15 of THM paper: https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf */

    /* Initialise the extra coefficients to zero, so calls to the amplitude ansatz function returns pure PN */
    /* The quantities inspC1 etc are now in the ampR, ampI arrays.  inspC1 [8], inspC2  [9], inspC3  [10] */
    


    double tinsppoints[3]     = {-2000., -250., -150.0}; // Collocation point times as defined in Eq. 16 of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf)

    /* We allocate a rank three linear system to solve the coefficients */
    p = gsl_permutation_alloc(3);
    b = gsl_vector_alloc(3);
    x = gsl_vector_alloc(3);
    A = gsl_matrix_alloc(3,3);

    double omega, xx, x4, x4half, x5; // Needed powers of PN parameter x=v^2=(\omega_orb)^(2/3)=(0.5\omega_22)^(2/3)
    double ampoffset; // Known PN part of the amplitude
    double bi; // CP value - known PN amplitude vector

    /* In this loop over collocation points, the components of the solution vector b and the basis matrix A are established.
    See equation 15 of THM paper: https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf */
    for (idx=0; idx<3; idx++)
    {
            omega = IMRPhenomTomega22(tinsppoints[idx], eta, IMRPT); // Twice the orbital frequency at the collocation point time
            xx = pow(0.5*omega,2./3.); // PN expansion parameter of the amplitude
            x4 = xx*xx*xx*xx; // Needed powers
            x4half = x4*sqrt(xx);
            x5 = x4*xx;

            ampoffset = creal(IMRPhenomTInspiralAmpAnsatzHM(xx, IMRPT)); // Real part of the known PN contribution
            bi = (1./IMRPT->A0/xx)*(ampInspCP[idx] - ampoffset); // Solution vector: collocation point value minus the know PN part of the ansatz, factored by the amplitude factor to not include it in each basis function (the powers of x)

            gsl_vector_set(b,idx,bi); // Set b vector
            
            /*Set basis matrix elements, Basis functions are the higher order powers of x that we add to the PN ansatz */
            gsl_matrix_set(A,idx,0,x4);
            gsl_matrix_set(A,idx,1,x4half);
            gsl_matrix_set(A,idx,2,x5);
    }

    /* We now solve the system A x = b via an LU decomposition */
    gsl_linalg_LU_decomp(A,p,&s);
    gsl_linalg_LU_solve(A,p,b,x);

    /* Set the extra pseudo-PN coefficients with the solutions of the system */
    IMRPT->ampR[8] = gsl_vector_get(x,0);
    IMRPT->ampR[9] = gsl_vector_get(x,1);
    IMRPT->ampR[10] = gsl_vector_get(x,2);

    /* Free the gsl solver */
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A);
    gsl_permutation_free(p);

    /***********************************************************/
    /************** MERGER COEFFICIENTS SOLUTION ***************/
    /***********************************************************/


    /* We need to solve for 4 unknown coefficients of the ansatz defined in IMRPhenomTMergerAmpAnsatzHM.
       Three of them are obtained by imposing continuity and differentiability at the boundary with the inspiral
       region (i.e in t = tCUT_Amp) and continuity with the ringdown region (i.e imposing amp(t_peak)=AmpPeak).
       Differentiability at the ringdown boundary is satisfied by default since both the merger and the ringdown
       ansatz describe a peak (derivative zero) at t_peak.
       4th coefficient is obtained by imposing the ansatz to match a collocation point at t=-25M.
       See equation 31 of THM paper: https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf */

    /* Reallocate gsl linear system, this time rank 4*/
    p = gsl_permutation_alloc(4);
    b = gsl_vector_alloc(4);
    x = gsl_vector_alloc(4);
    A = gsl_matrix_alloc(4,4);

    /* Set A_{0,i} and b_{0}: here we impose continuity with the inspiral region, essentially equating the value of the merger ansatz at the inspiral boundary time tCut
    with the value of the inspiral amplitude at that time. */

    xx = pow(0.5*IMRPhenomTomega22(tCut, eta, IMRPT),2./3.); // PN expansion parameter at tCut
    double ampinsp = copysign(1.0,creal(IMRPhenomTInspiralAmpAnsatzHM(xx, IMRPT)))*cabs(IMRPhenomTInspiralAmpAnsatzHM(xx, IMRPT)); // Value of the absolute inspiral amplitude, carrying the sign
    gsl_vector_set(b,0,ampinsp); // Set solution vector: Continuity with the inspiral region

    /* Here we compute the needed hyperbolic secant functions for the merger ansatz basis matrix */
    /* Time parameterisation of merger ansatz is in tau=t-tshift, so peak occurs at tau=0 */
    phi = gsl_complex_rect(IMRPT->alpha1RD*(tCut-IMRPT->tshift),0);
    gsl_complex sechphi = gsl_complex_sech(phi);
    double sech1 = GSL_REAL(sechphi);
    phi = gsl_complex_rect(2.*IMRPT->alpha1RD*(tCut-IMRPT->tshift),0);
    sechphi = gsl_complex_sech(phi);
    double sech2 = GSL_REAL(sechphi);

    /* Set the first row of the basis matrix. Just the functions that multiply each unknown coefficient of the ansatz */
    gsl_matrix_set(A,0,0,1.0);
    gsl_matrix_set(A,0,1,sech1);
    gsl_matrix_set(A,0,2,pow(sech2,1./7.));
    gsl_matrix_set(A,0,3,(tCut-IMRPT->tshift)*(tCut-IMRPT->tshift));


    /* Set A_{1,i} and b_{1}: here we impose a collocation point value, essentially equating the value of the merger ansatz at the collocation point time
    with the value of the collocation point. */

    gsl_vector_set(b,1,ampMergerCP1); // Imposing collocation point value

    /* Here we compute the needed hyperbolic secant functions for the merger ansatz basis matrix */
    phi = gsl_complex_rect(IMRPT->alpha1RD*(tcpMerger-IMRPT->tshift),0);
    sechphi = gsl_complex_sech(phi);
    sech1 = GSL_REAL(sechphi);
    phi = gsl_complex_rect(2.*IMRPT->alpha1RD*(tcpMerger-IMRPT->tshift),0);
    sechphi = gsl_complex_sech(phi);
    sech2 = GSL_REAL(sechphi);

    /* Set the second row of the basis matrix. Just the functions that multiply each unknown coefficient of the ansatz */
    gsl_matrix_set(A,1,0,1.0);
    gsl_matrix_set(A,1,1,sech1);
    gsl_matrix_set(A,1,2,pow(sech2,1./7.));
    gsl_matrix_set(A,1,3,(tcpMerger-IMRPT->tshift)*(tcpMerger-IMRPT->tshift));

    /* Set A_{2,i} and b_{2}: here we impose the peak amplitude value, essentially equating the value of the merger ansatz at the peak time tshift with
    with the value of the peak amplitude. */

    gsl_vector_set(b,2,ampPeak); // Imposing peak amplitude, that guarantees continuity with ringdown region.

    /* Set the second row of the basis matrix. Just the functions that multiply each unknown coefficient of the ansatz, once evaluated in the peak time */
    gsl_matrix_set(A,2,0,1.0);
    gsl_matrix_set(A,2,1,1.0); // sech(tau=0)=1
    gsl_matrix_set(A,2,2,1.0); // sech(tau=0)=1
    gsl_matrix_set(A,2,3,0.0); // tau*tau = 0 in tau=0

    /* Set A_{3,i} and b_{3}: here we impose the differentiability at the inspiral merger boundary, essentially by equating the value of the merger ansatz derivative at the
    boundary time tCut with the value of the inspiral amplitude derivative at that time. */

    /* First we compute the numerical derivatives with inspiral region for imposing differentiability at boundary.
    For this, first we compute the values of theta at two differentially close points in the boundary, from that we compute twice the orbital frequency at those points
    and then the value of the PN expansion parameter x at those points. Derivative is the amplitude difference between these two points weighted by the differential step.
    We carry the sign of the inspiral amplitude derivative. */

    double omega2 = IMRPhenomTomega22(tCut, eta, IMRPT);
    double omega1 = IMRPhenomTomega22(tCut-0.000001, eta, IMRPT);
    double x1 = pow(0.5*omega1,2./3.);
    double x2 = pow(0.5*omega2,2./3.);
    double dampMECO = copysign(1.0,creal(IMRPhenomTInspiralAmpAnsatzHM(x2, IMRPT)))*(cabs(IMRPhenomTInspiralAmpAnsatzHM(x2, IMRPT)) - cabs(IMRPhenomTInspiralAmpAnsatzHM(x1, IMRPT)))/0.000001; // Value of inspiral derivative at boundary time.

    gsl_vector_set(b,3,dampMECO); // We set this value to the solution vector

    /* Here we compute the needed hyperbolic  functions for the merger ansatz derivative basis matrix */
    phi = gsl_complex_rect(IMRPT->alpha1RD*(tCut-IMRPT->tshift),0);
    sechphi = gsl_complex_sech(phi);
    sech1 = GSL_REAL(sechphi);
    gsl_complex phi2 = gsl_complex_rect(2.*IMRPT->alpha1RD*(tCut-IMRPT->tshift),0);
    sechphi = gsl_complex_sech(phi2);
    sech2 = GSL_REAL(sechphi);
    tanhphi = gsl_complex_tanh(phi);
    double tanh = GSL_REAL(tanhphi);
    gsl_complex sinhphi = gsl_complex_sinh(phi2);
    double sinh = GSL_REAL(sinhphi);

    /* Basis functions of the analytical time derivative of the merger ansatz */
    double aux1 = -IMRPT->alpha1RD*sech1*tanh;
    double aux2 = (-2./7.)*IMRPT->alpha1RD*sinh*pow(sech2,8./7.);
    double aux3 = 2.*(tCut-IMRPT->tshift);

    /*We set the value of the basis matrix with the previous elements */
    gsl_matrix_set(A,3,0,0.0);
    gsl_matrix_set(A,3,1,aux1);
    gsl_matrix_set(A,3,2,aux2);
    gsl_matrix_set(A,3,3,aux3);

    /* Once we have set up the system, we now solve the system A x = b via an LU decomposition */
    gsl_linalg_LU_decomp(A,p,&s);
    gsl_linalg_LU_solve(A,p,b,x);

    /* Initialize the unkown merger ansatz coefficients with the solution of the linear system */
    IMRPT->mergerC1 = gsl_vector_get(x,0);
    IMRPT->mergerC2 = gsl_vector_get(x,1);
    IMRPT->mergerC3 = gsl_vector_get(x,2);
    IMRPT->mergerC4 = gsl_vector_get(x,3);

    /* Deallocate the gsl linear system objects */
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_matrix_free(A);
    gsl_permutation_free(p);


    /************************************************************/
    /*** COMPLEX AMPLITUDE PHASING AT INSPIRAL MERGER BOUNDARY **/
    /************************************************************/

    /* Inspiral amplitude is a complex quantity, and then its argument contributes to the phase and frequency
    of the modes. In order to not have a phase/frequency discontinuity between inspiral and merger regions,
    we need to store the value of this phase contribution at the boundary, so it can be added
    to the merger phase/frequency ansatz later. See discussion on Sec. IID of THM paper: https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf*/


    IMRPT->omegaCutPNAMP = -(ComplexAmpOrientation(x2, IMRPT) - ComplexAmpOrientation(x1, IMRPT))/0.000001; // Derivative of the phase contribution at boundary. Needed for frequency continuity
    IMRPT->phiCutPNAMP = atan2(cimag(IMRPhenomTInspiralAmpAnsatzHM(x2, IMRPT)),creal(IMRPhenomTInspiralAmpAnsatzHM(x2, IMRPT))); // Phase contribution at boundary. Needed for phase continuity

    /* We need to compute an extra Pi factor for adding to the merger phase if the inspiral amplitude real part is negative */
    if(copysign(1.0,creal(IMRPhenomTInspiralAmpAnsatzHM(x2, IMRPT)))==-1.0)
    {
            IMRPT->phiCutPNAMP += M_PI;
    }

    
    free(etapow);
    free(Spow);

    /* The aligned model is the special case afinal_prec=afinal. */
    IMRPhenomTSetPrecessingFinalSpin(IMRPT, Mfinal, afinal);

}

void MergerRingdownFit(double *params, struct IMRPhenomT *IMRPT)
{
    MergerRingdownFitInternal(params, 0, 0.0, IMRPT);
}

void MergerRingdownFitWithFinalSpin(double *params,
                                    double final_dimensionless_spin,
                                    struct IMRPhenomT *IMRPT)
{
    MergerRingdownFitInternal(params, 1, final_dimensionless_spin, IMRPT);
}

void IMRPhenomTSetPrecessingFinalSpin(struct IMRPhenomT *IMRPT,
                                      double final_mass_fraction,
                                      double final_dimensionless_spin)
{
    double fDAMP;
    double fDAMPn2;
    double amp_peak;
    double coshc3;
    double tanhc3;

    if(IMRPT == NULL || final_mass_fraction <= 0.0 ||
       !isfinite(final_dimensionless_spin) ||
       fabs(final_dimensionless_spin) >= 1.0)
        return;

    IMRPT->omegaRING_prec =
        2.0*M_PI*evaluate_QNMfit_fring22(final_dimensionless_spin)/
        final_mass_fraction;
    fDAMP = evaluate_QNMfit_fdamp22(final_dimensionless_spin)/
            final_mass_fraction;
    fDAMPn2 = evaluate_QNMfit_fdamp22n2(final_dimensionless_spin)/
              final_mass_fraction;
    IMRPT->alpha1RD_prec = 2.0*M_PI*fDAMP;
    IMRPT->alpha2RD_prec = 2.0*M_PI*fDAMPn2;
    IMRPT->alpha21RD_prec =
        0.5*(IMRPT->alpha2RD_prec-IMRPT->alpha1RD_prec);

    IMRPT->phase_c1_prec =
        (1.0+IMRPT->carray[3]+IMRPT->carray[4])*
        (IMRPT->omegaRING_prec-IMRPT->omegaPeak)/IMRPT->carray[2]/
        (IMRPT->carray[3]+2.0*IMRPT->carray[4]);

    coshc3 = cosh(IMRPT->c3);
    tanhc3 = tanh(IMRPT->c3);
    IMRPT->amp_c2_prec = IMRPT->alpha21RD_prec;
    if(fabs(IMRPT->amp_c2_prec) >
       fabs(0.5*IMRPT->alpha1RD_prec/tanhc3))
    {
        IMRPT->amp_c2_prec = -0.5*IMRPT->alpha1RD_prec/tanhc3;
    }
    amp_peak = IMRPT->c1*tanhc3+IMRPT->c4;
    IMRPT->amp_c1_prec = amp_peak*IMRPT->alpha1RD_prec*coshc3*coshc3/
                         IMRPT->amp_c2_prec;
    IMRPT->amp_c4_prec = amp_peak-IMRPT->amp_c1_prec*tanhc3;
}


/*************** RINGDOWN AND DAMPING QNM FREQUENCIES ***************/
 
/* FOR N=1, SAME CODE AS PHENOMXAS */
 
/*fRING*/
 
double evaluate_QNMfit_fring22(double finalDimlessSpin){
 
   double return_val;
 
   double x2= finalDimlessSpin*finalDimlessSpin;
   double x3= x2*finalDimlessSpin;
   double x4= x2*x2;
   double x5= x3*x2;
   double x6= x3*x3;
   double x7= x4*x3;
 
   return_val = (0.05947169566573468 - \
   0.14989771215394762*finalDimlessSpin + 0.09535606290986028*x2 + \
   0.02260924869042963*x3 - 0.02501704155363241*x4 - \
   0.005852438240997211*x5 + 0.0027489038393367993*x6 + \
   0.0005821983163192694*x7)/(1 - 2.8570126619966296*finalDimlessSpin + \
   2.373335413978394*x2 - 0.6036964688511505*x4 + \
   0.0873798215084077*x6);
   return return_val;
}
 
/* fDamp */
 
double evaluate_QNMfit_fdamp22(double finalDimlessSpin){
 
   double return_val;

   double x2= finalDimlessSpin*finalDimlessSpin;
   double x3= x2*finalDimlessSpin;
   double x4= x2*x2;
   double x5= x3*x2;
   double x6= x3*x3;
 
   return_val = (0.014158792290965177 - \
   0.036989395871554566*finalDimlessSpin + 0.026822526296575368*x2 + \
   0.0008490933750566702*x3 - 0.004843996907020524*x4 - \
   0.00014745235759327472*x5 + 0.0001504546201236794*x6)/(1 - \
   2.5900842798681376*finalDimlessSpin + 1.8952576220623967*x2 - \
   0.31416610693042507*x4 + 0.009002719412204133*x6);
   return return_val;
}
 
/* For n=2, new fits of damping frequency needed for PhenomT */
 
double evaluate_QNMfit_fdamp22n2(double finalDimlessSpin){
 
   double return_val;
 
   double x = finalDimlessSpin;
   double x2= x*x;
   double x3= x2*x;
   double x4= x2*x2;
   double x5= x3*x2;
   double x6= x3*x3;
   double x7= x4*x3;
   double x8= x4*x4;
 
   return_val = 0.043611742588188715 + (-0.004016191313442792*x - 0.0027646155943395426*x2 + 0.001141927763953028*x3 + 0.007938320030300492*x4 - 0.0008263166671238823*x5 - 0.014025760257115768*x6 + 0.001792158578158245*x7 + 0.008824138122361842*x8)/(2. - 1.9477781396815619*x);
    
   return return_val;
}

double IMRPhenomT_PeakFrequency_22(double *etapow, double *Spow, double dchi, double delta){
 
        double fit;
        double eta = etapow[1];
        double S = Spow[1];
    
 
   fit = 0.27212130745330404 + 0.40972689759932074*eta - 0.0018392172960247433*eta*dchi*dchi + S*(0.09558832959428547 - 0.04834585264918328*eta - 0.15275173823699056*etapow[2]) - 3.4232387074402153*etapow[2] + 32.853772442252605*etapow[3] - 1.4976829186605336*dchi*delta*(1. - 4.775645585721007*eta)*etapow[3] -
   0.9981117852179613*dchi*delta*(1. - 5.260098925354571*eta)*S*etapow[3] - 125.22505746137587*etapow[4] + 179.3797198714914*etapow[5] + (0.054391696704622204 - 0.1482682698299456*eta + 0.08938162810617255*etapow[2])*Spow[2] +
   (-0.020719540055375383 + 0.5090144456500953*eta - 1.5809441589349338*etapow[2])*Spow[3] + (0.024240736699062685 - 0.09490089674418004*eta + 0.09518501714836035*etapow[2])*Spow[4] + (0.09759303647532228 - 1.105520690228567*eta + 2.921271981239294*etapow[2])*Spow[5];
 
   return fit;
}

/* RD 22 Frequency Coefficient fits */
 
double IMRPhenomT_RD_Freq_D2_22(double *etapow, double *Spow, double dchi, double delta){
 
   double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
   fit = 0.1598180460429256 + 0.19120040104567676*eta + (-0.012853620630980167 - 0.006532392920798404*eta)*S - 0.7733759581766899*etapow[2] + 0.18151402648790957*dchi*delta*(1. - 9.041198282315879*eta)*etapow[2] + 0.27147713896183995*dchi*delta*(1. - 5.653323210961101*eta)*S*etapow[2] -
   0.01603489049446065*dchi*dchi*etapow[3] + (-0.046785083372074494 + 0.102759380109996*eta)*Spow[2] + (0.0009883572415502464 - 0.050384608002279486*eta)*Spow[3];
 
   return fit;
}
 
double IMRPhenomT_RD_Freq_D3_22(double *etapow, double *Spow, double dchi, double delta){
 
        double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
   fit = 2.6456463496860927 - 28.079375863863458*eta + 323.1691069138812*etapow[2] - 0.5040057675360762*dchi*delta*(1 + 21.786482297795278*eta)*etapow[2] + 1.561247215701216*dchi*delta*(1. - 1.7508069810164308*eta)*S*etapow[2] + S*(3.091917073632116 - 17.345283345692266*eta + 33.40735388809028*etapow[2]) -
   1490.8128941604907*etapow[3] + 0.1619056474567525*dchi*dchi*etapow[3] + 2376.3257196613886*etapow[4] + (0.734022429223849 - 0.029342234233198747*eta - 9.281610698291932*etapow[2])*Spow[2];
 
        return fit;
}

/******************************* 22 AMPLITUDE FITS ***************************/
 
/* Inspiral 22 amplitude collocation points */

double IMRPhenomT_Inspiral_Amp_CP1_22(double *etapow, double *Spow, double dchi, double delta){
 
        double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
        fit = 0.00006480771730217768*eta*dchi*dchi - 0.3543965558027252*dchi*delta*(1. - 2.463526130684083*eta)*etapow[3] + 0.01879295038873938*dchi*delta*(1. - 5.236796607517272*eta)*S*etapow[3] +
   S*(0.1472653807120573*eta - 1.9636752493349356*etapow[2] + 14.177521724634461*etapow[3] - 48.94620901701877*etapow[4] + 63.83730899015984*etapow[5]) +
   eta*(0.8493442097893826 - 13.211067914003836*eta + 311.99021467938235*etapow[2] - 4731.025904601601*etapow[3] + 44821.93042533854*etapow[4] - 264474.1374080295*etapow[5] + 943246.2317701122*etapow[6] - 1.8588135904328802e6*etapow[7] + 1.5524778581809246e6*etapow[8]) +
   (0.04902976057622393*eta - 1.0152511131279736*etapow[2] + 8.286289152216145*etapow[3] - 30.19775956110767*etapow[4] + 40.670065442751955*etapow[5])*Spow[2] +
   (0.04780630695082567*eta - 1.2177827888317065*etapow[2] + 11.505675146308567*etapow[3] - 46.733420749352135*etapow[4] + 68.40821782168776*etapow[5])*Spow[3];
 
        return fit;
}
 
double IMRPhenomT_Inspiral_Amp_CP2_22(double *etapow, double *Spow, double dchi, double delta){
 
        double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
        fit = 0.000100027278976821*eta*dchi*dchi - 0.7578403155712378*dchi*delta*(1. - 2.056456271350877*eta)*etapow[3] - 0.14126282637778914*dchi*delta*(1. - 2.5840771007494916*eta)*S*etapow[3] + S*(0.2331970217833686*eta - 1.5473968380422929*etapow[2] + 5.973401506474942*etapow[3] - 9.110484789161045*etapow[4]) +
   eta*(0.9904613241626621 - 6.708006572605403*eta + 127.40270095439482*etapow[2] - 1723.355339710798*etapow[3] + 15430.10086310527*etapow[4] - 88744.26044058547*etapow[5] + 313650.01696201024*etapow[6] - 617887.8122937253*etapow[7] + 518220.9267888211*etapow[8]) +
   (0.08934817374146888*eta - 0.8887847358339216*etapow[2] + 3.7233864099350784*etapow[3] - 5.814765403882651*etapow[4])*Spow[2] + (0.04471990627820145*eta - 0.642458648615624*etapow[2] + 3.393481171493086*etapow[3] - 6.092083983738554*etapow[4])*Spow[3];
 
        return fit;
}
 
double IMRPhenomT_Inspiral_Amp_CP3_22(double *etapow, double *Spow, double dchi, double delta){
 
        double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
        fit = 0.0002459376633671657*eta*dchi*dchi - 0.8794763631110696*dchi*delta*(1. - 2.0751630535350096*eta)*etapow[3] - 0.3319387797134261*dchi*delta*(1. - 3.1838055629892184*eta)*S*etapow[3] + S*(0.23505507416274007*eta - 1.2449030421324767*etapow[2] + 4.315803728759738*etapow[3] - 6.384257606413192*etapow[4]) +
   eta*(1.0208762064809185 - 3.3799457394243957*eta + 16.242639717123314*etapow[2] + 299.2297416582362*etapow[3] - 5913.920743907752*etapow[4] + 46388.231537995445*etapow[5] - 192261.0498470111*etapow[6] + 413750.14250475995*etapow[7] - 364403.84935539874*etapow[8]) +
   (0.09630827896641526*eta - 0.7915321134872877*etapow[2] + 2.86907420250287*etapow[3] - 4.038995403653199*etapow[4])*Spow[2] + (0.07395420485618898*eta - 1.0289224187583748*etapow[2] + 5.275845823734598*etapow[3] - 9.206158044409037*etapow[4])*Spow[3];
 
        return fit;
}
 
/* Merger 22 amplitude collocation points */
 
double IMRPhenomT_Merger_Amp_CP1_22(double *etapow, double *Spow, double dchi, double delta){
 
        double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
        fit = 0.0004059354652663733*eta*dchi*dchi - 0.9382383412276684*dchi*delta*(1. - 2.509151362054917*eta)*etapow[3] - 0.6560748977864668*dchi*delta*(1. - 3.426294113321932*eta)*S*etapow[3] + S*(0.23465398091766254*eta - 1.3398914201113978*etapow[2] + 5.9073801933446495*etapow[3] - 10.84221896204708*etapow[4]) +
   eta*(1.2946032382158479 - 3.3343035556341816*eta + 91.6430240976277*etapow[2] - 1687.6195123629968*etapow[3] + 19726.50907350641*etapow[4] - 140798.18973779568*etapow[5] + 594095.3303894227*etapow[6] - 1.358657562562124e6*etapow[7] + 1.2958912179017465e6*etapow[8]) +
   (0.03174875260265387*eta + 0.23082150180902375*etapow[2] - 1.9901867982613048*etapow[3] + 4.009389679757772*etapow[4])*Spow[2] + (-0.04033221614773138*eta + 0.8426888041517518*etapow[2] - 4.742283264846479*etapow[3] + 9.059923021547936*etapow[4])*Spow[3];
 
        return fit;
}
 
double IMRPhenomT_PeakAmp_22(double *etapow, double *Spow, double dchi, double delta){
 
        double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
        fit = 0.0017885007700308166*eta*dchi*dchi - 0.5846280668038513*dchi*delta*(1. - 4.879882766464646*eta)*etapow[3] - 0.874161608112943*dchi*delta*(1. - 1.690095043235707*eta)*S*etapow[3] + S*(0.203557188205307*eta - 2.4368458739010563*etapow[2] + 12.206344183078137*etapow[3] - 23.417979354674692*etapow[4]) +
   eta*(1.4701266133411792 - 1.387711607537906*eta + 25.641251409467607*etapow[2] - 186.013359336165*etapow[3] + 801.3039484150348*etapow[4] - 1893.8181854645718*etapow[5] + 1946.531703997353*etapow[6]) +
   (-0.0018659293826992745*eta - 0.1888206507658455*etapow[2] + 1.4677324802664107*etapow[3] - 1.4019283350536489*etapow[4])*Spow[2] + (-0.14699838946027494*eta + 2.6186847787143837*etapow[2] - 15.574381075605208*etapow[3] + 31.239292792717016*etapow[4])*Spow[3];
 
        return fit;
}
 
/*RD 22 Amplitude Coefficient Fits */
 
double IMRPhenomT_RD_Amp_C3_22(double *etapow, double *Spow){
 
        double fit;
    double eta = etapow[1];
    double S = Spow[1];
 
        fit = -0.48053994718185694 + 0.7023672141561462*eta + S*(-0.3597773028596323 + 1.4330280386796503*eta - 3.239121799338561*etapow[2]) - 0.1993836305574211*etapow[2] + (-0.2651107472061685 + 1.6433443489711386*eta - 2.757772023954491*etapow[2])*Spow[2] +
   (-0.01973537883495192 - 0.2410762147438714*eta + 2.7315015976869756*etapow[2])*Spow[3];
 
        return fit;
}

/* Wrapper to compute directly the phase contribution of the complex inspiral amplitude at a specific value of the PN expansion parameter x.
See equation 18 of PhenomTHM paper: https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf */
double ComplexAmpOrientation(double x, struct IMRPhenomT *IMRPT)
{
    int i;
    
    double *xarray;
    double xhalf = sqrt(x);
    double ampreal, ampimag;
    
    xarray = double_vector(11);

    xarray[0] = 1.0;
    for(i=1; i< 11; i++) xarray[i] = xarray[i-1]*xhalf;

    ampreal = 0.0;
    ampimag = 0.0;
    for(i=0; i< 11; i++)
    {
        ampreal += IMRPT->ampR[i]*xarray[i];
        ampimag += IMRPT->ampI[i]*xarray[i];
    }

    // log term
    ampreal -= 428./105.*log(16.*x)*xarray[6];
    
    free(xarray);
    
    return atan2(ampimag,ampreal);
    
}


double IMRPhenomTInspiralPhaseAnsatz22(double t, double thetabar, double eta, struct IMRPhenomT *IMRPT)
{
    double pow5_1_8 = pow(5.0, 0.125);
    double pow5_1_4 = pow(5.0, 0.25);
    double pow5_3_8 = pow(5.0, 0.375);
    double pow5_1_2 = pow(5.0, 0.5);
    double pow5_5_8 = pow(5.0, 0.625);
    double pow5_3_4 = pow(5.0, 0.75);
    double pow5_7_8 = pow(5.0, 0.875);
    double theta2 = thetabar*thetabar;
    double theta3 = theta2*thetabar;
    double theta4 = theta2*theta2;
    double theta5 = theta4*thetabar;
    double theta6 = theta3*theta3;
    double theta7 = theta6*thetabar;
    double bracket;
    double aux;

    bracket = 3.0*(-107.0 + 280.0*IMRPT->omega[6])*pow5_3_4 +
              /*
               * The fresh LAL source has log(theta) here.  This stripped
               * IMRPhenomT.c evaluates the 22 frequency with log(2*theta), so
               * the analytic antiderivative has to use the same convention for
               * a meaningful local cross-check.
               */
              321.0*log(2.0*thetabar*pow5_1_8)*pow5_3_4 +
              420.0*IMRPT->omega[7]*thetabar*pow5_7_8 +
              56.0*(25.0*IMRPT->omega[8] + 3.0*eta*t)*theta2 +
              1050.0*IMRPT->omega[9]*pow5_1_8*theta3 +
              280.0*(3.0*IMRPT->omega[10] + eta*IMRPT->omega[2]*t)*pow5_1_4*theta4 +
              140.0*(5.0*IMRPT->omega[11] + 3.0*eta*IMRPT->omega[3]*t)*pow5_3_8*theta5 +
              120.0*(5.0*IMRPT->omega[12] + 7.0*eta*IMRPT->omega[4]*t)*pow5_1_2*theta6 +
              525.0*IMRPT->omega[13]*pow5_5_8*theta7 +
              105.0*eta*IMRPT->omega[5]*t*log(-t)*pow5_5_8*theta7;

    aux = -(pow(5.0, -0.625)/(eta*eta))*(1.0/t)*pow(thetabar, -7.0)*bracket/84.0;

    return aux + IMRPT->phOffInsp;
}

double IMRPhenomTMergerPhaseAnsatz22(double t, struct IMRPhenomT *IMRPT)
{
    double x = asinh(IMRPT->alpha1RD*t);
    double x2 = x*x;
    double x3 = x2*x;
    double x4 = x2*x2;
    double alpha1RD = IMRPT->alpha1RD;
    double omegaPeak = IMRPT->omegaPeak;
    double domegaPeak = IMRPT->domegaPeak;
    double omegaRING = IMRPT->omegaRING;
    double cc = IMRPT->CMarray[1];
    double dd = IMRPT->CMarray[2];
    double ee = IMRPT->CMarray[3];
    double root = sqrt(1.0 + alpha1RD*alpha1RD*t*t);
    double aux;

    aux = omegaRING*t - omegaRING*(
        2.0*cc*t + 24.0*ee*t + 6.0*dd*t*x +
        domegaPeak*t*x/alpha1RD +
        t*(1.0 - omegaPeak/omegaRING) +
        cc*t*x2 + 12.0*ee*t*x2 + dd*t*x3 + ee*t*x4 -
        domegaPeak*root/(alpha1RD*alpha1RD) -
        6.0*dd*root/alpha1RD -
        2.0*cc*x*root/alpha1RD -
        24.0*ee*x*root/alpha1RD -
        3.0*dd*x2*root/alpha1RD -
        4.0*ee*x3*root/alpha1RD);

    return aux + IMRPT->phOffMerger;
}

double IMRPhenomTRDPhaseAnsatz22(double t, struct IMRPhenomT *IMRPT)
{
    double c3 = IMRPT->carray[3];
    double c4 = IMRPT->carray[4];
    double c2 = IMRPT->carray[2];
    double c1 = IMRPT->phase_c1_prec;
    double expC = exp(-c2*t);
    double expC2 = expC*expC;
    double num = 1.0 + c3*expC + c4*expC2;
    double den = 1.0 + c3 + c4;

    return c1*log(num/den) + IMRPT->omegaRING_prec*t + IMRPT->phOffRD;
}


/* 22 ringdown frequency ansatz is the analytical derivate of the expression in IMRPhenomTRDPhaseAnsatz22, described in equation 24 of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf) */
double IMRPhenomTRDOmegaAnsatz22(double t, struct IMRPhenomT *IMRPT){
 
   double c3 = IMRPT->carray[3];
   double c4 = IMRPT->carray[4];
   double c2 = IMRPT->carray[2];
   double c1 = IMRPT->carray[1];
 
   double expC = exp(-1.*c2*t);
   double expC2 = expC*expC;

 
   double num = c1*(-2*c2*c4*expC2 - c2*c3*expC);
   double den = 1. + c4*expC2 + c3*expC;
 
   return (num/den + IMRPT->omegaRING);
}

double IMRPhenomTRDPhaseOmegaAnsatz22(double t,
                                      struct IMRPhenomT *IMRPT)
{
   double c3 = IMRPT->carray[3];
   double c4 = IMRPT->carray[4];
   double c2 = IMRPT->carray[2];
   double c1 = IMRPT->phase_c1_prec;
   double expC = exp(-c2*t);
   double expC2 = expC*expC;
   double num = c1*(-2.0*c2*c4*expC2-c2*c3*expC);
   double den = 1.0+c4*expC2+c3*expC;

   return num/den+IMRPT->omegaRING_prec;
}

/* 22 merger frequency ansatz is a phenomenological expression described in equation [27-28] of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf)
   The ansatz described here corresponds to the rescaled frequency \bar{\omega}=1 - (\omega / \omega_ring) of equation 27. */
double IMRPhenomTMergerOmegaAnsatz22(double t, struct IMRPhenomT *IMRPT){
 
   gsl_complex phi = gsl_complex_rect(IMRPT->alpha1RD*t,0);
   gsl_complex arcsinhphi = gsl_complex_arcsinh(phi);
   double x = GSL_REAL(arcsinhphi);
   double x2 = x*x;
   double x3 = x2*x;
   double x4 = x2*x2;
 
   double out = 1. - IMRPT->omegaPeak/IMRPT->omegaRING + (IMRPT->domegaPeak/IMRPT->alpha1RD)*x + IMRPT->CMarray[1]*x2 + IMRPT->CMarray[2]*x3 + IMRPT->CMarray[3]*x4;
 
   return out;
}
 

  // Final Mass = 1 - Energy Radiated,  X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332
double XLALSimIMRPhenomXFinalMass2017(double eta, double chi1L, double chi2L) {
  
   double delta  = sqrt(1.0 - 4.0*eta);
   double m1     = 0.5 * (1.0 + delta);
   double m2     = 0.5 * (1.0 - delta);
   double eta2   =  eta*eta;
   double eta3   = eta2*eta;
   double eta4   = eta3*eta;
  
     double S    = (m1*m1*chi1L + m2*m2*chi2L)/(m1*m1 + m2*m2);
   double S2     =  S*S;
   double S3     = S2*S;
  
   double dchi   = chi1L - chi2L;
   double dchi2  = dchi*dchi;
  
   double noSpin, eqSpin, uneqSpin;
  
   noSpin = 0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta3 + 3.145145224278187*eta4;
  
   /* Because of the way this is written, we need to subtract the noSpin term */
   eqSpin = ((0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta3 + 3.145145224278187*eta4)*
     (    1.
       + (-0.13084389181783257 - 1.1387311580238488*eta + 5.49074464410971*eta2)*S
       + (-0.17762802148331427 + 2.176667900182948*eta2)*S2
       + (-0.6320191645391563 + 4.952698546796005*eta - 10.023747993978121*eta2)*S3))
       / (1 + (-0.9919475346968611 + 0.367620218664352*eta + 4.274567337924067*eta2)*S);
  
   eqSpin = eqSpin - noSpin;
  
   uneqSpin =  - 0.09803730445895877*dchi*delta*(1. - 3.2283713377939134*eta)*eta2
               + 0.01118530335431078*dchi2*eta3
               - 0.01978238971523653*dchi*delta*(1. - 4.91667749015812*eta)*eta*S;
  
   /* Mfinal = 1 - Erad, assuming that M = m1 + m2 = 1 */
   return (1.0 - (noSpin + eqSpin + uneqSpin));
  
  
 }
  

// Final Dimensionless Spin,  X Jimenez-Forteza et al, PRD, 95, 064024, (2017), arXiv:1611.00332
double XLALSimIMRPhenomXFinalSpin2017(double eta, double chi1L, double chi2L) {
  
         double delta  = sqrt(1.0 - 4.0*eta);
         double m1     = 0.5 * (1.0 + delta);
         double m2     = 0.5 * (1.0 - delta);
         double m1Sq   = m1*m1;
         double m2Sq   = m2*m2;
  
   double eta2   = eta*eta;
   double eta3   = eta2*eta;
  
         //double S  = (m1Sq * chi1L + m2Sq * chi2L) / (m1Sq + m2Sq);
   double S  = (m1*m1*chi1L + m2*m2*chi2L)/(m1*m1 + m2*m2);
   double S2 =  S*S;
   double S3 = S2*S;
  
   double dchi  = chi1L - chi2L;
   double dchi2 = dchi*dchi;
  
   double noSpin, eqSpin, uneqSpin;
  
   noSpin = (3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta3)/(1. + 7.2388440419467335*eta);
  
   eqSpin = (m1Sq + m2Sq)*S
   + ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta3)*S
   + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta3)*S2
   + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta3)*S3)
   / (1. + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta3)*S);
  
   uneqSpin = 0.3223660562764661*dchi*delta*(1. + 9.332575956437443*eta)*eta2           /* Linear in spin difference                */
   - 0.059808322561702126*dchi2*eta3                                                   /* Quadratic in spin difference             */
   + 2.3170397514509933*dchi*delta*(1. - 3.2624649875884852*eta)*eta3*S;                 /* Mixed spin difference + total spin term  */
  
  
         return (noSpin + eqSpin + uneqSpin);
 }
  

void InspiralT3(double *params, struct IMRPhenomT *IMRPT)
{
    /*PN coefficients corresponding to the TaylorT3 implementation of the model,
      as defined in Appendix A1 of Estelles et al 2020 (https://arxiv.org/pdf/2004.08302.pdf) */
    
    /* taken from LALSimIMRPhenomTHM_internals.c. Made minimal changes to avoid introducing errors
      even though using delta etc and not defining repeated combinations of terms is not the most efficient */
    
    /* I removed the log(theta) and log(x) terms from the frequency and phase coefficients and handle them directly when
     evaluating the frequency and amplitude */
    
    int idx;
    double m1, m2, M, delta, eta, eta2, eta3;
    double chi1, chi2;
    double chi1sq, chi2sq;

    m1 = params[0];
    m2 = params[1];
    chi1 = params[2];
    chi2 = params[3];
    
    M = m1+m2;
    
    delta = (m1-m2)/M;
    eta = m1*m2/(M*M);
    
    chi1sq = chi1*chi1;
    chi2sq = chi2*chi2;
    
    eta2 = eta*eta;
    eta3 = eta*eta2;
    
    IMRPT->omega[0] = 1.0;
    
    IMRPT->omega[1] = 0.0;
    
    IMRPT->omega[2] = 0.27641369047619047 + (11.*eta)/32.;

    IMRPT->omega[3] = (-19.*(chi1 + chi2)*eta)/80. + (-113*(chi2*(-1. + delta) - chi1*(1. + delta)) - 96.*M_PI)/320.;

    IMRPT->omega[4] = (1855099. + 1714608.*chi2*chi2*(-1. + delta) - 1714608.*chi1*chi1*(1. + delta))/1.4450688e7 + ((56975. + 61236.*chi1*chi1 - 119448.*chi1*chi2 + 61236.*chi2*chi2)*eta)/258048. + (371.*eta*eta)/2048.;

    IMRPT->omega[5] = (-17.*(chi1 + chi2)*eta*eta)/128. + (-146597.*(chi2*(-1. + delta) - chi1*(1. + delta)) - 46374.*M_PI)/129024. + (eta*(-2.*(chi1*(1213. - 63.*delta) + chi2*(1213. + 63.*delta)) + 117.*M_PI))/2304.;

    IMRPT->omega[6] = -2.499258364444952 - (16928263.*chi1*chi1)/1.376256e8 - (16928263.*chi2*chi2)/1.376256e8 - (16928263.*chi1*chi1*delta)/1.376256e8 + (16928263.*chi2*chi2*delta)/1.376256e8 + \
        ((-2318475. + 18767224.*chi1*chi1 - 54663952.*chi1*chi2 + 18767224.*chi2*chi2)*eta*eta)/1.376256e8 + (235925.*eta*eta*eta)/1.769472e6 + (107.*M_EULER)/280. - (6127.*chi1*M_PI)/12800. - \
        (6127.*chi2*M_PI)/12800. - (6127.*chi1*delta*M_PI)/12800. + (6127.*chi2*delta*M_PI)/12800. + (53.*M_PI*M_PI)/200. + \
         (eta*(632550449425. + 35200873512.*chi1*chi1 - 28527282000.*chi1*chi2 + 9605339856.*chi1*chi1*delta - 1512.*chi2*chi2*(-23281001. + 6352738.*delta) + 34172264448.*(chi1 + chi2)*M_PI - \
          22912243200.*M_PI*M_PI))/1.040449536e11;

    IMRPT->omega[7] = (-12029.*(chi1 + chi2)*eta*eta*eta)/92160. + (eta*eta*(507654.*chi1*chi2*chi2 - 838782.*chi2*chi2*chi2 + chi2*(-840149. + 507654.*chi1*chi1 - 870576.*delta) +
    chi1*(-840149. - 838782.*chi1*chi1 + 870576.*delta) + 1701228.*M_PI))/1.548288e7 +
     (eta*(218532006.*chi1*chi2*chi2*(-1. + delta) - 1134.*chi2*chi2*chi2*(-206917. + 71931.*delta) - chi2*(1496368361. - 429508815.*delta + 218532006.*chi1*chi1*(1. + delta)) +
    chi1*(-1496368361. - 429508815.*delta + 1134.*chi1*chi1*(206917. + 71931.*delta)) - 144.*(488825. + 923076.*chi1*chi1 - 1782648.*chi1*chi2 + 923076.*chi2*chi2)*M_PI))/1.8579456e8 +
    (-6579635551.*chi2*(-1. + delta) + 535759434.*chi2*chi2*chi2*(-1. + delta) - chi1*(-6579635551. + 535759434.*chi1*chi1)*(1. + delta) + (-565550067. - 465230304.*chi2*chi2*(-1. + delta) + 465230304.*chi1*chi1*(1. + delta))*M_PI)/1.30056192e9;
    
    IMRPT->A0 = 2.*eta*sqrt(16.*M_PI/5.); // Precomputed amplitude global factor (see eq. 12-14 of PhenomTHM paper https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf)
    
    IMRPT->ampR[0] = 1.0;
    IMRPT->ampI[0] = 0.0;
    
    IMRPT->ampR[1] = 0.0;
    IMRPT->ampI[1] = 0.0;

    IMRPT->ampR[2] = -2.5476190476190474 + (55.*eta)/42.;
    IMRPT->ampI[2] = 0.0;

    IMRPT->ampR[3] = (-2.*chi1)/3. - (2.*chi2)/3. - (2.*chi1*delta)/(3.*((1 - delta)/2. + (1. + delta)/2.)) + (2.*chi2*delta)/(3.*((1 - delta)/2. + (1. + delta)/2.)) + (2.*chi1*eta)/3. + (2.*chi2*eta)/3. + 2.*M_PI;
    IMRPT->ampI[3] = 0.0;

    IMRPT->ampR[4] = -1.437169312169312 + chi1sq/2. + chi2sq/2. + (chi1sq*delta)/2. - (chi2sq*delta)/2. - (1069.*eta)/216. - chi1sq*eta + 2.*chi1*chi2*eta - chi2sq*eta + (2047.*eta2)/1512.;
    IMRPT->ampI[4] = 0.0;

     IMRPT->ampR[5] = - (107.*M_PI)/21. + (34.*eta*M_PI)/21.;
     IMRPT->ampI[5]  =  -24.*eta;

    IMRPT->ampR[6] = 41.78634662956092 - (278185.*eta)/33264. - (20261.*eta2)/2772. + (114635.*eta3)/99792. - (856.*M_EULER)/105. + (2.*M_PI*M_PI)/3. + (41.*eta*M_PI*M_PI)/96.;
    IMRPT->ampI[6]= (428./105.)*M_PI;

    IMRPT->ampR[7] = (-2173.*M_PI)/756. - (2495.*eta*M_PI)/378. + (40.*eta2*M_PI)/27.;
    IMRPT->ampI[7] = (14333.*eta)/162. - (4066.*eta2)/945.;
    
    /* Initialise the extra coefficients to zero, so calls to the amplitude ansatz function returns pure PN when the merger
     ringdown fit is not called */
    for (idx=8; idx<11; idx++)
    {
        IMRPT->ampR[idx] = 0.0;
        IMRPT->ampI[idx] = 0.0;
    }
    
}

// similar to InspiralOmegaT3 but with M=1 and taking a given theta
double IMRPhenomTTaylorT3(double theta, struct IMRPhenomT *IMRPT)
{
    int i;
    double omt;
    double *tharray;
    
    tharray = double_vector(8);
    
    tharray[0] = 1.0;
    for(i=1; i< 8; i++) tharray[i] = tharray[i-1]*theta;
    
    omt = 0.0;
    for(i=0; i< 8; i++) omt += IMRPT->omega[i]*tharray[i];
    
    // log term(s)
    omt += 107./280.*log(2.*theta)*tharray[6];
    
    // overall PN scaling
    omt *= tharray[3]/4.0;
    
    free(tharray);
    
    return omt;
 
}


double InspiralOmegaT3(double u, double fac1, double fac2, struct IMRPhenomT *IMRPT)
{
    // fac1 = eta/(5 M)
    // fac2 = 1/(4 M)
    // u = t0-t
    int i;
    double omt, theta;
    double *tharray;
    
    tharray = double_vector(8);

    theta = pow(u*fac1, -0.125);
    
    tharray[0] = 1.0;
    for(i=1; i< 8; i++) tharray[i] = tharray[i-1]*theta;
    
    omt = 0.0;
    for(i=0; i< 8; i++) omt += IMRPT->omega[i]*tharray[i];
    
    // log term(s)
    omt += 107./280.*log(2.*theta)*tharray[6];
    
    // overall PN scaling
    omt *= tharray[3]*fac2;
    
    free(tharray);
    
    return omt;
    
}

double InspiralOmega(double u, double fac1, double fac2, struct IMRPhenomT *IMRPT)
{
    // fac1 = eta/(5 M)
    // fac2 = 1/(4 M)
    // u = t0-t
    int i;
    double omt, theta;
    double *tharray;
    
    tharray = double_vector(14);

    theta = pow(u*fac1, -0.125);
    
    tharray[0] = 1.0;
    for(i=1; i< 14; i++) tharray[i] = tharray[i-1]*theta;
    
    omt = 0.0;
    for(i=0; i< 14; i++) omt += IMRPT->omega[i]*tharray[i];
    
    // log term(s)
    omt += 107./280.*log(2.*theta)*tharray[6];
    
    // overall PN scaling
    omt *= tharray[3]*fac2;
    
    free(tharray);
    
    return omt;
    
}

// takes the previously computed array of x values and returns the real and imaginary parts of H22
void InspiralAmpT3(int NS, double *xarray, struct IMRPhenomT *IMRPT, double *AR, double *AI)
{
    int i, j;
    double x, sqrtx;
    double *xpow;
    
    xpow = double_vector(8);
    
    for(i=0; i<NS; i++)
    {
        x = xarray[i];
        sqrtx = sqrt(x);
        xpow[0] = 1.0;
        for(j=1; j<8; j++) xpow[j] = xpow[j-1]*sqrtx;
        
        AR[i] = 0.0;
        AI[i] = 0.0;
        for(j=0; j<8; j++)
        {
            AR[i] += IMRPT->ampR[j]*xpow[j];
            AI[i] += IMRPT->ampI[j]*xpow[j];
        }
        
        // log term
        AR[i] -= 428./105.*log(16.*x)*xpow[6];
        
        
    }
    
    free(xpow);
    
}

// some confuosn in the LAL code as to whether this is at theta = 0.33 or theta = 0.45
double IMRPhenomT_Inspiral_TaylorT3_t0(double *etapow, double *Spow, double dchi, double delta)
{
    int i;
    double fit;
    double eta, S;
    
    eta = etapow[1];
    S = Spow[1];
    
    // t0
    // theta = 0.45
    
    fit = (1.0/eta)*((-20.74399646637014 - 106.27711276502542*eta)/(1. + 0.6516016033332481*eta) + 0.0012450290074562259*dchi*delta*(1. - 4.701633367918768e6*eta)*etapow[2] -
                     111.5049997379579*dchi*delta*(1. + 19.95458485773613*eta)*S*etapow[2] + 1204.6829118499857*(1. - 4.025474056585855*eta)*dchi*dchi*etapow[3] +
                     S*(338.7318821277009 - 1553.5891860091408*eta + 19614.263378999745*etapow[2] - 156449.78737303324*etapow[3] + 577363.3090369126*etapow[4] - 802867.433363341*etapow[5]) +
                     (-55.75053935847546 - 290.36341163610575*eta + 7873.7667183299345*etapow[2] - 43585.59040070178*etapow[3] + 87229.84668746481*etapow[4] - 32469.263449695136*etapow[5])*Spow[2] +
                     (-102.8269343111326 + 5121.845705262981*eta - 93026.46878769135*etapow[2] + 650989.6793529999*etapow[3] - 1.8846061037110784e6*etapow[4] + 1.861602620702142e6*etapow[5])*Spow[3] +
                     (-7.294950933078567 + 314.24955197427136*eta - 3751.8509582195657*etapow[2] + 21205.339564205595*etapow[3] - 46448.94771114493*etapow[4] + 20310.512558558552*etapow[5])*Spow[4] +
                     (97.22312282683716 - 4556.60375328623*eta + 76308.73046927384*etapow[2] - 468784.4188333802*etapow[3] + 998692.0246600509*etapow[4] - 322905.9042578296*etapow[5])*Spow[5]);
    
    return(fit);
    
}

// extension to IMRPhenomTTaylorT3
double IMRPhenomTInspiralOmegaAnsatz22(double theta, struct IMRPhenomT *IMRPT){
 
   double theta8 = pow(theta,8.);
   double theta9 = theta8*theta;
   double theta10 = theta9*theta;
   double theta11 = theta10*theta;
   double theta12 = theta11*theta;
   double theta13 = theta12*theta;
 
   double fac = theta*theta*theta/8.;
 
   double taylort3 = IMRPhenomTTaylorT3(theta, IMRPT);
 
   double out =  IMRPT->omega[8]*theta8 + IMRPT->omega[9]*theta9 + IMRPT->omega[10]*theta10 + IMRPT->omega[11]*theta11 + IMRPT->omega[12]*theta12 + IMRPT->omega[13]*theta13;
 
   return taylort3 + 2.*fac*out;
}

void IMRPhenomT_Inspiral_Fit(double *InspiralFit, double *etapow, double *Spow, double dchi, double delta)
{
    int i;
    double fit;
    double eta, S;
    
    eta = etapow[1];
    S = Spow[1];

// IMRPhenomT_Inspiral_Freq_CP1_22
    // theta = 0.45

   fit = -0.014968864336704284*dchi*delta*(1. - 1.942061808318584*eta)*etapow[2] + 0.0017312772309375462*dchi*delta*(1. - 0.07106994121956058*eta)*S*etapow[2] + S*(0.0019208448318368731 - 0.0013579968243452476*eta - 0.0033501404728414627*etapow[2] + 0.008914420175326192*etapow[3]) +
   6.687615165457298e-6*dchi*dchi*etapow[3] + (0.02104073275966069 + 717.1534194224539*eta + 85.37320237350282*etapow[2] + 12.789214868358362*etapow[3] - 16.00243777208413*etapow[4])/(1. + 32934.586638893634*eta) +
   (-8.306810248117731e-6 + 0.00009918593182087119*eta - 0.003805916669791129*etapow[2] + 0.009854209286892323*etapow[3])*Spow[2] + (-5.578836442449699e-6 - 0.0030378960591856616*eta + 0.03746366675135751*etapow[2] - 0.10298471015315146*etapow[3])*Spow[3] +
   (0.00004425141111368952 - 0.0008702073302258368*eta + 0.006538604805919268*etapow[2] - 0.01578597166324495*etapow[3])*Spow[4] + (-0.000019469656288570753 + 0.002969863931498354*eta - 0.03643271052162611*etapow[2] + 0.09959495981802587*etapow[3])*Spow[5] +
   (-0.000042037164406446896 + 0.0007336074135429041*eta - 0.005603356997202016*etapow[2] + 0.013439843000090702*etapow[3])*Spow[6];
    
    InspiralFit[1] = fit;
 
// IMRPhenomT_Inspiral_Freq_CP2_22
   // theta = 0.55

   fit = -0.04486391236129559*dchi*delta*(1. - 1.8997912248414794*eta)*etapow[2] - 0.003531802135161727*dchi*delta*(1. - 8.001211450141325*eta)*S*etapow[2] + S*(0.0061664395419698285 - 0.0040934633081508905*eta - 0.009180337242551828*etapow[2] + 0.020338583755834694*etapow[3]) +
   0.00006524644306613066*dchi*dchi*etapow[3] + (0.03711511661217631 - 0.10663782888636487*eta - 0.09963406984414182*etapow[2] + 0.6597367702009397*etapow[3] - 2.777344875144891*etapow[4] + 4.220674345359693*etapow[5])/(1. - 3.2125452791404148*eta) +
   (0.00044302547647888445 + 0.000424246501303979*eta - 0.01394093576260671*etapow[2] + 0.02634851560709597*etapow[3])*Spow[2] + (0.00011582043047950321 - 0.008282652950117982*eta + 0.08965067576998058*etapow[2] - 0.23963885130463913*etapow[3])*Spow[3] +
   (0.0006123158975881322 - 0.007809160444435783*eta + 0.028517174579539676*etapow[2] - 0.03717957419042746*etapow[3])*Spow[4] + (-0.0000885530893214531 + 0.005939789043536808*eta - 0.07106551435109858*etapow[2] + 0.1891131957235774*etapow[3])*Spow[5] +
   (-0.0005110853374341054 + 0.0038762476596420855*eta + 0.005094077179675256*etapow[2] - 0.047971766995287136*etapow[3])*Spow[6];
    
    InspiralFit[2] = fit;
 
 //  IMRPhenomT_Inspiral_Freq_CP3_22
    // theta = 0.65
 
   fit = -0.10196878573773932*dchi*delta*(1. - 1.8918584778973513*eta)*etapow[2] - 0.018820536453940443*dchi*delta*(1. - 3.7307154599131183*eta)*S*etapow[2] - 0.00013162098437956188*dchi*dchi*etapow[3] +
   S*(0.0145572994468378 - 0.0017482433991394227*eta - 0.10299007619034371*etapow[2] + 0.4581039376357615*etapow[3] - 0.7123678787549022*etapow[4]) +
   (0.05489007025458171 + 5.852073438961151*eta + 2.74597705533403*etapow[2] + 4.834336623113389*etapow[3] - 26.931994454691022*etapow[4] + 57.67035368809743*etapow[5])/(1. + 105.52132834236778*eta) +
   (0.003001211395915229 + 0.0017929418998452987*eta - 0.13776590125456148*etapow[2] + 0.7471133710854526*etapow[3] - 1.3620323111858437*etapow[4])*Spow[2] +
   (0.001143282743686261 - 0.05793457776296727*eta + 0.7841331051705482*etapow[2] - 3.4936244160305323*etapow[3] + 4.802357041496856*etapow[4])*Spow[3] +
   (0.0009168588840889624 - 0.03261437094899735*eta + 0.3472881896838799*etapow[2] - 1.3634383958859384*etapow[3] + 1.7313939586675267*etapow[4])*Spow[4] +
   (-0.0002794014744432316 + 0.055911057147527664*eta - 0.8686311380514122*etapow[2] + 4.096191294930781*etapow[3] - 6.009676060669872*etapow[4])*Spow[5] +
   (-0.0005046018052528331 + 0.029804593053788925*eta - 0.3792653361049425*etapow[2] + 1.6366976231421981*etapow[3] - 2.26904099961476*etapow[4])*Spow[6];
    
    InspiralFit[3] = fit;
 
 // IMRPhenomT_Inspiral_Freq_CP4_22
    // theta = 0.75

 
   fit = -0.1831889759662071*dchi*delta*(1. - 1.8484261527766557*eta)*etapow[2] - 0.07586202965525136*dchi*delta*(1. - 3.2918162656371983*eta)*S*etapow[2] + 0.0019259052728265817*dchi*dchi*etapow[3] +
   S*(0.02685637375751212 + 0.013341664908359861*eta - 0.3057217933283597*etapow[2] + 1.395763446325911*etapow[3] - 2.2559396974665376*etapow[4]) +
   (0.0725639467287476 + 12.39400068457852*eta + 12.907450928972402*etapow[2] - 7.422660061864399*etapow[3] + 66.32985901506036*etapow[4] - 117.85875779454518*etapow[5])/(1. + 168.63492460136445*eta) +
   (0.0087781653701194 + 0.006944161553839352*eta - 0.3301149078235105*etapow[2] + 1.6835714783903248*etapow[3] - 2.950404929598742*etapow[4])*Spow[2] +
   (0.0037229746496019625 - 0.17155338099487646*eta + 2.5881802140836774*etapow[2] - 13.14710199375518*etapow[3] + 21.366803256010915*etapow[4])*Spow[3] +
   (0.00278507305662002 - 0.12475855143364532*eta + 1.8640209516178643*etapow[2] - 10.117078727717564*etapow[3] + 17.94244821676711*etapow[4])*Spow[4] +
   (0.0010273954584773936 + 0.1713357629442166*eta - 3.017249223460983*etapow[2] + 15.855096360798678*etapow[3] - 26.444621592311933*etapow[4])*Spow[5] +
   (-0.00012207946532225968 + 0.11709700788855186*eta - 2.0950821618097026*etapow[2] + 11.925324501640054*etapow[3] - 21.683978511818076*etapow[4])*Spow[6];
    
    InspiralFit[4] = fit;
 
 // IMRPhenomT_Inspiral_Freq_CP5_22
    // theta = 0.82

 
   fit = -0.2508206617297265*dchi*delta*(1. - 1.861010982421798*eta)*etapow[2] - 0.1392163711259171*dchi*delta*(1. - 3.2669366465555796*eta)*S*etapow[2] + 0.0023126403170013045*dchi*dchi*etapow[3] +
   S*(0.036750064163293766 + 0.036904343404333906*eta - 0.5238739410356437*etapow[2] + 2.3292117112945223*etapow[3] - 3.654184701923543*etapow[4]) + (0.08373610487663233 + 6.301736487754372*eta + 9.03911386193751*etapow[2] + 4.91153188278086*etapow[3])/(1. + 72.64820846804257*eta) +
   (0.014963449678540705 + 0.008354571522567225*eta - 0.41723078020683*etapow[2] + 2.2007932082378785*etapow[3] - 4.245354787320365*etapow[4])*Spow[2] +
   (0.005706180633326235 - 0.15748500622007494*eta + 2.3477109912232845*etapow[2] - 11.413877195221694*etapow[3] + 17.033120593116756*etapow[4])*Spow[3] +
   (0.003890296981717687 - 0.15985471334551038*eta + 2.560312006077997*etapow[2] - 14.400920672743332*etapow[3] + 26.10406142567958*etapow[4])*Spow[4] +
   (0.005305988847210204 + 0.10869207132210629*eta - 2.4201307115268875*etapow[2] + 12.544899744864924*etapow[3] - 19.550600837316903*etapow[4])*Spow[5] +
   (0.002917248769788225 + 0.11851143848720952*eta - 2.6640023622893416*etapow[2] + 15.993378498844761*etapow[3] - 29.752144941054446*etapow[4])*Spow[6];
    
    InspiralFit[5] = fit;


   }

/* Merger 22 frequency collocation points*/
 
double IMRPhenomT_Merger_Freq_CP1_22(double *etapow, double *Spow, double dchi, double delta) // theta=0.95
{
 
        double fit;
        double eta = etapow[1];
        double S = Spow[1];
 
   fit = -0.3926039690467202*dchi*delta*(1. - 2.359180951434749*eta)*etapow[2] - 0.28551098014898896*dchi*delta*(1 - 3.414696100901444*eta)*S*etapow[2] + 0.003414004344822246*dchi*dchi*etapow[3] +
   S*(0.05697014130854102 + 0.07170430925984912*eta - 0.9606499306623374*etapow[2] + 5.440955307244598*etapow[3] - 10.594319036394571*etapow[4]) +
   (0.10030959768350425 + 44.56725135920024*eta + 163.96290948585087*etapow[2] - 143.05635831020462*etapow[3] + 393.8084861740473*etapow[4])*pow(1. + 436.6494065618*eta,-1.) +
   (0.021213606590798472 + 0.2148355967310081*eta - 2.7747405367196265*etapow[2] + 13.771088220299802*etapow[3] - 25.128755397215368*etapow[4])*Spow[2] +
   (-0.003645992092251503 + 0.2137524962844931*eta - 0.644979226062801*etapow[2] - 1.7314849842209137*etapow[3] + 5.573297392347478*etapow[4])*Spow[3] + (0.029352214609533665 - 0.6020287633594307*eta + 7.014738679280164*etapow[2] - 36.027159248248296*etapow[3] + 63.42605850359639*etapow[4])*Spow[4] +
   (0.0356519646654399 - 0.5569780178251297*eta + 4.017784725334053*etapow[2] - 15.05881246593488*etapow[3] + 22.94821359434365*etapow[4])*Spow[5];
 
        return fit;
}



/* Ansatz for the ringdown amplitude of a general (l,m) mode. Equivalent to eq 4 of Damour&Nagar 2014 (https://arxiv.org/pdf/1406.0401.pdf).
   Also described in eq 25 of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf). */
double IMRPhenomTRDAmpAnsatzHM(double t, struct IMRPhenomT *IMRPT)
{
        double tpeak = IMRPT->tshift;
        double c3 = IMRPT->c3;
  double c4 = IMRPT->amp_c4_prec;
  double c2 = IMRPT->amp_c2_prec;
  double c1 = IMRPT->amp_c1_prec;
 
  gsl_complex phi = gsl_complex_rect(c2*(t-tpeak) + c3,0);
  gsl_complex tanhphi = gsl_complex_tanh(phi);
  double tanh = GSL_REAL(tanhphi);
 
  double expAlpha = exp(-IMRPT->alpha1RD_prec*(t-tpeak));
 
  return expAlpha*(c1*tanh + c4);
}

/* Ansatz for the merger amplitude of a general (l,m) mode. A phenomenological expression in terms of hyperbolic functions is employed,
   as described in eq 29 of PhenomTHM paper (https://dcc.ligo.org/DocDB/0172/P2000524/001/PhenomTHM_SH-3.pdf) */
double IMRPhenomTMergerAmpAnsatzHM(double t, struct IMRPhenomT *IMRPT)
{
        double tpeak = IMRPT->tshift;
 
        gsl_complex phi = gsl_complex_rect(IMRPT->alpha1RD*(t-tpeak),0);
        gsl_complex phi2 = gsl_complex_rect(2*IMRPT->alpha1RD*(t-tpeak),0);
        gsl_complex secphi = gsl_complex_sech(phi);
        gsl_complex secphi2 = gsl_complex_sech(phi2);
 
        double sech1 = GSL_REAL(secphi);
        double sech2 = GSL_REAL(secphi2);
 
        double aux = IMRPT->mergerC1 + IMRPT->mergerC2*sech1 + IMRPT->mergerC3*pow(sech2,1./7.) + IMRPT->mergerC4*(t-tpeak)*(t-tpeak);
        return aux;
}
