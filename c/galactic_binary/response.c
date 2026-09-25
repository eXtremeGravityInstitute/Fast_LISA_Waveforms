// Copyright (C) Neil Cornish.
// SPDX-License-Identifier: GPL-3.0-or-later
// Build with make in this directory; TDI-2 uses tdi2_response.c.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <complex.h>
#include <string.h>
#include <ctype.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_complex.h>

#include "wdm.h"
#include "tdi2_response.h"
#include <time.h>

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])


#define AU 1.4959787e11      // Astronomical Unit in meters
#define SECSYR 3.15581498e7
#define PARSEC 3.08568025e16    // Parsec in meters
#define CLIGHT 2.99792458e8     // Speed of light in m/s
#define sq3 1.7320508075688773
#define TSUN  4.92549232189886339689643862e-6  // mass to seconds conversion
#define IRTWO 0.707106781186547549

static const gsl_rng_type *rngtype;
static const gsl_rng *rng;
static int use_tdi2 = 0;

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


void orbit(int NT);

void wdmvalues(struct wdmshape *wdms);
void wdmband(double Tobs, double *params, struct wdmshape *wdms, int *kstart, int *kwidth);

void full_response(double *tarray, int N, double *params, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf);
void constellation(int Ns, double *tarray, double **Larray, double ***Parray, double ***Varray);
void fast_response(double *tarray, int N, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel *ARacc, gsl_spline *ARspline, gsl_interp_accel *PRacc, gsl_spline *PRspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf);
void ampphase(double t, double Amp, double Aplus, double Across, double cos2psi, double sin2psi, double phi0, double f0, double fdot, double fddot,  double *hp, double *hc,  double *hpf, double *hcf);
void UCB_ampphase(double t, gsl_interp_accel *ARacc, gsl_spline *ARspline, gsl_interp_accel *PRacc, gsl_spline *PRspline, double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf);
void TDI_spline(double *M, double *Mf, int a, int b, int c, double* tarray, int n, gsl_interp_accel *ARacc, gsl_spline *ARspline, gsl_interp_accel *PRacc, gsl_spline *PRspline, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
void TDI(double *M, double *Mf, int a, int b, int c, double* tarray, int n, double Amp, double Aplus, double Across, double cos2psi, double sin2psi, double phi0, double f0, double fdot, double fddot, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
void detector_time(double *tarray, double *tspace, double *params, gsl_interp_accel **SPacc, gsl_spline **SPspline, int N);

void wavemake(double df, double DF, int *Nfsam, double *fd, double *Phase, double *freq, double *freqd, double *Amp, double ***lookup, int *kmin, int *kmax, int *rlist, double *wave, double *wavef);

void wavelist(double df, double DF, int *Nfsam, double *fd, double *freqX, double *freqY, double *freqZ, double *fdotX, double *fdotY, double *fdotZ, double *waveX, double *waveY, double *waveZ, int *list, int *rlist, int *NM, int *kmin, int *kmax);



void unwrap(int Ns, double *phi);
void extractAP(int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
void UCBwaveform(double *params, int NS, double *times, double *Phase, double *Amp);

void phihetF(double *phihf, int Nfx);
void wdmtranF(int kx, int km, double *phihf, double *data, double **wdmout, double **wdmoutq);

void phihet(double *phih);
void wdmtranT(int K, int kx, double *phih, double *data, double *wdmout);

void wavelet_TDI_hetT(double Tobs, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, double df, double DF, double HBW, double *phih, int K, int *list, int *NM, double *waveX, double *waveY, double *waveZ);

void wavelet_TDI_hetF(double Tobs, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, double df, double HBW, int *list, int *NM, double *waveX, double *waveY, double *waveZ, struct wdmshape *wdms);


void wavelet_TDI(double Tobs, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, double df, double DF, double HBW, double ***lookup, int *Nfsam, double *fd, int *list, int *NM, double *waveX, double *waveY, double *waveZ, double *waveXf, double *waveYf, double *waveZf);

double phase(double t, double *params);
double freq(double t, double *params);

void spacecraft(double t, double *x, double *y, double *z);
void spacraft_loc(int i, double *x, double *y, double *z, double sa, double ca, double sb, double cb);
void spacraft_loc_Kepler(int i, double *x, double *y, double *z, double ci, double si, double alpha, double beta);

void coefficients(int *Nfsam, double *fd, double ***lookup);
void wavelet(int m, double *wave, int N, double nrm, double dom, double DOM, double A, double B, double insDOM);
double phitilde(double om, double insDOM, double A, double B);

void tukey(double *data, double alpha, int N);

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


int main(int argc, char **argv)
{
    int N, Ns;
    int i, j, k, K, mc;
    double Tobs, fny;
    double *params;
    double m1, m2, Mt, Mc, eta;
    double f, fdot, fddot, u, v, w, p, q, t;
    double DL, Amp, dtc;
    double *tarray, *X, *Y, *Z;
    double *Xf, *Yf, *Zf;
    double *phiB, *Dphi, *As;
    double DT, DF, Tfilt;
    double *fd;
    int *Nfsam;
    int Nsfmax;
    
    
    FILE *out;
    FILE *in;
    
    clock_t start, end;
    double cpu_time_used;
    
    int sparse_hetf = 0, dense_only_samples = 0;
    for(i=1; i<argc; i++)
    {
        if(strcmp(argv[i], "--sparse-hetF") == 0) sparse_hetf = 1;
        if(strcmp(argv[i], "--tdi2") == 0) use_tdi2 = 1;
        if(strcmp(argv[i], "--dense-only") == 0)
        {
            if(++i >= argc || (dense_only_samples = atoi(argv[i])) < 2)
            {
                fprintf(stderr, "--dense-only requires a sample count of at least 2\n");
                return 1;
            }
        }
    }

    struct wdmshape *wdms  = malloc(sizeof(struct wdmshape));
    
    wdmvalues(wdms);

    N = Nt*Nf;
    if(dense_only_samples > N)
    {
        fprintf(stderr, "--dense-only count exceeds the observation grid\n");
        return 1;
    }
    
    Tobs = dt*(double)(N);   // duration
    
    fny = 1.0/(2.0*dt);
    
    printf("Tobs = %e Nyquist = %e\n", Tobs, fny);
    
    DT = dt*(double)(Nf);           // width of wavelet pixel in time
    DF = 1.0/(2.0*dt*(double)(Nf));   // width of wavelet pixel in frequency
    
    printf("%e %e\n", DT, DF);
    
    
    // filter length
    K = mult*2*Nf;
    Tfilt = dt*(double)(K);
    
    params = double_vector(9);
    
    m1 = 0.6*TSUN;
    m2 = 0.7*TSUN;
    
    Mt = m1+m2;
    Mc = pow(m1*m2,3.0/5.0)/pow(Mt,1.0/5.0);
    eta = m1*m2/(Mt*Mt);
    
    // target frequency
     f = 5.0e-3;
    //  f = 5.0e-3+DF;
    //f = 5.0e-3-0.01*DF;
    f = 5.0e-3+0.6*DF;
    //f = 5.71e-03;
    
    fdot = 96.0*pow(M_PI,8.0/3.0)/5.0*pow(Mc,5.0/3.0)*pow(f,11.0/3.0);
    fddot = 11.0/3.0*fdot*fdot/f;
    
    printf("0PN %e %e\n", fdot, fddot);
    
    printf("frequency change over %f years fdot %e fddot %e\n", Tobs/SECSYR, fdot*Tobs, 0.5*fddot*Tobs*Tobs);
    
    printf("fmin %e fmax %e\n", f*(1.0-1e-4), (f+ fdot*Tobs + 0.5*fddot*Tobs*Tobs)*(1.0+1.0e-4));
    
    DL = 1.0e3*PARSEC/CLIGHT;
    Amp = 4.0*pow(Mc,5.0/3.0)*pow(M_PI*f,2.0/3.0)/DL;
    
    printf("Amplitude %e\n", Amp);
    
    
    params[0] = f;    // f

    params[1] = -0.783326909627;  // costh

    params[2] = 3.0;  // phi

    params[3] = Amp;  // Amp

    params[4] = 0.0707372016677; // cosi

    params[5] = 0.8;  // psi

    params[6] = 1.2;  // phi0

    params[7] = fdot; // fdot

    params[8] = fddot;// fddot
    
    u = DF*rint(f/DF);
    v = DF*rint((f+fdot*Tobs+0.5*fddot*Tobs*Tobs)/DF);
    
    out = fopen("frange.dat","w");
    fprintf(out,"%.15e %.15e\n", u-4.0*DF, v+4.0*DF);
    fclose(out);
    
    printf("spreads %e %e\n", fdot*Tobs*Tobs, f*1.0e-4*Tobs);
    
    
    if(!sparse_hetf)
    {
        int dense_count = dense_only_samples > 0 ? dense_only_samples : N;
        tarray = double_vector(dense_count);
        X = double_vector(dense_count);
        Y = double_vector(dense_count);
        Z = double_vector(dense_count);
        Xf = double_vector(dense_count);
        Yf = double_vector(dense_count);
        Zf = double_vector(dense_count);
        
        for(i=0; i<dense_count ;i++) tarray[i] = dt*(double)(i);
        
        // full, slow time domain TDI response
        
        start = clock();
        full_response(tarray, dense_count, params, X, Y, Z, Xf, Yf, Zf);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("time domain waveform calculation took %f seconds\n", cpu_time_used);
         
        
        out = fopen("X.dat","w");
        for(i=0; i<dense_count ;i++)
        {
            fprintf(out,"%.12e %.12e\n", tarray[i], X[i]);
        }
        fclose(out);
        
        out = fopen("Y.dat","w");
        for(i=0; i<dense_count ;i++)
        {
            fprintf(out,"%.12e %.12e\n", tarray[i], Y[i]);
        }
        fclose(out);
        
        out = fopen("Z.dat","w");
        for(i=0; i<dense_count ;i++)
        {
            fprintf(out,"%.12e %.12e\n", tarray[i], Z[i]);
        }
        fclose(out);
         
         
         free(tarray);
         free(X); free(Y), free(Z);
         free(Xf); free(Yf), free(Zf);
         if(dense_only_samples > 0) return 0;
    }
    
    double dtx;
    double **Larray;
    double ***Parray, ***Varray;
    
    Ns = (int)(200.0*Tobs/SECSYR);
    //printf("%d\n", Ns);
    if (Ns < 20) Ns = 20;
    dtx = (Tobs)/(double)(Ns-1);
    dtc = (Tobs+2.0*dtx)/(double)(Ns-1);  // tarray extends beyond observation time to allow for inter
    
    // The constellation orientation can be computed once and stored
    Larray = double_matrix(3,Ns);  // armlengths
    Parray = double_tensor(3,3,Ns);  // spacecraft positions
    Varray = double_tensor(3,3,Ns);  // arm vectors
    
    tarray = double_vector(Ns);
    for(i=0; i<Ns ;i++) tarray[i] = -dtx + dtc*(double)(i);   // coarse time sampling

    constellation(Ns, tarray, Larray, Parray, Varray);
    
    // now we spline the orbits. These can be used by all waveform types and only have to
    // be computed and stored once. Different waveform types will want to use different
    // time spacings for their interpolation, so we need to be able to access the orbit
    // information at any random time. For example, MBHBs will use time arrays that are
    // non-uniform (closer spaced near merger) and may terminate before the observation time
    // if the system merges.
    
    // Probably makes sense to put these splines into a spacecraft structure
 
    gsl_interp_accel **SLacc = malloc(3 * sizeof(gsl_interp_accel *));
    gsl_spline **SLspline = malloc(3 *sizeof(gsl_spline *));
    for(i=0; i<3; i++)
    {
        SLacc[i] = gsl_interp_accel_alloc();
        SLspline[i] = gsl_spline_alloc (gsl_interp_cspline, Ns);
    }
    
    for(i = 0 ; i< 3; i++)
    {
        gsl_spline_init(SLspline[i], tarray, Larray[i], Ns);
    }
    
    gsl_interp_accel **SPacc = malloc(9 * sizeof(gsl_interp_accel *));
    gsl_spline **SPspline = malloc(9 *sizeof(gsl_spline *));
    gsl_interp_accel **SVacc = malloc(9 * sizeof(gsl_interp_accel *));
    gsl_spline **SVspline = malloc(9 *sizeof(gsl_spline *));
    for(i=0; i<9; i++)
    {
        SPacc[i] = gsl_interp_accel_alloc();
        SPspline[i] = gsl_spline_alloc (gsl_interp_cspline, Ns);
        SVacc[i] = gsl_interp_accel_alloc();
        SVspline[i] = gsl_spline_alloc (gsl_interp_cspline, Ns);
    }
    
    for(i = 0 ; i< 3; i++)
    {
        for(j = 0 ; j< 3; j++)
        {
            k = j+i*3;
            gsl_spline_init(SPspline[k], tarray, Parray[i][j], Ns);
            gsl_spline_init(SVspline[k], tarray, Varray[i][j], Ns);
        }
    }
    
    
    double **wave, **wavef, ***lookup;
    double *waveX, *waveY, *waveZ;
    double *waveXf, *waveYf, *waveZf;
    double OM, DOM, A, B, BW, HBW, df;
    double c, s;
    int NM;
    int *list;
    
    // In sparse mode the live-pixel support is known from the UCB band
    // planner, so avoid allocating full Nt*Nf work arrays just to write
    // the compact comparison products.
    int Nwave = N;
    if(sparse_hetf)
    {
        int sparse_kx, sparse_kw;
        wdmband(Tobs, params, wdms, &sparse_kx, &sparse_kw);
        Nwave = Nt*sparse_kw;
    }
    list = int_vector(Nwave);
    
    waveX = double_vector(Nwave);  // live wavelet pixels for X
    waveY = double_vector(Nwave);  // live wavelet pixels for Y
    waveZ = double_vector(Nwave);  // live wavelet pixels for Z
    waveXf = NULL;
    waveYf = NULL;
    waveZf = NULL;
    wave = NULL;
    wavef = NULL;
    if(!sparse_hetf)
    {
        waveXf = double_vector(N);  // pi/2 shifted wavelet pixels for X
        waveYf = double_vector(N);  // pi/2 shifted wavelet pixels for Y
        waveZf = double_vector(N);  // pi/2 shifted wavelet pixels for Z
        wave = double_matrix(Nt,Nf);  // dense diagnostic unpacking of the signal
        wavef = double_matrix(Nt,Nf);  // dense diagnostic unpacking of the pi/2 shifted signal
    }
    
   // printf("%e %e %e\n", DF, DT, Tobs);
    
    OM = M_PI/dt;
      
    DOM = OM/(double)(Nf);
      
    B = Bfrac*DOM;
      
    A = (DOM-B)/2.0;
    
    // total width of wavelet in frequency
    BW = (A+B)/M_PI;
    HBW = 0.5*BW;
    
    // frequency spacing
    df = (BW)/(double)(Nsf);
    
    fd = double_vector(Nfd);
    
    double ddfdot;
    
    ddfdot = DF/Tfilt*dfdot;  // sets the f-dot increment
    

    for(j=0; j< Nfd; j++) fd[j] = -ddfdot*(double)(Nfd)/2.0 + (double)(j)*ddfdot;
    
    //printf("dfdot %e fdot min %e fdot max %e\n", ddfdot, fd[0], fd[Nfd-1]);
    
    if(sparse_hetf)
    {
        start = clock();
        wavelet_TDI_hetF(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, HBW, list, &NM, waveX, waveY, waveZ, wdms);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("sparse frequency domain wavelet calculation took %f seconds\n", cpu_time_used);
        printf("live pixels %d\n", NM);
        
        out = fopen("ucb_hetF_pixels.dat","w");
        fprintf(out,"# k list n m X Y Z\n");
        for(k=0; k<NM; k++)
        {
            j = (list[k]%Nt);
            i = (list[k]-j)/Nt;
            fprintf(out, "%d %d %d %d %.15e %.15e %.15e\n", k, list[k], j, i, waveX[k], waveY[k], waveZ[k]);
        }
        fclose(out);
        
        out = fopen("ucb_hetF_X_track_pixels.dat","w");
        for(k=0; k<NM; k++)
        {
            j = (list[k]%Nt);
            i = (list[k]-j)/Nt;
            fprintf(out, "%d %d %.15e\n", j, i, waveX[k]);
        }
        fclose(out);
        
        out = fopen("ucb_hetF_Y_track_pixels.dat","w");
        for(k=0; k<NM; k++)
        {
            j = (list[k]%Nt);
            i = (list[k]-j)/Nt;
            fprintf(out, "%d %d %.15e\n", j, i, waveY[k]);
        }
        fclose(out);
        
        out = fopen("ucb_hetF_Z_track_pixels.dat","w");
        for(k=0; k<NM; k++)
        {
            j = (list[k]%Nt);
            i = (list[k]-j)/Nt;
            fprintf(out, "%d %d %.15e\n", j, i, waveZ[k]);
        }
        fclose(out);
        
        return 0;
    }
    
    double *phih;
    int KX;
    
    KX = mult*2*4;
    phih = (double*)malloc(sizeof(double)* (KX));
    
    
    // set up phi(t) for the downsampled transform
    phihet(phih);
    
    out = fopen("phih.dat","w");
    for(j=0; j< KX; j++) fprintf(out,"%d %e\n", j, phih[j]);
    fclose(out);
    
    wavelet_TDI_hetT(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, DF, HBW, phih, KX, list, &NM, waveX, waveY, waveZ);
    
    start = clock();
    for(mc=0; mc<100; mc++)
    {
        wavelet_TDI_hetT(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, DF, HBW, phih, KX, list, &NM, waveX, waveY, waveZ);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("downsampled time domain wavelet calculation took %f seconds\n", cpu_time_used/100.0);
    
    
    start = clock();
    for(mc=0; mc<100; mc++)
    {
        wavelet_TDI_hetF(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, HBW, list, &NM, waveX, waveY, waveZ, wdms);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("downsampled frequency domain wavelet calculation took %f seconds\n", cpu_time_used/100.0);
    
    
    // number of terms in the lookup table at each frequency
    Nfsam = int_vector(Nfd);
    
    Nsfmax = 0;
    for(j=0; j< Nfd; j++)
    {
     Nfsam[j] = (int)((BW+fabs(fd[j])*Tfilt)/df);
     if(Nfsam[j]%2 != 0) Nfsam[j]++; // makes sure it is an even number
     //printf("%d %d %d\n", j, Nfsam[j], Nsf);
      if(Nfsam[j] > Nsfmax) Nsfmax = Nfsam[j];
    }
    
   
    lookup = double_tensor(Nfd,Nsfmax,2);  // records the cos and sin coefficients for the lookup table
    
    // compute the wavelet transform lookup table
    printf("computing the lookup tables\n");
    coefficients(Nfsam, fd, lookup);
    printf("finished computing the lookup tables\n");
    
    wavelet_TDI(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, DF, HBW, lookup, Nfsam, fd, list, &NM, waveX, waveY, waveZ, waveXf, waveYf, waveZf);
    
    //return 1;
    
    /*
    start = clock();
    for(mc=0; mc<100; mc++)
    {
        wavelet_TDI(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, DF, HBW, lookup, Nfsam, fd, list, &NM, waveX, waveY, waveZ, waveXf, waveYf, waveZf);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("fast wavelet calculation took %f seconds\n", cpu_time_used/100.0);
    */
    
    /* ************************************************************************
     
     Whichever waveform generator gets called last is the one the matches get computed for
     
    ************************************************************************* */
    
    // wavelet_TDI(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, DF, HBW, lookup, Nfsam, fd, list, &NM, waveX, waveY, waveZ, waveXf, waveYf, waveZf);
    
   // wavelet_TDI_hetT(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, DF, HBW, phih, KX, list, &NM, waveX, waveY, waveZ);
    
     wavelet_TDI_hetF(Tobs, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, df, HBW, list, &NM, waveX, waveY, waveZ, wdms);
    
    
    // initialize the array
    for(i=0; i< Nf; i++)
     {
      for(j=0; j< Nt; j++)
       {
           wave[j][i] = 0.0;
           wavef[j][i] = 0.0;
       }
    }
    
    // unpack the signal
    
    out = fopen("ucb.dat","w");
    for(k=0; k< NM; k++)
    {
        if(list[k] > 0)
        {
            j = (list[k]%Nt); // time index
            i = (list[k]-j)/Nt; // frequency index
            wave[j][i] = waveX[k];
            wavef[j][i] = waveXf[k];
            //printf("%d %d\n", j, i);
        }
        fprintf(out,"%d %.15e %.15e %.15e\n", list[k], waveX[k], waveY[k], waveZ[k]);
    }
    fclose(out);
    
    
    i = 92;
    out = fopen("f92.dat","w");
    for(j=0; j< Nt; j++)
     {
         fprintf(out, "%e %e %.14e\n", (double)(j)*DT, wave[j][i]*wave[j][i]+wavef[j][i]*wavef[j][i], atan2(wavef[j][i], wave[j][i]));
     }
    fclose(out);
    
    double px, pold, pw;
    int jx;
    
    jx = 0;
    i = 92;
    
    pold = atan2(wavef[jx][i], wave[jx][i]);
    pw = 0.0;
   
    out = fopen("f92_0.dat","w");
    for(j=jx; j< Nt; j += 4)
     {
         px = atan2(wavef[j][i], wave[j][i]);
         if(pold -px > 5.0) pw += 2.0*M_PI;
         if(px - pold > 5.0) pw -= 2.0*M_PI;
         pold = px;
         fprintf(out, "%e %e %.14e\n", (double)(j)*DT, wave[j][i]*wave[j][i]+wavef[j][i]*wavef[j][i], px+pw);
     }
    fclose(out);
    
    jx = 1;
    i = 92;
    
    pold = atan2(wavef[jx][i], wave[jx][i]);
    pw = 0.0;
   
    out = fopen("f92_1.dat","w");
    for(j=jx; j< Nt; j += 4)
     {
         px = atan2(wavef[j][i], wave[j][i]);
         if(pold -px > 5.0) pw += 2.0*M_PI;
         if(px - pold > 5.0) pw -= 2.0*M_PI;
         pold = px;
         fprintf(out, "%e %e %.14e\n", (double)(j)*DT, wave[j][i]*wave[j][i]+wavef[j][i]*wavef[j][i], px+pw);
     }
    fclose(out);
    
    jx = 2;
    i = 92;
    
    pold = atan2(wavef[jx][i], wave[jx][i]);
    pw = 0.0;
   
    out = fopen("f92_2.dat","w");
    for(j=jx; j< Nt; j += 4)
     {
         px = atan2(wavef[j][i], wave[j][i]);
         if(pold -px > 5.0) pw += 2.0*M_PI;
         if(px - pold > 5.0) pw -= 2.0*M_PI;
         pold = px;
         fprintf(out, "%e %e %.14e\n", (double)(j)*DT, wave[j][i]*wave[j][i]+wavef[j][i]*wavef[j][i], px+pw);
     }
    fclose(out);
    
    jx = 3;
    i = 92;
    
    pold = atan2(wavef[jx][i], wave[jx][i]);
    pw = 0.0;
   
    out = fopen("f92_3.dat","w");
    for(j=jx; j< Nt; j += 4)
     {
         px = atan2(wavef[j][i], wave[j][i]);
         if(pold -px > 5.0) pw += 2.0*M_PI;
         if(px - pold > 5.0) pw -= 2.0*M_PI;
         pold = px;
         fprintf(out, "%e %e %.14e\n", (double)(j)*DT, wave[j][i]*wave[j][i]+wavef[j][i]*wavef[j][i], px+pw);
     }
    fclose(out);
    
    
    
    i = 93;
    out = fopen("f93.dat","w");
    for(j=0; j< Nt; j++)
     {
         fprintf(out, "%e %e %.14e\n", (double)(j)*DT, wave[j][i]*wave[j][i]+wavef[j][i]*wavef[j][i], atan2(wavef[j][i], wave[j][i]));
     }
    fclose(out);
    
    
    

    out = fopen("FastX.dat","w");


        for(i=0; i< Nf; i++)
        {
           for(j=0; j< Nt; j++)
            {
            fprintf(out, "%e %e %.14e\n", (double)(j)*DT, (double)(i)*DF, wave[j][i]);
            }
                
        fprintf(out, "\n");
                
      }
     fclose(out);
    
    out = fopen("Xout.dat","w");

    for(j=0; j< Nt; j++)
     {
        for(i=0; i< Nf; i++)
        {
            fprintf(out, "%.14e ", wave[j][i]);
        }
                
        fprintf(out, "\n");
                
      }
     fclose(out);
    
    
    // initialize the array
    for(i=0; i< Nf; i++)
     {
      for(j=0; j< Nt; j++)
       {
           wave[j][i] = 0.0;
       }
    }
    
    // unpack the signal
    for(k=0; k< NM; k++)
    {
        if(list[k] > 0)
        {
            j = (list[k]%Nt);
            i = (list[k]-j)/Nt;
            wave[j][i] = waveY[k];
        }
    }
    

    out = fopen("FastY.dat","w");


        for(i=0; i< Nf; i++)
        {
           for(j=0; j< Nt; j++)
            {
            fprintf(out, "%e %e %.14e\n", (double)(j)*DT, (double)(i)*DF, wave[j][i]);
            }
                
        fprintf(out, "\n");
                
      }
     fclose(out);
    
    // initialize the array
    for(i=0; i< Nf; i++)
     {
      for(j=0; j< Nt; j++)
       {
           wave[j][i] = 0.0;
       }
    }
    
    // unpack the signal
    for(k=0; k< NM; k++)
    {
        if(list[k] > 0)
        {
            j = (list[k]%Nt);
            i = (list[k]-j)/Nt;
            wave[j][i] = waveZ[k];
        }
    }
    

    out = fopen("FastZ.dat","w");


        for(i=0; i< Nf; i++)
        {
           for(j=0; j< Nt; j++)
            {
            fprintf(out, "%e %e %.14e\n", (double)(j)*DT, (double)(i)*DF, wave[j][i]);
            }
                
        fprintf(out, "\n");
                
      }
     fclose(out);
    
    
    out = fopen("XYZ.dat","w");
    for(k=0; k< NM; k++)
    {
        j = (list[k]%Nt);
        i = (list[k]-j)/Nt;
        fprintf(out, "%d %d %d %d %e %e %e\n", k, list[k], j, i, waveX[k], waveY[k], waveZ[k]);
    }
    fclose(out);
    
    

    return 0;
}


void coefficients(int *Nfsam, double *fd, double ***lookup)
{
  int i, j, k, n, K;
  int Nstep;
  char filename[1024];
  double Tfilt, Tchunk, df;
  double DT, DF;
  double x, y, z, xa, xb, dx;
    int N, NC, M, NK, up;
    int Nx, NT;
    int n0, n1, n2, n3;
    double f, t;
    double c, s, delf;
    double phase, freq;
    double m1, m2, Mt, Mc, eta, tc, Theta, fisco, fdot;
    double tx, tmid;
    double fcent, delt;
    double odc, ods, evc, evs;
    double OM, DOM, A, B, BW;
    double insDOM;
    double om, fac, dom, nrm;
    double *phi, *DX;
    
    int ii, jj;
    
    double *wave;
    
    int prime;
    char ch;
    
     DT = dt*(double)(Nf);           // width of wavelet pixel in time
     DF = 1.0/(2.0*dt*(double)(Nf));   // width of wavelet pixel in frequency
     
     OM = M_PI/dt;
     
     DOM = OM/(double)(Nf);
     
     insDOM = 1.0/sqrt(DOM);
     
     B = Bfrac*DOM;
     
     A = (DOM-B)/2.0;
     
     K = mult*2*Nf;
     
     Tfilt = dt*(double)(K);
     
     printf("Filter length (seconds) %e\n", Tfilt);
     
     dom = 2.0*M_PI/Tfilt;  // max frequency is K/2*dom = pi/dt = OM
     
     printf("full filter bandwidth %e  samples %d\n", (A+B)/M_PI, (int)(((A+B)/M_PI)*Tfilt));

     DX = (double*)malloc(sizeof(double)* (2*K));

     wave = (double*)malloc(sizeof(double)* (K));
     
     //zero frequency
     REAL(DX,0) =  insDOM;
     IMAG(DX,0) =  0.0;
     
     // postive frequencies
     for(i=1; i<= K/2; i++)
     {
         om = (double)(i)*dom;
         z = phitilde(om, insDOM, A, B);
         REAL(DX,i) =  z;
         IMAG(DX,i) =  0.0;
     }
     
     // negative frequencies
     for(i=1; i< K/2; i++)
     {
         om = -(double)(i)*dom;
         z = phitilde(om, insDOM, A, B);
         REAL(DX,K-i) =  z;
         IMAG(DX,K-i) =  0.0;
     }
     
     gsl_fft_complex_radix2_backward(DX, 1, K);
     
     phi = (double*)malloc(sizeof(double)* (K));
     
            for(i=0; i < K/2; i++)
             {
              phi[i] = REAL(DX,K/2+i);
             }
            for(i=0; i< K/2; i++)
              {
              phi[K/2+i] = REAL(DX,i);
              }

     
     nrm = 0.0;
     for(i=0; i < K; i++) nrm += phi[i]*phi[i]*dt;
     nrm = sqrt(nrm);
    // printf("phi norm %e\n", nrm);
    
    // it turns out that all the wavelet layers are the same modulo a
    // shift in the reference frequency. Just have to do a single layer
    // we pick one far from the boundaries to avoid edge effects
    
    k = Nf/16;
    
    wavelet(k, wave, K, nrm, dom, DOM, A, B, insDOM);
    
    // total width of wavelet in frequency
    BW = (A+B)/M_PI;

    // frequency spacing
    df = BW/(double)(Nsf);

        // The odd wavelets coefficienst can be obtained from the even.
        // odd cosine = -even sine, odd sine = even cosine

            
        // each wavelet covers a frequency band of width DW
        // execept for the first and last wasvelets
        // there is some overlap. The wavelet pixels are of width
        // DOM/PI, except for the first and last which have width
        // half that
            
            
        fcent = (double)(k)*DF;
            
            for(jj=0; jj< Nfd; jj++)  // loop over f-dot slices
            {
                
            printf("%d %d\n", jj, Nfsam[jj]);
                
            for(j=0; j< Nfsam[jj]; j++)  // loop of frequency slices
            {
                f = fcent+((double)(j-Nfsam[jj]/2)+0.5)*df;

                evc = 0.0;
                evs = 0.0;
                
                for(i=0; i< K; i++)
                {
                    t = ((double)(i-K/2))*dt;
                    z = 2.0*M_PI*f*t+M_PI*fd[jj]*t*t;
                    c = cos(z)*dt;
                    s = sin(z)*dt;
                    evc += wave[i]*c;
                    evs += wave[i]*s;
                }
                
                
                lookup[jj][j][0] = evc;
                lookup[jj][j][1] = evs;
                
            }
                
            }

       
   free_double_vector(wave);
   free_double_vector(phi);
   free_double_vector(DX);
    
    return;

}

void wdmvalues(struct wdmshape *wdms)
{
    wdms->DT = dt*(double)(Nf);           // width of wavelet pixel in time
    wdms->DF = 1.0/(2.0*dt*(double)(Nf));   // width of wavelet pixel in frequency
    wdms->OM = M_PI/dt; // angular Nyquist frequency
    wdms->DOM = wdms->OM/(double)(Nf);
    /* This must match the window used by wdm_transform.c.  Changing B changes
       how power is shared by neighboring WDM frequency layers. */
    wdms->B = Bfrac*wdms->DOM;
    wdms->A = (wdms->DOM-wdms->B)/2.0;
    wdms->insDOM = 1.0/sqrt(wdms->DOM);
    wdms->FB = (wdms->A+wdms->B)/(2.0*M_PI); // extent
    wdms->DFA =  wdms->A/(2.0*M_PI); // half-width of non-overlapping frequency region
    wdms->Tfilt = dt*(double)(mult*2*Nf);
    wdms->Tobs = dt*(double)(Nt*Nf);
}

void wavelet(int m, double *wave, int N, double nrm, double dom, double DOM, double A, double B, double insDOM)
{
    
    int i;
    double om;
    double x, y, z;
    double *DE;
    
     DE = (double*)malloc(sizeof(double)* (2*N));

         // zero and postive frequencies
          for(i=0; i<= N/2; i++)
           {
            om = (double)(i)*dom;
            
            y = phitilde(om+(double)(m)*DOM, insDOM, A, B);
            z = phitilde(om-(double)(m)*DOM, insDOM, A, B);
               
               x = y+z;
            
               REAL(DE,i) = IRTWO*x;
               IMAG(DE,i) = 0.0;

           }
     
           // negative frequencies
            for(i=1; i< N/2; i++)
            {
             om = -(double)(i)*dom;
                         
              y = phitilde(om+(double)(m)*DOM, insDOM, A, B);
              z = phitilde(om-(double)(m)*DOM, insDOM, A, B);
                         
              x = y+z;
    
                REAL(DE,N-i) = IRTWO*x;
                IMAG(DE,N-i) = 0.0;
             
             }
    
          gsl_fft_complex_radix2_backward(DE, 1, N);
        
        
                 for(i=0; i < N/2; i++)
                    {
                        wave[i] = REAL(DE,N/2+i)/nrm;
                    }
                 for(i=0; i< N/2; i++)
                 {
                      wave[i+N/2] = REAL(DE,i)/nrm;
                 }

      free(DE);
    
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


void wavelet_TDI(double Tobs, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, double df, double DF, double HBW, double ***lookup, int *Nfsam, double *fd, int *list, int *NM, double *waveX, double *waveY, double *waveZ, double *waveXf, double *waveYf, double *waveZf)
{
    
    int i, j, k, Ns, flag;
    double DT, f;
    double *X, *Y, *Z, *Xf, *Yf, *Zf;
    double  *phiX, *AX, *phiY, *AY, *phiZ, *AZ;
    double *phiR, *AR, *phiRI, *phiB;
    double tinc;
    double *TF;
    double dtx, dtc;
    double *tarray, *times;
    double *pref, *fref, *fdref;
    int *kmin, *kmax;
    
    clock_t start, end;
    double cpu_time_used;
    
    FILE *out;
    
    DT = dt*(double)(Nf);           // width of wavelet pixel in time
    
    // each waveform type will want its own time spacing. For galactic binaries uniform spacing is fine
    // This section will need a flag to tell it what waveform type we are computing. For MBHMs the parameters
    // of the signal will impact the time spacing
    
    Ns = (int)(200.0*Tobs/SECSYR);
    if (Ns < 20) Ns = 20;
    dtx = (Tobs)/(double)(Ns-1);
    dtc = (Tobs+2.0*dtx)/(double)(Ns-1);  // tarray extends beyond observation time to allow for interpolation
    
    // phase and amplitude interpolation need extra points at each end to
    // allow for light travel times
    phiRI = double_vector(Ns);
    AR = double_vector(Ns);
    times = double_vector(Ns);
    tarray = double_vector(Ns);
    
    start = clock();
    
    // times array is used to interpoate the waveform
    // tarray is used to produce the TDI. times array needs to extend past tarray
    // for interpolation
    
    for (i=0; i< Ns; i++) times[i] = -dtx + (double)(i)*dtc;
    for (i=0; i< Ns; i++) tarray[i] = (double)(i)*dtx;
    
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
    

    
    // get amplitude and phase at reference times on a coarse grid
     UCBwaveform(params, Ns, times, phiRI, AR);
    
    // spline the reference amplitude and phase
    gsl_interp_accel *ARacc = gsl_interp_accel_alloc();
    gsl_spline *ARspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *PRacc = gsl_interp_accel_alloc();
    gsl_spline *PRspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_spline_init(ARspline, times, AR, Ns);
    gsl_spline_init(PRspline, times, phiRI, Ns);
    
    free(AR);
    free(phiRI);
    free(times);
    
    
    
    // coarse sampled phase at Barycenter
    phiB = double_vector(Ns);
    // coarse sampled phase at reference spacecraft
    phiR = double_vector(Ns);
    double *tspace;
    tspace = double_vector(Ns);
    // shift reference times from Barycenter to spacecraft 0
    detector_time(tarray, tspace, params, SPacc, SPspline, Ns);
    for (i=0; i< Ns; i++)
    {
        phiR[i] = gsl_spline_eval(PRspline, tspace[i], PRacc);
        phiB[i] = gsl_spline_eval(PRspline, tarray[i], PRacc);
    }
    free(tspace);
    

    pref = (double*)malloc(sizeof(double)* (Nt));
    fref = (double*)malloc(sizeof(double)* (Nt));
    fdref = (double*)malloc(sizeof(double)* (Nt));
    TF = (double*)malloc(sizeof(double)* (Nt));
    for (i=0; i< Nt; i++) TF[i] = ((double)(i))*DT;  // time center of the wavelet pixels
    
    tspace = double_vector(Nt);
    // shift reference times from Barycenter to spacecraft 0
    detector_time(TF, tspace, params, SPacc, SPspline, Nt);
    for (i=0; i< Nt; i++)
    {
        pref[i] = gsl_spline_eval(PRspline, tspace[i], PRacc);
        fref[i] = gsl_spline_eval_deriv(PRspline, tspace[i], PRacc)/(2.0*M_PI);
        fdref[i] = gsl_spline_eval_deriv2(PRspline, tspace[i], PRacc)/(2.0*M_PI);
    }
    free(tspace);
    
    fast_response(tarray, Ns, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, ARacc, ARspline, PRacc, PRspline, X, Y, Z, Xf, Yf, Zf);
    
    extractAP(Ns, AY, phiY, Y, Yf, phiR);
    
   /*
    out = fopen("Yextract.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%e %e %e %e\n", tarray[i], AY[i], phiY[i], phiR[i]);
    }
    fclose(out);
    */
    
    // remove any phase wraps
    unwrap(Ns, phiY);
    
    /*
    out = fopen("Yunwrap.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%e %e %e %e\n", tarray[i], AY[i], phiY[i], phiR[i]-phiB[i]);
    }
    fclose(out);
    */
    
    extractAP(Ns, AX, phiX, X, Xf, phiR);
    
    /*
    out = fopen("Xextract.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%e %e %e %e\n", tarray[i], AX[i], phiX[i], phiR[i]);
    }
    fclose(out);
    */
    
    // remove any phase wraps
    unwrap(Ns, phiX);
    
    /*
    out = fopen("Xunwrap.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%e %e %e %e\n", tarray[i], AX[i], phiX[i], phiR[i]-phiB[i]);
    }
    fclose(out);
    */

    
    extractAP(Ns, AZ, phiZ, Z, Zf, phiR);
    
    /*
    out = fopen("Zextract.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%e %e %e %e\n", tarray[i], AZ[i], phiZ[i], phiR[i]);
    }
    fclose(out);
    */

    // remove any phase wraps
    unwrap(Ns, phiZ);
    
    /*
    out = fopen("Zunwrap.dat","w");
    for (i=0; i< Ns; i++)
    {
        fprintf(out,"%e %e %e %e\n", tarray[i], AZ[i], phiZ[i], phiR[i]-phiB[i]);
    }
    fclose(out);
     */
    

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("TDI calculation took %f seconds\n", cpu_time_used);
     
    
    
    double *XPhase, *Xfreq, *XAmp, *Xfdot;
    double *YPhase, *Yfreq, *YAmp, *Yfdot;
    double *ZPhase, *Zfreq, *ZAmp, *Zfdot;
    
    XAmp = (double*)malloc(sizeof(double)* (Nt));
    XPhase = (double*)malloc(sizeof(double)* (Nt));
    Xfreq = (double*)malloc(sizeof(double)* (Nt));
    Xfdot = (double*)malloc(sizeof(double)* (Nt));
    YAmp = (double*)malloc(sizeof(double)* (Nt));
    YPhase = (double*)malloc(sizeof(double)* (Nt));
    Yfreq = (double*)malloc(sizeof(double)* (Nt));
    Yfdot = (double*)malloc(sizeof(double)* (Nt));
    ZAmp = (double*)malloc(sizeof(double)* (Nt));
    ZPhase = (double*)malloc(sizeof(double)* (Nt));
    Zfreq = (double*)malloc(sizeof(double)* (Nt));
    Zfdot = (double*)malloc(sizeof(double)* (Nt));

    gsl_interp_accel *Aacc = gsl_interp_accel_alloc();
    gsl_spline *Aspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *Pacc = gsl_interp_accel_alloc();
    gsl_spline *Pspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
  
    gsl_spline_init(Aspline, tarray, AX, Ns);
    gsl_spline_init(Pspline, tarray, phiX, Ns);
    
    
   // out = fopen("Xpffd.dat","w");
    for (i=0; i< Nt; i++)
    {
        XAmp[i] = gsl_spline_eval(Aspline, TF[i], Aacc);
        XPhase[i] = gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i];
    }
    
    for (i=0; i< Nt; i++)
    {
        Xfreq[i] = gsl_spline_eval_deriv(Pspline, TF[i], Pacc)/(2.0*M_PI) + fref[i];
        Xfdot[i] = gsl_spline_eval_deriv2(Pspline, TF[i], Pacc)/(2.0*M_PI) + fdref[i];
       // fprintf(out,"%e %e %e %e %e\n", TF[i], XAmp[i], XPhase[i]-pref[i], Xfreq[i], Xfdot[i]);
    }
   // fclose(out);
    
    double *Xtime;
    int N;
    
    N = Nt*Nf;
    
    double alpha, t;
    alpha = 1.0e5/Tobs;
    
    double *td, *pd, *ad, *tb;
    tb = double_vector(N);
    td = double_vector(N);
    pd = double_vector(N);
    ad = double_vector(N);
    
    for (i=0; i< N; i++)
    {
        t = (double)(i)*dt;
        tb[i] = t;
    }
    
    detector_time(tb, td, params, SPacc, SPspline, N);
    
    
    UCBwaveform(params, N, td, pd, ad);
    
    int Nslow;
    double dtslow;
    
    Nslow = 512;
    dtslow = (double)(N)/(double)(Nslow)*dt;
    
    double *tds, *pds, *ads, *tbs;
    double *twindow;
    tbs = double_vector(Nslow);
    tds = double_vector(Nslow);
    pds = double_vector(Nslow);
    ads = double_vector(Nslow);
    twindow = double_vector(Nslow);
    
    double fc;
    double *Xts, *PX;
    Xts = double_vector(Nslow);
    PX = double_vector(Nslow/2);
    Xtime = double_vector(N);
    
    printf("ratio %d %e %e\n", N/Nslow, Tobs, (double)(Nslow)*dtslow);
    
    for (i=0; i< N; i++) Xtime[i] = 1.0;
    tukey(Xtime,alpha,N);
    gsl_fft_real_radix2_transform(Xtime, 1, N);
    for (i=0; i< N; i++) Xtime[i] *= (2.0*dt);
    
    Xts[0] = Xtime[0];
    for (i=1; i< Nslow/2; i++)
    {
        Xts[i] = Xtime[i];
        Xts[Nslow-i] = Xtime[N-i];
    }
    gsl_fft_halfcomplex_radix2_inverse(Xts, 1, Nslow);
    for (i=0; i< Nslow; i++) Xts[i] /= (2.0*dtslow);
    
    for (i=0; i< Nslow; i++) Xtime[i] = 1.0;
    tukey(Xtime,alpha,Nslow);
    
    out = fopen("wt1.dat","w");
    for (i=0; i< Nslow; i++)
    {
        t = (double)(i)*dtslow;
        twindow[i] = Xts[i];
        fprintf(out,"%e %e %e\n", t, twindow[i], Xtime[i]);
    }
    fclose(out);
    
   
    for (i=0; i< N; i++)
    {
        t = (double)(i)*dt;
        Xtime[i] = 0.0;
        if(t < TF[Nt-1]) Xtime[i] = gsl_spline_eval(Aspline, t, Aacc)*cos(gsl_spline_eval(Pspline, t, Pacc)+pd[i]);
    }
    

    tukey(Xtime,alpha,N);
    
    out = fopen("X_time.dat","w");
    for (i=0; i< N; i++)
    {
        t = (double)(i)*dt;
        fprintf(out,"%e %e\n", t, Xtime[i]);
    }
    fclose(out);
    

    gsl_fft_real_radix2_transform(Xtime, 1, N);
    for (i=0; i< N; i++) Xtime[i] *= (2.0*dt);
    
    // average frequency
    f = params[0]+0.5*(params[7]*Tobs+0.5*params[8]*Tobs*Tobs);
    
    
    
    fc = floor(f*Tobs)/Tobs-(double)(Nslow/4)/Tobs;
    
    k = (int)(fc*Tobs);
     if(k%2 != 0) k++;  // needs to be even otherwise heterodyne phase is off by a bin
     fc = (double)(k)/Tobs;
    
    double x, y, z;
    double u;


    
    i = 1;
    out = fopen("Xf.dat","w");
    x = atan2(Xtime[N-(i+k)],Xtime[i+k]);
    y = 0.0;
    for (i=1; i< Nslow/2; i++)
    {
        f = (double)(i)/Tobs;
        z = atan2(Xtime[N-(i+k)],Xtime[i+k]);
        if(z-x > 2.0) y -= 2.0*M_PI;
        if(x-z > 2.0) y += 2.0*M_PI;
        x = z;
        fprintf(out,"%e %e %e\n", f+fc, Xtime[i+k]*Xtime[i+k]+Xtime[N-(i+k)]*Xtime[N-(i+k)], z);
    }
    fclose(out);
    
    
 
     for (i=0; i< Nslow; i++)
        {
            t = (double)(i)*dtslow;
            tbs[i] = t;
        }

    
    detector_time(tbs, tds, params, SPacc, SPspline, Nslow);
    
    UCBwaveform(params, Nslow, tds, pds, ads);
        
        for (i=0; i< Nslow; i++)
        {
            t = tbs[i];
            Xts[i] = 0.0;
            if(t < TF[Nt-1]) Xts[i] = gsl_spline_eval(Aspline, t, Aacc)*cos(gsl_spline_eval(Pspline, t, Pacc)+pds[i]-2.0*M_PI*fc*t);
        }
        
         //tukey(Xts,alpha,Nslow);
        
        // modifed Tukey
        for (i=0; i< Nslow; i++) Xts[i] *= twindow[i];
        
        out = fopen("Xslow_time.dat","w");
        for (i=0; i< Nslow; i++)
        {
            t = tbs[i];
            fprintf(out,"%e %e\n", t, Xts[i]);
        }
        fclose(out);
        
        gsl_fft_real_radix2_transform(Xts, 1, Nslow);
        for (i=0; i< Nslow; i++) Xts[i] *= (2.0*dtslow);
        
        
    x = 0.0;
    y = 0.0;
    z = 0.0;
    u = 0.0;
    out = fopen("Xfslow.dat","w");
        for (i=1; i< Nslow/2; i++)
        {
            f = (double)(i)/Tobs;
            fprintf(out,"%e %e %e %e\n", f+fc, Xts[i]*Xts[i]+Xts[Nslow-i]*Xts[Nslow-i], atan2(Xts[Nslow-i],Xts[i]), atan2(Xtime[N-(i+k)],Xtime[i+k]));
            x += Xts[i]*Xts[i]+Xts[Nslow-i]*Xts[Nslow-i];
            y += Xtime[i+k]*Xtime[i+k]+Xtime[N-(i+k)]*Xtime[N-(i+k)];
            z += Xtime[i+k]*Xts[i]+Xtime[N-(i+k)]*Xts[Nslow-i];
            u += Xtime[N-(i+k)]*Xts[i]-Xtime[i+k]*Xts[Nslow-i];
        }
    fclose(out);
        
    printf("Match = %e %e MM = %e\n", z/sqrt(x*y), sqrt(z*z+u*u)/sqrt(x*y), 1.0-z/sqrt(x*y));
    
    
    for (i=0; i< N; i++) Xtime[i] = 1.0;
    tukey(Xtime,alpha,N);
    gsl_fft_real_radix2_transform(Xtime, 1, N);
    for (i=0; i< N; i++) Xtime[i] *= (2.0*dt);
    
    for (i=0; i< Nslow; i++) Xts[i] = twindow[i];
    gsl_fft_real_radix2_transform(Xts, 1, Nslow);
    for (i=0; i< Nslow; i++) Xts[i] *= (2.0*dtslow);
    
    out = fopen("windows.dat","w");
    for (i=1; i< Nslow/2; i++)
    {
        f = (double)(i)/Tobs;
        fprintf(out,"%e %e %e %e %e %e %e\n", f, Xts[i]*Xts[i]+Xts[Nslow-i]*Xts[Nslow-i], Xtime[i]*Xtime[i]+Xtime[N-(i)]*Xtime[N-(i)], Xts[i], Xtime[i], Xts[Nslow-i], Xtime[N-i]);
    }
    fclose(out);
    
 

    
    
    /*
    double t, x, y, z;
    out = fopen("check.dat","w");
    for (i=0; i< 200; i++)
    {
        t = 1.86e7+1.2e6*(double)(i)/200.0;
        x = gsl_spline_eval(Pspline, t, Pacc);
        y = gsl_spline_eval_deriv(Pspline, t, Pacc)/(2.0*M_PI);
        z = gsl_spline_eval_deriv2(Pspline, t, Pacc)/(2.0*M_PI);
        fprintf(out,"%e %e %e %e\n", t, x, y, z);
    }
    fclose(out);
    */
    
    gsl_spline_init(Aspline, tarray, AY, Ns);
    gsl_spline_init(Pspline, tarray, phiY, Ns);
    //out = fopen("Ypffd.dat","w");
    for (i=0; i< Nt; i++)
    {
        YAmp[i] = gsl_spline_eval(Aspline, TF[i], Aacc);
        YPhase[i] = gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i];
        Yfreq[i] = gsl_spline_eval_deriv(Pspline, TF[i], Pacc)/(2.0*M_PI) + fref[i];
        Yfdot[i] = gsl_spline_eval_deriv2(Pspline, TF[i], Pacc)/(2.0*M_PI) + fdref[i];
       // fprintf(out,"%e %e %e %e %e\n", TF[i], YAmp[i], YPhase[i]-pref[i], Yfreq[i], Yfdot[i]);
    }
   // fclose(out);
    
    
   
    
    gsl_spline_init(Aspline, tarray, AZ, Ns);
    gsl_spline_init(Pspline, tarray, phiZ, Ns);
    //out = fopen("Zpffd.dat","w");
    for (i=0; i< Nt; i++)
    {
        ZAmp[i] = gsl_spline_eval(Aspline, TF[i], Aacc);
        ZPhase[i] = gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i];
        Zfreq[i] = gsl_spline_eval_deriv(Pspline, TF[i], Pacc)/(2.0*M_PI) + fref[i];
        Zfdot[i] = gsl_spline_eval_deriv2(Pspline, TF[i], Pacc)/(2.0*M_PI) + fdref[i];
       // fprintf(out,"%e %e %e %e %e\n", TF[i], ZAmp[i], ZPhase[i]-pref[i], Zfreq[i], Zfdot[i]);
    }
   //fclose(out);
    
    
    gsl_spline_free(Pspline);
    gsl_spline_free(Aspline);
    gsl_interp_accel_free(Pacc);
    gsl_interp_accel_free(Aacc);
    
    gsl_spline_free(PRspline);
    gsl_spline_free(ARspline);
    gsl_interp_accel_free(PRacc);
    gsl_interp_accel_free(ARacc);
    
    kmin = int_vector(Nt);
    kmax = int_vector(Nt);
    
    int *rlist;
    
    rlist = int_vector(Nt*Nf);
    
    wavelist(df, DF, Nfsam, fd, Xfreq, Yfreq, Zfreq, Xfdot, Yfdot, Zfdot, waveX, waveY, waveZ, list, rlist, NM, kmin, kmax);
    
    wavemake(df, DF, Nfsam, fd, XPhase, Xfreq, Xfdot, XAmp, lookup, kmin, kmax, rlist, waveX, waveXf);
    wavemake(df, DF, Nfsam, fd, YPhase, Yfreq, Yfdot, YAmp, lookup, kmin, kmax, rlist, waveY, waveYf);
    wavemake(df, DF, Nfsam, fd, ZPhase, Zfreq, Zfdot, ZAmp, lookup, kmin, kmax, rlist, waveZ, waveZf);
    
    free(rlist);
    
    
    free(X), free(Y), free(Z);
    free(Xf), free(Yf), free(Zf);
    free(phiR), free(phiX), free(phiY), free(phiZ);
    free(AX), free(AY), free(AZ);
    
    free(TF);
    free(pref), free(fref);
    free(XPhase), free(YPhase), free(ZPhase);
    free(XAmp), free(YAmp), free(ZAmp);
    free(Xfreq), free(Yfreq), free(Zfreq);
    free(Xfdot), free(Yfdot), free(Zfdot);
    
}

void wavelet_TDI_hetT(double Tobs, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, double df, double DF, double HBW, double *phih, int K, int *list, int *NM, double *waveX, double *waveY, double *waveZ)
{
    
    int i, j, k, Ns;
    int kw, mm;
    double DT, f, fx, delf;
    double *X, *Y, *Z, *Xf, *Yf, *Zf;
    double  *phiX, *AX, *phiY, *AY, *phiZ, *AZ;
    double *phiR, *AR, *phiRI;
    double tinc;
    double *TF;
    double dtx, dtc;
    double *tarray, *times;
    double *pref;
    int *kmin, *kmax;
    
    FILE *out;
    
    DT = dt*(double)(Nf);           // width of wavelet pixel in time
    
    // each waveform type will want its own time spacing. For galactic binaries uniform spacing is fine
    // This section will need a flag to tell it what waveform type we are computing. For MBHMs the parameters
    // of the signal will impact the time spacing
    
    Ns = (int)(200.0*Tobs/SECSYR);
    if (Ns < 20) Ns = 20;
    dtx = (Tobs)/(double)(Ns-1);
    dtc = (Tobs+2.0*dtx)/(double)(Ns-1);  // tarray extends beyond observation time to allow for interpolation
    
    // phase and amplitude interpolation need extra points at each end to
    // allow for light travel times
    phiRI = double_vector(Ns);
    AR = double_vector(Ns);
    times = double_vector(Ns);
    tarray = double_vector(Ns);
    
    // times array is used to interpoate the waveform
    // tarray is used to produce the TDI. times array needs to extend past tarray
    // for interpolation
    
    for (i=0; i< Ns; i++) times[i] = -dtx + (double)(i)*dtc;
    for (i=0; i< Ns; i++) tarray[i] = (double)(i)*dtx;
    
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
    
    // get amplitude and phase at reference times on a coarse grid
     UCBwaveform(params, Ns, times, phiRI, AR);
    
    // spline the reference amplitude and phase
    gsl_interp_accel *ARacc = gsl_interp_accel_alloc();
    gsl_spline *ARspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *PRacc = gsl_interp_accel_alloc();
    gsl_spline *PRspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_spline_init(ARspline, times, AR, Ns);
    gsl_spline_init(PRspline, times, phiRI, Ns);
    
    free(AR);
    free(phiRI);
    free(times);
    
    // coarse sampled phase at reference spacecraft
    phiR = double_vector(Ns);
    double *tspace;
    tspace = double_vector(Ns);
    // shift reference times from Barycenter to spacecraft 0
    detector_time(tarray, tspace, params, SPacc, SPspline, Ns);
    for (i=0; i< Ns; i++)
    {
        phiR[i] = gsl_spline_eval(PRspline, tspace[i], PRacc);
    }
    free(tspace);
    
    int Nx;
    
    kw = (int)(params[0]/DF);
    fx = (double)(kw-1)*DF;
    
   // printf("%d %e %e %e %e\n", kw, fx, params[0], params[0]-fx, DF);
    
    Nx = Nt*4;
    dtx = DT/4.0;

    pref = (double*)malloc(sizeof(double)* (Nx));
    TF = (double*)malloc(sizeof(double)* (Nx));
    for (i=0; i< Nx; i++) TF[i] = ((double)(i))*dtx;
    
    tspace = double_vector(Nx);
    // shift reference times from Barycenter to spacecraft 0
    detector_time(TF, tspace, params, SPacc, SPspline, Nx);
    for (i=0; i< Nx; i++)
    {
        pref[i] = gsl_spline_eval(PRspline, tspace[i], PRacc);
    }
    free(tspace);
    
    fast_response(tarray, Ns, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, ARacc, ARspline, PRacc, PRspline, X, Y, Z, Xf, Yf, Zf);
    
    extractAP(Ns, AX, phiX, X, Xf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiX);
    
    extractAP(Ns, AY, phiY, Y, Yf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiY);
    
    extractAP(Ns, AZ, phiZ, Z, Zf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiZ);
    
    double *XWave, *YWave, *ZWave;
    double *HPhase;  // heterodyning phase
    
    XWave = (double*)malloc(sizeof(double)* (Nx));
    YWave = (double*)malloc(sizeof(double)* (Nx));
    ZWave = (double*)malloc(sizeof(double)* (Nx));
    
    HPhase = (double*)malloc(sizeof(double)* (Nx));
    for (i=0; i< Nx; i++) HPhase[i] = 2.0*M_PI*fx*TF[i];

    gsl_interp_accel *Aacc = gsl_interp_accel_alloc();
    gsl_spline *Aspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *Pacc = gsl_interp_accel_alloc();
    gsl_spline *Pspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
  
    gsl_spline_init(Aspline, tarray, AX, Ns);
    gsl_spline_init(Pspline, tarray, phiX, Ns);
    for (i=0; i< Nx; i++)
    {
        XWave[i] = gsl_spline_eval(Aspline, TF[i], Aacc)*cos(gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i]-HPhase[i]);
    }
    
    gsl_spline_init(Aspline, tarray, AY, Ns);
    gsl_spline_init(Pspline, tarray, phiY, Ns);
    for (i=0; i< Nx; i++)
    {
        YWave[i] = gsl_spline_eval(Aspline, TF[i], Aacc)*cos(gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i]-HPhase[i]);
    }
    
    gsl_spline_init(Aspline, tarray, AZ, Ns);
    gsl_spline_init(Pspline, tarray, phiZ, Ns);
    for (i=0; i< Nx; i++)
    {
        ZWave[i] = gsl_spline_eval(Aspline, TF[i], Aacc)*cos(gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i]-HPhase[i]);
    }
    
    free(HPhase);
    
    free(X), free(Y), free(Z);
    free(Xf), free(Yf), free(Zf);
    free(phiR), free(phiX), free(phiY), free(phiZ);
    free(AX), free(AY), free(AZ);
    free(pref);

    gsl_spline_free(Pspline);
    gsl_spline_free(Aspline);
    gsl_interp_accel_free(Pacc);
    gsl_interp_accel_free(Aacc);

    gsl_spline_free(PRspline);
    gsl_spline_free(ARspline);
    gsl_interp_accel_free(PRacc);
    gsl_interp_accel_free(ARacc);
    

    mm = 0;
    for (i=0; i< Nt; i++)
    {
        for (k=kw; k<= kw+1; k++)  // a bit wasteful since upper layer might not get used (can improve this)
        {
            list[mm] = i+k*Nt;    // pixel reference
            mm++;
        }
    }
  
    *NM = mm;
    
    wdmtranT(K, kw, phih, XWave, waveX);
    wdmtranT(K, kw, phih, YWave, waveY);
    wdmtranT(K, kw, phih, ZWave, waveZ);
    
    free(TF);
    free(XWave), free(YWave), free(ZWave);
    
    
}

void wdmband(double Tobs, double *params, struct wdmshape *wdms, int *kstart, int *kwidth)
{
    double fs, fe;
    double fmin, fmax;
    double dfmin, dfmax;
    int k, kk;
    int kmin, kmax;
    
    
    fs = params[0];
    fe = (params[0]+params[7]*Tobs+0.5*params[8]*Tobs*Tobs);
    
    if(fs < fe)
    {
        fmin = fs*(1.0-1.0e-4);
        fmax = fe*(1.0+1.0e-4);
    }
    else
    {
        fmin = fe*(1.0-1.0e-4);
        fmax = fs*(1.0+1.0e-4);
    }
    
    kk = (int)(rint(fmin/wdms->DF));
    
    k = kk;
    // ensures the heterodyned doesn't lower any part of the signal into the zero layer
    // zero layer extends out to DF/2+(A+B)
    dfmin = fmin - (double)(k)*wdms->DF;
    do
    {
        if(dfmin < wdms->DF/2.0+wdms->FB)
        {
            k--;
            dfmin = fmin - (double)(k)*wdms->DF;
        }
    }while(dfmin < wdms->DF/2.0+wdms->FB);
     
    if(k%2 != 0) k--;  // has to be even. Both for the heterodyne and the WDM transform
    kmin = k;
    
    kmax = kk;
    dfmax = fmax - (double)(kmax+1)*wdms->DF;
    // ensures the heterodyned signal doesn't extend into the Nyquist layer
    // Nyquits extends down to to (kmax+1)DF-(A+B)
    do
    {
        if(dfmax+wdms->FB > 0.0)
        {
            kmax++;
            dfmax = fmax - (double)(kmax+1)*wdms->DF;
        }
    }while(dfmax+wdms->FB > 0.0);
    
    *kstart = kmin;
    *kwidth = kmax-kmin;

    
}

void wavelet_TDI_hetF(double Tobs, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, double df, double HBW, int *list, int *NM, double *waveX, double *waveY, double *waveZ, struct wdmshape *wdms)
{
    
    int i, j, k, Ns;
    int kx, kw, mm, odev;
    double f, fx, delf;
    double *X, *Y, *Z, *Xf, *Yf, *Zf;
    double  *phiX, *AX, *phiY, *AY, *phiZ, *AZ;
    double *phiR, *AR, *phiRI;
    double tinc;
    double *TF;
    double dtx, dtc;
    double *tarray, *times;
    double *pref;
    double *phihf;
    int *kmin, *kmax;
    
    FILE *out;
    
    // each waveform type will want its own time spacing. For galactic binaries uniform spacing is fine
    // This section will need a flag to tell it what waveform type we are computing. For MBHMs the parameters
    // of the signal will impact the time spacing
    
    Ns = (int)(200.0*Tobs/SECSYR);
    if (Ns < 20) Ns = 20;
    dtx = (Tobs)/(double)(Ns-1);
    dtc = (Tobs+2.0*dtx)/(double)(Ns-1);  // tarray extends beyond observation time to allow for interpolation
    
    // phase and amplitude interpolation need extra points at each end to
    // allow for light travel times
    phiRI = double_vector(Ns);
    AR = double_vector(Ns);
    times = double_vector(Ns);
    tarray = double_vector(Ns);
    
    // times array is used to interpoate the waveform
    // tarray is used to produce the TDI. times array needs to extend past tarray
    // for interpolation
    
    for (i=0; i< Ns; i++) times[i] = -dtx + (double)(i)*dtc;
    for (i=0; i< Ns; i++) tarray[i] = (double)(i)*dtx;
    
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
    

    
    // get amplitude and phase at reference times on a coarse grid
     UCBwaveform(params, Ns, times, phiRI, AR);
    
    // spline the reference amplitude and phase
    gsl_interp_accel *ARacc = gsl_interp_accel_alloc();
    gsl_spline *ARspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *PRacc = gsl_interp_accel_alloc();
    gsl_spline *PRspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_spline_init(ARspline, times, AR, Ns);
    gsl_spline_init(PRspline, times, phiRI, Ns);
    
    free(AR);
    free(phiRI);
    free(times);
    
    // coarse sampled phase at reference spacecraft
    phiR = double_vector(Ns);
    double *tspace;
    tspace = double_vector(Ns);
    // shift reference times from Barycenter to spacecraft 0
    detector_time(tarray, tspace, params, SPacc, SPspline, Ns);
    for (i=0; i< Ns; i++)
    {
        phiR[i] = gsl_spline_eval(PRspline, tspace[i], PRacc);
    }
    free(tspace);
    
    fast_response(tarray, Ns, params, SLacc, SLspline, SPacc, SPspline, SVacc, SVspline, ARacc, ARspline, PRacc, PRspline, X, Y, Z, Xf, Yf, Zf);
    
    extractAP(Ns, AX, phiX, X, Xf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiX);
    
    extractAP(Ns, AY, phiY, Y, Yf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiY);
    
    extractAP(Ns, AZ, phiZ, Z, Zf, phiR);
    // remove any phase wraps
    unwrap(Ns, phiZ);
    

    int Nx, Nfx;
    
    
    wdmband(Tobs, params, wdms, &kx, &kw);
    
    fx = (double)(kx)*wdms->DF;
    Nfx = kw+1;
    Nx = Nt*Nfx;
    dtx = wdms->DT/(double)(Nfx);
    
    phihf = (double*)malloc(sizeof(double)* (Nt/2+1));
    
    phihetF(phihf, Nfx);
    
    //printf("%d %d %e %e %e %e %e\n", kx, kw, fx, params[0], params[0]-fx, wdms->DF, wdms->FB);
    
   // printf("%e %e\n", 1.0/(2.0*dtx),  params[0]-fx);
    

    pref = (double*)malloc(sizeof(double)* (Nx));
    TF = (double*)malloc(sizeof(double)* (Nx));
    for (i=0; i< Nx; i++) TF[i] = ((double)(i))*dtx;
    
    tspace = double_vector(Nx);
    // shift reference times from Barycenter to spacecraft 0
    detector_time(TF, tspace, params, SPacc, SPspline, Nx);
    for (i=0; i< Nx; i++)
    {
        pref[i] = gsl_spline_eval(PRspline, tspace[i], PRacc);
    }
    free(tspace);
    
    double *XWave, *YWave, *ZWave;
    double *HPhase;  // heterodyning phase
    
    XWave = (double*)malloc(sizeof(double)* (Nx));
    YWave = (double*)malloc(sizeof(double)* (Nx));
    ZWave = (double*)malloc(sizeof(double)* (Nx));
    
    HPhase = (double*)malloc(sizeof(double)* (Nx));
    for (i=0; i< Nx; i++) HPhase[i] = 2.0*M_PI*fx*TF[i];

    gsl_interp_accel *Aacc = gsl_interp_accel_alloc();
    gsl_spline *Aspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
    gsl_interp_accel *Pacc = gsl_interp_accel_alloc();
    gsl_spline *Pspline = gsl_spline_alloc (gsl_interp_cspline, Ns);
  
    gsl_spline_init(Aspline, tarray, AX, Ns);
    gsl_spline_init(Pspline, tarray, phiX, Ns);
    
    for (i=0; i< Nx; i++)
    {
        XWave[i] = gsl_spline_eval(Aspline, TF[i], Aacc)*cos(gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i]-HPhase[i]);
    }

    gsl_spline_init(Aspline, tarray, AY, Ns);
    gsl_spline_init(Pspline, tarray, phiY, Ns);
    for (i=0; i< Nx; i++)
    {
        YWave[i] = gsl_spline_eval(Aspline, TF[i], Aacc)*cos(gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i]-HPhase[i]);
    }
    
    gsl_spline_init(Aspline, tarray, AZ, Ns);
    gsl_spline_init(Pspline, tarray, phiZ, Ns);
    for (i=0; i< Nx; i++)
    {
        ZWave[i] = gsl_spline_eval(Aspline, TF[i], Aacc)*cos(gsl_spline_eval(Pspline, TF[i], Pacc)+pref[i]-HPhase[i]);
    }
    
    free(HPhase);
    
    /* out = fopen("Xwave.dat","w");
    for (i=0; i< Nx; i++)
    {
       fprintf(out,"%e %e\n", TF[i], XWave[i]);
    }
   fclose(out);*/
    
    free(X), free(Y), free(Z);
    free(Xf), free(Yf), free(Zf);
    free(phiR), free(phiX), free(phiY), free(phiZ);
    free(AX), free(AY), free(AZ);
    free(pref);

    gsl_spline_free(Pspline);
    gsl_spline_free(Aspline);
    gsl_interp_accel_free(Pacc);
    gsl_interp_accel_free(Aacc);

    gsl_spline_free(PRspline);
    gsl_spline_free(ARspline);
    gsl_interp_accel_free(PRacc);
    gsl_interp_accel_free(ARacc);
    
    
 
    mm = 0;
    for (i=0; i< Nt; i++)
    {
        for (k=kx; k< kx+kw; k++)  // a bit wasteful since upper layer might not get used (can improve this)
        {
            list[mm] = i+(k+1)*Nt;    // pixel reference
            mm++;
        }
    }
  
    *NM = mm;
    
    double **wdmout, **wdmoutq;
    wdmout = double_matrix(Nt,Nfx);
    wdmoutq = double_matrix(Nt,Nfx);
    
    wdmtranF(kx, kw, phihf, XWave, wdmout, wdmoutq);
    mm = 0;
    for (i=0; i< Nt; i++)
    {
        for (k=kx; k< kx+kw; k++)
        {
            waveX[mm] = wdmout[i][k-kx];
            mm++;
        }
    }
    wdmtranF(kx, kw, phihf, YWave, wdmout, wdmoutq);
    mm = 0;
    for (i=0; i< Nt; i++)
    {
        for (k=kx; k< kx+kw; k++)
        {
            waveY[mm] = wdmout[i][k-kx];
            mm++;
        }
    }
    wdmtranF(kx, kw, phihf, ZWave, wdmout, wdmoutq);
    mm = 0;
    for (i=0; i< Nt; i++)
    {
        for (k=kx; k< kx+kw; k++)
        {
            waveZ[mm] = wdmout[i][k-kx];
            mm++;
        }
    }
    
    free_double_matrix(wdmout,Nt);
    free_double_matrix(wdmoutq,Nt);
    free(TF); free(phihf);
    free(XWave), free(YWave), free(ZWave);
    
    
}


void phihet(double *phih)
{
    
    double DT, DF;
    double dtx;
    double OM, DOM, insDOM;
    double om, z, dom;
    double A, B, T;
    double nrm, fac;
    double *DX;
    int M, L, K;
    int i;
    
    
    DT = dt*(double)(Nf);           // width of wavelet pixel in time
    DF = 1.0/(2.0*dt*(double)(Nf)); // width of wavelet pixel in frequency
    
    dtx = DT/4.0;
    
    OM = M_PI/dtx;
    
    M = 4;
    
    L = 2*M;
    
    DOM = OM/(double)(M);
    
    insDOM = 1.0/sqrt(DOM);

    
    B = Bfrac*DOM;
    
    A = (DOM-B)/2.0;
    
    K = mult*2*M;
    
    T = dtx*(double)(K);
    
    dom = 2.0*M_PI/T;  // max frequency is K/2*dom = pi/dtx = OM
    
    DX = (double*)malloc(sizeof(double)* (2*K+2));
    
    //zero frequency
    REAL(DX,0) =  insDOM;
    IMAG(DX,0) =  0.0;
    
    // postive frequencies
    for(i=1; i<= K/2; i++)
    {
        om = (double)(i)*dom;
        z = phitilde(om, insDOM, A, B);
        REAL(DX,i) =  z;
        IMAG(DX,i) =  0.0;
    }
    
    // negative frequencies
    for(i=1; i< K/2; i++)
    {
        om = -(double)(i)*dom;
        z = phitilde(om, insDOM, A, B);
        REAL(DX,K-i) =  z;
        IMAG(DX,K-i) =  0.0;
    }
    
    gsl_fft_complex_radix2_backward(DX, 1, K);
       
            for(i=0; i < K/2; i++)
             {
                phih[i] = REAL(DX,K/2+i);
             }
             for(i=0; i< K/2; i++)
             {
                 phih[K/2+i] = REAL(DX,i);
             }
    
    // need to figure out the normalization
    nrm = 0.0;
    for(i=0; i < K; i++) nrm += phih[i]*phih[i]*dtx;
    nrm = sqrt(nrm);
    
    fac = sqrt(2.0)*dtx/nrm;

    
    for(i=0; i < K; i++) phih[i] *= fac;
    
    free(DX);
    
}


void phihetF(double *phihf, int Nfx)
{
    
    double DT, DF;
    double dtx;
    double OM, DOM, insDOM;
    double om, z, dom;
    double A, B, T;
    double nrm, fac;
    double *DX;
    int M, L, K;
    int i, l;
    
    
    DT = dt*(double)(Nf);           // width of wavelet pixel in time
    DF = 1.0/(2.0*dt*(double)(Nf)); // width of wavelet pixel in frequency
    
    dtx = DT/(double)(Nfx);
    
    OM = M_PI/dtx;
    
    DOM = OM/(double)(Nfx);
    
    insDOM = 1.0/sqrt(DOM);

    B = Bfrac*DOM;
    
    A = (DOM-B)/2.0;
    
    K = mult*2*Nfx;
    
    T = DT*(double)(Nt);
    
    dom = 2.0*M_PI/T;


    for(i=0; i<= Nt/2; i++)
    {
        om = (double)(i)*dom;
        phihf[i] = phitilde(om, insDOM, A, B);
    }
    
    nrm = 0.0;
    for(l=-Nt/2; l<= Nt/2; l++) nrm += phihf[abs(l)]*phihf[abs(l)];
    nrm = sqrt(nrm/dtx);
    
     for(i=0; i<= Nt/2; i++) phihf[i] /= nrm;
    
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


void wdmtranF(int kx, int kw, double *phihf, double *data, double **wdmout, double **wdmoutq)
{
    int Nx, Nfx;
    double alpha, fac, x;
    double *DX;
    int n, m, i, j, jj, mm, mt;
    int flag;

    gsl_fft_real_wavetable * real;
    gsl_fft_real_workspace * work;
    
    Nfx = kw+1;

    Nx = Nfx*Nt;
    
    /* Match the frequency-domain Meyer normalization used by wd_viafreq and
       the IMRPhenomT/FEW sparse transforms.  The older UCB path omitted this
       conversion and was larger by sqrt(15/8). */
    fac = sqrt(8.0/15.0)/sqrt((double)(Nx/2));
    
    flag = 0;
    x = log2((double)(Nx));
    if(x-floor(x) == 0.0) flag = 1;
    //printf("%d %f %d %d\n", flag, x, Nt, kw);
    
    alpha = (8.0/(double)(Nt));
    
    tukey(data, alpha, Nx);
    
    if(flag == 0)
    {
        work = gsl_fft_real_workspace_alloc (Nx);
        real = gsl_fft_real_wavetable_alloc (Nx);
        gsl_fft_real_transform (data, 1, Nx, real, work);
        gsl_fft_real_wavetable_free (real);
        gsl_fft_real_workspace_free (work);
    }
    else
    {
        // Radix 2
        gsl_fft_real_radix2_transform(data, 1, Nx);
    }
    
    // for i > 0
    // GSL radix 2 packs Real i, Imaginary Nx-i
    // GSL general radix packs Real 2*i-1 Imaginary 2*i

    
    DX = double_vector(2*Nt);
    
     for(m=1; m< Nfx; m++)
     {
         mt = m+kx;
         
        for(j=-Nt/2; j< Nt/2; j++)
        {
            i = j+Nt/2;
            
            REAL(DX,i) = 0.0;
            IMAG(DX,i) = 0.0;
            
            jj = j + m*Nt/2;
            
            if(jj > 0 && jj < Nx/2)
            {
               
                if(flag == 0)
                {
                    // mixed radix
                    REAL(DX,i) = data[2*jj-1]*phihf[abs(j)];
                    IMAG(DX,i) = data[2*jj]*phihf[abs(j)];
                }
                else
                {
                    // radix 2
                    REAL(DX,i) = data[jj]*phihf[abs(j)];
                    IMAG(DX,i) = data[Nx-jj]*phihf[abs(j)];
                }
                
            }
            
                
        }
         
         gsl_fft_complex_radix2_backward(DX, 1, Nt);
         
        
         for(n=0; n < Nt; n++)
         {
             
             if(mt%2 == 0)
             {
                 
                 if((n+mt)%2 ==0)
                 {
                      wdmout[n][m-1] = fac*REAL(DX,n);
                      wdmoutq[n][m-1] = -fac*IMAG(DX,n);
                 }
                 else
                 {
                    wdmout[n][m-1] = fac*IMAG(DX,n);
                    wdmoutq[n][m-1] = fac*REAL(DX,n);
                 }
                 
             }
             else
             {
                 if((n+mt)%2 ==0)
                 {
                      wdmout[n][m-1] = fac*REAL(DX,n);
                      wdmoutq[n][m-1] = fac*IMAG(DX,n);
                 }
                 else
                 {
                    wdmout[n][m-1] = -fac*IMAG(DX,n);
                    wdmoutq[n][m-1] = fac*REAL(DX,n);
                 }
                 
             }
             
            
         }
         
       }
    
    free(DX);
    
    
}





void wdmtranT(int K, int kx, double *phih, double *data, double *wdmout)
{
    int i, j, jj, mm;
    int Nx, Nfx, ND;
    double *wdata;
    
    // windowed data packets
    wdata = (double*)malloc(sizeof(double)* (K));
    
    Nx = Nt*4;
    Nfx = 4;
    
    mm = 0;
    
    for(i=0; i< Nt; i++)
     {
         
        for(j=0; j< K; j++)
        {
            jj = i*Nfx-K/2+j;
            if(jj < 0) jj += Nx;  // periodically wrap the data
            if(jj >= Nx) jj -= Nx; // periodically wrap the data
            wdata[j] = data[jj]*phih[j];  // apply the window
        }
         
        gsl_fft_real_radix2_transform(wdata, 1, K);
         
         for(j=1; j<= 2; j++)
         {
             jj = kx-1+j;
            // printf("%d %d\n", i, j);
             if((i+jj)%2 ==0)
             {
                 wdmout[mm] = wdata[j*mult];
             }
             else
             {
                wdmout[mm] = -wdata[K-j*mult];
             }
             
             // not sure about this hack
             if((i+1)%2 == 0) wdmout[mm] *= -1.0;
             
             mm++;
         }
         
       
        
         
       }
    
    free(wdata);
    
}







/*
void wavelist(double df, double DF, double HBW, double *freqX, double *freqY, double *freqZ, int *list, int *kmin, int *kmax)
{
    
    int j, k, mm;
    double fx, fy, fz;
    double fmax, fmin;
    
    mm = 0;
    
    for(j=0; j< Nt; j++)
    {
        
        
        fx = freqX[j];
        fy = freqY[j];
        fz = freqZ[j];
        
        // find the largest and smallest frequencies
        fmin = fx;
        fmax = fx;
        if(fy < fmin) fmin = fy;
        if(fy > fmax) fmax = fy;
        if(fz < fmin) fmin = fz;
        if(fz > fmax) fmax = fz;
        
        // lowest frequency layer
        kmin[j] = (int)(ceil((fmin-HBW)/DF));
        
        // highest frequency layer
        kmax[j] = (int)(floor((fmax+HBW)/DF));
        
        for(k=kmin[j]; k<= kmax[j]; k++)
        {
        list[mm] = j+k*Nt;
        mm++;
        }  // end loop over frequency layers
        
    }
    
}
 */

void wavelist(double df, double DF, int *Nfsam, double *fd, double *freqX, double *freqY, double *freqZ, double *fdotX, double *fdotY, double *fdotZ, double *waveX, double *waveY, double *waveZ, int *list, int *rlist, int *NM, int *kmin, int *kmax)
{
    
    int i, j, k, ii, jj, kk, n, mm, mx;
    int flag;
    double dx, dy;
    double phase, f, fdotmax, fdotmin, fmid, fsam, A;
    double c, s, x, y, z, yy, zz;
    double fmx, fdmx, fdmn, dfd, HBW;
    int flagx, flagy, flagz;
    double fx, fy, fz;
    double fmax, fmin;
    int NH, NL;
    
     
    fmx = (double)(Nf-1)*DF; // maximum frequency
    fdmx = fd[Nfd-1]; // maximum frequency derivative
    fdmn = fd[0]; // minimum frequency derivative
    
    dfd = fd[1]-fd[0]; // f-dot increment
    
    
    mm = 0;

        for(j=0; j< Nt; j++)
        {
            
            fx = freqX[j];
            fy = freqY[j];
            fz = freqZ[j];
            
            // check to see if any of the channels is ok
            flagx = flagy = flagz = 0;
            if(fx < fmx) flagx = 1;
            if(fy < fmx) flagy = 1;
            if(fz < fmx) flagz = 1;
            
            // kill any channel that does not have valid fdots
            if(fdotX[j] < fdmn || fdotX[j] > fdmx) flagx = 0;
            if(fdotY[j] < fdmn || fdotY[j] > fdmx) flagy = 0;
            if(fdotZ[j] < fdmn || fdotZ[j] > fdmx) flagz = 0;
            
           // printf("%d %d %d %d\n", j, flagx, flagy, flagz);
            
            // if any of the channels has valid values continue
            if(flagx == 1 || flagy == 1 || flagz == 1)
            {
                
                // find the largest and smallest frequencies and frequency derivatives
                // but only use the valid channels
                
                if(flagx == 1) // start by assigning to X if valid
                {
                    fmin = fx;
                    fmax = fx;
                    fdotmax = fdotX[j];
                    fdotmin = fdotX[j];
                }
                else if (flagy == 1) // otherwise Y
                {
                    fmin = fy;
                    fmax = fy;
                    fdotmax = fdotY[j];
                    fdotmin = fdotY[j];
                }
                else // or if those not ok, Z
                {
                    fmin = fz;
                    fmax = fz;
                    fdotmax = fdotZ[j];
                    fdotmin = fdotZ[j];
                }
                
                if(flagy == 1) // see if Y sets the boundaries
                {
                    if(fy < fmin) fmin = fy;
                    if(fy > fmax) fmax = fy;
                    if(fdotY[j] < fdotmin) fdotmin = fdotY[j];
                    if(fdotY[j] > fdotmax) fdotmax = fdotY[j];
                }
                
                if(flagz == 1)
                {
                    if(fz < fmin) fmin = fz;
                    if(fz > fmax) fmax = fz;
                    if(fdotZ[j] < fdotmin) fdotmin = fdotZ[j];
                    if(fdotZ[j] > fdotmax) fdotmax = fdotZ[j];
                }
                
                
                if(fdotmax < fdmx && fdotmin > fdmn)
                {
                    
                    // lowest f-dot layer
                    n = (int)(floor((fdotmin-fdmn)/dfd));
                    NL = Nfsam[n];
                    
                    // highest f-dot layer
                    n = (int)(floor((fdotmax-fdmn)/dfd));
                    NH = Nfsam[n];
                    
                    // find which has the largest number of samples
                    if(NL > NH) NH = NL;
                    
                    HBW  = 0.5*(double)(NH-1)*df;
                    
                    // lowest frequency layer
                    kmin[j] = (int)(ceil((fmin-HBW)/DF));
                    
                    // highest frequency layer
                    kmax[j] = (int)(floor((fmax+HBW)/DF));
                    
                    if(kmin[j] < 0) kmin[j] = 0;
                    if(kmax[j] > Nf-1) kmax[j] = Nf-1;
                    
                    //printf("%d %d %d\n", j, kmin[j], kmax[j]);
                    
                    
                    for(k=kmin[j]; k<= kmax[j]; k++)
                    {
                        
                        list[mm] = j+k*Nt;    // pixel reference
                        rlist[j+k*Nt] = mm;   // reverse lookup
                        waveX[mm] = waveY[mm] = waveZ[mm] = 0.0;
                        
                        //printf("%d %d %d\n", j, k, mm);
                        
                        mm++;
                        
                    }  // end loop over frequency layers
                    
                }
                
            }
            
            
        }
    
    *NM = mm;
    
}


/*

void wavemake(double df, double DF, double HBW, double *Phase, double *freq, double *Amp, double **lookup, int *kmin, int *kmax, double *wave)
{
    
    int i, j, k, ii, jj, kk, n, mm, mx;
    double dx;
    double phase, f, fmid, fsam, A;
    double c, s, x, y, z;
    int flag;
    
       mm = 0;

        for(j=0; j< Nt; j++)
        {
    
           phase = Phase[j];
           f = freq[j];
           A = Amp[j];
            
            c = A*cos(phase);
            s = A*sin(phase);
            
            for(k=kmin[j]; k<= kmax[j]; k++)
            {
                
                // central frequency
                fmid = (double)(k)*DF;
                
                x = (f-(fmid+0.5*df))/df;
                ii = 1;
                if(x < 0.0)  ii = -1;
                kk = ii*(int)(floor(fabs(x)));
                fsam = fmid+((double)(kk)+0.5)*df;
                dx = (f-fsam)/df; // used for linear interpolation
                
                // interpolate over frequency
                jj = kk+Nsf/2;
                
                //printf("%d %d %d %d %e %e %e\n", j, k, kk, jj, f, fmid, dx);
                y = 0.0;
                z = 0.0;
                if(jj >= 0 && jj+1 < Nsf)
                {
                    y = (1.0-dx)*lookup[jj][0]+dx*lookup[jj+1][0];
                    z = (1.0-dx)*lookup[jj][1]+dx*lookup[jj+1][1];
                }
                
                if((j+k)%2 == 0)
                {
                    wave[mm] = (c*y-s*z);
                }
                else
                {
                    wave[mm] = -(c*z+s*y);
                }
                
                mm++;
                
            }  // end loop over frequency layers
            
        }
    
}
 
*/

void wavemake(double df, double DF, int *Nfsam, double *fd, double *Phase, double *freq, double *freqd, double *Amp, double ***lookup, int *kmin, int *kmax, int *rlist, double *wave, double *wavef)
{
    
    int i, j, k,  ii, jj, kk, n, mm, mx;
    double dx, dy;
    double phase, f, fdot, fmid, fsam, A;
    double fdmx, fdmn;
    double c, s, x, y, z, yy, zz;
    double fmx, dfd, HBW;
    int flag;
    
   fmx = (double)(Nf-1)*DF; // maximum frequency
   fdmx = fd[Nfd-1]; // maximum frequency derivative
   fdmn = fd[0]; // minimum frequency derivative
   
   dfd = fd[1]-fd[0]; // f-dot increment
    

        for(j=0; j< Nt; j++)
        {
    
           phase = Phase[j];
           f = freq[j];
           fdot = freqd[j];
           A = Amp[j];
            
           if(f < fmx && fdot < fdmx && fdot > fdmn)
           {
            
           // lower f-dot layer
           n = (int)(floor((fdot-fdmn)/dfd));
               
           dy = (fdot-fdmn)/dfd-(double)(n);
               
            HBW  = 0.5*(double)(Nfsam[n]-1)*df;
         
           c = A*cos(phase);
           s = A*sin(phase);
               
            for(k=kmin[j]; k<= kmax[j]; k++)
            {
            
            // central frequency
            fmid = (double)(k)*DF;
                
            kk = (int)(floor((f-(fmid+0.5*df))/df));
            fsam = fmid+((double)(kk)+0.5)*df;
            dx = (f-fsam)/df; // used for linear interpolation
            //printf("%d %d %d %d %d %e %f %f\n", j, k, kk, kk+Nfsam[n]/2, Nfsam[n], fsam, dx, dy);
                
            // interpolate over frequency
            jj = kk+Nfsam[n]/2;
            //printf("%d %d %d %d\n", j, k, n, jj);
                y = 0.0;
                z = 0.0;
                yy = 0.0;
                zz = 0.0;
            if(jj >=0 && jj < Nfsam[n]-1)
            {
                y = (1.0-dx)*lookup[n][jj][0]+dx*lookup[n][jj+1][0];
                z = (1.0-dx)*lookup[n][jj][1]+dx*lookup[n][jj+1][1];
            }
            jj = kk+Nfsam[n+1]/2;
            if(jj >=0 && jj < Nfsam[n]-1)
            {
                yy = (1.0-dx)*lookup[n+1][jj][0]+dx*lookup[n+1][jj+1][0];
                zz = (1.0-dx)*lookup[n+1][jj][1]+dx*lookup[n+1][jj+1][1];
            }
                
            // interpolate over fdot
            y = (1.0-dy)*y+dy*yy;
            z = (1.0-dy)*z+dy*zz;
                
            mm = rlist[j+k*Nt];
                
                
            if((j+k)%2 == 0)
            {
                wave[mm] = (c*y-s*z);
                wavef[mm] = -(s*y+c*z);
            }
            else
            {
                wave[mm] = -(c*z+s*y);
                wavef[mm] = s*z-c*y;
            }
           
           
                
            }  // end loop over frequency layers
            
        }
            
        }
    
   // printf("%d\n", mm);
    
}





void wavemakeold(double df, double DF, double HBW, double *Phase, double *freq, double *Amp, double **lookup, int *list, double *wave)
{
    
    int i, j, k, ii, jj, kk, n, mm, mx;
    int kmin, kmax;
    double dx;
    double phase, f, fmid, fsam, A;
    double c, s, x, y, z;
    int flag;
    
       mm = 0;

        for(j=0; j< Nt; j++)
        {
    
           phase = Phase[j];
           f = freq[j];
           A = Amp[j];
            
            c = A*cos(phase);
            s = A*sin(phase);
            
           // central frequency layer
           k = (int)(rint(f/DF));
            
            // lowest frequency layer
            kmin = (int)(ceil((f-HBW)/DF));
                          
            // highest frequency layer
            kmax = (int)(floor((f+HBW)/DF));
            
            
            //printf("%d %d %d\n", kmin, kmax, kmax-kmin);
            
            for(k=kmin; k<= kmax; k++)
            {
            
            // central frequency
            fmid = (double)(k)*DF;
                
            x = (f-(fmid+0.5*df))/df;
            ii = 1;
            if(x < 0.0)  ii = -1;
            kk = ii*(int)(floor(fabs(x)));
            fsam = fmid+((double)(kk)+0.5)*df;
            dx = (f-fsam)/df; // used for linear interpolation
                
            // interpolate over frequency
            jj = kk+Nsf/2;
                
            //printf("%d %d %d %d %e %e %e\n", j, k, kk, jj, f, fmid, dx);
            y = (1.0-dx)*lookup[jj][0]+dx*lookup[jj+1][0];
            z = (1.0-dx)*lookup[jj][1]+dx*lookup[jj+1][1];
            
            if((j+k)%2 == 0)
            {
                wave[mm] = (c*y-s*z);
            }
            else
            {
                wave[mm] = -(c*z+s*y);
            }
                
                
                
                
            list[mm] = j+k*Nt;
            mm++;
                
            }  // end loop over frequency layers
            
        }
    
   // printf("%d\n", mm);
    
}

/*
void extractAP(int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR)
{
    int i;
    double u, v;
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = sqrt(M[i]*M[i]+Mf[i]*Mf[i]);
        u = floor(phiR[i]/(2.0*M_PI));
        v = phiR[i]-u*(2.0*M_PI);  // reference phase mod 2pi
        Dphi[i] = -atan2(Mf[i],M[i])-v;
    }
    
}
*/

void extractAP(int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR)
{

    int i;
    double u, v;
    double dA1, dA2, dA3;
    
    double *flip, *pjump;
    
    flip = double_vector(Ns);
    pjump = double_vector(Ns);
    
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = sqrt(M[i]*M[i]+Mf[i]*Mf[i]);
    }
    
    // This catches sign flips in the amplitude. Can't catch flips at either end of array
    flip[0] = 1.0;
    pjump[0] = 0.0;

    i = 1;
    do
    {
        flip[i] = flip[i-1];
        pjump[i] = pjump[i-1];
        
        //local min
        if((As[i] < As[i-1]) && (As[i] < As[i+1]))
        {
            dA1 = As[i+1]+As[i-1]-2.0*As[i];  // regular second derivative
            dA2 = -As[i+1]+As[i-1]-2.0*As[i];  // second derivative if i+1 first negative value
            dA3 = -As[i+1]+As[i-1]+2.0*As[i];  // second derivative if i first negative value
            //printf("%d %e %e\n", i, fabs(dA2/dA1), fabs(dA3/dA1));
            if(fabs(dA2/dA1) < 0.1)
            {
               // printf("%d\n", i+1);
                flip[i+1] = -1.0*flip[i];
                pjump[i+1] = pjump[i]+M_PI;
                i++; // skip an extra place since i+1 already dealt with
            }
            if(fabs(dA3/dA1) < 0.1)
            {
                //printf("%d\n", i);
                flip[i] = -1.0*flip[i-1];
                pjump[i] = pjump[i-1]+M_PI;
            }
        }
        
        i++;
        
    }while(i < Ns-1);
    
    flip[Ns-1] = flip[Ns-2];
    pjump[Ns-1] = pjump[Ns-2];
    
    
    for(i=0; i<Ns ;i++)
    {
        As[i] = flip[i]*As[i];
        v = remainder(phiR[i], 2.0*M_PI);
        Dphi[i] = -atan2(Mf[i],M[i])+pjump[i]-v;
    }
    
    free(flip);
    free(pjump);
    
    
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


typedef struct
{
    gsl_interp_accel *amplitude_acc, *phase_acc;
    gsl_spline *amplitude_spline, *phase_spline;
    double aplus, across, cos2psi, sin2psi;
} GBTDI2Waveform;

static int gb_tdi2_polarizations(void *userdata, double source_time,
                                 double complex *hplus, double complex *hcross)
{
    GBTDI2Waveform *waveform = userdata;
    double hp, hc, hpf, hcf;
    if(source_time < waveform->amplitude_spline->x[0] ||
       source_time > waveform->amplitude_spline->x[
           waveform->amplitude_spline->size-1]) return 1;
    UCB_ampphase(source_time, waveform->amplitude_acc,
                 waveform->amplitude_spline, waveform->phase_acc,
                 waveform->phase_spline, waveform->aplus, waveform->across,
                 waveform->cos2psi, waveform->sin2psi,
                 &hp, &hc, &hpf, &hcf);
    *hplus = hp-I*hpf;
    *hcross = hc-I*hcf;
    return 0;
}

void fast_response(double *tarray, int N, double *params, gsl_interp_accel **SLacc, gsl_spline **SLspline, gsl_interp_accel **SPacc, gsl_spline **SPspline, gsl_interp_accel **SVacc, gsl_spline **SVspline, gsl_interp_accel *ARacc, gsl_spline *ARspline, gsl_interp_accel *PRacc, gsl_spline *PRspline, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf)
{
    if(use_tdi2)
    {
        GBTDI2Waveform waveform;
        int n;
        (void)SLacc; (void)SLspline; (void)SVacc; (void)SVspline;
        waveform.amplitude_acc = ARacc;
        waveform.amplitude_spline = ARspline;
        waveform.phase_acc = PRacc;
        waveform.phase_spline = PRspline;
        waveform.aplus = 0.5*(1.0+params[4]*params[4]);
        waveform.across = -params[4];
        waveform.cos2psi = cos(2.0*params[5]);
        waveform.sin2psi = sin(2.0*params[5]);
        for(n=0; n<N; n++)
        {
            double complex response[3];
            if(gb_tdi2_sample(tarray[n], params[1], params[2], SPacc,
                              SPspline, gb_tdi2_polarizations, &waveform,
                              response) != 0)
            {
                fprintf(stderr, "TDI-2 response failed at t=%.12e\n", tarray[n]);
                exit(1);
            }
            X[n] = creal(response[0]); Xf[n] = -cimag(response[0]);
            Y[n] = creal(response[1]); Yf[n] = -cimag(response[1]);
            Z[n] = creal(response[2]); Zf[n] = -cimag(response[2]);
        }
        return;
    }
    
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

    costh = params[1];      // costh
    phi = params[2];      // phi
    

    cosi = params[4];    // cosi
    
    // conventions are flipped relative to BH code
    psi =params[5];   // psi
    
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
          
          for(i=0;i<3;i++) 
          {
              Larm[i] = gsl_spline_eval(SLspline[i], t, SLacc[i]);
          }
          
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
          TDI_spline(X, Xf, 0, 1, 2, tarray, n, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kr, Larm);
          TDI_spline(Y, Yf, 1, 2, 0, tarray, n, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kr, Larm);
          TDI_spline(Z, Zf, 2, 0, 1, tarray, n, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, App, Apm, Acp, Acm, kr, Larm);

          
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

    costh = params[1];      // costh
    phi = params[2];      // phi
   
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


static int gb_dense_polarizations(void *userdata, double source_time,
                                  double complex *hplus, double complex *hcross)
{
    const double *params = userdata;
    double hp, hc, hpf, hcf;
    double cosi = params[4];
    ampphase(source_time, params[3], 0.5*(1.0+cosi*cosi), -cosi,
             cos(2.0*params[5]), sin(2.0*params[5]),
             params[6], params[0], params[7], params[8],
             &hp, &hc, &hpf, &hcf);
    *hplus = hp-I*hpf;
    *hcross = hc-I*hcf;
    return 0;
}

static int gb_dense_tdi2_response(double *tarray, int N, double *params,
                                  double *X, double *Y, double *Z,
                                  double *Xf, double *Yf, double *Zf)
{
    int i, j, n, ns, status = 1;
    double first, last, pad, x[3], y[3], z[3], *knots = NULL, *position = NULL;
    gsl_interp_accel *acc[9] = {NULL};
    gsl_spline *spline[9] = {NULL};
    if(N < 2) return 1;
    ns = (int)(200.0*(tarray[N-1]-tarray[0])/SECSYR);
    if(ns < 20) ns = 20;
    pad = fmax(1000.0, (tarray[N-1]-tarray[0])/(ns-1));
    first = tarray[0]-pad;
    last = tarray[N-1]+pad;
    knots = malloc((size_t)ns*sizeof(*knots));
    position = malloc((size_t)9*ns*sizeof(*position));
    if(knots == NULL || position == NULL) goto cleanup;
    for(i=0; i<ns; i++)
    {
        knots[i] = first+(last-first)*i/(ns-1);
        spacecraft(knots[i], x, y, z);
        for(j=0; j<3; j++)
        {
            position[(3*j+0)*ns+i] = x[j]/CLIGHT;
            position[(3*j+1)*ns+i] = y[j]/CLIGHT;
            position[(3*j+2)*ns+i] = z[j]/CLIGHT;
        }
    }
    for(j=0; j<9; j++)
    {
        acc[j] = gsl_interp_accel_alloc();
        spline[j] = gsl_spline_alloc(gsl_interp_cspline, ns);
        if(acc[j] == NULL || spline[j] == NULL ||
           gsl_spline_init(spline[j], knots, position+j*ns, ns) != 0)
            goto cleanup;
    }
    for(n=0; n<N; n++)
    {
        double complex response[3];
        if(gb_tdi2_sample(tarray[n], params[1], params[2], acc, spline,
                          gb_dense_polarizations, params, response) != 0)
            goto cleanup;
        X[n] = creal(response[0]); Xf[n] = -cimag(response[0]);
        Y[n] = creal(response[1]); Yf[n] = -cimag(response[1]);
        Z[n] = creal(response[2]); Zf[n] = -cimag(response[2]);
    }
    status = 0;
cleanup:
    for(j=0; j<9; j++)
    {
        gsl_interp_accel_free(acc[j]);
        gsl_spline_free(spline[j]);
    }
    free(position);
    free(knots);
    return status;
}

void full_response(double *tarray, int N, double *params, double *X, double *Y, double *Z, double *Xf, double *Yf, double *Zf)
{
    if(use_tdi2)
    {
        if(gb_dense_tdi2_response(tarray, N, params, X, Y, Z,
                                  Xf, Yf, Zf) != 0)
        {
            fprintf(stderr, "dense TDI-2 response failed\n");
            exit(1);
        }
        return;
    }
    
    /*   Indicies    */
    int i,j, k, n, m, a, M;
    
    /*   Gravitational Wave basis vectors   */
    double *u,*v,*kv;
    
    /*   Polarization basis tensors   */
    double **eplus, **ecross;
    
    /*   Spacecraft position and separation vector   */
    double *x, *y, *z;
    double *kr, *kn;
    double **nvec;  // n1, n2, n3 arm vectors
    double **rvec;  // spacecraft position vectors
    double *Larm;  // armlengths
    double *plus, *cross;
    
    double phi, cosi, psi;
    double costh, sinth, cosph, sinph;
    double cos2psi, sin2psi;
    double f, t, f0, fdot, fddot, Amp, phi0;
    double AA, Aplus, Across;
    double *App, *Apm, *Acp, *Acm;
    
    double hp, hc;

 
    /*   Allocating Arrays   */
    
    u = double_vector(3); v = double_vector(3); kv = double_vector(3);
    
    eplus  = double_matrix(3,3); ecross = double_matrix(3,3);
    
    nvec  = double_matrix(3,3); rvec = double_matrix(3,3);
    
    x = double_vector(3); y = double_vector(3); z = double_vector(3);
    
    kr = double_vector(3); kn = double_vector(3);
    
    plus = double_vector(3); cross = double_vector(3);
    
    App = double_vector(3); Apm = double_vector(3);
    Acp = double_vector(3); Acm = double_vector(3);
    
    
    Larm = double_vector(3);
    
    f0 = params[0];
    costh = params[1];      // costh
    phi = params[2];      // phi
    Amp = params[3]; // Amp
    

    cosi = params[4];    // cosi
    

    psi = params[5];   // psi
    
    phi0 = params[6];  // phi0
    fdot = params[7];  // fdot
    fddot = params[8];     // fddot
    
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
          
          spacecraft(tarray[n], x, y, z);
          
          // position vectors for each spacecraft
          for(i=0;i<3;i++)
          {
              rvec[i][0]  = x[i];
              rvec[i][1]  = y[i];
              rvec[i][2]  = z[i];
          }
          
          // k dot r_i
          for(i=0;i<3;i++)
          {
              kr[i] = 0.0;
              for(j=0;j<3;j++)
              {
                  kr[i] += rvec[i][j]*kv[j];
              }
          }
          
          // arm vectors
          for(i=0;i<3;i++)
          {
              nvec[0][i] = rvec[1][i]-rvec[2][i];
              nvec[1][i] = rvec[2][i]-rvec[0][i];
              nvec[2][i] = rvec[0][i]-rvec[1][i];
          }
          
          // arm lengths
          for(i=0;i<3;i++)
          {
              Larm[i] = 0.0;
              for(j=0;j<3;j++)
              {
                  Larm[i] += nvec[i][j]*nvec[i][j];
              }
              Larm[i] = sqrt(Larm[i]);
          }
          
          // turn n_i into unit vectors
          for(i=0;i<3;i++)
          {
              for(j=0;j<3;j++)
              {
                  nvec[i][j] /= Larm[i];
              }
          }
          
          // k dot n_i
          for(i=0;i<3;i++)
          {
              kn[i] = 0.0;
              for(j=0;j<3;j++)
              {
                  kn[i] += nvec[i][j]*kv[j];
              }
          }
          
          // convert to seconds
          for(i=0;i<3;i++)
          {
              Larm[i] /= CLIGHT;
              kr[i]  /= CLIGHT;
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
                      plus[i]  += (nvec[i][j]*nvec[i][k])*eplus[j][k];
                      cross[i] += (nvec[i][j]*nvec[i][k])*ecross[j][k];
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
          TDI(X, Xf, 0, 1, 2, tarray, n, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, App, Apm, Acp, Acm, kr, Larm);
          TDI(Y, Yf, 1, 2, 0, tarray, n, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, App, Apm, Acp, Acm, kr, Larm);
          TDI(Z, Zf, 2, 0, 1, tarray, n, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, App, Apm, Acp, Acm, kr, Larm);
          
          
      }
    
    
    free_double_vector(Larm);
    
    free_double_vector(u); free_double_vector(v); free_double_vector(kv);
        
    free_double_matrix(nvec,3);  free_double_matrix(rvec,3);
    
    free_double_matrix(eplus,3); free_double_matrix(ecross,3);

    free_double_vector(x); free_double_vector(y); free_double_vector(z);
    
    free_double_vector(kr); free_double_vector(kn);
    
    free_double_vector(plus); free_double_vector(cross);
    free_double_vector(App); free_double_vector(Apm);
    free_double_vector(Acp); free_double_vector(Acm);
    
    return;
}

void TDI_spline(double *M, double *Mf, int a, int b, int c, double* tarray, int n, gsl_interp_accel *ARacc, gsl_spline *ARspline, gsl_interp_accel *PRacc, gsl_spline *PRspline, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    double t, hp, hc, hpf, hcf;
    
    char name[1024];

              M[n] = 0.0;
              Mf[n] = 0.0;
              
              t = tarray[n] - kr[a]-2.0*Larm[c]-2.0*Larm[b];
              UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[c]+hc*Acp[c];
              M[n] -= hp*Apm[b]+hc*Acm[b];
              Mf[n] += hpf*App[c]+hcf*Acp[c];
              Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
              
              t = tarray[n] - kr[b]-Larm[c]-2.0*Larm[b];
              UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*App[c]+hc*Acp[c];
              M[n] += hp*Apm[c]+hc*Acm[c];
              Mf[n] -= hpf*App[c]+hcf*Acp[c];
              Mf[n] += hpf*Apm[c]+hcf*Acm[c];
              
              t = tarray[n] - kr[c]-Larm[b]-2.0*Larm[c];
              UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*Apm[b]+hc*Acm[b];
              M[n] -= hp*App[b]+hc*Acp[b];
              Mf[n] += hpf*Apm[b]+hcf*Acm[b];
              Mf[n] -= hpf*App[b]+hcf*Acp[b];
              
              t = tarray[n] - kr[a]-2.0*Larm[b];
             UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*Apm[c]+hc*Acm[c];
              M[n] += hp*Apm[b]+hc*Acm[b];
              Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
              Mf[n] += hpf*Apm[b]+hcf*Acm[b];
              
              t = tarray[n] - kr[a]-2.0*Larm[c];
              UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[b]+hc*Acp[b];
              M[n] -= hp*App[c]+hc*Acp[c];
              Mf[n] += hpf*App[b]+hcf*Acp[b];
              Mf[n] -= hpf*App[c]+hcf*Acp[c];
              
              t = tarray[n] - kr[c]- Larm[b];
              UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*Apm[b]+hc*Acm[b];
              M[n] += hp*App[b]+hc*Acp[b];
              Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
              Mf[n] += hpf*App[b]+hcf*Acp[b];
              
              t = tarray[n] - kr[b]- Larm[c];
              UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[c]+hc*Acp[c];
              M[n] -= hp*Apm[c]+hc*Acm[c];
              Mf[n] += hpf*App[c]+hcf*Acp[c];
              Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
              
              t = tarray[n] - kr[a];
              UCB_ampphase(t, ARacc, ARspline, PRacc, PRspline, Aplus, Across, cos2psi, sin2psi, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*App[b]+hc*Acp[b];
              M[n] += hp*Apm[c]+hc*Acm[c];
              Mf[n] -= hpf*App[b]+hcf*Acp[b];
              Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    
    
}


void TDI(double *M, double *Mf, int a, int b, int c, double* tarray, int n, double Amp, double Aplus, double Across, double cos2psi, double sin2psi, double phi0, double f0, double fdot, double fddot, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm)
{
    double t, hp, hc, hpf, hcf;
    
              M[n] = 0.0;
              Mf[n] = 0.0;
              
              t = tarray[n] - kr[a]-2.0*Larm[c]-2.0*Larm[b];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[c]+hc*Acp[c];
              M[n] -= hp*Apm[b]+hc*Acm[b];
              Mf[n] += hpf*App[c]+hcf*Acp[c];
              Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
              
              t = tarray[n] - kr[b]-Larm[c]-2.0*Larm[b];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*App[c]+hc*Acp[c];
              M[n] += hp*Apm[c]+hc*Acm[c];
              Mf[n] -= hpf*App[c]+hcf*Acp[c];
              Mf[n] += hpf*Apm[c]+hcf*Acm[c];
              
              t = tarray[n] - kr[c]-Larm[b]-2.0*Larm[c];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] += hp*Apm[b]+hc*Acm[b];
              M[n] -= hp*App[b]+hc*Acp[b];
              Mf[n] += hpf*Apm[b]+hcf*Acm[b];
              Mf[n] -= hpf*App[b]+hcf*Acp[b];
              
              t = tarray[n] - kr[a]-2.0*Larm[b];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*Apm[c]+hc*Acm[c];
              M[n] += hp*Apm[b]+hc*Acm[b];
              Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
              Mf[n] += hpf*Apm[b]+hcf*Acm[b];
              
              t = tarray[n] - kr[a]-2.0*Larm[c];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[b]+hc*Acp[b];
              M[n] -= hp*App[c]+hc*Acp[c];
              Mf[n] += hpf*App[b]+hcf*Acp[b];
              Mf[n] -= hpf*App[c]+hcf*Acp[c];
              
              t = tarray[n] - kr[c]- Larm[b];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*Apm[b]+hc*Acm[b];
              M[n] += hp*App[b]+hc*Acp[b];
              Mf[n] -= hpf*Apm[b]+hcf*Acm[b];
              Mf[n] += hpf*App[b]+hcf*Acp[b];
              
              t = tarray[n] - kr[b]- Larm[c];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] += hp*App[c]+hc*Acp[c];
              M[n] -= hp*Apm[c]+hc*Acm[c];
              Mf[n] += hpf*App[c]+hcf*Acp[c];
              Mf[n] -= hpf*Apm[c]+hcf*Acm[c];
              
              t = tarray[n] - kr[a];
              ampphase(t, Amp, Aplus, Across, cos2psi, sin2psi, phi0, f0, fdot, fddot, &hp, &hc, &hpf, &hcf);
              M[n] -= hp*App[b]+hc*Acp[b];
              M[n] += hp*Apm[c]+hc*Acm[c];
              Mf[n] -= hpf*App[b]+hcf*Acp[b];
              Mf[n] += hpf*Apm[c]+hcf*Acm[c];
    
    
}

double phase(double t, double *params)
{
    double phi0, f0, fdot, fddot;
    double phase;
    
    f0 = params[0];
    phi0 = params[6];
    fdot = params[7];
    fddot = params[8];
    
    phase = phi0+2.0*M_PI*(f0*t+0.5*fdot*t*t+1.0/6.0*fddot*t*t*t);
    
    return phase;
    
}

double freq(double t, double *params)
{
    double f0, fdot, fddot;
    double f;
    
    f0 = params[0];
    fdot = params[7];
    fddot = params[8];
    
    f = f0+fdot*t+0.5*fddot*t*t;
    
    return f;
    
}

void UCBwaveform(double *params, int NS, double *times, double *Phase, double *Amp)
{
    int n;
    double t, phi0, f0, fdot, fddot, A;
    
    f0 = params[0];
    A = params[3];
    phi0 = params[6];
    fdot = params[7];
    fddot = params[8];
    
    for(n=0; n< NS; n++)
    {
        t = times[n];
        Phase[n] = phi0+2.0*M_PI*(f0*t+0.5*fdot*t*t+1.0/6.0*fddot*t*t*t);
        Amp[n] =  A*(1.0+2.0/3.0*fdot/f0*t);
    }
    
              
}



void ampphase(double t, double Amp, double Aplus, double Across, double cos2psi, double sin2psi, double phi0, double f0, double fdot, double fddot,  double *hp, double *hc, double *hpf, double *hcf)
{
    double f, phase, A;
    double cp, sp;
    
    f = f0+fdot*t+0.5*fddot*t*t;
    phase = phi0+2.0*M_PI*(f0*t+0.5*fdot*t*t+1.0/6.0*fddot*t*t*t);
   // A = Amp*pow(f/f0,2.0/3.0);
   // A = Amp*(1.0+2.0/3.0*(fdot/f0*t+(f0*fddot-1.0/3.0*fdot*fdot)/(f0*f0)*t*t));
    A = Amp*(1.0+2.0/3.0*fdot/f0*t);
    cp = cos(phase);
    sp = sin(phase);
    *hp = A*(Aplus*cos2psi*cp+Across*sin2psi*sp);
    *hc = A*(Across*cos2psi*sp-Aplus*sin2psi*cp);
    
    *hpf = A*(-Aplus*cos2psi*sp+Across*sin2psi*cp);
    *hcf = A*(Across*cos2psi*cp+Aplus*sin2psi*sp);
              
}

void UCB_ampphase(double t, gsl_interp_accel *ARacc, gsl_spline *ARspline, gsl_interp_accel *PRacc, gsl_spline *PRspline, double Aplus, double Across, double cos2psi, double sin2psi,  double *hp, double *hc, double *hpf, double *hcf)
{
    double f, phase, A;
    double cp, sp;
    
    phase = gsl_spline_eval(PRspline, t, PRacc);
    f = gsl_spline_eval_deriv(PRspline, t, PRacc)/(2.0*M_PI);
    A = gsl_spline_eval(ARspline, t, ARacc);
    
    cp = cos(phase);
    sp = sin(phase);
    *hp = A*(Aplus*cos2psi*cp+Across*sin2psi*sp);
    *hc = A*(Across*cos2psi*sp-Aplus*sin2psi*cp);
    
    *hpf = A*(-Aplus*cos2psi*sp+Across*sin2psi*cp);
    *hcf = A*(Across*cos2psi*cp+Aplus*sin2psi*sp);
              
}



void spacecraft(double t, double *x, double *y, double *z)
{

  double alpha;
  double beta1, beta2, beta3;
  double sa, sb, ca, cb;
  double si, ci;
 
  alpha = 2.*M_PI*t*fm + kappa0;

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
  
    
    /*
    ci = cos(sq3*ec);
    si = sin(sq3*ec);
    
    spacraft_loc_Kepler(0, x, y, z, ci, si, alpha, beta1);
    spacraft_loc_Kepler(1, x, y, z, ci, si, alpha, beta2);
    spacraft_loc_Kepler(2, x, y, z, ci, si, alpha, beta3);
    */
    
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

void spacraft_loc_Kepler(int i, double *x, double *y, double *z, double ci, double si, double alpha, double beta)
{

      double r, psi, d, sd, cd;
      double gamma, sg, cg;
      double cb, sb;
    
      cb = cos(beta);
      sb = sin(beta);
    
      d = alpha-beta;
      sd = sin(d);
      cd = cos(d);
    
      // 4th order accurate
      gamma = d + 2.0*ec*sd + 5.0/2.0*sd*cd*ec*ec+ ec*ec*ec*(13.0*cd*cd*sd/3.0-4.0*sd/3.0)+ec*ec*ec*ec*sd*(206.0*cd*cd*cd-125.0*cd);
    
      sg = sin(gamma);
      cg = cos(gamma);
    
      r = AU*(1.0-ec*ec)/(1.0+ec*cg);
    
      x[i] = r*(ci*cb*cg-sb*sg);
      y[i] = r*(ci*sb*cg+cb*sg);
      z[i] = -r*si*cg;
    
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
