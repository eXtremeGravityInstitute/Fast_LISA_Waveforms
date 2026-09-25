/* Copyright (C) 2026 Neil Cornish.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "tdi2_response.h"

#include <math.h>
#include <stddef.h>

typedef struct {
    int receiver, emitter, sign, count;
    int delays[7];
} Term;

typedef struct {
    double light_time, derivative;
    double receiver_time, emitter_time;
    double receiver_rate, emitter_rate;
    double plus, cross;
} Link;

static const Term x_terms[16] = {
    {1,3, 1,0,{0}}, {3,1, 1,1,{13}},
    {1,2, 1,2,{13,31}}, {2,1, 1,3,{13,31,12}},
    {1,2,-1,0,{0}}, {2,1,-1,1,{12}},
    {1,3,-1,2,{12,21}}, {3,1,-1,3,{12,21,13}},
    {1,2, 1,4,{13,31,12,21}},
    {2,1, 1,5,{13,31,12,21,12}},
    {1,3, 1,6,{13,31,12,21,12,21}},
    {3,1, 1,7,{13,31,12,21,12,21,13}},
    {1,3,-1,4,{12,21,13,31}},
    {3,1,-1,5,{12,21,13,31,13}},
    {1,2,-1,6,{12,21,13,31,13,31}},
    {2,1,-1,7,{12,21,13,31,13,31,12}}
};

static int position(double time, int spacecraft, gsl_interp_accel **acc,
                    gsl_spline **spline, double value[3], double velocity[3])
{
    int coordinate;
    if(!isfinite(time) || time < spline[0]->x[0] ||
       time > spline[0]->x[spline[0]->size-1]) return 1;
    for(coordinate=0; coordinate<3; coordinate++)
    {
        int index = 3*spacecraft+coordinate;
        value[coordinate] = gsl_spline_eval(spline[index], time, acc[index]);
        if(velocity != NULL)
            velocity[coordinate] = gsl_spline_eval_deriv(spline[index], time,
                                                          acc[index]);
    }
    return 0;
}

static int directed_link(double time, int receiver, int emitter,
                         gsl_interp_accel **acc, gsl_spline **spline,
                         const double propagation[3],
                         const double eplus[3][3], const double ecross[3][3],
                         Link *link)
{
    double xr[3], xe[3], vr[3], ve[3], ve0[3], arm[3];
    double r2=0.0, rv=0.0, v2=0.0, length, projection=0.0;
    double pp=0.0, pc=0.0, numerator=0.0, denominator=1.0;
    int i, j;
    if(position(time, receiver, acc, spline, xr, vr) ||
       position(time, emitter, acc, spline, xe, ve0)) return 1;
    for(i=0; i<3; i++)
    {
        double radius = xr[i]-xe[i];
        r2 += radius*radius;
        rv += radius*ve0[i];
        v2 += ve0[i]*ve0[i];
    }
    if(r2 <= 0.0 || v2 >= 1.0) return 2;
    link->light_time = (rv+sqrt(rv*rv+(1.0-v2)*r2))/(1.0-v2);
    if(position(time-link->light_time, emitter, acc, spline, xe, ve)) return 3;
    length = 0.0;
    for(i=0; i<3; i++) length += (xr[i]-xe[i])*(xr[i]-xe[i]);
    length = sqrt(length);
    if(length <= 0.0) return 4;
    link->receiver_time = time;
    link->emitter_time = time-link->light_time;
    link->receiver_rate = 1.0;
    for(i=0; i<3; i++)
    {
        arm[i] = (xr[i]-xe[i])/length;
        projection += propagation[i]*arm[i];
        link->receiver_time -= propagation[i]*xr[i];
        link->emitter_time -= propagation[i]*xe[i];
        link->receiver_rate -= propagation[i]*vr[i];
        numerator += arm[i]*(vr[i]-ve[i]);
        denominator -= arm[i]*ve[i];
    }
    if(fabs(denominator) < 1.0e-12 || fabs(1.0-projection) < 1.0e-12)
        return 5;
    link->derivative = numerator/denominator;
    link->emitter_rate = 1.0-link->derivative;
    for(i=0; i<3; i++) link->emitter_rate -=
        (1.0-link->derivative)*propagation[i]*ve[i];
    for(i=0; i<3; i++) for(j=0; j<3; j++)
    {
        pp += arm[i]*arm[j]*eplus[i][j];
        pc += arm[i]*arm[j]*ecross[i][j];
    }
    link->plus = -0.5*pp/(1.0-projection);
    link->cross = -0.5*pc/(1.0-projection);
    return 0;
}

int gb_tdi2_sample(double output_time, double sky_costheta, double sky_longitude,
                   gsl_interp_accel **position_acc, gsl_spline **position_spline,
                   GBTDIPolarization polarization, void *userdata,
                   double complex output[3])
{
    Link links[3][3];
    double u[3], v[3], propagation[3], eplus[3][3], ecross[3][3];
    double sinth, cosph, sinph;
    int i, j, channel, term_index;
    if(position_acc == NULL || position_spline == NULL ||
       polarization == NULL || output == NULL ||
       fabs(sky_costheta) > 1.0) return 1;
    sinth = sqrt(fmax(0.0, 1.0-sky_costheta*sky_costheta));
    cosph = cos(sky_longitude);
    sinph = sin(sky_longitude);
    u[0] = -sky_costheta*cosph; u[1] = -sky_costheta*sinph; u[2] = sinth;
    v[0] = sinph; v[1] = -cosph; v[2] = 0.0;
    propagation[0] = -sinth*cosph;
    propagation[1] = -sinth*sinph;
    propagation[2] = -sky_costheta;
    for(i=0; i<3; i++) for(j=0; j<3; j++)
    {
        eplus[i][j] = v[i]*v[j]-u[i]*u[j];
        ecross[i][j] = u[i]*v[j]+v[i]*u[j];
    }
    for(i=0; i<3; i++) for(j=0; j<3; j++) if(i != j)
        if(directed_link(output_time, i, j, position_acc, position_spline,
                         propagation, eplus, ecross, &links[i][j])) return 2;
    for(channel=0; channel<3; channel++)
    {
        output[channel] = 0.0;
        for(term_index=0; term_index<16; term_index++)
        {
            const Term *term = &x_terms[term_index];
            int receiver = (term->receiver-1+channel)%3;
            int emitter = (term->emitter-1+channel)%3;
            const Link *link = &links[receiver][emitter];
            double lag=0.0, jacobian=1.0;
            double complex hpr, hcr, hpe, hce;
            for(j=0; j<term->count; j++)
            {
                int label = term->delays[j];
                int dr = (label/10-1+channel)%3;
                int de = (label%10-1+channel)%3;
                const Link *delay = &links[dr][de];
                lag += delay->light_time-delay->derivative*lag;
                jacobian *= 1.0-delay->derivative;
            }
            if(polarization(userdata,
                            link->receiver_time-link->receiver_rate*lag,
                            &hpr, &hcr) ||
               polarization(userdata,
                            link->emitter_time-link->emitter_rate*lag,
                            &hpe, &hce)) return 3;
            output[channel] += term->sign*jacobian*
                (link->plus*(hpr-hpe)+link->cross*(hcr-hce));
        }
    }
    return 0;
}
