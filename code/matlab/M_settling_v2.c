/*=================================================================
 * make_F_struct.c
 *
 *
 * Call as  
 * 
 * >>  M = M_settling(x, sigma2_tot, lambda, net_input, dt, dx)
 *
 *  OUTPUTS 
 * ----------
 *
 * M: The forward transition matrix, where F(i, j) contains the probability
 *    of ending up in spatial bin j at the next timestep given one is in spatial
 *    bin i at the current timestep.
 *
 *
 * PARAMETERS:
 * -----------
 *
 *  x             Bin centers
 *
 *  sigma2_tot    variance added in one timestep (one use of F)
 *
 *  lambda        inverse of accumulation time constant, in units of 1/sec
 *
 *  net_input     Total input to accumulator on this timestep -- divide by
 *                dt before passing it in.
 *
 *  dt            Timestep in secs
 * 
 *  dx            bin width

 *=================================================================*/

/* written by BWB Aug 2011 
   modified by CDB Apr 2014 
   adapted by BDD July 2017*/
#include "mex.h"
#include "matrix.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
   
   if ( nrhs!=6 ) mexErrMsgTxt("Need to have 6 input arguments");

   int i, j, k;
   int N;
   double *x, dx;
   double sigma2;
   double lambda;
   double h;
   double dt;
   double *F;
   int starting_j, ending_j;
   int ndeltas;
    
   /* inputs */
   N  = mxGetNumberOfElements(prhs[0]);
     
   x      = mxGetData(prhs[0]);
   sigma2 = mxGetScalar(prhs[1]);
   lambda = mxGetScalar(prhs[2]);
   h      = mxGetScalar(prhs[3]);
   dt     = mxGetScalar(prhs[4]);
   dx     = mxGetScalar(prhs[5]);
   
   plhs[0] = mxCreateDoubleMatrix(N, N, mxREAL);
   F = mxGetPr(plhs[0]);
      
   /* First column will have a 1 and then all zeros; last column will be all zeros and then a 1*/
   F[0] = 1;
   F[N*N-1] = 1;
   
   /* When we compute F matrix, we'll skip the first (zeroth) and last (N-1) columns */
   starting_j  = 1;
   ending_j    = N-2;

   /* define the slices of a gaussian with sigma2 variance */
   ndeltas = ceil(10*sqrt(sigma2)/dx); 
   if (ndeltas<70) {ndeltas = 70;}
   
   double deltas[2*ndeltas+1];
   double ps[2*ndeltas+1];
   if (sigma2 == 0) {
       ndeltas = 1;
       deltas[0] = 0;
       ps[0] = 1;
   }
   else {
       for (i=-ndeltas; i<=ndeltas; i++){ 
           deltas[i+ndeltas]  = i * (5*sqrt(sigma2))/ndeltas;
       } 
       ndeltas = ndeltas*2+1;
       double ps_sum=0;
       for (i=0; i<ndeltas; i++){
           ps[i] = exp(-pow(deltas[i],2)/(2*sigma2));
           ps_sum += ps[i];
       }
       for (i=0; i<ndeltas; i++){ ps[i] = ps[i]/ps_sum; } 
   }
   
   /* construct F and its derivatives, stepping through the bins j where we came from: */
   double mu;
   int hp, lp;
   for (j=starting_j; j<=ending_j; j++) {
	   double *myx;
	   myx = x;

	   /* mu is the center of mass where the probability starting at bin j will go */
        if (fabs(lambda) < 1.0e-10){ mu = myx[j] + h*dt; }
        else { mu = exp(lambda*dt)*(myx[j] + h/lambda) - h/lambda; }
        
       /* now we're going to look over all the slices of the gaussian */
        for (k=0; k<ndeltas; k++){
            double s = mu + deltas[k];
            if (s <= x[0]){ 
                F[j*N] = F[j*N] + ps[k];
            }
            else if (s >= x[N-1]){ 
                F[j*N+N-1] = F[j*N+N-1] + ps[k];
            }
            else{
                /* find the bin ids whose positions are just below or just above ss[k] */
                if (x[0]<s && s<x[1]) {lp = 0; hp = 1;}
                else if (x[N-2]<s && s<x[N-1]){lp = N-2; hp = N-1;}
                else {
                    hp = ceil( (s-x[1])/dx) + 1;
                    lp = floor((s-x[1])/dx) + 1;
                }
                
                if (hp == lp) { 
                    F[j*N+lp] = F[j*N+lp] + ps[k];
                }
                else { 
                    double dd = x[hp] - x[lp];
                    F[j*N+hp] = F[j*N+hp] + ps[k]*(s-x[lp])/dd;
                    F[j*N+lp] = F[j*N+lp] + ps[k]*(x[hp]-s)/dd;
                    
                }
                
            }
        }
   }
    
}
