#include <math.h>
#include "mex.h" 
#include <stdlib.h>


//  min (W\in [0,1] or W>0)   F2P2_NMF_GCDM(V,H,V2,H2,lamda)=|V - W* H|_F^2 + lamda1* |V2-W*H2|_F^2  +lamda2*|W|_1  

//   ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
//   ATTN2: This package was developed by Ms. Shao-Yuan Li (lisy@lamda.nju.edu.cn). For any problem concerning the code, please feel free to contact Ms. Li.
//  Some varables used in the code
//===============================================================================
//input: 
//		V, V2: the input n by t feature matrix  n*m  : 
//		H,H2: the k*m matrix
 
//      Winint: the initial value of W
//      c: tradeoff parameter
//		trace: 1: compute objective value per iteration.
//			   0: do not compute objective value per iteration. (default)
//
// output: 
//		NMF_GCDM will output matrices W, such that WH is an approximation of V and W is close to Wo
//		W: n by k  matrix
//		objGCD: objective values. 
//		timeGCD: time taken by GCD. 

//  Reference:
//    [1] S.-Y. Li, Y. Jiang nd Z.-H. Zhou. Partial Multi-View Clustering. In: Proceedings of the 28th AAAI Conference on Artificial Intelligence (AAAI'14),2014.
//    [2] C. Hsieh, and I. Dhillon. Fast coordinate descent methods with variable selection for non-negative matrix factorization. In Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining(KDD'11), 2011.
//===============================================================================

double *createMatrix(int row,int col){
    double * new_m = (double *)malloc(sizeof(double)*row*col);
    for (long i=0;i<row*col;i++)
        new_m[i] = 0;
    return new_m;
}
double *createDoubleMatrix(int row,int col){
    double * new_m = (double *)malloc(sizeof(double)*row*col);
    for (long i=0;i<row*col;i++)
        new_m[i] = 0;
    return new_m;
}
long double *createLongDoubleMatrix(int row,int col){
    long double * new_m = (long double *)malloc(sizeof(long double)*row*col);
    for (long i=0;i<row*col;i++)
        new_m[i] = 1e-8;
    return new_m;
}
long *createLongMatrix(int row,int col){
    long * new_m = (long *)malloc(sizeof(long)*row*col);
    for (long i=0;i<row*col;i++)
        new_m[i] = 0;
    return new_m;
}

void destroyMatrix(int *pMatrix){
    free(pMatrix);
}
void destroyMatrix(double *pMatrix){
    free(pMatrix);
}
void destroyMatrix(long *pMatrix){
    free(pMatrix);
}
void destroyMatrix(long double *pMatrix){
    free(pMatrix);
}
// void getHH(double *HH,double *H,int k,int m){  HH=H*H';
//  for i=0:k-1
//     for j=i:k-1
//          tmp=0;
//         for r=0:m
//               tmp+=H[i][r]*H[j][r];
//         end
//          HH[i][j]=tmp;// 
//          HH[j][i]=HH[i][j];
//   end end
//void getHH(double *HH,double *H,int k,int m)
// HHt=H*H'+c1*H2*H2';  k*k   H: k*m  H2: k*m2
  void  getHHt(double *HHt,double *H,double *H2,double c1,int k,int m, int m2){
    long i,j;
   // double *sums_hhir = createMatrix(k*(k+1)/2,1); //sum(H(j,:).*H(j,:)) for each j
    for (i=0;i<k;i++)
       for (j=0;j<k;j++){
          double sum = 0;
          double sum2=0;
          for (long d=0;d < m; d++){
              sum += H[d*k+i]*H[d*k+j];
          }
          for (long d=0;d < m2; d++){
              sum2 += H2[d*k+i]*H2[d*k+j];
          }
          HHt[j*k+i] = sum+c1*sum2;
          //printf("initial HHt [%d][%d] :  %lf !\n", i, j, HHt[j*k+i]); 
          
       }
  //   destroyMatrix(sums_hhir);
}

//getVH(VH,V,H,n,k,m);  VH=V*H';  n*k = (n*m�� *( m*k)

  //  VHt=V*H'+c1*V2*H2';
  void  getVHt(double *VHt,double *V,double *H,double *V2,double *H2,double c1,int n,int k,int m,int m2){
    long i,j;
   // double *sums_hhir = createMatrix(k*(k+1)/2,1); //sum(H(j,:).*H(j,:)) for each j
    for (i=0;i<n;i++)
       for (j=0;j<k;j++){
          double sum = 0;
          double sum2=0;
          for (long d=0;d < m; d++){
              sum += V[d*n+i]*H[d*k+j];
          }
          for (long d=0;d < m2; d++){
              sum2 += V2[d*n+i]*H2[d*k+j];
          }
          VHt[j*n+i] = sum+c1*sum2;
          //printf("initial VHt [%d][%d] :  %lf !\n", i, j, VHt[j*n+i]); 
       }
  //   destroyMatrix(sums_hhir);
}

 /*  SW= -GW./(GWW);  n*k
     *  GWW(i,r)= 2*( HHt(r,r))      
    */
  void getGWW(double *GWW,double *HHt,int n,int k){
      long i,j;
      for (i=0;i<n;i++)
        for (j=0;j<k;j++)
        {
          GWW[j*n+i]=2*HHt[j*k+j];    
          //printf("initial GWW [%d][%d] :  %lf !\n", i, j, GWW[j*n+i]); 
    } 
  }
 // obj=|V-WH|_F^2 + c*|V2-WH2|_F^2+c2*|W|1
  // objIner=getObj(V,W,H,V2,H2,n, m,k,c,m2);    H: k*m  H2: k*m2
 double getObj(double *V,double *W, double *H,double *V2, double *H2,int n, int m, int k, double c,double c2,int m2){
        long i,j,j2,j3,r;
         double objOut=0;
        for (i=0;i<n;i++)
        {
            for (j=0;j<m;j++)
            { 
                  double tmp=0;                 
                  for (r=0;r<k;r++){
                      tmp += (W[r*n+i]*H[j*k+r]);
                  }
                  objOut=objOut+pow(V[j*n+i]-tmp,2);        //Summation of the |V-WH| part  
            }
            for (j2=0;j2<m2;j2++)
            {                  
                  double tmp2=0;
                  for (r=0;r<k;r++){  
                      tmp2 += (W[r*n+i]*H2[j2*k+r]);
                  }
                  objOut=objOut+c*pow(V2[j2*n+i]-tmp2,2);  
            }            
        }
          for (j3=0;j3<n*k;j3++)
          {  
             objOut=objOut+c2*fabs(W[j3]);              //L1 Norm part
          }
        return objOut;
  }

 //    W: n*k     V: n*m   H: k*m     V2: n*m2  H2: k*m2   Winit: n*k
void initWs(double *HHt,double *VHt,double *GW,double *SW,double *DW,double *V,double *W,double *H,double *V2,double *H2,int n,int m, int k,double c1,double c2,double *GWW,int m2,int interval){
  //  GW=2*W*(H*H'+c1*H2*H2') -2*(V*H'+c1*V2*H2')+c2
    // W(n*k)* H(k*m)-V(n*m)       W(n*k)* H2(k*m2)-V2(n*m2) 
   // printf("Start initializing precomputations\n"); 
    
    long i,j,r;
    double tmp;
    double * tm = createMatrix(n,m);
    // HHt=H*H'+c1*H2*H2';  k*k
    getHHt(HHt,H,H2,c1,k,m,m2);
    
    getVHt(VHt,V,H,V2,H2,c1,n,k,m,m2);
    
    for (i=0;i<n;i++)
        for (j=0;j<k;j++){
            tmp = 0;  // W*HHt
            for (r=0;r<k;r++)
                tmp = tmp + W[r*n+i]*HHt[j*k+r];
            
           GW[j*n+i]=2*tmp-2*VHt[j*n+i]+c2;
           //printf("initial GW [%d][%d] :  %lf !\n", i, j, GW[j*n+i]); 
        }
   
  // printf("finish getGW !\n"); 
    /*  SW= -GW./(GWW);  n*k
     *  GWW(i,r)= 2*( HHt(r,r))   
    */
    getGWW(GWW,HHt,n,k);
    
   // printf("finish getGWW !\n"); 
   // SW= -GW./(GWW);  n*k
    for (i=0;i<n;i++)
        for (j=0;j<k;j++){
          SW[j*n+i] = -GW[j*n+i]/(GWW[j*n+i]==0?1e-10:GWW[j*n+i]);
          if (SW[j*n+i]+W[j*n+i]<0)
              SW[j*n+i]=-W[j*n+i];
          if (interval==1 && SW[j*n+i]+W[j*n+i]>1)
              SW[j*n+i]=1-W[j*n+i];
          //printf("initial SW [%d][%d] :  %lf !\n", i, j, SW[j*n+i]); 
        }
   
     //printf("finish getSW !\n"); 
    //DW=-GW.*SW-0.5*(GWW.*(SW.^2));
    for (i=0;i<n;i++)
        for (j=0;j<k;j++)
        {
             DW[j*n+i] = -GW[j*n+i]*SW[j*n+i] - 0.5*GWW[j*n+i]*SW[j*n+i]*SW[j*n+i]; 
        }    
    destroyMatrix(tm);
}
  
 // updateW(objGCD,W,V,H,V2,H2,n,m,k,c,tol,maxiter,trace,m2);
 //    W: n*k     V: n*m   H: k*m     V2: n*m2  H2: k*m2   Winit: n*k
void updateW(double *objGCD,double *W,double *V,double *H,double *V2,double *H2,int n,int m,int k, double c,double c2,double tol,int maxiter,int trace,int m2,int interval){
    //W :n*k;H:k*m

    long i,j,a,b,r;;
    long maxinner = k*k;
    double *HHt = createMatrix(k,k); //change1 ---
    double *VHt = createMatrix(n,k); //change1 ---
    double *GW = createMatrix(n,k);
    double *SW = createMatrix(n,k);
    double *DW = createMatrix(n,k);
    double * GWW= createMatrix(n,k);
    
   // void initWs(double *HHt,double *VHt,double *GW,double *SW,double *DW,double *V,double *W,double *H,double *V2,double *H2,int n,int m, int k,double c1,double *GWW){
    initWs(HHt,VHt,GW,SW,DW,V,W,H,V2,H2,n,m,k,c,c2,GWW,m2,interval);//----

  //  [maxV,maxI]=max(DW,[],2);
  //  init=max(maxV);
    long *VIndx = createLongMatrix(n,1); 
    double *maxV = createDoubleMatrix(n,1);
    double init = -1e15;
    for (i=0;i<n;i++){
        maxV[i] = -1e15;
        VIndx[i] = 0;
        for (j=0;j<k;j++){
            // printf("i, j  %d %d \n",i,j);
            if (DW[j*n+i]>maxV[i]){
                VIndx[i] = j;  //%%%%%%%%%%%%%
                maxV[i] = DW[j*n+i];
                
                //printf("maxI %d j %d for row %d \n", VIndx[i],j,i);
            }
        }
        if (maxV[i] > init) init = maxV[i];
        
    }

   //printf("Finish maxdecrease each row precomputation!\n");  
   for (int iter=0;iter<maxiter;iter++){ //-------------
      for (i=0;i<n;i++){
          for (long indx=0;indx<maxinner;indx++){
            double qv = maxV[i];
            long   qi = VIndx[i];
            if (sqrt(qv)<=1e-6) break;
            //if (qv<init*tol) break;
            double s = SW[qi*n+i];   
             W[qi*n+i] = W[qi*n+i] + s;        // W[i][qi] <-  W[i][qi]+s;     
             double mxv = -1e15;
             long mxi = 0;       
              for (j=0;j<k;j++){
                double tmp = 0;
                 GW[j*n+i] = GW[j*n+i] + 2*s*HHt[j*k+qi];
                SW[j*n+i] = -GW[j*n+i]/(GWW[j*n+i]==0?1e-10:GWW[j*n+i]);
                if (SW[j*n+i]+W[j*n+i]<0)
                     SW[j*n+i]=-W[j*n+i];
                if (interval==1 && SW[j*n+i]+W[j*n+i]>1)
                     SW[j*n+i]=1-W[j*n+i];
                DW[j*n+i] = -GW[j*n+i]*SW[j*n+i] - 0.5*GWW[j*n+i]*SW[j*n+i]*SW[j*n+i];
                
                 if (DW[j*n+i]>mxv){
                       mxv=DW[j*n+i];
                       mxi=j;
                      }
        }
             VIndx[i] = mxi; 
             maxV[i] = mxv;    
    }
   }
        // save the objective value in each iteration 	
       //obj=|V-WH|_F^2 + c*|V2-WH2|_F^2+c2*|W|1
       objGCD[iter]=getObj(V,W,H,V2,H2,n,m,k,c,c2,m2);  
       if (trace==1 && iter%20 ==0)// && iter%10 ==0
 	       printf("F2P2Constr Iteration %ld, objective value %lf\n", (long)(iter), objGCD[iter]);
        if  ( (iter>1 && (fabs(objGCD[iter]-objGCD[iter-1])/(objGCD[iter]+1e-6)) < (long double)1e-16) || objGCD[iter]<1e-6 )   // squre loss
            { printf("F2P2Constr Objective value converge to %lf at iteration %ld before the maxIteration reached \n",objGCD[iter],(long)(iter));
              objGCD[maxiter-1]=objGCD[iter];  break;} 
   }
   
    destroyMatrix(HHt);
    destroyMatrix(VHt);
    destroyMatrix(GW);
    destroyMatrix(SW);
    destroyMatrix(DW);
    destroyMatrix(GWW);
    destroyMatrix(maxV);
    destroyMatrix(VIndx);
}
   

//obj=doNMF(objGCD,W,V,H,V2,H2,Winit,n,m,k,c,maxiter, trace,m2);   
//    W: n*k     V: n*m   H: k*m     V2: n*m2  H2: k*m2   Winit: n*k
double doNMF(double *objGCD,double *W,double *V,double *H,double *V2,double *H2,double *Winit,int n,int m,int k,double c,double c2,int maxiter,int trace,int m2,int interval){
/*    double *SW = createMatrix(n,k);
    double *SH = createMatrix(k,n);
    double *DW = createMatrix(n,k);
    double *DH = createMatrix(k,n);*/
    long i,j,r,iter;
    double tol = 1e-8;
    printf("Start running NMF_GCDM\n");
    
    for (i=0;i<n*k;i++){
        W[i] = Winit[i];
    }
   // printf("Winit finishied \n");
   // for (iter=0;iter<maxiter;iter++){
     updateW(objGCD,W,V,H,V2,H2,n,m,k,c,c2,tol,maxiter,trace,m2,interval);//------------  
    return 0;
}

// mex function =============================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *V, *H,*V2,*H2,*Wo,*Winit,*Param; // input 
	long n,m,k, maxiter,trace,m2,interval;
	double c,c2;
   
    double *W;  // output
    double *objGCD;
    double obj;
	double *Wout;

 // doNMF(double *objGCD,double *W,double *V,double *H,double *Wo,double *Winit,int n,int m,int k,double c,int maxiter,int trace)
	bool flag = true;
	// Input Arguments
	V = mxGetPr(prhs[0]);
	long Vn = mxGetM(prhs[0]);
	long Vm = mxGetN(prhs[0]);
    
    H = mxGetPr(prhs[1]);
	long Hk = mxGetM(prhs[1]);
	long Hm = mxGetN(prhs[1]);
	if (Hm!=Vm) {flag = false;printf("Error Size of H!\n");}
    //    W: n*k     V: n*m   H: k*m     V2: n*m2  H2: k*m2   Winit: n*k
    V2 = mxGetPr(prhs[2]);
	long V2n = mxGetM(prhs[2]);
	long V2m = mxGetN(prhs[2]);
    if (V2n!=Vn) {flag = false;printf("Error Size of V2!\n");}
    
    H2 = mxGetPr(prhs[3]);
	long H2k = mxGetM(prhs[3]);
	long H2m = mxGetN(prhs[3]);
	if (H2m!=V2m) {flag = false;printf("Error Size of H!\n");}
    
    Winit = mxGetPr(prhs[4]);
	long Winitn = mxGetM(prhs[4]);
	long Winitk = mxGetN(prhs[4]);
	if (Winitn!=Vn || Winitk!=Hk) {flag = false;printf("Error Size of Winit!\n  Winit [%d %d]   Vn %d  Hk %d",Winitn,Winitk,Vn,Hk);}

	Param = mxGetPr(prhs[5]);
	long Pn = mxGetM(prhs[5]);
	long Pk = mxGetN(prhs[5]);
	long mxSize = Pn>Pk?Pn:Pk;
	long mnSize = Pn<Pk?Pn:Pk;
	if (mxSize!=5 || mnSize!=1) {flag = false;printf("Error Size of Param!\n");}

	if (!flag) {
		plhs[0] = NULL;
		plhs[1] = NULL;
		plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
		Wout = mxGetPr(plhs[2]);
		Wout[0] = -1;
		return;
	}
    
//    W: n*k     V: n*m   H: k*m     V2: n*m2  H2: k*m2   Winit: n*k
    n = Vn;
    m = Vm;
    m2 = V2m;
    k = Hk;

    c = Param[0]; //lamda1
    c2=Param[1]; //lamda2
    maxiter = (long)(Param[2]);
    trace = (int)(Param[3]);
    interval=(int)(Param[4]);  // interval==1: W\in[0,1], interval!=1ʱ��W>0

    W = createMatrix(n,k);
    objGCD = createMatrix(maxiter,1);

    //    W: n*k     V: n*m   H: k*m     V2: n*m2  H2: k*m2   Winit: n*k
   obj=doNMF(objGCD,W,V,H,V2,H2,Winit,n,m,k,c,c2,maxiter, trace,m2,interval);       
        
   long i;
	plhs[0] = mxCreateDoubleMatrix(n,k,mxREAL);
    Wout = mxGetPr(plhs[0]);
    for (i=0;i<n*k;i++)
		Wout[i] = W[i];
    
    plhs[1] = mxCreateDoubleMatrix(maxiter,1,mxREAL);
	Wout = mxGetPr(plhs[1]);
	 for (i=0;i<maxiter;i++)
		Wout[i] = objGCD[i]; 

    destroyMatrix(W);
    destroyMatrix(objGCD);
	return;	
}
