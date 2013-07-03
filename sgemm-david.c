#include <stdio.h>
#include <emmintrin.h> 
#include <omp.h>

void sgemm(int m, int n, float *A, float *C) {

  int n2 = n / 4 * 4;
  int m2 = m / 4 * 4;
  
  #pragma omp parallel for
  for(int j = 0; j < m2; j += 4) {
    int jm = j*m;
    int j1m = (j+1) * m;
    int j2m = (j+2) * m;
    int j3m = (j+3) * m;

    for(int k = 0; k < n2; k += 4){
    
      int km = k*m;
	  int k1m = (k + 1) * m; 
	  int k2m = (k + 2) * m;
	  int k3m = (k + 3) * m;
	  
	  __m128 AT0 = _mm_load1_ps(A+j + km);
	  __m128 AT1 = _mm_load1_ps(A + j + k1m);
	  __m128 AT2 = _mm_load1_ps(A+ j + k2m);
	  __m128 AT3 = _mm_load1_ps(A+ j + k3m);
	  
	  __m128 AT01 = _mm_load1_ps(A+j+1 + km);
	  __m128 AT11 = _mm_load1_ps(A + j+1 + k1m);
	  __m128 AT21 = _mm_load1_ps(A+ j+1 + k2m);
	  __m128 AT31 = _mm_load1_ps(A+ j+1 + k3m);
	  
	  __m128 AT02 = _mm_load1_ps(A+j+2 + km);
	  __m128 AT12 = _mm_load1_ps(A + j+2 + k1m);
	  __m128 AT22 = _mm_load1_ps(A+ j+2 + k2m);
	  __m128 AT32 = _mm_load1_ps(A+ j+2 + k3m);
	  
	  __m128 AT03 = _mm_load1_ps(A+j+3 + km);
	  __m128 AT13 = _mm_load1_ps(A + j+3 + k1m);
	  __m128 AT23 = _mm_load1_ps(A+ j+3 + k2m);
	  __m128 AT33 = _mm_load1_ps(A+ j+3 + k3m);
	    for(int  i = 0; i < m2; i+= 4 ) {
          __m128 A0 = _mm_loadu_ps(A+ i + km); 
          __m128 A1 = _mm_loadu_ps(A+ i + k1m);
          __m128 A2 = _mm_loadu_ps(A+ i + k2m);
          __m128 A3 = _mm_loadu_ps(A+ i + k3m);

           __m128 C0 = _mm_mul_ps(A0, AT0);
          C0 = _mm_add_ps(C0, _mm_mul_ps(A1 , AT1));
		  C0 = _mm_add_ps(C0, _mm_mul_ps(A2 , AT2));
		  C0 = _mm_add_ps(C0, _mm_mul_ps(A3 , AT3));
		
		   __m128 C1 = _mm_mul_ps(A0, AT01);
          C1 = _mm_add_ps(C1, _mm_mul_ps(A1 , AT11));
		  C1 = _mm_add_ps(C1, _mm_mul_ps(A2 , AT21));
		  C1 = _mm_add_ps(C1, _mm_mul_ps(A3 , AT31));
		  
		  __m128 C2 = _mm_mul_ps(A0, AT02);
          C2 = _mm_add_ps(C2, _mm_mul_ps(A1 , AT12));
		  C2 = _mm_add_ps(C2, _mm_mul_ps(A2 , AT22));
		  C2 = _mm_add_ps(C2, _mm_mul_ps(A3 , AT32));
		  
		  __m128 C3 = _mm_mul_ps(A0, AT03);
          C3 = _mm_add_ps(C3, _mm_mul_ps(A1 , AT13));
		  C3 = _mm_add_ps(C3, _mm_mul_ps(A2 , AT23));
		  C3 = _mm_add_ps(C3, _mm_mul_ps(A3 , AT33));
		  
		  __m128 C4 = _mm_loadu_ps(C+ i+jm);
		  __m128 C5 = _mm_add_ps(C0, C4);
		  _mm_storeu_ps(C+i+jm, C5);
		  
		  C4 = _mm_loadu_ps(C+ i+j1m);
		  C5 = _mm_add_ps(C1, C4);
		  _mm_storeu_ps(C+i+j1m, C5);
		  
		  C4 = _mm_loadu_ps(C+ i+j2m);
		  C5 = _mm_add_ps(C2, C4);
		  _mm_storeu_ps(C+i+j2m, C5);
		  
		  C4 = _mm_loadu_ps(C+ i+j3m);
		  C5 = _mm_add_ps(C3, C4);
		  _mm_storeu_ps(C+i+j3m, C5);

        }
        
       
	  for(int i = m2; i < m; i++) {
	       C[i+jm] += A[i+km] * A[j+km];
           C[i+j1m] += A[i+km] * A[(j + 1) + km];
           C[i+j2m] += A[i+km] * A[(j + 2) + km];
           C[i+j3m] += A[i+km] * A[(j + 3) + km];

	       C[i+jm] += A[i+(k+1)*m] * A[j+k1m];
           C[i+j1m] += A[i+k1m] * A[(j + 1) + k1m];
           C[i+j2m] += A[i+k1m] * A[(j + 2) + k1m];
           C[i+j3m] += A[i+k1m] * A[(j + 3) + k1m];

	       C[i+jm] += A[i+k2m] * A[j+k2m];
           C[i+j1m] += A[i+k2m] * A[(j + 1) + k2m];
           C[i+j2m] += A[i+k2m] * A[(j + 2) + k2m];
           C[i+j3m] += A[i+k2m] * A[(j + 3) + k2m];

	       C[i+jm] += A[i+k3m] * A[j+k3m];
           C[i+j1m] += A[i+k3m] * A[(j + 1) + k3m];
           C[i+j2m] += A[i+k3m] * A[(j + 2) + k3m];
           C[i+j3m] += A[i+k3m] * A[(j + 3) + k3m];
	  }

       
        
    }
        
    for(int k = n2; k < n; k ++){
    
      int km = k*m;
	  __m128 AT0 = _mm_load1_ps(A+j + km);
	  __m128 AT01 = _mm_load1_ps(A+j+1 + km);
	  __m128 AT02 = _mm_load1_ps(A+j+2 + km);
	  __m128 AT03 = _mm_load1_ps(A+j+3 + km);

	    for(int  i = 0; i < m2; i+= 4 ) {
          __m128 A0 = _mm_loadu_ps(A+ i + km); 
          __m128 C0 = _mm_mul_ps(A0, AT0);
		  __m128 C1 = _mm_mul_ps(A0, AT01);
		  __m128 C2 = _mm_mul_ps(A0, AT02);
		  __m128 C3 = _mm_mul_ps(A0, AT03);
		  __m128 C4 = _mm_loadu_ps(C+ i+jm);
		  __m128 C5 = _mm_add_ps(C0, C4);
		  _mm_storeu_ps(C+i+jm, C5);
		  
		  C4 = _mm_loadu_ps(C+ i+j1m);
		  C5 = _mm_add_ps(C1, C4);
		  _mm_storeu_ps(C+i+j1m, C5);
		  
		  C4 = _mm_loadu_ps(C+ i+j2m);
		  C5 = _mm_add_ps(C2, C4);
		  _mm_storeu_ps(C+i+j2m, C5);
		  
		  C4 = _mm_loadu_ps(C+ i+j3m);
		  C5 = _mm_add_ps(C3, C4);
		  _mm_storeu_ps(C+i+j3m, C5);

        }
        
        for(int i = m2; i < m;i++) {
          C[i+jm] += A[i+km] * A[j+km];
          C[i+j1m] += A[i+km] * A[j + 1+km];
          C[i+j2m] += A[i+km] * A[j+2+km];
          C[i+j3m] += A[i+km] * A[j+3+km];
        }
    }   
  }
  


  
  for(int j = m2; j < m; j++) {
    int jm = j*m;
    for(int k = 0; k < n2; k += 4){ 
      int km = k*m;
	  int k1m = (k + 1) * m; 
	  int k2m = (k + 2) * m;
	  int k3m = (k + 3) * m;
	  
	  __m128 AT0 = _mm_load1_ps(A+j + km);
	  __m128 AT1 = _mm_load1_ps(A + j + k1m);
	  __m128 AT2 = _mm_load1_ps(A+ j + k2m);
	  __m128 AT3 = _mm_load1_ps(A+ j + k3m);
	  
	    for(int  i = 0; i < m2; i+= 4 ) {
          __m128 A0 = _mm_loadu_ps(A+ i + km); 
          __m128 A1 = _mm_loadu_ps(A+ i + k1m);
          __m128 A2 = _mm_loadu_ps(A+ i + k2m);
          __m128 A3 = _mm_loadu_ps(A+ i + k3m);

           __m128 C0 = _mm_mul_ps(A0, AT0);
          C0 = _mm_add_ps(C0, _mm_mul_ps(A1 , AT1));
		  C0 = _mm_add_ps(C0, _mm_mul_ps(A2 , AT2));
		  C0 = _mm_add_ps(C0, _mm_mul_ps(A3 , AT3));
		
		  
		  __m128 C4 = _mm_loadu_ps(C+ i+j*m);
		  __m128 C5 = _mm_add_ps(C0, C4);
		  _mm_storeu_ps(C+i+j*m, C5);
		  
        }
        
        for(int i = m2; i < m; i++) {
          C[i+jm] += A[i+km] * A[j+km];
          C[i+jm] += A[i+k1m] * A[j+k1m];
          C[i+jm] += A[i+k2m] * A[j+k2m];
          C[i+jm] += A[i+k3m] * A[j+k3m];
        }
    }

    for(int k = n2; k < n; k ++){ 
      int km = k*m;

	  
	  __m128 AT0 = _mm_load1_ps(A+j + km);

	  
	    for(int  i = 0; i < m2; i+= 4 ) {
          __m128 A0 = _mm_loadu_ps(A+ i + km); 

           __m128 C0 = _mm_mul_ps(A0, AT0);
		  
		  __m128 C4 = _mm_loadu_ps(C+ i+jm);
		  __m128 C5 = _mm_add_ps(C0, C4);
		  _mm_storeu_ps(C+i+jm, C5);
		  
        }
        for(int i = m2; i < m; i++) {
          C[i+jm] += A[i+km] * A[j+km];
        }
    }

 
  }  

}