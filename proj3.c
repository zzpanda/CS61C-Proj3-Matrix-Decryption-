#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h> /* where intrinsics are defined */
#include <omp.h>

void sgemm( int m, int n, float *A, float *C)
{   
    int e = m/4*4;
    
    /*{   
        for (int j=0; j<m; j++) {
        	
            for (int i =0; i<e; i+=4) {
    	        __m128 c = _mm_loadu_ps( C+i+m*j );
            	
            	for (int k=0;k<n;k++) {
            	    __m128 b = _mm_load1_ps( A + j + m*k );
            	    __m128 a  = _mm_loadu_ps( A+ i + m*k );
            	    
            	    c = _mm_add_ps( c, _mm_mul_ps( a, b ) );
            	}
            	
            	_mm_storeu_ps( C+i+m*j, c );
            }
        }
    }*/
    
    {   
        for (int j=0; j<m; j++) {
            
            for (int k=0;k<n;k++) {
                __m128 b = _mm_load1_ps( A + j + m*k );
                
                for (int i =0; i<e; i+=4) {
                    __m128 c = _mm_loadu_ps( C+i+m*j );
                    //__m128 a  = _mm_loadu_ps( A+i+m*k );

                    c = _mm_add_ps( c, _mm_mul_ps( _mm_loadu_ps( A+i+m*k ), b ) );
                    _mm_storeu_ps( C+i+m*j, c );
                }
            }
        }
    }

    if (e != m) {
        for( int j = 0; j < n; j++ ) {
            int a1 = j*m;
            for( int k = 0; k < m; k++ ) {
                int a2 = k*m;
                for( int i = e; i < m; i++ ) {
                    *(C+i+a2) += *(A+i+a1) * (*(A+k+a1));
	        }
	    }
        }
    }
}

/*void sgemm1( int m, int n, float *A, float *C)
{
  for( int i = 0; i < m; i++ )
    for( int k = 0; k < n; k++ ) 
      for( int j = 0; j < m; j++ ) 
        C[i+j*m] += A[i+k*m] * A[j+k*m];
}   

int main( int argc, char **argv ) {
  int m =60;
  int n = 60,i;

  float *A = (float*)malloc( m*n*sizeof(float) );
  float *C = (float*)malloc( m*m*sizeof(float) );
  float *C1 = (float*)malloc( m*m*sizeof(float) );

  for( i = 0; i < m*n; i++ ) A[i] = (float)rand()/RAND_MAX;
  for( i = 0; i < m*m; i++ ) C[i] = (float) 0.00;
  for( i = 0; i < m*m; i++ ) C1[i] = (float) 0.00;

  sgemm(m, n, A, C);
  sgemm1(m, n, A, C1);  

  for( i = 0; i < m*m; i++ ) {
      printf("%d\n", i);
      printf("C1: %f\n ",C1[i]);
      printf("C:%f\n ",C[i]);
      if( C1[i] != C[i]) {
        printf("Error!!!! Transpose does not result in correct answer!!\n");
          exit( -1 );
     }
  }



  free( A );
  free( C );
  free( C1);
  return 0;
  }*/