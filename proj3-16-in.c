#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h> /* where intrinsics are defined */
#include <omp.h>

/*void sgemm1( int m, int n, float *A, float *C)
{
    for( int i = 0; i < m; i++ )
      for( int k = 0; k < n; k++ ) 
        for( int j = 0; j < m; j++ ) 
        C[i+j*m] += A[i+k*m] * A[j+k*m];
    } */


void sgemm( int m, int n, float *A, float *C)
{   
    int e = m/4*4; //for row's edge case
    int p = n/16*16;
    int q = (n-p)/8*8 + p;
    int w = (n-q)/4*4 + q;
    int km0,km1,km2,km3,km4,km5,km6,km7,km8,km9,km10,km11,km12,km13,km14,km15;
    float *t;  //b
    float *s;  //a
    float *c;  //c 

    #pragma omp parallel private(t,s,c,km0,km1,km2,km3,km4,km5,km6,km7,km8,km9,km10,km11,km12,km13,km14,km15)
    {   
        #pragma omp for
        for (int j=0; j<m; j++) {
            int mj = m*j;
            t = A+j;
            c = C+mj;
            //every four column1
            for (int k =0; k<p; k+=16){
                km0 = m*k;
                km1 = m*(k+1);
                km2 = m*(k+2);
                km3 = m*(k+3);
                km4 = m*(k+4);
                km5 = m*(k+5);
                km6 = m*(k+6);
                km7 = m*(k+7);
                km8 = m*(k+8);
                km9 = m*(k+9);
                km10 = m*(k+10);
                km11 = m*(k+11);
                km12 = m*(k+12);
                km13 = m*(k+13);
                km14 = m*(k+14);
                km15 = m*(k+15);


                __m128 b0 = _mm_load1_ps( t + km0 );
                __m128 b1 = _mm_load1_ps( t + km1 );
                __m128 b2 = _mm_load1_ps( t + km2 );
                __m128 b3 = _mm_load1_ps( t + km3 );
                __m128 b4 = _mm_load1_ps( t + km4 );
                __m128 b5 = _mm_load1_ps( t + km5 );
                __m128 b6 = _mm_load1_ps( t + km6 );
                __m128 b7 = _mm_load1_ps( t + km7 );
                __m128 b8 = _mm_load1_ps( t + km8 );
                __m128 b9 = _mm_load1_ps( t + km9 );
                __m128 b10 = _mm_load1_ps( t + km10);
                __m128 b11 = _mm_load1_ps( t + km11);
                __m128 b12 = _mm_load1_ps( t + km12);
                __m128 b13 = _mm_load1_ps( t + km13);
                __m128 b14 = _mm_load1_ps( t + km14);
                __m128 b15 = _mm_load1_ps( t + km15);


                for(int i = 0; i<e;i+=4){
                    s = A+i;
                    __m128 c0 = _mm_loadu_ps( c+i );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b0 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km1 ), b1 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km2 ), b2 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km3 ), b3 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km4 ), b4 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km5 ), b5 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km6 ), b6 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km7 ), b7 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km8 ), b8 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km9 ), b9 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km10 ), b10 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km11 ), b11 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km12 ), b12 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km13 ), b13 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km14 ), b14 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km15 ), b15 ) );

                    _mm_storeu_ps( c + i, c0 );
                }
            }

            for (int k =p; k<q; k+=8){
                km0 = m*k;
                km1 = m*(k+1);
                km2 = m*(k+2);
                km3 = m*(k+3);
                km4 = m*(k+4);
                km5 = m*(k+5);
                km6 = m*(k+6);
                km7 = m*(k+7);


                __m128 b0 = _mm_load1_ps( t + km0 );
                __m128 b1 = _mm_load1_ps( t + km1 );
                __m128 b2 = _mm_load1_ps( t + km2 );
                __m128 b3 = _mm_load1_ps( t + km3 );
                __m128 b4 = _mm_load1_ps( t + km4 );
                __m128 b5 = _mm_load1_ps( t + km5 );
                __m128 b6 = _mm_load1_ps( t + km6 );
                __m128 b7 = _mm_load1_ps( t + km7 );


                for(int i = 0; i<e;i+=4){
                    s = A+i;
                    __m128 c0 = _mm_loadu_ps( c+i );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b0 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km1 ), b1 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km2 ), b2 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km3 ), b3 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km4 ), b4 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km5 ), b5 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km6 ), b6 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km7 ), b7 ) );

                    _mm_storeu_ps( c + i, c0 );
                }
            }

            for (int k =q; k<w; k+=4){
                km0 = m*k;
                km1 = m*(k+1);
                km2 = m*(k+2);
                km3 = m*(k+3);

                __m128 b0 = _mm_load1_ps( t + km0 );
                __m128 b1 = _mm_load1_ps( t + km1 );
                __m128 b2 = _mm_load1_ps( t + km2 );
                __m128 b3 = _mm_load1_ps( t + km3 );

                for(int i = 0; i<e;i+=4){
                    s = A+i;
                    __m128 c0 = _mm_loadu_ps( c+i );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b0 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km1 ), b1 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km2 ), b2 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km3 ), b3 ) );

                    _mm_storeu_ps( c + i, c0 );
                }
            }

            /*for (int k =q; k<w; k+=2){
                km0 = m*k;
                km1 = m*(k+1);

                __m128 b0 = _mm_load1_ps( t + km0 );
                __m128 b1 = _mm_load1_ps( t + km1 );

                for(int i = 0; i<e;i+=4){
                    s = A+i;
                    __m128 c0 = _mm_loadu_ps( c+i );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b0 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km1 ), b1 ) );

                    _mm_storeu_ps( c + i, c0 );
                }
            }*/

            for (int k =w; k<n; k++){
                km0 = m*k;

                __m128 b0 = _mm_load1_ps( t + km0 );

                for(int i = 0; i<e;i+=4){
                    s = A+i;
                    __m128 c0 = _mm_loadu_ps( c+i );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b0 ) );

                    _mm_storeu_ps( c + i, c0 );
                }
            }
        }
    }

    if (e < m) {
        #pragma omp parallel for private(i,j,k,km,jm)
        for( j = 0; j < m; j++ ) {
            jm = j*m;
            t = A+j;
            for( k = 0; k < n; k++ ) {
                km = k*m;
                for( i = e; i < m; i++ ) {
                    *(C+i+jm) += *(A+i+km) * (*(t+km));
                }
            }
        }
    }
}



/* int main( int argc, char **argv ) {
    int m =41;
    int n = 35,i;

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
    }

*/
