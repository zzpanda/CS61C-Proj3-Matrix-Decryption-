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
    int q = n/8*8;
    int w = (n-q)/4*4 + q;
    int km0,km1,km2,km3,km4,km5,km6,km7,j,k,i,mj,km,jm;
    __m128 c0,b0,b1,b2,b3,b4,b5,b6,b7;
    float *t;  //b
    float *s;  //a
    float *c;  //c 

    #pragma omp parallel for private(t,s,c,km0,km1,km2,km3,km4,km5,km6,km7,j,k,i,mj,km,jm,c0,b0,b1,b2,b3,b4,b5,b6,b7)
        for (j=0; j<m; j++) {
            mj = m*j;
            t = A+j;
            c = C+mj;
            //every four column1

            for (k =0; k<q; k+=8){
                km0 = k*m;
                km1 = (k+1)*m;
                km2 = (k+2)*m;
                km3 = (k+3)*m;
                km4 = (k+4)*m;
                km5 = (k+5)*m;
                km6 = (k+6)*m;
                km7 = (k+7)*m;


                b0 = _mm_load1_ps( t + km0 );
                b1 = _mm_load1_ps( t + km1 );
                b2 = _mm_load1_ps( t + km2 );
                b3 = _mm_load1_ps( t + km3 );
                b4 = _mm_load1_ps( t + km4 );
                b5 = _mm_load1_ps( t + km5 );
                b6 = _mm_load1_ps( t + km6 );
                b7 = _mm_load1_ps( t + km7 );


                for(i = 0; i<e;i+=4){
                    s = A+i;
                    c0 = _mm_loadu_ps( c+i );
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

                for (i = e; i < m; i++) {
                    *(C+i+mj) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj) += *(A+i+km1) * (*(t+km1));
                    *(C+i+mj) += *(A+i+km2) * (*(t+km2));
                    *(C+i+mj) += *(A+i+km3) * (*(t+km3));
                    *(C+i+mj) += *(A+i+km4) * (*(t+km4));
                    *(C+i+mj) += *(A+i+km5) * (*(t+km5));
                    *(C+i+mj) += *(A+i+km6) * (*(t+km6));
                    *(C+i+mj) += *(A+i+km7) * (*(t+km7));
                }
            }

            for (k =q; k<w; k+=4){
                km0 = k*m;
                km1 = (k+1)*m;
                km2 = (k+2)*m;
                km3 = (k+3)*m;

                b0 = _mm_load1_ps( t + km0 );
                b1 = _mm_load1_ps( t + km1 );
                b2 = _mm_load1_ps( t + km2 );
                b3 = _mm_load1_ps( t + km3 );

                for(i = 0; i<e;i+=4){
                    s = A+i;
                    c0 = _mm_loadu_ps( c+i );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b0 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km1 ), b1 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km2 ), b2 ) );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km3 ), b3 ) );

                    _mm_storeu_ps( c + i, c0 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj) += *(A+i+km1) * (*(t+km1));
                    *(C+i+mj) += *(A+i+km2) * (*(t+km2));
                    *(C+i+mj) += *(A+i+km3) * (*(t+km3));
                }
            }

            for (k =w; k<n; k++){
                km0 = k*m;
                b0 = _mm_load1_ps( t + km0 );

                for(i = 0; i<e;i+=4){
                    s = A+i;
                    c0 = _mm_loadu_ps( c+i );
                    c0 = _mm_add_ps( c0, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b0 ) );

                    _mm_storeu_ps( c + i, c0 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj) += *(A+i+km0) * (*(t+km0));
                }
            }
        }
}



/*int main( int argc, char **argv ) {
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
    }*/
