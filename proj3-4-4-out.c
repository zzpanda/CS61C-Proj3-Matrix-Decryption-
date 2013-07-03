#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h> /* where intrinsics are defined */
#include <omp.h>

void sgemm( int m, int n, float *A, float *C)
{   
    int e = m/4*4; //for row's edge case
    int w = n/4*4;
    int km0,km1,km2,km3,j,k,i,mj0,mj1,mj2,mj3;
    __m128 c00,c10,c20,c30,b00,b01,b02,b03,b10,b11,b12,b13,b20,b21,b22,b23,b30,b31,b32,b33,a0,a1,a2,a3;
    float *t;  //b
    float *s;  //a
    float *c0,*c1,*c2,*c3;  //c 

        #pragma omp parallel for private(t,s,c0,c1,c2,c3,km0,km1,km2,km3,j,k,i,mj0,mj1,mj2,mj3,c00,c10,c20,c30,b00,b01,b02,b03,b10,b11,b12,b13,b20,b21,b22,b23,b30,b31,b32,b33,a0,a1,a2,a3)
        for (j=0; j<e; j+=4) {
            mj0 = m*j;
            mj1 = m*(j+1);
            mj2 = m*(j+2);
            mj3 = m*(j+3);
            t = A+j;
            c0 = C+mj0;
            c1 = C+mj1;
            c2 = C+mj2;
            c3 = C+mj3;
            //every four column1

            for (k =0; k<w; k+=4){
                km0 = k*m;
                km1 = (k+1)*m;
                km2 = (k+2)*m;
                km3 = (k+3)*m;


                b00 = _mm_load1_ps( t + km0 );
                b01 = _mm_load1_ps( t + km1 );
                b02 = _mm_load1_ps( t + km2 );
                b03 = _mm_load1_ps( t + km3 );

                b10 = _mm_load1_ps( t +1+ km0 );
                b11 = _mm_load1_ps( t +1+ km1 );
                b12 = _mm_load1_ps( t +1+ km2 );
                b13 = _mm_load1_ps( t +1+ km3 );

                b20 = _mm_load1_ps( t +2+ km0 );
                b21 = _mm_load1_ps( t +2+  km1 );
                b22 = _mm_load1_ps( t +2+ km2 );
                b23 = _mm_load1_ps( t +2+ km3 );


                b30 = _mm_load1_ps( t +3+ km0 );
                b31 = _mm_load1_ps( t +3+ km1 );
                b32 = _mm_load1_ps( t +3+ km2 );
                b33 = _mm_load1_ps( t +3+ km3 );


                for(i = 0; i<e;i+=4){
                    s = A+i;
                    a0 = _mm_loadu_ps( s + km0 );
                    a1 = _mm_loadu_ps( s + km1 );
                    a2 = _mm_loadu_ps( s + km2 );
                    a3 = _mm_loadu_ps( s + km3 );

                    c00 = _mm_loadu_ps( c0+i );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a0, b00 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a1, b01 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a2, b02 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a3, b03 ) );

                    _mm_storeu_ps( c0 + i, c00 );

                    c10 = _mm_loadu_ps( c1+i );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a0, b10 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a1, b11 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a2, b12 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a3, b13 ) );
                    _mm_storeu_ps( c1 + i, c10 );
                    
                    c20 = _mm_loadu_ps( c2+i );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a0, b20 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a1, b21 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a2, b22 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a3, b23 ) );

                    _mm_storeu_ps( c2 + i, c20 );

                    c30 = _mm_loadu_ps( c3+i );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a0, b30 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a1, b31 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a2, b32 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a3, b33 ) );

                    _mm_storeu_ps( c3 + i, c30 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj0) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj1) += *(A+i+km0) * (*(t+1+km0));
                    *(C+i+mj2) += *(A+i+km0) * (*(t+2+km0));
                    *(C+i+mj3) += *(A+i+km0) * (*(t+3+km0));

                    *(C+i+mj0) += *(A+i+km1) * (*(t+km1));
                    *(C+i+mj1) += *(A+i+km1) * (*(t+1+km1));
                    *(C+i+mj2) += *(A+i+km1) * (*(t+2+km1));
                    *(C+i+mj3) += *(A+i+km1) * (*(t+3+km1));

                    *(C+i+mj0) += *(A+i+km2) * (*(t+km2));
                    *(C+i+mj1) += *(A+i+km2) * (*(t+1+km2));
                    *(C+i+mj2) += *(A+i+km2) * (*(t+2+km2));
                    *(C+i+mj3) += *(A+i+km2) * (*(t+3+km2));

                    *(C+i+mj0) += *(A+i+km3) * (*(t+km3));
                    *(C+i+mj1) += *(A+i+km3) * (*(t+1+km3));
                    *(C+i+mj2) += *(A+i+km3) * (*(t+2+km3));
                    *(C+i+mj3) += *(A+i+km3) * (*(t+3+km3));

                }
            }

            for (k =w; k<n; k++){
                km0 = k*m;
                b00 = _mm_load1_ps( t + km0 );
                b10 = _mm_load1_ps( t +1+ km0 );
                b20 = _mm_load1_ps( t +2+ km0 );
                b30 = _mm_load1_ps( t +3+km0 );

                for(i = 0; i<e;i+=4){
                    //s = A+i;
                    a0 = _mm_loadu_ps( A+i + km0 );

                    c00 = _mm_loadu_ps( c0+i );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a0, b00 ) );
                    _mm_storeu_ps( c0 + i, c00 );

                    c10 = _mm_loadu_ps( c1+i );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a0, b10 ) );
                    _mm_storeu_ps( c1 + i, c10 );

                    c20 = _mm_loadu_ps( c2+i );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a0, b20 ) );
                    _mm_storeu_ps( c2 + i, c20 );

                    c30 = _mm_loadu_ps( c3+i );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a0, b30 ) );
                    _mm_storeu_ps( c3 + i, c30 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj0) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj1) += *(A+i+km0) * (*(t+1+km0));
                    *(C+i+mj2) += *(A+i+km0) * (*(t+2+km0));
                    *(C+i+mj3) += *(A+i+km0) * (*(t+3+km0));
                }
            }
        }
        
        #pragma omp parallel for private(t,s,c0,km0,km1,km2,km3,j,k,i,mj0,c00,b00,b01,b02,b03)
        // edge case
        for (j=e; j<m; j++) {
            mj0 = m*j;
            t = A+j;
            c0 = C+mj0;
            //every four column1

            for (k =0; k<w; k+=4){
                km0 = k*m;
                km1 = (k+1)*m;
                km2 = (k+2)*m;
                km3 = (k+3)*m;


                b00 = _mm_load1_ps( t + km0 );
                b01 = _mm_load1_ps( t + km1 );
                b02 = _mm_load1_ps( t + km2 );
                b03 = _mm_load1_ps( t + km3 );


                for(i = 0; i<e;i+=4){
                    s = A+i;
                    c00 = _mm_loadu_ps( c0+i );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b00 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km1 ), b01 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km2 ), b02 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km3 ), b03 ) );


                    _mm_storeu_ps( c0+i, c00 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj0) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj0) += *(A+i+km1) * (*(t+km1));
                    *(C+i+mj0) += *(A+i+km2) * (*(t+km2));
                    *(C+i+mj0) += *(A+i+km3) * (*(t+km3));

                }
            }

            for (k =w; k<n; k++){
                km0 = k*m;
                b00 = _mm_load1_ps( t + km0 );

                for(i = 0; i<e;i+=4){
                    s = A+i;
                    c00 = _mm_loadu_ps( c0+i );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b00 ) );

                    _mm_storeu_ps( c0 + i, c00 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj0) += *(A+i+km0) * (*(t+km0));
                }
            }
        }
}
