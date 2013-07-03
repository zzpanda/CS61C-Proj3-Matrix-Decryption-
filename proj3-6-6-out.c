#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h> /* where intrinsics are defined */
#include <omp.h>

void sgemm( int m, int n, float *A, float *C)
{   
    int e = m/4*4; //for row's edge case
    int f = m/6*6;
    int w = n/6*6;
    int km0,km1,km2,km3,km4,km5,j,k,i,mj0,mj1,mj2,mj3,mj4,mj5;
    __m128 c00,c10,c20,c30,c40,c50,b00,b01,b02,b03,b04,b05,b10,b11,b12,b13,b14,b15,b20,b21,b22,b23,b24,b25,b30,b31,b32,b33,b34,b35,b40,b41,b42,b43,b44,b45,b50,b51,b52,b53,b54,b55,a0,a1,a2,a3,a4,a5;
    float *t;  //b
    float *s;  //a
    float *c0,*c1,*c2,*c3,*c4,*c5;  //c 

        #pragma omp parallel for private(t,s,c0,c1,c2,c3,c4,c5,km0,km1,km2,km3,km4,km5,j,k,i,mj0,mj1,mj2,mj3,mj4,mj5,c00,c10,c20,c30,c40,c50,b00,b01,b02,b03,b04,b05,b10,b11,b12,b13,b14,b15,b20,b21,b22,b23,b24,b25,b30,b31,b32,b33,b34,b35,b40,b41,b42,b43,b44,b45,b50,b51,b52,b53,b54,b55,a0,a1,a2,a3,a4,a5)
        for (j=0; j<f; j+=6) {
            mj0 = m*j;
            mj1 = m*(j+1);
            mj2 = m*(j+2);
            mj3 = m*(j+3);
            mj4 = m*(j+4);
            mj5 = m*(j+5);
            t = A+j;
            c0 = C+mj0;
            c1 = C+mj1;
            c2 = C+mj2;
            c3 = C+mj3;
            c4 = C+mj4;
            c5 = C+mj5;
            //every four column1

            for (k =0; k<w; k+=6){
                km0 = k*m;
                km1 = (k+1)*m;
                km2 = (k+2)*m;
                km3 = (k+3)*m;
                km4 = (k+4)*m;
                km5 = (k+5)*m;

                b00 = _mm_load1_ps( t + km0 );
                b01 = _mm_load1_ps( t + km1 );
                b02 = _mm_load1_ps( t + km2 );
                b03 = _mm_load1_ps( t + km3 );
                b04 = _mm_load1_ps( t + km4 );
                b05 = _mm_load1_ps( t + km5 );

                b10 = _mm_load1_ps( t +1+ km0 );
                b11 = _mm_load1_ps( t +1+ km1 );
                b12 = _mm_load1_ps( t +1+ km2 );
                b13 = _mm_load1_ps( t +1+ km3 );
                b14 = _mm_load1_ps( t +1+ km4 );
                b15 = _mm_load1_ps( t +1+ km5 );

                b20 = _mm_load1_ps( t +2+ km0 );
                b21 = _mm_load1_ps( t +2+  km1 );
                b22 = _mm_load1_ps( t +2+ km2 );
                b23 = _mm_load1_ps( t +2+ km3 );
                b24 = _mm_load1_ps( t +2+ km4 );
                b25 = _mm_load1_ps( t +2+ km5 );

                b30 = _mm_load1_ps( t +3+ km0 );
                b31 = _mm_load1_ps( t +3+ km1 );
                b32 = _mm_load1_ps( t +3+ km2 );
                b33 = _mm_load1_ps( t +3+ km3 );
                b34 = _mm_load1_ps( t +3+ km4 );
                b35 = _mm_load1_ps( t +3+ km5 );

                b40 = _mm_load1_ps( t +4+ km0 );
                b41 = _mm_load1_ps( t +4+ km1 );
                b42 = _mm_load1_ps( t +4+ km2 );
                b43 = _mm_load1_ps( t +4+ km3 );
                b44 = _mm_load1_ps( t +4+ km4 );
                b45 = _mm_load1_ps( t +4+ km5 );

                b50 = _mm_load1_ps( t +5+ km0 );
                b51 = _mm_load1_ps( t +5+ km1 );
                b52 = _mm_load1_ps( t +5+ km2 );
                b53 = _mm_load1_ps( t +5+ km3 );
                b54 = _mm_load1_ps( t +5+ km4 );
                b55 = _mm_load1_ps( t +5+ km5 );

                for(i = 0; i<e;i+=4){
                    s = A+i;
                    a0 = _mm_loadu_ps( s + km0 );
                    a1 = _mm_loadu_ps( s + km1 );
                    a2 = _mm_loadu_ps( s + km2 );
                    a3 = _mm_loadu_ps( s + km3 );
                    a4 = _mm_loadu_ps( s + km4 );
                    a5 = _mm_loadu_ps( s + km5 );
                    
                    c00 = _mm_loadu_ps( c0+i );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a0, b00 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a1, b01 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a2, b02 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a3, b03 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a4, b04 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( a5, b05 ) );
                    _mm_storeu_ps( c0 + i, c00 );

                    c10 = _mm_loadu_ps( c1+i );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a0, b10 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a1, b11 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a2, b12 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a3, b13 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a4, b14 ) );
                    c10 = _mm_add_ps( c10, _mm_mul_ps( a5, b15 ) );
                    _mm_storeu_ps( c1 + i, c10 );
                    
                    c20 = _mm_loadu_ps( c2+i );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a0, b20 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a1, b21 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a2, b22 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a3, b23 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a4, b24 ) );
                    c20 = _mm_add_ps( c20, _mm_mul_ps( a5, b25 ) );
                    _mm_storeu_ps( c2 + i, c20 );

                    c30 = _mm_loadu_ps( c3+i );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a0, b30 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a1, b31 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a2, b32 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a3, b33 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a4, b34 ) );
                    c30 = _mm_add_ps( c30, _mm_mul_ps( a5, b35 ) );
                    _mm_storeu_ps( c3 + i, c30 );

                    c40 = _mm_loadu_ps( c4+i );
                    c40 = _mm_add_ps( c40, _mm_mul_ps( a0, b40 ) );
                    c40 = _mm_add_ps( c40, _mm_mul_ps( a1, b41 ) );
                    c40 = _mm_add_ps( c40, _mm_mul_ps( a2, b42 ) );
                    c40 = _mm_add_ps( c40, _mm_mul_ps( a3, b43 ) );
                    c40 = _mm_add_ps( c40, _mm_mul_ps( a4, b44 ) );
                    c40 = _mm_add_ps( c40, _mm_mul_ps( a5, b45 ) );
                    _mm_storeu_ps( c4 + i, c40 );

                    c50 = _mm_loadu_ps( c5+i );
                    c50 = _mm_add_ps( c50, _mm_mul_ps( a0, b50 ) );
                    c50 = _mm_add_ps( c50, _mm_mul_ps( a1, b51 ) );
                    c50 = _mm_add_ps( c50, _mm_mul_ps( a2, b52 ) );
                    c50 = _mm_add_ps( c50, _mm_mul_ps( a3, b53 ) );
                    c50 = _mm_add_ps( c50, _mm_mul_ps( a4, b54 ) );
                    c50 = _mm_add_ps( c50, _mm_mul_ps( a5, b55 ) );
                    _mm_storeu_ps( c5 + i, c50 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj0) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj1) += *(A+i+km0) * (*(t+1+km0));
                    *(C+i+mj2) += *(A+i+km0) * (*(t+2+km0));
                    *(C+i+mj3) += *(A+i+km0) * (*(t+3+km0));
                    *(C+i+mj4) += *(A+i+km0) * (*(t+4+km0));
                    *(C+i+mj5) += *(A+i+km0) * (*(t+5+km0));

                    *(C+i+mj0) += *(A+i+km1) * (*(t+km1));
                    *(C+i+mj1) += *(A+i+km1) * (*(t+1+km1));
                    *(C+i+mj2) += *(A+i+km1) * (*(t+2+km1));
                    *(C+i+mj3) += *(A+i+km1) * (*(t+3+km1));
                    *(C+i+mj4) += *(A+i+km1) * (*(t+4+km1));
                    *(C+i+mj5) += *(A+i+km1) * (*(t+5+km1));

                    *(C+i+mj0) += *(A+i+km2) * (*(t+km2));
                    *(C+i+mj1) += *(A+i+km2) * (*(t+1+km2));
                    *(C+i+mj2) += *(A+i+km2) * (*(t+2+km2));
                    *(C+i+mj3) += *(A+i+km2) * (*(t+3+km2));
                    *(C+i+mj4) += *(A+i+km2) * (*(t+4+km2));
                    *(C+i+mj5) += *(A+i+km2) * (*(t+5+km2));

                    *(C+i+mj0) += *(A+i+km3) * (*(t+km3));
                    *(C+i+mj1) += *(A+i+km3) * (*(t+1+km3));
                    *(C+i+mj2) += *(A+i+km3) * (*(t+2+km3));
                    *(C+i+mj3) += *(A+i+km3) * (*(t+3+km3));
                    *(C+i+mj4) += *(A+i+km3) * (*(t+4+km3));
                    *(C+i+mj5) += *(A+i+km3) * (*(t+5+km3));

                    *(C+i+mj0) += *(A+i+km4) * (*(t+km4));
                    *(C+i+mj1) += *(A+i+km4) * (*(t+1+km4));
                    *(C+i+mj2) += *(A+i+km4) * (*(t+2+km4));
                    *(C+i+mj3) += *(A+i+km4) * (*(t+3+km4));
                    *(C+i+mj4) += *(A+i+km4) * (*(t+4+km4));
                    *(C+i+mj5) += *(A+i+km4) * (*(t+5+km4));

                    *(C+i+mj0) += *(A+i+km5) * (*(t+km5));
                    *(C+i+mj1) += *(A+i+km5) * (*(t+1+km5));
                    *(C+i+mj2) += *(A+i+km5) * (*(t+2+km5));
                    *(C+i+mj3) += *(A+i+km5) * (*(t+3+km5));
                    *(C+i+mj4) += *(A+i+km5) * (*(t+4+km5));
                    *(C+i+mj5) += *(A+i+km5) * (*(t+5+km5));
                }
            }

            for (k =w; k<n; k++){
                km0 = k*m;
                b00 = _mm_load1_ps( t + km0 );
                b10 = _mm_load1_ps( t +1+ km0 );
                b20 = _mm_load1_ps( t +2+ km0 );
                b30 = _mm_load1_ps( t +3+km0 );
                b40 = _mm_load1_ps( t +4+km0 );
                b50 = _mm_load1_ps( t +5+km0 );

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

                    c40 = _mm_loadu_ps( c4+i );
                    c40 = _mm_add_ps( c40, _mm_mul_ps( a0, b40 ) );
                    _mm_storeu_ps( c4 + i, c40 );

                    c50 = _mm_loadu_ps( c5+i );
                    c50 = _mm_add_ps( c50, _mm_mul_ps( a0, b50 ) );
                    _mm_storeu_ps( c5 + i, c50 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj0) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj1) += *(A+i+km0) * (*(t+1+km0));
                    *(C+i+mj2) += *(A+i+km0) * (*(t+2+km0));
                    *(C+i+mj3) += *(A+i+km0) * (*(t+3+km0));
                    *(C+i+mj4) += *(A+i+km0) * (*(t+4+km0));
                    *(C+i+mj5) += *(A+i+km0) * (*(t+5+km0));
                }
            }
        }
        
        #pragma omp parallel for private(t,s,c0,km0,km1,km2,km3,km4,km5,j,k,i,mj0,c00,b00,b01,b02,b03,b04,b05)
        // edge case
        for (j=f; j<m; j++) {
            mj0 = m*j;
            t = A+j;
            c0 = C+mj0;
            //every four column1

            for (k =0; k<w; k+=6){
                km0 = k*m;
                km1 = (k+1)*m;
                km2 = (k+2)*m;
                km3 = (k+3)*m;
                km4 = (k+4)*m;
                km5 = (k+5)*m;

                b00 = _mm_load1_ps( t + km0 );
                b01 = _mm_load1_ps( t + km1 );
                b02 = _mm_load1_ps( t + km2 );
                b03 = _mm_load1_ps( t + km3 );
                b04 = _mm_load1_ps( t + km4 );
                b05 = _mm_load1_ps( t + km5 );

                for(i = 0; i<e;i+=4){
                    s = A+i;
                    c00 = _mm_loadu_ps( c0+i );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km0 ), b00 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km1 ), b01 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km2 ), b02 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km3 ), b03 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km4 ), b04 ) );
                    c00 = _mm_add_ps( c00, _mm_mul_ps( _mm_loadu_ps( s + km5 ), b05 ) );

                    _mm_storeu_ps( c0+i, c00 );
                }

                for (i = e; i < m; i++) {
                    *(C+i+mj0) += *(A+i+km0) * (*(t+km0));
                    *(C+i+mj0) += *(A+i+km1) * (*(t+km1));
                    *(C+i+mj0) += *(A+i+km2) * (*(t+km2));
                    *(C+i+mj0) += *(A+i+km3) * (*(t+km3));
                    *(C+i+mj0) += *(A+i+km4) * (*(t+km4));
                    *(C+i+mj0) += *(A+i+km5) * (*(t+km5));
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
