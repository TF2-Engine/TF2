
#ifndef OPENCL
#define OPENCL
#endif

#define Mreal int
#define real char

#ifdef OPENCL
#define STATIC
#define CONSTANT constant
#else
#define STATIC static
#define CONSTANT static const
#endif


#define BIT_IS_SET(value, bit_num) (((value) & (1ULL << (bit_num))) != 0)

inline Mreal MUL(real fea,real fil){
 if(BIT_IS_SET(fil,6))
  return 0;
 if(BIT_IS_SET(fil,7))
  fea=-fea;
 fil=0x1f&fil;
  Mreal data=fea<<fil;
 return data;
}

STATIC Mreal DotProduct(uchar16 feature_values, uchar16 filter_values ){
  int dot_accum = 0; // change from long int to int

  dot_accum += MUL(feature_values.s0,filter_values.s0);
  dot_accum += MUL(feature_values.s1,filter_values.s1);
  dot_accum += MUL(feature_values.s2,filter_values.s2);
  dot_accum += MUL(feature_values.s3,filter_values.s3);
  dot_accum += MUL(feature_values.s4,filter_values.s4);
  dot_accum += MUL(feature_values.s5,filter_values.s5);
  dot_accum += MUL(feature_values.s6,filter_values.s6);
  dot_accum += MUL(feature_values.s7,filter_values.s7);
  dot_accum += MUL(feature_values.s8,filter_values.s8);
  dot_accum += MUL(feature_values.s9,filter_values.s9);
  dot_accum += MUL(feature_values.sa,filter_values.sa);
  dot_accum += MUL(feature_values.sb,filter_values.sb);
  dot_accum += MUL(feature_values.sc,filter_values.sc);
  dot_accum += MUL(feature_values.sd,filter_values.sd);
  dot_accum += MUL(feature_values.se,filter_values.se);
  dot_accum += MUL(feature_values.sf,filter_values.sf);

  return dot_accum;
}
