#ifndef __CNN_WINOGRAD_H__
#define __CNN_WINOGRAD_H__

//////////////////////////////////////////
//                                      //
// F(2x2, 3x3) Winograd Transformations //
//                                      //
//////////////////////////////////////////

#define WGD_2x2_3x3_INPUT_ROWS 4
#define WGD_2x2_3x3_INPUT_COLUMNS 4

#define WGD_2x2_3x3_FILTER_ROWS 3
#define WGD_2x2_3x3_FILTER_COLUMNS 3

#define WGD_2x2_3x3_OUTPUT_ROWS 2
#define WGD_2x2_3x3_OUTPUT_COLUMNS 2

#define WGD_2x2_3x3_B_ROWS 4
#define WGD_2x2_3x3_B_COLUMNS 4
CONSTANT float WGD_2x2_3x3_B[WGD_2x2_3x3_B_ROWS][WGD_2x2_3x3_B_COLUMNS] = {
  {  1, 0,  0,  0 },
  {  0, 1, -1,  1 },
  { -1, 1,  1,  0 },
  {  0, 0,  0, -1 }
};

#define WGD_2x2_3x3_G_ROWS 4
#define WGD_2x2_3x3_G_COLUMNS 3
CONSTANT float WGD_2x2_3x3_G[WGD_2x2_3x3_G_ROWS][WGD_2x2_3x3_G_COLUMNS] = {
  { 1.0,    0,   0 },
  { 0.5,  0.5, 0.5 },
  { 0.5, -0.5, 0.5 },
  {   0,    0, 1.0 }
};

#define WGD_2x2_3x3_A_ROWS 4
#define WGD_2x2_3x3_A_COLUMNS 2
CONSTANT float WGD_2x2_3x3_A[WGD_2x2_3x3_A_ROWS][WGD_2x2_3x3_A_COLUMNS] = {
  { 1,  0 },
  { 1,  1 },
  { 1, -1 },
  { 0, -1 }
};

#define WINOGRAD_MATRIX_TRANSPOSE(X, Z, X_rows, X_columns)    \
  _Pragma("unroll")                                           \
  for(int r = 0; r < X_rows; r++) {                           \
    _Pragma("unroll")                                         \
    for(int c = 0; c < X_columns; c++) {                      \
      Z[c][r] = X[r][c];                                      \
    }                                                         \
  }

#define WINOGRAD_MATRIX_MULTIPLY(X, Y, Z, X_rows, X_columns, Y_columns) \
  _Pragma("unroll")                                                     \
  for(int r = 0; r < X_rows; r++) {                                     \
    _Pragma("unroll")                                                   \
    for(int c = 0; c < Y_columns; c++) {                                \
      float sum = 0;                                                    \
      _Pragma("unroll")                                                 \
      for(int i = 0; i < X_columns; i++) {                              \
        sum += X[r][i] * Y[i][c];                                       \
      }                                                                 \
      Z[r][c] = sum;                                                    \
    }                                                                   \
  }

#define WGD_2x2_3x3_TRANSFORM_FILTER(filter_in, filter_out)   \
      float G1[WGD_2x2_3x3_G_ROWS][WGD_2x2_3x3_FILTER_COLUMNS];         \
      WINOGRAD_MATRIX_MULTIPLY(WGD_2x2_3x3_G, filter_in, G1, WGD_2x2_3x3_G_ROWS, WGD_2x2_3x3_G_COLUMNS, WGD_2x2_3x3_FILTER_COLUMNS); \
      float GT[WGD_2x2_3x3_G_COLUMNS][WGD_2x2_3x3_G_ROWS];              \
      WINOGRAD_MATRIX_TRANSPOSE(WGD_2x2_3x3_G, GT, WGD_2x2_3x3_G_ROWS, WGD_2x2_3x3_G_COLUMNS); \
      WINOGRAD_MATRIX_MULTIPLY(G1, GT, filter_out, WGD_2x2_3x3_G_ROWS, WGD_2x2_3x3_FILTER_COLUMNS, WGD_2x2_3x3_G_ROWS);

#define WGD_2x2_3x3_TRANSFORM_INPUT(input_in, input_out)                \
      float BT[WGD_2x2_3x3_B_COLUMNS][WGD_2x2_3x3_B_ROWS];              \
      WINOGRAD_MATRIX_TRANSPOSE(WGD_2x2_3x3_B, BT, WGD_2x2_3x3_B_ROWS, WGD_2x2_3x3_B_COLUMNS); \
      float B1[WGD_2x2_3x3_B_COLUMNS][WGD_2x2_3x3_INPUT_COLUMNS];       \
      WINOGRAD_MATRIX_MULTIPLY(BT, input_in, B1, WGD_2x2_3x3_B_COLUMNS, WGD_2x2_3x3_B_ROWS, WGD_2x2_3x3_INPUT_COLUMNS); \
      WINOGRAD_MATRIX_MULTIPLY(B1, WGD_2x2_3x3_B, input_out, WGD_2x2_3x3_BT_ROWS, WGD_2x2_3x3_INPUT_COLUMNS, WGD_2x2_3x3_B_COLUMNS);
      
#define WGD_2x2_3x3_POINTWISE_MULTIPLY(input, filter, output)           \
  _Pragma("unroll")                                                     \
  for(int r = 0; r < WGD_2x2_3x3_G_ROWS; r++) {                         \
    _Pragma("unroll")                                                   \
    for(int c = 0; c < WGD_2x2_3x3_G_ROWS; c++) {                       \
      output[r][c] = input[r][c] * filter[r][c];                        \
    }                                                                   \
  }

#define WGD_2x2_3x3_TRANSFORM_OUTPUT(output_in, output_out)             \
  float AT[WGD_2x2_3x3_A_COLUMNS][WGD_2x2_3x3_A_ROWS];                  \
  WINOGRAD_MATRIX_TRANSPOSE(WGD_2x2_3x3_A, AT, WGD_2x2_3x3_A_ROWS, WGD_2x2_3x3_A_COLUMNS); \
  float M1[WGD_2x2_3x3_A_COLUMNS][WGD_2x2_3x3_G_ROWS];                  \
  WINOGRAD_MATRIX_MULTIPLY(AT, output_in, M1, WGD_2x2_3x3_A_COLUMNS, WGD_2x2_3x3_A_ROWS, WGD_2x2_3x3_G_ROWS); \
  WINOGRAD_MATRIX_MULTIPLY(M1, WGD_2x2_3x3_A, output_out, WGD_2x2_3x3_A_COLUMNS, WGD_2x2_3x3_G_ROWS, WGD_2x2_3x3_A_COLUMNS);

/////////////////////////////////////
//                                 //
// F(4,3) Winograd Transformations //
//                                 //
/////////////////////////////////////

#define WGD_4x3_B_ROWS 6
#define WGD_4x3_B_COLUMNS 6
CONSTANT float WGD_4x3_B[WGD_4x3_B_ROWS][WGD_4x3_B_COLUMNS] = {
  {  4,  0,  0,  0,  0,  0 },
  {  0, -4,  4, -2,  2,  4 },
  { -5, -4, -4, -1, -1,  0 },
  {  0,  1, -1,  2, -2, -5 },
  {  1,  1,  1,  1,  1,  0 },
  {  0,  0,  0,  0,  0,  1 }
};

#define WGD_4x3_G_ROWS 6
#define WGD_4x3_G_COLUMNS 3
CONSTANT float WGD_4x3_G[WGD_4x3_G_ROWS][WGD_4x3_G_COLUMNS] = {
  {  (float) 1/(float) 4,                   0,                  0 },
  {  (float)-1/(float) 6, (float)-1/(float) 6, (float)-1/(float)6 },
  {  (float)-1/(float) 6, (float) 1/(float) 6, (float)-1/(float)6 },
  {  (float) 1/(float)24, (float) 1/(float)12, (float) 1/(float)6 },
  {  (float) 1/(float)24, (float)-1/(float)12, (float) 1/(float)6 },
  {                    0,                   0,                  1 }
};

#define WGD_4x3_A_ROWS 6
#define WGD_4x3_A_COLUMNS 4
CONSTANT float WGD_4x3_A[WGD_4x3_A_ROWS][WGD_4x3_A_COLUMNS] = {
  { 1,  0, 0,  0 },
  { 1,  1, 1,  1 },
  { 1, -1, 1, -1 },
  { 1,  2, 4,  8 },
  { 1, -2, 4, -8 },
  { 0,  0, 0,  1 }
};

#define WGD_4x3_INPUT_ROWS 6
#define WGD_4x3_INPUT_COLUMNS 1

#define WGD_4x3_FILTER_ROWS 3
#define WGD_4x3_FILTER_COLUMNS 1

#define WGD_4x3_OUTPUT_ROWS 4
#define WGD_4x3_OUTPUT_COLUMNS 1

/*
#define WGD_4x3_TRANSFORM_FILTER(filter_in, filter_out)   \
  WINOGRAD_MATRIX_MULTIPLY(WGD_4x3_G, filter_in, filter_out, WGD_4x3_G_ROWS, WGD_4x3_G_COLUMNS, WGD_4x3_FILTER_COLUMNS);
*/

STATIC void wgd_4x3_transform_filter(
    float input[WGD_4x3_FILTER_ROWS][WGD_4x3_FILTER_COLUMNS],
    float output[WGD_4x3_G_ROWS][WGD_4x3_INPUT_COLUMNS]) {

  float in_a = input[0][0];
  float in_b = input[1][0];
  float in_c = input[2][0];

  float one_over_4  = ( 1.0f / 4.0f );
  float one_over_6  = ( 1.0f / 6.0f );
  float minus_one_over_6  = ( -1.0f / 6.0f );
  float one_over_12 = ( 1.0f / 12.0f );
  float one_over_24 = ( 1.0f / 24.0f );
  float a_plus_4c = one_over_24 * in_a + one_over_6 * in_c;
  float a_plus_c = minus_one_over_6 * in_a + minus_one_over_6 * in_c;
  float out_a = one_over_4 * in_a;
  float out_b = a_plus_c + minus_one_over_6 * in_b;
  float out_c = a_plus_c + one_over_6 * in_b;
  float out_d = a_plus_4c + one_over_12 * in_b;
  float out_e = a_plus_4c - one_over_12 * in_b;
  float out_f = in_c;

  output[0][0] = out_a;
  output[1][0] = out_b;
  output[2][0] = out_c;
  output[3][0] = out_d;
  output[4][0] = out_e;
  output[5][0] = out_f;
}

/*
#define WGD_4x3_TRANSFORM_INPUT(input_in, input_out)                    \
  float BT[WGD_4x3_B_COLUMNS][WGD_4x3_B_ROWS];                          \
  WINOGRAD_MATRIX_TRANSPOSE(WGD_4x3_B, BT, WGD_4x3_B_ROWS, WGD_4x3_B_COLUMNS);   \
  WINOGRAD_MATRIX_MULTIPLY(BT, input_in, input_out, WGD_4x3_B_COLUMNS, WGD_4x3_B_ROWS, WGD_4x3_INPUT_COLUMNS);
*/

STATIC void wgd_4x3_transform_input(
    real input[WGD_4x3_INPUT_ROWS][WGD_4x3_INPUT_COLUMNS],
    Mreal output[WGD_4x3_B_COLUMNS][WGD_4x3_INPUT_COLUMNS]) {

  real in_a = input[0][0];
  real in_b = input[1][0];
  real in_c = input[2][0];
  real in_d = input[3][0];
  real in_e = input[4][0];
  real in_f = input[5][0];
  //printf("winograd - in_a=%f in_b=%f in_c=%f in_d=%f in_e=%f in_f=%f\n",
  //        in_a, in_b, in_c, in_d, in_e, in_f);

  Mreal fourb_minus_d = 4 * in_b - in_d;
  Mreal b_minus_d = in_b - in_d;
  Mreal e_minus_fourc = in_e - 4 * in_c;
  Mreal e_minus_c = in_e - in_c;
  Mreal out_a = 4 * in_a + e_minus_fourc - in_c;
  Mreal out_b = e_minus_fourc - fourb_minus_d;
  Mreal out_c = e_minus_fourc + fourb_minus_d;
  Mreal out_d = e_minus_c - 2  * b_minus_d;
  Mreal out_e = e_minus_c + 2  * b_minus_d;
  Mreal out_f = 4 * in_b - 5 * in_d + in_f;

  //printf("winograd - out_a=%f out_b=%f out_c=%f out_d=%f out_e=%f out_f=%f\n",
  //        out_a, out_b, out_c, out_d, out_e, out_f);

  output[0][0] = out_a;
  output[1][0] = out_b;
  output[2][0] = out_c;
  output[3][0] = out_d;
  output[4][0] = out_e;
  output[5][0] = out_f;
  //printf("winograd - out_a=%f out_b=%f out_c=%f out_d=%f out_e=%f out_f=%f\n",
  //        output[0][0], output[1][0], output[2][0], output[3][0], output[4][0], output[5][0]);

  
}

#define WGD_4x3_POINTWISE_MULTIPLY(input, filter, output)           \
  _Pragma("unroll")                                                 \
  for(int r = 0; r < WGD_4x3_G_ROWS; r++) {                         \
    _Pragma("unroll")                                               \
    for(int c = 0; c < WGD_4x3_FILTER_COLUMNS; c++) {               \
      output[r][c] = input[r][c] * filter[r][c];                    \
    }                                                               \
  }

/*
#define WGD_4x3_TRANSFORM_OUTPUT(output_in, output_out)                 \
  float AT[WGD_4x3_A_COLUMNS][WGD_4x3_A_ROWS];                          \
  WINOGRAD_MATRIX_TRANSPOSE(WGD_4x3_A, AT, WGD_4x3_A_ROWS, WGD_4x3_A_COLUMNS);   \
  WINOGRAD_MATRIX_MULTIPLY(AT, output_in, output_out, WGD_4x3_A_COLUMNS, WGD_4x3_A_ROWS, WGD_4x3_FILTER_COLUMNS);
*/

/*inline real GET(Mreal data)
{
  if(data<=0) 
  return 0;
  Mreal trans=((data>>LOWER)+1)>>1;
  Mreal trans = data;
  real pk = trans > REALMAX ? REALMAX:
            trans < REALMIN ? REALMIN:
            (CHAR)trans;
  return pk;
}*/

inline real GET(Mreal data)
{
  real pk = data > REALMAX ? REALMAX:
            data < REALMIN ? REALMIN:
            (CHAR)data;
  return pk;
}

inline Sreal GETS(Mreal data)
{
  Sreal pk = data > SREALMAX ? SREALMAX:
            data < SREALMIN ? SREALMIN:
            (Sreal)data;
  return pk;
}

STATIC void wgd_4x3_transform_output(
    Mreal input[WGD_4x3_A_ROWS][WGD_4x3_FILTER_COLUMNS],
    real output[WGD_4x3_A_COLUMNS][WGD_4x3_FILTER_COLUMNS]) {

  Mreal in_a = input[0][0];
  Mreal in_b = input[1][0];
  Mreal in_c = input[2][0];
  Mreal in_d = input[3][0];
  Mreal in_e = input[4][0];
  Mreal in_f = input[5][0];

  // apply filter post transform
  Mreal b_plus_c = in_b + in_c;
  Mreal b_minus_c = in_b - in_c;
  Mreal d_plus_e = in_d + in_e;
  Mreal d_minus_e = in_d - in_e;
  Mreal out_a = in_a + b_plus_c + d_plus_e;
  Mreal out_b = b_minus_c + 2 * d_minus_e;
  Mreal out_c = b_plus_c + 4 * d_plus_e;
  Mreal out_d = b_minus_c + 8 * d_minus_e + in_f;

  output[0][0] = GET(out_a);
  output[1][0] = GET(out_b);
  output[2][0] = GET(out_c);
  output[3][0] = GET(out_d);
}

/////////////////////////////////////
//                                 //
// F(5,3) Winograd Transformations //
//                                 //
/////////////////////////////////////

#define WINO_INFLAT 120

#define WGD_5x3_BT_ROWS 7
#define WGD_5x3_BT_COLUMNS 7

CONSTANT float WGD_5x3_BT[WGD_5x3_BT_ROWS][WGD_5x3_BT_COLUMNS] = {
{ 12,  -4 ,  -15,  5 ,  3 ,  -1,  0 },
{ 0 ,  12 ,   8 ,  -7,  -2,  1 ,  0 },
{ 0 ,  -12,  16 ,  -1,  -4,  1 ,  0 },
{ 0 ,   6 ,   1 ,  -7,  -1,  1 ,  0 },
{ 0 ,  -6 ,   5 ,  5 ,  -5,  1 ,  0 },
{ 0 ,   4 ,   0 ,  -5,  0 ,  1 ,  0 },
{ 0 ,  -12,   4 ,  15,  -5,  -3,  1 }
};

#define WGD_5x3_G_ROWS 7
#define WGD_5x3_G_COLUMNS 3
CONSTANT float WGD_5x3_G[WGD_5x3_G_ROWS][WGD_5x3_G_COLUMNS] = {
  {  (float) 1/(float) 12,                    0,                   0 },
  {  (float) 1/(float) 12, (float) 1/(float) 12, (float) 1/(float)12 },
  {  (float) 1/(float) 24, (float)-1/(float) 24, (float) 1/(float)24 },
  {  (float)-1/(float) 24, (float)-1/(float) 12, (float)-1/(float) 6 },
  {  (float)-1/(float)120, (float) 1/(float) 60, (float)-1/(float)30 },
  {  (float) 1/(float)120, (float) 1/(float) 40, (float) 3/(float)40 },
  {                     0,                    0,                   1 }
};

#define WGD_5x3_AT_ROWS 5
#define WGD_5x3_AT_COLUMNS 7
CONSTANT float WGD_5x3_AT[WGD_5x3_AT_ROWS][WGD_5x3_AT_COLUMNS] = {
  { 1,  1,  1,  1,  1,  1,  0 },
  { 0,  1, -1,  2, -2,  3,  0 },
  { 0,  1,  1,  4,  4,  9,  0 },
  { 0,  1, -1,  8, -8, 27,  0 },
  { 0,  1,  1, 16, 16, 81,  1 },
};

#define WGD_5x3_INPUT_ROWS 7
#define WGD_5x3_INPUT_COLUMNS 1

#define WGD_5x3_FILTER_ROWS 3
#define WGD_5x3_FILTER_COLUMNS 1

#define WGD_5x3_OUTPUT_ROWS 7
#define WGD_5x3_OUTPUT_COLUMNS 1

/*
#define WGD_4x3_TRANSFORM_FILTER(filter_in, filter_out)   \
  WINOGRAD_MATRIX_MULTIPLY(WGD_4x3_G, filter_in, filter_out, WGD_4x3_G_ROWS, WGD_4x3_G_COLUMNS, WGD_4x3_FILTER_COLUMNS);
*/

STATIC void wgd_5x3_transform_filter(
    real input[WGD_5x3_FILTER_ROWS][WGD_5x3_FILTER_COLUMNS],
    Sreal output[WGD_5x3_G_ROWS][WGD_5x3_FILTER_COLUMNS] ) {

  real in_a = input[0][0];
  real in_b = input[1][0];
  real in_c = input[2][0];

  // float  f_1_6   = (  1.0f /   6.0f );
  // float  f_1_12  = (  1.0f /  12.0f );
  // float  f_1_24  = (  1.0f /  24.0f );
  // float  f_1_30  = (  1.0f /  30.0f );
  // float  f_1_40  = (  1.0f /  40.0f );
  // float  f_1_60  = (  1.0f /  60.0f );
  // float  f_1_120 = (  1.0f / 120.0f );
  
  Mreal a_c = in_a + in_c;
  
  // float out_a = f_1_12 * in_a;
  // float out_b = f_1_12 * ( a_c + in_b );
  // float out_c = f_1_24 * ( a_c - in_b ) ;
  // float out_d = - f_1_24 * in_a - f_1_12 * in_b - f_1_6 * in_c;
  // float out_e = - f_1_120 * in_a + f_1_60 * in_b - f_1_30 * in_c;
  // float out_f = f_1_120 * in_a + f_1_40 * ( in_b + 3 * in_c );
  // float out_g = in_c;

  Mreal out_a = 10 * in_a;
  Mreal out_b = 10 * ( a_c + in_b );
  Mreal out_c = 5 * ( a_c - in_b ) ;
  Mreal out_d = - 5 * in_a - 10 * in_b - 20 * in_c;
  Mreal out_e = - in_a + 2 * in_b - 4 * in_c;
  Mreal out_f = in_a + 3 * ( in_b + 3 * in_c );
  Mreal out_g = 120 * in_c;
  
  output[0][0] = GETS(out_a);
  output[1][0] = GETS(out_b);
  output[2][0] = GETS(out_c);
  output[3][0] = GETS(out_d);
  output[4][0] = GETS(out_e);
  output[5][0] = GETS(out_f);
  output[6][0] = GETS(out_g);
}

/*
#define WGD_4x3_TRANSFORM_INPUT(input_in, input_out)                    \
  float BT[WGD_4x3_B_COLUMNS][WGD_4x3_B_ROWS];                          \
  WINOGRAD_MATRIX_TRANSPOSE(WGD_4x3_B, BT, WGD_4x3_B_ROWS, WGD_4x3_B_COLUMNS);   \
  WINOGRAD_MATRIX_MULTIPLY(BT, input_in, input_out, WGD_4x3_B_COLUMNS, WGD_4x3_B_ROWS, WGD_4x3_INPUT_COLUMNS);
*/

STATIC void wgd_5x3_transform_input(
    real input[WGD_5x3_INPUT_ROWS][WGD_5x3_INPUT_COLUMNS],
    Sreal output[WGD_5x3_BT_ROWS][WGD_5x3_INPUT_COLUMNS]) {

  real in_a = input[0][0];
  real in_b = input[1][0];
  real in_c = input[2][0];
  real in_d = input[3][0];
  real in_e = input[4][0];
  real in_f = input[5][0];
  real in_g = input[6][0];
  //printf("winograd - in_a=%f in_b=%f in_c=%f in_d=%f in_e=%f in_f=%f\n",
  //        in_a, in_b, in_c, in_d, in_e, in_f);

  Mreal f__7d_f = - 7 * in_d + in_f;
  Mreal f_4b    =   4 * in_b;
  Mreal f_6b    =   6 * in_b;
  Mreal f_12b   =  12 * in_b;
  Mreal f_5d    =   5 * in_d;
  Mreal f_5e    =   5 * in_e;

  Mreal out_a =   12 * in_a -      f_4b - 15 * in_c +      f_5d +  3 * in_e -      in_f;
  Mreal out_b =                   f_12b +  8 * in_c             -  2 * in_e              + f__7d_f;
  Mreal out_c =             -     f_12b + 16 * in_c -      in_d -  4 * in_e +      in_f;
  Mreal out_d =                    f_6b +      in_c             -      in_e              + f__7d_f;
  Mreal out_e =             -      f_6b +  5 * in_c +      f_5d -      f_5e +      in_f;
  Mreal out_f =                    f_4b             -      f_5d             +      in_f;
  Mreal out_g =             -     f_12b +  4 * in_c + 15 * in_d -      f_5e -  3 * in_f  + in_g;

  //printf("winograd - out_a=%f out_b=%f out_c=%f out_d=%f out_e=%f out_f=%f\n",
  //        out_a, out_b, out_c, out_d, out_e, out_f);

  output[0][0] = GETS(out_a);
  output[1][0] = GETS(out_b);
  output[2][0] = GETS(out_c);
  output[3][0] = GETS(out_d);
  output[4][0] = GETS(out_e);
  output[5][0] = GETS(out_f);
  output[6][0] = GETS(out_g);
  //printf("winograd - out_a=%f out_b=%f out_c=%f out_d=%f out_e=%f out_f=%f\n",
  //        output[0][0], output[1][0], output[2][0], output[3][0], output[4][0], output[5][0]);
}

#define WGD_4x3_POINTWISE_MULTIPLY(input, filter, output)           \
  _Pragma("unroll")                                                 \
  for(int r = 0; r < WGD_4x3_G_ROWS; r++) {                         \
    _Pragma("unroll")                                               \
    for(int c = 0; c < WGD_4x3_FILTER_COLUMNS; c++) {               \
      output[r][c] = input[r][c] * filter[r][c];                    \
    }                                                               \
  }

/*
#define WGD_4x3_TRANSFORM_OUTPUT(output_in, output_out)                 \
  float AT[WGD_4x3_A_COLUMNS][WGD_4x3_A_ROWS];                          \
  WINOGRAD_MATRIX_TRANSPOSE(WGD_4x3_A, AT, WGD_4x3_A_ROWS, WGD_4x3_A_COLUMNS);   \
  WINOGRAD_MATRIX_MULTIPLY(AT, output_in, output_out, WGD_4x3_A_COLUMNS, WGD_4x3_A_ROWS, WGD_4x3_FILTER_COLUMNS);
*/

STATIC void wgd_5x3_transform_output(
    Mreal input[WGD_5x3_OUTPUT_ROWS][WGD_5x3_OUTPUT_COLUMNS],
    Mreal output[WGD_5x3_AT_ROWS][WGD_5x3_OUTPUT_COLUMNS]) {

  Mreal in_a = input[0][0];
  Mreal in_b = input[1][0];
  Mreal in_c = input[2][0];
  Mreal in_d = input[3][0];
  Mreal in_e = input[4][0];
  Mreal in_f = input[5][0];
  Mreal in_g = input[6][0];

  Mreal b_c  = in_b + in_c;
  Mreal b__c = in_b - in_c;
  Mreal d_e  = in_d + in_e;
  Mreal d__e = in_d - in_e;

  Mreal out_a = in_a + b_c + d_e + in_f; 
  Mreal out_b = b__c + 2 * d__e + 3 * in_f; 
  Mreal out_c = b_c + 4 * d_e + 9 * in_f;
  Mreal out_d = b__c + 8 * d__e + 27 * in_f;
  Mreal out_e = b_c + 16 * d_e + 81 * in_f + in_g;

  output[0][0] = out_a;
  output[1][0] = out_b;
  output[2][0] = out_c;
  output[3][0] = out_d;
  output[4][0] = out_e;
}

#endif
