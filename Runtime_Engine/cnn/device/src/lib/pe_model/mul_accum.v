/* Copyright 2019 Inspur Corporation. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


module mul_add(
	input clock,
	input resetn,
	input ivalid,
	input iready,
	output ovalid,
	output oready,
	
	input [127:0]feature_values,
	input [127:0]filter_values,
	output [31:0]dot_accum);
  
  wire [31:0]data[0:15];
  
  wire [31:0]feature0, shift0;
  assign feature0 = {32{~filter_values[6]}} & {{24{feature_values[7]}},feature_values[7:0]};
  assign shift0 = feature0 << filter_values[4:0];
  assign data[0] = filter_values[7] ? (~shift0[31:0] + 1) : shift0[31:0];
  
  wire [31:0]feature1, shift1;
  assign feature1 = {32{~filter_values[14]}} & {{24{feature_values[15]}},feature_values[15:8]};
  assign shift1 = feature1 << filter_values[12:8];
  assign data[1] = filter_values[15] ? (~shift1[31:0] + 1) : shift1[31:0];
  
  wire [31:0]feature2, shift2;
  assign feature2 = {32{~filter_values[22]}} & {{24{feature_values[23]}},feature_values[23:16]};
  assign shift2 = feature2 << filter_values[24:16];
  assign data[2] = filter_values[23] ? (~shift2[31:0] + 1) : shift2[31:0];
  
    wire [31:0]feature3, shift3;
  assign feature3 = {32{~filter_values[30]}} & {{24{feature_values[31]}},feature_values[31:24]};
  assign shift3 = feature3 << filter_values[28:24];
  assign data[3] = filter_values[31] ? (~shift3[31:0] + 1) : shift3[31:0];
  
    wire [31:0]feature4, shift4;
  assign feature4 = {32{~filter_values[38]}} & {{24{feature_values[39]}},feature_values[39:32]};
  assign shift4 = feature4 << filter_values[36:32];
  assign data[4] = filter_values[39] ? (~shift4[31:0] + 1) : shift4[31:0];
  
    wire [31:0]feature5, shift5;
  assign feature5 = {32{~filter_values[46]}} & {{24{feature_values[47]}},feature_values[47:40]};
  assign shift5 = feature5 << filter_values[44:40];
  assign data[5] = filter_values[47] ? (~shift5[31:0] + 1) : shift5[31:0];
  
    wire [31:0]feature6, shift6;
  assign feature6 = {32{~filter_values[54]}} & {{24{feature_values[55]}},feature_values[55:48]};
  assign shift6 = feature6 << filter_values[52:48];
  assign data[6] = filter_values[55] ? (~shift6[31:0] + 1) : shift6[31:0];
  
    wire [31:0]feature7, shift7;
  assign feature7 = {32{~filter_values[62]}} & {{24{feature_values[63]}},feature_values[63:56]};
  assign shift7 = feature7 << filter_values[60:56];
  assign data[7] = filter_values[63] ? (~shift0[31:0] + 1) : shift7[31:0];
  
    wire [31:0]feature8, shift8;
  assign feature8 = {32{~filter_values[70]}} & {{24{feature_values[71]}},feature_values[71:64]};
  assign shift8 = feature8 << filter_values[68:64];
  assign data[8] = filter_values[71] ? (~shift8[31:0] + 1) : shift8[31:0];
  
    wire [31:0]feature9, shift9;
  assign feature9 = {32{~filter_values[78]}} & {{24{feature_values[79]}},feature_values[79:72]};
  assign shift9 = feature9 << filter_values[76:72];
  assign data[9] = filter_values[79] ? (~shift9[31:0] + 1) : shift9[31:0];
  
    wire [31:0]feature10, shift10;
  assign feature10 = {32{~filter_values[86]}} & {{24{feature_values[87]}},feature_values[87:80]};
  assign shift10 = feature10 << filter_values[84:80];
  assign data[10] = filter_values[87] ? (~shift10[31:0] + 1) : shift10[31:0];
  
    wire [31:0]feature11, shift11;
  assign feature11 = {32{~filter_values[94]}} & {{24{feature_values[95]}},feature_values[95:88]};
  assign shift11 = feature11 << filter_values[92:88];
  assign data[11] = filter_values[95] ? (~shift11[31:0] + 1) : shift11[31:0];
  
    wire [31:0]feature12, shift12;
  assign feature12 = {32{~filter_values[102]}} & {{24{feature_values[103]}},feature_values[103:96]};
  assign shift12 = feature12 << filter_values[100:96];
  assign data[12] = filter_values[103] ? (~shift12[31:0] + 1) : shift12[31:0];
  
    wire [31:0]feature13, shift13;
  assign feature13 = {32{~filter_values[110]}} & {{24{feature_values[111]}},feature_values[111:104]};
  assign shift13 = feature13 << filter_values[108:104];
  assign data[13] = filter_values[111] ? (~shift13[31:0] + 1) : shift13[31:0];
  
    wire [31:0]feature14, shift14;
  assign feature14 = {32{~filter_values[118]}} & {{24{feature_values[119]}},feature_values[119:112]};
  assign shift14 = feature14 << filter_values[116:112];
  assign data[14] = filter_values[119] ? (~shift14[31:0] + 1) : shift14[31:0];
  
    wire [31:0]feature15, shift15;
  assign feature15 = {32{~filter_values[126]}} & {{24{feature_values[127]}},feature_values[127:120]};
  assign shift15 = feature15 << filter_values[124:120];
  assign data[15] = filter_values[127] ? (~shift15[31:0] + 1) : shift15[31:0];
  
	assign dot_accum = data[0] + data[1] + data[2] + data[3] + data[4] + data[5] + data[6] + data[7] + data[8] + data[9] + data[10] + data[11] + data[12] + data[13] + data[14] + data[15];
	assign ovalid = 1'b1;
	assign oready = 1'b1;
	
endmodule
