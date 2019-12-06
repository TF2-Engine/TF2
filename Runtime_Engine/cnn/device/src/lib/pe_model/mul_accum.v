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


module mul_accum(
	input clock,
	input resetn,
	input ivalid,
	input iready,
	output ovalid,
	output oready,
	
	input [255:0]feature_values,
	input [255:0]filter_values,
	output[31:0]dot_accum);
  
  wire aclr0       = {~resetn};
  wire aclr1       = {~resetn};
  wire clk0        = {clock};
  wire clk1        = {clock};
  wire clk2        = {clock};
  wire [2:0] ena        = {3{1'b1}};
  wire [31:0]data[0:7];
	
  efi_16x16_signed_dot_product mul0 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[255:240]),
	.ay(filter_values[255:240]),
	.bx(feature_values[239:224]),
	.by(filter_values[239:224]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[0])
	);
	efi_16x16_signed_dot_product mul1 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[223:208]),
	.ay(filter_values[223:208]),
	.bx(feature_values[207:192]),
	.by(filter_values[207:192]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[1])
	);
  efi_16x16_signed_dot_product mul2 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[191:176]),
	.ay(filter_values[191:176]),
	.bx(feature_values[175:160]),
	.by(filter_values[175:160]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[2])
	);
  efi_16x16_signed_dot_product mul3 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[159:144]),
	.ay(filter_values[159:144]),
	.bx(feature_values[143:128]),
	.by(filter_values[143:128]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[3])
	);
  efi_16x16_signed_dot_product mul4 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[127:112]),
	.ay(filter_values[127:112]),
	.bx(feature_values[111:96]),
	.by(filter_values[111:96]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[4])
	);
  efi_16x16_signed_dot_product mul5 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[95:80]),
	.ay(filter_values[95:80]),
	.bx(feature_values[79:64]),
	.by(filter_values[79:64]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[5])
	);
  efi_16x16_signed_dot_product mul6 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[63:48]),
	.ay(filter_values[63:48]),
	.bx(feature_values[47:32]),
	.by(filter_values[47:32]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[6])
	);
  efi_16x16_signed_dot_product mul7 (
	.aclr0(aclr0),
	.aclr1(aclr1),
	.ax(feature_values[31:16]),
	.ay(filter_values[31:16]),
	.bx(feature_values[15:0]),
	.by(filter_values[15:0]),
	.clk0(clk0),
	.clk1(clk1),
  .clk2(clk2),
	.ena(ena),
	.resulta(data[7])
	);

	assign dot_accum = data[0] + data[1] + data[2] + data[3] + data[4] + data[5] + data[6] + data[7];
	assign ovalid = 1'b1;
	assign oready = 1'b1;
	
endmodule
	
