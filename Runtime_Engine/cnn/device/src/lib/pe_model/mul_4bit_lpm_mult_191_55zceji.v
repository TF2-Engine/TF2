// (C) 2001-2019 Intel Corporation. All rights reserved.
// Your use of Intel Corporation's design tools, logic functions and other 
// software and tools, and its AMPP partner logic functions, and any output 
// files from any of the foregoing (including device programming or simulation 
// files), and any associated documentation or information are expressly subject 
// to the terms and conditions of the Intel Program License Subscription 
// Agreement, Intel FPGA IP License Agreement, or other applicable 
// license agreement, including, without limitation, that your use is for the 
// sole purpose of programming logic devices manufactured by Intel and sold by 
// Intel or its authorized distributors.  Please refer to the applicable 
// agreement for further details.


// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module  mul_4bit_lpm_mult_191_55zceji  (
            aclr,
            clock,
            dataa,
            datab,
            result);

            input  aclr;
            input  clock;
            input [3:0] dataa;
            input [3:0] datab;
            output [7:0] result;

            wire [7:0] sub_wire0;
            wire [7:0] result = sub_wire0[7:0];    

            lpm_mult        lpm_mult_component (
                                        .aclr (aclr),
                                        .clock (clock),
                                        .dataa (dataa),
                                        .datab (datab),
                                        .result (sub_wire0),
                                        .clken (1'b1),
                                        .sclr (1'b0),
                                        .sum (1'b0));
            defparam
                    lpm_mult_component.lpm_hint = "MAXIMIZE_SPEED=5",
                    lpm_mult_component.lpm_pipeline = 1,
                    lpm_mult_component.lpm_representation = "UNSIGNED",
                    lpm_mult_component.lpm_type = "LPM_MULT",
                    lpm_mult_component.lpm_widtha = 4,
                    lpm_mult_component.lpm_widthb = 4,
                    lpm_mult_component.lpm_widthp = 8;


endmodule


