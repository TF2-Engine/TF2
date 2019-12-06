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


// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module  efi_16x16_signed_dot_product (
            ax,
            ay,
            bx,
            by,
            ena,
            resulta,
            aclr0,
            aclr1,
            clk0,
            clk1,
            clk2);

            input [15:0] ax;
            input [15:0] ay;
            input [15:0] bx;
            input [15:0] by;
            input [2:0] ena;
            output [31:0] resulta;
            input  aclr0;
            input  aclr1;
            input  clk0;
            input  clk1;
            input  clk2;

            wire [31:0] sub_wire0;
            wire [31:0] resulta = sub_wire0[31:0];    

            twentynm_mac        twentynm_mac_component (
                                        .ax (ax),
                                        .ay (ay),
                                        .bx (bx),
                                        .by (by),
                                        .ena (ena),
                                        .resulta (sub_wire0),
                                        .aclr ({aclr1,aclr0}),
                                        .clk ({clk2,clk1,clk0}),
                                        .accumulate (),
                                        .az (),
                                        .bz (),
                                        .chainin (),
                                        .chainout (),
                                        .coefsela (),
                                        .coefselb (),
                                        .dftout (),
                                        .loadconst (),
                                        .negate (),
                                        .resultb (),
                                        .scanin (),
                                        .scanout (),
                                        .sub ());
            defparam
                    twentynm_mac_component.ax_width = 16,
                    twentynm_mac_component.ay_scan_in_width = 16,
                    twentynm_mac_component.bx_width = 16,
                    twentynm_mac_component.by_width = 16,
                    twentynm_mac_component.operation_mode = "m18x18_sumof2",
                    twentynm_mac_component.mode_sub_location = 0,
                    twentynm_mac_component.operand_source_max = "input",
                    twentynm_mac_component.operand_source_may = "input",
                    twentynm_mac_component.operand_source_mbx = "input",
                    twentynm_mac_component.operand_source_mby = "input",
                    twentynm_mac_component.signed_max = "true",
                    twentynm_mac_component.signed_may = "true",
                    twentynm_mac_component.signed_mbx = "true",
                    twentynm_mac_component.signed_mby = "true",
                    twentynm_mac_component.preadder_subtract_a = "false",
                    twentynm_mac_component.preadder_subtract_b = "false",
                    twentynm_mac_component.ay_use_scan_in = "false",
                    twentynm_mac_component.by_use_scan_in = "false",
                    twentynm_mac_component.delay_scan_out_ay = "false",
                    twentynm_mac_component.delay_scan_out_by = "false",
                    twentynm_mac_component.use_chainadder = "false",
                    twentynm_mac_component.enable_double_accum = "false",
                    twentynm_mac_component.load_const_value = 0,
                    twentynm_mac_component.coef_a_0 = 0,
                    twentynm_mac_component.coef_a_1 = 0,
                    twentynm_mac_component.coef_a_2 = 0,
                    twentynm_mac_component.coef_a_3 = 0,
                    twentynm_mac_component.coef_a_4 = 0,
                    twentynm_mac_component.coef_a_5 = 0,
                    twentynm_mac_component.coef_a_6 = 0,
                    twentynm_mac_component.coef_a_7 = 0,
                    twentynm_mac_component.coef_b_0 = 0,
                    twentynm_mac_component.coef_b_1 = 0,
                    twentynm_mac_component.coef_b_2 = 0,
                    twentynm_mac_component.coef_b_3 = 0,
                    twentynm_mac_component.coef_b_4 = 0,
                    twentynm_mac_component.coef_b_5 = 0,
                    twentynm_mac_component.coef_b_6 = 0,
                    twentynm_mac_component.coef_b_7 = 0,
                    twentynm_mac_component.ax_clock = "none",
                    twentynm_mac_component.ay_scan_in_clock = "none",
                    twentynm_mac_component.az_clock = "none",
                    twentynm_mac_component.bx_clock = "none",
                    twentynm_mac_component.by_clock = "none",
                    twentynm_mac_component.bz_clock = "none",
                    twentynm_mac_component.coef_sel_a_clock = "none",
                    twentynm_mac_component.coef_sel_b_clock = "none",
                    twentynm_mac_component.sub_clock = "none",
                    twentynm_mac_component.sub_pipeline_clock = "none",
                    twentynm_mac_component.negate_clock = "none",
                    twentynm_mac_component.negate_pipeline_clock = "none",
                    twentynm_mac_component.accumulate_clock = "none",
                    twentynm_mac_component.accum_pipeline_clock = "none",
                    twentynm_mac_component.load_const_clock = "none",
                    twentynm_mac_component.load_const_pipeline_clock = "none",
                    twentynm_mac_component.input_pipeline_clock = "none",
                    twentynm_mac_component.output_clock = "0",
                    twentynm_mac_component.scan_out_width = 16,
                    twentynm_mac_component.result_a_width = 32;


endmodule


