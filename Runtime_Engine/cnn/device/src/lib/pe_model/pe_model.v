module pe_model(
	input clock,
	input resetn,
	input ivalid,
	input iready,
	output ovalid,
	output oready,
	
	input [7:0]feature_val,
	input [7:0]filter_val,
	output[7:0]dot_accum);
	
	wire [7:0]product;
  reg  [7:0]result;
  
  reg  [7:0]buffer;
	
  mul_4bit_lpm_mult_191_55zceji lpm_mult_0 (
       .dataa  (filter_val[3:0]),  //   input,  width = 4,  dataa.dataa
       .result (product), //  output,  width = 8, result.result
       .datab  (feature_val[3:0]),  //   input,  width = 4,  datab.datab
       .clock  (clock),  //   input,  width = 1,  clock.clk
       .aclr   (1'b0)    //   input,  width = 1,   aclr.reset
  );
  
  always@(posedge clock or negedge resetn)
	begin
		if(!resetn)
      buffer <= 8'd0;
    else
      buffer <= filter_val;
	end

  always@(posedge clock or negedge resetn)
	begin
		if(!resetn)
			result <= 8'd0;
		else
		begin
			if(buffer[7:7])
				result <= -product[7:0];
			else
				result <= product[7:0];
		end
	end
	
  assign dot_accum = result;
	assign ovalid = 1'b1;
	assign oready = 1'b1;
	
endmodule
