`timescale 1ps / 1ps
module MUL(
	input clock,
	input resetn,
	input [7:0]fea, 
	input [7:0]fil,
	output reg[31:0] data
	);
	
	always@(posedge clock or negedge resetn)
	begin
		if(!resetn)
			data <= 32'd0;
		else
		begin
			if(fil[6:6])
				data <= 32'd0;
			else if(fil[7:7])
				data <= (32'hffffffff & (~fea+1'b1))<<fil[4:0];
			else
				data <= (32'd0 | fea)<<fil[4:0];	
		end
	end
endmodule 
