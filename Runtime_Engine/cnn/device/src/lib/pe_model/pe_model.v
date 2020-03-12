module pe_model(
	input clock,
	input resetn,
	input ivalid,
	input iready,
	output ovalid,
	output oready,
	
	input [127:0]feature_values,
	input [127:0]filter_values,
	output[31:0]dot_accum);
	
	wire [31:0]data[0:15];
	MUL mul0 (
	.clock(clock),
	.data(data[0]),
	.fea(feature_values[127:120]),
	.fil(filter_values[127:120]),
	.resetn(resetn)
	);
	MUL mul1 (
	.clock(clock),
	.data(data[1]),
	.fea(feature_values[119:112]),
	.fil(filter_values[119:112]),
	.resetn(resetn)
	);
	MUL mul2 (
	.clock(clock),
	.data(data[2]),
	.fea(feature_values[111:104]),
	.fil(filter_values[111:104]),
	.resetn(resetn)
	);
	MUL mul3 (
	.clock(clock),
	.data(data[3]),
	.fea(feature_values[103:96]),
	.fil(filter_values[103:96]),
	.resetn(resetn)
	);
	MUL mul4 (
	.clock(clock),
	.data(data[4]),
	.fea(feature_values[95:88]),
	.fil(filter_values[95:88]),
	.resetn(resetn)
	);
	MUL mul5 (
	.clock(clock),
	.data(data[5]),
	.fea(feature_values[87:80]),
	.fil(filter_values[87:80]),
	.resetn(resetn)
	);
	MUL mul6 (
	.clock(clock),
	.data(data[6]),
	.fea(feature_values[79:72]),
	.fil(filter_values[79:72]),
	.resetn(resetn)
	);
	MUL mul7 (
	.clock(clock),
	.data(data[7]),
	.fea(feature_values[71:64]),
	.fil(filter_values[71:64]),
	.resetn(resetn)
	);
	MUL mul8 (
	.clock(clock),
	.data(data[8]),
	.fea(feature_values[63:56]),
	.fil(filter_values[63:56]),
	.resetn(resetn)
	);

	MUL mul9 (
	.clock(clock),
	.data(data[9]),
	.fea(feature_values[55:48]),
	.fil(filter_values[55:48]),
	.resetn(resetn)
	);
	MUL mul10 (
	.clock(clock),
	.data(data[10]),
	.fea(feature_values[47:40]),
	.fil(filter_values[47:40]),
	.resetn(resetn)
	);
	MUL mul11 (
	.clock(clock),
	.data(data[11]),
	.fea(feature_values[39:32]),
	.fil(filter_values[39:32]),
	.resetn(resetn)
	);
	MUL mul12 (
	.clock(clock),
	.data(data[12]),
	.fea(feature_values[31:24]),
	.fil(filter_values[31:24]),
	.resetn(resetn)
	);
	MUL mul13 (
	.clock(clock),
	.data(data[13]),
	.fea(feature_values[23:16]),
	.fil(filter_values[23:16]),
	.resetn(resetn)
	);
	MUL mul14 (
	.clock(clock),
	.data(data[14]),
	.fea(feature_values[15:8]),
	.fil(filter_values[15:8]),
	.resetn(resetn)
	);
	MUL mul15 (
	.clock(clock),
	.data(data[15]),
	.fea(feature_values[7:0]),
	.fil(filter_values[7:0]),
	.resetn(resetn)
	);

	assign dot_accum = data[0]+data[1]+data[2]+data[3]+data[4]+data[5]+data[6]+data[7]+data[8]+data[9]+data[10]+data[11]+data[12]+data[13]+data[14]+data[15];
	assign ovalid = 1'b1;
	assign oready = 1'b1;
	
endmodule
	
