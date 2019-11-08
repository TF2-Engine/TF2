
find_package(Protobuf)

if(PROTOBUF_FOUND)
    include_directories(${PROTOBUF_INCLUDE_DIR})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)
    add_executable(caffe2fpga caffe2fpga.cpp fpganetworkinterface.cpp tools.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})
    set_target_properties(caffe2fpga PROPERTIES CXX_STANDARD 11)
    target_link_libraries(caffe2fpga ${PROTOBUF_LIBRARIES})
else()
    message(WARNING "Protobuf not found, caffe model convert tool won't be built")
endif()
