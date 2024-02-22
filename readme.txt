run server : python server.py
run client : python client.py
run  proto : python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. file_processor.proto
docker build -t grpc-server 
docker run -p 50051:50051 grpc-server

 127.0.0.1:50051
