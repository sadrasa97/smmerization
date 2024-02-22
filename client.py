import grpc
import file_processor_pb2
import file_processor_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = file_processor_pb2_grpc.FileProcessorStub(channel)

    with open(r"C:\\Users\\sadra\\Desktop\\N\\DataML_Engineer_Assignment.pdf", "rb") as file:
        content = file.read()

    request = file_processor_pb2.PdfFile(content=content)

    response = stub.ProcessPdf(request)
    print("Summary:", response.summary)

if __name__ == '__main__':
    run()
