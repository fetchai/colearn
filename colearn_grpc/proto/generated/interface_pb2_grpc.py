# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import colearn_grpc.proto.generated.interface_pb2 as interface__pb2


class GRPCLearnerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.QuerySupportedSystem = channel.unary_unary(
                '/contract_learn.grpc.GRPCLearner/QuerySupportedSystem',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=interface__pb2.ResponseSupportedSystem.FromString,
                )
        self.MLSetup = channel.unary_unary(
                '/contract_learn.grpc.GRPCLearner/MLSetup',
                request_serializer=interface__pb2.RequestMLSetup.SerializeToString,
                response_deserializer=interface__pb2.ResponseMLSetup.FromString,
                )
        self.ProposeWeights = channel.unary_stream(
                '/contract_learn.grpc.GRPCLearner/ProposeWeights',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=interface__pb2.Weights.FromString,
                )
        self.TestWeights = channel.unary_unary(
                '/contract_learn.grpc.GRPCLearner/TestWeights',
                request_serializer=interface__pb2.Weights.SerializeToString,
                response_deserializer=interface__pb2.ProposedWeights.FromString,
                )
        self.SetWeights = channel.unary_unary(
                '/contract_learn.grpc.GRPCLearner/SetWeights',
                request_serializer=interface__pb2.Weights.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.StatusStream = channel.stream_stream(
                '/contract_learn.grpc.GRPCLearner/StatusStream',
                request_serializer=interface__pb2.RequestStatus.SerializeToString,
                response_deserializer=interface__pb2.ResponseStatus.FromString,
                )


class GRPCLearnerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def QuerySupportedSystem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MLSetup(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ProposeWeights(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TestWeights(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetWeights(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StatusStream(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GRPCLearnerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'QuerySupportedSystem': grpc.unary_unary_rpc_method_handler(
                    servicer.QuerySupportedSystem,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=interface__pb2.ResponseSupportedSystem.SerializeToString,
            ),
            'MLSetup': grpc.unary_unary_rpc_method_handler(
                    servicer.MLSetup,
                    request_deserializer=interface__pb2.RequestMLSetup.FromString,
                    response_serializer=interface__pb2.ResponseMLSetup.SerializeToString,
            ),
            'ProposeWeights': grpc.unary_stream_rpc_method_handler(
                    servicer.ProposeWeights,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=interface__pb2.Weights.SerializeToString,
            ),
            'TestWeights': grpc.unary_unary_rpc_method_handler(
                    servicer.TestWeights,
                    request_deserializer=interface__pb2.Weights.FromString,
                    response_serializer=interface__pb2.ProposedWeights.SerializeToString,
            ),
            'SetWeights': grpc.unary_unary_rpc_method_handler(
                    servicer.SetWeights,
                    request_deserializer=interface__pb2.Weights.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'StatusStream': grpc.stream_stream_rpc_method_handler(
                    servicer.StatusStream,
                    request_deserializer=interface__pb2.RequestStatus.FromString,
                    response_serializer=interface__pb2.ResponseStatus.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'contract_learn.grpc.GRPCLearner', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GRPCLearner(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def QuerySupportedSystem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/contract_learn.grpc.GRPCLearner/QuerySupportedSystem',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            interface__pb2.ResponseSupportedSystem.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MLSetup(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/contract_learn.grpc.GRPCLearner/MLSetup',
            interface__pb2.RequestMLSetup.SerializeToString,
            interface__pb2.ResponseMLSetup.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ProposeWeights(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/contract_learn.grpc.GRPCLearner/ProposeWeights',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            interface__pb2.Weights.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def TestWeights(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/contract_learn.grpc.GRPCLearner/TestWeights',
            interface__pb2.Weights.SerializeToString,
            interface__pb2.ProposedWeights.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetWeights(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/contract_learn.grpc.GRPCLearner/SetWeights',
            interface__pb2.Weights.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StatusStream(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/contract_learn.grpc.GRPCLearner/StatusStream',
            interface__pb2.RequestStatus.SerializeToString,
            interface__pb2.ResponseStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
