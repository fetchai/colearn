# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: interface.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='interface.proto',
  package='contract_learn.grpc',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0finterface.proto\x12\x13\x63ontract_learn.grpc\x1a\x1bgoogle/protobuf/empty.proto\"\x83\x01\n\x0eRequestMLSetup\x12\x1b\n\x13\x64\x61taset_loader_name\x18\x01 \x01(\t\x12!\n\x19\x64\x61taset_loader_parameters\x18\x02 \x01(\t\x12\x17\n\x0fmodel_arch_name\x18\x03 \x01(\t\x12\x18\n\x10model_parameters\x18\x04 \x01(\t\"Z\n\x0fResponseMLSetup\x12\x32\n\x06status\x18\x01 \x01(\x0e\x32\".contract_learn.grpc.MLSetupStatus\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\"p\n\x0e\x44iffPrivBudget\x12\x16\n\x0etarget_epsilon\x18\x01 \x01(\x02\x12\x14\n\x0ctarget_delta\x18\x02 \x01(\x02\x12\x18\n\x10\x63onsumed_epsilon\x18\x03 \x01(\x02\x12\x16\n\x0e\x63onsumed_delta\x18\x04 \x01(\x02\"I\n\x0fTrainingSummary\x12\x36\n\tdp_budget\x18\x01 \x01(\x0b\x32#.contract_learn.grpc.DiffPrivBudget\"\x87\x01\n\x0bWeightsPart\x12\x0f\n\x07weights\x18\x01 \x01(\x0c\x12\x12\n\nbyte_index\x18\x02 \x01(\r\x12\x13\n\x0btotal_bytes\x18\x03 \x01(\x04\x12>\n\x10training_summary\x18\n \x01(\x0b\x32$.contract_learn.grpc.TrainingSummary\"G\n\x0fProposedWeights\x12\x12\n\nvote_score\x18\x01 \x01(\x02\x12\x12\n\ntest_score\x18\x02 \x01(\x02\x12\x0c\n\x04vote\x18\x03 \x01(\x08\"\x0f\n\rRequestStatus\"C\n\x0eResponseStatus\x12\x31\n\x06status\x18\x01 \x01(\x0e\x32!.contract_learn.grpc.SystemStatus\"=\n\x11\x44\x61tasetLoaderSpec\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1a\n\x12\x64\x65\x66\x61ult_parameters\x18\x02 \x01(\t\"9\n\rModelArchSpec\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1a\n\x12\x64\x65\x66\x61ult_parameters\x18\x02 \x01(\t\"D\n\x11\x43ompatibilitySpec\x12\x1a\n\x12model_architecture\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x61taloaders\x18\x02 \x03(\t\"\xd9\x01\n\x17ResponseSupportedSystem\x12<\n\x0c\x64\x61ta_loaders\x18\x01 \x03(\x0b\x32&.contract_learn.grpc.DatasetLoaderSpec\x12?\n\x13model_architectures\x18\x02 \x03(\x0b\x32\".contract_learn.grpc.ModelArchSpec\x12?\n\x0f\x63ompatibilities\x18\x03 \x03(\x0b\x32&.contract_learn.grpc.CompatibilitySpec*6\n\rMLSetupStatus\x12\r\n\tUNDEFINED\x10\x00\x12\x0b\n\x07SUCCESS\x10\x01\x12\t\n\x05\x45RROR\x10\x02*J\n\x0cSystemStatus\x12\x0b\n\x07WORKING\x10\x00\x12\x0c\n\x08NO_MODEL\x10\x01\x12\x12\n\x0eINTERNAL_ERROR\x10\x02\x12\x0b\n\x07UNKNOWN\x10\x03\x32\xe0\x04\n\x0bGRPCLearner\x12\\\n\x14QuerySupportedSystem\x12\x16.google.protobuf.Empty\x1a,.contract_learn.grpc.ResponseSupportedSystem\x12T\n\x07MLSetup\x12#.contract_learn.grpc.RequestMLSetup\x1a$.contract_learn.grpc.ResponseMLSetup\x12L\n\x0eProposeWeights\x12\x16.google.protobuf.Empty\x1a .contract_learn.grpc.WeightsPart0\x01\x12W\n\x0bTestWeights\x12 .contract_learn.grpc.WeightsPart\x1a$.contract_learn.grpc.ProposedWeights(\x01\x12H\n\nSetWeights\x12 .contract_learn.grpc.WeightsPart\x1a\x16.google.protobuf.Empty(\x01\x12O\n\x11GetCurrentWeights\x12\x16.google.protobuf.Empty\x1a .contract_learn.grpc.WeightsPart0\x01\x12[\n\x0cStatusStream\x12\".contract_learn.grpc.RequestStatus\x1a#.contract_learn.grpc.ResponseStatus(\x01\x30\x01\x62\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_empty__pb2.DESCRIPTOR,])

_MLSETUPSTATUS = _descriptor.EnumDescriptor(
  name='MLSetupStatus',
  full_name='contract_learn.grpc.MLSetupStatus',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1193,
  serialized_end=1247,
)
_sym_db.RegisterEnumDescriptor(_MLSETUPSTATUS)

MLSetupStatus = enum_type_wrapper.EnumTypeWrapper(_MLSETUPSTATUS)
_SYSTEMSTATUS = _descriptor.EnumDescriptor(
  name='SystemStatus',
  full_name='contract_learn.grpc.SystemStatus',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='WORKING', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NO_MODEL', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INTERNAL_ERROR', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1249,
  serialized_end=1323,
)
_sym_db.RegisterEnumDescriptor(_SYSTEMSTATUS)

SystemStatus = enum_type_wrapper.EnumTypeWrapper(_SYSTEMSTATUS)
UNDEFINED = 0
SUCCESS = 1
ERROR = 2
WORKING = 0
NO_MODEL = 1
INTERNAL_ERROR = 2
UNKNOWN = 3



_REQUESTMLSETUP = _descriptor.Descriptor(
  name='RequestMLSetup',
  full_name='contract_learn.grpc.RequestMLSetup',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dataset_loader_name', full_name='contract_learn.grpc.RequestMLSetup.dataset_loader_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dataset_loader_parameters', full_name='contract_learn.grpc.RequestMLSetup.dataset_loader_parameters', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_arch_name', full_name='contract_learn.grpc.RequestMLSetup.model_arch_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_parameters', full_name='contract_learn.grpc.RequestMLSetup.model_parameters', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=70,
  serialized_end=201,
)


_RESPONSEMLSETUP = _descriptor.Descriptor(
  name='ResponseMLSetup',
  full_name='contract_learn.grpc.ResponseMLSetup',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='contract_learn.grpc.ResponseMLSetup.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='description', full_name='contract_learn.grpc.ResponseMLSetup.description', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=203,
  serialized_end=293,
)


_DIFFPRIVBUDGET = _descriptor.Descriptor(
  name='DiffPrivBudget',
  full_name='contract_learn.grpc.DiffPrivBudget',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='target_epsilon', full_name='contract_learn.grpc.DiffPrivBudget.target_epsilon', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='target_delta', full_name='contract_learn.grpc.DiffPrivBudget.target_delta', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='consumed_epsilon', full_name='contract_learn.grpc.DiffPrivBudget.consumed_epsilon', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='consumed_delta', full_name='contract_learn.grpc.DiffPrivBudget.consumed_delta', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=295,
  serialized_end=407,
)


_TRAININGSUMMARY = _descriptor.Descriptor(
  name='TrainingSummary',
  full_name='contract_learn.grpc.TrainingSummary',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dp_budget', full_name='contract_learn.grpc.TrainingSummary.dp_budget', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=409,
  serialized_end=482,
)


_WEIGHTSPART = _descriptor.Descriptor(
  name='WeightsPart',
  full_name='contract_learn.grpc.WeightsPart',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='weights', full_name='contract_learn.grpc.WeightsPart.weights', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='byte_index', full_name='contract_learn.grpc.WeightsPart.byte_index', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_bytes', full_name='contract_learn.grpc.WeightsPart.total_bytes', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='training_summary', full_name='contract_learn.grpc.WeightsPart.training_summary', index=3,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=485,
  serialized_end=620,
)


_PROPOSEDWEIGHTS = _descriptor.Descriptor(
  name='ProposedWeights',
  full_name='contract_learn.grpc.ProposedWeights',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='vote_score', full_name='contract_learn.grpc.ProposedWeights.vote_score', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='test_score', full_name='contract_learn.grpc.ProposedWeights.test_score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vote', full_name='contract_learn.grpc.ProposedWeights.vote', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=622,
  serialized_end=693,
)


_REQUESTSTATUS = _descriptor.Descriptor(
  name='RequestStatus',
  full_name='contract_learn.grpc.RequestStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=695,
  serialized_end=710,
)


_RESPONSESTATUS = _descriptor.Descriptor(
  name='ResponseStatus',
  full_name='contract_learn.grpc.ResponseStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='contract_learn.grpc.ResponseStatus.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=712,
  serialized_end=779,
)


_DATASETLOADERSPEC = _descriptor.Descriptor(
  name='DatasetLoaderSpec',
  full_name='contract_learn.grpc.DatasetLoaderSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='contract_learn.grpc.DatasetLoaderSpec.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='default_parameters', full_name='contract_learn.grpc.DatasetLoaderSpec.default_parameters', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=781,
  serialized_end=842,
)


_MODELARCHSPEC = _descriptor.Descriptor(
  name='ModelArchSpec',
  full_name='contract_learn.grpc.ModelArchSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='contract_learn.grpc.ModelArchSpec.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='default_parameters', full_name='contract_learn.grpc.ModelArchSpec.default_parameters', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=844,
  serialized_end=901,
)


_COMPATIBILITYSPEC = _descriptor.Descriptor(
  name='CompatibilitySpec',
  full_name='contract_learn.grpc.CompatibilitySpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_architecture', full_name='contract_learn.grpc.CompatibilitySpec.model_architecture', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dataloaders', full_name='contract_learn.grpc.CompatibilitySpec.dataloaders', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=903,
  serialized_end=971,
)


_RESPONSESUPPORTEDSYSTEM = _descriptor.Descriptor(
  name='ResponseSupportedSystem',
  full_name='contract_learn.grpc.ResponseSupportedSystem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_loaders', full_name='contract_learn.grpc.ResponseSupportedSystem.data_loaders', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_architectures', full_name='contract_learn.grpc.ResponseSupportedSystem.model_architectures', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='compatibilities', full_name='contract_learn.grpc.ResponseSupportedSystem.compatibilities', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=974,
  serialized_end=1191,
)

_RESPONSEMLSETUP.fields_by_name['status'].enum_type = _MLSETUPSTATUS
_TRAININGSUMMARY.fields_by_name['dp_budget'].message_type = _DIFFPRIVBUDGET
_WEIGHTSPART.fields_by_name['training_summary'].message_type = _TRAININGSUMMARY
_RESPONSESTATUS.fields_by_name['status'].enum_type = _SYSTEMSTATUS
_RESPONSESUPPORTEDSYSTEM.fields_by_name['data_loaders'].message_type = _DATASETLOADERSPEC
_RESPONSESUPPORTEDSYSTEM.fields_by_name['model_architectures'].message_type = _MODELARCHSPEC
_RESPONSESUPPORTEDSYSTEM.fields_by_name['compatibilities'].message_type = _COMPATIBILITYSPEC
DESCRIPTOR.message_types_by_name['RequestMLSetup'] = _REQUESTMLSETUP
DESCRIPTOR.message_types_by_name['ResponseMLSetup'] = _RESPONSEMLSETUP
DESCRIPTOR.message_types_by_name['DiffPrivBudget'] = _DIFFPRIVBUDGET
DESCRIPTOR.message_types_by_name['TrainingSummary'] = _TRAININGSUMMARY
DESCRIPTOR.message_types_by_name['WeightsPart'] = _WEIGHTSPART
DESCRIPTOR.message_types_by_name['ProposedWeights'] = _PROPOSEDWEIGHTS
DESCRIPTOR.message_types_by_name['RequestStatus'] = _REQUESTSTATUS
DESCRIPTOR.message_types_by_name['ResponseStatus'] = _RESPONSESTATUS
DESCRIPTOR.message_types_by_name['DatasetLoaderSpec'] = _DATASETLOADERSPEC
DESCRIPTOR.message_types_by_name['ModelArchSpec'] = _MODELARCHSPEC
DESCRIPTOR.message_types_by_name['CompatibilitySpec'] = _COMPATIBILITYSPEC
DESCRIPTOR.message_types_by_name['ResponseSupportedSystem'] = _RESPONSESUPPORTEDSYSTEM
DESCRIPTOR.enum_types_by_name['MLSetupStatus'] = _MLSETUPSTATUS
DESCRIPTOR.enum_types_by_name['SystemStatus'] = _SYSTEMSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RequestMLSetup = _reflection.GeneratedProtocolMessageType('RequestMLSetup', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTMLSETUP,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.RequestMLSetup)
  })
_sym_db.RegisterMessage(RequestMLSetup)

ResponseMLSetup = _reflection.GeneratedProtocolMessageType('ResponseMLSetup', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSEMLSETUP,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.ResponseMLSetup)
  })
_sym_db.RegisterMessage(ResponseMLSetup)

DiffPrivBudget = _reflection.GeneratedProtocolMessageType('DiffPrivBudget', (_message.Message,), {
  'DESCRIPTOR' : _DIFFPRIVBUDGET,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.DiffPrivBudget)
  })
_sym_db.RegisterMessage(DiffPrivBudget)

TrainingSummary = _reflection.GeneratedProtocolMessageType('TrainingSummary', (_message.Message,), {
  'DESCRIPTOR' : _TRAININGSUMMARY,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.TrainingSummary)
  })
_sym_db.RegisterMessage(TrainingSummary)

WeightsPart = _reflection.GeneratedProtocolMessageType('WeightsPart', (_message.Message,), {
  'DESCRIPTOR' : _WEIGHTSPART,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.WeightsPart)
  })
_sym_db.RegisterMessage(WeightsPart)

ProposedWeights = _reflection.GeneratedProtocolMessageType('ProposedWeights', (_message.Message,), {
  'DESCRIPTOR' : _PROPOSEDWEIGHTS,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.ProposedWeights)
  })
_sym_db.RegisterMessage(ProposedWeights)

RequestStatus = _reflection.GeneratedProtocolMessageType('RequestStatus', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTSTATUS,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.RequestStatus)
  })
_sym_db.RegisterMessage(RequestStatus)

ResponseStatus = _reflection.GeneratedProtocolMessageType('ResponseStatus', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSESTATUS,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.ResponseStatus)
  })
_sym_db.RegisterMessage(ResponseStatus)

DatasetLoaderSpec = _reflection.GeneratedProtocolMessageType('DatasetLoaderSpec', (_message.Message,), {
  'DESCRIPTOR' : _DATASETLOADERSPEC,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.DatasetLoaderSpec)
  })
_sym_db.RegisterMessage(DatasetLoaderSpec)

ModelArchSpec = _reflection.GeneratedProtocolMessageType('ModelArchSpec', (_message.Message,), {
  'DESCRIPTOR' : _MODELARCHSPEC,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.ModelArchSpec)
  })
_sym_db.RegisterMessage(ModelArchSpec)

CompatibilitySpec = _reflection.GeneratedProtocolMessageType('CompatibilitySpec', (_message.Message,), {
  'DESCRIPTOR' : _COMPATIBILITYSPEC,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.CompatibilitySpec)
  })
_sym_db.RegisterMessage(CompatibilitySpec)

ResponseSupportedSystem = _reflection.GeneratedProtocolMessageType('ResponseSupportedSystem', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSESUPPORTEDSYSTEM,
  '__module__' : 'interface_pb2'
  # @@protoc_insertion_point(class_scope:contract_learn.grpc.ResponseSupportedSystem)
  })
_sym_db.RegisterMessage(ResponseSupportedSystem)



_GRPCLEARNER = _descriptor.ServiceDescriptor(
  name='GRPCLearner',
  full_name='contract_learn.grpc.GRPCLearner',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1326,
  serialized_end=1934,
  methods=[
  _descriptor.MethodDescriptor(
    name='QuerySupportedSystem',
    full_name='contract_learn.grpc.GRPCLearner.QuerySupportedSystem',
    index=0,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_RESPONSESUPPORTEDSYSTEM,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MLSetup',
    full_name='contract_learn.grpc.GRPCLearner.MLSetup',
    index=1,
    containing_service=None,
    input_type=_REQUESTMLSETUP,
    output_type=_RESPONSEMLSETUP,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ProposeWeights',
    full_name='contract_learn.grpc.GRPCLearner.ProposeWeights',
    index=2,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_WEIGHTSPART,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='TestWeights',
    full_name='contract_learn.grpc.GRPCLearner.TestWeights',
    index=3,
    containing_service=None,
    input_type=_WEIGHTSPART,
    output_type=_PROPOSEDWEIGHTS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetWeights',
    full_name='contract_learn.grpc.GRPCLearner.SetWeights',
    index=4,
    containing_service=None,
    input_type=_WEIGHTSPART,
    output_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetCurrentWeights',
    full_name='contract_learn.grpc.GRPCLearner.GetCurrentWeights',
    index=5,
    containing_service=None,
    input_type=google_dot_protobuf_dot_empty__pb2._EMPTY,
    output_type=_WEIGHTSPART,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StatusStream',
    full_name='contract_learn.grpc.GRPCLearner.StatusStream',
    index=6,
    containing_service=None,
    input_type=_REQUESTSTATUS,
    output_type=_RESPONSESTATUS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_GRPCLEARNER)

DESCRIPTOR.services_by_name['GRPCLearner'] = _GRPCLEARNER

# @@protoc_insertion_point(module_scope)
