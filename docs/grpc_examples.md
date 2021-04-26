# Mnist gRPC Example

To run the Keras Mnist gRPC example run:

```bash
python -m colearn_examples.grpc.run_grpc_demo.py --n_learners 5 --dataloader_tag KERAS_MNIST --model_tag KERAS_MNIST \
--data_locations /tmp/mnist/0,/tmp/mnist/1,/tmp/mnist/2,/tmp/mnist/3,/tmp/mnist/4
```

!!!Note 
    This requires `colearn[keras]`

You can verify that the example is working correctly by running the probe:

```bash
python -m colearn_grpc.scripts.probe_grpc_server.py --port 9995
```

For more about the gRPC components of Colearn see the [gRPC Tutorial](grpc_tutorial.md)
