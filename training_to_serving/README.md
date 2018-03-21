# Converting Training Code to Serving Code

The examples in this sub-directory show how to convert models trained using
Tensorflow Estimator and Keras APIs into servable saved models that can be
served using the [Kubeflow tf-serving component]()

## Setup

Create two virtual environments: a python 2 environment for the
TF client, and a python 3 environment for creating servable models.

```
VENV_DIR=<your-virtualenv-base-directory>
```

```
virtualenv -p python2 ${VENV_DIR}/client
virtualenv -p python3 ${VENV_DIR}/serving
```

From the training_to_serving directory, install requirements for both
environments.

```
source ${VENV_DIR}/client/bin/activate
pip install -r client_requirements.txt
deactivate
```

```
source ${VENV_DIR}/serving/bin/activate
pip install -r serving_requirements.txt
```

Stay in the python3 environment for the examples below.

## ResNet-50 ImageNet Examples

These examples shows how to convert pre-trained ResNet-50 models trained
using TF Estimator and Keras APIs into servable saved models with custom-defined
APIs for client-server RPC calls. Both examples build the same API, so the same
client can be used to send requests to and process responses from a model server
hosting either the Estimator or the Keras-trained model.

### TF Estimator Example

The following process will convert a
[pre-trained ResNet-50 model](https://github.com/tensorflow/models/tree/v1.4.0/official/resnet)
using the TF Estimator API into a servable saved model.

cd into the resnet/estimator path.
```
cd resnet/estimator
```

Set a directory to store the model checkpoint and the servable output:
```
MODEL_DIR=model
```
```
SERVABLE_DIR=servable
```

Download the ResNet-50 architecture code from the official Tensorflow
github repository:
```
python download_model.py
```

Download the checkpoint file into the 'model' directory, and 
```
python download_checkpoint.py \
  -s http://download.tensorflow.org/models/official/resnet50_2017_11_30.tar.gz \
  -d ${MODEL_DIR}
```

Run the following to create a servable saved model:

```
python training_to_serving.py -m ${MODEL_DIR} -o ${SERVABLE_DIR}
```

### TF Keras Example

The following converts the prepackaged Keras ResNet50 model trained on ImageNet
into a servable saved model with the same client-server RPC API:

```
cd resnet/keras
```

Set a directory to store the servable output:
```
SERVABLE_DIR=servable
```

Run the following to create a servable saved model:

```
python training_to_serving.py -o ${SERVABLE_DIR}
```

### Deploying the Model Server

Follow the instructions on 
[Kubeflow tf-serving](https://github.com/kubeflow/kubeflow/tree/master/components/k8s-model-server)
to start a Kubernetes cluster on GCP. 

At the [upload a model](https://github.com/kubeflow/kubeflow/tree/master/components/k8s-model-server#upload-a-model),
step, choose your servable directory to copy to your Google storage bucket
instead of inception:

```
BUCKET=gs://<your-bucket-name>
```

```
gsutil cp -r ${SERVABLE_DIR} ${BUCKET}/
```

Then use the modified ksonnet commands for deploying the resnet model server.
We will not add http proxy:

```
NAMESPACE=model-server

ks init model-server
cd model-server
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/master/kubeflow
ks pkg install kubeflow/tf-serving
ks env add cloud
ks env set cloud --namespace ${NAMESPACE}

MODEL_COMPONENT=serveResnet
MODEL_NAME=resnet
MODEL_PATH=${BUCKET}/${SERVABLE_DIR}
MODEL_SERVER_IMAGE=gcr.io/kubeflow/model-server:1.0
ks generate tf-serving ${MODEL_COMPONENT} --name=${MODEL_NAME}
ks param set --env=cloud ${MODEL_COMPONENT} modelPath $MODEL_PATH
# If you want to use your custom image.
ks param set --env=cloud ${MODEL_COMPONENT} modelServerImage $MODEL_SERVER_IMAGE
```

Apply your component to your k8s cluster:
```
ks apply cloud -c ${MODEL_COMPONENT}
```

### Client RPC Calls to Model Server on Cloud
2. Stop at the [use the served model](https://github.com/kubeflow/kubeflow/tree/master/components/k8s-model-server#use-the-served-model)
step. Follow the instructions below instead.

Activate your python2 environment for your client:

```
source ${VENV_DIR}/serving/bin/activate
```

Get the services running on your k8s cluster and find the one serving your
model name:

```
kubectl get services
NAME         TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)			 AGE
$MODEL_NAME  LoadBalancer   <INTERNAL IP>   <SERVICE IP>     <SERVICE PORT>:<NODE PORT>,<HTTP SERVICE PORT>:<NODE PORT>  <TIME SINCE DEPLOYMENT>
```

Then cd into the resnet/client directory and run the script with any RGB image
(local and web URL paths both work).

```
python client.py --server <SERVICE IP> \
  --port <SERVICE PORT> \
  <image-path-1> <image-path-2> ...
```

You can use an image in the `resnet/samples` directory as a test.


## Customizing and Debugging Model Serving API

### Unit Tests for TF Estimator
When modifying the model server API, it is useful to have unit tests
to run and verify that different Tensorflow graph components are hooking up
correctly. The Estimator API can be hard to debug without unit tests
because it waits for the very last (export) step before building the entire
graph and loading checkpoint parameters, generating a long stack trace.

Example unit tests are located in `training_to_serving_test.py`. Modify the file
based on your own client API, and run it to execute unit tests
(Make sure you are in your python3 virtual environment first!):

```
python training_to_serving_test.py
```
