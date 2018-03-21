# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creating a Resnet TF Estimator Servable Model."""

import argparse
import numpy as np
import os
import resnet_model
import sys
import tensorflow as tf
import urllib.request

# Constants: see v1.4.0/official/resnet/imagenet_main.py
_NUM_CHANNELS = 3


def convert_jpeg_to_image(encoded_image, height, width):
  '''Resize the image and normalize pixel values.

  Args:
    image: A jpeg-formatted byte stream represented as a string.

  Returns:
    A 3d tensor of image pixels normalized to be between -0.5 and 0.5, resized
      to height x width x 3.
      The normalization approximates the preprocess_for_train() and
      preprocess_for_eval() functions in
      https://github.com/tensorflow/models/blob/v1.4.0/official/resnet/vgg_preprocessing.py.
  '''

  image = tf.image.decode_jpeg(encoded_image, channels=3)
  image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  image = tf.to_float(image) / 255.0 - 0.5
  return image


def preprocess_input(features, height, width):
  '''Preprocess client request before feeding into the network.

  Use tf.map_fn and the convert_jpeg_to_image() helper function to convert the
  1D input tensor of jpeg strings into a list of single-precision floating
  point 3D tensors, which are normalized pixel values for the images.

  Then stack and reshape this list of tensors into a 4D tensor with
  appropriate dimensions.

  Args:
    features: request received from our client,
      a dictionary with a single element containing a tensor of multiple jpeg images
      {'images' : 1D_tensor_of_jpeg_byte_strings}

  Returns:
    a 4D tensor of normalized pixel values for the input images.

  '''

  def convert_jpeg_to_image_with_dim(encoded_image):
    return convert_jpeg_to_image(encoded_image, height, width)

  images = features['images']  # A tensor of tf.strings
  processed_images = tf.map_fn(convert_jpeg_to_image_with_dim,
                               images, dtype=tf.float32)
  processed_images = tf.stack(processed_images)
  processed_images = tf.reshape(
      tensor=processed_images,
      shape=[-1, height, width, 3]
  )
  return processed_images


def postprocess_output(logits, k=5):
  '''Return top k classes and probabilities from class logits.'''
  probs = tf.nn.softmax(logits)  # Converts logits to probabilities.
  top_k_probs, top_k_classes = tf.nn.top_k(probs, k=k)
  return {'classes': top_k_classes, 'probabilities': top_k_probs}


def serving_input_to_output(features, mode, params):
  # Preprocess inputs before sending tensors to the network.
  img_size = params.image_size
  processed_images = preprocess_input(features, img_size, img_size)

  # Create ResNet network
  network = resnet_model.imagenet_resnet_v2(params.resnet_size,
                               params.num_labels,
                               data_format=params.data_format)

  # Assign inputs and outputs of network.
  logits = network(
    inputs=processed_images,
    is_training=(mode == tf.estimator.ModeKeys.TRAIN)
  )

  # Postprocess network outputs and send top k predictions as response
  predictions = postprocess_output(logits, k=params.top_k)
  return predictions


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model-dir',
      '-m',
      required=True,
      help='Load latest model training checkpoint from this directory'
  )
  parser.add_argument(
      '--output-dir',
      '-o',
      required=True,
      help='Output directory for the servable saved model'
  )
  parser.add_argument(
      '--top-k',
      '-k',
      type=int,
      default=5,
      help='Number of top classes and probabilities to return'
  )
  parser.add_argument(
      '--resnet-size',
      '-r',
      type=int,
      default=50,
      help='ResNet model size: must be 18, 34, 50, 101, 152, or 200. '
           'Must be set to specific size of model used to produce checkpoint.'
  )
  parser.add_argument(
      '--num-labels',
      '-n',
      type=int,
      default=1001,
      help='Number of classes that the model was trained on.'
  )
  parser.add_argument(
      '--image-size',
      '-i',
      type=int,
      default=224,
      help='Number of classes that the model was trained on.'
  )
  parser.add_argument(
      '--data-format',
      '-d',
      type=str,
      default='channels_last',
      help='Process images with \'channels_first\' (NCHW) or '
           '\'channels_last\' (NHWC) configuration. \'channels_first\' is '
           'generally used for GPU accelerators.'
  )

  args = parser.parse_args()


  def serving_model_fn(features, labels, mode, params):
    '''Define the Tensorflow model server input-output API.

    See https://github.com/tensorflow/models/blob/v1.4.0/official/resnet/imagenet_main.py#L162
    resnet_model_fn() for comparison of training and serving model functions.

    Args:
      features: the client request, in this case, a dictionary:
        {'image': 1D tensor of jpeg strings}
      labels: None or not used since we are predicting only
      mode: TRAIN, EVAL, or PREDICT. Serving only uses PREDICT mode.

    Returns:
      If training or evaluating (should not happen), return a blank
        EstimatorSpec that does nothing.
      If predicting (always), return an EstimatorSpec that produces a response
        with top k classes and probabilities to send back to the client.
    '''

    # Move preprocessing, network, and postprocessing into a helper function.
    predictions = serving_input_to_output(features, mode, params)

    # Create PREDICT EstimatorSpec to send a proper response back to the client.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,  # Not used in serving, but must be provided.
        export_outputs={
          'predict': tf.estimator.export.PredictOutput(outputs=predictions)
        },
      )

    # Training and evaluation are not needed.
    # Returning a minimal EstimatorSpec.
    return tf.estimator.EstimatorSpec(mode=mode)

  # Define the estimator and specify model checkpoint directory
  estimator = tf.estimator.Estimator(
    model_fn=serving_model_fn,
    model_dir=args.model_dir,
    params=args
  )

  def serving_input_receiver_fn():
    return tf.estimator.export.build_raw_serving_input_receiver_fn(
      {'images': tf.placeholder(dtype=tf.string, shape=[None])}
    )()

  # Load latest checkpoint and export saved model
  estimator.export_savedmodel(
    export_dir_base=args.output_dir,
    serving_input_receiver_fn=serving_input_receiver_fn
  )

if __name__ == '__main__':
  main()
