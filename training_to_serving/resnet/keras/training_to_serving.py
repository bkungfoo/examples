import argparse
import keras.applications.resnet50 as resnet50
import os
import tensorflow as tf
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

_DEFAULT_IMAGE_SIZE = 224


def convert_jpeg_to_image(encoded_image,
                          height=_DEFAULT_IMAGE_SIZE,
                          width=_DEFAULT_IMAGE_SIZE
                          ):
  '''Preprocesses the image by subtracting out the mean from all channels.
  Args:
    image: A jpeg-formatted byte stream represented as a string.
  Returns:
    A 3d tensor of image pixels normalized for the Keras ResNet50 model.
      The canned ResNet50 pretrained model was trained after running
      keras.applications.resnet50.preprocess_input in 'caffe' mode, which
      flips the RGB channels and centers the pixel around the mean [103.939, 116.779, 123.68].
      There is no normalizing on the range.
  '''
  image = tf.image.decode_jpeg(encoded_image, channels=3)
  # TODO: see https://github.com/tensorflow/tensorflow/issues/4290
  image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  image = tf.to_float(image)
  image = resnet50.preprocess_input(image)
  return image


def preprocess_input(jpeg_tensor):
  '''Convert an array of jpeg strings into a 4D tensor.'''
  processed_images = tf.map_fn(convert_jpeg_to_image, jpeg_tensor,
                               dtype=tf.float32)  # Convert list of JPEGs to a list of tensors
  processed_images = tf.stack(
    processed_images)  # Convert list of tensors to tensor of tensors
  processed_images = tf.reshape(tensor=processed_images,
                                # Reshape to ensure TF graph knows the final dimensions
                                shape=[-1, _DEFAULT_IMAGE_SIZE,
                                       _DEFAULT_IMAGE_SIZE, 3])
  return processed_images


def postprocess_output(model_output, top_k):
    '''Return top k classes and probabilities.'''
    top_k_probs, top_k_classes = tf.nn.top_k(model_output, k=top_k)
    return {'classes': top_k_classes, 'probabilities': top_k_probs}


def main():
  parser = argparse.ArgumentParser()
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
      '--num-labels',
      '-n',
      type=int,
      default=1001,
      help='Number of classes that the model was trained on.'
  )
  parser.add_argument(
      '--model-version',
      '-v',
      type=int,
      default=-1,
      help='Model version number (required). If not specified, will find the '
           'latest version and add 1 to the number.'
  )

  args = parser.parse_args()

  model_version = args.model_version
  output_dir = args.output_dir

  if model_version <= 0:
    if not os.path.exists(output_dir):
      model_version = 1
    else:
      model_version = max([int(x) for x in os.listdir(output_dir)]) + 1

  # Create input placeholder for client requests
  images = tf.placeholder(dtype=tf.string, shape=[None])

  # Preprocess jpeg strings to 4D tensor
  images_tensor = preprocess_input(images)

  # Connect image tensor to ResNet50 model
  model = resnet50.ResNet50(input_tensor=images_tensor)
  model.name = 'resnet'

  # Connect output of model to postprocessor to return top k classes and probs
  predictions = postprocess_output(model.output, args.top_k)

  # Create a builder with input and output nodes specified
  builder = saved_model_builder.SavedModelBuilder(
    os.path.join(output_dir, str(model_version))
  )
  signature = predict_signature_def(inputs={'images': images},
                                    outputs=predictions)

  # Export the graph to a saved model
  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={
                                           'predict': signature})
    builder.save()


if __name__ == '__main__':
  main()