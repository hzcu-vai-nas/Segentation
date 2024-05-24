from io import BytesIO
import scipy.misc
import tensorflow as tf

class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def image_summary(self, tag, image, step):
        with self.writer.as_default():
            tf.summary.image(tag, image, step=step)

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Expand dimensions to make it rank 4 tensor
                img = tf.expand_dims(img, axis=0)
                tf.summary.image("{}/{}".format(tag, i), img, step=step)


# Example usage:
# logger = Logger("logs")
# logger.scalar_summary("loss", 0.5, 1)
# logger.image_summary("image", image_data, 1)
# logger.image_list_summary("images", list_of_image_data, 1)
