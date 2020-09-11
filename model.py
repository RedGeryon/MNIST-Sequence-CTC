import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class CTC(layers.Layer):
    def __init_(self, name=None):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Compute CTC loss, add directly; return preds
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At run time, just return the computed predictions
        return y_pred


class Embedding(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def conv_block(self, x, filters, kernel_size, pool_stride):
        x = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool_stride)(x)
        return x

    def clstm(self, input_img):
        x = self.conv_block(input_img,
                            filters=32,
                            kernel_size=3,
                            pool_stride=(2,2))
        x = self.conv_block(x,
                            filters=64,
                            kernel_size=3,
                            pool_stride=(2,2))

        # Image is now 4x smaller due to 2 pooling layers; to fit into RNN we need to
        # reshape by stacking across channels for each position in the width dimension.
        # We do this by multiplying by the num of channels (our final num of filters)
        new_dim = (self.w//4, 64 * self.h//4 )
        x = layers.Reshape(target_shape=new_dim, name="stack_channels")(x)

        # We can add an FC layer to compress each "timestep" in the width dimension
        # Width ("timesteps") will remain the same
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(.2)(x)

        # Now we can stack the LSTMs
        x = layers.LSTM(128, return_sequences=True, dropout=.2)(x)
        x = layers.LSTM(64, return_sequences=True, dropout=.2)(x)

        # Output has 10 digits plus 1 for empty string
        x = layers.Dense(11, activation="softmax", name='logits')(x)
        return x

    def ctc_compile(self):
        # Instantiate inputs and labels
        input_img = layers.Input(
            shape=(self.w, self.h, 1),
            dtype="float32",
            name="batch_images"
        )
        labels = layers.Input(shape=(None,), dtype="float32", name="labels")

        # Run image batch through our cnn-lstm embedding
        x = self.clstm(input_img)
        # Find loss through batch CTC
        output = CTC(name="CTC_preds")(labels, x)

        model = keras.models.Model(inputs=[input_img, labels], outputs=output)
        opt = keras.optimizers.Adam(lr=1e-4)
        model.compile(optimizer=opt)
        return model


class Inference(object):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def ctc_decoder(preds, label_len):
        # Since this is a simple example, we wont bother with beamsearch
        input_length = preds.shape[1] * np.ones(preds.shape[0])
        decoded = K.ctc_decode(preds, input_length, greedy=True)
        # Return only up to our desired label length
        return decoded[0][0][:, : label_len]

    def predict(self, data, label_len):
        # Extract the logit layer output > pass through model > decode
        inputs = self.model.get_layer(name="batch_images").input
        logits = self.model.get_layer(name="logits").output
        pred_model = keras.models.Model(inputs, logits)
        decoded = self.ctc_decoder(pred_model(data), label_len)
        return decoded.numpy()

