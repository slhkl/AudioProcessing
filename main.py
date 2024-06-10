import numpy as np
import pandas as pd
import tensorflow as tf
from jiwer import wer
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from tensorflow import keras
from tensorflow.keras import layers

wavs_path = "datasets/LJSpeech-1.1/wavs/"
metadata_path = "datasets/LJSpeech-1.1/metadata.csv"

metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
print(metadata_df.head(3))

metadata_df = metadata_df[:300]

number_of_test = int(len(metadata_df) * 0.15)
number_of_val = int(len(metadata_df) * 0.10)

df_test = metadata_df[:number_of_test]
df_val = metadata_df[number_of_test:number_of_val+number_of_test]
df_train = metadata_df[number_of_val+number_of_test:]


print(f"Size of the training set: {len(df_train)}")
print(f"Size of the test set: {len(df_test)}")
print(f"Size of the validation set: {len(df_val)}")

# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)


# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384


def encode_single_sample(wav_file, label):
    ###########################################
    #  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    #  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label


batch_size = 4
# Define the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_test["file_name"]), list(df_test["normalized_transcription"]))
)
test_dataset = (
    test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def decode_batch_predictions(pred, gr=False):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=gr)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def decode_all_predictions(model, ds, gr=False):
    predictions = []
    targets = []
    for batch in ds:
        X, y = batch
        batch_predictions = model.predict(X, verbose=0)
        batch_predictions = decode_batch_predictions(batch_predictions, gr=gr)
        predictions.extend(batch_predictions)
        for label in y:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            targets.append(label)
    return targets, predictions


# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)


def build_model(input_dim,
                output_dim,
                number_of_filters_conv_1,
                number_of_filters_conv_2,
                number_of_units_lstm_1,
                number_of_units_gru_1,
                number_of_units_gru_2,
                learning_rate):
    input_spectrogram = layers.Input((None, input_dim))

    sal = layers.Conv1D(filters=number_of_filters_conv_1, kernel_size=5, padding="same")(input_spectrogram)
    sal = layers.ReLU()(sal)
    sal = layers.GRU(units=number_of_units_gru_1, return_sequences=True)(sal)
    sal = layers.ReLU()(sal)

    ih = layers.LSTM(units=number_of_units_lstm_1, activation="relu", return_sequences=True)(input_spectrogram)
    ih = layers.ReLU()(ih)
    ih = layers.Conv1D(filters=number_of_filters_conv_2, kernel_size=5, padding="same")(ih)
    ih = layers.ReLU()(ih)

    salih = layers.Concatenate(axis=-1)([sal, ih])
    salih = layers.LSTM(units=number_of_units_lstm_1, activation="relu", return_sequences=True)(salih)
    salih = layers.ReLU()(salih)
    salih = layers.GRU(units=number_of_units_gru_2, return_sequences=True)(salih)
    salih = layers.ReLU()(salih)

    output = layers.Dense(units=output_dim + 1, activation="softmax")(salih)
    model = keras.Model(input_spectrogram, output)
    opt = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5, clipvalue=0.5)
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


learning_rates = Real(low=1e-4, high=1e-1, prior='log-uniform', name="learning_rate")
numbers_of_filters_conv_1 = Integer(low=16, high=256, name="number_of_filters_conv_1")
numbers_of_filters_conv_2 = Integer(low=16, high=256, name="number_of_filters_conv_2")
numbers_of_units_lstm_1 = Integer(low=16, high=256, name="number_of_units_lstm_1")
numbers_of_units_gru_1 = Integer(low=16, high=256, name="number_of_units_gru_1")
numbers_of_units_gru_2 = Integer(low=16, high=256, name="number_of_units_gru_2")
epochs = Integer(low=5, high=100, name="epoch")

param_grid = [learning_rates, numbers_of_filters_conv_1,
              numbers_of_filters_conv_2, numbers_of_units_lstm_1,
              numbers_of_units_gru_1, numbers_of_units_gru_2, epochs]


param_file = open("params.txt", "w")
param_file.close()


best_wer = float('inf')
best_model = tf.keras.Model()


@use_named_args(dimensions=param_grid)
def call_model(learning_rate,
               number_of_filters_conv_1,
               number_of_filters_conv_2,
               number_of_units_lstm_1,
               number_of_units_gru_1,
               number_of_units_gru_2,
               epoch):
    global best_wer, best_model

    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        learning_rate=learning_rate,
        number_of_filters_conv_1=number_of_filters_conv_1,
        number_of_filters_conv_2=number_of_filters_conv_2,
        number_of_units_lstm_1=number_of_units_lstm_1,
        number_of_units_gru_1=number_of_units_gru_1,
        number_of_units_gru_2=number_of_units_gru_2
    )

    model.fit(train_dataset, validation_data=validation_dataset,
              epochs=epoch, verbose=2)

    targets, predictions = decode_all_predictions(model, validation_dataset)

    wer_score = wer(targets, predictions)

    print("wer:", wer_score)

    param_file = open("params.txt", "a")
    line_to_write = "wer_score\t" + str(wer_score)
    line_to_write = line_to_write + "\tlearning_rate\t" + str(learning_rate)
    line_to_write = line_to_write + "\tnumber_of_filters_conv_1\t" + str(number_of_filters_conv_1)
    line_to_write = line_to_write + "\tnumber_of_filters_conv_2\t" + str(number_of_filters_conv_2)
    line_to_write = line_to_write + "\tnumber_of_units_lstm_1\t" + str(number_of_units_lstm_1)
    line_to_write = line_to_write + "\tnumber_of_units_gru_1\t" + str(number_of_units_gru_1)
    line_to_write = line_to_write + "\tnumber_of_units_gru_2\t" + str(number_of_units_gru_2)
    param_file.write(line_to_write + "\n")
    param_file.close()

    if wer_score < best_wer:
        best_model = model
        best_wer = wer_score

    return wer_score


sr = gp_minimize(call_model, dimensions=param_grid, acq_func='EI', n_calls=10)


last_targets, last_predictions = decode_all_predictions(best_model, test_dataset)
wer_score = wer(last_targets, last_predictions)
print("Wer Score of Last Model at Test:", wer_score)
