import numpy as np
from keras.models import load_model
from seq_param import max_seq_length, input_token_index, target_token_index
from keras.utils import plot_model
import pydot_ng

model = load_model("s2s.h5")
#plot_model(model, to_file='model.png', show_shapes=True)

num_samples = 100
data_path = 'pairs.txt'
input_texts = []
target_texts = []
lines = open(data_path).read().split('\n')


for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split(' ')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)

num_encoder_tokens = len(input_token_index)
num_decoder_tokens = len(target_token_index)
max_encoder_seq_length = max_seq_length
max_decoder_seq_length = max_seq_length

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


