from keras.models import load_model
from seq_param import max_seq_length, input_token_index, target_token_index
from keras.utils import plot_model
from lstm_seq2seq import *
from seq_param import *
import pydot_ng

model = load_model("s2s.h5")
plot_model(model, to_file='model.png', show_shapes=True)

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

# liczba slow, max dlugosc pojedynczej sekwencji, dlugosc slownika
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

print(input_texts)
print('#')
print(target_text)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.






# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())




for seq_index in range(100):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

while True:
    test_seq = input("Input word to rhyme: ")
    if test_seq == "":
        break
    test_np_array = generate_sequence(test_seq, max_encoder_seq_length, num_encoder_tokens)
    print("\noutput :", decode_sequence(test_np_array))



