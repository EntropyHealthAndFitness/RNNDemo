import math
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

MODE = "LOAD"

NUM_WORDS = 10000
NUM_DIGITS = 30

# uses the example model given at 
# https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
class ProofOfConceptTrainingModel:
  def __init__(self):
    self.features = []
    self.labels   = []
    self.model    = None
    self.init_dataset()
    self.init_model(load_from_file="./TRAINED_MODEL.h5")

  def init_dataset(self):
    with open("dataset.txt", "r") as dataset:
      for line in dataset:
        words   = line.split()
        feature = words[0]
        label   = words[1]

        # convert to proper value
        feature = [np.uint8(int(digit)) for digit in feature]
        self.features.append(feature)

        label = np.uint8(math.floor(float(label) * 255.0))
        self.labels.append(label)

    self.features = np.array(self.features)
    self.labels   = np.array(self.labels)

  def init_model(self, load_from_file=None, save_to_file="./TRAINED_MODEL.h5"):
    # if a file is specified, LOAD, don't train
    if load_from_file != None:
      self.model = load_model(load_from_file)
    else:
      # initialize model for training
      self.model = Sequential()

      # initialie layers
      self.model.add(
          Embedding(input_dim=NUM_WORDS,
                    input_length=NUM_DIGITS,
                    output_dim=100,
                    trainable=False,
                    mask_zero=True))
      self.model.add(Masking(mask_value=0.0))
      self.model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
      self.model.add(Dense(64, activation='relu'))
      self.model.add(Dropout(0.5))
      self.model.add(Dense(NUM_WORDS, activation='softmax'))

      self.model.compile(
          optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

      # now that the model is initialized, train it!
      callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                   ModelCheckpoint('./TRAINED_MODEL.h5', save_best_only=True, save_weights_only=False)]

      self.model.fit(self.features, self.labels,
                     batch_size=2048, epochs=2000,
                     callbacks=callbacks)

      self.model.save(save_to_file)

  def perform_tests(self):

    # TODO i should really generate some random tests and
    # ensure that they are not elements within the actual
    # training set....

    generated_test_set = []
    expected_output_set = []
    NUM_TEST_SET = 200

    for i in range(NUM_TEST_SET):
      prob_roll = random.random()
      digit_string = "".join(["1" if random.random() < prob_roll else "0" for j in range(NUM_DIGITS)])
      digit_inp = np.array([np.uint8(int(d)) for d in digit_string])

      # calculate expected result
      truebits = 0
      for d in digit_inp:
        if d == 1:
          truebits += 1

      # push into list
      expected_percentage = truebits/float(NUM_DIGITS)*100.0
      generated_test_set.append(digit_inp)
      expected_output_set.append(expected_percentage)

    # numpy conversions
    generated_test_set  = np.array(generated_test_set)
    expected_output_set = np.array(expected_output_set)

    # do the evaluation
    predictions = self.model.predict(generated_test_set)

    sum_err = 0

    # check results!
    for input_value, pred, expected in zip(generated_test_set, predictions, expected_output_set):
      prediction_value = float(pred.argmax())/255.0*100.0
      percent_error    = abs(prediction_value - expected)
      sum_err          += percent_error
      print("code:     {}".format(input_value))
      print("AI guess: {:.2f}%".format(prediction_value))
      print("actual:   {:.2f}%".format(expected))
      print("% error:  {:.2f}%\n".format(percent_error))

    print("-------------------------------------------")
    print("average error: {:.2f}%".format(sum_err/NUM_TEST_SET))

if __name__ == "__main__":
  model = ProofOfConceptTrainingModel()
  model.perform_tests()
