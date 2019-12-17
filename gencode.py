import math
import random

DIGITS = 30
CODES = 10000

def make_dataset():
  with open("dataset.txt", "w") as outfile:
    for i in range(CODES):
      prob_offset = random.random() - 0.50
      digits = [math.floor(random.random() + 0.5 + prob_offset) for _ in range(DIGITS)]
      score = sum(digits)/DIGITS

      digit_str = "".join(str(d) for d in digits)
      outfile.write("{} {:.2f}\n".format(digit_str, score))

if __name__ == "__main__":
  make_dataset()
