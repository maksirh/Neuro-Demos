import numpy as np
letters = {
 "M": [1,0,1,0,1,
       1,1,0,1,1,
       1,0,0,0,1,],

 "A": [1,0,0,0,1,
       1,1,1,1,1,
       1,0,0,0,1],

 "K": [1,0,0,1,0,
       1,1,1,0,0,
       1,0,0,1,0],

 "S": [0,1,1,1,0,
       1,0,0,0,1,
       0,1,1,1,0]
}

targets = {
 "M": [1,0,0,0],
 "A": [0,1,0,0],
 "K": [0,0,1,0],
 "S": [0,0,0,1]
}

inputs = np.array(list(letters.values()))
outputs = np.array(list(targets.values()))
weights = np.zeros((inputs.shape[1], outputs.shape[1]))
learning_rate = 0.1


def train_hebb(inputs, outputs):
    global weights
    for x, y in zip(inputs, outputs):
        weights += learning_rate * np.outer(x, y)

train_hebb(inputs, outputs)

def predict(input_vector):
    output = np.dot(input_vector, weights)
    return np.argmax(output)

test_letters = ["M", "A", "K", "S"]

for l in test_letters:
    input_vector = letters[l]
    predicted_index = predict(input_vector)
    predicted_letter = list(targets.keys())[predicted_index]
    print(f"Input: {l}, Predicted: {predicted_letter}")

noisy_M =  [1,0,1,1,1,
            1,1,0,1,1,
            1,0,1,0,1]

predicted_index = predict(noisy_M)
predicted_letter = list(targets.keys())[predicted_index]
print(f"Input: Noisy M, Predicted: {predicted_letter}")
