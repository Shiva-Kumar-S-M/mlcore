import numpy as np
from neural_network import NeuralNetworkScratch, relu, relu_derivative, sigmoid


def test_relu_zeros_out_negatives():
    z = np.array([-5, -1, 0, 1, 5])
    assert np.array_equal(relu(z), [0, 0, 0, 1, 5])


def test_relu_derivative_is_step_function():
    z = np.array([-2, -1, 0, 1, 2])
    assert np.array_equal(relu_derivative(z), [0, 0, 0, 1, 1])


def test_network_solves_xor():
    """XOR is the classic example a single linear model CANNOT solve
    but a network with a hidden layer can -- this is the textbook
    proof that hidden layers add real representational power."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])  # XOR pattern

    model = NeuralNetworkScratch(input_size=2, hidden_size=4,
                                    learning_rate=0.5, n_iterations=5000)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y)


def test_loss_decreases_over_training():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    model = NeuralNetworkScratch(input_size=2, hidden_size=4,
                                    learning_rate=0.5, n_iterations=2000)
    model.fit(X, y)

    assert model.loss_history[-1] < model.loss_history[0]


if __name__ == "__main__":
    test_relu_zeros_out_negatives()
    test_relu_derivative_is_step_function()
    test_network_solves_xor()
    test_loss_decreases_over_training()
    print("All tests passed!")