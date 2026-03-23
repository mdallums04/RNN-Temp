import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.title("RNN Temperature Prediction (Jena Climate)")

timesteps = st.slider("Timesteps", 24, 720, 168)
hidden_units = st.slider("Hidden Units", 10, 200, 100)
epochs = st.slider("Epochs", 5, 50, 25)
learning_rate = st.selectbox("Learning Rate", [1e-2, 1e-3, 1e-4, 1e-5], index=1)

df = pd.read_csv('jena_climate_2009_2016.csv')
temp = df['T (degC)'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(temp)
scaled_data = scaled_data[:5000]

def create_sequences(data, timesteps):
    x, y = [], []
    for i in range(len(data) - timesteps):
        x.append(data[i:i + timesteps])
        y.append(data[i + timesteps])
    return np.array(x), np.array(y)

x, y = create_sequences(scaled_data, timesteps)
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]


class RNN:
    def __init__(self, input_size, hidden_units, output_size):
        self.hidden_units = hidden_units

        # FIX 3: Better weight initialization (Xavier-style) + bias terms added
        self.Wx = np.random.randn(hidden_units, input_size) * np.sqrt(2.0 / (input_size + hidden_units))
        self.Wh = np.random.randn(hidden_units, hidden_units) * np.sqrt(2.0 / (hidden_units + hidden_units))
        self.Wy = np.random.randn(output_size, hidden_units) * 0.1
        self.bh = np.zeros((hidden_units, 1))   # hidden bias
        self.by = np.zeros((output_size, 1)) + 0.5    # output bias

    def forward(self, x_seq):
        """Run forward pass, cache states needed for BPTT."""
        T = len(x_seq)
        hs = [np.zeros((self.hidden_units, 1))]  # h0

        for t in range(T):
            xt = x_seq[t].reshape(-1, 1)
            ht = np.tanh(self.Wx @ xt + self.Wh @ hs[t] + self.bh)
            hs.append(ht)

        yt = self.Wy @ hs[-1] + self.by
        return yt, hs

    def train(self, x_data, y_data, epochs, lr):
        self.losses = []

        for epoch in range(epochs):
            total_loss = 0

            # Accumulate gradients over all samples
            dWx = np.zeros_like(self.Wx)
            dWh = np.zeros_like(self.Wh)
            dWy = np.zeros_like(self.Wy)
            dbh = np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)

            for i in range(len(x_data)):
                # --- Forward pass ---
                yt, hs = self.forward(x_data[i])
                error = yt - y_data[i].reshape(-1, 1)   # (output_size, 1)
                total_loss += float(np.sum(error ** 2))

                # FIX 1: BACKPROPAGATION THROUGH TIME (BPTT)
                # Output layer gradients
                dWy += error * hs[-1].T
                dby += error

                # Backprop through hidden states
                dh_next = self.Wy.T @ error   # gradient flowing into last hidden state
                T = len(x_data[i])

                for t in reversed(range(T)):
                    xt = x_data[i][t].reshape(-1, 1)
                    ht = hs[t + 1]
                    ht_prev = hs[t]

                    # tanh derivative: (1 - tanh²)
                    dtanh = (1 - ht ** 2) * dh_next

                    dWx += dtanh @ xt.T
                    dWh += dtanh @ ht_prev.T
                    dbh += dtanh

                    # Gradient for previous hidden state
                    dh_next = self.Wh.T @ dtanh

            # Average gradients and clip to prevent exploding gradients
            n = len(x_data)
            for grad in [dWx, dWh, dWy, dbh, dby]:
                np.clip(grad, -5, 5, out=grad)

            # FIX 1 (cont.): Actually update the weights
            self.Wx -= lr * dWx / n
            self.Wh -= lr * dWh / n
            self.Wy -= lr * dWy / n
            self.bh -= lr * dbh / n
            self.by -= lr * dby / n

            self.losses.append(total_loss / n)

    def predict(self, x_data):
        outputs = []
        for x_seq in x_data:
            yt, _ = self.forward(x_seq)
            outputs.append(yt.flatten())
        return np.array(outputs)


# FIX 2: Train only once
if st.button("Train Model"):
    st.write("Training started...")
    with st.spinner("Training model... please wait ⏳"):
        rnn = RNN(input_size=x_train.shape[2], hidden_units=hidden_units, output_size=y_train.shape[1])
        rnn.train(x_train, y_train, epochs, learning_rate)

        preds = rnn.predict(x_test).reshape(y_test.shape)
        preds_rescaled = scaler.inverse_transform(preds)
        y_test_rescaled = scaler.inverse_transform(y_test)
        rmse = np.sqrt(np.mean((preds_rescaled - y_test_rescaled) ** 2))

    st.write(f"RMSE (°C): {rmse:.4f}")

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(rnn.losses)
    ax_loss.set_title("Training Loss Over Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")
    st.pyplot(fig_loss)

    fig, ax = plt.subplots()
    ax.plot(y_test_rescaled, label="Actual Temperature")
    ax.plot(preds_rescaled, label="Predicted Temperature")
    ax.set_title("Actual vs Predicted Temperature")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)