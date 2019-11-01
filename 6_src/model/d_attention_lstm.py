import numpy as np
import pandas as pd
from config import config
import json
from keras.layers import Bidirectional, Input, LSTM, Softmax
from keras.layers import RepeatVector, Concatenate, Dense, Dot, Activation, Lambda
from keras.models import Model


class AttentionLstm():

    config_path = '../config/config.json'

    def __init__(self):
        self.load_config()

    def load_config(self):
        with open(AttentionLstm.config_path) as json_file:
            config = json.load(json_file)
        self.config = config
        self.dim = config['attention_lstm']['config']
        self.Tx = config['attention_lstm']['Tx']
        self.Ty = config['attention_lstm']['Ty']
        self.a_dim = config['attention_lstm']['a_dim']
        self.s_dim = config['attention_lstm']['s_dim']
        self.feature_dim = config['attention_lstm']['feature_dim']
        self.output_dim = config['attention_lstm']['output_dim']


    def compute_attention(self, a, s_prev):
        repeator = RepeatVector(self.Tx)
        s_prev = repeator(s_prev)

        concatenator = Concatenate(axis=-1)
        concat = concatenator([a, s_prev])

        densor = Dense(1, activation="relu")
        e = densor(concat) # e: scalar - un-normalized attention weight

        activator = Softmax(axis=-1)
        alphas = activator(e)  # alphas: scalar - normalized attention weight

        dotor = Dot(axes=1)
        context = dotor([alphas, a])

        return context

    def build_attention_lstm(self):
        # Step 1.1: Input
        X = Input(shape=(self.Tx, self.feature_dim))
        s0 = Input(shape=(self.s_dim,), name='s0')
        c0 = Input(shape=(self.s_dim,), name='c0')

        # Step 1.2: Pre-attention Bidirectional LSTM
        a = Bidirectional(LSTM(self.a_dim, return_sequences=True))(X)

        # Step 1.3: Post-attention Bidirectional LSTM
        post_attention_LSTM_cell = LSTM(self.s_dim, return_state=True)

        # Step 1.4: Output
        output_layer = Dense(self.output_dim, activation='softmax')

        # Step 2.1: Initialize
        outputs = []
        s = s0
        c = c0

        # Step 2.2: Iterate for Ty steps
        for t in range(self.Ty):
            context = self.compute_attention(a, s)
            s, _, c = post_attention_LSTM_cell(context, initial_state=[s, c])
            output = output_layer(s)
            outputs.append(output)

        # Step 3: Create model instance taking three inputs and returning the list of outputs
        model = Model(inputs=[X, s0, c0], outputs=outputs)

        return model

