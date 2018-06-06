import tensorflow as tf
import numpy as np


def lstm_dnn_construct(num_of_element = 51,
                               num_of_hidden_nodes = 1000,
                               num_layers = 5,
                               num_of_frm = 10,
                               hidden_size = [1000,500,100,51],
                               leaning_rate = 0.50):
    
    input_ph = tf.placeholder(tf.float32, [None, num_of_frm, num_of_element])
    sup_ph = tf.placeholder(tf.float32, [None, num_of_element])
    
    num_of_output_lstm_nodes = num_of_hidden_nodes
    #lstm
    lstm_w1 = tf.Variable(tf.truncated_normal([num_of_element, num_of_hidden_nodes], stddev=0.1))
    lstm_b1 = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1))
    lstm_w2 = tf.Variable(tf.truncated_normal( [num_of_hidden_nodes, num_of_output_lstm_nodes], stddev=0.1))
    lstm_b2 = tf.Variable(tf.truncated_normal([num_of_output_lstm_nodes], stddev=0.1))
    
    in1 = tf.transpose(input_ph, [1, 0, 2])
    in2 = tf.reshape(in1, [-1, num_of_element])
    in3 = tf.matmul(in2, lstm_w1) + lstm_b1
    in4 = tf.split(in3, num_of_frm, 0)
    
    lstm_state_ph = tf.placeholder(tf.float32, [num_layers, 2, None, num_of_hidden_nodes])
    l = tf.unstack(lstm_state_ph, axis=0)
    rnn_tuple_state = tuple( [tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1]) for idx in range(num_layers)] )
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_of_hidden_nodes, state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)
    lstm_output, lstm_states_op = tf.contrib.rnn.static_rnn(cell, in4, initial_state = rnn_tuple_state)
    lstm_output_op = tf.matmul(lstm_output[-1], lstm_w2) + lstm_b2
    
    #dnn?
    fc1_w = tf.Variable(tf.truncated_normal([num_of_output_lstm_nodes, hidden_size[0]], stddev=0.1), dtype=tf.float32)
    fc1_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[0]]), dtype=tf.float32)
    fc1 = tf.nn.sigmoid(tf.matmul(lstm_output_op, fc1_w) + fc1_b)

    fc2_w = tf.Variable(tf.truncated_normal([hidden_size[0], hidden_size[1]], stddev=0.1), dtype=tf.float32)
    fc2_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[1]]), dtype=tf.float32)
    fc2 = tf.nn.sigmoid(tf.matmul(fc1, fc2_w) + fc2_b)

    fc3_w = tf.Variable(tf.truncated_normal([hidden_size[1], hidden_size[2]], stddev=0.1), dtype=tf.float32)
    fc3_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[2]]), dtype=tf.float32)
    fc3 = tf.nn.sigmoid(tf.matmul(fc2, fc3_w) + fc3_b)

    fc4_w = tf.Variable(tf.truncated_normal([hidden_size[2], hidden_size[3]], stddev=0.1), dtype=tf.float32)
    fc4_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[3]]), dtype=tf.float32)
    fc4 = tf.nn.sigmoid(tf.matmul(fc3, fc4_w) + fc4_b)

    fc5_w = tf.Variable(tf.truncated_normal([hidden_size[3], num_of_element], stddev=0.1), dtype=tf.float32)
    fc5_b = tf.Variable(tf.constant(0.1, shape=[num_of_element]), dtype=tf.float32)
    y_pre = tf.matmul(fc4, fc5_w) + fc5_b
    predict = y_pre
    cross_entropy = tf.sqrt(tf.reduce_mean((y_pre - sup_ph)**2))
    train_step = tf.train.AdamOptimizer(leaning_rate).minimize(cross_entropy)
    return input_ph, sup_ph, lstm_state_ph,cross_entropy, train_step, predict