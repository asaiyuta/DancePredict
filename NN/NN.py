import tensorflow as tf
import numpy as np

def position2position_cnstruct(num_of_element = 75,
                               num_of_frm = 10,
                               hidden_size = [1000,500,100,75],
                               f_tuning = False,
                               leaning_rate = 0.01
                              ):
    #
    num_of_input_nodes = num_of_element * num_of_frm
    num_of_output_nodes = num_of_element
    input_elements_ph = tf.placeholder(tf.float32, (None, num_of_frm, num_of_element))
    supervisor_elements_ph = tf.placeholder(tf.float32, (None, num_of_output_nodes))
    in1_res = tf.reshape(input_elements_ph, [-1, num_of_input_nodes])

    fc1_w = tf.Variable(tf.truncated_normal([num_of_input_nodes, hidden_size[0]], stddev=0.1), dtype=tf.float32) # 入力層の重み
    fc1_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[0]]), dtype=tf.float32)
    fc1 = tf.nn.sigmoid(tf.matmul(in1_res, fc1_w) + fc1_b)

    fc2_w = tf.Variable(tf.truncated_normal([hidden_size[0], hidden_size[1]], stddev=0.1), dtype=tf.float32) # 隠れ層の重み
    fc2_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[1]]), dtype=tf.float32) # 隠れ層のバイアス
    fc2 = tf.nn.sigmoid(tf.matmul(fc1, fc2_w) + fc2_b)

    fc3_w = tf.Variable(tf.truncated_normal([hidden_size[1], hidden_size[2]], stddev=0.1), dtype=tf.float32) # 隠れ層の重み
    fc3_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[2]]), dtype=tf.float32) # 隠れ層のバイアス
    fc3 = tf.nn.sigmoid(tf.matmul(fc2, fc3_w) + fc3_b)

    fc4_w = tf.Variable(tf.truncated_normal([hidden_size[2], hidden_size[3]], stddev=0.1), dtype=tf.float32) # 隠れ層の重み
    fc4_b = tf.Variable(tf.constant(0.1, shape=[hidden_size[3]]), dtype=tf.float32) # 隠れ層のバイアス
    fc4 = tf.nn.sigmoid(tf.matmul(fc3, fc4_w) + fc4_b)

    fc5_w = tf.Variable(tf.truncated_normal([hidden_size[3], num_of_output_nodes], stddev=0.1), dtype=tf.float32, name="fine_tune_w") # 出力層の重み
    fc5_b = tf.Variable(tf.constant(0.1, shape=[num_of_output_nodes]), dtype=tf.float32, name="fine_tune_b") # 出力層のバイアス
    y_pre = tf.matmul(fc4, fc5_w)
    predict = y_pre
    cross_entropy = tf.sqrt(tf.reduce_mean((y_pre - supervisor_elements_ph)**2))
    train_step = 0
    if f_tuning:
        train_step = tf.train.AdamOptimizer(leaning_rate).minimize(cross_entropy, var_list=[fc4_w, fc4_b, fc5_w, fc5_b]) 
    else:
        train_step = tf.train.AdamOptimizer(leaning_rate).minimize(cross_entropy)
    return input_elements_ph, supervisor_elements_ph, cross_entropy, train_step, predict