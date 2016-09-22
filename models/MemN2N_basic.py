import random
import numpy as np
import tensorflow as tf

import os
import sys

import time

random.seed(1003)
np.random.seed(1003)

class MemN2N_QA_Basic(object): # m_i = A * x_i, version of Bow
    def __init__(self, config, sess, train, test):
        
        self.sess = sess
        
        self.basisConfig = config.basisConfig
        self.dictionary = config.dictionary
        self.voca_size = config.voca_size
        self.epi_size = config.epi_size
        self.max_words = config.max_words
        self.edim = config.edim


        self.train_story, self.train_questions, self.train_qstory = train
        self.test_story, self.test_questions, self.test_qstory = test

        self.batch_size = config.batch_size
        self.nhops = config.nhops

        self.init_std = config.init_std
        self.max_iter = config.max_iter

        self.display_interval = config.display_interval
        self.test_interval = config.test_interval

        self.g_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(config.learning_rate, self.g_step, len(config.train_range), 0.9)
        # self.lr_decay_op = self.lr.assign(self.lr * 0.98)
        
        self.train_range = config.train_range
        self.test_len = self.test_questions.shape[1]

        self.input_querys = tf.placeholder(tf.int32, [self.batch_size, 1, self.max_words])
        self.input_episodes = tf.placeholder(tf.int32, [self.batch_size, self.epi_size, self.max_words])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.voca_size])

        self.W = tf.Variable(tf.random_normal([self.edim, self.voca_size], stddev = self.init_std))
        self.A = tf.Variable(tf.random_normal([self.voca_size, self.edim], stddev = self.init_std))
        self.B = tf.Variable(tf.random_normal([self.voca_size, self.edim], stddev = self.init_std))
        self.C = tf.Variable(tf.random_normal([self.voca_size, self.edim], stddev = self.init_std))

        self.acc = 0

    def qa_model(self, data):
        # M = Memory
        M = tf.reduce_sum(tf.nn.embedding_lookup(self.A, data['input_episodes']), reduction_indices = 2)
        C = tf.reduce_sum(tf.nn.embedding_lookup(self.C, data['input_episodes']), reduction_indices = 2)
    
        u = tf.reduce_sum(tf.nn.embedding_lookup(self.B, data['input_querys']), reduction_indices = 2)
        pdb.set_trace()
        for h in xrange(self.nhops):
            
            p = tf.reshape(tf.batch_matmul(u, M, adj_y=True), [-1, self.epi_size])
            p = tf.reshape(tf.nn.softmax(p), [-1, self.epi_size, 1])

            o = tf.reduce_sum(tf.mul(C, p), reduction_indices = 1)
            
            o = tf.reshape(o, [-1, 1, self.edim])
            u = tf.add(o, u)
        
        u = tf.reshape(u, [-1, self.edim])
        self.z = tf.matmul(u, self.W)

        return self.z
            
# Memory should consist of coef
# ...
    def build_model(self, task_id):
        self.task_id = task_id
        print(" [*] Building Model...")
        self.train()

        self.test_data()
        self.test()
        print(" [*] Build Done")

        tf.initialize_all_variables().run()
        #self.saver = tf.train.Saver()

    def run(self):

        for epoch in xrange(self.max_iter+1):
            # pdb.set_trace()
            feed_dict = self.train_data()
            _, l, step, preds, lr = self.sess.run([self.opt, self.train_loss, self.g_step, self.train_prediction, self.lr], feed_dict = feed_dict)
            if (epoch % self.display_interval == 0):
                acc = accuracy(preds, feed_dict.values()[2])
                print(time.ctime() + " iter %d, train loss: %f, lr: %f, train_acc: %.1f%%" % (epoch, l, lr, acc))

            if (epoch % self.test_interval == 0):
                #self.saver.save(self.sess, '/home/jtlee/projects/NIPS/BasisMemQA/snapshots' + '/embow-model_t' + str(self.task_id), global_step=epoch)
                self.acc = accuracy(self.test_prediction.eval(), self.test_labels)
                print(time.ctime() + " Test accuracy: %.1f%%" % self.acc)
        print(time.ctime() + " Test accuracy: %.1f%%" % self.acc)
        print("Iteration terminate...")

    def train(self):
        data = { 'input_querys': self.input_querys, 'input_episodes': self.input_episodes }
        logits = self.qa_model(data)
        self.train_prediction = tf.nn.softmax(logits)
        self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.labels))
        regularizers = tf.nn.l2_loss(self.A) + tf.nn.l2_loss(self.B) + tf.nn.l2_loss(self.C)
        
        self.train_loss += 5e-4 * regularizers

        self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.train_loss, global_step = self.g_step)

    def train_data(self):
        batch = self.train_range[np.random.randint(len(self.train_range), size = self.batch_size)]
        batch_querys = np.zeros( (self.batch_size, 1, self.max_words), np.int32 )
        batch_labels = np.zeros( (self.batch_size, self.voca_size), np.float32 )
        batch_episodes = np.zeros( (self.batch_size, self.epi_size, self.max_words), np.int32 )

        for b in range(self.batch_size):
            d = self.train_story[:, :(1 + self.train_questions[1, batch[b]]), self.train_questions[0, batch[b]]]
            offset = max(0, d.shape[1] - self.epi_size)

            batch_episodes[b, :d.shape[1], :d.shape[0]] = d[:, offset:].T
            batch_querys[b, 0, :] = self.train_qstory[:, batch[b]]
            label = self.train_questions[2, batch[b]]   
            batch_labels[b, :] = bow(self.dictionary, np.array([label]))

        feed_dict = { self.input_querys: batch_querys, self.input_episodes: batch_episodes, self.labels: batch_labels }
        return feed_dict

    def test(self):
        data = { 'input_querys': self.test_querys, 'input_episodes': self.test_episodes }
        self.test_prediction = tf.nn.softmax(self.qa_model(data))

    def test_data(self):
        print(" [*] Test Data Processing...")
        self.test_querys = np.zeros( (self.test_len , 1, self.max_words), np.int32 )
        self.test_labels = np.zeros( (self.test_len , self.voca_size), np.float32 )
        self.test_episodes = np.zeros( (self.test_len , self.epi_size, self.max_words), np.int32 )

        for b in range(self.test_len):
            d = self.test_story[:, :(1 + self.test_questions[1, b]), self.test_questions[0, b]]
            offset = max(0, d.shape[1] - self.epi_size)
            self.test_episodes[b, :d.shape[1], :d.shape[0]] = d[:, offset:].T
            self.test_querys[b, 0, :] = self.test_qstory[:, b]
            label = self.test_questions[2, b]
            self.test_labels[b, :] = bow(self.dictionary, np.array([label]))
        
        self.test_querys = tf.constant(self.test_querys)
        self.test_episodes = tf.constant(self.test_episodes)
        
        print(" [*] Test Data Processing Done")

    def writeResult(self, path):
        fout = open(path, 'a')
        fout.write("task " + str(self.task_id) + ", accuracy: " + str(self.acc) + "\n")
        fout.close()
