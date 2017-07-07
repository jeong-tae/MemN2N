import random
import numpy as np
import tensorflow as tf
from tqdm import trange
import os
import sys
import time
from ops import *

random.seed(1003)
np.random.seed(1003)

class MemN2N_QA_Basic(object): # m_i = A * x_i, version of Bow
    def __init__(self, config, sess, train, test):

        self.sess = sess

        self.train_story, self.train_questions, self.train_qstory = train
        self.test_story, self.test_questions, self.test_qstory = test

        self.lr = tf.Variable(config.learning_rate, trainable = False, name = 'lr')
        self.lr_anneal = config.lr_anneal
        self.batch_size = config.batch_size
        self.nhops = config.nhops
        self.init_std = config.init_std
        self.max_epoch = config.max_epoch
        self.dictionary = config.dictionary
        self.voca_size = len(self.dictionary)
        self.max_mem_size = min(config.max_mem_size, self.train_story.shape[1])
        # lookup_size for word and time embedding together
        self.lookup_size = self.voca_size + self.max_mem_size
        self.max_words = len(self.train_story) # we can define separated buckets for this
        self.edim = config.edim
        self.max_grad_norm = config.max_grad_norm
        self.PE = config.PE
        self.TE = config.TE
        self.LS = config.LS
        self.RN = config.RN
        self.weight_tying = config.weight_tying
        self.initializer = tf.random_normal_initializer(stddev = self.init_std)

		# self.display_interval = config.display_interval
		# self.test_interval = config.test_interval

        self.train_len = self.train_questions.shape[1]
        self.test_len = self.test_questions.shape[1]

        self.input_querys = tf.placeholder(tf.int32, [None, 1, self.max_words+1])
        self.input_episodes = tf.placeholder(tf.int32, [None, self.max_mem_size, self.max_words+1])
        self.labels = tf.placeholder(tf.float32, [None, self.lookup_size])

    def get_variable(self, name, shape, stddev = None, scope = None, reuse = False):
        
        if scope:
            pass
        else:
            if stddev != None and reuse == False:
                w = tf.get_variable(name, shape, initializer = tf.truncated_normal_initializer(stddev = stddev), trainable = True)
                return w
            elif reuse:
                with tf.variable_scope(tf.get_variable_scope(), reuse = reuse):
                    w = tf.get_variable(name, trainable = True)
                return w

    def build_vars(self):
        nil_word = tf.zeros([1, self.edim])
        A = tf.concat(axis = 0, values = [ nil_word, self.initializer([self.lookup_size-1, self.edim]) ])
        C = tf.concat(axis = 0, values = [ nil_word, self.initializer([self.lookup_size-1, self.edim]) ])

        emb_A = tf.Variable(A, name = 'A1')
        emb_C = tf.Variable(C, name = 'C1')
        self.emb_A, self.emb_C = [emb_A], [emb_C]
        self.nil_vars = set([emb_A.name] + [emb_C.name])

        if self.weight_tying == 'LW':
            linear_H = tf.Variable(self.initializer([self.edim, self.edim]),
                        name = 'linear_H')
            self.linear_H = [linear_H]

        for h in range(1, self.nhops):
            if self.weight_tying == 'Adj':
                emb_A = emb_C
                emb_C = tf.Variable(tf.concat(axis = 0, values = [ nil_word, self.initializer([self.lookup_size-1, self.edim]) ]), name = 'C%d'%(h+1))

            elif self.weight_tying == 'LW':
                self.linear_H.append(linear_H)

            self.emb_A.append(emb_A)
            self.emb_C.append(emb_C)

            self.nil_vars.update([emb_A.name] + [emb_C.name])

    def nil_grad_to_zero(self, grad, name = None):
        # dummy should not be trained
        with tf.name_scope(name, "zero_grad", [grad]) as scope:
            t = tf.convert_to_tensor(grad, name = "t")
            s = tf.shape(t)[1]
            z = tf.zeros(tf.stack([1, s]))
            return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name = scope)

    def add_gradient_noise(self, grad, stddev = 1e-3, name = None):
        with tf.name_scope(name, "grad_noise", [grad, stddev]) as scope:
            t = tf.convert_to_tensor(grad, name = "t")
            gn = tf.random_normal(tf.shape(t), stddev = stddev)
            return tf.add(t, gn, name = scope)

    def qa_model(self, task_id):
        self.task_id = task_id
        print(" [*] Model building...")
        self.build_vars()

        if self.PE:
            pos_emb = np.ones((self.edim, self.max_words+1), np.float32)
            for j in range(1, self.edim+1):
                for k in range(1, self.max_words+1):
                    pos_emb[j-1, k-1] = (j - (self.edim+1)/2) * (k - (self.max_mem_size+1)/2)
            pos_emb = 1 + 4 * pos_emb / self.edim / self.max_mem_size
            # To void time word modification, last word is for time embedding
            pos_emb[:, -1] = 1.0
            self.pos_emb = np.transpose(pos_emb)

        #if self.RN:

        # m_i = sum A_ij * x_ij
        u = tf.nn.embedding_lookup(self.emb_A[0], self.input_querys)
        if self.PE:
            u = u * self.pos_emb
        u = tf.reduce_sum(u, reduction_indices = 2)
            
        for h in xrange(self.nhops):

            # m_i = sum A_ij * x_ij
            mem_M = tf.nn.embedding_lookup(self.emb_A[h], self.input_episodes)
            mem_C = tf.nn.embedding_lookup(self.emb_C[h], self.input_episodes)
            if self.PE:
                mem_M = mem_M * self.pos_emb
                mem_C = mem_C * self.pos_emb

            mem_M = tf.reduce_sum(mem_M, reduction_indices = 2)
            mem_C = tf.reduce_sum(mem_C, reduction_indices = 2)

            p = tf.reshape(tf.reduce_sum(u * mem_M, reduction_indices = 2), [-1, self.max_mem_size])
            p = tf.reshape(tf.nn.softmax(p), [-1, self.max_mem_size, 1])
            o = tf.reduce_sum(mem_C * p, reduction_indices = 1)
            o = tf.reshape(o, [-1, 1, self.edim])
            u = tf.add(o, u)
    
            if self.weight_tying == "LW": # linear project
                u = tf.reshape(u, [-1, self.edim])
                u = tf.add(tf.reshape(tf.matmul(u, self.linear_H[h]), [-1, 1, self.edim]), o)
                
            #else:
                #raise ValueError(" [!] Invaild weight tying type: %s"%self.weight_tying)

        if self.weight_tying == "Adj":
            self.W = tf.transpose(self.emb_C[self.nhops-1], [1, 0])
        else:
            self.W = tf.Variable(tf.random_normal([self.edim, self.lookup_size],
                            stddev = self.init_std), name = 'final_output')

        self.u = tf.reshape(u, [-1, self.edim])
        self.logits = tf.matmul(self.u, self.W)
        self.preds = tf.nn.softmax(self.logits)
        print(" [*] Model build done!")

    def optimization(self):
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.labels))
        opt = tf.train.GradientDescentOptimizer(self.lr)
        grads_vars = opt.compute_gradients(self.loss)
        grads_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_vars]
        grads_vars = [(self.add_gradient_noise(g), v) for g, v in grads_vars]
        nil_grads_vars = []
        for g, v in grads_vars:
            if v.name in self.nil_vars:
                nil_grads_vars.append((self.nil_grad_to_zero(g), v))
            else:
                nil_grads_vars.append((g, v))

        self.opt = opt.apply_gradients(nil_grads_vars, name = 'apply_grad')

    def run_epoch(self, dgen, desc = ""):
        iter_per_epoch = self.train_len / self.batch_size
        loss, acc = 0., 0.
        old_loss, counting = 999., 0
        pbar1 = trange(self.max_epoch)
        for epoch in pbar1:
            pbar1.set_description('Overall epoch')
            if ((epoch+1) % self.lr_anneal) == 0:
                self.lr.assign(self.lr * 0.5).eval()
            pbar2 = trange(iter_per_epoch)
            for step in pbar2:
                pbar2.set_description('Current epoch')
                querys, episodes, labels = dgen.next()
                _, loss, pred = self.sess.run([self.opt, self.loss, self.preds], 
                            feed_dict = { self.input_querys: querys,
                                        self.input_episodes: episodes,
                                        self.labels: labels 
                                        }
                            )
                acc = accuracy(pred, labels)
                pbar2.set_postfix(loss = loss, train_acc = "%.2f%%"%acc)

            test_acc, test_loss = self.test()
            pbar1.set_postfix(lr = self.sess.run(self.lr), loss = test_loss, test_acc = "%.2f%%"%test_acc)
            if old_loss > test_loss: 
                # no validation data, so we use test data as validation
                self.saver.save(self.sess,
                        os.path.join("checkpoints/", 
                                "memN2N_task%d_%s.model"%(self.task_id, desc)),
                        global_step = epoch)
                old_loss = test_loss
                counting = 0
            else:
                counting += 1
            # if you do validation check per step, indent all test code and change the ealy stopping criteria
            #if counting > 50:
                #break

    def train(self):
        dgen = self.data_iteration(desc = "train")

        self.run_epoch(dgen)

        if self.LS:
            self.lr.assign(0.005).eval()
            self.RW = tf.Variable(tf.random_normal([self.edim, self.lookup_size],
                            stddev = self.init_std), name = 'LS_softmax')
            self.logits = tf.matmul(self.u, self.RW)
            self.preds = tf.nn.softmax(self.logits)
            print(" [*] Training recommence")
            self.optimization()
            self.sess.run(tf.global_variables_initializer())
            self.run_epoch(dgen, desc = 'LS')
            print(" [*] Learning end...")

    def test(self):
        dgen = self.data_iteration(desc = "test")
        iter_per_epoch = self.test_len / self.batch_size

        actual_label = []
        model_pred = []
        total_loss = []
        for step in range(iter_per_epoch):
            querys, episodes, labels = dgen.next()
            loss, pred = self.sess.run([self.loss, self.preds], 
                        feed_dict = { self.input_querys: querys,
                                    self.input_episodes: episodes,
                                    self.labels: labels 
                                    }
                        )
            total_loss.append(loss)
            actual_label.append(labels)
            model_pred.append(pred)
        total_loss = np.mean(total_loss)
        actual_label = np.concatenate(actual_label, axis = 0)
        model_pred = np.concatenate(model_pred, axis = 0)
        return accuracy(actual_label, model_pred), total_loss

    def data_iteration(self, desc = 'Train'):
        step = 0 # step for test data
        # in the paper, sentences are indexed in reverse order but no reversed in here
        while True:
            querys = np.zeros((self.batch_size, 1, self.max_words+1), np.int32)
            # +1 for time embedding
            episodes = np.zeros((self.batch_size, self.max_mem_size, self.max_words+1), np.int32)
            labels = np.zeros((self.batch_size, self.lookup_size), np.float32)

            batch_idxs = np.zeros((self.batch_size), np.int32)
            if desc.lower() == "train":
                batch_idxs = np.arange(self.train_len)[np.random.randint(self.train_len, size = self.batch_size)]
                data_story = self.train_story
                data_qstory = self.train_qstory
                data_questions = self.train_questions
            else:
                batch_offset = (step * self.batch_size) % self.test_len
                if batch_offset < self.batch_size and batch_offset != 0:
                    batch_idxs[:self.batch_size - batch_offset] = np.arange((step-1) * self.batch_size, self.test_len)
                    batch_idxs[self.batch_size - batch_offset:] = np.arange(batch_offset)
                else:
                    batch_idxs = np.arange(batch_offset, batch_offset + self.batch_size)
                data_story = self.test_story
                data_qstory = self.test_qstory
                data_questions = self.test_questions

            for idx, b in enumerate(batch_idxs):
                d = data_story[:, :(1 + self.train_questions[1, b]), self.train_questions[0, b]]
                offset = max(0, d.shape[1] - self.max_mem_size)
                querys[idx, 0, :self.max_words] = data_qstory[:, b]
                episodes[idx, :d.shape[1], :d.shape[0]] = d[:, offset:].T
                import pdb
                for t in range(d.shape[1] - offset):
                    episodes[idx, t, -1] = self.lookup_size - self.max_mem_size - t + (d.shape[1] - offset) - 1
                label = data_questions[2, b]
                labels[idx, :] = bow(self.lookup_size, np.array([label]))
            yield querys, episodes, labels
