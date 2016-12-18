import os,sys,random
sys.path.insert(0,"../../python")
import numpy as np
import mxnet as mx
import string 
import math

from lstm import lstm_unroll

INPUT_SIGNAL_TYPES = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
                      "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
                      "total_acc_x_", "total_acc_y_", "total_acc_z_"]

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]

DATA_PATH = "data/"
TRAIN = "train/"
TEST = "test/"

BATCH_SIZE = 150

def load_X(X_signals_paths):
    X_signals = []
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'rb')
        X_signals.append([np.array(serie, dtype=np.float32) for serie in [
                        row.replace('  ',' ').strip().split(' ') for row in file
                         ]])
        file.close()
    return np.transpose(np.array(X_signals), (1,2,0))

def load_y(y_path):
    file = open(y_path, 'rb')
    y_ = np.array([elem for elem in [row.replace('  ',' ').strip().split(' ')
    for row in file]], dtype=np.int32)
    file.close()
    return y_ -1

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None

    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.label_names, self.label)]

def Accuracy(label, pred):
    SEQ_LEN = 128
    shape_1 = pred.shape
    shape_2 = label.shape
    hit = 0.
    total = 0.
    label = label.T.reshape(-1,1)
    #print label.shape
    for i in range(BATCH_SIZE*SEQ_LEN):
        maxIdx = np.argmax(pred[i])
        if maxIdx == int(label[i]):
            hit += 1.0
        total += 1.0
    return hit/total

def Loss(label, pred):
    SEQ_LEN = 128
    sum = 0.
    label = label.T.reshape(-1,1)
    for i in range(BATCH_SIZE * SEQ_LEN):
        maxIdx = np.argmax(pred[i])
        if maxIdx != int(label[i]):
            sum += (math.log(float(abs((maxIdx-int(label[i]))))))
    sum = sum/(BATCH_SIZE*SEQ_LEN)
    return sum

class LRCNIter(mx.io.DataIter):
    def __init__(self, dataset, labelset, num, batch_size, seq_len, init_states):
        super(LRCNIter, self).__init__()
        self.batch_size = batch_size
        self.count = num/batch_size
        self.dataset = dataset
        self.labelset = labelset
        self.seq_len = seq_len
        #self.data_shape = data_shape
        
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size,seq_len,9))]+init_states
        self.provide_label = [('label',(batch_size,seq_len,1))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                idx = k * batch_size + i 
                data.append(self.dataset[idx])
                label.append(self.labelset[idx])

            data_all = [mx.nd.array(data)]+self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data']+init_state_names
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':
    num_hidden = 32
    num_lstm_layer = 2
    batch_size = BATCH_SIZE

    num_epoch = 1000
    learning_rate = 0.0025
    momentum = 0.0015
    num_label = 6

    contexts = [mx.context.gpu(0)]

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, num_hidden, num_label)

    init_c = [('l%d_init_c'%l,(batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l,(batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h


    X_train_signals_paths = [DATA_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES]
    X_test_signals_paths = [DATA_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES]
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)
    
    y_train_path = DATA_PATH + TRAIN + "y_train.txt"
    y_test_path = DATA_PATH + TEST + "y_test.txt"
    y_train_1 = load_y(y_train_path)
    y_test_1 = load_y(y_test_path)

    y_train = []
    for i in range(len(y_train_1)):
        tmp = []
        for j in range(len(X_train[0])):
            tmp.append(y_train_1[i])
        y_train.append(tmp)
    y_train = np.array(y_train)

    y_test = []
    for i in range(len(y_test_1)):
        tmp = []
        for j in range(len(X_train[0])):
            tmp.append(y_test_1[i])
        y_test.append(tmp)
    y_test = np.array(y_test)

    #print y_train.shape,y_test.shape

    train_data_count = len(X_train)
    test_data_count = len(X_test)
    n_steps = len(X_train[0])
    n_input = len(X_train[0][0])

    #print n_steps, n_input

    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test = X_test.tolist()
    y_test = y_test.tolist()

    data_train = LRCNIter(X_train, y_train, train_data_count, batch_size, n_steps, init_states)
    data_test = LRCNIter(X_test, y_test, test_data_count, batch_size, n_steps, init_states)

    #print data_train.provide_data, data_train.provide_label

    symbol = sym_gen(n_steps)

    model = mx.model.FeedForward(ctx=contexts, 
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in",magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'begin fit'
    debug_metrics = [mx.metric.np(Accuracy), mx.metric.np(Loss)]
    debug_metric = mx.metric.create(debug_metrics)
    
    model.fit(X=data_train, eval_data=data_test, eval_metric=debug_metric,) 
              #batch_end_callback = batch_end_callback,
              #epoch_end_callback = checkpoint)
