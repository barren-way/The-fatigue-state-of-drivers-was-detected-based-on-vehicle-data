import tensorflow as tf
import random
import numpy as np
import pandas as pd

# 这个类用于产生序列样本
class ToySequenceData(object):

    def __init__(self,num_set,batch_size,classify):
        #生成使用的列表
        self.data = []
        self.labels = []
        self.seqlen = []

        #从文件导入数据
        df = pd.read_csv('yaw_lable1.csv', encoding="unicode_escape")  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
        df.columns = ["Col1", "Col2"]
        X = df[["Col1", "Col2"]]
        self.tensor = np.array(X)

        df2 = pd.read_csv('yaw_lable2.csv', encoding="unicode_escape")
        df2.columns = ["Col1", "Col2"]
        Y = df2[["Col1", "Col2"]]
        self.tensor2 = np.array(Y)
        print(self.tensor2.shape)

        seq_len=num_set
        #每组的数量
        n_samples = (self.tensor.shape[0] - self.tensor.shape[0] % seq_len-1000) / seq_len
        n_samples=int(n_samples-n_samples%batch_size)
        #数据1000个一组，分成几组
        choose=-1
        #数据导入列表中
        for i in range(n_samples):
            #清醒，疲劳数据依次导入
            if i%batch_size==0:
                choose*=-1
            if choose==1:
                #导入清醒数据
                self.seqlen.append(1000.0)
                num = i * num_set
                s = [self.tensor[j, 0] for j in range(num, num + 1000)]
                self.data.append(s)
                label = self.tensor[num + 1000, 1]
                #根据标签值进行分类，并将其导入
                if label < classify:
                    self.labels.append([1.,0.])
                else:
                    self.labels.append([0.,  1.])
                self.seqlen.append(1000.0)
            elif choose==-1:
                #导入疲劳数据
                s = [self.tensor2[j, 0] for j in range(num, num + 1000)]
                self.data.append(s)
                label = self.tensor2[num + 1000, 1]
                if label <classify:
                    self.labels.append([1.,0.])
                # elif label < 70:
                #     self.labels.append([0., 1., 0.])
                else:
                    self.labels.append([0., 1.])

        self.batch_id = 0

    def next(self, batch_size):
        """
        生成batch_size的样本。
        如果使用完了所有样本，会重新从头开始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels,batch_seqlen

#产生测试样本
class test_ToySequenceData(object):

    def __init__(self,num_set,classify):
        self.data = []
        self.labels = []
        self.seqlen = []


        #从文件导入数据
        df = pd.read_csv('yaw_test.csv', encoding="unicode_escape")  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
        df.columns = ["Col1", "Col2"]
        X = df[["Col1", "Col2"]]
        self.tensor = np.array(X)
        # print(self.tensor)

        seq_len = num_set
        # 每组的数量
        n_samples = (self.tensor.shape[0] - self.tensor.shape[0] % seq_len) / seq_len
        n_samples = (self.tensor.shape[0] - self.tensor.shape[0] % seq_len - 1000) / seq_len
        # n_samples=self.tensor.shape[0]-1002
        n_samples = int(n_samples)
        # 数据1000个一组，分成几组

        for i in range(n_samples):

            self.seqlen.append(1000.0)
            num = i * num_set
            s = [self.tensor[j, 0] for j in range(num, num + 1000)]
            self.data.append(s)
            label = self.tensor[num + 1000, 1]
            if label < classify:
                self.labels.append([1., 0.])
            # elif label<70:
            #     self.labels.append([0.,1.,  0.])

            else:
                self.labels.append([0., 1.])

        self.batch_id = 0




    def next(self, batch_size):
        """
        生成batch_size的样本。
        如果使用完了所有样本，会重新从头开始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels,batch_seqlen


def dynamicRNN(x, seqlen, weights, biases):


    # 输入x的形状： (batch_size, max_seq_len, n_input)
    # 输入seqlen的形状：(batch_size, )


    # 网络定义时的参数
    seq_max_len = 1000  # 序列长度
    n_hidden = 16 # 隐层的size


    n_neurons = n_hidden
    n_layers = 3#神经网络层数
    # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）

    lstm_cell = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons,forget_bias=1.0)
              for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(lstm_cell)

    # 使用tf.nn.dynamic_rnn展开时间维度
    # 此外sequence_length=seqlen也很重要，它告诉TensorFlow每一个序列应该运行多少步
    x = tf.expand_dims(x, axis=2)
    # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
    #                             sequence_length=seqlen)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, x ,dtype=tf.float32, sequence_length=seqlen)

    # outputs的形状为(batch_size, max_seq_len, n_hidden)
    # 如果有疑问可以参考上一章内容

    # 我们希望的是取出与序列长度相对应的输出。如一个序列长度为10，我们就应该取出第10个输出
    # 但是TensorFlow不支持直接对outputs进行索引，因此我们用下面的方法来做：

    batch_size = tf.shape(outputs)[0]
    # 得到每一个序列真正的index
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    #range创建数字序列(递增数列)变量

    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    # 给最后的输出
    return tf.matmul(outputs, weights['out']) + biases['out']
#将矩阵a乘以矩阵b，生成a * b。

def main():
    result=[]
    # 运行的参数
    learning_rate = 0.0001
    training_iters = 20000
    batch_size = 30
    display_step = 10

    # 网络定义时的参数
    seq_max_len = 1000  # 序列长度
    n_hidden = 16 # 隐层的size
    n_classes = 2 # 类别数
    classify=78#分类疲劳值

    trainset = ToySequenceData(100,batch_size,classify)
    #trainset = test_ToySequenceData(100, classify)


    testset = test_ToySequenceData(2010,classify)

    # x为输入，y为输出
    # None的位置实际为batch_size
    x = tf.placeholder("float", [None, seq_max_len])

    #数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
    y = tf.placeholder("float", [None, n_classes])
    # 这个placeholder存储了输入的x中，每个序列的实际长度
    seqlen = tf.placeholder(tf.int32, [None])

    # weights和bias在输出时会用到
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = dynamicRNN(x, seqlen, weights, biases)


    # 因为pred是logits，因此用tf.nn.softmax_cross_entropy_with_logits来定义损失
    #logits:未归一化的概率
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    #使用随机梯度下降算法，使参数沿着 梯度的反方向，即总损失减小的方向移动，实现更新参数

    # 分类准确率
    result_num=tf.argmax(pred, 1)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #判断两者是否相等
    #argmax：返回每行或者每列最大值的索引
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # 返回所有元素的平均值
    #cast数据类型转换

    # 初始化
    init = tf.global_variables_initializer()

    # 训练
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * batch_size < training_iters:
            batch_x, batch_y,batch_seqlen= trainset.next(batch_size)
            # 每run一次就会更新一次参数
            #batch_x = np.reshape(batch_x, (-1, 1000))
            #batch_y = np.reshape(batch_y, (-1, 1))
            #print(batch_y)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            result.append(format(loss))
            if step % display_step == 0:
                # 在这个batch内计算准确度
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
                # 在这个batch内计算损失
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})

                print(batch_y)
                #result.append(format(loss))


                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        np.savetxt('steel_angle_result7.csv', result, delimiter=',', fmt=' %s')

        # 测试集上计算一次准确度
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        result_2=sess.run(result_num, feed_dict={x: test_data, y: test_label,
                                            seqlen: test_seqlen})

        #print(len(result_2[1]))
        result=np.zeros(len(result_2))



        print("Testing Accuracy:", \
               sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                             seqlen: test_seqlen}))
        #输出测试集各数据的分类结果
        for i in range(len(result_2)):
            # if result_2[i]:
            #     result[i]=1
            # else:
            #     result[i] = 0
            result[i]=result_2[i]
        np.savetxt('yaw_result_test1.csv', result, delimiter=',', fmt=' %s')
        #np.savetxt('steel_angle_result2.csv', result_2, delimiter=',', fmt=' %s')



if __name__ == '__main__':
    main()