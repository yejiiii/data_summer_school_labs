import pandas as pd
import numpy as np

data_source = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"

names = ["Wife's age", "Wife's education", "Husband's education", "Number of children ever born", "Wife's religion", 
"Wife's now working?", "Husband's occupation", "Standard-of-living index", "Media exposure", "Contraceptive method used"]

dataframe = pd.read_csv(data_source, names=names)

for column in range(0,10):
    if column not in [0,3]:
        dataframe[names[column]] = dataframe[names[column]].astype("category")

y_data = dataframe.pop("Contraceptive method used")

x_data = dataframe
x_data.insert(0, "bias", 1)

x_data = pd.get_dummies(x_data).as_matrix()
y_data = pd.get_dummies(y_data).as_matrix()


from sklearn import preprocessing as p

min_max_scaler=p.MinMaxScaler()
x_data[:, 1:3] = min_max_scaler.fit_transform(x_data[:,1:3])

idx=np.random.randint(y_data.shape[0],size=int(y_data.shape[0]))
training_idx = idx[:int(y_data.shape[0]*0.8)]
test_idx = idx[int(y_data.shape[0]*0.8):]

x_training, x_test = x_data[training_idx], x_data[test_idx]
y_training, y_test = y_data[training_idx], y_data[test_idx]


import tensorflow as tf

W = tf.Variable(tf.zeros([len(x_data[0]), len(y_data[0])]))
                
X = tf.placeholder("float", [None, len(x_data[0])])
Y = tf.placeholder("float", [None, len(y_data[0])])


hypothesis = tf.nn.softmax(tf.matmul(X, W))

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

W_val = []
cost_val = []

a = tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(5000000):
    sess.run(train, feed_dict={X:x_training, Y:y_training})
    if step % 500000 == 0:
        print (step, sess.run(cost, feed_dict={X:x_training, Y:y_training}), sess.run(W))

hypothesis_value=sess.run(hypothesis,feed_dict={X:x_test})
result=[np.argmax(predict) == np.argmax(original_value) for predict, original_value in zip(hypothesis_value,y_test)]

sum(result)/len(y_test)