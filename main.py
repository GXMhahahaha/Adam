# Name : main.py
# Time : 2021/8/7 17:17
from utils import *
import matplotlib.pyplot as plt

train_X, train_Y = load_dataset()
print(train_X.shape, train_Y.shape)
layers_dims = (train_X.shape[0], 5, 2, 1)
para1, costs1 = l_layer_model(train_X, train_Y, layers_dims, num_iterations=10000, learning_rate=0.0007,
                              grad_check=True)
plot_decision_boundary(lambda x: predict(x, para1, len(layers_dims)), train_X, train_Y)
plt.title("normal model costs")
plt.plot(costs1)
plt.show()

para2, costs2 = model_minibatch(train_X, train_Y, layers_dims, mini_batch_size=64, grad_check=True, break_time=2000,
                                learning_rate=0.0007
                                , num_iterations=10000, Adam=True)

plot_decision_boundary(lambda x: predict(x, para2, len(layers_dims)), train_X, train_Y)
plt.title("Adam costs")
plt.plot(costs2)
plt.show()
