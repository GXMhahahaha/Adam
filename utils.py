# Name : utils.py
# Time : 2021/8/7 17:18
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

epsilon = 1e-7
np.seterr(all='raise')


def initialize(layers_dims, L):
    """
    初始化
    :param layers_dims: [test_X.shape[0], 20, 3, 1] each layers's node's num, from input to output
    :param L: len(layer_dims)
    :return: para--[W1, b1, W2, b2, WL-1, bL-1]
    """

    para = []
    for i in range(1, L):  # 从第一层到第L-1层
        W = np.random.randn(layers_dims[i], layers_dims[i - 1]) * (2 / np.sqrt(layers_dims[i - 1]))
        b = np.zeros((layers_dims[i], 1))
        para.append(W)
        para.append(b)

    return para  # W1,b1,W2,b2,...,WL-1,bL-1


def linear_forward(A, para_part):
    """
    线性正向传播
    :param A: A_prev
    :param para_part: [Wi,Ai]
    :return: Zi
    """
    # Z1 = W1A0 + b1
    Z = np.dot(para_part[0], A) + para_part[1]
    return Z


def sigmoid(Z):
    """
    sigmoid函数
    :param Z: Zi
    :return: Ai
    """
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    """
    relu函数
    :param Z: Zi
    :return: Ai
    """
    # A = np.where(Z > 0, Z, 0)
    A = np.maximum(0, Z)
    return A


def forward_prop(X, para, L):
    """
    正向传播
    :param X: input
    :param para: [W1,b1,...,WL-1,bL-1]
    :param L: len(layers_dims)
    :return: cache -- [Z1, A1, ..., ZL-1, AL-1]
    """
    cache = []
    A = np.copy(X)
    for i in range(L - 2):  # 0 ~ L-3
        Z = linear_forward(A, para[2 * i:2 * (i + 1)])
        A = relu(Z)
        cache.append(Z)
        cache.append(A)
    Z = linear_forward(A, para[2 * (L - 2):2 * (L - 1)])
    A = sigmoid(Z)
    cache.append(Z)
    cache.append(A)

    return cache  # Z1,A1,...,Z4,A4


def relu_back(dA, Z):
    """
    relu反向传播求导
    :param dA: dAi
    :param Z: Zi
    :return: dZi
    """
    dZ = np.where(Z > 0, dA, np.int64(0))
    return dZ


def backward_prop(cache, para, X, Y, L):
    """
    反向传播
    :param cache: [Z1, A1, ..., ZL-1, AL-1]
    :param para: [W1,b1,...,WL-1,bL-1]
    :param X: input
    :param Y: label
    :param L: len(layers_dims)
    :return: grad -- [dW1,db1,...,dWL-1,dbL-1]
    """
    m = Y.shape[1]
    A = cache[-1]
    dA = 0
    grad_orig = []
    for i in range(L - 1, 0, -1):  # L-1 ~ 1
        if i == L - 1:
            dZ = (1 / m) * (A - Y)

        else:
            dZ = relu_back(dA, cache[2 * (i - 1)])  # Zi
        if 2 * i - 3 > 0:
            dW = np.dot(dZ, cache[2 * i - 3].T)  # Ai-1
        else:
            dW = np.dot(dZ, X.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(para[2 * (i - 1)].T, dZ)
        grad_orig.append(db)
        grad_orig.append(dW)

    grad = list(reversed(grad_orig))
    return grad


def update_para(para, grad, learning_rate):
    """
    更新参数
    :param para: [W1,b1,...,WL-1,bL-1]
    :param grad: [dW1,db1,...,dWL-1,dbL-1]
    :param learning_rate: 学习率
    :return: para -- new[W1,b1,...,WL-1,bL-1]
    """
    length = len(para)
    assert (length == len(grad))
    for i in range(length):
        assert (para[i].shape == grad[i].shape)
        para[i] = para[i] - learning_rate * grad[i]

    return para


def compute_cost(A, Y):
    """
    计算损失
    :param A: 最后一层输出
    :param Y:
    :return: J
    """
    m = Y.shape[1]
    assert (A.shape == Y.shape)
    A = np.where(A == 0, A + epsilon, A)
    A_new = np.where(A == 1, A - epsilon, A)
    J = (-1 / m) * (np.dot(Y, np.log(A_new).T) + np.dot(1 - Y, np.log(1 - A_new).T))
    return np.squeeze(J)


def list_to_vec(para):
    """
    将矩阵列表形式的参数转换成一维向量形式的参数
    :param para: [W1,b1,...,WL-1,bL-1]
    :return: vec -- [w11,w12,...]
             shapes -- [W1.shape, b1.shape,...]
    """
    length = len(para)
    shapes = []
    for i in range(length):
        tmp = np.ravel(para[i])
        shapes.append(para[i].shape)
        if i == 0:
            vec = tmp
        else:
            vec = np.concatenate((vec, tmp), axis=0)

    vec = vec.reshape(vec.shape[0], 1)
    return vec, shapes


def vec_to_lst(vec, shapes):
    """
    将一维向量形式的参数转换为矩阵列表形式的参数
    :param vec: [w11,w12,...]
    :param shapes: [W1.shape, b1.shape,...]
    :return: para -- [W1,b1,...,WL-1,bL-1]
    """
    lst = []
    length = len(shapes)
    last, nxt = 0, 0
    for i in range(length):
        if len(shapes[i]) == 1:
            nxt = last + shapes[i][0]
        else:
            nxt = last + shapes[i][0] * shapes[i][1]
        tmp = vec[last:nxt].reshape(shapes[i])
        lst.append(tmp)
        last = nxt

    return lst


def gradient_check(grad, para, X, Y, L, epsilon=1e-7):
    """
    梯度测试
    :param grad: [dW1,db1,...,dWL-1,dbL-1]
    :param para: [W1,b1,...,WL-1,bL-1]
    :param X: input
    :param Y: label
    :param L: len(layers_dims)
    :param epsilon: 无穷小
    """
    para_vec, shapes = list_to_vec(para)
    total_num = para_vec.shape[0]
    grad_approx = np.zeros((total_num, 1))
    grad_vec, _ = list_to_vec(grad)
    assert (para_vec.shape == grad_vec.shape)

    for i in range(total_num):
        theta_plus_vec = np.copy(para_vec)
        theta_plus_vec[i] += epsilon
        theta_plus_lst = vec_to_lst(theta_plus_vec, shapes)
        cache = forward_prop(X, theta_plus_lst, L)
        J_plus = compute_cost(cache[-1], Y)

        theta_minus_vec = np.copy(para_vec)
        theta_minus_vec[i] -= epsilon
        theta_minus_lst = vec_to_lst(theta_minus_vec, shapes)
        cache = forward_prop(X, theta_minus_lst, L)
        J_minus = compute_cost(cache[-1], Y)

        grad_approx[i] = (J_plus - J_minus) / (2 * epsilon)

    diff = np.linalg.norm(grad_vec - grad_approx) / (np.linalg.norm(grad_vec) + np.linalg.norm(grad_approx))

    if diff < epsilon:
        print("BP success")
        # print(diff)
    else:
        print("BP unsuccess")
        print(diff)
        raise ValueError("BP unsuccess, please check your code")


def l_layer_model(X, Y, layers_dims, num_iterations=3000, learning_rate=0.0075, break_time=1000,
                  print_cost=True, grad_check=False):
    """
    普通l层神经网络模型
    :param X: input
    :param Y: label
    :param layers_dims: [test_X.shape[0], 20, 3, 1] each layers's node's num, from input to output
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param break_time: 中断进行梯度检测的时间
    :param print_cost: 是否输出损失
    :param grad_check: 是否进行梯度检测
    :return: para -- new[W1,b1,...,WL-1,bL-1]
    """
    L = len(layers_dims)  # 从输入层到输出层一共有L层
    para = initialize(layers_dims, L)
    costs = []
    for i in range(num_iterations):
        cache = forward_prop(X, para, L)
        grad = backward_prop(cache, para, X, Y, L)

        # 测试bp算法是否正确
        if i == break_time and grad_check:
            # diff = gradient_check(grad, para, X, Y, L)
            gradient_check(grad, para, X, Y, L)

        # 计算损失函数
        cost = compute_cost(cache[-1], Y)
        costs.append(cost)
        para = update_para(para, grad, learning_rate)
        if i % 1000 == 0 and print_cost:
            print(f'cost after {i} iterations:', cost)

    return para, costs


def random_mini_batches(X, Y, mini_batch_size):
    """
    打乱原数据， 根据mini_batch_size进行分组
    :param X: input
    :param Y: label
    :param mini_batch_size: mini_batch的尺寸
    :return: X_mini_lst -- [X分组1,X分组2,...]
             Y_mini_lst -- [Y分组1，Y分组2,...]
    """
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    num_complete_batches = m // mini_batch_size
    X_mini_lst = []
    Y_mini_lst = []
    for i in range(num_complete_batches):
        X_temp = X_shuffled[:, i * mini_batch_size: (i + 1) * mini_batch_size]
        Y_temp = Y_shuffled[:, i * mini_batch_size: (i + 1) * mini_batch_size]
        assert (X_temp.shape == (X.shape[0], mini_batch_size))
        assert (Y_temp.shape == (1, mini_batch_size))
        X_mini_lst.append(X_temp)
        Y_mini_lst.append(Y_temp)
    mod = m % mini_batch_size
    assert (mini_batch_size * num_complete_batches + mod == m)
    X_temp = X_shuffled[:, -mod::]
    Y_temp = Y_shuffled[:, -mod::]
    assert (X_temp.shape == (X.shape[0], mod))
    assert (Y_temp.shape == (1, mod))
    X_mini_lst.append(X_temp)
    Y_mini_lst.append(Y_temp)
    assert (len(X_mini_lst) == num_complete_batches + 1)
    assert (len(Y_mini_lst) == num_complete_batches + 1)

    return X_mini_lst, Y_mini_lst


def initialize_v_s(para):
    """
    初始化v和s
    :param para:
    :return: v，s
    """
    v = []
    s = []
    l = len(para)
    for i in range(l):
        v.append(np.zeros(para[i].shape))
        s.append(np.zeros(para[i].shape))
    return v, s


def adam(v, s, grad, t, beta1, beta2):
    """
    Adam算法
    :param v:
    :param s:
    :param grad:
    :param t: 每进行一轮参数更新，t就 +1
    :param beta1: 0.9
    :param beta2: 0.99
    :return: v_corrected, s_corrected
    """
    l = len(grad)
    v_corrected = []
    s_corrected = []
    for i in range(l):
        # v和s会跟着一起改变
        v[i] = beta1 * v[i] + (1 - beta1) * grad[i]
        v_corrected.append(v[i] / (1 - beta1 ** t))
        s[i] = beta2 * s[i] + (1 - beta2) * (grad[i] ** 2)
        s_corrected.append(s[i] / (1 - beta2 ** t))

    assert (len(v) == len(v_corrected))
    assert (len(s) == len(s_corrected))
    return v_corrected, s_corrected


def update_para_adam(para, learning_rate, v_corrected, s_corrected, epsilon):
    l = len(para)
    for i in range(l):
        para[i] = para[i] - learning_rate * (v_corrected[i] / (np.sqrt(s_corrected[i]) + epsilon))

    return para


def model_minibatch(X, Y, layers_dims, mini_batch_size, num_iterations=3000, learning_rate=0.0075, break_time=1000,
                    print_cost=True, grad_check=False, Adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8):
    X_mini_lst, Y_mini_lst = random_mini_batches(X, Y, mini_batch_size)
    L = len(layers_dims)
    minibatch_sizes = len(X_mini_lst)
    para = initialize(layers_dims, L)
    costs = []
    if not Adam:
        for i in range(num_iterations):

            for j in range(minibatch_sizes):
                current_X = X_mini_lst[j]
                current_Y = Y_mini_lst[j]
                cache = forward_prop(current_X, para, L)
                grad = backward_prop(cache, para, current_X, current_Y, L)
                if grad_check and i == break_time:
                    gradient_check(grad, para, current_X, current_Y, L)
                para = update_para(para, grad, learning_rate)
                cost = compute_cost(cache[-1], current_Y)
            if i % 1000 == 0 and print_cost:
                print(f'cost after {i + 1} epochs', cost)
            if i % 100 == 0:
                costs.append(cost)
    else:
        v, s = initialize_v_s(para)
        t = 0
        for i in range(num_iterations):

            for j in range(minibatch_sizes):
                t += 1
                current_X = X_mini_lst[j]
                current_Y = Y_mini_lst[j]
                cache = forward_prop(current_X, para, L)
                grad = backward_prop(cache, para, current_X, current_Y, L)
                v_corrected, s_corrected = adam(v, s, grad, t, beta1, beta2)
                para = update_para_adam(para, learning_rate, v_corrected, s_corrected, epsilon)
                cost = compute_cost(cache[-1], current_Y)
            if i % 1000 == 0 and print_cost:
                print(f'cost after {i + 1} epochs', cost)
            if i % 100 == 0:
                costs.append(cost)


    return para, costs


def predict(X_test, para, L):
    """
    预测
    :param X_test: 测试集
    :param para: [W1,b1,...,WL-1,bL-1]
    :param L: len(layers_dims)
    :return: y_hat 预测
    """
    cache = forward_prop(X_test, para, L)
    AL = cache[-1]
    Y_hat = np.where(AL > 0.5, 1, 0)
    return Y_hat


def compute_accuracy(Y_hat, Y):
    """
    计算准确度
    :param Y_hat:
    :param Y: label
    :return: accuracy 百分数
    """
    m = Y.shape[1]
    part1 = np.squeeze(np.dot(Y_hat, Y.T))
    part2 = np.squeeze(np.dot(1 - Y_hat, (1 - Y).T))
    acc = (part1 + part2) / m
    return str(acc * 100) + '%'


def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X, train_Y


def plot_decision_boundary(model, X, Y):
    """
    画出预测分界线
    :param model: predict
    :param X:
    :param Y:
    :return: None
    """
    xx = X[0, :]
    yy = X[1, :]

    xmin = xx.min() - 0.1
    xmax = xx.max() + 0.1
    ymin = yy.min() - 0.1
    ymax = yy.max() + 0.1

    x_range = np.arange(xmin, xmax, 0.01)
    y_range = np.arange(ymin, ymax, 0.01)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    x_flatten = np.ravel(x_mesh)
    y_flatten = np.ravel(y_mesh)
    x_flatten = x_flatten.reshape(1, x_flatten.shape[0])
    y_flatten = y_flatten.reshape(1, y_flatten.shape[0])
    data_points = np.concatenate((x_flatten, y_flatten), axis=0)
    Y_hat = model(data_points)
    Y_hat = Y_hat.reshape(x_mesh.shape)

    plt.contourf(x_mesh, y_mesh, Y_hat, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.title("decision boundary")
    plt.show()
