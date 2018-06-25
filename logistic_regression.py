import numpy as np


class MyLogisticRegression:
    """logistic regression classifier"""
    def __init__(self, lamda=0.1, max_iter=500, eps=0.01):
        """constructor

        Args:
            lamda: (float) regularization coefficient
            max_iter: (int) optional [default=1] maximum training iterations
            eps: (float) optional [default=0.001] gradient stopping criteria
        """
        self.coef = None

        self._eps = eps
        self._eta = None
        self._lamda = lamda
        self._max_iter = max_iter
        self._pos, self._neg = None, None

        self.__log_queue = None
        self.__x,  self.__y  = None, None
        self.__n,  self.__d  = None, None

    def predict(self, x, beta=None):
        """prediction probabilities

        Args:
            x: nXd (ndarray) of input values
            beta: dX1 (ndarray) optional weight coefficients
        Returns:
            nX1 ndarray of class labels
        """
        beta = self.coef if beta is None else beta
        return [self._pos if xi @ beta > 0 else self._neg for xi in x]

    def predict_proba(self, x, beta=None):
        """prediction probabilities

        Args:
            x: nXd (ndarray) of input values
            beta: dX1 (ndarray) optional weight coefficients
        Returns:
            nX1 ndarray of probabilites towards the positive class label
        """
        beta = self.coef if beta is None else beta
        return [np.exp(xi@beta)/(1 + np.exp(xi@beta)) for xi in x]

    def fit(self, x_train, y_train, pos=1, neg=-1, eta=None, queue=None):
        """fit the classifier

        Args:
            x_train: nXd (ndarray) of training samples
            y_train: nX1 (ndarray) of true labels
            pos: (object) positive class label
            neg: (object) optional negative class label
            eta: (float) optional learning rate
            queue: (Queue) optional logging queue
        Returns:
            trained classifier
        """
        self._pos, self._neg = pos, neg

        self.__x,  self.__y  = x_train, y_train
        self.__n,  self.__d  = x_train.shape
        self._eta = self.__calc_t_init() if eta is not None else eta

        self.__log_queue = queue
        self.__fgrad()

        self.__log(dict(
            klass=self._pos,
            iter='END',
            obj=self.__objective(self.coef),
            coef=self.coef)
        )
        return self

    def __backtracking(self, beta, t_eta=0.5, alpha=0.5):
        """backtracking line search

        Args:
            beta: dX1 (ndarray) weight coefficients
            t_eta: (float) optional [default=0.5] 0 < t_eta < 1 learning rate for eta
            alpha: (float) optional [default=0.5] 0 < alpha < 1 tune stopping condition

        Returns:
            float: optimum learning rate
        """
        l, t = self._lamda, 1

        gb = self.__gradient(beta)
        n_gb = np.linalg.norm(gb)

        found_t, i = False, 0
        while not found_t and i < 100:
            if self.__objective(beta - t*gb) < self.__objective(beta) - alpha*t*n_gb**2:
                found_t = True
            elif i == self._max_iter-1:
                break
            else:
                t *= t_eta
                i += 1

        self.__eta = t
        return self.__eta

    def __calc_t_init(self):
        """calculate optimum initial learning rate"""
        x, l, n = self.__x, self._lamda, self.__n

        m = np.max(1/n * np.linalg.eigvals(x.T @ x)) + l
        return 1 / np.float(m)

    def __gradient(self, b):
        """calculate gradient

        Args:
            b: dX1 (ndarray) weight coefficients

        Returns:
            dX1 ndarray gradient
        """
        x, y, l, n = self.__x, self.__y, self._lamda, self.__n

        p = (1 + np.exp(y*(x @ b)))**-1
        return 2*l*b - (x.T @ np.diag(p) @ y)/n

    def __fgrad(self):
        """fast gradient descent"""
        b0 = np.zeros(self.__d)

        theta = np.copy(b0)
        grad = self.__gradient(theta)

        i = 0
        while np.linalg.norm(grad) > self._eps and i < self._max_iter:
            t = self.__backtracking(b0)
            grad = self.__gradient(theta)

            b1 = theta - t*grad
            theta = b1 + (i/(i+3))*(b1-b0)

            b0 = b1
            i += 1

            self.coef = b0
            self.__log(dict(
                klass=self._pos,
                iter=i,
                obj=self.__objective(b0),
                coef=b0)
            )

    def __log(self, args):
        """logging function

        if a Queue has been supplied to the classifier in the 'fit'
        method, log output will be given to it. Otherwise args will
        be printed to the console
        Args:
            args: [object] a list of objects with an implmement __str__ method

        Returns:
            None
        """
        if self.__log_queue is not None:
            self.__log_queue.put(args)
        else:
            print(','.join([str(v) for k, v in args.items() if k != 'coef']))

    def __objective(self, beta):
        """calculate objective value

        Args:
            beta: dX1 (ndarray) weight coefficients

        Returns:
            float
        """
        x, y, n, l = self.__x, self.__y, self.__n, self._lamda

        loss = np.sum([np.log(1 + np.exp(-yi*xi.T@beta)) for xi, yi in zip(x, y)])
        return loss/n + l*np.linalg.norm(beta)**2

    def __repr__(self):
        return f'<MyLogisticRegression(C={self._lamda})>'

    def __str__(self):
        return f'MyLogisticRegression: C={self._lamda}'
