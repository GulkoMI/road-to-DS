from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, 
                 dimension: int, 
                 lambda_: float = 1e-3, 
                 loss_function: LossFunction = LossFunction.MSE,
                 huber_delta: float = 1):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        :huber_delta: delta for Huber Loss calculation
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.huber_delta: float = huber_delta

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss_mse(self, x: np.ndarray, diff: np.ndarray):
        """
        Calculate MSE loss for x and y with our weights
        :param x: features array
        :param diff: difference between target and prediction
        :return: loss: float
        """
        return (1 / x.shape[0]) * diff.T @ diff
    
    def calc_loss_logcosh(self, diff: np.ndarray):
        """
        Calculate LogCosh loss for x and y with our weights
        :param diff: difference between target and prediction
        :return: loss: float
        """
        return np.mean(np.log(np.cosh(diff)))  
    
    def calc_loss_mae(self, diff: np.ndarray):
        """
        Calculate MAE loss for x and y with our weights
        :param diff: difference between target and prediction
        :return: loss: float
        """
        return np.mean(np.abs(diff))

    def calc_loss_huber(self, x: np.ndarray, diff: np.ndarray):
        """
        Calculate Huber loss for x and y with our weights
        :param x: features array
        :param diff: difference between target and prediction
        :return: loss: float
        """
        delta = self.huber_delta
        abs_diff = np.abs(diff)
        checker = abs_diff <= delta
        first_part_loss = 0.5 * (diff[checker] ** 2)
        second_part_loss  = delta * (abs_diff[~checker] - 0.5 * delta)
        l = x.shape[0]
        return (np.sum(first_part_loss) + np.sum(second_part_loss)) / l

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        y_pred = self.predict(x)
        diff = y - y_pred

        if self.loss_function is LossFunction.MSE:
            loss = self.calc_loss_mse(x, diff)
            return loss
        
        elif self.loss_function is LossFunction.LogCosh:
            loss =  self.calc_loss_logcosh(diff)
            return loss
        
        elif self.loss_function is LossFunction.MAE:
            loss = self.calc_loss_mae(diff)
            return loss

        elif self.loss_function is LossFunction.Huber:
            loss = self.calc_loss_huber(x, diff)
            return loss
        
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        y_pred = x @ self.w
        return y_pred


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        etta = self.lr()
        w_new = self.w - etta * gradient
        weight_diff = w_new - self.w
        self.w = w_new
        return weight_diff

    # В идеале тут бы добавить calc_gradient_{Loss}
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = self.predict(x)
        #Меняем (y - y_pred) ---> (y_pred - y), чтобы избавиться от минуса в градиентах
        diff = y_pred - y
        l = x.shape[0]
        if self.loss_function is LossFunction.MSE:
            return (2 / l) * (x.T @ diff)
        
        elif self.loss_function is LossFunction.LogCosh:
            return (1 / l) * (x.T @ np.tanh(diff))
        
        elif self.loss_function is LossFunction.MAE:
            return (1 / l) * (x.T @ np.sign(diff))

        elif self.loss_function is LossFunction.Huber:
            delta = self.huber_delta
            abs_diff = np.abs(diff)
            checker = abs_diff < delta
            pre_gradient = np.empty_like(diff)
            pre_gradient[checker] = diff[checker]
            pre_gradient[~checker] = delta * np.sign(diff[~checker])
            return (1 / l) * (x.T @ pre_gradient)

        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")        
        


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, batch_size: int = 50, *args, **kwargs):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        idx = np.random.choice(x.shape[0], size=self.batch_size, replace=False)
        x_new = x[idx]
        y_new = y[idx]
        return super().calc_gradient(x_new, y_new)


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros_like(self.w)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        etta = self.lr()
        h_new = self.alpha * self.h + etta * gradient
        w_new = self.w - h_new
        w_diff = w_new - self.w
        self.w = w_new
        self.h = h_new
        return w_diff


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.999, eps: float = 1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps: float = eps

        self.m: np.ndarray = np.zeros_like(self.w)
        self.v: np.ndarray = np.zeros_like(self.w)

        self.beta_1: float = beta_1
        self.beta_2: float = beta_2

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        
        m_new = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        v_new = self.beta_2 * self.v + (1 - self.beta_2) * (gradient**2)

        self.iteration +=1
        m_hat = m_new / (1 - self.beta_1**self.iteration)
        v_hat = v_new / (1 - self.beta_2**self.iteration)

        etta = self.lr()
        w_new = self.w - (etta / (np.sqrt(v_hat) + self.eps)) * m_hat
        w_diff = w_new - self.w

        self.w = w_new
        self.m = m_new
        self.v = v_new

        return w_diff
    
class AdaMax(VanillaGradientDescent):

    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.999, eps: float = 1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps: float = eps

        self.m: np.ndarray = np.zeros_like(self.w)
        self.v: np.ndarray = np.zeros_like(self.w)

        self.beta_1: float = beta_1
        self.beta_2: float = beta_2

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        m_new = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        v_new = np.maximum(self.beta_2 * self.v, np.abs(gradient))

        self.iteration += 1
        m_hat = m_new / (1 - self.beta_1**self.iteration)
        etta = self.lr()
        w_new = self.w - (etta / (v_new + self.eps)) * m_hat
        w_diff = w_new - self.w

        self.w = w_new
        self.m = m_new
        self.v = v_new
        return w_diff

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient = self.w

        return super().calc_gradient(x, y) + self.mu * l2_gradient 


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """

class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """

class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """

class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AdaMaxReg(BaseDescentReg, AdaMax):
    """
    AdaMax  algorithm with regularization class
    """

def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'adamax': AdaMax if not regularized else AdaMaxReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
