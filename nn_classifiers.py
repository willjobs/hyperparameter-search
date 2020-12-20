##########################################################
# The purpose of this code is to create the fully connected neural networks and
# CNN used in the experiments comparing grid search, random search, and Bayesian
# hyperparameter optimization. It wraps PyTorch modules with the code to make a
# custom scikit-learn classifier so that the neural networks can be trained via
# the GridSearchCV, RandomizedSearchCV, and BayesSearchCV approaches. The code is
# written to allow its use in an environment with GPUs, or without GPUs.
#
# Note that there are ways this code could be simplified and generalized. This code
# currently uses specific architectures for the NNs (with several hyperparameters
# that control aspects of the architecture and learning), but it could be modified to
# allow for more general NNs and additional styles. Also, the fit() and predict() functions
# for each of the NNs are very similar and could be abstracted away into a single function.
##########################################################

import gc
from functools import wraps
import inspect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin  # so we can make a custom classifier
from sklearn.metrics import make_scorer
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler

CPU = torch.device('cpu')

def initializer(func):
    """
    Adapted from: https://stackoverflow.com/a/1389216
    Automatically assigns the parameters to instance variables.

    Example:
    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, _, _, defaults = inspect.getfullargspec(func)[0:4]

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def get_accuracy(loader_val, model):
    """Calculate the accuracy of a PyTorch model using a validation set. This is called periodically
    during the training phase (frequency determined by the print_every parameter) as a way to
    monitor the training progress. Adapted from UMass CS 682.

    Args:
        loader_val (DataLoader): Loads the validation dataset in increments of `batch_size`
        model (nn.Module): Trained PyTorch model (up to a certain epoch/iteration)

    Returns:
        float: accuracy (# correct / # samples) computed by the model on the loader's dataset
    """
    n_correct = 0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader_val:
            scores = model(x)
            _, preds = scores.max(1)
            n_correct += (preds == y).sum()
            n_samples += preds.size(0)
        return float(n_correct) / n_samples


def _custom_scorer(clf, X_val, y_true_val):
    """Used to calculate accuracy in GridSearchCV, RandomizedSearchCV, and BayesSearchCV.
    This is because the default 'accuracy' scoring method would calculate accuracy across
    the entire validation set, which can cause an out-of-memory issue. This custom scorer
    uses the get_accuracy function defined above to use a DataLoader to calculate the accuracy
    on batches rather than all at once.

    Note: This is meant to be used immediately by the search algorithm (e.g., GridSearchCV) when
    the loader_val object has been instantiated and is in scope. Otherwise this function won't have
    access to that object. This is a necessary but unfortunate compromise, because GridSearchCV (and
    RandomizedSearchCV, and BayesSearchCV) require this function signature for a custom scorer, and
    the alternative approach using make_scorer doesn't work because when the scorer object created
    by make_scorer is called by GridSearchCV, it doesn't see the fitted model object for some reason.

    See https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object
    """
    return get_accuracy(loader_val, clf.model_)


class FCNet(nn.Module):
    """Fully connected neural network with hidden layers of all the same size. Includes
    ReLU or Leaky ReLU nonlinearities, and optional dropout after the last nonlinearity.
    """
    @initializer
    def __init__(self,
                 n_layers=2,
                 in_size=3072,
                 hidden_size=4000,
                 n_classes=10,
                 leaky_relu=False,
                 p_dropout=0.5):
        """Initialize.

        Args:
            n_layers (int, optional): Number of layers (including the output layer). Defaults to 2.
            in_size (int, optional): Product of the width and height dimensions of the image. For
                example, if the images are 32x32, the in_size is 3072. Defaults to 3072.
            hidden_size (int, optional): Size of the hidden layers (same for all hidden layers).
                Defaults to 4000.
            n_classes (int, optional): Number of classes in the dataset (e.g., 10 for CIFAR-10).
                Defaults to 10.
            leaky_relu (bool, optional): Whether to use a ReLU (True) or Leaky ReLU (False).
                Defaults to False.
            p_dropout (float, optional): How much dropout to use after the last nonlinearity, if
                any (0 = no dropout). Defaults to 0.5.
        """
        super(FCNet, self).__init__()
        self.mods = nn.ModuleList()
        self.mods.append(nn.Flatten())
        in_sizes = [in_size, *([hidden_size] * (n_layers - 1))]
        out_sizes = [*([hidden_size] * (n_layers - 1)), n_classes]

        for i in range(n_layers):
            tmp = nn.Linear(in_sizes[i], out_sizes[i])

            self.mods.append(tmp)
            if not leaky_relu:
                self.mods.append(nn.ReLU())
            else:
                self.mods.append(nn.LeakyReLU())

        self.mods.append(nn.Dropout(p=p_dropout, inplace=True))

    def forward(self, x, **kwargs):
        for layer in self.mods:
            x = layer(x)

        return x


class FCNetClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper of FCNet that allows it to be used like a scikit-learn classifier, and thus
    to be used easily with GridSearchCV, RandomizedSearchCV, and BayesSearchCV.

    This classifier uses the FCNet module defined above, cross-entropy loss, and the
    AdamW optimizer (not Adam). It also handles whether calculations are computed on a GPU
    or a CPU.
    """

    @initializer
    def __init__(self,
                 n_layers=2,
                 in_size=3072,
                 hidden_size=4000,
                 n_classes=10,
                 n_epochs=3,
                 batch_size=64,
                 leaky_relu=False,
                 p_dropout=0.5,
                 learning_rate=1e-5,
                 weight_decay=1e-3,
                 loader_val=None,
                 print_every=200,
                 seed=682,
                 device=CPU):
        """Initialize.

        Args:
            n_layers (int, optional): Number of layers (including the output layer). Defaults to 2.
            in_size (int, optional): Product of the width and height dimensions of the image. For
                example, if the images are 32x32, the in_size is 3072.
                Defaults to 3072.
            hidden_size (int, optional): Size of the hidden layers (same for all hidden layers).
                Defaults to 4000.
            n_classes (int, optional): Number of classes in the dataset (e.g., 10 for CIFAR-10).
                Defaults to 10.
            n_epochs (int, optional): Number of epochs to train for. Defaults to 3.
            batch_size (int, optional): Batch size for each iteration. Defaults to 64.
            leaky_relu (bool, optional): Whether to use a ReLU (True) or Leaky ReLU (False).
                Defaults to False.
            p_dropout (float, optional): How much dropout to use after the last nonlinearity, if
                any (0 = no dropout). Defaults to 0.5.
            learning_rate (float, optional): Learning rate for the AdamW optimizer.
                Defaults to 1e-5.
            weight_decay (float, optional): Weight decay for the AdamW optimizer. Defaults to 1e-3.
            loader_val (DataLoader, optional): Loads the validation dataset in increments of
                `batch_size`. Defaults to None.
            print_every (int, optional): After this many iterations, print the accuracy to check in
                on progress. Defaults to 200.
            seed (int, optional): For reproducibility. Defaults to 682.
            device (PyTorch device, optional): Where calculations will be performed. Defaults to CPU.
        """
        pass  # automatically initialize all instance variables via initializer()

    def fit(self, X, y):
        """Train the fully connected neural network (FCNet) using the AdamW optimizer and
        cross-entropy loss. The final trained model is stored in self.model_.

        Args:
            X (numpy array): Training data
            y (numpy array): Training labels
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        if X.dtype == np.float64:
            X.astype(np.float32)

        X = torch.from_numpy(X).to(device=self.device, dtype=dtype)
        y = torch.from_numpy(y).to(device=self.device, dtype=torch.long)

        train_data = TensorDataset(X, y)
        loader_train = DataLoader(train_data,
                                  batch_size=self.batch_size,
                                  sampler=RandomSampler(train_data))

        model = FCNet(n_layers=self.n_layers,
                      in_size=self.in_size,
                      hidden_size=self.hidden_size,
                      n_classes=self.n_classes,
                      leaky_relu=self.leaky_relu,
                      p_dropout=self.p_dropout)
        model.to(device=self.device)

        optimizer = optim.AdamW(model.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        print('\n------------\n    Beginning training loop with params = {}...'.format(self.get_params()))
        for epoch in range(self.n_epochs):
            for i, (X, y) in enumerate(loader_train):
                model.train()
                scores = model(X)
                loss = F.cross_entropy(scores, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % self.print_every == (self.print_every - 1) or i == 0:
                    if self.loader_val is not None:
                        acc = get_accuracy(self.loader_val, model)
                        print('        Epoch {}, iteration {}: loss = {}, accuracy = {}%'.format(epoch+1, i+1, loss.item(), round(acc * 100, 3)))
                    else:
                        print('        Epoch {}, iteration {}: loss = {}'.format(epoch+1, i+1, loss.item()))

            if self.loader_val is not None:
                acc = get_accuracy(self.loader_val, model)
                print('        End of Epoch {}: accuracy = {}%'.format(epoch+1, round(acc * 100, 2)))

        # save the fitted model
        self.model_ = model

    def predict(self, X):
        """Function used to predict class labels using the model trained by `fit`.

        Args:
            X (numpy array): Data for which to predict class labels

        Returns:
            numpy array: Predicted class labels
        """
        try:
            model = self.model_
        except:
            raise RuntimeError('Model has not been fitted')

        if X.dtype == np.float64:
            X.astype(np.float32)

        X = torch.from_numpy(X).to(device=self.device, dtype=dtype)
        scores = model(X)
        _, preds = scores.max(1)
        preds = preds.cpu()
        return preds


class CNN(nn.Module):
    """Convolutional neural network (CNN) with the same number of filters, filter size, and stride
    at every layer. Zero-padding is automatically calculated by trying widths from 0 to 5 until an
    integer next layer output size is achieved. Successive convolutional layers are followed by
    either a ReLU or Leaky ReLU nonlinearity, and the last nonlinearity is followed by optional
    dropout and a fully-connected layer.
    """
    @initializer
    def __init__(self,
                 n_layers=5,
                 img_width=32,
                 n_classes=10,
                 filter_size=3,
                 n_filters=32,
                 stride=1,
                 leaky_relu=False,
                 p_dropout=0.5):
        """Initialize.

        Args:
            n_layers (int, optional): Number of layers (including the output layer). Defaults to 5.
            img_width (int, optional): Length (in pixels) of one size of input image (assuming a
                square input image). Defaults to 32.
            n_classes (int, optional): Number of classes in the dataset (e.g., 10 for CIFAR-10).
                Defaults to 10.
            filter_size (int, optional): Also called "kernel size", this measures one side
                of a square that is convolved over the input. Defaults to 3.
            n_filters (int, optional): Number of filters stacked in a given layer. In this
                implementation the same number of filters will be used in all convolutional layers.
                Defaults to 32.
            stride (int, optional): When convolving the filters over each layer, this specifies
                how many pixels over/down each filter moves at each step. Defaults to 1.
            leaky_relu (bool, optional): Whether to use a ReLU (True) or Leaky ReLU (False).
                Defaults to False.
            p_dropout (float, optional): How much dropout to use after the last nonlinearity, if
                any (0 = no dropout). Defaults to 0.5.
        """
        super(CNN, self).__init__()
        self.mods = nn.ModuleList()

        in_channels = [3, *([n_filters] * (n_layers - 1))]
        out_channels = [n_filters] * n_layers
        output_width = img_width

        for i in range(n_layers):
            if stride == 1:
                padding = 0
            else:
                # determine how much padding is needed; try padding values between 0 and 5 (inclusive)
                for padding in range(6):
                    tmp = (output_width - filter_size + 2*padding) / stride + 1
                    if round(tmp) == tmp and tmp > 0:
                        # we found a padding that results in an integer output
                        break
                    if padding == 4:
                        raise RuntimeError('No valid padding found')
            output_width = (output_width - filter_size + 2*padding) / stride + 1

            tmp = nn.Conv2d(in_channels[i],
                            out_channels[i],
                            kernel_size=filter_size,
                            stride=stride,
                            padding=padding)
            self.mods.append(tmp)

            if not leaky_relu:
                self.mods.append(nn.ReLU())
            else:
                self.mods.append(nn.LeakyReLU())

        self.mods.append(nn.Flatten())
        self.mods.append(nn.Dropout(p=p_dropout))
        self.mods.append(nn.Linear(int(output_width ** 2) * out_channels[-1], n_classes))

    def forward(self, x, **kwargs):
        if len(x.shape) == 2:
            x = x.reshape((-1, 3, self.img_width, self.img_width))

        for layer in self.mods:
            x = layer(x)

        return x


class CNNClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper of CNN that allows it to be used like a scikit-learn classifier, and thus
    to be used easily with GridSearchCV, RandomizedSearchCV, and BayesSearchCV.

    This classifier uses the CNN module defined above, cross-entropy loss, and the
    AdamW optimizer (not Adam). It also handles whether calculations are computed on a GPU
    or a CPU.
    """

    @initializer
    def __init__(self,
                 n_layers=5,
                 img_width=32,
                 n_classes=10,
                 n_epochs=3,
                 batch_size=64,
                 filter_size=3,
                 n_filters=32,
                 stride=1,
                 leaky_relu=False,
                 p_dropout=0.5,
                 learning_rate=1e-5,
                 weight_decay=1e-3,
                 loader_val=None,
                 print_every=200,
                 seed=682,
                 device=CPU):
        """Initialize.

        Args:
            n_layers (int, optional): Number of layers (including the output layer). Defaults to 5.
            img_width (int, optional): Length (in pixels) of one size of input image (assuming a
                square input image). Defaults to 32.
            n_classes (int, optional): Number of classes in the dataset (e.g., 10 for CIFAR-10).
                Defaults to 10.
            n_epochs (int, optional): Number of epochs to train for. Defaults to 3.
            batch_size (int, optional): Batch size for each iteration. Defaults to 64.
            filter_size (int, optional): Also called "kernel size", this measures one side
                of a square that is convolved over the input. Defaults to 3.
            n_filters (int, optional): Number of filters stacked in a given layer. In this
                implementation the same number of filters will be used in all convolutional layers.
                Defaults to 32.
            stride (int, optional): When convolving the filters over each layer, this specifies
                how many pixels over/down each filter moves at each step. Defaults to 1.
            leaky_relu (bool, optional): Whether to use a ReLU (True) or Leaky ReLU (False).
                Defaults to False.
            p_dropout (float, optional): How much dropout to use after the last nonlinearity, if
                any (0 = no dropout). Defaults to 0.5.
            learning_rate (float, optional): Learning rate for the AdamW optimizer.
                Defaults to 1e-5.
            weight_decay (float, optional): Weight decay for the AdamW optimizer. Defaults to 1e-3.
            loader_val (DataLoader, optional): Loads the validation dataset in increments of
                `batch_size`. Defaults to None.
            print_every (int, optional): After this many iterations, print the accuracy to check in
                on progress. Defaults to 200.
            seed (int, optional): For reproducibility. Defaults to 682.
            device (PyTorch device, optional): Where calculations will be performed. Defaults to CPU.
        """
        pass  # automatically initialize all instance variables via initializer()

    def fit(self, X, y):
        """Train the CNN using the AdamW optimizer and cross-entropy loss. The final trained
        model is stored in self.model_.

        Args:
            X (numpy array): Training data
            y (numpy array): Training labels
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        if X.dtype == np.float64:
            X.astype(np.float32)

        X = X.reshape((-1, 3, self.img_width, self.img_width))
        X = torch.from_numpy(X).to(device=self.device, dtype=dtype)
        y = torch.from_numpy(y).to(device=self.device, dtype=torch.long)

        train_data = TensorDataset(X, y)
        loader_train = DataLoader(train_data,
                                  batch_size=self.batch_size,
                                  sampler=RandomSampler(train_data))

        model = CNN(n_layers=self.n_layers,
                    img_width=self.img_width,
                    n_classes=self.n_classes,
                    filter_size=self.filter_size,
                    n_filters=self.n_filters,
                    stride=self.stride,
                    leaky_relu=self.leaky_relu,
                    p_dropout=self.p_dropout)
        model.to(device=self.device)

        optimizer = optim.AdamW(model.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        print('\n------------\n    Beginning training loop with params = {}...'.format(self.get_params()))
        for epoch in range(self.n_epochs):
            for i, (X, y) in enumerate(loader_train):
                model.train()
                scores = model(X)
                loss = F.cross_entropy(scores, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % self.print_every == (self.print_every - 1) or i == 0:
                    if self.loader_val is not None:
                        acc = get_accuracy(self.loader_val, model)
                        print('        Epoch {}, iteration {}: loss = {}, accuracy = {}%'.format(epoch+1, i+1, loss.item(), round(acc * 100, 3)))
                    else:
                        print('        Epoch {}, iteration {}: loss = {}'.format(epoch+1, i+1, loss.item()))


            if self.loader_val is not None:
                acc = get_accuracy(self.loader_val, model)
                print('        End of Epoch {}: accuracy = {}%'.format(epoch+1, round(acc * 100, 2)))

        # save the fitted model
        self.model_ = model

    def predict(self, X):
        """Function used to predict class labels using the model trained by `fit`.

        Args:
            X (numpy array): Data for which to predict class labels

        Returns:
            numpy array: Predicted class labels
        """
        try:
            model = self.model_
        except:
            raise RuntimeError('Model has not been fitted')

        if X.dtype == np.float64:
            X.astype(np.float32)

        X = X.reshape((-1, 3, self.img_width, self.img_width))
        X = torch.from_numpy(X).to(device=self.device, dtype=dtype)
        scores = model(X)
        _, preds = scores.max(1)
        preds = preds.cpu()
        return preds
