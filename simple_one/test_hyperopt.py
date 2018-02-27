

from hyperopt import fmin, tpe, hp, STATUS_OK, rand, Trials
import networkx
import time

def obj(x):

    return {'loss': x**2,
            'status': STATUS_OK,
           }


if __name__ == "__main__":
    print(networkx.__version__)


