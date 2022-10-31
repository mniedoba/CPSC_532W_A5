# Standard imports
import torch
from enum import Enum
from time import time
from uuid import uuid4
from tqdm import trange

# Project imports
from primitives import primitives
from utils import log_sample

# Parameters
run_name = 'start'

class ExpressionType(Enum):

    SYMBOL = 0
    CONSTANT = 1
    IF_BLOCK = 2
    EXPR_LIST = 3
    SAMPLE = 4
    OBSERVE = 5
    FUNCTION = 6

    @classmethod
    def parse_type(cls, expr):

        if isinstance(expr, str) and expr[0] != "\"" and expr[0] != "\'":
            return ExpressionType.SYMBOL
        elif not isinstance(expr, list):
            return ExpressionType.CONSTANT
        elif expr[0] == 'sample':
            return ExpressionType.SAMPLE
        elif expr[0] == 'observe':
            return ExpressionType.OBSERVE
        elif expr[0] == 'fn':
            return ExpressionType.FUNCTION
        elif expr[0] == 'if':
            return ExpressionType.IF_BLOCK
        else:
            return ExpressionType.EXPR_LIST # I don't know if this is actually used.



class Env(dict):
    'An environment: a dict of {var: val} pairs, with an outer environment'
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        'Get var from the innermost env.'
        if var in self:
            result = self[var]
        elif var[:4] == 'addr':
            return var + "_" + str(uuid4())
        else:
            if self.outer is None:
                raise ValueError(f'Outer limit of environment reached, {var} not found.')
            else:
                result = self.outer.find(var)
        return result


class Procedure(object):
    'A user-defined HOPPL procedure'
    def __init__(self, params:list, body:list, sig:dict, env:Env):
        self.params, self.body, self.sig, self.env = params, body, sig, env
    def __call__(self, *args):
        return eval(self.body, self.sig, Env(self.params, args, self.env))


def standard_env():
    'An environment with some standard procedures'
    env = Env()
    env.update(primitives)
    return env


def eval(e, sig:dict, env:Env, verbose=False):
    '''
    The eval routine
    @params
        e: expression
        sig: side-effects
        env: environment
    '''
    expr_type = ExpressionType.parse_type(e)
    match expr_type:
        case ExpressionType.SYMBOL:
            return env.find(e)
        case ExpressionType.CONSTANT:
            if isinstance(e, str):
                return e
            return torch.tensor(e, dtype=torch.float)
        case ExpressionType.IF_BLOCK:
            _, pred, cons, ante = e
            if eval(pred, sig, env):
                return eval(cons, sig, env)
            else:
                return eval(ante, sig, env)
        case ExpressionType.EXPR_LIST:
            evaluated = []
            for sub_expr in e:
                evaluated_sub_expr = eval(sub_expr, sig, env)
                evaluated.append(evaluated_sub_expr)
            proc = evaluated[0]
            return proc(*evaluated[1:])
        case ExpressionType.SAMPLE:
            _, addr_expr, dist_expr = e
            new_alpha = eval(addr_expr, sig, env)
            env = Env(['alpha'], [new_alpha], outer=env)
            dist = eval(dist_expr, sig, env)
            return dist.sample()
        case ExpressionType.OBSERVE:
            pass
        case ExpressionType.FUNCTION:
            _, params, body = e
            return Procedure(params, body, sig, env)
    return


def evaluate(ast:dict, verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    sig = {}; env = standard_env()
    exp = eval(ast, sig, env, verbose)(run_name) # NOTE: Must run as function with *any* argument
    return exp


def get_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a FOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in trange(num_samples, leave=False):
        sample = evaluate(ast, verbose)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and time() > max_time: break
    return samples
