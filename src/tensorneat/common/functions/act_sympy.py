import numpy as np
import sympy as sp

SCALE = 3


class SympySigmoid(sp.Function):
    @classmethod
    def eval(cls, z):
        return 1 / (1 + sp.exp(-z))


class SympyScaledSigmoid(sp.Function):
    @classmethod
    def eval(cls, z):
        return SympySigmoid(z) * SCALE


class SympyTanh(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.tanh(z)


class SympyScaledTanh(sp.Function):
    @classmethod
    def eval(cls, z):
        return SympyTanh(z) * SCALE


class SympySin(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.sin(z)


class SympyRelu(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Max(z, 0)


class SympyLelu(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Piecewise((z, z > 0), (0.005 * z, True))


class SympyIdentity(sp.Function):
    @classmethod
    def eval(cls, z):
        return z


class SympyInv(sp.Function):
    @classmethod
    def eval(cls, z):
        safe = sp.Piecewise((sp.Max(z, 1e-7), z > 0), (sp.Min(z, -1e-7), True))
        return 1 / safe


class SympyLog(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.log(sp.Max(z, 1e-7))


class SympyExp(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.exp(SympyClip(z, -10, 10))


class SympyAbs(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.Abs(z)


class SympyClip(sp.Function):
    @classmethod
    def eval(cls, val, min_val, max_val):
        if val.is_Number and min_val.is_Number and max_val.is_Number:
            return sp.Piecewise(
                (min_val, val < min_val),
                (max_val, val > max_val),
                (val, True),
            )
        return None

    @staticmethod
    def numerical_eval(val, min_val, max_val, backend=np):
        return backend.clip(val, min_val, max_val)

    def _sympystr(self, printer):
        return f"clip({self.args[0]}, {self.args[1]}, {self.args[2]})"

    def _latex(self, printer):
        return rf"\mathrm{{clip}}\left({sp.latex(self.args[0])}, {self.args[1]}, {self.args[2]}\right)"