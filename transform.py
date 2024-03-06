from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, TypeVarTuple, Callable as Fn, ParamSpec
import functools

P = ParamSpec("P")
Ts = TypeVarTuple("Ts")
Rs = TypeVarTuple("Rs")
T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class Data(Generic[*Ts]):
    """Monad for carrying tuples of data through a pipeline."""

    data: tuple[*Ts]

    @property
    def value(self: Data[T]) -> T:
        return self.data[0]

    def and_then(self, f: Fn[[*Ts], Data[*Rs]]) -> Data[*Rs]:
        return f(*self.data)


def data(*x: *Ts) -> Data[*Ts]:
    return Data(x)


@dataclass(frozen=True)
class transform(Generic[P, R]):
    """Monad of functions with function composition."""

    fn: Fn[P, R]

    def and_then(
        self: transform[P, Data[*Ts]],
        f: Fn[[*Ts], Data[*Rs]],
    ) -> transform[P, Data[*Rs]]:
        def wrap(*args: P.args, **kwargs: P.kwargs) -> Data[*Rs]:
            x = self.fn(*args, **kwargs).data
            return f(*x)

        return transform(wrap)

    def and_then_lift(
        self: transform[P, Data[*Ts]],
        f: Fn[[*Ts], tuple[*Rs]],
    ) -> transform[P, Data[*Rs]]:
        return self.and_then(lift_func(f))

    def __call__(
        self: transform[P, Data[*Rs]], *args: P.args, **kwargs: P.kwargs
    ) -> Data[*Rs]:
        return self.fn(*args, **kwargs)


z = data("foo")
x = data(1, 2.0, "3")
y = x.and_then(lambda x, y, z: Data((2 + x, y / 4, z + "foo")))
print(x, y)


@transform
def foo(x: int, y: float, z: str) -> Data[int, float, str]:
    return data(2 + x, y / 4, z + "foo")


def bar(x: int, y: float, z: str) -> Data[int, float, str]:
    return data(2 + x, y / 4, z + "bar")


baz = foo.and_then(bar).and_then(bar).and_then(lambda x, y, z: data(z))

print(baz(1, 2.0, "3"))
print(x.and_then(baz))


def lift_func(f: Fn[P, tuple[*Rs]]) -> transform[P, Data[*Rs]]:
    @functools.wraps(f)
    def wrap(*args: P.args, **kwargs: P.kwargs) -> Data[*Rs]:
        return data(*f(*args, **kwargs))

    return transform(wrap)


xx = data(x.and_then(lambda x, y, z: data(y))), x.and_then(
    lambda x, y, z: data(z),
)
print(xx)
