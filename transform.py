from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, Generic, TypeVarTuple, Callable as Fn, ParamSpec


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

    def switch(self, f: Fn[[*Ts], R]) -> R:
        """
        Leave the Data monad through a call to f.

        This can be convinient when you want to continue with another monad,
        where you would otherwise need to call .data or .value to extract the
        data and then start a separate monad.
        """
        return f(*self.data)


def data(*x: *Ts) -> Data[*Ts]:
    return Data(x)


@dataclass(frozen=True)
class transform(Generic[P, *Rs]):
    """Monad of functions with function composition."""

    fn: Fn[P, Data[*Rs]]

    def and_then(self, f: Fn[[*Rs], Data[*Ts]]) -> transform[P, *Ts]:
        def wrap(*args: P.args, **kwargs: P.kwargs) -> Data[*Ts]:
            return self.fn(*args, **kwargs).and_then(f)

        return transform(wrap)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Data[*Rs]:
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


baz = (
    foo.and_then(bar)
    .and_then(bar)
    .and_then(lambda x, y, z: data(-x, -y, z))
    .and_then(bar)
    .and_then(foo)
)

print(baz(1, 2.0, "3"))
print(x.and_then(baz))
print(x.and_then(lambda x, y, z: data(z)).value)
