import functools
from typing import Any, Callable, Iterable, Iterator, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")


def log(logging_func: Callable[..., Any]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Returns a logging-decorator with given logging severity.

    -----
    Parameters:

        logging_func: Callable[..., Any]

            Logging function with appropriate severity level.

            Examples: logging.info, logging.warning, ...

    -----
    Returns:

        Callable
            Logging-decorator with desired severity.
    """

    def log_wrapper(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            logging_func(f"Calling {func.__name__}")
            logging_func(f"args: {args}")
            logging_func(f"kwargs: {kwargs}")

            value = func(*args, **kwargs)

            logging_func(f"Finished {func.__name__}")

            return value

        return wrapper

    return log_wrapper


def log_iterations(
    iterable: Iterable[T], logging_func: Callable[..., Any], every_n: int = 1
) -> Iterator[T]:
    """
    Iterator to log the end of each iteration in a for loops.

    -----
    Parameters:

        iterable: Iterable[T]
            Iterable to be looped.

        logging_func: Callable[..., Any]

            Logging function with appropriate severity level.

            Examples: logging.info, logging.warning, ...

        every_n: int = 1
            Log every n iterations.

    -----
    Returns:

        Iterator[T]
            Returns the same element that would be returned by iterable.

            The only difference is that the number of the iteration was logged.
    """

    for iteration, value in enumerate(iterable):
        yield value

        if iteration % every_n == 0:
            logging_func(f"Ended {iteration=}")

    logging_func(f"Ended last iteration: {iteration}")
