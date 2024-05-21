from pathlib import Path
from typing import Callable, Iterable, Protocol, TypeVar

import pandas as pd

from logs import turning_logger

T = TypeVar("T")


def parse_value_or_iterable(parameter: T | Iterable[T]) -> list[T]:
    """
    ----
    Parameters:
        parameter: T | Iterable[T]

        Remark: strings are not treated as an iterable.

    ----
    Returns:
        list[T]
            Parameter converted to list: `parameter` or [`parameter`].
    """
    if not isinstance(parameter, Iterable):
        return [parameter]

    if isinstance(parameter, str):
        return [parameter]

    return list(parameter)


FnInput, FnOutput = TypeVar("FnInput"), TypeVar("FnOutput")
Fn = Callable[[Path, FnInput], FnOutput]

DecoratedFnOutput = dict[str, FnOutput]
DecoratedFn = Callable[[str | Iterable[str], Path, FnInput], DecoratedFnOutput]


def run_for_all_filenames(
    fn: Fn,
    filenames: str | list[str],
    read_directory: Path,
    *fn_args,
    **fn_kwargs,
) -> DecoratedFn:
    """
    Run a function (`fn`) for different filenames.

    ----
    Parameters:
        fn: Callable[
            [
                Path,      #: full filepath
                *args,     #: `fn` args
                **kwargs,  #: `fn` kwargs
            ],
            FnOutput
        ]
            File path is the only required parameter (always the first one).

        filenames: str | list[str]
            Desired filenames (only the stem, no suffix)

        read_directory: Path
            Where filenames should be read from.
    ----
    Returns:
        dict[str, FnOutput]
            Mapping of each filename to its respective result
    """
    filename_to_result = {}

    for filename in parse_value_or_iterable(filenames):

        filepath = read_directory / f"{filename}.csv"

        if not filepath.exists():
            turning_logger.warning(f"No file: {filepath}")
            continue

        filename_to_result[filename] = fn(filepath, *fn_args, **fn_kwargs)

    return filename_to_result


class DfContainer(Protocol):
    df: pd.DataFrame


def save_filename_to_df(
    filename_to_df_container: dict[str, DfContainer],
    save_dir: Path,
) -> None:
    """
    Saves all dataframes.

    ---
    Parameters:
        fiename_to_df_container: dict[
            str,         #: filename (only the stem, no suffix necessary)
            DfContainer  #: dataframe container (`.df` must be the dataframe)
        ]

        save_dir: Path
            Where to save the dataframe to.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for filename, container in filename_to_df_container.items():
        container.df.to_csv(save_dir / f"{filename}.csv")
