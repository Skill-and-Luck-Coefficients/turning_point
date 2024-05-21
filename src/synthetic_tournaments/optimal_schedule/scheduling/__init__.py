from . import good_vs_bad_first, good_vs_bad_last

KEY_TO_SCHEDULING_FUNCTION = {
    "tp_minimizer": good_vs_bad_first.create_double_rr,
    "tp_minimizer_reversed": good_vs_bad_first.create_reversed_double_rr,
    "tp_maximizer": good_vs_bad_last.create_double_rr,
    "tp_maximizer_reversed": good_vs_bad_last.create_reversed_double_rr,
    "tp_minimizer_random": good_vs_bad_first.create_random_double_rr,
    "tp_minimizer_random_reversed": good_vs_bad_first.create_random_reversed_double_rr,
    "tp_maximizer_random": good_vs_bad_last.create_random_double_rr,
    "tp_maximizer_random_reversed": good_vs_bad_last.create_random_reversed_double_rr,
}

__all__ = ["good_vs_bad_first", "good_vs_bad_last", "KEY_TO_SCHEDULING_FUNCTION"]
