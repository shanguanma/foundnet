#!/usr/bin/env python3

import subprocess
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union
import kaldialign
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import k2
import torch
import logging
from pathlib import Path
Pathlike = Union[str, Path]
## this function is from icefall
@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a
        # cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz
        name = subprocess.check_output("hostname -f", shell=True, text=True)
        if name.strip().endswith(".clsp.jhu.edu"):
            import plz
            from distributed import Client

            with plz.setup_cluster() as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except Exception:
        pass
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")



def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
    sclite_mode: bool = False,
) -> float:
    """Write statistics based on predicted results and reference transcripts.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR, sclite_mode=sclite_mode)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    return float(tot_err_rate)

def write_error_stats_with_timestamps(
    f: TextIO,
    test_set_name: str,
    results: List[
        Tuple[
            str,
            List[str],
            List[str],
            List[Union[float, Tuple[float, float]]],
            List[Union[float, Tuple[float, float]]],
        ]
    ],
    enable_log: bool = True,
    with_end_time: bool = False,
) -> Tuple[float, Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]]:
    """Write statistics based on predicted results and reference transcripts
    as well as their timestamps.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
      with_end_time:
        Whether use end timestamps.

    Returns:
      Return total word error rate and mean delay.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    # Compute mean alignment delay on the correct words
    all_delay = []
    for cut_id, ref, hyp, time_ref, time_hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        has_time = len(time_ref) > 0 and len(time_hyp) > 0
        if has_time:
            # pointer to timestamp_hyp
            p_hyp = 0
            # pointer to timestamp_ref
            p_ref = 0
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
                if has_time:
                    p_hyp += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
                if has_time:
                    p_ref += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
                if has_time:
                    p_hyp += 1
                    p_ref += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
                if has_time:
                    if with_end_time:
                        all_delay.append(
                            (
                                time_hyp[p_hyp][0] - time_ref[p_ref][0],
                                time_hyp[p_hyp][1] - time_ref[p_ref][1],
                            )
                        )
                    else:
                        all_delay.append(time_hyp[p_hyp] - time_ref[p_ref])
                    p_hyp += 1
                    p_ref += 1
        if has_time:
            assert p_hyp == len(hyp), (p_hyp, len(hyp))
            assert p_ref == len(ref), (p_ref, len(ref))

    ref_len = sum([len(r) for _, r, _, _, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = float("%.2f" % (100.0 * tot_errs / ref_len))

    if with_end_time:
        mean_delay = (float("inf"), float("inf"))
        var_delay = (float("inf"), float("inf"))
    else:
        mean_delay = float("inf")
        var_delay = float("inf")
    num_delay = len(all_delay)
    if num_delay > 0:
        if with_end_time:
            all_delay_start = [i[0] for i in all_delay]
            mean_delay_start = sum(all_delay_start) / num_delay
            var_delay_start = (
                sum([(i - mean_delay_start) ** 2 for i in all_delay_start]) / num_delay
            )

            all_delay_end = [i[1] for i in all_delay]
            mean_delay_end = sum(all_delay_end) / num_delay
            var_delay_end = (
                sum([(i - mean_delay_end) ** 2 for i in all_delay_end]) / num_delay
            )

            mean_delay = (
                float("%.3f" % mean_delay_start),
                float("%.3f" % mean_delay_end),
            )
            var_delay = (float("%.3f" % var_delay_start), float("%.3f" % var_delay_end))
        else:
            mean_delay = sum(all_delay) / num_delay
            var_delay = sum([(i - mean_delay) ** 2 for i in all_delay]) / num_delay
            mean_delay = float("%.3f" % mean_delay)
            var_delay = float("%.3f" % var_delay)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )
        logging.info(
            f"[{test_set_name}] %symbol-delay mean (s): "
            f"{mean_delay}, variance: {var_delay} "  # noqa
            f"computed on {num_delay} correct words"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp, _, _ in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    return float(tot_err_rate), float(mean_delay), float(var_delay)

class MetricsTracker(defaultdict):
    def __init__(self):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super(MetricsTracker, self).__init__(int)

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            norm_value = (
                float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            )
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)



def store_transcripts(
    filename: Pathlike, texts: Iterable[Tuple[str, str, str]]
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
        If it is a multi-talker ASR system, the ref and hyp may also be lists of
        strings.
    Returns:
      Return None.
    """
    with open(filename, "w") as f:
        for cut_id, ref, hyp in texts:
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)
def concat(ragged: k2.RaggedTensor, value: int, direction: str) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      value:
        The value to prepend or append.
      direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
      Return a new ragged tensor, whose sublists either start with
      or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans

def add_sos(ragged: k2.RaggedTensor, sos_id: int) -> k2.RaggedTensor:
    """Add SOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      sos_id:
        The ID of the SOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist starts with SOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_sos(a, sos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, sos_id, direction="left")


def add_eos(ragged: k2.RaggedTensor, eos_id: int) -> k2.RaggedTensor:
    """Add EOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      eos_id:
        The ID of the EOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist ends with EOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_eos(a, eos_id=0)
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    return concat(ragged, eos_id, direction="right")


