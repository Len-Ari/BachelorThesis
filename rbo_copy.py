# This is a copy of the rank-biased overlap implementation taken from:
# https://github.com/changyaochen/rbo/blob/master/rbo/rbo.py
# I copied code as i was unable to use conda install and I want my environment to be stable
# The code is edited and dummed down to fit my usage
from tqdm.autonotebook import tqdm

def rbo(L1, L2, k=None, p=1.0, ext=False):
    """
    This the weighted non-conjoint measures, namely, rank-biased overlap.
    Unlike Kendall tau which is correlation based, this is intersection
    based.
    The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the
    RBO paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf

    If p = 1, it returns to the un-bounded set-intersection overlap,
    according to Fagin et al.
    https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf

    The fig. 5 in that RBO paper can be used as test case.
    Note there the choice of p is of great importance, since it
    essentially control the "top-weightness". Simply put, to an extreme,
    a small p value will only consider first few items, whereas a larger p
    value will consider more items. See Eq. (21) for quantitative measure.

    Args:
        k: The depth of evaluation.
        p: Weight of each agreement at depth d:
            p**(d-1). When set to 1.0, there is no weight, the rbo returns
            to average overlap.
        ext: If True, we will extrapolate the rbo, as in Eq. (23).

    Returns:
        The rbo at depth k (or extrapolated beyond).
    """

    if not len(L1) and not len(L2):
        return 1  # both lists are empty

    if not len(L1) or not len(L2):
        return 0  # one list empty, one non-empty

    if k is None:
        k = float("inf")
    k = min(len(L1), len(L2), k)

    # initialize the agreement and average overlap arrays
    A, AO = [0] * k, [0] * k
    if p == 1.0:
        weights = [1.0 for _ in range(k)]
    else:
        assert 0.0 < p < 1.0
        weights = [1.0 * (1 - p) * p**d for d in range(k)]

    # using dict for O(1) look up
    L1_running, L2_running = {str(L1[0]): True}, {str(L2[0]): True}
    A[0] = 1 if L1[0] == L2[0] else 0
    AO[0] = weights[0] if L1[0] == L2[0] else 0

    for d in tqdm(range(1, k), disable=True):

        tmp = 0
        # if the new item from S is in T already
        if str(L1[d]) in L2_running:
            tmp += 1
        # if the new item from T is in S already
        if str(L2[d]) in L1_running:
            tmp += 1
        # if the new items are the same, which also means the previous
        # two cases did not happen
        if str(L1[d]) == str(L2[d]):
            tmp += 1

        # update the agreement array
        A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

        # update the average overlap array
        if p == 1.0:
            AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
        else:  # weighted average
            AO[d] = AO[d - 1] + weights[d] * A[d]

        # add the new item to the running set (dict)
        L1_running[str(L1[d])] = True
        L2_running[str(L2[d])] = True

    if ext and p < 1:
        return min(1, max(0, (AO[-1] + A[-1] * p**k)))

    return min(1, max(0, (AO[-1])))