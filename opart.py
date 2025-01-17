import numpy as np

def squared_error(segment):
    if len(segment) == 0:  # Handle empty segments
        return 0
    mean = np.mean(segment)
    return np.sum((segment - mean) ** 2)

def opart_k_segments(sequence, num_changepoints):
    n = len(sequence)

    # Initialize dynamic programming table and changepoint traceback
    dp = np.full((num_changepoints + 1, n + 1), np.inf)
    changepoint_trace = np.full((num_changepoints + 1, n + 1), -1)

    # Base case: no changepoints, full sequence cost
    for i in range(n + 1):
        dp[0][i] = squared_error(sequence[:i])

    # Fill DP table
    for k in range(1, num_changepoints + 1):
        for t in range(k, n + 1):  # At least k data points are needed for k changepoints
            for s in range(k - 1, t):  # Previous changepoint
                cost = dp[k - 1][s] + squared_error(sequence[s:t])
                if cost < dp[k][t]:
                    dp[k][t] = cost
                    changepoint_trace[k][t] = s

    # Backtrace to find changepoints
    changepoints = [n]
    t = n
    for k in range(num_changepoints, 0, -1):
        s = changepoint_trace[k][t]
        changepoints.append(s)
        t = s

    changepoints.reverse()

    partition_means_sequence = np.zeros_like(sequence, dtype=float)
    prev_cp = 0
    for cp in changepoints:
        partition = sequence[prev_cp:cp]
        mean_value = np.mean(partition)
        partition_means_sequence[prev_cp:cp] = mean_value
        prev_cp = cp

    return changepoints, partition_means_sequence


# Get cumulative sum vectors
def get_cumsum(sequence):
    y = np.cumsum(sequence)
    z = np.cumsum(np.square(sequence))
    return np.append([0], y), np.append([0], z)


# function to create loss value from 'start' to 'end' given cumulative sum vector y (data) and z (square)
def L(start, end, y, z):
    _y = y[end+1] - y[start]
    _z = z[end+1] - z[start]
    return _z - np.square(_y)/(end-start+1)


# function to get the list of changepoint from vector tau_star
def trace_back(tau_star):
    tau = tau_star[-1]
    chpnt = np.array([len(tau_star)], dtype=int)
    while tau > 0:
        chpnt = np.append(tau, chpnt)
        tau = tau_star[tau-1]
    return np.append(0, chpnt)

def error_count(chpnt, neg_start, neg_end, pos_start, pos_end):
    chpnt = chpnt[:-1]
    fp = 0
    fn = 0
    for ns, ne in zip(neg_start, neg_end):
        count = sum(1 for cp in chpnt if ns <= cp < ne)
        if count >= 1:
            fp += 1
    for ps, pe in zip(pos_start, pos_end):
        count = sum(1 for cp in chpnt if ps <= cp < pe)
        if count >= 2:
            fp += 1
        elif count == 0:
            fn += 1
    return fp, fn

def opart_penalty(sequence, lda):
    sequence = np.append(0, sequence)
    y, z = get_cumsum(sequence)             # cumsum vector
    sequence_length = len(sequence)-1       # length of sequence 

    # Set up
    C = np.zeros(sequence_length + 1)
    C[0] = -lda

    # Get tau_star
    tau_star = np.zeros(sequence_length+1, dtype=int)
    for t in range(1, sequence_length+1):
        V = C[:t] + lda + L(1 + np.arange(t), t, y, z)  # calculate set V
        last_chpnt = np.argmin(V)                       # get optimal tau from set V
        C[t] = V[last_chpnt]                            # update C_i
        tau_star[t] = last_chpnt                        # update tau_star

    set_of_chpnt = trace_back(tau_star[1:])             # get set of changepoints
    set_of_chpnt = set_of_chpnt[1:] - 1
    partition_means_sequence = np.zeros_like(sequence, dtype=float)
    prev_cp = 0
    for cp in set_of_chpnt:
        partition = sequence[prev_cp:cp]
        mean_value = np.mean(partition)
        partition_means_sequence[prev_cp:cp] = mean_value
        prev_cp = cp
    return set_of_chpnt, partition_means_sequence