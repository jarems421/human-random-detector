import random


def generate_true_random(length=50):
    return ''.join(random.choice(['0', '1']) for _ in range(length))


def human_near_alternating(length):
    if length <= 0:
        return ''

    seq = [random.choice(['0', '1'])]
    last = seq[0]

    for _ in range(1, length):
        if random.random() < 0.85:
            next_val = _opposite_bit(last)
        else:
            next_val = last
        seq.append(next_val)
        last = next_val

    return ''.join(seq)


def human_balanced_streak_avoidant(length):
    seq = []
    counts = {'0': 0, '1': 0}

    for _ in range(length):
        if _current_run_length(seq) >= 3:
            next_val = _opposite_bit(seq[-1])
        elif counts['0'] < counts['1']:
            next_val = '0' if random.random() < 0.75 else '1'
        elif counts['1'] < counts['0']:
            next_val = '1' if random.random() < 0.75 else '0'
        else:
            next_val = random.choice(['0', '1'])

        if len(seq) >= 2 and next_val == seq[-1] == seq[-2] and random.random() < 0.8:
            next_val = _opposite_bit(next_val)
        seq.append(next_val)
        counts[next_val] += 1

    return ''.join(seq)


def human_chunk_pattern(length):
    motifs = ['001', '011', '010', '101']
    seq = []

    while len(seq) < length:
        seq.extend(random.choice(motifs))

    seq = seq[:length]

    if seq and random.random() < 0.25:
        index = random.randrange(len(seq))
        seq[index] = _opposite_bit(seq[index])

    return ''.join(seq)


def human_soft_biased(length):
    bias = random.uniform(0.55, 0.65)
    return ''.join('1' if random.random() < bias else '0' for _ in range(length))


def human_noisy(length):
    return generate_true_random(length)


def generate_human_like(length=50):
    behaviours = [
        human_near_alternating,
        human_balanced_streak_avoidant,
        human_chunk_pattern,
        human_soft_biased,
        human_noisy,
    ]
    weights = [35, 25, 20, 10, 10]

    behaviour = random.choices(behaviours, weights=weights, k=1)[0]
    return behaviour(length)


def create_dataset(n=1000, length=50):
    data = []
    labels = []

    for _ in range(n):
        data.append(generate_true_random(length))
        labels.append(0)

        data.append(generate_human_like(length))
        labels.append(1)

    return data, labels


def _opposite_bit(bit):
    return '1' if bit == '0' else '0'


def _current_run_length(seq):
    if not seq:
        return 0

    run = 1

    for bit in reversed(seq[:-1]):
        if bit != seq[-1]:
            break
        run += 1

    return run
