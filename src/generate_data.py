import random


def generate_true_random(length=50):
    return ''.join(random.choice(['0', '1']) for _ in range(length))


def human_alternator(length):
    seq = []
    last = random.choice(['0', '1'])

    for _ in range(length):
        if random.random() < 0.8:
            next_val = '1' if last == '0' else '0'
        else:
            next_val = last

        seq.append(next_val)
        last = next_val

    return ''.join(seq)


def human_repeater(length):
    seq = []
    last = random.choice(['0', '1'])

    for _ in range(length):
        if random.random() < 0.7:
            next_val = last
        else:
            next_val = '1' if last == '0' else '0'

        seq.append(next_val)
        last = next_val

    return ''.join(seq)


def human_biased(length):
    bias = random.uniform(0.6, 0.8)
    return ''.join('1' if random.random() < bias else '0' for _ in range(length))


def human_noisy(length):
    return ''.join(random.choice(['0', '1']) for _ in range(length))


def generate_human_like(length=50):
    behaviours = [
        human_alternator,
        human_repeater,
        human_biased,
        human_noisy
    ]

    behaviour = random.choice(behaviours)
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