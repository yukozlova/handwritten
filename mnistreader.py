import random
def get_train_samples(batch_size):
    with open("train.csv", "r") as file:
        text = file.read()
    textlines_all = text.strip().split("\n")

    # random.shuffle(textlines_all)
    textlines = textlines_all[:100]
    start = 0

    while start < len(textlines):
        labels = []
        targets = []
        inputs = []
        end = start + batch_size
        for text in textlines[start: end]:
            cells = text.split(",")
            labels.append(int(cells[0]))
            targets.append([float(c) for c in cells[1:11]])
            inputs.append([float(c) for c in cells[11:]])

        yield labels, targets, inputs
        start += batch_size


def get_test_samples():
    with open("test.csv") as file:
        text = file.read()
    textlines = text.strip().split("\n")

    labels = []
    targets = []
    inputs = []

    for line in textlines:
        cells = line.split(",")
        labels.append(int(cells[0]))
        targets.append([float(c) for c in cells[1:11]])
        inputs.append([float(c) for c in cells[11:]])

    return labels, targets, inputs


def plot_number(inputs):
    line = ""
    for p in inputs:
        line += ".░▒▓█"[round(p*4)]
        if len(line) > 27:
            print(line)
            line = ""
