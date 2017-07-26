import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
cmap = sns.color_palette("Paired", 4)


def findnth(string, substring, n):

    parts = string.split(substring, n+1)

    if len(parts) <= n+1:
        return -1

    return len(string)-len(parts[-1])-len(substring)


runs = [i for i in os.listdir('../code_outputs') if i.startswith('2017_07')]

for run in runs:

    filename = os.path.join('../code_outputs', run, 'out.txt')

    elbos = []
    kls = []
    x_test = []
    test_set_elbos = []

    with open(filename, 'rb') as f:

        last_line_test = False

        for l in f.readlines():

            try:
                l = l.decode('ascii')
            except UnicodeDecodeError:
                continue

            if l.startswith('Iteration'):

                elbos.append(float(l[findnth(l, ' ', 3) + 1: findnth(l, ' ', 4)]))

                try:
                    kls.append(float(l[findnth(l, 'KL', 0) + 5: findnth(l, ')', 0)]))
                except:
                    pass

                if last_line_test:
                    x_test.append(len(elbos))

                last_line_test = False

            elif l.startswith('Test set ELBO') and l.strip().endswith('per data point'):

                test_set_elbos.append(float(l[findnth(l, ' ', 3) + 1: findnth(l, ' ', 4)]))

                last_line_test = True

    test_set_elbos = [i for i in test_set_elbos if i != 0]

    moving_average = 10

    x_test_avg = []
    test_set_avg = []

    for t in range(moving_average - 1, len(test_set_elbos)):
        x_test_avg.append(x_test[t])
        test_set_avg.append(np.mean(test_set_elbos[t - (moving_average - 1): t + 1]))

    fig, ax1 = plt.subplots()

    ax1.plot(elbos, label='train', c=cmap[0], zorder=2)
    ax1.plot(x_test_avg, test_set_avg, c=cmap[1], zorder=3)

    ax1.set_ylim(bottom=min((max(elbos) * 3, min(elbos[-20000:]) - 10)), top=round(max(elbos), -1) + 10)

    ax1.tick_params('y', colors=cmap[1])
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('L(X)', color=cmap[1])

    ax2 = ax1.twinx()
    ax2.plot(kls, label='kl divergence', c=cmap[2], zorder=1)

    try:
        ax2.set_ylim(bottom=0, top=max([kls[-1] * 3, 5]))
    except IndexError:
        pass

    ax2.tick_params('y', colors=cmap[3])
    ax2.set_ylabel('KL divergence', color=cmap[3])

    ax1.set_yticks(ax1.get_yticks())
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    ax2.grid(None)

    fig.tight_layout()

    plt.savefig('../pics/elbos/' + run + '.png')

    plt.clf()
