import numpy as np
import matplotlib.pyplot as plt
import click

import _loadingdata as ld

@click.command()
@click.option('--_data', type=str)


def main(_data):
    
    shaps = ld.load_data(_data)

    fig, ax = plt.subplots(1, 1, figsize =(20,5))
    for i in range(np.shape(shaps)[0]):
        plt.plot(shaps[i], label = '%s' % (i+1))
    
    plt.legend()
    fig.savefig(f'../plots/{_data}.pdf', bbox_inches='tight')

    
if __name__ == '__main__':
    main()