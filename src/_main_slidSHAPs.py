import _slidSHAPs
import _loadingdata
import click
import numpy as np


@click.command()
@click.option('--_data', type=str)
@click.option('--_window_width', type=int)
@click.option('--_overlap', type=int)
@click.option('--_type', type=str)
@click.option('--_subsets_bound', type=int)
def main(_data, _window_width, _overlap, _type, _subsets_bound, _approx='max'):
    if _data == 'concept_drifted_data':
        _mydata = _loadingdata.load_data_CD()
    else:
        _mydata = _loadingdata.load_data(_data)
    print(f'running for {_data} with _window_width {_window_width} and _overlap {_overlap}')
    data_shaps = _slidSHAPs.run_slidshaps(_mydata, _window_width, _overlap, _type, _subsets_bound, _approx)
    _path = '../results/'
    _file_name = f'shapleyvalues_data{_data}_windowwidth{_window_width}_overlap{_overlap}_CHECK2.txt'      
    np.savetxt(_path + _file_name, data_shaps, delimiter =", ")


if __name__ == '__main__':
    main()