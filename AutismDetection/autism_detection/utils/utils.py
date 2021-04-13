"""
This is utils.py contains some utils function (currently get_config() only).
and the main() execution code.
| AUTHOR |    UPDATE    |   EMAIL                                 |
| LiuQM  |  2020/10/06  | contact:liuqm19@mails.tsinghua.edu.cn   |

TODO:

BUG:

Tricky:

"""
#%%
import os

import json


def get_config(config_file) -> tuple:
    """
    Get user configuration.

    Args:
        config_file: [str], config file path.

    Return:
        exs_conf: [dict], experiments design configuration.
        output_conf: [dict], some csv path, which contain intermediate results.

    Raise:
        Config file does not exist!
    """
    if not os.path.exists(config_file):
        raise Exception("Config file: {} does not exist! Please check!".format(config_file))

    # Read Configuration.
    with open(config_file) as f:
        config = json.load(f)
        f.close()

        exs_conf = config['ex_conf']
        output_conf = config['output_conf']

        return exs_conf, output_conf


def main():
    config_file = "D:/src/AutismDetection/AutismDetection/docs/face_config.json"  # need changed by user

    exs_conf, output_conf = get_config(config_file)

    print("Experiments configuration: {}\n".format(exs_conf))

    print("Csv configuration: {}\n".format(output_conf))


if __name__ == '__main__':
    main()

# %%
