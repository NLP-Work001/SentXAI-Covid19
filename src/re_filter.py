"""
This script is run during `filter_params.sh` to ensure that no numpy data types exist
in the parameter score file.
"""

import re
import numpy as np
import sys

if __name__ == "__main__":
    with open(f"{sys.argv[1]}/best_param_scores_.txt", "r") as f:
        data = f.readlines()
        for idx, line in enumerate(data):
            line = line.strip()
            if idx ==0:
                print(line)
            else:
                print(re.sub(r"np\.\w+\(|\)", "", line))
