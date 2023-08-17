# conda init zsh
conda activate main_env
. ./main/.envrc


#$ pip install stable-baselines3==1.7.0
#>>>  WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f66a3be4790>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/stable-baselines3/
#disconnect the ethernets (using the Ubuntu UI) then try again


# % python ./main/main.py               
    # Traceback (most recent call last):
    #   File "./main/main.py", line 10, in <module>
    #     from specific_tools.train_ppo import * # required because of pickle lookup
    #   File "/home/administrator/repos/action-adjuster/main/specific_tools/train_ppo.py", line 15, in <module>
    #     import scipy.signal
    #   File "/home/administrator/anaconda3/lib/python3.8/site-packages/scipy/__init__.py", line 155, in <module>
    #     from . import fft
    #   File "/home/administrator/anaconda3/lib/python3.8/site-packages/scipy/fft/__init__.py", line 79, in <module>
    #     from ._helper import next_fast_len
    #   File "/home/administrator/anaconda3/lib/python3.8/site-packages/scipy/fft/_helper.py", line 3, in <module>
    #     from ._pocketfft import helper as _helper
    #   File "/home/administrator/anaconda3/lib/python3.8/site-packages/scipy/fft/_pocketfft/__init__.py", line 3, in <module>
    #     from .basic import *
    #   File "/home/administrator/anaconda3/lib/python3.8/site-packages/scipy/fft/_pocketfft/basic.py", line 6, in <module>
    #     from . import pypocketfft as pfft
    # ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.22' not found (required by /home/administrator/anaconda3/lib/python3.8/site-packages/scipy/fft/_pocketfft/pypocketfft.cpython-38-x86_64-linux-gnu.so)
    # administrator@cpr-tamu11 ~/repos/action-adjuster


# conda install libgcc=5.2.0

    # Collecting package metadata (current_repodata.json): done
    # Solving environment: failed with initial frozen solve. Retrying with flexible solve.
    # Collecting package metadata (repodata.json): done
    # Solving environment: failed with initial frozen solve. Retrying with flexible solve.

    # PackagesNotFoundError: The following packages are not available from current channels:

    #   - libgcc=5.2.0

    # Current channels:

    #   - https://repo.anaconda.com/pkgs/main/linux-64
    #   - https://repo.anaconda.com/pkgs/main/noarch
    #   - https://repo.anaconda.com/pkgs/r/linux-64
    #   - https://repo.anaconda.com/pkgs/r/noarch

    # To search for alternate channels that may provide the conda package you're
    # looking for, navigate to

    #     https://anaconda.org

    # and use the search bar at the top of the page.
    
# conda config --append channels conda-forge
# conda install libgcc=5.2.0

# $ bash
# $ pp
# $ cd repos/action-adjuster
# $ . ./main/.envrc
# $ conda install ros-roscore ros-rospy 

# Collecting package metadata (current_repodata.json): done
# Solving environment: failed with initial frozen solve. Retrying with flexible solve.
# Collecting package metadata (repodata.json): done
# Solving environment: failed with initial frozen solve. Retrying with flexible solve.

# PackagesNotFoundError: The following packages are not available from current channels:

#   - ros-roscore

# Current channels:

#   - https://repo.anaconda.com/pkgs/main/linux-64
#   - https://repo.anaconda.com/pkgs/main/noarch
#   - https://repo.anaconda.com/pkgs/r/linux-64
#   - https://repo.anaconda.com/pkgs/r/noarch
#   - https://conda.anaconda.org/conda-forge/linux-64
#   - https://conda.anaconda.org/conda-forge/noarch

# To search for alternate channels that may provide the conda package you're
# looking for, navigate to

#     https://anaconda.org

# and use the search bar at the top of the page.
# conda install ros-rospy
# conda install ros-common-msgs
# conda install ros-message-filters