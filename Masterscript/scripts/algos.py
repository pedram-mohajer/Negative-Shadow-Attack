#########################################
import subprocess
import os
from time import sleep

def check_docker() -> bool:
    """ Checks if there is a running clrernet dev container

    Returns:
        bool value denotaing the existence of the dev container
    """
    check_cmd = ['docker ps -a -f name=clrernet_dev_run -f status=running | grep -w "Up"']
    status = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    clrernet_img = 'clrernet_dev'
    return clrernet_img in status.stdout

def run_clrernet(clrernet_dir: os.path) -> None:
    """ Runs CLRerNet image processing script as a subprocess

    Args:
        clrernet_dir: path to the root directory of the clrernet folder
    """
    print("Running CLRerNet")
    if not check_docker():
        print('No docker container found running, opening docker and waiting 5 seconds')
        subprocess.Popen(['/bin/sh','./script_run.sh'], cwd=clrernet_dir)
        sleep(5) # Popen doesn't properly wait until the docker container starts, this is a safeguard

    get_docker_img = "docker ps -a -f name=clrernet_dev_run -f status=running | awk ' {print $1 }'"
    docker_img = subprocess.run(get_docker_img, shell=True, capture_output=True, text=True)
    img_list= docker_img.stdout[len('CONTAINER\n'):][:-len('\n')]
    first_img = img_list.splitlines()[0]
    clrernet_dir = './Shadow_Attack/CLRerNet/'

    print('Running CLRerNet script...')
    docker_cmd = f'docker exec -i  {first_img} /bin/bash python run_clrernet.py data/input configs/clrernet/culane/clrernet_culane_dla34_ema.py clrernet_culane_dla34_ema.pth --out-file result.png'
    os.system(docker_cmd)


def run_hybridnets(hybridnets_dir: os.path) -> None:
    """ Runs hybridnets image processing script

    Args:
        hybridnets_dir: path to the root directory of the hybridnets folder
    """
    print('Running HybridNets')
    hybridnets_cmd = f'{hybridnets_dir}process_inputs.sh'
    subprocess.run(hybridnets_cmd, shell=True, cwd=hybridnets_dir)
    # os.chdir(oldwd)


def run_twinlitenet(twinlitenet_dir: os.path) -> None:
    """ Runs TwinLiteNets image processing script

    Args:
        twinlitenet_dir: path to the root directory of the twinlitenet folder
    """
    print('Running TwinLiteNet')
    twinlitenet_cmd = f'{twinlitenet_dir}process_images.sh'
    subprocess.run(twinlitenet_cmd, shell=True, cwd=twinlitenet_dir)
    
def run_all(config: dict) -> None:
    """ Runs all three lane detection algortihm processing scripts

    Args:
        config: dict with path to all parent folders
    """
    run_clrernet(config['clrernet_dir'])
    run_hybridnets(config['hybridnets_dir'])
    run_twinlitenet(config['twinlitenet_dir'])