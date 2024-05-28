import os, sys
from scripts.move import move2algos, move2root, resize_images, create_folders, clear_outputs
from scripts.apply_overlay import apply_overlay
from scripts.algos import run_all, run_clrernet, run_hybridnets, run_twinlitenet
from scripts.evaluation import evaluate_attacks
from scripts.analyze_results import analyze

NUM_CSV = 320

def build_config_dict(data_dir: os.path, clrernet_dir: os.path, hybridnets_dir: os.path, \
                      twinlitenet_dir: os.path) -> dict: 
    """ Builds a single dict that contains all necessary folders for this script

    Args:
        data_dir: root directory of masterscript data directory
        clrernet_dir: root directory of clrnernet
        hybridnets_dir: root directory of hybridnets directory
        hybridnets_env: name of hybridnets conda environment
    
    Returns:
        dict with all necessary configuration information
    """
    input_folder = os.path.join(data_dir, 'input')
    output_folder = os.path.join(data_dir, 'output')
    results_folder = os.path.join(data_dir, 'results')
    attr_folder = os.path.join(data_dir, 'attr_files')

    config = {'data_dir': data_dir,
              'input_folder': input_folder, 'output_folder': output_folder,
              'attr_folder': attr_folder, 'results_folder': results_folder, 
              'clrernet_dir': clrernet_dir, 'hybridnets_dir': hybridnets_dir,
              'twinlitenet_dir': twinlitenet_dir} 
    
    return config

def stepwise_progess(stepwise: bool) -> None:
    """ Pauses progress if stepwise is true
    
    Args:
        stepwise: whether or not to pause operation in between steps
    """
    if stepwise:
        print('Enter Q to exit, or press any other key to continue')
        choice = input().upper()
        if choice == 'Q':
            sys.exit()

def run_specified_algo(config: dict) -> None:
    """ Prompts for and runs a specific lane detection algorithm

    Args:
        config: dict containing all configuration information
    """
    print('\t1: CLRerNet')
    print('\t2: Hybridnets')
    print('\t3: TwinLiteNet')
    print('\t4: All algorithms')
    algo_choice = int(input())

    if algo_choice == 4:
        run_all(config)
    elif algo_choice == 1:
        run_clrernet(config['clrernet_dir'])
    elif algo_choice == 2:
        run_hybridnets(config['hybridnets_dir'])
    elif algo_choice == 3:
        run_twinlitenet(config['twinlitenet_dir'])
    else:
        print(f'Unknown choice: {algo_choice}. Try again')


def run_specified_step(config):
    print('Choose a step of the script to run')
    print('\t1: Resize images in input folder')
    print('\t2: Apply Overlay to images in input folder')
    print('\t3: Move input and output images to lane detection folders')
    print('\t4: Run all lane detection algorithms on their input images')
    print('\t5. Move output of lane detection algorithms to evaluation folder')
    print('\t6: Evaluate attacks')
    print('\t7: Delete input/output folders')
    print('\t8: Analyze')

    choice = int(input())

    if choice == 1:
        resize_images(config['input_folder'])
    elif choice == 2:
        apply_overlay(config, 0, verbose=True)
    elif choice == 3:
        move2algos(config)
    elif choice == 4:
        run_all(config)
    elif choice == 5:
        move2root(config)
    elif choice == 6:
        evaluate_attacks(config, 0)
    elif choice == 7:
        clear_outputs(config)
    elif choice == 8:
        analyze(config)
    else:
        print(f'Unsupported option: {choice}!')

def test_install(config):
    """ Runs small batch of shadow attacks to test functionality
    """
    resize_images(config['input_folder'])
    apply_overlay(config, -1, verbose=True)
    move2algos(config)
    run_all(config)
    move2root(config)
    evaluate_attacks(config, -1)
    
def run_masterscript(config, stepwise=False) -> None:
    """ Runs the entirety of the shadow attack process flow

    Args:
        data_dir: path to the root directory of the data folder
        clrernet_dir: path to the root directory of the clrernet folder
        hybridnets_dir: path to the root directory of the hybridnets folder
        twinlitenet_dir: path to the root directory of the twinlitenet folder
        hybridnets_env: name of the hybridnets conda environment
        stepwise: Enables stepwise progression of script
    """

    resize_images(config['input_folder'])
    stepwise_progess(stepwise)

    for i in range(NUM_CSV):

        clear_outputs(config) 

        apply_overlay(config, i, verbose=True)
        stepwise_progess(stepwise)

        move2algos(config)
        stepwise_progess(stepwise)

        run_clrernet(config['clrernet_dir'])
        # run_hybridnets(config['hybridnets_dir'])
        stepwise_progess(stepwise)

        move2root(config)
        stepwise_progess(stepwise)

        evaluate_attacks(config, i)

    sys.exit()

def main():
    try:
        with open('config.txt', 'r') as config_file:
            lines = config_file.read().splitlines()
    except:
        print('Error reading config file!')
        exit()

    data_dir = lines[0]
    clrerent_dir = lines[1]
    hybridnets_dir = lines[2]
    twinlitenet_dir =lines[3]

    config = build_config_dict(data_dir, clrerent_dir, hybridnets_dir, twinlitenet_dir)
    create_folders(config)

    while True:
        print('Choose an option below')
        print('\tEnter R to run entire script')
        print('\tEnter S to run entire script in steps')
        print('\tEnter A to run lane detection algorithms')
        print('\tEnter N to run specific step of script')
        print('\tEnter T to run a small test batch')
        print('\tEnter Q to Quit')
        choice = input().upper()

        if choice == 'Q':
            sys.exit()

        elif choice == 'R':
            run_masterscript(config)

        elif choice == 'S':
            run_masterscript(config, stepwise=True)

        elif choice == 'A':
            run_specified_algo(config)

        elif choice == 'N':
            run_specified_step(config)
        
        elif choice =='T':
            test_install(config)

        else:
            print(f'Unknown command {choice}! Please enter a recognized command')

if __name__ == '__main__':
    main()
