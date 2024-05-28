Ensure all requirements are met for each subfolder. 

Masterscript:
    python >= 3.9 (Script developed in python 3.10.12)
    python package requirements in requirements.txt

CLRerNet:
    docker-compose
    nvidia-container-toolkit
    python requirements defined in masterscript
    Make sure that script_run.sh runs without issues and without sudo

    If having issues, make sure the following is true as docker-compose is VERY buggy if not using specific python package versions listed in masterscript/requirements.txt:
        docker==6.1.3
        urllib3<2.0
        requests<2.29.0

    Look here for nms install issues: https://github.com/hirotomusiker/CLRerNet/blob/main/docs/INSTALL.md

HybridNets:
    python requirements defined in masterscript
    
    if having issues with torch.backends, run pip install torch --force-reinstall

TwinLiteNet:
    python requirements defined in masterscript

There is an option in masterscript.py that runs a very small batch of shadows to test for successful installation