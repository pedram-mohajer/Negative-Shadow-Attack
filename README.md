# Project Setup Guide

This guide helps ensure that all requirements are met for each subfolder in the project.

---

## 🛠️ Masterscript

- **Python Version**: `>= 3.9` (Developed in Python 3.10.12)
- **Dependencies**: Install the Python package requirements listed in `requirements.txt`.

---

## 🚀 CLRerNet

- **Required Tools**:
  - `docker-compose`
  - `nvidia-container-toolkit`
  
- **Dependencies**: Python requirements are defined in the `masterscript`.
  
- **Setup**: Make sure `script_run.sh` runs without issues and **without** using `sudo`.

### ⚠️ Troubleshooting

If you're facing issues, ensure the following specific package versions are installed (as `docker-compose` can be **buggy** with different versions):

- `docker==6.1.3`
- `urllib3<2.0`
- `requests<2.29.0`

For more details on installation issues, particularly for `nms`, check out the [installation guide](https://github.com/hirotomusiker/CLRerNet/blob/main/docs/INSTALL.md).

---

## 🧠 HybridNets

- **Dependencies**: Python requirements are defined in the `masterscript`.

### ⚠️ Torch Backend Issues?

If you encounter issues with `torch.backends`, run the following command:

```bash
pip install torch --force-reinstall
