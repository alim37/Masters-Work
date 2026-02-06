# AINDANE (Python conversion)

Converted from the provided MATLAB files:
- AINDANE_ALE.m  -> aindane_ale.py
- AINDANE_ACE.m  -> aindane_ace.py
- AINDANE_CR.m   -> aindane_cr.py
- AINDANE_main.m -> aindane_main.py

## Install deps
```bash
pip install numpy opencv-python scikit-image matplotlib scipy
```

## Run
```bash
python aindane_main.py --image image.bmp --save_dir aindane_out
```

Outputs will be saved in `aindane_out/`:
- original.png
- ahe.png
- aindane.png
- quantitative_evaluation.png
