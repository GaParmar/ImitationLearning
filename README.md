# ImitationLearning

## 0. Setup
 - `conda env create --file environment.yml`
 - `conda activate dagger_car`
 - `mkdir .tmp && cd .tmp`
 - `git clone https://github.com/tawnkramer/gym-donkeycar`
 - `cd gym-donkeycar && pip install -e .`

## 1. Collect Expert trajectories using simulator here
 - https://github.com/GaParmar/CarSimulator

## 2. Train the network with behavior cloning
 - `python main_bc.py` 


# References
 - 