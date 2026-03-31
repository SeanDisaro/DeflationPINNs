# DeflationPIML

This is the repository to the paper ["Deflation-PINNs: Learning Multiple Solutions for PDEs and Landau-de Gennes"]: http://arxiv.org/abs/2603.27936 in which we present a PINN and DeepONet based model to approximate multiple solutions at once for a Landau de Gennes problem from liquid crystal theory.

## Installation
Set up a python 3.12 environment.
Make sure, that you have a cuda GPU!
Now make sure, that you are in the directory of this repository with your console. Then run

    pip install -r ./requirements.txt

to install the necessary packages. Afterwards run

    pip install -e .

to do some basic set up.

## Structure of the repo
The experiment described in the paper runs, if you run `python main.py` with the environment described above. You can find how the pictures and metrics from the paper were computed in `metricsAndPics4Paper.ipynb`!

_____________________________________________________
These files contain the solution for the problem computed via FEM. We compare the solutions from our Deflation PINN to these FEM solutions.

```
📦data
 ┗ 📂Reduced2DimLDG
 ┃ ┗ 📂trueSolution
 ┃ ┃ ┣ 📜data_LDG_D1_solution.mat
 ┃ ┃ ┣ 📜data_LDG_D2_solution.mat
 ┃ ┃ ┣ 📜data_LDG_R1_solution.mat
 ┃ ┃ ┣ 📜data_LDG_R2_solution.mat
 ┃ ┃ ┣ 📜data_LDG_R3_solution.mat
 ┃ ┃ ┗ 📜data_LDG_R4_solution.mat
 ```
________________________________________________________


`src` contains model implementations, loss functions and the star domain extrapolation implementation for the boundary hard constraint described in the paper.
📦src
 ┣ 📂architectures
 ┃ ┗ 📜DeflationPINN.py
 ┣ 📂lossFunctions
 ┃ ┣ 📜DeflationLoss.py
 ┃ ┣ 📜DeflationPINNLoss.py
 ┃ ┗ 📜LDGPINNLoss.py
 ┣ 📂starDomainExtrapolation
 ┃ ┗ 📜starDomain.py
 ```

________________________________________________________

`tests` contains the implementation of the training loop and some plotting functions used in the experiment from the paper. The pictures can be found in `tests/pictures/deflationPINNTest`. If you want to play around with the hyperparameters yourself, then you can simply adjust them in `tests/deflationPINNTest/run.py`. Once you have adjusted what you need, you can then run again `python main.py`

The files ending with `_zero.png` show what the random initialization of the model looks like the others show the results. This is only to show, that some training occured.
For the evaluation of the results, check out the notebook `metricsAndPics4Paper.ipynb`.


## Cite the paper


[Link to arXiv]: http://arxiv.org/abs/2603.27936


## Team

| Name        | Email                 |
|-------------|-----------------------|
| Sean Disarò | seandisaro@gmail.com  |
| Aras Bacho  | bacho@caltech.edu     |
| Ruma Maity  | rumamaity081@gmail.com|

## Questions
If you have any questions, please write an email to seandisaro@gmail.com

## License
This project is licensed under the GNU license. You may use it however you want!


