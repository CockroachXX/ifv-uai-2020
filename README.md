# Verifying Individual Fairness in Machine Learning Models

Authors: Philips George John, Deepak Vijayakeerthy, Diptikalyan Saha

_Paper accepted in UAI 2020 conference._

This is the Python implementation used for the bias verification experiments in the paper.

To run the experiments:

### Linear model (logistic regression)

``python run_linear.py <ds-name> <model-file-name> [mark-prot-attrs=True|False]``

Where ``<ds-name>`` is ``german``, ``adult``, ``fraud`` or ``credit``, ``model-file-name`` can be anything (the trained model is saved to that file, and if the file exists, the script will try to load the model from it), and ``mask-prot-attrs`` being set to ``True`` will result in the protected attribute(s) for the specified dataset being masked (set to 0). The default for `mask-prot-attrs` is `False`.

### Polynomial kernel SVM

``python run_polynomial.py <ds-name> <model-file-name> [mark-prot-attrs=True|False]``

The parameters have the same interpretation as in the linear case. As mentioned in the paper, the polynomial kernel used will have degree $2$, $C = 1$, $\gamma = 0.001$, and $r = 0$ (parameters as per ``sklearn.svm.SVC`` docs).

### RBF kernel SVM

``python run_rbf.py <ds-name> <model-file-name> [mark-prot-attrs=True|False] [model-type=1|2]``

The parameters other than ``model-type`` have the same interpretation as in the linear/polynomial case. The parameter ``model-type`` is used to choose between the two sets of rbf kernel parameters mentioned in the paper. ``model-type=1`` corresponds to $C = 1000 : \gamma = 10^{-4}$, and ``model-type=2`` corresponds to $C = 1 : \gamma = 0.5$ (parameters as per ``sklearn.svm.SVC`` docs).

## Acknowledgements

We have included preprocessed versions of the following publicly available datasets --- German Credit (UCI ML Repo), Fraud Detection (Kaggle), Credit (ISLR - Gareth James), Modified Adult Dataset (UCI ML Repo) --- in the datasets folder.

We have used an adapted version of `sdp.py` from `Irene` (Mehdi Ghasemi -- https://github.com/mghasemi/Irene/) in our implementation. This code is available under the MIT License.

