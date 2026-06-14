# RFQC Stage Report Figure Captions

**Figure 1. Structural model selection during five-fold cross-validation.**
Mean validation G-mean is shown for the q* prevalence threshold during (a, b) the
quick search over split rule, `mtry`, and terminal node size and (c) local refinement
under Gini splitting. Cell text reports the fold mean and standard deviation. The
outlined cell marks the structure locked for final fitting (`mtry = 24`, terminal node
size = 20).

**Figure 2. Training out-of-bag performance across candidate classification thresholds.**
The filled square and blue vertical line identify the locked q* prevalence threshold.
The open circle and grey vertical line identify the threshold with the maximum
training OOB G-mean. The q* threshold retained near-maximal G-mean while providing a
smaller sensitivity-specificity imbalance and was fixed before test evaluation.

**Figure 3. Discrimination performance on the untouched final test set.**
(a) Receiver operating characteristic curve and (b) precision-recall curve. Filled
squares show the operating point produced by the pre-locked q* threshold. The dashed
horizontal line in panel (b) is the test-set fraud prevalence. The precision axis is
restricted to 0-0.12 to display the operationally relevant portion of the curve.
