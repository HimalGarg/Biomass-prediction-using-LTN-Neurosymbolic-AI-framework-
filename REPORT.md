# Neuro-Symbolic Biomass Prediction with Logical Tensor Networks

## 1. Project Objective & Context
This project implements a neuro-symbolic framework for predicting pasture biomass by fusing visual image features (pasture plots) with categorical and numerical agricultural metadata. The target biomass categories are:
- `Dry_Clover_g` (dry mass of clover components)
- `Dry_Dead_g` (dry mass of dead plant components)
- `Dry_Green_g` (dry mass of green plant components)
- `Dry_Total_g` (aggregate total dry mass)
- `GDM_g` (Green Dry Matter mass)

A pure deep learning approach can learn to predict these values independently but often violates critical physical/logical rules of composition. A pure symbolic model can enforce the equations but fails to learn from sensory inputs like images. The **Logical Tensor Network (LTN)** framework bridges this gap by grounding a neural network regressor as an LTN function and adding differentiable logical constraints (axioms) directly to the loss objective.

---

## 2. Advanced Data Processing & Imbalance Handling

### State-Stratified Splitting
The pasture records exhibit geographical imbalances across different regions/states. To ensure robust testing and representation:
- Data is split into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets.
- The split is stratified according to the geographical `State` column. This guarantees that each partition contains a representative distribution of pasture locations.
- A fallback check ensures that if a state has fewer than 3 samples, it is grouped gracefully to prevent splitting errors.

### Feature & Target Normalization
- **Tabular Features**: Scaled using training-set statistics (mean and standard deviation).
- **Biomass Targets**: Scaled by a single, shared scaling factor computed from the training split. Fusing targets under a single global scale factor is critical because it preserves the linear additive structure in the normalized space (i.e., if $Total = Clover + Dead + Green$, then $Total_{norm} = Clover_{norm} + Dead_{norm} + Green_{norm}$).

---

## 3. Multimodal Neural Architecture

The primary perception network utilizes a multi-stream fusion architecture:
1.  **Visual Stream**: Resized pasture images are processed through either a Custom CNN or a pre-trained **MobileNetV2** backbone.
2.  **Metadata Stream**: Numerical metrics (e.g., NDVI, average height) and categorical features (one-hot state/species) are processed through dense layers.
3.  **Fusion Layer**: The visual and metadata embeddings are concatenated into a shared multimodal representation.
4.  **Non-Saturating Output Regressor**:
    *   **The Dead ReLU Challenge**: Early tests using ReLU outputs resulted in the model collapsing into predicting exact `0.0` values for zero-heavy pasture targets (like Clover).
    *   **The Softplus Solution**: To allow continuous, positive-only regression gradients without bounds or saturation limits, we use the `softplus` activation function:
        $$f(x) = \log(1 + e^x)$$
    *   **Inverse Softplus Bias Initialization**: The output layer's bias weights are initialized using the inverse softplus of target means to match the ground truth prior distribution and stabilize early training epochs:
        $$b_{init} = \log(e^{\mu} - 1)$$

---

## 4. Robust Regression Loss

Standard regression frameworks often use Mean Squared Error (MSE). However, pasture biomass data has skewed target distributions and extreme outliers. To mitigate this:
- **Huber Loss ($\delta = 0.15$)** is implemented as the base supervised loss:
$$
\mathcal{L}_{\text{Huber}}(y,\hat{y})=
\begin{cases}
\frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \le \delta \\
\delta(|y-\hat{y}|-\frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$
- This makes the regression objective robust to outliers while maintaining stable gradients near zero.

---

## 5. Logical Grounding & Axioms

Using the Logic Tensor Networks package, the regressor network is grounded as a mathematical function:
$$f(\text{image}, \text{metadata}) \rightarrow [clover, dead, green, total, gdm]$$

We define real-valued regression predicates that measure how closely a statement is satisfied on a scale of `[0, 1]`. 

### The Knowledge Base Axioms:
1.  **Data Fitting**: The predicted values should match the observed targets ($y$):
    $$\forall x, \text{Close}(f(x), y)$$
2.  **Total Mass Decomposition**: The sum of the parts must equal the total mass:
    $$\forall x, \text{Dry Total (g)} \approx \text{Dry Clover (g)} + \text{Dry Dead (g)} + \text{Dry Green (g)}$$
3.  **Green Dry Matter (GDM) Composition**: Green dry matter must equal green plus clover biomass:
    $$\forall x, \text{GDM (g)} \approx \text{Dry Clover (g)} + \text{Dry Green (g)}$$
4.  **Logical Ordering (Part-Whole Constraint)**: Aggregate masses must be greater than or equal to their component parts:
    $$\forall x, \text{Dry Total (g)} \ge \text{Dry Clover (g)}$$
    $$\forall x, \text{Dry Total (g)} \ge \text{Dry Dead (g)}$$
    $$\forall x, \text{Dry Total (g)} \ge \text{Dry Green (g)}$$
    $$\forall x, \text{GDM (g)} \ge \text{Dry Clover (g)}$$
    $$\forall x, \text{GDM (g)} \ge \text{Dry Green (g)}$$

### Optimization Objective:
The loss function combines the supervised regression loss with the logical satisfiability ($\phi$) calculated using fuzzy logic operators (p-mean error):
$$\mathcal{L}_{total} = \mathcal{L}_{Huber} + \lambda_{ltn} \cdot (1 - \text{Sat}(\Phi))$$

*   **Warm-Up Period**: Supervised warm-up runs for the first few epochs to establish scale before logical constraints are gradually ramped in, preventing shortcut learning (collapsing to zero).

---

## 6. Tuning Hyperparameters & Experimental Results

### The Tuned Hyperparameters:
- **LTN weight ($\lambda_{ltn}$)**: `0.01` (lowered from `0.1` to prevent shortcut learning where the network predicts zero to satisfy logical rules at the cost of actual regression accuracy).
- **Rule Tolerance**: `0.12` (allowing soft deviations to accommodate measurement noise).
- **Backbone**: MobileNetV2 with ImageNet weights.
- **Base Loss**: Huber Loss ($\delta = 0.15$).

### Empirical Performance Comparison:

| Metric | Symbolic Baseline | Neural Model (Only Huber) | LTN (Neuro-Symbolic) |
| :--- | :---: | :---: | :---: |
| **Combined $R^2$** | -0.122 | 0.312 | **0.347** |
| **Green $R^2$** | -0.054 | 0.589 | **0.631** |
| **GDM $R^2$** | -0.021 | 0.612 | **0.667** |
| **Total Sum Violation (g)** | **0.000** | 18.25 | **4.74** *(74% reduction)* |
| **Order Violation (g)** | **0.000** | 9.42 | **0.75** *(92% reduction)* |

### Key Takeaways:
1.  **Logical Consistency**: The LTN model reduces physical rule violations dramatically (92% reduction in component ordering violations and 74% reduction in summation violations) compared to the standard neural model.
2.  **Regression Performance**: Enforcing logical constraints actually acts as a regularizer, improving generalized prediction quality on unseen data and leading to a higher $R^2$ score across all targets.