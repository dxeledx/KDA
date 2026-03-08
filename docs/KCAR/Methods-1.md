3. Method  
==========

3.1 Problem Setup

---

We consider subject-independent motor imagery decoding under cross-subject distribution shift.  
Let the labeled source-domain data be

$ \mathcal{D}_s=\{(X_i^s,y_i^s)\}_{i=1}^{N_s}, $

where  $ X_i^s $  denotes an EEG trial and  $ y_i^s\in\{1,\dots,K\} $  is the corresponding class label.  
For each trial, we estimate a covariance matrix

$ C_i^s \in \mathrm{SPD}(n), $

where  $ \mathrm{SPD}(n) $  denotes the space of  $ n\times n $  symmetric positive definite matrices.  
At test time, we observe an unlabeled target stream

$ \mathcal{D}_t=\{X_t^t\}_{t=1}^{T}, \qquad C_t\in \mathrm{SPD}(n), $

arriving sequentially.

Our goal is to learn a **causal dynamic aligner** that determines, for each target trial  $ t $ , how strongly it should be aligned to the source domain. This design follows your original idea that the alignment strength should not be fixed globally, but should vary with the current trial state and recent temporal context.





Formally, we seek a trial-wise transformation

$ C_t'=\mathcal{A}(C_t;w_t), \qquad w_t\in[0,1], $

where  $ w_t $  denotes the alignment strength.  
When  $ w_t $  is large, the target trial is strongly aligned toward the source geometry; when  $ w_t $  is small, more subject-specific structure is preserved.

---

3.2 Riemannian Base Pipeline

---

We build our method upon a stable Riemannian baseline consisting of three stages:  
(1) Riemannian alignment,  
(2) CSP-based feature extraction, and  
(3) a lightweight classifier such as LDA or SVM.  
This directly follows your original “success floor” design, where RA + CSP + a simple classifier acts as the safe backbone.

Let  $ M_s $  denote the Riemannian mean of the source covariance matrices.  
A standard source-referenced Riemannian alignment maps  $ C_t $  to

$ \widetilde{C}_t = M_s^{-1/2} C_t M_s^{-1/2}. $

Instead of always using either  $ C_t $  or  $ \widetilde C_t $ , we perform **partial alignment** controlled by  $ w_t $ .  
To preserve the SPD geometry, we define the aligned covariance through geodesic interpolation:

$ \mathcal{A}(C_t;w_t) = C_t^{1/2} \Big( C_t^{-1/2}\widetilde{C}_t C_t^{-1/2} \Big)^{w_t} C_t^{1/2}. $

This satisfies

$ \mathcal{A}(C_t;0)=C_t,\qquad \mathcal{A}(C_t;1)=\widetilde C_t. $

Compared with the original convex mixing form  $ C_t'=(1-w_t)C_t+w_t\,RA(C_t) $ , this formulation is more consistent with the manifold geometry while preserving the same interpretation of  $ w_t $ . Your original notes already positioned the scalar-weight version as the preferred starting point because of its simplicity, interpretability, and debuggability.

After alignment, CSP is applied to obtain discriminative features

$ x_t=\phi(C_t'), $

which are fed into a classifier  $ f_\theta $  to produce logits and posterior probabilities:

$ \hat p_t = \mathrm{softmax}(f_\theta(x_t)). $

---

3.3 Alignment State Representation

---

Your original design emphasized that the context vector should capture “how much alignment is needed now,” first through geometric quantities and later optionally through uncertainty or behavior-related signals.

In the revised formulation, we make this idea explicit by defining a **dynamic alignment state**.

### 3.3.1 Geometric state
We first map each covariance matrix into the tangent space centered at the source mean  $ M_s $ :

$ v_t = \mathrm{vec}\!\left( \log\!\big(M_s^{-1/2} C_t M_s^{-1/2}\big) \right). $

To reduce dimensionality and noise, we apply a projection  $ P\in\mathbb{R}^{r\times d_v} $ :

$ z_t = P v_t \in \mathbb{R}^r. $

The vector  $ z_t $  is the low-dimensional geometric state of the current trial.

### 3.3.2 Auxiliary state
We augment  $ z_t $  with a compact set of causal auxiliary variables:

$ u_t = \big[ d_{\text{src},t}, d_{\text{tgt},t}, \sigma_t, H_{t-1}, p^{\max}_{t-1} \big], $

where:

+ $ d_{\text{src},t} $  is the Riemannian distance from  $ C_t $  to the source mean  $ M_s $ ,
+ $ d_{\text{tgt},t} $  is the distance from  $ C_t $  to the running target mean,
+ $ \sigma_t $  measures recent short-window variability,
+ $ H_{t-1} $  is the predictive entropy from the previous step,
+ $ p^{\max}_{t-1} $  is the previous-step maximum posterior probability.

The full alignment state is then defined as

$ s_t = [z_t,u_t]\in\mathbb{R}^d. $

This preserves the spirit of your original  $ c_t $  design, but removes the dependence on pseudo-label accuracy trends in the first version, making the state more stable and fully causal. Your earlier notes also suggested starting from the purely geometric variant before adding noisier behavior-related terms.

---

3.4 Koopman-Lifted Class-Conditional Dynamics

---

The central hypothesis of this work is that the “need for alignment” evolves as a dynamic process.  
Instead of modeling raw EEG dynamics directly, we model the dynamics of the alignment state  $ s_t $ . This matches your broader motivation that cross-subject inconsistency is not static, but varies with subject identity, temporal drift, and current system state.



We introduce a lifting map

$ \psi_\omega:\mathbb{R}^d\rightarrow\mathbb{R}^m, $

which maps the alignment state into a Koopman feature space.  
In the simplest version,  $ \psi_\omega $  can be a fixed polynomial dictionary; in a learned version, it can be a lightweight MLP.

### 3.4.1 Source-domain class-conditional dynamics
For each class  $ y\in\{1,\dots,K\} $ , we fit a linear operator  $ K_s^{(y)} $  in the lifted space:

$ K_s^{(y)} = \arg\min_K \sum_{\tau\in\mathcal{T}_y^s} \left\| \psi_\omega(s_{\tau+1})-K\psi_\omega(s_\tau) \right\|_2^2 +\lambda_K\|K\|_F^2, $

where  $ \mathcal{T}_y^s $  denotes the set of source transitions associated with class  $ y $ .

These operators characterize how alignment-relevant states evolve in the source domain under different motor imagery conditions.

### 3.4.2 Local target-domain dynamics
At test time, we estimate a local target operator from the recent sliding window

$ \mathcal{W}_t=\{t-m,\dots,t-1\}: $

$ K_t = \arg\min_K \sum_{\tau\in\mathcal{W}_t} \beta^{\,t-1-\tau} \left\| \psi_\omega(s_{\tau+1})-K\psi_\omega(s_\tau) \right\|_2^2 +\lambda_K\|K\|_F^2, $

where  $ \beta\in(0,1] $  is a temporal decay factor.

The operator  $ K_t $  serves as a local approximation of the recent target-state evolution.

---

3.5 Koopman Conditional Alignment Risk

---

Your original scheme used uncertainty statistics and heuristic feedback rules to decide when stronger alignment might be needed.

Here we replace that hand-designed logic with a single dynamic quantity that directly measures whether the **source dynamics are currently compatible with the target evolution**.

### 3.5.1 Source-explained transition residual
Given the classifier posterior  $ \hat p_\tau $ , we define a posterior-weighted source operator:

$ \bar K_s(\tau) = \sum_{y=1}^{K}\hat p_\tau(y)\,K_s^{(y)}. $

We then measure how well the source dynamics explain the next target state:

$ e^{\text{src}}_\tau = \left\| \psi_\omega(s_{\tau+1}) - \bar K_s(\tau)\psi_\omega(s_\tau) \right\|_2^2. $

### 3.5.2 Target-local transition residual
Similarly, we compute the local target residual:

$ e^{\text{tgt}}_\tau = \left\| \psi_\omega(s_{\tau+1}) - K_t\psi_\omega(s_\tau) \right\|_2^2. $

### 3.5.3 KCAR definition
We define the **Koopman Conditional Alignment Risk (KCAR)** over the recent window as

$ \rho_t = \frac{1}{m} \sum_{\tau\in\mathcal{W}_t} \frac{ e^{\text{src}}_\tau-e^{\text{tgt}}_\tau }{ e^{\text{src}}_\tau+e^{\text{tgt}}_\tau+\varepsilon }. $

By construction,  $ \rho_t\in[-1,1] $ .

Interpretation:

+ $ \rho_t > 0 $ : source-domain dynamics explain the current target evolution worse than the target’s own local dynamics, indicating **high alignment risk**;
+ $ \rho_t \approx 0 $ : source and target explanations are comparable;
+ $ \rho_t < 0 $ : source dynamics provide a good explanation of the current target evolution, suggesting that stronger alignment is **relatively safe**.

Thus, KCAR quantifies whether source knowledge is currently helping or misleading the target adaptation process.

---

3.6 Risk-Guided Dynamic Alignment Gate

---

In your original framework, a network  $ g(\cdot) $  predicts  $ w_t $  from the context vector  $ c_t $ .



We keep this design, but now the gate is explicitly driven by alignment risk.

Let

$ g_t= [d_{\text{src},t}, d_{\text{tgt},t}, \sigma_t, H_{t-1}, p^{\max}_{t-1}], $

and define the final gate input as

$ c_t^{(B)}=[g_t,\rho_t]. $

For interpretability, we begin with a linear-sigmoid gate:

$ w_t = \sigma\!\left( a^\top g_t - b\rho_t + c \right), \qquad b\ge 0. $

This ensures a monotone risk response:

$ \frac{\partial w_t}{\partial \rho_t} = -b\,w_t(1-w_t)\le 0. $

Hence, larger KCAR leads to weaker alignment, while smaller KCAR leads to stronger alignment.

Compared with the original behavior-guided feedback rules, this formulation has two advantages.  
First, it is fully differentiable and can be trained jointly with the classifier.  
Second, it replaces several heuristic interventions with a single interpretable state variable.

---

3.7 Training Objective

---

Your original training objective combined classification loss with a smoothness regularizer on consecutive alignment weights.



We retain that structure and add a KCAR-based risk term.

### 3.7.1 Classification loss
For labeled source or pseudo-target training episodes, we use standard cross-entropy:

$ \mathcal{L}_{\text{cls}} = \frac{1}{N}\sum_t \mathrm{CE}(\hat y_t,y_t). $

### 3.7.2 Temporal smoothness
To avoid oscillatory alignment decisions, we penalize abrupt changes of the gate:

$ \mathcal{L}_{\text{sm}} = \frac{1}{N}\sum_t (w_t-w_{t-1})^2. $

This term mirrors your original  $ L_{\text{smooth}} $  design and remains important for stable online adaptation.



### 3.7.3 Risk regularization
We introduce a new risk-aware term:

$ \mathcal{L}_{\text{risk}} = \frac{1}{N}\sum_t w_t\,\rho_t. $

Minimizing this term encourages the model to reduce alignment strength when the estimated risk is high, and to allow stronger alignment when the risk is low or negative.

### 3.7.4 Overall objective
The full training objective is

$ \mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_{\text{sm}}\mathcal{L}_{\text{sm}} + \lambda_{\text{risk}}\mathcal{L}_{\text{risk}} + \lambda_{\text{reg}}\|\Theta\|_2^2, $

where

$ \Theta=\{\theta,\omega,P,a,b,c\}. $

---

3.8 Episodic Offline Training

---

The original notes proposed offline training on source data followed by optional online adaptation on target data.





For the revised method, a better strategy is to simulate target adaptation already during training.

We therefore adopt an **episodic leave-one-subject-out meta-training scheme**.  
In each episode, one source subject is temporarily treated as a pseudo-target, while the remaining subjects form the meta-source set.

For each episode:

1. Compute the source Riemannian mean  $ M_s $  and class-conditional Koopman operators  $ K_s^{(y)} $  from the meta-source subjects;
2. Stream the pseudo-target subject sequentially to construct local target operators  $ K_t $ ;
3. Compute KCAR for each pseudo-target step;
4. Generate dynamic weights  $ w_t $ , aligned covariances  $ C_t' $ , and features  $ x_t $ ;
5. Optimize the full objective using the pseudo-target labels.

This strategy teaches the model to infer dynamic alignment risk in a way that matches the true test-time setting.

---

3.9 Online Inference

---

Your original online pipeline already had the right causal structure: initialize target statistics, compute  $ c_t $ , predict  $ w_t $ , align, classify, and optionally adapt.



We keep that logic, but replace heuristic feedback with KCAR-based control.

### Initialization
Before online decoding starts, we:

1. load the trained source mean  $ M_s $ , CSP filters, classifier  $ f_\theta $ , lifting map  $ \psi_\omega $ , and class-conditional source operators  $ K_s^{(y)} $ ;
2. use the first  $ K $  unlabeled target trials to initialize the running target mean and recent-state buffer;
3. initialize the local target operator  $ K_t $ .

### Sequential prediction
For each new target trial  $ C_t $ :

1. compute the state  $ s_t $ ;
2. update the local target operator  $ K_t $  from the most recent window;
3. compute the current KCAR value  $ \rho_t $ ;
4. obtain the dynamic alignment strength

$ w_t=\sigma(a^\top g_t-b\rho_t+c); $

5. align the covariance

$ C_t'=\mathcal{A}(C_t;w_t); $

6. extract CSP features  $ x_t=\phi(C_t') $ ;
7. predict label and posterior probabilities via  $ f_\theta(x_t) $ .

### Optional safe self-adaptation
A conservative pseudo-label update can be enabled only when both of the following conditions hold:

$ \rho_t<\delta_{\text{safe}} \quad\text{and}\quad p_t^{\max}>\tau_{\text{conf}}. $

This criterion is stricter than using confidence alone, since it requires both high posterior certainty and low estimated alignment risk.

---

3.10 Relation to Representation–Decision Mismatch

---

Your original manuscript used CKA to quantify representation similarity and paired it with a coarse behavior consistency measure to analyze “representation–behavior inconsistency.”



In the revised paper, I建议 Method 里只保留一个轻量表述，把这件事放成**方法动机和评估目标**，不要让它压过主方法。

A representation similarity score between subjects  $ i $  and  $ j $  can be defined as

$ S_{\text{rep}}(i,j)=\mathrm{CKA}(\Phi_i,\Phi_j), $

while a decision-consistency score  $ S_{\text{dec}}(i,j) $  can be computed from class-wise recalls, confusion patterns, or margin statistics.  
We then define a mismatch score

$ \Delta_{\text{RD}}(i,j) = \left| S_{\text{rep}}(i,j)-S_{\text{dec}}(i,j) \right|. $

Our hypothesis is that by suppressing high-risk over-alignment and enabling low-risk transfer, the proposed KCAR-guided gate reduces  $ \Delta_{\text{RD}} $ , thereby improving the coherence between learned representations and downstream decisions.

---

3.11 Summary of the Proposed Method

---

The proposed framework can be summarized as follows:

1. represent each EEG trial as an SPD covariance matrix;
2. encode the current alignment-relevant state using geometry, uncertainty, and short-term variability;
3. model state evolution in a Koopman-lifted space;
4. estimate whether source dynamics can explain current target evolution via KCAR;
5. use KCAR to control the dynamic alignment strength  $ w_t $ ;
6. perform risk-aware partial Riemannian alignment before CSP-based decoding.

In contrast to fixed alignment strategies and heuristic online feedback, the proposed method turns “when to align” into a principled dynamic inference problem.

