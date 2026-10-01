"""Benchmark Model Implementations for PD-MCI Tabular Deep Learning & Foundation Models.

This module provides unified, reproducible interfaces for:
1. LR: the v1.0 pipeline (LASSO 1SE selection -> unpenalized LR -> bootstrap uniform shrinkage)
2. SVM and Random Forest (classical machine learning)
3. XGBoost, LightGBM and CatBoost (GBDT)
4. RealMLP (NeurIPS 2024) and TabM (ICLR 2025) (tabular deep learning trained from scratch)
5. TabICLv2 (Tabular Foundation Model with Zero-shot In-Context Learning, 2026)
6. TabPFN v3.5 (Prior-Data Fitted Tabular Foundation Model, local gated checkpoint)
7. TabFM (Google Research Zero-shot Tabular Foundation Model, 2026)

Models trained from scratch take their hyperparameters as one ``params`` dict (fixed
settings from benchmark_config.json merged with the step 15 tuning result). Models
that use early stopping set ``uses_early_stopping = True``; the evaluation runner then
carves an inner early-stopping split out of the training data they receive, so the
rows being scored are never used for early stopping. No model uses class weighting.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
import copy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
from xgboost import XGBClassifier

# Tabular DL and Foundation Model Imports
try:
    from tabm import TabM
except ImportError:
    TabM = None

# pytabkit (RealMLP) is imported lazily in RealMLPBenchmarkModel.fit: it pulls in
# pytorch-lightning, which would otherwise load in every joblib worker of the LR bootstrap.
# pytorch-lightning logs several lines per fit.
for _logger_name in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

try:
    from tabicl import TabICLClassifier
except ImportError:
    TabICLClassifier = None

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import ModelSource, prepend_cache_path
except ImportError:
    TabPFNClassifier = None

try:
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch
except ImportError:
    TabFMClassifier = None
    tabfm_v1_0_0_pytorch = None


# =====================================================================
# Base Model Interface
# =====================================================================

class BaseBenchmarkModel(abc.ABC):
    """Abstract base class for all benchmark classifiers."""

    uses_early_stopping = False

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.is_fitted = False

    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> BaseBenchmarkModel:
        """Fit model on training data, optionally using validation set for early stopping."""
        pass

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class (PD-MCI, Y=1). Returns 1D array of shape (N,)."""
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels based on a probability threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# =====================================================================
# 1. Logistic Regression: v1.0 LASSO + shrinkage pipeline (primary comparator)
# =====================================================================

def _bootstrap_slope(seed: int, X: np.ndarray, y: np.ndarray) -> Optional[float]:
    from step_14_shrinkage import one_bootstrap

    result = one_bootstrap(0, seed, X, y)
    return None if result is None else result["slope_orig"]


class LassoShrunkLRModel(BaseBenchmarkModel):
    """
    The complete v1.0 LR algorithm, refit on whatever training data it receives:
    LASSO feature selection (10-fold CV, AUPRC, 1SE rule; step 12) -> unpenalized
    LR (step 13) -> bootstrap uniform shrinkage with re-estimated intercept (step 14).
    Reuses the step 14 functions, so a fit on the full training set reproduces the
    v1.0 shrunk model.
    """

    def __init__(
        self,
        name: str = "LR",
        random_state: int = 42,
        n_shrinkage_bootstrap: int = 500,
        n_jobs: int = -1,
    ):
        super().__init__(name=name, random_state=random_state)
        self.n_shrinkage_bootstrap = n_shrinkage_bootstrap
        self.n_jobs = n_jobs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> LassoShrunkLRModel:
        from step_14_shrinkage import fit_unpenalized, lasso_select, shrink_model

        self.mask_, _ = lasso_select(X_train, y_train)
        apparent = fit_unpenalized(X_train[:, self.mask_], y_train)

        # Same seed stream as step 14, so the full-training-set fit reproduces v1.0
        seeds = np.random.SeedSequence(self.random_state).generate_state(
            self.n_shrinkage_bootstrap
        )
        slopes = Parallel(n_jobs=self.n_jobs)(
            delayed(_bootstrap_slope)(int(seed), X_train, y_train) for seed in seeds
        )
        slopes = [s for s in slopes if s is not None]
        self.shrinkage_ = float(np.mean(slopes))
        self.intercept_, self.coef_ = shrink_model(
            X_train[:, self.mask_], y_train, apparent.coef_[0], self.shrinkage_
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from step_14_shrinkage import predict_prob

        return predict_prob(X[:, self.mask_], self.intercept_, self.coef_)


# =====================================================================
# 1b. Plain Logistic Regression on all inputs (optional supplementary model)
# =====================================================================

class LRBenchmarkModel(BaseBenchmarkModel):
    """Unpenalized / weakly-regularized Logistic Regression baseline."""

    def __init__(
        self,
        name: str = "LR",
        random_state: int = 42,
        C: float = 1e5,
        max_iter: int = 1000,
    ):
        super().__init__(name=name, random_state=random_state)
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> LRBenchmarkModel:
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# =====================================================================
# 2. Classical machine learning: SVM and Random Forest
# =====================================================================

class SVMBenchmarkModel(BaseBenchmarkModel):
    """
    Support vector machine. Probabilities come from Platt scaling, which scikit-learn
    fits with an internal 5-fold CV on the training rows it receives.
    """

    def __init__(self, name: str = "SVM", random_state: int = 42, params: Optional[dict] = None):
        super().__init__(name=name, random_state=random_state)
        self.params = dict(params or {})
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> SVMBenchmarkModel:
        self.model = SVC(probability=True, random_state=self.random_state, **self.params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class RFBenchmarkModel(BaseBenchmarkModel):
    """Random forest."""

    def __init__(self, name: str = "RF", random_state: int = 42, params: Optional[dict] = None):
        super().__init__(name=name, random_state=random_state)
        self.params = dict(params or {})
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> RFBenchmarkModel:
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **self.params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# =====================================================================
# 3. GBDT: XGBoost, LightGBM, CatBoost (early stopping on the inner split)
# =====================================================================

class XGBoostBenchmarkModel(BaseBenchmarkModel):
    """XGBoost; ``n_estimators`` is the maximum, early stopping picks the number of trees."""

    uses_early_stopping = True

    def __init__(self, name: str = "XGBoost", random_state: int = 42, params: Optional[dict] = None,
                 early_stopping_rounds: int = 50):
        super().__init__(name=name, random_state=random_state)
        self.params = dict(params or {})
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> XGBoostBenchmarkModel:
        has_val = X_val is not None and y_val is not None
        self.model = XGBClassifier(
            eval_metric="logloss",
            early_stopping_rounds=self.early_stopping_rounds if has_val else None,
            random_state=self.random_state,
            **self.params,
        )
        if has_val:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class LightGBMBenchmarkModel(BaseBenchmarkModel):
    """LightGBM; ``n_estimators`` is the maximum, early stopping picks the number of trees."""

    uses_early_stopping = True

    def __init__(self, name: str = "LightGBM", random_state: int = 42, params: Optional[dict] = None,
                 early_stopping_rounds: int = 50):
        super().__init__(name=name, random_state=random_state)
        self.params = dict(params or {})
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> LightGBMBenchmarkModel:
        self.model = LGBMClassifier(
            random_state=self.random_state,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
            **self.params,
        )
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[lgb_early_stopping(self.early_stopping_rounds, verbose=False)],
            )
        else:
            self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class CatBoostBenchmarkModel(BaseBenchmarkModel):
    """
    CatBoost; ``iterations`` is the maximum, early stopping picks the number of trees.
    Without an early-stopping split no eval_set is passed, so CatBoost's default
    use_best_model cannot select trees on scored rows.
    """

    uses_early_stopping = True

    def __init__(self, name: str = "CatBoost", random_state: int = 42, params: Optional[dict] = None,
                 early_stopping_rounds: int = 50):
        super().__init__(name=name, random_state=random_state)
        self.params = dict(params or {})
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> CatBoostBenchmarkModel:
        self.model = CatBoostClassifier(
            random_seed=self.random_state,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
            **self.params,
        )
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# =====================================================================
# 4a. RealMLP (pytabkit)
# =====================================================================

class RealMLPBenchmarkModel(BaseBenchmarkModel):
    """
    RealMLP (Holzmueller et al., NeurIPS 2024) via pytabkit. RealMLP trains for a fixed
    number of epochs and keeps the epoch with the best validation loss; the validation
    rows are the inner early-stopping split, never the rows being scored.
    """

    uses_early_stopping = True

    def __init__(self, name: str = "RealMLP", random_state: int = 42, params: Optional[dict] = None,
                 device: str = "cpu"):
        super().__init__(name=name, random_state=random_state)
        self.params = dict(params or {})
        self.device = device
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> RealMLPBenchmarkModel:
        from pytabkit import RealMLP_TD_Classifier

        params = dict(self.params)
        if isinstance(params.get("hidden_sizes"), str):
            # Stored as e.g. "256x3" so the tuning space can use it as a categorical value
            width, depth = params["hidden_sizes"].split("x")
            params["hidden_sizes"] = [int(width)] * int(depth)
        self.model = RealMLP_TD_Classifier(
            device=self.device,
            random_state=self.random_state,
            val_metric_name="cross_entropy",
            verbosity=0,
            **params,
        )
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# =====================================================================
# 4b. TabM (Task-Specific Tabular Deep Learning with BatchEnsemble)
# =====================================================================


class _SingleTabMNetwork:
    """Helper class for training a single TabM BatchEnsemble instance."""

    def __init__(
        self,
        n_features: int,
        k: int = 32,
        d_block: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 60,
        patience: int = 15,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.n_features = n_features
        self.k = k
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.best_state_dict = None

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = TabM(
            n_num_features=n_features,
            d_out=1,
            arch_type="tabm",
            k=k,
            d_block=d_block,
            n_blocks=n_blocks,
            dropout=dropout,
            start_scaling_init="random-signs",
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        device = self.device
        self.model.to(device)

        X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_tr = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(-1)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
            y_v = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(-1)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        n_samples = len(X_train)
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                bx = X_tr[batch_idx]
                by = y_tr[batch_idx]

                optimizer.zero_grad()
                out = self.model(bx)  # (batch_size, k, 1)
                # Expand target across ensemble members
                target = by.unsqueeze(1).expand(-1, self.k, -1)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation evaluation for early stopping
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_v)
                    val_target = y_v.unsqueeze(1).expand(-1, self.k, -1)
                    val_loss = criterion(val_out, val_target).item()

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            else:
                self.best_state_dict = copy.deepcopy(self.model.state_dict())

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        self.model.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            out = self.model(X_tensor)  # (batch_size, k, 1)
            # Sigmoid on logits, then average across k ensemble submodels
            probs = torch.sigmoid(out.squeeze(-1)).mean(dim=1).cpu().numpy()
        return probs


class TabMBenchmarkModel(BaseBenchmarkModel):
    """TabM model with multi-seed ensembling (default 5 seeds as recommended in review)."""

    uses_early_stopping = True

    def __init__(
        self,
        name: str = "TabM",
        random_state: int = 42,
        k: int = 32,
        d_block: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 60,
        patience: int = 15,
        seeds: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        super().__init__(name=name, random_state=random_state)
        self.device = device
        self.k = k
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seeds = seeds if seeds is not None else [random_state + i for i in range(5)]
        self.submodels: List[_SingleTabMNetwork] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TabMBenchmarkModel:
        if TabM is None:
            raise ImportError("TabM is not installed. Please install tabm via pip.")

        self.submodels = []
        n_features = X_train.shape[1]

        for seed in self.seeds:
            net = _SingleTabMNetwork(
                n_features=n_features,
                k=self.k,
                d_block=self.d_block,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.patience,
                random_state=seed,
                device=self.device,
            )
            net.fit(X_train, y_train, X_val, y_val)
            self.submodels.append(net)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or not self.submodels:
            raise RuntimeError("TabM model is not fitted yet.")

        # Average predicted probabilities across all random seeds
        seed_predictions = [net.predict_proba(X) for net in self.submodels]
        return np.mean(seed_predictions, axis=0)


# =====================================================================
# 4. TabICLv2 (Tabular Foundation Model - Zero-Shot ICL)
# =====================================================================

class TabICLBenchmarkModel(BaseBenchmarkModel):
    """TabICLv2 zero-shot in-context learning classifier with local HF checkpoint."""

    def __init__(
        self,
        name: str = "TabICLv2",
        random_state: int = 42,
        n_estimators: int = 8,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
        device: str = "cpu",
    ):
        super().__init__(name=name, random_state=random_state)
        self.device = device
        self.n_estimators = n_estimators
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.checkpoint_version = checkpoint_version
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TabICLBenchmarkModel:
        if TabICLClassifier is None:
            raise ImportError("TabICL is not installed. Please install tabicl via pip.")

        self.model = TabICLClassifier(
            n_estimators=self.n_estimators,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=self.class_shuffle_method,
            checkpoint_version=self.checkpoint_version,
            random_state=self.random_state,
            device=self.device,
            verbose=False,
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TabICL model is not fitted.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]


# =====================================================================
# 5. TabPFN (Tabular Foundation Model)
# =====================================================================

def tabpfn_checkpoint_path(model_version: str = "v2.5") -> str:
    """Local cache path of the default classifier checkpoint for a TabPFN version."""
    if TabPFNClassifier is None:
        raise ImportError("TabPFN is not installed. Please install tabpfn via pip.")
    sources = {
        "v2": ModelSource.get_classifier_v2,
        "v2.5": ModelSource.get_classifier_v2_5,
        "v3.5": ModelSource.get_v3_5,
    }
    return str(prepend_cache_path(sources[model_version]().default_filename))


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TabPFNBenchmarkModel(BaseBenchmarkModel):
    """
    TabPFN classifier pinned to one model version (v3.5 in the benchmark) and loaded
    only from the local checkpoint cache. The weights are gated on
    Hugging Face, so they must be downloaded manually after accepting the licence;
    this class never downloads and fails if the file is missing or its SHA-256
    differs from the configured value.
    """

    def __init__(
        self,
        name: str = "TabPFN",
        random_state: int = 42,
        model_version: str = "v2.5",
        checkpoint_sha256: Optional[str] = None,
        n_estimators: int = 8,
        balance_probabilities: bool = False,
        device: str = "cpu",
    ):
        super().__init__(name=name, random_state=random_state)
        self.model_version = model_version
        self.checkpoint_sha256 = checkpoint_sha256
        self.n_estimators = n_estimators
        self.balance_probabilities = balance_probabilities
        self.device = device
        self.model = None

    def _verified_checkpoint(self) -> str:
        path = tabpfn_checkpoint_path(self.model_version)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"TabPFN {self.model_version} checkpoint not found at {path}. Accept the "
                "licence on Hugging Face and download it manually; the benchmark does not "
                "download model weights."
            )
        if self.checkpoint_sha256 and file_sha256(path) != self.checkpoint_sha256:
            raise RuntimeError(f"TabPFN checkpoint hash mismatch: {path}")
        return path

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TabPFNBenchmarkModel:
        self._verified_checkpoint()

        # TabPFN refuses >1,000 training rows on CPU unless explicitly allowed
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
        self.model = TabPFNClassifier.create_default_for_version(
            ModelVersion(self.model_version),
            n_estimators=self.n_estimators,
            balance_probabilities=self.balance_probabilities,
            ignore_pretraining_limits=True,
            random_state=self.random_state,
            device=self.device,
            show_progress_bar=False,
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TabPFN model is not fitted.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]


# =====================================================================
# 6. TabFM (Google Research Zero-Shot Tabular Foundation Model)
# =====================================================================

class TabFMBenchmarkModel(BaseBenchmarkModel):
    """TabFM zero-shot classifier using pre-trained google/tabfm-1.0.0-pytorch."""

    _cached_tabfm_model = None

    def __init__(
        self,
        name: str = "TabFM",
        random_state: int = 42,
        n_estimators: int = 8,
        max_num_rows: Optional[int] = None,
        norm_methods: Optional[str] = "quantile",
        device: str = "cpu",
    ):
        super().__init__(name=name, random_state=random_state)
        self.n_estimators = n_estimators
        # None = use every training row as in-context examples (no context subsampling)
        self.max_num_rows = max_num_rows
        self.norm_methods = norm_methods
        self.device = device
        self.model = None

    @classmethod
    def get_or_load_weights(cls, device: str = "cpu"):
        """Load and cache TabFM PyTorch weights."""
        if cls._cached_tabfm_model is None:
            if tabfm_v1_0_0_pytorch is None:
                raise ImportError("TabFM is not installed. Please install tabfm via pip.")
            cls._cached_tabfm_model = tabfm_v1_0_0_pytorch.load(
                model_type="classification",
                device=device,
                dtype=torch.float32,
            )
        return cls._cached_tabfm_model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TabFMBenchmarkModel:
        if TabFMClassifier is None:
            raise ImportError("TabFM is not installed. Please install tabfm via pip.")

        pretrained_model = self.get_or_load_weights(self.device)
        self.model = TabFMClassifier(
            model=pretrained_model,
            n_estimators=self.n_estimators,
            max_num_rows=self.max_num_rows,
            norm_methods=self.norm_methods,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TabFM model is not fitted.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]


# =====================================================================
# Factory Function
# =====================================================================

TUNED_MODELS = ["SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "RealMLP", "TabM"]


def model_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Hyperparameters of a tuned model: fixed settings overridden by the tuning result."""
    if "fixed_params" not in cfg:
        raise KeyError("Tuned models need 'fixed_params' in their configuration")
    return {**cfg["fixed_params"], **cfg.get("tuned_params", {})}


def create_benchmark_model(
    model_key: str,
    config: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> BaseBenchmarkModel:
    """Factory function to instantiate models by name with configuration."""
    cfg = config or {}
    key = model_key.upper()
    device = cfg.get("device", "cpu")

    if key in ["LR", "LOGISTIC_REGRESSION"]:
        return LassoShrunkLRModel(
            name="LR",
            random_state=random_state,
            n_shrinkage_bootstrap=cfg.get("n_shrinkage_bootstrap", 500),
        )
    elif key in ["LR_PLAIN"]:
        return LRBenchmarkModel(
            name="LR_plain",
            random_state=random_state,
            C=cfg.get("C", 1e5),
            max_iter=cfg.get("max_iter", 1000),
        )
    elif key == "SVM":
        return SVMBenchmarkModel(random_state=random_state, params=model_params(cfg))
    elif key == "RF":
        return RFBenchmarkModel(random_state=random_state, params=model_params(cfg))
    elif key == "XGBOOST":
        return XGBoostBenchmarkModel(
            random_state=random_state, params=model_params(cfg),
            early_stopping_rounds=cfg["early_stopping_rounds"],
        )
    elif key == "LIGHTGBM":
        return LightGBMBenchmarkModel(
            random_state=random_state, params=model_params(cfg),
            early_stopping_rounds=cfg["early_stopping_rounds"],
        )
    elif key == "CATBOOST":
        return CatBoostBenchmarkModel(
            random_state=random_state, params=model_params(cfg),
            early_stopping_rounds=cfg["early_stopping_rounds"],
        )
    elif key == "REALMLP":
        return RealMLPBenchmarkModel(random_state=random_state, params=model_params(cfg), device=device)
    elif key == "TABM":
        p = model_params(cfg)
        return TabMBenchmarkModel(
            name="TabM",
            random_state=random_state,
            k=p["k"],
            d_block=p["d_block"],
            n_blocks=p["n_blocks"],
            dropout=p["dropout"],
            learning_rate=p["learning_rate"],
            weight_decay=p["weight_decay"],
            batch_size=p["batch_size"],
            max_epochs=p["max_epochs"],
            patience=p["patience"],
            seeds=p["seeds"],
            device=device,
        )
    elif key in ["TABICL", "TABICLV2"]:
        return TabICLBenchmarkModel(
            name="TabICLv2",
            random_state=random_state,
            n_estimators=cfg.get("n_estimators", 8),
            feat_shuffle_method=cfg.get("feat_shuffle_method", "latin"),
            class_shuffle_method=cfg.get("class_shuffle_method", "shift"),
            checkpoint_version=cfg.get("checkpoint", "tabicl-classifier-v2-20260212.ckpt"),
            device=device,
        )
    elif key.startswith("TABPFN"):
        return TabPFNBenchmarkModel(
            name=model_key,
            random_state=random_state,
            model_version=cfg.get("model_version", "v2.5"),
            checkpoint_sha256=cfg.get("checkpoint_sha256"),
            n_estimators=cfg.get("n_estimators", 8),
            balance_probabilities=cfg.get("balance_probabilities", False),
            device=device,
        )
    elif key in ["TABFM"]:
        return TabFMBenchmarkModel(
            name="TabFM",
            random_state=random_state,
            n_estimators=cfg.get("n_estimators", 8),
            max_num_rows=cfg.get("max_num_rows", None),
            norm_methods=cfg.get("norm_methods", "quantile"),
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown model key: {model_key}. Available models: "
            "['LR', 'SVM', 'RF', 'XGBoost', 'LightGBM', 'CatBoost', 'RealMLP', 'TabM', "
            "'TabICLv2', 'TabPFN', 'TabFM', 'LR_plain']"
        )



# LightGBM 4.6 records Column_0..Column_n as feature_names_in_ even when fitted on a
# NumPy array, so scikit-learn >= 1.6 warns on every NumPy predict_proba call. Columns are
# matched by position either way. Kept at the end of the module so that the source line
# numbers of the model classes (hashed into the step 15 tuning fingerprint) do not shift.
import warnings  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
