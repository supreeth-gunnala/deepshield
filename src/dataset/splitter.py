"""
DeepShield Generator-Disjoint Dataset Splitter Module
Implements generator-disjoint train/val/test splits to evaluate unseen generator generalization.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("DeepShield.DatasetSplitter")

class GeneratorDisjointSplitter:
    """Enforces generator-disjoint data splitting to prevent synthetic generator leakage."""

    def __init__(
        self,
        train_generators: list[str] = None,
        val_generators: list[str] = None,
        unseen_test_generators: list[str] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.20,
        random_seed: int = 42
    ):
        self.train_generators = train_generators or ["generator_A", "generator_B"]
        self.val_generators = val_generators or ["generator_A", "generator_B"]
        self.unseen_test_generators = unseen_test_generators or ["generator_C"]
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def apply_generator_disjoint_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies generator-disjoint or stratified train/val/test splitting:
        - Fake audio from unseen_test_generators -> strictly assigned to 'test'.
        - Fake audio from known/general generators -> split proportionally across 'train', 'val', and 'test'.
        - Real human audio -> split proportionally across 'train', 'val', and 'test'.
        - Background noise samples -> assigned to 'background_noise'.
        """
        if df.empty:
            return df

        df = df.copy()
        df["split"] = "unassigned"

        # 1. Isolate Background Noise Samples
        bg_mask = df["label"] == "background_noise"
        df.loc[bg_mask, "split"] = "background_noise"

        # 2. Isolate Real Human Audio vs Fake Audio
        real_mask = df["label"] == "real"
        fake_mask = df["label"] == "fake"

        # 3. Handle Unseen Generator Fake samples if configured
        if self.unseen_test_generators:
            unseen_fake_mask = fake_mask & df["generator"].isin(self.unseen_test_generators)
            df.loc[unseen_fake_mask, "split"] = "test"
            known_fake_mask = fake_mask & (~df["generator"].isin(self.unseen_test_generators))
        else:
            known_fake_mask = fake_mask

        # 4. Split Known/General Fake Audio across train, val, test
        fake_indices = df[known_fake_mask].index
        if len(fake_indices) > 0:
            train_val_fake, test_fake_idx = train_test_split(
                fake_indices,
                test_size=self.test_ratio,
                random_state=self.random_seed
            )
            val_rel_ratio = self.val_ratio / (1.0 - self.test_ratio)
            train_fake_idx, val_fake_idx = train_test_split(
                train_val_fake,
                test_size=val_rel_ratio,
                random_state=self.random_seed
            )
            df.loc[train_fake_idx, "split"] = "train"
            df.loc[val_fake_idx, "split"] = "val"
            df.loc[test_fake_idx, "split"] = "test"

        # 5. Split Real Human Audio proportionally across train, val, and test
        real_indices = df[real_mask].index
        if len(real_indices) > 0:
            train_val_real, test_real_idx = train_test_split(
                real_indices,
                test_size=self.test_ratio,
                random_state=self.random_seed
            )
            val_rel_ratio = self.val_ratio / (1.0 - self.test_ratio)
            train_real_idx, val_real_idx = train_test_split(
                train_val_real,
                test_size=val_rel_ratio,
                random_state=self.random_seed
            )
            df.loc[train_real_idx, "split"] = "train"
            df.loc[val_real_idx, "split"] = "val"
            df.loc[test_real_idx, "split"] = "test"

        # 6. Handle any remaining unassigned records
        unassigned_mask = df["split"] == "unassigned"
        df.loc[unassigned_mask, "split"] = "train"

        return df

    def verify_no_leakage(self, df: pd.DataFrame) -> dict:
        """
        Verifies that unseen test generators do not appear in train or validation sets.
        """
        train_gen = set(df[df["split"] == "train"]["generator"].unique()) - {"human"}
        val_gen = set(df[df["split"] == "val"]["generator"].unique()) - {"human"}
        test_gen = set(df[df["split"] == "test"]["generator"].unique()) - {"human"}

        unseen_set = set(self.unseen_test_generators)
        train_overlap = train_gen.intersection(unseen_set)
        val_overlap = val_gen.intersection(unseen_set)

        duplicate_paths = df["file_path"].duplicated().sum()
        duplicate_filenames = df["filename"].duplicated().sum()

        is_disjoint = len(train_overlap) == 0 and len(val_overlap) == 0

        verification = {
            "is_disjoint": is_disjoint,
            "train_generators": list(train_gen),
            "val_generators": list(val_gen),
            "test_generators": list(test_gen),
            "train_unseen_overlap": list(train_overlap),
            "val_unseen_overlap": list(val_overlap),
            "duplicate_file_paths": int(duplicate_paths),
            "duplicate_filenames": int(duplicate_filenames)
        }
        return verification
