from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import TARGET_NAMES


@dataclass
class PreparedData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_scale: float
    image_size: int

    @property
    def tabular_dim(self) -> int:
        return len(self.feature_columns)


def load_biomass_frame(data_dir: str | Path) -> pd.DataFrame:
    """Load the long CSV and return one supervised row per pasture image."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "train.csv"
    images_dir = data_dir / "Images"

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Expected {images_dir}")

    long_df = pd.read_csv(csv_path)
    long_df["target"] = pd.to_numeric(long_df["target"], errors="coerce")

    meta_cols = [
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]
    meta = long_df[meta_cols].drop_duplicates("image_path").set_index("image_path")
    targets = long_df.pivot_table(
        index="image_path",
        columns="target_name",
        values="target",
        aggfunc="first",
    )

    missing_targets = sorted(set(TARGET_NAMES).difference(targets.columns))
    if missing_targets:
        raise ValueError(f"Missing target columns in train.csv: {missing_targets}")

    frame = meta.join(targets[TARGET_NAMES]).reset_index()
    frame["image_id"] = frame["image_path"].map(lambda value: Path(str(value)).name)
    frame["image_file"] = frame["image_id"].map(lambda name: str(images_dir / name))

    missing_images = [p for p in frame["image_file"] if not Path(p).exists()]
    if missing_images:
        preview = ", ".join(Path(p).name for p in missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} images referenced by CSV are missing: {preview}")

    frame = _add_date_features(frame)
    frame = _add_tabular_features(frame)
    return frame


def prepare_data(
    data_dir: str | Path,
    image_size: int = 128,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    limit_samples: int | None = None,
) -> PreparedData:
    """Create deterministic train/validation/test splits and scale tabular inputs."""
    frame = load_biomass_frame(data_dir)
    if limit_samples is not None:
        limit_samples = min(limit_samples, len(frame))
        frame = frame.sample(n=limit_samples, random_state=seed).reset_index(drop=True)

    tabular_cols = [c for c in frame.columns if c.startswith("feat_")]

    stratify_by = frame["State"] if (frame["State"].value_counts().min() >= 3) else None

    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_by,
    )
    relative_val = val_size / (1.0 - test_size)
    
    stratify_val = train_val["State"] if (train_val["State"].value_counts().min() >= 3) else None
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        shuffle=True,
        stratify=stratify_val,
    )

    scaler = StandardScaler()
    scaler.fit(train[tabular_cols].astype("float32"))

    scaled_cols = [f"scaled_{c}" for c in tabular_cols]
    splits = []
    for split in (train, val, test):
        split = split.copy()
        split[scaled_cols] = scaler.transform(split[tabular_cols].astype("float32"))
        splits.append(split.reset_index(drop=True))
    train, val, test = splits

    target_scale = float(np.nanmax(train[TARGET_NAMES].to_numpy(dtype="float32")))
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Could not compute a positive target scale from the training split.")

    return PreparedData(
        train=train,
        val=val,
        test=test,
        feature_columns=scaled_cols,
        target_scale=target_scale,
        image_size=image_size,
    )


def make_dataset(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    target_scale: float,
    image_size: int,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    cache: bool = False,
    for_keras: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from a prepared split."""
    paths = frame["image_file"].astype(str).to_numpy()
    tabular = frame[list(feature_columns)].to_numpy(dtype="float32")
    targets = frame[TARGET_NAMES].to_numpy(dtype="float32") / np.float32(target_scale)

    ds = tf.data.Dataset.from_tensor_slices((paths, tabular, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(frame), seed=seed, reshuffle_each_iteration=True)

    def load_item(path: tf.Tensor, tab: tf.Tensor, target: tf.Tensor):
        image = load_image(path, image_size=image_size, augment=augment)
        if for_keras:
            return (image, tab), target
        return image, tab, target

    ds = ds.map(load_item, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_prediction_dataset(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    image_size: int,
    batch_size: int,
    cache: bool = False,
) -> tf.data.Dataset:
    paths = frame["image_file"].astype(str).to_numpy()
    tabular = frame[list(feature_columns)].to_numpy(dtype="float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, tabular))

    def load_inputs(path: tf.Tensor, tab: tf.Tensor):
        return {
            "image": load_image(path, image_size=image_size, augment=False),
            "tabular": tab,
        }

    ds = ds.map(load_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def target_array(frame: pd.DataFrame, target_scale: float | None = None) -> np.ndarray:
    values = frame[TARGET_NAMES].to_numpy(dtype="float32")
    if target_scale is not None:
        values = values / np.float32(target_scale)
    return values


def _add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    dates = pd.to_datetime(frame["Sampling_Date"], errors="coerce")
    day = dates.dt.dayofyear.fillna(dates.dt.dayofyear.median()).astype("float32")
    year = dates.dt.year.fillna(dates.dt.year.median()).astype("float32")

    frame["feat_day_sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["feat_day_cos"] = np.cos(2 * np.pi * day / 365.25)
    frame["feat_year"] = year
    return frame


def _add_tabular_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ("Pre_GSHH_NDVI", "Height_Ave_cm"):
        values = pd.to_numeric(frame[col], errors="coerce")
        frame[f"feat_{col}"] = values.fillna(values.median()).astype("float32")

    one_hot = pd.get_dummies(
        frame[["State", "Species"]].fillna("Unknown"),
        prefix=["feat_state", "feat_species"],
        dtype="float32",
    )
    return pd.concat([frame, one_hot], axis=1)


def load_image(path: tf.Tensor, image_size: int, augment: bool = False) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (image_size, image_size), antialias=True)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image
