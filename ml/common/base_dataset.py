"""Abstract base class for all datasets in the building analyzer system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset, ABC):
    """Common interface for every dataset in the building analyzer system.

    Concrete implementations must supply :meth:`__len__` and
    :meth:`__getitem__` as normal PyTorch datasets, plus :meth:`num_classes`
    and :meth:`class_names` properties so that downstream trainers and
    inferencers can query the label space without hard-coding it.
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        """
        Args:
            root: Root directory that contains images (and possibly annotations).
            split: Dataset split — one of ``"train"``, ``"val"``, or ``"test"``.
        """
        valid_splits = {"train", "val", "test"}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got '{split}'")
        self.root = Path(root)
        self.split = split

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of target classes (excluding background where applicable)."""

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Ordered list of class name strings, length == :attr:`num_classes`."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in this split."""

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return a single sample.

        The return type is task-specific (segmentation, detection, or
        classification), so it is intentionally left as ``Any``.  See
        concrete dataset implementations for the precise signature.
        """

    def build_dataloader(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool | None = None,
        **kwargs: Any,
    ) -> DataLoader:
        """Convenience factory that wraps *self* in a :class:`DataLoader`.

        Args:
            batch_size: Number of samples per batch.
            num_workers: Worker processes for data loading.
            shuffle: Whether to shuffle.  Defaults to ``True`` for the
                ``"train"`` split and ``False`` otherwise.
            **kwargs: Extra keyword arguments forwarded to :class:`DataLoader`.

        Returns:
            A configured :class:`DataLoader` for this dataset.
        """
        if shuffle is None:
            shuffle = self.split == "train"
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )
