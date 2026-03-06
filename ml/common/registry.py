"""Model registry — a lightweight mechanism for registering and instantiating
ML models by name, enabling model swapping without changing caller code.

Usage::

    from ml.common.registry import ModelRegistry
    from ml.common.base_model import BaseSegmentationModel

    @ModelRegistry.register("my_seg_model")
    class MySegModel(BaseSegmentationModel):
        ...

    model = ModelRegistry.build("my_seg_model", num_classes=7, class_names=[...])
"""

from __future__ import annotations

from typing import Any, Type


class ModelRegistry:
    """Global registry mapping string keys to model classes.

    Models are registered with :meth:`register` (as a decorator) and
    instantiated with :meth:`build`.  Multiple registries can coexist by
    using different *namespace* arguments, but the default namespace
    (``"default"``) is sufficient for most use-cases.
    """

    _registry: dict[str, dict[str, type]] = {}

    @classmethod
    def register(cls, name: str, namespace: str = "default"):
        """Class decorator that registers a model under *name* in *namespace*.

        Args:
            name: Unique key used to look up the model.
            namespace: Logical group (e.g. ``"segmentation"``, ``"detection"``).

        Returns:
            The original class unchanged so the decorator is transparent.

        Raises:
            ValueError: If *name* is already registered in *namespace*.
        """

        def decorator(model_cls: type) -> type:
            ns = cls._registry.setdefault(namespace, {})
            if name in ns:
                raise ValueError(
                    f"Model '{name}' is already registered in namespace '{namespace}'. "
                    "Use a different name or explicitly remove the existing entry first."
                )
            ns[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def build(cls, name: str, namespace: str = "default", **kwargs: Any) -> Any:
        """Instantiate and return the model registered under *name*.

        Args:
            name: Key used at registration time.
            namespace: The namespace to look in.
            **kwargs: Constructor keyword arguments forwarded to the model class.

        Returns:
            A new instance of the registered model class.

        Raises:
            KeyError: If *name* is not registered in *namespace*.
        """
        ns = cls._registry.get(namespace, {})
        if name not in ns:
            available = list(ns.keys())
            raise KeyError(
                f"Model '{name}' not found in namespace '{namespace}'. "
                f"Available: {available}"
            )
        return ns[name](**kwargs)

    @classmethod
    def list_models(cls, namespace: str = "default") -> list[str]:
        """Return the list of registered model names in *namespace*."""
        return list(cls._registry.get(namespace, {}).keys())

    @classmethod
    def get_class(cls, name: str, namespace: str = "default") -> Type:
        """Return the model *class* (not an instance) for *name*.

        Useful when you need access to class methods or want to inspect the
        model structure before instantiation.
        """
        ns = cls._registry.get(namespace, {})
        if name not in ns:
            available = list(ns.keys())
            raise KeyError(
                f"Model '{name}' not found in namespace '{namespace}'. "
                f"Available: {available}"
            )
        return ns[name]

    @classmethod
    def clear(cls, namespace: str | None = None) -> None:
        """Remove registered models.  Primarily useful in tests.

        Args:
            namespace: If given, clear only that namespace; otherwise clear all.
        """
        if namespace is None:
            cls._registry.clear()
        else:
            cls._registry.pop(namespace, None)
