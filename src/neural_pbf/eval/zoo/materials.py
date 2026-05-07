"""MaterialZoo — a registry of named material presets with ID/OOD tagging."""
from __future__ import annotations

from dataclasses import dataclass

from neural_pbf.physics.material import MaterialConfig


@dataclass(frozen=True)
class ZooEntry:
    """One entry in the material zoo."""

    name: str
    mat_cfg: MaterialConfig
    in_distribution: bool


class MaterialZoo:
    """Registry of :class:`MaterialConfig` presets for benchmarking.

    Materials tagged ``in_distribution=True`` were seen during training;
    ``False`` marks OOD generalisation targets.
    """

    def __init__(self) -> None:
        self._registry: dict[str, ZooEntry] = {}

    def register(
        self,
        name: str,
        mat_cfg: MaterialConfig,
        in_distribution: bool = True,
    ) -> None:
        """Add or overwrite a zoo entry."""
        self._registry[name] = ZooEntry(
            name=name, mat_cfg=mat_cfg, in_distribution=in_distribution
        )

    def get(self, name: str) -> ZooEntry:
        """Retrieve an entry by name (raises :exc:`KeyError` if missing)."""
        return self._registry[name]

    def all(self) -> list[ZooEntry]:
        """All registered entries."""
        return list(self._registry.values())

    def id_materials(self) -> list[ZooEntry]:
        """In-distribution entries."""
        return [e for e in self._registry.values() if e.in_distribution]

    def ood_materials(self) -> list[ZooEntry]:
        """Out-of-distribution entries."""
        return [e for e in self._registry.values() if not e.in_distribution]

    @classmethod
    def default(cls) -> MaterialZoo:
        """Return a zoo pre-populated with SS316L (ID) and three OOD materials."""
        zoo = cls()
        zoo.register("ss316l", MaterialConfig.ss316l_preset(), in_distribution=True)
        zoo.register("ss304", MaterialConfig.ss304_preset(), in_distribution=True)
        zoo.register("ti64", MaterialConfig.ti64_preset(), in_distribution=False)
        zoo.register(
            "alsi10mg", MaterialConfig.alsi10mg_preset(), in_distribution=False
        )
        zoo.register("in718", MaterialConfig.in718_preset(), in_distribution=False)
        return zoo
