"""Executor + Ray-client factories.

Three Curator executors are exposed, all Ray-backed:

* ``xenna`` (default) -- :class:`nemo_curator.backends.xenna.XennaExecutor`
  on Cosmos-Xenna. Streaming autoscaler, the production path.
* ``ray_actor_pool`` -- :class:`nemo_curator.backends.ray_actor_pool.RayActorPoolExecutor`.
  Useful when the Ray head node also serves models (Xenna refuses to
  co-schedule GPU stages with Ray Serve).
* ``ray_data`` -- :class:`nemo_curator.backends.ray_data.RayDataExecutor`.
  Transforms-as-Ray-Data path; best for single-stage vectorized fits
  (large reducers, fuzzy dedup) and when the pipeline fits in a batch.

For remote Ray clusters, :func:`init_ray` wraps ``ray.init(...)``
so the driver connects once before ``Pipeline.run``. ``None`` means
local single-node; ``"auto"`` discovers an existing local cluster;
``"ray://<head>:10001"`` is Ray Client mode.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

EXECUTOR_CHOICES = ("xenna", "ray_actor_pool", "ray_data")


def build_executor(cfg: Any):
    """Return a :class:`nemo_curator.backends.base.BaseExecutor` instance.

    ``cfg.executor.name`` selects the backend. Per-backend knobs live
    under ``cfg.executor.*`` and are forwarded verbatim as the
    executor's ``config=`` dict.
    """
    name = str(cfg.executor.name).lower()
    base_config = _base_executor_config(cfg)

    if name == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        return XennaExecutor(
            config={
                "execution_mode": str(cfg.executor.mode),
                "logging_interval": int(cfg.executor.logging_interval),
                "autoscale_interval_s": int(cfg.executor.autoscale_interval_s),
                "cpu_allocation_percentage": float(
                    cfg.executor.cpu_allocation_percentage
                ),
                "ignore_failures": bool(cfg.executor.ignore_failures),
                **base_config,
            },
            ignore_head_node=False,
        )
    if name == "ray_actor_pool":
        from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor

        return RayActorPoolExecutor(
            config=base_config,
            ignore_head_node=bool(cfg.executor.ignore_head_node),
        )
    if name == "ray_data":
        from nemo_curator.backends.ray_data import RayDataExecutor

        return RayDataExecutor(
            config=base_config,
            ignore_head_node=bool(cfg.executor.ignore_head_node),
        )
    raise ValueError(
        f"unknown executor: {name!r}; expected one of {EXECUTOR_CHOICES}"
    )


def _base_executor_config(cfg: Any) -> dict[str, Any]:
    """Executor-agnostic extras merged into every backend config."""
    overrides = cfg.get("stage_overrides", {}) or {}
    base: dict[str, Any] = {}
    if "executor" in overrides:
        base.update(dict(overrides.executor))
    return base


# ---------------------------------------------------------------- ray client


def init_ray(cfg: Any) -> bool:
    """Connect (or start) a Ray runtime from ``cfg.ray``. Returns ``True`` on init.

    When ``cfg.ray.address`` is unset and Ray is already initialised in
    this process (e.g. the executor itself started it), this is a
    no-op. Idempotent -- safe to call from ``__main__``.
    """
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "ray is not installed. Install `nemo-curator[text_cpu]` or add "
            "`ray` to the environment to use any Curator executor."
        ) from exc

    if ray.is_initialized():
        logger.info("ray already initialised; skipping init_ray")
        return False

    address = cfg.ray.get("address", None)
    runtime_env = dict(cfg.ray.get("runtime_env", {}) or {})
    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": bool(cfg.ray.get("ignore_reinit_error", True)),
    }
    if runtime_env:
        init_kwargs["runtime_env"] = runtime_env
    if address:
        init_kwargs["address"] = str(address)
        logger.info("connecting to ray cluster at %s", address)
    else:
        num_cpus = cfg.ray.get("num_cpus", None)
        num_gpus = cfg.ray.get("num_gpus", None)
        if num_cpus is not None:
            init_kwargs["num_cpus"] = int(num_cpus)
        if num_gpus is not None:
            init_kwargs["num_gpus"] = int(num_gpus)
        logger.info(
            "starting local ray runtime (num_cpus=%s num_gpus=%s)",
            num_cpus, num_gpus,
        )
    ray.init(**init_kwargs)
    return True


def shutdown_ray() -> None:
    """Tear down the Ray runtime if it is up. Safe to call from ``finally``."""
    try:
        import ray
    except ImportError:
        return
    if ray.is_initialized():
        ray.shutdown()


__all__ = [
    "EXECUTOR_CHOICES",
    "build_executor",
    "init_ray",
    "shutdown_ray",
]
