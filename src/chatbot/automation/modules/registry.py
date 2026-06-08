from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from chatbot.automation.modules.base import FulfillmentMode, HookModule
from chatbot.automation.modules.core.orders import CoreOrdersModule
from chatbot.automation.modules.erpnext.quote import ErpNextQuoteModule
from chatbot.automation.modules.quickbooks.quote import QuickBooksQuoteModule
from chatbot.domain.models.hook import HookEvent

logger = logging.getLogger(__name__)

_BUILTIN: tuple[HookModule, ...] = (
    CoreOrdersModule(),
    ErpNextQuoteModule(),
    QuickBooksQuoteModule(),
)


def all_modules() -> tuple[HookModule, ...]:
    return _BUILTIN


def get_module(module_id: str) -> HookModule | None:
    for mod in _BUILTIN:
        if mod.id == module_id:
            return mod
    return None


def module_for_hook_type(hook_type: str) -> HookModule | None:
    for mod in _BUILTIN:
        if mod.matches(hook_type):
            return mod
    return None


class ModuleRegistry:
    def __init__(self, modules: tuple[HookModule, ...] | None = None) -> None:
        self._modules = modules or all_modules()

    def list_modules(self) -> tuple[HookModule, ...]:
        return self._modules

    def get(self, module_id: str) -> HookModule | None:
        for mod in self._modules:
            if mod.id == module_id:
                return mod
        return None

    def module_for_hook_type(self, hook_type: str) -> HookModule | None:
        for mod in self._modules:
            if mod.matches(hook_type):
                return mod
        return None

    def dispatch_worker(self, session: Session, hook: HookEvent) -> None:
        mod = self.module_for_hook_type(hook.type)
        if mod is None:
            raise ValueError(f"No handler for hook type: {hook.type}")
        if mod.fulfillment_mode != FulfillmentMode.WORKER:
            logger.info(
                "Skipping worker for hook type %s (module %s uses %s fulfillment)",
                hook.type,
                mod.id,
                mod.fulfillment_mode.value,
            )
            return
        mod.handle_worker(session, hook)


_default_registry = ModuleRegistry()


def get_registry() -> ModuleRegistry:
    return _default_registry


def dispatch_hook(session: Session, hook: HookEvent) -> None:
    get_registry().dispatch_worker(session, hook)


def enabled_modules_for_tenant(
    module_ids: list[str],
    *,
    active_integrations: set[str],
) -> tuple[HookModule, ...]:
    out: list[HookModule] = []
    for mod_id in module_ids:
        mod = get_module(mod_id)
        if mod is None:
            continue
        if mod.requires_integration and mod.requires_integration not in active_integrations:
            continue
        out.append(mod)
    return tuple(out)
