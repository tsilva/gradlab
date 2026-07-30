from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, TemplateError, meta
from jinja2.sandbox import SandboxedEnvironment
from omegaconf import OmegaConf
import yaml
from yaml.constructor import ConstructorError


YAML_EXTENSIONS = {".yaml", ".yml"}
TEMPLATE_VARS_KEY = "template_vars"
RECIPE_TEMPLATE_VALUES: dict[str, Any] = {
    "campaign_id": "b-test",
    "seed": 123,
    "recipe_id": "candidate",
    "timestamp": "20260626T120000Z",
    "utc": "20260626T120000Z",
}
RECIPE_TEMPLATE_FIELDS = frozenset(RECIPE_TEMPLATE_VALUES)

_LEVEL_ID_RE = re.compile(r"^Level(?P<world>\d+)-(?P<level>\d+)$", re.IGNORECASE)
_TEMPLATE_ENV = SandboxedEnvironment(
    autoescape=False,
    undefined=StrictUndefined,
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    seen: dict[Any, yaml.Node] = {}
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen[key] = key_node
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ComposedDocument:
    document: dict[str, Any]
    sources: tuple[Path, ...]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    cfg = OmegaConf.merge(OmegaConf.create(dict(base)), OmegaConf.create(dict(override)))
    return _plain_dict(cfg)


def dotlist_to_mapping(overrides: Sequence[str], *, label: str = "overrides") -> dict[str, Any]:
    cleaned = [str(item).strip() for item in overrides if str(item).strip()]
    if not cleaned:
        return {}
    try:
        cfg = OmegaConf.from_dotlist(cleaned)
    except Exception as exc:
        raise ValueError(f"failed to parse {label}: {exc}") from exc
    return _plain_dict(cfg)


def apply_dotlist_overrides(
    document: Mapping[str, Any],
    overrides: Sequence[str],
    *,
    label: str = "overrides",
) -> dict[str, Any]:
    override_mapping = dotlist_to_mapping(overrides, label=label)
    if not override_mapping:
        return dict(document)
    return deep_merge(document, override_mapping)


def slugify_template_value(value: Any) -> str:
    chars: list[str] = []
    for char in str(value or "").strip().lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-")


def _concrete_template_source(value: Any) -> str:
    text = str(value or "").strip()
    return "" if "{" in text or "}" in text else text


def _environment_mapping_from_document(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for candidate in (document, document.get("goal")):
        if not isinstance(candidate, Mapping):
            continue
        train = candidate.get("train")
        if isinstance(train, Mapping) and isinstance(train.get("environment"), Mapping):
            return train["environment"]
        if isinstance(candidate.get("environment"), Mapping):
            return candidate["environment"]
    return None


def _environment_template_context_from_document(document: Mapping[str, Any]) -> dict[str, str]:
    environment = _environment_mapping_from_document(document)
    if not isinstance(environment, Mapping):
        return {}
    env_provider = _concrete_template_source(environment.get("env_provider"))
    env_config = environment.get("env_config")
    game = (
        _concrete_template_source(env_config.get("game")) if isinstance(env_config, Mapping) else ""
    )
    context = {}
    if env_provider:
        context["env_provider"] = env_provider
    if game:
        context["env_id"] = game
    return context


def template_context_from_path(
    path: Path, document: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Build stable template variables from a goal/recipe path and optional document."""

    resolved = path.resolve()
    goal_id = ""
    game = ""
    recipe_slug = ""
    if resolved.parent.name == "recipes":
        recipe_slug = resolved.stem
        goal_id = resolved.parent.parent.name
        game = (
            resolved.parent.parent.parent.name
            if resolved.parent.parent.parent.name != "goals"
            else ""
        )
    else:
        goal_id = resolved.parent.name
        game = resolved.parent.parent.name if resolved.parent.parent.name != "goals" else ""

    if isinstance(document, Mapping):
        environment_context = _environment_template_context_from_document(document)
        raw_goal = document.get("goal")
        if isinstance(raw_goal, Mapping):
            goal_id = (
                _concrete_template_source(raw_goal.get("goal_id"))
                or _concrete_template_source(raw_goal.get("goal"))
                or goal_id
            )
        elif isinstance(raw_goal, str) and raw_goal.strip():
            goal_id = _concrete_template_source(raw_goal) or goal_id
        goal_id = (
            _concrete_template_source(document.get("goal_id"))
            or _concrete_template_source(document.get("goal_slug"))
            or goal_id
        )
        recipe_slug = _concrete_template_source(document.get("recipe_id")) or recipe_slug

    game_slug = slugify_template_value(game)
    goal_slug = slugify_template_value(goal_id)
    level_match = _LEVEL_ID_RE.match(goal_id)
    level_short = (
        f"l{level_match.group('world')}{level_match.group('level')}" if level_match else goal_slug
    )
    return {
        key: value
        for key, value in {
            "env_id": environment_context.get("env_id", "")
            if isinstance(document, Mapping)
            else "",
            "env_provider": environment_context.get("env_provider", "")
            if isinstance(document, Mapping)
            else "",
            "game": game,
            "game_slug": game_slug,
            "goal_id": goal_id,
            "goal_slug": goal_id,
            "level_short": level_short,
            "level_tag": goal_slug,
            "slug": recipe_slug,
            "recipe_id": recipe_slug,
            "recipe_slug": recipe_slug,
        }.items()
        if value
    }


def _template_fields(value: str) -> set[str]:
    return set(meta.find_undeclared_variables(_TEMPLATE_ENV.parse(value)))


def _referenced_template_fields(value: Any) -> set[str]:
    if isinstance(value, str):
        return _template_fields(value)
    if isinstance(value, Mapping):
        return set().union(
            *(
                _referenced_template_fields(nested)
                for key, nested in value.items()
                if key != TEMPLATE_VARS_KEY
            ),
            set(),
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return set().union(*map(_referenced_template_fields, value), set())
    return set()


def _validate_template_var_usage(document: Mapping[str, Any], *, label: str) -> None:
    raw_vars = document.get(TEMPLATE_VARS_KEY)
    if not isinstance(raw_vars, Mapping):
        return
    declared = {key for key in raw_vars if isinstance(key, str)}
    used = declared & _referenced_template_fields(document)
    pending = list(used)
    while pending:
        key = pending.pop()
        dependencies = declared & _referenced_template_fields(raw_vars[key])
        new_dependencies = dependencies - used
        used.update(new_dependencies)
        pending.extend(new_dependencies)
    unused = sorted(declared - used)
    if unused:
        raise ValueError(f"{label}.{TEMPLATE_VARS_KEY} declares unused fields: {', '.join(unused)}")


def _template_vars_from_document(
    document: Mapping[str, Any],
    *,
    base_context: Mapping[str, Any],
    label: str,
) -> dict[str, str]:
    raw_vars = document.get(TEMPLATE_VARS_KEY)
    if raw_vars is None:
        return {}
    if not isinstance(raw_vars, Mapping):
        raise ValueError(f"{label}.{TEMPLATE_VARS_KEY} must be an object")
    rendered: dict[str, str] = {}
    for key, value in raw_vars.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label}.{TEMPLATE_VARS_KEY} keys must be non-empty strings")
        if not isinstance(value, str | int | float | bool):
            raise ValueError(f"{label}.{TEMPLATE_VARS_KEY}.{key} must be a scalar")
        text = str(value)
        if isinstance(value, str):
            text = _render_template_string(
                value,
                context={**base_context, **rendered},
                deferred_fields=frozenset(),
                label=f"{label}.{TEMPLATE_VARS_KEY}.{key}",
            )
        rendered[key] = text
    return rendered


def _render_template_string(
    value: str,
    *,
    context: Mapping[str, Any],
    deferred_fields: frozenset[str],
    label: str,
) -> str:
    try:
        fields = _template_fields(value)
        unknown = sorted(fields - set(context) - deferred_fields)
        if unknown:
            allowed = ", ".join(sorted({*context, *deferred_fields}))
            raise ValueError(
                f"{label} uses unknown template field {unknown[0]!r}; allowed: {allowed}"
            )
        deferred_context = {
            field: "{{ " + field + " }}"
            for field in fields & deferred_fields
        }
        return _TEMPLATE_ENV.from_string(value).render({**context, **deferred_context})
    except TemplateError as exc:
        raise ValueError(f"{label} is not a valid Jinja template: {exc}") from exc


def validate_template_string(
    value: str,
    *,
    allowed_values: Mapping[str, Any],
    required_fields: frozenset[str] = frozenset(),
    label: str,
) -> frozenset[str]:
    try:
        fields = frozenset(_template_fields(value))
    except TemplateError as exc:
        raise ValueError(f"{label} is not a valid Jinja template: {exc}") from exc
    unknown = sorted(fields - set(allowed_values))
    if unknown:
        raise ValueError(f"{label} uses unsupported template field(s): {', '.join(unknown)}")
    missing = sorted(required_fields - fields)
    if missing:
        raise ValueError(f"{label} must include template field(s): {', '.join(missing)}")
    _render_template_string(
        value,
        context=allowed_values,
        deferred_fields=frozenset(),
        label=label,
    )
    return fields


def _render_template_value(
    value: Any,
    *,
    context: Mapping[str, Any],
    deferred_fields_by_path: Mapping[tuple[str, ...], frozenset[str]],
    path: tuple[str, ...],
    label: str,
) -> Any:
    if isinstance(value, str):
        deferred_fields = deferred_fields_by_path.get(path, frozenset())
        return _render_template_string(
            value,
            context=context,
            deferred_fields=deferred_fields,
            label=label,
        )
    if isinstance(value, Mapping):
        return {
            key: _render_template_value(
                nested,
                context=context,
                deferred_fields_by_path=deferred_fields_by_path,
                path=(*path, str(key)),
                label=f"{label}.{key}",
            )
            for key, nested in value.items()
            if key != TEMPLATE_VARS_KEY
        }
    if isinstance(value, list):
        return [
            _render_template_value(
                item,
                context=context,
                deferred_fields_by_path=deferred_fields_by_path,
                path=(*path, str(index)),
                label=f"{label}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    return value


def render_template_vars(
    document: Mapping[str, Any],
    *,
    path: Path,
    label: str,
    extra_context: Mapping[str, Any] | None = None,
    deferred_fields_by_path: Mapping[tuple[str, ...], frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Render checked-in Jinja template variables and remove `template_vars`.

    This is intentionally stricter than OmegaConf interpolation: unknown fields fail
    unless the caller marks a specific document path as a deferred runtime template.
    """

    base_context = {
        **template_context_from_path(path, document),
        **{key: str(value) for key, value in (extra_context or {}).items()},
    }
    _validate_template_var_usage(document, label=label)
    template_vars = _template_vars_from_document(document, base_context=base_context, label=label)
    context = {**base_context, **template_vars}
    return _render_template_value(
        dict(document),
        context=context,
        deferred_fields_by_path=deferred_fields_by_path or {},
        path=(),
        label=label,
    )


def load_config_document(path: Path, *, default: Any = None) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in YAML_EXTENSIONS:
        loaded = yaml.load(text, Loader=_UniqueKeySafeLoader)
    else:
        loaded = json.loads(text)
    return default if loaded is None else loaded


def load_mapping_document(path: Path, *, label: str | None = None) -> dict[str, Any]:
    payload = load_config_document(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label or path} must contain a JSON/YAML object")
    return dict(payload)


def _plain_dict(value: Any) -> dict[str, Any]:
    payload = OmegaConf.to_container(value, resolve=False)
    if not isinstance(payload, Mapping):
        raise ValueError("composed config must contain a JSON/YAML object")
    return dict(payload)


def _resolve_default_path(default_path: str, *, base_dir: Path, config_root: Path) -> Path:
    is_absolute_default = default_path.startswith("/")
    path = default_path.lstrip("/")
    if path.endswith(".yaml") or path.endswith(".yml") or path.endswith(".json"):
        candidate = Path(path)
    else:
        candidate = Path(f"{path}.yaml")
    if not candidate.is_absolute():
        candidate = (config_root if is_absolute_default else base_dir) / candidate
    return candidate.resolve()


def _parse_default_reference(entry: Any) -> tuple[str, str]:
    if not isinstance(entry, str):
        raise ValueError(f"config default must be a string: {entry!r}")
    default_path, separator, package = entry.strip().partition("@")
    if not separator or not default_path or not package:
        raise ValueError(f"config default must declare an explicit path@package: {entry!r}")
    return default_path, package


def _package_document(document: Mapping[str, Any], package: str) -> dict[str, Any]:
    if package == "_global_":
        return dict(document)
    parts = package.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid config package: {package!r}")
    result: dict[str, Any] = dict(document)
    for part in reversed(parts):
        result = {part: result}
    return result


def _compose_mapping_document(
    path: Path,
    *,
    config_root: Path,
    stack: tuple[Path, ...] = (),
) -> ComposedDocument:
    resolved_path = path.resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"cyclic config defaults chain: {chain}")
    document = load_mapping_document(resolved_path, label=str(path))
    defaults = document.pop("defaults", ["_self_"])
    if not isinstance(defaults, Sequence) or isinstance(defaults, str | bytes):
        raise ValueError(f"{path}.defaults must be a sequence")
    entries = list(defaults)
    if entries.count("_self_") != 1:
        raise ValueError(f"{path}.defaults must contain exactly one _self_ entry")

    composed: dict[str, Any] = {}
    sources: list[Path] = []
    for entry in entries:
        if entry == "_self_":
            composed = deep_merge(composed, document)
            sources.append(resolved_path)
            continue
        default_path, package = _parse_default_reference(entry)
        source = _resolve_default_path(
            default_path,
            base_dir=resolved_path.parent,
            config_root=config_root,
        )
        if not source.is_file():
            raise FileNotFoundError(f"config default not found: {source}")
        child = _compose_mapping_document(
            source,
            config_root=config_root,
            stack=(*stack, resolved_path),
        )
        composed = deep_merge(
            composed,
            _package_document(child.document, package),
        )
        sources.extend(child.sources)
    return ComposedDocument(document=composed, sources=tuple(sources))


def load_composed_mapping(
    path: Path,
    *,
    cycle_label: str = "config",
    overrides: Sequence[str] = (),
) -> ComposedDocument:
    resolved_path = path.resolve()
    if resolved_path.suffix.lower() not in YAML_EXTENSIONS:
        document = load_mapping_document(resolved_path, label=str(path))
        return ComposedDocument(
            document=apply_dotlist_overrides(
                document,
                overrides,
                label=f"{cycle_label} overrides for {path}",
            ),
            sources=(resolved_path,),
        )
    try:
        composed = _compose_mapping_document(
            resolved_path,
            config_root=resolved_path.parent,
        )
    except Exception as exc:
        raise ValueError(f"failed to compose {cycle_label} config {path}: {exc}") from exc
    document = apply_dotlist_overrides(
        composed.document,
        overrides,
        label=f"{cycle_label} overrides for {path}",
    )
    return ComposedDocument(document=document, sources=composed.sources)
