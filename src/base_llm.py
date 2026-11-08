from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

from messages import Message


class BaseLLM(ABC):
    """Common interface for every LLM provider implementation.

    Subclasses register themselves automatically using the ``provider`` class
    keyword. This lets callers create any implementation through a single
    entry point instead of importing and instantiating a concrete class:

        llm = BaseLLM.create("openai", model="...", api_key="...")

    A new implementation only needs to subclass ``BaseLLM`` with a provider
    name; scripts that rely on ``create``/``from_config`` do not change.
    """

    _implementations: ClassVar[dict[str, type["BaseLLM"]]] = {}

    def __init_subclass__(
        cls,
        *,
        provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        name = provider or cls.__name__.removeprefix("LLM")
        BaseLLM._implementations[name.lower()] = cls

    def __init__(
            self,
            model: str,
            api_key: str
        ):
        self.model: str = model
        self.api_key: str = api_key

    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> "BaseLLM":
        """Create the implementation registered for ``provider``."""
        try:
            implementation = cls._implementations[provider.lower()]
        except KeyError:
            available = ", ".join(sorted(cls._implementations)) or "none"
            raise ValueError(
                f"Unknown LLM provider {provider!r}. "
                f"Available providers: {available}."
            ) from None

        return implementation(**kwargs)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BaseLLM":
        """Create an implementation from a provider-agnostic configuration.

        The mapping must contain a ``provider`` key; the remaining keys are
        forwarded to that implementation's constructor.
        """
        options = dict(config)
        try:
            provider = options.pop("provider")
        except KeyError:
            raise ValueError(
                "LLM configuration must include a 'provider' key."
            ) from None

        return cls.create(provider, **options)

    @classmethod
    def available_providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._implementations))

    @abstractmethod
    def chat(
        self,
        messages,
        *args, 
        **kwargs
    ) -> Message:
        pass