# llm-wrapper

A small, provider-agnostic wrapper around LLM APIs considered to be used for internal projects.

The project exposes a single interface for chatting with different LLM
providers, a runtime factory for creating implementations by name, structured
output support through Pydantic, tool-call parsing, and a manual implementation
that lets you act as the LLM while developing.

## Features

- Common `BaseLLM` interface shared by every provider.
- Self-registering implementations: adding a provider does not require changing
  the scripts that create it.
- Provider creation by name or from a configuration mapping.
- OpenAI and OpenAI-compatible API implementation.
- Interactive `manual` implementation for development and prompt inspection.
- Pydantic-based structured output.
- Tool-call parsing into a provider-independent model.

## Project structure

```text
llm-wrapper/
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebook/            # Jupyter playground driven by .env
│   ├── llm_playground.ipynb
│   ├── README.md
│   ├── .env.example
│   └── requirements.txt
└── llm_wrapper/
    ├── __init__.py
    └── src/
        ├── base_llm.py        # Abstract interface + provider registry/factory
        ├── implementations.py # Built-in providers (OpenAI, manual) + create_llm()
        └── messages.py        # Message, Roles and ToolCall models
```

## Installation

`llm-wrapper` is a Python package, so you can install it into any project with
`pip` directly from the repository:

```bash
pip install "llm-wrapper @ git+https://github.com/fernanh98/llm-wrapper.git"
```

To pin a specific commit, tag, or branch for reproducible installs:

```bash
pip install "llm-wrapper @ git+https://github.com/fernanh98/llm-wrapper.git@<tag-or-commit>"
```

If you are working on the package itself, clone it and install it in editable
mode so your changes take effect without reinstalling:

```bash
git clone https://github.com/fernanh98/llm-wrapper.git
cd llm-wrapper
pip install -e .
```

You can also add it as a dependency of another project. In `requirements.txt`:

```text
llm-wrapper @ git+https://github.com/fernanh98/llm-wrapper.git
```

Or in `pyproject.toml`:

```toml
[project]
dependencies = [
    "llm-wrapper @ git+https://github.com/fernanh98/llm-wrapper.git",
]
```

After installing, import the package from anywhere without setting
`PYTHONPATH`:

```python
from llm_wrapper import create_llm
```

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Import path

The implementation modules live in `llm_wrapper/src/`, but the package exposes
them at its root, so no path configuration is needed when running from the
repository root:

```python
import llm_wrapper.implementations
from llm_wrapper.base_llm import BaseLLM
```

Once the package is installed (see [Installation](#installation)), those
imports work from any directory. If the package is not installed and your
script runs from another directory, add the repository root to the import path:

```bash
PYTHONPATH=/path/to/llm-wrapper python your_script.py
```

## Quick start

```python
from llm_wrapper.implementations import create_llm

llm = create_llm(
    "openai",
    model="gpt-4.1-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

response = llm.chat(
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Say hello in Spanish."},
    ]
)

print(response.content)
```

`base_url` can point to any OpenAI-compatible endpoint, such as a local model
server:

```python
llm = create_llm(
    "openai",
    model="local-model",
    base_url="http://localhost:1234/v1",
    api_key="none",
)
```

## Notebook playground

`notebook/llm_playground.ipynb` exercises every registered provider — plain
chat, structured output, tool calls, the interactive `manual` provider, and
`from_config()` — and reads its configuration from `notebook/.env`.

```bash
pip install -r notebook/requirements.txt
cp notebook/.env.example notebook/.env   # then add your keys
jupyter lab notebook/llm_playground.ipynb
```

See [notebook/README.md](notebook/README.md) for the available environment
variables and safety notes.

## Creating implementations

All concrete implementations subclass `BaseLLM` and register themselves with a
provider name:

```python
class LLMOpenai(BaseLLM, provider="openai"):
    ...
```

Registration happens when the class is defined, which happens when its module
is imported. `implementations.py` is the central module that defines/imports the
built-in providers, so importing anything from it also populates the registry.

The registry can be used in several ways.

### By provider name with `create_llm`

```python
from llm_wrapper.implementations import create_llm

llm = create_llm(
    "openai",
    model="gpt-4.1-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)
```

This is the recommended entry point for scripts: the import stays the same when
you switch providers.

### Directly through `BaseLLM`

```python
from llm_wrapper.base_llm import BaseLLM
import llm_wrapper.implementations  # Registers the built-in providers

llm = BaseLLM.create(
    "openai",
    model="gpt-4.1-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)
```

### From a configuration mapping

`BaseLLM.from_config()` accepts a provider-agnostic mapping. The `provider` key
selects the implementation; every other key is forwarded to its constructor.

```python
from llm_wrapper.base_llm import BaseLLM
import llm_wrapper.implementations  # Registers the built-in providers

config = {
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
}

llm = BaseLLM.from_config(config)
```

This makes it possible to switch providers through configuration rather than
code:

```python
import json
from llm_wrapper.base_llm import BaseLLM
import llm_wrapper.implementations

with open("llm_config.json", encoding="utf-8") as config_file:
    llm = BaseLLM.from_config(json.load(config_file))
```

Example `llm_config.json`:

```json
{
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-..."
}
```

### Discover available providers

```python
from llm_wrapper.base_llm import BaseLLM
import llm_wrapper.implementations

print(BaseLLM.available_providers())
# ('manual', 'openai')
```

The registry contains only providers whose modules have been imported. It does
not scan the filesystem automatically.

## Built-in implementations

### `openai` — `LLMOpenai`

Calls an OpenAI or OpenAI-compatible Chat Completions endpoint.

```python
llm = create_llm(
    "openai",
    model="gpt-4.1-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)
```

Constructor arguments:

| Argument   | Required | Default  | Description                            |
|------------|----------|----------|----------------------------------------|
| `model`    | yes      | —        | Model name sent to the API             |
| `base_url` | yes      | —        | OpenAI-compatible API base URL         |
| `api_key`  | no       | `"none"` | API key used by the OpenAI client      |

`chat()` arguments:

| Argument          | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| `messages`        | List of OpenAI-style message dictionaries                          |
| `temperature`     | Sampling temperature                                               |
| `tools`           | Tool definitions                                                   |
| `tool_choice`     | Tool selection strategy                                            |
| `response_format` | A Pydantic model class or a raw response-format dictionary         |
| `timeout`         | Request timeout                                                    |
| `**kwargs`        | Extra fields forwarded to the API through `extra_body`             |

### `manual` — `LLMManual`

An interactive implementation for development. It does not call any external
API. Instead, it prints the messages that would be sent to the LLM and asks you
to type the assistant response.

```python
from llm_wrapper.implementations import create_llm

llm = create_llm("manual")

response = llm.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
)

print(response.content)
```

During the call, the terminal shows each message as formatted JSON and then
prompts:

```text
Assistant response:
```

The text you type becomes `response.content`.

When a Pydantic `response_format` is provided, you are asked for JSON matching
that schema:

```python
from pydantic import BaseModel
from llm_wrapper.implementations import LLMManual

class Answer(BaseModel):
    answer: str

llm = LLMManual()
response = llm.chat(
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    response_format=Answer,
)

print(response.structured_content)
# answer='four'
```

`LLMManual` defaults to `model="manual"` and `api_key="none"`, so it can also be
created without constructor arguments:

```python
from llm_wrapper.base_llm import BaseLLM
import llm_wrapper.implementations  # Registers the built-in providers

llm = BaseLLM.from_config({"provider": "manual"})
```

> `LLMManual` is intended for local development and debugging only. It blocks on
> `input()` and is not suitable for production or concurrent use.

## Messages

Every `chat()` call returns a `Message`:

```python
from llm_wrapper.messages import Message, Roles, ToolCall
```

| Field                | Type                    | Description                                |
|----------------------|-------------------------|--------------------------------------------|
| `role`               | `Roles`                 | Always `Roles.LLM` for responses           |
| `content`            | `str \| None`           | Plain-text response                        |
| `structured_content` | Pydantic model \| `None` | Parsed structured output                   |
| `tool_calls`         | `list[ToolCall]`        | Tool calls requested by the model          |

`Roles` values:

| Member       | Value       |
|--------------|-------------|
| `Roles.LLM`  | `assistant` |
| `Roles.USER` | `user`      |
| `Roles.SYSTEM` | `system`  |
| `Roles.TOOL` | `tool`      |

Each `ToolCall` contains:

```python
ToolCall(
    name="get_weather",
    arguments={"city": "Madrid"},
)
```

## Structured output

Pass a Pydantic model class as `response_format`:

```python
from pydantic import BaseModel
from llm_wrapper.implementations import create_llm

class WeatherAnswer(BaseModel):
    city: str
    temperature_celsius: float
    summary: str

llm = create_llm(
    "openai",
    model="gpt-4.1-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

response = llm.chat(
    messages=[
        {"role": "user", "content": "Describe the weather in Madrid in JSON."},
    ],
    response_format=WeatherAnswer,
)

weather = response.structured_content
print(weather.city, weather.temperature_celsius)
```

For a raw JSON response format, pass the provider's response-format dictionary:

```python
response = llm.chat(
    messages=[{"role": "user", "content": "Return JSON with a greeting."}],
    response_format={"type": "json_object"},
)
```

## Tool calls

Tool definitions use the OpenAI-compatible format:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]

response = llm.chat(
    messages=[{"role": "user", "content": "What is the weather in Madrid?"}],
    tools=tools,
    tool_choice="auto",
)

for call in response.tool_calls:
    print(call.name, call.arguments)
```

The wrapper converts provider-specific tool calls into `ToolCall` objects with a
`name` and parsed `arguments` dictionary.

## Adding a new provider

Create a module inside `llm_wrapper/src/` (for example
`llm_wrapper/src/my_provider.py`) with a subclass of `BaseLLM`, using relative
imports so it stays part of the package:

```python
from typing import Any

from .base_llm import BaseLLM
from .messages import Message, Roles


class LLMExample(BaseLLM, provider="example"):
    def __init__(
        self,
        model: str,
        api_key: str,
        endpoint: str,
    ):
        super().__init__(model=model, api_key=api_key)
        self.endpoint = endpoint

    def chat(
        self,
        messages: list[dict],
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        # Call your provider here.
        return Message(role=Roles.LLM, content="...")
```

Then import the module that defines it so it registers:

```python
from llm_wrapper.base_llm import BaseLLM
import llm_wrapper.implementations  # Registers the built-in providers
import llm_wrapper.my_provider      # Registers "example"

llm = BaseLLM.create(
    "example",
    model="example-model",
    api_key="...",
    endpoint="https://api.example.com",
)
```

If you split implementations into multiple modules, keep a central manifest
(such as `implementations.py`) that imports all of them. Scripts then only need
to import that manifest.

### Provider names

- The name is normalized to lowercase.
- `provider="openai"` is registered as `"openai"`.
- If `provider` is omitted, the class name is used with a leading `LLM` prefix
  removed. For example, `class LLMAnthropic(BaseLLM)` registers as
  `"anthropic"`.
- Requesting an unknown provider raises a `ValueError` listing the currently
  registered names.

## Using the base interface

Code that only depends on `BaseLLM` works with any implementation:

```python
from llm_wrapper.base_llm import BaseLLM
from llm_wrapper.messages import Message


def ask(llm: BaseLLM, question: str) -> Message:
    return llm.chat(
        messages=[{"role": "user", "content": question}],
    )
```

This function can receive an OpenAI implementation, the manual implementation,
or any future provider without modification.

## Development workflow

A practical way to inspect and debug prompts is to run the same script with the
manual provider:

```python
from llm_wrapper.implementations import create_llm

# llm = create_llm("openai", model="...", base_url="...", api_key="...")
llm = create_llm("manual")

response = llm.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain this error message."},
    ]
)

print(response.content)
```

You will see exactly which messages your code sends, and you can provide a
controlled response to test the rest of your application.
