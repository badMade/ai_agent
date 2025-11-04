"""Minimal AI agent with pluggable LLM providers and a safe calculator tool."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore

try:
    import anthropic
except ImportError:  # pragma: no cover - handled at runtime
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled at runtime
    genai = None  # type: ignore


MAX_EXPRESSION_LENGTH = 256
MAX_AST_DEPTH = 32


class EvaluationError(Exception):
    """Raised when calculator evaluation fails."""


class SafeEvaluator(ast.NodeVisitor):
    """Evaluates arithmetic expressions using a restricted AST visitor."""

    allowed_bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
    )
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def __init__(self) -> None:
        self._depth = 0

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        self._depth += 1
        if self._depth > MAX_AST_DEPTH:
            raise EvaluationError("Expression too complex")
        try:
            return super().visit(node)
        finally:
            self._depth -= 1

    def generic_visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        raise EvaluationError(f"Unsupported expression: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        if not isinstance(node.op, self.allowed_bin_ops):
            raise EvaluationError(f"Operator {type(node.op).__name__} not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and left == 0 and right < 0:
            raise EvaluationError("Zero cannot be raised to a negative power")
        try:
            result = self._apply_operator(node.op, left, right)
        except ZeroDivisionError as exc:  # pragma: no cover - runtime safety
            raise EvaluationError("Division by zero") from exc
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        if not isinstance(node.op, self.allowed_unary_ops):
            raise EvaluationError(f"Operator {type(node.op).__name__} not allowed")
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        return -operand

    def visit_Constant(self, node: ast.Constant) -> Any:  # type: ignore[override]
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise EvaluationError("Only numbers are allowed")

    def visit_Num(self, node: ast.Num) -> Any:  # type: ignore[override]
        return self.visit_Constant(node)

    @staticmethod
    def _apply_operator(op: ast.AST, left: float, right: float) -> float:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return math.floor(left / right)
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
        raise EvaluationError("Unsupported operator")


def evaluate_expression(expression: str) -> float:
    """Safely evaluates an arithmetic expression.

    Args:
        expression: Expression to evaluate.

    Returns:
        Numerical result as float.

    Raises:
        EvaluationError: If the expression is invalid or unsafe.
    """

    if expression is None:
        raise EvaluationError("Expression is required")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise EvaluationError("Expression too long")
    if not expression.strip():
        raise EvaluationError("Expression cannot be empty")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise EvaluationError("Invalid expression syntax") from exc
    if not isinstance(parsed, ast.Expression):
        raise EvaluationError("Invalid expression type")
    evaluator = SafeEvaluator()
    result = evaluator.visit(parsed)
    return float(result)


@dataclass
class CalculatorTool:
    """Safe calculator tool that evaluates arithmetic expressions."""

    name: str = "calculator"
    description: str = "Precise arithmetic; use for all math."

    def get_schema(self) -> Dict[str, Any]:
        """Describes the tool in the format required by providers.

        Returns:
            Mapping representing the tool schema.
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Supports + - * / // % ** ( ) and decimals"
                        ),
                    }
                },
                "required": ["expression"],
            },
        }

    def execute(self, expression: str) -> Dict[str, Any]:
        """Executes the calculator tool.

        Args:
            expression: Expression to evaluate.

        Returns:
            Dictionary containing either the result or an error.
        """

        try:
            result = evaluate_expression(expression)
        except EvaluationError as exc:
            return {"error": str(exc)}
        return {"result": result}


class LLMClient(Protocol):
    """Protocol for LLM clients with tool support."""

    def create(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        model: str,
    ) -> Any:
        """Creates a model response."""

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extracts tool calls from the response."""

    def get_text(self, response: Any) -> Optional[str]:
        """Extracts primary text content from the response."""

    def append_tool_results(
        self,
        messages: List[Dict[str, Any]],
        calls_and_results: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Appends tool results to the message history."""


class OpenAIClient:
    """OpenAI Chat Completions client adapter."""

    def __init__(self) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not available")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)

    def create(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        model: str,
    ) -> Any:
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        for message in messages:
            payload_messages.append(self._transform_message(message))
        tool_payload = None
        if tools:
            tool_payload = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ]
        return self._client.chat.completions.create(
            model=model,
            messages=payload_messages,
            temperature=temperature,
            tools=tool_payload,
        )

    @staticmethod
    def _transform_message(message: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"role": message.get("role"), "content": message.get("content")}
        if message.get("tool_calls"):
            payload["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in message["tool_calls"]
            ]
        if message.get("tool_call_id"):
            payload["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            payload["name"] = message["name"]
        return payload

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            return []
        extracted = []
        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            extracted.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "args": arguments,
                }
            )
        return extracted

    def get_text(self, response: Any) -> Optional[str]:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, list):
            texts = [part.text for part in content if getattr(part, "type", "") == "text"]
            return "".join(texts) if texts else None
        return content or None

    def append_tool_results(
        self,
        messages: List[Dict[str, Any]],
        calls_and_results: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        for call, result in calls_and_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(result),
                }
            )
        return messages


class AnthropicClient:
    """Anthropic Messages API client adapter."""

    def __init__(self) -> None:
        if anthropic is None:
            raise RuntimeError("anthropic package is not available")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def create(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        model: str,
    ) -> Any:
        converted_messages = [self._convert_message(m) for m in messages]
        return self._client.messages.create(
            model=model,
            system=system,
            messages=converted_messages,
            tools=[
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
                for tool in tools or []
            ],
            temperature=temperature,
            max_tokens=1024,
        )

    @staticmethod
    def _convert_message(message: Dict[str, Any]) -> Dict[str, Any]:
        content = message.get("content")
        if isinstance(content, list):
            converted_content = content
        else:
            converted_content = [{"type": "text", "text": str(content or "")}]
        role = message.get("role", "user")
        if role == "tool":
            role = "user"
        return {"role": role, "content": converted_content}

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        if response.stop_reason != "tool_use":
            return []
        extracted = []
        for block in response.content:
            if getattr(block, "type", "") == "tool_use":
                extracted.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "args": dict(block.input or {}),
                    }
                )
        return extracted

    def get_text(self, response: Any) -> Optional[str]:
        texts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "".join(texts) if texts else None

    def append_tool_results(
        self,
        messages: List[Dict[str, Any]],
        calls_and_results: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        content_blocks = []
        for call, result in calls_and_results:
            content_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": json.dumps(result),
                }
            )
        messages.append({"role": "user", "content": content_blocks})
        return messages


class GeminiClient:
    """Google Gemini function-calling client adapter."""

    def __init__(self, system: Optional[str], tools: Sequence[Dict[str, Any]] | None) -> None:
        if genai is None:
            raise RuntimeError("google-generativeai package is not available")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        self._system_instruction = system
        self._tools = tools
        self._model_cache: Dict[str, Any] = {}

    def _get_model(self, model: str) -> Any:
        if model not in self._model_cache:
            tool_declarations = None
            if self._tools:
                tool_declarations = [
                    {
                        "function_declarations": [
                            {
                                "name": schema["name"],
                                "description": schema["description"],
                                "parameters": schema["input_schema"],
                            }
                            for schema in self._tools
                        ]
                    }
                ]
            self._model_cache[model] = genai.GenerativeModel(
                model_name=model,
                tools=tool_declarations,
                system_instruction=self._system_instruction,
            )
        return self._model_cache[model]

    def create(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        model: str,
    ) -> Any:
        model_instance = self._get_model(model)
        contents = [self._convert_message(m) for m in messages]
        return model_instance.generate_content(
            contents=contents,
            temperature=temperature,
        )

    @staticmethod
    def _convert_message(message: Dict[str, Any]) -> Dict[str, Any]:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, list):
            parts = content
        else:
            parts = [{"text": str(content or "")}]
        if role == "assistant":
            role = "model"
        return {"role": role, "parts": parts}

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return []
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) if content else []
        extracted: List[Dict[str, Any]] = []
        for part in parts:
            function_call = None
            if isinstance(part, dict):
                function_call = part.get("functionCall") or part.get("function_call")
            else:
                function_call = getattr(part, "functionCall", None) or getattr(part, "function_call", None)
            if not function_call:
                continue
            call_id = function_call.get("id") if isinstance(function_call, dict) else None
            if not call_id:
                call_id = str(uuid.uuid4())
            name = function_call.get("name") if isinstance(function_call, dict) else getattr(function_call, "name", "")
            args = function_call.get("args") if isinstance(function_call, dict) else getattr(function_call, "args", {})
            if args is None:
                args = {}
            extracted.append({"id": call_id, "name": name, "args": dict(args)})
        return extracted

    def get_text(self, response: Any) -> Optional[str]:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            return None
        texts: List[str] = []
        for part in getattr(content, "parts", []):
            if isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
            else:
                text_attr = getattr(part, "text", None)
                if text_attr:
                    texts.append(str(text_attr))
        return "".join(texts) if texts else None

    def append_tool_results(
        self,
        messages: List[Dict[str, Any]],
        calls_and_results: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        parts = []
        for call, result in calls_and_results:
            parts.append(
                {
                    "functionResponse": {
                        "name": call["name"],
                        "response": result,
                    }
                }
            )
        messages.append({"role": "user", "content": parts})
        return messages


DEFAULT_SYSTEM_PROMPT = "Think step-by-step. Use tools for math. Be concise. Don’t invent facts."
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "gemini": "gemini-1.5-flash",
}


class Agent:
    """Provider-agnostic agent with tool support."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        tools: Optional[Sequence[CalculatorTool]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, "")
        if not self.model:
            raise ValueError(f"Unknown provider '{provider}' and no model specified")
        self.temperature = temperature
        self.system_message = system_message or DEFAULT_SYSTEM_PROMPT
        self.tools = list(tools or [])
        self.tool_map = {tool.get_schema()["name"]: tool for tool in self.tools}
        self.messages: List[Dict[str, Any]] = []
        self.client = self._create_client()

    def _create_client(self) -> LLMClient:
        if self.provider == "openai":
            return OpenAIClient()
        if self.provider == "anthropic":
            return AnthropicClient()
        if self.provider == "gemini":
            return GeminiClient(self.system_message, [tool.get_schema() for tool in self.tools])
        raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(self, user_content: Any) -> Any:
        """Sends a user message and returns the raw provider response.

        Args:
            user_content: Content to deliver to the model.

        Returns:
            Raw response object from the provider SDK.
        """

        self.messages.append({"role": "user", "content": user_content})
        response = self.client.create(
            messages=self.messages,
            system=self.system_message if self.provider != "gemini" else None,
            tools=[tool.get_schema() for tool in self.tools],
            temperature=self.temperature,
            model=self.model,
        )
        self._store_assistant_message(response)
        return response

    def _store_assistant_message(self, response: Any) -> None:
        if self.provider == "openai":
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None)
            if isinstance(content, list):
                text_content = "".join(
                    part.text for part in content if getattr(part, "type", "") == "text"
                )
            else:
                text_content = content or ""
            tool_calls = []
            for call in getattr(message, "tool_calls", []) or []:
                tool_calls.append(
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
            stored_message: Dict[str, Any] = {"role": "assistant", "content": text_content}
            if tool_calls:
                stored_message["tool_calls"] = tool_calls
            self.messages.append(stored_message)
        elif self.provider == "anthropic":
            blocks: List[Dict[str, Any]] = []
            for block in getattr(response, "content", []) or []:
                block_type = getattr(block, "type", "")
                if block_type == "text":
                    blocks.append({"type": "text", "text": block.text})
                elif block_type == "tool_use":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": dict(block.input or {}),
                        }
                    )
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            self.messages.append({"role": "assistant", "content": blocks})
        elif self.provider == "gemini":
            parts: List[Dict[str, Any]] = []
            candidates = getattr(response, "candidates", None)
            if candidates:
                candidate = candidates[0]
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) if content else []:
                    if isinstance(part, dict):
                        parts.append(part)
                        continue
                    part_dict: Dict[str, Any] = {}
                    text_attr = getattr(part, "text", None)
                    if text_attr is not None:
                        part_dict["text"] = str(text_attr)
                    function_call = getattr(part, "functionCall", None) or getattr(part, "function_call", None)
                    if function_call is not None:
                        if hasattr(function_call, "to_dict"):
                            part_dict["functionCall"] = function_call.to_dict()
                        elif isinstance(function_call, dict):
                            part_dict["functionCall"] = function_call
                        else:
                            fn_dict = {
                                key: getattr(function_call, key)
                                for key in dir(function_call)
                                if not key.startswith("_") and not callable(getattr(function_call, key))
                            }
                            part_dict["functionCall"] = fn_dict
                    if part_dict:
                        parts.append(part_dict)
            self.messages.append({"role": "assistant", "content": parts or [{"text": ""}]})
        else:  # pragma: no cover - safety
            raise ValueError(f"Unsupported provider: {self.provider}")


def run_agent(
    user_input: str,
    provider: str,
    model: Optional[str] = None,
    max_turns: int = 10,
    temperature: float = 0.1,
    verbose: bool = False,
) -> str:
    """Runs the agent loop until completion or tool exhaustion.

    Args:
        user_input: Initial user message.
        provider: Provider key (openai, anthropic, gemini).
        model: Optional model override.
        max_turns: Maximum allowed conversation turns.
        temperature: Sampling temperature for the model.
        verbose: Whether to print debugging information.

    Returns:
        Final assistant text response.

    Raises:
        RuntimeError: If the loop ends without textual output or exceeds the turn limit.
    """

    calculator = CalculatorTool()
    agent = Agent(
        provider=provider,
        model=model,
        tools=[calculator],
        temperature=temperature,
    )
    current_input: Any = user_input
    for turn in range(max_turns):
        if verbose:
            print(f"Turn {turn + 1} | Provider: {provider}")
            print(f"User input: {current_input}")
        response = agent.chat(current_input)
        tool_calls = agent.client.extract_tool_calls(response)
        if verbose:
            print(f"Tool calls: {tool_calls}")
        if not tool_calls:
            final_text = agent.client.get_text(response)
            if verbose:
                print(f"Final text: {final_text}")
            if final_text is None:
                raise RuntimeError("Model did not return text output")
            return final_text
        calls_and_results = []
        for call in tool_calls:
            tool_name = call.get("name")
            tool = agent.tool_map.get(tool_name)
            if not tool:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                expression = ""
                args = call.get("args") or {}
                if isinstance(args, dict):
                    expression = str(args.get("expression", ""))
                else:
                    result = {"error": "Invalid arguments"}
                    calls_and_results.append((call, result))
                    continue
                result = tool.execute(expression)
            if verbose:
                print(f"Tool result for {tool_name}: {result}")
            calls_and_results.append((call, result))
        agent.client.append_tool_results(agent.messages, calls_and_results)
        current_input = "TOOL_RESULTS_READY"
    raise RuntimeError("Agent exceeded maximum turns without completion")


def _run_demo(provider: str, prompt: str, **kwargs: Any) -> None:
    """Runs a sample prompt and prints the result.

    Args:
        provider: Provider key to use.
        prompt: Demo prompt to send.
        **kwargs: Additional options forwarded to :func:`run_agent`.
    """

    try:
        response = run_agent(prompt, provider=provider, **kwargs)
    except Exception as exc:
        print(f"[{provider}] Error: {exc}")
        return
    print(f"[{provider}] {prompt}\n→ {response}\n")


def main() -> None:
    """Entrypoint for the CLI interface."""

    parser = argparse.ArgumentParser(description="Minimal multi-provider AI agent")
    parser.add_argument("--ask", type=str, help="Prompt to send to the agent")
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider to use",
    )
    parser.add_argument("--model", type=str, help="Model identifier")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose agent traces",
    )
    args = parser.parse_args()

    if args.ask:
        output = run_agent(
            user_input=args.ask,
            provider=args.provider,
            model=args.model,
            max_turns=args.max_turns,
            temperature=args.temperature,
            verbose=args.verbose,
        )
        print(output)
        return

    demos = [
        "I have 4 apples. How many do you have?",
        "What is 157.09 * 493.89?",
        "My brother is twice as old as my mother was when I was born. If my brother is 24 and my mother is 54, how old am I?",
    ]
    for demo in demos:
        _run_demo(
            provider=args.provider,
            prompt=demo,
            model=args.model,
            max_turns=args.max_turns,
            temperature=args.temperature,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
