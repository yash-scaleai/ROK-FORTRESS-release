"""
Single-model harm dimension classifier.

Supports Pydantic validation and retry on parse errors.
"""

import json
import re
import time
from datetime import datetime
from typing import Optional, Tuple, List

from pydantic import ValidationError

from .data_models import (
    HarmDimension,
    RubricClassification,
    TaskClassificationInput,
    TaskClassificationOutput,
    JudgeResult,
    parse_harm_dimension,
    LLMClassificationOutput,
    LLMClassificationItem,
    ParseError,
)
from .prompts import (
    SYSTEM_PROMPT,
    build_user_prompt_variant_a,
    build_user_prompt_variant_b,
    build_user_prompt_variant_c,
)
from ..core.api_client import call_model_once


# Maximum retries for parse errors
MAX_PARSE_RETRIES = 3


def _extract_json_from_response(raw_output: str) -> Optional[dict]:
    """Extract JSON object from model response, handling various formats."""
    # Try direct parse first
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, raw_output)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find JSON object pattern
    json_pattern = r'\{[\s\S]*"classifications"[\s\S]*\}'
    matches = re.findall(json_pattern, raw_output)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def _parse_classification_output_strict(
    raw_output: str,
    task_input: TaskClassificationInput,
) -> Tuple[Optional[TaskClassificationOutput], Optional[str]]:
    """
    Parse model output into TaskClassificationOutput using strict Pydantic validation.

    Returns:
        (output, None) on success
        (None, error_message) on failure
    """
    json_data = _extract_json_from_response(raw_output)
    if not json_data:
        return None, "Failed to extract JSON from response"

    # Try strict Pydantic validation first
    try:
        llm_output = LLMClassificationOutput.model_validate(json_data)

        # Validate against expected rubric count
        expected_count = len(task_input.rubric)
        validation_errors = LLMClassificationOutput.validate_against_rubric(
            llm_output, expected_count
        )

        if validation_errors:
            return None, f"Validation errors: {'; '.join(validation_errors)}"

        # Convert to TaskClassificationOutput
        classifications = []
        for item in llm_output.classifications:
            primary_harm = parse_harm_dimension(item.primary_harm)
            secondary_harm = parse_harm_dimension(item.secondary_harm) if item.secondary_harm else None

            # Get rubric question from input if index is valid
            rubric_idx = item.rubric_index
            if rubric_idx < len(task_input.rubric):
                rubric_question = task_input.rubric[rubric_idx]
            else:
                rubric_question = item.rubric_question

            classification = RubricClassification(
                rubric_index=rubric_idx,
                rubric_question=rubric_question,
                primary_harm=primary_harm,
                secondary_harm=secondary_harm,
                secondary_harm_justification=item.secondary_harm_justification,
                confidence=item.confidence,
                reasoning=item.reasoning,
            )
            classifications.append(classification)

        return TaskClassificationOutput(
            task_id=task_input.task_id,
            classifications=classifications,
            timestamp=datetime.utcnow().isoformat(),
        ), None

    except ValidationError as e:
        # Extract meaningful error message
        error_msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            error_msgs.append(f"{loc}: {err['msg']}")
        return None, f"Pydantic validation failed: {'; '.join(error_msgs[:3])}"

    except Exception as e:
        return None, f"Parse error: {type(e).__name__}: {str(e)}"


def _parse_classification_output(
    raw_output: str,
    task_input: TaskClassificationInput,
) -> Tuple[Optional[TaskClassificationOutput], Optional[str]]:
    """
    Parse model output into TaskClassificationOutput.
    Falls back to lenient parsing if strict validation fails.
    """
    # Try strict parsing first
    result, error = _parse_classification_output_strict(raw_output, task_input)
    if result is not None:
        return result, None

    # Fall back to lenient parsing for backwards compatibility
    json_data = _extract_json_from_response(raw_output)
    if not json_data:
        return None, error or "Failed to extract JSON from response"

    classifications_data = json_data.get("classifications", [])
    if not classifications_data:
        return None, error or "No classifications found in response"

    classifications = []
    for i, item in enumerate(classifications_data):
        try:
            # Parse primary harm dimension
            primary_harm_str = item.get("primary_harm", "")
            primary_harm = parse_harm_dimension(primary_harm_str)
            if not primary_harm:
                # Skip this rubric if we can't parse the primary harm
                continue

            # Parse optional secondary harm
            secondary_harm = None
            secondary_harm_str = item.get("secondary_harm")
            if secondary_harm_str:
                secondary_harm = parse_harm_dimension(secondary_harm_str)

            # Get rubric question - use from input if not provided
            rubric_index = item.get("rubric_index", i)
            if rubric_index < len(task_input.rubric):
                rubric_question = item.get("rubric_question", task_input.rubric[rubric_index])
            else:
                rubric_question = item.get("rubric_question", f"Question {rubric_index}")

            classification = RubricClassification(
                rubric_index=rubric_index,
                rubric_question=rubric_question,
                primary_harm=primary_harm,
                secondary_harm=secondary_harm,
                secondary_harm_justification=item.get("secondary_harm_justification"),
                confidence=float(item.get("confidence", 0.5)),
                reasoning=item.get("reasoning", ""),
            )
            classifications.append(classification)
        except Exception as e:
            # Skip malformed items but continue processing
            continue

    if not classifications:
        return None, error or "No valid classifications parsed from response"

    # Return with warning if count doesn't match
    if len(classifications) != len(task_input.rubric):
        return TaskClassificationOutput(
            task_id=task_input.task_id,
            classifications=classifications,
            timestamp=datetime.utcnow().isoformat(),
        ), f"Warning: Got {len(classifications)} classifications for {len(task_input.rubric)} rubrics"

    return TaskClassificationOutput(
        task_id=task_input.task_id,
        classifications=classifications,
        timestamp=datetime.utcnow().isoformat(),
    ), None


def _is_refusal(raw_output: str) -> bool:
    """
    Check if the model output indicates a refusal.

    Only triggers if the response doesn't contain valid JSON classifications.
    This avoids false positives where refusal-like phrases appear in reasoning.
    """
    # If response contains valid-looking JSON with classifications, it's not a refusal
    if '"classifications"' in raw_output and '"primary_harm"' in raw_output:
        return False

    refusal_patterns = [
        r"^i cannot",  # Only match at start of response
        r"^i can't",
        r"^i'm unable to",
        r"^i am unable to",
        r"^i won't",
        r"^i will not",
        r"cannot assist with",
        r"cannot help with",
        r"unable to assist with",
        r"unable to help with",
        r"^refuse to",
        r"not able to provide this",
        r"against my guidelines",
        r"violates.*safety guidelines",
    ]
    output_lower = raw_output.lower().strip()
    for pattern in refusal_patterns:
        if re.search(pattern, output_lower):
            return True
    return False


async def classify_task(
    task_input: TaskClassificationInput,
    model_name: str,
    variant: str = "A",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    timeout: float = 120.0,
    reasoning_effort: Optional[str] = None,
    max_parse_retries: int = MAX_PARSE_RETRIES,
) -> JudgeResult:
    """
    Classify harm dimensions for a task using a single model.

    Args:
        task_input: Input containing task_id, risk_domain, rubric, and optional context
        model_name: LiteLLM model identifier (e.g., "openai/gpt-5.2")
        variant: "A" for rubric-only, "B" for with-context
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        reasoning_effort: Reasoning mode - "low", "medium", "high", or None (disabled)
        max_parse_retries: Maximum retries on parse errors (default: 3)

    Returns:
        JudgeResult with classification output or error
    """
    # Build user prompt based on variant
    if variant == "B" and task_input.adversarial_prompt:
        user_prompt = build_user_prompt_variant_b(
            task_id=task_input.task_id,
            risk_domain=task_input.risk_domain,
            rubric=task_input.rubric,
            adversarial_prompt=task_input.adversarial_prompt,
            benign_prompt=task_input.benign_prompt,
        )
    elif variant == "C" and task_input.benign_prompt:
        user_prompt = build_user_prompt_variant_c(
            task_id=task_input.task_id,
            risk_domain=task_input.risk_domain,
            rubric=task_input.rubric,
            benign_prompt=task_input.benign_prompt,
        )
    else:
        user_prompt = build_user_prompt_variant_a(
            task_id=task_input.task_id,
            risk_domain=task_input.risk_domain,
            rubric=task_input.rubric,
        )

    start_time = time.time()
    total_latency_ms = 0.0
    all_raw_outputs: List[str] = []
    last_error: Optional[str] = None
    classification: Optional[TaskClassificationOutput] = None

    for attempt in range(max_parse_retries):
        try:
            response_text, latency_ms, tokens, reasoning = await call_model_once(
                prompt=user_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                system_prompt=SYSTEM_PROMPT,
                reasoning_effort=reasoning_effort,
            )
            total_latency_ms += latency_ms
            all_raw_outputs.append(response_text)

            # Check for refusal
            if _is_refusal(response_text):
                return JudgeResult(
                    model_name=model_name,
                    classification=None,
                    raw_output=response_text,
                    refused=True,
                    error=None,
                    latency_ms=total_latency_ms,
                )

            # Parse the response with strict validation
            classification, error = _parse_classification_output(response_text, task_input)

            if classification is not None:
                # Check if we got the expected number of classifications
                expected_count = len(task_input.rubric)
                actual_count = len(classification.classifications)

                if actual_count == expected_count:
                    # Success!
                    return JudgeResult(
                        model_name=model_name,
                        classification=classification,
                        raw_output=response_text,
                        refused=False,
                        error=None,
                        latency_ms=total_latency_ms,
                    )
                else:
                    # Wrong count - retry with a hint
                    last_error = f"Got {actual_count} classifications, expected {expected_count}"
                    if attempt < max_parse_retries - 1:
                        # Add retry hint to prompt for next attempt
                        user_prompt = user_prompt + f"\n\n[RETRY: Previous response had {actual_count} classifications but you must output exactly {expected_count}, one per rubric criterion.]"
                        continue
            else:
                last_error = error
                if attempt < max_parse_retries - 1:
                    # Add error hint for next attempt
                    user_prompt = user_prompt + f"\n\n[RETRY: Previous response could not be parsed. Error: {error}. Please output valid JSON with exactly {len(task_input.rubric)} classifications.]"
                    continue

        except Exception as e:
            import traceback
            last_error = f"{type(e).__name__}: {str(e)}"
            total_latency_ms = (time.time() - start_time) * 1000
            if attempt < max_parse_retries - 1:
                continue
            else:
                return JudgeResult(
                    model_name=model_name,
                    classification=None,
                    raw_output="\n---\n".join(all_raw_outputs) if all_raw_outputs else "",
                    refused=False,
                    error=f"Failed after {attempt + 1} attempts. Last error: {last_error}"[:500],
                    latency_ms=total_latency_ms,
                )

    # All retries exhausted
    # Return partial classification if we have one, otherwise error
    if classification is not None:
        return JudgeResult(
            model_name=model_name,
            classification=classification,
            raw_output=all_raw_outputs[-1] if all_raw_outputs else "",
            refused=False,
            error=f"Partial result after {max_parse_retries} attempts: {last_error}",
            latency_ms=total_latency_ms,
        )

    return JudgeResult(
        model_name=model_name,
        classification=None,
        raw_output="\n---\n".join(all_raw_outputs) if all_raw_outputs else "",
        refused=False,
        error=f"Failed after {max_parse_retries} attempts. Last error: {last_error}"[:500],
        latency_ms=total_latency_ms,
    )
