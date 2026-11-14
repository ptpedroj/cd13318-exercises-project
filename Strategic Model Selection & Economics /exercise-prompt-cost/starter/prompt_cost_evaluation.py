# Prompt Engineering and Cost Evaluation Exercise - Starter Template
# TODO: Complete this script to optimize prompts for effectiveness and cost efficiency

# TODO: Import necessary libraries
# Hint: You'll need openai, pandas, time, json, typing, matplotlib, seaborn, datetime, numpy
import openai
from openai import OpenAI
# TODO: Add remaining imports here
import os
import time
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; we can still read a pre-set env var.
    load_dotenv = None

# Read the key from the environment at runtime (never commit real keys).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


_client: Optional[OpenAI] = None

def get_client(api_key=OPENAI_API_KEY) -> OpenAI:
    """Create the OpenAI client once and reuse it for every call."""
    global _client
    if _client is None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openai.vocareum.com/v1" if api_key.startswith("voc") else None
        )
    return _client


def embed(texts: list[str]) -> list[np.array]:
    """Embed one or more texts in a single API call; one ndarray per text."""
    client = get_client()
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(d.embedding, dtype=float) for d in response.data]


def cosine(a, b) -> float:
    """Cosine similarity of two vectors (numpy)."""
    a = np.asarray(a, dtype=float) # allows for bare lists though ndarrays are expected
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.


PLOT_OUTPUT_DIR = Path(__file__).parent / "outputs" # charts land next to the script, not in the cwd

CONFIG_MINIMAL = "minimal"
CONFIG_STANDARD = "standard"
CONFIG_PREMIUM = "premium"

# TODO: Define prompt configurations for different cost-effectiveness strategies
# Create a dictionary with three configurations:
# 1. "minimal" - lowest cost but potentially less effective
# 2. "standard" - balanced cost-performance ratio
# 3. "premium" - highest quality but most expensive
PROMPT_CONFIGS = {
    CONFIG_MINIMAL: {
        "parameters": {
            # TODO: Choose most cost-effective model (hint: gpt-4o-mini is cheapest)
            "model": "gpt-4o-mini",
            # TODO: Set temperature for balanced results
            "temperature": 0.7,
            # TODO: Set low max_completion_tokens to control costs
            "max_completion_tokens": 200
        },
        "description": "Minimal prompts with cost-effective model for budget optimization"
    },
    CONFIG_STANDARD: {
        "parameters": {
            # TODO: Choose balanced model (hint: gpt-4o offers good performance)
            "model": "gpt-4o",
            # TODO: Set temperature for consistent results
            "temperature": 0.6,
            # TODO: Set moderate max_completion_tokens
            "max_completion_tokens": 400
        },
        "description": "Standard prompts with balanced cost-performance ratio"
    },
    CONFIG_PREMIUM: {
        "parameters": {
            # TODO: Choose high-performance model
            "model": "gpt-4.1", # replaced gpt-4-turbo since it is no longer available (deprecated)
            # TODO: Set temperature for consistency (hint: lower for more consistent)
            "temperature": 0.5,
            # TODO: Set higher max_completion_tokens for detailed responses
            "max_completion_tokens": 800
        },
        "description": "Detailed prompts with premium model for maximum effectiveness"
    }
}


# TODO: Define pricing information for cost calculations
# Research current OpenAI pricing and fill in the rates per 1K tokens in USD
MODEL_PRICING = {
    "gpt-4o-mini": {
        # TODO: Add input token price (hint: check OpenAI pricing page)
        "input": 0.15 / 1_000_000,
        # TODO: Add output token price
        "output": 0.6 / 1_000_000
    },
    "gpt-4o": {
        # TODO: Add input token price
        "input": 2.5 / 1_000_000,
        # TODO: Add output token price
        "output": 10. / 1_000_000
    },
    "gpt-4.1": { # replaced gpt-4-turbo since it is no longer available (deprecated)
        # TODO: Add input token price
        "input": 2. / 1_000_000,
        # TODO: Add output token price
        "output": 8. / 1_000_000
    }
}


CATEGORY_TASK_COMPLETION = "task_completion"
CATEGORY_CREATIVE_WRITING = "creative_writing"
CATEGORY_PROBLEM_SOLVING = "problem_solving"

TEXT_PLACEHOLDER = "[TEXT_PLACEHOLDER]"
# TODO: Define prompt strategies for different task categories
# Create test cases that demonstrate how prompt complexity affects quality and cost
PROMPT_STRATEGIES = {
    CATEGORY_TASK_COMPLETION: {
        CONFIG_MINIMAL: {
            # TODO: Create a basic summarization prompt
            "prompt": f"Summarize this text: {TEXT_PLACEHOLDER}",
            "description": "Basic instruction without context or examples"
        },
        CONFIG_STANDARD: {
            # TODO: Create a more detailed summarization prompt with guidance
            "prompt": f"Please provide a concise summary of the following text, focusing on the main points: {TEXT_PLACEHOLDER}",
            "description": "Clear instruction with specific guidance"
        },
        CONFIG_PREMIUM: {
            # TODO: Create a comprehensive prompt with role-playing and detailed requirements
            "prompt": f"You are an expert analyst. Provide a comprehensive summary with: 1) Main points, 2) Key evidence, 3) Conclusions. Format with headings: {TEXT_PLACEHOLDER}",
            "description": "Detailed instruction with role, structure, and formatting requirements"
        }
    },
    CATEGORY_CREATIVE_WRITING: {
        CONFIG_MINIMAL: {
            # TODO: Create a simple creative writing prompt
            "prompt": "Write a story about a robot.",
            "description": "Simple creative prompt without constraints"
        },
        CONFIG_STANDARD: {
            # TODO: Create a structured creative prompt with specifications
            "prompt": "Write a short story (200-300 words) about a robot discovering emotions. Include dialogue.",
            "description": "Structured creative prompt with length and content specifications"
        },
        CONFIG_PREMIUM: {
            # TODO: Create a comprehensive creative prompt with detailed requirements
            "prompt": "You are a skilled author. Write a compelling story (200-300 words) about a robot discovering emotions. Requirements: 1) Meaningful dialogue, 2) Emotional journey, 3) Narrative arc, 4) Vivid descriptions.",
            "description": "Comprehensive creative prompt with role-playing, detailed requirements, and quality guidelines"
        }
    },
    CATEGORY_PROBLEM_SOLVING: {
        CONFIG_MINIMAL: {
            # TODO: Create a direct business question
            "prompt": "How do I reduce customer churn?",
            "description": "Direct question without context"
        },
        CONFIG_STANDARD: {
            # TODO: Create a contextualized business question
            "prompt": "I'm running a SaaS business with 15% monthly churn. What strategies can reduce churn?",
            "description": "Contextualized question with specific details"
        },
        CONFIG_PREMIUM: {
            # TODO: Create a comprehensive business consultation prompt
            "prompt": "You are a business consultant. My SaaS has 15% churn, $50 ARPU, 6-month LTV. Provide: 1) Root cause analysis, 2) Actionable strategies, 3) Implementation timeline, 4) ROI projections.",
            "description": "Expert consultation prompt with detailed context, specific deliverables, and structured output requirements"
        }
    }
}


# TODO: Create sample text for testing summarization prompts
# Write a substantial paragraph (200-300 words) about a relevant topic
SAMPLE_TEXT = """
Most people think they're good at prompting. They're not — they're good at chatting with AI, and that skill is rapidly becoming table stakes. Same model, same Tuesday, and the difference between the person who spends forty minutes cleaning up 70%-right output and the person who comes back to finished deliverables isn't talent. It's that she's practicing a discipline most people don't know exists yet. Three model releases in February alone shipped autonomous-agent capabilities that make chat-based prompting feel like bringing a phone book to a search-engine fight. The models stopped being conversation partners and started being workers — workers that run for days without checking in. And directing a worker you can't supervise in real time is a fundamentally different skill than having a productive conversation.

I'm naming four disciplines hiding inside the word "prompting." Prompt Craft — the original, synchronous skill — is now just the ability to type, no longer differentiating. Context Engineering is where the action is: your prompt might be 200 tokens, but the context window it lands in might be a million. The other 99.98% is context engineering. Anthropic's data is brutal — on a needle-in-a-million-tokens test, Sonnet 4.5 scored 18.5%. The window is a filing cabinet with no index.

Intent Engineering tells agents what to want. Klarna learned this the hard way: excellent context, missing intent, and $40 million in projected savings cratered into a customer-satisfaction crisis. Specification Engineering — the newest discipline — is writing documents autonomous agents can execute over days: complete, structured, with acceptance criteria and constraints baked in.

The 2025 mental model — "write good instructions" — only works if you're always in the room. Agents running at 2 AM aren't. So the quality you used to supply in real time has to be embedded before the run starts.

Here's the part nobody puts in "how to prompt" courses: this discipline transfers. Tobi Lütke found that stating problems with complete context made him a better CEO, not just a better AI user. Clear requests, stated acceptance criteria, articulated constraints, decomposed projects — those are the four disciplines of AI input, and they're also just the disciplines of effective leadership. The specification is the prompt now. And the specification, done right, is just what clear thinking always looked like — finally made explicit, because the machines won't let us be lazy anymore.
"""


def call_openai_api(prompt: str, config: Dict) -> Dict:
    """
    Make API call to OpenAI and capture comprehensive response metrics including cost analysis.
    
    TODO: Complete this function to:
    1. Initialize OpenAI client
    2. Make API call with given configuration
    3. Measure response time
    4. Calculate costs based on token usage
    5. Return structured results with cost metrics
    
    Args:
        prompt (str): The input prompt to send to the model
        config (Dict): Model configuration including model name, temperature, etc.
    
    Returns:
        Dict: Response data including content, performance metrics, cost analysis, and error handling
    """
    parameters = config['parameters']
    print(f"  🔄 Calling {parameters['model']} (temperature: {parameters['temperature']}, max_completion_tokens: {parameters['max_completion_tokens']})...")
    
    # TODO: Record start time for latency measurement
    start_time = time.time()
    
    try:
        # TODO: Initialize OpenAI client with your API key
        # SECURITY NOTE: Use environment variables for API keys in production
        client = get_client()
        
        # TODO: Make the API call
        # Hint: Use client.chat.completions.create() with:
        # - model from config
        # - messages with user role and prompt content
        # - temperature from config
        # - max_completion_tokens from config

        response = client.chat.completions.create(
            messages = [{"role": "user", "content": prompt}],
            **parameters
        )
        
        # TODO: Calculate latency in milliseconds
        end_time = time.time()
        latency = round((end_time - start_time) * 1000, 2)
        
        # TODO: Calculate cost based on token usage and model pricing
        # Hint: cost = (tokens / 1000) * price_per_1k_tokens
        model_name = parameters["model"]
        input_cost = response.usage.prompt_tokens * MODEL_PRICING[model_name]["input"]
        output_cost = response.usage.completion_tokens * MODEL_PRICING[model_name]["output"]
        total_cost = input_cost + output_cost
        
        # TODO: Structure the successful response
        # Include: response content, latency, token usage, cost analysis, success status
        result = {
            "response": response.choices[0].message.content or "",
            "latency_ms": latency,
            "tokens_used": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_token": total_cost / response.usage.total_tokens if response.usage.total_tokens else 0,
            "success": True,
            "error": None
        }
        
        print(f"  ✅ Success! Latency: {result['latency_ms']}ms, Tokens: {result['tokens_used']}, Cost: ${result['total_cost']:.6f}")
        return result
        
    except Exception as e:
        # TODO: Handle API errors gracefully
        print(f"Error: {str(e)}")
        return {
            "response": None, "latency_ms": None, "tokens_used": None,
            "prompt_tokens": None, "completion_tokens": None,
            "input_cost": 0, "output_cost": 0, "total_cost": 0, "cost_per_token": 0,
            "success": False, "error": str(e)
        }


# --- Heuristic tuning knobs (weights per category sum to 1.0) ---
TASK_MIN_WORDS, TASK_MAX_WORDS = 25, 200
CREATIVE_MIN_WORDS, CREATIVE_MAX_WORDS = 200, 300  # the creative prompts ask for 200-300 words
# Counts at which a feature earns full credit (more markers than this adds nothing)
STRUCTURE_TARGET = 6    # bullets / numbered items / "first, second, finally" markers
DIALOGUE_TARGET = 3     # quoted lines
PARAGRAPH_TARGET = 3    # blank-line paragraph breaks
PROBLEM_MIN_WORDS, PROBLEM_FULL_WORDS = 100, 200
# Observed cosine(summary, source) with text-embedding-3-small spans ~0.55 (truncated) to ~0.87 (full);
# rescale that band to 0..1 so good summaries can actually reach the top of the scale
COSINE_LOW, COSINE_HIGH = 0.55, 0.90
# Blended scores observed in practice span ~0.2 (cut-off answers) to ~0.85 (good answers), so a
# plain 0..1 -> 1..5 mapping never reaches 5; stretch that band over the full scale
BLEND_LOW, BLEND_HIGH = 0.2, 0.85
MIN_CONTENT_WORD_LEN = 3      # words must be longer than this to count as "content" words
TARGET_SOURCE_COVERAGE = 0.25 # a good summary should reuse ~25% of the source's key terms
GROUNDING_WEIGHT = 0.5        # share of the task score given to source grounding
DEFAULT_HEURISTIC = 0.5       # score for unknown categories
MAX_INCOMPLETE_SCORE = 4      # a truncated answer (cut off by the token cap) can't reach 5
DISTINCT_WINDOW = 50          # words per window for length-independent lexical variety
REQUIREMENT_WEIGHT = 0.3      # share of the score given to "did it cover what the prompt asked for"
REQUIREMENT_PREFIX_LEN = 5    # match "analysis"/"analyses" etc. on their first 5 letters
MIN_REQUIREMENTS = 2          # a prompt needs at least this many numbered items to be checked

HEURISTIC_WEIGHTS = {
    CATEGORY_TASK_COMPLETION:  {"min_len": 0.1, "max_len": 0.2, "distinct": 0.3, "structure": 0.4},
    CATEGORY_CREATIVE_WRITING: {"length": 0.1, "dialogue": 0.3, "distinct": 0.4, "paragraphs": 0.2},
    CATEGORY_PROBLEM_SOLVING:  {"min_len": 0.2, "structure": 0.4, "distinct": 0.3, "full_len": 0.1},
}

_STRUCTURE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s|\b(?:step|first|second|finally)\b", re.I | re.M)
_DIALOGUE_RE = re.compile(r'["“”]')
_REQUIREMENT_RE = re.compile(r"\d+\)\s*([^,.\d]+)")  # "1) Root cause analysis," -> "Root cause analysis"
_REQUIREMENT_STOPWORDS = {"and", "the", "for", "with", "that"}
# Only categories whose requirements show up as words in the text; creative asks for qualities
# ("narrative arc", "meaningful dialogue") that keyword matching would wrongly mark as missing.
REQUIREMENT_CATEGORIES = {CATEGORY_TASK_COMPLETION, CATEGORY_PROBLEM_SOLVING}


def _ramp_up(n, target):
    """0 -> 1 as n grows toward `target` (no cliff at the threshold)."""
    return min(n / target, 1.0)


def _ramp_down(n, limit):
    """1 up to `limit`, then decays smoothly (limit/n) for longer responses."""
    return 1.0 if n <= limit else limit / n


def _lexical_variety(words, window=DISTINCT_WINDOW) -> float:
    """Moving-average type-token ratio: unique/total averaged over sliding windows.
    Plain unique/total shrinks as text grows, which penalises longer responses."""
    n = len(words)
    if n <= window:
        return len(set(words)) / max(n, 1)
    ratios = [len(set(words[i:i + window])) / window for i in range(n - window + 1)]
    return sum(ratios) / len(ratios)


def _requirement_coverage(words, prompt) -> Optional[float]:
    """Share of the prompt's numbered requirements ("1) ..., 2) ...") the response mentions.
    Each requirement is scored by the fraction of its keywords found, then averaged.
    Returns None when the prompt lists fewer than MIN_REQUIREMENTS items (nothing to check)."""
    requirements = [[w for w in re.findall(r"\w+", item.lower()) if len(w) >= 3 and w not in _REQUIREMENT_STOPWORDS]
                    for item in _REQUIREMENT_RE.findall(prompt)]
    requirements = [keywords for keywords in requirements if keywords]
    if len(requirements) < MIN_REQUIREMENTS:
        return None
    stems = {w[:REQUIREMENT_PREFIX_LEN] for w in words}
    found = [sum(k[:REQUIREMENT_PREFIX_LEN] in stems for k in keywords) / len(keywords) for keywords in requirements]
    return sum(found) / len(found)


def _source_grounding(words, source) -> float:
    """Blend of precision (response stays on source) and capped coverage (response captures source)."""
    content = lambda toks: {w for w in toks if len(w) > MIN_CONTENT_WORD_LEN}
    src = content(re.findall(r"\w+", source.lower()))
    resp = content(words)
    if not src or not resp:
        return 0.0
    overlap = len(resp & src)
    precision = overlap / len(resp)
    coverage = min(1.0, (overlap / len(src)) / TARGET_SOURCE_COVERAGE)
    return 0.5 * precision + 0.5 * coverage


def _heuristic_signal(response, category, source=None, prompt=None) -> float:
    words = re.findall(r"\w+", response.lower())
    n = len(words)
    distinct = _lexical_variety(words) # lexical variety
    # graded 0..1: how much of each feature is present, not just whether it is
    structure = _ramp_up(len(_STRUCTURE_RE.findall(response)), STRUCTURE_TARGET)
    dialogue = _ramp_up(len(_DIALOGUE_RE.findall(response)) // 2, DIALOGUE_TARGET) # 2 quote marks = 1 line
    paragraphs = _ramp_up(response.strip().count("\n\n"), PARAGRAPH_TARGET)

    w = HEURISTIC_WEIGHTS.get(category)
    if category == CATEGORY_TASK_COMPLETION:
        score = (w["min_len"] * _ramp_up(n, TASK_MIN_WORDS)
                 + w["max_len"] * _ramp_down(n, TASK_MAX_WORDS)
                 + w["distinct"] * distinct
                 + w["structure"] * structure)
        if source:
            score = (1 - GROUNDING_WEIGHT) * score + GROUNDING_WEIGHT * _source_grounding(words, source)
    elif category == CATEGORY_CREATIVE_WRITING:
        score = (w["length"] * _ramp_up(n, CREATIVE_MIN_WORDS) * _ramp_down(n, CREATIVE_MAX_WORDS)
                 + w["dialogue"] * dialogue
                 + w["distinct"] * distinct
                 + w["paragraphs"] * paragraphs)
    elif category == CATEGORY_PROBLEM_SOLVING:
        score = (w["min_len"] * _ramp_up(n, PROBLEM_MIN_WORDS)
                 + w["structure"] * structure
                 + w["distinct"] * distinct
                 + w["full_len"] * _ramp_up(n, PROBLEM_FULL_WORDS))
    else:
        score = DEFAULT_HEURISTIC

    if prompt and category in REQUIREMENT_CATEGORIES:
        coverage = _requirement_coverage(words, prompt)
        if coverage is not None:
            score = (1 - REQUIREMENT_WEIGHT) * score + REQUIREMENT_WEIGHT * coverage

    return max(0.0, min(1.0, score))


def _to_scale(blended: float) -> int:
    """Stretch the BLEND_LOW..BLEND_HIGH band over 0..1, then map to the 1-5 scale."""
    stretched = max(0.0, min(1.0, (blended - BLEND_LOW) / (BLEND_HIGH - BLEND_LOW)))
    return 1 + round(stretched * 4)


def _is_truncated(response: str) -> bool:
    """True when the response was likely cut off by the token cap.

    A truncated answer ends mid-thought rather than on terminal punctuation
    (. ! ?), so it never actually finished the task and shouldn't be eligible
    for a perfect 5 — this keeps a barely-complete answer from tying a
    fully-written one. Closing quotes/parentheses are peeled back first so a
    response that ends on a quoted line still reads as complete."""
    text = response.strip()
    if not text:
        return True
    if text[-1] in "\"”')]":
        text = text[:-1].rstrip()
    if not text:
        return True
    return text[-1] not in ".!?"


# per-category weight on EMBEDDINGS (meaning); heuristics get (1 - alpha)
ALPHA = {
    CATEGORY_TASK_COMPLETION:  0.55,   # correctness of content matters most
    CATEGORY_PROBLEM_SOLVING:  0.35,
    CATEGORY_CREATIVE_WRITING: 0.30,   # quality is subjective → trust heuristics more
}


def score_response_quality(response: str, category: str, source: Optional[str] = None,
                           prompt: Optional[str] = None) -> int:
    """
    Evaluate the quality of responses on a 1-5 scale based on task category.
    
    TODO: Complete this function to:
    1. Check response content for category-specific indicators
    2. Evaluate completeness and quality
    3. Return appropriate score based on criteria
    
    Args:
        response (str): The model's response to evaluate
        category (str): The task category (task_completion, creative_writing, problem_solving)
    
    Returns:
        int: Score from 1-5 where 5 is excellent quality, 1 is poor quality
    """

    """ 
     Commented out because the scoring is drastically refactored    
     response_lower = response.lower()
     word_count = len(response.split())
     
     # TODO: Implement scoring for task_completion category
     if category == CATEGORY_TASK_COMPLETION:
         # Hint: Look for summary indicators like "main", "key", "important"
         # Check for appropriate length and structure
         # Return score based on completeness and quality
         pass
             
     # TODO: Implement scoring for creative_writing category
     elif category == CATEGORY_CREATIVE_WRITING:
         # Hint: Look for narrative elements, dialogue, descriptive language
         # Check for story completeness and creativity
         # Consider word count and narrative structure
         pass
             
     # TODO: Implement scoring for problem_solving category
     elif category == CATEGORY_PROBLEM_SOLVING:
         # Hint: Look for strategy words, analysis terms, actionable advice
         # Check for comprehensiveness and practical value
         # Consider depth of analysis and actionability
         pass
     
     return 3  # Default middle score
     """
 
    if not response:
        return 0

    heuristic = _heuristic_signal(response, category, source, prompt)
    if not source:
        # Nothing to compare against: skip the embedding call and rely on heuristics alone
        score = _to_scale(heuristic)
    else:
        resp_emb, src_emb = embed([response, source])
        similarity = cosine(resp_emb, src_emb)
        embedding = max(0.0, min(1.0, (similarity - COSINE_LOW) / (COSINE_HIGH - COSINE_LOW))) # rescale + clip to 0..1
        alpha = ALPHA.get(category, 0.4)
        blended = (1 - alpha) * heuristic + alpha * embedding
        score = _to_scale(blended)

    # A truncated answer was cut off by the model's completion cap and never
    # finished the task, so it can't earn a perfect 5 even if its heuristics
    # scored high. This keeps a barely-complete answer from tying a full one.
    if _is_truncated(response):
        score = min(score, MAX_INCOMPLETE_SCORE)
    return score
        
 
def calculate_cost_effectiveness(quality_score: int, total_cost: float) -> float:
    """
    Calculate cost-effectiveness ratio for prompt strategies.
    
    TODO: Complete this function to:
    1. Handle zero cost cases
    2. Calculate quality points per dollar
    3. Return meaningful ratio for comparison
    
    Args:
        quality_score (int): Quality score from 1-5
        total_cost (float): Total cost in USD for the API call
    
    Returns:
        float: Cost-effectiveness ratio (higher is better)
    """
    # TODO: Handle zero cost case
    if total_cost == 0:
        return 0.0
    
    # TODO: Calculate and return cost-effectiveness ratio
    # Hint: quality_score / total_cost, multiply by 1000 for readability
    return quality_score / total_cost 


def test_prompt_strategy(category: str, strategy_type: str) -> Optional[Dict]:
    """
    Test a specific prompt strategy and analyze its cost-effectiveness.
    
    TODO: Complete this function to:
    1. Find the specified strategy from PROMPT_STRATEGIES
    2. Get the appropriate configuration
    3. Make API call and evaluate results
    4. Display comprehensive metrics
    5. Return structured results
    
    Args:
        category (str): The task category to test
        strategy_type (str): The prompt strategy type (minimal, standard, premium)
    
    Returns:
        dict: Results including quality scores, costs, and effectiveness metrics
    """
    # TODO: Find the specified strategy
    strategy_category = PROMPT_STRATEGIES.get(category)
    if not strategy_category:
        print(f"❌ Category '{category}' not found")
        return None
       
    # TODO: Get prompt info and config
    prompt_info = strategy_category.get(strategy_type)
    if not prompt_info:
        print(f"❌ Strategy type '{strategy_type}' in category {category}")
        return None
    
    config = PROMPT_CONFIGS.get(strategy_type)
    if not config:
        print(f"❌ Config not found in strategy type '{strategy_type}' in category {category}")
        return None
    
    parameters = config['parameters']
    print(f"\n🧪 TESTING PROMPT STRATEGY: {category.upper()} - {strategy_type.upper()}")
    print(f"Description: {prompt_info['description']}")
    print(f"Model: {parameters['model']} | Temp: {parameters['temperature']} | Max Completion Tokens: {parameters['max_completion_tokens']}")
    print("=" * 80)
    
    # TODO: Prepare the prompt (replace placeholder if needed)
    prompt = prompt_info["prompt"]
    # Hint: Replace [TEXT_PLACEHOLDER] with SAMPLE_TEXT if present
    if TEXT_PLACEHOLDER in prompt:
        prompt = prompt.replace(TEXT_PLACEHOLDER, SAMPLE_TEXT)
    
    # TODO: Make API call with current configuration
    result = call_openai_api(prompt, config)
    
    if result['success']:
        # TODO: Score the response quality
        source = SAMPLE_TEXT if category == CATEGORY_TASK_COMPLETION else None
        quality_score = score_response_quality(result["response"], category, source, prompt_info["prompt"])
        
        # TODO: Calculate cost-effectiveness
        cost_effectiveness = calculate_cost_effectiveness(quality_score, result["total_cost"])
        
        # TODO: Display results with clear formatting
        print(f"\n📝 RESPONSE:")
        print("-" * 60)
        print(result['response'])
        print("-" * 60)
        
        print(f"\n📊 METRICS:")
        print(f"✅ Quality Score: {quality_score}/5")
        print(f"⏱️  Latency: {result['latency_ms']}ms")
        print(f"🔢 Tokens Used: {result['tokens_used']} (Input: {result['prompt_tokens']}, Output: {result['completion_tokens']})")
        print(f"💰 Cost Breakdown:")
        print(f"   Input Cost: ${result['input_cost']:.6f}")
        print(f"   Output Cost: ${result['output_cost']:.6f}")
        print(f"   Total Cost: ${result['total_cost']:.6f}")
        print(f"   Cost per Token: ${result['cost_per_token']:.8f}")
        print(f"📈 Cost-Effectiveness: {cost_effectiveness:.2f} quality points per $")
        
        # TODO: Return structured results
        return {
            'category': category,
            'strategy_type': strategy_type,
            'response': result['response'],
            'quality_score': quality_score,
            'latency_ms': result['latency_ms'],
            'tokens_used': result['tokens_used'],
            'prompt_tokens': result['prompt_tokens'],
            'completion_tokens': result['completion_tokens'],
            'total_cost': result['total_cost'],
            'cost_effectiveness': cost_effectiveness,
            'model': parameters['model']
        }
    else:
        print(f"❌ Failed: {result['error']}")
        return None


def plot_cost_vs_quality(results: Dict, category: str) -> str:
    """Plot one point per strategy (cost vs quality) and save it as a PNG in outputs/.

    Model tiers differ in cost by orders of magnitude, so the x-axis is log-scaled; otherwise the
    cheap strategies hug one edge and the trade-off is invisible. Marker area scales with
    cost-effectiveness (quality points per dollar), so the chart shows the same ranking the
    comparison table prints. Strategies are told apart by colour and legend, not text labels,
    which collide when two strategies cost about the same. Returns the saved file path.
    """
    strategies = list(results.values())
    top = max(r["cost_effectiveness"] for r in strategies) or 1 # avoid dividing by 0
    palette = sns.color_palette("deep", len(strategies))

    with sns.axes_style("whitegrid"): # scoped: leaves matplotlib's global style untouched
        fig, ax = plt.subplots(figsize=(9, 5))
        for r, color in zip(strategies, palette):
            ax.scatter(r["total_cost"], r["quality_score"],
                       s=100 + 500 * r["cost_effectiveness"] / top, # area ~ cost-effectiveness
                       color=color, alpha=0.8, zorder=3,
                       label=f"{r['strategy_type']} (cost-effectiveness {r['cost_effectiveness']:,.0f})")

        ax.set_xscale("log")
        ax.margins(x=0.3) # keep markers off the axis edges
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks(range(1, 6))
        ax.set_xlabel("Total cost (USD, log scale)")
        ax.set_ylabel("Quality score (/5)")
        ax.set_title(f"Cost vs Quality trade-off - {category.replace('_', ' ').title()}")
        ax.legend(title="Strategy (marker size = cost-effectiveness)", loc="best")
        fig.tight_layout()

    PLOT_OUTPUT_DIR.mkdir(exist_ok=True)
    path = PLOT_OUTPUT_DIR / f"cost_vs_quality_{category}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"💾 Saved visualization: {path}")
    return str(path)


def compare_prompt_strategies(category: str):
    """
    Compare all prompt strategies for a given category and analyze trade-offs.
    
    TODO: Complete this function to:
    1. Test all three strategy types for the given category
    2. Collect and analyze results
    3. Create comparison table using pandas
    4. Provide recommendations based on different criteria
    
    Args:
        category (str): The task category to compare strategies for
    
    Returns:
        dict: Comparison results with recommendations
    """
    print(f"\n🔍 COMPARING PROMPT STRATEGIES FOR: {category.upper()}")
    print("=" * 80)
    
    results = {}
    strategy_types = [CONFIG_MINIMAL, CONFIG_STANDARD, CONFIG_PREMIUM]
    
    # TODO: Test each strategy type
    for strategy_type in strategy_types:
        # TODO: Call test_prompt_strategy and store results
        # Add small delay between calls to avoid rate limiting
        res = test_prompt_strategy(category, strategy_type)
        if res:
            results[strategy_type] = res
    
    # TODO: Analyze and display comparison
    if len(results) > 1:
        print(f"\n📊 STRATEGY COMPARISON SUMMARY:")
        print("=" * 80)
        
        # TODO: Create comparison table using pandas
        # Include: Strategy, Quality, Cost, Tokens, Cost-Effectiveness, Model
        columns = ["strategy_type", "quality_score", "total_cost", "tokens_used", "cost_effectiveness", "model"]
        table = pd.DataFrame(
            [{k: results[st].get(k) for k in columns} for st in strategy_types if st in results],
            columns=columns
        )
        print(table.to_string(index=False))
        
        # TODO: Provide recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # TODO: Find and display best cost-effectiveness
        best_ce = max(results.values(), key=lambda r: r['cost_effectiveness'])
        print(f"    Best cost-effectiveness : {best_ce['strategy_type']}")

        # TODO: Find and display highest quality
        best_q = max(results.values(), key=lambda r: r['quality_score'])
        print(f"    Highest quality         : {best_q['strategy_type']}")

        # TODO: Find and display lowest cost
        lowest_c = min(results.values(), key=lambda r: r['total_cost'])
        print(f"    Lowest cost             : { lowest_c['strategy_type'] }")

        # TODO: Visualize the cost vs quality trade-off
        plot_cost_vs_quality(results, category)        
        
        return results
    else:
        print("❌ Insufficient results for comparison")
        return results


def main():
    # TODO: Example usage - uncomment and test when ready
    # Test a single strategy
    # minimal_summary = test_prompt_strategy(CATEGORY_TASK_COMPLETION, CONFIG_MINIMAL)

    # TODO: Additional test examples you can run:
    # Compare all strategies for a category
    task_completion_results = compare_prompt_strategies("task_completion")
    creative_writing_results = compare_prompt_strategies("creative_writing")
    problem_solving_results = compare_prompt_strategies("problem_solving")

    # Single strategy tests:
    # standard_story = test_prompt_strategy("creative_writing", "standard")
    # premium_business = test_prompt_strategy("problem_solving", "premium")


if __name__ == "__main__":
    main()


"""
EXERCISE COMPLETION CHECKLIST:
✅ Import all necessary libraries
✅ Complete PROMPT_CONFIGS with appropriate models and parameters
✅ Fill in MODEL_PRICING with current OpenAI pricing
✅ Create comprehensive PROMPT_STRATEGIES for all categories and types
✅ Write substantial SAMPLE_TEXT for summarization testing
✅ Implement call_openai_api() function with cost calculation
✅ Complete score_response_quality() with category-specific logic
✅ Implement calculate_cost_effectiveness() function
✅ Complete test_prompt_strategy() function
✅ Implement compare_prompt_strategies() function
✅ Test your implementation with the example usage
✅ Add your own API key and test the complete workflow

BONUS CHALLENGES:
✅ Add visualization of cost vs quality trade-offs using matplotlib
□ Implement batch testing with multiple runs for statistical analysis
□ Add confidence intervals for quality scores
□ Create a budget optimization function that recommends strategies based on cost constraints
□ Add support for custom prompt templates
□ Implement A/B testing framework for prompt comparison
□ Add export functionality for results (CSV, JSON)
□ Create a cost forecasting tool based on usage patterns
"""
