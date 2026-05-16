"""Tools the Researcher (Strategy Agent) can call.

Three tools:
  - tavily_search       — general web search for ML practice / blog posts
  - arxiv_search        — academic paper search
  - estimate_param_count — quick parameter-count estimator (so the LLM can
                           sanity-check that a proposed arch fits the envelope
                           BEFORE recommending it)

Each tool is exposed in two forms:
  - a plain Python function (callable directly for tests / smoke)
  - an OpenAI-style JSON-schema definition in `RESEARCH_TOOL_SCHEMAS` (passed
    to Nemotron via the `tools=` parameter)

`RESEARCH_TOOL_DISPATCH` maps tool name → function for the LLM client's
tool-use loop to dispatch into.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from config import TAVILY_API_KEY


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: Tavily web search
# ---------------------------------------------------------------------------
def tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """General web search. Returns a slim payload the LLM can read fast."""
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY not set in .env"}
    try:
        from tavily import TavilyClient
    except ImportError:
        return {"error": "tavily-python not installed"}

    max_results = max(1, min(int(max_results), 10))
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    results = []
    for r in resp.get("results", [])[:max_results]:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content", "") or "")[:400],
        })
    return {"query": query, "results": results}


# ---------------------------------------------------------------------------
# Tool 2: arXiv search
# ---------------------------------------------------------------------------
def arxiv_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Academic paper search via arXiv. Returns title, url, authors, abstract."""
    try:
        import arxiv
    except ImportError:
        return {"error": "arxiv package not installed"}

    max_results = max(1, min(int(max_results), 10))
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        results_iter = arxiv.Client().results(search)
        results = []
        for r in results_iter:
            results.append({
                "title": r.title,
                "url": r.entry_id,
                "authors": [a.name for a in r.authors[:5]],
                "published": str(r.published.date()) if r.published else None,
                "abstract": (r.summary or "")[:400],
            })
        return {"query": query, "results": results}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Tool 3: parameter-count estimator
# ---------------------------------------------------------------------------
def estimate_param_count(
    arch_family: str,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Order-of-magnitude parameter-count estimate.

    Lets the LLM quickly check "does this architecture fit my envelope?"
    before recommending it. Not a real model build — just heuristics.
    """
    hp = hyperparams or {}
    f = (arch_family or "").lower().strip()

    # Gradient-boosted trees: rough "tree-size × n_trees"
    if f in {"xgboost", "lightgbm", "catboost", "gradient_boost", "gbm"}:
        n_est = int(hp.get("n_estimators", 200))
        max_depth = int(hp.get("max_depth", 6))
        # ~ #leaves per tree × #trees. Cap depth to avoid silly numbers.
        params = n_est * (2 ** min(max_depth, 12))
        return {
            "arch_family": arch_family,
            "estimated_params": params,
            "note": "GBT 'params' counted as total tree leaves; not directly comparable to NN params.",
        }

    if f in {"logistic_regression", "logreg", "linear"}:
        n_features = int(hp.get("n_features", 100))
        return {
            "arch_family": arch_family,
            "estimated_params": n_features + 1,
        }

    if f in {"mlp", "feedforward", "fcn"}:
        # Sum of layer weights+biases
        n_features = int(hp.get("n_features", 100))
        hidden = hp.get("hidden_sizes", [128, 64])
        if isinstance(hidden, int):
            hidden = [hidden]
        n_classes = int(hp.get("n_classes", 2))
        prev = n_features
        total = 0
        for h in hidden:
            total += prev * int(h) + int(h)
            prev = int(h)
        total += prev * n_classes + n_classes
        return {"arch_family": arch_family, "estimated_params": total}

    # Common CNNs / ViTs — published param counts
    known = {
        "resnet18":          11_700_000,
        "resnet34":          21_800_000,
        "resnet50":          25_600_000,
        "resnet101":         44_500_000,
        "efficientnet_b0":   5_300_000,
        "efficientnet_b1":   7_800_000,
        "efficientnet_b3":   12_000_000,
        "mobilenet_v2":      3_500_000,
        "vit_tiny":          5_700_000,
        "vit_small":         22_000_000,
        "vit_base":          86_000_000,
        "vit_large":         307_000_000,
        "convnext_tiny":     28_600_000,
        "convnext_small":    50_200_000,
    }
    if f in known:
        return {"arch_family": arch_family, "estimated_params": known[f]}

    return {
        "arch_family": arch_family,
        "estimated_params": None,
        "note": f"Unknown architecture family {arch_family!r}. "
                "Recognized: xgboost, lightgbm, catboost, gbm, logistic_regression, "
                "mlp, resnet{18,34,50,101}, efficientnet_b{0,1,3}, mobilenet_v2, "
                "vit_{tiny,small,base,large}, convnext_{tiny,small}.",
    }


# ---------------------------------------------------------------------------
# OpenAI tool-schema definitions (consumed by Nemotron via `tools=` kwarg)
# ---------------------------------------------------------------------------
RESEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "Search the web for ML practice, blog posts, library comparisons, "
                "recent benchmark reports. Use this for breadth: 'which library is "
                "people using for tabular churn in 2025?' Returns up to N hits "
                "with title + url + 400-char snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Be specific — mention the task, the data shape, the year if recency matters.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (1-10).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search arXiv for academic papers. Use this for depth: find the "
                "canonical paper on a method, get the precise architecture details, "
                "cite something credible. Returns up to N papers with title, url, "
                "authors, published date, and 400-char abstract."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for arXiv. Method names, model families, technique terms work well.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (1-10).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_param_count",
            "description": (
                "Estimate the parameter count of a proposed architecture. ALWAYS "
                "call this before recommending an architecture so you can verify "
                "it fits the hardware envelope. Returns an order-of-magnitude "
                "estimate. For GBT models the 'params' are total tree leaves "
                "(rough). For NN models it's actual weight count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arch_family": {
                        "type": "string",
                        "description": (
                            "Architecture family. Examples: xgboost, lightgbm, "
                            "catboost, logistic_regression, mlp, resnet18, resnet50, "
                            "efficientnet_b0, vit_base, mobilenet_v2."
                        ),
                    },
                    "hyperparams": {
                        "type": "object",
                        "description": (
                            "Hyperparameters that affect size. For trees: "
                            "n_estimators, max_depth. For MLPs: n_features, "
                            "hidden_sizes (list), n_classes. For pre-built CNN/ViT "
                            "families this can be empty {}."
                        ),
                    },
                },
                "required": ["arch_family", "hyperparams"],
                "additionalProperties": False,
            },
        },
    },
]


RESEARCH_TOOL_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "tavily_search": tavily_search,
    "arxiv_search": arxiv_search,
    "estimate_param_count": estimate_param_count,
}
