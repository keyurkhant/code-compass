import json
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.table import Table
from codecompass.eval.runner import EvalResult

console = Console()


def print_eval_table(result: EvalResult) -> None:
    table = Table(title=f"Eval Results ({result.num_queries} queries)", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green", justify="right")
    table.add_row("Recall@1", f"{result.recall_at_1:.3f}")
    table.add_row("Recall@5", f"{result.recall_at_5:.3f}")
    table.add_row("Recall@10", f"{result.recall_at_10:.3f}")
    table.add_row("MRR", f"{result.mrr:.3f}")
    console.print(table)


def save_eval_result(result: EvalResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"eval_{ts}.json"
    data = {
        "timestamp": ts,
        "num_queries": result.num_queries,
        "aggregate": {
            "recall_at_1": result.recall_at_1,
            "recall_at_5": result.recall_at_5,
            "recall_at_10": result.recall_at_10,
            "mrr": result.mrr,
        },
        "per_query": [
            {
                "question": pq.question,
                "relevant_paths": pq.relevant_paths,
                "retrieved_paths": pq.retrieved_paths,
                "recall_at_1": pq.recall_at_1,
                "recall_at_5": pq.recall_at_5,
                "recall_at_10": pq.recall_at_10,
                "rr": pq.rr,
            }
            for pq in result.per_query
        ],
    }
    out_path.write_text(json.dumps(data, indent=2))
    return out_path
