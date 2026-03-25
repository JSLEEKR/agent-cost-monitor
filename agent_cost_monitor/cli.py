import sys
from .tracker import CostTracker


def demo():
    tracker = CostTracker(budget=1.00)

    tracker.record("claude-sonnet-4-6", input_tokens=2000, output_tokens=800)
    tracker.record("gpt-4o-mini", input_tokens=5000, output_tokens=2000)
    tracker.record("claude-sonnet-4-6", input_tokens=1500, output_tokens=600)
    tracker.record("gemini-2.5-flash", input_tokens=3000, output_tokens=1200)
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=400)

    print(tracker.report())
    print()
    print("--- JSON export (first 3 lines) ---")
    json_lines = tracker.to_json().splitlines()
    for line in json_lines[:3]:
        print(line)
    print("...")
    print()
    print("--- CSV export ---")
    print(tracker.to_csv())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        print("Usage: python -m agent_cost_monitor <command>")
        print()
        print("Commands:")
        print("  demo    Run a quick demo showing the tracker in action")


if __name__ == "__main__":
    main()
