from evaluator import load_traces, compute_recall_at_k

traces = load_traces()
for t in traces:
    tid = t["id"]
    desc = t["description"]
    n = len(t["expected_assessments"])
    print(f"  {tid}: {desc}  [expected: {n} items]")

print()

# Test recall metric — 2/3 expected found
expected = ["Core Java (Advanced Level) (New)", "Java 8 (New)", "Java Design Patterns (New)"]
returned = [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/x", "test_type": "K"},
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/y", "test_type": "K"},
    {"name": "Python Basics", "url": "https://www.shl.com/z", "test_type": "K"},
]
r = compute_recall_at_k(expected, returned, k=10)
print("Recall@10 2/3 match:", round(r, 3), "(expected 0.667)")

r2 = compute_recall_at_k(expected, [], k=10)
print("Recall@10 empty:    ", round(r2, 3), "(expected 0.0)")

r3 = compute_recall_at_k([], returned, k=10)
print("Recall@10 no expect:", round(r3, 3), "(expected 1.0 vacuously)")
