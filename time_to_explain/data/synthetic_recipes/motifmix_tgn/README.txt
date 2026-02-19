MotifMix-TGN synthetic temporal dataset (edge-addition stream with causal ground truth)

Files:
- events.csv : event stream sorted by timestamp
- nodes.csv  : node metadata (community)

events.csv columns:
- event_id: integer (0..n_events-1)
- src, dst: node ids
- timestamp: integer time (monotone increasing, equals event_id)
- label: always 1 (positive events)
- rule_id, rule_name: generation mechanism (metadata; do NOT feed to model if you want a fair task)
- cause_event_ids: JSON list of event_ids that *caused* this edge
- n_causes: number of causal edges
- split: train/val/test (time split)

Generator params used:
- n_nodes=1000
- n_events=50000
- seed_edges=2000
- seed=42
- weights=(0.15, 0.35, 0.25, 0.15, 0.1)
- W1=200, W2=400, W3=800, W4=1200
- K=3
- communities=20, p_intra=0.7

Sanity check: in a sample of events, causes were earlier than the event:
ok=3000, bad=0 (should be bad=0)

Training note (TGN style):
- Use events in train split as positives. Sample negatives (u, v') at each t.
- Evaluate on val/test by time.
- You can also evaluate explanation: compare your model's retrieved "important past edges" to cause_event_ids.
