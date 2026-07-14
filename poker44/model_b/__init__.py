"""Model B: an independent, decorrelated Poker44 bot-detection model.

Model B is deliberately built on a different signal family from the primary
model (``poker44.model``):

* features focus on commitment / stack geometry, bet-sizing relative to the
  effective stack, raise_to / call_to magnitudes, the inter-actor response
  graph (reraise / check-raise / donk / fold-to-raise), and showdown dynamics
  -- signals the primary model's ratio / entropy / signature-replay features do
  not directly exploit;
* aggregation uses different moments (mean/std/median/p10/p90);
* the learner is a level-wise histogram GBDT (sklearn HistGradientBoosting)
  optionally blended with an attention-pooled DeepSets net (no recurrence),
  rather than a leaf-wise LightGBM + GRU;
* calibration is isotonic + an operating-point shift.

The goal is a second miner that earns on its own merits with genuinely
decorrelated predictions -- not a disguised copy of the primary model.
"""
