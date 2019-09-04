export PYTHONUNBUFFERED=1
python rollout.py --yaml-path data/ppo-300-agents.yaml --force-rewrite --num-workers 16
wait
python rollout.py --yaml-path data/test_ablation_result.yaml --force-rewrite --num-workers 16
wait