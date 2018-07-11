# DeepMine
AI Research project with Open AI platform.

# How to run?

1. **DQN**

```cd Open-AI/DQN2```

To train the model (takes too long: ~1,5 days)

```python3 dqn.py 0```

To load the model add model number as an extra parameter. Model is already trained and output files are commited. So if you want to load final model with parameter 0:
```python3 dqn.py 0 0```

In each 100 episode model is saved with '.h5' extension. So in order to load model after 30000 episodes:
```python3 dqn.py 0 300```

2. **Prioritized Experience Replay**

```cd Open-AI/DQN-PER```

  All the other part is same with DQN's running.
  
3. **Policy Gradient**

```cd Open-AI/PG```
```python3 0.001_loss_steps.py.py```

4. **A3C**

```cd Open-AI/A3C```
```python3 raw_A3C.py```

