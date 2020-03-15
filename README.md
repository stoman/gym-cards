# gym-cards

This is a collection of gym environments for training reinforcement
algorithms to play simple card games. This environment
runs inside the [OpenAI Gym](https://github.com/openai/gym).

## Available environments

### gym_cards:Wizards-v0

A simple card game with thew following features:

* multiple agents (parameter `players`)
* simulated agents play legal cards in order they were dealed
* cards of multiple suits and different values (parameters `suits` and `max_card`)
* players have to play the same suit as the first player or lose the game (-100 points)
* the highest card wins a round, each round awards one point to the winner

## Running environments

Install this package using the following commands.

```
git clone https://github.com/stoman/gym-cards.git
pip3 install -e gym-cards
```

All environments can now be used as the default Gym environments.

In particular, you can run the
[baselines](http://github.com/openai/baselines) after installing them:

```
export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'
python3 -m baselines.run --env=gym_cards:Wizards-v0 --log_path=~/logs/wizards --save_path=~/models/wizards_ppo2 --num_timesteps=1e7
```

To monitor progress run tensorboard:

```
tensorboard --logdir ~/logs/wizards
```