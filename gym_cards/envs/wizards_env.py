import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class WizardsEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, suits=2, max_card=5, players=2, seed=None):
    self.suits = suits
    self.max_card = max_card
    self.players = players
    self.cards_per_player = (suits * max_card) // players
  
    self.action_space = spaces.Discrete(self.cards_per_player)
    # self.action_space = spaces.Box(low=0.0, high=1.0, shape=[self.cards_per_player], dtype=np.float32)
    self.observation_space = spaces.Tuple((
      spaces.MultiDiscrete([max_card + 1, suits + 1] * self.cards_per_player), # cards in hand
      spaces.MultiDiscrete([max_card + 1, suits + 1] * players), # cards played this turn
      spaces.MultiDiscrete([self.cards_per_player + 1] * players) # scores
    ))
    
    self.seed(seed)
    self.reset()
    
  def seed(self, seed):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
      
  def step(self, action):
    assert self.action_space.contains(action)
    
    # play the card the agent has chosen
    card_id = action # .argmax()
    chosen_card = np.array(self.hands[0][card_id])
    if chosen_card[0] == 0:
      return self._lost("played card that was played before") 
    self.hands[0][card_id] = [0, 0]
    self.played_cards[0] = chosen_card
    
    # check whether the correct suit was played
    played_suits = [card[1] for card in self.played_cards[1:] if not card[1] == 0]
    if len(played_suits) > 0:
      called_suite = played_suits[0]
      if not chosen_card[1] == called_suite and len([card for card in self.hands[0] if card[1] == called_suite]) > 0:
        return self._lost("did not play suite called for")
    else:
      called_suite = chosen_card[1]
    
    # play cards by other agents until the round is finished
    for i in range(1, self.players):
      if not self.played_cards[i][0] == 0:
        continue
      self._play_card(i, called_suite)
      
    # determin winner of the round and distribute scores
    winner = max(range(self.players), key=lambda player: self.played_cards[player][0] + (1 if self.played_cards[player][1] == called_suite else 0) * self.max_card)
    self.scores[winner] += 1
    self.played_cards = [[0, 0]] * self.players
    called_suite = -1

    # set up return values
    reward = self.scores[0]
    done = max([card[0] for card in self.hands[0]]) == 0
    info = {}

    # simulate beginning of next round until the agent has to take an action
    if not done and not winner == 0:
      for i in range(max(1, winner), self.players):
        called_suite = self._play_card(i, called_suite)
    
    return self._get_observations(), reward, done, info
      
  def _lost(self, reason):
    return self._get_observations(), -100 + 10*sum(self.scores) + self.scores[0], True, {"reason": reason}

  def reset(self):
    deck = [[i, suit] for i in range(1, self.max_card + 1) for suit in range(1, self.suits + 1)]
    shuffled_deck = self.np_random.permutation(deck) 
    self.hands = shuffled_deck[:self.players * self.cards_per_player].reshape(self.players, -1, 2)
    self.scores = [0] * self.players
    self.played_cards = [[0, 0]] * self.players
    return self._get_observations()
      
  def render(self, mode='human'):
    print("scores: {}".format(self.scores))
    print("hands: {}".format(self.hands))
    print("played cards: {}".format(self.played_cards))
      
  def close(self):
    pass
    
  def _play_card(self, player, called_suite):
    # play cards left to right for now
    legal_cards = [i for i in range(self.cards_per_player) if self.hands[player][i][1] == called_suite]
    if len(legal_cards) == 0:
      legal_cards = [i for i in range(self.cards_per_player) if not self.hands[player][i][0] == 0]
    card_idx = min(legal_cards)
    self.played_cards[player] = np.array(self.hands[player][card_idx])
    self.hands[player][card_idx] = [0, 0]
    return called_suite if called_suite > 0 else self.played_cards[player][1]
      
  def _get_observations(self):
    # return (self.hands[0], self.played_cards, self.scores)
    first = min([i for i in range(self.players) if not self.played_cards[i][0] == 0], default=0)
    return (self.hands[0], self.played_cards[first:] + self.played_cards[:first], self.scores)
