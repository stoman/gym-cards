from gym.envs.registration import register

register(
    id='Wizards-v0',
    entry_point='gym_cards.envs:WizardsEnv',
)