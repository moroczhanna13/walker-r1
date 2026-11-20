import gymnasium as gym


class PushWrapper(gym.Wrapper):
    """Időnként vízszintesen meglöki a robot törzsét.

    force:    a lökés ereje newtonban (0 = nincs lökés)
    interval: átlagosan hány lépésenként jön lökés
    duration: hány lépésig tart egy lökés
    side:     0 = véletlen irány, +1 = mindig hátulról (előre lök), -1 = mindig elölről
    """

    def __init__(self, env, force=100.0, interval=200, duration=5, side=0):
        super().__init__(env)
        self.force = force
        self.interval = interval
        self.duration = duration
        self.side = side
        self.torso_id = env.unwrapped.model.body("torso").id
        self.steps_left = 0          # ennyi lépés van még hátra az aktuális lökésből
        self.direction = 1.0
        self.push_count = 0

    def reset(self, **kwargs):
        self.steps_left = 0
        self.push_count = 0
        self.unwrapped.data.xfrc_applied[self.torso_id, :] = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Minden lépésben 1/interval eséllyel új lökés indul
        if self.force > 0 and self.steps_left == 0 and self.np_random.random() < 1.0 / self.interval:
            self.steps_left = self.duration
            self.direction = self.side if self.side != 0 else self.np_random.choice([-1.0, 1.0])
            self.push_count += 1

        # xfrc_applied: külső erő egy testre; a 0. elem az x irányú (előre-hátra) erő
        pushing = self.steps_left > 0
        self.unwrapped.data.xfrc_applied[self.torso_id, 0] = self.direction * self.force if pushing else 0.0
        if pushing:
            self.steps_left -= 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["pushing"] = pushing
        info["push_count"] = self.push_count
        return obs, reward, terminated, truncated, info