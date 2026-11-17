import pyglet
from pyglet import shapes
from random import random
from numpy import array, sin, cos, pi, floor, ceil, zeros, arange, asarray
from numpy.random import shuffle
from torch import save, load, Tensor, tensor, no_grad, zeros as torch_zeros, multinomial, float as torch_float, \
    long as torch_long, sum as torch_sum, clamp, minimum, log
from torch.optim import Adam, AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import mse_loss
from Game_AI.Pong.create_model import Model as Actor, Critic

output_file = 'version1.pth'
replay_buffer = []
trajectory_count = 0

actor, critic = Actor(), Critic()
actor_optim = Adam(actor.parameters(), lr=3e-4)
critic_optim = AdamW(critic.parameters(), lr=1e-4)

actor.eval()
critic.eval()

game_width, game_height = 600, 400
paddle_width, paddle_height = 8, 60
paddle_buffer = 25
paddle_speed = 300.0
score_x_buffer, score_y_buffer = 50, 25

ball_start_speed, ball_speed = 250.0, 450.0
ball_width, ball_height = 10, 10
max_angle = pi / 3
m = (2 * max_angle) / (ball_height + paddle_height)
b = -max_angle + ball_height * m
start_angle = pi / 24
white, green, red = (255, 255, 255), (0, 255, 0), (255, 0, 0)

hit_reward = 1.0
scored_on_reward = -1.0
track_reward_scale = 0.01

max_buffer_size = 2000
iterations = 0
max_iterations = 250
num_inputs = 5

grid_x, grid_y = 3, 2
window_width = grid_x * game_width
window_height = grid_y * game_height
window = pyglet.window.Window(width=window_width, height=window_height, caption="Pong AI")
batch = pyglet.graphics.Batch()


def clip(x, a, c):
    # a < c
    return max(min(x, c), a)


def rect_collision(rect1, rect2):
    if rect2.x < rect1.x + rect1.width and rect1.x < rect2.x + rect2.width and rect2.y < rect1.y + rect1.height \
            and rect1.y < rect2.y + rect2.height:
        return True
    return False


def scaled_distance(x, y):
    # 1 = max distance
    return abs(x - y) / game_height


def reinforcement_training(data, temperature=1.0, gamma=0.99, ent=0.01, eps=0.2, kl=0.015, c_epochs=10, a_epochs=5):
    global actor, critic, actor_optim, critic_optim, num_inputs
    print('\rTraining...', end='')
    s = tensor(data[:, 0:num_inputs], dtype=torch_float)
    r = data[:, num_inputs + 1]
    t = data[:, -1]
    a = tensor(data[:, num_inputs], dtype=torch_long)
    n_trajectories = int(data[-1, -1]) + 1
    del data
    count = 0
    future_r = zeros(s.shape[0])
    for k in range(n_trajectories):
        idx = t == k
        r_k = r[idx]
        for v in range(r_k.shape[0]):
            future_r[count + v] = (r_k[v:] * gamma ** arange(r_k.shape[0] - v)).sum()
        count += idx.sum()

    r = Tensor(r)
    future_r = Tensor(future_r)
    indices = arange(r.shape[0])
    shuffle(indices)
    n = round(0.9 * r.shape[0])
    train = indices[0:n]
    val = indices[n:]
    del indices

    train_dataset = TensorDataset(s[train], future_r[train])
    val_dataset = TensorDataset(s[val], future_r[val])

    train_loader = DataLoader(train_dataset, batch_size=64, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)

    min_loss = float('inf')
    for _ in range(c_epochs):
        critic.train()
        val_loss = 0.0
        for s_b, y_b in train_loader:
            loss = mse_loss(critic.forward(s_b).view(y_b.shape), y_b)

            critic_optim.zero_grad(set_to_none=True)
            loss.backward()
            critic_optim.step()

        with no_grad():
            critic.eval()
            for s_b, y_b in val_loader:
                loss = mse_loss(critic.forward(s_b).view(y_b.shape), y_b)
                val_loss += loss.item() * s_b.shape[0]

        val_loss /= val.shape[0]
        if val_loss < min_loss:
            min_loss = val_loss
            save(critic.state_dict(), "critic_versions/critic.pth")
            save(critic_optim.state_dict(), 'critic_versions/optim.pth')

    critic.load_state_dict(load('critic_versions/critic.pth', weights_only=True))
    critic_optim.load_state_dict(load('critic_versions/optim.pth', weights_only=True))
    del train_loader, train_dataset, val_loader, val_dataset, train, val

    advantage = torch_zeros(t.shape[0] - n_trajectories)
    indices = []
    count = 0

    for k in range(n_trajectories):
        idx = t == k
        r_k = r[idx]
        s_k = s[idx]

        indices.extend(list(asarray(idx).nonzero()[0][:-1]))

        with no_grad():
            value = critic.forward(s_k).view(r_k.shape)
            advantage[count: count + r_k.shape[0] - 1] = r_k[:-1] + (gamma * value[1:] - value[:-1])
        count += r_k.shape[0] - 1

    with no_grad():
        advantage = (advantage - advantage.mean()) / advantage.std()
    del t, future_r, r
    s = s[indices]
    a = a[indices]

    actor.eval()
    with no_grad():
        logits = actor.forward(s)
    dataset = TensorDataset(s[-4000:], a[-4000:], logits[-4000:], advantage[-4000:])
    loader = DataLoader(dataset, batch_size=64, drop_last=True, shuffle=True)
    actor.train()

    for _ in range(a_epochs):
        kl_est = 0.0
        for s_b, a_b, L_b, A_b in loader:
            idx = arange(a_b.shape[0])
            actor_p = actor.forward(s_b).softmax(dim=-1)
            old_p = (L_b / temperature).softmax(dim=-1)

            ratio = actor_p[idx, a_b] / old_p[idx, a_b]

            clipped = clamp(ratio, 1 - eps, 1 + eps)
            entropy = -torch_sum(actor_p * log(actor_p + 1e-8), dim=-1).mean()
            objective = -(minimum(ratio * A_b, clipped * A_b).mean()) - ent * entropy

            actor_optim.zero_grad(set_to_none=True)
            objective.backward()
            actor_optim.step()

            with no_grad():
                kl_est += torch_sum(actor_p * (log(actor_p + 1e-8) -
                                               log(L_b.softmax(dim=-1) + 1e-8)), dim=-1).mean().item()

        if kl_est >= kl:
            break
    actor.eval()
    print('\rTraining complete!', end='')


class AI:
    def __init__(self, start_x, start_y, is_left):
        self.velocity = 0.0
        self.reward_sum = 0.0
        self.reward = 0

        self.trajectory = []

        self.start_y = start_y

        self.rect = shapes.Rectangle(start_x, start_y, paddle_width, paddle_height, batch=batch)
        self.is_left = is_left

        self.model = actor

    def action(self, feature):
        # 0 is don't move, 1 is move up and 2 is move down
        global paddle_speed, game_height
        with no_grad():
            feature_tensor = tensor(feature, dtype=torch_float).view((1, -1))
            logits = self.model.forward(feature_tensor)
            p = logits.softmax(dim=-1)
            out = multinomial(p, num_samples=1)

        if out == 0:
            self.velocity = 0
        elif out == 1:
            self.velocity = paddle_speed
        else:
            self.velocity = -paddle_speed

        self.trajectory.append(feature + [out.item()])

    def reset(self):
        self.rect.y = self.start_y
        self.trajectory.clear()
        self.reward = 0.0
        self.reward_sum = 0.0
        self.velocity = 0

    def update_reward(self):
        self.reward_sum += self.reward
        self.trajectory[-1].append(self.reward)


class Ball:
    def __init__(self, start_x, start_y):
        self.velocity = array([0.0, 0.0])
        self.rect = shapes.Rectangle(start_x, start_y, ball_width, ball_height, batch=batch)
        self.target_rect = shapes.Rectangle(0, 0, ball_width, ball_height, batch=batch)
        self.start_x, self.start_y = start_x, start_y
        self.x, self.y = start_x - game_width // 2, start_y - game_height // 2
        self.serving = True
        self.can_hit_left = 0
        self.angle_reward = 0.0
        self.reset()

    def find_target(self):
        global game_height, game_width, paddle_width, ball_height, ball_width
        # calculates where the ball will be when a player will be able to hit it
        self.target_rect.visible = True
        a = game_width - paddle_buffer + paddle_width
        z = self.velocity[1] * (a - self.rect.x + self.x) / self.velocity[0] + (self.rect.y - self.y)
        n = floor(z / game_height)
        if n % 2 == 0:
            k = 1
        else:
            k = -1

        if n >= 0:
            self.target_rect.y = k * z - 2 * game_height * k * ceil(n / 2)
        else:
            self.target_rect.y = k * (z + 2 * game_height * ceil((-n - 1) / 2))
        self.target_rect.y = clip(self.target_rect.y, 0, game_height - ball_height) + self.y
        self.target_rect.x = a + self.x

    def reset(self):
        global ball_speed, start_angle
        self.serving = True
        a = (2 * random() - 1) * start_angle
        self.can_hit_left = 0

        self.velocity = array([cos(a), sin(a)]) * ball_start_speed
        self.rect.x, self.rect.y = self.start_x, self.start_y
        self.find_target()

    def was_hit(self, player_object):
        global m, b, ball_speed, max_angle
        self.can_hit_left = 1
        self.serving = False
        a = (self.rect.y - player_object.rect.y) * m + b
        self.angle_reward = abs(a / max_angle)

        if not player_object.is_left:
            a = pi - a

        self.velocity = array([cos(a), sin(a)]) * ball_speed
        self.target_rect.visible = False

    def hit_wall(self):
        global max_angle, ball_speed, ball_start_speed
        a = (2 * random() - 1) * max_angle
        if random() < 0.9:
            self.velocity = array([cos(a), sin(a)]) * ball_speed
        else:
            self.velocity = array([cos(a), sin(a)]) * ball_start_speed
        self.target_rect.visible = True
        self.find_target()
        self.can_hit_left = 0


class Game:
    def __init__(self, n_x, n_y):
        self.x = game_width * n_x
        self.y = game_height * n_y
        paddle_y = self.y + game_height // 2 - paddle_height // 2
        self.player2 = AI(self.x + game_width - paddle_buffer, paddle_y, 0)
        self.ball = Ball(self.x + game_width // 2 - ball_width // 2, self.y + game_height // 2 - ball_height // 2)
        self.max_x = self.x + game_width
        self.max_y = self.y + game_height

        self.score2 = 0
        self.score_label2 = pyglet.text.Label('0', x=self.x + game_width // 2 + score_x_buffer,
                                              y=self.y + game_height - score_y_buffer, batch=batch)

        self.reward_sum2 = pyglet.text.Label('0.00', x=self.x + game_width // 2 + score_x_buffer,
                                             y=self.y + game_height - 2 * score_y_buffer, batch=batch)

    def calculate_reward(self):
        global paddle_height, ball_height, track_reward_scale, ball_width, paddle_width, game_height

        if self.ball.velocity[0] < 0:
            if (self.player2.rect.y - self.y < game_height / 20 and self.player2.velocity <= 0) or (self.player2.rect.y + paddle_height - self.y > game_height * 19 / 20 and self.player2.velocity >= 0):
                self.player2.reward = -1.0
            else:
                self.player2.reward = 0.0
        else:
            if self.player2.rect.y < self.ball.target_rect.y < self.player2.rect.y + paddle_height or \
                    self.player2.rect.y < self.ball.target_rect.y + ball_height < self.player2.rect.y + paddle_height:
                self.player2.reward = 1.0
            elif self.player2.rect.y > self.ball.target_rect.y + ball_height:
                if self.player2.velocity < 0:
                    self.player2.reward = 1.0
                else:
                    self.player2.reward = -1.0
            else:
                if self.player2.velocity > 0:
                    self.player2.reward = 1.0
                else:
                    self.player2.reward = -1.0

        self.player2.reward *= track_reward_scale

    def add_trajectory(self):
        global trajectory_count
        for t in self.player2.trajectory:
            t.append(trajectory_count)
        replay_buffer.extend(self.player2.trajectory)
        trajectory_count += 1

    def point_scored(self):
        global replay_buffer, max_buffer_size, trajectory_count, scored_on_reward, max_iterations, iterations
        self.player2.reward += scored_on_reward
        self.player2.update_reward()

        self.add_trajectory()

    def ball_hit(self, player_object):
        global hit_reward
        self.ball.was_hit(player_object)
        player_object.reward += hit_reward  # + self.ball.angle_reward

        self.score2 += 1
        if self.score2 >= 30:
            self.player2.update_reward()
            self.add_trajectory()
            self.player2.trajectory.clear()
        self.score_label2.text = str(self.score2)

    def restart(self):
        self.ball.reset()
        self.player2.reset()
        self.score2 = 0
        self.score_label2.text = '0'

    def reset(self):
        self.point_scored()
        self.restart()

    def update_game(self, dt):
        global game_width, ball_width, ball_height, paddle_height, hit_reward, \
            paddle_buffer, paddle_width, white, green, red
        dt = min(dt, 1 / 60)

        state = [self.player2.rect.y - self.y, self.ball.rect.x - self.x,
                 self.ball.rect.y - self.y, self.ball.velocity[0], self.ball.velocity[1]]
        state[0], state[1], state[2] = state[0] / game_height, state[1] / game_width, state[2] / game_height
        state[3], state[4] = state[3] / ball_speed, state[4] / ball_speed

        self.player2.action(state)
        self.player2.rect.y = clip(self.player2.rect.y + dt * self.player2.velocity, self.y, self.max_y - paddle_height)

        self.ball.rect.x += self.ball.velocity[0] * dt
        self.ball.rect.y += self.ball.velocity[1] * dt

        if self.ball.rect.y < self.y or self.ball.rect.y > self.max_y - ball_height:
            self.ball.velocity[1] = - self.ball.velocity[1]
            self.ball.rect.y = clip(self.ball.rect.y, self.y, self.max_y - ball_height)
            if not self.ball.can_hit_left:
                self.ball.find_target()

        self.calculate_reward()
        if 0.0 == self.player2.reward:
            self.player2.rect.color = white
        elif self.player2.reward > 0.0:
            self.player2.rect.color = green
        else:
            self.player2.rect.color = red

        # hit wall
        if self.ball.rect.x < self.x + paddle_buffer + paddle_width:
            self.ball.hit_wall()

        # contact/miss check
        if (not self.ball.can_hit_left) and self.ball.rect.x > self.max_x - game_width / 10 and \
                rect_collision(self.player2.rect, self.ball.rect):
            self.ball_hit(self.player2)

        if self.ball.rect.x > self.player2.rect.x + paddle_width:
            self.reset()
        else:
            self.player2.update_reward()

        self.reward_sum2.text = str(round(self.player2.reward_sum, 2))


game_list = []
for i in range(grid_x):
    for j in range(grid_y):
        game_list.append(Game(i, j))


@window.event
def on_draw():
    window.clear()
    batch.draw()


def update(dt):
    global iterations, replay_buffer, max_buffer_size, max_iterations, trajectory_count, actor, output_file
    print(f'\riteration: {iterations}, buffer size: {len(replay_buffer)} out of {max_buffer_size}', end='')

    if len(replay_buffer) >= max_buffer_size:
        # state_t (5), action_t (1), reward_t (1), trajectory_count (1)
        reinforcement_training(array(replay_buffer))
        replay_buffer.clear()
        iterations += 1
        if iterations == max_iterations:
            save(actor.state_dict(), output_file)
            pyglet.app.exit()
        trajectory_count = 0
        for game in game_list:
            game.restart()
    else:
        for game in game_list:
            game.update_game(dt)


pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()
