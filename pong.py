from pyglet import clock, app
from pyglet.media import load as sound_load
from pyglet.window import Window, key
from pyglet.graphics import Batch
from pyglet.shapes import Rectangle, Circle
from pyglet.text import Label
from random import random, randint
from numpy import array, sin, cos, pi
from torch import Tensor, no_grad, multinomial, softmax, load
from create_model import Model


window_width, window_height = 600, 400
paddle_width, paddle_height = 8, 60
paddle_buffer = 25
paddle_speed = 300.0
ball_start_speed, ball_speed = 250.0, 450.0
ball_width, ball_height = 10, 10
score_x_buffer, score_y_buffer = 100, 50
max_player_y = window_height - paddle_height

collision_range1 = window_width / 10
collision_range2 = window_width * 9 / 10

game_state = 0
states = {0: 'main menu', 1: 'playing game', 2: 'paused', 3: 'game over', 4: 'scored'}

window = Window(width=window_width, height=window_height, caption="Pong")
keys = key.KeyStateHandler()
window.push_handlers(keys)

reset_delay = 15
reset_timer = reset_delay

game_batch = Batch()
menu_batch = Batch()

one_player = Label('1 player', x=284, y=200, batch=menu_batch)
two_player = Label('2 player', x=284, y=170, batch=menu_batch)
selector_offset = 6
selector = Circle(x=one_player.x - 30, y=one_player.y + selector_offset, radius=5.0, batch=menu_batch)
select_up = False
select_down = False
select_position = 0
chosen_mode = False

p1_controls = Label('Player 1 Controls - W and S keys', x=35, y=90, batch=menu_batch)
p2_controls = Label('Player 2 Controls - Up and Down keys', x=35, y=60, batch=menu_batch)
extra_controls = Label('Press P while playing to pause. And press Escape to terminate the program.', x=35, y=30,
                       batch=menu_batch)
title = Label('PONG', x=230, y=300, batch=menu_batch, font_size=40)
select_text = Label('Hit Enter to select an option.', x=200, y=270, batch=menu_batch)

over_delay = 2 * 60
over_timer = over_delay
over_text = Label(x=204, y=200, batch=game_batch, font_size=25)
over_text.visible = False

pause_text1 = Label('Paused.', x=274, y=230, batch=game_batch)
pause_text2 = Label('Press P to resume.', x=236, y=200, batch=game_batch)
pause_text3 = Label('Hit enter to return to main menu.', x=184, y=170, batch=game_batch)
score_font = 25


def pause(visible):
    global pause_text1, pause_text2, pause_text3
    pause_text1.visible = visible
    pause_text2.visible = visible
    pause_text3.visible = visible


def rect_collision(rect1, rect2):
    if rect2.x < rect1.x + rect1.width and rect1.x < rect2.x + rect2.width and rect2.y < rect1.y + rect1.height \
            and rect1.y < rect2.y + rect2.height:
        return True
    return False


def clip(x, a, b):
    # a < b
    return max(min(x, b), a)


class Player:
    def __init__(self, is_left):
        self.is_ai = False
        self.up_pressed = False
        self.down_pressed = False
        self.velocity = 0.0
        self.score = 0
        if is_left:
            self.score_label = Label('0', x=window_width//2 - score_x_buffer, y=window_height - score_y_buffer,
                                     batch=game_batch, font_size=score_font)
            start_x, self.start_y = paddle_buffer, window_height // 2 - paddle_height / 2
            self.rect = Rectangle(start_x, self.start_y, paddle_width, paddle_height, batch=game_batch)
        else:
            self.score_label = Label('0', x=window_width//2 + score_x_buffer, y=window_height - score_y_buffer,
                                     batch=game_batch, font_size=score_font)
            start_x, self.start_y = window_width - paddle_buffer, window_height // 2 - paddle_height / 2
            self.rect = Rectangle(start_x, self.start_y, paddle_width, paddle_height, batch=game_batch)
        self.is_left = is_left

    def reset(self):
        self.rect.y = self.start_y

    def update_score(self):
        self.score += 1
        self.score_label.text = str(self.score)

    def restart(self):
        self.score = 0
        self.score_label.text = '0'
        self.reset()

    def move(self):
        global keys, paddle_speed
        if self.up_pressed and not self.down_pressed:
            self.velocity = paddle_speed
        elif not self.up_pressed and self.down_pressed:
            self.velocity = - paddle_speed
        else:
            self.velocity = 0.0


class AI(Player):
    def __init__(self, is_left, ver=''):
        super().__init__(is_left)
        self.is_ai = True
        self.model = Model()
        self.model.load_state_dict(load(ver, weights_only=True))
        self.model.eval()

    def action(self, feature):
        # 0 is don't move, 1 is move up and 2 is move down
        global paddle_speed
        feature = Tensor(feature)
        with no_grad():
            logits = self.model.forward(feature.view((1, -1)))
            p = softmax(logits, dim=-1)
            out = multinomial(p, num_samples=1).item()
        if out == 0:
            self.velocity = 0
        elif out == 1:
            self.velocity = paddle_speed
        else:
            self.velocity = -paddle_speed


class Ball:
    def __init__(self):
        self.start_x, self.start_y = window_width // 2 - ball_width // 2, window_height // 2 - ball_height // 2
        self.rect = Rectangle(self.start_x, self.start_y, ball_width, ball_height, batch=game_batch)
        self.velocity = array([0, 0])
        self.show_target = False
        self.start_angle = pi / 24

        max_angle = pi / 3
        self.m = (2 * max_angle) / (ball_height + paddle_height)
        self.b = -max_angle + ball_height * self.m
        self.can_hit_left = 1
        self.sounds = []
        for i in range(1, 7):
            self.sounds.append(sound_load(f"sound effects\\pong{i}.mp3", streaming=False))
        self.reset()

    def reset(self, go_left=None):
        global ball_speed
        a = (2 * random() - 1) * self.start_angle
        if go_left is None:
            self.can_hit_left = randint(0, 1)
            a += self.can_hit_left * pi
        elif go_left:
            a += pi

        self.velocity = array([cos(a), sin(a)]) * ball_start_speed
        self.rect.x, self.rect.y = self.start_x, self.start_y

    def was_hit(self, player_object):
        global ball_speed
        self.can_hit_left = (1 + self.can_hit_left) % 2
        a = (self.rect.y - player_object.rect.y) * self.m + self.b

        if not player_object.is_left:
            a = pi - a
            self.rect.x = player_object.rect.x - player_object.rect.width
        else:
            self.rect.x = player_object.rect.x + player_object.rect.width

        self.velocity = array([cos(a), sin(a)]) * ball_speed
        self.sounds[randint(0, 5)].play()

    def flip_y(self):
        global window_height
        self.velocity[1] = - self.velocity[1]
        self.rect.y = clip(self.rect.y, 0, window_height - self.rect.height)
        self.sounds[randint(0, 5)].play()


points_to_win = 11
pause(False)
ball = Ball()
player1 = Player(1)
player2 = Player(0)


@window.event
def on_key_press(symbol, modifiers):
    global game_state, player1, player2, select_up, select_down, chosen_mode, ball
    if game_state == 0:
        if symbol == key.W or symbol == key.UP:
            select_up = True
        elif symbol == key.S or symbol == key.DOWN:
            select_down = True
        elif symbol == key.ENTER:
            chosen_mode = True
    else:
        if symbol == key.P:
            if game_state == 1:
                game_state = 2
                pause(True)
                ball.rect.visible = False
            elif game_state == 2:
                game_state = 1
                pause(False)
                ball.rect.visible = True
        elif symbol == key.ENTER and game_state == 2:
            game_state = 0
            pause(False)
            player1.restart()
            player2.restart()
            ball.reset()
            ball.rect.visible = True
        elif symbol == key.W:
            player1.up_pressed = True
        elif symbol == key.S:
            player1.down_pressed = True
        elif not player2.is_ai:
            if symbol == key.UP:
                player2.up_pressed = True
            elif symbol == key.DOWN:
                player2.down_pressed = True


@window.event
def on_key_release(symbol, modifiers):
    global player1, player2, select_down, select_up, game_state
    if game_state == 0:
        if symbol == key.W or symbol == key.UP:
            select_up = False
        elif symbol == key.S or symbol == key.DOWN:
            select_down = False
    else:
        if symbol == key.W:
            player1.up_pressed = False
        elif symbol == key.S:
            player1.down_pressed = False
        elif not player2.is_ai:
            if symbol == key.UP:
                player2.up_pressed = False
            elif symbol == key.DOWN:
                player2.down_pressed = False


@window.event
def on_draw():
    global game_state
    window.clear()
    if game_state == 0:
        menu_batch.draw()
    else:
        game_batch.draw()


def update_menu():
    global select_up, select_down, select_position, selector, one_player, two_player, game_state, player2, chosen_mode, selector_offset
    if not select_up and not select_down:
        pass
    elif select_up:
        select_position = (select_position + 1) % 2
        select_up = False
    else:
        select_position = (select_position - 1) % 2
        select_down = False

    if select_position == 0:
        selector.y = one_player.y + selector_offset
    else:
        selector.y = two_player.y + selector_offset

    if chosen_mode:
        game_state = 1
        chosen_mode = False
        if select_position == 0:
            player2 = AI(0, 'final_model.pth')
        else:
            player2 = Player(0)


def scored(left_scored):
    global player1, player2, points_to_win, ball, game_state, over_text
    game_state = 4

    if left_scored:
        player1.update_score()
        if player1.score == points_to_win:
            game_state = 3
            over_text.text = 'Player 1 wins!'
    else:
        player2.update_score()
        if player2.score == points_to_win:
            game_state = 3
            over_text.text = 'Player 2 wins!'

    if game_state == 3:
        over_text.visible = True
        ball.reset()
        ball.rect.visible = False
    else:
        ball.reset(go_left=not left_scored)
        player1.reset()
        player2.reset()


def update_game(dt):
    global player1, player2, ball, points_to_win, game_state, max_player_y, collision_range1, collision_range2, \
        paddle_speed, keys, reset_timer, reset_delay, over_timer, over_delay

    if game_state == 3:
        over_timer -= 1
        if over_timer < 0:
            game_state = 2
            pause(True)
            over_timer = over_delay
            over_text.visible = False
            player1.restart()
            player2.restart()
    elif game_state == 4:
        reset_timer -= 1
        if reset_timer < 0:
            game_state = 1
            reset_timer = reset_delay
    else:
        if player2.is_ai:
            state = [player2.rect.y / window_height, ball.rect.x / window_width, ball.rect.y / window_height,
                     ball.velocity[0] / ball_speed, ball.velocity[1] / ball_speed]
            player2.action(state)
        else:
            player2.move()

        player1.move()

        # move paddles up / down
        player1.rect.y = clip(player1.rect.y + player1.velocity * dt, 0, max_player_y)
        player2.rect.y = clip(player2.rect.y + player2.velocity * dt, 0, max_player_y)

        # move ball
        ball.rect.x += ball.velocity[0] * dt
        ball.rect.y += ball.velocity[1] * dt

        # check if either player has scored and reset or end the game
        if ball.rect.x < ball.rect.width:
            scored(False)
        elif ball.rect.x > window_width - ball.rect.width:
            scored(True)

        # look for collisions if ball is close enough to either paddle
        if ball.rect.x < collision_range1 and ball.can_hit_left and rect_collision(player1.rect, ball.rect):
            ball.was_hit(player1)
        elif ball.rect.x > collision_range2 and not ball.can_hit_left and rect_collision(player2.rect, ball.rect):
            ball.was_hit(player2)

        # reflect y-component of ball velocity if it hits the top/bottom of the screen
        if ball.rect.y < 0 or ball.rect.y > window_height - ball.rect.height:
            ball.flip_y()


def update(dt):
    global game_state
    if game_state == 0:
        update_menu()
    elif game_state == 2:
        pass
    else:
        update_game(dt)


clock.schedule_interval(update, 1/60.0)
app.run()
