# controller.py

import numpy as np
import pygame
"""
🕹️ Key Mapping

WASD → Move joints 1 & 2

Arrow Keys → Move joints 3 & 4

Q / E → Joint 5

R / F → Joint 6

T / G → Joint 7

Y / H → Joint 8

Space → Close gripper

C → Open gripper
"""
class Controller:

    def __init__(self):
        self.gripper_closed = None

        pygame.init()
        self.screen = pygame.display.set_mode((200, 200))  # tiny window to capture key events
        pygame.display.set_caption("Keyboard Controller")

    def get_action(self):
        action = np.zeros(9)
        gripper_button_pressed = False

        # process pygame events (important for key state updates)
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # ---- Movement mapping ----
        # Left stick replacement (WASD)
        if keys[pygame.K_a]:   # left
            action[0] = -1
        if keys[pygame.K_d]:   # right
            action[0] = 1
        if keys[pygame.K_w]:   # up
            action[1] = 1
        if keys[pygame.K_s]:   # down
            action[1] = -1

        # Right stick replacement (Arrow keys)
        if keys[pygame.K_LEFT]:
            action[2] = -1
        if keys[pygame.K_RIGHT]:
            action[2] = 1
        if keys[pygame.K_UP]:
            action[3] = 1
        if keys[pygame.K_DOWN]:
            action[3] = -1

        # ---- Buttons mapping ----
        if keys[pygame.K_q]:
            action[4] = -1
            print("Q pressed")
        elif keys[pygame.K_e]:
            action[4] = 1
            print("E pressed")

        if keys[pygame.K_SPACE]:
            self.gripper_closed = True
            gripper_button_pressed = True
            print("Space (close gripper) pressed")
        elif keys[pygame.K_c]:
            self.gripper_closed = False
            gripper_button_pressed = True
            print("C (open gripper) pressed")

        if keys[pygame.K_r]:
            action[5] = 1
        elif keys[pygame.K_f]:
            action[5] = -1

        if keys[pygame.K_t]:
            action[6] = -1
        elif keys[pygame.K_g]:
            action[6] = 1

        if keys[pygame.K_y]:
            action[7] = 1
        elif keys[pygame.K_h]:
            action[7] = -1

        # ---- Apply thresholds / gripper state ----
        mask = np.abs(action) >= 0.1
        action = action * mask
        action = np.where(action == -0.0, 0.0, action)

        if np.all(action == 0) and not gripper_button_pressed:
            action = None
        else:
            if self.gripper_closed is True:
                action[7] = -1.0
                action[8] = -1.0
            elif self.gripper_closed is False:
                action[7] = 1.0
                action[8] = 1.0

        return action
