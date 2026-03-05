import pygame
import sys
import settings
from entities import Player, Alien, Bullet
from ui import GameUI


class MainGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("Space Invaders")
        self.clock = pygame.time.Clock()
        self.ui = GameUI()

        self.reset_game()

    def reset_game(self):
        self.score = 0
        self.state = "PLAYING"
        self.alien_direction = 1

        self.player_group = pygame.sprite.GroupSingle(Player())
        self.alien_group = pygame.sprite.Group()
        self.bullet_group = pygame.sprite.Group()

        # Reset difficulty changes
        settings.ALIEN_SPEED = 2

        self._spawn_aliens()

    def _spawn_aliens(self):
        rows = 5
        cols = 10
        x_offset = 60
        y_offset = 60

        for row in range(rows):
            for col in range(cols):
                x = 100 + col * x_offset
                y = 80 + row * y_offset
                alien = Alien(x, y)
                self.alien_group.add(alien)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()

            if self.state == "PLAYING":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.SPACE:
                        self.shoot_bullet()
            elif self.state == "GAME_OVER":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.ui.is_restart_clicked(event.pos):
                        self.reset_game()

    def shoot_bullet(self):
        # Allow a limited number of bullets on screen simultaneously
        if len(self.bullet_group) < 3:
            player = self.player_group.sprite
            if player:
                bullet = Bullet(player.rect.centerx, player.rect.top)
                self.bullet_group.add(bullet)

    def update_aliens(self):
        if not self.alien_group:
            self._spawn_aliens()
            self.score += 500  # Bonus for clearing the wave
            settings.ALIEN_SPEED += 0.5  # Increase difficulty

        move_down = False
        for alien in self.alien_group.sprites():
            if alien.rect.right >= settings.SCREEN_WIDTH or alien.rect.left <= 0:
                self.alien_direction *= -1
                move_down = True
                break

        horiz_move = self.alien_direction * settings.ALIEN_SPEED
        self.alien_group.update(horiz_move, move_down)

        for alien in self.alien_group.sprites():
            if alien.rect.bottom >= settings.SCREEN_HEIGHT:
                self.state = "GAME_OVER"
            if pygame.sprite.spritecollide(alien, self.player_group, False):
                self.state = "GAME_OVER"

    def check_collisions(self):
        hits = pygame.sprite.groupcollide(
            self.bullet_group, self.alien_group, True, True
        )
        for hit_bullet, hit_aliens in hits.items():
            self.score += 10 * len(hit_aliens)

    def update(self):
        if self.state == "PLAYING":
            keys = pygame.key.get_pressed()
            self.player_group.update(keys)
            self.bullet_group.update()

            self.update_aliens()
            self.check_collisions()

    def draw(self):
        self.screen.fill(settings.BG_COLOR)

        self.player_group.draw(self.screen)
        self.alien_group.draw(self.screen)
        self.bullet_group.draw(self.screen)

        self.ui.draw_score(self.screen, self.score)

        if self.state == "GAME_OVER":
            self.ui.draw_game_over(self.screen, self.score)

        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(settings.FPS)

    def quit_game(self):
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = MainGame()
    game.run()
