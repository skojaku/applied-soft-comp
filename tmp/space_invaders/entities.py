import pygame
import settings


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((settings.PLAYER_WIDTH, settings.PLAYER_HEIGHT))
        self.image.fill(settings.PLAYER_COLOR)
        self.rect = self.image.get_rect(
            midbottom=(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT - 20)
        )
        self.speed = settings.PLAYER_SPEED

    def update(self, keys):
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < settings.SCREEN_WIDTH:
            self.rect.x += self.speed


class Alien(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((settings.ALIEN_WIDTH, settings.ALIEN_HEIGHT))
        self.image.fill(settings.ALIEN_COLOR)
        self.rect = self.image.get_rect(topleft=(x, y))

    def update(self, move_horizontally, move_down):
        self.rect.x += move_horizontally
        if move_down:
            self.rect.y += settings.ALIEN_DROP


class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((settings.BULLET_WIDTH, settings.BULLET_HEIGHT))
        self.image.fill(settings.BULLET_COLOR)
        self.rect = self.image.get_rect(midbottom=(x, y))

    def update(self):
        self.rect.y -= settings.BULLET_SPEED
        if self.rect.bottom < 0:
            self.kill()
