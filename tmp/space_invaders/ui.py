import pygame
import settings


class GameUI:
    def __init__(self):
        pygame.font.init()
        self.title_font = pygame.font.SysFont("arial", 64, bold=True)
        self.score_font = pygame.font.SysFont("arial", 32, bold=True)
        self.button_font = pygame.font.SysFont("arial", 28, bold=True)

        self.restart_rect = pygame.Rect(0, 0, 200, 50)
        self.restart_rect.center = (
            settings.SCREEN_WIDTH // 2,
            settings.SCREEN_HEIGHT // 2 + 50,
        )

    def draw_score(self, screen, score):
        score_surf = self.score_font.render(
            f"SCORE: {score}", True, settings.TEXT_COLOR
        )
        screen.blit(score_surf, (20, 20))

    def draw_game_over(self, screen, score):
        # Semi-transparent overlay
        overlay = pygame.Surface(
            (settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        # Game Over Text
        go_surf = self.title_font.render("GAME OVER", True, settings.ALIEN_COLOR)
        go_rect = go_surf.get_rect(
            center=(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2 - 50)
        )
        screen.blit(go_surf, go_rect)

        # Final Score Text
        fs_surf = self.score_font.render(
            f"FINAL SCORE: {score}", True, settings.TEXT_COLOR
        )
        fs_rect = fs_surf.get_rect(
            center=(settings.SCREEN_WIDTH // 2, settings.SCREEN_HEIGHT // 2)
        )
        screen.blit(fs_surf, fs_rect)

        # Restart Button
        pygame.draw.rect(
            screen, settings.BULLET_COLOR, self.restart_rect, border_radius=10
        )
        btn_surf = self.button_font.render("RESTART", True, settings.BG_COLOR)
        btn_rect = btn_surf.get_rect(center=self.restart_rect.center)
        screen.blit(btn_surf, btn_rect)

    def is_restart_clicked(self, mouse_pos):
        return self.restart_rect.collidepoint(mouse_pos)
