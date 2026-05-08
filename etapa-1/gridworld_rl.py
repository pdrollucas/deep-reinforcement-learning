"""
Grid World - Aprendizado por Reforço com Pygame
================================================
Ambiente 4x4 | Q-Learning | Visualização Animada

Respostas Teóricas:
-------------------
1. CÁLCULO DO RETORNO (Gt) para trajetória:
   (0,0)→(0,1)→(1,1)→(2,1)→(3,1)→(3,2)→(3,3)
   Recompensas: R1=-1, R2=-1, R3=-1, R4=-1, R5=-1, R6=+10
   γ = 0.9

   Gt = R1 + γ·R2 + γ²·R3 + γ³·R4 + γ⁴·R5 + γ⁵·R6
      = -1 + 0.9·(-1) + 0.81·(-1) + 0.729·(-1) + 0.6561·(-1) + 0.59049·(10)
      = -1 - 0.9 - 0.81 - 0.729 - 0.6561 + 5.9049
      = **1.9098**

2. CONVERGÊNCIA DA FUNÇÃO DE VALOR V:
   V(s) converge quando a diferença máxima entre iterações consecutivas
   for menor que um limiar θ (ex: θ = 0.001):
   max_s |V_novo(s) - V_antigo(s)| < θ
   Isso indica que a política estabilizou e novos episódios
   não alteram significativamente as estimativas de valor.

Execução:
---------
pip install pygame numpy
python gridworld_rl.py
"""

import pygame
import numpy as np
import sys
import time
import math

# ─── CONFIGURAÇÕES ────────────────────────────────────────────────────────────
GRID_SIZE    = 4
CELL_SIZE    = 130
MARGIN       = 60
INFO_WIDTH   = 320
FPS          = 60

WIN_W = GRID_SIZE * CELL_SIZE + 2 * MARGIN + INFO_WIDTH
WIN_H = GRID_SIZE * CELL_SIZE + 2 * MARGIN + 80

# ─── CORES ────────────────────────────────────────────────────────────────────
BG          = (15,  17,  26)
GRID_LINE   = (40,  50,  80)
CELL_EMPTY  = (22,  28,  48)
CELL_START  = (30,  80,  60)
CELL_GOAL   = (180, 140,  20)
CELL_HOVER  = (35,  45,  70)
AGENT_CLR   = (80, 180, 240)
AGENT_TRAIL = (50, 100, 160)
TEXT_MAIN   = (220, 230, 255)
TEXT_DIM    = (100, 120, 160)
TEXT_GOLD   = (255, 200,  60)
TEXT_GREEN  = ( 80, 220, 120)
TEXT_RED    = (220,  80,  80)
ARROW_CLR   = (160, 200, 255)
PANEL_BG    = (18,  22,  40)
PANEL_BORDER= (50,  65, 110)
REWARD_POS  = ( 80, 220, 120)
REWARD_NEG  = (220,  80,  80)
REWARD_WALL = (220, 140,  40)

# ─── AMBIENTE ─────────────────────────────────────────────────────────────────
class GridWorld:
    def __init__(self):
        self.size   = GRID_SIZE
        self.start  = (0, 0)
        self.goal   = (3, 3)
        self.reset()

    def reset(self):
        self.pos = list(self.start)
        return tuple(self.pos)

    def step(self, action):
        """Ações: 0=Norte, 1=Sul, 2=Leste, 3=Oeste"""
        r, c = self.pos
        deltas = [(-1,0),(1,0),(0,1),(0,-1)]
        dr, dc = deltas[action]
        nr, nc = r+dr, c+dc

        if 0 <= nr < self.size and 0 <= nc < self.size:
            self.pos = [nr, nc]
            reward = -1
        else:
            reward = -5  # colisão com borda

        done = (self.pos[0] == self.goal[0] and self.pos[1] == self.goal[1])
        if done:
            reward = 10
        return tuple(self.pos), reward, done


# ─── AGENTE Q-LEARNING ────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self):
        self.q      = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.gamma  = 0.9
        self.alpha  = 0.3
        self.eps    = 1.0
        self.eps_min= 0.05
        self.eps_dec= 0.995
        self.episode= 0
        self.steps  = 0
        self.total_reward = 0
        self.best_reward  = -999
        self.rewards_hist = []

    def choose_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(4)
        r, c = state
        return int(np.argmax(self.q[r, c]))

    def learn(self, s, a, r, s2, done):
        r0, c0 = s
        r1, c1 = s2
        best_next = 0 if done else np.max(self.q[r1, c1])
        target = r + self.gamma * best_next
        self.q[r0, c0, a] += self.alpha * (target - self.q[r0, c0, a])

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_dec)

    def greedy_path(self, env):
        """Retorna o caminho greedy atual"""
        env.reset()
        path = [tuple(env.pos)]
        for _ in range(30):
            r, c = env.pos
            a = int(np.argmax(self.q[r, c]))
            s2, _, done = env.step(a)
            path.append(s2)
            if done:
                break
        return path


# ─── UTILITÁRIOS DE DESENHO ───────────────────────────────────────────────────
def cell_rect(row, col):
    x = MARGIN + col * CELL_SIZE
    y = MARGIN + row * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

def cell_center(row, col):
    r = cell_rect(row, col)
    return r.centerx, r.centery

def lerp(a, b, t):
    return a + (b - a) * t

def draw_rounded_rect(surf, color, rect, radius=12, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)

def draw_arrow(surf, color, cx, cy, direction, size=18, alpha=200):
    """Desenha seta de política"""
    dx = [0, 0, 1, -1][direction]
    dy = [-1, 1, 0, 0][direction]
    tip = (cx + dx*size, cy + dy*size)
    base_perp = (dy*size*0.45, dx*size*0.45)
    p1 = (cx - dx*size*0.4 + base_perp[0], cy - dy*size*0.4 + base_perp[1])
    p2 = (cx - dx*size*0.4 - base_perp[0], cy - dy*size*0.4 - base_perp[1])
    s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    pts = [
        (tip[0] - (cx - CELL_SIZE//2), tip[1] - (cy - CELL_SIZE//2)),
        (p1[0]  - (cx - CELL_SIZE//2), p1[1]  - (cy - CELL_SIZE//2)),
        (p2[0]  - (cx - CELL_SIZE//2), p2[1]  - (cy - CELL_SIZE//2)),
    ]
    pygame.draw.polygon(s, (*color, alpha), pts)
    surf.blit(s, (cx - CELL_SIZE//2, cy - CELL_SIZE//2))


# ─── VISUALIZADOR ─────────────────────────────────────────────────────────────
class Visualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Grid World — Q-Learning Agent")
        self.clock  = pygame.time.Clock()

        self.font_xl  = pygame.font.SysFont("segoeui", 28, bold=True)
        self.font_lg  = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_md  = pygame.font.SysFont("segoeui", 16)
        self.font_sm  = pygame.font.SysFont("segoeui", 13)
        self.font_val = pygame.font.SysFont("consolas", 13)

        self.env   = GridWorld()
        self.agent = QLearningAgent()

        # Estado da animação
        self.agent_pos    = list(self.env.start)   # posição visual (float)
        self.agent_target = list(self.env.start)
        self.agent_state  = tuple(self.env.start)
        self.anim_t       = 1.0
        self.anim_speed   = 0.12
        self.trail        = []
        self.reward_popups= []   # [(x,y,reward,timer)]

        # Controle de episódio
        self.action       = None
        self.ep_reward    = 0
        self.ep_steps     = 0
        self.done         = False
        self.running_ep   = True
        self.speed        = 3   # passos por segundo (1..10)
        self.step_timer   = 0.0
        self.paused       = False
        self.show_qvals   = True

        # Trajetória greedy
        self.greedy       = []

        self.env.reset()
        self.agent_state = tuple(self.env.pos)
        self.agent_pos   = list(self.env.pos)
        self.agent_target= list(self.env.pos)

    # ── loop principal ────────────────────────────────────────────────────────
    def run(self):
        dt = 0.0
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            if not self.paused:
                self._update(dt)
            self._draw()
            pygame.display.flip()

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_q:
                    self.show_qvals = not self.show_qvals
                elif ev.key == pygame.K_UP:
                    self.speed = min(10, self.speed + 1)
                elif ev.key == pygame.K_DOWN:
                    self.speed = max(1,  self.speed - 1)
                elif ev.key == pygame.K_r:
                    self.agent = QLearningAgent()
                    self._new_episode()
                elif ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

    def _update(self, dt):
        # Animação de movimento
        if self.anim_t < 1.0:
            self.anim_t = min(1.0, self.anim_t + self.anim_speed)
            t = self._ease(self.anim_t)
            self.agent_pos[0] = lerp(self.agent_pos[0], self.agent_target[0], t)
            self.agent_pos[1] = lerp(self.agent_pos[1], self.agent_target[1], t)
            return  # espera animação terminar

        # Timer de passo
        self.step_timer += dt
        step_interval = 1.0 / self.speed
        if self.step_timer < step_interval:
            return
        self.step_timer = 0.0

        # Decay popup timers
        self.reward_popups = [(x,y,r,t-dt) for x,y,r,t in self.reward_popups if t > 0]

        if self.done:
            self._new_episode()
            return

        # Escolhe e executa ação
        s = self.agent_state
        a = self.agent.choose_action(s)
        s2, reward, done = self.env.step(a)

        self.agent.learn(s, a, reward, s2, done)
        self.agent.decay_eps()

        # Atualiza estado e animação
        self.agent_state = s2
        self.trail.append(list(self.agent_target))
        if len(self.trail) > 12:
            self.trail.pop(0)
        self.agent_target = list(s2)
        self.anim_t = 0.0

        # Popup de recompensa
        cx, cy = cell_center(s2[0], s2[1])
        self.reward_popups.append((cx, cy - 20, reward, 1.2))

        self.ep_reward += reward
        self.ep_steps  += 1
        self.done = done
        if self.ep_steps > 60:
            self.done = True

    def _new_episode(self):
        self.agent.episode += 1
        self.agent.steps   += self.ep_steps
        self.agent.total_reward += self.ep_reward
        self.agent.rewards_hist.append(self.ep_reward)
        if self.ep_reward > self.agent.best_reward:
            self.agent.best_reward = self.ep_reward

        self.env.reset()
        self.agent_state  = tuple(self.env.pos)
        self.agent_target = list(self.env.pos)
        self.agent_pos    = list(self.env.pos)
        self.anim_t = 1.0
        self.trail  = []
        self.ep_reward = 0
        self.ep_steps  = 0
        self.done = False
        if self.agent.episode % 10 == 0:
            self.greedy = self.agent.greedy_path(GridWorld())

    def _ease(self, t):
        return t * t * (3 - 2*t)

    # ── desenho ───────────────────────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG)
        self._draw_title()
        self._draw_grid()
        self._draw_trail()
        self._draw_greedy_path()
        self._draw_policy_arrows()
        self._draw_agent()
        self._draw_reward_popups()
        self._draw_info_panel()
        self._draw_controls()

    def _draw_title(self):
        t = self.font_xl.render("Grid World — Q-Learning", True, TEXT_MAIN)
        self.screen.blit(t, (MARGIN, 14))

    def _draw_grid(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = cell_rect(r, c)
                # Cor da célula
                if (r,c) == self.env.goal:
                    color = CELL_GOAL
                elif (r,c) == self.env.start:
                    color = CELL_START
                else:
                    color = CELL_EMPTY
                draw_rounded_rect(self.screen, color, rect.inflate(-4,-4), 10,
                                  1, GRID_LINE)

                # Valor Q máximo na célula
                if self.show_qvals and (r,c) != self.env.goal:
                    vmax = np.max(self.agent.q[r,c])
                    txt  = f"{vmax:.2f}"
                    s = self.font_val.render(txt, True, TEXT_DIM)
                    self.screen.blit(s, (rect.x+6, rect.y+6))

                # Labels especiais
                if (r,c) == self.env.goal:
                    s = self.font_lg.render("GOAL", True, (30,20,5))
                    self.screen.blit(s, s.get_rect(center=rect.center))
                elif (r,c) == self.env.start:
                    s = self.font_sm.render("START", True, (200,255,220))
                    self.screen.blit(s, s.get_rect(center=(rect.centerx, rect.centery+28)))

                # Coordenada
                coord = self.font_sm.render(f"({r},{c})", True, TEXT_DIM)
                self.screen.blit(coord, (rect.right-42, rect.bottom-18))

    def _draw_trail(self):
        for i, pos in enumerate(self.trail):
            alpha = int(180 * (i+1) / len(self.trail))
            cx, cy = cell_center(int(pos[0]), int(pos[1]))
            r = max(4, int(10 * (i+1)/len(self.trail)))
            s = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*AGENT_TRAIL, alpha), (r+1,r+1), r)
            self.screen.blit(s, (cx-r-1, cy-r-1))

    def _draw_greedy_path(self):
        if len(self.greedy) < 2:
            return
        for i in range(len(self.greedy)-1):
            r0,c0 = self.greedy[i]
            r1,c1 = self.greedy[i+1]
            x0,y0 = cell_center(r0,c0)
            x1,y1 = cell_center(r1,c1)
            s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            pygame.draw.line(s, (255,200,60,60), (x0,y0),(x1,y1), 3)
            self.screen.blit(s,(0,0))

    def _draw_policy_arrows(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r,c) == self.env.goal:
                    continue
                best_a = int(np.argmax(self.agent.q[r,c]))
                cx, cy = cell_center(r, c)
                draw_arrow(self.screen, ARROW_CLR, cx, cy, best_a, 20, 160)

    def _draw_agent(self):
        t = self._ease(self.anim_t)
        r0 = self.agent_pos[0]
        c0 = self.agent_pos[1]
        r1 = self.agent_target[0]
        c1 = self.agent_target[1]
        vr = lerp(r0, r1, t)
        vc = lerp(c0, c1, t)
        x = MARGIN + vc * CELL_SIZE + CELL_SIZE//2
        y = MARGIN + vr * CELL_SIZE + CELL_SIZE//2
        # sombra
        pygame.draw.circle(self.screen, (0,0,0,60), (int(x)+3, int(y)+4), 22)
        # corpo
        pygame.draw.circle(self.screen, AGENT_CLR,  (int(x), int(y)), 22)
        pygame.draw.circle(self.screen, (200,240,255),(int(x)-5, int(y)-6), 8)

    def _draw_reward_popups(self):
        for (x,y,r,t) in self.reward_popups:
            alpha = int(255 * min(1, t))
            dy    = int((1.2 - t) * 30)
            color = REWARD_POS if r > 0 else (REWARD_WALL if r == -5 else REWARD_NEG)
            sign  = "+" if r >= 0 else ""
            txt   = f"{sign}{r}"
            s = self.font_lg.render(txt, True, color)
            s.set_alpha(alpha)
            self.screen.blit(s, (x - s.get_width()//2, y - dy))

    def _draw_info_panel(self):
        px = MARGIN + GRID_SIZE * CELL_SIZE + 20
        pw = INFO_WIDTH - 10
        py = MARGIN

        # Fundo do painel
        panel = pygame.Rect(px, py, pw, GRID_SIZE*CELL_SIZE)
        draw_rounded_rect(self.screen, PANEL_BG, panel, 14, 1, PANEL_BORDER)

        x = px + 16
        y = py + 16

        def line(text, color=TEXT_MAIN, font=None, indent=0):
            nonlocal y
            f = font or self.font_md
            s = f.render(text, True, color)
            self.screen.blit(s, (x+indent, y))
            y += s.get_height() + 4

        def sep():
            nonlocal y
            pygame.draw.line(self.screen, PANEL_BORDER, (x, y+2), (x+pw-32, y+2))
            y += 10

        line("📊 Estatísticas", TEXT_GOLD, self.font_lg)
        sep()
        line(f"Episódio:   {self.agent.episode:>6}")
        line(f"Passos ep.: {self.ep_steps:>6}")
        line(f"Exploração: {self.agent.eps*100:>5.1f}%")

        ep_color = TEXT_GREEN if self.ep_reward >= 0 else TEXT_RED
        line(f"Ret. atual: ", TEXT_MAIN)
        y -= 22
        val = self.font_md.render(f"{self.ep_reward:>+.0f}", True, ep_color)
        self.screen.blit(val, (x+110, y-2))
        y += 22

        best_color = TEXT_GOLD if self.agent.best_reward > 0 else TEXT_RED
        line(f"Melhor ret: ", TEXT_MAIN)
        y -= 22
        bv = self.font_md.render(f"{self.agent.best_reward:>+.0f}", True, best_color)
        self.screen.blit(bv, (x+110, y-2))
        y += 22

        sep()
        line("📈 Mini-gráfico", TEXT_GOLD, self.font_lg)
        self._draw_mini_chart(x, y, pw-32, 70)
        y += 78

        sep()
        line("⚙  Parâmetros", TEXT_GOLD, self.font_lg)
        line(f"γ (desconto): 0.9",    TEXT_DIM, self.font_sm)
        line(f"α (aprendiz.): 0.3",   TEXT_DIM, self.font_sm)
        line(f"ε mín:        0.05",   TEXT_DIM, self.font_sm)
        line(f"Velocidade:  {self.speed:>2}/10",  TEXT_DIM, self.font_sm)

        sep()
        # Gt da trajetória fixa
        line("🔢 Gt trajetória fixa", TEXT_GOLD, self.font_lg)
        line("(0,0)→…→(3,3)  6 passos", TEXT_DIM, self.font_sm)
        line("Gt = 1.9098", TEXT_GREEN, self.font_lg)

    def _draw_mini_chart(self, x, y, w, h):
        hist = self.agent.rewards_hist[-60:]
        if len(hist) < 2:
            return
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (12,16,32), rect, border_radius=6)
        vmin, vmax = min(hist), max(hist)
        rng  = max(1, vmax - vmin)
        pts  = []
        for i, v in enumerate(hist):
            px_ = x + int(i / (len(hist)-1) * w)
            py_ = y + h - int((v - vmin) / rng * (h-4)) - 2
            pts.append((px_, py_))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, TEXT_GOLD, False, pts, 2)
        # zero line
        zy = y + h - int((0 - vmin) / rng * (h-4)) - 2
        pygame.draw.line(self.screen, PANEL_BORDER, (x, zy), (x+w, zy), 1)

    def _draw_controls(self):
        controls = [
            ("ESPAÇO", "Pausar/Retomar"),
            ("↑ / ↓",  "Velocidade"),
            ("Q",       "Ocultar Q-valores"),
            ("R",       "Reiniciar treinamento"),
            ("ESC",     "Sair"),
        ]
        base_y = WIN_H - 48
        x = MARGIN
        for key, desc in controls:
            ks = self.font_sm.render(f"[{key}]", True, TEXT_GOLD)
            self.screen.blit(ks, (x, base_y))
            x += ks.get_width() + 2
            ds = self.font_sm.render(desc, True, TEXT_DIM)
            self.screen.blit(ds, (x, base_y))
            x += ds.get_width() + 18
            if x > WIN_W - 160:
                x = MARGIN
                base_y += 18

        # status pausa
        if self.paused:
            s = self.font_lg.render("⏸  PAUSADO", True, TEXT_GOLD)
            r = s.get_rect(center=(MARGIN + GRID_SIZE*CELL_SIZE//2, WIN_H - 28))
            self.screen.blit(s, r)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(__doc__)

    # ── Cálculo explícito do Gt ──
    gamma = 0.9
    rewards = [-1, -1, -1, -1, -1, 10]
    Gt = sum(r * gamma**t for t, r in enumerate(rewards))
    print(f"\n{'='*55}")
    print(f"  CÁLCULO DO RETORNO Gt")
    print(f"{'='*55}")
    print(f"  Trajetória: (0,0)→(0,1)→(1,1)→(2,1)→(3,1)→(3,2)→(3,3)")
    print(f"  Recompensas: {rewards}")
    for t, r in enumerate(rewards):
        print(f"    t={t}: {r} × {gamma}^{t} = {r * gamma**t:.4f}")
    print(f"  {'─'*40}")
    print(f"  Gt = {Gt:.4f}")
    print(f"\n{'='*55}")
    print(f"  CONVERGÊNCIA DE V(s)")
    print(f"{'='*55}")
    print(f"  V(s) converge quando:")
    print(f"  max_s |V_novo(s) - V_antigo(s)| < θ  (θ ≈ 0.001)")
    print(f"\n  Iniciando visualização Pygame...")
    print(f"  Controles: ESPAÇO=pausar | ↑↓=velocidade | R=reset | ESC=sair")
    print(f"{'='*55}\n")

    try:
        viz = Visualizer()
        viz.run()
    except Exception as e:
        print(f"Erro ao iniciar Pygame: {e}")
        print("Certifique-se de instalar: pip install pygame numpy")