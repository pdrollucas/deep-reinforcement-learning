"""
╔══════════════════════════════════════════════════════════════════╗
║   GRID WORLD ESTOCÁSTICO 5×5  —  Q-Learning com Fantasmas       ║
║   Etapa 2 · Ambiente Dinâmico e Incerto                         ║
╚══════════════════════════════════════════════════════════════════╝

RESPOSTAS TEÓRICAS
══════════════════

1. EQUAÇÃO DE BELLMAN ESTOCÁSTICA — V(1,0) com ação "Leste"
   ─────────────────────────────────────────────────────────
   Ação "Leste" a partir de (1,0):
     • 80% → move para (1,1)  [direção desejada]
     • 10% → desliza para (0,0) [esquerda relativa = Norte]
     • 10% → desliza para (2,0) [direita relativa  = Sul]

   Equação de Bellman estocástica:
   V(1,0) = 0.80 · [R(1,0→1,1) + γ·V(1,1)]
           + 0.10 · [R(1,0→0,0) + γ·V(0,0)]
           + 0.10 · [R(1,0→2,0) + γ·V(2,0)]

   Supondo V(1,1)=0.5, V(0,0)=0.0, V(2,0)=0.3, γ=0.9, R=-0.1:
   V(1,0) = 0.80·(-0.1 + 0.9·0.5)
           + 0.10·(-0.1 + 0.9·0.0)
           + 0.10·(-0.1 + 0.9·0.3)
          = 0.80·(0.35) + 0.10·(-0.1) + 0.10·(0.17)
          = 0.280 - 0.010 + 0.017 = 0.287

   No mundo DETERMINÍSTICO: V(1,0) = R + γ·V(1,1) = -0.1 + 0.45 = 0.35
   A estocasticidade reduz o valor esperado pois as derrapagens podem
   levar a estados menos valiosos, forçando o agente a considerar
   trajetórias não intencionais.

2. RELAÇÃO α × γ NA ESTABILIDADE COM 20% DE INCERTEZA
   ─────────────────────────────────────────────────────
   • α alto (ex: 0.9) + γ alto (ex: 0.99): instabilidade. As atualizações
     Q sobrescrevem agressivamente experiências passadas. Com 20% de
     derrapagem, o ruído se propaga e os Q-valores oscilam sem convergir.

   • α baixo (ex: 0.01) + γ alto: convergência lenta mas estável.
     O agente leva centenas de episódios para aprender o impacto das
     ações futuras com incerteza.

   • Configuração IDEAL para 20% de incerteza:
     α ∈ [0.1, 0.3] + γ ∈ [0.9, 0.95]
     Isso equilibra plasticidade (aprender erros) e estabilidade
     (não esquecer o que foi aprendido). A condição de convergência
     exige Σα_t = ∞ e Σα_t² < ∞ (Robbins-Monro).

3. Q(s,a) COMO BARREIRA CONTRA ESTADOS PERIGOSOS
   ─────────────────────────────────────────────────
   Durante a exploração (ε-greedy), o agente visita estados próximos
   ao poço. As recompensas -100 propagam-se via TD-backup:

   Q(s_prev, a) ← Q(s_prev, a) + α·[-100 + γ·max_a'Q(s_poço, a') - Q(s_prev, a)]

   Como max Q(s_poço, a') ≈ -100, o Q-valor das ações que levam ao
   poço torna-se muito negativo. Na política ε-greedy, a ação greedy
   evita essas rotas. A função Q age como um "campo de repulsão":
   quanto mais próximo do perigo, mais negativos os Q-valores vizinhos,
   criando uma zona de exclusão implícita aprendida por experiência.

EXECUÇÃO
════════
  pip install pygame numpy
  python gridworld_stochastic.py
"""

import pygame
import numpy as np
import sys
import math
import random
import time

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES GLOBAIS
# ═══════════════════════════════════════════════════════════════════
GRID       = 5
CELL       = 110
MARGIN_L   = 50
MARGIN_T   = 70
PANEL_W    = 300
WIN_W      = GRID*CELL + MARGIN_L*2 + PANEL_W
WIN_H      = GRID*CELL + MARGIN_T   + 90
FPS        = 60
GAMMA      = 0.9
ALPHA      = 0.15
EPS_START  = 1.0
EPS_MIN    = 0.05
EPS_DECAY  = 0.997

# Posições especiais
START      = (0, 0)
GOAL       = (4, 4)
PIT        = (3, 2)       # poço fixo
GHOST_CELLS= [(2,2),(2,4),(1,3),(3,4),(0,3)]  # células onde fantasmas vagam

# Recompensas
R_GOAL     =  100.0
R_PIT      = -100.0
R_GHOST    =  -50.0
R_STEP     =   -0.1

# Ações: 0=N 1=S 2=L 3=O
ACTIONS    = 4
DIRS       = [(-1,0),(1,0),(0,1),(0,-1)]
DIR_NAMES  = ["Norte","Sul","Leste","Oeste"]
DIR_SYMS   = ["▲","▼","▶","◀"]

# ─── PALETA ────────────────────────────────────────────────────────
BG         = ( 8, 10, 20)
C_EMPTY    = (16, 20, 38)
C_GOAL     = (20, 80, 30)
C_PIT      = (80, 10, 10)
C_GHOST_Z  = (40, 10, 60)
C_START    = (10, 40, 60)
C_TRAIL    = (30, 80,130)
C_AGENT    = (60,200,255)
C_GHOST    = (180, 40,220)
C_GRID     = (28, 35, 65)
C_PANEL    = (10, 13, 28)
C_BORDER   = (45, 60,110)
C_GOLD     = (255,195, 40)
C_GREEN    = ( 60,230,120)
C_RED      = (230, 60, 60)
C_BLUE     = ( 80,160,255)
C_PURPLE   = (180, 80,240)
C_DIM      = ( 90,110,155)
C_WHITE    = (220,230,255)

# ═══════════════════════════════════════════════════════════════════
#  MDP ESTOCÁSTICO
# ═══════════════════════════════════════════════════════════════════
class StochasticGridWorld:
    """Transições 80/10/10: direção desejada / esquerda relativa / direita relativa"""

    # deslizamentos relativos a cada ação
    SLIP = {
        0: [0, 3, 2],  # Norte  → slip: Oeste, Leste
        1: [1, 2, 3],  # Sul    → slip: Leste, Oeste
        2: [2, 0, 1],  # Leste  → slip: Norte, Sul
        3: [3, 1, 0],  # Oeste  → slip: Sul,   Norte
    }

    def __init__(self):
        self.ghosts = [list(GHOST_CELLS[0]), list(GHOST_CELLS[1])]
        self.ghost_timer = 0
        self.reset()

    def reset(self):
        self.pos = list(START)
        return tuple(self.pos)

    def _clamp(self, r, c):
        return max(0, min(GRID-1, r)), max(0, min(GRID-1, c))

    def _move(self, pos, action):
        dr, dc = DIRS[action]
        nr, nc = self._clamp(pos[0]+dr, pos[1]+dc)
        return [nr, nc]

    def step(self, action):
        # Escolhe ação real (80/10/10)
        roll = random.random()
        opts = self.SLIP[action]
        if   roll < 0.80: act = opts[0]
        elif roll < 0.90: act = opts[1]
        else:             act = opts[2]

        npos = self._move(self.pos, act)
        self.pos = npos

        # Mover fantasmas periodicamente
        self.ghost_timer += 1
        if self.ghost_timer % 8 == 0:
            self._move_ghosts()

        state = tuple(self.pos)
        reward, done = R_STEP, False

        if state == GOAL:
            reward, done = R_GOAL, True
        elif state == PIT:
            reward, done = R_PIT, True
        else:
            for g in self.ghosts:
                if state == tuple(g):
                    reward = R_GHOST
                    self.pos = list(START)
                    state = START
                    break

        return state, reward, done, tuple(self.pos)

    def _move_ghosts(self):
        for g in self.ghosts:
            candidates = GHOST_CELLS
            nxt = random.choice(candidates)
            g[0], g[1] = nxt

    def ghost_positions(self):
        return [tuple(g) for g in self.ghosts]


# ═══════════════════════════════════════════════════════════════════
#  AGENTE Q-LEARNING
# ═══════════════════════════════════════════════════════════════════
class QAgent:
    def __init__(self):
        self.Q        = np.zeros((GRID, GRID, ACTIONS))
        self.eps      = EPS_START
        self.episode  = 0
        self.ep_steps = 0
        self.ep_reward= 0.0
        self.best_r   = -9999
        self.hist     = []          # recompensa por episódio
        self.deaths   = 0
        self.wins     = 0
        self.total_st = 0

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, ACTIONS-1)
        r, c = state
        return int(np.argmax(self.Q[r, c]))

    def learn(self, s, a, r, s2, done):
        r0, c0 = s
        r1, c1 = s2
        best_next = 0.0 if done else float(np.max(self.Q[r1, c1]))
        td = r + GAMMA * best_next - self.Q[r0, c0, a]
        self.Q[r0, c0, a] += ALPHA * td

    def decay(self):
        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)

    def best_action(self, state):
        r, c = state
        return int(np.argmax(self.Q[r, c]))

    def best_path(self, env_cls):
        env = env_cls()
        path = [START]
        for _ in range(40):
            s = tuple(env.pos)
            a = self.best_action(s)
            s2, _, done, _ = env.step(a)
            path.append(s2)
            if done: break
        return path


# ═══════════════════════════════════════════════════════════════════
#  HELPERS DE DESENHO
# ═══════════════════════════════════════════════════════════════════
def cell_rect(r, c):
    return pygame.Rect(MARGIN_L + c*CELL, MARGIN_T + r*CELL, CELL, CELL)

def cell_center(r, c):
    rect = cell_rect(r, c)
    return rect.centerx, rect.centery

def lerp(a, b, t): return a + (b-a)*t
def ease(t):       return t*t*(3-2*t)

def draw_rrect(surf, color, rect, rad=10, bw=0, bc=None):
    pygame.draw.rect(surf, color, rect, border_radius=rad)
    if bw and bc:
        pygame.draw.rect(surf, bc, rect, bw, border_radius=rad)

def hsv_color(h, s=0.8, v=0.9):
    import colorsys
    r,g,b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

def blend(c1, c2, t):
    return tuple(int(lerp(a,b,t)) for a,b in zip(c1,c2))

def draw_arrow(surf, color, cx, cy, direction, size=16, alpha=200):
    dx = [0, 0, 1, -1][direction]
    dy = [-1, 1, 0, 0][direction]
    tip = (cx + dx*size, cy + dy*size)
    perp = (dy*size*0.42, dx*size*0.42)
    base = (cx - dx*size*0.35, cy - dy*size*0.35)
    p1 = (base[0]+perp[0], base[1]+perp[1])
    p2 = (base[0]-perp[0], base[1]-perp[1])
    s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    pygame.draw.polygon(s, (*color, alpha), [tip, p1, p2])
    surf.blit(s, (0,0))

def draw_glow(surf, color, cx, cy, radius, alpha=60):
    for r2 in range(radius, 0, -4):
        a = int(alpha * (1 - r2/radius))
        s = pygame.Surface((r2*2+2, r2*2+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, a), (r2+1, r2+1), r2)
        surf.blit(s, (cx-r2-1, cy-r2-1))


# ═══════════════════════════════════════════════════════════════════
#  PARTÍCULAS
# ═══════════════════════════════════════════════════════════════════
class Particle:
    def __init__(self, x, y, color, vx=None, vy=None, life=None, size=None):
        self.x = x; self.y = y
        self.color = color
        self.vx = vx if vx else random.uniform(-2, 2)
        self.vy = vy if vy else random.uniform(-3, -0.5)
        self.life = life if life else random.uniform(0.4, 1.0)
        self.max_life = self.life
        self.size = size if size else random.randint(2, 5)

    def update(self, dt):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.1
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        alpha = int(255 * self.life / self.max_life)
        r = max(1, int(self.size * self.life / self.max_life))
        s = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (r+1,r+1), r)
        surf.blit(s, (int(self.x)-r-1, int(self.y)-r-1))


# ═══════════════════════════════════════════════════════════════════
#  POPUP DE EVENTO
# ═══════════════════════════════════════════════════════════════════
class Popup:
    def __init__(self, x, y, text, color, size=20, life=1.8):
        self.x = x; self.y = y; self.text = text
        self.color = color; self.life = life; self.max_life = life
        self.size = size; self.dy = 0

    def update(self, dt):
        self.life -= dt
        self.dy   += 30 * dt
        return self.life > 0

    def draw(self, surf, font):
        alpha = int(255 * min(1, self.life / self.max_life * 2))
        s = font.render(self.text, True, self.color)
        s.set_alpha(alpha)
        surf.blit(s, (int(self.x - s.get_width()//2), int(self.y - self.dy)))


# ═══════════════════════════════════════════════════════════════════
#  VISUALIZADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Grid World Estocástico 5×5 — Q-Learning")
        self.clock  = pygame.time.Clock()
        self.tick   = 0
        self.time   = 0.0

        # Fontes
        self.f_title = pygame.font.SysFont("segoeui", 26, bold=True)
        self.f_head  = pygame.font.SysFont("segoeui", 17, bold=True)
        self.f_body  = pygame.font.SysFont("segoeui", 14)
        self.f_small = pygame.font.SysFont("segoeui", 12)
        self.f_mono  = pygame.font.SysFont("consolas", 12)
        self.f_popup = pygame.font.SysFont("segoeui", 18, bold=True)
        self.f_sym   = pygame.font.SysFont("segoeui", 22)

        # RL
        self.env   = StochasticGridWorld()
        self.agent = QAgent()

        # Animação do agente
        self.apos   = [0.0, 0.0]   # posição visual (float row, col)
        self.atgt   = [0.0, 0.0]
        self.anim_t = 1.0
        self.anim_spd = 0.14
        self.astate = START

        # Fantasmas visuais
        self.gpos   = [[float(g[0]), float(g[1])] for g in self.env.ghost_positions()]
        self.gtgt   = [list(g) for g in self.env.ghost_positions()]

        # Controles
        self.speed  = 4
        self.paused = False
        self.show_q = True
        self.show_arrows = True
        self.step_timer = 0.0
        self.ep_done= False

        # Trilha
        self.trail  = []

        # FX
        self.particles = []
        self.popups    = []
        self.screen_flash = 0.0
        self.flash_color  = (0,0,0)
        self.best_path = []

        # Stats episódio
        self.ep_reward = 0.0
        self.ep_steps  = 0

    # ──────────────────────────────────────────────────────────────
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)
            self.time += dt
            self._events()
            if not self.paused:
                self._update(dt)
            self._draw()
            pygame.display.flip()

    # ──────────────────────────────────────────────────────────────
    def _events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_SPACE:  self.paused = not self.paused
                elif ev.key == pygame.K_q:       self.show_q = not self.show_q
                elif ev.key == pygame.K_a:       self.show_arrows = not self.show_arrows
                elif ev.key == pygame.K_UP:      self.speed = min(15, self.speed+1)
                elif ev.key == pygame.K_DOWN:    self.speed = max(1,  self.speed-1)
                elif ev.key == pygame.K_r:
                    self.agent = QAgent()
                    self._reset_episode()
                elif ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

    # ──────────────────────────────────────────────────────────────
    def _update(self, dt):
        self.tick += 1

        # Partículas
        self.particles = [p for p in self.particles if p.update(dt)]
        self.popups    = [p for p in self.popups    if p.update(dt)]
        self.screen_flash = max(0, self.screen_flash - dt*3)

        # Animar agente
        if self.anim_t < 1.0:
            self.anim_t = min(1.0, self.anim_t + self.anim_spd)
            t = ease(self.anim_t)
            self.apos[0] = lerp(self.apos[0], self.atgt[0], t)
            self.apos[1] = lerp(self.apos[1], self.atgt[1], t)

        # Animar fantasmas
        for i, g in enumerate(self.gpos):
            t = ease(min(1.0, self.anim_t * 1.5))
            g[0] = lerp(g[0], float(self.gtgt[i][0]), t*0.08)
            g[1] = lerp(g[1], float(self.gtgt[i][1]), t*0.08)

        if self.anim_t < 1.0:
            return

        # Passo de RL
        self.step_timer += dt
        interval = 1.0 / self.speed
        if self.step_timer < interval:
            return
        self.step_timer = 0.0

        if self.ep_done:
            self._reset_episode()
            return

        s = self.astate
        a = self.agent.act(s)
        s2, reward, done, actual_pos = self.env.step(a)

        self.agent.learn(s, a, reward, s2, done)
        self.agent.decay()

        # Trilha
        self.trail.append(list(self.atgt))
        if len(self.trail) > 20: self.trail.pop(0)

        # Atualizar posição visual
        self.astate = actual_pos
        self.atgt   = [float(actual_pos[0]), float(actual_pos[1])]
        self.anim_t = 0.0

        # Atualizar fantasmas visuais
        for i, g in enumerate(self.env.ghost_positions()):
            self.gtgt[i] = [float(g[0]), float(g[1])]

        self.ep_reward += reward
        self.ep_steps  += 1

        # FX por evento
        cx, cy = cell_center(s2[0], s2[1])
        if done and reward > 0:
            self._burst_particles(cx, cy, C_GOLD, 40)
            self.popups.append(Popup(cx, cy-30, "+100 META!", C_GOLD, 22, 2.5))
            self.screen_flash = 1.0; self.flash_color = (20,60,10)
            self.agent.wins += 1
            self.best_path = self.agent.best_path(StochasticGridWorld)
        elif done and reward < 0:
            self._burst_particles(cx, cy, C_RED, 30)
            self.popups.append(Popup(cx, cy-30, "-100 POÇO!", C_RED, 22, 2.0))
            self.screen_flash = 0.8; self.flash_color = (60,5,5)
            self.agent.deaths += 1
        elif reward == R_GHOST:
            self._burst_particles(cx, cy, C_PURPLE, 25)
            self.popups.append(Popup(cx, cy-30, "-50 FANTASMA!", C_PURPLE, 20, 2.0))
            self.screen_flash = 0.5; self.flash_color = (30,5,40)
        else:
            sign = "+" if reward >= 0 else ""
            self.popups.append(Popup(cx, cy-20, f"{sign}{reward:.1f}", C_DIM, 14, 0.8))

        self.ep_done = done
        if self.ep_steps >= 80:
            self.ep_done = True

    def _reset_episode(self):
        self.agent.episode  += 1
        self.agent.ep_steps += self.ep_steps
        self.agent.hist.append(self.ep_reward)
        if self.ep_reward > self.agent.best_r:
            self.agent.best_r = self.ep_reward

        self.env.reset()
        self.astate   = START
        self.atgt     = [0.0, 0.0]
        self.apos     = [0.0, 0.0]
        self.anim_t   = 1.0
        self.trail    = []
        self.ep_reward= 0.0
        self.ep_steps = 0
        self.ep_done  = False

        if self.agent.episode % 20 == 0:
            self.best_path = self.agent.best_path(StochasticGridWorld)

    def _burst_particles(self, cx, cy, color, n=20):
        for _ in range(n):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(1.5, 5)
            self.particles.append(Particle(
                cx, cy, color,
                vx=math.cos(angle)*speed,
                vy=math.sin(angle)*speed,
                life=random.uniform(0.5, 1.4),
                size=random.randint(3, 7)
            ))

    # ──────────────────────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG)

        # Flash de tela
        if self.screen_flash > 0:
            s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            a = int(self.screen_flash * 80)
            s.fill((*self.flash_color, a))
            self.screen.blit(s, (0,0))

        self._draw_title()
        self._draw_grid()
        self._draw_best_path()
        self._draw_trail()
        self._draw_arrows()
        self._draw_q_values()
        self._draw_ghosts()
        self._draw_agent()
        self._draw_particles()
        self._draw_popups()
        self._draw_panel()
        self._draw_legend()
        self._draw_controls()

    def _draw_title(self):
        t = self.f_title.render("Grid World Estocástico  5×5  —  Q-Learning", True, C_WHITE)
        sub = self.f_small.render("Transições 80/10/10 · Fantasmas Dinâmicos · Poço de Penalidade", True, C_DIM)
        self.screen.blit(t,   (MARGIN_L, 12))
        self.screen.blit(sub, (MARGIN_L, 42))

    def _draw_grid(self):
        for r in range(GRID):
            for c in range(GRID):
                rect = cell_rect(r, c).inflate(-3,-3)
                state = (r, c)

                # Cor base
                if state == GOAL:
                    col = C_GOAL
                elif state == PIT:
                    col = C_PIT
                elif state == START:
                    col = C_START
                elif state in GHOST_CELLS:
                    col = blend(C_EMPTY, C_GHOST_Z, 0.5)
                else:
                    col = C_EMPTY

                # Gradiente de "perigo" baseado em Q mínimo
                q_min = float(np.min(self.agent.Q[r, c]))
                if q_min < -20:
                    danger = min(1.0, abs(q_min) / 100)
                    col = blend(col, (60, 5, 5), danger * 0.4)

                draw_rrect(self.screen, col, rect, 9, 1, C_GRID)

                # Labels especiais
                if state == GOAL:
                    self._blit_center(rect, "★ META", self.f_head, C_GOLD)
                    self._blit_center_off(rect, "+100", self.f_small, C_GREEN, 18)
                elif state == PIT:
                    self._blit_center(rect, "☠ POÇO", self.f_head, C_RED)
                    self._blit_center_off(rect, "-100", self.f_small, C_RED, 18)
                elif state == START:
                    self._blit_center_off(rect, "START", self.f_small, (100,200,255), -20)

                # Coordenada
                coord = self.f_small.render(f"({r},{c})", True, C_DIM)
                self.screen.blit(coord, (rect.right-38, rect.bottom-16))

    def _blit_center(self, rect, text, font, color):
        s = font.render(text, True, color)
        self.screen.blit(s, s.get_rect(center=rect.center))

    def _blit_center_off(self, rect, text, font, color, dy=0):
        s = font.render(text, True, color)
        cx, cy = rect.center
        self.screen.blit(s, s.get_rect(center=(cx, cy+dy)))

    def _draw_best_path(self):
        if len(self.best_path) < 2: return
        for i in range(len(self.best_path)-1):
            r0,c0 = self.best_path[i]
            r1,c1 = self.best_path[i+1]
            x0,y0 = cell_center(r0,c0)
            x1,y1 = cell_center(r1,c1)
            s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            pygame.draw.line(s, (255,200,60, 45), (x0,y0),(x1,y1), 4)
            self.screen.blit(s,(0,0))

    def _draw_trail(self):
        n = len(self.trail)
        for i, pos in enumerate(self.trail):
            alpha = int(160 * (i+1)/n)
            cx, cy = cell_center(int(round(pos[0])), int(round(pos[1])))
            sz = max(3, int(9 * (i+1)/n))
            s = pygame.Surface((sz*2+2, sz*2+2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*C_TRAIL, alpha), (sz+1,sz+1), sz)
            self.screen.blit(s, (cx-sz-1, cy-sz-1))

    def _draw_arrows(self):
        if not self.show_arrows: return
        for r in range(GRID):
            for c in range(GRID):
                if (r,c) in [GOAL, PIT]: continue
                best = int(np.argmax(self.agent.Q[r, c]))
                cx, cy = cell_center(r, c)
                draw_arrow(self.screen, C_BLUE, cx, cy, best, 14, 130)

    def _draw_q_values(self):
        if not self.show_q: return
        for r in range(GRID):
            for c in range(GRID):
                if (r,c) in [GOAL, PIT]: continue
                vmax = float(np.max(self.agent.Q[r, c]))
                color = C_GREEN if vmax > 0 else (C_RED if vmax < -5 else C_DIM)
                t = self.f_mono.render(f"{vmax:+.1f}", True, color)
                rect = cell_rect(r, c)
                self.screen.blit(t, (rect.x+5, rect.y+5))

    def _draw_ghosts(self):
        for i, gpos in enumerate(self.gpos):
            cx = int(MARGIN_L + gpos[1]*CELL + CELL//2)
            cy = int(MARGIN_T + gpos[0]*CELL + CELL//2)

            # Pulsar
            pulse = 0.85 + 0.15 * math.sin(self.time * 4 + i * math.pi)
            r = int(20 * pulse)

            # Brilho
            draw_glow(self.screen, C_GHOST, cx, cy, r+16, 40)

            # Corpo
            s = pygame.Surface((r*2+4, r*2+4), pygame.SRCALPHA)
            pygame.draw.circle(s, (*C_GHOST, 200), (r+2, r+2), r)
            # Olhos
            pygame.draw.circle(s, (255,255,255), (r-5, r-3), 5)
            pygame.draw.circle(s, (255,255,255), (r+5, r-3), 5)
            pygame.draw.circle(s, (30,0,50),     (r-4, r-3), 3)
            pygame.draw.circle(s, (30,0,50),     (r+6, r-3), 3)
            self.screen.blit(s, (cx-r-2, cy-r-2))

            # Label
            lbl = self.f_small.render(f"G{i+1}", True, C_PURPLE)
            self.screen.blit(lbl, (cx-10, cy+r+2))

    def _draw_agent(self):
        t = ease(self.anim_t)
        vr = lerp(self.apos[0], self.atgt[0], t)
        vc = lerp(self.apos[1], self.atgt[1], t)
        cx = int(MARGIN_L + vc*CELL + CELL//2)
        cy = int(MARGIN_T + vr*CELL + CELL//2)

        # Glow
        draw_glow(self.screen, C_AGENT, cx, cy, 28, 50)

        # Sombra
        pygame.draw.circle(self.screen, (0,0,0), (cx+3, cy+4), 20)
        # Corpo
        pygame.draw.circle(self.screen, C_AGENT, (cx, cy), 20)
        # Reflexo
        pygame.draw.circle(self.screen, (180,230,255), (cx-5, cy-6), 7)
        # Borda
        pygame.draw.circle(self.screen, (160,220,255), (cx, cy), 20, 2)

    def _draw_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _draw_popups(self):
        for p in self.popups:
            p.draw(self.screen, self.f_popup)

    # ──────────────────────────────────────────────────────────────
    def _draw_panel(self):
        px = MARGIN_L + GRID*CELL + 18
        pw = PANEL_W - 10
        py = MARGIN_T
        ph = GRID*CELL

        draw_rrect(self.screen, C_PANEL, pygame.Rect(px, py, pw, ph), 14, 1, C_BORDER)

        x  = px + 14
        y  = py + 14

        def line(text, color=C_WHITE, font=None, indent=0):
            nonlocal y
            f = font or self.f_body
            s = f.render(text, True, color)
            self.screen.blit(s, (x+indent, y))
            y += s.get_height() + 5

        def sep(thick=1):
            nonlocal y
            pygame.draw.line(self.screen, C_BORDER, (x,y+2),(x+pw-28,y+2), thick)
            y += 11

        line("📊 Estatísticas", C_GOLD, self.f_head)
        sep(2)

        ep_r_col = C_GREEN if self.ep_reward >= 0 else C_RED
        stats = [
            ("Episódio",    f"{self.agent.episode:>6}",        C_WHITE),
            ("Passos",      f"{self.ep_steps:>6}",             C_WHITE),
            ("ε (explor.)", f"{self.agent.eps*100:>5.1f}%",    C_BLUE),
            ("Ret. atual",  f"{self.ep_reward:>+.1f}",         ep_r_col),
            ("Melhor ret.", f"{self.agent.best_r:>+.1f}",      C_GOLD),
            ("Vitórias",    f"{self.agent.wins:>6}",           C_GREEN),
            ("Mortes",      f"{self.agent.deaths:>6}",         C_RED),
        ]
        for label, val, col in stats:
            s1 = self.f_body.render(f"{label}:", True, C_DIM)
            s2 = self.f_body.render(val, True, col)
            self.screen.blit(s1, (x, y))
            self.screen.blit(s2, (x + pw - s2.get_width() - 28, y))
            y += s1.get_height() + 5

        sep()
        line("📈 Recompensa / Ep.", C_GOLD, self.f_head)
        self._mini_chart(x, y, pw-28, 65)
        y += 73

        sep()
        line("⚙ Parâmetros MDP", C_GOLD, self.f_head)
        params = [
            ("γ desconto",  "0.90"),
            ("α aprendiz.", "0.15"),
            ("Trans. 80%",  "→ direção"),
            ("Trans. 10%",  "→ esq./dir."),
            ("Vel. sim.",   f"{self.speed}/15"),
        ]
        for k, v in params:
            s1 = self.f_small.render(f"{k}:", True, C_DIM)
            s2 = self.f_small.render(v, True, C_WHITE)
            self.screen.blit(s1, (x, y))
            self.screen.blit(s2, (x + pw - s2.get_width() - 28, y))
            y += s1.get_height() + 4

        sep()
        line("🔢 Bellman V(1,0) 'Leste'", C_GOLD, self.f_head)
        bellman = [
            "0.80·(-0.1+0.9·V(1,1))",
            "+0.10·(-0.1+0.9·V(0,0))",
            "+0.10·(-0.1+0.9·V(2,0))",
            "≈ 0.287",
        ]
        for b in bellman:
            line(b, C_BLUE if "≈" not in b else C_GREEN, self.f_mono)

    def _mini_chart(self, x, y, w, h):
        hist = self.agent.hist[-50:]
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (8,10,22), rect, border_radius=6)
        if len(hist) < 2: return
        vmin, vmax = min(hist), max(hist)
        rng = max(1, vmax - vmin)
        pts = []
        for i, v in enumerate(hist):
            px_ = x + int(i / (len(hist)-1) * w)
            py_ = y + h - int((v-vmin)/rng*(h-4)) - 2
            pts.append((px_, py_))
        pygame.draw.lines(self.screen, C_GOLD, False, pts, 2)
        # linha zero
        zy = y + h - int((0-vmin)/rng*(h-4)) - 2
        pygame.draw.line(self.screen, C_BORDER, (x, zy), (x+w, zy), 1)

    def _draw_legend(self):
        items = [
            (C_GOAL,    "Meta (+100)"),
            (C_PIT,     "Poço (-100)"),
            (C_GHOST,   "Fantasma (-50)"),
            (C_AGENT,   "Agente"),
            (C_TRAIL,   "Trilha"),
        ]
        bx = MARGIN_L
        by = MARGIN_T + GRID*CELL + 8
        for color, label in items:
            pygame.draw.circle(self.screen, color, (bx+6, by+8), 6)
            s = self.f_small.render(label, True, C_DIM)
            self.screen.blit(s, (bx+16, by+1))
            bx += s.get_width() + 36

    def _draw_controls(self):
        keys = [
            "ESPAÇO=Pausa", "↑↓=Vel", "Q=Q-vals",
            "A=Setas", "R=Reset", "ESC=Sair"
        ]
        bx = MARGIN_L
        by = WIN_H - 22
        for k in keys:
            parts = k.split("=")
            s1 = self.f_small.render(f"[{parts[0]}]", True, C_GOLD)
            s2 = self.f_small.render(f" {parts[1]}  ", True, C_DIM)
            self.screen.blit(s1, (bx, by))
            self.screen.blit(s2, (bx + s1.get_width(), by))
            bx += s1.get_width() + s2.get_width()

        if self.paused:
            s = self.f_head.render("⏸  PAUSADO", True, C_GOLD)
            self.screen.blit(s, s.get_rect(
                center=(MARGIN_L + GRID*CELL//2, WIN_H - 10)))


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(__doc__)

    # Demonstração do cálculo de Bellman
    print("=" * 62)
    print("  EQUAÇÃO DE BELLMAN — V(1,0) ação=Leste")
    print("=" * 62)
    V  = {(1,1): 0.5, (0,0): 0.0, (2,0): 0.3}
    g  = 0.9; R = -0.1
    v10 = (0.80*(R + g*V[(1,1)]) +
           0.10*(R + g*V[(0,0)]) +
           0.10*(R + g*V[(2,0)]))
    print(f"  V(1,1)={V[(1,1)]}, V(0,0)={V[(0,0)]}, V(2,0)={V[(2,0)]}, γ={g}")
    print(f"  = 0.80·({R}+{g}·{V[(1,1)]}) + 0.10·({R}+{g}·{V[(0,0)]}) + 0.10·({R}+{g}·{V[(2,0)]})")
    print(f"  = 0.80·{R+g*V[(1,1)]:.3f} + 0.10·{R+g*V[(0,0)]:.3f} + 0.10·{R+g*V[(2,0)]:.3f}")
    print(f"  V(1,0) ≈ {v10:.4f}")
    det = R + g*V[(1,1)]
    print(f"\n  Determinístico: V(1,0) = {R} + {g}·{V[(1,1)]} = {det:.4f}")
    print(f"  Diferença pela estocasticidade: {v10-det:+.4f}")
    print("=" * 62)
    print("\n  Iniciando Pygame... [pip install pygame numpy]")
    print("  Controles: ESPAÇO pausar | ↑↓ velocidade | R reset | ESC sair")
    print("=" * 62 + "\n")

    try:
        App().run()
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback; traceback.print_exc()
        print("\nInstale as dependências: pip install pygame numpy")