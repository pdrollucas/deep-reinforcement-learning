# Grid World — Aprendizado por Reforço

Dois ambientes de Grid World implementados em Python puro com visualização em Pygame, cobrindo conceitos fundamentais de RL: Q-Learning, Equação de Bellman e políticas estocásticas.

---

## Etapas

### Etapa 1 — Grid World Determinístico (4×4)
Ambiente simples onde o agente aprende a navegar de `(0,0)` até `(3,3)` com recompensas de -1 por passo, -5 por colisão com borda e +10 ao atingir a meta.

```
python gridworld_rl.py
```

Resultado:

<img width="972" height="762" alt="image" src="https://github.com/user-attachments/assets/8131ea83-102c-422e-bca7-cf15f8b9091f" />


### Etapa 2 — Grid World Estocástico (5×5)
Ambiente com incerteza nas transições (80/10/10), fantasmas dinâmicos e um poço de penalidade severa. O agente navega de `(0,0)` até `(4,4)`.

```
python gridworld_stochastic.py
```

Resultado:

<img width="964" height="756" alt="image" src="https://github.com/user-attachments/assets/c883f94a-d066-4760-96dc-a2e05e74e842" />


---

## Instalação

```bash
pip install pygame numpy
```

---

## Controles (ambos os scripts)

| Tecla | Ação |
|---|---|
| `ESPAÇO` | Pausar / Retomar |
| `↑` / `↓` | Aumentar / Diminuir velocidade |
| `Q` | Mostrar/ocultar Q-valores |
| `R` | Reiniciar treinamento |
| `ESC` | Sair |

---

## Conceitos Abordados

- Q-Learning com ε-greedy
- Equação de Bellman (determinística e estocástica)
- Fator de desconto γ e taxa de aprendizado α
- Transições estocásticas 80/10/10
- Convergência da função de valor V(s)
