#pip install pygame-ce 설치하고ㅜ cmd에다가 python christmas_tree.py 이거 작성

import sys
import time
import numpy as np

TAU = np.pi * 2

# ───────────────────────── 색 ─────────────────────────
GREEN = (40, 170, 70)
RED = (235, 40, 40)
GOLD = (255, 210, 60)
WARM = (255, 235, 150)
BROWN = (120, 75, 40)
ORANGE = (255, 140, 0)
MAGENTA = (255, 80, 150)
SNOW_WHITE = (240, 245, 255)

# ─────────────────────── 점 종류 / 문자 ───────────────────────
NEEDLE, TRUNK, GARLAND, BAUBLE, LIGHT, GIFT, STAR, SNOW_HEAP = range(8)

# 밝기가 낮은 것 -> 높은 것 순서 (10글자로 맞춤)
_RAMPS = {
    NEEDLE: ".,:;+xvX%&",
    TRUNK: "-=+=#%#%##",
    GARLAND: ".:o0o0O0O@",
    BAUBLE: "..oo0OO0@@",
    LIGHT: ".oO0@#@$%&",
    GIFT: "-=+#+#%#%@",
    STAR: "..**##@@&&",
    SNOW_HEAP: ".:*=+*#%@@",
}
RAMP = np.array([[ord(c) for c in _RAMPS[k]] for k in range(8)], dtype=np.uint8)

MESSAGE = "MERRY CHRISTMAS 2026"
T_MSG = 12.0  # 이 시간(초) 이후 문구가 타이핑됨
LAYERS = 9  # 가지 층 수


# ───────────────────────── 트리 모양 ─────────────────────────
def y_world(y):
    """0(바닥)~1(꼭대기) -> 월드 좌표 높이"""
    return (y - 0.45) * 1.6


def surf_r(y):
    """높이 y에서 트리 표면 반지름. 층마다 아래가 넓은 계단 모양."""
    env = 0.62 * (1 - y)
    f = (y * LAYERS) % 1.0
    return env * (0.62 + 0.38 * (1 - f))


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)


def _sphere(rng, c, rad, n=70):
    d = _unit(rng.normal(size=(n, 3)))
    return np.asarray(c) + d * rad, d


def _box(rng, c, size, body, rib, n=1400):
    """상자 표면에 점을 뿌리고 십자 리본 색을 입힌다."""
    size = np.array(size, float)
    area = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])
    ax = rng.choice(3, n, p=area / area.sum())
    sg = rng.choice([-1.0, 1.0], n)
    sg[ax == 1] = 1.0  # 바닥은 안 보이니 위쪽 면만
    u = rng.uniform(-0.5, 0.5, (n, 3))
    p = u * size
    idx = np.arange(n)
    p[idx, ax] = sg * 0.5 * size[ax]
    nrm = np.zeros((n, 3))
    nrm[idx, ax] = sg
    au = np.abs(u / 0.5)
    au[idx, ax] = 9.0
    ribbon = au.min(axis=1) < 0.18
    col = np.where(ribbon[:, None], np.array(rib, float), np.array(body, float))
    return p + np.asarray(c), nrm, col


def build(seed=7):
    """장면 전체를 (위치, 법선, 색, 종류, 등장시간, 위상) 배열로 만든다."""
    rng = np.random.default_rng(seed)
    parts = []
    
    def add(P, N, C, kind, birth, phase=None):
        n = len(P)
        parts.append((
            np.asarray(P, float), _unit(np.asarray(N, float)), np.asarray(C, float),
            np.full(n, kind), np.broadcast_to(np.asarray(birth, float), (n,)).copy(),
            rng.random(n) if phase is None else np.broadcast_to(phase, (n,)).copy(),
        ))

    # 잎 (아래부터 위로 자라듯 등장)
    m = 18000
    y = rng.random(m) ** 0.9
    r = surf_r(y) * (1 - 0.5 * rng.random(m) ** 3)
    phi = rng.random(m) * TAU
    P = np.stack([np.cos(phi) * r, y_world(y), np.sin(phi) * r], 1)
    N = np.stack([np.cos(phi), np.full(m, 0.5), np.sin(phi)], 1)
    C = np.array(GREEN) * (0.6 + 0.5 * rng.random((m, 1)))
    add(P, N, C, NEEDLE, 0.4 + 4.5 * y)

    # 줄기
    m = 1000
    Y = rng.uniform(-1.05, -0.65, m)
    phi = rng.random(m) * TAU
    r = 0.08 * (0.6 + 0.4 * rng.random(m))
    P = np.stack([np.cos(phi) * r, Y, np.sin(phi) * r], 1)
    N = np.stack([np.cos(phi), np.zeros(m), np.sin(phi)], 1)
    add(P, N, np.array(BROWN) * (0.7 + 0.5 * rng.random((m, 1))), TRUNK, 0.0)

    # 바닥에 쌓인 눈 더미
    m_snow = 1200
    sx = rng.uniform(-0.9, 0.9, m_snow)
    sz = rng.uniform(-0.9, 0.9, m_snow)
    sy = np.full(m_snow, -1.02) + rng.normal(0, 0.02, m_snow)
    P = np.stack([sx, sy, sz], 1)
    N = np.stack([np.zeros(m_snow), np.ones(m_snow), np.zeros(m_snow)], 1)
    C = np.array(SNOW_WHITE) * (0.8 + 0.2 * rng.random((m_snow, 1)))
    add(P, N, C, SNOW_HEAP, 0.0)

    # 금빛 전구줄 (줄 자체를 두껍게 만들기 위해 나선 주위로 폭을 넓혀서 여러 겹의 미세 점들을 밀집시킴)
    n = 3500
    y = np.linspace(0.05, 0.91, n)
    phi = y * 36 + rng.normal(0, 0.03, n)
    # 두께감을 주기 위해 표면 바깥쪽 두께 범위를 넓힘 (0.02 ~ 0.045)
    r = surf_r(y) + rng.uniform(0.02, 0.045, n)
    P = np.stack([np.cos(phi) * r, y_world(y), np.sin(phi) * r], 1)
    N = np.stack([np.cos(phi), np.full(n, 0.3), np.sin(phi)], 1)
    add(P, N, np.array(GOLD) * np.ones((n, 1)), GARLAND, 5.0 + 3.0 * y)

    # 반짝이는 동그라미 전구 (서로 겹치지 않게 간격을 두고 고르게 분산 배치)
    n_lights = 180
    y_lights = np.linspace(0.1, 0.85, n_lights) + rng.normal(0, 0.02, n_lights)
    phi_lights = np.linspace(0, TAU, n_lights, endpoint=False) + rng.uniform(-0.1, 0.1, n_lights)
    # 각도 순서를 살짝 섞어주되 겹침 방지
    rng.shuffle(phi_lights)
    r_lights = surf_r(y_lights) * 1.06 + 0.03
    P_lights = np.stack([np.cos(phi_lights) * r_lights, y_world(y_lights), np.sin(phi_lights) * r_lights], 1)
    N_lights = np.stack([np.cos(phi_lights), np.full(n_lights, 0.3), np.sin(phi_lights)], 1)
    pal = np.array([WARM, RED, GOLD, ORANGE, MAGENTA, WARM], float)
    add(P_lights, N_lights, pal[rng.integers(0, len(pal), n_lights)], LIGHT, 6.0 + 3.0 * rng.random(n_lights))

    # 오너먼트 (방울 장식도 겹치지 않게 분산)
    n_baubles = 30
    for k in range(n_baubles):
        y = 0.1 + (k / n_baubles) * 0.75 + rng.uniform(-0.01, 0.01)
        phi = (k * 2.4) + rng.uniform(-0.1, 0.1) # 일정한 간격 유지
        r = surf_r(y) * 0.98 + 0.02
        c = (np.cos(phi) * r, y_world(y), np.sin(phi) * r)
        P, N = _sphere(rng, c, 0.04 + 0.02 * rng.random(), n=50)
        col = RED if k % 3 == 0 else (GOLD if k % 3 == 1 else MAGENTA)
        add(P, N, np.array(col) * np.ones((len(P), 1)), BAUBLE, 4.5 + 2.0 * rng.random())

    # 선물 상자 (고정형)
    gifts = [
        (0.60, 0.32, (0.26, 0.22, 0.26), RED, GOLD),
        (-0.68, 0.25, (0.32, 0.18, 0.24), MAGENTA, GOLD),
        (0.18, 0.78, (0.24, 0.20, 0.22), (40, 160, 80), RED),
        (-0.35, -0.78, (0.22, 0.24, 0.22), GOLD, RED),
    ]
    for i, (gx, gz, size, body, rib) in enumerate(gifts):
        c = (gx, -1.0 + size[1] / 2, gz)
        P, N, C = _box(rng, c, size, body, rib)
        add(P, N, C, GIFT, 7.5 + i * 0.4)

    # 꼭대기 별 (작고 아담하게 유지)
    n = 800
    th = rng.random(n) * TAU
    rad = 0.03 + 0.08 * np.abs(np.cos(2.5 * (th - np.pi / 2))) ** 2.5
    rho = np.sqrt(rng.random(n)) * rad
    z = rng.uniform(-0.02, 0.02, n)
    P = np.stack([rho * np.cos(th), y_world(1.0) + 0.04 + rho * np.sin(th), z], 1)
    N = np.stack([0.5 * np.cos(th), 0.5 * np.sin(th), np.sign(z) * 1.0], 1)
    add(P, N, np.array(GOLD) * np.ones((n, 1)), STAR, 9.0)

    cat = lambda i: np.concatenate([p[i] for p in parts])
    return tuple(cat(i) for i in range(6))


# ───────────────────────── 눈 (배경 내리는 눈) ─────────────────────────
class Snow:
    def __init__(self, cols, rows, n=400, seed=3):
        rng = np.random.default_rng(seed)
        self.cols, self.rows = cols, rows
        self.c0 = rng.random(n) * cols
        self.r0 = rng.random(n) * rows
        self.speed = rng.uniform(1.5, 6.0, n)
        self.ph = rng.random(n) * TAU
        self.size = rng.integers(0, 3, n)

    def draw(self, ch, col, t):
        r = ((self.r0 + self.speed * t) % self.rows).astype(int)
        c = ((self.c0 + np.sin(t * 0.7 + self.ph) * 2) % self.cols).astype(int)
        ch[r, c] = np.array([ord("."), ord("+"), ord("*")], np.uint8)[self.size]
        lvl = np.array([130, 190, 245])[self.size]
        col[r, c] = lvl[:, None]


# ───────────────────────── 렌더링 ─────────────────────────
LIGHT_DIR = np.array([-0.4, 0.6, 0.7])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def render(scene, snow, t, cols, rows, aspect):
    P, N, C, K, B, PH = scene
    ch = np.full((rows, cols), 32, np.uint8)
    col = np.zeros((rows, cols, 3), np.uint8)
    
    snow.draw(ch, col, t)

    vis = B <= t
    p, n, c, k, ph = P[vis], N[vis], C[vis], K[vis], PH[vis]

    is_rot = (k != GIFT) & (k != STAR) & (k != SNOW_HEAP)
    
    p_rot, n_rot = p[is_rot], n[is_rot]
    p_fix, n_fix = p[~is_rot], n[~is_rot]

    a, tilt = t * 0.40, 0.12
    ca, sa, ct, st = np.cos(a), np.sin(a), np.cos(tilt), np.sin(tilt)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    R = Rx @ Ry
    
    p_rot = p_rot @ R.T
    n_rot = n_rot @ R.T

    p_final = np.empty_like(p)
    n_final = np.empty_like(n)
    p_final[is_rot] = p_rot
    p_final[~is_rot] = p_fix
    n_final[is_rot] = n_rot
    n_final[~is_rot] = n_fix
    p, n = p_final, n_final

    dist = 4.0
    s = dist / (dist - p[:, 2])
    rpu = rows * 0.42  
    cpu = rpu * aspect 
    cc = np.round(cols / 2 + p[:, 0] * s * cpu).astype(int)
    rr = np.round(rows * 0.44 - p[:, 1] * s * rpu).astype(int)
    ok = (cc >= 0) & (cc < cols) & (rr >= 0) & (rr < rows)
    p, n, c, k, ph, cc, rr = p[ok], n[ok], c[ok], k[ok], ph[ok], cc[ok], rr[ok]

    order = np.argsort(-p[:, 2])
    _, first = np.unique((rr * cols + cc)[order], return_index=True)
    sel = order[first]
    p, n, c, k, ph, cc, rr = p[sel], n[sel], c[sel], k[sel], ph[sel], cc[sel], rr[sel]

    lum = 0.30 + 0.70 * np.clip(n @ LIGHT_DIR, 0, 1)
    tw = 0.5 + 0.5 * np.sin(t * 4.0 + ph * TAU)
    lum = np.where(k == LIGHT, 0.75 + 0.25 * tw, lum)
    lum = np.where(k == STAR, 0.95 + 0.05 * np.sin(t * 3.0), lum)
    lum = np.where(k == GARLAND, 0.65 + 0.35 * lum, lum)
    lum = np.where(k == SNOW_HEAP, 0.8 + 0.2 * np.sin(t + ph), lum)
    lum = np.clip(lum, 0, 1)

    ch[rr, cc] = RAMP[k, (lum * 9.99).astype(int)]
    ch[rr, cc] = np.where(k == LIGHT, ord("O"), ch[rr, cc])
    ch[rr, cc] = np.where(k == GARLAND, ord("o"), ch[rr, cc])
    ch[rr, cc] = np.where(k == STAR, ord("*"), ch[rr, cc])
    ch[rr, cc] = np.where(k == SNOW_HEAP, ord("*"), ch[rr, cc])
    
    glow = np.where((k == LIGHT) | (k == STAR), 0.90 + 0.10 * lum, 0.35 + 0.65 * lum)
    col[rr, cc] = np.clip(c * glow[:, None], 0, 255).astype(np.uint8)

    if t > T_MSG:
        msg = "  ".join(" ".join(w) for w in MESSAGE.split())
        shown = int((t - T_MSG) * 10)
        row, x0 = rows - 2, (cols - len(msg)) // 2
        palette = [RED, GOLD, MAGENTA]
        for i, chr_ in enumerate(msg[:shown]):
            if chr_ != " " and 0 <= x0 + i < cols:
                ch[row, x0 + i] = ord(chr_)
                tint = palette[(i // 2 + int(t * 1.5)) % 3]
                col[row, x0 + i] = np.array(tint) * (0.85 + 0.15 * np.sin(t * 3 + i))
    return ch, col


# ───────────────────────── pygame 창 ─────────────────────────
def run_pygame():
    import pygame

    COLS, ROWS, CW, CH, FONT_PX = 160, 75, 7, 14, 13
    pygame.init()
    screen = pygame.display.set_mode((COLS * CW, ROWS * CH))
    pygame.display.set_caption("ASCII Christmas Tree - Widescreen Edition")
    font = pygame.font.SysFont("consolas,menlo,dejavusansmono,couriernew,monospace", FONT_PX)
    hint = font.render("Space pause - R restart - S screenshot - Esc quit", True, (120, 130, 150))

    scene, snow = build(), Snow(COLS, ROWS)
    cache = {}

    def glyph(code, rgb):
        key = (code, int(rgb[0]) // 24, int(rgb[1]) // 24, int(rgb[2]) // 24)
        g = cache.get(key)
        if g is None:
            g = cache[key] = font.render(chr(code), True, tuple(int(v) for v in rgb))
        return g

    clock, t, paused = pygame.time.Clock(), 0.0, False
    while True:
        dt = clock.tick(30) / 1000
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return
                if e.key == pygame.K_SPACE:
                    paused = not paused
                if e.key == pygame.K_r:
                    t = 0.0
                if e.key == pygame.K_s:
                    pygame.image.save(screen, "tree_widescreen.png")
        if not paused:
            t += dt

        ch, col = render(scene, snow, t, COLS, ROWS, CH / CW)
        screen.fill((0, 0, 0))
        for y, x in zip(*np.nonzero(ch != 32)):
            g = glyph(ch[y, x], col[y, x])
            screen.blit(g, (x * CW + (CW - g.get_width()) // 2,
                            y * CH + (CH - g.get_height()) // 2))
        screen.blit(hint, (8, 6))
        pygame.display.flip()


# ───────────────────────── 터미널(cmd) ─────────────────────────
def run_text():
    import os
    import shutil

    os.system("")
    size = shutil.get_terminal_size((160, 50))
    cols, rows = min(size.columns - 1, 160), min(size.lines - 2, 60)

    scene, snow = build(), Snow(cols, rows)
    out = sys.stdout
    out.write("\x1b[2J\x1b[?25l")
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            ch, col = render(scene, snow, t, cols, rows, 2.0)
            col = (col // 16) * 16
            lines = []
            for y in range(rows):
                buf, last = [], None
                for x in range(cols):
                    if ch[y, x] == 32:
                        buf.append(" ")
                        continue
                    rgb = (int(col[y, x, 0]), int(col[y, x, 1]), int(col[y, x, 2]))
                    if rgb != last:
                        buf.append("\x1b[38;2;%d;%d;%dm" % rgb)
                        last = rgb
                    buf.append(chr(ch[y, x]))
                lines.append("".join(buf) + "\x1b[0m")
            out.write("\x1b[H" + "\n".join(lines))
            out.flush()
            time.sleep(1 / 20)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\n")


if __name__ == "__main__":
    if "--text" in sys.argv:
        run_text()
    else:
        try:
            import pygame  # noqa: F401
        except ImportError:
            print("pygame이 없어서 터미널 모드로 실행합니다. (pip install pygame 하면 창 모드 가능)")
            time.sleep(1.5)
            run_text()
        else:
            run_pygame()
