"""
Лабораторная работа №1
Число обусловленности и чувствительность на дискретной модели:
число решений уравнения a^y = c (mod b)
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. РЕАЛИЗАЦИИ ДЛЯ ПОЛНОГО ДИАПАЗОНА (Раздел 4)
# =====================================================================

def count_naive(a: int, b: int, c: int) -> int:
    """
    Наивная реализация: для каждого y вычисляет a^y mod b
    через y последовательных модульных умножений.
    Соглашение: 0^0 = 1.
    """
    count = 0
    for y in range(b):
        if y == 0:
            res = 1 % b
        else:
            res = 1 % b
            for _ in range(y):
                res = (res * a) % b
        if res == c:
            count += 1
    return count


def count_fastpow(a: int, b: int, c: int) -> int:
    """
    Быстрое возведение в степень через встроенный pow(a, y, b).
    Соглашение: 0^0 = 1.
    """
    count = 0
    for y in range(b):
        if y == 0:
            res = 1 % b
        else:
            res = pow(a, y, b)
        if res == c:
            count += 1
    return count


def count_iter(a: int, b: int, c: int) -> int:
    """
    Итеративная реализация: один проход по y,
    каждый следующий остаток получается умножением предыдущего на a mod b.
    """
    count = 0
    cur = 1 % b  # y = 0: a^0 = 1
    if cur == c:
        count += 1
    for y in range(1, b):
        cur = (cur * a) % b
        if cur == c:
            count += 1
    return count


# =====================================================================
# 2. ЧУВСТВИТЕЛЬНОСТЬ И ЧИСЛО ОБУСЛОВЛЕННОСТИ (Раздел 7)
# =====================================================================

def get_perturbations(p: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """
    Возвращает список допустимых возмущений (до 6 соседей):
    - a' = (a +- 1) mod b, b' = b, c' = c
    - c' = (c +- 1) mod b, a' = a, b' = b
    - b' = b +- 1 (при b' >= 2), a' = a mod b', c' = c mod b'
    """
    a, b, c = p
    neighbors = []

    # Возмущения a
    for da in [-1, 1]:
        neighbors.append(((a + da) % b, b, c))

    # Возмущения c
    for dc in [-1, 1]:
        neighbors.append((a, b, (c + dc) % b))

    # Возмущения b
    for db in [-1, 1]:
        b_prime = b + db
        if b_prime >= 2:
            neighbors.append((a % b_prime, b_prime, c % b_prime))

    return neighbors


def sensitivity(p: tuple[int, int, int], p_prime: tuple[int, int, int], 
                reg: bool = True) -> float:
    """
    Вычисление чувствительности S(p, p' - p).
    reg = True использует регуляризацию ||x|| -> max(|x|, 1).
    reg = False возвращает float('inf') при x = 0 и Delta x != 0.
    """
    a, b, c = p
    ap, bp, cp = p_prime

    x = count_iter(a, b, c)
    x_prime = count_iter(ap, bp, cp)
    delta_x = abs(x_prime - x)

    # Относительное изменение решения
    if reg:
        norm_x = max(abs(x), 1)
        rel_x = delta_x / norm_x
    else:
        if x == 0:
            rel_x = float('inf') if delta_x != 0 else 0.0
        else:
            rel_x = delta_x / abs(x)

    # Относительное изменение параметров: max_i (|Delta p_i| / max(|p_i|, 1))
    dp_a = abs(ap - a) / max(abs(a), 1)
    dp_b = abs(bp - b) / max(abs(b), 1)
    dp_c = abs(cp - c) / max(abs(c), 1)
    rel_p = max(dp_a, dp_b, dp_c)

    if rel_p == 0.0:
        return 0.0

    return rel_x / rel_p


def kappa_lab(p: tuple[int, int, int]) -> float:
    """
    Оценка числа обусловленности kappa_lab(p) по всем допустимым возмущениям.
    """
    neighbors = get_perturbations(p)
    s_vals = []
    for p_prime in neighbors:
        s = sensitivity(p, p_prime, reg=True)
        s_vals.append(s)

    return max(s_vals) if s_vals else 0.0


# =====================================================================
# 3. РАЗРЕЖЕННАЯ МОДЕЛЬ (Раздел 8)
# =====================================================================

def get_sparse_Y(b: int, k: int = 500) -> list[int]:
    """Генерация разреженного множества Y(b)."""
    return [math.floor(i * (b - 1) / (k - 1)) for i in range(k)]


def count_sp_naive(a: int, b: int, c: int, k: int = 500) -> int:
    """Разреженная модель: наивное возведение для каждого y in Y(b)."""
    Y = get_sparse_Y(b, k)
    count = 0
    for y in Y:
        if y == 0:
            res = 1 % b
        else:
            res = 1 % b
            for _ in range(y):
                res = (res * a) % b
        if res == c:
            count += 1
    return count


def count_sp_fastpow(a: int, b: int, c: int, k: int = 500) -> int:
    """Разреженная модель: быстрое возведение для каждого y in Y(b)."""
    Y = get_sparse_Y(b, k)
    count = 0
    for y in Y:
        res = (1 % b) if y == 0 else pow(a, y, b)
        if res == c:
            count += 1
    return count


def count_sp_iter(a: int, b: int, c: int, k: int = 500) -> int:
    """
    Разреженная модель: сканирование полного диапазона y in {0, ..., b-1}
    с инкрементальным пересчетом и проверкой принадлежности к Y(b).
    """
    Y_set = set(get_sparse_Y(b, k))
    count = 0
    cur = 1 % b
    if 0 in Y_set and cur == c:
        count += 1
    for y in range(1, b):
        cur = (cur * a) % b
        if y in Y_set and cur == c:
            count += 1
    return count


# =====================================================================
# 4. ВЫЧИСЛИТЕЛЬНЫЕ ЭКСПЕРИМЕНТЫ И ПОСТРОЕНИЕ ГРАФИКОВ
# =====================================================================

def run_all_experiments():
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ 1: Проверочная таблица (Раздел 6.1)")
    print("=" * 70)
    test_cases = [
        (2, 7, 4),
        (2, 7, 3),
        (3, 11, 1),
        (0, 7, 0),
        (0, 7, 1)
    ]
    print(f"{'a':<5}{'b':<5}{'c':<5}{'Naive':<10}{'Fastpow':<10}{'Iter':<10}{'Совпадение':<12}")
    for a, b, c in test_cases:
        n_res = count_naive(a, b, c)
        f_res = count_fastpow(a, b, c)
        i_res = count_iter(a, b, c)
        match = (n_res == f_res == i_res)
        print(f"{a:<5}{b:<5}{c:<5}{n_res:<10}{f_res:<10}{i_res:<10}{str(match):<12}")

    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 2: Замеры времени на полном диапазоне (Раздел 6.2)")
    print("=" * 70)
    b_vals = [100, 300, 1000, 3000, 10000]
    a_fixed, c_fixed = 2, 1
    times_naive, times_fastpow, times_iter = [], [], []

    for b in b_vals:
        # count_naive
        t0 = time.perf_counter()
        count_naive(a_fixed, b, c_fixed)
        times_naive.append(time.perf_counter() - t0)

        # count_fastpow
        t0 = time.perf_counter()
        count_fastpow(a_fixed, b, c_fixed)
        times_fastpow.append(time.perf_counter() - t0)

        # count_iter
        t0 = time.perf_counter()
        count_iter(a_fixed, b, c_fixed)
        times_iter.append(time.perf_counter() - t0)

    print(f"{'b':<10}{'t_naive (c)':<18}{'t_fastpow (c)':<18}{'t_iter (c)':<18}")
    for i, b in enumerate(b_vals):
        print(f"{b:<10}{times_naive[i]:<18.6f}{times_fastpow[i]:<18.6f}{times_iter[i]:<18.6f}")

    # Оценка наклонов в log-шкале: log(t) = k * log(b) + const
    log_b = np.log(b_vals)
    slope_naive, _ = np.polyfit(log_b, np.log(times_naive), 1)
    slope_fastpow, _ = np.polyfit(log_b, np.log(times_fastpow), 1)
    slope_iter, _ = np.polyfit(log_b, np.log(times_iter), 1)

    print(f"\nНаклоны графиков в log-шкале:")
    print(f"count_naive:   {slope_naive:.2f} (теория: 2)")
    print(f"count_fastpow: {slope_fastpow:.2f} (теория: ~1)")
    print(f"count_iter:    {slope_iter:.2f} (теория: 1)")
    print(f"При b = 10^4 count_naive медленнее count_iter в {times_naive[-1]/times_iter[-1]:.1f} раз")

    # Построение графика времени
    plt.figure(figsize=(9, 6))
    plt.loglog(b_vals, times_naive, 'o-', label=f'count_naive (slope ≈ {slope_naive:.2f})')
    plt.loglog(b_vals, times_fastpow, 's-', label=f'count_fastpow (slope ≈ {slope_fastpow:.2f})')
    plt.loglog(b_vals, times_iter, '^-', label=f'count_iter (slope ≈ {slope_iter:.2f})')
    plt.xlabel('Параметр b (размер диапазона)')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Зависимость времени выполнения от b на полном диапазоне (log-log)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('timing_full_range.png', dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 3: Мини-исследование при b <= 50 (Раздел 7.2, п. 5)")
    print("=" * 70)
    n_zero_count = 0
    n_ge_2_count = 0
    total_points = 0
    all_N = []
    max_kappa = -1.0
    max_kappa_point = None
    stable_point = None

    delta_N_fraction_by_b = {}

    for b in range(2, 51):
        non_zero_delta_for_b = 0
        total_b_pts = 0
        for a in range(b):
            for c in range(b):
                p = (a, b, c)
                total_points += 1
                total_b_pts += 1
                val = count_iter(a, b, c)
                all_N.append(val)
                if val == 0:
                    n_zero_count += 1
                if val >= 2:
                    n_ge_2_count += 1

                # Оценка kappa_lab
                k_val = kappa_lab(p)
                if k_val > max_kappa:
                    max_kappa = k_val
                    max_kappa_point = p

                if k_val == 0.0 and stable_point is None:
                    stable_point = p

                # Проверка, есть ли хотя бы одно Delta N != 0 среди возмущений
                has_delta = False
                for p_prime in get_perturbations(p):
                    if count_iter(p_prime[0], p_prime[1], p_prime[2]) != val:
                        has_delta = True
                        break
                if has_delta:
                    non_zero_delta_for_b += 1

        delta_N_fraction_by_b[b] = non_zero_delta_for_b / total_b_pts

    print(f"Всего точек (b <= 50): {total_points}")
    print(f"Количество точек с N = 0: {n_zero_count} ({n_zero_count/total_points*100:.2f}%)")
    print(f"Количество точек с N >= 2: {n_ge_2_count} ({n_ge_2_count/total_points*100:.2f}%)")
    print(f"Точка с максимальной оценкой kappa_lab: p = {max_kappa_point}, kappa = {max_kappa:.4f}")
    print(f"Пример стабильной точки (kappa_lab = 0): p = {stable_point}")

    # Гистограмма N
    plt.figure(figsize=(9, 5))
    plt.hist(all_N, bins=range(0, max(all_N) + 2), align='left', rwidth=0.8, color='steelblue', edgecolor='black')
    plt.xlabel('Значение N(a, b, c)')
    plt.ylabel('Количество точек')
    plt.title('Распределение значений N(a, b, c) при b ≤ 50')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('hist_N_distribution.png', dpi=300)
    plt.close()

    # График доли точек с Delta N != 0
    plt.figure(figsize=(9, 5))
    b_vals_plot = list(delta_N_fraction_by_b.keys())
    fracs_plot = list(delta_N_fraction_by_b.values())
    plt.plot(b_vals_plot, fracs_plot, 'o-', color='darkred')
    plt.xlabel('Параметр b')
    plt.ylabel('Доля точек с ΔN ≠ 0')
    plt.title('Зависимость доли чувствительных точек (ΔN ≠ 0) от b')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.savefig('fraction_sensitive_points.png', dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 4: Разреженная модель (Раздел 8.3)")
    print("=" * 70)
    # Проверка совпадения ответов при k=500, b <= 10^5
    b_test = 10000
    a_test, c_test = 3, 7
    r_n = count_sp_naive(a_test, b_test, c_test, 500)
    r_f = count_sp_fastpow(a_test, b_test, c_test, 500)
    r_i = count_sp_iter(a_test, b_test, c_test, 500)
    print(f"Тест совпадения при b={b_test}, a={a_test}, c={c_test}: Naive={r_n}, Fastpow={r_f}, Iter={r_i}")

    # Замеры времени для разреженной модели
    b_sparse_list = [10**3, 10**5, 10**7, 10**9, 10**12, 10**15]
    print(f"\n{'b':<10}{'count_sp_naive':<20}{'count_sp_fastpow':<20}{'count_sp_iter':<20}")

    for b in b_sparse_list:
        # naive
        if b <= 10**5:
            t0 = time.perf_counter()
            count_sp_naive(2, b, 1, 500)
            t_n_str = f"{time.perf_counter() - t0:.6f} c"
        else:
            t_n_str = "не выполнять"

        # fastpow
        t0 = time.perf_counter()
        count_sp_fastpow(2, b, 1, 500)
        t_f_str = f"{time.perf_counter() - t0:.6f} c"

        # iter
        if b <= 10**7:
            t0 = time.perf_counter()
            count_sp_iter(2, b, 1, 500)
            t_i_str = f"{time.perf_counter() - t0:.6f} c"
        else:
            t_i_str = "не выполнять"

        print(f"{b:<10}{t_n_str:<20}{t_f_str:<20}{t_i_str:<20}")

    print("\nВсе вычисления успешно завершены! Графики сохранены в текущей папке.")

if __name__ == '__main__':
    run_all_experiments()