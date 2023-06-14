# Animación de la epidemia a través de agentes.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm, trange
import os

colores = ["blue", "red", "green", "black"]
archivos = [f"data/anim_simulacion_{i}.txt" for i in range(1,10)]
# ==============================================================================#
# Animacion
# ==============================================================================#
def trayectoria(archivo, n_sim, tpause=0.01):
    N = 100
    L = 50
    nsteps = np.loadtxt(archivo, usecols=0).size / N
    fig, ax = plt.subplots()
    loop_range = int(nsteps)

    mkdir_flag = True

    for i in trange(loop_range, mininterval=40):
        x = np.loadtxt(archivo, usecols=0, skiprows=N * i, max_rows=N)
        y = np.loadtxt(archivo, usecols=1, skiprows=N * i, max_rows=N)
        t = np.loadtxt(archivo, usecols=2, skiprows=N * i, max_rows=1)
        estado = np.loadtxt(archivo, usecols=3, skiprows=N * i, max_rows=N, dtype=int)

        plt.cla()
        plt.title("Agent system")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")

        plt.axis("square")
        plt.grid()
        plt.xlim(-1, L + 1)
        plt.ylim(-1, L + 1)

        population_count = np.array([0, 0, 0], dtype=int)
        for j in range(N):
            circ = patches.Circle((x[j], y[j]), 1, alpha=0.7, fc=colores[estado[j]])
            ax.add_patch(circ)
            if estado[j] == 0:
                population_count[0] += 1
            if estado[j] == 1:
                population_count[1] += 1
            if estado[j] == 2:
                population_count[2] += 1

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="%i" % population_count[0],
                markerfacecolor="b",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="%i" % population_count[1],
                markerfacecolor="r",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="%i" % population_count[2],
                markerfacecolor="g",
                markersize=10,
            ),
        ]

        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

        if mkdir_flag:
            try:
                os.mkdir(f"video/video_{n_sim}")
            except Exception as error:
                print(error)

            mkdir_flag = False

        plt.savefig(f"video/video_{n_sim}/pic{i}.png", dpi=70)



if __name__ == "__main__":
    for n_sim, file in enumerate(archivos):
        trayectoria(file, n_sim)
