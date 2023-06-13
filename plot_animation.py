# Animación de la epidemia a través de agentes.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm, trange
import os

colores = [
    "blue",
    "red",
    "green",
    "black"
]

archivo = [f"data/anim_simulacion_{i}.txt" for i in range(10)]

print(archivo)

# Animacion
def trayectoria(archivo, tpause=0.01):

    N = 100
    L = 30

    nsteps = np.loadtxt(archivo, usecols=0).size / N
    fig, ax = plt.subplots()
    loop_range = int(nsteps)
    n_sim = 0

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

        # if t == 0:
        #     try:
        #         os.mkdir("video/video%i" % n_sim)
        #         # os.system(f'mkdir video/video{n_sim}')
        #     except:
        #         pass
        #     finally:
        #         n_sim += 1

        #plt.savefig("video/video%i/pic%.4i.png" % (n_sim - 1, i), dpi=70)
        plt.savefig(f"video/pic{i}.png", dpi=70)
        # plt.pause(tpause)

    """
	if animation:
		path = "C:/Users/Admin/Desktop/GIT/Agentes/video"
		print(os.getcwd())
		os.chdir(path)
		print(os.getcwd())
		os.system('cmd /k "ffmpeg -r 30 -f image2 -s 1920x1080 -i pic%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test_0.mp4"')
	"""


if __name__ == "__main__":
    print("main")
    trayectoria()
