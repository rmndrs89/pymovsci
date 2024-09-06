import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# imagebox = OffsetImage(logo, zoom = 0.15)
# ab = AnnotationBbox(imagebox, (5, 700), frameon = False)
# ax.add_artist(ab)

def main() -> None:
    theta = 35. * np.pi / 180.  # angle in radians
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    t = np.array([3, 2, 0])

    fig, ax = plt.subplots()
    ax.plot(
        [t[0], t[0] + x[0]],
        [t[1], t[1] + x[1]],
        c=(1., 0., 1.), lw=2, label="u"
    )
    ax.plot(
        [t[0], t[0] + y[0]],
        [t[1], t[1] + y[1]],
        c=(0., 1., 0.), lw=2, label="v"
    )
    ax.set_xlim((-1, 8))
    ax.set_ylim((-1, 8))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_position("zero")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)
    plt.show()

    return


if __name__ == "__main__":
    main()