import numpy as np

class StructuredMesh2D:
    def __init__(self, nx, ny, x_range=(0, 1), y_range=(0, 1)):
        self.nx, self.ny = nx, ny
        self.x_range, self.y_range = x_range, y_range

        # uniform spacing
        self.dx = (x_range[1] - x_range[0]) / nx
        self.dy = (y_range[1] - y_range[0]) / ny

        # store cells as (i, j) indices
        self.cells = [(i, j) for j in range(ny) for i in range(nx)]

    def cell_bounds(self, i, j):
        """Return physical coordinates of cell (i, j)."""
        x0 = self.x_range[0] + i * self.dx
        x1 = x0 + self.dx
        y0 = self.y_range[0] + j * self.dy
        y1 = y0 + self.dy
        return (x0, x1, y0, y1)

    def neighbors(self, i, j):
        """Return neighbors (left, right, bottom, top).
           None if boundary.
        """
        neigh = {}
        neigh["left"]   = (i-1, j) if i > 0 else None
        neigh["right"]  = (i+1, j) if i < self.nx-1 else None
        neigh["bottom"] = (i, j-1) if j > 0 else None
        neigh["top"]    = (i, j+1) if j < self.ny-1 else None
        return neigh

    def face_normals(self):
        """Normals for left, right, bottom, top faces."""
        return {
            "left":   np.array([1.0, 0.0]),
            "right":  np.array([-1.0, 0.0]),
            "bottom": np.array([0.0, 1.0]),
            "top":    np.array([0.0, -1.0]),
        }
