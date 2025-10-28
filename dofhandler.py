import numpy as np

class DoFHandler2D:
    def __init__(self, mesh, dofs_per_cell):
        self.mesh = mesh
        self.dofs_per_cell = dofs_per_cell

        # assign global DoF indices per cell
        self.cell_to_dofs = {}
        counter = 0
        for cell in mesh.cells:
            i, j = cell
            self.cell_to_dofs[(i, j)] = list(range(counter, counter + dofs_per_cell))
            counter += dofs_per_cell
        self.ndofs = counter

    def iter_cells(self):
        """Iterator over all cells."""
        for cell in self.mesh.cells:
            yield cell, self.cell_to_dofs[cell]

    def iter_faces(self):
        """Iterator over all faces (cell, neighbor, normal, dofs)."""
        normals = self.mesh.face_normals()
        for (i, j) in self.mesh.cells:
            neighs = self.mesh.neighbors(i, j)
            for face, neighbor in neighs.items():
                yield {
                    "cell": (i, j),
                    "neighbor": neighbor,
                    "normal": normals[face],
                    "dofs_cell": self.cell_to_dofs[(i, j)],
                    "dofs_neighbor": None if neighbor is None else self.cell_to_dofs[neighbor],
                    "face": face
                }