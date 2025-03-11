import numpy as np
from scipy.integrate import dblquad
import pandas as pd  # Импортируем pandas для работы с таблицами
import matplotlib.pyplot as plt

class BasisFunctions:
    """
    Класс для работы с базисными функциями и их производными.
    """
    @staticmethod
    def phi(i, xi, eta):
        """
        Базисные функции в локальной системе координат.
        """
        if i == 0:
            return 0.25 * (1 - xi) * (1 - eta)
        elif i == 1:
            return 0.25 * (1 + xi) * (1 - eta)
        elif i == 2:
            return 0.25 * (1 + xi) * (1 + eta)
        elif i == 3:
            return 0.25 * (1 - xi) * (1 + eta)
        else:
            raise ValueError("Invalid index for basis function")

    @staticmethod
    def dphi_dxi(i, xi, eta):
        """
        Производные базисных функций по xi.
        """
        if i == 0:
            return -0.25 * (1 - eta)
        elif i == 1:
            return 0.25 * (1 - eta)
        elif i == 2:
            return 0.25 * (1 + eta)
        elif i == 3:
            return -0.25 * (1 + eta)
        else:
            raise ValueError("Invalid index for basis function")

    @staticmethod
    def dphi_deta(i, xi, eta):
        """
        Производные базисных функций по eta.
        """
        if i == 0:
            return -0.25 * (1 - xi)
        elif i == 1:
            return -0.25 * (1 + xi)
        elif i == 2:
            return 0.25 * (1 + xi)
        elif i == 3:
            return 0.25 * (1 - xi)
        else:
            raise ValueError("Invalid index for basis function")

    @staticmethod
    def jacobian(xi, eta, nodes):
        """
        Вычисляет якобиан преобразования для элемента.
        nodes - глобальные координаты узлов элемента в порядке [ (x0, y0), (x1, y1), (x2, y2), (x3, y3) ].
        """
        dx_dxi = 0.0
        dx_deta = 0.0
        dy_dxi = 0.0
        dy_deta = 0.0

        for i in range(4):
            dx_dxi += nodes[i][0] * BasisFunctions.dphi_dxi(i, xi, eta)
            dx_deta += nodes[i][0] * BasisFunctions.dphi_deta(i, xi, eta)
            dy_dxi += nodes[i][1] * BasisFunctions.dphi_dxi(i, xi, eta)
            dy_deta += nodes[i][1] * BasisFunctions.dphi_deta(i, xi, eta)

        J = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])
        return J


class Mesh:
    """
    Класс для хранения информации о сетке.
    """
    def __init__(self):
        self.global_nodes = []  # Глобальные координаты узлов
        self.element_to_global = []  # Соответствие между локальными и глобальными узлами
        self.global_node_labels=[]

    def set_global_nodes(self, nodes):
        """
        Устанавливает глобальные координаты узлов.
        """
        self.global_nodes = nodes

    def set_element_to_global(self, element_to_global):
        """
        Устанавливает соответствие между локальными и глобальными узлами.
        """
        self.element_to_global = element_to_global

    def set_node_labels(self,labels):

        self.global_node_labels=labels

    def get_node_labels(self):

        return self.global_node_labels

    def get_global_nodes(self):
        """
        Возвращает глобальные координаты узлов.
        """
        return self.global_nodes

    def get_element_to_global(self):
        """
        Возвращает соответствие между локальными и глобальными узлами.
        """
        return self.element_to_global

    def get_num_global_nodes(self):
        """
        Возвращает количество глобальных узлов.
        """
        return len(self.global_nodes)
    
    def plot_mesh(self):

        x_coords, y_coords = zip(*self.global_nodes)

        # Create the plot
        plt.figure(figsize=(6, 6))  # Set the figure size
        plt.scatter(x_coords, y_coords, color='blue', label='Points')  # Plot the points

        for i, (xi, yi, label) in enumerate(zip(x_coords, y_coords, self.global_node_labels)):
            plt.annotate(
                label,              # Text to display
                (xi, yi),          # Coordinates of the point
                textcoords="offset points",  # Offset for the text
                xytext=(10, 5),     # Distance from the point (x, y)
                ha='center'         # Horizontal alignment of the text
            )

        E2G=self.get_element_to_global()
        for nodes in E2G:
            ND=[self.global_nodes[i] for i in nodes]
            x_coor, y_coor = zip(*ND)
            x0=x_coor[0]
            y0=y_coor[0]
            x_coor=list(x_coor)
            y_coor=list(y_coor)
            x_coor.append(float(x0))
            y_coor.append(float(y0))
            plt.plot(x_coor, y_coor, linestyle='--', color='red', label='Line')  # Optional: Connect points with a line

        
        # Set aspect ratio to 'equal'
        plt.gca().set_aspect('equal')
        plt.show()


class FiniteElementSolver:
    """
    Класс для формирования локальных и глобальных матриц жесткости.
    """
    def __init__(self, mesh, basis_functions):
        self.mesh = mesh
        self.basis_functions = basis_functions

    def stiffness_element_global(self, i, j, nodes):
        """
        Вычисляет элемент матрицы жесткости в глобальной системе координат.
        """
        def integrand(xi, eta):
            J = self.basis_functions.jacobian(xi, eta, nodes)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Преобразование производных в глобальную систему
            dphi_dx = invJ[0, 0] * self.basis_functions.dphi_dxi(i, xi, eta) + invJ[0, 1] * self.basis_functions.dphi_deta(i, xi, eta)
            dphi_dy = invJ[1, 0] * self.basis_functions.dphi_dxi(i, xi, eta) + invJ[1, 1] * self.basis_functions.dphi_deta(i, xi, eta)
            dpsi_dx = invJ[0, 0] * self.basis_functions.dphi_dxi(j, xi, eta) + invJ[0, 1] * self.basis_functions.dphi_deta(j, xi, eta)
            dpsi_dy = invJ[1, 0] * self.basis_functions.dphi_dxi(j, xi, eta) + invJ[1, 1] * self.basis_functions.dphi_deta(j, xi, eta)

            return (dphi_dx * dpsi_dx + dphi_dy * dpsi_dy) * detJ

        result, _ = dblquad(integrand, -1, 1, lambda x: -1, lambda x: 1)
        return result

    def local_stiffness_matrix_global(self, nodes):
        """
        Вычисляет локальную матрицу жесткости в глобальной системе координат.
        """
        K_local = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                K_local[i, j] = self.stiffness_element_global(i, j, nodes)
        return K_local

    def assemble_global_stiffness_matrix(self):
        """
        Собирает глобальную матрицу жесткости.
        """
        num_global_nodes = self.mesh.get_num_global_nodes()
        K_global = np.zeros((num_global_nodes, num_global_nodes))

        for element in self.mesh.get_element_to_global():
            # Глобальные координаты узлов текущего элемента
            nodes = [self.mesh.get_global_nodes()[i] for i in element]
            # Локальная матрица жесткости в глобальной системе
            K_local = self.local_stiffness_matrix_global(nodes)
            # Добавление в глобальную матрицу
            for i in range(4):
                for j in range(4):
                    K_global[element[i], element[j]] += K_local[i, j]

        return K_global

    def display_global_stiffness_matrix(self, K_global):
        """
        Отображает глобальную матрицу жесткости в виде таблицы.
        """
        # Преобразуем матрицу в DataFrame
        df = pd.DataFrame(K_global)
        # Отображаем таблицу
        print("Глобальная матрица жесткости:")
        print(df)


# Пример использования
if __name__ == "__main__":
    # Создаем объект сетки
    mesh = Mesh()
    mesh.set_global_nodes([
        (0.0, 0.0),  # Узел 0
        (1.0, 0.0),  # Узел 1
        (2.0, 0.0),  # Узел 2
        (3.0, 0.0),  # Узел 3
        (3.0, 1.0),  # Узел 4
        (2.0, 1.0),  # Узел 5
        (1.0, 1.0),  # Узел 6
        (0.0, 1.0),  # Узел 7
        
    ])
    mesh.set_element_to_global([
        [0, 1, 6, 7],  # Элемент 1
        [1, 2, 5, 6],  # Элемент 2
        [2, 3, 4, 5]   # Элемент 3
    ])

    mesh.set_node_labels([
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7'
   ])

    mesh.plot_mesh()

    # Создаем объект для базисных функций
    basis_functions = BasisFunctions()

    # Создаем решатель
    solver = FiniteElementSolver(mesh, basis_functions)

    # Собираем глобальную матрицу жесткости
    K_global = solver.assemble_global_stiffness_matrix()

    # Отображаем глобальную матрицу в виде таблицы
    solver.display_global_stiffness_matrix(K_global)