# -*- coding: utf-8 -*-
'''
Модуль со вспомогательными классами и функциями, не связанные напрямую с
методом FDTD
'''
from abc import ABCMeta, abstractmethod
import pylab
from matplotlib import pyplot as plt
import numpy
from typing import List
from scipy.fft import fft, fftfreq


def fract(A, B):
    if B != 0:
        return A/B
    else:
        return 0

class HarmonicPlaneWave:
    ''' Класс с уравнением плоской волны для гармонического сигнала в дискретном виде
    Nl - количество ячеек на длину волны.
    phi0 - начальная фаза.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''
    def __init__(self, Nl, phi0, Sc=1.0, eps=1.0, mu=1.0):
        self.Nl = Nl
        self.phi0 = phi0
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return numpy.sin(2 * numpy.pi / self.Nl * (self.Sc * q - numpy.sqrt(self.mu * self.eps) * m) + self.phi0)

class Probe:
    '''
    Класс для хранения временного сигнала в датчике.
    '''
    def __init__(self, position: int, maxTime: int):
        '''
        position - положение датчика (номер ячейки).
        maxTime - максимально количество временных
            шагов для хранения в датчике.
        '''
        self.position = position

        # Временные сигналы для полей E и H
        self.E = numpy.zeros(maxTime)
        self.H = numpy.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E: List[float], H: List[float]):
        '''
        Добавить данные по полям E и H в датчик.
        '''
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1


class AnimateFieldDisplay:
    '''
    Класс для отображения анимации распространения ЭМ волны в пространстве
    '''

    def __init__(self,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str):
        '''
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y
        '''
        self.maxXSize = maxXSize
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, отсчет'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'

    def activate(self):
        '''
        Инициализировать окно с анимацией
        '''
        self._xList = numpy.arange(self.maxXSize)

        # Включить интерактивный режим для анимации
        pylab.ion()

        # Создание окна для графика
        self._fig, self._ax = pylab.subplots()

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, numpy.zeros(self.maxXSize))

    def drawProbes(self, probesPos: List[int]):
        '''
        Нарисовать датчики.

        probesPos - список координат датчиков для регистрации временных
            сигналов (в отсчетах).
        '''
        # Отобразить положение датчиков
        self._ax.plot(probesPos, [0] * len(probesPos), self._probeStyle)

    def drawSources(self, sourcesPos: List[int]):
        '''
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        '''
        # Отобразить положение источников
        self._ax.plot(sourcesPos, [0] * len(sourcesPos), self._sourceStyle)

    def drawBoundary(self, position: int):
        '''
        Нарисовать границу в области моделирования.

        position - координата X границы (в отсчетах).
        '''
        self._ax.plot([position, position],
                      [self.minYSize, self.maxYSize],
                      '--k')

    def stop(self):
        '''
        Остановить анимацию
        '''
        pylab.ioff()

    def updateData(self, data: List[float], timeCount: int):
        '''
        Обновить данные с распределением поля в пространстве
        '''
        self._line.set_ydata(data)
        self._ax.set_title(str(timeCount))
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


def ResultGraphs(probes: List[Probe], dt: float, fmin: float, fmax: float):
    
    N = int(len(probes[0].E))
    minYSize = -2.1
    maxYSize = 2.1
    xf = fftfreq(N, dt)
    # Создание окна с графиками
    plt.figure()
    plt.subplot(2, 2, 1)

    # Настройка внешнего вида графикa
    plt.xlim(0, N*dt*1e9)
    plt.ylim(minYSize, maxYSize)
    plt.xlabel('t, нс')
    plt.ylabel('Ez, В/м')
    plt.title("Падающий и отраженный\nсигналы")
    plt.grid()

    # Вывод сигналов в окно
    t = numpy.linspace(0, N*dt, N)
    for probe in probes:
        plt.plot(t*1e9, probe.E)
    

    # Создание и отображение легенды на графике
    legend = ['Probe x = {}'.format(probe.position) for probe in probes]
    plt.legend(legend)
    
    # Создание графикa спектра
    plt.subplot(2, 2, 2)
    E_f = [fft(probe.E) for probe in probes]
    plt.xlim(fmin, fmax)
    plt.xlabel('f, ГГц')
    plt.ylabel('|Ez(f)|')
    for En in E_f:
        plt.plot(xf*1e-9, numpy.abs(En))
    plt.legend(legend)
    plt.title("АЧХ падающего и\nотраженного сигнала")
    plt.grid()

    # Создание графикa |Г|
    plt.subplot(2, 1, 2)
    Gf = [fract(E_f[0][n], E_f[1][n]) for n in range(3, 12)]
    plt.xlim(fmin, fmax)
    f = [2*round(2e-9*max(xf)/len(xf),2)*m for m in range(1, 10)]
    plt.xlabel('f, ГГц')
    plt.ylabel('|Г(f)|')
    plt.xlim(fmin, fmax)
    plt.plot(f, numpy.abs(Gf))
    plt.grid()
    plt.title("Зависимость коэффициента отражения от частоты")
    # Показать окно с графиками
    plt.show()


class BoundaryBase(metaclass=ABCMeta):
    @abstractmethod
    def updateField(self, E, H):
        pass


class ABCSecondBase(BoundaryBase, metaclass=ABCMeta):
    '''
    Поглощающие граничные условия второй степени
    '''
    def __init__(self, eps, mu, Sc):
        Sc1 = Sc / numpy.sqrt(mu * eps)
        self.k1 = -1 / (1 / Sc1 + 2 + Sc1)
        self.k2 = 1 / Sc1 - 2 + Sc1
        self.k3 = 2 * (Sc1 - 1 / Sc1)
        self.k4 = 4 * (1 / Sc1 + Sc1)

        # E в предыдущий момент времени (q)
        self.oldE1 = numpy.zeros(3)

        # E в пред-предыдущий момент времени (q - 1)
        self.oldE2 = numpy.zeros(3)


class ABCSecondLeft(ABCSecondBase):
    def updateField(self, E, H):
        E[0] = (self.k1 * (self.k2 * (E[2] + self.oldE2[0]) +
                           self.k3 * (self.oldE1[0] + self.oldE1[2] - E[1] - self.oldE2[1]) -
                           self.k4 * self.oldE1[1]) - self.oldE2[2])

        self.oldE2[:] = self.oldE1[:]
        self.oldE1[:] = E[0: 3]


class ABCSecondRight(ABCSecondBase):
    def updateField(self, E, H):
        E[-1] = (self.k1 * (self.k2 * (E[-3] + self.oldE2[-1]) +
                            self.k3 * (self.oldE1[-1] + self.oldE1[-3] - E[-2] - self.oldE2[-2]) -
                            self.k4 * self.oldE1[-2]) - self.oldE2[-3])

        self.oldE2[:] = self.oldE1[:]
        self.oldE1[:] = E[-3:]

