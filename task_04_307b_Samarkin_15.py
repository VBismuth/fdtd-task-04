# -*- encoding: utf-8 -*-

import numpy
import tools


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0
    
    dx = 2e-3
    dt = 2e-12

    #Максимальная и минимальная частота, GHz
    fmin = 5 
    fmax = 13

    # Положение начала диэлектрика

    d0 = 100
    d1 = d0 + int(0.05/dx)
    d2 = d1 + int(0.03/dx)
    d3 = d2 + int(0.01/dx)

    # Время расчета в отсчетах
    maxTime = 500

    # Размер области моделирования в отсчетах
    maxSize = d3+int(0.01/dx)

    print(" maxSize = "+str(maxSize*dx)+"м\n",
    "maxTime = "+str(maxTime*dt*1e9)+"нс\n",
    "d0 = "+str(d0*dx)+"м", "d1 = "+str(d1*dx)+"м\n",
    "d2 = "+str(d2*dx)+"м","d3 = "+str(d3*dx)+"м")

    # Положение источника в отсчетах
    sourcePos = 60

    # Датчики для регистрации поля
    probesPos = [50, 60]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[d0:] = 9.2
    eps[d1:] = 1.7
    eps[d2:] = 2.2
    eps[d3:] = 2.0

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    # Создание экземпляров классов граничных условий
    boundary_left = tools.ABCSecondLeft(eps[0], mu[0], Sc)
    boundary_right = tools.ABCSecondRight(eps[-1], mu[-1], Sc)

    source = tools.GaussianPlaneWave(40.0, 12.0, Sc)


    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    for dn in [d0, d1, d2, d3]:
        display.drawBoundary(dn)

    for q in range(1, maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= (Sc / W0) * source.getE(0, q)


        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += Sc * source.getE(-0.5, q + 0.5)

        boundary_left.updateField(Ez, Hy)
        boundary_right.updateField(Ez, Hy)

        # Регистрация поля в датчиках
        for probe in probes:
            if probe.position != sourcePos or \
            q < 80:
                probe.addData(Ez, Hy)

        if q % 2 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение графиков
    tools.ResultGraphs(probes, dt, fmin, fmax)
