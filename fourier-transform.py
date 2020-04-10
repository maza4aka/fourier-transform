import numpy as np
import matplotlib.pyplot as plt


# Случайная амплитуда или фаза. :return: амплитуда или фаза
amplitude = angle_phi = lambda lower, upper: np.random.uniform(lower, upper)


def generate_signal(harmonics, frequency, sampling):
    """
    Случайный сигнал...
    x(t) = sum( amp * sin(wp * t + phi) )
    :param harmonics: количество гармоник
    :param frequency: граничная частота
    :param sampling: степень дискретизации
    :return: сигнал
    """

    # Wгр / (n - 1) - шаг частоты для расчёта wp ниже
    step = frequency / (harmonics - 1)

    # создаём матрицу: строки - гармоники, столбцы - сигналы
    harmonics_matrix = np.zeros((harmonics, sampling))

    # генерируем гармоники, заполняем строки матрицы сигналами
    for h in range(harmonics):
        wp = frequency - h * step
        for t in range(sampling):
            harmonics_matrix[h, t] = \
                amplitude(-5, 5) * np.sin(wp * t + angle_phi(0, 360))

    # суммируем гармоники.
    return np.array([np.sum(v) for v in harmonics_matrix.T])


def discrete_fourier_transform(signal):
    """
    Дискретное преобразование Фурье.
    F(p) = sum( x(k)*W_N^(p*k) ), where W_N^(p*k) = e^(i*(-2pi/N)*p*k) = cos( (2pi/N)*p*k ) - i*sin( (2pi/N)*p*k )
    :param signal: сигнал для преобразования
    :return: результат F.
    """

    N = len(signal)

    # массив F
    F = np.zeros(N, dtype=complex)

    for p in range(N):
        for k in range(N):
            F[p] += complex(signal[k] * np.cos( (2*np.pi*p*k)/N ), - signal[k] * np.sin( (2*np.pi*p*k)/N ))

    return F


if __name__ == '__main__':
    """
    Исследование дискретного преобразования Фурье...
    """

    # количество генерируемых гармоник
    number_of_harmonics = 14

    # граничная/максимальная частота
    cutoff_frequency = 2000

    # степень дискретизации (количество точек на графике)
    sampling_rate = 256

    #signal = np.sin(np.linspace(0, np.pi*64, sampling_rate))
    signal = generate_signal(number_of_harmonics, cutoff_frequency, sampling_rate)

    # получаем данные для N синусоид
    DFT = discrete_fourier_transform(signal)

    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    axs[1].set_title('Дискретное преобразование Фурье\n')
    axs[1].plot([np.sqrt(z.real*z.real + z.imag*z.imag)/len(DFT) for z in DFT]) # амплитудный спектр
    axs[2].plot([np.arctan(z.imag/z.real) for z in DFT]) # спектр фаз

    axs[1].set_xlabel('синусоида')
    axs[1].set_ylabel('амплитуда')
    axs[1].grid(True)

    axs[2].set_xlabel('синусоида')
    axs[2].set_ylabel('фаза')
    axs[2].grid(True)

    axs[0].set_title('Случайный сигнал\n')
    axs[0].plot(signal)

    axs[0].set_xlabel('время')
    axs[0].set_ylabel('сигнал')

    fig.canvas.set_window_title('')

    plt.show()

