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
    Сложность: О(n^2)
    F(p) = sum( x(k)*W_N^(p*k) ), where W_N^(p*k) = e^(i*(-2pi/N)*p*k) = cos( (2pi/N)*p*k ) - i*sin( (2pi/N)*p*k )
    :param signal: сигнал для преобразования
    :return: результат F.
    """

    N = len(signal)

    W = lambda p, k: complex(np.cos( (2*np.pi*p*k)/N ), - np.sin( (2*np.pi*p*k)/N ))

    # массив F
    F = np.zeros(N, dtype=complex)

    for p in range(N):
        for k in range(N):
            F[p] += signal[k] * W(p, k)

    return F


def fast_fourier_transform(signal):
    """
    Быстрое преобразование Фурье!
    Модификация алгоритма выше, со сложностью O(nlog(n))
    =P
    """

    #return np.fft.fft(signal)

    N = len(signal)

    # массив F
    F = np.zeros(N, dtype=complex)

    if N % 2 == 0: # должно быть степенью двух.
        even = fast_fourier_transform(signal[0::2])
        odd = fast_fourier_transform(signal[1::2])

        W = lambda k: np.exp(-2j*np.pi*k/N)

        for k in range(N//2):
            F[k] = even[k] + W(k) * odd[k]
            F[k + N//2] = even[k] - W(k) * odd[k]
    elif N == 1: # замыкаем рекурсию...
        return signal
    else:
        raise ValueError("размер массива signal должен быть степенью 2`ки!")

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

    signal = generate_signal(number_of_harmonics, cutoff_frequency, sampling_rate)

    # получаем данные для N синусоид
    DFT = discrete_fourier_transform(signal)
    FFT = fast_fourier_transform(signal)

    # проверим, правильно ли работают алгоритмы?..
    assert np.allclose(DFT, np.fft.fft(signal)), "DFT дал неверный результат!"
    assert np.allclose(DFT, FFT), "DFT и FFT дали разные результаты?"
    ft = FFT

    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    x = np.arange(0, sampling_rate, 1)

    axs[1].set_title('Быстрое преобразование Фурье...\nThe Cooley-Tukey algorithm!\n')
    axs[1].bar(x, [np.sqrt(z.real*z.real + z.imag*z.imag)/len(ft) for z in ft]) # амплитудный спектр: |X_k|/N = sqrt(z.real^2 + z.imag^2)/N
    axs[2].bar(x, [np.arctan(z.imag/z.real) for z in ft]) # спектр фаз: phi = arctg( z.imag/z.real )

    axs[1].set_xlabel('n`ая синусоида')
    axs[1].set_ylabel('амплитуда')
    axs[1].grid(True)

    #axs[2].set_xlabel('n`ая синусоида')
    axs[2].set_ylabel('фаза')
    axs[2].grid(True)

    axs[0].set_title('Случайный сигнал\n')
    axs[0].plot(signal)

    axs[0].set_xlabel('время')
    axs[0].set_ylabel('сигнал')

    fig.canvas.set_window_title('')

    plt.show()

