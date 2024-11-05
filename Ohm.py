import random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import erfc

# รับจำนวนบิต
n_bit = input("Nbits : ")
if n_bit.isdigit() and int(n_bit) > 0:
    rd.seed(1)
    N_bit = []
    for i in range(int(n_bit)):
        N_bit.append(i)

    data = []
    for _ in range(int(n_bit)):
        data.append(rd.randint(0, 1))

    Polar_NRZ = []
    for bit in data:
        if bit == 0:
            Polar_NRZ.append(-1)
        else:
            Polar_NRZ.append(1)

    Polar_RZ = []
    for bit in Polar_NRZ:
        if bit == -1:
            Polar_RZ.extend([-1, 0])
        else:
            Polar_RZ.extend([1, 0])

    D_NRZ = []
    previous_signal = 0
    for bit in data:
        if bit == 1:
            previous_signal = 1 - previous_signal
        D_NRZ.append(previous_signal)

    DNRZ = []
    for bit in D_NRZ:
        if bit == 0:
            DNRZ.append(-1)
        else:
            DNRZ.append(1)

    # ฟังก์ชันสร้าง RRC filter
    def root_raised_cosine(num_taps, Ts, Fs, beta):
        t = np.arange(-num_taps // 2, num_taps // 2) / Fs
        h = np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
        h[np.abs(t) == Ts / (2 * beta)] = np.pi / 4 * np.sinc(1 / (2 * beta))
        return h

    # ฟังก์ชันคำนวณ BER ตามทฤษฎี
    def calculate_ber(Eb_N0_dB):
        Eb_N0 = 10 ** (Eb_N0_dB / 10)
        return 0.5 * erfc(np.sqrt(Eb_N0))

    def calculate_ber_rz(Eb_N0_dB):
        Eb_N0 = 10 ** (Eb_N0_dB / 10)
        return 0.5 * erfc(np.sqrt(Eb_N0 / 2))

    # สร้าง RRC Filter
    beta = 0.25
    Ts = 1
    Fs = 8
    num_taps = 101
    rrc_filter = root_raised_cosine(num_taps, Ts, Fs, beta)

    # ใช้ RRC Filter กับสัญญาณที่ส่ง
    Polar_NRZ_Tx = signal.convolve(Polar_NRZ, rrc_filter, mode='full')[num_taps//2 : -(num_taps//2)]
    Polar_RZ_Tx = signal.convolve(Polar_RZ, rrc_filter, mode='full')[num_taps//2 : -(num_taps//2)]
    DNRZ_Tx = signal.convolve(DNRZ, rrc_filter, mode='full')[num_taps//2 : -(num_taps//2)]

    Eb_N0_dB_range = np.arange(0, 10, 1)
    BER_Polar_NRZ_values = []
    BER_Polar_RZ_values = []
    BER_DNRZ_values = []
    BER_theory_values = []
    for Eb_N0_dB in Eb_N0_dB_range:
      BER_theory_values.append(calculate_ber(Eb_N0_dB))

    BER_theory_rz_values = []
    for Eb_N0_dB in Eb_N0_dB_range:
      BER_theory_rz_values.append(calculate_ber_rz(Eb_N0_dB))

    # คำนวณพลังงานสัญญาณ
    signal_power_NRZ = np.mean(np.array(Polar_NRZ)**2)
    signal_power_RZ = np.mean(np.array(Polar_RZ)**2) / 2  # เนื่องจาก RZ มี duty cycle 50%
    signal_power_DNRZ = np.mean(np.array(DNRZ)**2)

    for Eb_N0_dB in Eb_N0_dB_range:
        # คำนวณค่า sigma ตามพลังงานสัญญาณของแต่ละรูปแบบการเข้ารหัส
        Eb_N0 = 10 ** (Eb_N0_dB / 10)
        sigma_NRZ = np.sqrt((1 / 2) * (10) * (10 ** (-Eb_N0_dB / 10)))
        sigma_RZ = np.sqrt((1 / 2) * (10) * (10 ** (-Eb_N0_dB / 10)))
        sigma_DNRZ = np.sqrt((1 / 2) * (10) * (10 ** (-Eb_N0_dB / 10)))

        # เพิ่ม noise แบบ AWGN
        AWGN_NRZ = np.random.normal(0, sigma_NRZ, len(Polar_NRZ_Tx))
        AWGN_RZ = np.random.normal(0, sigma_RZ, len(Polar_RZ_Tx))
        AWGN_DNRZ = np.random.normal(0, sigma_DNRZ, len(DNRZ_Tx))

        # สัญญาณที่รับ
        Polar_NRZ_Rx = Polar_NRZ_Tx + AWGN_NRZ
        Polar_RZ_Rx = Polar_RZ_Tx + AWGN_RZ
        DNRZ_Rx = DNRZ_Tx + AWGN_DNRZ

        # Decision
        Polar_NRZ_Rx_decision = (Polar_NRZ_Rx > 0).astype(int)
        Polar_RZ_Rx_decision = (Polar_RZ_Rx > 0).astype(int)[::2]  # ตัดสินใจที่ duty cycle 50%

        # Differential NRZ Decision
        DNRZ_Rx_decision = (DNRZ_Rx > 0).astype(int)
        DNRZ_Rx_data = []
        previous_bit = 0
        for i in range(len(DNRZ_Rx_decision)):
            if DNRZ_Rx_decision[i] != previous_bit:
                DNRZ_Rx_data.append(1)
            else:
                DNRZ_Rx_data.append(0)
            previous_bit = DNRZ_Rx_decision[i]

        # คำนวณ BER
        BER_Polar_NRZ = np.sum(np.array(data) != Polar_NRZ_Rx_decision) / len(data)
        BER_Polar_RZ = np.sum(np.array(data) != Polar_RZ_Rx_decision) / len(data)
        BER_DNRZ = np.sum(np.array(data) != np.array(DNRZ_Rx_data)) / len(data)

        BER_Polar_NRZ_values.append(BER_Polar_NRZ)
        BER_Polar_RZ_values.append(BER_Polar_RZ)
        BER_DNRZ_values.append(BER_DNRZ)

    N_bit_ext = np.arange(len(Polar_NRZ_Tx)) * Ts / Fs  # แกนเวลา
    N_bit_RZ_ext = np.arange(len(Polar_RZ_Tx)) * Ts / Fs
    # วาดกราฟสัญญาณที่ผ่าน RRC filter
    plt.figure(figsize=(10, 12))

    # สัญญาณ Polar NRZ ที่ส่งและรับ (หลังผ่าน RRC filter)
    plt.subplot(3, 2, 1)
    plt.title("Tx Polar NRZ (RRC Filtered)")
    plt.plot(N_bit_ext, Polar_NRZ_Tx)
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.title("Rx Polar NRZ (RRC Filtered)")
    plt.plot(N_bit_ext, Polar_NRZ_Rx)
    plt.grid()

    # สัญญาณ Polar RZ ที่ส่งและรับ (หลังผ่าน RRC filter)
    plt.subplot(3, 2, 3)
    plt.title("Tx Polar RZ (RRC Filtered)")
    plt.plot(N_bit_RZ_ext, Polar_RZ_Tx)
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.title("Rx Polar RZ (RRC Filtered)")
    plt.plot(N_bit_RZ_ext, Polar_RZ_Rx)
    plt.grid()


    # สัญญาณ Differential NRZ ที่ส่งและรับ (หลังผ่าน RRC filter)
    plt.subplot(3, 2, 5)
    plt.title("Tx Differential NRZ (RRC Filtered)")
    plt.plot(N_bit_ext, DNRZ_Tx)
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.title("Rx Differential NRZ (RRC Filtered)")
    plt.plot(N_bit_ext, DNRZ_Rx)
    plt.grid()
    print("ขนาดของ Eb_N0_dB_range:", len(Eb_N0_dB_range))
    print("ขนาดของ BER_Polar_NRZ_values:", len(BER_Polar_NRZ_values))
    print("ขนาดของ BER_Polar_RZ_values:", len(BER_Polar_RZ_values))
    print("ขนาดของ BER_DNRZ_values:", len(BER_DNRZ_values))

    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.semilogy(Eb_N0_dB_range, BER_Polar_NRZ_values, 'o-', label='Simulated Polar NRZ')
    plt.semilogy(Eb_N0_dB_range, BER_Polar_RZ_values, 's-', label='Simulated Polar RZ')
    plt.semilogy(Eb_N0_dB_range, BER_DNRZ_values, '^-', label='Simulated Differential NRZ')
    #plt.plot(Eb_N0_dB_range, BER_theory_values, 'r--', label='Theoretical Polar NRZ')
    #plt.plot(Eb_N0_dB_range, BER_theory_rz_values, 'm--', label='Theoretical Polar RZ')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER vs Eb/N0 for Different Encoding Schemes')
    plt.legend()
    plt.grid(True)
    plt.show()
